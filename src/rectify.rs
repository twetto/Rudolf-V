// rectify.rs — Stereo rectification: turn both cameras of a calibrated rig so
// that their epipolar lines become image rows, and resample raw images into a
// shared pinhole camera.
//
// The rotations follow Bouguet's method as OpenCV's stereoRectify computes it
// with CALIB_ZERO_DISPARITY: each camera turns by half the rotation between
// them, so that both look the same way, then both turn so that the x-axis lies
// along the baseline. The shared pinhole camera (fx = fy, cx, cy) is the
// caller's choice: the ZED SDK's rectified camera, for instance, puts the
// images on the grid of the SDK's own rectified images and depth.
//
// A map per eye holds, for every rectified pixel, the raw-image position that
// sees the same ray: K⁻¹·(u, v, 1), turned back into the raw camera, through
// its distortion model. Rectifying a frame samples the raw image there,
// bilinearly; the GPU does the same while building its pyramid. The map is kept
// as each pixel's bilinear tap (the raw pixel at the top left and the two
// weights), computed once, so a frame needs no clamping or conversion per
// pixel.

use crate::camera::{CameraIntrinsics, StereoRig};
use crate::image::Image;
use camera_geometry::CameraModel;
use nalgebra::{Matrix3, Rotation3, Vector3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// One rectified pixel's bilinear tap in the raw image: the raw pixel at the
/// top left, x0 | y0 << 16, and the weights of the pixels right of it (wx)
/// and below it (wy). The GPU reads the same layout
/// (pyramid_convert_rect.wgsl).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RemapTap {
    pub pos: u32,
    pub wx: f32,
    pub wy: f32,
}

impl RemapTap {
    /// No raw position (the camera model has no projection): the pixel is 0.
    pub const NONE: Self = Self {
        pos: u32::MAX,
        wx: 0.0,
        wy: 0.0,
    };

    /// The tap at raw position (x, y) of a w × h image, clamped into it.
    fn at([x, y]: [f32; 2], w: usize, h: usize) -> Self {
        if !(x.is_finite() && y.is_finite()) {
            return Self::NONE;
        }
        let x = x.clamp(0.0, (w - 1) as f32);
        let y = y.clamp(0.0, (h - 1) as f32);
        let (x0, y0) = ((x as usize).min(w - 2), (y as usize).min(h - 2));
        Self {
            pos: x0 as u32 | (y0 as u32) << 16,
            wx: x - x0 as f32,
            wy: y - y0 as f32,
        }
    }

    /// The raw position the tap samples (clamped into the image); None for
    /// `NONE`.
    pub fn position(self) -> Option<[f32; 2]> {
        let (x0, y0) = ((self.pos & 0xffff) as f32, (self.pos >> 16) as f32);
        (self != Self::NONE).then_some([x0 + self.wx, y0 + self.wy])
    }
}

/// Rectification of a stereo rig into a pinhole pair with row-aligned
/// epipolar lines.
#[derive(Debug, Clone)]
pub struct StereoRectification {
    width: usize,
    height: usize,
    fx: f64,
    cx: f64,
    cy: f64,
    /// Per eye, the rotation from the raw camera's frame into the rectified
    /// one: x_rect = R·x_raw.
    rotations: [[[f64; 3]; 3]; 2],
    /// The rectified right camera's translation (x_right = x_left + t).
    t: [f64; 3],
    /// Per eye, the bilinear tap of each rectified pixel, row-major.
    maps: [Vec<RemapTap>; 2],
}

impl StereoRectification {
    /// Rectification of `rig` (its raw cameras and their distortion) into a
    /// shared pinhole camera with focal length `fx` (= fy) and principal point
    /// (cx, cy), at the left camera's resolution.
    pub fn new(rig: &StereoRig, fx: f64, cx: f64, cy: f64) -> Self {
        let r = Matrix3::from_fn(|i, j| rig.r_10[i][j]);
        let t = Vector3::from(rig.t_10);
        // Each camera turns by half the rotation between them.
        let om = Rotation3::from_matrix_unchecked(r).scaled_axis();
        let half = Rotation3::from_scaled_axis(-om * 0.5);
        let th = half * t;
        // Then both turn the baseline onto the x-axis (the y-axis for a
        // vertical rig).
        let axis = if th.x.abs() > th.y.abs() { 0 } else { 1 };
        let mut u = Vector3::zeros();
        u[axis] = th[axis].signum();
        let w = th.cross(&u);
        let turn = if w.norm() > 0.0 {
            Rotation3::from_scaled_axis(w * ((th[axis].abs() / th.norm()).acos() / w.norm()))
        } else {
            Rotation3::identity()
        };
        let r0 = turn * half.inverse();
        let r1 = turn * half;
        // The rectified translation lies on the chosen axis, up to rounding.
        let tr = r1 * t;
        let mut t_rect = [0.0; 3];
        t_rect[axis] = tr[axis];

        let [width, height] = rig.cam0.resolution;
        assert!(width >= 2 && height >= 2, "image too small to rectify");
        let to_array = |m: Rotation3<f64>| {
            let m = m.matrix();
            [0, 1, 2].map(|i| [m[(i, 0)], m[(i, 1)], m[(i, 2)]])
        };
        let rotations = [to_array(r0), to_array(r1)];
        let projections = [&rig.cam0_projection, &rig.cam1_projection];
        let maps = [0, 1].map(|eye| {
            let rt = [r0, r1][eye].inverse();
            let mut map = Vec::with_capacity(width * height);
            for v in 0..height {
                for u in 0..width {
                    let n = Vector3::new((u as f64 - cx) / fx, (v as f64 - cy) / fx, 1.0);
                    map.push(match projections[eye].project(rt * n) {
                        Some(p) => RemapTap::at([p.x() as f32, p.y() as f32], width, height),
                        None => RemapTap::NONE,
                    });
                }
            }
            map
        });
        Self {
            width,
            height,
            fx,
            cx,
            cy,
            rotations,
            t: t_rect,
            maps,
        }
    }

    /// The rectified pair: both cameras the shared pinhole camera, no
    /// rotation between them.
    pub fn rig(&self) -> StereoRig {
        let cam =
            CameraIntrinsics::new(self.fx, self.fx, self.cx, self.cy, self.width, self.height);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        StereoRig::new(cam.clone(), cam, identity, self.t)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The shared pinhole camera: (fx = fy, cx, cy).
    pub fn camera(&self) -> (f64, f64, f64) {
        (self.fx, self.cx, self.cy)
    }

    /// Rotation from eye `eye`'s raw camera frame into its rectified frame
    /// (0 left, 1 right). A rectified ray is R·(raw ray), and the range along
    /// it is the same.
    pub fn rotation(&self, eye: usize) -> [[f64; 3]; 3] {
        self.rotations[eye]
    }

    /// The bilinear tap of every rectified pixel of eye `eye` in its raw
    /// image, row-major.
    pub fn map(&self, eye: usize) -> &[RemapTap] {
        &self.maps[eye]
    }

    /// Rectify eye `eye` of a raw frame into `out` (grey levels as f32):
    /// bilinear samples of `raw` at the map, with the image's edge repeated
    /// beyond it, and 0 where the map has no position.
    pub fn remap(&self, eye: usize, raw: &Image<u8>, out: &mut Image<f32>) {
        let (w, h) = (self.width, self.height);
        assert_eq!((raw.width(), raw.height()), (w, h), "raw image size");
        out.clear_resize(w, h);
        let map = &self.maps[eye];
        let (data, stride) = (raw.as_slice(), raw.stride());
        let row = |y: usize, o: &mut [f32]| {
            for (v, t) in o.iter_mut().zip(&map[y * w..(y + 1) * w]) {
                if t.pos == u32::MAX {
                    *v = 0.0;
                    continue;
                }
                let j = (t.pos >> 16) as usize * stride + (t.pos & 0xffff) as usize;
                let (r0, r1) = (&data[j..j + 2], &data[j + stride..j + stride + 2]);
                let (a, b, c, d) = (r0[0] as f32, r0[1] as f32, r1[0] as f32, r1[1] as f32);
                *v = (1.0 - t.wy) * ((1.0 - t.wx) * a + t.wx * b)
                    + t.wy * ((1.0 - t.wx) * c + t.wx * d);
            }
        };
        let dst = out.as_mut_slice();
        #[cfg(feature = "parallel")]
        dst.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, o)| row(y, o));
        #[cfg(not(feature = "parallel"))]
        dst.chunks_mut(w).enumerate().for_each(|(y, o)| row(y, o));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use camera_geometry::Pixel;

    /// A raw rig like the ZED 2i's: rational distortion, a small rotation and
    /// offsets between the eyes.
    fn zed_like() -> StereoRig {
        let cam = |fx: f64, cx: f64, cy: f64, k: [f64; 8]| CameraIntrinsics {
            fx,
            fy: fx + 0.5,
            cx,
            cy,
            resolution: [1280, 720],
            distortion: k.to_vec(),
            model: crate::camera::DistortionModel::Rational,
        };
        let k = [-4.26, 46.5, 4.2e-4, 9.1e-6, -23.3, -4.17, 46.1, -19.8];
        let r = Rotation3::from_scaled_axis(Vector3::new(0.003, 0.004, 0.0035));
        let m = r.matrix();
        StereoRig::new(
            cam(958.2, 651.6, 353.6, k),
            cam(956.1, 653.0, 359.9, k),
            [0, 1, 2].map(|i| [m[(i, 0)], m[(i, 1)], m[(i, 2)]]),
            [-0.12, -2.5e-4, 4.3e-4],
        )
    }

    #[test]
    fn rectified_rays_meet_on_the_same_row() {
        // A point seen by both raw cameras lands on the same row of both
        // rectified images, and the rectified rig is a pinhole pair along x.
        let rig = zed_like();
        let rect = StereoRectification::new(&rig, 951.4, 658.9, 356.7);
        let r0 = Matrix3::from_fn(|i, j| rect.rotation(0)[i][j]);
        let r1 = Matrix3::from_fn(|i, j| rect.rotation(1)[i][j]);
        let r = Matrix3::from_fn(|i, j| rig.r_10[i][j]);
        let t = Vector3::from(rig.t_10);
        let (f, cx, cy) = rect.camera();
        let project = |p: Vector3<f64>| (f * p.x / p.z + cx, f * p.y / p.z + cy);
        for p in [
            Vector3::new(0.3, -0.2, 1.0),
            Vector3::new(-1.0, 0.5, 3.0),
            Vector3::new(0.0, 0.0, 0.4),
        ] {
            let left = project(r0 * p);
            let right = project(r1 * (r * p + t));
            assert!((left.1 - right.1).abs() < 1e-9, "rows {left:?} {right:?}");
            assert!(left.0 > right.0, "positive disparity");
        }
        let rr = rect.rig();
        assert!(rr.t_10[0] < 0.0 && rr.t_10[1] == 0.0 && rr.t_10[2] == 0.0);
        assert!((rr.baseline_meters() - t.norm()).abs() < 1e-12);
    }

    #[test]
    fn map_follows_the_raw_projection() {
        // The map of a rectified pixel is the raw camera's projection of its
        // ray, turned back by the eye's rotation.
        let rig = zed_like();
        let rect = StereoRectification::new(&rig, 951.4, 658.9, 356.7);
        let (f, cx, cy) = rect.camera();
        for eye in 0..2 {
            let rt = Matrix3::from_fn(|i, j| rect.rotation(eye)[i][j]).transpose();
            let proj = if eye == 0 {
                &rig.cam0_projection
            } else {
                &rig.cam1_projection
            };
            for (u, v) in [(0usize, 0usize), (640, 360), (1279, 719), (100, 600)] {
                let n = Vector3::new((u as f64 - cx) / f, (v as f64 - cy) / f, 1.0);
                let p: Pixel = proj.project(rt * n).expect("inside the model");
                let m = rect.map(eye)[v * rect.width() + u]
                    .position()
                    .expect("a tap");
                assert!((m[0] as f64 - p.x()).abs() < 1e-3 && (m[1] as f64 - p.y()).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn remap_of_an_identity_map_is_the_image() {
        // A pinhole rig without rotation rectifies to itself.
        let cam = CameraIntrinsics::new(500.0, 500.0, 31.5, 23.5, 64, 48);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rig = StereoRig::new(cam.clone(), cam, identity, [-0.1, 0.0, 0.0]);
        let rect = StereoRectification::new(&rig, 500.0, 31.5, 23.5);
        let data: Vec<u8> = (0..64 * 48).map(|i| (i * 7 % 256) as u8).collect();
        let raw = Image::from_vec(64, 48, data);
        let mut out = Image::new(1, 1);
        rect.remap(0, &raw, &mut out);
        for y in 0..48 {
            for x in 0..64 {
                assert!(
                    (out.get(x, y) - raw.get(x, y) as f32).abs() < 1e-3,
                    "({x},{y})"
                );
            }
        }
    }
}
