// dense_stereo.rs — Dense stereo depth: DIS-style patch inverse search along
// the epipolar curve, variational refinement, and per-pixel validity.
//
// Follows OpenCV's DIS optical flow (Kroeger et al., "Fast Optical Flow using
// Dense Inverse Search", ECCV 2016; modules/video/src/dis_flow.cpp and
// variational_refinement.cpp), with the correspondence search reduced to one
// parameter because the rig is calibrated:
//
//   1. Pyramid of both images (the frontend's Gaussian pyramid). No
//      equalization: CLAHE's tone curve varies across the image, so on a
//      flat surface it paints ramps fixed to the image, not to the scene,
//      which both eyes see in the same place: they match at zero disparity
//      (a textureless wall at 2 m read as 6 m).
//   2. Patch inverse search, coarse to fine. Overlapping patches (8×8, stride
//      4), each running inverse-compositional Gauss-Newton on one parameter D
//      (below). A patch slides along its centre's epipolar curve
//      π1(R·b + ρ·t) in the right image, so raw (distorted) images work as
//      well as rectified ones. The coarsest level starts from an exhaustive
//      1-D search, every finer level from the coarser level's result.
//      Spatial propagation in red-black sweeps: a patch first tries its four
//      neighbours' estimates and keeps the one with the lowest SSD. (DIS
//      propagates sequentially along rows; red-black gives the GPU the same
//      result as this reference.)
//   3. Densification: each pixel is the weighted mean of the patches over it,
//      weight 1/max(1, |photometric residual|) as in DIS, times the patch's
//      inverse variance.
//   4. Variational refinement at the finest level (DIS: at every level, which
//      `refine_coarse_levels` restores): Brox et al. 2004's energy as
//      OpenCV has it (colour and gradient constancy, normalised; TV
//      smoothness), with one unknown per pixel and the data term's derivative
//      taken along the epipolar direction. Fixed-point iterations, red-black
//      SOR. Optionally the smoothness weight falls where the left image has an
//      edge, so depth may break where the grey image does.
//   5. Validity, at the finest level: a pixel has depth only if a patch over
//      it has texture along its epipolar direction, matched (residual)
//      distinctly (its SSD rises when D moves two pixels either way: noise
//      matches every D about equally), lies inside the right image, and the
//      pixel is not occluded in the right image (z-buffer of the
//      forward-warped left image). Everything else is
//      NaN: no depth is invented across textureless regions or occlusions.
//   6. Coarse fallback (`coarse_fallback`): where the finest level found no
//      texture, a coarser level's patches, larger in the scene, may have:
//      such a pixel takes the nearest coarser level's valid estimate, with
//      that level's variance, unless a finer measurement within the coarse
//      estimate's support disagrees with it (the coarse patch straddled a
//      depth edge and may have matched the other surface).
//
// THE UNKNOWN D. Per pixel, inverse range scaled to pixels of the current
// level:  D = ρ · f1 · |t| / 2^level,  ρ = 1 / range,  where the right image
// sees the left pixel's ray b at π1(R·b + ρ·t) (central cameras: scaling the
// point by ρ does not move its projection). ρ = 0 is the point at infinity.
// For a rectified pair, D at the principal point is the disparity in pixels of
// that level; one level up D halves, like a flow field.
//
// Coordinates: level-l pixel (x, y) is level-0 pixel (x·2^l, y·2^l), the
// pyramid's alignment (pyrdown samples even pixels).

use crate::camera::{DistortionModel, StereoRig};
use crate::image::Image;
use crate::pyramid::{Pyramid, PyramidScratch};
use camera_geometry::{CameraModel, CameraProjection, Pixel};
use nalgebra::Vector3;

/// Floor on a patch's residual standard deviation (grey levels) when its
/// variance is estimated: a perfect match is not infinitely certain.
const NOISE_FLOOR: f32 = 1.0;
/// OpenCV's variational refinement constants: ζ regularises the data-term
/// normalisation and the system's diagonal, ε the robust penalty Ψ.
const ZETA2: f32 = 0.01;
const EPS2: f32 = 1e-6;
/// GN stops when a step moves D by less than this (pixels of the level).
const GN_MIN_STEP: f32 = 1e-3;
/// Candidates are compared on squared residuals capped at this (grey
/// levels), so on a patch straddling a depth edge the surface most of the
/// patch shows wins, not the one that best hides a few mismatched pixels.
const CANDIDATE_TRUNCATION: f32 = 10.0;
/// Largest supported patch side (stack buffer for one patch's residuals).
const MAX_PATCH: usize = 16;
/// Two candidates closer than this (pixels) are the same candidate.
const SAME_CANDIDATE: f32 = 0.25;
/// Patches over one pixel whose D differ by more than this (pixels) lie on
/// different surfaces.
const DISAGREE: f32 = 1.0;
/// Neighbours in one speckle region differ by at most this much D (pixels).
const SPECKLE_STEP: f32 = 1.0;
/// At a depth edge, the winning surface's local photometric cost must be this
/// many times lower than the other surface's, or the pixel has no depth.
const AMBIGUITY_RATIO: f32 = 2.0;
/// A finer measurement agrees with a coarse estimate (scaled to its level)
/// when their D differ by at most DISAGREE or this fraction of its D.
const FALLBACK_AGREE: f32 = 0.1;
/// `DenseDepth::source_level` where there is no depth.
const NO_SOURCE: u8 = u8::MAX;
/// Distinct match: moving D by DISTINCT_STEP (pixels) either way must raise
/// the patch's SSD by at least DISTINCT_RISE of what `min_texture` predicts
/// (n · min_texture² · step²). Noise rises by about zero.
const DISTINCT_STEP: f32 = 2.0;
const DISTINCT_RISE: f32 = 0.5;
/// Along a row, a textureless run is where |∂I/∂x| stays under this many
/// times `min_texture`; its ends are checked this many pixels inside.
const FLAT_GRADIENT: f32 = 4.0;
const FLAT_END_MARGIN: usize = 2;

#[derive(Debug, Clone)]
pub struct DenseStereoConfig {
    /// Finest pyramid level computed. The depth map has this level's size
    /// (0 = full resolution, 1 = half).
    pub finest_level: usize,
    /// Coarsest level; None chooses it from the image size as DIS does.
    pub coarsest_level: Option<usize>,
    /// Patch side in pixels (DIS: 8).
    pub patch_size: usize,
    /// Distance between neighbouring patches (DIS: 4).
    pub patch_stride: usize,
    /// Gauss-Newton iterations per patch and level, split over the sweeps
    /// (DIS: 16).
    pub grad_descent_iterations: usize,
    /// Red-black propagation sweeps per level (DIS: 2 passes).
    pub propagation_sweeps: usize,
    /// Compare patches after removing their mean (robust to exposure and
    /// vignetting differences between the eyes).
    pub mean_normalization: bool,
    /// Fixed-point iterations of the variational refinement (0 disables it).
    pub variational_iterations: usize,
    /// Refine every level, as DIS does, not only the finest. On EuRoC and
    /// synthetic scenes this changed no error measurably and took 11 % longer.
    pub refine_coarse_levels: bool,
    /// SOR iterations per fixed-point iteration.
    pub sor_iterations: usize,
    /// SOR relaxation factor.
    pub sor_omega: f32,
    /// Smoothness weight (OpenCV: 20).
    pub alpha: f32,
    /// Gradient-constancy weight (OpenCV: 10).
    pub gamma: f32,
    /// Colour-constancy weight (OpenCV: 5).
    pub delta: f32,
    /// Edge-aware smoothness: the smoothness weight is divided by
    /// 1 + |∇I|² / edge_sigma² (grey levels per pixel). 0 disables it.
    pub edge_sigma: f32,
    /// Nearest range searched (metres).
    pub min_range: f64,
    /// Farthest range searched (metres).
    pub max_range: f64,
    /// A valid patch has at least this RMS image gradient along its epipolar
    /// direction (grey levels per pixel, after mean removal). Sensor noise
    /// alone that passes it fails the distinctness test (header, step 5)
    /// down to about half this: with noise σ = 1.5 (a ZED 2i's is 1.6),
    /// depth appeared on flat surfaces from 0.25. Lower finds more faint
    /// texture: on a ZED walk 1.0 → 0.5 took coverage of the SDK's depth
    /// from 52 % to 59 % and errors over 25 % from 4.6 % to 4.2 %.
    pub min_texture: f32,
    /// A valid patch matches with at most this RMS residual (grey levels).
    pub max_residual: f32,
    /// Drop pixels whose point is hidden in the right image behind a nearer
    /// one.
    pub occlusion_check: bool,
    /// Drop connected regions of consistent depth smaller than this many
    /// pixels of the finest level (isolated mismatches); 0 keeps them.
    pub speckle_size: usize,
    /// Where the finest level has no valid estimate, use one of this many
    /// coarser levels' (header, step 6); 0 disables it. Coarser patches
    /// straddle more depth edges: on a ZED walk (finest level 1), fills at
    /// 0.3-0.5 m were too far by over 25 % for 18 % of level-2 pixels, 29 %
    /// of level 3, 36 % of level 4 and 62 % of level 5. Those levels are
    /// validated too, so they seed finer ones from valid patches only, as the
    /// finest level's depth is (on that walk this moved 0.7 % of measured
    /// pixels by over 1 %).
    pub coarse_fallback: usize,
}

impl Default for DenseStereoConfig {
    fn default() -> Self {
        Self {
            finest_level: 1,
            coarsest_level: None,
            patch_size: 8,
            patch_stride: 4,
            grad_descent_iterations: 16,
            propagation_sweeps: 2,
            mean_normalization: true,
            variational_iterations: 5,
            refine_coarse_levels: false,
            sor_iterations: 5,
            sor_omega: 1.6,
            alpha: 20.0,
            gamma: 10.0,
            delta: 5.0,
            edge_sigma: 0.0,
            min_range: 0.25,
            max_range: 30.0,
            min_texture: 0.5,
            max_residual: 12.0,
            occlusion_check: true,
            speckle_size: 100,
            coarse_fallback: 2,
        }
    }
}

/// Dense depth of the left image, at the finest computed pyramid level.
#[derive(Debug, Clone)]
pub struct DenseDepth {
    pub width: usize,
    pub height: usize,
    /// Pyramid level: pixel (x, y) here is left-image pixel (x·2^level, y·2^level).
    pub level: usize,
    /// Natural log of the range (metres) along the pixel's ray; NaN where
    /// there is no estimate. Row-major, `width` per row.
    pub log_range: Vec<f32>,
    /// Variance of `log_range`; NaN where there is no estimate.
    pub variance: Vec<f32>,
    /// Pyramid level each estimate was measured at: `level`, or a coarser
    /// one where the coarse fallback filled it; u8::MAX where there is none.
    pub source_level: Vec<u8>,
}

impl DenseDepth {
    /// log-range at map pixel (x, y), NaN if none.
    pub fn log_range_at(&self, x: usize, y: usize) -> f32 {
        self.log_range[y * self.width + x]
    }

    /// Range in metres at map pixel (x, y).
    pub fn range_at(&self, x: usize, y: usize) -> Option<f32> {
        let l = self.log_range_at(x, y);
        l.is_finite().then(|| l.exp())
    }

    /// Left-image pixel coordinates of map pixel (x, y).
    pub fn full_res_pixel(&self, x: usize, y: usize) -> (f64, f64) {
        let s = (1usize << self.level) as f64;
        (x as f64 * s, y as f64 * s)
    }

    /// Fraction of map pixels that have depth.
    pub fn coverage(&self) -> f32 {
        let n = self.log_range.iter().filter(|v| v.is_finite()).count();
        n as f32 / self.log_range.len().max(1) as f32
    }
}

// ---------------------------------------------------------------------------
// Epipolar geometry
// ---------------------------------------------------------------------------

/// How the right image sees a left-image ray as a function of D.
#[derive(Debug, Clone)]
enum EpipolarKind {
    /// Pinhole pair with equal intrinsics, no rotation, baseline along x:
    /// x1 = x + sign·D/b_z, y1 = y. Exact, and no projection per sample.
    Rectified { sign: f32 },
    /// Anything else: π1(R·b + ρ·t) through camera-geometry.
    General,
}

#[derive(Debug, Clone)]
struct Epipolar {
    kind: EpipolarKind,
    t: [f64; 3],
    cam1: CameraProjection,
}

impl Epipolar {
    fn new(rig: &StereoRig) -> Self {
        let kind = match rectified_sign(rig) {
            Some(sign) => EpipolarKind::Rectified { sign },
            None => EpipolarKind::General,
        };
        Self {
            kind,
            t: rig.t_10,
            cam1: rig.cam1_projection.clone(),
        }
    }

    /// Where the right image sees `ray` (R·b, unit) at D, in pixels of the
    /// level, and the derivative of that position with respect to D:
    /// `[x1, y1, dx1/dD, dy1/dD]`. (x, y) is the ray's left pixel in the level.
    #[inline]
    fn locate(&self, g: &LevelGeom, ray: [f32; 3], x: f32, y: f32, d: f32) -> Option<[f32; 4]> {
        match self.kind {
            EpipolarKind::Rectified { sign } => {
                if !(ray[2] > 1e-6) {
                    return None;
                }
                let n = sign / ray[2];
                Some([x + d * n, y, n, 0.0])
            }
            EpipolarKind::General => {
                let rho = d as f64 / g.s;
                let t = self.t;
                let p = Vector3::new(
                    ray[0] as f64 + rho * t[0],
                    ray[1] as f64 + rho * t[1],
                    ray[2] as f64 + rho * t[2],
                );
                let px = self.cam1.project(p)?;
                let j = self.cam1.project_jacobian(p)?;
                let du = j[(0, 0)] * t[0] + j[(0, 1)] * t[1] + j[(0, 2)] * t[2];
                let dv = j[(1, 0)] * t[0] + j[(1, 1)] * t[1] + j[(1, 2)] * t[2];
                let inv = 1.0 / g.scale;
                let k = inv / g.s;
                Some([
                    (px.x() * inv) as f32,
                    (px.y() * inv) as f32,
                    (du * k) as f32,
                    (dv * k) as f32,
                ])
            }
        }
    }
}

/// `Some(sign of t_x)` when the rig is a rectified pinhole pair.
fn rectified_sign(rig: &StereoRig) -> Option<f32> {
    let pinhole = |c: &crate::camera::CameraIntrinsics| {
        matches!(c.model, DistortionModel::None | DistortionModel::RadTan)
            && c.distortion.iter().all(|&k| k == 0.0)
    };
    let (c0, c1) = (&rig.cam0, &rig.cam1);
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);
    let identity = (0..3).all(|r| (0..3).all(|c| same(rig.r_10[r][c], (r == c) as u8 as f64)));
    let t = rig.t_10;
    let along_x = t[0] != 0.0 && t[1].abs() <= 1e-9 * t[0].abs() && t[2].abs() <= 1e-9 * t[0].abs();
    (pinhole(c0)
        && pinhole(c1)
        && same(c0.fx, c1.fx)
        && same(c0.fy, c1.fy)
        && same(c0.cx, c1.cx)
        && same(c0.cy, c1.cy)
        && identity
        && along_x)
        .then(|| t[0].signum() as f32)
}

/// Per-level geometry, fixed for the rig: rays and the patch grid.
struct LevelGeom {
    w: usize,
    h: usize,
    /// 2^level.
    scale: f64,
    /// D per unit inverse range: f1 · |t| / 2^level.
    s: f64,
    /// R·b for every pixel (NaN where the left camera has no ray).
    rays: Vec<[f32; 3]>,
    /// Patch grid: `ws` × `hs` patches, top-left corner (js·stride, is·stride).
    ws: usize,
    hs: usize,
    patch_rays: Vec<[f32; 3]>,
}

impl LevelGeom {
    fn new(rig: &StereoRig, level: usize, w0: usize, h0: usize, psz: usize, stride: usize) -> Self {
        let scale = (1usize << level) as f64;
        let (w, h) = (w0 >> level, h0 >> level);
        let r = rig.r_10;
        let ray = |x: f64, y: f64| -> [f32; 3] {
            match rig
                .cam0_projection
                .unproject(Pixel::new(x * scale, y * scale))
            {
                Some(b) => {
                    let b = b.vector();
                    let mut a = [0.0f32; 3];
                    for (i, ai) in a.iter_mut().enumerate() {
                        *ai = (r[i][0] * b.x + r[i][1] * b.y + r[i][2] * b.z) as f32;
                    }
                    a
                }
                None => [f32::NAN; 3],
            }
        };
        let mut rays = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                rays.push(ray(x as f64, y as f64));
            }
        }
        let (ws, hs) = if w >= psz && h >= psz {
            (1 + (w - psz) / stride, 1 + (h - psz) / stride)
        } else {
            (0, 0)
        };
        let c = (psz as f64 - 1.0) * 0.5;
        let mut patch_rays = Vec::with_capacity(ws * hs);
        for is in 0..hs {
            for js in 0..ws {
                patch_rays.push(ray((js * stride) as f64 + c, (is * stride) as f64 + c));
            }
        }
        Self {
            w,
            h,
            scale,
            s: rig.cam1.fx * rig.baseline_meters() / scale,
            rays,
            ws,
            hs,
            patch_rays,
        }
    }
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

/// Replicate-padded copy of a level, so patches may be sampled partly
/// outside the image.
struct Padded {
    data: Vec<f32>,
    stride: usize,
    border: usize,
    w: usize,
    h: usize,
}

impl Padded {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            stride: 0,
            border: 0,
            w: 0,
            h: 0,
        }
    }

    fn fill(&mut self, img: &Image<f32>, border: usize) {
        let (w, h) = (img.width(), img.height());
        let stride = w + 2 * border;
        self.data.resize(stride * (h + 2 * border), 0.0);
        for py in 0..h + 2 * border {
            let sy = (py as isize - border as isize).clamp(0, h as isize - 1) as usize;
            let src = img.row(sy);
            let dst = &mut self.data[py * stride..(py + 1) * stride];
            dst[..border].fill(src[0]);
            dst[border..border + w].copy_from_slice(src);
            dst[border + w..].fill(src[w - 1]);
        }
        self.stride = stride;
        self.border = border;
        self.w = w;
        self.h = h;
    }

    /// Bilinear sample at level coordinates, clamped into the padding.
    #[inline]
    fn sample(&self, x: f32, y: f32) -> f32 {
        let b = self.border as f32;
        // Positive after the shift into the padding: truncation is the floor.
        let x = x.clamp(-b + 1.0, self.w as f32 + b - 2.0) + b;
        let y = y.clamp(-b + 1.0, self.h as f32 + b - 2.0) + b;
        let (x0, y0) = (x as usize, y as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i = y0 * self.stride + x0;
        let d = &self.data;
        (1.0 - fy) * ((1.0 - fx) * d[i] + fx * d[i + 1])
            + fy * ((1.0 - fx) * d[i + self.stride] + fx * d[i + self.stride + 1])
    }
}

/// Bilinear samples of the right image at one patch's shift, for the pixels
/// that use the patch: the fractional part of the shift, and so the weights,
/// is shared. `fast` when all of them stay inside the padding; otherwise
/// samples go through the clamping `Padded::sample`.
#[derive(Clone, Copy, Default)]
struct ShiftSampler {
    shift: [f32; 2],
    /// Offset from a pixel's padded index to its sample's top-left corner.
    off: isize,
    w: [f32; 4],
    fast: bool,
}

impl ShiftSampler {
    /// For a patch at shift `shift` used by pixels x in xs, y in ys.
    fn new(img: &Padded, shift: [f32; 2], xs: (usize, usize), ys: (usize, usize)) -> Self {
        let (ox, oy) = (shift[0].floor(), shift[1].floor());
        let (fx, fy) = (shift[0] - ox, shift[1] - oy);
        let b = img.border as isize;
        let (ox, oy) = (ox as isize, oy as isize);
        let rows = (img.data.len() / img.stride) as isize;
        let fast = xs.0 as isize + b + ox >= 0
            && xs.1 as isize + b + ox + 1 < img.stride as isize
            && ys.0 as isize + b + oy >= 0
            && ys.1 as isize + b + oy + 1 < rows;
        Self {
            shift,
            off: oy * img.stride as isize + ox,
            w: [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ],
            fast,
        }
    }

    #[inline]
    fn sample(&self, img: &Padded, x: usize, y: usize) -> f32 {
        if !self.fast {
            return img.sample(x as f32 + self.shift[0], y as f32 + self.shift[1]);
        }
        let i = ((y + img.border) * img.stride + x + img.border) as isize + self.off;
        let i = i as usize;
        let d = &img.data;
        self.w[0] * d[i]
            + self.w[1] * d[i + 1]
            + self.w[2] * d[i + img.stride]
            + self.w[3] * d[i + img.stride + 1]
    }
}

/// Central difference along x, replicate border, true derivative scale.
fn diff_x(src: &[f32], w: usize, h: usize, out: &mut Vec<f32>) {
    out.resize(w * h, 0.0);
    for y in 0..h {
        let r = &src[y * w..(y + 1) * w];
        let o = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let l = r[x.saturating_sub(1)];
            let rr = r[(x + 1).min(w - 1)];
            o[x] = 0.5 * (rr - l);
        }
    }
}

/// Central difference along y, replicate border, true derivative scale.
fn diff_y(src: &[f32], w: usize, h: usize, out: &mut Vec<f32>) {
    out.resize(w * h, 0.0);
    for y in 0..h {
        let up = &src[y.saturating_sub(1) * w..][..w];
        let dn = &src[(y + 1).min(h - 1) * w..][..w];
        let o = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            o[x] = 0.5 * (dn[x] - up[x]);
        }
    }
}

fn copy_level(img: &Image<f32>, out: &mut Vec<f32>) {
    let (w, h) = (img.width(), img.height());
    out.resize(w * h, 0.0);
    for y in 0..h {
        out[y * w..(y + 1) * w].copy_from_slice(img.row(y));
    }
}

/// Bilinear sample of a dense w×h field, clamped to the image.
#[inline]
/// Gradient sums of every patch of the grid.
fn patch_sums(
    gx: &[f32],
    gy: &[f32],
    g: &LevelGeom,
    psz: usize,
    st: usize,
    out: &mut Vec<PatchSums>,
) {
    out.clear();
    for is in 0..g.hs {
        for js in 0..g.ws {
            let mut s = PatchSums::default();
            for y in is * st..is * st + psz {
                for x in js * st..js * st + psz {
                    let (gx, gy) = (gx[y * g.w + x], gy[y * g.w + x]);
                    s.xx += gx * gx;
                    s.xy += gx * gy;
                    s.yy += gy * gy;
                    s.x += gx;
                    s.y += gy;
                }
            }
            out.push(s);
        }
    }
}

fn sample_field(f: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    (1.0 - fy) * ((1.0 - fx) * f[y0 * w + x0] + fx * f[y0 * w + x1])
        + fy * ((1.0 - fx) * f[y1 * w + x0] + fx * f[y1 * w + x1])
}

// ---------------------------------------------------------------------------
// Patch inverse search
// ---------------------------------------------------------------------------

/// Sums over a patch of the left image's gradients (the structure tensor and,
/// for mean normalisation, the plain sums).
#[derive(Clone, Copy, Default)]
struct PatchSums {
    xx: f32,
    xy: f32,
    yy: f32,
    x: f32,
    y: f32,
}

/// One patch against the right image at a bilinear position: the right
/// image's samples from `base` (row stride `stride`, bilinear `weights`),
/// the left image, its gradients from `off` (row stride `w`).
struct PatchRows<'a> {
    i1: &'a [f32],
    base: usize,
    stride: usize,
    weights: [f32; 4],
    i0: &'a [f32],
    gx: &'a [f32],
    gy: &'a [f32],
    off: usize,
    w: usize,
    psz: usize,
}

impl PatchRows<'_> {
    /// Σ r, Σ r², Σ r·gx and Σ r·gy over the patch, r = I1 − I0, and with
    /// COST each r in `res` (row-major, psz² of them). AVX2+FMA (x86_64, runtime check)
    /// or NEON (aarch64) for patch sides that fit their vectors; the sums'
    /// order differs from the scalar loop's by rounding only.
    #[inline(always)]
    fn residuals<const COST: bool>(&self, res: &mut [f32]) -> [f32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if self.psz % 8 == 0
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
            {
                // SAFETY: the CPU has AVX2 and FMA; the rows are sliced to
                // their lengths before any load.
                return unsafe { self.residuals_avx2::<COST>(res) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.psz % 4 == 0 {
                // SAFETY: NEON is part of aarch64; rows are sliced first.
                return unsafe { self.residuals_neon::<COST>(res) };
            }
        }
        self.residuals_scalar::<COST>(res)
    }

    /// One row's slices: right image rows r0 and r1 (psz + 1 samples), and
    /// the left image and its gradients (psz).
    #[inline(always)]
    fn row(&self, row: usize) -> [&[f32]; 5] {
        let (psz, o) = (self.psz, self.off + row * self.w);
        [
            &self.i1[self.base + row * self.stride..][..psz + 1],
            &self.i1[self.base + (row + 1) * self.stride..][..psz + 1],
            &self.i0[o..o + psz],
            &self.gx[o..o + psz],
            &self.gy[o..o + psz],
        ]
    }

    fn residuals_scalar<const COST: bool>(&self, res: &mut [f32]) -> [f32; 4] {
        let [w00, w10, w01, w11] = self.weights;
        let psz = self.psz;
        let (mut sr, mut srr, mut srx, mut sry) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for row in 0..psz {
            let [r0, r1, t, gxr, gyr] = self.row(row);
            let res_row: &mut [f32] = if COST {
                &mut res[row * psz..(row + 1) * psz]
            } else {
                &mut []
            };
            for c in 0..psz {
                let v = w00 * r0[c] + w10 * r0[c + 1] + w01 * r1[c] + w11 * r1[c + 1];
                let r = v - t[c];
                if COST {
                    res_row[c] = r;
                }
                sr += r;
                srr += r * r;
                srx += r * gxr[c];
                sry += r * gyr[c];
            }
        }
        [sr, srr, srx, sry]
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn residuals_avx2<const COST: bool>(&self, res: &mut [f32]) -> [f32; 4] {
        use std::arch::x86_64::*;
        let psz = self.psz;
        let [w00, w10, w01, w11] = self.weights.map(|v| _mm256_set1_ps(v));
        let (mut ar, mut arr, mut arx, mut ary) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        for row in 0..psz {
            let [r0, r1, t, gxr, gyr] = self.row(row);
            let out = if COST {
                res[row * psz..(row + 1) * psz].as_mut_ptr()
            } else {
                std::ptr::null_mut()
            };
            for c in (0..psz).step_by(8) {
                // r0, r1 hold psz + 1 samples, the others psz: c + 8 <= psz.
                let v = _mm256_mul_ps(w00, _mm256_loadu_ps(r0.as_ptr().add(c)));
                let v = _mm256_fmadd_ps(w10, _mm256_loadu_ps(r0.as_ptr().add(c + 1)), v);
                let v = _mm256_fmadd_ps(w01, _mm256_loadu_ps(r1.as_ptr().add(c)), v);
                let v = _mm256_fmadd_ps(w11, _mm256_loadu_ps(r1.as_ptr().add(c + 1)), v);
                let r = _mm256_sub_ps(v, _mm256_loadu_ps(t.as_ptr().add(c)));
                if COST {
                    _mm256_storeu_ps(out.add(c), r);
                }
                ar = _mm256_add_ps(ar, r);
                arr = _mm256_fmadd_ps(r, r, arr);
                arx = _mm256_fmadd_ps(r, _mm256_loadu_ps(gxr.as_ptr().add(c)), arx);
                ary = _mm256_fmadd_ps(r, _mm256_loadu_ps(gyr.as_ptr().add(c)), ary);
            }
        }
        [hsum256(ar), hsum256(arr), hsum256(arx), hsum256(ary)]
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn residuals_neon<const COST: bool>(&self, res: &mut [f32]) -> [f32; 4] {
        use std::arch::aarch64::*;
        let psz = self.psz;
        let [w00, w10, w01, w11] = self.weights.map(|v| vdupq_n_f32(v));
        let (mut ar, mut arr, mut arx, mut ary) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        for row in 0..psz {
            let [r0, r1, t, gxr, gyr] = self.row(row);
            let out = if COST {
                res[row * psz..(row + 1) * psz].as_mut_ptr()
            } else {
                std::ptr::null_mut()
            };
            for c in (0..psz).step_by(4) {
                let v = vmulq_f32(w00, vld1q_f32(r0.as_ptr().add(c)));
                let v = vfmaq_f32(v, w10, vld1q_f32(r0.as_ptr().add(c + 1)));
                let v = vfmaq_f32(v, w01, vld1q_f32(r1.as_ptr().add(c)));
                let v = vfmaq_f32(v, w11, vld1q_f32(r1.as_ptr().add(c + 1)));
                let r = vsubq_f32(v, vld1q_f32(t.as_ptr().add(c)));
                if COST {
                    vst1q_f32(out.add(c), r);
                }
                ar = vaddq_f32(ar, r);
                arr = vfmaq_f32(arr, r, r);
                arx = vfmaq_f32(arx, r, vld1q_f32(gxr.as_ptr().add(c)));
                ary = vfmaq_f32(ary, r, vld1q_f32(gyr.as_ptr().add(c)));
            }
        }
        [
            vaddvq_f32(ar),
            vaddvq_f32(arr),
            vaddvq_f32(arx),
            vaddvq_f32(ary),
        ]
    }
}

/// Horizontal sum of 8 f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehdup_ps(s));
    let s = _mm_add_ss(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(s)
}

/// One patch's estimate.
#[derive(Clone, Copy)]
struct PatchState {
    d: f32,
    /// SSD at `d` (after mean removal if configured).
    ssd: f32,
    /// Truncated SSD at `d`, for comparing candidates.
    cost: f32,
    /// Gauss-Newton Hessian Σ G_d² at `d` (after mean removal).
    hess: f32,
    /// Where the patch lands in the right image, minus where it is in the left.
    shift: [f32; 2],
    /// The patch lands fully inside the right image.
    inside: bool,
    /// `locate` succeeded (the ray reaches the right image).
    ok: bool,
}

impl PatchState {
    const FAILED: Self = Self {
        d: 0.0,
        ssd: f32::INFINITY,
        cost: f32::INFINITY,
        hess: 0.0,
        shift: [0.0; 2],
        inside: false,
        ok: false,
    };
}

/// SSD, GN gradient and Hessian of one patch at one D.
struct Eval {
    ssd: f32,
    /// Truncated SSD; only computed for candidates (NaN otherwise).
    cost: f32,
    grad: f32,
    hess: f32,
    shift: [f32; 2],
    inside: bool,
}

/// Everything a level's patch search and densification read.
struct LevelView<'a> {
    g: &'a LevelGeom,
    epi: &'a Epipolar,
    cfg: &'a DenseStereoConfig,
    i0: &'a [f32],
    gx: &'a [f32],
    gy: &'a [f32],
    i1: &'a Padded,
    sums: &'a [PatchSums],
    dmin: f32,
    dmax: f32,
}

impl LevelView<'_> {
    #[inline]
    fn origin(&self, k: usize) -> (usize, usize) {
        let st = self.cfg.patch_stride;
        ((k % self.g.ws) * st, (k / self.g.ws) * st)
    }

    #[inline]
    fn centre(&self, k: usize) -> (f32, f32) {
        let (j, i) = self.origin(k);
        let c = (self.cfg.patch_size as f32 - 1.0) * 0.5;
        (j as f32 + c, i as f32 + c)
    }

    /// The patch at D. `COST` also computes the truncated SSD that
    /// candidates are compared on; Gauss-Newton steps do not need it.
    fn eval<const COST: bool>(&self, k: usize, d: f32) -> Option<Eval> {
        let psz = self.cfg.patch_size;
        let n = (psz * psz) as f32;
        let (j, i) = self.origin(k);
        let (cx, cy) = self.centre(k);
        let loc = self.epi.locate(self.g, self.g.patch_rays[k], cx, cy, d)?;
        let shift = [loc[0] - cx, loc[1] - cy];
        let (w, h) = (self.g.w, self.g.h);
        let lim = psz as f32 - 1.0;
        let tlx_raw = j as f32 + shift[0];
        let tly_raw = i as f32 + shift[1];
        // Every sample on the right image (a sample past the last row or
        // column reads the replicated border with a weight below one).
        let inside = tlx_raw >= 0.0
            && tly_raw >= 0.0
            && tlx_raw <= (w - psz) as f32
            && tly_raw <= (h - psz) as f32;
        // As DIS: keep at least one row and column of the patch on the
        // image. In padded coordinates the corner is never negative, so
        // truncation is the floor.
        let b = self.i1.border as f32;
        let tlx = tlx_raw.clamp(-lim, (w - 1) as f32) + b;
        let tly = tly_raw.clamp(-lim, (h - 1) as f32) + b;
        let (x0, y0) = (tlx as usize, tly as usize);
        let (fx, fy) = (tlx - x0 as f32, tly - y0 as f32);
        let (w00, w10, w01, w11) = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        );
        let stride = self.i1.stride;
        let base = y0 * stride + x0;

        // Residuals are kept only to compare candidates (COST).
        let mut res = [0.0f32; MAX_PATCH * MAX_PATCH];
        let res: &mut [f32] = if COST { &mut res } else { &mut [] };
        let rows = PatchRows {
            i1: &self.i1.data,
            base,
            stride,
            weights: [w00, w10, w01, w11],
            i0: self.i0,
            gx: self.gx,
            gy: self.gy,
            off: i * w + j,
            w,
            psz,
        };
        let [sr, srr, srx, sry] = rows.residuals::<COST>(res);
        let mean = if self.cfg.mean_normalization {
            sr / n
        } else {
            0.0
        };
        let cost = if COST {
            let cap = CANDIDATE_TRUNCATION * CANDIDATE_TRUNCATION;
            res[..psz * psz]
                .iter()
                .map(|r| ((r - mean) * (r - mean)).min(cap))
                .sum()
        } else {
            f32::NAN
        };
        let s = &self.sums[k];
        let (ex, ey) = (loc[2], loc[3]);
        let mut hess = ex * ex * s.xx + 2.0 * ex * ey * s.xy + ey * ey * s.yy;
        let (ssd, grad) = if self.cfg.mean_normalization {
            let gd = ex * s.x + ey * s.y;
            hess -= gd * gd / n;
            (
                srr - sr * sr / n,
                ex * (srx - sr * s.x / n) + ey * (sry - sr * s.y / n),
            )
        } else {
            (srr, ex * srx + ey * sry)
        };
        Some(Eval {
            ssd,
            cost,
            grad,
            hess,
            shift,
            inside,
        })
    }

    fn state(&self, d: f32, e: &Eval) -> PatchState {
        PatchState {
            d,
            ssd: e.ssd,
            cost: e.cost,
            hess: e.hess,
            shift: e.shift,
            inside: e.inside,
            ok: true,
        }
    }

    /// Inverse-compositional Gauss-Newton from D `start` (evaluated as
    /// `e`), accepting only steps that lower the SSD. Moving more than a
    /// patch side from the start discards the result, as DIS does. Also
    /// whether it converged (stopped before the iteration limit).
    fn refine(&self, k: usize, start: f32, mut e: Eval, iterations: usize) -> (PatchState, bool) {
        let first = self.state(start, &e);
        let mut d = start;
        let mut converged = false;
        for _ in 0..iterations {
            if !(e.hess > 1e-6) {
                converged = true;
                break;
            }
            let next = (d - e.grad / e.hess).clamp(self.dmin, self.dmax);
            if (next - d).abs() < GN_MIN_STEP {
                converged = true;
                break;
            }
            match self.eval::<false>(k, next) {
                Some(n) if n.ssd < e.ssd => {
                    d = next;
                    e = n;
                }
                _ => {
                    converged = true;
                    break;
                }
            }
        }
        if (d - start).abs() > self.cfg.patch_size as f32 {
            return (first, false);
        }
        (self.state(d, &e), converged)
    }

    /// The best of the candidate D values by truncated SSD, with its
    /// evaluation. Candidates within SAME_CANDIDATE of one already tried are
    /// skipped.
    fn best_candidate(&self, k: usize, candidates: &[f32]) -> Option<(f32, Eval)> {
        let mut tried = [0.0f32; 8];
        let mut n_tried = 0;
        let mut best: Option<(f32, Eval)> = None;
        for &d in candidates {
            if !d.is_finite()
                || tried[..n_tried]
                    .iter()
                    .any(|t| (d - t).abs() < SAME_CANDIDATE)
            {
                continue;
            }
            if n_tried < tried.len() {
                tried[n_tried] = d;
                n_tried += 1;
            }
            if let Some(e) = self.eval::<true>(k, d) {
                if best.as_ref().is_none_or(|(_, b)| e.cost < b.cost) {
                    best = Some((d, e));
                }
            }
        }
        best
    }

    /// Exhaustive 1-D search over [dmin, dmax] in steps of one pixel, for the
    /// coarsest level. Positions where the patch leaves the right image are
    /// not candidates.
    fn search(&self, k: usize) -> PatchState {
        let mut best = PatchState::FAILED;
        let n = ((self.dmax - self.dmin).ceil() as usize).max(1);
        for s in 0..=n {
            let d = (self.dmin + s as f32).min(self.dmax);
            if let Some(e) = self.eval::<true>(k, d) {
                if e.inside && e.cost < best.cost {
                    best = self.state(d, &e);
                }
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// The matcher
// ---------------------------------------------------------------------------

/// Per-frame working buffers, reused across frames.
#[derive(Default)]
struct Buffers {
    i0: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    sums: Vec<PatchSums>,
    patches: Vec<PatchState>,
    // Propagation: per patch, whether its D moved in its latest evaluation
    // and whether that evaluation converged.
    moved: Vec<bool>,
    settled: Vec<bool>,
    /// Dense D of the current level.
    d: Vec<f32>,
    /// The coarser level's D, upsampled.
    d_init: Vec<f32>,
    // Densification: per patch, its inverse variance, whether it is valid,
    // and its right-image sampler.
    precision: Vec<f32>,
    valid_patch: Vec<bool>,
    samplers: Vec<ShiftSampler>,
    // Finest-level statistics from the valid patches.
    valid: Vec<bool>,
    var: Vec<f32>,
    // Variational refinement: the solved pixels, the linearised warp's mean
    // and difference images, and the solver.
    act: Vec<u32>,
    avg: Vec<f32>,
    iz: Vec<f32>,
    rb: RedBlack,
    // Occlusion test: nearest D per right-image pixel, and each left pixel's
    // right-image pixel.
    zbuf: Vec<f32>,
    target: Vec<usize>,
    // Coarse fallback: each coarser level's validated depth (indexed by
    // level), the integral image of cells a finer measurement disagrees
    // with, and the level each finest pixel's estimate comes from.
    coarse: Vec<LevelDepth>,
    disagree: Vec<u32>,
    source: Vec<u8>,
    speckle: SpeckleScratch,
}

/// remove_speckles' working memory: region labels, the flood fill's stack
/// and the current region.
#[derive(Default)]
struct SpeckleScratch {
    label: Vec<u32>,
    stack: Vec<usize>,
    region: Vec<usize>,
}

/// The variational refinement's unknowns and coefficients, split by colour
/// for red-black SOR: colour c holds the pixels with (x + y) % 2 == c,
/// pixel x of row y at column q = x / 2 of that colour's row y. Each colour
/// is an (h + 2) × (hw + 2) grid, hw = ⌈w / 2⌉, with a zero border, the two
/// one after the other. With s = (y + c) % 2, a pixel's right and left
/// neighbours are the other colour's columns q + s and q + s − 1 of the
/// same row, the ones above and below column q of rows y ∓ 1. Every update
/// is then a straight row loop the compiler vectorizes. A pixel not solved
/// (or past its row's end) has zero coefficients: it stays put and
/// constrains nothing.
#[derive(Default)]
struct RedBlack {
    hw: usize,
    stride: usize,
    /// Elements per colour.
    size: usize,
    /// D, its increment, and the weights of each pixel's right and lower
    /// edges.
    d: Vec<f32>,
    dd: Vec<f32>,
    wr: Vec<f32>,
    wd: Vec<f32>,
    /// 1/A and the right-hand side's constant part of each pixel's
    /// equation A·dD = c + Σ w·dD_neighbour.
    inv_a: Vec<f32>,
    c: Vec<f32>,
    /// 1 where the pixel is solved; where it and its right (lower)
    /// neighbour are; the smoothness weight's edge factor.
    solved: Vec<f32>,
    right: Vec<f32>,
    down: Vec<f32>,
    edge: Vec<f32>,
    /// The data term, fixed over the fixed-point iterations: I1(warp) − I0
    /// and its gradient; the mean image's gradient along the epipolar
    /// direction (id) and that gradient's derivatives; the normalisations.
    iz: Vec<f32>,
    id: Vec<f32>,
    ixd: Vec<f32>,
    iyd: Vec<f32>,
    ixz: Vec<f32>,
    iyz: Vec<f32>,
    inv_nd: Vec<f32>,
    inv_nx: Vec<f32>,
    inv_ny: Vec<f32>,
}

/// `n` elements of `v` from `start`.
#[inline(always)]
fn slice_row(v: &[f32], start: usize, n: usize) -> &[f32] {
    &v[start..start + n]
}

/// The constants of one refinement.
#[derive(Clone, Copy)]
struct Solve {
    h: usize,
    iterations: usize,
    sor_iterations: usize,
    alpha2: f32,
    gamma2: f32,
    delta2: f32,
    omega: f32,
}

impl RedBlack {
    fn reset(&mut self, w: usize, h: usize) {
        self.hw = w.div_ceil(2);
        self.stride = self.hw + 2;
        self.size = self.stride * (h + 2);
        let n = 2 * self.size;
        for v in [
            &mut self.d,
            &mut self.dd,
            &mut self.wr,
            &mut self.wd,
            &mut self.inv_a,
            &mut self.c,
            &mut self.solved,
            &mut self.right,
            &mut self.down,
            &mut self.edge,
            &mut self.iz,
            &mut self.id,
            &mut self.ixd,
            &mut self.iyd,
            &mut self.ixz,
            &mut self.iyz,
            &mut self.inv_nd,
            &mut self.inv_nx,
            &mut self.inv_ny,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
    }

    /// Index of pixel (x, y).
    fn index(&self, x: usize, y: usize) -> usize {
        (x + y) % 2 * self.size + (y + 1) * self.stride + x / 2 + 1
    }

    /// Start of row y of colour c, and of the other colour's row y.
    fn rows(&self, c: usize, y: usize) -> (usize, usize) {
        let r = (y + 1) * self.stride + 1;
        (c * self.size + r, (1 - c) * self.size + r)
    }

    /// The fixed-point iterations, with AVX2 where the CPU has it (no FMA:
    /// the results match the plain build's bit for bit).
    fn solve(&mut self, k: Solve) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: the CPU has AVX2.
                unsafe { self.solve_avx2(k) };
                return;
            }
        }
        self.solve_rows(k);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn solve_avx2(&mut self, k: Solve) {
        self.solve_rows(k);
    }

    #[inline(always)]
    fn solve_rows(&mut self, k: Solve) {
        let (hw, stride) = (self.hw, self.stride);
        for _ in 0..k.iterations {
            // Smoothness weights from the forward differences of D + dD; each
            // pixel's weight belongs to its right and lower edges, and only
            // edges between two solved pixels have one.
            for c in 0..2 {
                for y in 0..k.h {
                    let (me, ot) = self.rows(c, y);
                    let s = (y + c) % 2;
                    let (d, dd) = (&self.d, &self.dd);
                    let (ud, udd) = (&d[me..me + hw], &dd[me..me + hw]);
                    let (rd, rdd) = (&d[ot + s..ot + s + hw], &dd[ot + s..ot + s + hw]);
                    let (bd, bdd) = (&d[ot + stride..][..hw], &dd[ot + stride..][..hw]);
                    let (right, down) = (&self.right[me..me + hw], &self.down[me..me + hw]);
                    let edge = &self.edge[me..me + hw];
                    let wr = &mut self.wr[me..me + hw];
                    let wd = &mut self.wd[me..me + hw];
                    for q in 0..hw {
                        let u = ud[q] + udd[q];
                        let ux = (rd[q] + rdd[q] - u) * right[q];
                        let uy = (bd[q] + bdd[q] - u) * down[q];
                        let ws = k.alpha2 / (ux * ux + uy * uy + EPS2).sqrt() * edge[q];
                        wr[q] = ws * right[q];
                        wd[q] = ws * down[q];
                    }
                }
            }
            // Data terms linearised around D + dD, and the parts of each
            // pixel's equation that stay fixed during SOR.
            for c in 0..2 {
                for y in 0..k.h {
                    let (me, ot) = self.rows(c, y);
                    let s = (y + c) % 2;
                    let (dd, d0) = (slice_row(&self.dd, me, hw), slice_row(&self.d, me, hw));
                    let (iz, id, ixd, iyd) = (
                        slice_row(&self.iz, me, hw),
                        slice_row(&self.id, me, hw),
                        slice_row(&self.ixd, me, hw),
                        slice_row(&self.iyd, me, hw),
                    );
                    let (ixz, iyz) = (slice_row(&self.ixz, me, hw), slice_row(&self.iyz, me, hw));
                    let (inv_nd, inv_nx, inv_ny) = (
                        slice_row(&self.inv_nd, me, hw),
                        slice_row(&self.inv_nx, me, hw),
                        slice_row(&self.inv_ny, me, hw),
                    );
                    let (wr, wd, solved) = (
                        slice_row(&self.wr, me, hw),
                        slice_row(&self.wd, me, hw),
                        slice_row(&self.solved, me, hw),
                    );
                    let wl = &self.wr[ot + s - 1..ot + s - 1 + hw];
                    let wu = &self.wd[ot - stride..ot - stride + hw];
                    let rd = &self.d[ot + s..ot + s + hw];
                    let ld = &self.d[ot + s - 1..ot + s - 1 + hw];
                    let bd = &self.d[ot + stride..ot + stride + hw];
                    let td = &self.d[ot - stride..ot - stride + hw];
                    let inv_a = &mut self.inv_a[me..me + hw];
                    let cst = &mut self.c[me..me + hw];
                    for q in 0..hw {
                        let ddq = dd[q];
                        let z = iz[q] + id[q] * ddq;
                        let wc = k.delta2 * inv_nd[q] / (z * z * inv_nd[q] + EPS2).sqrt();
                        let mut a = wc * id[q] * id[q] + ZETA2;
                        let mut rhs = -wc * iz[q] * id[q];

                        let zx = ixz[q] + ixd[q] * ddq;
                        let zy = iyz[q] + iyd[q] * ddq;
                        let wg =
                            k.gamma2 / (zx * zx * inv_nx[q] + zy * zy * inv_ny[q] + EPS2).sqrt();
                        a += wg * (ixd[q] * ixd[q] * inv_nx[q] + iyd[q] * iyd[q] * inv_ny[q]);
                        rhs -= wg * (ixd[q] * ixz[q] * inv_nx[q] + iyd[q] * iyz[q] * inv_ny[q]);

                        let d = d0[q];
                        a += wr[q] + wl[q] + wd[q] + wu[q];
                        rhs += wr[q] * (rd[q] - d)
                            + wl[q] * (ld[q] - d)
                            + wd[q] * (bd[q] - d)
                            + wu[q] * (td[q] - d);
                        inv_a[q] = solved[q] * (1.0 / a);
                        cst[q] = solved[q] * rhs;
                    }
                }
            }
            // Red-black SOR: each colour's update reads only the other's.
            for _ in 0..k.sor_iterations {
                for c in 0..2 {
                    let size = self.size;
                    let (lo, hi) = self.dd.split_at_mut(size);
                    let (mine, other): (&mut [f32], &[f32]) =
                        if c == 0 { (lo, hi) } else { (hi, lo) };
                    for y in 0..k.h {
                        // As rows(): row y within each colour, and overall.
                        let local = (y + 1) * stride + 1;
                        let olocal = local;
                        let (me, ot) = (c * size + local, (1 - c) * size + local);
                        let s = (y + c) % 2;
                        let cst = &self.c[me..me + hw];
                        let inv_a = &self.inv_a[me..me + hw];
                        let (wr, wd) = (&self.wr[me..me + hw], &self.wd[me..me + hw]);
                        let wl = &self.wr[ot + s - 1..ot + s - 1 + hw];
                        let wu = &self.wd[ot - stride..ot - stride + hw];
                        let rdd = &other[olocal + s..olocal + s + hw];
                        let ldd = &other[olocal + s - 1..olocal + s - 1 + hw];
                        let bdd = &other[olocal + stride..olocal + stride + hw];
                        let tdd = &other[olocal - stride..olocal - stride + hw];
                        let dd = &mut mine[local..local + hw];
                        for q in 0..hw {
                            let sum = cst[q]
                                + wr[q] * rdd[q]
                                + wl[q] * ldd[q]
                                + wd[q] * bdd[q]
                                + wu[q] * tdd[q];
                            dd[q] += k.omega * (sum * inv_a[q] - dd[q]);
                        }
                    }
                }
            }
        }
    }
}

/// A coarser level's depth after validation, for the coarse fallback.
#[derive(Default)]
struct LevelDepth {
    w: usize,
    h: usize,
    d: Vec<f32>,
    var: Vec<f32>,
    valid: Vec<bool>,
}

/// Dense stereo depth for a calibrated rig.
pub struct DenseStereo {
    config: DenseStereoConfig,
    epi: Epipolar,
    width: usize,
    height: usize,
    coarsest: usize,
    /// Indexed by level; levels finer than `finest_level` are empty.
    levels: Vec<Option<LevelGeom>>,
    pyr0: Pyramid,
    pyr1: Pyramid,
    pyr_scratch: PyramidScratch,
    i1: Padded,
    buf: Buffers,
}

impl DenseStereo {
    /// The left camera's resolution sets the image size.
    pub fn new(rig: StereoRig, config: DenseStereoConfig) -> Self {
        let [w, h] = rig.cam0.resolution;
        let psz = config.patch_size;
        assert!(
            (2..=MAX_PATCH).contains(&psz) && config.patch_stride >= 1,
            "bad patch size or stride"
        );
        assert!(config.min_range > 0.0 && config.max_range > config.min_range);
        assert!(rig.baseline_meters() > 0.0, "stereo rig needs a baseline");
        // DIS's choice: motions up to a quarter of the image, and the coarsest
        // level still holds a patch.
        let auto = ((w.max(h) as f64 / (4.0 * psz as f64)).log2() + 0.5) as usize;
        let fits = ((w.min(h) as f64 / psz as f64).log2()).floor().max(0.0) as usize;
        let coarsest = config
            .coarsest_level
            .unwrap_or(auto)
            .min(fits)
            .max(config.finest_level);
        assert!(
            (w >> coarsest) >= psz && (h >> coarsest) >= psz,
            "image too small for the finest level and patch size"
        );
        let levels = (0..=coarsest)
            .map(|l| {
                (l >= config.finest_level)
                    .then(|| LevelGeom::new(&rig, l, w, h, psz, config.patch_stride))
            })
            .collect();
        let blank = Image::<u8>::new(w, h);
        Self {
            epi: Epipolar::new(&rig),
            width: w,
            height: h,
            coarsest,
            levels,
            pyr0: Pyramid::build(&blank, coarsest + 1, 1.0),
            pyr1: Pyramid::build(&blank, coarsest + 1, 1.0),
            pyr_scratch: PyramidScratch::new(w, h, 1.0),
            i1: Padded::new(),
            buf: Buffers::default(),
            config,
        }
    }

    pub fn config(&self) -> &DenseStereoConfig {
        &self.config
    }

    /// Coarsest pyramid level used.
    pub fn coarsest_level(&self) -> usize {
        self.coarsest
    }

    /// True when the rig is a rectified pinhole pair (no projection per sample).
    pub fn is_rectified(&self) -> bool {
        matches!(self.epi.kind, EpipolarKind::Rectified { .. })
    }

    /// Depth of the left image from a left/right pair (mono8, raw or
    /// rectified as the rig describes them).
    pub fn compute(&mut self, left: &Image<u8>, right: &Image<u8>) -> DenseDepth {
        let mut out = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        self.compute_into(left, right, &mut out);
        out
    }

    /// As `compute`, reusing `out`'s buffers.
    pub fn compute_into(&mut self, left: &Image<u8>, right: &Image<u8>, out: &mut DenseDepth) {
        assert_eq!((left.width(), left.height()), (self.width, self.height));
        assert_eq!((right.width(), right.height()), (self.width, self.height));
        let n_levels = self.coarsest + 1;
        self.pyr0.build_reuse(left, n_levels, &mut self.pyr_scratch);
        self.pyr1
            .build_reuse(right, n_levels, &mut self.pyr_scratch);

        let finest = self.config.finest_level;
        // Levels the coarse fallback takes depth from: finest+1..=fallback.
        let fallback = self.last_validated_level();
        for level in (finest..=self.coarsest).rev() {
            self.prepare_level(level);
            if level == self.coarsest {
                self.search_coarsest(level);
            } else {
                self.upsample_init(level);
            }
            self.propagate(level);
            let validate = level <= fallback;
            self.densify(level, validate);
            self.refine_variational(level);
            if validate {
                self.mask(level);
            }
            if level > finest && level <= fallback {
                self.keep_level(level);
            }
        }
        self.coarse_fallback();
        self.reject_contradicted_fills();
        self.finish(out);
    }

    /// The coarsest level whose patches are validated: the finest, and the
    /// coarse fallback's.
    fn last_validated_level(&self) -> usize {
        (self.config.finest_level + self.config.coarse_fallback).min(self.coarsest)
    }

    // The per-level stages run once per level; kept out of line so profiles
    // attribute time to them.
    #[inline(never)]
    fn prepare_level(&mut self, level: usize) {
        let g = self.levels[level].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let (psz, st) = (self.config.patch_size, self.config.patch_stride);
        let b = &mut self.buf;
        copy_level(self.pyr0.level(level), &mut b.i0);
        diff_x(&b.i0, w, h, &mut b.gx);
        diff_y(&b.i0, w, h, &mut b.gy);
        self.i1.fill(self.pyr1.level(level), psz + 2);
        patch_sums(&b.gx, &b.gy, g, psz, st, &mut b.sums);
    }

    fn view(&self, level: usize) -> LevelView<'_> {
        let g = self.levels[level].as_ref().unwrap();
        LevelView {
            g,
            epi: &self.epi,
            cfg: &self.config,
            i0: &self.buf.i0,
            gx: &self.buf.gx,
            gy: &self.buf.gy,
            i1: &self.i1,
            sums: &self.buf.sums,
            dmin: (g.s / self.config.max_range) as f32,
            dmax: (g.s / self.config.min_range) as f32,
        }
    }

    #[inline(never)]
    fn search_coarsest(&mut self, level: usize) {
        let mut patches = std::mem::take(&mut self.buf.patches);
        let v = self.view(level);
        let n = v.g.ws * v.g.hs;
        patches.clear();
        patches.extend((0..n).map(|k| {
            let mut p = v.search(k);
            if !p.ok {
                // Nothing inside the right image: start from far away.
                p.d = v.dmin;
            }
            p
        }));
        self.buf.patches = patches;
    }

    /// Start every patch from the coarser level's D at its centre.
    #[inline(never)]
    fn upsample_init(&mut self, level: usize) {
        let g = self.levels[level].as_ref().unwrap();
        let coarse = self.levels[level + 1].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let b = &mut self.buf;
        b.d_init.resize(w * h, 0.0);
        for y in 0..h {
            for x in 0..w {
                b.d_init[y * w + x] =
                    2.0 * sample_field(&b.d, coarse.w, coarse.h, x as f32 * 0.5, y as f32 * 0.5);
            }
        }
        let mut patches = std::mem::take(&mut self.buf.patches);
        let v = self.view(level);
        let n = v.g.ws * v.g.hs;
        patches.clear();
        patches.extend((0..n).map(|k| {
            let (cx, cy) = v.centre(k);
            let mut p = PatchState::FAILED;
            p.d = sample_field(&self.buf.d_init, w, h, cx, cy);
            p
        }));
        self.buf.patches = patches;
    }

    /// Red-black sweeps: each patch takes the best of its own and its four
    /// neighbours' estimates, then runs Gauss-Newton from there. After the
    /// first sweep, a patch whose Gauss-Newton converged and whose
    /// neighbours have not moved since is skipped: it would start from its
    /// own converged estimate and stop there.
    #[inline(never)]
    fn propagate(&mut self, level: usize) {
        let sweeps = self.config.propagation_sweeps.max(1);
        let iters = (self.config.grad_descent_iterations / sweeps).max(1);
        let mut patches = std::mem::take(&mut self.buf.patches);
        let mut moved = std::mem::take(&mut self.buf.moved);
        let mut settled = std::mem::take(&mut self.buf.settled);
        let v = self.view(level);
        let (ws, hs) = (v.g.ws, v.g.hs);
        moved.clear();
        moved.resize(ws * hs, true);
        settled.clear();
        settled.resize(ws * hs, false);
        for _ in 0..sweeps {
            for colour in 0..2 {
                for is in 0..hs {
                    for js in ((is + colour) % 2..ws).step_by(2) {
                        let k = is * ws + js;
                        let neighbours_moved = (js > 0 && moved[k - 1])
                            || (js + 1 < ws && moved[k + 1])
                            || (is > 0 && moved[k - ws])
                            || (is + 1 < hs && moved[k + ws]);
                        if settled[k] && !neighbours_moved {
                            moved[k] = false;
                            continue;
                        }
                        let own = patches[k].d;
                        let mut cand = [own, f32::NAN, f32::NAN, f32::NAN, f32::NAN];
                        if js > 0 {
                            cand[1] = patches[k - 1].d;
                        }
                        if js + 1 < ws {
                            cand[2] = patches[k + 1].d;
                        }
                        if is > 0 {
                            cand[3] = patches[k - ws].d;
                        }
                        if is + 1 < hs {
                            cand[4] = patches[k + ws].d;
                        }
                        let (p, converged) = match v.best_candidate(k, &cand) {
                            Some((d, e)) => v.refine(k, d, e, iters),
                            None => (
                                PatchState {
                                    d: own,
                                    ..PatchState::FAILED
                                },
                                false,
                            ),
                        };
                        moved[k] = !((p.d - own).abs() < GN_MIN_STEP);
                        settled[k] = converged;
                        patches[k] = p;
                    }
                }
            }
        }
        self.buf.patches = patches;
        self.buf.moved = moved;
        self.buf.settled = settled;
    }

    /// Dense D from the patches. With `validate` (the finest level, and every
    /// level for the coarse fallback), also which pixels a valid patch covers
    /// and their variance.
    #[inline(never)]
    fn densify(&mut self, level: usize, validate: bool) {
        let mut d = std::mem::take(&mut self.buf.d);
        let mut valid = std::mem::take(&mut self.buf.valid);
        let mut var = std::mem::take(&mut self.buf.var);
        let mut precision = std::mem::take(&mut self.buf.precision);
        let mut valid_patch = std::mem::take(&mut self.buf.valid_patch);
        let mut samplers = std::mem::take(&mut self.buf.samplers);
        let v = self.view(level);
        let (w, h) = (v.g.w, v.g.h);
        let (ws, hs) = (v.g.ws, v.g.hs);
        let psz = self.config.patch_size;
        let st = self.config.patch_stride;
        let n = (psz * psz) as f32;
        let patches = &self.buf.patches;
        let has_init = level < self.coarsest;

        let min_hess = self.config.min_texture * self.config.min_texture * n;
        let max_ssd = self.config.max_residual * self.config.max_residual * n;
        precision.clear();
        precision.extend(patches.iter().map(|p| {
            if !p.ok {
                return 0.0;
            }
            let sigma2 = (p.ssd / (n - 1.0)).max(NOISE_FLOOR * NOISE_FLOOR);
            p.hess.max(1e-6) / sigma2
        }));
        valid_patch.clear();
        let min_rise = DISTINCT_RISE * min_hess * DISTINCT_STEP * DISTINCT_STEP;
        valid_patch.extend(patches.iter().enumerate().map(|(k, p)| {
            if !(p.ok && p.inside && p.hess >= min_hess && p.ssd <= max_ssd) {
                return false;
            }
            // Distinctness, where validity counts (header, step 5).
            let rise = |d: f32| v.eval::<false>(k, d).map_or(0.0, |e| e.ssd - p.ssd);
            !validate || rise(p.d - DISTINCT_STEP).min(rise(p.d + DISTINCT_STEP)) >= min_rise
        }));
        // Pixels using a patch: its footprint, extended to the image's far
        // edge for the last row and column of patches, and by one pixel all
        // round for the local cost at depth edges.
        samplers.clear();
        samplers.extend(patches.iter().enumerate().map(|(k, p)| {
            let (js, is) = (k % ws, k / ws);
            let xs = (
                js * st,
                if js + 1 == ws {
                    w - 1
                } else {
                    js * st + psz - 1
                },
            );
            let ys = (
                is * st,
                if is + 1 == hs {
                    h - 1
                } else {
                    is * st + psz - 1
                },
            );
            let grow = |r: (usize, usize), n: usize| (r.0.saturating_sub(1), (r.1 + 1).min(n - 1));
            ShiftSampler::new(v.i1, p.shift, grow(xs, w), grow(ys, h))
        }));

        d.resize(w * h, 0.0);
        if validate {
            valid.clear();
            valid.resize(w * h, false);
            var.clear();
            var.resize(w * h, f32::NAN);
        }
        // Patches covering coordinate c along one axis: first..=last.
        let span = |c: usize, count: usize| -> (usize, usize) {
            let last = (c / st).min(count - 1);
            let first = if c + 1 >= psz {
                (c + 1 - psz).div_ceil(st)
            } else {
                0
            };
            (first.min(last), last)
        };
        // Photometric cost of patch k's shift on the 3×3 pixels around (x, y).
        let local_cost = |x: usize, y: usize, k: usize| -> f32 {
            let mut c = 0.0;
            for dy in -1isize..=1 {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for dx in -1isize..=1 {
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    let i1 = samplers[k].sample(v.i1, xx, yy);
                    c += (i1 - v.i0[yy * w + xx]).abs();
                }
            }
            c
        };
        // Valid patches over the current pixel: (patch, weight).
        const MAX_COVER: usize = 16;
        let mut cand = [(0usize, 0.0f32); MAX_COVER];
        for y in 0..h {
            let (is0, is1) = span(y, hs);
            for x in 0..w {
                let (js0, js1) = span(x, ws);
                let i0 = v.i0[y * w + x];
                let (mut sw, mut swd) = (0.0f32, 0.0f32);
                let mut nc = 0;
                for is in is0..=is1 {
                    for js in js0..=js1 {
                        let k = is * ws + js;
                        if precision[k] == 0.0 {
                            continue;
                        }
                        let i1 = samplers[k].sample(v.i1, x, y);
                        let photo = 1.0 / (i1 - i0).abs().max(1.0);
                        let wk = photo * precision[k];
                        sw += wk;
                        swd += wk * patches[k].d;
                        if validate && valid_patch[k] && nc < MAX_COVER {
                            cand[nc] = (k, wk);
                            nc += 1;
                        }
                    }
                }
                let idx = y * w + x;
                d[idx] = if sw > 0.0 {
                    swd / sw
                } else if has_init {
                    self.buf.d_init[idx]
                } else {
                    v.dmin
                };
                if nc == 0 {
                    continue;
                }
                let cand = &cand[..nc];
                // Where valid patches cover the pixel, its D is theirs alone.
                // Patches that disagree by more than a pixel lie on different
                // surfaces (a depth edge): the one whose shift explains the
                // pixel's neighbourhood best wins, and only patches agreeing
                // with it are averaged, so no depth between the surfaces.
                let (lo, hi) = cand
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(k, _)| {
                        (lo.min(patches[k].d), hi.max(patches[k].d))
                    });
                let centre = if hi - lo > DISAGREE {
                    let mut costs = [(0.0f32, 0.0f32); MAX_COVER];
                    for (c, &(k, _)) in costs.iter_mut().zip(cand) {
                        *c = (local_cost(x, y, k), patches[k].d);
                    }
                    let costs = &costs[..nc];
                    let (win_cost, win_d) = costs
                        .iter()
                        .copied()
                        .fold((f32::INFINITY, lo), |b, c| if c.0 < b.0 { c } else { b });
                    // The other surface's best; a winner that is not clearly
                    // better leaves the pixel without depth.
                    let rival = costs
                        .iter()
                        .filter(|c| (c.1 - win_d).abs() > DISAGREE)
                        .fold(f32::INFINITY, |b, c| b.min(c.0));
                    if win_cost * AMBIGUITY_RATIO > rival {
                        continue;
                    }
                    Some(win_d)
                } else {
                    None
                };
                let (mut vw, mut vwd, mut vwd2, mut vwvar) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for &(k, wk) in cand {
                    let pd = patches[k].d;
                    if centre.is_some_and(|c| (pd - c).abs() > DISAGREE) {
                        continue;
                    }
                    vw += wk;
                    vwd += wk * pd;
                    vwd2 += wk * pd * pd;
                    vwvar += wk / precision[k];
                }
                let mean = vwd / vw;
                let spread = (vwd2 / vw - mean * mean).max(0.0);
                d[idx] = mean;
                valid[idx] = true;
                var[idx] = vwvar / vw + spread;
            }
        }
        self.buf.d = d;
        self.buf.valid = valid;
        self.buf.var = var;
        self.buf.precision = precision;
        self.buf.valid_patch = valid_patch;
        self.buf.samplers = samplers;
    }

    /// Brox et al. refinement of the dense D, as OpenCV's
    /// VariationalRefinement with one unknown per pixel. At the finest level
    /// only pixels a valid patch covers are solved for: the others get no
    /// depth, and their values must not leak into valid neighbours through
    /// the smoothness term.
    #[inline(never)]
    fn refine_variational(&mut self, level: usize) {
        let cfg = &self.config;
        if cfg.variational_iterations == 0 {
            return;
        }
        if level != cfg.finest_level && !cfg.refine_coarse_levels {
            return;
        }
        let g = self.levels[level].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let n = w * h;
        let masked = level == cfg.finest_level;
        let b = &mut self.buf;
        let solved = |valid: &[bool], i: usize| !masked || valid[i];
        b.act.clear();
        b.act
            .extend((0..n as u32).filter(|&i| solved(&b.valid, i as usize)));

        // Linearise the warp at the current D: mean and difference of the
        // left and warped right images (an unsolved pixel, or one whose ray
        // misses the right image, sees its own grey), and the epipolar
        // direction.
        b.avg.clear();
        b.avg.extend_from_slice(&b.i0);
        b.iz.clear();
        b.iz.resize(n, 0.0);
        let rb = &mut b.rb;
        rb.reset(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = rb.index(x, y);
                rb.d[r] = b.d[y * w + x];
            }
        }
        for &i in &b.act {
            let i = i as usize;
            let (x, y) = (i % w, i / w);
            let r = rb.index(x, y);
            if let Some(loc) = self.epi.locate(g, g.rays[i], x as f32, y as f32, b.d[i]) {
                let warped = self.i1.sample(loc[0], loc[1]);
                b.avg[i] = 0.5 * (b.i0[i] + warped);
                b.iz[i] = warped - b.i0[i];
                // The epipolar direction, until the derivatives below.
                (rb.id[r], rb.ixd[r]) = (loc[2], loc[3]);
            }
        }
        // Derivatives at the solved pixels: central differences clamped at
        // the border (diff_x, diff_y), the second ones composed of the first.
        let (avg, iz) = (&b.avg, &b.iz);
        let xm = |x: usize| x.saturating_sub(1);
        let xp = |x: usize| (x + 1).min(w - 1);
        let ym = |y: usize| y.saturating_sub(1);
        let yp = |y: usize| (y + 1).min(h - 1);
        let dx = |f: &[f32], x: usize, y: usize| 0.5 * (f[y * w + xp(x)] - f[y * w + xm(x)]);
        let dy = |f: &[f32], x: usize, y: usize| 0.5 * (f[yp(y) * w + x] - f[ym(y) * w + x]);
        let inv_edge = if cfg.edge_sigma > 0.0 {
            1.0 / (cfg.edge_sigma * cfg.edge_sigma)
        } else {
            0.0
        };
        for &i in &b.act {
            let i = i as usize;
            let (x, y) = (i % w, i / w);
            let r = rb.index(x, y);
            let (ex, ey) = (rb.id[r], rb.ixd[r]);
            let (ix, iy) = (dx(avg, x, y), dy(avg, x, y));
            let ixx = 0.5 * (dx(avg, xp(x), y) - dx(avg, xm(x), y));
            let ixy = 0.5 * (dx(avg, x, yp(y)) - dx(avg, x, ym(y)));
            let iyy = 0.5 * (dy(avg, x, yp(y)) - dy(avg, x, ym(y)));
            // Along the epipolar direction: id = ∇I·e, ixd = ∂x∇I·e,
            // iyd = ∂y∇I·e.
            let id = ix * ex + iy * ey;
            let ixd = ixx * ex + ixy * ey;
            let iyd = ixy * ex + iyy * ey;
            rb.id[r] = id;
            rb.ixd[r] = ixd;
            rb.iyd[r] = iyd;
            rb.iz[r] = iz[i];
            rb.ixz[r] = dx(iz, x, y);
            rb.iyz[r] = dy(iz, x, y);
            rb.inv_nd[r] = 1.0 / (id * id + ZETA2);
            rb.inv_nx[r] = 1.0 / (ixd * ixd + ZETA2);
            rb.inv_ny[r] = 1.0 / (iyd * iyd + ZETA2);
            rb.edge[r] = 1.0 / (1.0 + (b.gx[i] * b.gx[i] + b.gy[i] * b.gy[i]) * inv_edge);
            rb.solved[r] = 1.0;
            if x + 1 < w && solved(&b.valid, i + 1) {
                rb.right[r] = 1.0;
            }
            if i + w < n && solved(&b.valid, i + w) {
                rb.down[r] = 1.0;
            }
        }

        rb.solve(Solve {
            h,
            iterations: cfg.variational_iterations,
            sor_iterations: cfg.sor_iterations,
            alpha2: cfg.alpha * 0.5,
            gamma2: cfg.gamma * 0.5,
            delta2: cfg.delta * 0.5,
            omega: cfg.sor_omega,
        });

        // The refinement's own change counts towards the variance.
        let var = b.var.len() == n;
        for &i in &b.act {
            let i = i as usize;
            let change = rb.dd[rb.index(i % w, i / w)];
            b.d[i] += change;
            if var {
                b.var[i] += change * change;
            }
        }
    }

    /// Clamp D to the searched range, and drop occluded pixels and speckles
    /// from `valid`.
    #[inline(never)]
    fn mask(&mut self, level: usize) {
        let g = self.levels[level].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let n = w * h;
        let dmin = (g.s / self.config.max_range) as f32;
        let dmax = (g.s / self.config.min_range) as f32;
        let b = &mut self.buf;
        for i in 0..n {
            b.d[i] = b.d[i].clamp(dmin, dmax);
        }

        if self.config.occlusion_check {
            // Forward-warp D into the right image, keep the nearest (largest
            // D) per right pixel, and drop left pixels that lost by more than
            // a pixel of D.
            const MARGIN: f32 = 1.0;
            b.zbuf.clear();
            b.zbuf.resize(n, f32::NEG_INFINITY);
            b.target.clear();
            b.target.resize(n, usize::MAX);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if !b.valid[i] {
                        continue;
                    }
                    if let Some(loc) = self.epi.locate(g, g.rays[i], x as f32, y as f32, b.d[i]) {
                        let (qx, qy) = (loc[0].round(), loc[1].round());
                        if qx >= 0.0 && qy >= 0.0 && qx < w as f32 && qy < h as f32 {
                            let q = qy as usize * w + qx as usize;
                            b.target[i] = q;
                            b.zbuf[q] = b.zbuf[q].max(b.d[i]);
                        }
                    }
                }
            }
            for i in 0..n {
                let q = b.target[i];
                if q != usize::MAX && b.zbuf[q] > b.d[i] + MARGIN {
                    b.valid[i] = false;
                }
            }
        }

        // The size is in finest-level pixels: a quarter per level coarser.
        let speckle = self.config.speckle_size >> (2 * (level - self.config.finest_level));
        if speckle > 0 {
            remove_speckles(&mut b.valid, &b.d, w, h, speckle, &mut b.speckle);
        }
    }

    /// Keep a coarser level's validated depth for the coarse fallback.
    fn keep_level(&mut self, level: usize) {
        let g = self.levels[level].as_ref().unwrap();
        let b = &mut self.buf;
        if b.coarse.len() <= level {
            b.coarse.resize_with(level + 1, LevelDepth::default);
        }
        let c = &mut b.coarse[level];
        (c.w, c.h) = (g.w, g.h);
        c.d.clone_from(&b.d);
        c.var.clone_from(&b.var);
        c.valid.clone_from(&b.valid);
    }

    /// Fill finest-level pixels without a valid estimate from the nearest
    /// coarser level that has one, finest first. A coarse estimate is used
    /// only if every finest-level measurement within its support (the
    /// patches over its cell) agrees with it: a coarse patch over a depth
    /// edge may have matched the far surface while the near one is
    /// textureless (a white box before a textured wall), and filling the box
    /// with the wall's depth would hide it.
    #[inline(never)]
    fn coarse_fallback(&mut self) {
        let finest = self.config.finest_level;
        let g = self.levels[finest].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let reach = self.config.patch_size - 1;
        let b = &mut self.buf;
        b.source.clear();
        b.source.extend(
            b.valid
                .iter()
                .map(|&v| if v { finest as u8 } else { NO_SOURCE }),
        );
        let fallback = (finest + self.config.coarse_fallback).min(self.coarsest);
        for level in finest + 1..=fallback {
            let k = level - finest;
            let f = (1usize << k) as f32;
            let c = &b.coarse[level];
            let (cw, ch) = (c.w, c.h);
            // Finest pixel x is coarse pixel x / 2^k: its nearest cell.
            let cell = |x: usize, y: usize| {
                let half = 1usize << (k - 1);
                (((y + half) >> k).min(ch - 1), ((x + half) >> k).min(cw - 1))
            };
            // Count the disagreeing measurements per cell, then integrate:
            // cell (cx, cy) counts at (cx + 1, cy + 1).
            let iw = cw + 1;
            b.disagree.clear();
            b.disagree.resize(iw * (ch + 1), 0);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let (cy, cx) = cell(x, y);
                    let j = cy * cw + cx;
                    if !b.valid[i] || !c.valid[j] {
                        continue;
                    }
                    let (df, dc) = (b.d[i], f * c.d[j]);
                    if (df - dc).abs() > DISAGREE.max(FALLBACK_AGREE * df) {
                        b.disagree[(cy + 1) * iw + cx + 1] += 1;
                    }
                }
            }
            for cy in 1..=ch {
                for cx in 1..=cw {
                    let p = cy * iw + cx;
                    b.disagree[p] = b.disagree[p] + b.disagree[p - 1] + b.disagree[p - iw]
                        - b.disagree[p - iw - 1];
                }
            }
            let ii = &b.disagree;
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if b.source[i] != NO_SOURCE {
                        continue;
                    }
                    let (cy, cx) = cell(x, y);
                    let j = cy * cw + cx;
                    if !c.valid[j] {
                        continue;
                    }
                    let (x0, x1) = (cx.saturating_sub(reach), (cx + reach + 1).min(cw));
                    let (y0, y1) = (cy.saturating_sub(reach), (cy + reach + 1).min(ch));
                    if ii[y1 * iw + x1] + ii[y0 * iw + x0] > ii[y0 * iw + x1] + ii[y1 * iw + x0] {
                        continue;
                    }
                    b.d[i] = f * c.d[j];
                    b.var[i] = f * f * c.var[j];
                    b.source[i] = level as u8;
                }
            }
        }
    }

    /// A textureless region matches the right image at many depths, but not
    /// at its outline: filled with the wall's depth, a textureless box
    /// before the wall lands on the wall's texture along one side, a strip as
    /// wide as the disparity error. Disparity moves pixels along the
    /// epipolar line, so the test runs along it: along each image row (the
    /// epipolar line of a rectified pair, close to it for a side-by-side
    /// rig), a fill is dropped if its own pixel, or either end of its
    /// textureless run (FLAT_GRADIENT; FLAT_END_MARGIN inside, at the fill's
    /// D give or take half a pixel of its level), differs from the right
    /// image by more than `max_residual`, or the right image does not see it.
    #[inline(never)]
    fn reject_contradicted_fills(&mut self) {
        if self.config.coarse_fallback == 0 {
            return;
        }
        let finest = self.config.finest_level;
        let g = self.levels[finest].as_ref().unwrap();
        let (w, h) = (g.w, g.h);
        let max_residual = self.config.max_residual;
        let flat_gradient = FLAT_GRADIENT * self.config.min_texture;
        let (epi, i1) = (&self.epi, &self.i1);
        let b = &mut self.buf;
        let (i0, gx) = (&b.i0, &b.gx);
        let filled = |src: u8| src != NO_SOURCE && src as usize != finest;
        // Right-image grey at pixel (x, y) seen at D, if inside the image.
        let seen_at = |x: usize, y: usize, d: f32| -> Option<f32> {
            let loc = epi.locate(g, g.rays[y * w + x], x as f32, y as f32, d)?;
            let inside = loc[0] >= 0.0
                && loc[1] >= 0.0
                && loc[0] <= (w - 1) as f32
                && loc[1] <= (h - 1) as f32;
            inside.then(|| i1.sample(loc[0], loc[1]))
        };
        let agrees = |x: usize, y: usize, d: f32| {
            seen_at(x, y, d).is_some_and(|v| (v - i0[y * w + x]).abs() <= max_residual)
        };
        for y in 0..h {
            let row = y * w;
            let mut x = 0;
            while x < w {
                // The next run: textureless pixels, or a single textured one.
                let first = x;
                let flat = gx[row + x].abs() < flat_gradient;
                x += 1;
                while flat && x < w && gx[row + x].abs() < flat_gradient {
                    x += 1;
                }
                let last = x - 1;
                let m = FLAT_END_MARGIN.min((last - first) / 2);
                for xx in first..=last {
                    let src = b.source[row + xx];
                    if !filled(src) {
                        continue;
                    }
                    let d = b.d[row + xx];
                    let half = 0.5 * (1usize << (src as usize - finest)) as f32;
                    // An end at the image border meets no texture.
                    let end = |xe: usize, border: bool| {
                        border || [d, d - half, d + half].iter().any(|&dd| agrees(xe, y, dd))
                    };
                    let ok = agrees(xx, y, d)
                        && (!flat || (end(first + m, first == 0) && end(last - m, last == w - 1)));
                    if !ok {
                        b.source[row + xx] = NO_SOURCE;
                    }
                }
            }
        }
    }

    /// Conversion to log-range at the finest level.
    #[inline(never)]
    fn finish(&mut self, out: &mut DenseDepth) {
        let level = self.config.finest_level;
        let g = self.levels[level].as_ref().unwrap();
        let n = g.w * g.h;
        let b = &self.buf;
        let ln_s = (g.s as f32).ln();
        out.width = g.w;
        out.height = g.h;
        out.level = level;
        out.log_range.resize(n, f32::NAN);
        out.variance.resize(n, f32::NAN);
        out.source_level.clear();
        out.source_level.extend_from_slice(&b.source);
        for i in 0..n {
            if b.source[i] != NO_SOURCE {
                let d = b.d[i];
                out.log_range[i] = ln_s - d.ln();
                out.variance[i] = b.var[i] / (d * d);
            } else {
                out.log_range[i] = f32::NAN;
                out.variance[i] = f32::NAN;
            }
        }
    }
}

/// Clear `keep` on connected regions (4-neighbours whose D differ by at most
/// SPECKLE_STEP) smaller than `min_size` pixels, as OpenCV's filterSpeckles.
#[inline(never)]
fn remove_speckles(
    keep: &mut [bool],
    d: &[f32],
    w: usize,
    h: usize,
    min_size: usize,
    scratch: &mut SpeckleScratch,
) {
    const UNSEEN: u32 = u32::MAX;
    let SpeckleScratch {
        label,
        stack,
        region,
    } = scratch;
    label.clear();
    label.resize(w * h, UNSEEN);
    stack.clear();
    for start in 0..w * h {
        if !keep[start] || label[start] != UNSEEN {
            continue;
        }
        label[start] = start as u32;
        stack.push(start);
        region.clear();
        while let Some(i) = stack.pop() {
            region.push(i);
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if keep[j] && label[j] == UNSEEN && (d[j] - d[i]).abs() <= SPECKLE_STEP {
                    label[j] = start as u32;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if region.len() < min_size {
            for &i in region.iter() {
                keep[i] = false;
            }
        }
    }
}
