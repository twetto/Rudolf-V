// Dense stereo against ray-cast scenes with exact depth: textured planes seen
// through both cameras, so occlusions, slant and lens distortion are real.

use camera_geometry::{CameraModel, Pixel};
use nalgebra::{Matrix3, Vector3};
use rudolf_v::camera::{CameraIntrinsics, DistortionModel, StereoRig};
use rudolf_v::dense_stereo::{DenseDepth, DenseStereo, DenseStereoConfig};
use rudolf_v::image::Image;
use rudolf_v::rectify::StereoRectification;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Texture {
    /// Multi-octave value noise; `cell` is the coarsest cell in metres.
    Noise { seed: u64, cell: f64 },
}

#[derive(Clone)]
struct Plane {
    origin: Vector3<f64>,
    /// Orthonormal in-plane axes; texture coordinates are along them.
    u: Vector3<f64>,
    v: Vector3<f64>,
    /// [u_min, u_max, v_min, v_max] in metres from `origin`.
    bounds: [f64; 4],
    texture: Texture,
    /// Where `flat` applies within the plane, a flat (textureless) rectangle.
    flat: Option<([f64; 4], f64)>,
}

impl Plane {
    fn facing(z: f64, half: f64, texture: Texture) -> Self {
        Plane {
            origin: Vector3::new(0.0, 0.0, z),
            u: Vector3::x(),
            v: Vector3::y(),
            bounds: [-half, half, -half, half],
            texture,
            flat: None,
        }
    }
}

fn hash(i: i64, j: i64, seed: u64) -> f64 {
    let mut h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ seed.wrapping_mul(0x165667B19E3779F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(u: f64, v: f64, seed: u64) -> f64 {
    let (i, j) = (u.floor(), v.floor());
    let (fu, fv) = (u - i, v - j);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (su, sv) = (s(fu), s(fv));
    let (i, j) = (i as i64, j as i64);
    let a = hash(i, j, seed) * (1.0 - su) + hash(i + 1, j, seed) * su;
    let b = hash(i, j + 1, seed) * (1.0 - su) + hash(i + 1, j + 1, seed) * su;
    a * (1.0 - sv) + b * sv
}

fn shade(p: &Plane, a: f64, b: f64) -> f64 {
    if let Some((r, level)) = p.flat {
        if a >= r[0] && a <= r[1] && b >= r[2] && b <= r[3] {
            return level;
        }
    }
    match p.texture {
        Texture::Noise { seed, cell } => {
            let mut sum = 0.0;
            let mut amp = 1.0;
            let mut c = cell;
            for o in 0..5 {
                sum += amp * value_noise(a / c, b / c, seed + o);
                amp *= 0.55;
                c *= 0.5;
            }
            30.0 + 195.0 * (sum / 2.1).clamp(0.0, 1.0)
        }
    }
}

/// Nearest hit along origin + s·dir: (s, intensity).
fn cast(scene: &[Plane], o: Vector3<f64>, dir: Vector3<f64>) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for p in scene {
        let n = p.u.cross(&p.v);
        let den = n.dot(&dir);
        if den.abs() < 1e-12 {
            continue;
        }
        let s = n.dot(&(p.origin - o)) / den;
        if s <= 1e-9 || best.is_some_and(|(b, _)| b <= s) {
            continue;
        }
        let q = o + dir * s - p.origin;
        let (a, b) = (q.dot(&p.u), q.dot(&p.v));
        if a < p.bounds[0] || a > p.bounds[1] || b < p.bounds[2] || b > p.bounds[3] {
            continue;
        }
        best = Some((s, shade(p, a, b)));
    }
    best
}

/// Sensor noise added to the renders (grey levels, standard deviation).
const NOISE_SIGMA: f64 = 1.5;

/// Camera `cam` (1 = right) of the rig, 2×2 supersampled, with sensor noise.
fn render(rig: &StereoRig, cam: usize, scene: &[Plane]) -> Image<u8> {
    let c = if cam == 0 { &rig.cam0 } else { &rig.cam1 };
    let proj = c.projection();
    let [w, h] = c.resolution;
    let r = Matrix3::from_fn(|i, j| rig.r_10[i][j]);
    let t = Vector3::from(rig.t_10);
    // Camera 1 in camera-0 (world) coordinates.
    let (o, m) = if cam == 0 {
        (Vector3::zeros(), Matrix3::identity())
    } else {
        (-(r.transpose() * t), r.transpose())
    };
    let mut data = vec![0u8; w * h];
    let mut rng = 0x2545F4914F6CDD1Du64 ^ cam as u64;
    let mut gauss = || {
        // Irwin-Hall: the sum of 12 uniforms, minus 6, is close to N(0, 1).
        let mut s = -6.0;
        for _ in 0..12 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            s += (rng >> 11) as f64 / (1u64 << 53) as f64;
        }
        s
    };
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (dx, dy) in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] {
                let b = proj
                    .unproject(Pixel::new(x as f64 + dx, y as f64 + dy))
                    .expect("ray")
                    .vector();
                sum += cast(scene, o, m * b).map_or(0.0, |(_, i)| i);
            }
            data[y * w + x] = (sum * 0.25 + NOISE_SIGMA * gauss())
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(w, h, data)
}

/// True log-range at every pixel of the depth map's level.
fn truth(rig: &StereoRig, scene: &[Plane], depth: &DenseDepth) -> Vec<f32> {
    let proj = rig.cam0.projection();
    let mut out = Vec::with_capacity(depth.width * depth.height);
    for y in 0..depth.height {
        for x in 0..depth.width {
            let (u, v) = depth.full_res_pixel(x, y);
            let b = proj.unproject(Pixel::new(u, v)).expect("ray").vector();
            out.push(cast(scene, Vector3::zeros(), b).map_or(f32::NAN, |(s, _)| s.ln() as f32));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Rigs
// ---------------------------------------------------------------------------

fn rectified_rig() -> StereoRig {
    let cam = CameraIntrinsics::new(500.0, 500.0, 319.5, 239.5, 640, 480);
    StereoRig::new(
        cam.clone(),
        cam,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [-0.11, 0.0, 0.0],
    )
}

/// EuRoC-like raw pair: radial-tangential distortion and a small rotation.
fn raw_rig() -> StereoRig {
    let cam = |fx: f64, cx: f64, k: [f64; 4]| CameraIntrinsics {
        fx,
        fy: fx * 0.998,
        cx,
        cy: 241.0,
        resolution: [640, 480],
        distortion: k.to_vec(),
        model: DistortionModel::RadTan,
    };
    let cam0 = cam(458.0, 322.0, [-0.28, 0.074, 2e-4, 2e-5]);
    let cam1 = cam(457.0, 318.0, [-0.283, 0.075, -1e-4, -4e-5]);
    let (yaw, roll): (f64, f64) = (0.012, -0.006);
    let rz = Matrix3::new(
        roll.cos(),
        -roll.sin(),
        0.0,
        roll.sin(),
        roll.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let ry = Matrix3::new(
        yaw.cos(),
        0.0,
        yaw.sin(),
        0.0,
        1.0,
        0.0,
        -yaw.sin(),
        0.0,
        yaw.cos(),
    );
    let r = ry * rz;
    StereoRig::new(
        cam0,
        cam1,
        [
            [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
            [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
            [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
        ],
        [-0.110, 0.0004, -0.0008],
    )
}

/// ZED 2i raw pair at half size (640×360): rational distortion, factory extrinsics.
fn zed_raw_rig() -> StereoRig {
    let cam = |k: [f64; 4], d: [f64; 8]| CameraIntrinsics {
        fx: k[0] * 0.5,
        fy: k[1] * 0.5,
        cx: (k[2] + 0.5) * 0.5 - 0.5,
        cy: (k[3] + 0.5) * 0.5 - 0.5,
        resolution: [640, 360],
        distortion: d.to_vec(),
        model: DistortionModel::Rational,
    };
    let cam0 = cam(
        [958.05, 958.365, 643.225, 362.359],
        [
            -0.576428,
            28.0253,
            -3.6843e-05,
            -0.000274993,
            -11.4656,
            -0.473068,
            27.7565,
            -8.83319,
        ],
    );
    let cam1 = cam(
        [955.285, 955.54, 644.78, 367.3435],
        [
            -4.01805,
            19.5407,
            -2.1755e-05,
            -0.000179312,
            -8.1645,
            -3.93723,
            19.1884,
            -6.66662,
        ],
    );
    StereoRig::new(
        cam0,
        cam1,
        [
            [0.999984351, 0.003146175, -0.004622436],
            [-0.003142531, 0.999994737, 0.000795437],
            [0.004624914, -0.000780899, 0.999989027],
        ],
        [-0.119868005, 0.000731885, -0.001205486],
    )
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct Stats {
    coverage: f32,
    median: f32,
    p95: f32,
    /// Fraction of estimates more than 10 % off.
    bad: f32,
}

fn stats(depth: &DenseDepth, truth: &[f32], mask: impl Fn(usize, usize) -> bool) -> Stats {
    let mut err = Vec::new();
    let mut total = 0usize;
    for y in 0..depth.height {
        for x in 0..depth.width {
            let i = y * depth.width + x;
            if !mask(x, y) || !truth[i].is_finite() {
                continue;
            }
            total += 1;
            let e = depth.log_range[i];
            if e.is_finite() {
                err.push((e - truth[i]).abs());
            }
        }
    }
    err.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| {
        err.get(((err.len() as f32 * f) as usize).min(err.len().saturating_sub(1)))
            .copied()
            .unwrap_or(f32::NAN)
    };
    let bad = err.iter().filter(|&&e| e > 0.1f32.ln_1p()).count();
    Stats {
        coverage: err.len() as f32 / total.max(1) as f32,
        median: q(0.5),
        p95: q(0.95),
        bad: bad as f32 / err.len().max(1) as f32,
    }
}

fn run(rig: &StereoRig, scene: &[Plane], config: DenseStereoConfig) -> (DenseDepth, Vec<f32>) {
    let left = render(rig, 0, scene);
    let right = render(rig, 1, scene);
    let mut stereo = DenseStereo::new(rig.clone(), config);
    let depth = stereo.compute(&left, &right);
    let gt = truth(rig, scene, &depth);
    (depth, gt)
}

fn noise(seed: u64) -> Texture {
    Texture::Noise { seed, cell: 0.12 }
}

/// Map pixels whose left-image ray the right camera can see at depth `z`
/// (drops the strip at the left border that the right eye never sees).
fn seen_at(depth: &DenseDepth, rig: &StereoRig, z: f64) -> impl Fn(usize, usize) -> bool {
    let disparity = rig.cam1.fx * rig.baseline_meters() / z;
    let lim = (disparity / (1usize << depth.level) as f64) as usize + 6;
    let (w, h) = (depth.width, depth.height);
    move |x, y| x >= lim && x + 6 < w && y >= 6 && y + 6 < h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn rectified_plane() {
    let rig = rectified_rig();
    let scene = [Plane::facing(2.0, 10.0, noise(1))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    assert!(DenseStereo::new(rig.clone(), DenseStereoConfig::default()).is_rectified());
    let s = stats(&depth, &gt, seen_at(&depth, &rig, 2.0));
    eprintln!(
        "rectified plane: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.97, "coverage {}", s.coverage);
    assert!(s.median < 0.005, "median log error {}", s.median);
    assert!(s.p95 < 0.02, "p95 log error {}", s.p95);
    assert!(s.bad < 0.002, "bad {}", s.bad);
}

#[test]
fn raw_distorted_plane() {
    let rig = raw_rig();
    let scene = [Plane::facing(2.0, 10.0, noise(2))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    assert!(!DenseStereo::new(rig.clone(), DenseStereoConfig::default()).is_rectified());
    let s = stats(&depth, &gt, seen_at(&depth, &rig, 2.0));
    eprintln!(
        "raw plane: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.95, "coverage {}", s.coverage);
    assert!(s.median < 0.006, "median log error {}", s.median);
    assert!(s.p95 < 0.025, "p95 log error {}", s.p95);
    assert!(s.bad < 0.005, "bad {}", s.bad);
}

#[test]
fn raw_zed_rational_plane() {
    let rig = zed_raw_rig();
    let scene = [Plane::facing(2.0, 10.0, noise(9))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    let s = stats(&depth, &gt, seen_at(&depth, &rig, 2.0));
    eprintln!(
        "raw ZED plane: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.95, "coverage {}", s.coverage);
    assert!(s.median < 0.006, "median log error {}", s.median);
    assert!(s.p95 < 0.025, "p95 log error {}", s.p95);
    assert!(s.bad < 0.005, "bad {}", s.bad);
}

#[test]
fn raw_zed_plane_rectified_first() {
    // The same raw ZED pair, rectified before matching: the depth is on the
    // rectified left grid, whose rays are the rectified camera's, turned
    // back into the raw left camera by the rectification.
    let rig = zed_raw_rig();
    let scene = [Plane::facing(2.0, 10.0, noise(9))];
    let left = render(&rig, 0, &scene);
    let right = render(&rig, 1, &scene);
    let (f, cx, cy) = (rig.cam0.fx * 0.99, rig.cam0.cx, rig.cam0.cy);
    let rect = StereoRectification::new(&rig, f, cx, cy);
    let r0 = Matrix3::from_fn(|i, j| rect.rotation(0)[i][j]);
    let mut stereo = DenseStereo::new_rectifying(rect, DenseStereoConfig::default());
    assert!(stereo.is_rectified());
    let depth = stereo.compute(&left, &right);
    let mut gt = Vec::with_capacity(depth.width * depth.height);
    for y in 0..depth.height {
        for x in 0..depth.width {
            let (u, v) = depth.full_res_pixel(x, y);
            let ray = r0.transpose() * Vector3::new((u - cx) / f, (v - cy) / f, 1.0).normalize();
            gt.push(cast(&scene, Vector3::zeros(), ray).map_or(f32::NAN, |(s, _)| s.ln() as f32));
        }
    }
    let s = stats(&depth, &gt, seen_at(&depth, &rig, 2.0));
    eprintln!(
        "raw ZED plane, rectified first: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.95, "coverage {}", s.coverage);
    assert!(s.median < 0.006, "median log error {}", s.median);
    assert!(s.p95 < 0.025, "p95 log error {}", s.p95);
    assert!(s.bad < 0.005, "bad {}", s.bad);
}

#[test]
fn slanted_floor() {
    let rig = rectified_rig();
    // Floor 0.6 m below the camera, seen from 1 to 10 m; a wall at 10 m.
    let floor = Plane {
        origin: Vector3::new(0.0, 0.6, 0.0),
        u: Vector3::x(),
        v: Vector3::z(),
        bounds: [-10.0, 10.0, 0.3, 10.0],
        texture: Texture::Noise {
            seed: 3,
            cell: 0.08,
        },
        flat: None,
    };
    let scene = [floor, Plane::facing(10.0, 20.0, noise(4))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    let below = |x: usize, y: usize| y > depth.height / 2 + 12 && x > 50 && x + 6 < depth.width;
    let s = stats(&depth, &gt, below);
    eprintln!(
        "floor: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.9, "coverage {}", s.coverage);
    assert!(s.median < 0.01, "median log error {}", s.median);
    assert!(s.bad < 0.01, "bad {}", s.bad);
}

#[test]
fn occluding_box() {
    let rig = rectified_rig();
    let mut front = Plane::facing(1.0, 0.25, noise(5));
    front.origin.x = 0.05;
    let scene = [front, Plane::facing(3.0, 10.0, noise(6))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());

    // Away from the box's outline, both surfaces are measured well.
    let (w, h) = (depth.width, depth.height);
    let near_edge = |x: usize, y: usize| {
        let i = y * w + x;
        let mut edge = false;
        for dy in -8i32..=8 {
            for dx in -8i32..=8 {
                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                    edge |= (gt[yy as usize * w + xx as usize] - gt[i]).abs() > 0.1;
                }
            }
        }
        edge
    };
    let seen = seen_at(&depth, &rig, 3.0);
    let s = stats(&depth, &gt, |x, y| seen(x, y) && !near_edge(x, y));
    eprintln!(
        "box, away from edges: coverage {:.3} median {:.4} p95 {:.4} bad {:.4}",
        s.coverage, s.median, s.p95, s.bad
    );
    assert!(s.coverage > 0.95, "coverage {}", s.coverage);
    assert!(s.median < 0.006, "median log error {}", s.median);
    assert!(s.bad < 0.005, "bad {}", s.bad);

    // Near the outline, an estimate is one of the two surfaces or nothing:
    // no depth in between.
    let all = stats(&depth, &gt, |x, y| seen(x, y));
    let mut between = 0usize;
    let proj = rig.cam0.projection();
    for y in 0..h {
        for x in 0..w {
            let e = depth.log_range[y * w + x];
            if !e.is_finite() {
                continue;
            }
            let (u, v) = depth.full_res_pixel(x, y);
            let b = proj.unproject(Pixel::new(u, v)).unwrap().vector();
            // Range to each surface along this ray, ignoring the box's bounds.
            let on = [1.0, 3.0]
                .iter()
                .any(|z| ((z / b.z).ln() as f32 - e).abs() < 0.1);
            between += !on as usize;
        }
    }
    let n_valid = depth.log_range.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "box, everywhere: coverage {:.3} bad {:.4}; between the surfaces: {between} of {n_valid}",
        all.coverage, all.bad
    );
    assert!(
        (between as f32) < 0.005 * n_valid as f32,
        "{between} estimates between the surfaces"
    );
}

/// Pixels within `margin` of the flat rectangle's interior on a wall facing
/// the camera at `z` (map pixels, rectified_rig).
fn flat_interior(
    depth: &DenseDepth,
    z: f64,
    half: [f64; 2],
    margin: f64,
) -> impl Fn(usize, usize) -> bool {
    let s = (1usize << depth.level) as f64;
    let (cx, cy) = (319.5 / s, 239.5 / s);
    let (hx, hy) = (half[0] * 500.0 / z / s, half[1] * 500.0 / z / s);
    move |x, y| (x as f64 - cx).abs() < hx - margin && (y as f64 - cy).abs() < hy - margin
}

#[test]
fn textureless_wall_has_no_depth() {
    let rig = rectified_rig();
    let mut wall = Plane::facing(2.0, 10.0, noise(7));
    // A 1.2 m × 1.0 m flat rectangle in the middle of the view.
    wall.flat = Some(([-0.6, 0.6, -0.5, 0.5], 140.0));
    let scene = [wall];
    let strict = DenseStereoConfig {
        coarse_fallback: 0,
        ..Default::default()
    };
    let (depth, gt) = run(&rig, &scene, strict);
    // 0.6 m at 2 m is 150 px at full size, 75 at level 1; 10 px is the patch
    // side and densification overlap at this level.
    let inside = flat_interior(&depth, 2.0, [0.6, 0.5], 10.0);
    let (w, h) = (depth.width, depth.height);
    let mut n = 0usize;
    let mut with_depth = 0usize;
    for y in 0..h {
        for x in 0..w {
            if inside(x, y) {
                n += 1;
                with_depth += depth.log_range[y * w + x].is_finite() as usize;
            }
        }
    }
    eprintln!("flat rectangle, finest level only: {with_depth} of {n} pixels have depth");
    assert!(n > 1000);
    assert_eq!(with_depth, 0, "depth invented on a textureless wall");

    // The coarse fallback may fill it from coarser patches that reach its
    // textured surroundings: then with the wall's depth, and less certain
    // than the measured wall around it.
    let (depth, _) = run(&rig, &scene, DenseStereoConfig::default());
    let (mut filled, mut wrong, mut var_filled, mut var_measured, mut measured) =
        (0usize, 0usize, 0.0f64, 0.0f64, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let e = depth.log_range[i];
            if !e.is_finite() {
                continue;
            }
            if inside(x, y) {
                filled += 1;
                wrong += ((e - gt[i]).abs() > 0.1f32.ln_1p()) as usize;
                var_filled += depth.variance[i] as f64;
                assert!(depth.source_level[i] as usize > depth.level);
            } else if depth.source_level[i] as usize == depth.level {
                measured += 1;
                var_measured += depth.variance[i] as f64;
            }
        }
    }
    let (var_filled, var_measured) = (
        var_filled / filled.max(1) as f64,
        var_measured / measured as f64,
    );
    eprintln!(
        "flat rectangle, coarse fallback: {filled} of {n} pixels, {wrong} off by over 10 %; \
         variance {var_filled:.2e} vs {var_measured:.2e} measured"
    );
    assert_eq!(
        wrong, 0,
        "the fallback filled the rectangle with a wrong depth"
    );
    assert!(filled == 0 || var_filled > var_measured);
}

#[test]
fn textureless_box_keeps_its_own_depth() {
    // A box at 1 m without any texture before a textured wall at 3 m. Coarse
    // patches over its outline see mostly the wall and may match it: the
    // fallback must not fill the box with the wall's depth (an obstacle
    // would vanish).
    let rig = rectified_rig();
    let mut white = Plane::facing(1.0, 0.25, noise(9));
    white.flat = Some(([-1.0, 1.0, -1.0, 1.0], 215.0));
    let scene = [white, Plane::facing(3.0, 10.0, noise(10))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    // The box's interior, a patch side in from its outline: patches over the
    // outline match the wall and carry its depth that far in.
    let inside = flat_interior(&depth, 1.0, [0.25, 0.25], 8.0);
    let (w, h) = (depth.width, depth.height);
    let (mut n, mut with_depth, mut wrong) = (0usize, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if !inside(x, y) {
                continue;
            }
            let i = y * w + x;
            n += 1;
            let e = depth.log_range[i];
            if e.is_finite() {
                with_depth += 1;
                wrong += ((e - gt[i]).abs() > 0.1) as usize;
            }
        }
    }
    eprintln!("textureless box: {with_depth} of {n} pixels have depth, {wrong} not the box's");
    assert!(n > 5000);
    assert_eq!(wrong, 0, "the textureless box took another surface's depth");
}

#[test]
fn variance_tracks_error() {
    let rig = rectified_rig();
    let scene = [Plane::facing(4.0, 20.0, noise(8))];
    let (depth, gt) = run(&rig, &scene, DenseStereoConfig::default());
    let seen = seen_at(&depth, &rig, 4.0);
    let (mut sum_e2, mut sum_var, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 0..depth.height {
        for x in 0..depth.width {
            let i = y * depth.width + x;
            if seen(x, y) && depth.log_range[i].is_finite() {
                let e = (depth.log_range[i] - gt[i]) as f64;
                sum_e2 += e * e;
                sum_var += depth.variance[i] as f64;
                n += 1;
            }
        }
    }
    let (mse, mvar) = (sum_e2 / n as f64, sum_var / n as f64);
    eprintln!("variance: mean squared error {mse:.3e}, mean variance {mvar:.3e}");
    assert!(depth.variance.iter().all(|v| v.is_nan() || *v >= 0.0));
    // Within an order of magnitude of the actual error.
    assert!(
        mvar > mse / 10.0 && mvar < mse * 10.0,
        "mse {mse:.3e} vs variance {mvar:.3e}"
    );
}

#[test]
fn one_eye_object_has_no_depth() {
    // A bright object so close to the left camera that the right one cannot
    // see it, before a textured wall at 3 m. Compared on shape alone (after
    // mean removal), the object's outline matched the wall's texture in the
    // right image, and the object took the wall's depth: an obstacle reading
    // 3 m away. Its brightness differs from the wall's, so it must not.
    let rig = rectified_rig();
    let z = 0.15;
    // The left image's columns from its border to about 200.
    let mut near = Plane::facing(z, 1.0, noise(21));
    near.bounds = [-0.64 * z, -0.24 * z, -0.5, 0.5];
    near.flat = Some(([-1.0, 1.0, -1.0, 1.0], 252.0));
    let scene = [near, Plane::facing(3.0, 10.0, noise(22))];
    for level in [1, 3] {
        let config = DenseStereoConfig {
            finest_level: level,
            ..Default::default()
        };
        let (depth, gt) = run(&rig, &scene, config);
        let on_object = |i: usize| (gt[i] - (z as f32).ln()).abs() < 0.5;
        let n = (0..gt.len()).filter(|&i| on_object(i)).count();
        let with_depth = (0..gt.len())
            .filter(|&i| on_object(i) && depth.log_range[i].is_finite())
            .count();
        eprintln!("one-eye object, level {level}: {with_depth} of {n} pixels have depth");
        assert!(n > 1000);
        assert_eq!(
            with_depth, 0,
            "level {level}: the one-eye object took a depth"
        );
    }
}
