// dense_stereo_common.wgsl — types shared by the GPU dense stereo kernels
// (gpu/dense_stereo.rs prepends this to each of them). The CPU reference is
// dense_stereo.rs; the comments there explain the method.
//
// {{PSZ}}: patch side (the patch kernels run one 8×8 patch per 64-thread
// workgroup, a thread per pixel). {{BORDER}}: the CPU's padding of the right
// image (patch side + 2), so that bilinear weights round as the CPU's do.

// One level's dispatch parameters (gpu/dense_stereo.rs `Params`).
struct Params {
    w: u32,           // level size
    h: u32,
    ws: u32,          // patch grid
    hs: u32,
    stride: u32,      // patch stride
    colour: u32,      // propagation: the red-black colour this pass updates
    per_row: u32,     // propagation: patches of one colour per patch row
    iters: u32,       // Gauss-Newton iterations per patch and sweep
    dmin: f32,        // D's search range
    dmax: f32,
    mean_norm: u32,   // compare patches after removing their mean
    cw: u32,          // the coarser level's size (upsampling)
    ch: u32,
    validate: u32,    // the level's patches are validated
    has_init: u32,    // there is a coarser level's D for uncovered pixels
    min_speckle: u32, // mask: smallest region kept (pixels of the level)
    min_hess: f32,    // validity: least Hessian (texture), ...
    max_ssd: f32,     // ... largest SSD (residual),
    max_bright: f32,  // ... largest brightness offset,
    min_rise: f32,    // ... least SSD rise for a distinct match
}

// A patch's estimate (dense_stereo.rs `PatchState`, plus propagation's
// moved/settled flags).
struct Patch {
    d: f32,
    ssd: f32,
    cost: f32,
    hess: f32,
    shift_x: f32,
    shift_y: f32,
    offset: f32,
    flags: u32,
}

// A patch's inverse variance (0: not used) and validity, for densification.
struct PatchInfo {
    prec: f32,       // inverse variance (0: not used)
    valid: u32,
}

// Σ gx², Σ gx·gy, Σ gy², Σ gx, Σ gy over a patch (`PatchSums`).
struct Sums {
    xx: f32,
    xy: f32,
    yy: f32,
    x: f32,
    y: f32,
}

const PSZ: u32 = {{PSZ}}u;
const BORDER: i32 = {{BORDER}};

// Patch flags.
const OK: u32 = 1u;        // the ray reaches the right image
const INSIDE: u32 = 2u;    // the patch lands fully inside the right image
const MOVED: u32 = 4u;     // D moved in the latest evaluation
const SETTLED: u32 = 8u;   // that evaluation's Gauss-Newton converged

// Stands for +∞ (WGSL may assume there are none).
const BIG: f32 = 3.0e38;

// A patch without an estimate at D `d` (`PatchState::FAILED`).
fn failed(d: f32) -> Patch {
    return Patch(d, BIG, BIG, 0.0, 0.0, 0.0, BIG, 0u);
}
