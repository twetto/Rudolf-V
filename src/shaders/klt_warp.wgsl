// klt_warp.wgsl — Wavefront-per-patch KLT tracker (§4).
//
// KEY CHANGE FROM klt.wgsl:
//   Old: 1 thread = 1 feature (sequential pixel iteration)
//   New: 1 workgroup = 1 feature (cooperative pixel iteration)
//
// Each workgroup has WG_SIZE threads that cooperate on one feature's patch.
// Pixels are distributed across threads in a strided pattern:
//   thread 0: pixels 0, WG_SIZE, 2*WG_SIZE, ...
//   thread 1: pixels 1, WG_SIZE+1, 2*WG_SIZE+1, ...
//
// Partial sums are reduced via shared memory + workgroupBarrier().
// This cuts the per-thread work from PATCH bilinear lookups down to
// ceil(PATCH / WG_SIZE).  With WG_SIZE=16 and PATCH=225 (window=7),
// each thread does ~14 lookups instead of 225 — a 16× reduction.
//
//
// DISPATCH MODEL
// ──────────────
// dispatch_workgroups(n_features, 1, 1)
// workgroup_id.x  = feature index
// local_id.x      = thread index within the cooperating group
//
//
// SHARED MEMORY USAGE
// ───────────────────
// 5 × WG_SIZE floats for Hessian/residual reduction arrays.
// 3 scalars for broadcasting dx, dy, and status across the workgroup.
// Total: ~5 × 16 × 4 + 12 = 332 bytes at WG_SIZE=16.
//
//
// COMPILE-TIME CONSTANTS (substituted by gpu/klt.rs)
// ──────────────────────────────────────────────────
//   {{HALF}}     — window half-size W
//   {{SIDE}}     — 2*W+1
//   {{PATCH}}    — SIDE²
//   {{WG_SIZE}}  — threads per workgroup (must be power of 2)

// ---------------------------------------------------------------------------
// Status codes — must match TrackStatus in klt.rs
// ---------------------------------------------------------------------------

const STATUS_TRACKED: u32 = 0u;
const STATUS_LOST:    u32 = 1u;
const STATUS_OOB:     u32 = 2u;
const LOST_SENTINEL:  f32 = 1.0e20;

// ---------------------------------------------------------------------------
// Bindings (same layout as klt.wgsl — shared bind group layout)
// ---------------------------------------------------------------------------

@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var curr_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read>       features:      array<GpuKltFeature>;
@group(0) @binding(3) var<storage, read_write> displacements: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> results:       array<GpuTrackResult>;
@group(0) @binding(5) var<uniform>             params:        KltParams;
@group(0) @binding(6) var<storage, read_write> t_buf:         array<f32>;
@group(0) @binding(7) var<storage, read_write> gx_buf:        array<f32>;
@group(0) @binding(8) var<storage, read_write> gy_buf:        array<f32>;
@group(0) @binding(9) var<storage, read_write> h_inv:         array<vec4<f32>>;

// ---------------------------------------------------------------------------
// Structs (identical to klt.wgsl)
// ---------------------------------------------------------------------------

struct GpuKltFeature {
    x: f32, y: f32, score: f32, _pad: f32,
}

struct GpuTrackResult {
    x: f32, y: f32, status: u32, _pad: u32,
}

struct KltParams {
    n_features:     u32,
    max_iterations: u32,
    epsilon_sq:     f32,
    level:          u32,
    level_scale:    f32,
    img0_width:     u32,
    img0_height:    u32,
    _pad:           u32,
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

var<workgroup> sh_a: array<f32, {{WG_SIZE}}>;   // reduction scratch A
var<workgroup> sh_b: array<f32, {{WG_SIZE}}>;   // reduction scratch B
var<workgroup> sh_c: array<f32, {{WG_SIZE}}>;   // reduction scratch C
var<workgroup> sh_dx: f32;                       // broadcast dx
var<workgroup> sh_dy: f32;                       // broadcast dy
var<workgroup> sh_stop: u32;                     // 0=continue, 1=lost, 2=converged

// ---------------------------------------------------------------------------
// Bilinear interpolation (identical to klt.wgsl)
// ---------------------------------------------------------------------------

fn bilinear(tex: texture_2d<f32>, x: f32, y: f32) -> f32 {
    let dims = textureDimensions(tex);
    let max_x = f32(dims.x) - 1.0;
    let max_y = f32(dims.y) - 1.0;
    let cx = clamp(x, 0.0, max_x);
    let cy = clamp(y, 0.0, max_y);
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let x1 = min(x0 + 1, i32(dims.x) - 1);
    let y1 = min(y0 + 1, i32(dims.y) - 1);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let v00 = textureLoad(tex, vec2<i32>(x0, y0), 0).r;
    let v10 = textureLoad(tex, vec2<i32>(x1, y0), 0).r;
    let v01 = textureLoad(tex, vec2<i32>(x0, y1), 0).r;
    let v11 = textureLoad(tex, vec2<i32>(x1, y1), 0).r;
    return v00 * (1.0 - fx) * (1.0 - fy)
         + v10 * fx          * (1.0 - fy)
         + v01 * (1.0 - fx) * fy
         + v11 * fx          * fy;
}

// ---------------------------------------------------------------------------
// Shared-memory tree reduction (3-wide: reduces sh_a, sh_b, sh_c in lockstep)
// ---------------------------------------------------------------------------

fn reduce_abc(tid: u32) {
    // WG_SIZE must be a power of 2.
    for (var stride = {{WG_SIZE}}u / 2u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            sh_a[tid] += sh_a[tid + stride];
            sh_b[tid] += sh_b[tid + stride];
            sh_c[tid] += sh_c[tid + stride];
        }
        workgroupBarrier();
    }
}

// Reduce only sh_a and sh_b (for Phase 2 residual).
fn reduce_ab(tid: u32) {
    for (var stride = {{WG_SIZE}}u / 2u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            sh_a[tid] += sh_a[tid + stride];
            sh_b[tid] += sh_b[tid + stride];
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size({{WG_SIZE}}, 1, 1)
fn track_level(
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let feat_idx = wid.x;
    let tid      = lid.x;

    if feat_idx >= params.n_features { return; }

    // --- Load feature and displacement ---
    let feat = features[feat_idx];
    let disp = displacements[feat_idx];

    // --- Check lost sentinel (uniform branch — all threads take same path) ---
    if disp.x >= LOST_SENTINEL {
        if tid == 0u && params.level == 0u {
            results[feat_idx] = GpuTrackResult(feat.x, feat.y, STATUS_LOST, 0u);
        }
        return;
    }

    let fx = feat.x * params.level_scale;
    let fy = feat.y * params.level_scale;
    let buf_base = feat_idx * {{PATCH}}u;
    let half = {{HALF}};

    // ===================================================================
    // IC Phase 1: cooperative template precompute + Hessian accumulation
    // ===================================================================

    var p_h00 = 0.0;
    var p_h01 = 0.0;
    var p_h11 = 0.0;

    // Strided pixel iteration: thread tid handles pixels tid, tid+WG, tid+2*WG, ...
    var pidx = tid;
    while pidx < {{PATCH}}u {
        let py = i32(pidx / {{SIDE}}u) - half;
        let px = i32(pidx % {{SIDE}}u) - half;
        let tx = fx + f32(px);
        let ty = fy + f32(py);

        let t_val = bilinear(prev_tex, tx, ty);
        let gx = 0.5 * (bilinear(prev_tex, tx + 1.0, ty)
                       - bilinear(prev_tex, tx - 1.0, ty));
        let gy = 0.5 * (bilinear(prev_tex, tx, ty + 1.0)
                       - bilinear(prev_tex, tx, ty - 1.0));

        let si = buf_base + pidx;
        t_buf[si]  = t_val;
        gx_buf[si] = gx;
        gy_buf[si] = gy;

        p_h00 += gx * gx;
        p_h01 += gx * gy;
        p_h11 += gy * gy;

        pidx += {{WG_SIZE}}u;
    }

    // Reduce Hessian: sh_a=h00, sh_b=h01, sh_c=h11
    sh_a[tid] = p_h00;
    sh_b[tid] = p_h01;
    sh_c[tid] = p_h11;
    workgroupBarrier();
    reduce_abc(tid);

    // Thread 0: invert Hessian, check singularity, broadcast status + dx/dy.
    if tid == 0u {
        let h00 = sh_a[0];
        let h01 = sh_b[0];
        let h11 = sh_c[0];
        let det = h00 * h11 - h01 * h01;

        if abs(det) < 1.0e-6 {
            sh_stop = 1u;  // lost
            displacements[feat_idx] = vec2<f32>(LOST_SENTINEL, LOST_SENTINEL);
            if params.level == 0u {
                results[feat_idx] = GpuTrackResult(
                    feat.x + disp.x, feat.y + disp.y, STATUS_LOST, 0u);
            }
        } else {
            sh_stop = 0u;
            let inv_det = 1.0 / det;
            h_inv[feat_idx] = vec4<f32>(
                inv_det * h11, -inv_det * h01, inv_det * h00, 0.0);
        }
        sh_dx = disp.x;
        sh_dy = disp.y;
    }
    workgroupBarrier();

    if sh_stop != 0u { return; }

    // All threads read shared Hessian inverse and initial displacement.
    let hv = h_inv[feat_idx];
    var dx = sh_dx;
    var dy = sh_dy;

    // ===================================================================
    // IC Phase 2: cooperative iterative solver
    // ===================================================================

    for (var iter: u32 = 0u; iter < params.max_iterations; iter++) {
        // Each thread accumulates partial residual over its pixel subset.
        var p_b0 = 0.0;
        var p_b1 = 0.0;

        pidx = tid;
        while pidx < {{PATCH}}u {
            let py = i32(pidx / {{SIDE}}u) - half;
            let px = i32(pidx % {{SIDE}}u) - half;
            let wx = fx + dx + f32(px);
            let wy = fy + dy + f32(py);

            let i_val = bilinear(curr_tex, wx, wy);
            let si = buf_base + pidx;
            let e = t_buf[si] - i_val;
            p_b0 += gx_buf[si] * e;
            p_b1 += gy_buf[si] * e;

            pidx += {{WG_SIZE}}u;
        }

        // Reduce residuals: sh_a=b0, sh_b=b1.
        sh_a[tid] = p_b0;
        sh_b[tid] = p_b1;
        workgroupBarrier();
        reduce_ab(tid);

        // Thread 0: solve for delta, update displacement, check convergence.
        if tid == 0u {
            let b0 = sh_a[0];
            let b1 = sh_b[0];
            let delta_x = hv.x * b0 + hv.y * b1;
            let delta_y = hv.y * b0 + hv.z * b1;
            dx += delta_x;
            dy += delta_y;
            sh_dx = dx;
            sh_dy = dy;

            if delta_x * delta_x + delta_y * delta_y < params.epsilon_sq {
                sh_stop = 2u;  // converged
            }
        }
        workgroupBarrier();

        // All threads sync updated displacement.
        dx = sh_dx;
        dy = sh_dy;
        if sh_stop == 2u { break; }
    }

    // ===================================================================
    // Write results (thread 0 only)
    // ===================================================================

    if tid == 0u {
        if params.level == 0u {
            let new_x = feat.x + dx;
            let new_y = feat.y + dy;
            let oob = new_x < 0.0 || new_x >= f32(params.img0_width)
                   || new_y < 0.0 || new_y >= f32(params.img0_height);
            let status = select(STATUS_TRACKED, STATUS_OOB, oob);
            results[feat_idx] = GpuTrackResult(new_x, new_y, status, 0u);
            displacements[feat_idx] = vec2<f32>(dx, dy);
        } else {
            displacements[feat_idx] = vec2<f32>(dx * 2.0, dy * 2.0);
        }
    }
}
