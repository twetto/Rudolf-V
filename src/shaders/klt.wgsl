// klt.wgsl — Pyramidal KLT optical flow tracker (Inverse Compositional).
//
// Each thread tracks one feature. Features are processed in parallel across
// the workgroup; there is no cooperation between threads (no shared memory,
// no barriers). The GPU's occupancy comes from running hundreds of features
// simultaneously, each following the same instruction stream.
//
//
// ALGORITHM: INVERSE COMPOSITIONAL (Baker & Matthews, 2004)
// ──────────────────────────────────────────────────────────
// The CPU lk_inverse_compositional() in klt.rs is ported here verbatim.
// Brief recap:
//
//   IC precomputes template values T(x) and their gradients ∇T once per
//   level, along with the constant Hessian H = Σ ∇T·∇Tᵀ and its inverse.
//   Each iteration only requires one bilinear lookup per patch pixel
//   (the warped current-frame pixel I(x+d)).
//
//   Cost per level:
//     Setup:     (2W+1)² × 3 bilinear lookups (template + 2 gradient)
//     Per iter:  (2W+1)² × 1 bilinear lookup (warped only)
//     vs FA:     (2W+1)² × 5 per iteration
//
//   This is exactly the reason the GAP8 C tracker uses IC.
//
//
// MULTI-PASS DISPATCH
// ────────────────────
// `track_level` is dispatched once per pyramid level (coarse → fine).
// Between passes the CPU multiplies all displacements by 2, propagating
// the coarse estimate to the finer level. The final pass (level == 0)
// writes the tracked position and status to the results buffer.
//
//
// LOST SENTINEL
// ─────────────
// If the Hessian is singular (flat region, no gradient), the feature is
// considered lost. We write LOST_SENTINEL (1e20) to the displacement
// buffer so subsequent finer-level passes can skip this feature cheaply.
//
//
// PATCH BUFFERS IN STORAGE (not private) ADDRESS SPACE
// ─────────────────────────────────────────────────────
// The three per-feature arrays (template values, gx, gy) are stored in
// global storage buffers indexed as [feat_idx * PATCH + pixel_idx].
// This avoids per-thread private memory pressure — critical for
// VideoCore VI (RPi 4) which has severely limited private memory and
// silently corrupts spilled arrays. It also enables future wavefront-
// per-patch cooperation (§4) where multiple threads share patch data.
//
//
// NEW WGSL CONCEPTS
// ─────────────────
// - `select(a, b, cond)` — branchless ternary: returns b if cond, else a.
//   Used for the bounds check on the final status.
// - Function-scope `var` arrays — unlike `var<workgroup>`, these are
//   private to each thread and do not require barriers.
// - `textureDimensions(tex)` — returns the size of the texture as vec2<u32>.
//   Used here to get current-level image dimensions for bounds clamping.

// ---------------------------------------------------------------------------
// Compile-time constants (substituted by gpu/klt.rs)
// ---------------------------------------------------------------------------
//
//   {{HALF}}     — window half-size W (e.g. 7 for a 15×15 patch)
//   {{SIDE}}     — 2*W+1 (e.g. 15)
//   {{PATCH}}    — SIDE² (e.g. 225)
//   {{WG_SIZE}}  — 1-D workgroup size (e.g. 64)

// ---------------------------------------------------------------------------
// Status codes — must match TrackStatus in klt.rs
// ---------------------------------------------------------------------------

const STATUS_TRACKED:    u32 = 0u;
const STATUS_LOST:       u32 = 1u;
const STATUS_OOB:        u32 = 2u;

/// Displacement magnitude above this value means the feature was marked lost
/// at a coarser level. Any real displacement is well below 1e10 pixels.
const LOST_SENTINEL: f32 = 1.0e20;

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Previous frame, current pyramid level (R32Float, values in [0, 255]).
@group(0) @binding(0) var prev_tex: texture_2d<f32>;

/// Current frame, current pyramid level.
@group(0) @binding(1) var curr_tex: texture_2d<f32>;

/// Input feature positions from FAST (or CPU). Read-only for KLT.
@group(0) @binding(2) var<storage, read>       features:     array<GpuKltFeature>;

/// Per-feature displacement estimate, updated each level pass.
/// On entry to the coarsest level: all zero (zeroed by the caller).
/// After each non-final level: scaled by ×2 by the caller before the
/// next finer-level dispatch.
@group(0) @binding(3) var<storage, read_write> displacements: array<vec2<f32>>;

/// Final tracking results, written only on the level-0 pass.
@group(0) @binding(4) var<storage, read_write> results:       array<GpuTrackResult>;

/// Constant parameters for this pass.
@group(0) @binding(5) var<uniform>             params:        KltParams;

/// Per-feature patch buffers in global memory (storage, not private).
/// Indexed as [feat_idx * {{PATCH}}u + patch_pixel_idx].
/// Using storage buffers instead of private arrays avoids per-thread stack
/// pressure — critical for VideoCore VI (RPi 4) which silently corrupts
/// spilled private arrays. Also enables future wavefront-per-patch (§4)
/// where multiple threads cooperate on one patch.
@group(0) @binding(6) var<storage, read_write> t_buf:  array<f32>;
@group(0) @binding(7) var<storage, read_write> gx_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> gy_buf: array<f32>;

/// Per-feature Hessian inverse: (ih00, ih01, ih11) packed as vec4 (w unused).
/// Stored in global memory to reduce register pressure during Phase 2 —
/// VideoCore VI corrupts values when too many f32s are live across a loop
/// containing storage buffer reads.
@group(0) @binding(9) var<storage, read_write> h_inv:  array<vec4<f32>>;

// ---------------------------------------------------------------------------
// Structs (repr(C) equivalents in Rust)
// ---------------------------------------------------------------------------

struct GpuKltFeature {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,
}

struct GpuTrackResult {
    x:      f32,
    y:      f32,
    status: u32,   // STATUS_TRACKED / STATUS_LOST / STATUS_OOB
    _pad:   u32,
}

struct KltParams {
    n_features:   u32,
    max_iterations: u32,
    epsilon_sq:   f32,   // convergence threshold squared (epsilon²)
    level:        u32,   // 0 = finest (level 0 pass writes final results)
    level_scale:  f32,   // 1.0 / (1 << level) — feature → level coords
    img0_width:   u32,   // level-0 image width  (for final OOB check)
    img0_height:  u32,   // level-0 image height
    _pad:         u32,
}

// ---------------------------------------------------------------------------
// Bilinear interpolation (clamp-to-edge border, matching CPU implementation)
// ---------------------------------------------------------------------------

/// Sample a single-channel float texture at a sub-pixel (x, y) coordinate.
///
/// Uses bilinear interpolation with clamp-to-edge border handling.
/// Returns values in [0, 255] (the R32Float textures store raw pixel values).
///
/// This mirrors `interpolate_bilinear` in image.rs — same formula, same
/// border behaviour.
fn bilinear(tex: texture_2d<f32>, x: f32, y: f32) -> f32 {
    let dims = textureDimensions(tex);
    let max_x = f32(dims.x) - 1.0;
    let max_y = f32(dims.y) - 1.0;

    // Clamp to [0, dim-1].
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
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size({{WG_SIZE}}, 1, 1)
fn track_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let feat_idx = gid.x;
    if feat_idx >= params.n_features { return; }

    // --- Check lost sentinel ---
    // A prior coarse-level pass marked this feature as lost.
    // All threads must execute the same code path on the same set of
    // iterations (GPU control flow), but we still skip to avoid
    // accumulating garbage into the result buffer.
    let feat = features[feat_idx];
    var disp = displacements[feat_idx];
    if disp.x >= LOST_SENTINEL {
        if params.level == 0u {
            results[feat_idx] = GpuTrackResult(feat.x, feat.y, STATUS_LOST, 0u);
        }
        return;
    }

    var dx = disp.x;
    var dy = disp.y;

    // Scale feature position to this pyramid level.
    // level_scale = 1.0 / (1 << level), so:
    //   level 0: scale = 1.0  → no change
    //   level 1: scale = 0.5  → half resolution
    //   level 2: scale = 0.25 → quarter resolution
    let fx = feat.x * params.level_scale;
    let fy = feat.y * params.level_scale;

    // -----------------------------------------------------------------------
    // IC Phase 1: precompute template patch, gradients, and Hessian.
    // All of these are constant across iterations (the IC advantage).
    // -----------------------------------------------------------------------

    // Storage buffer base offset for this feature's patch data.
    let buf_base = feat_idx * {{PATCH}}u;

    var h00 = 0.0;
    var h01 = 0.0;
    var h11 = 0.0;

    let half = {{HALF}};  // window half-size (e.g. 7)

    var bidx: u32 = 0u;
    for (var py: i32 = -half; py <= half; py++) {
        for (var px: i32 = -half; px <= half; px++) {
            let tx = fx + f32(px);
            let ty = fy + f32(py);

            let t_val = bilinear(prev_tex, tx, ty);
            // Central-difference gradient at template position.
            let gx = 0.5 * (bilinear(prev_tex, tx + 1.0, ty)
                          - bilinear(prev_tex, tx - 1.0, ty));
            let gy = 0.5 * (bilinear(prev_tex, tx, ty + 1.0)
                          - bilinear(prev_tex, tx, ty - 1.0));

            let si = buf_base + bidx;
            t_buf[si]  = t_val;
            gx_buf[si] = gx;
            gy_buf[si] = gy;

            // Accumulate symmetric 2×2 Hessian.
            h00 += gx * gx;
            h01 += gx * gy;
            h11 += gy * gy;

            bidx++;
        }
    }

    // Invert H. det(H) = h00*h11 - h01².
    // A near-zero determinant means the patch has no directional gradient
    // (flat region, aperture problem) — the feature is untrakable.
    let det = h00 * h11 - h01 * h01;
    if abs(det) < 1.0e-6 {
        // Mark lost via sentinel. Later levels will skip this thread.
        displacements[feat_idx] = vec2<f32>(LOST_SENTINEL, LOST_SENTINEL);
        if params.level == 0u {
            results[feat_idx] = GpuTrackResult(feat.x + dx, feat.y + dy, STATUS_LOST, 0u);
        }
        return;
    }
    let inv_det = 1.0 / det;
    // H⁻¹ = (1/det) * [[h11, -h01], [-h01, h00]]
    // Store to global memory immediately — keeping these live in registers
    // across the Phase 2 loop causes V3DV (VideoCore VI) to corrupt spilled
    // values. The store lets the compiler kill these registers before Phase 2.
    h_inv[feat_idx] = vec4<f32>(inv_det * h11, -inv_det * h01, inv_det * h00, 0.0);

    // -----------------------------------------------------------------------
    // IC Phase 2: iterate — only warped pixel needs to be re-sampled.
    // -----------------------------------------------------------------------

    // Reload Hessian inverse from storage — fresh let bindings with minimal
    // live range keep register pressure below V3DV's spill threshold.
    let hv = h_inv[feat_idx];
    let ih00 = hv.x;
    let ih01 = hv.y;
    let ih11 = hv.z;

    for (var iter: u32 = 0u; iter < params.max_iterations; iter++) {
        var b0 = 0.0;
        var b1 = 0.0;

        bidx = 0u;
        for (var py: i32 = -half; py <= half; py++) {
            for (var px: i32 = -half; px <= half; px++) {
                // Warped position in current frame.
                let wx = fx + dx + f32(px);
                let wy = fy + dy + f32(py);

                let i_val = bilinear(curr_tex, wx, wy);
                let si = buf_base + bidx;
                let e = t_buf[si] - i_val;

                // Accumulate right-hand side b = Σ ∇T * e.
                b0 += gx_buf[si] * e;
                b1 += gy_buf[si] * e;

                bidx++;
            }
        }

        // Solve H * delta = b via the precomputed H⁻¹.
        let delta_x = ih00 * b0 + ih01 * b1;
        let delta_y = ih01 * b0 + ih11 * b1;

        dx += delta_x;
        dy += delta_y;

        // Convergence: stop when |delta|² < ε².
        if delta_x * delta_x + delta_y * delta_y < params.epsilon_sq {
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Write results
    // -----------------------------------------------------------------------

    if params.level == 0u {
        // Final level: compute absolute position and write result.
        let new_x = feat.x + dx;
        let new_y = feat.y + dy;

        // Out-of-bounds check at level-0 resolution.
        let oob = new_x < 0.0 || new_x >= f32(params.img0_width)
               || new_y < 0.0 || new_y >= f32(params.img0_height);

        let status = select(STATUS_TRACKED, STATUS_OOB, oob);
        results[feat_idx] = GpuTrackResult(new_x, new_y, status, 0u);
        // Also update displacement (useful if the caller wants to chain passes).
        displacements[feat_idx] = vec2<f32>(dx, dy);
    } else {
        // Non-final level: scale displacement by ×2 for the next finer level.
        // The CPU multiplies by 2 between levels; here we bake it in to avoid
        // a separate scaling pass.
        displacements[feat_idx] = vec2<f32>(dx * 2.0, dy * 2.0);
    }
}
