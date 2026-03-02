// klt.rs — KLT (Kanade-Lucas-Tomasi) pyramidal optical flow tracker.
//
// Implements two formulations of Lucas-Kanade:
//
// 1. FORWARD ADDITIVE (vilib's approach):
//    Gradients evaluated at the warped position in the current frame
//    each iteration → Hessian recomputed every iteration.
//    More robust to large displacements.
//
// 2. INVERSE COMPOSITIONAL (Baker & Matthews, 2004):
//    Gradients evaluated at the template position in the previous frame
//    once → Hessian is constant across iterations. Only the error image
//    is recomputed. Cheaper per iteration. This is what your C tracker
//    on the GAP8 uses.
//
// Both share the same coarse-to-fine pyramid strategy.
//
// The IC path is optimized for throughput:
// - Constant bilinear weights (integer patch offsets → frac is invariant)
// - Row-pointer access (one stride multiply per row, not per pixel)
// - Two-phase iteration (extract → accumulate over contiguous buffers)
// - Hoisted scratch buffers (zero per-feature allocation)
//
// NEW RUST CONCEPTS:
// - Enums with variants (TrackStatus, LkMethod) — Rust enums can carry
//   data, making them algebraic data types (tagged unions).
// - Borrowing two pyramids simultaneously (&prev_pyramid, &curr_pyramid).
// - `match` on enums to dispatch between algorithm variants.

use crate::fast::Feature;
use crate::image::{interpolate_bilinear, interpolate_bilinear_unchecked, Image};
use crate::pyramid::Pyramid;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Status of a tracked feature after one frame-to-frame tracking pass.
///
/// This is a Rust enum — like a C enum but the compiler enforces that
/// you handle every variant in match expressions (exhaustive matching).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackStatus {
    /// Successfully tracked to a new position.
    Tracked,
    /// Lost: the iterative solver diverged or the Hessian was singular.
    Lost,
    /// The tracked position fell outside the image bounds.
    OutOfBounds,
}

/// Lucas-Kanade algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LkMethod {
    /// Forward additive: gradients at warped position in current frame.
    /// Hessian recomputed every iteration. vilib's approach.
    ForwardAdditive,
    /// Inverse compositional: gradients at template position in previous
    /// frame. Hessian precomputed once. Your GAP8 C tracker's approach.
    InverseCompositional,
}

/// A feature with its tracking status after a track() call.
#[derive(Debug, Clone)]
pub struct TrackedFeature {
    /// The feature with updated (x, y) position.
    /// If status != Tracked, the position may be unreliable.
    pub feature: Feature,
    /// Tracking outcome.
    pub status: TrackStatus,
}

/// Pre-allocated scratch buffers for KLT tracking.
///
/// Eliminates per-feature heap allocation. The IC precompute needs
/// (t_buf, gx_buf, gy_buf) and the two-phase iteration adds warped_buf.
/// All are sized to the largest patch: (2 * window_size + 1)².
///
/// GPU EQUIVALENT: Pre-allocated storage buffers bound once per dispatch,
/// not re-created per workgroup invocation.
pub struct KltScratch {
    t_buf: Vec<f32>,
    gx_buf: Vec<f32>,
    gy_buf: Vec<f32>,
    warped_buf: Vec<f32>,
}

impl KltScratch {
    /// Create scratch buffers for the given window half-size.
    pub fn new(window_size: usize) -> Self {
        let patch_size = (2 * window_size + 1) * (2 * window_size + 1);
        KltScratch {
            t_buf: vec![0.0; patch_size],
            gx_buf: vec![0.0; patch_size],
            gy_buf: vec![0.0; patch_size],
            warped_buf: vec![0.0; patch_size],
        }
    }

    /// Ensure buffers are large enough (no-op if already sized).
    #[inline]
    fn ensure_size(&mut self, patch_size: usize) {
        if self.t_buf.len() < patch_size {
            self.t_buf.resize(patch_size, 0.0);
            self.gx_buf.resize(patch_size, 0.0);
            self.gy_buf.resize(patch_size, 0.0);
            self.warped_buf.resize(patch_size, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

/// Pyramidal KLT optical flow tracker.
///
/// Configuration follows vilib's defaults:
/// - window_size (half-size): 11 → 23×23 patch
/// - max_iterations: 30 per pyramid level
/// - epsilon: 0.01 pixels convergence threshold
/// - max_levels: matched to pyramid depth
pub struct KltTracker {
    /// Patch half-size. The actual patch is (2*window_size + 1)².
    /// vilib uses 11 (23×23 patch). Your C tracker used 4 (9×9 patch)
    /// for the tiny 64×64 GAP8 images.
    pub window_size: usize,
    /// Maximum Gauss-Newton iterations per pyramid level.
    pub max_iterations: usize,
    /// Convergence threshold in pixels. Iteration stops when
    /// |delta| < epsilon.
    pub epsilon: f32,
    /// Number of pyramid levels to use. Should match or be ≤
    /// the pyramid's num_levels().
    pub max_levels: usize,
    /// LK algorithm variant.
    pub method: LkMethod,
}

impl KltTracker {
    /// Create a tracker with the given parameters (forward additive by default).
    pub fn new(
        window_size: usize,
        max_iterations: usize,
        epsilon: f32,
        max_levels: usize,
    ) -> Self {
        KltTracker {
            window_size,
            max_iterations,
            epsilon,
            max_levels,
            method: LkMethod::ForwardAdditive,
        }
    }

    /// Create a tracker with a specific LK method.
    pub fn with_method(
        window_size: usize,
        max_iterations: usize,
        epsilon: f32,
        max_levels: usize,
        method: LkMethod,
    ) -> Self {
        KltTracker {
            window_size,
            max_iterations,
            epsilon,
            max_levels,
            method,
        }
    }

    /// Track features from the previous frame to the current frame.
    ///
    /// Convenience wrapper that allocates a temporary scratch buffer.
    /// For per-frame use in a pipeline, prefer `track_into_opt` with
    /// a persistent `KltScratch` to avoid allocation each call.
    pub fn track(
        &self,
        prev_pyramid: &Pyramid,
        curr_pyramid: &Pyramid,
        features: &[Feature],
    ) -> Vec<TrackedFeature> {
        let mut results = Vec::with_capacity(features.len());
        let mut scratch = KltScratch::new(self.window_size);
        self.track_into_opt(prev_pyramid, curr_pyramid, features, &mut results, &mut scratch);
        results
    }

    /// Track features into a pre-allocated result buffer.
    ///
    /// Convenience wrapper that allocates a temporary scratch buffer.
    /// For per-frame use in a pipeline, prefer `track_into_opt`.
    pub fn track_into(
        &self,
        prev_pyramid: &Pyramid,
        curr_pyramid: &Pyramid,
        features: &[Feature],
        results: &mut Vec<TrackedFeature>,
    ) {
        let mut scratch = KltScratch::new(self.window_size);
        self.track_into_opt(prev_pyramid, curr_pyramid, features, results, &mut scratch);
    }

    /// Track features using pre-allocated scratch and result buffers.
    ///
    /// This is the primary entry point for production use. Both `results`
    /// and `scratch` persist across frames, eliminating all per-frame and
    /// per-feature allocation.
    ///
    /// GPU EQUIVALENT: Writing tracking results into a pre-allocated
    /// storage buffer bound to the compute shader dispatch.
    pub fn track_into_opt(
        &self,
        prev_pyramid: &Pyramid,
        curr_pyramid: &Pyramid,
        features: &[Feature],
        results: &mut Vec<TrackedFeature>,
        scratch: &mut KltScratch,
    ) {
        let num_levels = self.max_levels
            .min(prev_pyramid.num_levels())
            .min(curr_pyramid.num_levels());

        let side = 2 * self.window_size + 1;
        scratch.ensure_size(side * side);

        results.clear();
        results.reserve(features.len());

        for feat in features {
            results.push(self.track_single(
                prev_pyramid, curr_pyramid, feat, num_levels, scratch,
            ));
        }
    }

    /// Track a single feature through the pyramid, coarse-to-fine.
    fn track_single(
        &self,
        prev_pyr: &Pyramid,
        curr_pyr: &Pyramid,
        feature: &Feature,
        num_levels: usize,
        scratch: &mut KltScratch,
    ) -> TrackedFeature {
        // Start with zero displacement at the coarsest level.
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        for level in (0..num_levels).rev() {
            let prev_img = &prev_pyr.levels[level];
            let curr_img = &curr_pyr.levels[level];

            // Scale feature position to this pyramid level.
            let scale = 1.0 / (1u32 << level) as f32;
            let feat_x = feature.x * scale;
            let feat_y = feature.y * scale;

            // No explicit bounds check here — bilinear interpolation
            // clamps to image borders, so the iteration won't crash.
            // If the patch lands mostly outside the image, gradients
            // will be degenerate → singular Hessian → returns Lost.
            // This matches vilib's approach and your C tracker.

            // Run iterative Lucas-Kanade at this level.
            let result = match self.method {
                LkMethod::ForwardAdditive => {
                    self.lk_forward_additive(prev_img, curr_img, feat_x, feat_y, dx, dy)
                }
                LkMethod::InverseCompositional => {
                    self.lk_inverse_compositional(prev_img, curr_img, feat_x, feat_y, dx, dy, scratch)
                }
            };

            match result {
                LkResult::Converged(new_dx, new_dy) | LkResult::MaxIter(new_dx, new_dy) => {
                    dx = new_dx;
                    dy = new_dy;
                }
                LkResult::Singular => {
                    return TrackedFeature {
                        feature: Feature {
                            x: feature.x + dx / scale,
                            y: feature.y + dy / scale,
                            score: feature.score,
                            level: feature.level,
                            id: feature.id,
                        },
                        status: TrackStatus::Lost,
                    };
                }
            }

            // Propagate displacement to the next finer level: d *= 2.
            if level > 0 {
                dx *= 2.0;
                dy *= 2.0;
            }
        }

        // Final tracked position at level 0.
        let new_x = feature.x + dx;
        let new_y = feature.y + dy;

        // Final bounds check at full resolution.
        let w = prev_pyr.levels[0].width() as f32;
        let h = prev_pyr.levels[0].height() as f32;
        let status = if new_x >= 0.0 && new_x < w && new_y >= 0.0 && new_y < h {
            TrackStatus::Tracked
        } else {
            TrackStatus::OutOfBounds
        };

        TrackedFeature {
            feature: Feature {
                x: new_x,
                y: new_y,
                score: feature.score,
                level: feature.level,
                id: feature.id,
            },
            status,
        }
    }

    // =====================================================================
    // Forward additive (reference implementation)
    // =====================================================================

    /// Iterative forward-additive Lucas-Kanade at a single pyramid level.
    ///
    /// Gradients are evaluated at the warped position (feat + d) in the
    /// current frame each iteration. The Hessian is recomputed every
    /// iteration because the gradient changes as d changes.
    fn lk_forward_additive(
        &self,
        prev_img: &Image<f32>,
        curr_img: &Image<f32>,
        feat_x: f32,
        feat_y: f32,
        mut dx: f32,
        mut dy: f32,
    ) -> LkResult {
        let half_i = self.window_size as isize;

        // Check if template window is in bounds (constant across iterations).
        let tmpl_in_bounds = (feat_x - half_i as f32) >= 0.0
            && (feat_y - half_i as f32) >= 0.0
            && (feat_x + half_i as f32) < prev_img.width() as f32
            && (feat_y + half_i as f32) < prev_img.height() as f32;

        for _iter in 0..self.max_iterations {
            // Accumulators for the 2×2 Hessian and 2×1 right-hand side.
            let mut h00 = 0.0f32;
            let mut h01 = 0.0f32;
            let mut h11 = 0.0f32;
            let mut b0 = 0.0f32;
            let mut b1 = 0.0f32;

            // Check if warped window + gradient margin is in bounds.
            let warp_in_bounds = (feat_x + dx - half_i as f32 - 1.0) >= 0.0
                && (feat_y + dy - half_i as f32 - 1.0) >= 0.0
                && (feat_x + dx + half_i as f32 + 1.0) < curr_img.width() as f32
                && (feat_y + dy + half_i as f32 + 1.0) < curr_img.height() as f32;

            let both_in_bounds = tmpl_in_bounds && warp_in_bounds;

            for py in -half_i..=half_i {
                for px in -half_i..=half_i {
                    let px_f = px as f32;
                    let py_f = py as f32;

                    let (t_val, i_val, gx, gy);

                    if both_in_bounds {
                        // SAFETY: both template and warped windows verified in bounds.
                        unsafe {
                            t_val = interpolate_bilinear_unchecked(
                                prev_img, feat_x + px_f, feat_y + py_f);

                            let wx = feat_x + dx + px_f;
                            let wy = feat_y + dy + py_f;
                            i_val = interpolate_bilinear_unchecked(curr_img, wx, wy);

                            gx = 0.5
                                * (interpolate_bilinear_unchecked(curr_img, wx + 1.0, wy)
                                    - interpolate_bilinear_unchecked(curr_img, wx - 1.0, wy));
                            gy = 0.5
                                * (interpolate_bilinear_unchecked(curr_img, wx, wy + 1.0)
                                    - interpolate_bilinear_unchecked(curr_img, wx, wy - 1.0));
                        }
                    } else {
                        t_val = interpolate_bilinear(
                            prev_img, feat_x + px_f, feat_y + py_f);

                        let wx = feat_x + dx + px_f;
                        let wy = feat_y + dy + py_f;
                        i_val = interpolate_bilinear(curr_img, wx, wy);

                        gx = 0.5
                            * (interpolate_bilinear(curr_img, wx + 1.0, wy)
                                - interpolate_bilinear(curr_img, wx - 1.0, wy));
                        gy = 0.5
                            * (interpolate_bilinear(curr_img, wx, wy + 1.0)
                                - interpolate_bilinear(curr_img, wx, wy - 1.0));
                    }

                    let e = t_val - i_val;

                    // Accumulate Hessian (symmetric, so h10 = h01).
                    h00 += gx * gx;
                    h01 += gx * gy;
                    h11 += gy * gy;

                    // Accumulate right-hand side.
                    b0 += gx * e;
                    b1 += gy * e;
                }
            }

            // Solve the 2×2 system: H * delta = b.
            // det(H) = h00*h11 - h01²
            let det = h00 * h11 - h01 * h01;
            if det.abs() < 1e-6 {
                return LkResult::Singular;
            }
            let inv_det = 1.0 / det;

            // delta = H^{-1} * b
            let delta_x = inv_det * (h11 * b0 - h01 * b1);
            let delta_y = inv_det * (h00 * b1 - h01 * b0);

            dx += delta_x;
            dy += delta_y;

            // Convergence check.
            if delta_x * delta_x + delta_y * delta_y < self.epsilon * self.epsilon {
                return LkResult::Converged(dx, dy);
            }
        }

        LkResult::MaxIter(dx, dy)
    }

    // =====================================================================
    // Inverse compositional (optimized)
    // =====================================================================

    /// Optimized inverse-compositional Lucas-Kanade at a single level.
    ///
    /// Baker & Matthews (2004): gradients are evaluated at the template
    /// position in the previous frame. Since the template doesn't change,
    /// the Hessian H = J^T * J is constant across iterations — computed
    /// once, then only the error image is recomputed per iteration.
    ///
    /// Optimizations over a naive implementation:
    ///
    /// - CONSTANT BILINEAR WEIGHTS: patch offsets (px, py) are integers, so
    ///   frac(feat_x + px) = frac(feat_x) for all px. The four bilinear
    ///   weights are computed once per feature, not per pixel. This is NOT
    ///   an approximation — it's algebraically exact.
    ///
    /// - ROW-POINTER ACCESS: pre-compute raw pointers to image rows, then
    ///   index horizontally with ptr.add(x). Eliminates y * stride
    ///   multiplication from the inner loop.
    ///
    /// - TWO-PHASE ITERATION:
    ///     Phase 1: extract all warped pixels into contiguous warped_buf.
    ///     Phase 2: accumulate b0 += gx*e, b1 += gy*e over contiguous data.
    ///   Phase 2 is a paired dot-product — ready for SIMD.
    ///
    /// - HOISTED SCRATCH: t_buf, gx_buf, gy_buf, warped_buf are pre-allocated
    ///   in KltScratch and reused across all features and pyramid levels.
    fn lk_inverse_compositional(
        &self,
        prev_img: &Image<f32>,
        curr_img: &Image<f32>,
        feat_x: f32,
        feat_y: f32,
        mut dx: f32,
        mut dy: f32,
        scratch: &mut KltScratch,
    ) -> LkResult {
        let half = self.window_size as isize;
        let side = (2 * self.window_size + 1) as isize;
        let patch_size = (side * side) as usize;

        // =================================================================
        // PRECOMPUTE: template values, gradients, Hessian
        // =================================================================

        // Constant bilinear weights for template sampling.
        let fx = feat_x - feat_x.floor();
        let fy = feat_y - feat_y.floor();
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        // Base integer coordinate: top-left corner of the patch.
        let base_x_i = feat_x.floor() as isize - half;
        let base_y_i = feat_y.floor() as isize - half;

        // Gradient margin: gx needs columns ± 1, gy needs rows ± 1.
        let tmpl_in_bounds = base_x_i >= 1
            && base_y_i >= 1
            && base_x_i + side + 2 <= prev_img.width() as isize
            && base_y_i + side + 2 <= prev_img.height() as isize;

        let mut h00 = 0.0f32;
        let mut h01 = 0.0f32;
        let mut h11 = 0.0f32;

        let t_buf = &mut scratch.t_buf[..patch_size];
        let gx_buf = &mut scratch.gx_buf[..patch_size];
        let gy_buf = &mut scratch.gy_buf[..patch_size];

        if tmpl_in_bounds {
            // ── Fast path: row-pointer + constant-weight bilinear ──
            let base_x = base_x_i as usize;
            let base_y = base_y_i as usize;

            let mut idx = 0;
            for ly in 0..side as usize {
                let iy = base_y + ly;
                unsafe {
                    // Four rows covering template + gradient neighborhoods:
                    //   template:   (iy, iy+1)
                    //   gx:         same rows, columns ± 1
                    //   gy-:        (iy-1, iy)
                    //   gy+:        (iy+1, iy+2)
                    let r_m1 = prev_img.row_ptr(iy - 1);
                    let r_0  = prev_img.row_ptr(iy);
                    let r_1  = prev_img.row_ptr(iy + 1);
                    let r_p2 = prev_img.row_ptr(iy + 2);

                    for lx in 0..side as usize {
                        let ix = base_x + lx;

                        // Template value.
                        let t_val = bilerp_ptr(r_0, r_1, ix, w00, w10, w01, w11);

                        // Gradient gx: central difference in x, same rows.
                        let gx = 0.5 * (
                            bilerp_ptr(r_0, r_1, ix + 1, w00, w10, w01, w11)
                          - bilerp_ptr(r_0, r_1, ix - 1, w00, w10, w01, w11)
                        );

                        // Gradient gy: central difference in y, same columns.
                        let gy = 0.5 * (
                            bilerp_ptr(r_1, r_p2, ix, w00, w10, w01, w11)
                          - bilerp_ptr(r_m1, r_0, ix, w00, w10, w01, w11)
                        );

                        t_buf[idx] = t_val;
                        gx_buf[idx] = gx;
                        gy_buf[idx] = gy;

                        h00 += gx * gx;
                        h01 += gx * gy;
                        h11 += gy * gy;

                        idx += 1;
                    }
                }
            }
        } else {
            // ── Border fallback: clamped bilinear (rare — edge features) ──
            let mut idx = 0;
            for py in -half..=half {
                for px in -half..=half {
                    let tx = feat_x + px as f32;
                    let ty = feat_y + py as f32;

                    let t_val = interpolate_bilinear(prev_img, tx, ty);
                    let gx = 0.5 * (
                        interpolate_bilinear(prev_img, tx + 1.0, ty)
                      - interpolate_bilinear(prev_img, tx - 1.0, ty)
                    );
                    let gy = 0.5 * (
                        interpolate_bilinear(prev_img, tx, ty + 1.0)
                      - interpolate_bilinear(prev_img, tx, ty - 1.0)
                    );

                    t_buf[idx] = t_val;
                    gx_buf[idx] = gx;
                    gy_buf[idx] = gy;

                    h00 += gx * gx;
                    h01 += gx * gy;
                    h11 += gy * gy;

                    idx += 1;
                }
            }
        }

        // Precompute H^{-1} (constant across all iterations).
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-6 {
            return LkResult::Singular;
        }
        let inv_det = 1.0 / det;
        let ih00 =  inv_det * h11;
        let ih01 = -inv_det * h01;
        let ih11 =  inv_det * h00;

        // =================================================================
        // ITERATE: two-phase extraction + accumulation
        // =================================================================

        let warped_buf = &mut scratch.warped_buf[..patch_size];

        for _iter in 0..self.max_iterations {
            // ── Phase 1: extract warped pixels into contiguous buffer ──

            // Constant bilinear weights for this iteration's warp position.
            let wx = feat_x + dx;
            let wy = feat_y + dy;
            let fx_w = wx - wx.floor();
            let fy_w = wy - wy.floor();
            let ww00 = (1.0 - fx_w) * (1.0 - fy_w);
            let ww10 = fx_w * (1.0 - fy_w);
            let ww01 = (1.0 - fx_w) * fy_w;
            let ww11 = fx_w * fy_w;

            let bx_i = wx.floor() as isize - half;
            let by_i = wy.floor() as isize - half;

            // No gradient margin needed — just the bilinear +1 neighbor.
            let warp_in_bounds = bx_i >= 0
                && by_i >= 0
                && bx_i + side + 1 <= curr_img.width() as isize
                && by_i + side + 1 <= curr_img.height() as isize;

            if warp_in_bounds {
                let bx = bx_i as usize;
                let by = by_i as usize;

                let mut idx = 0;
                for ly in 0..side as usize {
                    unsafe {
                        let r0 = curr_img.row_ptr(by + ly);
                        let r1 = curr_img.row_ptr(by + ly + 1);

                        for lx in 0..side as usize {
                            warped_buf[idx] = bilerp_ptr(
                                r0, r1, bx + lx, ww00, ww10, ww01, ww11,
                            );
                            idx += 1;
                        }
                    }
                }
            } else {
                // Border fallback.
                let mut idx = 0;
                for py in -half..=half {
                    for px in -half..=half {
                        warped_buf[idx] = interpolate_bilinear(
                            curr_img,
                            wx + px as f32,
                            wy + py as f32,
                        );
                        idx += 1;
                    }
                }
            }

            // ── Phase 2: accumulate b = J^T * error (contiguous data) ──
            let (b0, b1) = accumulate_ic(t_buf, warped_buf, gx_buf, gy_buf);

            // Solve: delta = H^{-1} * b.
            let delta_x = ih00 * b0 + ih01 * b1;
            let delta_y = ih01 * b0 + ih11 * b1;

            dx += delta_x;
            dy += delta_y;

            if delta_x * delta_x + delta_y * delta_y < self.epsilon * self.epsilon {
                return LkResult::Converged(dx, dy);
            }
        }

        LkResult::MaxIter(dx, dy)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Internal result of iterative LK at one pyramid level.
enum LkResult {
    Converged(f32, f32),
    MaxIter(f32, f32),
    Singular,
}

/// Bilinear interpolation from two pre-fetched row pointers with constant
/// weights.
///
/// Given row pointers `r0` (row y0) and `r1` (row y0+1), samples the
/// 2×2 neighborhood at column `ix`:
///
///     r0[ix]   r0[ix+1]
///     r1[ix]   r1[ix+1]
///
/// The weights (w00, w10, w01, w11) correspond to:
///     w00 = (1-fx)(1-fy)    w10 = fx(1-fy)
///     w01 = (1-fx)fy        w11 = fx·fy
///
/// # Safety
/// Caller must ensure `ix + 1` is a valid offset from both `r0` and `r1`.
#[inline(always)]
unsafe fn bilerp_ptr(
    r0: *const f32, r1: *const f32,
    ix: usize,
    w00: f32, w10: f32, w01: f32, w11: f32,
) -> f32 {
    w00 * *r0.add(ix) + w10 * *r0.add(ix + 1)
  + w01 * *r1.add(ix) + w11 * *r1.add(ix + 1)
}

/// Accumulate the IC right-hand side vector over contiguous buffers.
///
///   b0 = Σ gx[i] * (t[i] - w[i])
///   b1 = Σ gy[i] * (t[i] - w[i])
///
/// This is a paired dot-product of (gx, gy) against the error vector
/// (t - warped). Written as a standalone function on contiguous slices
/// so it can be replaced with a SIMD version later without touching
/// the rest of the tracker.
///
/// GPU EQUIVALENT: A reduction shader over the patch — exactly what
/// the wavefront-per-patch cooperative KLT shader does.
#[inline]
fn accumulate_ic(t: &[f32], w: &[f32], gx: &[f32], gy: &[f32]) -> (f32, f32) {
    debug_assert_eq!(t.len(), w.len());
    debug_assert_eq!(t.len(), gx.len());
    debug_assert_eq!(t.len(), gy.len());

    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;

    // Scalar loop on contiguous data. The compiler can auto-vectorize
    // this with -C target-feature=+avx2,+fma, but we'll replace it
    // with explicit SIMD later.
    for i in 0..t.len() {
        let e = unsafe { *t.get_unchecked(i) - *w.get_unchecked(i) };
        b0 += unsafe { *gx.get_unchecked(i) } * e;
        b1 += unsafe { *gy.get_unchecked(i) } * e;
    }

    (b0, b1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test image with a bright square on dark background.
    fn make_test_image(w: usize, h: usize, sq_x: usize, sq_y: usize, sq_size: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![30u8; w * h]);
        for y in sq_y..(sq_y + sq_size).min(h) {
            for x in sq_x..(sq_x + sq_size).min(w) {
                img.set(x, y, 200);
            }
        }
        img
    }

    #[test]
    fn test_zero_motion() {
        // Same image for both frames → displacement should be ~0.
        // Use a large enough image so the feature survives pyramid scaling:
        // L2 position = 41/4 = 10.25, margin = 5+1 = 6 → OK.
        let img = make_test_image(120, 120, 40, 40, 30);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        // Feature at the top-left corner of the square where the patch
        // straddles the intensity transition → strong gradient.
        let features = vec![Feature {
            x: 41.0,
            y: 41.0,
            score: 100.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(
            dx.abs() < 0.5 && dy.abs() < 0.5,
            "zero motion test: displacement ({dx}, {dy}) should be near zero"
        );
    }

    #[test]
    fn test_known_horizontal_shift() {
        // Shift the image 3px to the right → tracker should recover dx ≈ 3.
        // Large image so pyramid L2 position stays in bounds:
        // feature at 41 → L2: 41/4=10.25, margin=7+1=8 → OK.
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 40, 30); // shifted right by 3

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        // Feature near the top-left corner of the square where the patch
        // straddles both the horizontal and vertical edges → good 2D gradient.
        let features = vec![Feature {
            x: 41.0,
            y: 41.0,
            score: 100.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;

        assert!(
            (dx - 3.0).abs() < 1.5,
            "horizontal shift: dx = {dx}, expected ~3.0"
        );
        assert!(
            dy.abs() < 1.5,
            "horizontal shift: dy = {dy}, expected ~0.0"
        );
    }

    #[test]
    fn test_known_diagonal_shift() {
        // Shift (2, 2) diagonally.
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 42, 42, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        // Feature at the top-left corner of the square.
        let features = vec![Feature {
            x: 41.0,
            y: 41.0,
            score: 100.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;

        assert!(
            (dx - 2.0).abs() < 1.5,
            "diagonal shift: dx = {dx}, expected ~2.0"
        );
        assert!(
            (dy - 2.0).abs() < 1.5,
            "diagonal shift: dy = {dy}, expected ~2.0"
        );
    }

    #[test]
    fn test_multiple_features() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 42, 40, 30); // shift right by 2

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        // Features near different edges/corners of the square.
        let features = vec![
            Feature { x: 41.0, y: 50.0, score: 100.0, level: 0, id: 1 },
            Feature { x: 55.0, y: 41.0, score: 90.0, level: 0, id: 2 },
            Feature { x: 69.0, y: 55.0, score: 80.0, level: 0, id: 3 },
        ];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results.len(), 3);

        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.feature.id, features[i].id, "ID should be preserved");
        }
    }

    #[test]
    fn test_feature_at_border_degrades_gracefully() {
        // Feature near the edge: no pre-emptive rejection.
        // Bilinear clamp handles the border. The patch will mostly see
        // clamped (flat) pixels → likely singular Hessian → Lost.
        // Or if it somehow tracks, the final bounds check catches it.
        // Either way, it should not panic.
        let img = make_test_image(40, 40, 10, 10, 20);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 3.0,
            y: 3.0,
            score: 50.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        // Should not panic. Status is either Lost or Tracked — both OK.
        assert!(
            results[0].status == TrackStatus::Lost
                || results[0].status == TrackStatus::Tracked,
            "border feature should degrade gracefully, got {:?}",
            results[0].status,
        );
    }

    #[test]
    fn test_flat_region_singular() {
        // A completely flat image has zero gradient → singular Hessian.
        let img = Image::from_vec(60, 60, vec![128u8; 3600]);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        let features = vec![Feature {
            x: 30.0,
            y: 30.0,
            score: 50.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        // Should be Lost (singular Hessian) or Tracked with zero displacement.
        // Either is acceptable — the point is it doesn't crash.
        assert!(
            results[0].status == TrackStatus::Lost
                || (results[0].feature.x - 30.0).abs() < 0.5,
            "flat region should be lost or stationary"
        );
    }

    #[test]
    fn test_id_preserved() {
        let img = make_test_image(120, 120, 40, 40, 30);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        let features = vec![
            Feature { x: 41.0, y: 41.0, score: 100.0, level: 0, id: 42 },
            Feature { x: 69.0, y: 69.0, score: 80.0, level: 0, id: 99 },
        ];

        let results = tracker.track(&pyr, &pyr, &features);
        assert_eq!(results[0].feature.id, 42);
        assert_eq!(results[1].feature.id, 99);
    }

    #[test]
    fn test_subpixel_shift() {
        // Create a smooth gradient image where sub-pixel shifts are meaningful.
        let w = 80;
        let h = 80;
        let mut data1 = vec![0u8; w * h];
        let mut data2 = vec![0u8; w * h];

        // Gaussian-ish blob centered at (40, 40).
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 40.0;
                let dy = y as f32 - 40.0;
                let v = (255.0 * (-0.005 * (dx * dx + dy * dy)).exp()) as u8;
                data1[y * w + x] = v;

                // Shift blob by (1.5, 0.5)
                let dx2 = x as f32 - 41.5;
                let dy2 = y as f32 - 40.5;
                let v2 = (255.0 * (-0.005 * (dx2 * dx2 + dy2 * dy2)).exp()) as u8;
                data2[y * w + x] = v2;
            }
        }

        let img1 = Image::from_vec(w, h, data1);
        let img2 = Image::from_vec(w, h, data2);
        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 40.0,
            y: 40.0,
            score: 100.0,
            level: 0,
            id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 40.0;
        let dy = results[0].feature.y - 40.0;

        // Sub-pixel accuracy: should recover (1.5, 0.5) within ~0.5px.
        assert!(
            (dx - 1.5).abs() < 0.5,
            "subpixel shift: dx = {dx}, expected ~1.5"
        );
        assert!(
            (dy - 0.5).abs() < 0.5,
            "subpixel shift: dy = {dy}, expected ~0.5"
        );
    }

    // ===== Inverse Compositional tests =====
    // Mirror the forward-additive tests to verify both methods agree.

    fn make_ic_tracker(window_size: usize, max_levels: usize) -> KltTracker {
        KltTracker::with_method(window_size, 30, 0.01, max_levels, LkMethod::InverseCompositional)
    }

    #[test]
    fn test_ic_zero_motion() {
        let img = make_test_image(120, 120, 40, 40, 30);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = make_ic_tracker(5, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(
            dx.abs() < 0.5 && dy.abs() < 0.5,
            "IC zero motion: ({dx}, {dy}) should be near zero"
        );
    }

    #[test]
    fn test_ic_horizontal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 40, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = make_ic_tracker(7, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(
            (dx - 3.0).abs() < 1.5,
            "IC horizontal: dx = {dx}, expected ~3.0"
        );
        assert!(dy.abs() < 1.5, "IC horizontal: dy = {dy}, expected ~0.0");
    }

    #[test]
    fn test_ic_diagonal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 42, 42, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = make_ic_tracker(7, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!((dx - 2.0).abs() < 1.5, "IC diagonal: dx = {dx}, expected ~2.0");
        assert!((dy - 2.0).abs() < 1.5, "IC diagonal: dy = {dy}, expected ~2.0");
    }

    #[test]
    fn test_ic_subpixel_shift() {
        let w = 80;
        let h = 80;
        let mut data1 = vec![0u8; w * h];
        let mut data2 = vec![0u8; w * h];

        for y in 0..h {
            for x in 0..w {
                let d1 = (x as f32 - 40.0).powi(2) + (y as f32 - 40.0).powi(2);
                data1[y * w + x] = (255.0 * (-0.005 * d1).exp()) as u8;

                let d2 = (x as f32 - 41.5).powi(2) + (y as f32 - 40.5).powi(2);
                data2[y * w + x] = (255.0 * (-0.005 * d2).exp()) as u8;
            }
        }

        let img1 = Image::from_vec(w, h, data1);
        let img2 = Image::from_vec(w, h, data2);
        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = make_ic_tracker(7, 3);
        let features = vec![Feature {
            x: 40.0, y: 40.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 40.0;
        let dy = results[0].feature.y - 40.0;
        assert!((dx - 1.5).abs() < 0.5, "IC subpixel: dx = {dx}, expected ~1.5");
        assert!((dy - 0.5).abs() < 0.5, "IC subpixel: dy = {dy}, expected ~0.5");
    }

    #[test]
    fn test_ic_flat_region() {
        let img = Image::from_vec(60, 60, vec![128u8; 3600]);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = make_ic_tracker(5, 3);
        let features = vec![Feature {
            x: 30.0, y: 30.0, score: 50.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        // Flat → singular Hessian → Lost.
        assert_eq!(results[0].status, TrackStatus::Lost);
    }

    #[test]
    fn test_fa_and_ic_agree() {
        // Both methods should recover approximately the same displacement
        // on a clean synthetic shift.
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 42, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let fa_tracker = KltTracker::new(7, 30, 0.01, 3);
        let ic_tracker = make_ic_tracker(7, 3);

        let fa_results = fa_tracker.track(&pyr1, &pyr2, &features);
        let ic_results = ic_tracker.track(&pyr1, &pyr2, &features);

        assert_eq!(fa_results[0].status, TrackStatus::Tracked);
        assert_eq!(ic_results[0].status, TrackStatus::Tracked);

        let fa_dx = fa_results[0].feature.x - 41.0;
        let fa_dy = fa_results[0].feature.y - 41.0;
        let ic_dx = ic_results[0].feature.x - 41.0;
        let ic_dy = ic_results[0].feature.y - 41.0;

        // Both should agree within ~1 pixel on a clean synthetic scene.
        assert!(
            (fa_dx - ic_dx).abs() < 1.0,
            "FA vs IC dx: {fa_dx:.2} vs {ic_dx:.2}"
        );
        assert!(
            (fa_dy - ic_dy).abs() < 1.0,
            "FA vs IC dy: {fa_dy:.2} vs {ic_dy:.2}"
        );
    }
}
