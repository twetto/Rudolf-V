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
// - AVX2+FMA SIMD for accumulation loops (runtime-detected)
// - RAYON PARALLELISM: feature loop is embarrassingly parallel —
//   each feature tracked independently. map_init creates one KltScratch
//   per worker thread, reused across all features on that thread.
//
// SIMD STRATEGY:
// We use std::arch intrinsics with #[target_feature(enable = "avx2,fma")]
// on individual functions, guarded by is_x86_feature_detected! at runtime.
// This means:
// - No global -C target-feature flags needed (compiles for generic x86_64)
// - AVX2+FMA used automatically on 12400F, 8845H, etc.
// - Scalar fallback on RPi 4 / VideoCore VI (ARM)
// - Each SIMD function processes 8 f32s per iteration (256-bit lanes)
//
// GPU MAPPING: The accumulation SIMD mirrors the warp-level reduction
// in the wavefront-per-patch cooperative KLT shader. The extraction
// SIMD mirrors the texture-gather pattern.

use crate::fast::Feature;
use crate::image::{interpolate_bilinear, interpolate_bilinear_unchecked, Image};
use crate::pyramid::Pyramid;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Status of a tracked feature after one frame-to-frame tracking pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackStatus {
    Tracked,
    Lost,
    OutOfBounds,
}

/// Lucas-Kanade algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LkMethod {
    ForwardAdditive,
    InverseCompositional,
}

/// A feature with its tracking status after a track() call.
#[derive(Debug, Clone)]
pub struct TrackedFeature {
    pub feature: Feature,
    pub status: TrackStatus,
}

/// Pre-allocated scratch buffers for KLT tracking.
///
/// Eliminates per-feature heap allocation. The IC precompute needs
/// (t_buf, gx_buf, gy_buf) and the two-phase iteration adds warped_buf.
/// All are sized to the largest patch: (2 * window_size + 1)².
pub struct KltScratch {
    t_buf: Vec<f32>,
    gx_buf: Vec<f32>,
    gy_buf: Vec<f32>,
    warped_buf: Vec<f32>,
}

impl KltScratch {
    pub fn new(window_size: usize) -> Self {
        let patch_size = (2 * window_size + 1) * (2 * window_size + 1);
        KltScratch {
            t_buf: vec![0.0; patch_size],
            gx_buf: vec![0.0; patch_size],
            gy_buf: vec![0.0; patch_size],
            warped_buf: vec![0.0; patch_size],
        }
    }

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
pub struct KltTracker {
    pub window_size: usize,
    pub max_iterations: usize,
    pub epsilon: f32,
    pub max_levels: usize,
    pub method: LkMethod,
}

impl KltTracker {
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
        let patch_size = side * side;

        // ── Parallel path (feature-gated) ───────────────────────────────
        // Each feature is independent. map_init creates one KltScratch per
        // rayon worker thread (not per feature), reusing it across all
        // features assigned to that thread. For 200 features on 8 cores,
        // that's ~25 features per thread at ~5µs each = ~125µs per thread.
        #[cfg(feature = "parallel")]
        {
            results.clear();
            let par_results: Vec<TrackedFeature> = features
                .par_iter()
                .map_init(
                    || {
                        let mut s = KltScratch::new(self.window_size);
                        s.ensure_size(patch_size);
                        s
                    },
                    |thread_scratch, feat| {
                        self.track_single(
                            prev_pyramid, curr_pyramid, feat, num_levels, thread_scratch,
                        )
                    },
                )
                .collect();
            *results = par_results;
            return;
        }

        // ── Sequential path ─────────────────────────────────────────────
        #[cfg(not(feature = "parallel"))]
        {
            scratch.ensure_size(patch_size);

            results.clear();
            results.reserve(features.len());

            for feat in features {
                results.push(self.track_single(
                    prev_pyramid, curr_pyramid, feat, num_levels, scratch,
                ));
            }
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
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;

        for level in (0..num_levels).rev() {
            let prev_img = &prev_pyr.levels[level];
            let curr_img = &curr_pyr.levels[level];

            let scale = 1.0 / (1u32 << level) as f32;
            let feat_x = feature.x * scale;
            let feat_y = feature.y * scale;

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

            if level > 0 {
                dx *= 2.0;
                dy *= 2.0;
            }
        }

        let new_x = feature.x + dx;
        let new_y = feature.y + dy;

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
    // Forward additive (reference — not SIMD-optimized)
    // =====================================================================

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

        let tmpl_in_bounds = (feat_x - half_i as f32) >= 0.0
            && (feat_y - half_i as f32) >= 0.0
            && (feat_x + half_i as f32) < prev_img.width() as f32
            && (feat_y + half_i as f32) < prev_img.height() as f32;

        for _iter in 0..self.max_iterations {
            let mut h00 = 0.0f32;
            let mut h01 = 0.0f32;
            let mut h11 = 0.0f32;
            let mut b0 = 0.0f32;
            let mut b1 = 0.0f32;

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

                    h00 += gx * gx;
                    h01 += gx * gy;
                    h11 += gy * gy;

                    b0 += gx * e;
                    b1 += gy * e;
                }
            }

            let det = h00 * h11 - h01 * h01;
            if det.abs() < 1e-6 {
                return LkResult::Singular;
            }
            let inv_det = 1.0 / det;

            let delta_x = inv_det * (h11 * b0 - h01 * b1);
            let delta_y = inv_det * (h00 * b1 - h01 * b0);

            dx += delta_x;
            dy += delta_y;

            if delta_x * delta_x + delta_y * delta_y < self.epsilon * self.epsilon {
                return LkResult::Converged(dx, dy);
            }
        }

        LkResult::MaxIter(dx, dy)
    }

    // =====================================================================
    // Inverse compositional (SIMD-optimized)
    // =====================================================================

    /// Optimized inverse-compositional Lucas-Kanade at a single level.
    ///
    /// Optimizations:
    /// - CONSTANT BILINEAR WEIGHTS: frac(feat_x + px) = frac(feat_x)
    ///   for integer px. Weights computed once, not per pixel.
    /// - ROW-POINTER ACCESS: eliminates y * stride from inner loop.
    /// - TWO-PHASE ITERATION: extract warped → SIMD accumulate.
    /// - AVX2+FMA SIMD: accumulate_ic processes 8 floats per cycle.
    ///   accumulate_hessian does the same for the precompute.
    ///   extract_warped_row_simd vectorizes the bilinear extraction.
    /// - HOISTED SCRATCH: zero per-feature allocation.
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

        let base_x_i = feat_x.floor() as isize - half;
        let base_y_i = feat_y.floor() as isize - half;

        // Gradient margin: gx needs columns ±1, gy needs rows ±1.
        let tmpl_in_bounds = base_x_i >= 1
            && base_y_i >= 1
            && base_x_i + side + 2 <= prev_img.width() as isize
            && base_y_i + side + 2 <= prev_img.height() as isize;

        let t_buf = &mut scratch.t_buf[..patch_size];
        let gx_buf = &mut scratch.gx_buf[..patch_size];
        let gy_buf = &mut scratch.gy_buf[..patch_size];

        // Fill buffers (template values + gradients).
        // Hessian is accumulated separately via SIMD after filling.
        if tmpl_in_bounds {
            // ── Fast path: row-pointer + constant-weight bilinear ──
            let base_x = base_x_i as usize;
            let base_y = base_y_i as usize;

            let mut idx = 0;
            for ly in 0..side as usize {
                let iy = base_y + ly;
                unsafe {
                    let r_m1 = prev_img.row_ptr(iy - 1);
                    let r_0  = prev_img.row_ptr(iy);
                    let r_1  = prev_img.row_ptr(iy + 1);
                    let r_p2 = prev_img.row_ptr(iy + 2);

                    for lx in 0..side as usize {
                        let ix = base_x + lx;

                        let t_val = bilerp_ptr(r_0, r_1, ix, w00, w10, w01, w11);

                        let gx = 0.5 * (
                            bilerp_ptr(r_0, r_1, ix + 1, w00, w10, w01, w11)
                          - bilerp_ptr(r_0, r_1, ix - 1, w00, w10, w01, w11)
                        );

                        let gy = 0.5 * (
                            bilerp_ptr(r_1, r_p2, ix, w00, w10, w01, w11)
                          - bilerp_ptr(r_m1, r_0, ix, w00, w10, w01, w11)
                        );

                        t_buf[idx] = t_val;
                        gx_buf[idx] = gx;
                        gy_buf[idx] = gy;

                        idx += 1;
                    }
                }
            }
        } else {
            // ── Border fallback: clamped bilinear ──
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

                    idx += 1;
                }
            }
        }

        // Hessian: SIMD-accelerated triple accumulation over filled buffers.
        let (h00, h01, h11) = accumulate_hessian(gx_buf, gy_buf);

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
        let side_u = side as usize;

        for _iter in 0..self.max_iterations {
            // ── Phase 1: extract warped pixels into contiguous buffer ──

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

            let warp_in_bounds = bx_i >= 0
                && by_i >= 0
                && bx_i + side + 1 <= curr_img.width() as isize
                && by_i + side + 1 <= curr_img.height() as isize;

            if warp_in_bounds {
                let bx = bx_i as usize;
                let by = by_i as usize;

                let mut idx = 0;
                for ly in 0..side_u {
                    unsafe {
                        let r0 = curr_img.row_ptr(by + ly);
                        let r1 = curr_img.row_ptr(by + ly + 1);

                        // SIMD extraction: process 8 pixels at a time.
                        extract_warped_row(
                            r0, r1, bx, side_u,
                            ww00, ww10, ww01, ww11,
                            &mut warped_buf[idx..idx + side_u],
                        );
                    }
                    idx += side_u;
                }
            } else {
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

            // ── Phase 2: SIMD-accelerated accumulate b = J^T * error ──
            let (b0, b1) = accumulate_ic(t_buf, warped_buf, gx_buf, gy_buf);

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

// ==========================================================================
// Private helpers
// ==========================================================================

enum LkResult {
    Converged(f32, f32),
    MaxIter(f32, f32),
    Singular,
}

/// Bilinear interpolation from two row pointers with constant weights.
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

// ==========================================================================
// SIMD dispatch layer
//
// Each function checks for AVX2+FMA at runtime (cached after first call),
// then dispatches to the SIMD or scalar implementation.
// ==========================================================================

/// Accumulate the IC right-hand side: b0 = Σ gx·(t-w), b1 = Σ gy·(t-w).
///
/// This runs every IC iteration × every feature × every level — it's the
/// single hottest loop in the entire frontend pipeline.
#[inline]
fn accumulate_ic(t: &[f32], w: &[f32], gx: &[f32], gy: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { accumulate_ic_avx2(t, w, gx, gy) };
        }
    }
    accumulate_ic_scalar(t, w, gx, gy)
}

/// Accumulate the Hessian: h00 = Σ gx², h01 = Σ gx·gy, h11 = Σ gy².
///
/// Runs once per feature per level during IC precompute.
#[inline]
fn accumulate_hessian(gx: &[f32], gy: &[f32]) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { accumulate_hessian_avx2(gx, gy) };
        }
    }
    accumulate_hessian_scalar(gx, gy)
}

/// Extract one row of warped pixels via bilinear interpolation.
///
/// # Safety
/// Caller must ensure r0, r1 have at least `bx + count + 1` valid elements.
#[inline]
unsafe fn extract_warped_row(
    r0: *const f32, r1: *const f32,
    bx: usize, count: usize,
    w00: f32, w10: f32, w01: f32, w11: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            extract_warped_row_avx2(r0, r1, bx, count, w00, w10, w01, w11, out);
            return;
        }
    }
    extract_warped_row_scalar(r0, r1, bx, count, w00, w10, w01, w11, out);
}

// ==========================================================================
// Scalar implementations (fallback for non-x86 or missing AVX2)
// ==========================================================================

#[inline]
fn accumulate_ic_scalar(t: &[f32], w: &[f32], gx: &[f32], gy: &[f32]) -> (f32, f32) {
    debug_assert_eq!(t.len(), w.len());
    debug_assert_eq!(t.len(), gx.len());
    debug_assert_eq!(t.len(), gy.len());

    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;

    for i in 0..t.len() {
        let e = unsafe { *t.get_unchecked(i) - *w.get_unchecked(i) };
        b0 += unsafe { *gx.get_unchecked(i) } * e;
        b1 += unsafe { *gy.get_unchecked(i) } * e;
    }

    (b0, b1)
}

#[inline]
fn accumulate_hessian_scalar(gx: &[f32], gy: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(gx.len(), gy.len());

    let mut h00 = 0.0f32;
    let mut h01 = 0.0f32;
    let mut h11 = 0.0f32;

    for i in 0..gx.len() {
        let gxi = unsafe { *gx.get_unchecked(i) };
        let gyi = unsafe { *gy.get_unchecked(i) };
        h00 += gxi * gxi;
        h01 += gxi * gyi;
        h11 += gyi * gyi;
    }

    (h00, h01, h11)
}

#[inline]
unsafe fn extract_warped_row_scalar(
    r0: *const f32, r1: *const f32,
    bx: usize, count: usize,
    w00: f32, w10: f32, w01: f32, w11: f32,
    out: &mut [f32],
) {
    for lx in 0..count {
        let ix = bx + lx;
        *out.get_unchecked_mut(lx) = bilerp_ptr(r0, r1, ix, w00, w10, w01, w11);
    }
}

// ==========================================================================
// AVX2 + FMA implementations
// ==========================================================================

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 f32 lanes → single f32.
    #[inline(always)]
    pub(super) unsafe fn hsum256(v: __m256) -> f32 {
        // [a0+a4, a1+a5, a2+a6, a3+a7] (128-bit)
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        // [s0+s2, s1+s3, s2+s0, s3+s1]
        let shuf = _mm_movehdup_ps(sum128); // [s1, s1, s3, s3]
        let sum64 = _mm_add_ps(sum128, shuf);
        let hi64 = _mm_movehl_ps(sum64, sum64);
        let sum32 = _mm_add_ss(sum64, hi64);
        _mm_cvtss_f32(sum32)
    }
}

/// AVX2+FMA paired dot-product for IC accumulation.
///
/// Processes 8 f32s per iteration. For 529 elements (23×23 patch):
/// 66 full AVX2 iterations + 1-element scalar tail.
/// Each iteration: 1 sub + 2 FMA = 3 AVX2 instructions on 8 lanes.
///
/// vs scalar: 529 iterations × (1 sub + 2 mul + 2 add) = 2645 ops.
/// Theoretical speedup: ~8× (limited by FMA throughput, not latency,
/// since the accumulator dependency chain is broken by out-of-order exec).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_ic_avx2(t: &[f32], w: &[f32], gx: &[f32], gy: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;

    let n = t.len();
    let chunks = n / 8;

    let mut sum_b0 = _mm256_setzero_ps();
    let mut sum_b1 = _mm256_setzero_ps();

    let t_ptr = t.as_ptr();
    let w_ptr = w.as_ptr();
    let gx_ptr = gx.as_ptr();
    let gy_ptr = gy.as_ptr();

    for i in 0..chunks {
        let off = i * 8;
        let vt  = _mm256_loadu_ps(t_ptr.add(off));
        let vw  = _mm256_loadu_ps(w_ptr.add(off));
        let ve  = _mm256_sub_ps(vt, vw);          // e = t - w

        let vgx = _mm256_loadu_ps(gx_ptr.add(off));
        let vgy = _mm256_loadu_ps(gy_ptr.add(off));

        sum_b0 = _mm256_fmadd_ps(vgx, ve, sum_b0); // b0 += gx * e
        sum_b1 = _mm256_fmadd_ps(vgy, ve, sum_b1); // b1 += gy * e
    }

    let mut b0 = simd_avx2::hsum256(sum_b0);
    let mut b1 = simd_avx2::hsum256(sum_b1);

    // Scalar tail (0–7 elements).
    for i in (chunks * 8)..n {
        let e = *t_ptr.add(i) - *w_ptr.add(i);
        b0 += *gx_ptr.add(i) * e;
        b1 += *gy_ptr.add(i) * e;
    }

    (b0, b1)
}

/// AVX2+FMA triple accumulation for Hessian precompute.
///
/// h00 = Σ gx², h01 = Σ gx·gy, h11 = Σ gy²
/// 3 FMAs per 8-element chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_hessian_avx2(gx: &[f32], gy: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let n = gx.len();
    let chunks = n / 8;

    let mut sum_h00 = _mm256_setzero_ps();
    let mut sum_h01 = _mm256_setzero_ps();
    let mut sum_h11 = _mm256_setzero_ps();

    let gx_ptr = gx.as_ptr();
    let gy_ptr = gy.as_ptr();

    for i in 0..chunks {
        let off = i * 8;
        let vgx = _mm256_loadu_ps(gx_ptr.add(off));
        let vgy = _mm256_loadu_ps(gy_ptr.add(off));

        sum_h00 = _mm256_fmadd_ps(vgx, vgx, sum_h00); // h00 += gx * gx
        sum_h01 = _mm256_fmadd_ps(vgx, vgy, sum_h01); // h01 += gx * gy
        sum_h11 = _mm256_fmadd_ps(vgy, vgy, sum_h11); // h11 += gy * gy
    }

    let mut h00 = simd_avx2::hsum256(sum_h00);
    let mut h01 = simd_avx2::hsum256(sum_h01);
    let mut h11 = simd_avx2::hsum256(sum_h11);

    for i in (chunks * 8)..n {
        let gxi = *gx_ptr.add(i);
        let gyi = *gy_ptr.add(i);
        h00 += gxi * gxi;
        h01 += gxi * gyi;
        h11 += gyi * gyi;
    }

    (h00, h01, h11)
}

/// AVX2+FMA vectorized bilinear extraction for one patch row.
///
/// For 8 consecutive pixels at column offsets [ix, ix+1, ..., ix+7]:
///   out[k] = w00*r0[ix+k] + w10*r0[ix+k+1] + w01*r1[ix+k] + w11*r1[ix+k+1]
///
/// Each 8-pixel chunk: 4 loads + 1 mul + 3 FMAs.
/// For a 23-wide patch: 2 full chunks + 7 scalar tail per row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn extract_warped_row_avx2(
    r0: *const f32, r1: *const f32,
    bx: usize, count: usize,
    w00: f32, w10: f32, w01: f32, w11: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let chunks = count / 8;
    let r0b = r0.add(bx);
    let r1b = r1.add(bx);

    let vw00 = _mm256_set1_ps(w00);
    let vw10 = _mm256_set1_ps(w10);
    let vw01 = _mm256_set1_ps(w01);
    let vw11 = _mm256_set1_ps(w11);

    for i in 0..chunks {
        let off = i * 8;
        // r0[ix..ix+8], r0[ix+1..ix+9] (shifted by 1)
        let v_r0   = _mm256_loadu_ps(r0b.add(off));
        let v_r0_1 = _mm256_loadu_ps(r0b.add(off + 1));
        let v_r1   = _mm256_loadu_ps(r1b.add(off));
        let v_r1_1 = _mm256_loadu_ps(r1b.add(off + 1));

        // w00 * r0[ix] + w10 * r0[ix+1] + w01 * r1[ix] + w11 * r1[ix+1]
        let mut acc = _mm256_mul_ps(vw00, v_r0);
        acc = _mm256_fmadd_ps(vw10, v_r0_1, acc);
        acc = _mm256_fmadd_ps(vw01, v_r1, acc);
        acc = _mm256_fmadd_ps(vw11, v_r1_1, acc);

        _mm256_storeu_ps(out.as_mut_ptr().add(off), acc);
    }

    // Scalar tail.
    for lx in (chunks * 8)..count {
        let ix = bx + lx;
        *out.get_unchecked_mut(lx) = bilerp_ptr(r0, r1, ix, w00, w10, w01, w11);
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        let img = make_test_image(120, 120, 40, 40, 30);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(
            dx.abs() < 0.5 && dy.abs() < 0.5,
            "zero motion: ({dx}, {dy}) should be near zero"
        );
    }

    #[test]
    fn test_known_horizontal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 40, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!((dx - 3.0).abs() < 1.5, "horizontal: dx = {dx}, expected ~3.0");
        assert!(dy.abs() < 1.5, "horizontal: dy = {dy}, expected ~0.0");
    }

    #[test]
    fn test_known_diagonal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 42, 42, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 41.0, y: 41.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!((dx - 2.0).abs() < 1.5, "diagonal: dx = {dx}, expected ~2.0");
        assert!((dy - 2.0).abs() < 1.5, "diagonal: dy = {dy}, expected ~2.0");
    }

    #[test]
    fn test_multiple_features() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 42, 40, 30);

        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
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
        let img = make_test_image(40, 40, 10, 10, 20);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 3.0, y: 3.0, score: 50.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
        assert!(
            results[0].status == TrackStatus::Lost
                || results[0].status == TrackStatus::Tracked,
            "border feature should degrade gracefully, got {:?}", results[0].status,
        );
    }

    #[test]
    fn test_flat_region_singular() {
        let img = Image::from_vec(60, 60, vec![128u8; 3600]);
        let pyr = Pyramid::build(&img, 3, 1.0);

        let tracker = KltTracker::new(5, 30, 0.01, 3);
        let features = vec![Feature {
            x: 30.0, y: 30.0, score: 50.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr, &pyr, &features);
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
        let w = 80;
        let h = 80;
        let mut data1 = vec![0u8; w * h];
        let mut data2 = vec![0u8; w * h];

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 40.0;
                let dy = y as f32 - 40.0;
                data1[y * w + x] = (255.0 * (-0.005 * (dx * dx + dy * dy)).exp()) as u8;

                let dx2 = x as f32 - 41.5;
                let dy2 = y as f32 - 40.5;
                data2[y * w + x] = (255.0 * (-0.005 * (dx2 * dx2 + dy2 * dy2)).exp()) as u8;
            }
        }

        let img1 = Image::from_vec(w, h, data1);
        let img2 = Image::from_vec(w, h, data2);
        let pyr1 = Pyramid::build(&img1, 3, 1.0);
        let pyr2 = Pyramid::build(&img2, 3, 1.0);

        let tracker = KltTracker::new(7, 30, 0.01, 3);
        let features = vec![Feature {
            x: 40.0, y: 40.0, score: 100.0, level: 0, id: 1,
        }];

        let results = tracker.track(&pyr1, &pyr2, &features);
        assert_eq!(results[0].status, TrackStatus::Tracked);

        let dx = results[0].feature.x - 40.0;
        let dy = results[0].feature.y - 40.0;
        assert!((dx - 1.5).abs() < 0.5, "subpixel: dx = {dx}, expected ~1.5");
        assert!((dy - 0.5).abs() < 0.5, "subpixel: dy = {dy}, expected ~0.5");
    }

    // ===== Inverse Compositional tests =====

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
        assert!((dx - 3.0).abs() < 1.5, "IC horizontal: dx = {dx}, expected ~3.0");
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
        assert_eq!(results[0].status, TrackStatus::Lost);
    }

    #[test]
    fn test_fa_and_ic_agree() {
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

        assert!((fa_dx - ic_dx).abs() < 1.0, "FA vs IC dx: {fa_dx:.2} vs {ic_dx:.2}");
        assert!((fa_dy - ic_dy).abs() < 1.0, "FA vs IC dy: {fa_dy:.2} vs {ic_dy:.2}");
    }

    // ===== SIMD-specific correctness tests =====

    #[test]
    fn test_accumulate_ic_simd_matches_scalar() {
        let n = 529; // 23×23 patch
        let t: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.05).collect();
        let gx: Vec<f32> = (0..n).map(|i| ((i * 7) as f32).sin()).collect();
        let gy: Vec<f32> = (0..n).map(|i| ((i * 13) as f32).cos()).collect();

        let (s_b0, s_b1) = accumulate_ic_scalar(&t, &w, &gx, &gy);
        let (d_b0, d_b1) = accumulate_ic(&t, &w, &gx, &gy);

        assert!(
            (s_b0 - d_b0).abs() < 1e-2,
            "accumulate_ic b0: scalar={s_b0}, dispatch={d_b0}"
        );
        assert!(
            (s_b1 - d_b1).abs() < 1e-2,
            "accumulate_ic b1: scalar={s_b1}, dispatch={d_b1}"
        );
    }

    #[test]
    fn test_accumulate_hessian_simd_matches_scalar() {
        let n = 529;
        let gx: Vec<f32> = (0..n).map(|i| ((i * 7) as f32).sin()).collect();
        let gy: Vec<f32> = (0..n).map(|i| ((i * 13) as f32).cos()).collect();

        let (s_h00, s_h01, s_h11) = accumulate_hessian_scalar(&gx, &gy);
        let (d_h00, d_h01, d_h11) = accumulate_hessian(&gx, &gy);

        assert!((s_h00 - d_h00).abs() < 1e-2, "hessian h00: {s_h00} vs {d_h00}");
        assert!((s_h01 - d_h01).abs() < 1e-2, "hessian h01: {s_h01} vs {d_h01}");
        assert!((s_h11 - d_h11).abs() < 1e-2, "hessian h11: {s_h11} vs {d_h11}");
    }

    #[test]
    fn test_accumulate_ic_odd_length() {
        // Non-multiple-of-8 length to test scalar tail.
        let n = 23; // 23 elements = 2 chunks + 7 tail
        let t: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let w: Vec<f32> = vec![0.0; n];
        let gx: Vec<f32> = vec![1.0; n];
        let gy: Vec<f32> = vec![2.0; n];

        let (b0, b1) = accumulate_ic(&t, &w, &gx, &gy);

        // b0 = Σ 1.0 * i = n*(n-1)/2 = 253
        // b1 = Σ 2.0 * i = 2 * 253 = 506
        let expected_b0 = (n * (n - 1) / 2) as f32;
        let expected_b1 = 2.0 * expected_b0;

        assert!((b0 - expected_b0).abs() < 1e-3, "odd len b0: {b0} vs {expected_b0}");
        assert!((b1 - expected_b1).abs() < 1e-3, "odd len b1: {b1} vs {expected_b1}");
    }
}
