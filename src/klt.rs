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
// NEW RUST CONCEPTS:
// - Enums with variants (TrackStatus, LkMethod) — Rust enums can carry
//   data, making them algebraic data types (tagged unions).
// - Borrowing two pyramids simultaneously (&prev_pyramid, &curr_pyramid).
// - `match` on enums to dispatch between algorithm variants.

use crate::fast::Feature;
use crate::image::{interpolate_bilinear, Image};
use crate::pyramid::Pyramid;

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
    /// Takes two pre-built pyramids (both f32, as produced by
    /// `Pyramid::build`) and a list of features detected in the
    /// previous frame.
    ///
    /// Returns a `TrackedFeature` for each input feature, with
    /// updated position and status.
    pub fn track(
        &self,
        prev_pyramid: &Pyramid,
        curr_pyramid: &Pyramid,
        features: &[Feature],
    ) -> Vec<TrackedFeature> {
        let num_levels = self.max_levels.min(prev_pyramid.num_levels()).min(curr_pyramid.num_levels());

        features
            .iter()
            .map(|feat| self.track_single(prev_pyramid, curr_pyramid, feat, num_levels))
            .collect()
    }

    /// Track a single feature through the pyramid, coarse-to-fine.
    fn track_single(
        &self,
        prev_pyr: &Pyramid,
        curr_pyr: &Pyramid,
        feature: &Feature,
        num_levels: usize,
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
                    self.lk_inverse_compositional(prev_img, curr_img, feat_x, feat_y, dx, dy)
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

        for _iter in 0..self.max_iterations {
            // Accumulators for the 2×2 Hessian and 2×1 right-hand side.
            let mut h00 = 0.0f32;
            let mut h01 = 0.0f32;
            let mut h11 = 0.0f32;
            let mut b0 = 0.0f32;
            let mut b1 = 0.0f32;
            for py in -half_i..=half_i {
                for px in -half_i..=half_i {
                    let px_f = px as f32;
                    let py_f = py as f32;

                    // Template pixel from previous frame (at original feature position).
                    let t_val = interpolate_bilinear(
                        prev_img,
                        feat_x + px_f,
                        feat_y + py_f,
                    );

                    // Warped pixel from current frame (at feature + displacement).
                    let wx = feat_x + dx + px_f;
                    let wy = feat_y + dy + py_f;
                    let i_val = interpolate_bilinear(curr_img, wx, wy);

                    // Error.
                    let e = t_val - i_val;

                    // Forward additive: gradients at the warped position
                    // in the current frame. Central differences, ±1 pixel.
                    let gx = 0.5
                        * (interpolate_bilinear(curr_img, wx + 1.0, wy)
                            - interpolate_bilinear(curr_img, wx - 1.0, wy));
                    let gy = 0.5
                        * (interpolate_bilinear(curr_img, wx, wy + 1.0)
                            - interpolate_bilinear(curr_img, wx, wy - 1.0));

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

    /// Iterative inverse-compositional Lucas-Kanade at a single pyramid level.
    ///
    /// Baker & Matthews (2004): gradients are evaluated at the template
    /// position in the previous frame. Since the template doesn't change,
    /// the Hessian H = J^T * J is constant across iterations — computed
    /// once, then only the error image is recomputed per iteration.
    ///
    /// This is what your C tracker does on the GAP8. The key insight:
    /// instead of asking "where did the template warp to?", IC asks
    /// "what incremental warp of the template best explains the error?"
    /// and then inverts that to update the displacement.
    ///
    /// Cost per iteration: (2W+1)² × 3 bilinear lookups (template + warped + error)
    /// vs. forward additive's (2W+1)² × 5 (template + warped + 2 gradient + error).
    /// The Hessian inverse is just a 2×2 constant.
    fn lk_inverse_compositional(
        &self,
        prev_img: &Image<f32>,
        curr_img: &Image<f32>,
        feat_x: f32,
        feat_y: f32,
        mut dx: f32,
        mut dy: f32,
    ) -> LkResult {
        let half_i = self.window_size as isize;

        // --- Precompute template gradients and Hessian (constant across iterations) ---
        let patch_size = (2 * self.window_size + 1) * (2 * self.window_size + 1);
        let mut gx_buf = vec![0.0f32; patch_size];
        let mut gy_buf = vec![0.0f32; patch_size];

        let mut h00 = 0.0f32;
        let mut h01 = 0.0f32;
        let mut h11 = 0.0f32;

        let mut idx = 0;
        for py in -half_i..=half_i {
            for px in -half_i..=half_i {
                let tx = feat_x + px as f32;
                let ty = feat_y + py as f32;

                // Gradient of the template (previous frame) at the
                // original feature position. Central differences, ±1 pixel.
                let gx = 0.5
                    * (interpolate_bilinear(prev_img, tx + 1.0, ty)
                        - interpolate_bilinear(prev_img, tx - 1.0, ty));
                let gy = 0.5
                    * (interpolate_bilinear(prev_img, tx, ty + 1.0)
                        - interpolate_bilinear(prev_img, tx, ty - 1.0));

                gx_buf[idx] = gx;
                gy_buf[idx] = gy;

                h00 += gx * gx;
                h01 += gx * gy;
                h11 += gy * gy;

                idx += 1;
            }
        }

        // Precompute H^{-1} (constant!).
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-6 {
            return LkResult::Singular;
        }
        let inv_det = 1.0 / det;

        // H^{-1} elements (2×2 symmetric).
        let ih00 = inv_det * h11;
        let ih01 = -inv_det * h01;
        let ih11 = inv_det * h00;

        // --- Iterate: only recompute error and b each iteration ---
        for _iter in 0..self.max_iterations {
            let mut b0 = 0.0f32;
            let mut b1 = 0.0f32;

            let mut idx = 0;
            for py in -half_i..=half_i {
                for px in -half_i..=half_i {
                    let px_f = px as f32;
                    let py_f = py as f32;

                    // Template pixel (previous frame).
                    let t_val = interpolate_bilinear(
                        prev_img,
                        feat_x + px_f,
                        feat_y + py_f,
                    );

                    // Warped pixel (current frame at feature + displacement).
                    let i_val = interpolate_bilinear(
                        curr_img,
                        feat_x + dx + px_f,
                        feat_y + dy + py_f,
                    );

                    let e = t_val - i_val;

                    // Use precomputed template gradients.
                    b0 += gx_buf[idx] * e;
                    b1 += gy_buf[idx] * e;

                    idx += 1;
                }
            }

            // delta = H^{-1} * b (using precomputed inverse).
            let delta_x = ih00 * b0 + ih01 * b1;
            let delta_y = ih01 * b0 + ih11 * b1;

            // In IC, the update is conceptually applied to the template warp
            // and then inverted. For pure translation, this simplifies to
            // the same additive update as forward additive:
            dx += delta_x;
            dy += delta_y;

            if delta_x * delta_x + delta_y * delta_y < self.epsilon * self.epsilon {
                return LkResult::Converged(dx, dy);
            }
        }

        LkResult::MaxIter(dx, dy)
    }
}

/// Internal result of iterative LK at one pyramid level.
enum LkResult {
    Converged(f32, f32),
    MaxIter(f32, f32),
    Singular,
}

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
