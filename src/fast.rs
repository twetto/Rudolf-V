// fast.rs — FAST corner detector (Features from Accelerated Segment Test).
//
// Reference: Rosten & Drummond, "Machine learning for high-speed corner
// detection" (ECCV 2006). vilib includes Rosten's original C implementation.
//
// Algorithm:
//   For each pixel, sample 16 points on a Bresenham circle of radius 3.
//   Classify each as BRIGHTER, DARKER, or SIMILAR relative to the center ±
//   threshold. A corner exists if ≥ N contiguous circle pixels are all
//   BRIGHTER or all DARKER.
//
// The "contiguous" check must wrap around the circle (index 15 is adjacent
// to index 0). The standard trick is to duplicate the 16-element classification
// array into a 32-element array and scan for a run of length N.
//
// This mirrors vilib's `fast_gpu_cuda_tools.cu`.
//
// NEW RUST CONCEPTS:
// - `struct` with methods via `impl` blocks
// - `Vec<Feature>` — dynamic collection of results
// - Const arrays for lookup tables (the Bresenham offsets)

use crate::image::Image;

/// Bresenham circle of radius 3: 16 (dx, dy) offsets.
/// Listed clockwise starting from 12 o'clock, matching Rosten's convention.
const CIRCLE_OFFSETS: [(isize, isize); 16] = [
    ( 0, -3), ( 1, -3), ( 2, -2), ( 3, -1),
    ( 3,  0), ( 3,  1), ( 2,  2), ( 1,  3),
    ( 0,  3), (-1,  3), (-2,  2), (-3,  1),
    (-3,  0), (-3, -1), (-2, -2), (-1, -3),
];

/// A detected feature point.
#[derive(Debug, Clone)]
pub struct Feature {
    /// Sub-pixel x coordinate (integer cast to f32 for FAST; sub-pixel
    /// refinement comes later with KLT tracking).
    pub x: f32,
    /// Sub-pixel y coordinate.
    pub y: f32,
    /// Corner response score (sum of |circle[i] - center| - threshold
    /// for qualifying pixels). Higher = stronger corner.
    pub score: f32,
    /// Pyramid level where this feature was detected (0 = original resolution).
    pub level: usize,
    /// Unique feature ID. 0 for newly detected features; assigned by the
    /// tracker when a feature is first tracked across frames.
    pub id: u64,
}

/// FAST-N corner detector.
///
/// Configurable threshold and arc length (N in FAST-N).
/// Common choices: FAST-9 (more features, some noise) or FAST-12 (fewer,
/// more robust).
pub struct FastDetector {
    /// Intensity difference threshold. A circle pixel is classified as
    /// BRIGHTER/DARKER only if it differs from the center by more than
    /// this value. Typical: 20–40 for u8 images.
    pub threshold: u8,
    /// Minimum number of contiguous circle pixels required.
    /// Must be in [9, 12]. FAST-9 and FAST-12 are the most common.
    pub arc_length: usize,
}

impl FastDetector {
    /// Create a new FAST detector.
    ///
    /// # Panics
    /// Panics if `arc_length` is not in the range [9, 12].
    pub fn new(threshold: u8, arc_length: usize) -> Self {
        assert!(
            (9..=12).contains(&arc_length),
            "arc_length must be 9..=12 (got {arc_length})"
        );
        FastDetector {
            threshold,
            arc_length,
        }
    }

    /// Detect FAST corners in a u8 grayscale image.
    ///
    /// Features are returned with `level = 0` and `id = 0`.
    /// Apply NMS afterwards to suppress weak nearby detections.
    pub fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        self.detect_at_level(image, 0)
    }

    /// Detect FAST corners, tagging features with the given pyramid level.
    ///
    /// Used internally when running FAST on each pyramid level.
    pub fn detect_at_level(&self, image: &Image<u8>, level: usize) -> Vec<Feature> {
        let w = image.width();
        let h = image.height();
        let mut features = Vec::new();

        // Skip a 3-pixel border since the Bresenham circle has radius 3.
        if w <= 6 || h <= 6 {
            return features;
        }

        let thresh = self.threshold as i16;
        let min_cardinals: u8 = if self.arc_length >= 12 { 3 } else { 2 };

        for y in 3..(h - 3) {
            for x in 3..(w - 3) {
                // SAFETY: x in [3, w-3) and y in [3, h-3), and all circle
                // offsets are at most ±3, so every access is in bounds.
                unsafe {
                let center = image.get_unchecked(x, y) as i16;

                // --- Quick rejection (Rosten's high-speed test) ---
                // Check 4 cardinal points (top, right, bottom, left).
                let p0 = image.get_unchecked(
                    (x as isize + CIRCLE_OFFSETS[0].0) as usize,
                    (y as isize + CIRCLE_OFFSETS[0].1) as usize) as i16;
                let p4 = image.get_unchecked(
                    (x as isize + CIRCLE_OFFSETS[4].0) as usize,
                    (y as isize + CIRCLE_OFFSETS[4].1) as usize) as i16;
                let p8 = image.get_unchecked(
                    (x as isize + CIRCLE_OFFSETS[8].0) as usize,
                    (y as isize + CIRCLE_OFFSETS[8].1) as usize) as i16;
                let p12 = image.get_unchecked(
                    (x as isize + CIRCLE_OFFSETS[12].0) as usize,
                    (y as isize + CIRCLE_OFFSETS[12].1) as usize) as i16;

                let bright_count = (p0 > center + thresh) as u8
                    + (p4 > center + thresh) as u8
                    + (p8 > center + thresh) as u8
                    + (p12 > center + thresh) as u8;
                let dark_count = (p0 < center - thresh) as u8
                    + (p4 < center - thresh) as u8
                    + (p8 < center - thresh) as u8
                    + (p12 < center - thresh) as u8;

                if bright_count < min_cardinals && dark_count < min_cardinals {
                    continue;
                }

                // --- Full 16-point test ---
                let mut circle_vals = [0i16; 16];
                for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                    circle_vals[i] = image.get_unchecked(
                        (x as isize + dx) as usize,
                        (y as isize + dy) as usize,
                    ) as i16;
                }

                let (is_corner, score) =
                    self.check_contiguous_and_score(center, &circle_vals, thresh);

                if is_corner {
                    features.push(Feature {
                        x: x as f32,
                        y: y as f32,
                        score,
                        level,
                        id: 0,
                    });
                }
                } // unsafe
            }
        }

        features
    }

    /// Check whether N contiguous circle pixels are all brighter or all
    /// darker than center ± threshold, and compute the corner score.
    ///
    /// Uses a bitmask approach instead of a doubled-array scan:
    ///   1. Build u16 bright_mask / dark_mask from threshold classification.
    ///   2. Check for N contiguous set bits via repeated AND-shift.
    ///   3. If corner, find the longest run and score it.
    ///
    /// The repeated AND-shift is branchless and maps directly to GPU
    /// bitwise ops in WGSL (u32 & (u32 >> 1) etc.).
    ///
    /// Returns (is_corner, score). Score = sum of (|diff| - threshold)
    /// for the qualifying pixels on the best arc.
    fn check_contiguous_and_score(
        &self,
        center: i16,
        circle: &[i16; 16],
        thresh: i16,
    ) -> (bool, f32) {
        let n = self.arc_length;

        // Build bitmasks: bit i set if circle[i] is brighter/darker.
        let mut bright_mask: u16 = 0;
        let mut dark_mask: u16 = 0;
        for i in 0..16 {
            let diff = circle[i] - center;
            if diff > thresh {
                bright_mask |= 1 << i;
            } else if diff < -thresh {
                dark_mask |= 1 << i;
            }
        }

        // Quick popcount rejection: need at least N bits set.
        let bright_has = bright_mask.count_ones() as usize >= n;
        let dark_has = dark_mask.count_ones() as usize >= n;
        if !bright_has && !dark_has {
            return (false, 0.0);
        }

        // Check for N contiguous set bits in a circular 16-bit pattern.
        // Method: double the mask into u32 to handle wrap-around, then
        // AND-shift N-1 times. If result is nonzero, there's a run of N.
        let mut best_score = -1.0f32;

        if bright_has {
            let m32 = (bright_mask as u32) | ((bright_mask as u32) << 16);
            let mut acc = m32;
            for _ in 1..n {
                acc &= acc >> 1;
            }
            if acc != 0 {
                // Corner found. Score the longest bright arc.
                let score = self.bitmask_best_arc_score(
                    center, circle, thresh, bright_mask);
                best_score = best_score.max(score);
            }
        }

        if dark_has {
            let m32 = (dark_mask as u32) | ((dark_mask as u32) << 16);
            let mut acc = m32;
            for _ in 1..n {
                acc &= acc >> 1;
            }
            if acc != 0 {
                let score = self.bitmask_best_arc_score(
                    center, circle, thresh, dark_mask);
                best_score = best_score.max(score);
            }
        }

        if best_score >= 0.0 {
            (true, best_score)
        } else {
            (false, 0.0)
        }
    }

    /// Find the longest contiguous arc in a circular 16-bit mask and
    /// compute its score. Used only for confirmed corners (rare path).
    #[inline]
    fn bitmask_best_arc_score(
        &self,
        center: i16,
        circle: &[i16; 16],
        thresh: i16,
        mask: u16,
    ) -> f32 {
        // Find all runs and pick the longest. Use the doubled u32 trick.
        let m32 = (mask as u32) | ((mask as u32) << 16);
        let mut best_start = 0usize;
        let mut best_len = 0usize;
        let mut i = 0u32;
        while i < 16 {
            if m32 & (1 << i) == 0 {
                i += 1;
                continue;
            }
            let start = i;
            while i < 32 && (m32 & (1 << i)) != 0 {
                i += 1;
            }
            let run_len = (i - start) as usize;
            if run_len > best_len {
                best_len = run_len;
                best_start = start as usize;
            }
        }

        // Score the best arc.
        let mut score = 0.0f32;
        for j in best_start..best_start + best_len {
            let idx = j % 16;
            let diff = (circle[idx] - center).abs() - thresh;
            score += diff.max(0) as f32;
        }
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a blank image and plant a bright cross pattern that should
    /// trigger FAST at the center.
    fn make_fast_corner_image(size: usize, center_val: u8, ring_val: u8) -> Image<u8> {
        let mut img = Image::from_vec(size, size, vec![center_val; size * size]);
        let cx = size / 2;
        let cy = size / 2;
        // Set all 16 circle pixels to ring_val.
        for &(dx, dy) in &CIRCLE_OFFSETS {
            let px = (cx as isize + dx) as usize;
            let py = (cy as isize + dy) as usize;
            img.set(px, py, ring_val);
        }
        img
    }

    #[test]
    fn test_bright_corner() {
        // Center = 50, ring = 200. diff = 150, well above any threshold.
        let img = make_fast_corner_image(20, 50, 200);
        let det = FastDetector::new(30, 9);
        let features = det.detect(&img);
        // Should detect at least one corner near the center.
        assert!(
            !features.is_empty(),
            "expected at least one bright corner"
        );
        // The planted ring creates corners at and around (10,10).
        // Check that at least one feature is within 4px of the center.
        let near_center = features.iter().any(|f| {
            (f.x - 10.0).abs() <= 4.0 && (f.y - 10.0).abs() <= 4.0
        });
        assert!(near_center, "expected a feature near (10, 10)");
        assert!(features[0].score > 0.0);
    }

    #[test]
    fn test_dark_corner() {
        // Center = 200, ring = 20. All circle pixels are darker.
        let img = make_fast_corner_image(20, 200, 20);
        let det = FastDetector::new(30, 9);
        let features = det.detect(&img);
        assert!(
            !features.is_empty(),
            "expected at least one dark corner"
        );
    }

    #[test]
    fn test_no_corner_flat() {
        // Uniform image — no corners anywhere.
        let img = Image::from_vec(20, 20, vec![128u8; 400]);
        let det = FastDetector::new(20, 9);
        let features = det.detect(&img);
        assert!(features.is_empty(), "flat image should have no corners");
    }

    #[test]
    fn test_threshold_sensitivity() {
        // With a small intensity difference, a high threshold should reject.
        let img = make_fast_corner_image(20, 100, 115); // diff = 15
        let det_low = FastDetector::new(10, 9);  // threshold 10 → detect
        let det_high = FastDetector::new(20, 9); // threshold 20 → reject

        assert!(!det_low.detect(&img).is_empty(), "low threshold should detect");
        assert!(det_high.detect(&img).is_empty(), "high threshold should reject");
    }

    #[test]
    fn test_arc_length_sensitivity() {
        // Plant only 10 contiguous bright pixels on the circle (not all 16).
        let mut img = Image::from_vec(20, 20, vec![100u8; 400]);
        let cx = 10usize;
        let cy = 10usize;
        for i in 0..10 {
            let (dx, dy) = CIRCLE_OFFSETS[i];
            img.set((cx as isize + dx) as usize, (cy as isize + dy) as usize, 200);
        }
        // FAST-9 should detect a corner at (10,10) (10 ≥ 9).
        // FAST-12 should NOT detect at (10,10) (10 < 12).
        // Note: the planted ring pixels may themselves trigger corners
        // elsewhere, so we check specifically for the center pixel.
        let det9 = FastDetector::new(20, 9);
        let det12 = FastDetector::new(20, 12);

        let has_center = |features: &[Feature]| {
            features.iter().any(|f| f.x as usize == cx && f.y as usize == cy)
        };

        let f9 = det9.detect(&img);
        let f12 = det12.detect(&img);
        assert!(has_center(&f9), "FAST-9 should detect corner at center");
        assert!(!has_center(&f12), "FAST-12 should not detect corner at center");
    }

    #[test]
    fn test_border_exclusion() {
        // Features within 3px of the border should not be detected.
        // Place a "corner" at (2, 2) — inside the skip zone.
        let mut img = Image::from_vec(20, 20, vec![100u8; 400]);
        // This would be out of bounds for circle sampling anyway,
        // so FAST should skip it.
        img.set(2, 2, 200);
        let det = FastDetector::new(10, 9);
        let features = det.detect(&img);
        for f in &features {
            assert!(f.x >= 3.0 && f.y >= 3.0, "feature too close to border");
        }
    }

    #[test]
    fn test_image_too_small() {
        // 6×6 or smaller → no room for the 3-pixel border on both sides.
        let img: Image<u8> = Image::new(6, 6);
        let det = FastDetector::new(20, 9);
        assert!(det.detect(&img).is_empty());
    }

    #[test]
    fn test_score_increases_with_contrast() {
        // Higher contrast should yield a higher score.
        let img_low = make_fast_corner_image(20, 100, 140);  // diff = 40
        let img_high = make_fast_corner_image(20, 100, 220);  // diff = 120

        let det = FastDetector::new(20, 9);
        let f_low = det.detect(&img_low);
        let f_high = det.detect(&img_high);

        assert!(!f_low.is_empty() && !f_high.is_empty());
        assert!(
            f_high[0].score > f_low[0].score,
            "higher contrast should give higher score: {} vs {}",
            f_high[0].score,
            f_low[0].score,
        );
    }

    #[test]
    fn test_detect_at_level() {
        let img = make_fast_corner_image(20, 50, 200);
        let det = FastDetector::new(30, 9);
        let features = det.detect_at_level(&img, 3);
        assert!(!features.is_empty());
        assert_eq!(features[0].level, 3);
    }

    #[test]
    #[should_panic(expected = "arc_length")]
    fn test_invalid_arc_length() {
        FastDetector::new(20, 7);
    }
}
