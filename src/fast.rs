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
// OPTIMIZATIONS:
//
// - PRECOMPUTED FLAT OFFSETS: Instead of computing (y+dy)*stride+(x+dx)
//   per circle pixel, precompute 16 flat offsets as dy*stride+dx once.
//   The inner loop becomes base pointer + indexed reads — eliminates
//   20 multiplies per pixel candidate (16 full + 4 cardinal).
//
// - DIRECT SLICE ACCESS: Work on the raw image slice with flat offsets,
//   bypassing per-pixel method call overhead.
//
// - CARDINAL EARLY REJECT (Rosten's high-speed test): Check 4 cardinal
//   points {0,4,8,12} first. At least 2 of 4 (3 for FAST-12) must be
//   brighter or darker. Rejects ~85% of non-corner pixels with 4
//   comparisons instead of 16.
//
// - BITMASK CONTIGUOUS CHECK: Build u16 bright/dark masks, popcount
//   reject, then AND-shift for N contiguous bits. Branchless, maps
//   directly to GPU bitwise ops in WGSL.
//
// GPU MAPPING: Each pixel maps to one thread. The flat-offset pattern
// mirrors texture sampling with fixed offsets. The bitmask approach
// maps to WGSL u32 bitwise ops.

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

        // Skip a 3-pixel border since the Bresenham circle has radius 3.
        if w <= 6 || h <= 6 {
            return Vec::new();
        }

        let stride = image.stride();
        let slice = image.as_slice();

        // Precompute flat offsets: circle_off[i] = dy * stride + dx.
        // Adding this to (y * stride + x) gives the flat index of circle
        // pixel i. Eliminates a multiply per circle pixel in the inner loop.
        let circle_off: [isize; 16] = {
            let mut off = [0isize; 16];
            for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
                off[i] = dy * stride as isize + dx;
            }
            off
        };

        // Cardinal offsets for the quick-reject test (indices 0, 4, 8, 12).
        let card = [circle_off[0], circle_off[4], circle_off[8], circle_off[12]];

        let thresh = self.threshold as i16;
        let arc_length = self.arc_length;
        let min_cardinals: u8 = if arc_length >= 12 { 3 } else { 2 };

        let mut features = Vec::new();

        for y in 3..(h - 3) {
            let row_base = y * stride;

            for x in 3..(w - 3) {
                let base = row_base + x;

                // SAFETY: x in [3, w-3), y in [3, h-3), all circle offsets
                // are at most ±3 in each dimension, so base + offset is
                // always within the image slice.
                unsafe {
                let center = *slice.get_unchecked(base) as i16;
                let hi = center + thresh;
                let lo = center - thresh;

                // --- Quick rejection (Rosten's high-speed test) ---
                // Check 4 cardinal points. At least min_cardinals must be
                // brighter (or darker) to have any chance of an N-arc.
                let p0  = *slice.get_unchecked((base as isize + card[0]) as usize) as i16;
                let p4  = *slice.get_unchecked((base as isize + card[1]) as usize) as i16;
                let p8  = *slice.get_unchecked((base as isize + card[2]) as usize) as i16;
                let p12 = *slice.get_unchecked((base as isize + card[3]) as usize) as i16;

                let bright_count = (p0 > hi) as u8
                    + (p4 > hi) as u8
                    + (p8 > hi) as u8
                    + (p12 > hi) as u8;
                let dark_count = (p0 < lo) as u8
                    + (p4 < lo) as u8
                    + (p8 < lo) as u8
                    + (p12 < lo) as u8;

                if bright_count < min_cardinals && dark_count < min_cardinals {
                    continue;
                }

                // --- Full 16-point test ---
                // Build bitmasks and cache circle values in one pass.
                let mut bright_mask: u16 = 0;
                let mut dark_mask: u16 = 0;
                let mut circle_vals = [0i16; 16];

                for i in 0..16 {
                    let v = *slice.get_unchecked(
                        (base as isize + circle_off[i]) as usize
                    ) as i16;
                    circle_vals[i] = v;
                    if v > hi {
                        bright_mask |= 1 << i;
                    } else if v < lo {
                        dark_mask |= 1 << i;
                    }
                }

                // Quick popcount rejection: need at least N bits set.
                let bright_has = bright_mask.count_ones() as usize >= arc_length;
                let dark_has = dark_mask.count_ones() as usize >= arc_length;
                if !bright_has && !dark_has {
                    continue;
                }

                // Check for N contiguous set bits in a circular 16-bit pattern.
                // Double the mask into u32 to handle wrap-around, then
                // AND-shift N-1 times. Nonzero result = run of N exists.
                let mut best_score = -1.0f32;

                if bright_has {
                    let m32 = (bright_mask as u32) | ((bright_mask as u32) << 16);
                    let mut acc = m32;
                    for _ in 1..arc_length {
                        acc &= acc >> 1;
                    }
                    if acc != 0 {
                        let score = bitmask_best_arc_score(
                            center, &circle_vals, thresh, bright_mask);
                        best_score = best_score.max(score);
                    }
                }

                if dark_has {
                    let m32 = (dark_mask as u32) | ((dark_mask as u32) << 16);
                    let mut acc = m32;
                    for _ in 1..arc_length {
                        acc &= acc >> 1;
                    }
                    if acc != 0 {
                        let score = bitmask_best_arc_score(
                            center, &circle_vals, thresh, dark_mask);
                        best_score = best_score.max(score);
                    }
                }

                if best_score >= 0.0 {
                    features.push(Feature {
                        x: x as f32,
                        y: y as f32,
                        score: best_score,
                        level,
                        id: 0,
                    });
                }
                } // unsafe
            }
        }

        features
    }
}

/// Find the longest contiguous arc in a circular 16-bit mask and
/// compute its score. Used only for confirmed corners (rare path).
#[inline]
fn bitmask_best_arc_score(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a blank image and plant a bright cross pattern that should
    /// trigger FAST at the center.
    fn make_fast_corner_image(size: usize, center_val: u8, ring_val: u8) -> Image<u8> {
        let mut img = Image::from_vec(size, size, vec![center_val; size * size]);
        let cx = size / 2;
        let cy = size / 2;
        for &(dx, dy) in &CIRCLE_OFFSETS {
            let px = (cx as isize + dx) as usize;
            let py = (cy as isize + dy) as usize;
            img.set(px, py, ring_val);
        }
        img
    }

    #[test]
    fn test_bright_corner() {
        let img = make_fast_corner_image(20, 50, 200);
        let det = FastDetector::new(30, 9);
        let features = det.detect(&img);
        assert!(!features.is_empty(), "expected at least one bright corner");
        let near_center = features.iter().any(|f| {
            (f.x - 10.0).abs() <= 4.0 && (f.y - 10.0).abs() <= 4.0
        });
        assert!(near_center, "expected a feature near (10, 10)");
        assert!(features[0].score > 0.0);
    }

    #[test]
    fn test_dark_corner() {
        let img = make_fast_corner_image(20, 200, 20);
        let det = FastDetector::new(30, 9);
        let features = det.detect(&img);
        assert!(!features.is_empty(), "expected at least one dark corner");
    }

    #[test]
    fn test_no_corner_flat() {
        let img = Image::from_vec(20, 20, vec![128u8; 400]);
        let det = FastDetector::new(20, 9);
        let features = det.detect(&img);
        assert!(features.is_empty(), "flat image should have no corners");
    }

    #[test]
    fn test_threshold_sensitivity() {
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
        let mut img = Image::from_vec(20, 20, vec![100u8; 400]);
        img.set(2, 2, 200);
        let det = FastDetector::new(10, 9);
        let features = det.detect(&img);
        for f in &features {
            assert!(f.x >= 3.0 && f.y >= 3.0, "feature too close to border");
        }
    }

    #[test]
    fn test_image_too_small() {
        let img: Image<u8> = Image::new(6, 6);
        let det = FastDetector::new(20, 9);
        assert!(det.detect(&img).is_empty());
    }

    #[test]
    fn test_score_increases_with_contrast() {
        let img_low = make_fast_corner_image(20, 100, 140);  // diff = 40
        let img_high = make_fast_corner_image(20, 100, 220);  // diff = 120

        let det = FastDetector::new(20, 9);
        let f_low = det.detect(&img_low);
        let f_high = det.detect(&img_high);

        assert!(!f_low.is_empty() && !f_high.is_empty());
        assert!(
            f_high[0].score > f_low[0].score,
            "higher contrast should give higher score: {} vs {}",
            f_high[0].score, f_low[0].score,
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
