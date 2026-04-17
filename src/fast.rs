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
// OPTIMIZATIONS:
//
// - PRECOMPUTED FLAT OFFSETS: 16 circle offsets as dy*stride+dx.
//   Inner loop becomes base + indexed reads — eliminates per-pixel
//   address arithmetic.
//
// - CARDINAL EARLY REJECT (Rosten's high-speed test): Check 4 cardinal
//   points {0,4,8,12} first. Rejects ~85% of non-corner pixels.
//
// - BITMASK CONTIGUOUS CHECK: u16 bright/dark masks, popcount reject,
//   AND-shift for N contiguous bits. Branchless.
//
// - INLINE OCCUPANCY SKIP: When an occupancy grid is provided, the x-loop
//   checks one bool per cell boundary and jumps past occupied cells.
//   No mask allocation, no Vec of ranges — just a while loop with an
//   integer division every cell_size pixels.
//
// - RAYON ROW PARALLELISM (feature-gated): Each row is independent.
//
// GPU MAPPING: Each pixel maps to one thread. The flat-offset pattern
// mirrors texture sampling with fixed offsets. The bitmask approach
// maps to WGSL u32 bitwise ops.

use crate::image::Image;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub level: usize,
    pub id: u64,
    pub descriptor: u16,
}

/// FAST-N corner detector.
///
/// Configurable threshold and arc length (N in FAST-N).
/// Common choices: FAST-9 (more features, some noise) or FAST-12 (fewer,
/// more robust).
pub struct FastDetector {
    /// Intensity difference threshold. Typical: 20–40 for u8 images.
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

    /// Detect FAST corners in the entire image.
    ///
    /// Features are returned with `level = 0` and `id = 0`.
    pub fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        // Empty grid → no occupancy skipping, scans everything.
        self.detect_unoccupied(image, &[], 0, 1)
    }

    /// Detect FAST corners, skipping occupied cells in the occupancy grid.
    ///
    /// At each cell boundary in the x-loop, checks one bool in the grid.
    /// If occupied, jumps past the entire cell — no mask image needed.
    ///
    /// Pass an empty `grid` slice to scan the entire image (same as `detect`).
    ///
    /// # Arguments
    /// * `grid` — flat bool array, row-major, `true` = occupied. Empty = no grid.
    /// * `grid_cols` — number of grid columns (ignored if grid is empty)
    /// * `cell_size` — pixel width of each grid cell (ignored if grid is empty)
    pub fn detect_unoccupied(
        &self,
        image: &Image<u8>,
        grid: &[bool],
        grid_cols: usize,
        cell_size: usize,
    ) -> Vec<Feature> {
        let w = image.width();
        let h = image.height();

        if w <= 6 || h <= 6 {
            return Vec::new();
        }

        let stride = image.stride();
        let slice = image.as_slice();
        let (circle_off, card) = precompute_offsets(stride);

        let thresh = self.threshold as i16;
        let arc_length = self.arc_length;
        let min_cardinals: u8 = if arc_length >= 12 { 3 } else { 2 };

        #[cfg(feature = "parallel")]
        {
            return (3..(h - 3))
                .into_par_iter()
                .flat_map(|y| {
                    let mut row_features = Vec::new();
                    detect_row(
                        slice, stride, w, y, 0,
                        &circle_off, &card,
                        thresh, arc_length, min_cardinals,
                        grid, grid_cols, cell_size,
                        &mut row_features,
                    );
                    row_features
                })
                .collect();
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut features = Vec::new();
            for y in 3..(h - 3) {
                detect_row(
                    slice, stride, w, y, 0,
                    &circle_off, &card,
                    thresh, arc_length, min_cardinals,
                    grid, grid_cols, cell_size,
                    &mut features,
                );
            }
            features
        }
    }

    /// Simple helper for coarser levels
    pub fn detect_at_level(&self, image: &Image<u8>, level: usize) -> Vec<Feature> {
        let mut features = self.detect(image);
        for f in &mut features {
            f.level = level;
        }
        features
    }
}

// ==========================================================================
// Internal helpers
// ==========================================================================

/// Precompute flat circle offsets and cardinal offsets for the given stride.
#[inline]
fn precompute_offsets(stride: usize) -> ([isize; 16], [isize; 4]) {
    let mut circle_off = [0isize; 16];
    for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
        circle_off[i] = dy * stride as isize + dx;
    }
    let card = [circle_off[0], circle_off[4], circle_off[8], circle_off[12]];
    (circle_off, card)
}

/// Process one row of FAST detection with inline occupancy skipping.
///
/// When `grid` is non-empty, checks one bool per cell boundary and jumps
/// past occupied cells. When `grid` is empty, scans the full row.
///
/// Uses a `while` loop so we can jump x forward by an entire cell width
/// when we hit an occupied cell.
#[inline]
fn detect_row(
    slice: &[u8],
    stride: usize,
    w: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    cardinal_off: &[isize; 4],
    thresh: i16,
    arc_length: usize,
    min_cardinals: u8,
    grid: &[bool],
    grid_cols: usize,
    cell_size: usize,
    features: &mut Vec<Feature>,
) {
    let row_base = y * stride;
    let has_grid = !grid.is_empty();
    let grid_row_off = if has_grid { (y / cell_size) * grid_cols } else { 0 };

    let x_min = 3usize;
    let x_max = w - 3;

    let mut x = x_min;
    while x < x_max {
        // ── Occupancy skip: one bool check per cell boundary ──
        if has_grid {
            let gc = x / cell_size;
            if gc < grid_cols && grid[grid_row_off + gc] {
                // Jump to the start of the next cell.
                x = (gc + 1) * cell_size;
                continue;
            }
        }

        let base = row_base + x;

        // SAFETY: x in [3, w-3), y in [3, h-3), all circle offsets
        // are at most ±3 in each dimension, so base + offset is
        // always within the image slice.
        unsafe {
        let center = *slice.get_unchecked(base) as i16;
        let hi = center + thresh;
        let lo = center - thresh;

        // --- Quick rejection (Rosten's high-speed test) ---
        let p0  = *slice.get_unchecked((base as isize + cardinal_off[0]) as usize) as i16;
        let p4  = *slice.get_unchecked((base as isize + cardinal_off[1]) as usize) as i16;
        let p8  = *slice.get_unchecked((base as isize + cardinal_off[2]) as usize) as i16;
        let p12 = *slice.get_unchecked((base as isize + cardinal_off[3]) as usize) as i16;

        let bright_count = (p0 > hi) as u8
            + (p4 > hi) as u8
            + (p8 > hi) as u8
            + (p12 > hi) as u8;
        let dark_count = (p0 < lo) as u8
            + (p4 < lo) as u8
            + (p8 < lo) as u8
            + (p12 < lo) as u8;

        if bright_count < min_cardinals && dark_count < min_cardinals {
            x += 1;
            continue;
        }

        // --- Full 16-point test ---
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

        // Quick popcount rejection.
        let bright_has = bright_mask.count_ones() as usize >= arc_length;
        let dark_has = dark_mask.count_ones() as usize >= arc_length;
        if !bright_has && !dark_has {
            x += 1;
            continue;
        }

        // Contiguous-arc check + scoring.
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
            // Compute RI-LBP descriptor (native radius 3).
            let mut lbp: u16 = 0;
            for i in 0..16 {
                if circle_vals[i] >= center {
                    lbp |= 1 << i;
                }
            }
            let descriptor = compute_min_rotation(lbp);

            features.push(Feature {
                x: x as f32,
                y: y as f32,
                score: best_score,
                level,
                id: 0,
                descriptor,
            });
        }
        } // unsafe

        x += 1;
    }
}

/// Compute the rotation-invariant LBP by finding the minimum value
/// among all 16 cyclic shifts.
#[inline]
fn compute_min_rotation(mut v: u16) -> u16 {
    let mut min_v = v;
    for _ in 0..15 {
        v = v.rotate_right(1);
        if v < min_v {
            min_v = v;
        }
    }
    min_v
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

    let mut score = 0.0f32;
    for j in best_start..best_start + best_len {
        let idx = j % 16;
        let diff = (circle[idx] - center).abs() - thresh;
        score += diff.max(0) as f32;
    }
    score
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_ri_lbp_stability() {
        // Create an image with a clear corner.
        let img = make_fast_corner_image(20, 100, 200);
        let det = FastDetector::new(50, 9);
        let f1 = det.detect(&img);
        assert!(!f1.is_empty());

        // "Rotate" the circle by shifting pixel values.
        // Since make_fast_corner_image uses CIRCLE_OFFSETS, we can't easily
        // rotate the whole image, but we can verify that compute_min_rotation
        // works on shifted masks.
        let mut v: u16 = 0b1111111110000000;
        let d_ref = compute_min_rotation(v);
        for _ in 0..16 {
            v = v.rotate_right(1);
            assert_eq!(compute_min_rotation(v), d_ref, "RI-LBP should be invariant to rotation");
        }
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
        let img = make_fast_corner_image(20, 100, 115);
        let det_low = FastDetector::new(10, 9);
        let det_high = FastDetector::new(20, 9);

        assert!(!det_low.detect(&img).is_empty(), "low threshold should detect");
        assert!(det_high.detect(&img).is_empty(), "high threshold should reject");
    }

    #[test]
    fn test_arc_length_sensitivity() {
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
        let img_low = make_fast_corner_image(20, 100, 140);
        let img_high = make_fast_corner_image(20, 100, 220);

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
    #[should_panic(expected = "arc_length")]
    fn test_invalid_arc_length() {
        FastDetector::new(20, 7);
    }

    // ===== Grid-skip tests =====

    #[test]
    fn test_detect_unoccupied_all_empty() {
        let img = make_fast_corner_image(40, 50, 200);
        let det = FastDetector::new(30, 9);

        let cols = (40 + 15) / 16;
        let rows = (40 + 15) / 16;
        let grid = vec![false; cols * rows];

        let full = det.detect(&img);
        let skip = det.detect_unoccupied(&img, &grid, cols, 16);

        assert_eq!(full.len(), skip.len(),
            "all-empty grid should match full detect");
    }

    #[test]
    fn test_detect_unoccupied_all_occupied() {
        let img = make_fast_corner_image(40, 50, 200);
        let det = FastDetector::new(30, 9);

        let cols = (40 + 15) / 16;
        let rows = (40 + 15) / 16;
        let grid = vec![true; cols * rows];

        let skip = det.detect_unoccupied(&img, &grid, cols, 16);
        assert!(skip.is_empty(), "all-occupied grid should detect nothing");
    }

    #[test]
    fn test_detect_unoccupied_filters_correctly() {
        let img = make_fast_corner_image(40, 50, 200);
        let det = FastDetector::new(30, 9);

        let cols = 3;
        let rows = 3;
        let mut grid = vec![false; cols * rows];
        grid[1 * cols + 1] = true; // center cell

        let full = det.detect(&img);
        let skip = det.detect_unoccupied(&img, &grid, cols, 16);

        assert!(full.iter().any(|f| f.x >= 16.0 && f.x < 32.0 && f.y >= 16.0 && f.y < 32.0),
            "full detect should find corner in center cell");
        assert!(!skip.iter().any(|f| f.x >= 16.0 && f.x < 32.0 && f.y >= 16.0 && f.y < 32.0),
            "grid-skip should not find corner in occupied center cell");
    }

    #[test]
    fn test_detect_delegates_to_unoccupied() {
        // detect() should produce identical results to detect_unoccupied with empty grid.
        let img = make_fast_corner_image(40, 50, 200);
        let det = FastDetector::new(30, 9);

        let a = det.detect(&img);
        let b = det.detect_unoccupied(&img, &[], 0, 1);

        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa.x, fb.x);
            assert_eq!(fa.y, fb.y);
            assert_eq!(fa.score, fb.score);
        }
    }
}
