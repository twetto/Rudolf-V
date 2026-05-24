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
// - CELL-RANGE OCCUPANCY SKIP: When an occupancy grid is provided, the
//   outer loop iterates grid cells with bit-shift indexing (cell_size must
//   be power of 2), skipping occupied cells entirely. The inner pixel loop
//   has zero grid overhead.
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
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
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

        let cell_shift = if !grid.is_empty() {
            debug_assert!(cell_size.is_power_of_two(), "cell_size must be power of 2");
            cell_size.trailing_zeros()
        } else {
            0
        };

        #[cfg(feature = "parallel")]
        {
            let mut features: Vec<_> = (3..(h - 3))
                .into_par_iter()
                .flat_map(|y| {
                    let mut row_features = Vec::new();
                    detect_row(
                        slice,
                        stride,
                        w,
                        y,
                        0,
                        &circle_off,
                        &card,
                        thresh,
                        arc_length,
                        min_cardinals,
                        grid,
                        grid_cols,
                        cell_shift,
                        &mut row_features,
                    );
                    row_features
                })
                .collect();
            features.sort_by(|a, b| {
                a.y.total_cmp(&b.y)
                    .then_with(|| a.x.total_cmp(&b.x))
                    .then_with(|| b.score.total_cmp(&a.score))
            });
            return features;
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut features = Vec::new();
            for y in 3..(h - 3) {
                detect_row(
                    slice,
                    stride,
                    w,
                    y,
                    0,
                    &circle_off,
                    &card,
                    thresh,
                    arc_length,
                    min_cardinals,
                    grid,
                    grid_cols,
                    cell_shift,
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

/// Process one row of FAST detection with cell-range occupancy skipping.
///
/// Outer loop iterates grid cells (~6 per row for 752px / 128-cell);
/// inner loop scans pixels within each unoccupied cell with zero grid
/// overhead. When `grid` is empty, scans the full row in one range.
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
    cell_shift: u32,
    features: &mut Vec<Feature>,
) {
    let row_base = y * stride;
    let has_grid = !grid.is_empty();
    let grid_row_off = if has_grid {
        (y >> cell_shift) * grid_cols
    } else {
        0
    };

    let x_min = 3usize;
    let x_max = w - 3;

    // No grid → single range covering the full row.
    if !has_grid {
        detect_row_range(
            slice,
            row_base,
            x_min,
            x_max,
            y,
            level,
            circle_off,
            cardinal_off,
            thresh,
            arc_length,
            min_cardinals,
            features,
        );
        return;
    }

    // ── Cell-range iteration: one grid check per cell boundary ──
    let mut cell_x = x_min;
    while cell_x < x_max {
        let gc = cell_x >> cell_shift;
        if gc < grid_cols && grid[grid_row_off + gc] {
            // Jump past the entire occupied cell.
            cell_x = (gc + 1) << cell_shift;
            continue;
        }
        let run_end = (((gc + 1) << cell_shift) as usize).min(x_max);
        detect_row_range(
            slice,
            row_base,
            cell_x,
            run_end,
            y,
            level,
            circle_off,
            cardinal_off,
            thresh,
            arc_length,
            min_cardinals,
            features,
        );
        cell_x = run_end;
    }
}

/// Scan pixels in [x_start, x_end) for FAST corners. No grid checks.
#[inline]
fn detect_row_range(
    slice: &[u8],
    row_base: usize,
    x_start: usize,
    x_end: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    cardinal_off: &[isize; 4],
    thresh: i16,
    arc_length: usize,
    min_cardinals: u8,
    features: &mut Vec<Feature>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                detect_row_range_avx2(
                    slice,
                    row_base,
                    x_start,
                    x_end,
                    y,
                    level,
                    circle_off,
                    cardinal_off,
                    thresh,
                    arc_length,
                    min_cardinals,
                    features,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        detect_row_range_neon(
            slice,
            row_base,
            x_start,
            x_end,
            y,
            level,
            circle_off,
            cardinal_off,
            thresh,
            arc_length,
            min_cardinals,
            features,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    detect_row_range_scalar(
        slice,
        row_base,
        x_start,
        x_end,
        y,
        level,
        circle_off,
        cardinal_off,
        thresh,
        arc_length,
        min_cardinals,
        features,
    );
}

/// Full 16-point FAST test + scoring + LBP for a single pixel that passed
/// the cardinal early-reject. Shared by both scalar and AVX2 paths.
#[inline]
unsafe fn fast_full_test(
    slice: &[u8],
    base: usize,
    x: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    thresh: i16,
    arc_length: usize,
    features: &mut Vec<Feature>,
) {
    let center = *slice.get_unchecked(base) as i16;
    let hi = center + thresh;
    let lo = center - thresh;

    // --- Full 16-point test ---
    let mut bright_mask: u16 = 0;
    let mut dark_mask: u16 = 0;
    let mut circle_vals = [0i16; 16];

    for i in 0..16 {
        let v = *slice.get_unchecked((base as isize + circle_off[i]) as usize) as i16;
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
        return;
    }

    // Contiguous-arc check + scoring.
    let mut best_score = -1.0f32;

    if bright_has {
        if has_contiguous_arc(bright_mask, arc_length) {
            let score = bitmask_best_arc_score(center, &circle_vals, thresh, bright_mask);
            best_score = best_score.max(score);
        }
    }

    if dark_has {
        if has_contiguous_arc(dark_mask, arc_length) {
            let score = bitmask_best_arc_score(center, &circle_vals, thresh, dark_mask);
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
}

/// Scalar scan — fallback for non-AVX2 and for the AVX2 tail.
#[inline]
fn detect_row_range_scalar(
    slice: &[u8],
    row_base: usize,
    x_start: usize,
    x_end: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    cardinal_off: &[isize; 4],
    thresh: i16,
    arc_length: usize,
    min_cardinals: u8,
    features: &mut Vec<Feature>,
) {
    let mut x = x_start;
    while x < x_end {
        let base = row_base + x;

        // SAFETY: x in [3, w-3), y in [3, h-3), all circle offsets
        // are at most ±3 in each dimension, so base + offset is
        // always within the image slice.
        unsafe {
            let center = *slice.get_unchecked(base) as i16;
            let hi = center + thresh;
            let lo = center - thresh;

            // --- Quick rejection (Rosten's high-speed test) ---
            let p0 = *slice.get_unchecked((base as isize + cardinal_off[0]) as usize) as i16;
            let p4 = *slice.get_unchecked((base as isize + cardinal_off[1]) as usize) as i16;
            let p8 = *slice.get_unchecked((base as isize + cardinal_off[2]) as usize) as i16;
            let p12 = *slice.get_unchecked((base as isize + cardinal_off[3]) as usize) as i16;

            let bright_count =
                (p0 > hi) as u8 + (p4 > hi) as u8 + (p8 > hi) as u8 + (p12 > hi) as u8;
            let dark_count = (p0 < lo) as u8 + (p4 < lo) as u8 + (p8 < lo) as u8 + (p12 < lo) as u8;

            if bright_count >= min_cardinals || dark_count >= min_cardinals {
                fast_full_test(
                    slice, base, x, y, level, circle_off, thresh, arc_length, features,
                );
            }
        } // unsafe

        x += 1;
    }
}

/// AVX2: process 32 center pixels at a time for cardinal early-reject,
/// then scalar full test for the ~15% that survive.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detect_row_range_avx2(
    slice: &[u8],
    row_base: usize,
    x_start: usize,
    x_end: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    cardinal_off: &[isize; 4],
    thresh: i16,
    arc_length: usize,
    min_cardinals: u8,
    features: &mut Vec<Feature>,
) {
    use std::arch::x86_64::*;

    let ptr = slice.as_ptr();
    let thresh_v = _mm256_set1_epi8(thresh as i8);
    let one = _mm256_set1_epi8(1);
    let min_card_v = _mm256_set1_epi8(min_cardinals as i8);

    let mut x = x_start;

    // --- AVX2 main loop: 32 pixels per iteration ---
    while x + 32 <= x_end {
        let base = row_base + x;
        let base_ptr = ptr.add(base);

        // Load 32 center pixels.
        let center = _mm256_loadu_si256(base_ptr as *const __m256i);

        // Saturating thresholds: c_hi = min(center + thresh, 255),
        //                        c_lo = max(center - thresh, 0).
        let c_hi = _mm256_adds_epu8(center, thresh_v);
        let c_lo = _mm256_subs_epu8(center, thresh_v);

        // Load 32 pixels at each cardinal offset.
        // SAFETY: same bounds guarantee as scalar — all offsets ±3 rows/cols.
        let p0 = _mm256_loadu_si256(base_ptr.offset(cardinal_off[0]) as *const __m256i);
        let p4 = _mm256_loadu_si256(base_ptr.offset(cardinal_off[1]) as *const __m256i);
        let p8 = _mm256_loadu_si256(base_ptr.offset(cardinal_off[2]) as *const __m256i);
        let p12 = _mm256_loadu_si256(base_ptr.offset(cardinal_off[3]) as *const __m256i);

        // Bright cardinal count: subs(p, c_hi) > 0 means p > center + thresh.
        // min(nonzero, 1) converts to 0/1, then sum the 4 cardinals.
        let b0 = _mm256_min_epu8(_mm256_subs_epu8(p0, c_hi), one);
        let b4 = _mm256_min_epu8(_mm256_subs_epu8(p4, c_hi), one);
        let b8 = _mm256_min_epu8(_mm256_subs_epu8(p8, c_hi), one);
        let b12 = _mm256_min_epu8(_mm256_subs_epu8(p12, c_hi), one);
        let bright_count = _mm256_add_epi8(_mm256_add_epi8(b0, b4), _mm256_add_epi8(b8, b12));

        // Dark cardinal count: subs(c_lo, p) > 0 means p < center - thresh.
        let d0 = _mm256_min_epu8(_mm256_subs_epu8(c_lo, p0), one);
        let d4 = _mm256_min_epu8(_mm256_subs_epu8(c_lo, p4), one);
        let d8 = _mm256_min_epu8(_mm256_subs_epu8(c_lo, p8), one);
        let d12 = _mm256_min_epu8(_mm256_subs_epu8(c_lo, p12), one);
        let dark_count = _mm256_add_epi8(_mm256_add_epi8(d0, d4), _mm256_add_epi8(d8, d12));

        // Pass if bright_count >= min_cardinals OR dark_count >= min_cardinals.
        // Unsigned compare a >= b: max(a, b) == a.
        let bright_ok = _mm256_cmpeq_epi8(_mm256_max_epu8(bright_count, min_card_v), bright_count);
        let dark_ok = _mm256_cmpeq_epi8(_mm256_max_epu8(dark_count, min_card_v), dark_count);
        let pass = _mm256_or_si256(bright_ok, dark_ok);
        let mut pass_mask = _mm256_movemask_epi8(pass) as u32;

        // Skip entire 32-pixel chunk if no candidates.
        if pass_mask == 0 {
            x += 32;
            continue;
        }

        // Full SIMD FAST-arc gate for cardinal survivors. Scoring,
        // descriptor computation, and Vec compaction stay scalar.
        pass_mask &= full_ring_pass_mask_avx2(ptr, base, circle_off, c_hi, c_lo, arc_length);

        // Scalar score/descriptor for each true FAST candidate.
        while pass_mask != 0 {
            let bit = pass_mask.trailing_zeros() as usize;
            pass_mask &= pass_mask - 1;
            fast_full_test(
                slice,
                row_base + x + bit,
                x + bit,
                y,
                level,
                circle_off,
                thresh,
                arc_length,
                features,
            );
        }

        x += 32;
    }

    // --- Scalar tail for remaining < 32 pixels ---
    detect_row_range_scalar(
        slice,
        row_base,
        x,
        x_end,
        y,
        level,
        circle_off,
        cardinal_off,
        thresh,
        arc_length,
        min_cardinals,
        features,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn full_ring_pass_mask_avx2(
    ptr: *const u8,
    base: usize,
    circle_off: &[isize; 16],
    c_hi: std::arch::x86_64::__m256i,
    c_lo: std::arch::x86_64::__m256i,
    arc_length: usize,
) -> u32 {
    use std::arch::x86_64::*;

    let base_ptr = ptr.add(base);
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi8(1);
    let arc = _mm256_set1_epi8(arc_length as i8);
    let mut bright_run = zero;
    let mut dark_run = zero;
    let mut max_run = zero;

    for k in 0..(16 + arc_length - 1) {
        let p = _mm256_loadu_si256(base_ptr.offset(circle_off[k & 15]) as *const __m256i);

        // p > c_hi and p < c_lo using saturated subtract. Clamp to 0/1 before
        // comparison so large unsigned differences are not misread as signed.
        let bright = _mm256_cmpgt_epi8(_mm256_min_epu8(_mm256_subs_epu8(p, c_hi), one), zero);
        let dark = _mm256_cmpgt_epi8(_mm256_min_epu8(_mm256_subs_epu8(c_lo, p), one), zero);

        bright_run = _mm256_and_si256(_mm256_add_epi8(bright_run, one), bright);
        dark_run = _mm256_and_si256(_mm256_add_epi8(dark_run, one), dark);
        max_run = _mm256_max_epu8(max_run, _mm256_max_epu8(bright_run, dark_run));
    }

    _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_max_epu8(max_run, arc), max_run)) as u32
}

/// NEON: process 16 center pixels at a time for cardinal early-reject.
/// Mirrors `detect_row_range_avx2` with half-width vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn detect_row_range_neon(
    slice: &[u8],
    row_base: usize,
    x_start: usize,
    x_end: usize,
    y: usize,
    level: usize,
    circle_off: &[isize; 16],
    cardinal_off: &[isize; 4],
    thresh: i16,
    arc_length: usize,
    min_cardinals: u8,
    features: &mut Vec<Feature>,
) {
    use std::arch::aarch64::*;

    let ptr = slice.as_ptr();
    let thresh_v = vdupq_n_u8(thresh as u8);
    let one = vdupq_n_u8(1);
    let min_card_v = vdupq_n_u8(min_cardinals);

    let mut x = x_start;

    while x + 16 <= x_end {
        let base = row_base + x;
        let base_ptr = ptr.add(base);

        let center = vld1q_u8(base_ptr);
        let c_hi = vqaddq_u8(center, thresh_v);
        let c_lo = vqsubq_u8(center, thresh_v);

        let p0 = vld1q_u8(base_ptr.offset(cardinal_off[0]));
        let p4 = vld1q_u8(base_ptr.offset(cardinal_off[1]));
        let p8 = vld1q_u8(base_ptr.offset(cardinal_off[2]));
        let p12 = vld1q_u8(base_ptr.offset(cardinal_off[3]));

        // Bright: vqsubq_u8(p, c_hi) > 0 means p > center + thresh.
        let b0 = vminq_u8(vqsubq_u8(p0, c_hi), one);
        let b4 = vminq_u8(vqsubq_u8(p4, c_hi), one);
        let b8 = vminq_u8(vqsubq_u8(p8, c_hi), one);
        let b12 = vminq_u8(vqsubq_u8(p12, c_hi), one);
        let bright_count = vaddq_u8(vaddq_u8(b0, b4), vaddq_u8(b8, b12));

        // Dark: vqsubq_u8(c_lo, p) > 0 means p < center - thresh.
        let d0 = vminq_u8(vqsubq_u8(c_lo, p0), one);
        let d4 = vminq_u8(vqsubq_u8(c_lo, p4), one);
        let d8 = vminq_u8(vqsubq_u8(c_lo, p8), one);
        let d12 = vminq_u8(vqsubq_u8(c_lo, p12), one);
        let dark_count = vaddq_u8(vaddq_u8(d0, d4), vaddq_u8(d8, d12));

        let bright_ok = vcgeq_u8(bright_count, min_card_v); // 0xFF if pass, else 0x00
        let dark_ok = vcgeq_u8(dark_count, min_card_v);
        let pass = vorrq_u8(bright_ok, dark_ok);

        // Early-skip via horizontal max.
        if vmaxvq_u8(pass) == 0 {
            x += 16;
            continue;
        }

        // Pack 16 lanes of 0xFF/0x00 into 64 bits, one nibble per lane.
        let nibble = vshrn_n_u16::<4>(vreinterpretq_u16_u8(pass));
        let mut m: u64 = vget_lane_u64::<0>(vreinterpret_u64_u8(nibble));

        // Full SIMD FAST-arc gate for cardinal survivors. Scoring,
        // descriptor computation, and Vec compaction stay scalar.
        m &= full_ring_pass_mask_neon(ptr, base, circle_off, c_hi, c_lo, arc_length);

        while m != 0 {
            let lane = (m.trailing_zeros() as usize) / 4;
            m &= !(0xFu64 << (lane * 4));
            fast_full_test(
                slice,
                row_base + x + lane,
                x + lane,
                y,
                level,
                circle_off,
                thresh,
                arc_length,
                features,
            );
        }

        x += 16;
    }

    detect_row_range_scalar(
        slice,
        row_base,
        x,
        x_end,
        y,
        level,
        circle_off,
        cardinal_off,
        thresh,
        arc_length,
        min_cardinals,
        features,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn full_ring_pass_mask_neon(
    ptr: *const u8,
    base: usize,
    circle_off: &[isize; 16],
    c_hi: std::arch::aarch64::uint8x16_t,
    c_lo: std::arch::aarch64::uint8x16_t,
    arc_length: usize,
) -> u64 {
    use std::arch::aarch64::*;

    let base_ptr = ptr.add(base);
    let one = vdupq_n_u8(1);
    let arc = vdupq_n_u8(arc_length as u8);
    let mut bright_run = vdupq_n_u8(0);
    let mut dark_run = vdupq_n_u8(0);
    let mut max_run = vdupq_n_u8(0);

    for k in 0..(16 + arc_length - 1) {
        let p = vld1q_u8(base_ptr.offset(circle_off[k & 15]));
        let bright = vcgtq_u8(p, c_hi);
        let dark = vcgtq_u8(c_lo, p);

        bright_run = vandq_u8(vaddq_u8(bright_run, one), bright);
        dark_run = vandq_u8(vaddq_u8(dark_run, one), dark);
        max_run = vmaxq_u8(max_run, vmaxq_u8(bright_run, dark_run));
    }

    let pass = vcgeq_u8(max_run, arc);
    let nibble = vshrn_n_u16::<4>(vreinterpretq_u16_u8(pass));
    vget_lane_u64::<0>(vreinterpret_u64_u8(nibble))
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

#[inline]
fn has_contiguous_arc(mask: u16, arc_length: usize) -> bool {
    let mut acc = (mask as u32) | ((mask as u32) << 16);
    for _ in 1..arc_length {
        acc &= acc >> 1;
    }
    acc != 0
}

/// Find the longest contiguous arc in a circular 16-bit mask and
/// compute its score. Used only for confirmed corners (rare path).
#[inline]
fn bitmask_best_arc_score(center: i16, circle: &[i16; 16], thresh: i16, mask: u16) -> f32 {
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
        let near_center = features
            .iter()
            .any(|f| (f.x - 10.0).abs() <= 4.0 && (f.y - 10.0).abs() <= 4.0);
        assert!(near_center, "expected a feature near (10, 10)");
        assert!(features[0].score > 0.0);
    }

    #[test]
    fn test_avx2_wide_bright_corner_with_large_contrast() {
        let img = make_fast_corner_image(80, 10, 255);
        let det = FastDetector::new(20, 9);
        let features = det.detect(&img);
        let near_center = features
            .iter()
            .any(|f| (f.x - 40.0).abs() <= 4.0 && (f.y - 40.0).abs() <= 4.0);
        assert!(
            near_center,
            "wide image should detect high-contrast bright corner"
        );
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
            assert_eq!(
                compute_min_rotation(v),
                d_ref,
                "RI-LBP should be invariant to rotation"
            );
        }
    }

    fn naive_has_contiguous_arc(mask: u16, arc_length: usize) -> bool {
        for start in 0..16 {
            let mut ok = true;
            for offset in 0..arc_length {
                let bit = (start + offset) & 15;
                if mask & (1 << bit) == 0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                return true;
            }
        }
        false
    }

    #[test]
    fn test_contiguous_arc_helper_exhaustive() {
        for arc_length in 9..=12 {
            for mask in 0u32..=u16::MAX as u32 {
                let mask = mask as u16;
                assert_eq!(
                    has_contiguous_arc(mask, arc_length),
                    naive_has_contiguous_arc(mask, arc_length),
                    "mask={mask:#018b} arc_length={arc_length}"
                );
            }
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

        assert!(
            !det_low.detect(&img).is_empty(),
            "low threshold should detect"
        );
        assert!(
            det_high.detect(&img).is_empty(),
            "high threshold should reject"
        );
    }

    #[test]
    fn test_arc_length_sensitivity() {
        let mut img = Image::from_vec(20, 20, vec![100u8; 400]);
        let cx = 10usize;
        let cy = 10usize;
        for i in 0..10 {
            let (dx, dy) = CIRCLE_OFFSETS[i];
            img.set(
                (cx as isize + dx) as usize,
                (cy as isize + dy) as usize,
                200,
            );
        }

        let det9 = FastDetector::new(20, 9);
        let det12 = FastDetector::new(20, 12);

        let has_center = |features: &[Feature]| {
            features
                .iter()
                .any(|f| f.x as usize == cx && f.y as usize == cy)
        };

        let f9 = det9.detect(&img);
        let f12 = det12.detect(&img);
        assert!(has_center(&f9), "FAST-9 should detect corner at center");
        assert!(
            !has_center(&f12),
            "FAST-12 should not detect corner at center"
        );
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
            f_high[0].score,
            f_low[0].score,
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

        assert_eq!(
            full.len(),
            skip.len(),
            "all-empty grid should match full detect"
        );
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

        assert!(
            full.iter()
                .any(|f| f.x >= 16.0 && f.x < 32.0 && f.y >= 16.0 && f.y < 32.0),
            "full detect should find corner in center cell"
        );
        assert!(
            !skip
                .iter()
                .any(|f| f.x >= 16.0 && f.x < 32.0 && f.y >= 16.0 && f.y < 32.0),
            "grid-skip should not find corner in occupied center cell"
        );
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
