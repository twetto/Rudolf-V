// histeq.rs — Histogram equalization for brightness normalization.
//
// Auto-exposure on cameras like EuRoC's MT9V034 causes sudden brightness
// jumps between frames. The KLT tracker matches templates by SSD — if
// the overall brightness shifts, the error landscape changes and features
// either jump to wrong locations or get lost.
//
// Histogram equalization normalizes the brightness distribution to be
// approximately uniform, making the image intensity statistics consistent
// frame-to-frame regardless of exposure changes.
//
// Two variants:
//
// 1. GLOBAL histogram equalization: compute one histogram over the entire
//    image, build the CDF, and remap every pixel. Simple, fast, and
//    effective when auto-exposure is the main problem.
//
// 2. CLAHE (Contrast Limited Adaptive Histogram Equalization): divide the
//    image into tiles, equalize each tile independently with a clip limit,
//    then bilinearly interpolate between tile CDFs for smooth transitions.
//
// OPTIMIZATIONS:
// - Direct slice access (no per-pixel method call overhead)
// - equalize_histogram_into() avoids allocation by writing into caller's buffer
// - Scratch buffer for histeq output can be held in Frontend (reused per frame)
// - RAYON PARALLELISM (feature-gated): histogram via fold/reduce with per-thread
//   local histograms (no atomics), remap via parallel row iteration.
//
// GPU NOTES:
// - Global histEq: atomic histogram in shared memory, prefix sum for CDF,
//   per-pixel LUT remap. Three dispatches.
// - CLAHE: one workgroup per tile, then per-pixel interpolation pass.

use crate::image::Image;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Histogram equalization method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistEqMethod {
    /// No preprocessing.
    None,
    /// Global histogram equalization.
    Global,
    /// CLAHE with the given tile size and clip limit.
    Clahe {
        tile_size: usize,
        clip_limit: f32,
    },
}

// ============================================================
// Global histogram equalization
// ============================================================

/// Apply global histogram equalization, allocating output.
pub fn equalize_histogram(image: &Image<u8>) -> Image<u8> {
    let mut out = Image::new(image.width(), image.height());
    equalize_histogram_into(image, &mut out);
    out
}

/// Apply global histogram equalization into a pre-allocated buffer.
///
/// Avoids per-frame allocation when called repeatedly (e.g., from Frontend).
/// Uses direct slice access for both histogram and remap passes.
/// When the `parallel` feature is enabled, both passes use rayon:
/// - Histogram: fold/reduce with per-thread local histograms (no atomics).
/// - Remap: parallel row iteration with LUT lookup.
pub fn equalize_histogram_into(image: &Image<u8>, out: &mut Image<u8>) {
    let w = image.width();
    let h = image.height();
    let n = w * h;

    out.clear_resize(w, h);

    if n == 0 {
        return;
    }

    let src = image.as_slice();
    let src_stride = image.stride();

    // Step 1: Histogram.
    #[cfg(feature = "parallel")]
    let hist = {
        // Each thread builds a local [u32; 256], then we reduce by summing.
        // No atomics, no contention on the 256 bins.
        (0..h).into_par_iter()
            .fold(
                || [0u32; 256],
                |mut local_hist, y| {
                    let row = y * src_stride;
                    unsafe {
                        for x in 0..w {
                            let v = *src.get_unchecked(row + x) as usize;
                            *local_hist.get_unchecked_mut(v) += 1;
                        }
                    }
                    local_hist
                },
            )
            .reduce(
                || [0u32; 256],
                |mut a, b| {
                    for i in 0..256 { a[i] += b[i]; }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let hist = {
        let mut hist = [0u32; 256];
        for y in 0..h {
            let row = y * src_stride;
            unsafe {
                for x in 0..w {
                    let v = *src.get_unchecked(row + x) as usize;
                    *hist.get_unchecked_mut(v) += 1;
                }
            }
        }
        hist
    };

    // Step 2: CDF → LUT (256 elements, trivially fast, always sequential).
    let lut = build_lut(&hist, n);

    // Step 3: Remap.
    let dst_stride = out.stride();
    let dst = out.as_mut_slice();

    #[cfg(feature = "parallel")]
    {
        let dst_base = dst.as_mut_ptr() as usize;
        (0..h).into_par_iter().for_each(|y| {
            let src_off = y * src_stride;
            let dst_off = y * dst_stride;
            unsafe {
                let dp = dst_base as *mut u8;
                for x in 0..w {
                    let v = *src.get_unchecked(src_off + x) as usize;
                    *dp.add(dst_off + x) = *lut.get_unchecked(v);
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..h {
            let src_off = y * src_stride;
            let dst_off = y * dst_stride;
            unsafe {
                for x in 0..w {
                    let v = *src.get_unchecked(src_off + x) as usize;
                    *dst.get_unchecked_mut(dst_off + x) = *lut.get_unchecked(v);
                }
            }
        }
    }
}

/// Wrapper to send a raw pointer across threads.
/// SAFETY: The caller must guarantee non-overlapping writes.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}

/// Build a 256-entry lookup table from a histogram and total pixel count.
fn build_lut(hist: &[u32; 256], total: usize) -> [u8; 256] {
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Find the first non-zero CDF value (skip fully black bins).
    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);

    let mut lut = [0u8; 256];
    let denom = total as f32 - cdf_min as f32;
    if denom <= 0.0 {
        return lut;
    }

    for i in 0..256 {
        let val = (cdf[i] as f32 - cdf_min as f32) / denom * 255.0;
        lut[i] = val.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

// ============================================================
// CLAHE
// ============================================================

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
pub fn equalize_clahe(image: &Image<u8>, tile_size: usize, clip_limit: f32) -> Image<u8> {
    let mut out = Image::new(image.width(), image.height());
    equalize_clahe_into(image, tile_size, clip_limit, &mut out);
    out
}

/// Apply CLAHE into a pre-allocated buffer.
pub fn equalize_clahe_into(
    image: &Image<u8>,
    tile_size: usize,
    clip_limit: f32,
    out: &mut Image<u8>,
) {
    let w = image.width();
    let h = image.height();

    out.clear_resize(w, h);

    if w == 0 || h == 0 || tile_size == 0 {
        return;
    }

    let cols = (w + tile_size - 1) / tile_size;
    let rows = (h + tile_size - 1) / tile_size;

    let src = image.as_slice();
    let src_stride = image.stride();

    // Compute per-tile LUTs.
    let mut tile_luts = vec![[0u8; 256]; cols * rows];

    for ty in 0..rows {
        for tx in 0..cols {
            let x0 = tx * tile_size;
            let y0 = ty * tile_size;
            let x1 = (x0 + tile_size).min(w);
            let y1 = (y0 + tile_size).min(h);
            let tile_pixels = (x1 - x0) * (y1 - y0);

            let mut hist = [0u32; 256];
            for y in y0..y1 {
                let row = y * src_stride;
                unsafe {
                    for x in x0..x1 {
                        let v = *src.get_unchecked(row + x) as usize;
                        *hist.get_unchecked_mut(v) += 1;
                    }
                }
            }

            if clip_limit > 0.0 {
                clip_histogram(&mut hist, tile_pixels, clip_limit);
            }

            tile_luts[ty * cols + tx] = build_lut(&hist, tile_pixels);
        }
    }

    // Remap each pixel using bilinear interpolation between 4 nearest tiles.
    let dst_stride = out.stride();
    let dst = out.as_mut_slice();

    let tile_cx = |tx: usize| -> f32 { (tx as f32 + 0.5) * tile_size as f32 };
    let tile_cy = |ty: usize| -> f32 { (ty as f32 + 0.5) * tile_size as f32 };

    for y in 0..h {
        let src_off = y * src_stride;
        let dst_off = y * dst_stride;
        let py = y as f32;
        let fy = (py / tile_size as f32) - 0.5;
        let ty0 = (fy.floor() as isize).max(0) as usize;
        let ty1 = (ty0 + 1).min(rows - 1);
        let ay = if ty0 == ty1 {
            0.0
        } else {
            ((py - tile_cy(ty0)) / (tile_cy(ty1) - tile_cy(ty0))).clamp(0.0, 1.0)
        };

        for x in 0..w {
            let px = x as f32;
            let fx = (px / tile_size as f32) - 0.5;
            let tx0 = (fx.floor() as isize).max(0) as usize;
            let tx1 = (tx0 + 1).min(cols - 1);
            let ax = if tx0 == tx1 {
                0.0
            } else {
                ((px - tile_cx(tx0)) / (tile_cx(tx1) - tile_cx(tx0))).clamp(0.0, 1.0)
            };

            let v = unsafe { *src.get_unchecked(src_off + x) as usize };

            let v00 = tile_luts[ty0 * cols + tx0][v] as f32;
            let v10 = tile_luts[ty0 * cols + tx1][v] as f32;
            let v01 = tile_luts[ty1 * cols + tx0][v] as f32;
            let v11 = tile_luts[ty1 * cols + tx1][v] as f32;

            let val = v00 * (1.0 - ax) * (1.0 - ay)
                + v10 * ax * (1.0 - ay)
                + v01 * (1.0 - ax) * ay
                + v11 * ax * ay;

            unsafe {
                *dst.get_unchecked_mut(dst_off + x) = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Clip histogram bins and redistribute excess counts.
fn clip_histogram(hist: &mut [u32; 256], total_pixels: usize, clip_multiplier: f32) {
    let clip_val = ((total_pixels as f32 / 256.0) * clip_multiplier).ceil() as u32;

    let mut excess = 0u32;
    for bin in hist.iter_mut() {
        if *bin > clip_val {
            excess += *bin - clip_val;
            *bin = clip_val;
        }
    }

    let per_bin = excess / 256;
    let remainder = (excess % 256) as usize;
    for (i, bin) in hist.iter_mut().enumerate() {
        *bin += per_bin;
        if i < remainder {
            *bin += 1;
        }
    }
}

/// Convenience function: apply the specified equalization method.
pub fn apply_histeq(image: &Image<u8>, method: HistEqMethod) -> Image<u8> {
    match method {
        HistEqMethod::None => image.clone(),
        HistEqMethod::Global => equalize_histogram(image),
        HistEqMethod::Clahe { tile_size, clip_limit } => {
            equalize_clahe(image, tile_size, clip_limit)
        }
    }
}

/// Apply histogram equalization into a pre-allocated buffer.
///
/// Use this from Frontend to avoid per-frame allocation.
pub fn apply_histeq_into(image: &Image<u8>, method: HistEqMethod, out: &mut Image<u8>) {
    match method {
        HistEqMethod::None => {
            // Copy instead of clone — reuses out's allocation.
            out.clear_resize(image.width(), image.height());
            let w = image.width();
            let src_stride = image.stride();
            let dst_stride = out.stride();
            let src = image.as_slice();
            let dst = out.as_mut_slice();
            for y in 0..image.height() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(y * src_stride),
                        dst.as_mut_ptr().add(y * dst_stride),
                        w,
                    );
                }
            }
        }
        HistEqMethod::Global => equalize_histogram_into(image, out),
        HistEqMethod::Clahe { tile_size, clip_limit } => {
            equalize_clahe_into(image, tile_size, clip_limit, out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_uniform_input() {
        let mut img = Image::new(256, 1);
        for x in 0..256 {
            img.set(x, 0, x as u8);
        }
        let out = equalize_histogram(&img);
        for x in 0..256 {
            let diff = (out.get(x, 0) as i32 - x as i32).abs();
            assert!(diff <= 1, "pixel {x}: expected ~{x}, got {}", out.get(x, 0));
        }
    }

    #[test]
    fn test_global_constant_image() {
        let img = Image::from_vec(10, 10, vec![128u8; 100]);
        let out = equalize_histogram(&img);
        let v = out.get(0, 0);
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(out.get(x, y), v);
            }
        }
    }

    #[test]
    fn test_global_low_contrast() {
        let w = 110;
        let h = 1;
        let mut img = Image::new(w, h);
        for x in 0..w {
            img.set(x, 0, (100 + x % 11) as u8);
        }
        let out = equalize_histogram(&img);
        let min_val = (0..w).map(|x| out.get(x, 0)).min().unwrap();
        let max_val = (0..w).map(|x| out.get(x, 0)).max().unwrap();
        assert!(max_val - min_val > 100, "range {min_val}..{max_val} not expanded enough");
    }

    #[test]
    fn test_global_preserves_ordering() {
        let img = Image::from_vec(5, 1, vec![10, 50, 100, 150, 200]);
        let out = equalize_histogram(&img);
        for i in 1..5 {
            assert!(
                out.get(i, 0) >= out.get(i - 1, 0),
                "monotonicity violated at {i}"
            );
        }
    }

    #[test]
    fn test_global_output_range() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }
        let out = equalize_histogram(&img);
        assert_eq!(out.width(), 50);
        assert_eq!(out.height(), 50);
    }

    #[test]
    fn test_into_matches_alloc() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }

        let alloc = equalize_histogram(&img);
        let mut into = Image::new(1, 1);
        equalize_histogram_into(&img, &mut into);

        assert_eq!(alloc.width(), into.width());
        assert_eq!(alloc.height(), into.height());
        for y in 0..50 {
            for x in 0..50 {
                assert_eq!(alloc.get(x, y), into.get(x, y),
                    "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_clahe_basic() {
        let w = 64;
        let h = 64;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, ((x * 4) % 256) as u8);
            }
        }
        let out = equalize_clahe(&img, 16, 2.0);
        assert_eq!(out.width(), w);
        assert_eq!(out.height(), h);
    }

    #[test]
    fn test_clahe_vs_global_on_bimodal() {
        let w = 64;
        let h = 32;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = if x < w / 2 { 30 } else { 200 };
                let noise = ((x + y * 7) % 20) as u8;
                img.set(x, y, base + noise);
            }
        }

        let global = equalize_histogram(&img);
        let clahe = equalize_clahe(&img, 16, 2.0);

        let g_range = range(&global);
        let c_range = range(&clahe);
        assert!(g_range > 50, "global range too small: {g_range}");
        assert!(c_range > 50, "clahe range too small: {c_range}");
    }

    #[test]
    fn test_clahe_non_divisible() {
        let img = Image::from_vec(100, 75, vec![128u8; 100 * 75]);
        let out = equalize_clahe(&img, 16, 3.0);
        assert_eq!(out.width(), 100);
        assert_eq!(out.height(), 75);
    }

    #[test]
    fn test_apply_histeq_none() {
        let img = Image::from_vec(4, 4, vec![42u8; 16]);
        let out = apply_histeq(&img, HistEqMethod::None);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(x, y), 42);
            }
        }
    }

    #[test]
    fn test_apply_histeq_into_global() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }

        let alloc = apply_histeq(&img, HistEqMethod::Global);
        let mut into = Image::new(1, 1);
        apply_histeq_into(&img, HistEqMethod::Global, &mut into);

        for y in 0..50 {
            for x in 0..50 {
                assert_eq!(alloc.get(x, y), into.get(x, y));
            }
        }
    }

    fn range(img: &Image<u8>) -> u8 {
        let mut lo = 255u8;
        let mut hi = 0u8;
        for y in 0..img.height() {
            for x in 0..img.width() {
                let v = img.get(x, y);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi - lo
    }
}
