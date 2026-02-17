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
//    effective when auto-exposure is the main problem — because AE already
//    does local adaptation, global histEq just stabilizes the overall
//    distribution.
//
// 2. CLAHE (Contrast Limited Adaptive Histogram Equalization): divide the
//    image into tiles, equalize each tile independently with a clip limit,
//    then bilinearly interpolate between tile CDFs for smooth transitions.
//    Better for scenes with large dynamic range (e.g., indoor/outdoor
//    boundary) but can amplify noise and create artificial gradients when
//    combined with auto-exposure.
//
// GPU NOTES:
// - Global histEq: histogram accumulation uses atomic adds into shared
//   memory (one workgroup), prefix sum for CDF is a classic parallel
//   primitive, and the remap is a per-pixel LUT lookup. Three dispatches.
// - CLAHE: one workgroup per tile for independent histograms (embarrassingly
//   parallel), then a per-pixel interpolation pass. The tile-independent
//   structure maps beautifully to GPU compute.

use crate::image::Image;

/// Histogram equalization method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistEqMethod {
    /// No preprocessing.
    None,
    /// Global histogram equalization.
    Global,
    /// CLAHE with the given tile size and clip limit.
    Clahe {
        /// Tile size in pixels (e.g., 8 means 8×8 tiles).
        tile_size: usize,
        /// Clip limit as a multiplier on the "uniform" bin count.
        /// 2.0–4.0 is typical. Higher = less clipping = more contrast.
        clip_limit: f32,
    },
}

// ============================================================
// Global histogram equalization
// ============================================================

/// Apply global histogram equalization to a grayscale image.
///
/// Algorithm:
///   1. Compute 256-bin histogram (one pass).
///   2. Build cumulative distribution function (CDF).
///   3. Remap: output[i] = round(CDF[input[i]] * 255).
///
/// This is the textbook algorithm from Gonzalez & Woods.
pub fn equalize_histogram(image: &Image<u8>) -> Image<u8> {
    let w = image.width();
    let h = image.height();
    let n = w * h;

    if n == 0 {
        return Image::new(0, 0);
    }

    // Step 1: Histogram.
    let mut hist = [0u32; 256];
    for y in 0..h {
        for x in 0..w {
            hist[image.get(x, y) as usize] += 1;
        }
    }

    // Step 2: CDF → LUT.
    let lut = build_lut(&hist, n);

    // Step 3: Remap.
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            out.set(x, y, lut[image.get(x, y) as usize]);
        }
    }
    out
}

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
        // Degenerate: all pixels are the same value.
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
///
/// Algorithm:
///   1. Divide image into tiles of `tile_size × tile_size`.
///   2. For each tile, compute histogram, clip, redistribute, build CDF.
///   3. For each pixel, bilinearly interpolate between the CDFs of the
///      four nearest tile centers.
///
/// Reference: Zuiderveld (1994), "Contrast Limited Adaptive Histogram
/// Equalization", Graphics Gems IV.
pub fn equalize_clahe(image: &Image<u8>, tile_size: usize, clip_limit: f32) -> Image<u8> {
    let w = image.width();
    let h = image.height();

    if w == 0 || h == 0 || tile_size == 0 {
        return Image::new(w, h);
    }

    // Number of tiles (round up to cover the image).
    let cols = (w + tile_size - 1) / tile_size;
    let rows = (h + tile_size - 1) / tile_size;

    // Compute per-tile LUTs.
    let mut tile_luts = vec![[0u8; 256]; cols * rows];

    for ty in 0..rows {
        for tx in 0..cols {
            let x0 = tx * tile_size;
            let y0 = ty * tile_size;
            let x1 = (x0 + tile_size).min(w);
            let y1 = (y0 + tile_size).min(h);
            let tile_pixels = (x1 - x0) * (y1 - y0);

            // Histogram for this tile.
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[image.get(x, y) as usize] += 1;
                }
            }

            // Clip histogram and redistribute.
            if clip_limit > 0.0 {
                clip_histogram(&mut hist, tile_pixels, clip_limit);
            }

            // Build LUT from clipped histogram.
            tile_luts[ty * cols + tx] = build_lut(&hist, tile_pixels);
        }
    }

    // Remap each pixel using bilinear interpolation between 4 nearest tiles.
    let mut out = Image::new(w, h);

    // Tile centers.
    let tile_cx = |tx: usize| -> f32 { (tx as f32 + 0.5) * tile_size as f32 };
    let tile_cy = |ty: usize| -> f32 { (ty as f32 + 0.5) * tile_size as f32 };

    for y in 0..h {
        for x in 0..w {
            let px = x as f32;
            let py = y as f32;

            // Find the four surrounding tile indices.
            // The pixel lies between tile centers; find the two nearest
            // in each dimension.
            let fx = (px / tile_size as f32) - 0.5;
            let fy = (py / tile_size as f32) - 0.5;

            let tx0 = (fx.floor() as isize).max(0) as usize;
            let ty0 = (fy.floor() as isize).max(0) as usize;
            let tx1 = (tx0 + 1).min(cols - 1);
            let ty1 = (ty0 + 1).min(rows - 1);

            // Interpolation weights.
            let ax = if tx0 == tx1 {
                0.0
            } else {
                ((px - tile_cx(tx0)) / (tile_cx(tx1) - tile_cx(tx0))).clamp(0.0, 1.0)
            };
            let ay = if ty0 == ty1 {
                0.0
            } else {
                ((py - tile_cy(ty0)) / (tile_cy(ty1) - tile_cy(ty0))).clamp(0.0, 1.0)
            };

            let v = image.get(x, y) as usize;

            // Look up remapped value in each of the 4 tile LUTs.
            let v00 = tile_luts[ty0 * cols + tx0][v] as f32;
            let v10 = tile_luts[ty0 * cols + tx1][v] as f32;
            let v01 = tile_luts[ty1 * cols + tx0][v] as f32;
            let v11 = tile_luts[ty1 * cols + tx1][v] as f32;

            // Bilinear interpolation.
            let val = v00 * (1.0 - ax) * (1.0 - ay)
                + v10 * ax * (1.0 - ay)
                + v01 * (1.0 - ax) * ay
                + v11 * ax * ay;

            out.set(x, y, val.round().clamp(0.0, 255.0) as u8);
        }
    }

    out
}

/// Clip histogram bins and redistribute excess counts.
///
/// The clip limit is expressed as a multiplier on the "uniform" bin count
/// (total_pixels / 256). Bins exceeding the limit are clipped, and the
/// excess is redistributed evenly across all bins.
fn clip_histogram(hist: &mut [u32; 256], total_pixels: usize, clip_multiplier: f32) {
    let clip_val = ((total_pixels as f32 / 256.0) * clip_multiplier).ceil() as u32;

    // Clip and accumulate excess.
    let mut excess = 0u32;
    for bin in hist.iter_mut() {
        if *bin > clip_val {
            excess += *bin - clip_val;
            *bin = clip_val;
        }
    }

    // Distribute excess evenly.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_uniform_input() {
        // Already uniform distribution → output should be similar.
        let mut img = Image::new(256, 1);
        for x in 0..256 {
            img.set(x, 0, x as u8);
        }
        let out = equalize_histogram(&img);
        // Each value appears once; CDF is linear → output ≈ input.
        for x in 0..256 {
            let diff = (out.get(x, 0) as i32 - x as i32).abs();
            assert!(diff <= 1, "pixel {x}: expected ~{x}, got {}", out.get(x, 0));
        }
    }

    #[test]
    fn test_global_constant_image() {
        // All same value → output should be all same (mapped to some value).
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
        // Image with values only in [100, 110] → output should spread to [0, 255].
        let w = 110;
        let h = 1;
        let mut img = Image::new(w, h);
        for x in 0..w {
            img.set(x, 0, (100 + x % 11) as u8);
        }
        let out = equalize_histogram(&img);
        let min_val = (0..w).map(|x| out.get(x, 0)).min().unwrap();
        let max_val = (0..w).map(|x| out.get(x, 0)).max().unwrap();
        // Should expand the range significantly.
        assert!(max_val - min_val > 100, "range {min_val}..{max_val} not expanded enough");
    }

    #[test]
    fn test_global_preserves_ordering() {
        // Brighter input pixels should map to >= output of darker inputs.
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
        // Output should be in [0, 255].
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }
        let out = equalize_histogram(&img);
        for y in 0..50 {
            for x in 0..50 {
                assert!(out.get(x, y) <= 255);
            }
        }
    }

    #[test]
    fn test_clahe_basic() {
        // CLAHE on a gradient image — should not crash and should
        // produce output in valid range.
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
        for y in 0..h {
            for x in 0..w {
                assert!(out.get(x, y) <= 255);
            }
        }
    }

    #[test]
    fn test_clahe_vs_global_on_bimodal() {
        // Image with left half dark, right half bright.
        // CLAHE should provide better local contrast than global.
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

        // Both should expand the range.
        let g_range = range(&global);
        let c_range = range(&clahe);
        assert!(g_range > 50, "global range too small: {g_range}");
        assert!(c_range > 50, "clahe range too small: {c_range}");
    }

    #[test]
    fn test_clahe_non_divisible() {
        // Image size not a multiple of tile size.
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
