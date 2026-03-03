// pyramid.rs — Gaussian image pyramid (fused blur + downsample).
//
// Mirrors vilib's pyramid_gpu.cu: Gaussian blur then 2× downsample at each
// level. The pyramid is the backbone of multi-scale detection and tracking —
// FAST runs on each level, and KLT does coarse-to-fine refinement.
//
// OPTIMIZATION: Fused blur + downsample with hardcoded [1,4,6,4,1]/16 kernel.
//
// The standard pyramid kernel [1,4,6,4,1]/16 is the same one OpenCV uses in
// pyrDown(). It's a binomial approximation to a Gaussian with sigma ≈ 1.0.
//
// Previous approach:
//   1. convolve_separable_into() → full-size blurred image
//   2. downsample_2x_into() → read blurred, write every other pixel
//   Cost: 2 × (w × h × kernel_len) multiply-adds + w/2 × h/2 copy
//
// Fused approach:
//   1. Horizontal pass: blur all rows with 5-tap kernel → intermediate
//   2. Vertical pass + downsample: accumulate 5 intermediate rows,
//      write only surviving output pixels (every 2nd row × 2nd column)
//   Cost: w × h × 5 + w/2 × h/2 × 5 = 6.25 × w × h multiply-adds
//   Saves 75% of vertical-pass work, eliminates downsample read pass.
//
// GPU MAPPING: The fused single-pass structure maps directly to a compute
// shader with output-size dispatch — each thread computes one output pixel
// by gathering from the source texture with hardware sampling.

use crate::image::{Image, Pixel};

// Re-export convolution utilities for other modules that may still need them.
// The pyramid itself no longer uses generic convolution.
#[allow(unused_imports)]
use crate::convolution::{gaussian_kernel_1d, ConvolveScratch};

/// A Gaussian image pyramid.
///
/// `levels[0]` is the original resolution (converted to f32).
/// `levels[n]` is approximately `(width / 2^n, height / 2^n)`.
///
/// All levels are stored as `Image<f32>` because:
/// - Blur accumulation needs f32 precision.
/// - KLT operates in f32 for sub-pixel accuracy.
/// - Storing f32 avoids repeated u8↔f32 conversions at each level.
pub struct Pyramid {
    /// Pyramid levels, from finest (index 0) to coarsest.
    pub levels: Vec<Image<f32>>,
}

/// Pre-allocated scratch buffers for pyramid construction.
///
/// Contains the intermediate buffer for the horizontal blur pass.
/// The fused approach eliminates the need for separate convolution and
/// downsample scratch buffers — just one flat Vec<f32> for the h-pass.
///
/// Flamegraph showed ~9% of runtime in `asm_exc_page_fault` from repeated
/// alloc/dealloc of these buffers. With scratch reuse, pages stay mapped.
pub struct PyramidScratch {
    /// Horizontal-pass intermediate buffer.
    /// Sized for the largest pyramid level (level 0 dimensions).
    h_buf: Vec<f32>,
}

impl PyramidScratch {
    /// Create scratch buffers for the given image dimensions.
    ///
    /// The `sigma` parameter is accepted for API compatibility but ignored —
    /// the fused pyrdown always uses the hardcoded [1,4,6,4,1]/16 kernel
    /// (equivalent to sigma ≈ 1.0, matching OpenCV's pyrDown).
    pub fn new(width: usize, height: usize, _sigma: f32) -> Self {
        PyramidScratch {
            h_buf: vec![0.0f32; width * height],
        }
    }
}

impl Pyramid {
    /// Build a Gaussian pyramid from an input image.
    ///
    /// Allocates fresh buffers each call. For repeated use (e.g., per-frame
    /// in a pipeline), use `build_reuse` with a `PyramidScratch` instead.
    pub fn build<T: Pixel>(src: &Image<T>, num_levels: usize, _sigma: f32) -> Self {
        assert!(num_levels >= 1, "pyramid must have at least 1 level");

        let mut levels = Vec::with_capacity(num_levels);

        // Level 0: convert source to f32 (no blur on the original).
        let level0 = to_f32_image(src);
        levels.push(level0);

        // Scratch for horizontal pass — sized for level 0 (largest).
        let mut h_buf = vec![0.0f32; src.width() * src.height()];

        // Each subsequent level: fused blur + 2× downsample.
        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let mut down = Image::new(prev.width() / 2, prev.height() / 2);
            pyrdown_fused(prev, &mut down, &mut h_buf);
            levels.push(down);
        }

        Pyramid { levels }
    }

    /// Build a Gaussian pyramid, reusing pre-allocated buffers.
    ///
    /// On the first call, `self.levels` may be empty — they'll be allocated.
    /// On subsequent calls, existing level buffers are reused via `clear_resize`,
    /// avoiding page faults from repeated alloc/dealloc.
    pub fn build_reuse<T: Pixel>(
        &mut self,
        src: &Image<T>,
        num_levels: usize,
        scratch: &mut PyramidScratch,
    ) {
        assert!(num_levels >= 1);

        // Ensure we have enough level slots.
        while self.levels.len() < num_levels {
            self.levels.push(Image::new(1, 1));
        }
        self.levels.truncate(num_levels);

        // Level 0: convert source to f32, reusing buffer.
        to_f32_image_into(src, &mut self.levels[0]);

        // Levels 1+: fused blur + downsample from previous level.
        for i in 1..num_levels {
            let (prev_levels, curr_levels) = self.levels.split_at_mut(i);
            let prev = &prev_levels[i - 1];
            pyrdown_fused(prev, &mut curr_levels[0], &mut scratch.h_buf);
        }
    }

    /// Number of pyramid levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get a reference to a specific level.
    pub fn level(&self, level: usize) -> &Image<f32> {
        &self.levels[level]
    }
}

// =============================================================================
// Fused blur + downsample — the heart of the optimization
// =============================================================================

/// Hardcoded kernel weights: [1, 4, 6, 4, 1] / 16.
///
/// This is the standard binomial pyramid kernel (OpenCV pyrDown).
/// Effective Gaussian sigma ≈ 1.0. Using f32 constants avoids
/// integer→float conversion in the inner loop.
const K0: f32 = 1.0 / 16.0;  // 0.0625
const K1: f32 = 4.0 / 16.0;  // 0.25
const K2: f32 = 6.0 / 16.0;  // 0.375

/// Fused Gaussian blur + 2× downsample for one pyramid level.
///
/// Two-pass separable convolution with [1,4,6,4,1]/16:
///   Phase 1: Horizontal blur of all src rows → h_buf
///   Phase 2: Vertical blur + 2× downsample → dst
///
/// The vertical pass only computes output rows (h/2) and reads only
/// the columns that survive the 2× x-downsample, cutting 75% of
/// vertical-pass work vs. a full vertical blur followed by downsample.
///
/// Border handling: clamp (replicate edge pixel), matching vilib and
/// GPU texture clamp-to-edge addressing.
fn pyrdown_fused(
    src: &Image<f32>,
    dst: &mut Image<f32>,
    h_buf: &mut Vec<f32>,
) {
    let sw = src.width();
    let sh = src.height();
    let dw = sw / 2;
    let dh = sh / 2;
    dst.clear_resize(dw, dh);

    // Tiny images: nothing to do.
    if dw == 0 || dh == 0 {
        return;
    }

    // Ensure h_buf is large enough.
    let h_needed = sw * sh;
    if h_buf.len() < h_needed {
        h_buf.resize(h_needed, 0.0);
    }

    let src_slice = src.as_slice();
    let src_stride = src.stride();

    // ── Phase 1: Horizontal pass ──────────────────────────────────────────
    // Blur each row of src with [K0, K1, K2, K1, K0].
    // Output: h_buf[y * sw + x] for all (x, y).
    //
    // Interior loop uses unchecked access; border pixels (x < 2, x >= sw-2)
    // use clamped access. For typical 640+ wide images, the border is < 1%
    // of pixels — the interior loop dominates.

    for y in 0..sh {
        let row_off = y * src_stride;
        let hrow_off = y * sw;

        // Left border: x = 0..min(2, sw)
        for x in 0..2.min(sw) {
            let sx = |dx: isize| -> f32 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src_slice[row_off + idx]
            };
            h_buf[hrow_off + x] = K0 * sx(-2) + K1 * sx(-1) + K2 * sx(0)
                                 + K1 * sx(1)  + K0 * sx(2);
        }

        // Interior: x in [2, sw-2) — no clamping needed.
        if sw > 4 {
            let src_row = &src_slice[row_off..row_off + sw];
            let h_row = &mut h_buf[hrow_off..hrow_off + sw];
            unsafe {
                for x in 2..sw - 2 {
                    *h_row.get_unchecked_mut(x) =
                          K0 * *src_row.get_unchecked(x - 2)
                        + K1 * *src_row.get_unchecked(x - 1)
                        + K2 * *src_row.get_unchecked(x)
                        + K1 * *src_row.get_unchecked(x + 1)
                        + K0 * *src_row.get_unchecked(x + 2);
                }
            }
        }

        // Right border: x = max(2, sw-2)..sw
        let right_start = if sw > 2 { sw - 2 } else { 2.min(sw) };
        for x in right_start..sw {
            let sx = |dx: isize| -> f32 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src_slice[row_off + idx]
            };
            h_buf[hrow_off + x] = K0 * sx(-2) + K1 * sx(-1) + K2 * sx(0)
                                 + K1 * sx(1)  + K0 * sx(2);
        }
    }

    // ── Phase 2: Vertical pass + downsample ──────────────────────────────
    // For each output pixel (ox, oy):
    //   source center = (ox*2, oy*2) in h_buf
    //   accumulate 5 rows: [oy*2-2, oy*2-1, oy*2, oy*2+1, oy*2+2]
    //   with weights [K0, K1, K2, K1, K0]
    //
    // Row pointers are computed once per output row (not per pixel).
    // The inner loop is a simple 5-way accumulation with stride-1 reads
    // from h_buf — very cache-friendly since consecutive ox values
    // read adjacent memory locations (at stride 2 in h_buf).

    let dst_stride = dst.stride();
    let dst_slice = dst.as_mut_slice();

    for oy in 0..dh {
        let sy = oy * 2;

        // Clamped row offsets into h_buf.
        let clamp_row = |dy: isize| -> usize {
            (sy as isize + dy).clamp(0, (sh - 1) as isize) as usize * sw
        };
        let r0 = clamp_row(-2);
        let r1 = clamp_row(-1);
        let r2 = sy * sw;
        let r3 = clamp_row(1);
        let r4 = clamp_row(2);

        let out_off = oy * dst_stride;

        unsafe {
            for ox in 0..dw {
                let sx = ox * 2;
                let v = K0 * *h_buf.get_unchecked(r0 + sx)
                      + K1 * *h_buf.get_unchecked(r1 + sx)
                      + K2 * *h_buf.get_unchecked(r2 + sx)
                      + K1 * *h_buf.get_unchecked(r3 + sx)
                      + K0 * *h_buf.get_unchecked(r4 + sx);
                *dst_slice.get_unchecked_mut(out_off + ox) = v;
            }
        }
    }
}

// =============================================================================
// Reference (non-fused) downsample — kept for standalone use if needed
// =============================================================================

/// Downsample an image by 2× in both dimensions (no blur).
///
/// Takes every other pixel: `dst(x, y) = src(2*x, 2*y)`.
/// Output dimensions: `(width / 2, height / 2)` using integer division
/// (odd dimensions drop the last row/column).
#[allow(dead_code)]
fn downsample_2x(src: &Image<f32>) -> Image<f32> {
    let new_w = src.width() / 2;
    let new_h = src.height() / 2;
    let mut dst = Image::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            unsafe { dst.set_unchecked(x, y, src.get_unchecked(x * 2, y * 2)); }
        }
    }
    dst
}

/// Downsample into a pre-allocated buffer (no blur).
#[allow(dead_code)]
fn downsample_2x_into(src: &Image<f32>, dst: &mut Image<f32>) {
    let new_w = src.width() / 2;
    let new_h = src.height() / 2;
    dst.clear_resize(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            unsafe { dst.set_unchecked(x, y, src.get_unchecked(x * 2, y * 2)); }
        }
    }
}

// =============================================================================
// u8 <-> f32 conversion helpers
// =============================================================================

/// Convert any Pixel image to f32, preserving raw values.
fn to_f32_image<T: Pixel>(src: &Image<T>) -> Image<f32> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, src.get(x, y).to_f32());
        }
    }
    dst
}

/// Convert into a pre-allocated f32 buffer.
fn to_f32_image_into<T: Pixel>(src: &Image<T>, dst: &mut Image<f32>) {
    dst.clear_resize(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            unsafe { dst.set_unchecked(x, y, src.get(x, y).to_f32()); }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_dimensions() {
        let img = Image::<f32>::new(100, 80);
        let down = downsample_2x(&img);
        assert_eq!(down.width(), 50);
        assert_eq!(down.height(), 40);
    }

    #[test]
    fn test_downsample_odd_dimensions() {
        let img = Image::<f32>::new(7, 5);
        let down = downsample_2x(&img);
        assert_eq!(down.width(), 3);
        assert_eq!(down.height(), 2);
    }

    #[test]
    fn test_downsample_preserves_values() {
        let mut img = Image::<f32>::new(4, 4);
        img.set(0, 0, 1.0);
        img.set(2, 0, 2.0);
        img.set(0, 2, 3.0);
        img.set(2, 2, 4.0);

        let down = downsample_2x(&img);
        assert_eq!(down.width(), 2);
        assert_eq!(down.height(), 2);
        assert!((down.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((down.get(1, 0) - 2.0).abs() < 1e-6);
        assert!((down.get(0, 1) - 3.0).abs() < 1e-6);
        assert!((down.get(1, 1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_level_dimensions() {
        let img: Image<u8> = Image::new(640, 480);
        let pyr = Pyramid::build(&img, 5, 1.0);

        assert_eq!(pyr.num_levels(), 5);
        assert_eq!(pyr.levels[0].width(), 640);
        assert_eq!(pyr.levels[0].height(), 480);
        assert_eq!(pyr.levels[1].width(), 320);
        assert_eq!(pyr.levels[1].height(), 240);
        assert_eq!(pyr.levels[2].width(), 160);
        assert_eq!(pyr.levels[2].height(), 120);
        assert_eq!(pyr.levels[3].width(), 80);
        assert_eq!(pyr.levels[3].height(), 60);
        assert_eq!(pyr.levels[4].width(), 40);
        assert_eq!(pyr.levels[4].height(), 30);
    }

    #[test]
    fn test_pyramid_single_level() {
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let img = Image::from_vec(2, 2, data);
        let pyr = Pyramid::build(&img, 1, 1.0);

        assert_eq!(pyr.num_levels(), 1);
        assert!((pyr.levels[0].get(0, 0) - 10.0).abs() < 1e-6);
        assert!((pyr.levels[0].get(1, 1) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_constant_image() {
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]);
        let pyr = Pyramid::build(&img, 4, 1.0);

        for (lvl, level) in pyr.levels.iter().enumerate() {
            for (x, y, v) in level.pixels() {
                assert!(
                    (v - 128.0).abs() < 0.5,
                    "level {lvl} pixel ({x},{y}) = {v}, expected 128.0"
                );
            }
        }
    }

    #[test]
    fn test_pyramid_decreasing_variance() {
        let mut data = vec![0u8; 128 * 128];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if (i / 128 + i % 128) % 2 == 0 { 255 } else { 0 };
        }
        let img = Image::from_vec(128, 128, data);
        let pyr = Pyramid::build(&img, 5, 1.0);

        let variance = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        let mut prev_var = variance(&pyr.levels[0]);
        for lvl in 1..pyr.num_levels() {
            let var = variance(&pyr.levels[lvl]);
            assert!(
                var <= prev_var * 1.2 + 1e-3,
                "variance increased too much from level {} ({prev_var}) to level {lvl} ({var})",
                lvl - 1
            );
            prev_var = var;
        }
    }

    #[test]
    fn test_pyramid_f32_input() {
        let img = Image::from_vec(32, 32, vec![50.0f32; 32 * 32]);
        let pyr = Pyramid::build(&img, 3, 1.0);
        assert_eq!(pyr.num_levels(), 3);
        assert!((pyr.levels[0].get(0, 0) - 50.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn test_pyramid_zero_levels_panics() {
        let img: Image<u8> = Image::new(10, 10);
        Pyramid::build(&img, 0, 1.0);
    }

    // ===== Fused pyrdown-specific tests =====

    #[test]
    fn test_fused_constant_passthrough() {
        let src = Image::from_vec(20, 20, vec![42.0f32; 400]);
        let mut dst = Image::new(10, 10);
        let mut h_buf = vec![0.0f32; 400];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        for (x, y, v) in dst.pixels() {
            assert!(
                (v - 42.0).abs() < 1e-4,
                "fused constant: ({x},{y}) = {v}, expected 42.0"
            );
        }
    }

    #[test]
    fn test_fused_dimensions() {
        let src = Image::<f32>::new(100, 80);
        let mut dst = Image::new(1, 1);
        let mut h_buf = vec![0.0f32; 8000];
        pyrdown_fused(&src, &mut dst, &mut h_buf);
        assert_eq!(dst.width(), 50);
        assert_eq!(dst.height(), 40);
    }

    #[test]
    fn test_fused_reduces_variance() {
        let mut data = vec![0.0f32; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                data[y * 64 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
            }
        }
        let src = Image::from_vec(64, 64, data);
        let mut dst = Image::new(32, 32);
        let mut h_buf = vec![0.0f32; 64 * 64];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        let var = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        let var_src = var(&src);
        let var_dst = var(&dst);
        assert!(
            var_dst < var_src,
            "fused pyrdown should reduce variance: src={var_src}, dst={var_dst}"
        );
    }

    #[test]
    fn test_fused_symmetry() {
        let mut data = vec![0.0f32; 100 * 100];
        for y in 0..100 {
            for x in 0..100 {
                data[y * 100 + x] = x as f32 * 2.55;
            }
        }
        let src = Image::from_vec(100, 100, data);
        let mut dst = Image::new(50, 50);
        let mut h_buf = vec![0.0f32; 10000];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        for y in 2..48 {
            for x in 2..48 {
                assert!(
                    dst.get(x, y) >= dst.get(x - 1, y) - 0.1,
                    "monotonicity violated at ({x},{y}): {} < {}",
                    dst.get(x, y), dst.get(x - 1, y)
                );
            }
        }
    }

    #[test]
    fn test_build_reuse_matches_build() {
        let data: Vec<u8> = (0..100).cycle().take(640 * 480).collect();
        let img = Image::from_vec(640, 480, data);

        let pyr_alloc = Pyramid::build(&img, 4, 1.0);

        let mut pyr_reuse = Pyramid { levels: Vec::new() };
        let mut scratch = PyramidScratch::new(640, 480, 1.0);
        pyr_reuse.build_reuse(&img, 4, &mut scratch);

        for lvl in 0..4 {
            let a = &pyr_alloc.levels[lvl];
            let b = &pyr_reuse.levels[lvl];
            assert_eq!(a.width(), b.width(), "level {lvl} width mismatch");
            assert_eq!(a.height(), b.height(), "level {lvl} height mismatch");

            for (x, y, va) in a.pixels() {
                let vb = b.get(x, y);
                assert!(
                    (va - vb).abs() < 1e-4,
                    "level {lvl} ({x},{y}): build={va}, build_reuse={vb}"
                );
            }
        }
    }
}
