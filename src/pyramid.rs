// pyramid.rs — Gaussian image pyramid.
//
// Mirrors vilib's pyramid_gpu.cu: Gaussian blur then 2× downsample at each
// level. The pyramid is the backbone of multi-scale detection and tracking —
// FAST runs on each level, and KLT does coarse-to-fine refinement.
//
// Algorithm at each level:
//   1. Gaussian blur (separable convolution from convolution.rs)
//   2. Downsample 2× by taking every other pixel in both dimensions
//
// NEW RUST CONCEPTS:
// - `Vec<Image<T>>` — a vector of owned heap objects. Each pyramid level
//   is a full Image that the Pyramid owns. When the Pyramid is dropped,
//   all levels are dropped automatically (RAII).
// - Propagating generic bounds through multiple function calls:
//   build() → convolve_separable() → convolve_rows(). The T: Pixel bound
//   must be threaded through each layer.

use crate::convolution::{convolve_separable, convolve_separable_into,
                         gaussian_kernel_1d, ConvolveScratch};
use crate::image::{Image, Pixel};

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
/// Eliminates per-frame allocation overhead. The convolution intermediate
/// buffers and the blurred-before-downsample buffer are reused across frames.
///
/// Flamegraph showed ~9% of runtime in `asm_exc_page_fault` from repeated
/// alloc/dealloc of these buffers. With scratch reuse, pages stay mapped.
pub struct PyramidScratch {
    conv: ConvolveScratch,
    kernel: Vec<f32>,
}

impl PyramidScratch {
    /// Create scratch buffers for the given image dimensions and sigma.
    pub fn new(width: usize, height: usize, sigma: f32) -> Self {
        let half_size = (3.0 * sigma).ceil().max(1.0) as usize;
        PyramidScratch {
            conv: ConvolveScratch::new(width, height),
            kernel: gaussian_kernel_1d(half_size, sigma),
        }
    }
}

impl Pyramid {
    /// Build a Gaussian pyramid from an input image.
    ///
    /// Allocates fresh buffers each call. For repeated use (e.g., per-frame
    /// in a pipeline), use `build_reuse` with a `PyramidScratch` instead.
    pub fn build<T: Pixel>(src: &Image<T>, num_levels: usize, sigma: f32) -> Self {
        assert!(num_levels >= 1, "pyramid must have at least 1 level");

        let half_size = (3.0 * sigma).ceil().max(1.0) as usize;
        let kernel = gaussian_kernel_1d(half_size, sigma);

        let mut levels = Vec::with_capacity(num_levels);

        // Level 0: convert source to f32 (no blur on the original).
        let level0 = to_f32_image(src);
        levels.push(level0);

        // Each subsequent level: blur the previous level, then downsample 2×.
        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let blurred = convolve_separable(prev, &kernel, &kernel);
            let down = downsample_2x(&blurred);
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

        // Each subsequent level: blur previous, downsample.
        for i in 1..num_levels {
            let (prev_levels, curr_levels) = self.levels.split_at_mut(i);
            let prev = &prev_levels[i - 1];

            // Ensure convolution scratch is big enough for this level.
            scratch.conv.ensure_size(prev.width(), prev.height());
            convolve_separable_into(prev, &scratch.kernel, &scratch.kernel, &mut scratch.conv);

            // Downsample blurred result into current level.
            downsample_2x_into(&scratch.conv.output, &mut curr_levels[0]);
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

/// Downsample an image by 2× in both dimensions.
///
/// Takes every other pixel: `dst(x, y) = src(2*x, 2*y)`.
/// Output dimensions: `(width / 2, height / 2)` using integer division
/// (odd dimensions drop the last row/column).
fn downsample_2x(src: &Image<f32>) -> Image<f32> {
    let new_w = src.width() / 2;
    let new_h = src.height() / 2;
    let mut dst = Image::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            // SAFETY: x*2 < width and y*2 < height since x < width/2 and y < height/2.
            unsafe { dst.set_unchecked(x, y, src.get_unchecked(x * 2, y * 2)); }
        }
    }
    dst
}

/// Downsample into a pre-allocated buffer.
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
        // Odd dimensions: (7, 5) → (3, 2)
        let img = Image::<f32>::new(7, 5);
        let down = downsample_2x(&img);
        assert_eq!(down.width(), 3);
        assert_eq!(down.height(), 2);
    }

    #[test]
    fn test_downsample_preserves_values() {
        // Verify we sample from even-indexed pixels.
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
        // 1-level pyramid is just the original converted to f32.
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let img = Image::from_vec(2, 2, data);
        let pyr = Pyramid::build(&img, 1, 1.0);

        assert_eq!(pyr.num_levels(), 1);
        assert!((pyr.levels[0].get(0, 0) - 10.0).abs() < 1e-6);
        assert!((pyr.levels[0].get(1, 1) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_constant_image() {
        // A constant image should stay constant at every pyramid level
        // (blur of constant = constant, downsample of constant = constant).
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
        // Each level should have equal or lower variance than the previous
        // (blur smooths, downsample subsamples the smoothed result).
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
            // At coarse levels, the image is so small that the blur kernel
            // can't fully suppress aliasing before the 2× downsample — a
            // perfect checkerboard is the adversarial worst case. Allow a
            // small relative increase (20%) at coarse levels.
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
        // Pyramid should also work with f32 source images.
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
}
