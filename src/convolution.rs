// convolution.rs — Separable 1D convolution for Image<T>.
//
// This mirrors vilib's GPU kernel split:
//   conv_filter_row.cu → convolve_rows()    (horizontal pass)
//   conv_filter_col.cu → convolve_cols()     (vertical pass)
//   combined           → convolve_separable()
//
// A 2D convolution with a separable kernel K = k_col * k_row^T decomposes
// into two 1D passes, reducing cost from O(k²) to O(2k) per pixel.
//
// BORDER HANDLING: Clamp (replicate edge pixels).
// When the kernel window extends beyond the image boundary, out-of-bounds
// indices are clamped to the nearest edge pixel. This is what vilib does
// and is standard for pyramid construction.
//
// NEW RUST CONCEPTS:
// - Slice parameters (`&[f32]`) — borrowed view into a contiguous array.
//   The caller can pass &Vec<f32>, &[f32; 5], or a sub-slice — all work.
// - Passing generic types through function boundaries: convolve_separable
//   calls convolve_rows and convolve_cols, both generic over T: Pixel.

use crate::image::{Image, Pixel};

/// Convolve each row of `src` with a 1D kernel (horizontal pass).
///
/// The kernel is applied centered: for a kernel of length K, the center
/// element is at index K/2. So a 5-tap kernel [-2, -1, 0, +1, +2] has
/// its center at index 2.
///
/// Optimized: the interior pixels (where the kernel doesn't touch the
/// border) use unchecked access. Border pixels use clamped access.
/// This mirrors GPU texture sampling with clamp-to-edge addressing.
pub fn convolve_rows<T: Pixel>(src: &Image<T>, kernel: &[f32]) -> Image<f32> {
    assert!(!kernel.is_empty(), "kernel must not be empty");
    assert!(kernel.len() % 2 == 1, "kernel length must be odd (got {})", kernel.len());

    let w = src.width();
    let h = src.height();
    let half = kernel.len() / 2;
    let mut dst = Image::<f32>::new(w, h);

    for y in 0..h {
        // Left border: x in [0, half)
        for x in 0..half.min(w) {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize) + (ki as isize) - (half as isize);
                let sx = sx.clamp(0, (w - 1) as isize) as usize;
                acc += src.get(sx, y).to_f32() * kv;
            }
            dst.set(x, y, acc);
        }

        // Interior: x in [half, w - half) — no bounds checks needed.
        if w > 2 * half {
            for x in half..(w - half) {
                let mut acc = 0.0f32;
                // SAFETY: x - half >= 0 and x + half < w, all within bounds.
                unsafe {
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sx = x + ki - half;
                        acc += src.get_unchecked(sx, y).to_f32() * kv;
                    }
                    dst.set_unchecked(x, y, acc);
                }
            }
        }

        // Right border: x in [w - half, w)
        let right_start = if w > half { w - half } else { half.min(w) };
        for x in right_start..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize) + (ki as isize) - (half as isize);
                let sx = sx.clamp(0, (w - 1) as isize) as usize;
                acc += src.get(sx, y).to_f32() * kv;
            }
            dst.set(x, y, acc);
        }
    }
    dst
}

/// Convolve each column of `src` with a 1D kernel (vertical pass).
///
/// Input is f32 (the output of convolve_rows). Output is also f32.
/// Optimized with interior/border split like convolve_rows.
pub fn convolve_cols(src: &Image<f32>, kernel: &[f32]) -> Image<f32> {
    assert!(!kernel.is_empty(), "kernel must not be empty");
    assert!(kernel.len() % 2 == 1, "kernel length must be odd (got {})", kernel.len());

    let w = src.width();
    let h = src.height();
    let half = kernel.len() / 2;
    let mut dst = Image::<f32>::new(w, h);

    // Top border rows: y in [0, half)
    for y in 0..half.min(h) {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize) + (ki as isize) - (half as isize);
                let sy = sy.clamp(0, (h - 1) as isize) as usize;
                acc += src.get(x, sy) * kv;
            }
            dst.set(x, y, acc);
        }
    }

    // Interior rows: y in [half, h - half) — no bounds checks needed.
    if h > 2 * half {
        for y in half..(h - half) {
            for x in 0..w {
                let mut acc = 0.0f32;
                // SAFETY: y - half >= 0 and y + half < h, all within bounds.
                unsafe {
                    for (ki, &kv) in kernel.iter().enumerate() {
                        let sy = y + ki - half;
                        acc += src.get_unchecked(x, sy) * kv;
                    }
                    dst.set_unchecked(x, y, acc);
                }
            }
        }
    }

    // Bottom border rows: y in [h - half, h)
    let bottom_start = if h > half { h - half } else { half.min(h) };
    for y in bottom_start..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize) + (ki as isize) - (half as isize);
                let sy = sy.clamp(0, (h - 1) as isize) as usize;
                acc += src.get(x, sy) * kv;
            }
            dst.set(x, y, acc);
        }
    }
    dst
}

/// Full separable 2D convolution: horizontal pass then vertical pass.
///
/// Returns an `Image<f32>` regardless of input pixel type, because the
/// intermediate accumulation is in f32. The caller can convert back to
/// the original type with `Pixel::from_f32` if needed (e.g., via
/// `convert::f32_raw_to_u8`).
///
/// For a Gaussian blur with kernel g, call:
///   `convolve_separable(&img, &g, &g)`
/// since the Gaussian is symmetric (same kernel for rows and columns).
///
/// # Panics
/// Panics if either kernel is empty or has even length.
pub fn convolve_separable<T: Pixel>(
    src: &Image<T>,
    kernel_row: &[f32],
    kernel_col: &[f32],
) -> Image<f32> {
    let intermediate = convolve_rows(src, kernel_row);
    convolve_cols(&intermediate, kernel_col)
}

/// Generate a 1D Gaussian kernel with the given half-size and sigma.
///
/// Returns a kernel of length `2 * half_size + 1`, normalized so the
/// coefficients sum to 1.0.
///
/// # Examples
/// ```
/// let k = rudolf_v::convolution::gaussian_kernel_1d(2, 1.0);
/// assert_eq!(k.len(), 5);
/// assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
pub fn gaussian_kernel_1d(half_size: usize, sigma: f32) -> Vec<f32> {
    assert!(sigma > 0.0, "sigma must be positive");
    let len = 2 * half_size + 1;
    let mut kernel = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..len {
        let x = i as f32 - half_size as f32;
        kernel.push((-x * x / two_sigma_sq).exp());
    }

    // Normalize so coefficients sum to 1 (preserves image brightness).
    let sum: f32 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_properties() {
        let k = gaussian_kernel_1d(2, 1.0);
        assert_eq!(k.len(), 5);
        // Sums to 1.
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Symmetric.
        assert!((k[0] - k[4]).abs() < 1e-6);
        assert!((k[1] - k[3]).abs() < 1e-6);
        // Center is the largest.
        assert!(k[2] > k[1]);
        assert!(k[1] > k[0]);
    }

    #[test]
    fn test_identity_kernel() {
        // A kernel [0, 0, 1, 0, 0] should reproduce the input exactly.
        let data: Vec<u8> = (0..12).collect();
        let img = Image::from_vec(4, 3, data);
        let kernel = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let out = convolve_separable(&img, &kernel, &kernel);
        for y in 0..3 {
            for x in 0..4 {
                assert!(
                    (out.get(x, y) - img.get(x, y).to_f32()).abs() < 1e-6,
                    "identity mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn test_constant_image_unchanged() {
        // Blurring a constant image should return the same constant.
        let img = Image::from_vec(5, 5, vec![100.0f32; 25]);
        let k = gaussian_kernel_1d(2, 1.0);
        let out = convolve_separable(&img, &k, &k);
        for (x, y, v) in out.pixels() {
            assert!(
                (v - 100.0).abs() < 1e-4,
                "constant image changed at ({x}, {y}): {v}"
            );
        }
    }

    #[test]
    fn test_blur_reduces_variance() {
        // A noisy image should have lower variance after blurring.
        let mut data = vec![0.0f32; 64];
        // Checkerboard: high local variance.
        for y in 0..8 {
            for x in 0..8 {
                data[y * 8 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
            }
        }
        let img = Image::from_vec(8, 8, data);
        let k = gaussian_kernel_1d(2, 1.0);
        let blurred = convolve_separable(&img, &k, &k);

        // Compute variance of original and blurred.
        let var = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        assert!(
            var(&blurred) < var(&img),
            "variance should decrease after blur"
        );
    }

    #[test]
    fn test_box_filter_3x3() {
        // A 3-tap box filter [1/3, 1/3, 1/3] applied separably should
        // give the mean of a 3×3 neighborhood.
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.0,
            0.0, 9.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let img = Image::from_vec(3, 3, data);
        let k = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let out = convolve_separable(&img, &k, &k);

        // Center pixel: mean of 3×3 = 9/9 = 1.0
        assert!((out.get(1, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_border() {
        // 1D image [10, 20, 30], kernel [0.25, 0.5, 0.25].
        // At x=0: clamp gives pixel[-1]=pixel[0]=10.
        //   result = 0.25*10 + 0.5*10 + 0.25*20 = 12.5
        let data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let img = Image::from_vec(3, 1, data);
        let k = vec![0.25, 0.5, 0.25];
        let out = convolve_rows(&img, &k);
        assert!((out.get(0, 0) - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_single_pixel() {
        let img = Image::from_vec(1, 1, vec![42.0f32]);
        let k = gaussian_kernel_1d(2, 1.0);
        let out = convolve_separable(&img, &k, &k);
        // All kernel taps clamp to the same pixel → output = 42.0 * sum(k) = 42.0
        assert!((out.get(0, 0) - 42.0).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "odd")]
    fn test_even_kernel_panics() {
        let img = Image::from_vec(4, 4, vec![0.0f32; 16]);
        let k = vec![0.5, 0.5]; // even length
        convolve_rows(&img, &k);
    }
}
