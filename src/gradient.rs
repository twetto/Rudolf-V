// gradient.rs — Image gradient computation via Sobel operators.
//
// Mirrors vilib's gradient computation used in harris_gpu_cuda_tools.cu.
//
// Sobel kernels are separable:
//   Sobel_x = col_kernel * row_kernel^T
//     row: [-1, 0, 1]   (derivative along x)
//     col: [ 1, 2, 1]   (smoothing along y)
//
//   Sobel_y = col_kernel * row_kernel^T
//     row: [ 1, 2, 1]   (smoothing along x)
//     col: [-1, 0, 1]   (derivative along y)
//
// We reuse convolve_separable from convolution.rs, so border handling
// (clamp/replicate) is inherited automatically.

use crate::convolution::convolve_separable;
use crate::image::{Image, Pixel};

/// Sobel kernels (unnormalized — standard convention).
const SOBEL_DERIV: [f32; 3] = [-1.0, 0.0, 1.0];
const SOBEL_SMOOTH: [f32; 3] = [1.0, 2.0, 1.0];

/// Compute the horizontal gradient Ix using the Sobel operator.
///
/// Positive values indicate intensity increasing to the right.
/// Output is unnormalized (range roughly [-1020, 1020] for u8 input).
pub fn sobel_x<T: Pixel>(src: &Image<T>) -> Image<f32> {
    // Sobel_x: derivative along rows, smooth along columns.
    convolve_separable(src, &SOBEL_DERIV, &SOBEL_SMOOTH)
}

/// Compute the vertical gradient Iy using the Sobel operator.
///
/// Positive values indicate intensity increasing downward.
pub fn sobel_y<T: Pixel>(src: &Image<T>) -> Image<f32> {
    // Sobel_y: smooth along rows, derivative along columns.
    convolve_separable(src, &SOBEL_SMOOTH, &SOBEL_DERIV)
}

/// Compute both gradients at once, avoiding redundant work if the caller
/// needs both (which Harris always does).
pub fn sobel_xy<T: Pixel>(src: &Image<T>) -> (Image<f32>, Image<f32>) {
    (sobel_x(src), sobel_y(src))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_gradient() {
        // Vertical step edge: left half = 0, right half = 100.
        // Sobel_x should produce a strong positive response at the edge.
        let mut img = Image::<u8>::new(20, 10);
        for y in 0..10 {
            for x in 10..20 {
                img.set(x, y, 100);
            }
        }

        let ix = sobel_x(&img);

        // At the edge (x=10), gradient should be strongly positive.
        let edge_response = ix.get(10, 5);
        assert!(
            edge_response > 50.0,
            "expected strong positive Ix at edge, got {edge_response}"
        );

        // Far from edge, gradient should be near zero.
        let flat_response = ix.get(5, 5);
        assert!(
            flat_response.abs() < 1.0,
            "expected near-zero Ix in flat region, got {flat_response}"
        );
    }

    #[test]
    fn test_vertical_gradient() {
        // Horizontal step edge: top half = 0, bottom half = 100.
        let mut img = Image::<u8>::new(10, 20);
        for y in 10..20 {
            for x in 0..10 {
                img.set(x, y, 100);
            }
        }

        let iy = sobel_y(&img);

        let edge_response = iy.get(5, 10);
        assert!(
            edge_response > 50.0,
            "expected strong positive Iy at edge, got {edge_response}"
        );

        let flat_response = iy.get(5, 5);
        assert!(
            flat_response.abs() < 1.0,
            "expected near-zero Iy in flat region, got {flat_response}"
        );
    }

    #[test]
    fn test_constant_image_zero_gradient() {
        let img = Image::from_vec(10, 10, vec![128u8; 100]);
        let ix = sobel_x(&img);
        let iy = sobel_y(&img);

        for (x, y, v) in ix.pixels() {
            assert!(v.abs() < 1e-6, "Ix nonzero at ({x},{y}): {v}");
        }
        for (x, y, v) in iy.pixels() {
            assert!(v.abs() < 1e-6, "Iy nonzero at ({x},{y}): {v}");
        }
    }

    #[test]
    fn test_linear_horizontal_gradient() {
        // Image where value = x. Ix should be constant (positive),
        // Iy should be zero.
        let mut img = Image::<f32>::new(20, 10);
        for y in 0..10 {
            for x in 0..20 {
                img.set(x, y, x as f32);
            }
        }

        let ix = sobel_x(&img);
        let iy = sobel_y(&img);

        // Interior pixels should have constant Ix.
        // Sobel_x on a linear ramp f(x)=x gives: [-1,0,1]*x = 2 per row,
        // then [1,2,1] smooth column = 4. So Ix = 2 * (1+2+1) = 2*4... wait,
        // actually Sobel on linear ramp: conv([-1,0,1], x) at any point = 2,
        // then conv([1,2,1], constant) = 4. So result = 2*4 = 8? No...
        // Separable: row pass gives 2.0 everywhere (interior), col pass
        // multiplies by sum of [1,2,1] = 4. So Ix = 8 for interior pixels.
        for y in 2..8 {
            for x in 2..18 {
                let v = ix.get(x, y);
                assert!(
                    (v - 8.0).abs() < 1e-3,
                    "Ix at ({x},{y}) = {v}, expected ~8.0"
                );
            }
        }

        // Iy should be zero everywhere (no vertical variation).
        for y in 1..9 {
            for x in 1..19 {
                assert!(
                    iy.get(x, y).abs() < 1e-3,
                    "Iy at ({x},{y}) = {}, expected 0",
                    iy.get(x, y)
                );
            }
        }
    }
}
