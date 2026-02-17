// harris.rs — Harris corner detector.
//
// Mirrors vilib's harris_gpu_cuda_tools.cu.
//
// The Harris detector finds corners by analyzing the structure tensor
// (second moment matrix) of image gradients. Unlike FAST, which uses
// a binary arc test, Harris uses eigenvalue analysis — making it robust
// to rotation and better at detecting true geometric corners like
// chessboard intersections.
//
// Algorithm:
//   1. Compute Sobel gradients: Ix, Iy
//   2. Compute element-wise products: Ix², Iy², Ix·Iy
//   3. Gaussian-blur each product (window function for structure tensor)
//   4. At each pixel, form the 2×2 structure tensor M from blurred products
//   5. Corner response: R = det(M) - k·trace(M)²
//   6. Threshold R, apply NMS
//
// NEW RUST CONCEPTS:
// - Multiple intermediate Image<f32> objects alive simultaneously.
//   Each is owned — no lifetime issues, but you're juggling several
//   heap allocations. The borrow checker ensures you can't accidentally
//   alias them.

use crate::convolution::{convolve_separable, gaussian_kernel_1d};
use crate::fast::Feature;
use crate::gradient::sobel_xy;
use crate::image::{Image, Pixel};

/// Harris corner detector.
pub struct HarrisDetector {
    /// Harris parameter. Controls sensitivity to corner vs. edge.
    /// Typical range: 0.04–0.06. Lower values detect more corners.
    pub k: f32,
    /// Minimum corner response threshold. Only pixels with R > threshold
    /// are considered corners. Scale depends on image intensity range —
    /// for u8 input, values in the range 1e6–1e8 are common.
    pub threshold: f32,
    /// Half-size of the Gaussian window for the structure tensor.
    /// block_size=1 → 3×3 window, block_size=2 → 5×5 window.
    /// Typical: 1 or 2.
    pub block_size: usize,
}

impl HarrisDetector {
    /// Create a new Harris detector with the given parameters.
    pub fn new(k: f32, threshold: f32, block_size: usize) -> Self {
        HarrisDetector {
            k,
            threshold,
            block_size,
        }
    }

    /// Compute the Harris corner response image.
    ///
    /// Returns an Image<f32> where each pixel contains the response value R.
    /// Positive R → corner, negative R → edge, near-zero R → flat.
    ///
    /// Exposed publicly so you can visualize the response map.
    pub fn corner_response<T: Pixel>(&self, image: &Image<T>) -> Image<f32> {
        let w = image.width();
        let h = image.height();

        // Step 1: Compute Sobel gradients.
        let (ix, iy) = sobel_xy(image);

        // Step 2: Element-wise products.
        let mut ix2 = Image::<f32>::new(w, h);
        let mut iy2 = Image::<f32>::new(w, h);
        let mut ixiy = Image::<f32>::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let gx = ix.get(x, y);
                let gy = iy.get(x, y);
                ix2.set(x, y, gx * gx);
                iy2.set(x, y, gy * gy);
                ixiy.set(x, y, gx * gy);
            }
        }

        // Step 3: Gaussian blur of each product (structure tensor window).
        let sigma = self.block_size as f32 * 0.5 + 0.5; // reasonable default
        let kernel = gaussian_kernel_1d(self.block_size, sigma);

        let sxx = convolve_separable(&ix2, &kernel, &kernel);
        let syy = convolve_separable(&iy2, &kernel, &kernel);
        let sxy = convolve_separable(&ixiy, &kernel, &kernel);

        // Steps 4 & 5: Harris response at each pixel.
        //   M = [[Sxx, Sxy], [Sxy, Syy]]
        //   det(M)   = Sxx * Syy - Sxy²
        //   trace(M) = Sxx + Syy
        //   R = det(M) - k * trace(M)²
        let mut response = Image::<f32>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let a = sxx.get(x, y);
                let b = syy.get(x, y);
                let c = sxy.get(x, y);
                let det = a * b - c * c;
                let trace = a + b;
                response.set(x, y, det - self.k * trace * trace);
            }
        }

        response
    }

    /// Detect Harris corners in a grayscale image.
    ///
    /// Returns features sorted by score (descending). Features have
    /// `level = 0` and `id = 0`.
    pub fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        self.detect_at_level(image, 0)
    }

    /// Detect Harris corners, tagging with the given pyramid level.
    pub fn detect_at_level(&self, image: &Image<u8>, level: usize) -> Vec<Feature> {
        let response = self.corner_response(image);
        let w = image.width();
        let h = image.height();

        // Step 6: Threshold the response.
        let mut features = Vec::new();

        // Skip a border of block_size + 1 to avoid edge artifacts from
        // the Sobel and Gaussian convolutions.
        let border = self.block_size + 2;
        if w <= 2 * border || h <= 2 * border {
            return features;
        }

        for y in border..(h - border) {
            for x in border..(w - border) {
                let r = response.get(x, y);
                if r > self.threshold {
                    features.push(Feature {
                        x: x as f32,
                        y: y as f32,
                        score: r,
                        level,
                        id: 0,
                    });
                }
            }
        }

        // Sort by score descending — strongest corners first.
        features.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nms::OccupancyNms;

    /// Create a chessboard image. This is the canonical Harris test —
    /// FAST can't detect these junction corners, but Harris can.
    fn make_chessboard(img_size: usize, cell_size: usize, lo: u8, hi: u8) -> Image<u8> {
        let mut img = Image::new(img_size, img_size);
        for y in 0..img_size {
            for x in 0..img_size {
                let cx = x / cell_size;
                let cy = y / cell_size;
                let val = if (cx + cy) % 2 == 0 { lo } else { hi };
                img.set(x, y, val);
            }
        }
        img
    }

    #[test]
    fn test_chessboard_detects_corners() {
        let img = make_chessboard(80, 10, 20, 230);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect(&img);

        // Chessboard has multiple junction corners. Should detect many.
        assert!(
            features.len() >= 10,
            "expected many Harris corners on chessboard, got {}",
            features.len()
        );
    }

    #[test]
    fn test_chessboard_corners_near_junctions() {
        let cell = 10;
        let img = make_chessboard(80, cell, 20, 230);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect(&img);

        // Each corner should be near a cell boundary intersection.
        let tolerance = (cell as f32) / 2.0;
        for f in &features {
            let nearest_ix = (f.x / cell as f32).round() * cell as f32;
            let nearest_iy = (f.y / cell as f32).round() * cell as f32;
            let dist = ((f.x - nearest_ix).powi(2) + (f.y - nearest_iy).powi(2)).sqrt();
            assert!(
                dist <= tolerance,
                "Harris corner at ({:.0},{:.0}) is {:.1}px from nearest junction",
                f.x, f.y, dist,
            );
        }
    }

    #[test]
    fn test_flat_image_no_corners() {
        let img = Image::from_vec(40, 40, vec![128u8; 1600]);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect(&img);
        assert!(features.is_empty(), "flat image should have no Harris corners");
    }

    #[test]
    fn test_edge_not_corner() {
        // A single vertical edge: left=50, right=200.
        // Harris should produce low response along edges (negative R)
        // and only corners where the edge terminates (which it doesn't here).
        let mut img = Image::new(60, 60);
        for y in 0..60 {
            for x in 0..60 {
                img.set(x, y, if x < 30 { 50 } else { 200 });
            }
        }
        let det = HarrisDetector::new(0.04, 1e8, 2);
        let features = det.detect(&img);

        // A full-image edge shouldn't produce many corners (if any).
        assert!(
            features.len() < 5,
            "straight edge produced too many Harris corners: {}",
            features.len()
        );
    }

    #[test]
    fn test_response_image_properties() {
        let img = make_chessboard(40, 10, 20, 230);
        let det = HarrisDetector::new(0.04, 0.0, 2);
        let response = det.corner_response(&img);

        assert_eq!(response.width(), img.width());
        assert_eq!(response.height(), img.height());

        // Response should have both positive (corners) and negative (edges) values.
        let mut has_positive = false;
        let mut has_negative = false;
        for (_, _, v) in response.pixels() {
            if v > 1e4 {
                has_positive = true;
            }
            if v < -1e4 {
                has_negative = true;
            }
        }
        assert!(has_positive, "response should have positive values (corners)");
        assert!(has_negative, "response should have negative values (edges)");
    }

    #[test]
    fn test_k_sensitivity() {
        // Higher k should produce fewer corners (more selective).
        let img = make_chessboard(80, 10, 20, 230);
        let det_low_k = HarrisDetector::new(0.04, 1e6, 2);
        let det_high_k = HarrisDetector::new(0.15, 1e6, 2);

        let f_low = det_low_k.detect(&img);
        let f_high = det_high_k.detect(&img);

        assert!(
            f_high.len() <= f_low.len(),
            "higher k should detect fewer or equal corners: k=0.04→{}, k=0.15→{}",
            f_low.len(),
            f_high.len(),
        );
    }

    #[test]
    fn test_threshold_sensitivity() {
        let img = make_chessboard(80, 10, 20, 230);
        let det_low = HarrisDetector::new(0.04, 1e5, 2);
        let det_high = HarrisDetector::new(0.04, 1e8, 2);

        let f_low = det_low.detect(&img);
        let f_high = det_high.detect(&img);

        assert!(
            f_high.len() <= f_low.len(),
            "higher threshold should detect fewer corners: low→{}, high→{}",
            f_low.len(),
            f_high.len(),
        );
    }

    #[test]
    fn test_sorted_by_score() {
        let img = make_chessboard(80, 10, 20, 230);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect(&img);

        for i in 1..features.len() {
            assert!(
                features[i - 1].score >= features[i].score,
                "features not sorted by score at index {i}"
            );
        }
    }

    #[test]
    fn test_with_nms() {
        let img = make_chessboard(80, 10, 20, 230);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let raw = det.detect(&img);
        let nms = OccupancyNms::new(12);
        let suppressed = nms.suppress(&raw, img.width(), img.height());

        assert!(suppressed.len() <= raw.len());
        assert!(!suppressed.is_empty());
    }

    #[test]
    fn test_image_too_small() {
        let img = Image::from_vec(6, 6, vec![128u8; 36]);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect(&img);
        assert!(features.is_empty());
    }

    #[test]
    fn test_detect_at_level() {
        let img = make_chessboard(80, 10, 20, 230);
        let det = HarrisDetector::new(0.04, 1e6, 2);
        let features = det.detect_at_level(&img, 2);
        for f in &features {
            assert_eq!(f.level, 2);
        }
    }
}
