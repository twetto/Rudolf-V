// nms.rs — Grid-based non-maximum suppression for detected features.
//
// Mirrors vilib's `detector_base_gpu_cuda_tools.cu` NMS strategy:
// Divide the image into a grid of cells. Within each cell, keep only
// the feature with the highest score. This ensures spatial distribution
// of features across the image — critical for VIO where you want
// features spread across the field of view, not clustered on one texture.
//
// This is sometimes called "occupancy-based NMS" or "bucketed NMS."

use crate::fast::Feature;

/// Grid-based non-maximum suppression.
pub struct OccupancyNms {
    /// Cell size in pixels. Each cell keeps at most one feature.
    /// Typical values: 16–32 for 640×480 images.
    pub cell_size: usize,
}

impl OccupancyNms {
    /// Create a new NMS grid with the given cell size.
    ///
    /// # Panics
    /// Panics if `cell_size == 0`.
    pub fn new(cell_size: usize) -> Self {
        assert!(cell_size > 0, "cell_size must be > 0");
        OccupancyNms { cell_size }
    }

    /// Suppress non-maximum features on a grid.
    ///
    /// For each grid cell, retains only the feature with the highest score.
    /// Features are not modified — surviving features are cloned into the
    /// output vector.
    ///
    /// # Arguments
    /// * `features` — Input features (e.g., from `FastDetector::detect`).
    /// * `img_w` — Image width (used to determine grid dimensions).
    /// * `img_h` — Image height.
    ///
    /// # Returns
    /// A new `Vec<Feature>` containing at most one feature per grid cell,
    /// each being the highest-scoring feature that fell in that cell.
    pub fn suppress(&self, features: &[Feature], img_w: usize, img_h: usize) -> Vec<Feature> {
        if features.is_empty() {
            return Vec::new();
        }

        // Grid dimensions: ceil(img_dim / cell_size).
        let grid_cols = (img_w + self.cell_size - 1) / self.cell_size;
        let grid_rows = (img_h + self.cell_size - 1) / self.cell_size;

        // Each cell stores the index of the best feature seen so far (or None).
        // This is idiomatic Rust: Option<usize> is the same size as usize on
        // most platforms thanks to niche optimization, so this grid is cheap.
        let mut grid: Vec<Option<usize>> = vec![None; grid_rows * grid_cols];

        for (i, feat) in features.iter().enumerate() {
            let col = (feat.x as usize) / self.cell_size;
            let row = (feat.y as usize) / self.cell_size;

            // Clamp to grid bounds (defensive — shouldn't happen if features
            // are within image bounds, but better safe than UB).
            let col = col.min(grid_cols - 1);
            let row = row.min(grid_rows - 1);
            let cell = row * grid_cols + col;

            match grid[cell] {
                None => {
                    grid[cell] = Some(i);
                }
                Some(prev_i) => {
                    if feat.score > features[prev_i].score {
                        grid[cell] = Some(i);
                    }
                }
            }
        }

        // Collect survivors.
        grid.iter()
            .filter_map(|&cell| cell.map(|i| features[i].clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(x: f32, y: f32, score: f32) -> Feature {
        Feature {
            x,
            y,
            score,
            level: 0,
            id: 0,
        }
    }

    #[test]
    fn test_empty_input() {
        let nms = OccupancyNms::new(32);
        let result = nms.suppress(&[], 640, 480);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_feature_survives() {
        let nms = OccupancyNms::new(32);
        let features = vec![make_feature(100.0, 200.0, 50.0)];
        let result = nms.suppress(&features, 640, 480);
        assert_eq!(result.len(), 1);
        assert!((result[0].score - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_same_cell_keeps_best() {
        let nms = OccupancyNms::new(32);
        // Three features in the same 32×32 cell.
        let features = vec![
            make_feature(10.0, 10.0, 30.0),
            make_feature(15.0, 15.0, 80.0), // best
            make_feature(20.0, 20.0, 50.0),
        ];
        let result = nms.suppress(&features, 640, 480);
        assert_eq!(result.len(), 1);
        assert!((result[0].score - 80.0).abs() < 1e-6);
        assert!((result[0].x - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_different_cells_all_survive() {
        let nms = OccupancyNms::new(32);
        // Four features, each in a different cell.
        let features = vec![
            make_feature(10.0, 10.0, 50.0),   // cell (0, 0)
            make_feature(40.0, 10.0, 50.0),   // cell (1, 0)
            make_feature(10.0, 40.0, 50.0),   // cell (0, 1)
            make_feature(40.0, 40.0, 50.0),   // cell (1, 1)
        ];
        let result = nms.suppress(&features, 640, 480);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_no_two_survivors_in_same_cell() {
        let nms = OccupancyNms::new(20);
        // Scatter many features.
        let mut features = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                features.push(make_feature(
                    x as f32 * 5.0 + 2.0,
                    y as f32 * 5.0 + 2.0,
                    (x * 10 + y) as f32,
                ));
            }
        }
        let result = nms.suppress(&features, 100, 100);

        // Verify no two survivors share a cell.
        for i in 0..result.len() {
            for j in (i + 1)..result.len() {
                let ci = (result[i].x as usize / 20, result[i].y as usize / 20);
                let cj = (result[j].x as usize / 20, result[j].y as usize / 20);
                assert_ne!(
                    ci, cj,
                    "features at ({}, {}) and ({}, {}) share cell {:?}",
                    result[i].x, result[i].y, result[j].x, result[j].y, ci
                );
            }
        }
    }

    #[test]
    fn test_suppression_reduces_count() {
        let nms = OccupancyNms::new(32);
        // 50 features all crammed into a small region (within one cell).
        let features: Vec<Feature> = (0..50)
            .map(|i| make_feature(10.0 + i as f32 * 0.1, 10.0, i as f32))
            .collect();
        let result = nms.suppress(&features, 640, 480);
        assert_eq!(result.len(), 1);
        // Best score should be 49.0 (the last one).
        assert!((result[0].score - 49.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "cell_size")]
    fn test_zero_cell_size_panics() {
        OccupancyNms::new(0);
    }
}
