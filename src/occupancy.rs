// occupancy.rs — Occupancy grid for spatial feature distribution.
//
// Mirrors vilib's detector_base_gpu_cuda_tools.cu occupancy logic.
//
// The occupancy grid divides the image into cells and tracks which cells
// already contain a tracked feature. New feature detection is restricted
// to unoccupied cells, ensuring even spatial distribution — critical for
// VIO where geometric constraints rely on features being well-spread
// across the field of view.
//
// This is conceptually similar to NMS but serves a different purpose:
// - NMS: given a set of NEW detections, keep only the strongest per cell.
// - Occupancy: given EXISTING tracked features, prevent detection in
//   cells that are already covered, then detect in empty cells.

use crate::image::Image;

/// 2D occupancy grid over the image.
///
/// Each cell is a boolean indicating whether a tracked feature already
/// occupies that region. The grid is rebuilt every frame from the
/// currently tracked feature positions.
pub struct OccupancyGrid {
    /// Flattened 2D boolean grid. `true` = occupied.
    grid: Vec<bool>,
    /// Number of grid columns.
    cols: usize,
    /// Number of grid rows.
    rows: usize,
    /// Size of each cell in pixels.
    cell_size: usize,
    /// Image width (for bounds checking).
    img_w: usize,
    /// Image height.
    img_h: usize,
}

impl OccupancyGrid {
    /// Create a new occupancy grid for an image of the given dimensions.
    pub fn new(img_w: usize, img_h: usize, cell_size: usize) -> Self {
        let cols = (img_w + cell_size - 1) / cell_size;
        let rows = (img_h + cell_size - 1) / cell_size;
        OccupancyGrid {
            grid: vec![false; cols * rows],
            cols,
            rows,
            cell_size,
            img_w,
            img_h,
        }
    }

    /// Mark the cell containing pixel (x, y) as occupied.
    pub fn mark(&mut self, x: f32, y: f32) {
        if let Some(idx) = self.cell_index(x, y) {
            self.grid[idx] = true;
        }
    }

    /// Check if the cell containing pixel (x, y) is occupied.
    pub fn is_occupied(&self, x: f32, y: f32) -> bool {
        match self.cell_index(x, y) {
            Some(idx) => self.grid[idx],
            None => true, // out-of-bounds counts as occupied (don't detect there)
        }
    }

    /// Clear the entire grid (all cells become unoccupied).
    pub fn clear(&mut self) {
        self.grid.fill(false);
    }

    /// Generate a binary mask image: 255 = unoccupied (detect here),
    /// 0 = occupied (skip). One pixel per image pixel.
    ///
    /// This mask can be passed to the detector to restrict detection
    /// to empty cells.
    pub fn unoccupied_mask(&self) -> Image<u8> {
        let mut mask = Image::new(self.img_w, self.img_h);
        for y in 0..self.img_h {
            for x in 0..self.img_w {
                let col = x / self.cell_size;
                let row = y / self.cell_size;
                let idx = row * self.cols + col;
                let val = if self.grid[idx] { 0 } else { 255 };
                mask.set(x, y, val);
            }
        }
        mask
    }

    /// Number of unoccupied cells.
    pub fn count_empty(&self) -> usize {
        self.grid.iter().filter(|&&occupied| !occupied).count()
    }

    /// Total number of cells.
    pub fn total_cells(&self) -> usize {
        self.cols * self.rows
    }

    /// Grid dimensions (cols, rows).
    pub fn dims(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    fn cell_index(&self, x: f32, y: f32) -> Option<usize> {
        if x < 0.0 || y < 0.0 || x >= self.img_w as f32 || y >= self.img_h as f32 {
            return None;
        }
        let col = x as usize / self.cell_size;
        let row = y as usize / self.cell_size;
        if col >= self.cols || row >= self.rows {
            return None;
        }
        Some(row * self.cols + col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_grid_all_empty() {
        let grid = OccupancyGrid::new(64, 48, 16);
        assert_eq!(grid.cols, 4);
        assert_eq!(grid.rows, 3);
        assert_eq!(grid.total_cells(), 12);
        assert_eq!(grid.count_empty(), 12);
    }

    #[test]
    fn test_mark_and_query() {
        let mut grid = OccupancyGrid::new(64, 64, 16);
        assert!(!grid.is_occupied(8.0, 8.0));

        grid.mark(8.0, 8.0);
        assert!(grid.is_occupied(8.0, 8.0));
        // Same cell, different pixel.
        assert!(grid.is_occupied(0.0, 0.0));
        assert!(grid.is_occupied(15.0, 15.0));
        // Different cell.
        assert!(!grid.is_occupied(16.0, 8.0));
    }

    #[test]
    fn test_clear() {
        let mut grid = OccupancyGrid::new(64, 64, 16);
        grid.mark(8.0, 8.0);
        grid.mark(32.0, 32.0);
        assert_eq!(grid.count_empty(), 14); // 16 - 2

        grid.clear();
        assert_eq!(grid.count_empty(), 16);
    }

    #[test]
    fn test_out_of_bounds() {
        let grid = OccupancyGrid::new(64, 64, 16);
        assert!(grid.is_occupied(-1.0, 10.0));
        assert!(grid.is_occupied(10.0, -1.0));
        assert!(grid.is_occupied(64.0, 10.0));
        assert!(grid.is_occupied(10.0, 64.0));
    }

    #[test]
    fn test_unoccupied_mask() {
        let mut grid = OccupancyGrid::new(32, 32, 16);
        // Mark top-left cell.
        grid.mark(8.0, 8.0);

        let mask = grid.unoccupied_mask();
        assert_eq!(mask.width(), 32);
        assert_eq!(mask.height(), 32);

        // Pixel in occupied cell → 0.
        assert_eq!(mask.get(5, 5), 0);
        // Pixel in unoccupied cell → 255.
        assert_eq!(mask.get(20, 5), 255);
        assert_eq!(mask.get(5, 20), 255);
    }

    #[test]
    fn test_non_divisible_image() {
        // Image size not a multiple of cell size.
        let grid = OccupancyGrid::new(100, 75, 16);
        // 100/16 = 6.25 → 7 cols, 75/16 = 4.69 → 5 rows
        assert_eq!(grid.cols, 7);
        assert_eq!(grid.rows, 5);

        // Edge pixel should map to valid cell.
        assert!(!grid.is_occupied(99.0, 74.0));
    }

    #[test]
    fn test_count_empty() {
        let mut grid = OccupancyGrid::new(48, 48, 16);
        // 3×3 = 9 cells.
        assert_eq!(grid.count_empty(), 9);

        grid.mark(8.0, 8.0);
        grid.mark(40.0, 40.0);
        assert_eq!(grid.count_empty(), 7);

        // Marking same cell again doesn't change count.
        grid.mark(0.0, 0.0);
        assert_eq!(grid.count_empty(), 7);
    }
}
