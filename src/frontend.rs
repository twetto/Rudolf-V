// frontend.rs — Visual frontend pipeline.
//
// Mirrors vilib_ros.cpp's processing loop.
//
// This is the top-level module that ties every component together
// into the frame-by-frame loop that a VIO backend would call:
//
//   1. Build pyramid from input image
//   2. If previous frame exists → track existing features via KLT
//   3. Remove lost features, update occupancy grid
//   4. Detect new features in unoccupied cells (FAST or Harris)
//   5. Add new features up to max_features, assign unique IDs
//   6. Store pyramid for next frame
//   7. Return feature list
//
// NEW RUST CONCEPTS:
// - `Option<T>` for the first-frame case (no previous pyramid).
//   Option is Rust's null-safety mechanism: Option::None replaces
//   NULL pointers, and the compiler forces you to handle both cases.
// - `&mut self` methods that mutate internal state across frames.
// - Trait objects (`Box<dyn Detector>`) for runtime detector dispatch —
//   more idiomatic than enum matching.

use crate::fast::{FastDetector, Feature};
use crate::harris::HarrisDetector;
use crate::histeq::{self, HistEqMethod};
use crate::image::Image;
use crate::klt::{KltTracker, LkMethod, TrackStatus};
use crate::nms::OccupancyNms;
use crate::occupancy::OccupancyGrid;
use crate::pyramid::Pyramid;

/// Which corner detector to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectorType {
    Fast,
    Harris,
}

/// Detector trait — unifies FAST and Harris behind a common interface.
///
/// This is more idiomatic Rust than enum dispatch: you can add new
/// detector types without modifying existing code (open/closed principle).
pub trait Detector {
    /// Detect features in the image.
    fn detect(&self, image: &Image<u8>) -> Vec<Feature>;

    /// Detect features only where mask > 0.
    fn detect_masked(&self, image: &Image<u8>, mask: &Image<u8>) -> Vec<Feature> {
        // Default implementation: detect everywhere, then filter.
        // Individual detectors can override for efficiency.
        let features = self.detect(image);
        features
            .into_iter()
            .filter(|f| {
                let x = f.x as usize;
                let y = f.y as usize;
                x < mask.width() && y < mask.height() && mask.get(x, y) > 0
            })
            .collect()
    }
}

impl Detector for FastDetector {
    fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        // Call the inherent method explicitly to avoid recursion.
        FastDetector::detect(self, image)
    }
}

impl Detector for HarrisDetector {
    fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        HarrisDetector::detect(self, image)
    }
}

/// Frontend configuration.
pub struct FrontendConfig {
    /// Which detector to use.
    pub detector: DetectorType,
    /// FAST threshold (only used if detector == Fast).
    pub fast_threshold: u8,
    /// FAST arc length (9, 10, 11, or 12).
    pub fast_arc_length: usize,
    /// Harris k parameter (only used if detector == Harris).
    pub harris_k: f32,
    /// Harris response threshold.
    pub harris_threshold: f32,
    /// Harris block size.
    pub harris_block_size: usize,
    /// Maximum number of tracked features.
    pub max_features: usize,
    /// Occupancy grid / NMS cell size in pixels.
    pub cell_size: usize,
    /// Number of Gaussian pyramid levels.
    pub pyramid_levels: usize,
    /// Gaussian pyramid sigma.
    pub pyramid_sigma: f32,
    /// KLT tracker window half-size.
    pub klt_window: usize,
    /// KLT maximum iterations per pyramid level.
    pub klt_max_iter: usize,
    /// KLT convergence threshold (pixels).
    pub klt_epsilon: f32,
    /// KLT algorithm variant.
    pub klt_method: LkMethod,
    /// Histogram equalization preprocessing.
    /// Stabilizes brightness across frames when auto-exposure is active.
    pub histeq: HistEqMethod,
}

impl Default for FrontendConfig {
    /// Reasonable defaults matching vilib's configuration.
    fn default() -> Self {
        FrontendConfig {
            detector: DetectorType::Fast,
            fast_threshold: 20,
            fast_arc_length: 9,
            harris_k: 0.04,
            harris_threshold: 1e6,
            harris_block_size: 2,
            max_features: 200,
            cell_size: 16,
            pyramid_levels: 3,
            pyramid_sigma: 1.0,
            klt_window: 7,
            klt_max_iter: 30,
            klt_epsilon: 0.01,
            klt_method: LkMethod::ForwardAdditive,
            histeq: HistEqMethod::None,
        }
    }
}

/// The visual frontend: manages the detect-track-replenish loop.
///
/// This is what vilib_ros.cpp does on the CPU side — calling into
/// GPU kernels for each step. Our CPU reference mirrors that pipeline.
pub struct Frontend {
    config: FrontendConfig,
    /// Previous frame's pyramid. `None` on the very first frame.
    /// This is a Rust `Option<T>` — the type-safe replacement for
    /// NULL pointers. The compiler forces you to handle both cases.
    prev_pyramid: Option<Pyramid>,
    /// Currently tracked features with persistent IDs.
    features: Vec<Feature>,
    /// Next feature ID to assign. Monotonically increasing.
    next_id: u64,
    /// Occupancy grid for spatial distribution.
    grid: OccupancyGrid,
    /// Image dimensions (for validation).
    img_w: usize,
    img_h: usize,
}

/// Statistics returned after processing each frame.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Number of features tracked from previous frame.
    pub tracked: usize,
    /// Number of features lost during tracking.
    pub lost: usize,
    /// Number of new features detected to replenish.
    pub new_detections: usize,
    /// Total features after this frame.
    pub total: usize,
    /// Number of occupied cells in the grid.
    pub occupied_cells: usize,
    /// Total grid cells.
    pub total_cells: usize,
}

impl Frontend {
    /// Create a new frontend for images of the given dimensions.
    pub fn new(config: FrontendConfig, img_w: usize, img_h: usize) -> Self {
        let grid = OccupancyGrid::new(img_w, img_h, config.cell_size);
        Frontend {
            config,
            prev_pyramid: None,
            features: Vec::new(),
            next_id: 1,
            grid,
            img_w,
            img_h,
        }
    }

    /// Process one frame. Returns the currently tracked feature list
    /// and frame statistics.
    ///
    /// This is the core loop — called once per camera frame by the
    /// VIO pipeline.
    pub fn process(&mut self, image: &Image<u8>) -> (&[Feature], FrameStats) {
        assert_eq!(image.width(), self.img_w);
        assert_eq!(image.height(), self.img_h);

        // Step 0: Histogram equalization (brightness normalization).
        // Applied before pyramid construction so that all levels benefit
        // from consistent intensity statistics.
        let equalized;
        let input = if self.config.histeq != HistEqMethod::None {
            equalized = histeq::apply_histeq(image, self.config.histeq);
            &equalized
        } else {
            image
        };

        // Step 1: Build pyramid.
        let curr_pyramid = Pyramid::build(input, self.config.pyramid_levels, self.config.pyramid_sigma);

        let mut stats = FrameStats {
            tracked: 0,
            lost: 0,
            new_detections: 0,
            total: 0,
            occupied_cells: 0,
            total_cells: self.grid.total_cells(),
        };

        // Step 2: Track existing features if we have a previous frame.
        if let Some(ref prev_pyr) = self.prev_pyramid {
            if !self.features.is_empty() {
                let tracker = KltTracker::with_method(
                    self.config.klt_window,
                    self.config.klt_max_iter,
                    self.config.klt_epsilon,
                    self.config.pyramid_levels,
                    self.config.klt_method,
                );

                let results = tracker.track(prev_pyr, &curr_pyramid, &self.features);

                // Keep only successfully tracked features.
                let mut tracked = Vec::new();
                for r in &results {
                    if r.status == TrackStatus::Tracked {
                        tracked.push(r.feature.clone());
                        stats.tracked += 1;
                    } else {
                        stats.lost += 1;
                    }
                }
                self.features = tracked;
            }
        }

        // Step 3: Update occupancy grid from tracked features.
        self.grid.clear();
        for f in &self.features {
            self.grid.mark(f.x, f.y);
        }

        // Step 4: Detect new features in unoccupied cells.
        let slots_available = self.config.max_features.saturating_sub(self.features.len());

        if slots_available > 0 {
            // Convert pyramid level 0 to u8 for detection.
            let level0 = &curr_pyramid.levels[0];
            let mut u8_img = Image::new(level0.width(), level0.height());
            for (x, y, v) in level0.pixels() {
                u8_img.set(x, y, v.clamp(0.0, 255.0).round() as u8);
            }

            // Generate mask from occupancy grid.
            let mask = self.grid.unoccupied_mask();

            // Detect and filter by mask.
            let new_features = match self.config.detector {
                DetectorType::Fast => {
                    let det = FastDetector::new(
                        self.config.fast_threshold,
                        self.config.fast_arc_length,
                    );
                    det.detect_masked(&u8_img, &mask)
                }
                DetectorType::Harris => {
                    let det = HarrisDetector::new(
                        self.config.harris_k,
                        self.config.harris_threshold,
                        self.config.harris_block_size,
                    );
                    det.detect_masked(&u8_img, &mask)
                }
            };

            // NMS on new detections only, then take up to slots_available.
            let nms = OccupancyNms::new(self.config.cell_size);
            let suppressed = nms.suppress(&new_features, self.img_w, self.img_h);

            // Step 5: Add new features with unique IDs.
            for f in suppressed.iter().take(slots_available) {
                let new_feat = Feature {
                    x: f.x,
                    y: f.y,
                    score: f.score,
                    level: f.level,
                    id: self.next_id,
                };
                self.next_id += 1;
                self.features.push(new_feat);
                self.grid.mark(f.x, f.y);
                stats.new_detections += 1;
            }
        }

        // Step 6: Store pyramid for next frame.
        self.prev_pyramid = Some(curr_pyramid);

        stats.total = self.features.len();
        stats.occupied_cells = self.grid.total_cells() - self.grid.count_empty();

        (&self.features, stats)
    }

    /// Get the current feature list (without processing a new frame).
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Number of frames processed so far.
    pub fn has_prev_frame(&self) -> bool {
        self.prev_pyramid.is_some()
    }

    /// Reset the frontend (clear all features, discard previous frame).
    pub fn reset(&mut self) {
        self.prev_pyramid = None;
        self.features.clear();
        self.grid.clear();
        // Don't reset next_id — IDs should be globally unique.
    }

    /// Get the current histogram equalization method.
    pub fn histeq(&self) -> HistEqMethod {
        self.config.histeq
    }

    /// Change the histogram equalization method at runtime.
    pub fn set_histeq(&mut self, method: HistEqMethod) {
        self.config.histeq = method;
    }
}

/// Convenience method on FastDetector for masked detection.
impl FastDetector {
    /// Detect features only where mask > 0.
    pub fn detect_masked(&self, image: &Image<u8>, mask: &Image<u8>) -> Vec<Feature> {
        let features = self.detect(image);
        features
            .into_iter()
            .filter(|f| {
                let x = f.x as usize;
                let y = f.y as usize;
                x < mask.width() && y < mask.height() && mask.get(x, y) > 0
            })
            .collect()
    }
}

/// Convenience method on HarrisDetector for masked detection.
impl HarrisDetector {
    /// Detect features only where mask > 0.
    pub fn detect_masked(&self, image: &Image<u8>, mask: &Image<u8>) -> Vec<Feature> {
        let features = self.detect(image);
        features
            .into_iter()
            .filter(|f| {
                let x = f.x as usize;
                let y = f.y as usize;
                x < mask.width() && y < mask.height() && mask.get(x, y) > 0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a textured scene with multiple squares — good for detection + tracking.
    fn make_scene(w: usize, h: usize, shift_x: usize, shift_y: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![25u8; w * h]);
        let rects: [(usize, usize, usize, usize, u8); 6] = [
            (30, 25, 20, 20, 200),
            (70, 20, 25, 15, 180),
            (110, 30, 18, 22, 210),
            (25, 65, 22, 25, 190),
            (75, 60, 30, 20, 170),
            (115, 70, 20, 18, 205),
        ];
        for &(rx, ry, rw, rh, val) in &rects {
            let rx = rx + shift_x;
            let ry = ry + shift_y;
            for y in ry..(ry + rh).min(h) {
                for x in rx..(rx + rw).min(w) {
                    img.set(x, y, val);
                }
            }
        }
        img
    }

    #[test]
    fn test_first_frame_detects() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (features, stats) = frontend.process(&img);
        assert!(features.len() > 0, "first frame should detect features");
        assert_eq!(stats.tracked, 0, "nothing to track on first frame");
        assert_eq!(stats.lost, 0);
        assert!(stats.new_detections > 0);
        assert_eq!(stats.total, features.len());
    }

    #[test]
    fn test_features_get_unique_ids() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (features, _) = frontend.process(&img);
        let mut ids: Vec<u64> = features.iter().map(|f| f.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), features.len(), "all IDs should be unique");
    }

    #[test]
    fn test_tracking_across_frames() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (f1, _) = frontend.process(&img1);
        let n1 = f1.len();
        assert!(n1 > 0);

        let (f2, stats2) = frontend.process(&img2);
        // Some features should survive tracking.
        assert!(stats2.tracked > 0, "should track some features: {stats2:?}");
        // Total should stay near max_features (replenished).
        assert!(f2.len() > 0);
    }

    #[test]
    fn test_ids_persist_across_frames() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (f1, _) = frontend.process(&img1);
        let ids_f1: Vec<u64> = f1.iter().map(|f| f.id).collect();

        let (f2, _) = frontend.process(&img2);
        let ids_f2: Vec<u64> = f2.iter().map(|f| f.id).collect();

        // Tracked features should retain their IDs from frame 1.
        let persisted: Vec<u64> = ids_f2.iter().filter(|id| ids_f1.contains(id)).cloned().collect();
        assert!(
            !persisted.is_empty(),
            "some feature IDs should persist across frames"
        );
    }

    #[test]
    fn test_max_features_respected() {
        let img = make_scene(160, 120, 0, 0);
        let max = 15;
        let config = FrontendConfig {
            max_features: max,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (features, _) = frontend.process(&img);
        assert!(
            features.len() <= max,
            "features {} exceeds max {max}",
            features.len()
        );
    }

    #[test]
    fn test_replenishment_after_loss() {
        let img1 = make_scene(160, 120, 0, 0);
        // Large shift — many features will be lost.
        let img2 = make_scene(160, 120, 15, 10);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats2) = frontend.process(&img2);

        // Should detect new features to replace lost ones.
        assert!(
            stats2.new_detections > 0,
            "should replenish after losses: {stats2:?}"
        );
    }

    #[test]
    fn test_three_frame_sequence() {
        let frames: Vec<Image<u8>> = (0..3)
            .map(|i| make_scene(160, 120, i * 2, i))
            .collect();

        let config = FrontendConfig {
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let mut prev_total = 0;
        for (i, frame) in frames.iter().enumerate() {
            let (features, stats) = frontend.process(frame);
            println!(
                "Frame {i}: tracked={}, lost={}, new={}, total={}",
                stats.tracked, stats.lost, stats.new_detections, stats.total
            );
            if i > 0 {
                assert!(stats.tracked > 0, "frame {i}: should track some features");
            }
            prev_total = features.len();
        }
        assert!(prev_total > 0);
    }

    #[test]
    fn test_harris_detector_type() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            detector: DetectorType::Harris,
            max_features: 30,
            harris_threshold: 1e5, // lower threshold for synthetic images
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (features, stats) = frontend.process(&img);
        assert!(features.len() > 0, "Harris should detect features");
        assert!(stats.new_detections > 0);
    }

    #[test]
    fn test_reset() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig::default();
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img);
        assert!(!frontend.features().is_empty());

        frontend.reset();
        assert!(frontend.features().is_empty());
        assert!(!frontend.has_prev_frame());
    }

    #[test]
    fn test_inverse_compositional_method() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            klt_method: LkMethod::InverseCompositional,
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats) = frontend.process(&img2);
        assert!(stats.tracked > 0, "IC method should track features");
    }

    #[test]
    fn test_global_histeq_tracking() {
        // With global histEq: same scene at different brightness levels
        // should still track successfully.
        let img1 = make_scene(160, 120, 0, 0);
        // Simulate brightness jump: multiply all pixels.
        let mut img2_bright = make_scene(160, 120, 2, 1);
        for y in 0..120 {
            for x in 0..160 {
                let v = img2_bright.get(x, y);
                img2_bright.set(x, y, (v as u16 * 3 / 2).min(255) as u8);
            }
        }

        let config = FrontendConfig {
            histeq: HistEqMethod::Global,
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats) = frontend.process(&img2_bright);
        assert!(stats.tracked > 0, "histEq should help track across brightness change");
    }

    #[test]
    fn test_clahe_tracking() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            histeq: HistEqMethod::Clahe { tile_size: 32, clip_limit: 2.0 },
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats) = frontend.process(&img2);
        assert!(stats.tracked > 0, "CLAHE frontend should track features");
    }
}
