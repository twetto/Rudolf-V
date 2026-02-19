// gpu/frontend.rs — GPU visual frontend pipeline.
//
// Drop-in GPU replacement for frontend.rs. The `process()` method returns
// the same (&[Feature], FrameStats) pair so euroc_live.rs works unchanged
// after substituting `GpuFrontend` for `Frontend`.
//
//
// STAGE ALLOCATION TABLE
// ──────────────────────
//   Stage              CPU or GPU   Notes
//   ─────────────────  ───────────  ──────────────────────────────────────
//   HistEq             CPU          Applied before GPU upload.
//   Image upload       GPU          u8 → R32Float texture via staging buf.
//   Pyramid build      GPU          GpuPyramidPipeline (Gaussian + downsample)
//   KLT tracking       GPU          GpuKltTracker (inverse compositional)
//   FAST detection     GPU          GpuFastDetector (Bresenham circle + arcs)
//   NMS                CPU          OccupancyNms — microseconds, not worth porting
//   Occupancy grid     CPU          OccupancyGrid — byte mask, trivial
//   RANSAC             CPU          essential::estimate_essential_ransac — pure
//                                   linear algebra on O(N) tracked positions
//
// PYRAMID LIFETIME
// ─────────────────
// Unlike the CPU frontend which double-buffers two Pyramid structs to avoid
// re-allocation, GpuPyramid is cheap to recreate: it's just a set of
// wgpu::Texture handles. The GPU driver manages the actual VRAM allocation
// and typically reuses pages across frames. We store prev_pyramid as
// Option<GpuPyramid> and build a fresh curr_pyramid each frame.
//
// RANSAC NOTE
// ────────────
// RANSAC runs on the tracked positions after KLT readback. It's CPU-side
// work on a small Vec<Correspondence> (one entry per tracked feature).
// With 200 features at 200 iterations this is ~1ms — not worth porting.
// Signature: essential::estimate_essential_ransac(&corrs, &config) → Option<RansacResult>
// where RansacResult.inliers: Vec<bool> matches the corrs slice indices.

use std::time::Instant;

use crate::camera::CameraIntrinsics;
use crate::essential::{self, Correspondence, RansacConfig};
use crate::fast::Feature;
use crate::frontend::{FrameStats, TimingStats};
use crate::gpu::device::GpuDevice;
use crate::gpu::fast::GpuFastDetector;
use crate::gpu::klt::GpuKltTracker;
use crate::gpu::pyramid::{GpuPyramid, GpuPyramidPipeline};
use crate::histeq::{self, HistEqMethod};
use crate::image::Image;
use crate::klt::TrackStatus;
use crate::nms::OccupancyNms;
use crate::occupancy::OccupancyGrid;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GPU visual frontend.
///
/// Mirrors `FrontendConfig` from frontend.rs. Fields that select between
/// CPU algorithm variants (LkMethod, DetectorType) are absent because the
/// GPU frontend always uses IC KLT and GPU FAST.
#[derive(Clone)]
pub struct GpuFrontendConfig {
    /// FAST corner detection threshold (pixel intensity difference).
    pub fast_threshold: u8,
    /// FAST arc length — minimum contiguous bright/dark pixels (9–12).
    pub fast_arc_length: usize,
    /// Maximum number of tracked features.
    pub max_features: usize,
    /// Occupancy grid / NMS cell size in pixels.
    pub cell_size: usize,
    /// Number of Gaussian pyramid levels.
    pub pyramid_levels: usize,
    /// Gaussian pyramid sigma.
    pub pyramid_sigma: f32,
    /// KLT patch half-size W. Patch is (2W+1)².
    pub klt_window: usize,
    /// KLT maximum Gauss-Newton iterations per pyramid level.
    pub klt_max_iter: usize,
    /// KLT convergence threshold in pixels.
    pub klt_epsilon: f32,
    /// Histogram equalization applied before GPU upload.
    /// Stabilizes brightness across frames when auto-exposure is active.
    pub histeq: HistEqMethod,
    /// Camera intrinsics for geometric verification (optional).
    /// If Some, RANSAC essential-matrix outlier rejection runs after KLT.
    pub camera: Option<CameraIntrinsics>,
    /// RANSAC configuration for essential matrix estimation.
    pub ransac: RansacConfig,
}

impl Default for GpuFrontendConfig {
    fn default() -> Self {
        GpuFrontendConfig {
            fast_threshold:  20,
            fast_arc_length: 9,
            max_features:    200,
            cell_size:       16,
            pyramid_levels:  3,
            pyramid_sigma:   1.0,
            klt_window:      7,
            klt_max_iter:    30,
            klt_epsilon:     0.01,
            histeq:          HistEqMethod::None,
            camera:          None,
            ransac:          RansacConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuFrontend
// ---------------------------------------------------------------------------

/// GPU visual frontend.
///
/// Create once with `GpuFrontend::new()`; call `process()` every frame.
/// GPU pipelines are compiled at construction time.
///
/// # Example
/// ```ignore
/// let gpu = GpuDevice::new().unwrap();
/// let config = GpuFrontendConfig {
///     max_features: 150,
///     pyramid_levels: 4,
///     klt_window: 7,
///     camera: Some(cam),
///     ..Default::default()
/// };
/// let mut frontend = GpuFrontend::new(&gpu, config, img_w, img_h);
///
/// loop {
///     let (features, stats) = frontend.process(&gpu, &frame);
///     // features: &[Feature] with persistent IDs, ready for VIO backend
///     println!("{}", stats.timing);
/// }
/// ```
pub struct GpuFrontend {
    config: GpuFrontendConfig,

    // GPU pipelines (compiled once).
    pyr_pipeline: GpuPyramidPipeline,
    fast:         GpuFastDetector,
    klt:          GpuKltTracker,

    // CPU post-processing.
    nms:  OccupancyNms,
    grid: OccupancyGrid,

    // Per-frame state.
    prev_pyramid:  Option<GpuPyramid>,
    features:      Vec<Feature>,
    prev_features: Vec<Feature>,
    next_id:       u64,
    has_prev:      bool,

    img_w: usize,
    img_h: usize,
}

impl GpuFrontend {
    /// Create a new GPU frontend for images of the given dimensions.
    ///
    /// This compiles three compute shaders (pyramid, FAST, KLT). Call once
    /// at startup, not every frame.
    pub fn new(gpu: &GpuDevice, config: GpuFrontendConfig, img_w: usize, img_h: usize) -> Self {
        let pyr_pipeline = GpuPyramidPipeline::new(gpu);
        let fast         = GpuFastDetector::new(gpu, config.fast_threshold, config.fast_arc_length);
        let klt          = GpuKltTracker::new(
            gpu,
            config.klt_window,
            config.klt_max_iter,
            config.klt_epsilon,
            config.pyramid_levels,
            config.max_features,
        );
        let nms  = OccupancyNms::new(config.cell_size);
        let grid = OccupancyGrid::new(img_w, img_h, config.cell_size);

        GpuFrontend {
            config,
            pyr_pipeline, fast, klt, nms, grid,
            prev_pyramid:  None,
            features:      Vec::new(),
            prev_features: Vec::new(),
            next_id:       1,
            has_prev:      false,
            img_w, img_h,
        }
    }

    /// Process one frame. Returns the tracked feature list and statistics.
    ///
    /// The returned `&[Feature]` slice is valid until the next `process()`
    /// call. Feature IDs are persistent: a feature keeps its ID as long as
    /// it tracks successfully.
    pub fn process<'a>(
        &'a mut self,
        gpu:   &GpuDevice,
        image: &Image<u8>,
    ) -> (&'a [Feature], FrameStats) {
        assert_eq!(image.width(),  self.img_w, "image width mismatch");
        assert_eq!(image.height(), self.img_h, "image height mismatch");

        let t_total = Instant::now();
        let mut timing = TimingStats::default();

        // ── Step 0: Histogram equalization (CPU, before GPU upload) ──────────
        let t0 = Instant::now();
        let equalized;
        let input: &Image<u8> = if self.config.histeq != HistEqMethod::None {
            equalized = histeq::apply_histeq(image, self.config.histeq);
            &equalized
        } else {
            image
        };
        timing.histeq = t0.elapsed().as_secs_f64();

        // ── Step 1: Build GPU pyramid ────────────────────────────────────────
        let t0 = Instant::now();
        let curr_pyramid = self.pyr_pipeline.build(
            gpu, input, self.config.pyramid_levels, self.config.pyramid_sigma,
        );
        timing.pyramid = t0.elapsed().as_secs_f64();

        let mut stats = FrameStats {
            tracked: 0, lost: 0, rejected: 0,
            new_detections: 0, total: 0,
            occupied_cells: 0,
            total_cells: self.grid.total_cells(),
            timing: TimingStats::default(),
        };

        // ── Step 2: KLT tracking ─────────────────────────────────────────────
        let t0 = Instant::now();
        if self.has_prev && !self.features.is_empty() {
            // Clone features into a local slice to satisfy the borrow checker:
            // self.klt.track() needs &mut self.klt, but self.features and
            // self.prev_pyramid are also part of self — Rust can't split the
            // borrow across fields through an if-let like this without help.
            // The clone is one Vec<Feature> copy (~N×16 bytes), negligible.
            let feats_snap: Vec<Feature> = self.features.clone();
            let prev = self.prev_pyramid.as_ref().unwrap();
            let results = self.klt.track(gpu, prev, &curr_pyramid, &feats_snap);

            // Filter in-place: keep only Tracked features.
            let mut write = 0;
            for (_i, result) in results.iter().enumerate() {
                if result.status == TrackStatus::Tracked {
                    self.features[write] = result.feature.clone();
                    write += 1;
                    stats.tracked += 1;
                } else {
                    stats.lost += 1;
                }
            }
            self.features.truncate(write);
        }
        timing.klt = t0.elapsed().as_secs_f64();

        // ── Step 2b: Geometric verification (RANSAC, CPU) ────────────────────
        let t0 = Instant::now();
        if let Some(ref cam) = self.config.camera {
            if self.features.len() >= 8 {
                // Build correspondences: match by feature ID between
                // prev_features (positions in frame t-1) and current
                // features (positions in frame t after KLT).
                let corrs: Vec<(usize, Correspondence)> = self.features.iter()
                    .enumerate()
                    .filter_map(|(idx, f)| {
                        self.prev_features.iter()
                            .find(|pf| pf.id == f.id)
                            .map(|pf| {
                                let (x1, y1) = cam.normalize_undistorted(pf.x as f64, pf.y as f64);
                                let (x2, y2) = cam.normalize_undistorted(f.x as f64, f.y as f64);
                                (idx, Correspondence { x1, y1, x2, y2 })
                            })
                    })
                    .collect();

                if corrs.len() >= 8 {
                    let corr_only: Vec<Correspondence> =
                        corrs.iter().map(|(_, c)| *c).collect();

                    if let Some(result) = essential::estimate_essential_ransac(
                        &corr_only, &self.config.ransac,
                    ) {
                        let mut inliers = Vec::new();
                        for (ci, (feat_idx, _)) in corrs.iter().enumerate() {
                            if result.inliers[ci] {
                                inliers.push(self.features[*feat_idx].clone());
                            } else {
                                stats.rejected += 1;
                            }
                        }
                        // Preserve features that had no previous match
                        // (newly detected this frame — can't form a correspondence).
                        let matched_ids: Vec<u64> =
                            corrs.iter().map(|(idx, _)| self.features[*idx].id).collect();
                        for f in &self.features {
                            if !matched_ids.contains(&f.id) {
                                inliers.push(f.clone());
                            }
                        }
                        stats.tracked = stats.tracked.saturating_sub(stats.rejected);
                        self.features = inliers;
                    }
                }
            }
        }
        timing.ransac = t0.elapsed().as_secs_f64();

        // ── Step 3: Update occupancy grid from surviving tracked features ─────
        self.grid.clear();
        for f in &self.features {
            self.grid.mark(f.x, f.y);
        }

        // ── Step 4: Detect + replenish ────────────────────────────────────────
        let t0 = Instant::now();
        let slots = self.config.max_features.saturating_sub(self.features.len());

        if slots > 0 {
            // Run GPU FAST on pyramid level 0.
            let mut raw = self.fast.detect(gpu, &curr_pyramid.levels[0]);

            // Filter to unoccupied cells only — mirrors CPU frontend's
            // detect_masked() call. The occupancy grid was just updated from
            // surviving tracked features, so this ensures new detections
            // don't pile on top of existing ones.
            let mask = self.grid.unoccupied_mask();
            raw.retain(|f| {
                let x = f.x as usize;
                let y = f.y as usize;
                x < mask.width() && y < mask.height() && mask.get(x, y) > 0
            });

            // Sort by score descending so NMS picks the strongest corner per cell.
            raw.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            // NMS over the unoccupied-only candidates, then assign IDs.
            let suppressed = self.nms.suppress(&raw, self.img_w, self.img_h);

            for f in suppressed.iter().take(slots) {
                let new_feat = Feature {
                    x:     f.x,
                    y:     f.y,
                    score: f.score,
                    level: f.level,
                    id:    self.next_id,
                };
                self.next_id += 1;
                self.features.push(new_feat);
                self.grid.mark(f.x, f.y);
                stats.new_detections += 1;
            }
        }
        timing.detect = t0.elapsed().as_secs_f64();

        // ── Step 5: Advance state ─────────────────────────────────────────────
        self.prev_pyramid  = Some(curr_pyramid);
        self.prev_features = self.features.clone();
        self.has_prev      = true;

        timing.total = t_total.elapsed().as_secs_f64();
        stats.total          = self.features.len();
        stats.occupied_cells = self.grid.total_cells() - self.grid.count_empty();
        stats.timing         = timing;

        (&self.features, stats)
    }

    /// Currently tracked features (without processing a new frame).
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Whether at least one frame has been processed.
    pub fn has_prev_frame(&self) -> bool {
        self.has_prev
    }

    /// Reset frontend state (features, previous frame, occupancy grid).
    /// Does NOT reset next_id — feature IDs remain globally unique.
    /// Does NOT recompile shaders.
    pub fn reset(&mut self) {
        self.has_prev = false;
        self.features.clear();
        self.prev_features.clear();
        self.prev_pyramid = None;
        self.grid.clear();
    }

    /// Current histogram equalization setting.
    pub fn histeq(&self) -> HistEqMethod {
        self.config.histeq
    }

    /// Change histogram equalization at runtime (takes effect next frame).
    pub fn set_histeq(&mut self, method: HistEqMethod) {
        self.config.histeq = method;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run_gpu_test(name: &str) -> String {
        let out = std::process::Command::new("cargo")
            .args(["test", "--lib", "--", name, "--exact", "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed: {e}"));
        String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr)
    }

    fn make_scene(w: usize, h: usize, shift_x: usize, shift_y: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![25u8; w * h]);
        for &(rx, ry, rw, rh, val) in &[
            (30usize, 25usize, 20usize, 20usize, 200u8),
            (70,  20, 25, 15, 180),
            (110, 30, 18, 22, 210),
            (25,  65, 22, 25, 190),
            (75,  60, 30, 20, 170),
            (115, 70, 20, 18, 205),
        ] {
            for y in (ry + shift_y)..((ry + shift_y + rh).min(h)) {
                for x in (rx + shift_x)..((rx + shift_x + rw).min(w)) {
                    img.set(x, y, val);
                }
            }
        }
        img
    }

    // ---- inner GPU tests (subprocess-isolated) ----------------------------

    #[test]
    #[ignore = "GPU integration"]
    fn inner_first_frame_detects() {
        let img = make_scene(160, 120, 0, 0);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 50, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let (features, stats) = fe.process(&gpu, &img);
        assert!(features.len() > 0, "first frame should detect features");
        assert_eq!(stats.tracked, 0, "nothing to track on first frame");
        assert!(stats.new_detections > 0);
        assert_eq!(stats.total, features.len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_unique_ids() {
        let img = make_scene(160, 120, 0, 0);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 50, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let (features, _) = fe.process(&gpu, &img);
        let mut ids: Vec<u64> = features.iter().map(|f| f.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), features.len(), "IDs must be unique");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_tracking_across_frames() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 50, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let n1 = fe.process(&gpu, &img1).0.len();
        assert!(n1 > 0);

        let (_, stats) = fe.process(&gpu, &img2);
        assert!(stats.tracked > 0, "should track some features: {stats:?}");
        println!("GPU_TEST_OK");
        drop(fe); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_ids_persist() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 50, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let ids1: Vec<u64> = fe.process(&gpu, &img1).0.iter().map(|f| f.id).collect();
        let ids2: Vec<u64> = fe.process(&gpu, &img2).0.iter().map(|f| f.id).collect();

        let persisted = ids2.iter().filter(|id| ids1.contains(id)).count();
        assert!(persisted > 0, "some IDs should persist across frames");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_max_features_respected() {
        let img = make_scene(160, 120, 0, 0);
        let gpu = GpuDevice::new().unwrap();
        let max = 15usize;
        let config = GpuFrontendConfig { max_features: max, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let (features, _) = fe.process(&gpu, &img);
        assert!(features.len() <= max, "features {} > max {max}", features.len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_replenishment_after_loss() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 15, 10); // large shift → many lost
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 50, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        fe.process(&gpu, &img1);
        let (_, stats) = fe.process(&gpu, &img2);
        assert!(stats.new_detections > 0, "should replenish: {stats:?}");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_reset_clears_state() {
        let img = make_scene(160, 120, 0, 0);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig::default();
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        fe.process(&gpu, &img);
        assert!(!fe.features().is_empty());

        fe.reset();
        assert!(fe.features().is_empty());
        assert!(!fe.has_prev_frame());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_three_frame_sequence() {
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig { max_features: 30, ..Default::default() };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        for i in 0..3usize {
            let img = make_scene(160, 120, i * 2, i);
            let (features, stats) = fe.process(&gpu, &img);
            eprintln!("frame {i}: tracked={} lost={} new={} total={}",
                stats.tracked, stats.lost, stats.new_detections, stats.total);
            if i > 0 {
                assert!(stats.tracked > 0, "frame {i}: should track some features");
            }
            assert!(!features.is_empty());
        }
        println!("GPU_TEST_OK");
    }

    // ---- outer subprocess wrappers ----------------------------------------

    macro_rules! gpu_test {
        ($outer:ident, $inner:ident) => {
            #[test]
            #[ignore = "requires a real Vulkan GPU"]
            fn $outer() {
                let out = run_gpu_test(concat!("gpu::frontend::tests::", stringify!($inner)));
                assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
            }
        };
    }

    gpu_test!(test_first_frame_detects,    inner_first_frame_detects);
    gpu_test!(test_unique_ids,             inner_unique_ids);
    gpu_test!(test_tracking_across_frames, inner_tracking_across_frames);
    gpu_test!(test_ids_persist,            inner_ids_persist);
    gpu_test!(test_max_features_respected, inner_max_features_respected);
    gpu_test!(test_replenishment,          inner_replenishment_after_loss);
    gpu_test!(test_reset,                  inner_reset_clears_state);
    gpu_test!(test_three_frame_sequence,   inner_three_frame_sequence);
}
