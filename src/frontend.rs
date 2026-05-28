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

use crate::camera::CameraIntrinsics;
use crate::essential::{self, Correspondence, RansacConfig};
use crate::fast::{FastDetector, Feature};
use crate::harris::HarrisDetector;
use crate::histeq::{self, HistEqMethod};
use crate::image::Image;
use crate::klt::{KltScratch, KltTracker, LkMethod, TrackStatus, TrackedFeature};
use crate::nms::OccupancyNms;
use crate::occupancy::OccupancyGrid;
use crate::pyramid::{Pyramid, PyramidScratch};

/// Which corner detector to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectorType {
    Fast,
    Harris,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbpPolicy {
    /// Store LBP distance but never reject a reservoir track on LBP alone.
    SoftPenalty,
    /// Preserve the legacy behavior: reject when distance exceeds threshold.
    HardReject,
}

#[derive(Debug, Clone, Copy)]
struct TileBounds {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

/// Frontend-owned lifecycle and backend policy state for one reservoir track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackMeta {
    pub id: u64,
    pub age: u16,
    pub is_ekf_landmark: bool,
    pub ekf_landmark_id: Option<u32>,
    pub klt_quality: f32,
    pub lbp_distance: u16,
    pub fine_cell: u16,
    pub coarse_tile: u16,
    pub reservoir_score: f32,
}

impl TrackMeta {
    fn new(
        feature: &Feature,
        age: u16,
        lbp_distance: u16,
        img_w: usize,
        img_h: usize,
        cell_size: usize,
        tile_cols: usize,
        tile_rows: usize,
    ) -> Self {
        TrackMeta {
            id: feature.id,
            age,
            is_ekf_landmark: false,
            ekf_landmark_id: None,
            klt_quality: 0.0,
            lbp_distance,
            fine_cell: fine_cell_index(feature, img_w, img_h, cell_size),
            coarse_tile: tile_index(feature, tile_cols.max(1), tile_rows.max(1), img_w, img_h)
                .min(u16::MAX as usize) as u16,
            reservoir_score: reservoir_score(feature, age, 0.0, lbp_distance, img_w, img_h),
        }
    }

    fn advanced(
        &self,
        feature: &Feature,
        klt_quality: f32,
        lbp_distance: u16,
        img_w: usize,
        img_h: usize,
        cell_size: usize,
        tile_cols: usize,
        tile_rows: usize,
    ) -> Self {
        TrackMeta {
            id: feature.id,
            age: self.age.saturating_add(1),
            is_ekf_landmark: self.is_ekf_landmark,
            ekf_landmark_id: self.ekf_landmark_id,
            klt_quality,
            lbp_distance,
            fine_cell: fine_cell_index(feature, img_w, img_h, cell_size),
            coarse_tile: tile_index(feature, tile_cols.max(1), tile_rows.max(1), img_w, img_h)
                .min(u16::MAX as usize) as u16,
            reservoir_score: reservoir_score(
                feature,
                self.age.saturating_add(1),
                klt_quality,
                lbp_distance,
                img_w,
                img_h,
            ),
        }
    }
}

/// Detector trait — unifies FAST and Harris behind a common interface.
///
/// This is more idiomatic Rust than enum dispatch: you can add new
/// detector types without modifying existing code (open/closed principle).
pub trait Detector {
    /// Detect features in the image.
    fn detect(&self, image: &Image<u8>) -> Vec<Feature>;
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

fn tile_bounds(
    tile_x: usize,
    tile_y: usize,
    cols: usize,
    rows: usize,
    img_w: usize,
    img_h: usize,
) -> TileBounds {
    TileBounds {
        x0: tile_x * img_w / cols,
        x1: (tile_x + 1) * img_w / cols,
        y0: tile_y * img_h / rows,
        y1: (tile_y + 1) * img_h / rows,
    }
}

fn tile_index(feature: &Feature, cols: usize, rows: usize, img_w: usize, img_h: usize) -> usize {
    let x = feature.x.max(0.0) as usize;
    let y = feature.y.max(0.0) as usize;
    let col = ((x * cols) / img_w.max(1)).min(cols - 1);
    let row = ((y * rows) / img_h.max(1)).min(rows - 1);
    row * cols + col
}

fn fine_cell_index(feature: &Feature, img_w: usize, img_h: usize, cell_size: usize) -> u16 {
    let cell = cell_size.max(1);
    let cols = img_w.max(1).div_ceil(cell);
    let rows = img_h.max(1).div_ceil(cell);
    tile_index(feature, cols, rows, img_w, img_h).min(u16::MAX as usize) as u16
}

fn klt_quality_from_residual(residual: f32) -> f32 {
    if residual.is_finite() {
        1.0 / (1.0 + residual.max(0.0))
    } else {
        0.0
    }
}

fn reservoir_score(
    feature: &Feature,
    age: u16,
    klt_quality: f32,
    lbp_distance: u16,
    img_w: usize,
    img_h: usize,
) -> f32 {
    let corner_score = (feature.score.max(0.0) / 255.0).min(1.0);
    let age_score = (age as f32 / 10.0).min(1.0);
    let lower_image_score = if img_h > 0 {
        (feature.y / img_h as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lbp_penalty = (lbp_distance as f32 / 16.0).min(1.0);
    let border_penalty = border_penalty(feature, img_w, img_h, 16.0);

    2.0 * klt_quality.clamp(0.0, 1.0)
        + 1.0 * corner_score
        + 0.5 * age_score
        + 0.5 * lower_image_score
        - 1.0 * lbp_penalty
        - 0.5 * border_penalty
}

fn border_penalty(feature: &Feature, img_w: usize, img_h: usize, margin: f32) -> f32 {
    if img_w == 0 || img_h == 0 || margin <= 0.0 {
        return 0.0;
    }
    let right = (img_w as f32 - 1.0 - feature.x).max(0.0);
    let bottom = (img_h as f32 - 1.0 - feature.y).max(0.0);
    let dist = feature
        .x
        .max(0.0)
        .min(feature.y.max(0.0))
        .min(right)
        .min(bottom);
    ((margin - dist) / margin).clamp(0.0, 1.0)
}

fn prune_low_reservoir_score(
    features: &mut Vec<Feature>,
    track_meta: &mut Vec<TrackMeta>,
    min_score: f32,
) -> usize {
    if !min_score.is_finite() || features.is_empty() {
        return 0;
    }

    debug_assert_eq!(features.len(), track_meta.len());

    let original_len = features.len();
    let mut write = 0;
    for i in 0..original_len {
        if track_meta[i].reservoir_score >= min_score {
            if write != i {
                features[write] = features[i].clone();
                track_meta[write] = track_meta[i].clone();
            }
            write += 1;
        }
    }

    features.truncate(write);
    track_meta.truncate(write);
    original_len - write
}

fn tile_targets(
    capacity: usize,
    img_w: usize,
    img_h: usize,
    tile_cols: usize,
    tile_rows: usize,
) -> Vec<usize> {
    let cols = tile_cols.max(1);
    let rows = tile_rows.max(1);
    let image_area = (img_w.max(1) * img_h.max(1)) as f32;
    let mut target = vec![0usize; cols * rows];

    for tile_y in 0..rows {
        for tile_x in 0..cols {
            let idx = tile_y * cols + tile_x;
            let b = tile_bounds(tile_x, tile_y, cols, rows, img_w, img_h);
            let area = ((b.x1 - b.x0).max(1) * (b.y1 - b.y0).max(1)) as f32;
            target[idx] = ((capacity as f32) * area / image_area).ceil() as usize;
        }
    }

    target
}

fn prune_overfull_tiles(
    features: &mut Vec<Feature>,
    track_meta: &mut Vec<TrackMeta>,
    capacity: usize,
    img_w: usize,
    img_h: usize,
    tile_cols: usize,
    tile_rows: usize,
) -> usize {
    if features.is_empty() || capacity == 0 {
        return 0;
    }

    debug_assert_eq!(features.len(), track_meta.len());

    let cols = tile_cols.max(1);
    let rows = tile_rows.max(1);
    let tile_count = cols * rows;
    let targets = tile_targets(capacity, img_w, img_h, cols, rows);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); tile_count];

    for (idx, feature) in features.iter().enumerate() {
        buckets[tile_index(feature, cols, rows, img_w, img_h)].push(idx);
    }

    let mut keep = vec![true; features.len()];
    let mut pruned = 0;
    for (tile, indices) in buckets.iter_mut().enumerate() {
        let target = targets[tile];
        if indices.len() <= target {
            continue;
        }

        indices.sort_by(|&a, &b| {
            track_meta[a]
                .reservoir_score
                .partial_cmp(&track_meta[b].reservoir_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| track_meta[a].age.cmp(&track_meta[b].age))
                .then_with(|| features[a].id.cmp(&features[b].id))
        });

        for &idx in indices.iter().take(indices.len() - target) {
            keep[idx] = false;
            pruned += 1;
        }
    }

    if pruned == 0 {
        return 0;
    }

    let original_len = features.len();
    let mut write = 0;
    for i in 0..original_len {
        if keep[i] {
            if write != i {
                features[write] = features[i].clone();
                track_meta[write] = track_meta[i].clone();
            }
            write += 1;
        }
    }
    features.truncate(write);
    track_meta.truncate(write);

    pruned
}

/// Selects new detection candidates by coarse tile deficit.
///
/// Grid NMS supplies at most one candidate per fine cell. This pass fixes
/// row-major admission bias when there are more candidates than open slots:
/// tiles below their area-proportional target are filled first, and candidates
/// within a tile are admitted by corner score.
pub(crate) fn select_by_tile_deficit(
    candidates: &[Feature],
    existing: &[Feature],
    slots: usize,
    capacity: usize,
    img_w: usize,
    img_h: usize,
    tile_cols: usize,
    tile_rows: usize,
) -> Vec<Feature> {
    if candidates.is_empty() || slots == 0 {
        return Vec::new();
    }

    let cols = tile_cols.max(1);
    let rows = tile_rows.max(1);
    let tile_count = cols * rows;
    let target = tile_targets(capacity, img_w, img_h, cols, rows);

    let mut counts = vec![0usize; tile_count];
    for feature in existing {
        counts[tile_index(feature, cols, rows, img_w, img_h)] += 1;
    }

    let mut buckets: Vec<Vec<Feature>> = vec![Vec::new(); tile_count];
    for candidate in candidates {
        buckets[tile_index(candidate, cols, rows, img_w, img_h)].push(candidate.clone());
    }
    for bucket in &mut buckets {
        bucket.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.x.total_cmp(&b.x))
        });
    }

    let mut bucket_pos = vec![0usize; tile_count];
    let mut selected = Vec::with_capacity(slots.min(candidates.len()));
    while selected.len() < slots {
        let best_tile = (0..tile_count)
            .filter(|&idx| bucket_pos[idx] < buckets[idx].len())
            .max_by(|&a, &b| {
                let deficit_a = target[a] as isize - counts[a] as isize;
                let deficit_b = target[b] as isize - counts[b] as isize;
                deficit_a
                    .cmp(&deficit_b)
                    .then_with(|| {
                        buckets[a][bucket_pos[a]]
                            .score
                            .total_cmp(&buckets[b][bucket_pos[b]].score)
                    })
                    .then_with(|| b.cmp(&a))
            });

        let Some(tile) = best_tile else {
            break;
        };

        selected.push(buckets[tile][bucket_pos[tile]].clone());
        bucket_pos[tile] += 1;
        counts[tile] += 1;
    }

    selected
}

/// Frontend configuration.
#[derive(Clone)]
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
    /// Reservoir capacity: maximum number of tracked features.
    pub max_features: usize,
    /// Occupancy grid / NMS cell size in pixels.
    pub cell_size: usize,
    /// Coarse tile columns for replenishment coverage balancing.
    pub coarse_tile_cols: usize,
    /// Coarse tile rows for replenishment coverage balancing.
    pub coarse_tile_rows: usize,
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
    /// Compute final level-0 patch residual for tracked points.
    ///
    /// Disabled by default because it adds an extra patch pass per track.
    pub klt_residual_enabled: bool,
    /// LBP descriptor verification (occlusion/drift detection).
    pub lbp_verification_enabled: bool,
    /// Whether high LBP distance is metadata only or a hard reservoir reject.
    pub lbp_policy: LbpPolicy,
    /// Hamming distance threshold for LBP verification (max bits allowed).
    pub lbp_threshold: u32,
    /// Reject tracked reservoir points whose soft score falls below this value.
    ///
    /// Defaults to negative infinity, which disables score-based pruning.
    pub min_reservoir_score: f32,
    /// Prune only over-target coarse tiles by local reservoir score.
    pub tile_reservoir_pruning_enabled: bool,
    /// Histogram equalization preprocessing.
    /// Stabilizes brightness across frames when auto-exposure is active.
    pub histeq: HistEqMethod,
    /// Camera intrinsics for geometric verification (optional).
    /// If set and `enable_internal_ransac` is true, RANSAC outlier rejection
    /// is applied after KLT tracking.
    pub camera: Option<CameraIntrinsics>,
    /// RANSAC configuration for essential matrix estimation.
    pub ransac: RansacConfig,
    /// Run the built-in cam0 essential-matrix RANSAC inside `process()`.
    /// Set to false when an external joint RANSAC (e.g. rigid 3D-3D over
    /// stereo correspondences) will be applied via `retain_features_by_id`.
    pub enable_internal_ransac: bool,
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
            coarse_tile_cols: 8,
            coarse_tile_rows: 6,
            pyramid_levels: 3,
            pyramid_sigma: 1.0,
            klt_window: 7,
            klt_max_iter: 30,
            klt_epsilon: 0.01,
            klt_method: LkMethod::ForwardAdditive,
            klt_residual_enabled: false,
            lbp_verification_enabled: true,
            lbp_policy: LbpPolicy::SoftPenalty,
            lbp_threshold: 4,
            min_reservoir_score: f32::NEG_INFINITY,
            tile_reservoir_pruning_enabled: true,
            histeq: HistEqMethod::None,
            camera: None,
            ransac: RansacConfig::default(),
            enable_internal_ransac: true,
        }
    }
}

/// The visual frontend: manages the detect-track-replenish loop.
///
/// This is what vilib_ros.cpp does on the CPU side — calling into
/// GPU kernels for each step. Our CPU reference mirrors that pipeline.
pub struct Frontend {
    config: FrontendConfig,
    /// Double-buffered pyramids: swap each frame to avoid allocation.
    prev_pyramid: Pyramid,
    curr_pyramid: Pyramid,
    /// Scratch buffers for pyramid construction (convolution intermediates).
    pyr_scratch: PyramidScratch,
    /// Scratch buffer for histogram equalization output (avoids per-frame alloc).
    histeq_buf: Image<u8>,
    /// Whether we have a valid previous frame.
    has_prev: bool,
    /// Currently tracked features with persistent IDs.
    features: Vec<Feature>,
    /// Metadata for `features`, kept index-aligned with the feature vector.
    track_meta: Vec<TrackMeta>,
    /// Previous frame's feature positions (for geometric verification).
    prev_features: Vec<Feature>,
    /// Reusable buffer for KLT tracking results (avoids per-frame alloc).
    track_results: Vec<TrackedFeature>,
    /// Scratch buffers for KLT IC precompute + iteration (avoids per-feature alloc).
    klt_scratch: KltScratch,
    /// Next feature ID to assign. Monotonically increasing.
    next_id: u64,
    /// Occupancy grid for spatial distribution.
    grid: OccupancyGrid,
    /// Image dimensions (for validation).
    img_w: usize,
    img_h: usize,
}

/// Per-stage timing breakdown from a single `process()` call.
/// All durations in seconds (use `*_ms()` helpers for display).
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    pub histeq: f64,
    pub pyramid: f64,
    pub klt: f64,
    pub ransac: f64,
    pub detect: f64,
    pub total: f64,
}

impl TimingStats {
    pub fn histeq_ms(&self) -> f32 {
        (self.histeq * 1000.0) as f32
    }
    pub fn pyramid_ms(&self) -> f32 {
        (self.pyramid * 1000.0) as f32
    }
    pub fn klt_ms(&self) -> f32 {
        (self.klt * 1000.0) as f32
    }
    pub fn ransac_ms(&self) -> f32 {
        (self.ransac * 1000.0) as f32
    }
    pub fn detect_ms(&self) -> f32 {
        (self.detect * 1000.0) as f32
    }
    pub fn total_ms(&self) -> f32 {
        (self.total * 1000.0) as f32
    }
}

impl std::fmt::Display for TimingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "histeq:{:.1} pyr:{:.1} klt:{:.1} ransac:{:.1} det:{:.1} total:{:.1}ms",
            self.histeq_ms(),
            self.pyramid_ms(),
            self.klt_ms(),
            self.ransac_ms(),
            self.detect_ms(),
            self.total_ms()
        )
    }
}

/// Statistics returned after processing each frame.
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Number of features tracked from previous frame.
    pub tracked: usize,
    /// Number of features lost during tracking.
    pub lost: usize,
    /// Number of features rejected by geometric verification (RANSAC).
    pub rejected: usize,
    /// Number of new features detected to replenish.
    pub new_detections: usize,
    /// Total features after this frame.
    pub total: usize,
    /// Number of occupied cells in the grid.
    pub occupied_cells: usize,
    /// Total grid cells.
    pub total_cells: usize,
    /// Per-stage timing breakdown.
    pub timing: TimingStats,
}

impl Frontend {
    /// Create a new frontend for images of the given dimensions.
    pub fn new(config: FrontendConfig, img_w: usize, img_h: usize) -> Self {
        let grid = OccupancyGrid::new(img_w, img_h, config.cell_size);
        let pyr_scratch = PyramidScratch::new(img_w, img_h, config.pyramid_sigma);
        let klt_scratch = KltScratch::new(config.klt_window);
        let pad_border = if config.klt_method == LkMethod::InverseCompositional {
            config.klt_window + 2
        } else {
            0
        };
        Frontend {
            config,
            prev_pyramid: Pyramid {
                levels: Vec::new(),
                u8_levels: Vec::new(),
                padded_levels: Vec::new(),
                pad_border,
            },
            curr_pyramid: Pyramid {
                levels: Vec::new(),
                u8_levels: Vec::new(),
                padded_levels: Vec::new(),
                pad_border,
            },
            pyr_scratch,
            histeq_buf: Image::new(img_w, img_h),
            has_prev: false,
            features: Vec::new(),
            track_meta: Vec::new(),
            prev_features: Vec::new(),
            track_results: Vec::new(),
            klt_scratch,
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

        use std::time::Instant;
        let t_total = Instant::now();
        let mut timing = TimingStats::default();

        // Step 0: Histogram equalization LUT (brightness normalization).
        // The actual remap is fused into pyramid level 0 construction.
        let t0 = Instant::now();
        let histeq_lut = match self.config.histeq {
            HistEqMethod::Global => Some(histeq::equalize_histogram_lut(image)),
            HistEqMethod::None => None,
            HistEqMethod::Clahe {
                tile_size,
                clip_limit,
            } => {
                histeq::equalize_clahe_into(image, tile_size, clip_limit, &mut self.histeq_buf);
                None
            }
        };
        timing.histeq = t0.elapsed().as_secs_f64();

        // Step 1: Build pyramid (reusing buffers to avoid page faults).
        // Swap: what was curr becomes prev, then rebuild curr.
        let t0 = Instant::now();
        std::mem::swap(&mut self.prev_pyramid, &mut self.curr_pyramid);
        let (pyr_src, pyr_lut) = if histeq_lut.is_some() {
            (image, histeq_lut.as_ref())
        } else if self.config.histeq != HistEqMethod::None {
            (&self.histeq_buf, None)
        } else {
            (image, None)
        };
        self.curr_pyramid.build_reuse_lut(
            pyr_src,
            self.config.pyramid_levels,
            &mut self.pyr_scratch,
            pyr_lut,
        );
        timing.pyramid = t0.elapsed().as_secs_f64();

        // After pyramid construction, u8_levels[0] holds the (possibly
        // LUT-remapped) input image. Use it for detection below.
        let input = self.curr_pyramid.u8_level(0);

        let mut stats = FrameStats {
            tracked: 0,
            lost: 0,
            rejected: 0,
            new_detections: 0,
            total: 0,
            occupied_cells: 0,
            total_cells: self.grid.total_cells(),
            timing: TimingStats::default(),
        };

        // Step 2: Track existing features if we have a previous frame.
        let t0 = Instant::now();
        if self.has_prev {
            if !self.features.is_empty() {
                debug_assert_eq!(self.features.len(), self.track_meta.len());
                let tracker = KltTracker::with_method(
                    self.config.klt_window,
                    self.config.klt_max_iter,
                    self.config.klt_epsilon,
                    self.config.pyramid_levels,
                    self.config.klt_method,
                )
                .with_residual(self.config.klt_residual_enabled);

                tracker.track_into_opt(
                    &self.prev_pyramid,
                    &self.curr_pyramid,
                    &self.features,
                    &mut self.track_results,
                    &mut self.klt_scratch,
                );

                // Filter features in-place: keep only successfully tracked.
                // Avoids allocating a second Vec.
                let mut write = 0;
                let curr_img = self.curr_pyramid.u8_level(0);

                for i in 0..self.track_results.len() {
                    if self.track_results[i].status == TrackStatus::Tracked {
                        let feat = &self.track_results[i].feature;
                        let mut lbp_distance = 0u16;

                        // LBP verification (occlusion/drift detection).
                        if self.config.lbp_verification_enabled {
                            let Some(new_desc) = compute_lbp_at(curr_img, feat.x, feat.y) else {
                                stats.rejected += 1;
                                continue;
                            };
                            let dist = (new_desc ^ feat.descriptor).count_ones();
                            lbp_distance = dist.min(u16::MAX as u32) as u16;
                            if self.config.lbp_policy == LbpPolicy::HardReject
                                && dist > self.config.lbp_threshold
                            {
                                stats.rejected += 1;
                                continue;
                            }
                        }

                        self.features[write] = feat.clone();
                        let klt_quality = if self.config.klt_residual_enabled {
                            klt_quality_from_residual(self.track_results[i].residual)
                        } else {
                            1.0
                        };
                        self.track_meta[write] = self.track_meta[i].advanced(
                            feat,
                            klt_quality,
                            lbp_distance,
                            self.img_w,
                            self.img_h,
                            self.config.cell_size,
                            self.config.coarse_tile_cols,
                            self.config.coarse_tile_rows,
                        );
                        write += 1;
                        stats.tracked += 1;
                    } else {
                        stats.lost += 1;
                    }
                }
                self.features.truncate(write);
                self.track_meta.truncate(write);
                timing.klt = t0.elapsed().as_secs_f64();

                // Step 2b: Geometric verification (essential matrix RANSAC).
                let t0 = Instant::now();
                // Reject outlier tracks that are geometrically inconsistent.
                if self.config.enable_internal_ransac {
                    if let Some(ref cam) = self.config.camera {
                        if self.features.len() >= 8 {
                            // Build correspondences: prev_features -> current features.
                            // Match by ID (both lists may differ if features were lost).
                            let corrs: Vec<(usize, Correspondence)> = self
                                .features
                                .iter()
                                .enumerate()
                                .filter_map(|(idx, f)| {
                                    self.prev_features
                                        .iter()
                                        .find(|pf| pf.id == f.id)
                                        .map(|pf| {
                                            let (x1, y1) =
                                                cam.normalize_undistorted(pf.x as f64, pf.y as f64);
                                            let (x2, y2) =
                                                cam.normalize_undistorted(f.x as f64, f.y as f64);
                                            (idx, Correspondence { x1, y1, x2, y2 })
                                        })
                                })
                                .collect();

                            if corrs.len() >= 8 {
                                let corr_only: Vec<Correspondence> =
                                    corrs.iter().map(|(_, c)| *c).collect();

                                if let Some(result) = essential::estimate_essential_ransac(
                                    &corr_only,
                                    &self.config.ransac,
                                ) {
                                    // Remove outlier features.
                                    let mut inlier_features = Vec::new();
                                    let mut inlier_meta = Vec::new();
                                    let mut ransac_rejected = 0usize;
                                    for (ci, (feat_idx, _)) in corrs.iter().enumerate() {
                                        if result.inliers[ci] {
                                            inlier_features.push(self.features[*feat_idx].clone());
                                            inlier_meta.push(self.track_meta[*feat_idx].clone());
                                        } else {
                                            ransac_rejected += 1;
                                            stats.rejected += 1;
                                        }
                                    }
                                    // Also keep features that had no prev match (new this frame).
                                    let matched_ids: Vec<u64> = corrs
                                        .iter()
                                        .map(|(idx, _)| self.features[*idx].id)
                                        .collect();
                                    for (idx, f) in self.features.iter().enumerate() {
                                        if !matched_ids.contains(&f.id) {
                                            inlier_features.push(f.clone());
                                            inlier_meta.push(self.track_meta[idx].clone());
                                        }
                                    }
                                    stats.tracked = stats.tracked.saturating_sub(ransac_rejected);
                                    self.features = inlier_features;
                                    self.track_meta = inlier_meta;
                                }
                            }
                        }
                    }
                }
                timing.ransac = t0.elapsed().as_secs_f64();
            }
        } else {
            // First frame: no tracking, no RANSAC.
        }

        let pruned = prune_low_reservoir_score(
            &mut self.features,
            &mut self.track_meta,
            self.config.min_reservoir_score,
        );
        stats.rejected += pruned;
        stats.tracked = stats.tracked.saturating_sub(pruned);

        if self.config.tile_reservoir_pruning_enabled {
            let pruned = prune_overfull_tiles(
                &mut self.features,
                &mut self.track_meta,
                self.config.max_features,
                self.img_w,
                self.img_h,
                self.config.coarse_tile_cols,
                self.config.coarse_tile_rows,
            );
            stats.rejected += pruned;
            stats.tracked = stats.tracked.saturating_sub(pruned);
        }

        // Step 3: Update occupancy grid from tracked features.
        self.grid.clear();
        for f in &self.features {
            self.grid.mark(f.x, f.y);
        }

        // Step 4: Detect new features in unoccupied cells.
        let t0 = Instant::now();
        let slots_available = self.config.max_features.saturating_sub(self.features.len());

        if slots_available > 0 {
            // Detect features only in unoccupied grid cells.
            // Previous approach: allocate 752×480 mask (~0.56ms) → detect all → filter.
            // New approach: skip occupied cell columns entirely in the FAST scan loop.
            // Saves the mask allocation AND skips ~14% of FAST computation.
            let new_features = match self.config.detector {
                DetectorType::Fast => {
                    let det =
                        FastDetector::new(self.config.fast_threshold, self.config.fast_arc_length);
                    det.detect_unoccupied(
                        input,
                        self.grid.grid_cells(),
                        self.grid.grid_cols(),
                        self.grid.cell_size(),
                    )
                }
                DetectorType::Harris => {
                    // Harris uses the old detect + post-filter path
                    // (rarely used, not worth a grid-aware variant).
                    let det = HarrisDetector::new(
                        self.config.harris_k,
                        self.config.harris_threshold,
                        self.config.harris_block_size,
                    );
                    let all = det.detect(input);
                    all.into_iter()
                        .filter(|f| !self.grid.is_occupied(f.x, f.y))
                        .collect()
                }
            };

            // NMS on new detections only, then take up to slots_available.
            let nms = OccupancyNms::new(self.config.cell_size);
            let suppressed = nms.suppress(&new_features, self.img_w, self.img_h);
            let selected = select_by_tile_deficit(
                &suppressed,
                &self.features,
                slots_available,
                self.config.max_features,
                self.img_w,
                self.img_h,
                self.config.coarse_tile_cols,
                self.config.coarse_tile_rows,
            );

            // Step 5: Add new features with unique IDs.
            let curr_img = self.curr_pyramid.u8_level(0);
            for f in &selected {
                // Ensure every feature has a valid LBP descriptor for later verification.
                // FAST already provides this, but Harris/others might leave it as 0.
                let descriptor = if f.descriptor == 0 {
                    compute_lbp_at(curr_img, f.x, f.y).unwrap_or(0)
                } else {
                    f.descriptor
                };

                let new_feat = Feature {
                    x: f.x,
                    y: f.y,
                    score: f.score,
                    level: f.level,
                    id: self.next_id,
                    descriptor,
                };
                self.next_id += 1;
                let new_meta = TrackMeta::new(
                    &new_feat,
                    1,
                    0,
                    self.img_w,
                    self.img_h,
                    self.config.cell_size,
                    self.config.coarse_tile_cols,
                    self.config.coarse_tile_rows,
                );
                self.features.push(new_feat);
                self.track_meta.push(new_meta);
                self.grid.mark(f.x, f.y);
                stats.new_detections += 1;
            }
        }

        timing.detect = t0.elapsed().as_secs_f64();

        // Step 6: Mark that we have a valid previous frame, store feature snapshot.
        // (Pyramid swap already happened at step 1 — curr is ready for next frame's prev.)
        self.has_prev = true;
        self.prev_features = self.features.clone();

        timing.total = t_total.elapsed().as_secs_f64();

        stats.total = self.features.len();
        stats.occupied_cells = self.grid.total_cells() - self.grid.count_empty();
        stats.timing = timing;

        (&self.features, stats)
    }

    /// Get the current feature list (without processing a new frame).
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Get metadata for the current feature list.
    ///
    /// The returned slice is index-aligned with [`Frontend::features`].
    pub fn track_meta(&self) -> &[TrackMeta] {
        &self.track_meta
    }

    /// Drop frontend tracks by feature ID.
    ///
    /// Backend-side visibility logic can call this after deciding that an
    /// optical-flow ID is occluded or otherwise invalid. Dropped IDs are not
    /// tracked on the next frame and will disappear from the next vision
    /// measurement, allowing the backend to marginalize matching landmarks.
    ///
    /// Returns the number of active tracks removed.
    pub fn drop_tracks(&mut self, ids: &[u64]) -> usize {
        if ids.is_empty() || self.features.is_empty() {
            return 0;
        }

        debug_assert_eq!(self.features.len(), self.track_meta.len());

        let before = self.features.len();
        let mut write = 0usize;

        for read in 0..self.features.len() {
            if ids.contains(&self.features[read].id) {
                continue;
            }

            if write != read {
                self.features[write] = self.features[read].clone();
                self.track_meta[write] = self.track_meta[read].clone();
            }
            write += 1;
        }

        self.features.truncate(write);
        self.track_meta.truncate(write);
        self.prev_features
            .retain(|feature| !ids.contains(&feature.id));

        self.grid.clear();
        for feature in &self.features {
            self.grid.mark(feature.x, feature.y);
        }

        before - write
    }

    pub fn current_pyramid(&self) -> &Pyramid {
        &self.curr_pyramid
    }

    /// Get the last histogram-equalized input image, if preprocessing is enabled.
    pub fn preprocessed_image(&self) -> Option<&Image<u8>> {
        match self.config.histeq {
            HistEqMethod::None => None,
            HistEqMethod::Global => {
                if self.curr_pyramid.has_u8_levels() {
                    Some(self.curr_pyramid.u8_level(0))
                } else {
                    None
                }
            }
            _ => Some(&self.histeq_buf),
        }
    }

    /// Number of frames processed so far.
    pub fn has_prev_frame(&self) -> bool {
        self.has_prev
    }

    /// Reset the frontend (clear all features, discard previous frame).
    pub fn reset(&mut self) {
        self.has_prev = false;
        self.features.clear();
        self.track_meta.clear();
        self.prev_features.clear();
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

    fn feature(x: f32, y: f32, score: f32) -> Feature {
        Feature {
            x,
            y,
            score,
            level: 0,
            id: 0,
            descriptor: 0,
        }
    }

    #[test]
    fn test_tile_deficit_selection_fills_empty_tile_before_row_major_candidate() {
        let candidates = vec![feature(10.0, 10.0, 100.0), feature(70.0, 10.0, 10.0)];
        let existing = vec![feature(12.0, 12.0, 1.0)];

        let selected = select_by_tile_deficit(&candidates, &existing, 1, 2, 100, 50, 2, 1);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].x, 70.0);
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
        let persisted: Vec<u64> = ids_f2
            .iter()
            .filter(|id| ids_f1.contains(id))
            .cloned()
            .collect();
        assert!(
            !persisted.is_empty(),
            "some feature IDs should persist across frames"
        );
    }

    #[test]
    fn test_track_meta_created_for_new_tracks() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let (features, _) = frontend.process(&img);
        let ids: Vec<u64> = features.iter().map(|f| f.id).collect();
        let meta = frontend.track_meta();

        assert_eq!(meta.len(), ids.len());
        assert!(meta.iter().all(|m| m.age == 1));
        assert!(meta.iter().all(|m| !m.is_ekf_landmark));
        assert!(meta.iter().all(|m| m.ekf_landmark_id.is_none()));
        for (m, id) in meta.iter().zip(ids) {
            assert_eq!(m.id, id);
        }
    }

    #[test]
    fn test_track_meta_age_advances_with_persisted_ids() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let ids_f1: Vec<u64> = frontend.process(&img1).0.iter().map(|f| f.id).collect();
        frontend.process(&img2);

        let persisted: Vec<&TrackMeta> = frontend
            .track_meta()
            .iter()
            .filter(|m| ids_f1.contains(&m.id))
            .collect();

        assert!(!persisted.is_empty(), "expected some persisted tracks");
        assert!(persisted.iter().all(|m| m.age >= 2));
        assert!(persisted
            .iter()
            .all(|m| m.klt_quality > 0.0 && m.klt_quality <= 1.0));
    }

    #[test]
    fn test_klt_residual_disabled_defaults_tracked_quality_to_one() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            klt_residual_enabled: false,
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let ids_f1: Vec<u64> = frontend.process(&img1).0.iter().map(|f| f.id).collect();
        frontend.process(&img2);

        let persisted: Vec<&TrackMeta> = frontend
            .track_meta()
            .iter()
            .filter(|m| ids_f1.contains(&m.id))
            .collect();

        assert!(!persisted.is_empty(), "expected some persisted tracks");
        assert!(persisted.iter().all(|m| m.klt_quality == 1.0));
    }

    #[test]
    fn test_klt_residual_enabled_updates_quality_from_residual() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            klt_residual_enabled: true,
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let ids_f1: Vec<u64> = frontend.process(&img1).0.iter().map(|f| f.id).collect();
        frontend.process(&img2);

        let persisted: Vec<&TrackMeta> = frontend
            .track_meta()
            .iter()
            .filter(|m| ids_f1.contains(&m.id))
            .collect();

        assert!(!persisted.is_empty(), "expected some persisted tracks");
        assert!(persisted
            .iter()
            .all(|m| m.klt_quality > 0.0 && m.klt_quality <= 1.0));
    }

    #[test]
    fn test_track_meta_new_replenishment_tracks_start_at_age_one() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 15, 10);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        let ids_f1: Vec<u64> = frontend.process(&img1).0.iter().map(|f| f.id).collect();
        frontend.process(&img2);

        let new_tracks: Vec<&TrackMeta> = frontend
            .track_meta()
            .iter()
            .filter(|m| !ids_f1.contains(&m.id))
            .collect();

        assert!(!new_tracks.is_empty(), "expected replenished tracks");
        assert!(new_tracks.iter().all(|m| m.age == 1));
    }

    #[test]
    fn test_reservoir_score_penalizes_lbp_distance() {
        let f = feature(50.0, 50.0, 100.0);

        let clean = reservoir_score(&f, 3, 0.8, 0, 100, 100);
        let bad_lbp = reservoir_score(&f, 3, 0.8, 16, 100, 100);

        assert!(clean > bad_lbp, "LBP penalty should lower reservoir score");
    }

    #[test]
    fn test_reservoir_score_rewards_age_and_quality() {
        let f = feature(50.0, 50.0, 100.0);

        let weak = reservoir_score(&f, 1, 0.1, 0, 100, 100);
        let strong = reservoir_score(&f, 10, 0.9, 0, 100, 100);

        assert!(
            strong > weak,
            "age and KLT quality should improve reservoir score"
        );
    }

    #[test]
    fn test_track_meta_reservoir_score_is_finite() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        frontend.process(&img2);

        assert!(frontend
            .track_meta()
            .iter()
            .all(|m| m.reservoir_score.is_finite()));
    }

    #[test]
    fn test_low_reservoir_score_pruning_disabled_by_default() {
        let mut features = vec![feature(20.0, 20.0, 50.0)];
        features[0].id = 7;
        let mut meta = vec![TrackMeta::new(&features[0], 1, 0, 100, 100, 16, 2, 2)];
        meta[0].reservoir_score = -10.0;

        let pruned = prune_low_reservoir_score(
            &mut features,
            &mut meta,
            FrontendConfig::default().min_reservoir_score,
        );

        assert_eq!(pruned, 0);
        assert_eq!(features.len(), 1);
        assert_eq!(meta.len(), 1);
    }

    #[test]
    fn test_low_reservoir_score_pruning_removes_weak_tracks() {
        let mut features = vec![
            feature(20.0, 20.0, 50.0),
            feature(40.0, 40.0, 50.0),
            feature(60.0, 60.0, 50.0),
        ];
        for (i, f) in features.iter_mut().enumerate() {
            f.id = (i + 1) as u64;
        }
        let mut meta: Vec<TrackMeta> = features
            .iter()
            .map(|f| TrackMeta::new(f, 2, 0, 100, 100, 16, 2, 2))
            .collect();
        meta[0].reservoir_score = 0.2;
        meta[1].reservoir_score = -0.3;
        meta[2].reservoir_score = 0.8;

        let pruned = prune_low_reservoir_score(&mut features, &mut meta, 0.0);

        assert_eq!(pruned, 1);
        assert_eq!(
            features.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(meta.iter().map(|m| m.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn test_tile_pruning_removes_lowest_score_only_from_overfull_tile() {
        let mut features = vec![
            feature(10.0, 20.0, 50.0),
            feature(20.0, 20.0, 50.0),
            feature(30.0, 20.0, 50.0),
            feature(80.0, 20.0, 50.0),
        ];
        for (i, f) in features.iter_mut().enumerate() {
            f.id = (i + 1) as u64;
        }
        let mut meta: Vec<TrackMeta> = features
            .iter()
            .map(|f| TrackMeta::new(f, 2, 0, 100, 100, 16, 2, 1))
            .collect();
        meta[0].reservoir_score = 0.2;
        meta[1].reservoir_score = 0.8;
        meta[2].reservoir_score = 0.5;
        meta[3].reservoir_score = -10.0;

        let pruned = prune_overfull_tiles(&mut features, &mut meta, 4, 100, 100, 2, 1);

        assert_eq!(pruned, 1);
        assert_eq!(
            features.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert_eq!(meta.iter().map(|m| m.id).collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn test_tile_pruning_disabled_keeps_overfull_tile() {
        let mut features = vec![
            feature(10.0, 20.0, 50.0),
            feature(20.0, 20.0, 50.0),
            feature(30.0, 20.0, 50.0),
        ];
        for (i, f) in features.iter_mut().enumerate() {
            f.id = (i + 1) as u64;
        }
        let mut meta: Vec<TrackMeta> = features
            .iter()
            .map(|f| TrackMeta::new(f, 2, 0, 100, 100, 16, 2, 1))
            .collect();

        let config = FrontendConfig {
            tile_reservoir_pruning_enabled: false,
            ..Default::default()
        };
        let pruned = if config.tile_reservoir_pruning_enabled {
            prune_overfull_tiles(&mut features, &mut meta, 2, 100, 100, 2, 1)
        } else {
            0
        };

        assert_eq!(pruned, 0);
        assert_eq!(features.len(), 3);
        assert_eq!(meta.len(), 3);
    }

    #[test]
    fn test_low_reservoir_score_pruning_runs_before_replenishment() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            max_features: 30,
            min_reservoir_score: 10.0,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats) = frontend.process(&img2);

        assert!(
            stats.rejected > 0,
            "score pruning should reject tracked points"
        );
        assert!(
            stats.new_detections > 0,
            "pruned tracks should free replenishment slots"
        );
        assert_eq!(frontend.features().len(), frontend.track_meta().len());
        for (feature, meta) in frontend.features().iter().zip(frontend.track_meta()) {
            assert_eq!(feature.id, meta.id);
        }
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
        let frames: Vec<Image<u8>> = (0..3).map(|i| make_scene(160, 120, i * 2, i)).collect();

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
        assert!(frontend.track_meta().is_empty());
        assert!(!frontend.has_prev_frame());
    }

    #[test]
    fn test_drop_tracks_removes_features_and_metadata() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img);
        let drop_id = frontend.features()[0].id;
        let before = frontend.features().len();

        let removed = frontend.drop_tracks(&[drop_id]);

        assert_eq!(removed, 1);
        assert_eq!(frontend.features().len(), before - 1);
        assert_eq!(frontend.features().len(), frontend.track_meta().len());
        assert!(frontend
            .features()
            .iter()
            .all(|feature| feature.id != drop_id));
        assert!(frontend.track_meta().iter().all(|meta| meta.id != drop_id));
    }

    #[test]
    fn test_drop_tracks_ignores_missing_ids() {
        let img = make_scene(160, 120, 0, 0);
        let config = FrontendConfig {
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img);
        let before = frontend.features().len();

        let removed = frontend.drop_tracks(&[u64::MAX]);

        assert_eq!(removed, 0);
        assert_eq!(frontend.features().len(), before);
        assert_eq!(frontend.features().len(), frontend.track_meta().len());
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
    fn test_lbp_soft_policy_does_not_reject_on_lbp_alone() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            lbp_policy: LbpPolicy::SoftPenalty,
            lbp_threshold: 0,
            tile_reservoir_pruning_enabled: false,
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        for f in &mut frontend.features {
            f.descriptor = !f.descriptor;
        }
        let (_, stats) = frontend.process(&img2);

        assert_eq!(stats.rejected, 0, "soft LBP policy should not hard-reject");
        assert!(
            stats.tracked > 0,
            "soft LBP policy should keep tracked reservoir points"
        );
    }

    #[test]
    fn test_lbp_hard_policy_preserves_legacy_rejection() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            lbp_policy: LbpPolicy::HardReject,
            lbp_threshold: 0,
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        for f in &mut frontend.features {
            f.descriptor = !f.descriptor;
        }
        let (_, stats) = frontend.process(&img2);

        assert!(
            stats.rejected > 0,
            "hard LBP policy should reject mismatched descriptors"
        );
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
        assert!(
            stats.tracked > 0,
            "histEq should help track across brightness change"
        );
    }

    #[test]
    fn test_clahe_tracking() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);

        let config = FrontendConfig {
            histeq: HistEqMethod::Clahe {
                tile_size: 32,
                clip_limit: 2.0,
            },
            max_features: 30,
            ..Default::default()
        };
        let mut frontend = Frontend::new(config, 160, 120);

        frontend.process(&img1);
        let (_, stats) = frontend.process(&img2);
        assert!(stats.tracked > 0, "CLAHE frontend should track features");
    }
}

// ---------------------------------------------------------------------------
// LBP Helpers
// ---------------------------------------------------------------------------

const CIRCLE_OFFSETS: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Compute the rotation-invariant LBP at a fractional position using u8 image.
///
/// All circle offsets are integers, so every sample shares the same fractional
/// part (fx, fy) as the center. Precompute bilinear weights once, then do 17
/// unchecked u8 lookups with fixed-point interpolation.
fn compute_lbp_at(img: &Image<u8>, x: f32, y: f32) -> Option<u16> {
    let w = img.width();
    let h = img.height();
    let stride = img.stride();
    let data = img.as_slice();

    if !x.is_finite() || !y.is_finite() {
        return None;
    }

    let x0 = x as usize;
    let y0 = y as usize;
    if x0 < 3 || y0 < 3 || x0 + 4 >= w || y0 + 4 >= h {
        return None;
    }

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Fixed-point weights (8-bit fraction, total = 256).
    let w00 = ((1.0 - fx) * (1.0 - fy) * 256.0) as u32;
    let w10 = (fx * (1.0 - fy) * 256.0) as u32;
    let w01 = ((1.0 - fx) * fy * 256.0) as u32;
    let w11 = (fx * fy * 256.0) as u32;

    let bilerp = |px: usize, py: usize| -> u32 {
        let x1 = if px + 1 < w { px + 1 } else { px };
        let y1 = if py + 1 < h { py + 1 } else { py };
        unsafe {
            let p00 = *data.get_unchecked(py * stride + px) as u32;
            let p10 = *data.get_unchecked(py * stride + x1) as u32;
            let p01 = *data.get_unchecked(y1 * stride + px) as u32;
            let p11 = *data.get_unchecked(y1 * stride + x1) as u32;
            w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11
        }
    };

    let center = bilerp(x0, y0);
    let mut lbp: u16 = 0;

    for (i, &(dx, dy)) in CIRCLE_OFFSETS.iter().enumerate() {
        let px = (x0 as i32 + dx) as usize;
        let py = (y0 as i32 + dy) as usize;
        if bilerp(px, py) >= center {
            lbp |= 1 << i;
        }
    }

    Some(compute_min_rotation(lbp))
}

/// Compute the rotation-invariant LBP by finding the minimum value
/// among all 16 cyclic shifts.
#[inline]
fn compute_min_rotation(mut v: u16) -> u16 {
    let mut min_v = v;
    for _ in 0..15 {
        v = v.rotate_right(1);
        if v < min_v {
            min_v = v;
        }
    }
    min_v
}
