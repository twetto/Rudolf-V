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
//   Image upload       GPU          u8 via write_texture, widened to R32Float on GPU.
//   Pyramid build      GPU          GpuPyramidPipeline (Gaussian + downsample)
//   KLT tracking       GPU          ┐
//   FAST detection     GPU          ├─ Fused: one encoder, one submit, one poll
//   NMS                GPU          ┘
//   LBP verification   CPU          RI-LBP on the u8 input, same rules as CPU
//   Reservoir pruning  CPU          TrackMeta scores, shared with frontend.rs
//   Occupancy grid     CPU          OccupancyGrid — byte mask, trivial
//   RANSAC             CPU          essential::estimate_essential_ransac — pure
//                                   linear algebra on O(N) tracked positions
//
// PYRAMID LIFETIME
// ─────────────────
// Like the CPU frontend, we double-buffer: two GpuPyramids are allocated once
// (GpuPyramidPipeline::allocate) and alternate between "prev" and "curr".
// Each frame rebuilds the curr slot in place (record_rebuild), so steady
// state creates no textures, buffers or bind groups — the KLT and FAST bind
// group caches only ever see these two pyramids. Reusing a slot is safe
// because every frame waits for its submission before returning.
//
// SUBMISSION
// ──────────
// Fused (default, one wait per frame): upload + pyramid + KLT + FAST + NMS
// are recorded into a single encoder. FAST always runs, even when the
// reservoir is full before KLT: KLT losses (and RANSAC rejections) would
// otherwise need a second blocking round-trip just for detection.
//
// Separate keeps the original flow: a fresh pyramid every frame (own
// submit), then KLT and FAST as independent submits. It does not use the
// persistent pyramids. On the Tegra Vulkan driver (wgpu 22), persistent
// pyramid textures consumed by a *later* submit than the one that built them
// intermittently read stale contents — output became nondeterministic.
// Fused never does that: each pyramid is built and first consumed in one
// command buffer. Keep that invariant if you restructure this.
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
use crate::essential::{self, BearingCorrespondence, RansacConfig};
use crate::fast::Feature;
use crate::frontend::{
    compute_lbp_at_lut, prune_low_reservoir_score, prune_overfull_tiles, select_by_tile_deficit,
    FrameStats, LbpPolicy, TimingStats, TrackMeta,
};
use crate::gpu::device::GpuDevice;
use crate::gpu::fast::{GpuFastDetector, NmsStrategy};
use crate::gpu::histeq::GpuGlobalHistEq;
use crate::gpu::klt::{GpuKltTracker, KltSampling};
use crate::gpu::pyramid::{GpuPyramid, GpuPyramidPipeline};
use crate::histeq::{self, HistEqMethod};
use crate::image::Image;
use crate::klt::TrackStatus;
use crate::occupancy::OccupancyGrid;
use camera_geometry::{CameraModel, CameraProjection, Pixel};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls how KLT and FAST are submitted to the GPU each frame.
///
/// `Fused` is the default and the fast, verified path. `Pipelined` is faster
/// still but only verified with a fixed GPU clock (see its docs). `Separate`
/// is the original pipeline, kept for comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubmitStrategy {
    /// Original flow: fresh pyramid per frame, KLT and FAST in separate
    /// submits with a CPU wait each.
    Separate,
    /// Pyramid + KLT + FAST recorded into one encoder: one submit, one wait
    /// per frame, persistent pyramids and bind groups (no per-frame
    /// allocation). Default.
    #[default]
    Fused,
    /// Like `Fused`, but FAST + NMS go in a second submit so the CPU runs LBP,
    /// RANSAC and reservoir pruning while the GPU is still detecting (collect
    /// waits for the first submit, then for the second only before
    /// replenishment). Same results as `Fused`.
    ///
    /// Caveat: FAST reads the pyramid in a later submit than the one that
    /// built it. On the Jetson Orin Nano (wgpu Vulkan) that pattern returned
    /// stale texels while the GPU clock governor was switching frequencies;
    /// it was deterministic with the clock fixed. Use only with a fixed GPU
    /// clock (e.g. min_freq = max_freq, or jetson_clocks).
    Pipelined,
}

/// How `collect()` waits for GPU work that has not finished yet.
///
/// Only matters when `collect()` is called before the GPU is done (e.g.
/// `process()`, or a caller with little work between `submit()` and
/// `collect()`); otherwise there is nothing to wait for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GpuWait {
    /// `SleepPoll { interval_us: 20 }` on NVIDIA Tegra (Jetson) adapters,
    /// `Block` elsewhere.
    #[default]
    Auto,
    /// The driver's blocking wait (`vkWaitSemaphores` via wgpu). On Jetson
    /// Orin Nano the NVIDIA driver busy-waits inside it, keeping a CPU core
    /// at 100% for the whole wait.
    Block,
    /// Check for completion without blocking (zero-timeout wait) and sleep
    /// `interval_us` between checks, leaving the core to other threads.
    /// Jetson Orin Nano, Global histeq, nothing between submit and collect:
    /// CPU time 1.26 → 0.56 ms/frame for +0.04–0.06 ms frame time
    /// (20–50 µs intervals).
    SleepPoll { interval_us: u32 },
}

impl GpuWait {
    /// The concrete strategy for this device (`Auto` resolved).
    fn resolve(self, gpu: &GpuDevice) -> GpuWait {
        match self {
            GpuWait::Auto if gpu.adapter_info.name.contains("Tegra") => {
                GpuWait::SleepPoll { interval_us: 20 }
            }
            GpuWait::Auto => GpuWait::Block,
            other => other,
        }
    }

    /// Wait until `submission` (or, if `None`, all submitted work) has
    /// completed and its map callbacks have run.
    fn wait(self, gpu: &GpuDevice, submission: Option<&wgpu::SubmissionIndex>) {
        match self {
            GpuWait::SleepPoll { interval_us } => {
                let interval = std::time::Duration::from_micros(interval_us as u64);
                loop {
                    match gpu.device.poll(wgpu::PollType::Wait {
                        submission_index: submission.cloned(),
                        timeout: Some(std::time::Duration::ZERO),
                    }) {
                        Ok(_) => return,
                        Err(wgpu::PollError::Timeout) => std::thread::sleep(interval),
                        Err(e) => panic!("GPU poll failed: {e}"),
                    }
                }
            }
            GpuWait::Block | GpuWait::Auto => {
                gpu.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: submission.cloned(),
                        timeout: None,
                    })
                    .expect("GPU poll failed");
            }
        }
    }
}

/// Configuration for the GPU visual frontend.
///
/// Mirrors `FrontendConfig` from frontend.rs. Fields that select between
/// CPU algorithm variants (LkMethod, DetectorType) are absent because the
/// GPU frontend always uses IC KLT and GPU FAST.
#[derive(Clone)]
pub struct GpuFrontendConfig {
    /// GPU submit strategy: one fused submit per frame, or the original
    /// separate submits. See [`SubmitStrategy`]. Defaults to `Fused`.
    pub submit_strategy: SubmitStrategy,
    /// NMS strategy: CPU readback (iGPU) or GPU NMS pass (discrete GPU).
    /// See [`NmsStrategy`] for guidance. Defaults to `Cpu`.
    pub nms_strategy: NmsStrategy,
    /// FAST corner detection threshold (pixel intensity difference).
    pub fast_threshold: u8,
    /// FAST arc length — minimum contiguous bright/dark pixels (9–12).
    pub fast_arc_length: usize,
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
    /// KLT patch half-size W. Patch is (2W+1)².
    pub klt_window: usize,
    /// KLT maximum Gauss-Newton iterations per pyramid level.
    pub klt_max_iter: usize,
    /// KLT convergence threshold in pixels.
    pub klt_epsilon: f32,
    /// How KLT interpolates the pyramid: manual 4-tap bilinear (`Manual`,
    /// default) or hardware texture filtering (`Hardware` / `Auto`, ~3% faster
    /// frontend, slightly more KLT outliers; only verified with `Fused`).
    /// See [`KltSampling`].
    pub klt_sampling: KltSampling,
    /// LBP descriptor verification (occlusion/drift detection), as in the
    /// CPU frontend. Computed on the CPU from the (preprocessed) input image.
    pub lbp_verification_enabled: bool,
    /// Whether high LBP distance is metadata only or a hard reservoir reject.
    pub lbp_policy: LbpPolicy,
    /// Hamming distance threshold for `LbpPolicy::HardReject` (max bits).
    pub lbp_threshold: u32,
    /// Reject tracked reservoir points whose soft score falls below this
    /// value. Negative infinity (default) disables score-based pruning.
    pub min_reservoir_score: f32,
    /// Prune only over-target coarse tiles by local reservoir score.
    pub tile_reservoir_pruning_enabled: bool,
    /// Run `HistEqMethod::Global` on the GPU (histogram + LUT in the frame's
    /// submit; bit-identical to the CPU equalization). Fused strategy only;
    /// CLAHE and the Separate strategy always equalize on the CPU.
    pub gpu_histeq: bool,
    /// How `collect()` waits for unfinished GPU work. See [`GpuWait`].
    pub gpu_wait: GpuWait,
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
            submit_strategy: SubmitStrategy::Fused,
            nms_strategy: NmsStrategy::Cpu,
            fast_threshold: 20,
            fast_arc_length: 9,
            max_features: 200,
            cell_size: 16,
            coarse_tile_cols: 8,
            coarse_tile_rows: 6,
            pyramid_levels: 3,
            pyramid_sigma: 1.0,
            klt_window: 7,
            klt_max_iter: 30,
            klt_epsilon: 0.01,
            klt_sampling: KltSampling::Manual,
            // Reservoir policy defaults match FrontendConfig::default().
            lbp_verification_enabled: true,
            lbp_policy: LbpPolicy::SoftPenalty,
            lbp_threshold: 4,
            min_reservoir_score: f32::NEG_INFINITY,
            tile_reservoir_pruning_enabled: true,
            gpu_histeq: true,
            gpu_wait: GpuWait::Auto,
            histeq: HistEqMethod::None,
            camera: None,
            ransac: RansacConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuFrontend
// ---------------------------------------------------------------------------

/// State of a frame between `submit()` and `collect()`.
struct PendingFrame {
    /// histeq + pyramid timings measured in submit().
    timing: TimingStats,
    submit_secs: f64,
    fused: bool,
    /// Pipelined strategy: KLT (+ histeq, pyramid) were submitted first.
    pipelined: bool,
    first_submit: Option<wgpu::SubmissionIndex>,
    /// KLT was run (fused: recorded) for this frame.
    tracking: bool,
    /// Features and metadata as they were at submit(), index-aligned with the
    /// KLT results.
    feats_snap: Vec<Feature>,
    meta_snap: Vec<TrackMeta>,
    /// Separate strategy only: this frame's fresh pyramid.
    separate_pyr: Option<GpuPyramid>,
    /// IDs dropped via drop_tracks() while in flight.
    dropped: Vec<u64>,
    /// Set by reset(); drained by the next submit().
    discarded: bool,
    /// Global histogram equalization ran on the GPU; its LUT is being read back.
    gpu_lut: bool,
}

/// GPU visual frontend.
///
/// Create once with `GpuFrontend::new()`; call `process()` every frame, or
/// split it into `submit()` + `collect()` to overlap the GPU work with other
/// CPU work. GPU pipelines are compiled at construction time.
///
/// # Example
/// ```text
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
///
/// // Overlapped: run the backend on frame N while the GPU works on N+1.
/// frontend.submit(&gpu, &frame_n1);
/// backend.update(&features_n);          // may call frontend.drop_tracks(..)
/// let (features_n1, _) = frontend.collect(&gpu);
/// ```
pub struct GpuFrontend {
    config: GpuFrontendConfig,
    camera_projection: Option<CameraProjection>,

    // GPU pipelines (compiled once).
    pyr_pipeline: GpuPyramidPipeline,
    fast: GpuFastDetector,
    klt: GpuKltTracker,

    // CPU post-processing.
    grid: OccupancyGrid,

    // Per-frame state. `pyramids[curr_slot]` is rebuilt each frame; the
    // other slot holds the previous frame's pyramid.
    pyramids: [GpuPyramid; 2],
    curr_slot: usize,
    // `SubmitStrategy::Separate` keeps the original flow instead: a fresh
    // pyramid per frame, with the previous one held here. See SUBMISSION.
    separate_prev: Option<GpuPyramid>,
    features: Vec<Feature>,
    // Index-aligned with `features` (same semantics as the CPU frontend).
    track_meta: Vec<TrackMeta>,
    prev_features: Vec<Feature>,
    next_id: u64,
    // This frame's (preprocessed) input, kept from submit() to collect() for
    // LBP verification and new-feature descriptors.
    input_buf: Image<u8>,
    // GPU global histogram equalization; its LUT buffer is bound to both
    // pyramids' convert pass. Holds the identity when not equalizing on GPU.
    histeq_gpu: GpuGlobalHistEq,
    lut_is_identity: bool,
    // LUT of the frame being collected (GPU Global equalization), else None.
    frame_lut: Option<[u8; 256]>,
    // Track ID → bearing unprojected at the track's current position (RANSAC
    // b2); reused as next frame's b1.
    bearing_cache: std::collections::HashMap<u64, [f64; 3]>,
    has_prev: bool,
    // Frame between submit() and collect().
    pending: Option<PendingFrame>,
    // `config.gpu_wait` resolved for this device.
    wait: GpuWait,

    img_w: usize,
    img_h: usize,
}

impl GpuFrontend {
    /// Create a new GPU frontend for images of the given dimensions.
    ///
    /// This compiles three compute shaders (pyramid, FAST, KLT). Call once
    /// at startup, not every frame.
    pub fn new(gpu: &GpuDevice, config: GpuFrontendConfig, img_w: usize, img_h: usize) -> Self {
        let camera_projection = config.camera.as_ref().map(CameraIntrinsics::projection);
        let config_wait = config.gpu_wait.resolve(gpu);
        let pyr_pipeline = GpuPyramidPipeline::new(gpu);
        let fast = GpuFastDetector::new(
            gpu,
            config.fast_threshold,
            config.fast_arc_length,
            img_w,
            img_h,
            config.cell_size,
            config.nms_strategy,
        );
        let klt = GpuKltTracker::new_with_sampling(
            gpu,
            config.klt_window,
            config.klt_max_iter,
            config.klt_epsilon,
            config.pyramid_levels,
            config.max_features,
            config.klt_sampling,
        );
        let grid = OccupancyGrid::new(img_w, img_h, config.cell_size);
        let histeq_gpu = GpuGlobalHistEq::new(gpu, img_w, img_h);
        let pyramids = [
            pyr_pipeline.allocate_with_lut(
                gpu,
                img_w,
                img_h,
                config.pyramid_levels,
                histeq_gpu.lut_buffer(),
            ),
            pyr_pipeline.allocate_with_lut(
                gpu,
                img_w,
                img_h,
                config.pyramid_levels,
                histeq_gpu.lut_buffer(),
            ),
        ];

        GpuFrontend {
            config,
            camera_projection,
            pyr_pipeline,
            fast,
            klt,
            grid,
            pyramids,
            curr_slot: 0,
            separate_prev: None,
            input_buf: Image::new(img_w, img_h),
            histeq_gpu,
            lut_is_identity: true,
            frame_lut: None,
            bearing_cache: std::collections::HashMap::new(),
            features: Vec::new(),
            track_meta: Vec::new(),
            pending: None,
            wait: config_wait,
            prev_features: Vec::new(),
            next_id: 1,
            has_prev: false,
            img_w,
            img_h,
        }
    }

    /// Process one frame (blocking). Returns the tracked feature list and
    /// statistics. Equivalent to [`submit`] followed by [`collect`].
    ///
    /// The returned `&[Feature]` slice is valid until the next `process()` /
    /// `collect()` call. Feature IDs are persistent: a feature keeps its ID as
    /// long as it tracks successfully.
    ///
    /// [`submit`]: GpuFrontend::submit
    /// [`collect`]: GpuFrontend::collect
    pub fn process<'a>(
        &'a mut self,
        gpu: &GpuDevice,
        image: &Image<u8>,
    ) -> (&'a [Feature], FrameStats) {
        self.submit(gpu, image);
        self.collect(gpu)
    }

    /// Start processing a frame without waiting for the GPU.
    ///
    /// Copies (or histogram-equalizes) `image` into an internal buffer, then —
    /// with `SubmitStrategy::Fused` — records pyramid + KLT + FAST + NMS and
    /// submits them. Returns as soon as the work is queued, so the calling
    /// thread can do other work (e.g. the VIO backend on the previous frame's
    /// features) while the GPU runs. Finish the frame with [`collect`].
    ///
    /// Tracks the features as they are *now*. [`drop_tracks`] may be called
    /// between `submit` and `collect`; dropped IDs are removed from this
    /// frame's results too.
    ///
    /// With `SubmitStrategy::Separate` only the pyramid is submitted here; KLT
    /// and FAST run (blocking) inside `collect`.
    ///
    /// # Panics
    /// If the previous submitted frame has not been collected.
    ///
    /// [`collect`]: GpuFrontend::collect
    /// [`drop_tracks`]: GpuFrontend::drop_tracks
    pub fn submit(&mut self, gpu: &GpuDevice, image: &Image<u8>) {
        if let Some(p) = &self.pending {
            assert!(
                p.discarded,
                "GpuFrontend::submit: previous frame was not collected"
            );
            self.drain_discarded(gpu);
        }
        assert_eq!(image.width(), self.img_w, "image width mismatch");
        assert_eq!(image.height(), self.img_h, "image height mismatch");

        let t_start = Instant::now();
        let mut timing = TimingStats::default();

        // ── Step 0: Histogram equalization / input copy (CPU) ────────────────
        // `input_buf` holds this frame's image until collect(): LBP
        // verification and new-feature descriptors read it there. It is the
        // CPU-equalized image, or — when Global equalization runs on the GPU —
        // the raw image, with the GPU's LUT applied at the sampled pixels.
        let t0 = Instant::now();
        let fused = matches!(
            self.config.submit_strategy,
            SubmitStrategy::Fused | SubmitStrategy::Pipelined
        );
        let pipelined = self.config.submit_strategy == SubmitStrategy::Pipelined;
        let mut first_submit = None;
        let gpu_lut = fused && self.config.gpu_histeq && self.config.histeq == HistEqMethod::Global;
        if self.config.histeq != HistEqMethod::None && !gpu_lut {
            histeq::apply_histeq_into(image, self.config.histeq, &mut self.input_buf);
        } else if self.input_buf.stride() == image.stride() {
            self.input_buf
                .as_mut_slice()
                .copy_from_slice(image.as_slice());
        } else {
            self.input_buf = image.clone();
        }
        timing.histeq = t0.elapsed().as_secs_f64();

        // ── Step 1: GPU pyramid (+ KLT + FAST when fused) ────────────────────
        // Fused: rebuild the persistent curr slot and record KLT + FAST in the
        //   same encoder; one submit, no wait.
        // Separate: build a fresh pyramid with its own submit (original path);
        //   KLT and FAST run in collect().
        let t0 = Instant::now();
        let curr = self.curr_slot;
        let prev = 1 - curr;
        let can_track = self.has_prev && !self.features.is_empty();
        let feats_snap: Vec<Feature> = if can_track {
            self.features.clone()
        } else {
            Vec::new()
        };
        let meta_snap: Vec<TrackMeta> = if can_track {
            self.track_meta.clone()
        } else {
            Vec::new()
        };
        let mut tracking = false;
        let mut separate_pyr = None;

        if fused {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GpuFrontend frame"),
                });
            self.pyr_pipeline
                .upload(gpu, &self.input_buf, &self.pyramids[curr]);
            if !gpu_lut && !self.lut_is_identity {
                // Equalization switched off or moved to the CPU at runtime.
                self.histeq_gpu.write_identity(gpu);
                self.lut_is_identity = true;
            }
            tracking = can_track
                && self
                    .klt
                    .prepare(gpu, &feats_snap, &self.pyramids[prev], &self.pyramids[curr]);
            // FAST joins the same pass when it uses the fused GPU-NMS kernel
            // and is not deferred to a second submit (Pipelined).
            let fast_in_pass = !pipelined && self.fast.uses_gpu_nms();

            // One compute pass for the whole frame: histogram → LUT → pyramid
            // → KLT levels → FAST+NMS. Every dispatch is its own
            // synchronization scope, so wgpu orders the dependencies; one pass
            // instead of five saves ~17 µs of CPU recording per pass.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GpuFrontend frame"),
                    timestamp_writes: None,
                });
                if gpu_lut {
                    self.histeq_gpu
                        .record_dispatches(gpu, &mut pass, &self.pyramids[curr]);
                    self.lut_is_identity = false;
                }
                self.pyr_pipeline
                    .record_build_in_pass(gpu, &mut pass, &self.pyramids[curr]);
                if tracking {
                    self.klt.record_dispatches(&mut pass);
                }
                if fast_in_pass {
                    self.fast
                        .record_dispatch(gpu, &mut pass, &self.pyramids[curr].levels[0]);
                }
            }
            if gpu_lut {
                self.histeq_gpu.record_readback(&mut encoder);
            }
            if tracking {
                self.klt.record_readback(&mut encoder);
            }
            if fast_in_pass {
                self.fast.record_readback(&mut encoder);
            }
            if pipelined {
                // First submit: histeq + pyramid + KLT. collect() only waits
                // for this before LBP / RANSAC / pruning.
                first_submit = Some(gpu.queue.submit(std::iter::once(encoder.finish())));
                if tracking {
                    self.klt.arm_readback();
                }
                if gpu_lut {
                    self.histeq_gpu.arm_readback();
                }
                encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("GpuFrontend frame (FAST)"),
                    });
            }
            if !fast_in_pass {
                // Pipelined (second submit) or the dense CPU-NMS path.
                self.fast
                    .record_into(gpu, &mut encoder, &self.pyramids[curr].levels[0]);
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));
            if !pipelined {
                if tracking {
                    self.klt.arm_readback();
                }
                if gpu_lut {
                    self.histeq_gpu.arm_readback();
                }
            }
            self.fast.arm_readback();
        } else {
            separate_pyr = Some(self.pyr_pipeline.build(
                gpu,
                &self.input_buf,
                self.config.pyramid_levels,
                self.config.pyramid_sigma,
            ));
            tracking = can_track && self.separate_prev.is_some();
        }
        timing.pyramid = t0.elapsed().as_secs_f64();

        self.pending = Some(PendingFrame {
            timing,
            submit_secs: t_start.elapsed().as_secs_f64(),
            fused,
            pipelined,
            first_submit,
            tracking,
            feats_snap,
            meta_snap,
            separate_pyr,
            dropped: Vec::new(),
            discarded: false,
            gpu_lut,
        });
    }

    /// Non-blocking check: `true` once the GPU work of the submitted frame has
    /// finished, i.e. [`collect`] will not wait for the GPU.
    ///
    /// [`collect`]: GpuFrontend::collect
    pub fn poll_ready(&self, gpu: &GpuDevice) -> bool {
        gpu.device
            .poll(wgpu::PollType::Poll)
            .is_ok_and(|s| s.is_queue_empty())
    }

    /// Whether a frame has been submitted and not yet collected.
    pub fn is_pending(&self) -> bool {
        self.pending.as_ref().is_some_and(|p| !p.discarded)
    }

    /// Finish the frame started by [`submit`]: wait for the GPU (if it is still
    /// running), then LBP verification, RANSAC, reservoir pruning and
    /// replenishment on the CPU. Returns the tracked feature list and stats.
    ///
    /// `stats.timing.total` counts only time spent inside `submit` and
    /// `collect`, not the caller's work in between.
    ///
    /// If the GPU is still busy, `collect` waits according to
    /// [`GpuFrontendConfig::gpu_wait`] (by default sleep-polling on Jetson,
    /// where the driver's blocking wait spins a core). Work done between
    /// `submit` and `collect` overlaps with the GPU; [`poll_ready`] checks
    /// without waiting. (`SubmitStrategy::Separate` keeps the original blocking
    /// waits inside its stages.)
    ///
    /// # Panics
    /// If no frame was submitted.
    ///
    /// [`submit`]: GpuFrontend::submit
    /// [`poll_ready`]: GpuFrontend::poll_ready
    pub fn collect<'a>(&'a mut self, gpu: &GpuDevice) -> (&'a [Feature], FrameStats) {
        let p = self
            .pending
            .take()
            .filter(|p| !p.discarded)
            .expect("GpuFrontend::collect: no frame submitted");
        let t_collect = Instant::now();
        let mut timing = p.timing.clone();

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

        // ── Step 2: KLT results (+ LBP verification, CPU) ────────────────────
        let t0 = Instant::now();
        let (results, fused_winners) = if p.fused {
            self.wait.wait(gpu, p.first_submit.as_ref());
            if p.gpu_lut {
                self.frame_lut = Some(self.histeq_gpu.collect_lut());
            } else {
                self.frame_lut = None;
            }
            let results = if p.tracking {
                self.klt.collect_results(&p.feats_snap)
            } else {
                Vec::new()
            };
            // Pipelined: FAST is still running; collected before replenishment.
            let winners = if p.pipelined {
                None
            } else {
                Some(self.fast.collect_winners(0))
            };
            (results, winners)
        } else {
            let curr_pyr = p
                .separate_pyr
                .as_ref()
                .expect("separate pyramid built in submit");
            let results = match (p.tracking, self.separate_prev.as_ref()) {
                (true, Some(prev_pyr)) => self.klt.track(gpu, prev_pyr, curr_pyr, &p.feats_snap),
                _ => Vec::new(),
            };
            (results, None)
        };

        if p.tracking {
            // Same rules as Frontend::process (CPU): LBP descriptor distance is
            // track metadata under SoftPenalty and a hard reject under
            // HardReject; a track whose LBP cannot be computed (too close to
            // the border) is rejected under both.
            let mut features = Vec::with_capacity(results.len());
            let mut track_meta = Vec::with_capacity(results.len());
            for (result, meta) in results.iter().zip(&p.meta_snap) {
                if p.dropped.contains(&result.feature.id) {
                    continue; // dropped by the caller while the frame was in flight
                }
                if result.status != TrackStatus::Tracked {
                    stats.lost += 1;
                    continue;
                }
                let feat = &result.feature;
                let mut lbp_distance = 0u16;
                if self.config.lbp_verification_enabled {
                    let Some(new_desc) = compute_lbp_at_lut(
                        &self.input_buf,
                        self.frame_lut.as_ref(),
                        feat.x,
                        feat.y,
                    ) else {
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
                // The GPU tracker computes no residual; this matches the CPU
                // frontend with klt_residual_enabled = false (its default).
                track_meta.push(meta.advanced(
                    feat,
                    1.0,
                    lbp_distance,
                    self.img_w,
                    self.img_h,
                    self.config.cell_size,
                    self.config.coarse_tile_cols,
                    self.config.coarse_tile_rows,
                ));
                features.push(feat.clone());
                stats.tracked += 1;
            }
            self.features = features;
            self.track_meta = track_meta;
        }
        timing.klt = t0.elapsed().as_secs_f64();

        // ── Step 2b: Geometric verification (RANSAC, CPU) ────────────────────
        let t0 = Instant::now();
        // Bearings unprojected this frame, by track ID. A track's position at
        // the end of this frame is its "previous" position next frame, so its
        // b1 then is exactly this b2 (same pixel, same deterministic
        // unprojection). Rebuilt every frame; empty if RANSAC is skipped.
        let prev_bearings = std::mem::take(&mut self.bearing_cache);
        if let Some(ref cam) = self.camera_projection {
            if self.features.len() >= 8 {
                let prev_by_id: std::collections::HashMap<u64, &Feature> =
                    self.prev_features.iter().map(|pf| (pf.id, pf)).collect();
                let mut corrs: Vec<(usize, BearingCorrespondence)> =
                    Vec::with_capacity(self.features.len());
                for (idx, f) in self.features.iter().enumerate() {
                    let Some(pf) = prev_by_id.get(&f.id) else {
                        continue;
                    };
                    let b1 = match prev_bearings.get(&f.id) {
                        Some(b) => *b,
                        None => match cam.unproject(Pixel::new(pf.x as f64, pf.y as f64)) {
                            Some(b) => {
                                let v = b.vector();
                                [v.x, v.y, v.z]
                            }
                            None => continue,
                        },
                    };
                    let Some(b2) = cam.unproject(Pixel::new(f.x as f64, f.y as f64)) else {
                        continue;
                    };
                    let b2 = b2.vector();
                    let b2 = [b2.x, b2.y, b2.z];
                    self.bearing_cache.insert(f.id, b2);
                    corrs.push((idx, BearingCorrespondence { b1, b2 }));
                }

                if corrs.len() >= 8 {
                    let corr_only: Vec<BearingCorrespondence> =
                        corrs.iter().map(|(_, c)| *c).collect();

                    if let Some(result) = essential::estimate_essential_ransac_bearings(
                        &corr_only,
                        &self.config.ransac,
                    ) {
                        let mut inliers = Vec::new();
                        let mut inlier_meta = Vec::new();
                        let mut ransac_rejected = 0usize;
                        for (ci, (feat_idx, _)) in corrs.iter().enumerate() {
                            if result.inliers[ci] {
                                inliers.push(self.features[*feat_idx].clone());
                                inlier_meta.push(self.track_meta[*feat_idx].clone());
                            } else {
                                ransac_rejected += 1;
                            }
                        }
                        let matched_ids: Vec<u64> = corrs
                            .iter()
                            .map(|(idx, _)| self.features[*idx].id)
                            .collect();
                        for (idx, f) in self.features.iter().enumerate() {
                            if !matched_ids.contains(&f.id) {
                                inliers.push(f.clone());
                                inlier_meta.push(self.track_meta[idx].clone());
                            }
                        }
                        stats.rejected += ransac_rejected;
                        stats.tracked = stats.tracked.saturating_sub(ransac_rejected);
                        self.features = inliers;
                        self.track_meta = inlier_meta;
                    }
                }
            }
        }
        timing.ransac = t0.elapsed().as_secs_f64();

        // ── Step 2c: Reservoir pruning (same as the CPU frontend) ────────────
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

        // ── Step 3: Update occupancy grid from surviving tracked features ─────
        self.grid.clear();
        for f in &self.features {
            self.grid.mark(f.x, f.y);
        }

        // ── Step 4: Detect + replenish ────────────────────────────────────────
        // slots computed after KLT/RANSAC/pruning — reflects true feature gap.
        // Fused: winners were computed on the GPU in the same submit as KLT.
        // Separate: run FAST now as its own submit.
        let t0 = Instant::now();
        let slots = self.config.max_features.saturating_sub(self.features.len());

        let winners = if let Some(w) = fused_winners {
            w
        } else if p.pipelined {
            self.wait.wait(gpu, None);
            self.fast.collect_winners(0)
        } else if slots > 0 {
            let curr_pyr = p
                .separate_pyr
                .as_ref()
                .expect("separate pyramid built in submit");
            self.fast.detect(gpu, &curr_pyr.levels[0], 0)
        } else {
            Vec::new()
        };

        if slots > 0 {
            // Look winners up in the occupancy grid directly — same result as
            // `unoccupied_mask()`, without filling a full-resolution mask image
            // (that alone cost ~0.7 ms/frame on Jetson Orin Nano).
            let cells = self.grid.grid_cells();
            let cols = self.grid.grid_cols();
            let cs = self.grid.cell_size();
            let unoccupied: Vec<Feature> = winners
                .iter()
                .filter_map(|f| {
                    let x = f.x as usize;
                    let y = f.y as usize;
                    if x < self.img_w && y < self.img_h && !cells[(y / cs) * cols + x / cs] {
                        Some(Feature {
                            x: f.x,
                            y: f.y,
                            score: f.score,
                            level: f.level,
                            id: 0,
                            descriptor: 0,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            let selected = select_by_tile_deficit(
                &unoccupied,
                &self.features,
                slots,
                self.config.max_features,
                self.img_w,
                self.img_h,
                self.config.coarse_tile_cols,
                self.config.coarse_tile_rows,
            );
            for f in selected.iter().take(slots) {
                // GPU FAST provides no descriptor; compute the RI-LBP on the
                // (preprocessed) input exactly as the CPU frontend does for
                // detectors without one.
                let new_feat = Feature {
                    x: f.x,
                    y: f.y,
                    score: f.score,
                    level: f.level,
                    id: self.next_id,
                    descriptor: compute_lbp_at_lut(
                        &self.input_buf,
                        self.frame_lut.as_ref(),
                        f.x,
                        f.y,
                    )
                    .unwrap_or(0),
                };
                self.next_id += 1;
                self.track_meta.push(TrackMeta::new(
                    &new_feat,
                    1,
                    0,
                    self.img_w,
                    self.img_h,
                    self.config.cell_size,
                    self.config.coarse_tile_cols,
                    self.config.coarse_tile_rows,
                ));
                self.grid.mark(new_feat.x, new_feat.y);
                self.features.push(new_feat);
                stats.new_detections += 1;
            }
        }
        timing.detect = t0.elapsed().as_secs_f64();

        // ── Step 5: Advance state ─────────────────────────────────────────────
        if p.fused {
            self.curr_slot = 1 - self.curr_slot;
        } else {
            self.separate_prev = p.separate_pyr;
        }
        self.prev_features = self.features.clone();
        self.has_prev = true;

        timing.total = p.submit_secs + t_collect.elapsed().as_secs_f64();
        stats.total = self.features.len();
        stats.occupied_cells = self.grid.total_cells() - self.grid.count_empty();
        stats.timing = timing;

        (&self.features, stats)
    }

    /// Finish a frame that was discarded by `reset()` while in flight: wait
    /// for its GPU work and unmap its readback buffers, so the next frame can
    /// reuse them.
    fn drain_discarded(&mut self, gpu: &GpuDevice) {
        let Some(p) = self.pending.take() else { return };
        debug_assert!(p.discarded);
        if p.fused {
            self.wait.wait(gpu, None);
            if p.tracking {
                let _ = self.klt.collect_results(&p.feats_snap);
            }
            let _ = self.fast.collect_winners(0);
            if p.gpu_lut {
                let _ = self.histeq_gpu.collect_lut();
            }
        }
    }

    /// Currently tracked features (without processing a new frame).
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Metadata for the current feature list, index-aligned with
    /// [`features`](GpuFrontend::features). Same semantics as
    /// [`Frontend::track_meta`](crate::frontend::Frontend::track_meta); the
    /// GPU tracker has no residual, so `klt_quality` is always 1.0.
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
    /// May be called between [`submit`](GpuFrontend::submit) and
    /// [`collect`](GpuFrontend::collect): the IDs are then also removed from
    /// the in-flight frame's results.
    ///
    /// Returns the number of active tracks removed.
    pub fn drop_tracks(&mut self, ids: &[u64]) -> usize {
        if ids.is_empty() {
            return 0;
        }
        if let Some(p) = self.pending.as_mut().filter(|p| !p.discarded) {
            p.dropped.extend_from_slice(ids);
        }
        if self.features.is_empty() {
            return 0;
        }

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

    /// Whether at least one frame has been processed.
    pub fn has_prev_frame(&self) -> bool {
        self.has_prev
    }

    /// Reset frontend state (features, previous frame, occupancy grid).
    /// Does NOT reset next_id — feature IDs remain globally unique.
    /// Does NOT recompile shaders.
    ///
    /// A frame submitted but not yet collected is discarded (its GPU work is
    /// drained on the next `submit`).
    pub fn reset(&mut self) {
        if let Some(p) = self.pending.as_mut() {
            p.discarded = true;
        }
        self.has_prev = false;
        self.features.clear();
        self.track_meta.clear();
        self.prev_features.clear();
        self.bearing_cache.clear();
        self.separate_prev = None;
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
            .args([
                "test",
                "--lib",
                "--",
                name,
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed: {e}"));
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr)
    }

    fn make_scene(w: usize, h: usize, shift_x: usize, shift_y: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![25u8; w * h]);
        for &(rx, ry, rw, rh, val) in &[
            (30usize, 25usize, 20usize, 20usize, 200u8),
            (70, 20, 25, 15, 180),
            (110, 30, 18, 22, 210),
            (25, 65, 22, 25, 190),
            (75, 60, 30, 20, 170),
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
        let config = GpuFrontendConfig {
            max_features: 50,
            ..Default::default()
        };
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
        let config = GpuFrontendConfig {
            max_features: 50,
            ..Default::default()
        };
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
        let config = GpuFrontendConfig {
            max_features: 50,
            ..Default::default()
        };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let n1 = fe.process(&gpu, &img1).0.len();
        assert!(n1 > 0);

        let (_, stats) = fe.process(&gpu, &img2);
        assert!(stats.tracked > 0, "should track some features: {stats:?}");
        println!("GPU_TEST_OK");
        drop(fe);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_ids_persist() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 2, 1);
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig {
            max_features: 50,
            ..Default::default()
        };
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
        let config = GpuFrontendConfig {
            max_features: max,
            ..Default::default()
        };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        let (features, _) = fe.process(&gpu, &img);
        assert!(
            features.len() <= max,
            "features {} > max {max}",
            features.len()
        );
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_replenishment_after_loss() {
        let img1 = make_scene(160, 120, 0, 0);
        let img2 = make_scene(160, 120, 15, 10); // large shift → many lost
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig {
            max_features: 50,
            ..Default::default()
        };
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
        let config = GpuFrontendConfig {
            max_features: 30,
            ..Default::default()
        };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);

        for i in 0..3usize {
            let img = make_scene(160, 120, i * 2, i);
            let (features, stats) = fe.process(&gpu, &img);
            eprintln!(
                "frame {i}: tracked={} lost={} new={} total={}",
                stats.tracked, stats.lost, stats.new_detections, stats.total
            );
            if i > 0 {
                assert!(stats.tracked > 0, "frame {i}: should track some features");
            }
            assert!(!features.is_empty());
        }
        println!("GPU_TEST_OK");
    }

    /// Scene with enough texture that LBP descriptors are well defined.
    fn textured_scene(shift_x: usize, shift_y: usize) -> Image<u8> {
        let (w, h) = (160usize, 120usize);
        let mut img = make_scene(w, h, shift_x, shift_y);
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x.wrapping_sub(shift_x), y.wrapping_sub(shift_y));
                let v =
                    img.get(x, y) as u32 + ((sx.wrapping_mul(7) ^ sy.wrapping_mul(13)) % 23) as u32;
                img.set(x, y, v.min(255) as u8);
            }
        }
        img
    }

    fn snapshot(fe: &GpuFrontend) -> Vec<(u64, u32, u32, u16, u16)> {
        fe.features()
            .iter()
            .zip(fe.track_meta())
            .map(|(f, m)| (f.id, f.x.to_bits(), f.y.to_bits(), f.descriptor, m.age))
            .collect()
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_split_matches_process() {
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig {
            max_features: 40,
            ..Default::default()
        };
        let mut a = GpuFrontend::new(&gpu, config.clone(), 160, 120);
        let mut b = GpuFrontend::new(&gpu, config, 160, 120);
        for i in 0..6usize {
            let img = textured_scene(i * 2, i);
            a.process(&gpu, &img);
            b.submit(&gpu, &img);
            assert!(b.is_pending());
            b.collect(&gpu);
            assert!(!b.is_pending());
            assert_eq!(
                snapshot(&a),
                snapshot(&b),
                "frame {i}: split differs from process"
            );
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_track_meta_aligned() {
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig {
            max_features: 40,
            ..Default::default()
        };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);
        for i in 0..5usize {
            fe.process(&gpu, &textured_scene(i * 2, i));
            assert_eq!(fe.features().len(), fe.track_meta().len());
            for (f, m) in fe.features().iter().zip(fe.track_meta()) {
                assert_eq!(f.id, m.id, "meta not index-aligned");
                assert!(m.age >= 1);
            }
        }
        // Tracks surviving 5 frames have aged.
        assert!(fe.track_meta().iter().any(|m| m.age > 1), "no track aged");
        // New GPU-detected features get an RI-LBP descriptor on the CPU.
        assert!(
            fe.features().iter().any(|f| f.descriptor != 0),
            "no descriptors computed"
        );
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_drop_tracks_in_flight() {
        let gpu = GpuDevice::new().unwrap();
        let config = GpuFrontendConfig {
            max_features: 40,
            ..Default::default()
        };
        let mut fe = GpuFrontend::new(&gpu, config, 160, 120);
        fe.process(&gpu, &textured_scene(0, 0));
        let ids: Vec<u64> = fe.features().iter().take(3).map(|f| f.id).collect();
        assert_eq!(ids.len(), 3);

        fe.submit(&gpu, &textured_scene(2, 1));
        assert_eq!(fe.drop_tracks(&ids), 3);
        let (features, _) = fe.collect(&gpu);
        assert!(
            features.iter().all(|f| !ids.contains(&f.id)),
            "dropped IDs came back from the in-flight frame"
        );
        assert_eq!(fe.features().len(), fe.track_meta().len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_reset_while_pending() {
        let gpu = GpuDevice::new().unwrap();
        let mut fe = GpuFrontend::new(&gpu, GpuFrontendConfig::default(), 160, 120);
        fe.process(&gpu, &textured_scene(0, 0));
        fe.submit(&gpu, &textured_scene(2, 1));
        fe.reset();
        assert!(!fe.is_pending());
        // The discarded frame is drained; processing continues normally.
        let (features, stats) = fe.process(&gpu, &textured_scene(0, 0));
        assert!(!features.is_empty());
        assert_eq!(stats.tracked, 0, "first frame after reset tracks nothing");
        let (_, stats) = fe.process(&gpu, &textured_scene(2, 1));
        assert!(stats.tracked > 0);
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_lbp_policies() {
        // Mirrors frontend.rs test_lbp_{soft,hard}_policy_*: corrupt every
        // stored descriptor, then re-process the same image.
        let gpu = GpuDevice::new().unwrap();
        let img = textured_scene(0, 0);
        for policy in [LbpPolicy::SoftPenalty, LbpPolicy::HardReject] {
            let config = GpuFrontendConfig {
                max_features: 40,
                lbp_policy: policy,
                lbp_threshold: 0,
                tile_reservoir_pruning_enabled: false,
                ..Default::default()
            };
            let mut fe = GpuFrontend::new(&gpu, config, 160, 120);
            fe.process(&gpu, &img);
            for f in &mut fe.features {
                f.descriptor = !f.descriptor;
            }
            let (_, stats) = fe.process(&gpu, &img);
            match policy {
                LbpPolicy::SoftPenalty => {
                    assert!(
                        stats.tracked > 0,
                        "soft policy must not reject on LBP alone"
                    );
                    assert!(fe.track_meta().iter().any(|m| m.lbp_distance > 0));
                }
                LbpPolicy::HardReject => {
                    assert_eq!(
                        stats.tracked, 0,
                        "hard policy should reject mismatched descriptors"
                    );
                    assert!(stats.rejected > 0);
                }
            }
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

    gpu_test!(test_first_frame_detects, inner_first_frame_detects);
    gpu_test!(test_unique_ids, inner_unique_ids);
    gpu_test!(test_tracking_across_frames, inner_tracking_across_frames);
    gpu_test!(test_ids_persist, inner_ids_persist);
    gpu_test!(test_max_features_respected, inner_max_features_respected);
    gpu_test!(test_replenishment, inner_replenishment_after_loss);
    gpu_test!(test_reset, inner_reset_clears_state);
    gpu_test!(test_three_frame_sequence, inner_three_frame_sequence);
    gpu_test!(test_split_matches_process, inner_split_matches_process);
    gpu_test!(test_track_meta_aligned, inner_track_meta_aligned);
    gpu_test!(test_drop_tracks_in_flight, inner_drop_tracks_in_flight);
    gpu_test!(test_reset_while_pending, inner_reset_while_pending);
    gpu_test!(test_lbp_policies, inner_lbp_policies);
}
