// gpu/fast.rs — GPU FAST-N corner detector with selectable NMS strategy.
//
// NmsStrategy::Cpu  (default, best for iGPU / unified memory)
// ─────────────────────────────────────────────────────────────
//   record_into: FAST dispatch → copy score_buf (1.4 MB) → rb
//   collect:     iterate scores on CPU, cell-max NMS inline
//
//   Buffers are allocated fresh each call so the driver zero-inits them —
//   no cross-frame WAW dependency, no explicit clear needed.
//   On unified memory the 1.4 MB "copy" is essentially free; CPU NMS on
//   ~few-hundred corners is sub-microsecond.
//
// NmsStrategy::Gpu  (best for discrete GPU with PCIe)
// ─────────────────────────────────────────────────────
//   record_into: clear score_buf → FAST → nms_cells → copy winners (~17 KB) → rb
//   collect:     read winners array directly — already one per cell
//
//   Pre-allocated buffers avoid per-frame VRAM allocation.
//   Saves ~175 µs of PCIe transfer (1.4 MB @ 8 GB/s) at the cost of an
//   extra compute pass that is negligible on a discrete GPU.

use wgpu::util::DeviceExt;
use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramidLevel;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls where NMS runs after GPU FAST detection.
///
/// | Hardware                         | Recommended | Reason                                        |
/// |----------------------------------|-------------|-----------------------------------------------|
/// | iGPU / SoC (780M, Jetson, RPi5) | `Cpu`       | Unified memory — 1.4 MB readback is free      |
/// | Discrete GPU (RTX, RX series)    | `Gpu`       | Saves ~175 µs PCIe transfer per frame         |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NmsStrategy {
    /// Full score buffer readback (~1.4 MB for 752×480), CPU cell-max NMS.
    /// Per-call buffer allocation avoids cross-frame GPU dependencies.
    /// Default: best for iGPU / unified memory.
    #[default]
    Cpu,
    /// GPU NMS pass, winners-only readback (~17 KB for 752×480, cell=16).
    /// Pre-allocated buffers + `clear_buffer` each frame.
    /// Best for discrete GPU where PCIe bandwidth is the bottleneck.
    Gpu,
}

// ---------------------------------------------------------------------------
// Internal structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    img_width:  u32,
    img_height: u32,
    threshold:  f32,
    arc_length: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NmsParams {
    img_width:  u32,
    img_height: u32,
    cell_size:  u32,
    n_cells_x:  u32,
    n_cells_y:  u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CellWinner {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,
}

const WG_SIZE_NMS: u32 = 64;

// ---------------------------------------------------------------------------
// Strategy-specific state
// ---------------------------------------------------------------------------

struct CpuNmsState {
    _score_buf:  wgpu::Buffer,  // kept alive until GPU finishes; never read by CPU
    _params_buf: wgpu::Buffer,  // kept alive until submit
    rb_buf:      wgpu::Buffer,
    img_w:       u32,
    img_h:       u32,
}

struct GpuNmsState {
    nms_pipeline:     wgpu::ComputePipeline,
    nms_bgl:          wgpu::BindGroupLayout,
    score_buf:        wgpu::Buffer,
    winners_buf:      wgpu::Buffer,
    rb_buf:           wgpu::Buffer,
    nms_params_buf:   wgpu::Buffer,
    n_cells_recorded: u32,
    img_w:     u32,
    img_h:     u32,
}

enum NmsState {
    Cpu(Option<CpuNmsState>),  // None between frames
    Gpu(GpuNmsState),
}

// ---------------------------------------------------------------------------
// GpuFastDetector
// ---------------------------------------------------------------------------

/// GPU FAST-N corner detector with selectable NMS strategy.
///
/// Create once at startup; call [`detect`] each frame.
/// For pipeline fusion with KLT use the split API:
/// `record_into` → (submit) → `arm_readback` → (poll) → `collect_winners`.
pub struct GpuFastDetector {
    fast_pipeline:   wgpu::ComputePipeline,
    fast_bgl:        wgpu::BindGroupLayout,
    fast_params_buf: wgpu::Buffer,

    pub threshold:  u8,
    pub arc_length: usize,
    pub cell_size:  usize,

    nms: NmsState,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuFastDetector {
    /// Create a GPU FAST detector.
    ///
    /// `img_w` / `img_h` — expected image dimensions (level 0).
    /// `cell_size` — NMS grid cell size in pixels (e.g. 16).
    /// `nms_strategy` — where NMS runs; see [`NmsStrategy`] for guidance.
    pub fn new(
        gpu:          &GpuDevice,
        threshold:    u8,
        arc_length:   usize,
        img_w:        usize,
        img_h:        usize,
        cell_size:    usize,
        nms_strategy: NmsStrategy,
    ) -> Self {
        assert!((9..=12).contains(&arc_length),
            "arc_length must be 9..=12 (got {arc_length})");

        // ── FAST shader (shared by both strategies) ───────────────────────────
        let fast_src = include_str!("../shaders/fast.wgsl")
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string());
        let fast_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fast.wgsl"), source: wgpu::ShaderSource::Wgsl(fast_src.into()),
        });

        let fast_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuFast BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let fast_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("detect_corners"),
            layout: Some(&gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&fast_bgl], push_constant_ranges: &[],
            })),
            module: &fast_shader, entry_point: "detect_corners",
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let fast_params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast params"), size: std::mem::size_of::<FastParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Strategy-specific setup ───────────────────────────────────────────
        let nms = match nms_strategy {
            NmsStrategy::Cpu => NmsState::Cpu(None),

            NmsStrategy::Gpu => {
                let w  = img_w as u32;
                let h  = img_h as u32;
                let cs = cell_size as u32;
                let n_cells_x = w.div_ceil(cs);
                let n_cells_y = h.div_ceil(cs);

                let nms_src = include_str!("../shaders/nms.wgsl")
                    .replace("{{WG_SIZE}}", &WG_SIZE_NMS.to_string());
                let nms_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("nms.wgsl"), source: wgpu::ShaderSource::Wgsl(nms_src.into()),
                });

                let nms_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GpuNms BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false, min_binding_size: None,
                            }, count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false, min_binding_size: None,
                            }, count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false, min_binding_size: None,
                            }, count: None,
                        },
                    ],
                });

                let nms_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("nms_cells"),
                    layout: Some(&gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None, bind_group_layouts: &[&nms_bgl], push_constant_ranges: &[],
                    })),
                    module: &nms_shader, entry_point: "nms_cells",
                    compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
                });

                let score_bytes   = (img_w * img_h) as u64 * 4;
                let n_cells       = (n_cells_x * n_cells_y) as u64;
                let winners_bytes = n_cells * std::mem::size_of::<CellWinner>() as u64;

                let score_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuFast scores"), size: score_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let winners_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuNms winners"), size: winners_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuNms readback"), size: winners_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let nms_params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuNms params"), size: std::mem::size_of::<NmsParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                NmsState::Gpu(GpuNmsState {
                    nms_pipeline, nms_bgl,
                    score_buf, winners_buf, rb_buf, nms_params_buf,
                    n_cells_recorded: 0,
                    img_w: w, img_h: h,
                })
            }
        };

        GpuFastDetector {
            fast_pipeline, fast_bgl, fast_params_buf,
            threshold, arc_length, cell_size,
            nms, readback_rx: None,
        }
    }

    // ── Split API ─────────────────────────────────────────────────────────────

    /// Record FAST (+ optional GPU NMS) passes into an existing encoder.
    ///
    /// `NmsStrategy::Cpu`: allocates fresh `score_buf` + `rb_buf` each call —
    /// driver zero-inits them, no cross-frame WAW dependency, no clear needed.
    ///
    /// `NmsStrategy::Gpu`: uses pre-allocated buffers; emits `clear_buffer`
    /// to prevent cross-frame write-after-write barriers on RADV.
    pub fn record_into(
        &mut self,
        gpu:     &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        level:   &GpuPyramidLevel,
    ) {
        let w  = level.width;
        let h  = level.height;
        let cs = self.cell_size as u32;

        match &mut self.nms {
            NmsState::Cpu(slot) => {
                let score_bytes = (w * h) as u64 * 4;

                // All three buffers allocated fresh each frame, exactly as 8a26674:
                // driver zero-inits score_buf, no cross-frame WAW dependency,
                // no explicit clear needed. params_buf uses create_buffer_init
                // (mapped-at-creation) to avoid the write_buffer staging copy.
                let score_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuFast scores"), size: score_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuFast readback"), size: score_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label:    Some("GpuFast params"),
                    contents: bytemuck::bytes_of(&FastParams {
                        img_width: w, img_height: h,
                        threshold: self.threshold as f32, arc_length: self.arc_length as u32,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GpuFast BG"), layout: &self.fast_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level.read_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: score_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                    ],
                });

                let (wg_x, wg_y) = gpu.dispatch_size(w, h);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("detect_corners"), timestamp_writes: None });
                    pass.set_pipeline(&self.fast_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                encoder.copy_buffer_to_buffer(&score_buf, 0, &rb_buf, 0, score_bytes);

                *slot = Some(CpuNmsState { _score_buf: score_buf, _params_buf: params_buf, rb_buf, img_w: w, img_h: h });
            }

            NmsState::Gpu(s) => {
                gpu.queue.write_buffer(&self.fast_params_buf, 0, bytemuck::bytes_of(&FastParams {
                    img_width: w, img_height: h,
                    threshold: self.threshold as f32, arc_length: self.arc_length as u32,
                }));
                assert!(w <= s.img_w && h <= s.img_h,
                    "image ({w}×{h}) larger than pre-allocated buffers ({}×{})", s.img_w, s.img_h);

                let n_cells_x     = w.div_ceil(cs);
                let n_cells_y     = h.div_ceil(cs);
                let n_cells       = n_cells_x * n_cells_y;
                let score_bytes   = (w * h) as u64 * 4;
                let winners_bytes = n_cells as u64 * std::mem::size_of::<CellWinner>() as u64;

                gpu.queue.write_buffer(&s.nms_params_buf, 0, bytemuck::bytes_of(&NmsParams {
                    img_width: w, img_height: h,
                    cell_size: cs, n_cells_x, n_cells_y,
                    _pad0: 0, _pad1: 0, _pad2: 0,
                }));

                let fast_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GpuFast BG"), layout: &self.fast_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level.read_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: s.score_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.fast_params_buf.as_entire_binding() },
                    ],
                });
                let nms_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GpuNms BG"), layout: &s.nms_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: s.score_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: s.winners_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: s.nms_params_buf.as_entire_binding() },
                    ],
                });

                let (wg_x, wg_y) = gpu.dispatch_size(w, h);
                let nms_wg = (n_cells + WG_SIZE_NMS - 1) / WG_SIZE_NMS;

                // clear_buffer prevents cross-frame WAW barriers on the pre-allocated
                // score_buf. Without it RADV stalls until the previous frame's NMS
                // pass has finished reading, producing an alternating ~3 ms penalty.
                encoder.clear_buffer(&s.score_buf, 0, Some(score_bytes));
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("detect_corners"), timestamp_writes: None });
                    pass.set_pipeline(&self.fast_pipeline);
                    pass.set_bind_group(0, &fast_bg, &[]);
                    pass.dispatch_workgroups(wg_x, wg_y, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("nms_cells"), timestamp_writes: None });
                    pass.set_pipeline(&s.nms_pipeline);
                    pass.set_bind_group(0, &nms_bg, &[]);
                    pass.dispatch_workgroups(nms_wg, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&s.winners_buf, 0, &s.rb_buf, 0, winners_bytes);

                s.n_cells_recorded = n_cells;
            }
        }
    }

    /// Map the readback buffer asynchronously.
    /// Call after `queue.submit()`, before `device.poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        match &self.nms {
            NmsState::Cpu(Some(s)) => {
                let score_bytes = (s.img_w * s.img_h) as u64 * 4;
                s.rb_buf.slice(..score_bytes)
                    .map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            }
            NmsState::Gpu(s) => {
                let winners_bytes = s.n_cells_recorded as u64
                    * std::mem::size_of::<CellWinner>() as u64;
                s.rb_buf.slice(..winners_bytes)
                    .map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            }
            NmsState::Cpu(None) => panic!("call record_into() before arm_readback()"),
        }
        self.readback_rx = Some(rx);
    }

    /// Collect detected corners. Call after `device.poll(Wait)`.
    /// Returns one `Feature` per NMS cell (already suppressed).
    pub fn collect_winners(&mut self, pyramid_level: usize) -> Vec<Feature> {
        let rx = self.readback_rx.take()
            .expect("call arm_readback() before collect_winners()");
        rx.recv().unwrap().expect("GpuFast readback failed");

        match &mut self.nms {
            NmsState::Cpu(slot) => {
                let s = slot.take().expect("record_into() state missing");
                let score_bytes = (s.img_w * s.img_h) as u64 * 4;
                let mapped = s.rb_buf.slice(..score_bytes).get_mapped_range();
                let scores: &[f32] = bytemuck::cast_slice(&mapped);

                // Cell-max NMS — mirrors nms.wgsl logic on CPU.
                let cs        = self.cell_size as u32;
                let n_cells_x = s.img_w.div_ceil(cs);
                let n_cells_y = s.img_h.div_ceil(cs);
                let mut features = Vec::new();

                for cy in 0..n_cells_y {
                    for cx in 0..n_cells_x {
                        let x0 = cx * cs;
                        let y0 = cy * cs;
                        let x1 = (x0 + cs).min(s.img_w);
                        let y1 = (y0 + cs).min(s.img_h);

                        let mut best_score = 0.0f32;
                        let mut best_x = 0u32;
                        let mut best_y = 0u32;
                        for y in y0..y1 {
                            for x in x0..x1 {
                                let sc = scores[(y * s.img_w + x) as usize];
                                if sc > best_score {
                                    best_score = sc;
                                    best_x = x;
                                    best_y = y;
                                }
                            }
                        }
                        if best_score > 0.0 {
                            features.push(Feature {
                                x: best_x as f32, y: best_y as f32,
                                score: best_score, level: pyramid_level, id: 0,
                            });
                        }
                    }
                }

                drop(mapped);
                s.rb_buf.unmap();
                features
            }

            NmsState::Gpu(s) => {
                let winners_bytes = s.n_cells_recorded as u64
                    * std::mem::size_of::<CellWinner>() as u64;
                let mapped = s.rb_buf.slice(..winners_bytes).get_mapped_range();
                let winners: &[CellWinner] = bytemuck::cast_slice(&mapped);
                let features = winners.iter()
                    .filter(|w| w.score > 0.0)
                    .map(|w| Feature { x: w.x, y: w.y, score: w.score, level: pyramid_level, id: 0 })
                    .collect();
                drop(mapped);
                s.rb_buf.unmap();
                features
            }
        }
    }

    /// Convenience wrapper: record + submit + poll + collect in one call.
    /// Use for standalone detection (tests, benchmarks, examples).
    /// For pipeline fusion use `record_into` → `arm_readback` → `collect_winners`.
    pub fn detect(
        &mut self,
        gpu:           &GpuDevice,
        level:         &GpuPyramidLevel,
        pyramid_level: usize,
    ) -> Vec<Feature> {
        let mut enc = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuFast standalone") });
        self.record_into(gpu, &mut enc, level);
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.arm_readback();
        gpu.device.poll(wgpu::Maintain::Wait);
        self.collect_winners(pyramid_level)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::FastDetector;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::image::Image;

    fn run_gpu_test(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args(["test", "--lib", "--", test_name, "--exact", "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed for {test_name}: {e}"));
        let out = String::from_utf8_lossy(&output.stdout).into_owned()
                + &String::from_utf8_lossy(&output.stderr);
        print!("{out}"); out
    }

    // ── Cpu strategy ─────────────────────────────────────────────────────────

    #[test]
    #[ignore = "GPU integration"]
    fn inner_cpu_no_corners_flat() {
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 20, 9, 64, 64, 16, NmsStrategy::Cpu);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(features.is_empty(), "flat image should have no corners, got {}", features.len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_cpu_detects_corner() {
        let mut pixels = vec![20u8; 64 * 64];
        for y in 20..44usize { for x in 20..44usize { pixels[y * 64 + x] = 220; } }
        let src = Image::<u8>::from_vec(64, 64, pixels);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 64, 64, 16, NmsStrategy::Cpu);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(!features.is_empty(), "bright rectangle should produce corners");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_cpu_matches_cpu_ref() {
        let mut rng = 99991u32;
        let pixels: Vec<u8> = (0..128*128).map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        }).collect();
        let src = Image::<u8>::from_vec(128, 128, pixels.clone());

        let cpu_det = FastDetector::new(30, 9);
        let cpu_raw = cpu_det.detect(&src);

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 128, 128, 16, NmsStrategy::Cpu);
        let gpu_features = det.detect(&gpu, &pyr.levels[0], 0);

        eprintln!("[test] CPU raw: {}  GPU+CpuNMS: {}",
            cpu_raw.len(), gpu_features.len());

        for gf in &gpu_features {
            let found = cpu_raw.iter().any(|cf|
                (cf.x as i32 - gf.x as i32).abs() <= 1 &&
                (cf.y as i32 - gf.y as i32).abs() <= 1
            );
            assert!(found, "GPU+CpuNMS corner ({},{}) not in CPU detections", gf.x, gf.y);
        }
        println!("GPU_TEST_OK");
    }

    // ── Gpu strategy ─────────────────────────────────────────────────────────

    #[test]
    #[ignore = "GPU integration"]
    fn inner_gpu_no_corners_flat() {
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 20, 9, 64, 64, 16, NmsStrategy::Gpu);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(features.is_empty(), "flat image should have no corners, got {}", features.len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_gpu_detects_corner() {
        let mut pixels = vec![20u8; 64 * 64];
        for y in 20..44usize { for x in 20..44usize { pixels[y * 64 + x] = 220; } }
        let src = Image::<u8>::from_vec(64, 64, pixels);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 64, 64, 16, NmsStrategy::Gpu);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(!features.is_empty(), "bright rectangle should produce corners");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_gpu_matches_cpu_positions() {
        let mut rng = 99991u32;
        let pixels: Vec<u8> = (0..128*128).map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        }).collect();
        let src = Image::<u8>::from_vec(128, 128, pixels.clone());

        let cpu_det = FastDetector::new(30, 9);
        let cpu_raw = cpu_det.detect(&src);

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 128, 128, 16, NmsStrategy::Gpu);
        let gpu_features = det.detect(&gpu, &pyr.levels[0], 0);

        eprintln!("[test] CPU raw: {}  GPU+GpuNMS: {}",
            cpu_raw.len(), gpu_features.len());

        for gf in &gpu_features {
            let found = cpu_raw.iter().any(|cf|
                (cf.x as i32 - gf.x as i32).abs() <= 1 &&
                (cf.y as i32 - gf.y as i32).abs() <= 1
            );
            assert!(found, "GPU corner ({},{}) not in CPU detections", gf.x, gf.y);
        }
        println!("GPU_TEST_OK");
    }

    // ── Subprocess wrappers ───────────────────────────────────────────────────

    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_cpu_no_corners_flat() {
        assert!(run_gpu_test("gpu::fast::tests::inner_cpu_no_corners_flat").contains("GPU_TEST_OK"));
    }
    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_cpu_detects_corner() {
        assert!(run_gpu_test("gpu::fast::tests::inner_cpu_detects_corner").contains("GPU_TEST_OK"));
    }
    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_cpu_matches_cpu_ref() {
        assert!(run_gpu_test("gpu::fast::tests::inner_cpu_matches_cpu_ref").contains("GPU_TEST_OK"));
    }
    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_no_corners_flat() {
        assert!(run_gpu_test("gpu::fast::tests::inner_gpu_no_corners_flat").contains("GPU_TEST_OK"));
    }
    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_detects_corner() {
        assert!(run_gpu_test("gpu::fast::tests::inner_gpu_detects_corner").contains("GPU_TEST_OK"));
    }
    #[test] #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu_positions() {
        assert!(run_gpu_test("gpu::fast::tests::inner_gpu_matches_cpu_positions").contains("GPU_TEST_OK"));
    }
}
