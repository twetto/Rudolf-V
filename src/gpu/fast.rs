// gpu/fast.rs — GPU FAST-N corner detector + GPU occupancy-grid NMS.
//
// FAST and NMS are chained in a single command encoder:
//
//   clear_buffer(score_buf)          — GPU memset, no CPU involvement
//   dispatch detect_corners          — writes score per pixel
//   dispatch nms_cells               — one thread per cell, finds max
//   copy_buffer winners → rb         — readback only winners (~17 KB)
//   submit + poll(Wait)              — one round-trip total
//
// Previously: FAST wrote scores → CPU read back 1.4 MB → CPU NMS.
// Now: FAST writes scores → GPU NMS → CPU reads ~17 KB winners only.
// Eliminates one ~4 ms dzn synchronisation round-trip per frame.
//
//
// PRE-ALLOCATED BUFFERS
// ──────────────────────
// All GPU buffers are allocated once in new() and reused every frame:
//   score_buf   — img_w × img_h × 4 bytes  (R/W storage)
//   winners_buf — n_cells × 16 bytes        (R/W storage + COPY_SRC)
//   rb_buf      — n_cells × 16 bytes        (MAP_READ + COPY_DST)
//
// The score buffer is zeroed with encoder.clear_buffer() each frame,
// which is a GPU-side memset with no staging copy.

use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramidLevel;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WG_SIZE_NMS: u32 = 64;

// ---------------------------------------------------------------------------
// GPU-side structs
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

/// Cell winner as written by the NMS shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CellWinner {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,
}

// ---------------------------------------------------------------------------
// GpuFastDetector
// ---------------------------------------------------------------------------

/// GPU FAST-N corner detector with integrated GPU occupancy-grid NMS.
///
/// Create once; call [`detect`] each frame. All GPU buffers are pre-allocated
/// and reused — no per-frame VRAM allocation after construction.
///
/// The public interface is identical to the old version: `detect()` returns
/// `Vec<Feature>`. NMS happens on-GPU; the caller no longer needs to run
/// `OccupancyNms` on the result.
pub struct GpuFastDetector {
    // FAST pipeline
    fast_pipeline: wgpu::ComputePipeline,
    fast_bgl:      wgpu::BindGroupLayout,

    // NMS pipeline
    nms_pipeline:  wgpu::ComputePipeline,
    nms_bgl:       wgpu::BindGroupLayout,

    // Pre-allocated buffers sized for img_w × img_h at construction time.
    score_buf:    wgpu::Buffer,  // img_w × img_h × 4  — STORAGE | COPY_DST
    winners_buf:  wgpu::Buffer,  // n_cells × 16       — STORAGE | COPY_SRC
    rb_buf:       wgpu::Buffer,  // n_cells × 16       — MAP_READ | COPY_DST

    // Params buffers (written via write_buffer each frame, no re-alloc).
    fast_params_buf: wgpu::Buffer,
    nms_params_buf:  wgpu::Buffer,

    pub threshold:  u8,
    pub arc_length: usize,
    pub cell_size:  usize,

    // Dimensions this detector was built for.
    _img_w:    u32,
    _img_h:    u32,
    _n_cells_x: u32,
    _n_cells_y: u32,

    // State set by record_into(), consumed by arm_readback()/collect_winners().
    // Stored here so record_into() and collect_winners() share context without
    // the caller needing to thread it through.
    n_cells_recorded:  u32,
    score_bytes_rec:   u64,
    fast_bg_rec:       Option<wgpu::BindGroup>,
    nms_bg_rec:        Option<wgpu::BindGroup>,
    wg_x_rec: u32, wg_y_rec: u32, nms_wg_rec: u32,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuFastDetector {
    /// Create a GPU FAST+NMS detector.
    ///
    /// `img_w` / `img_h` — expected image dimensions (level 0).
    /// `cell_size` — NMS grid cell size in pixels (e.g. 16).
    pub fn new(
        gpu:        &GpuDevice,
        threshold:  u8,
        arc_length: usize,
        img_w:      usize,
        img_h:      usize,
        cell_size:  usize,
    ) -> Self {
        assert!((9..=12).contains(&arc_length),
            "arc_length must be 9..=12 (got {arc_length})");

        // ── FAST shader ───────────────────────────────────────────────────────
        let fast_src = include_str!("../shaders/fast.wgsl")
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string());
        let fast_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fast.wgsl"), source: wgpu::ShaderSource::Wgsl(fast_src.into()),
        });

        let fast_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuFast BGL"),
            entries: &[
                // 0 — input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    }, count: None,
                },
                // 1 — score buffer (storage r/w)
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // 2 — params uniform
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

        // ── NMS shader ────────────────────────────────────────────────────────
        let nms_src = include_str!("../shaders/nms.wgsl")
            .replace("{{WG_SIZE}}", &WG_SIZE_NMS.to_string());
        let nms_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nms.wgsl"), source: wgpu::ShaderSource::Wgsl(nms_src.into()),
        });

        let nms_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuNms BGL"),
            entries: &[
                // 0 — scores (storage read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // 1 — winners (storage r/w)
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                // 2 — params uniform
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

        // ── Pre-allocate buffers ──────────────────────────────────────────────
        let w = img_w as u32;
        let h = img_h as u32;
        let cs = cell_size as u32;
        let n_cells_x = w.div_ceil(cs);
        let n_cells_y = h.div_ceil(cs);
        let n_cells   = n_cells_x * n_cells_y;

        let score_bytes   = (img_w * img_h * std::mem::size_of::<f32>()) as u64;
        let winners_bytes = (n_cells as usize * std::mem::size_of::<CellWinner>()) as u64;

        let score_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast scores"), size: score_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let winners_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast winners"), size: winners_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast readback"), size: winners_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params buffers — content updated via write_buffer each frame.
        let fast_params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast params"), size: std::mem::size_of::<FastParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nms_params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuNms params"), size: std::mem::size_of::<NmsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuFastDetector {
            fast_pipeline, fast_bgl, nms_pipeline, nms_bgl,
            score_buf, winners_buf, rb_buf,
            fast_params_buf, nms_params_buf,
            threshold, arc_length, cell_size,
            _img_w: w, _img_h: h, _n_cells_x: n_cells_x, _n_cells_y: n_cells_y,
            n_cells_recorded: 0, score_bytes_rec: 0,
            fast_bg_rec: None, nms_bg_rec: None,
            wg_x_rec: 0, wg_y_rec: 0, nms_wg_rec: 0,
            readback_rx: None,
        }
    }

    /// Detect and NMS-suppress corners in a pyramid level.
    ///
    /// Returns at most one `Feature` per NMS cell. Feature `id` is 0
    /// (assigned by the frontend). Feature `level` is set to `pyramid_level`.
    ///
    /// The result is already NMS-suppressed — callers do NOT need to run
    /// `OccupancyNms` on the output. The occupancy grid should still be used
    /// to skip cells already occupied by tracked features.
    /// Write params and record FAST+NMS passes into `encoder`.
    ///
    /// Call order for full pipeline fusion (one submit, one poll):
    /// ```text
    /// fast.record_into(gpu, &mut encoder, level);
    /// klt.record_into(&mut encoder);          // other work
    /// queue.submit(encoder.finish());
    /// fast.arm_readback();
    /// klt.arm_readback();
    /// device.poll(Wait);                      // single shared poll
    /// let features = fast.collect_winners(0);
    /// ```
    pub fn record_into(
        &mut self,
        gpu:     &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        level:   &GpuPyramidLevel,
    ) {
        let w  = level.width;
        let h  = level.height;
        let cs = self.cell_size as u32;

        let n_cells_x = w.div_ceil(cs);
        let n_cells_y = h.div_ceil(cs);
        let n_cells   = n_cells_x * n_cells_y;
        let score_bytes   = (w * h) as u64 * 4;
        let winners_bytes = n_cells as u64 * std::mem::size_of::<CellWinner>() as u64;

        assert!(score_bytes   <= self.score_buf.size(),   "image too large for score_buf");
        assert!(winners_bytes <= self.winners_buf.size(), "too many cells for winners_buf");

        // Params go via staging ring — valid at submit time regardless of
        // when record_into() is called relative to other write_buffer calls.
        gpu.queue.write_buffer(&self.fast_params_buf, 0, bytemuck::bytes_of(&FastParams {
            img_width: w, img_height: h,
            threshold: self.threshold as f32, arc_length: self.arc_length as u32,
        }));
        gpu.queue.write_buffer(&self.nms_params_buf, 0, bytemuck::bytes_of(&NmsParams {
            img_width: w, img_height: h,
            cell_size: cs, n_cells_x, n_cells_y,
            _pad0: 0, _pad1: 0, _pad2: 0,
        }));

        // BindGroup creation is cheap (descriptor writes, no VRAM).
        // wgpu resources are Arc-backed, so storing them in self is fine.
        let fast_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuFast BG"), layout: &self.fast_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level.read_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.score_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.fast_params_buf.as_entire_binding() },
            ],
        });
        let nms_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuNms BG"), layout: &self.nms_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.score_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.winners_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.nms_params_buf.as_entire_binding() },
            ],
        });

        let (wg_x, wg_y) = gpu.dispatch_size(w, h);
        let nms_wg = (n_cells + WG_SIZE_NMS - 1) / WG_SIZE_NMS;

        encoder.clear_buffer(&self.score_buf, 0, Some(score_bytes));
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
            pass.set_pipeline(&self.nms_pipeline);
            pass.set_bind_group(0, &nms_bg, &[]);
            pass.dispatch_workgroups(nms_wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.winners_buf, 0, &self.rb_buf, 0, winners_bytes);

        // Stash dispatch context for collect_winners().
        self.n_cells_recorded = n_cells;
        self.score_bytes_rec  = score_bytes;
        self.fast_bg_rec      = Some(fast_bg);
        self.nms_bg_rec       = Some(nms_bg);
        self.wg_x_rec = wg_x; self.wg_y_rec = wg_y; self.nms_wg_rec = nms_wg;
    }

    /// Map the readback buffer asynchronously.
    /// Must be called after `queue.submit()` and before `device.poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let winners_bytes = self.n_cells_recorded as u64
            * std::mem::size_of::<CellWinner>() as u64;
        let (tx, rx) = std::sync::mpsc::channel();
        self.rb_buf
            .slice(..winners_bytes)
            .map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.readback_rx = Some(rx);
    }

    /// Collect detected corners. Must be called after `device.poll(Wait)`.
    pub fn collect_winners(&mut self, pyramid_level: usize) -> Vec<Feature> {
        let rx = self.readback_rx.take()
            .expect("call arm_readback() before collect_winners()");
        rx.recv().unwrap().expect("GpuFast readback failed");

        let winners_bytes = self.n_cells_recorded as u64
            * std::mem::size_of::<CellWinner>() as u64;
        let slice  = self.rb_buf.slice(..winners_bytes);
        let mapped = slice.get_mapped_range();
        let winners: &[CellWinner] = bytemuck::cast_slice(&mapped);
        let features = winners.iter()
            .filter(|w| w.score > 0.0)
            .map(|w| Feature { x: w.x, y: w.y, score: w.score, level: pyramid_level, id: 0 })
            .collect();
        drop(mapped);
        self.rb_buf.unmap();
        // Drop cached bind groups — textures from this frame shouldn't outlive it.
        self.fast_bg_rec = None;
        self.nms_bg_rec  = None;
        features
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
            &wgpu::CommandEncoderDescriptor { label: Some("GpuFast+NMS standalone") });
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

    #[test]
    #[ignore = "GPU integration"]
    fn inner_no_corners_on_flat_image() {
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 20, 9, 64, 64, 16);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(features.is_empty(), "flat image should have no corners, got {}", features.len());
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_detects_bright_corner() {
        let mut pixels = vec![20u8; 64 * 64];
        for y in 20..44usize { for x in 20..44usize { pixels[y * 64 + x] = 220; } }
        let src = Image::<u8>::from_vec(64, 64, pixels);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 64, 64, 16);
        let features = det.detect(&gpu, &pyr.levels[0], 0);
        assert!(!features.is_empty(), "bright rectangle should produce corners");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_gpu_matches_cpu_positions() {
        // After NMS, GPU should find corners at similar positions to CPU+NMS.
        // We check that GPU corners are a subset of CPU corners (NMS may choose
        // slightly different winners due to floating-point, but positions should
        // be within the same cell).
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
        let mut det = GpuFastDetector::new(&gpu, 30, 9, 128, 128, 16);
        let gpu_features = det.detect(&gpu, &pyr.levels[0], 0);

        eprintln!("[test] CPU raw: {} corners, GPU+NMS: {} corners",
            cpu_raw.len(), gpu_features.len());

        // GPU+NMS produces at most one corner per cell.
        // All GPU winners should correspond to actual CPU-detected corners.
        for gf in &gpu_features {
            let found = cpu_raw.iter().any(|cf|
                (cf.x as i32 - gf.x as i32).abs() <= 1 &&
                (cf.y as i32 - gf.y as i32).abs() <= 1
            );
            assert!(found, "GPU corner ({},{}) not in CPU detections", gf.x, gf.y);
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_no_corners_on_flat_image() {
        let out = run_gpu_test("gpu::fast::tests::inner_no_corners_on_flat_image");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_detects_bright_corner() {
        let out = run_gpu_test("gpu::fast::tests::inner_detects_bright_corner");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu_positions() {
        let out = run_gpu_test("gpu::fast::tests::inner_gpu_matches_cpu_positions");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
