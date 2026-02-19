// gpu/klt.rs — GPU pyramidal KLT tracker (Inverse Compositional).
//
// Design overview:
//
//   GpuKltTracker is created once (expensive shader compilation).
//   `track()` is called each frame with two GpuPyramids and a feature list.
//
//   Under the hood, `track()` dispatches one compute pass per pyramid level,
//   coarse → fine. Between passes, the displacement buffer is scaled by ×2
//   on the CPU (cheap: a few bytes, no GPU round-trip needed since the
//   buffer stays on GPU). The final pass (level 0) writes the tracked
//   positions and statuses to the results buffer, which is then read back.
//
//
// SHADER PARAMETERS BAKED AT COMPILE TIME
// ─────────────────────────────────────────
// Like the pyramid and FAST shaders, window_size is baked into the shader
// source via string substitution before compilation:
//   {{HALF}}    = window_size         (e.g. 7)
//   {{SIDE}}    = 2*window_size + 1   (e.g. 15)
//   {{PATCH}}   = SIDE²               (e.g. 225)
//   {{WG_SIZE}} = 1-D workgroup size  (e.g. 64)
//
// This makes the template buffer a compile-time-sized array in the shader's
// private address space, avoiding dynamic allocation.
//
//
// NEW WGPU CONCEPTS
// ─────────────────
// - **Scaling a storage buffer on CPU between dispatches**: rather than
//   reading back and re-uploading the displacement buffer, we call
//   `queue.write_buffer` directly. The GPU has already finished (after
//   `device.poll(Wait)` is replaced by submitting a pipeline barrier via
//   a dummy submit), but for simplicity we use a CPU-side intermediate
//   for the ×2 scale step.
//
//   More precisely: we submit the coarse-level dispatch, then block with
//   `device.poll(Wait)`, read back the 8n bytes, scale on CPU, write back.
//   This adds one round-trip per level but keeps the code simple. A future
//   optimisation would use a tiny "scale" compute shader to avoid the
//   CPU involvement entirely.
//
// - **Reusing a pipeline across multiple bind groups**: the pipeline is
//   compiled once; each level creates a new bind group with the correct
//   texture pair. Bind groups are cheap to create.

use wgpu::util::DeviceExt;

use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramid;
use crate::klt::{TrackedFeature, TrackStatus};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 1-D workgroup size. Each thread handles one feature independently.
/// 64 is a good default: enough parallelism, fits in a single wavefront on
/// AMD (64 lanes) and two warps on NVIDIA (32 lanes each).
const WG_SIZE: u32 = 64;

/// Sentinel displacement value indicating a lost feature.
/// Must match LOST_SENTINEL in klt.wgsl.

// ---------------------------------------------------------------------------
// GPU-side structs (must match WGSL layout exactly — repr(C))
// ---------------------------------------------------------------------------

/// Feature input layout. Matches GpuFeature in fast.rs / fast.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuKltFeature {
    pub x:     f32,
    pub y:     f32,
    pub score: f32,
    pub _pad:  f32,
}

impl From<&Feature> for GpuKltFeature {
    fn from(f: &Feature) -> Self {
        GpuKltFeature { x: f.x, y: f.y, score: f.score, _pad: 0.0 }
    }
}

/// Tracking result written by the level-0 shader pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTrackResult {
    x:      f32,
    y:      f32,
    status: u32,  // 0 = Tracked, 1 = Lost, 2 = OutOfBounds
    _pad:   u32,
}

/// Uniform parameters for one level pass (must match KltParams in klt.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KltParams {
    n_features:     u32,
    max_iterations: u32,
    epsilon_sq:     f32,
    level:          u32,
    level_scale:    f32,
    img0_width:     u32,
    img0_height:    u32,
    _pad:           u32,
}

// ---------------------------------------------------------------------------
// GpuKltTracker
// ---------------------------------------------------------------------------

/// GPU inverse-compositional KLT tracker.
///
/// Create once with `GpuKltTracker::new()`; call `track()` every frame.
/// The compute pipeline is compiled at construction time.
pub struct GpuKltTracker {
    pipeline:       wgpu::ComputePipeline,
    bgl:            wgpu::BindGroupLayout,
    pub window_size:    usize,
    pub max_iterations: usize,
    pub epsilon:        f32,
    pub max_levels:     usize,

    // Pre-allocated GPU buffers — reused every frame to avoid VRAM allocation
    // overhead. Sized for max_features at construction.
    max_features:   usize,
    feature_buf:    wgpu::Buffer,   // STORAGE | COPY_DST  — feature positions
    disp_buf:       wgpu::Buffer,   // STORAGE | COPY_DST  — displacements (zeroed each frame)
    results_buf:    wgpu::Buffer,   // STORAGE | COPY_SRC  — track results
    rb_buf:         wgpu::Buffer,   // MAP_READ | COPY_DST — CPU readback
    params_bufs:    Vec<wgpu::Buffer>, // one UNIFORM | COPY_DST per level
}

impl GpuKltTracker {
    /// Create a GPU KLT tracker.
    ///
    /// `window_size` is the patch half-width W (patch = (2W+1)²).
    /// Typical values: 4 (GAP8 / small images), 7 (vilib / HD cameras).
    pub fn new(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
    ) -> Self {
        let side  = 2 * window_size + 1;
        let patch = side * side;

        // Bake all compile-time constants into the shader source.
        let shader_template = include_str!("../shaders/klt.wgsl");
        let shader_src = shader_template
            .replace("{{HALF}}",    &window_size.to_string())
            .replace("{{SIDE}}",    &side.to_string())
            .replace("{{PATCH}}",   &patch.to_string())
            .replace("{{WG_SIZE}}", &WG_SIZE.to_string());

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("klt.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout mirrors @group(0) in klt.wgsl.
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuKlt BGL"),
            entries: &[
                // 0 — prev_tex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 1 — curr_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // 2 — features (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3 — displacements (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4 — results (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5 — params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuKlt pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some("track_level"),
                layout:              Some(&pipeline_layout),
                module:              &shader,
                entry_point:         "track_level",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:               None,
            });

        // Pre-allocate buffers sized for max_features.
        // write_buffer / clear_buffer are used each frame to update content —
        // both go through the queue's staging ring and avoid VRAM re-allocation.
        let feat_bytes   = (max_features * std::mem::size_of::<GpuKltFeature>()) as u64;
        let disp_bytes   = (max_features * 2 * std::mem::size_of::<f32>()) as u64;
        let result_bytes = (max_features * std::mem::size_of::<GpuTrackResult>()) as u64;

        let feature_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt features"), size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let disp_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt displacements"), size: disp_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt results"), size: result_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt readback"), size: result_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One params buffer per level — content updated via write_buffer each frame.
        let params_bufs: Vec<wgpu::Buffer> = (0..max_levels)
            .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuKlt params"), size: std::mem::size_of::<KltParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        GpuKltTracker {
            pipeline, bgl,
            window_size, max_iterations, epsilon, max_levels,
            max_features,
            feature_buf, disp_buf, results_buf, rb_buf, params_bufs,
        }
    }

    pub fn track(
        &mut self,
        gpu:          &GpuDevice,
        prev_pyramid: &GpuPyramid,
        curr_pyramid: &GpuPyramid,
        features:     &[Feature],
    ) -> Vec<TrackedFeature> {
        if features.is_empty() {
            return Vec::new();
        }

        let n = features.len();
        assert!(n <= self.max_features,
            "GpuKltTracker: {} features exceeds max_features={}", n, self.max_features);

        let n_u32 = n as u32;
        let num_levels = self.max_levels
            .min(prev_pyramid.levels.len())
            .min(curr_pyramid.levels.len());

        let img0_w = prev_pyramid.levels[0].width;
        let img0_h = prev_pyramid.levels[0].height;

        // ── Upload features (write_buffer into pre-allocated storage buf) ─────
        let gpu_features: Vec<GpuKltFeature> =
            features.iter().map(GpuKltFeature::from).collect();
        let feat_bytes = (n * std::mem::size_of::<GpuKltFeature>()) as u64;
        gpu.queue.write_buffer(&self.feature_buf, 0, bytemuck::cast_slice(&gpu_features));

        // ── Write params for each level into pre-allocated uniform bufs ───────
        let result_bytes = (n * std::mem::size_of::<GpuTrackResult>()) as u64;
        let disp_bytes   = (n * 2 * std::mem::size_of::<f32>()) as u64;

        for (i, level) in (0..num_levels).rev().enumerate() {
            let params = KltParams {
                n_features:     n_u32,
                max_iterations: self.max_iterations as u32,
                epsilon_sq:     self.epsilon * self.epsilon,
                level:          level as u32,
                level_scale:    1.0f32 / (1u32 << level) as f32,
                img0_width:     img0_w,
                img0_height:    img0_h,
                _pad:           0,
            };
            gpu.queue.write_buffer(&self.params_bufs[i], 0, bytemuck::bytes_of(&params));
        }

        // ── Build bind groups (cheap — just descriptor writes, no VRAM alloc) ─
        // Textures change every frame so bind groups can't be pre-built.
        let bind_groups: Vec<wgpu::BindGroup> = (0..num_levels).rev()
            .enumerate()
            .map(|(i, level)| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:  Some("GpuKlt BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &prev_pyramid.levels[level].read_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &curr_pyramid.levels[level].read_view),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: self.feature_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.disp_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.results_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.params_bufs[i].as_entire_binding() },
                    ],
                })
            })
            .collect();

        // ── Single encoder: clear displacements + all level passes + readback ─
        let workgroups = (n_u32 + WG_SIZE - 1) / WG_SIZE;
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuKlt") },
        );

        // Zero the displacement buffer — clear_buffer is a GPU memset,
        // faster than write_buffer (no staging copy) and stays in the encoder.
        encoder.clear_buffer(&self.disp_buf, 0, Some(disp_bytes));

        for bg in &bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("track_level"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.results_buf, 0, &self.rb_buf, 0, result_bytes);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // ── Readback ──────────────────────────────────────────────────────────
        let slice = self.rb_buf.slice(..result_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("KLT readback map failed");

        let mapped = slice.get_mapped_range();
        let gpu_results: &[GpuTrackResult] = bytemuck::cast_slice(&mapped);

        let tracked: Vec<TrackedFeature> = gpu_results[..n].iter()
            .zip(features.iter())
            .map(|(r, f)| {
                let status = match r.status {
                    0 => TrackStatus::Tracked,
                    1 => TrackStatus::Lost,
                    _ => TrackStatus::OutOfBounds,
                };
                TrackedFeature {
                    feature: Feature { x: r.x, y: r.y, score: f.score, level: f.level, id: f.id },
                    status,
                }
            })
            .collect();

        drop(mapped);
        self.rb_buf.unmap();
        tracked
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::image::Image;
    use crate::pyramid::Pyramid;

    fn run_gpu_test(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args([
                "test", "--lib", "--",
                test_name, "--exact", "--ignored", "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed: {e}"));
        let out = String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr);
        print!("{out}");
        out
    }

    // ---- helpers -----------------------------------------------------------

    /// Bright square on dark background — the standard test scene.
    fn make_test_image(w: usize, h: usize, sq_x: usize, sq_y: usize, sq_size: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![30u8; w * h]);
        for y in sq_y..(sq_y + sq_size).min(h) {
            for x in sq_x..(sq_x + sq_size).min(w) {
                img.set(x, y, 200);
            }
        }
        img
    }

    /// Make a feature at (x, y) for use in track() calls.
    fn feat(x: f32, y: f32) -> Feature {
        Feature { x, y, score: 100.0, level: 0, id: 1 }
    }

    // ---- inner GPU tests (subprocess-isolated) ----------------------------

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_zero_motion() {
        let img = make_test_image(120, 120, 40, 40, 30);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &img, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let results = tracker.track(&gpu, &pyr, &pyr, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked);
        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(dx.abs() < 0.5 && dy.abs() < 0.5,
            "zero motion: ({dx:.3}, {dy:.3}) should be ~0");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_horizontal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 40, 30); // shifted right 3px
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked,
            "status = {:?}", results[0].status);
        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!((dx - 3.0).abs() < 1.5, "horizontal shift: dx={dx:.3}, expected ~3");
        assert!(dy.abs() < 1.5,         "horizontal shift: dy={dy:.3}, expected ~0");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_flat_region_is_lost() {
        let img = Image::from_vec(60, 60, vec![128u8; 3600]);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &img, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 5, 30, 0.01, 3, 256);

        let features = vec![feat(30.0, 30.0)];
        let results = tracker.track(&gpu, &pyr, &pyr, &features);

        assert_eq!(results[0].status, TrackStatus::Lost,
            "flat region should be Lost");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_matches_cpu() {
        // Both GPU (IC) and CPU (IC) should recover the same displacement on a
        // clean synthetic shift. Tolerance: 0.5 pixels (same as CPU-only test).
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 42, 30); // +3, +2

        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let gpu_results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        // CPU reference (IC).
        use crate::klt::{KltTracker, LkMethod};
        let cpu_pyr1 = Pyramid::build(&img1, 3, 1.0);
        let cpu_pyr2 = Pyramid::build(&img2, 3, 1.0);
        let cpu_tracker = KltTracker::with_method(7, 30, 0.01, 3, LkMethod::InverseCompositional);
        let cpu_results = cpu_tracker.track(&cpu_pyr1, &cpu_pyr2, &features);

        eprintln!("[test] GPU: ({:.3}, {:.3}) status={:?}",
            gpu_results[0].feature.x, gpu_results[0].feature.y, gpu_results[0].status);
        eprintln!("[test] CPU: ({:.3}, {:.3}) status={:?}",
            cpu_results[0].feature.x, cpu_results[0].feature.y, cpu_results[0].status);

        assert_eq!(gpu_results[0].status, TrackStatus::Tracked);
        assert_eq!(cpu_results[0].status, TrackStatus::Tracked);

        let gpu_dx = gpu_results[0].feature.x - 41.0;
        let cpu_dx = cpu_results[0].feature.x - 41.0;
        let gpu_dy = gpu_results[0].feature.y - 41.0;
        let cpu_dy = cpu_results[0].feature.y - 41.0;

        assert!((gpu_dx - cpu_dx).abs() < 0.5,
            "dx mismatch: GPU={gpu_dx:.3} CPU={cpu_dx:.3}");
        assert!((gpu_dy - cpu_dy).abs() < 0.5,
            "dy mismatch: GPU={gpu_dy:.3} CPU={cpu_dy:.3}");

        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_subpixel_shift() {
        // Gaussian blob shifted by (1.5, 0.5) — tests sub-pixel accuracy.
        let w = 80usize;
        let h = 80usize;
        let mut d1 = vec![0u8; w * h];
        let mut d2 = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let r1 = (x as f32 - 40.0).powi(2) + (y as f32 - 40.0).powi(2);
                d1[y * w + x] = (255.0 * (-0.005 * r1).exp()) as u8;
                let r2 = (x as f32 - 41.5).powi(2) + (y as f32 - 40.5).powi(2);
                d2[y * w + x] = (255.0 * (-0.005 * r2).exp()) as u8;
            }
        }

        let img1 = Image::from_vec(w, h, d1);
        let img2 = Image::from_vec(w, h, d2);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(40.0, 40.0)];
        let results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked);
        let dx = results[0].feature.x - 40.0;
        let dy = results[0].feature.y - 40.0;
        assert!((dx - 1.5).abs() < 0.5, "subpixel dx={dx:.3}, expected ~1.5");
        assert!((dy - 0.5).abs() < 0.5, "subpixel dy={dy:.3}, expected ~0.5");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    // ---- outer wrappers (non-GPU CI, spawn subprocess) --------------------

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_zero_motion() {
        let out = run_gpu_test("gpu::klt::tests::inner_zero_motion");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_horizontal_shift() {
        let out = run_gpu_test("gpu::klt::tests::inner_horizontal_shift");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_flat_region_is_lost() {
        let out = run_gpu_test("gpu::klt::tests::inner_flat_region_is_lost");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu() {
        let out = run_gpu_test("gpu::klt::tests::inner_gpu_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_subpixel_shift() {
        let out = run_gpu_test("gpu::klt::tests::inner_subpixel_shift");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
