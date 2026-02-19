// gpu/fast.rs — GPU FAST-N corner detector.
//
// OUTPUT STRATEGY: dense score buffer (no atomics)
// ─────────────────────────────────────────────────
// Each thread writes its FAST score to scores[y * width + x] — its own slot,
// no contention, no synchronisation needed. The CPU scans the flat buffer
// after readback and collects all nonzero entries.
//
// This replaces the previous atomic-counter + keypoint-buffer pattern, which
// triggered a naga SPIR-V memory-semantics bug (UniformMemory bit without
// AcquireRelease order) that caused vkCreateShaderModule to fail on strict
// Vulkan validation layers.
//
// Buffer size: img_w × img_h × 4 bytes (≈1.4 MB for EuRoC 752×480).

use wgpu::util::DeviceExt;

use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramidLevel;

// ---------------------------------------------------------------------------
// Uniform params (must match WGSL struct FastParams exactly)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    img_width:  u32,
    img_height: u32,
    threshold:  f32,
    arc_length: u32,
}

// ---------------------------------------------------------------------------
// GpuFastDetector
// ---------------------------------------------------------------------------

/// GPU FAST-N corner detector.
///
/// Create once; call [`detect`] or [`detect_at_level`] each frame.
pub struct GpuFastDetector {
    pipeline:   wgpu::ComputePipeline,
    bgl:        wgpu::BindGroupLayout,
    pub threshold:  u8,
    pub arc_length: usize,
}

impl GpuFastDetector {
    pub fn new(gpu: &GpuDevice, threshold: u8, arc_length: usize) -> Self {
        assert!((9..=12).contains(&arc_length),
            "arc_length must be 9..=12 (got {arc_length})");

        let shader_template = include_str!("../shaders/fast.wgsl");
        let shader_src = shader_template
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string());

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("fast.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuFast BGL"),
            entries: &[
                // 0 — input texture
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
                // 1 — dense score buffer (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2 — params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                label: Some("GpuFast pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some("detect_corners"),
                layout:              Some(&pipeline_layout),
                module:              &shader,
                entry_point:         "detect_corners",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:               None,
            });

        GpuFastDetector { pipeline, bgl, threshold, arc_length }
    }

    /// Detect corners in pyramid level 0 (convenience wrapper).
    pub fn detect(&self, gpu: &GpuDevice, level: &GpuPyramidLevel) -> Vec<Feature> {
        self.detect_at_level(gpu, level, 0)
    }

    /// Detect corners in an arbitrary pyramid level. The returned features
    /// have their `level` field set to `pyramid_level`.
    pub fn detect_at_level(
        &self,
        gpu:           &GpuDevice,
        level:         &GpuPyramidLevel,
        pyramid_level: usize,
    ) -> Vec<Feature> {
        let w = level.width;
        let h = level.height;
        let n_pixels = (w * h) as usize;

        // Allocate dense score buffer (zero-filled by wgpu).
        let score_buf_size = (n_pixels * std::mem::size_of::<f32>()) as u64;
        let score_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("GpuFast scores"),
            size:               score_buf_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = FastParams {
            img_width:  w,
            img_height: h,
            threshold:  self.threshold as f32,
            arc_length: self.arc_length as u32,
        };
        let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("GpuFast params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("GpuFast BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level.read_view) },
                wgpu::BindGroupEntry { binding: 1, resource: score_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        let (wg_x, wg_y) = gpu.dispatch_size(w, h);
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuFast dispatch") },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("detect_corners"), timestamp_writes: None },
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Readback buffer.
        let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("GpuFast readback"),
            size:               score_buf_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&score_buf, 0, &rb, 0, score_buf_size);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("GpuFast score readback failed");

        let mapped = slice.get_mapped_range();
        // SAFETY: buffer is f32-aligned, size = n_pixels * 4.
        let scores: &[f32] = bytemuck::cast_slice(&mapped);

        let features = scores.iter().enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(idx, &score)| {
                let x = (idx % w as usize) as f32;
                let y = (idx / w as usize) as f32;
                Feature { x, y, score, level: pyramid_level, id: 0 }
            })
            .collect();

        drop(mapped);
        rb.unmap();
        features
    }
}

// ---------------------------------------------------------------------------
// Tests — same suite as before, no changes needed in test logic
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::FastDetector;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::image::Image;

    fn run_gpu_test_in_subprocess(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args(["test", "--lib", "--", test_name, "--exact", "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed for {test_name}: {e}"));
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        print!("{stdout}"); eprint!("{stderr}");
        stdout + &stderr
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_no_corners_on_flat_image() {
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 20, 9);
        let features = detector.detect(&gpu, &pyr.levels[0]);
        assert!(features.is_empty(), "flat image should have no corners, got {}", features.len());
        println!("GPU_TEST_OK");
        drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_detects_bright_corner() {
        let mut pixels = vec![20u8; 64 * 64];
        for y in 20..44usize { for x in 20..44usize { pixels[y * 64 + x] = 220; } }
        let src = Image::<u8>::from_vec(64, 64, pixels);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 30, 9);
        let features = detector.detect(&gpu, &pyr.levels[0]);
        assert!(!features.is_empty(), "bright rectangle should produce corners");
        println!("GPU_TEST_OK");
        drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_matches_cpu() {
        let mut rng = 99991u32;
        let pixels: Vec<u8> = (0..128*128).map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        }).collect();
        let src = Image::<u8>::from_vec(128, 128, pixels.clone());

        let cpu_det = FastDetector::new(30, 9);
        let mut cpu_features = cpu_det.detect(&src);
        cpu_features.sort_by_key(|f| (f.y as u32, f.x as u32));

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 30, 9);
        let mut gpu_features = detector.detect(&gpu, &pyr.levels[0]);
        gpu_features.sort_by_key(|f| (f.y as u32, f.x as u32));

        eprintln!("[test] CPU: {} corners, GPU: {} corners", cpu_features.len(), gpu_features.len());
        assert_eq!(gpu_features.len(), cpu_features.len(),
            "count mismatch: GPU={} CPU={}", gpu_features.len(), cpu_features.len());

        for (i, (g, c)) in gpu_features.iter().zip(cpu_features.iter()).enumerate() {
            assert_eq!(g.x as u32, c.x as u32, "feature {i}: x mismatch");
            assert_eq!(g.y as u32, c.y as u32, "feature {i}: y mismatch");
            assert!((g.score - c.score).abs() < 0.5,
                "feature {i} score diff: GPU={:.3} CPU={:.3}", g.score, c.score);
        }
        println!("GPU_TEST_OK");
        drop(gpu_features); drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_no_corners_on_flat_image() {
        let out = run_gpu_test_in_subprocess("gpu::fast::tests::inner_no_corners_on_flat_image");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_detects_bright_corner() {
        let out = run_gpu_test_in_subprocess("gpu::fast::tests::inner_detects_bright_corner");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu() {
        let out = run_gpu_test_in_subprocess("gpu::fast::tests::inner_gpu_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
