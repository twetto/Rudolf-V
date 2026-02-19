// gpu/fast.rs — GPU FAST-N corner detector.
//
// Wraps the `fast.wgsl` compute shader. The public API mirrors `FastDetector`
// in fast.rs so callers can swap CPU ↔ GPU without changing downstream code.
//
//
// OUTPUT STRATEGY: atomic counter + pre-allocated keypoint buffer
// ───────────────────────────────────────────────────────────────
// The number of detected corners isn't known before the shader runs, so we
// can't allocate exactly the right output buffer. Two storage buffers are
// used:
//
//   counter_buf  — one u32, initialised to 0. Threads atomically increment
//                  it to claim output slots.
//   keypoints_buf — pre-allocated for MAX_FEATURES GpuFeature entries.
//
// After the dispatch, we read `counter` to find how many corners were found,
// then read the first `counter` entries from `keypoints_buf`.
//
// If more corners are found than MAX_FEATURES, the excess are silently
// dropped (the shader guards against buffer overflow). In practice, with
// NMS applied afterward, MAX_FEATURES = width * height / 8 is generous.
//
//
// NEW WGPU CONCEPTS
// ─────────────────
// - `BufferUsages::STORAGE` — enables binding as `var<storage>` in WGSL.
//   Distinct from UNIFORM (which is read-only and size-limited to 64 KiB).
// - Zeroing a buffer before each dispatch — use `queue.write_buffer` or
//   create it fresh. We re-create the counter buffer (1 u32) each frame
//   since it's tiny. The keypoints buffer is reused across frames.
// - Reading a storage buffer back requires `BufferUsages::COPY_SRC` on the
//   storage buffer and a separate `MAP_READ | COPY_DST` readback buffer.

use wgpu::util::DeviceExt;

use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramidLevel;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of keypoints the GPU output buffer can hold.
/// Sized generously — real images with NMS produce far fewer.
/// 65536 × 16 bytes = 1 MiB, acceptable GPU allocation.
const MAX_FEATURES: u32 = 65_536;

// ---------------------------------------------------------------------------
// GPU-side feature layout (must match WGSL struct GpuFeature)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFeature {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Uniform params (must match WGSL struct FastParams)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FastParams {
    img_width:    u32,
    img_height:   u32,
    threshold:    f32,
    arc_length:   u32,
    max_features: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// GpuFastDetector
// ---------------------------------------------------------------------------

/// GPU FAST-N corner detector.
///
/// Create once; call [`detect`] each frame. The keypoint output buffer is
/// allocated once and reused — no per-frame GPU allocations after init.
pub struct GpuFastDetector {
    pipeline:     wgpu::ComputePipeline,
    bgl:          wgpu::BindGroupLayout,
    /// Pre-allocated keypoint output buffer (STORAGE | COPY_SRC).
    keypoints_buf: wgpu::Buffer,
    /// Intensity difference threshold (same scale as u8 pixel values).
    pub threshold:  u8,
    /// Arc length N for FAST-N (9..=12).
    pub arc_length: usize,
}

impl GpuFastDetector {
    /// Create a GPU FAST detector.
    ///
    /// # Panics
    /// Panics if `arc_length` is not in 9..=12, matching the CPU detector.
    pub fn new(gpu: &GpuDevice, threshold: u8, arc_length: usize) -> Self {
        assert!(
            (9..=12).contains(&arc_length),
            "arc_length must be 9..=12 (got {arc_length})"
        );

        // Bake workgroup size into shader (naga doesn't support override in
        // @workgroup_size — same workaround as pyramid.rs).
        let shader_template = include_str!("../shaders/fast.wgsl");
        let shader_src = shader_template
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string());

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fast.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout — mirrors @group(0) bindings in fast.wgsl.
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuFast BGL"),
            entries: &[
                // Binding 0 — input texture (R32Float pyramid level)
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
                // Binding 1 — atomic counter (storage read_write)
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
                // Binding 2 — keypoints output (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3 — params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                label: Some("detect_corners"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "detect_corners",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Pre-allocate keypoint output buffer.
        let keypoints_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast keypoints"),
            size: (MAX_FEATURES as usize * std::mem::size_of::<GpuFeature>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuFastDetector { pipeline, bgl, keypoints_buf, threshold, arc_length }
    }

    /// Detect FAST corners in a GPU pyramid level.
    ///
    /// Returns CPU-side `Vec<Feature>` (same type as the CPU detector), with
    /// `level = 0` and `id = 0`. Ordering is non-deterministic (GPU threads
    /// race for output slots) — sort or apply NMS afterward if needed.
    ///
    /// This is synchronous: it submits a dispatch, then blocks until the GPU
    /// copy completes and the results are mapped to CPU.
    pub fn detect(&self, gpu: &GpuDevice, level: &GpuPyramidLevel) -> Vec<Feature> {
        self.detect_at_level(gpu, level, 0)
    }

    /// Detect corners, tagging features with the given pyramid level index.
    pub fn detect_at_level(
        &self,
        gpu: &GpuDevice,
        level: &GpuPyramidLevel,
        pyramid_level: usize,
    ) -> Vec<Feature> {
        let w = level.width;
        let h = level.height;

        // --- Zero the atomic counter ---
        // We create a fresh 4-byte buffer each call (tiny, constant cost).
        // Using write_buffer on a reused buffer would also work but requires
        // an extra COPY_DST usage flag and a submit/poll cycle.
        let counter_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuFast counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // --- Params uniform ---
        let params = FastParams {
            img_width:    w,
            img_height:   h,
            threshold:    self.threshold as f32,
            arc_length:   self.arc_length as u32,
            max_features: MAX_FEATURES,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuFast params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- Bind group ---
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuFast bind group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&level.read_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.keypoints_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // --- Dispatch ---
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuFast::detect") },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("detect_corners"),
                    timestamp_writes: None,
                },
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (dx, dy) = gpu.dispatch_size(w, h);
            pass.dispatch_workgroups(dx, dy, 1);
        }

        // --- Readback: counter then keypoints ---
        // Two readback buffers: one for the counter (4 bytes), one for
        // the keypoints (up to MAX_FEATURES × 16 bytes).
        let counter_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast counter readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kp_rb_size =
            (MAX_FEATURES as usize * std::mem::size_of::<GpuFeature>()) as u64;
        let kp_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuFast keypoints readback"),
            size: kp_rb_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&counter_buf, 0, &counter_rb, 0, 4);
        encoder.copy_buffer_to_buffer(
            &self.keypoints_buf, 0, &kp_rb, 0, kp_rb_size,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map counter.
        let n_features = {
            let s = counter_rb.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            s.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().expect("counter map failed");
            let mapped = s.get_mapped_range();
            let count = u32::from_le_bytes(mapped[0..4].try_into().unwrap());
            drop(mapped);
            counter_rb.unmap();
            count.min(MAX_FEATURES) as usize
        };

        if n_features == 0 {
            return Vec::new();
        }

        // Map keypoints (only read the populated prefix).
        let kp_byte_len = n_features * std::mem::size_of::<GpuFeature>();
        let s = kp_rb.slice(..kp_byte_len as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        s.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("keypoints map failed");

        let mapped = s.get_mapped_range();
        // SAFETY: GpuFeature is repr(C), Pod, and we allocated exactly
        // n_features × size_of::<GpuFeature>() bytes.
        let gpu_features: &[GpuFeature] = bytemuck::cast_slice(&mapped);

        let features = gpu_features.iter().map(|gf| Feature {
            x:     gf.x,
            y:     gf.y,
            score: gf.score,
            level: pyramid_level,
            id:    0,
        }).collect();

        drop(mapped);
        kp_rb.unmap();

        features
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::FastDetector;
    use crate::image::Image;
    use crate::gpu::pyramid::GpuPyramidPipeline;

    fn run_gpu_test_in_subprocess(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args([
                "test", "--lib", "--",
                test_name, "--exact", "--ignored", "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed for {test_name}: {e}"));
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        print!("{stdout}");
        eprint!("{stderr}");
        stdout + &stderr
    }

    // ---- Inner tests --------------------------------------------------------

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_no_corners_on_flat_image() {
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 20, 9);
        let features = detector.detect(&gpu, &pyr.levels[0]);
        assert!(features.is_empty(),
            "flat image should have no corners, got {}", features.len());
        println!("GPU_TEST_OK");
        drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_detects_bright_corner() {
        // Bright rectangle on dark background — the rectangle corners have
        // a large contiguous dark arc on the Bresenham circle, making them
        // reliable FAST corners. Checkerboards do NOT work because the
        // alternating bright/dark quadrants break the contiguous arc.
        let mut pixels = vec![20u8; 64 * 64];
        for y in 20..44usize {
            for x in 20..44usize {
                pixels[y * 64 + x] = 220;
            }
        }
        let src = Image::<u8>::from_vec(64, 64, pixels);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 30, 9);
        let features = detector.detect(&gpu, &pyr.levels[0]);
        assert!(!features.is_empty(),
            "bright rectangle should produce corners");
        eprintln!("[test] found {} corners", features.len());
        println!("GPU_TEST_OK");
        drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_matches_cpu() {
        // The key validation: GPU must find the same corners as CPU.
        // We compare sets of (x, y) positions — ordering differs because
        // GPU threads race for output slots.
        //
        // Tolerance: exact position match (both are integer pixel centres).
        // Score tolerance: 0.5 (same rounding as pyramid validation).

        // Pseudo-random test image.
        let mut rng = 99991u32;
        let pixels: Vec<u8> = (0..128 * 128)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 24) as u8
            })
            .collect();
        let src = Image::<u8>::from_vec(128, 128, pixels.clone());

        // CPU reference.
        let cpu_det = FastDetector::new(30, 9);
        let mut cpu_features = cpu_det.detect(&src);
        cpu_features.sort_by(|a, b| {
            (a.y as u32, a.x as u32).cmp(&(b.y as u32, b.x as u32))
        });

        // GPU.
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        // Build a 1-level pyramid so level 0 is just the u8→f32 upload.
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);
        let detector = GpuFastDetector::new(&gpu, 30, 9);
        let mut gpu_features = detector.detect(&gpu, &pyr.levels[0]);
        gpu_features.sort_by(|a, b| {
            (a.y as u32, a.x as u32).cmp(&(b.y as u32, b.x as u32))
        });

        eprintln!("[test] CPU: {} corners, GPU: {} corners",
            cpu_features.len(), gpu_features.len());

        // Same count.
        assert_eq!(
            gpu_features.len(), cpu_features.len(),
            "corner count mismatch: GPU={} CPU={}",
            gpu_features.len(), cpu_features.len()
        );

        // Same positions and scores.
        for (i, (g, c)) in gpu_features.iter().zip(cpu_features.iter()).enumerate() {
            assert_eq!(g.x as u32, c.x as u32,
                "feature {i}: x mismatch GPU={} CPU={}", g.x, c.x);
            assert_eq!(g.y as u32, c.y as u32,
                "feature {i}: y mismatch GPU={} CPU={}", g.y, c.y);
            let score_diff = (g.score - c.score).abs();
            assert!(score_diff < 0.5,
                "feature {i} at ({},{}) score diff too large: GPU={:.3} CPU={:.3}",
                g.x, g.y, g.score, c.score);
        }

        println!("GPU_TEST_OK");
        drop(gpu_features); drop(pyr); drop(detector); drop(pipeline); drop(gpu);
    }

    // ---- Outer wrappers -----------------------------------------------------

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_no_corners_on_flat_image() {
        let out = run_gpu_test_in_subprocess(
            "gpu::fast::tests::inner_no_corners_on_flat_image");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_detects_bright_corner() {
        let out = run_gpu_test_in_subprocess(
            "gpu::fast::tests::inner_detects_bright_corner");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu() {
        let out = run_gpu_test_in_subprocess(
            "gpu::fast::tests::inner_gpu_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
