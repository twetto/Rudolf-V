// gpu/pyramid.rs — GPU Gaussian image pyramid.
//
// Mirrors the CPU `Pyramid::build()` in pyramid.rs:
//   Level 0  : source image converted to f32 (no blur), uploaded from CPU.
//   Level n+1: Gaussian blur + 2× downsample of level n via compute shader.
//
// All pyramid levels are stored as R32Float textures on the GPU, with values
// in the same [0, 255] range as the CPU implementation. This makes
// pixel-perfect validation straightforward: GPU readback == CPU value.
//
//
// WHY R32FLOAT AND NOT R8UNORM?
// ──────────────────────────────
// The CPU pyramid stores `Image<f32>` at every level. Repeated Gaussian blur
// accumulates floating-point sums; storing as u8 after each level would
// introduce quantisation error that compounds over 4+ levels. R32Float
// preserves the same precision as the CPU and makes the per-pixel diff in
// tests meaningful (we expect < 0.5 error, not < 128/255).
//
//
// PIPELINE LIFETIME
// ─────────────────
// `GpuPyramidPipeline` is expensive to create (shader compilation). Create it
// once and reuse it every frame:
//
//   let pipeline = GpuPyramidPipeline::new(&gpu);
//   loop {
//       let pyr = pipeline.build(&gpu, &frame_image, 5, 1.0);
//       // ... use pyr.levels[i] in subsequent kernels
//   }
//
// `GpuPyramid` is cheap to create per frame — it's just textures and
// pre-recorded GPU commands.
//
//
// NEW WGPU / RUST CONCEPTS
// ─────────────────────────
// - `wgpu::BindGroupLayout` — describes the *types* of bindings a shader
//   expects (texture, storage texture, uniform buffer). Created once from the
//   pipeline, shared across bind groups.
//
// - `wgpu::BindGroup` — the *actual* resources for one dispatch. Created per
//   level pair because each level uses different input/output textures.
//
// - `texture_storage_2d` write access — requires `TextureUsages::STORAGE_BINDING`
//   on the texture AND `StorageTextureAccess::WriteOnly` in the BGL entry.
//
// - `include_str!` — embeds a file's contents as a `&'static str` at compile
//   time. The path is relative to the *source file*, not the crate root.
//
// - `bytemuck::bytes_of` — safely reinterprets a `#[repr(C)]` struct as
//   `&[u8]` for upload to a GPU buffer. Avoids unsafe pointer casts.

use wgpu::util::DeviceExt;

use crate::gpu::device::GpuDevice;
use crate::gpu::image::align_to;
use crate::image::Image;

// ---------------------------------------------------------------------------
// GPU pyramid level
// ---------------------------------------------------------------------------

/// One level of the GPU pyramid: an R32Float texture + views for reading
/// (TEXTURE_BINDING) and writing (STORAGE_BINDING).
pub struct GpuPyramidLevel {
    /// The underlying R32Float texture.
    pub texture: wgpu::Texture,
    /// View for binding as `texture_2d<f32>` input to the next level's shader.
    pub read_view: wgpu::TextureView,
    /// View for binding as `texture_storage_2d<r32float, write>` output.
    pub write_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl GpuPyramidLevel {
    /// Allocate a new R32Float texture.
    fn new(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST   // for level 0 CPU upload
                | wgpu::TextureUsages::COPY_SRC,  // for test readback
            view_formats: &[],
        });
        // Both views reference the full texture; the distinction is only in
        // how we bind them in the shader (read vs write).
        let read_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let write_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        GpuPyramidLevel { texture, read_view, write_view, width, height }
    }
}

// ---------------------------------------------------------------------------
// GpuPyramid
// ---------------------------------------------------------------------------

/// A Gaussian image pyramid resident on the GPU.
///
/// All levels are R32Float textures with values in [0, 255], matching the
/// CPU `Pyramid` which stores `Image<f32>`.
///
/// Create via [`GpuPyramidPipeline::build`].
pub struct GpuPyramid {
    /// Pyramid levels, finest (index 0) to coarsest. Level 0 is at the
    /// source image resolution; level n is approximately 1/2^n scale.
    pub levels: Vec<GpuPyramidLevel>,
}

impl GpuPyramid {
    /// Read one pyramid level back to CPU memory.
    ///
    /// **Expensive and synchronous** — stalls the GPU. Use only in tests.
    ///
    /// Returns a flat `Vec<f32>` of length `width * height`, row-major,
    /// no padding (stride == width), matching `Image<f32>::as_slice()`.
    pub fn readback_level(&self, gpu: &GpuDevice, level: usize) -> Vec<f32> {
        let lvl = &self.levels[level];

        // R32Float: 4 bytes per pixel.
        let bytes_per_pixel: u32 = 4;
        let aligned_bytes_per_row = align_to(lvl.width * bytes_per_pixel, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_size = (aligned_bytes_per_row * lvl.height) as u64;

        let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPyramid::readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuPyramid::readback") },
        );
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &lvl.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(lvl.height),
                },
            },
            wgpu::Extent3d {
                width: lvl.width,
                height: lvl.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let buf_slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("readback channel closed");
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("readback callback never fired")
            .expect("readback map failed");

        let mapped = buf_slice.get_mapped_range();
        let bytes_per_row_w = lvl.width as usize * 4;

        // Strip alignment padding and interpret bytes as f32.
        let mut out = vec![0.0f32; (lvl.width * lvl.height) as usize];
        for y in 0..lvl.height as usize {
            let src_byte_start = y * aligned_bytes_per_row as usize;
            let dst_start = y * lvl.width as usize;
            let src_bytes = &mapped[src_byte_start..src_byte_start + bytes_per_row_w];
            // SAFETY: src_bytes is aligned to 4 bytes because the GPU wrote
            // properly-aligned R32Float data; f32 and [u8; 4] have the same
            // size; and we're only reading (not mutating).
            let src_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    src_bytes.as_ptr() as *const f32,
                    lvl.width as usize,
                )
            };
            out[dst_start..dst_start + lvl.width as usize].copy_from_slice(src_f32);
        }
        drop(mapped);
        readback_buf.unmap();
        out
    }
}

// ---------------------------------------------------------------------------
// Kernel params uniform (must match WGSL struct layout exactly)
// ---------------------------------------------------------------------------

/// Gaussian kernel parameters uploaded as a uniform buffer.
///
/// Layout must match `PyramidParams` in `pyramid.wgsl`:
///   offset  0: dst_width  (u32)
///   offset  4: dst_height (u32)
///   offset  8: half_size  (u32)
///   offset 12: _pad       (u32)
///   offset 16: coeffs     (4 × vec4<f32> = 16 × f32)
///   total:  80 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PyramidParams {
    dst_width:  u32,
    dst_height: u32,
    half_size:  u32,
    _pad:       u32,
    /// coeffs[i/4][i%4] = Gaussian kernel coefficient for index i.
    /// Maximum half_size = 15 (16 coefficients), covering σ ≈ 5.
    coeffs: [[f32; 4]; 4],
}

impl PyramidParams {
    fn new(dst_width: u32, dst_height: u32, kernel: &[f32]) -> Self {
        // `kernel` comes from `gaussian_kernel_1d(half_size, sigma)`.
        // It has length 2*half_size+1; we only need indices 0..=half_size
        // (the right half, which equals the left half by symmetry).
        let full_len = kernel.len();
        assert!(full_len % 2 == 1, "kernel must have odd length");
        let half_size = (full_len - 1) / 2;
        assert!(half_size <= 15, "half_size > 15 — increase coeffs array size");

        // Extract the right half (index half_size .. end), which gives
        // coefficients for offsets 0, 1, 2, ..., half_size.
        let right_half = &kernel[half_size..];

        let mut coeffs = [[0.0f32; 4]; 4];
        for (i, &c) in right_half.iter().enumerate() {
            coeffs[i / 4][i % 4] = c;
        }

        PyramidParams {
            dst_width,
            dst_height,
            half_size: half_size as u32,
            _pad: 0,
            coeffs,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPyramidPipeline
// ---------------------------------------------------------------------------

/// Compiled GPU pipeline for pyramid construction.
///
/// Create once at startup; reuse every frame via [`GpuPyramidPipeline::build`].
///
/// Contains:
/// - The compiled `blur_downsample` compute pipeline.
/// - The `BindGroupLayout` used to create per-level bind groups.
pub struct GpuPyramidPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuPyramidPipeline {
    /// Compile the `pyramid.wgsl` shader and create the compute pipeline.
    ///
    /// Specialises workgroup size from the `GpuDevice`'s active profile
    /// via `PipelineCompilationOptions::constants`.
    pub fn new(gpu: &GpuDevice) -> Self {
        // `include_str!` embeds the WGSL source at compile time.
        // Path is relative to this source file (src/gpu/pyramid.rs →
        // src/shaders/pyramid.wgsl).
        //
        // naga (wgpu's WGSL compiler) does not yet support `override`
        // expressions inside @workgroup_size(), so we bake the workgroup
        // dimensions directly into the shader source via string replacement.
        // {{WG_X}} and {{WG_Y}} are placeholder tokens in the WGSL file.
        let shader_template = include_str!("../shaders/pyramid.wgsl");
        let shader_src = shader_template
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string());

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout: mirrors the @group(0) bindings in pyramid.wgsl.
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuPyramid BGL"),
            entries: &[
                // Binding 0 — input texture (read as texture_2d<f32>)
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
                // Binding 1 — output texture (storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 2 — kernel params uniform buffer
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
                label: Some("GpuPyramid pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("blur_downsample"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "blur_downsample",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        GpuPyramidPipeline { pipeline, bgl }
    }

    /// Build a GPU pyramid from a CPU source image.
    ///
    /// # Arguments
    /// - `src`        — CPU source image (`Image<u8>`).
    /// - `num_levels` — number of pyramid levels (must be ≥ 1).
    /// - `sigma`      — Gaussian blur sigma. Must match what the CPU pipeline
    ///                  uses so the GPU and CPU pyramids agree pixel-for-pixel.
    ///
    /// # How level 0 is uploaded
    /// The source pixels are converted from `u8` to `f32` on the CPU (raw
    /// values, not normalised — same as `u8::to_f32()` in the CPU pipeline).
    /// This matches `Pyramid::build()` which stores level 0 as `Image<f32>`
    /// with raw u8 values.
    pub fn build(
        &self,
        gpu: &GpuDevice,
        src: &Image<u8>,
        num_levels: usize,
        sigma: f32,
    ) -> GpuPyramid {
        assert!(num_levels >= 1, "pyramid must have at least 1 level");

        // Compute Gaussian kernel — same formula as pyramid.rs so GPU and
        // CPU results agree. We reproduce the kernel computation here to
        // avoid a dependency on the convolution module.
        let kernel = gaussian_kernel_1d_for_gpu(sigma);

        let mut levels = Vec::with_capacity(num_levels);

        // --- Level 0: upload from CPU as R32Float ---
        let w0 = src.width() as u32;
        let h0 = src.height() as u32;
        let level0 = GpuPyramidLevel::new(&gpu.device, w0, h0, "pyramid level 0");
        upload_f32_level(gpu, &level0, src);
        levels.push(level0);

        // --- Levels 1..(num_levels-1): blur + downsample via compute ---
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuPyramid::build") },
        );

        for lvl_idx in 1..num_levels {
            let prev = &levels[lvl_idx - 1];
            let dst_w = (prev.width / 2).max(1);
            let dst_h = (prev.height / 2).max(1);

            let label = format!("pyramid level {lvl_idx}");
            let curr = GpuPyramidLevel::new(&gpu.device, dst_w, dst_h, &label);

            // Build the uniform buffer for this level's kernel params.
            let params = PyramidParams::new(dst_w, dst_h, &kernel);
            let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PyramidParams"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Create a bind group for this level pair.
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pyramid level bind group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&prev.read_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&curr.write_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            // Dispatch the compute pass for this level.
            {
                let mut pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("blur_downsample"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                let (dx, dy) = gpu.dispatch_size(dst_w, dst_h);
                pass.dispatch_workgroups(dx, dy, 1);
            }

            levels.push(curr);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        GpuPyramid { levels }
    }
}

// ---------------------------------------------------------------------------
// Level-0 upload (CPU u8 → GPU R32Float)
// ---------------------------------------------------------------------------

/// Upload a CPU `Image<u8>` to a pre-allocated R32Float `GpuPyramidLevel`.
///
/// Converts each pixel from `u8` to `f32` (raw value, not normalised) and
/// handles stride-compaction, exactly as `GpuImage::upload` does for u8.
fn upload_f32_level(gpu: &GpuDevice, dst: &GpuPyramidLevel, src: &Image<u8>) {
    let width = src.width() as u32;
    let height = src.height() as u32;

    // R32Float: 4 bytes per pixel. Alignment applies to bytes, not pixels.
    let bytes_per_pixel: u32 = 4;
    let aligned_bytes_per_row =
        align_to(width * bytes_per_pixel, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    let staging_byte_len = (aligned_bytes_per_row * height) as usize;
    let mut staging = vec![0u8; staging_byte_len];

    let src_data = src.as_slice();
    let src_stride = src.stride();

    for y in 0..height as usize {
        let src_row_start = y * src_stride;
        // Each f32 occupies 4 bytes; destination row is aligned_bytes_per_row bytes wide.
        let dst_row_byte_start = y * aligned_bytes_per_row as usize;

        for x in 0..width as usize {
            let f32_val: f32 = src_data[src_row_start + x] as f32;
            let f32_bytes = f32_val.to_le_bytes();
            let dst_byte_off = dst_row_byte_start + x * 4;
            staging[dst_byte_off..dst_byte_off + 4].copy_from_slice(&f32_bytes);
        }
    }

    let staging_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pyramid level 0 staging"),
        contents: &staging,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let mut encoder = gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("upload_f32_level") },
    );
    encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: &staging_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::ImageCopyTexture {
            texture: &dst.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));
}

// ---------------------------------------------------------------------------
// Kernel generation (reproduced from convolution.rs to avoid coupling)
// ---------------------------------------------------------------------------

/// Compute a normalised 1D Gaussian kernel matching `gaussian_kernel_1d` in
/// convolution.rs. Returns a full-length kernel (length = 2*half_size + 1).
///
/// Formula: k[i] = exp(−i² / (2σ²)), normalised so Σk = 1.
/// half_size = ceil(3σ).max(1) — the same formula used in PyramidScratch::new.
fn gaussian_kernel_1d_for_gpu(sigma: f32) -> Vec<f32> {
    let half_size = (3.0 * sigma).ceil().max(1.0) as usize;
    let len = 2 * half_size + 1;
    let mut k = vec![0.0f32; len];
    let two_sigma_sq = 2.0 * sigma * sigma;
    for i in 0..=half_size as isize {
        let v = (-(i * i) as f32 / two_sigma_sq).exp();
        k[half_size as usize - i as usize] = v;
        k[half_size as usize + i as usize] = v;
    }
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    // ---- Pure CPU tests (no GPU) -------------------------------------------

    #[test]
    fn test_pyramid_params_layout() {
        // Verify the struct is 80 bytes to match the WGSL uniform layout.
        assert_eq!(std::mem::size_of::<PyramidParams>(), 80);
    }

    #[test]
    fn test_pyramid_params_coefficients() {
        // sigma=1.0: half_size=3, kernel has 7 taps.
        // Right half (indices 0..=3) maps to coeffs[0][0..3] and coeffs[0][3].
        let kernel = gaussian_kernel_1d_for_gpu(1.0);
        assert_eq!(kernel.len(), 7);
        let half_size = 3;
        let params = PyramidParams::new(320, 240, &kernel);
        assert_eq!(params.half_size, half_size as u32);
        // coeffs[0][0] should be the centre weight (largest).
        assert!(params.coeffs[0][0] > params.coeffs[0][1],
            "centre weight should be largest");
        // All weights should be positive.
        for row in &params.coeffs {
            for &c in row {
                assert!(c >= 0.0);
            }
        }
    }

    #[test]
    fn test_kernel_matches_cpu() {
        // Verify our GPU kernel matches the CPU half_size formula.
        let sigma = 1.5f32;
        let expected_half_size = (3.0 * sigma).ceil().max(1.0) as usize;
        let k = gaussian_kernel_1d_for_gpu(sigma);
        assert_eq!(k.len(), 2 * expected_half_size + 1);
        // Must normalise to 1.
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "kernel sum = {sum}, expected 1.0");
        // Must be symmetric.
        let half = expected_half_size;
        for i in 0..half {
            assert!((k[i] - k[k.len() - 1 - i]).abs() < 1e-7,
                "kernel not symmetric at index {i}");
        }
    }

    // ---- GPU integration tests (subprocess-isolated) -----------------------
    //
    // Same subprocess isolation pattern as gpu::device and gpu::image.
    // The `inner_*` tests run in a child process; `test_*` wrappers spawn
    // the child and assert "GPU_TEST_OK" appears in the output.

    #[cfg(test)]
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

    // Inner tests ─────────────────────────────────────────────────────────────

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_pipeline_creation() {
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let _pipeline = GpuPyramidPipeline::new(&gpu);
        println!("GPU_TEST_OK");
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_level0_round_trip() {
        // Upload level 0, read back, check pixel values are exact u8→f32.
        let pixels: Vec<u8> = (0u8..=99).collect();
        let src = Image::<u8>::from_vec(10, 10, pixels.clone());

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 1, 1.0);

        let readback = pyr.readback_level(&gpu, 0);
        assert_eq!(readback.len(), 100);
        for (i, (&expected, &got)) in pixels.iter().zip(readback.iter()).enumerate() {
            let diff = (expected as f32 - got).abs();
            assert!(diff < 1e-3,
                "level0 pixel {i}: expected {}, got {got}", expected as f32);
        }
        println!("GPU_TEST_OK");
        drop(pyr);
        drop(pipeline);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_pyramid_dimensions() {
        // Verify level dimensions halve at each step.
        let src = Image::<u8>::new(640, 480);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 5, 1.0);

        assert_eq!(pyr.levels[0].width,  640); assert_eq!(pyr.levels[0].height, 480);
        assert_eq!(pyr.levels[1].width,  320); assert_eq!(pyr.levels[1].height, 240);
        assert_eq!(pyr.levels[2].width,  160); assert_eq!(pyr.levels[2].height, 120);
        assert_eq!(pyr.levels[3].width,   80); assert_eq!(pyr.levels[3].height,  60);
        assert_eq!(pyr.levels[4].width,   40); assert_eq!(pyr.levels[4].height,  30);

        println!("GPU_TEST_OK");
        drop(pyr);
        drop(pipeline);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_constant_image_preserved() {
        // A constant image should remain constant at all pyramid levels.
        let src = Image::<u8>::from_vec(64, 64, vec![128u8; 64 * 64]);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &src, 4, 1.0);

        for lvl in 0..4 {
            let data = pyr.readback_level(&gpu, lvl);
            for (i, &v) in data.iter().enumerate() {
                assert!((v - 128.0).abs() < 0.5,
                    "level {lvl} pixel {i}: expected 128, got {v}");
            }
        }
        println!("GPU_TEST_OK");
        drop(pyr);
        drop(pipeline);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_matches_cpu() {
        // The most important test: GPU pyramid must agree with CPU pyramid.
        //
        // We use a pseudo-random image to avoid the degenerate constant case.
        // Error tolerance is 0.5 (half an integer step), accounting for the
        // difference in floating-point associativity between CPU and GPU.
        use crate::pyramid::Pyramid;

        // Simple LCG for deterministic test data without extra deps.
        let mut rng = 12345u32;
        let pixels: Vec<u8> = (0..128 * 128)
            .map(|_| { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); (rng >> 24) as u8 })
            .collect();

        let src = Image::<u8>::from_vec(128, 128, pixels.clone());
        let cpu_pyr = Pyramid::build(&src, 4, 1.0);

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let gpu_pyr = pipeline.build(&gpu, &src, 4, 1.0);

        let mut max_err = 0.0f32;
        for lvl in 0..4 {
            let gpu_data = gpu_pyr.readback_level(&gpu, lvl);
            let cpu_level = cpu_pyr.level(lvl);
            let cpu_data = cpu_level.as_slice();

            assert_eq!(gpu_data.len(), cpu_data.len(),
                "level {lvl} length mismatch: GPU {} vs CPU {}", gpu_data.len(), cpu_data.len());

            for (i, (&g, &c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
                let diff = (g - c).abs();
                if diff > max_err { max_err = diff; }
                assert!(diff < 0.5,
                    "level {lvl} pixel {i}: GPU={g:.4} CPU={c:.4} diff={diff:.4}");
            }
        }
        eprintln!("[test] max GPU/CPU pyramid error: {max_err:.4}");
        println!("GPU_TEST_OK");
        drop(gpu_pyr);
        drop(pipeline);
        drop(gpu);
    }

    // Outer wrappers ──────────────────────────────────────────────────────────

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_pipeline_creation() {
        let out = run_gpu_test_in_subprocess("gpu::pyramid::tests::inner_pipeline_creation");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_level0_round_trip() {
        let out = run_gpu_test_in_subprocess("gpu::pyramid::tests::inner_level0_round_trip");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_pyramid_dimensions() {
        let out = run_gpu_test_in_subprocess("gpu::pyramid::tests::inner_pyramid_dimensions");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_constant_image_preserved() {
        let out = run_gpu_test_in_subprocess("gpu::pyramid::tests::inner_constant_image_preserved");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu() {
        let out = run_gpu_test_in_subprocess("gpu::pyramid::tests::inner_gpu_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
