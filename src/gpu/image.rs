// gpu/image.rs — GPU image representation and CPU→GPU upload.
//
// RESPONSIBILITIES
// ─────────────────
// 1. `GpuImage` — a grayscale frame resident on the GPU as a wgpu::Texture.
//    This is the input to every subsequent GPU kernel (pyramid, FAST, KLT).
//
// 2. `GpuImage::upload()` — copy a CPU `Image<u8>` to the GPU via a staging
//    buffer, handling the stride-compaction problem described below.
//
// 3. Round-trip validation — read the texture back to CPU and assert
//    pixel-perfect agreement with the original. Used only in tests.
//
//
// THE STRIDE-COMPACTION PROBLEM
// ──────────────────────────────
// Our CPU `Image<u8>` may have stride > width (alignment padding per row):
//
//   CPU layout (stride=5, width=4):
//     row0: [p00 p01 p02 p03 _padding_]
//     row1: [p10 p11 p12 p13 _padding_]
//     ...
//
// wgpu's `copy_buffer_to_texture` requires that the source buffer has rows
// packed at `bytes_per_row` alignment (must be a multiple of 256). The
// stride in the CPU image is measured in *elements*, not bytes, and is
// typically far from 256-aligned.
//
// Strategy: always compact into a staging buffer before upload.
//   - Allocate a staging buffer sized `width * height` bytes.
//   - Copy each row's active pixels (indices [y*stride .. y*stride+width])
//     contiguously, skipping padding.
//   - Set `bytes_per_row` to `width` rounded up to the nearest 256-byte
//     boundary.
//
// This means one staging buffer write per upload, which is acceptable —
// uploads happen once per frame and the CPU-side memcpy is bandwidth-bound,
// not latency-bound.
//
//
// NEW RUST CONCEPTS
// ──────────────────
// - `wgpu::Texture` vs `wgpu::Buffer` — textures have dimensionality and
//   sampling metadata; buffers are flat byte arrays. We use a texture so
//   the pyramid kernel can sample with hardware bilinear interpolation.
// - `wgpu::TextureView` — a "window" into a texture (which mip/layer to
//   use). Shaders bind TextureViews, not Textures directly.
// - `wgpu::BufferUsages::MAP_READ | COPY_DST` — the combination needed to
//   map a buffer back to CPU for readback. `MAP_READ` alone is insufficient.
// - `buffer.slice(..).map_async(MapMode::Read, cb)` — asynchronous map
//   request. We poll the device until the callback fires.
// - `bytes_per_row` must be a multiple of
//   `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (= 256). Violating this panics
//   at runtime inside wgpu's validation layer.

use wgpu::util::DeviceExt;

use crate::gpu::device::GpuDevice;
use crate::image::Image;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// wgpu requires that the number of bytes per row in a buffer→texture copy
/// is a multiple of this value. Padding is added to the staging buffer rows
/// to meet the alignment.
const COPY_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

// ---------------------------------------------------------------------------
// GpuImage
// ---------------------------------------------------------------------------

/// A grayscale `u8` image resident on the GPU as a 2D texture.
///
/// Created via [`GpuImage::upload`]. The texture uses format
/// `R8Unorm` — one channel, 8 bits, normalised to [0, 1] in shaders
/// (WGSL: `textureLoad` returns a `vec4<f32>` with `.r` in [0, 1]).
///
/// # Lifetime
/// `GpuImage` borrows nothing and owns its wgpu resources. Dropping it
/// releases the GPU texture memory.
pub struct GpuImage {
    /// The underlying wgpu texture on the GPU.
    pub texture: wgpu::Texture,
    /// Default view: full texture, all mips, all layers.
    /// Passed to compute pipelines as a binding.
    pub view: wgpu::TextureView,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl GpuImage {
    // -----------------------------------------------------------------------
    // Upload
    // -----------------------------------------------------------------------

    /// Upload a CPU `Image<u8>` to the GPU.
    ///
    /// # What this does
    /// 1. Allocates a `GpuImage` (empty `R8Unorm` 2D texture).
    /// 2. Creates a staging buffer with compacted pixel rows
    ///    (stride padding removed, rows aligned to 256 bytes for wgpu).
    /// 3. Submits a `copy_buffer_to_texture` command.
    /// 4. Returns the `GpuImage` immediately — the copy runs asynchronously
    ///    on the GPU timeline. If you need the data to be visible to a
    ///    subsequent kernel, submit them in the same `CommandEncoder` or
    ///    use `queue.submit` + `device.poll(Wait)` between them.
    ///
    /// # Stride handling
    /// If the CPU image has `stride > width`, pixel rows are compacted
    /// (padding stripped) before being written to the staging buffer.
    /// If `stride == width`, rows are copied directly.
    pub fn upload(gpu: &GpuDevice, src: &Image<u8>) -> Self {
        let width = src.width() as u32;
        let height = src.height() as u32;

        // --- Create destination texture ---
        //
        // TextureUsages we need:
        //   TEXTURE_BINDING — bind to compute shaders for sampling.
        //   COPY_DST        — accept data from a buffer copy.
        //   COPY_SRC        — allow readback to a buffer (for tests).
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GpuImage"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // R8Unorm: single-channel u8 stored as float [0,1] in shaders.
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Build the staging buffer ---
        //
        // wgpu requires `bytes_per_row` to be a multiple of 256.
        // Round width up to the next 256-byte boundary.
        let aligned_bytes_per_row = align_to(width, COPY_ALIGNMENT);
        let staging_size = (aligned_bytes_per_row * height) as usize;

        let mut staging: Vec<u8> = vec![0u8; staging_size];

        // Compact rows from the CPU image into the staging buffer.
        // Each destination row is `aligned_bytes_per_row` bytes wide;
        // only the first `width` bytes carry real pixel data.
        let src_data = src.as_slice();
        let src_stride = src.stride();
        for y in 0..height as usize {
            let src_row_start = y * src_stride;
            let dst_row_start = y * aligned_bytes_per_row as usize;
            staging[dst_row_start..dst_row_start + width as usize]
                .copy_from_slice(&src_data[src_row_start..src_row_start + width as usize]);
        }

        // Upload the staging buffer as an initialised buffer (avoids a
        // separate write_buffer call).
        let staging_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuImage::staging"),
            contents: &staging,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // --- Submit copy command ---
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::upload"),
            });

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
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        GpuImage { texture, view, width, height }
    }

    // -----------------------------------------------------------------------
    // Readback (for tests / debug)
    // -----------------------------------------------------------------------

    /// Read the GPU texture back to CPU memory.
    ///
    /// This is an **expensive, synchronous** operation — it stalls the GPU
    /// pipeline and blocks the CPU until the copy is complete. Use only in
    /// tests or offline debug tools, never on the hot path.
    ///
    /// Returns a flat `Vec<u8>` of length `width * height`, in row-major
    /// order with no padding (stride == width).
    pub fn readback(&self, gpu: &GpuDevice) -> Vec<u8> {
        let aligned_bytes_per_row = align_to(self.width, COPY_ALIGNMENT);
        let readback_size = (aligned_bytes_per_row * self.height) as u64;

        // Allocate a readback buffer (MAP_READ + COPY_DST).
        let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuImage::readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encode the texture → buffer copy.
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuImage::readback"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer — this is async in wgpu's API but we block here
        // via device.poll(Wait) after requesting the map.
        let buf_slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("readback channel closed");
        });

        // Poll until the GPU copy is done and the map callback fires.
        gpu.device.poll(wgpu::Maintain::Wait);
        receiver.recv().expect("readback map callback never fired")
            .expect("readback map failed");

        // Extract pixel data, stripping the alignment padding from each row.
        let mapped = buf_slice.get_mapped_range();
        let mut out = vec![0u8; (self.width * self.height) as usize];
        for y in 0..self.height as usize {
            let src_start = y * aligned_bytes_per_row as usize;
            let dst_start = y * self.width as usize;
            out[dst_start..dst_start + self.width as usize].copy_from_slice(
                &mapped[src_start..src_start + self.width as usize],
            );
        }
        drop(mapped);
        readback_buf.unmap();

        out
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `alignment`.
/// Used to compute `aligned_bytes_per_row` for wgpu copy operations.
///
/// Examples:
///   align_to(100, 256) = 256
///   align_to(256, 256) = 256
///   align_to(257, 256) = 512
///   align_to(640, 256) = 768   (rounds up to next multiple)
#[inline]
pub(crate) fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) / alignment * alignment
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    // ---- align_to (pure, no GPU needed) ------------------------------------

    #[test]
    fn test_align_to_already_aligned() {
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(512, 256), 512);
        assert_eq!(align_to(768, 256), 768);
    }

    #[test]
    fn test_align_to_rounds_up() {
        assert_eq!(align_to(1, 256), 256);
        assert_eq!(align_to(100, 256), 256);
        assert_eq!(align_to(255, 256), 256);
        assert_eq!(align_to(257, 256), 512);
        assert_eq!(align_to(641, 256), 768);
    }

    #[test]
    fn test_align_to_zero() {
        // Zero should stay zero (0 + 256 - 1 = 255; 255 / 256 = 0; 0 * 256 = 0).
        assert_eq!(align_to(0, 256), 0);
    }

    // ---- Staging buffer compaction (pure, no GPU) --------------------------
    //
    // Verify the row-compaction logic independently of wgpu by reproducing
    // the same loop and checking the output.

    #[test]
    fn test_staging_compaction_no_padding() {
        // stride == width: no padding, rows should copy verbatim.
        let img = Image::<u8>::from_vec(
            2,   // width
            3,   // height
            vec![1, 2, 3, 4, 5, 6],
        );
        let aligned = align_to(2, 256);
        let mut staging = vec![0u8; (aligned * 3) as usize];
        for y in 0..3usize {
            let src_start = y * img.stride();
            let dst_start = y * aligned as usize;
            staging[dst_start..dst_start + 2]
                .copy_from_slice(&img.as_slice()[src_start..src_start + 2]);
        }
        // First 2 bytes of each aligned row should hold the pixel data.
        assert_eq!(staging[0], 1);
        assert_eq!(staging[1], 2);
        assert_eq!(staging[aligned as usize], 3);
        assert_eq!(staging[aligned as usize + 1], 4);
        assert_eq!(staging[2 * aligned as usize], 5);
        assert_eq!(staging[2 * aligned as usize + 1], 6);
    }

    #[test]
    fn test_staging_compaction_with_padding() {
        // stride = 4, width = 3: one padding byte per row.
        // Pixel data: row0=[10,20,30,_], row1=[40,50,60,_].
        let img = Image::<u8>::from_vec_with_stride(
            3,   // width
            2,   // height
            4,   // stride > width
            vec![10, 20, 30, 0,
                 40, 50, 60, 0],
        );
        let aligned = align_to(3, 256); // = 256
        let mut staging = vec![0u8; (aligned * 2) as usize];
        for y in 0..2usize {
            let src_start = y * img.stride();
            let dst_start = y * aligned as usize;
            staging[dst_start..dst_start + 3]
                .copy_from_slice(&img.as_slice()[src_start..src_start + 3]);
        }
        assert_eq!(&staging[0..3], &[10, 20, 30]);
        assert_eq!(&staging[aligned as usize..aligned as usize + 3], &[40, 50, 60]);
        // Padding region of first row should be zeros (never written).
        assert_eq!(&staging[3..aligned as usize], &vec![0u8; (aligned - 3) as usize]);
    }

    // ---- GPU round-trip tests (subprocess-isolated) ------------------------
    //
    // Same subprocess isolation pattern as gpu::device — dzn crashes on exit.
    // The inner_* tests run inside a child process; outer test_* wrappers
    // spawn the child and assert "GPU_TEST_OK" appears in the output.

    #[cfg(test)]
    fn run_gpu_test_in_subprocess(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args([
                "test", "--lib", "--",
                test_name, "--exact", "--ignored", "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn subprocess for {test_name}: {e}"));
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        print!("{stdout}");
        eprint!("{stderr}");
        stdout + &stderr
    }

    // Inner tests ─────────────────────────────────────────────────────────────

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_upload_round_trip_no_padding() {
        // 4×3 image, stride == width (contiguous rows).
        let pixels: Vec<u8> = (0u8..12).collect();
        let src = Image::<u8>::from_vec(4, 3, pixels.clone());

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let gpu_img = GpuImage::upload(&gpu, &src);

        assert_eq!(gpu_img.width, 4);
        assert_eq!(gpu_img.height, 3);

        let readback = gpu_img.readback(&gpu);
        assert_eq!(readback, pixels, "round-trip mismatch (no padding)");

        println!("GPU_TEST_OK");
        drop(gpu_img);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_upload_round_trip_with_padding() {
        // 3×2 image with stride=5 (2 padding bytes per row).
        // Active pixels: row0=[10,20,30], row1=[40,50,60].
        let src = Image::<u8>::from_vec_with_stride(
            3, 2, 5,
            vec![10, 20, 30, 0, 0,
                 40, 50, 60, 0, 0],
        );

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let gpu_img = GpuImage::upload(&gpu, &src);

        let readback = gpu_img.readback(&gpu);
        // Readback is compact (stride == width), padding should be gone.
        assert_eq!(readback, vec![10, 20, 30, 40, 50, 60],
            "round-trip mismatch (with padding)");

        println!("GPU_TEST_OK");
        drop(gpu_img);
        drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_upload_large_gradient() {
        // 640×480 ramp — stress test for alignment and larger transfers.
        let pixels: Vec<u8> = (0..(640 * 480))
            .map(|i| (i % 256) as u8)
            .collect();
        let src = Image::<u8>::from_vec(640, 480, pixels.clone());

        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let gpu_img = GpuImage::upload(&gpu, &src);

        let readback = gpu_img.readback(&gpu);
        assert_eq!(readback, pixels, "large gradient round-trip mismatch");

        println!("GPU_TEST_OK");
        drop(gpu_img);
        drop(gpu);
    }

    // Outer wrappers ──────────────────────────────────────────────────────────

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_upload_round_trip_no_padding() {
        let out = run_gpu_test_in_subprocess(
            "gpu::image::tests::inner_upload_round_trip_no_padding",
        );
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_upload_round_trip_with_padding() {
        let out = run_gpu_test_in_subprocess(
            "gpu::image::tests::inner_upload_round_trip_with_padding",
        );
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_upload_large_gradient() {
        let out = run_gpu_test_in_subprocess(
            "gpu::image::tests::inner_upload_large_gradient",
        );
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
