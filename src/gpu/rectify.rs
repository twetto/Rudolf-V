// gpu/rectify.rs — stereo rectification on the GPU, for pairs read back to
// the CPU.
//
// The raw u8 pair goes up as two R8Uint textures; rectify.wgsl samples each
// through its eye's bilinear taps (StereoRectification's maps) with the CPU's
// arithmetic, rounds as the CPU rounds, and packs the rectified pixels four
// to a u32; both eyes then copy into one readback buffer, all in the command
// buffer that first reads the upload (the Tegra stale-read rule; see
// gpu/frontend.rs).
//
// Against StereoRectification::remap rounded to u8, it differs only where the
// GPU's f32 arithmetic (fused multiply-adds) lands the other side of a .5:
// one grey level, on a few pixels in a million.

use wgpu::util::DeviceExt;

use crate::gpu::device::GpuDevice;
use crate::image::Image;
use crate::rectify::StereoRectification;

/// Threads per workgroup in rectify.wgsl; each does four pixels.
const WORKGROUP: u32 = 64;

/// A StereoRectification's two eyes on the GPU, for one image size.
pub struct GpuRectification {
    w: usize,
    h: usize,
    /// u32 words per eye: four pixels each.
    words: u64,
    raw: [wgpu::Texture; 2],
    out: [wgpu::Buffer; 2],
    readback: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind: [wgpu::BindGroup; 2],
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuRectification {
    /// The GPU side of `rect`: its maps uploaded once, buffers for its size.
    pub fn new(gpu: &GpuDevice, rect: &StereoRectification) -> Self {
        let dev = &gpu.device;
        let (w, h) = (rect.width(), rect.height());
        let words = (w * h).div_ceil(4) as u64;

        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuRectification BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Uint,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
            ],
        });
        let module = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
            label: Some("rectify.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rectify.wgsl").into()),
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rectify.wgsl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rectify.wgsl"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("rectify"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let raw = [0, 1].map(|eye| {
            dev.create_texture(&wgpu::TextureDescriptor {
                label: Some(["GpuRectification raw left", "GpuRectification raw right"][eye]),
                size: wgpu::Extent3d {
                    width: w as u32,
                    height: h as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let out = [0, 1].map(|_| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuRectification out"),
                size: words * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuRectification readback"),
            size: 2 * words * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = [0, 1].map(|eye| {
            let taps = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuRectification taps"),
                contents: bytemuck::cast_slice(rect.map(eye)),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let view = raw[eye].create_view(&wgpu::TextureViewDescriptor::default());
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuRectification bind group"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: taps.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out[eye].as_entire_binding(),
                    },
                ],
            })
        });
        Self {
            w,
            h,
            words,
            raw,
            out,
            readback,
            pipeline,
            bind,
            readback_rx: None,
        }
    }

    /// Stage the raw pair (`queue.write_texture`; lands before the next
    /// submitted command buffer).
    pub fn upload(&self, gpu: &GpuDevice, left: &Image<u8>, right: &Image<u8>) {
        for (texture, img) in self.raw.iter().zip([left, right]) {
            assert_eq!(
                (img.width(), img.height()),
                (self.w, self.h),
                "raw image size"
            );
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                img.as_slice(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(img.stride() as u32),
                    rows_per_image: Some(self.h as u32),
                },
                wgpu::Extent3d {
                    width: self.w as u32,
                    height: self.h as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Rectify both eyes and copy them to the readback buffer (after
    /// `upload`, in the command buffer that first reads it).
    pub fn record_into(&self, encoder: &mut wgpu::CommandEncoder) {
        let groups = (self.words as u32).div_ceil(WORKGROUP);
        // At most 65535 workgroups a dimension: wrap into rows of them.
        let (gx, gy) = (groups.min(65535), groups.div_ceil(65535));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rectify"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for bind in &self.bind {
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        let bytes = self.words * 4;
        for (eye, out) in self.out.iter().enumerate() {
            encoder.copy_buffer_to_buffer(out, 0, &self.readback, eye as u64 * bytes, bytes);
        }
    }

    /// Map the readback. Call after `queue.submit()`, before `poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        self.readback_rx = Some(rx);
    }

    /// The rectified pair into `out` (left, right). Call after
    /// `device.poll(Wait)`.
    pub fn collect_into(&mut self, out: &mut [Image<u8>; 2]) {
        let rx = self
            .readback_rx
            .take()
            .expect("call arm_readback() before collect_into()");
        rx.recv().unwrap().expect("rectification readback failed");
        let mapped = self
            .readback
            .slice(..)
            .get_mapped_range()
            .expect("buffer mapping failed");
        let n = self.w * self.h;
        let bytes = self.words as usize * 4;
        for (eye, img) in out.iter_mut().enumerate() {
            img.clear_resize(self.w, self.h);
            img.as_mut_slice()
                .copy_from_slice(&mapped[eye * bytes..eye * bytes + n]);
        }
        drop(mapped);
        self.readback.unmap();
    }

    /// The rectified pair of a raw one: upload, record, submit, wait and
    /// read back (for standalone use; a pipeline sharing the GPU records
    /// `record_into` into its own command buffer).
    pub fn rectify_into(
        &mut self,
        gpu: &GpuDevice,
        left: &Image<u8>,
        right: &Image<u8>,
        out: &mut [Image<u8>; 2],
    ) {
        self.upload(gpu, left, right);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.record_into(&mut enc);
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.arm_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        self.collect_into(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraIntrinsics, DistortionModel, StereoRig};

    fn run_gpu_test(name: &str) -> String {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed: {e}"));
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr)
    }

    /// A ZED 2i-like raw rig (rational distortion, a small rotation between
    /// the eyes) at w × h.
    fn zed_like(w: usize, h: usize) -> StereoRig {
        let s = w as f64 / 1280.0;
        let cam = |fx: f64, cx: f64, cy: f64, k: [f64; 8]| CameraIntrinsics {
            fx: fx * s,
            fy: (fx + 0.5) * s,
            cx: cx * s,
            cy: cy * s,
            resolution: [w, h],
            distortion: k.to_vec(),
            model: DistortionModel::Rational,
        };
        StereoRig::new(
            cam(
                958.2,
                651.6,
                353.6,
                [-4.26, 46.5, 4.2e-4, 9.1e-6, -23.3, -4.17, 46.1, -19.8],
            ),
            cam(
                956.1,
                653.1,
                359.9,
                [-5.12, 47.7, 3.2e-4, 2.3e-5, -22.8, -5.03, 47.2, -19.3],
            ),
            [
                [0.99999, -0.00347, 0.00392],
                [0.00348, 0.99999, -0.00305],
                [-0.00391, 0.00306, 0.99999],
            ],
            [-0.12, -2.5e-4, 4.3e-4],
        )
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_matches_cpu() {
        // The CPU's remap rounded as the Python binding rounds it, against the
        // GPU's, on a width that leaves a partial word (322 × 181) and on the
        // ZED's HD720.
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let mut rng = 99u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        };
        for (w, h) in [(322usize, 181usize), (1280, 720)] {
            let s = w as f64 / 1280.0;
            let rect = StereoRectification::new(&zed_like(w, h), 951.4 * s, 658.9 * s, 356.7 * s);
            let raw: Vec<Image<u8>> = (0..2)
                .map(|_| Image::from_vec(w, h, (0..w * h).map(|_| next()).collect()))
                .collect();
            let mut on_gpu = GpuRectification::new(&gpu, &rect);
            let mut got = [Image::new(1, 1), Image::new(1, 1)];
            for _ in 0..2 {
                // twice: the second frame reuses every buffer
                on_gpu.rectify_into(&gpu, &raw[0], &raw[1], &mut got);
            }
            let mut f = Image::new(1, 1);
            for eye in 0..2 {
                rect.remap(eye, &raw[eye], &mut f);
                let (mut off, mut worst) = (0usize, 0i32);
                for (g, c) in got[eye].as_slice().iter().zip(f.as_slice()) {
                    let c = c.round().clamp(0.0, 255.0) as i32;
                    let d = (*g as i32 - c).abs();
                    off += (d > 0) as usize;
                    worst = worst.max(d);
                }
                eprintln!(
                    "[test] {w}x{h} eye {eye}: {off} of {} pixels off, by at most {worst}",
                    w * h
                );
                assert!(worst <= 1, "{w}x{h} eye {eye}: off by {worst}");
                assert!(off * 10_000 < w * h, "{w}x{h} eye {eye}: {off} pixels off");
            }
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_matches_cpu() {
        let out = run_gpu_test("gpu::rectify::tests::inner_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
