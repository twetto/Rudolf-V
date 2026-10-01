// gpu/histeq.rs — global histogram equalization on the GPU.
//
// Replaces the CPU `equalize_histogram` step of the GPU frontend with two
// passes recorded into the frame's command buffer, between the raw upload and
// the pyramid build:
//
//   histeq_hist.wgsl  raw R8Uint frame → 256-bin histogram (workgroup atomics)
//   histeq_lut.wgsl   histogram → 256-entry LUT (same f32 formula as the CPU)
//
// The pyramid's level-0 convert pass then reads through the LUT buffer
// (`GpuPyramidPipeline::allocate_with_lut`), so the equalized image is never
// materialized on the CPU. The LUT itself (1 KB) is copied to a readback
// buffer in the same submit: CPU-side consumers that need equalized pixel
// values at a few positions (LBP verification) apply it to the raw image,
// which gives exactly the values the GPU used.

use wgpu::util::DeviceExt;

use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramid;

/// Workgroups for the histogram pass (each thread handles ~total/(N·256) px).
const HIST_WORKGROUPS: u32 = 64;
/// LUT buffer size: 256 × u32.
const LUT_BYTES: u64 = 256 * 4;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HistEqParams {
    total: u32,
    _pad: [u32; 3],
}

/// GPU global histogram equalization (histogram + LUT) for one image size.
pub struct GpuGlobalHistEq {
    hist_pipeline: wgpu::ComputePipeline,
    hist_bgl: wgpu::BindGroupLayout,
    lut_pipeline: wgpu::ComputePipeline,
    lut_bg: wgpu::BindGroup,
    hist_buf: wgpu::Buffer,
    lut_buf: wgpu::Buffer,
    rb_buf: wgpu::Buffer,
    // Histogram bind groups per source texture (the frontend's two pyramids).
    bg_cache: Vec<(wgpu::Texture, wgpu::BindGroup)>,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuGlobalHistEq {
    /// Compile the passes for `width × height` frames.
    pub fn new(gpu: &GpuDevice, width: usize, height: usize) -> Self {
        let dev = &gpu.device;
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

        let hist_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuHistEq hist BGL"),
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
                storage(1, false),
            ],
        });
        let lut_bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuHistEq LUT BGL"),
            entries: &[
                storage(0, true),
                storage(1, false),
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

        let pipeline = |src: &str, label: &str, entry: &str, bgl: &wgpu::BindGroupLayout| {
            let module = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let hist_pipeline = pipeline(
            include_str!("../shaders/histeq_hist.wgsl"),
            "histeq_hist.wgsl",
            "histogram",
            &hist_bgl,
        );
        let lut_pipeline = pipeline(
            include_str!("../shaders/histeq_lut.wgsl"),
            "histeq_lut.wgsl",
            "build_lut",
            &lut_bgl,
        );

        let hist_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuHistEq histogram"),
            size: LUT_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Starts as the identity so pyramids bound to it convert unchanged
        // until the first LUT pass runs.
        let lut_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuHistEq LUT"),
            contents: bytemuck::cast_slice(&crate::gpu::pyramid::identity_lut_u32()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let rb_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuHistEq LUT readback"),
            size: LUT_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuHistEq params"),
            contents: bytemuck::bytes_of(&HistEqParams {
                total: (width * height) as u32,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let lut_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuHistEq LUT BG"),
            layout: &lut_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        GpuGlobalHistEq {
            hist_pipeline,
            hist_bgl,
            lut_pipeline,
            lut_bg,
            hist_buf,
            lut_buf,
            rb_buf,
            bg_cache: Vec::new(),
            readback_rx: None,
        }
    }

    /// The LUT buffer (256 × u32) to bind as the pyramid's conversion LUT.
    pub fn lut_buffer(&self) -> &wgpu::Buffer {
        &self.lut_buf
    }

    /// Record histogram + LUT passes for the raw frame already uploaded into
    /// `pyr` (`GpuPyramidPipeline::upload`), plus the LUT readback copy.
    /// Record this *before* `GpuPyramidPipeline::record_build` for `pyr`.
    pub fn record(
        &mut self,
        gpu: &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        pyr: &GpuPyramid,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("histeq"),
                timestamp_writes: None,
            });
            self.record_dispatches(gpu, &mut pass, pyr);
        }
        self.record_readback(encoder);
    }

    /// Histogram + LUT dispatches into an existing compute pass. Zeroes the
    /// histogram through the queue (lands before the next submit), so no
    /// encoder-level clear is needed. Pair with [`record_readback`] after the
    /// pass.
    ///
    /// [`record_readback`]: GpuGlobalHistEq::record_readback
    pub fn record_dispatches(
        &mut self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        pyr: &GpuPyramid,
    ) {
        let raw_view = pyr
            .raw_view()
            .expect("GpuGlobalHistEq needs a pyramid from GpuPyramidPipeline::allocate*");
        let key = pyr.levels[0].texture.clone();
        let idx = match self.bg_cache.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GpuHistEq hist BG"),
                    layout: &self.hist_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(raw_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.hist_buf.as_entire_binding(),
                        },
                    ],
                });
                if self.bg_cache.len() >= 2 {
                    self.bg_cache.remove(0);
                }
                self.bg_cache.push((key, bg));
                self.bg_cache.len() - 1
            }
        };

        gpu.queue
            .write_buffer(&self.hist_buf, 0, &[0u8; LUT_BYTES as usize]);
        pass.set_pipeline(&self.hist_pipeline);
        pass.set_bind_group(0, &self.bg_cache[idx].1, &[]);
        pass.dispatch_workgroups(HIST_WORKGROUPS, 1, 1);
        pass.set_pipeline(&self.lut_pipeline);
        pass.set_bind_group(0, &self.lut_bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Copy the LUT to the readback buffer (record after the pass).
    pub fn record_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.lut_buf, 0, &self.rb_buf, 0, LUT_BYTES);
    }

    /// Reset the LUT to the identity (for frames that skip `record`).
    pub fn write_identity(&self, gpu: &GpuDevice) {
        gpu.queue.write_buffer(
            &self.lut_buf,
            0,
            bytemuck::cast_slice(&crate::gpu::pyramid::identity_lut_u32()),
        );
    }

    /// Map the LUT readback. Call after `queue.submit()`, before `poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rb_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        self.readback_rx = Some(rx);
    }

    /// The frame's LUT as bytes. Call after `device.poll(Wait)`.
    pub fn collect_lut(&mut self) -> [u8; 256] {
        let rx = self
            .readback_rx
            .take()
            .expect("call arm_readback() before collect_lut()");
        rx.recv().unwrap().expect("histeq LUT readback failed");
        let mapped = self
            .rb_buf
            .slice(..)
            .get_mapped_range()
            .expect("buffer mapping failed");
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let lut: [u8; 256] = std::array::from_fn(|i| words[i] as u8);
        drop(mapped);
        self.rb_buf.unmap();
        lut
    }

    /// Convenience for tests: histogram + LUT for one image, blocking.
    pub fn compute_lut_blocking(
        &mut self,
        gpu: &GpuDevice,
        pipeline: &crate::gpu::pyramid::GpuPyramidPipeline,
        pyr: &GpuPyramid,
        src: &crate::image::Image<u8>,
    ) -> [u8; 256] {
        pipeline.upload(gpu, src, pyr);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.record(gpu, &mut enc, pyr);
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.arm_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        self.collect_lut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::histeq::equalize_histogram_lut;
    use crate::image::Image;

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

    #[test]
    #[ignore = "GPU integration"]
    fn inner_gpu_lut_matches_cpu() {
        // The GPU LUT must equal histeq::equalize_histogram_lut bit for bit on
        // varied content: random, low-contrast, constant, and sparse images.
        let (w, h) = (97usize, 61usize);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let mut heq = GpuGlobalHistEq::new(&gpu, w, h);
        let pyr = pipeline.allocate_with_lut(&gpu, w, h, 1, heq.lut_buffer());

        let mut rng = 777u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 24) as u8
        };
        let images: Vec<Vec<u8>> = vec![
            (0..w * h).map(|_| next()).collect(),
            (0..w * h).map(|_| 100 + next() % 20).collect(),
            vec![42u8; w * h],
            (0..w * h)
                .map(|i| if i % 13 == 0 { 255 } else { 3 })
                .collect(),
            (0..w * h).map(|i| (i % 256) as u8).collect(),
        ];
        for (k, px) in images.into_iter().enumerate() {
            let img = Image::from_vec(w, h, px);
            let gpu_lut = heq.compute_lut_blocking(&gpu, &pipeline, &pyr, &img);
            let cpu_lut = equalize_histogram_lut(&img);
            assert_eq!(gpu_lut, cpu_lut, "image {k}: GPU LUT differs from CPU");
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_lut_matches_cpu() {
        let out = run_gpu_test("gpu::histeq::tests::inner_gpu_lut_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
