// gpu/checkerboard.rs — the checkerboard detector's per-pixel part on the GPU.
//
// The ChESS response and its local maxima (the candidates) are the detector's
// per-pixel work, about half of its time on an Orin Nano. Here they come from
// one submission for both scales, the image and its half-size image
// (checkerboard_response.wgsl, then checkerboard_candidates.wgsl, per scale);
// the lattice search and refinement run on the CPU as before. What is read back
// is the maxima, with their 5R and their neighbours': the CPU makes them into
// candidates as candidates_in does, so the corners are the CPU detector's, bit
// for bit.
//
// The response kernel works from a shared-memory tile per workgroup (each texel
// read about twice, not 21 times). As the frontend's persistent pyramids (Phase
// 3c), the frame goes into an R8Uint texture by `queue.write_texture`, and the
// texture, buffers and bind groups are kept for the largest image so far; a
// smaller one (a window around a board) runs in them with its own sizes, so a
// frame allocates nothing. Both scales are computed every time, though the half-size
// candidates are only searched when the full-size ones hold no board: one round
// trip, not two.
//
// Standalone: `detect`. In a caller's submission: `record_into`, submit,
// `arm_readback`, poll, `collect`.

use std::sync::mpsc::Receiver;

use wgpu::util::DeviceExt;

use crate::checkerboard::{
    Candidate, CheckerboardConfig, CheckerboardDetector, Maximum, candidates_from, candidates_of,
    grow_r5, half_size,
};
use crate::gpu::device::GpuDevice;
use crate::image::Image;

/// Uniform for checkerboard_response.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResponseParams {
    width: u32,
    height: u32,
    half: u32,
    _pad: u32,
}

/// Uniform for checkerboard_candidates.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CandidateParams {
    width: u32,
    height: u32,
    threshold: i32,
    capacity: u32,
}

/// Maxima a scale's buffer holds. Two are at least 3 px apart, but in practice
/// a 1280×720 frame has a few hundred; a frame with more than this is done on
/// the CPU.
const CAPACITY: u32 = 4096;
/// Bytes before the maxima in the candidates buffer: the count, padded to a
/// vec4<u32>'s alignment.
const HEADER: u64 = 16;

/// A checkerboard detector whose responses and candidates come from the GPU.
pub struct GpuCheckerboardDetector {
    /// The CPU part, and the configuration.
    pub detector: CheckerboardDetector,
    response_pipeline: wgpu::ComputePipeline,
    response_bgl: wgpu::BindGroupLayout,
    /// Nearest, clamped: for textureGather of 2×2 blocks.
    sampler: wgpu::Sampler,
    candidate_pipeline: wgpu::ComputePipeline,
    candidate_bgl: wgpu::BindGroupLayout,
    workgroup: (u32, u32),
    /// The GPU resources, for the largest image so far.
    frame: Option<Frame>,
    readback_rx: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Also read the responses back (for checking them).
    read_responses: bool,
}

/// The GPU side for images up to cap_w × cap_h: the frame's texture and both
/// scales.
struct Frame {
    cap_w: u32,
    cap_h: u32,
    texture: wgpu::Texture,
    scales: Vec<Scale>,
}

/// One scale: its response and candidates passes and their buffers, sized
/// for the frame's capacity.
struct Scale {
    half: bool,
    /// This frame's size at this scale; inactive when empty.
    width: u32,
    height: u32,
    active: bool,
    response_bg: wgpu::BindGroup,
    response_groups: (u32, u32),
    candidate_bg: wgpu::BindGroup,
    candidate_groups: (u32, u32),
    /// u32 per response row: the row's pixel pairs.
    pairs: u32,
    response: wgpu::Buffer,
    response_readback: wgpu::Buffer,
    found: wgpu::Buffer,
    found_readback: wgpu::Buffer,
    response_params: wgpu::Buffer,
    candidate_params: wgpu::Buffer,
}

impl Scale {
    fn response_bytes(&self) -> u64 {
        (self.pairs * self.height) as u64 * 4
    }

    /// This scale at width × height (within its capacity): the shaders' sizes
    /// (`write_buffer`, before the next submission) and the dispatch.
    fn resize(&mut self, gpu: &GpuDevice, width: u32, height: u32, threshold: i32, wg: (u32, u32)) {
        self.width = width;
        self.height = height;
        self.pairs = width.div_ceil(2);
        let response = ResponseParams {
            width,
            height,
            half: self.half as u32,
            _pad: 0,
        };
        let candidates = CandidateParams {
            width,
            height,
            threshold,
            capacity: CAPACITY,
        };
        gpu.queue
            .write_buffer(&self.response_params, 0, bytemuck::bytes_of(&response));
        gpu.queue
            .write_buffer(&self.candidate_params, 0, bytemuck::bytes_of(&candidates));
        self.response_groups = (self.pairs.div_ceil(wg.0), height.div_ceil(wg.1));
        self.candidate_groups = (width.div_ceil(wg.0), height.div_ceil(wg.1));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn pipeline(
    gpu: &GpuDevice,
    label: &str,
    src: String,
    bgl: &wgpu::BindGroupLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(
                &gpu.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(bgl)],
                        immediate_size: 0,
                    }),
            ),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

impl GpuCheckerboardDetector {
    pub fn new(gpu: &GpuDevice, config: CheckerboardConfig) -> Self {
        let (wx, wy) = (gpu.workgroup_size.x, gpu.workgroup_size.y);
        // The response tile: the workgroup's pixels and 2·RADIUS of border,
        // in whole 2×2 blocks (one textureGather each).
        let (tile_w, tile_h) = (2 * wx + 10, (wy + 10).next_multiple_of(2));
        let with_sizes = |src: &str| {
            src.replace("{{WG_X}}", &wx.to_string())
                .replace("{{WG_Y}}", &wy.to_string())
                .replace("{{TILE_W}}", &tile_w.to_string())
                .replace("{{TILE_LEN}}", &(tile_w * tile_h).to_string())
        };
        let response_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuCheckerboard response BGL"),
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
                    storage_entry(1, false),
                    uniform_entry(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
        let candidate_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuCheckerboard candidates BGL"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            });
        Self {
            detector: CheckerboardDetector::new(config),
            response_pipeline: pipeline(
                gpu,
                "checkerboard_response.wgsl",
                with_sizes(include_str!("../shaders/checkerboard_response.wgsl")),
                &response_bgl,
                "chess_response",
            ),
            response_bgl,
            sampler: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("GpuCheckerboard nearest"),
                ..Default::default()
            }),
            candidate_pipeline: pipeline(
                gpu,
                "checkerboard_candidates.wgsl",
                with_sizes(include_str!("../shaders/checkerboard_candidates.wgsl")),
                &candidate_bgl,
                "candidates",
            ),
            candidate_bgl,
            workgroup: (wx, wy),
            frame: None,
            readback_rx: None,
            read_responses: false,
        }
    }

    /// The board's `cols × rows` inner corners in pixels (row-major), or None:
    /// `CheckerboardDetector::detect`'s result, with the responses and
    /// candidates from the GPU.
    pub fn detect(&mut self, gpu: &GpuDevice, image: &Image<u8>) -> Option<Vec<[f32; 2]>> {
        self.run(gpu, image);
        self.collect(image)
    }

    /// As `detect`, within the region (x, y, w, h) of `image`; corners in the
    /// whole image's pixels.
    pub fn detect_region(
        &mut self,
        gpu: &GpuDevice,
        image: &Image<u8>,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Option<Vec<[f32; 2]>> {
        let w = w.min(image.width().saturating_sub(x));
        let h = h.min(image.height().saturating_sub(y));
        let crop = image.sub_image(x, y, w, h).to_owned_image();
        let corners = self.detect(gpu, &crop)?;
        Some(
            corners
                .into_iter()
                .map(|[u, v]| [u + x as f32, v + y as f32])
                .collect(),
        )
    }

    /// The candidates in `image` and in its half-size image, from the GPU.
    pub fn candidates(
        &mut self,
        gpu: &GpuDevice,
        image: &Image<u8>,
    ) -> (Vec<Candidate>, Vec<Candidate>) {
        self.run(gpu, image);
        self.read_candidates(image)
    }

    /// The responses (5R) of `image` and of its half-size image, from the GPU.
    pub fn responses(&mut self, gpu: &GpuDevice, image: &Image<u8>) -> (Image<i16>, Image<i16>) {
        self.read_responses = true;
        self.run(gpu, image);
        self.read_responses = false;
        let frame = self.frame.as_ref().expect("run() allocates the frame");
        let mut out = [Image::new(0, 0), Image::new(0, 0)];
        for s in frame.scales.iter().filter(|s| s.active) {
            out[s.half as usize] = read_response(s);
        }
        self.read_candidates(image); // unmaps the candidates
        let [full, half] = out;
        (full, half)
    }

    /// Record, submit, map and wait.
    fn run(&mut self, gpu: &GpuDevice, image: &Image<u8>) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuCheckerboard"),
            });
        self.record_into(gpu, &mut encoder, image);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        self.arm_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
    }

    /// Stage `image` (`queue.write_texture`, landing before the next submitted
    /// command buffer) and record both scales' passes and readback copies into
    /// `encoder`.
    pub fn record_into(
        &mut self,
        gpu: &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        image: &Image<u8>,
    ) {
        let (w, h) = (image.width() as u32, image.height() as u32);
        let (cap_w, cap_h) = self.frame.as_ref().map_or((0, 0), |f| (f.cap_w, f.cap_h));
        if self.frame.is_none() || w > cap_w || h > cap_h {
            self.frame = Some(self.allocate(gpu, w.max(cap_w), h.max(cap_h)));
        }
        let (threshold, wg) = (grow_r5(&self.detector.config) as i32, self.workgroup);
        let frame = self.frame.as_mut().unwrap();
        for s in &mut frame.scales {
            let (sw, sh) = if s.half { (w / 2, h / 2) } else { (w, h) };
            s.active = sw > 0 && sh > 0;
            if s.active && (sw, sh) != (s.width, s.height) {
                s.resize(gpu, sw, sh, threshold, wg);
            }
        }
        let frame = self.frame.as_ref().unwrap();
        if w == 0 || h == 0 {
            return;
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.as_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.stride() as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let active = || frame.scales.iter().filter(|s| s.active);
        for s in active() {
            encoder.clear_buffer(&s.found, 0, Some(HEADER));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("checkerboard"),
                timestamp_writes: None,
            });
            for s in active() {
                pass.set_pipeline(&self.response_pipeline);
                pass.set_bind_group(0, &s.response_bg, &[]);
                pass.dispatch_workgroups(s.response_groups.0, s.response_groups.1, 1);
                pass.set_pipeline(&self.candidate_pipeline);
                pass.set_bind_group(0, &s.candidate_bg, &[]);
                pass.dispatch_workgroups(s.candidate_groups.0, s.candidate_groups.1, 1);
            }
        }
        for s in active() {
            encoder.copy_buffer_to_buffer(&s.found, 0, &s.found_readback, 0, s.found.size());
            if self.read_responses {
                let bytes = s.response_bytes();
                encoder.copy_buffer_to_buffer(&s.response, 0, &s.response_readback, 0, bytes);
            }
        }
    }

    /// Map the readback buffers; call after submitting `record_into`'s
    /// encoder, before `device.poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        for s in self
            .frame
            .iter()
            .flat_map(|f| &f.scales)
            .filter(|s| s.active)
        {
            let mut buffers = vec![(&s.found_readback, s.found.size())];
            if self.read_responses {
                buffers.push((&s.response_readback, s.response_bytes()));
            }
            for (buf, bytes) in buffers {
                let tx = tx.clone();
                buf.slice(..bytes)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            }
        }
        self.readback_rx = Some(rx);
    }

    /// The corners for `image`, the image given to `record_into`; call after
    /// `device.poll(Wait)`.
    pub fn collect(&mut self, image: &Image<u8>) -> Option<Vec<[f32; 2]>> {
        let (full, half) = self.read_candidates(image);
        self.detector.detect_with_candidates(image, full, half)
    }

    /// Both scales' candidates from the mapped maxima, then unmapped; a scale
    /// with more maxima than its buffer holds is done on the CPU.
    fn read_candidates(&mut self, image: &Image<u8>) -> (Vec<Candidate>, Vec<Candidate>) {
        for r in self
            .readback_rx
            .take()
            .expect("call arm_readback() before collecting")
        {
            r.expect("GpuCheckerboard readback failed");
        }
        let config = &self.detector.config;
        let mut out = [Vec::new(), Vec::new()];
        let frame = self.frame.as_ref().expect("call record_into() first");
        for s in frame.scales.iter().filter(|s| s.active) {
            let mapped = s
                .found_readback
                .slice(..s.found.size())
                .get_mapped_range()
                .expect("buffer mapping failed");
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            let count = words[0];
            out[s.half as usize] = if count > CAPACITY {
                let scaled;
                candidates_of(
                    if s.half {
                        scaled = half_size(image);
                        &scaled
                    } else {
                        image
                    },
                    config,
                )
            } else {
                let items = &words[HEADER as usize / 4..][..4 * count as usize];
                let mut maxima: Vec<Maximum> = items
                    .chunks_exact(4)
                    .map(|m| Maximum {
                        x: m[0] as u16,
                        y: (m[0] >> 16) as u16,
                        r5: m[1] as i16,
                        near: [
                            (m[1] >> 16) as i16,
                            m[2] as i16,
                            (m[2] >> 16) as i16,
                            m[3] as i16,
                        ],
                    })
                    .collect();
                candidates_from(&mut maxima, config)
            };
            drop(mapped);
            s.found_readback.unmap();
        }
        let [full, half] = out;
        (full, half)
    }

    /// The texture and both scales' buffers and bind groups for images up to
    /// w × h; their sizes are set by `record_into`.
    fn allocate(&self, gpu: &GpuDevice, w: u32, h: u32) -> Frame {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GpuCheckerboard frame"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let scales = [(w, h, false), (w / 2, h / 2, true)]
            .into_iter()
            .map(|(sw, sh, half)| self.allocate_scale(gpu, &view, sw.max(1), sh.max(1), half))
            .collect();
        Frame {
            cap_w: w,
            cap_h: h,
            texture,
            scales,
        }
    }

    fn allocate_scale(
        &self,
        gpu: &GpuDevice,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        half: bool,
    ) -> Scale {
        let buffer = |label, size, usage| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let uniform = |contents: &[u8]| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GpuCheckerboard params"),
                    contents,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        };
        let bind_group = |layout, entries: &[wgpu::BindGroupEntry]| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuCheckerboard BG"),
                layout,
                entries,
            })
        };
        use wgpu::BufferUsages as U;
        let pairs = width.div_ceil(2);
        let response_bytes = (pairs * height) as u64 * 4;
        let found_bytes = HEADER + CAPACITY as u64 * 16;
        let response = buffer(
            "GpuCheckerboard response",
            response_bytes,
            U::STORAGE | U::COPY_SRC,
        );
        let response_readback = buffer(
            "GpuCheckerboard response readback",
            response_bytes,
            U::MAP_READ | U::COPY_DST,
        );
        let found = buffer(
            "GpuCheckerboard maxima",
            found_bytes,
            U::STORAGE | U::COPY_SRC | U::COPY_DST,
        );
        let found_readback = buffer(
            "GpuCheckerboard maxima readback",
            found_bytes,
            U::MAP_READ | U::COPY_DST,
        );
        let response_params = uniform(bytemuck::bytes_of(&ResponseParams {
            width,
            height,
            half: half as u32,
            _pad: 0,
        }));
        let candidate_params = uniform(bytemuck::bytes_of(&CandidateParams {
            width,
            height,
            threshold: grow_r5(&self.detector.config) as i32,
            capacity: CAPACITY,
        }));
        let response_bg = bind_group(
            &self.response_bgl,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: response.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: response_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        );
        let candidate_bg = bind_group(
            &self.candidate_bgl,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: response.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: found.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidate_params.as_entire_binding(),
                },
            ],
        );
        // Sized by the first `record_into` (resize).
        Scale {
            half,
            width: 0,
            height: 0,
            active: false,
            response_bg,
            response_groups: (0, 0),
            candidate_bg,
            candidate_groups: (0, 0),
            pairs: 0,
            response,
            response_readback,
            found,
            found_readback,
            response_params,
            candidate_params,
        }
    }
}

/// A scale's mapped response as an image (stride = width), then unmapped.
fn read_response(s: &Scale) -> Image<i16> {
    let (w, h) = (s.width as usize, s.height as usize);
    let mut out = Image::new(w, h);
    {
        let mapped = s
            .response_readback
            .slice(..s.response_bytes())
            .get_mapped_range()
            .expect("buffer mapping failed");
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut out.as_mut_slice()[..w * h]);
        // Rows of pairs: an odd width ends each row in half a pair.
        let row = 4 * s.pairs as usize;
        for (d, m) in dst.chunks_exact_mut(2 * w).zip(mapped.chunks_exact(row)) {
            d.copy_from_slice(&m[..2 * w]);
        }
    }
    s.response_readback.unmap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkerboard::chess_response_r5;

    fn noise(w: usize, h: usize, seed: u32) -> Image<u8> {
        let mut rng = seed;
        let pixels = (0..w * h)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 24) as u8
            })
            .collect();
        Image::from_vec(w, h, pixels)
    }

    /// A 10×7-square board (9×6 inner corners) of `square` px on grey.
    fn board(w: usize, h: usize, x0: usize, y0: usize, square: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![128u8; w * h]);
        for y in y0..y0 + 9 * square {
            for x in x0..x0 + 12 * square {
                let (r, c) = ((y - y0) / square, (x - x0) / square);
                let inside = (1..8).contains(&r) && (1..11).contains(&c);
                img.set(x, y, if inside && (r + c) % 2 == 0 { 0 } else { 255 });
            }
        }
        img
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_responses_match_cpu() {
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let mut det = GpuCheckerboardDetector::new(&gpu, CheckerboardConfig::default());
        // Odd sizes too: pairs then span rows, and the half size rounds down.
        for (w, h, seed) in [
            (1280, 720, 1),
            (641, 359, 2),
            (37, 23, 3),
            (11, 11, 4),
            (7, 3, 5),
        ] {
            let img = noise(w, h, seed);
            let (full, small) = det.responses(&gpu, &img);
            assert_eq!(
                full.as_slice(),
                chess_response_r5(&img).as_slice(),
                "{w}x{h}"
            );
            assert_eq!(
                small.as_slice(),
                chess_response_r5(&half_size(&img)).as_slice(),
                "{w}x{h} half"
            );
        }
        println!("GPU_TEST_OK");
    }

    /// Noise in `block`×`block` squares: flat patches, so responses tie (plateaus).
    fn blocky(w: usize, h: usize, block: usize, seed: u32) -> Image<u8> {
        let coarse = noise(w.div_ceil(block), h.div_ceil(block), seed);
        let mut img = Image::from_vec(w, h, vec![0u8; w * h]);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, coarse.get(x / block, y / block));
            }
        }
        img
    }

    fn bits(c: &[Candidate]) -> Vec<(u32, u32, u32)> {
        c.iter()
            .map(|c| (c.x.to_bits(), c.y.to_bits(), c.response.to_bits()))
            .collect()
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_candidates_match_cpu() {
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let config = CheckerboardConfig::default();
        let mut det = GpuCheckerboardDetector::new(&gpu, config.clone());
        for img in [
            blocky(1280, 720, 4, 7),
            blocky(641, 359, 3, 8),
            noise(640, 480, 9),
            board(640, 480, 100, 80, 30),
        ] {
            let (full, half) = det.candidates(&gpu, &img);
            assert!(!full.is_empty());
            assert_eq!(bits(&full), bits(&candidates_of(&img, &config)));
            assert_eq!(bits(&half), bits(&candidates_of(&half_size(&img), &config)));
        }
        println!("GPU_TEST_OK");
    }

    /// One detector through changing sizes (as a window around a board): a
    /// smaller image runs in the larger one's texture and buffers, whose stale
    /// pixels past its edge must not reach its responses or candidates.
    #[test]
    #[ignore = "GPU integration"]
    fn inner_sizes_change() {
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let config = CheckerboardConfig::default();
        let mut det = GpuCheckerboardDetector::new(&gpu, config.clone());
        for (k, (w, h)) in [
            (640, 480),
            (321, 157),
            (640, 480),
            (97, 301),
            (1, 1),
            (700, 100),
        ]
        .into_iter()
        .enumerate()
        {
            let img = blocky(w, h, 3, 20 + k as u32);
            let (full, small) = det.responses(&gpu, &img);
            assert_eq!(
                full.as_slice(),
                chess_response_r5(&img).as_slice(),
                "{w}x{h}"
            );
            let half = half_size(&img);
            assert_eq!(
                small.as_slice(),
                chess_response_r5(&half).as_slice(),
                "{w}x{h} half"
            );
            let (c_full, c_half) = det.candidates(&gpu, &img);
            assert_eq!(
                bits(&c_full),
                bits(&candidates_of(&img, &config)),
                "{w}x{h}"
            );
            assert_eq!(
                bits(&c_half),
                bits(&candidates_of(&half, &config)),
                "{w}x{h} half"
            );
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration"]
    fn inner_detects_as_cpu() {
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let mut det = GpuCheckerboardDetector::new(&gpu, CheckerboardConfig::default());
        let cpu = CheckerboardDetector::new(CheckerboardConfig::default());
        // Small squares, big squares (half size may be needed), no board.
        for img in [
            board(640, 480, 100, 80, 30),
            board(1280, 720, 200, 40, 70),
            noise(320, 240, 6),
        ] {
            assert_eq!(det.detect(&gpu, &img), cpu.detect(&img));
        }
        assert!(det.detect(&gpu, &board(640, 480, 100, 80, 30)).is_some());
        println!("GPU_TEST_OK");
    }

    fn run_gpu_test(test_name: &str) -> String {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test_name, "--exact", "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed for {test_name}: {e}"));
        let out = String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr);
        print!("{out}");
        out
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_responses_match_cpu() {
        assert!(
            run_gpu_test("gpu::checkerboard::tests::inner_responses_match_cpu")
                .contains("GPU_TEST_OK")
        );
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_candidates_match_cpu() {
        assert!(
            run_gpu_test("gpu::checkerboard::tests::inner_candidates_match_cpu")
                .contains("GPU_TEST_OK")
        );
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_sizes_change() {
        assert!(
            run_gpu_test("gpu::checkerboard::tests::inner_sizes_change").contains("GPU_TEST_OK")
        );
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_detects_as_cpu() {
        assert!(
            run_gpu_test("gpu::checkerboard::tests::inner_detects_as_cpu").contains("GPU_TEST_OK")
        );
    }
}
