// gpu/sgbm.rs — Semi-global block matching on the GPU (sgbm.wgsl): sgbm.rs's
// steps on the same integers, one command buffer per frame.
//
// The pyramids are built on the GPU (GpuPyramidPipeline); `spans` reads the
// matched level's textures. The cost volumes hold 16-bit costs, two
// disparities per word, as OpenCV's (every volume stage is bound by memory
// traffic; halving it cut a frame on the Orin Nano by a fifth). Pixel costs
// are one thread per pixel, block sums one per volume word; the path costs
// run four scanlines per workgroup, a 32-thread group per scanline, all
// directions in one dispatch, added into S with integer atomics, so their
// order does not matter and S equals the CPU's. The left-right check takes
// each right pixel's best match with atomicMin on the packed (S << 8 | d);
// speckles use the same lock-free union-find as dense_stereo_mask.wgsl.
//
// Orin Nano 15 W, 160x90, 59 disparities, 8 paths: 4.6 ms a frame with the
// readback (upload 0.5, pyramids 0.9, pixel and block costs 0.7, paths 1.8,
// winners to speckles 0.4).
//
// From the CPU's spans every stage matches the CPU exactly (the tests feed
// them in); the GPU pyramid differs from the CPU's by up to ~1e-4 grey levels,
// which moves a rounded pixel cost here and there.

use crate::dense_stereo::{DenseDepth, SPECKLE_STEP};
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::{GpuPyramid, GpuPyramidPipeline};
use crate::image::Image;
use crate::sgbm::{DIRS, Sgbm, SgbmConfig, depth_from_disparity};
use wgpu::util::DeviceExt;

/// sgbm.wgsl's `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    nd: u32,
    half: u32,
    p1: u32,
    p2: u32,
    uniqueness: u32,
    max_diff: u32,
    min_speckle: u32,
    n_lines: u32,
    cap: f32,
    ndp: u32,
}

const NO_CHECK: u32 = u32::MAX;

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

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
        },
        count: None,
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

fn res(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    b.as_entire_binding()
}

fn tex(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

/// A compute pipeline for `entry`, with one bind group of `entries`.
fn pipeline(
    gpu: &GpuDevice,
    module: &wgpu::ShaderModule,
    entry: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::ComputePipeline {
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(entry),
            entries,
        });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

/// Every path direction's scanlines (first pixel, step): the pixels whose
/// predecessor along the direction is outside the image.
fn scanlines(w: usize, h: usize, paths: usize) -> Vec<[i32; 4]> {
    let mut out = Vec::new();
    for &(dx, dy) in &DIRS[..paths] {
        for y in 0..h as isize {
            for x in 0..w as isize {
                let (qx, qy) = (x - dx, y - dy);
                if qx < 0 || qy < 0 || qx >= w as isize || qy >= h as isize {
                    out.push([x as i32, y as i32, dx as i32, dy as i32]);
                }
            }
        }
    }
    out
}

/// sgbm.wgsl without the kernels that update the path sums (`box_cols`,
/// `paths`), which need them atomic.
fn winners_only(wgsl: &str) -> String {
    let mut out = wgsl.to_string();
    for entry in ["fn box_cols(", "fn paths("] {
        let start = out.find(entry).expect("kernel");
        let start = out[..start].rfind("@compute").expect("attribute");
        let end = start + out[start..].find("\n}\n").expect("end") + 3;
        out.replace_range(start..end, "");
    }
    out
}

/// A dispatch of `groups` workgroups as (x, y) within the per-dimension limit.
fn grid(groups: u32) -> (u32, u32) {
    let x = groups.clamp(1, 65535);
    (x, groups.div_ceil(x))
}

struct Pipelines {
    spans: wgpu::ComputePipeline,
    pixel_costs: wgpu::ComputePipeline,
    box_rows: wgpu::ComputePipeline,
    box_cols: wgpu::ComputePipeline,
    paths: wgpu::ComputePipeline,
    winners: wgpu::ComputePipeline,
    left_right: wgpu::ComputePipeline,
    unite: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

/// Bind groups, in Pipelines' order.
struct BindGroups([wgpu::BindGroup; 10]);

/// The buffers the tests read back.
#[cfg_attr(not(test), allow(dead_code))]
struct Buffers {
    params: wgpu::Buffer,
    spans: wgpu::Buffer,
    cost: wgpu::Buffer,
    tmp: wgpu::Buffer,
    total: wgpu::Buffer,
    scanlines: wgpu::Buffer,
    disp: wgpu::Buffer,
    best: wgpu::Buffer,
    right_best: wgpu::Buffer,
    valid: wgpu::Buffer,
    parent: wgpu::Buffer,
    region: wgpu::Buffer,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
}

/// Semi-global block matching on the GPU, for an `Sgbm`'s rig and settings.
pub struct GpuSgbm {
    config: SgbmConfig,
    w: usize,
    h: usize,
    nd: usize,
    n_lines: usize,
    log_scale: Vec<f32>,
    pyramid: GpuPyramidPipeline,
    pyr: [GpuPyramid; 2],
    pipelines: Pipelines,
    bind: BindGroups,
    buf: Buffers,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    disp: Vec<f32>,
}

impl GpuSgbm {
    pub fn new(gpu: &GpuDevice, cpu: &Sgbm) -> Self {
        Self::with_shader(gpu, cpu, include_str!("../shaders/sgbm.wgsl"))
    }

    /// As `new`, from sgbm.wgsl's text (the tests time variants of it).
    fn with_shader(gpu: &GpuDevice, cpu: &Sgbm, shader: &str) -> Self {
        let config = cpu.config.clone();
        let (w, h, nd) = (cpu.w, cpu.h, cpu.nd);
        // 16-bit costs, two per word (sgbm.wgsl): S must stay under 2^16. A
        // pixel cost is at most 2 cap + 64 (prefiltered, plus a quarter of
        // 255, rounded), a path cost a block's plus P2.
        let cost_max = (2 * config.pre_filter_cap + 64) as usize;
        let block_max = config.block_size * config.block_size * cost_max;
        let sum_max = config.paths * (block_max + config.p2 as usize);
        assert!(
            sum_max < 1 << 16,
            "the GPU's 16-bit path sums could reach {sum_max}: a smaller block_size, P2 or pre_filter_cap"
        );
        let ndp = nd.div_ceil(2);
        let volume = 4 * w * h * ndp;
        let limit = gpu.device.limits().max_storage_buffer_binding_size as usize;
        assert!(
            volume <= limit,
            "a {w}x{h}x{nd} cost volume ({volume} bytes) exceeds the GPU's storage binding limit; match a coarser level"
        );
        let wgsl = shader
            .replace("{{WG_X}}", &gpu.workgroup_size.x.to_string())
            .replace("{{WG_Y}}", &gpu.workgroup_size.y.to_string())
            .replace("{{SPECKLE_STEP}}", &format!("{SPECKLE_STEP:?}"));
        // `winners` only reads the sums: a module declaring them a plain
        // array lets those reads go through the cache like any other (atomic
        // loads need not).
        let plain = wgsl
            .replace(
                "var<storage, read_write> total: array<atomic<u32>>;",
                "var<storage, read_write> total: array<u32>;",
            )
            .replace("atomicLoad(&total[", "(total[");
        // (`sum_at` reads the plain array then too.)
        let plain_sums = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
            label: Some("sgbm.wgsl, plain sums"),
            source: wgpu::ShaderSource::Wgsl(winners_only(&plain).into()),
        });
        let module = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
            label: Some("sgbm.wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let u = uniform_entry;
        let rw = |b| storage_entry(b, false);
        let pipelines = Pipelines {
            spans: pipeline(
                gpu,
                &module,
                "spans_main",
                &[u(0), texture_entry(1), texture_entry(2), rw(3)],
            ),
            pixel_costs: pipeline(gpu, &module, "pixel_costs", &[u(0), rw(3), rw(4)]),
            box_rows: pipeline(gpu, &module, "box_rows", &[u(0), rw(4), rw(5)]),
            box_cols: pipeline(
                gpu,
                &module,
                "box_cols",
                &[u(0), rw(4), rw(5), rw(7), rw(11)],
            ),
            paths: pipeline(
                gpu,
                &module,
                "paths",
                &[u(0), rw(4), rw(7), storage_entry(8, true)],
            ),
            winners: pipeline(
                gpu,
                &plain_sums,
                "winners",
                &[u(0), rw(7), rw(9), rw(10), rw(11), rw(12)],
            ),
            left_right: pipeline(
                gpu,
                &module,
                "left_right",
                &[u(0), rw(9), rw(11), rw(12), rw(13), rw(14)],
            ),
            unite: pipeline(gpu, &module, "unite", &[u(0), rw(9), rw(12), rw(13)]),
            count: pipeline(gpu, &module, "count", &[u(0), rw(12), rw(13), rw(14)]),
            finish: pipeline(
                gpu,
                &module,
                "finish",
                &[u(0), rw(9), rw(12), rw(13), rw(14), rw(15)],
            ),
        };

        let lines = scanlines(w, h, config.paths);
        let params = Params {
            w: w as u32,
            h: h as u32,
            nd: nd as u32,
            half: (config.block_size / 2) as u32,
            p1: config.p1,
            p2: config.p2,
            uniqueness: config.uniqueness,
            max_diff: config.disp12_max_diff.unwrap_or(NO_CHECK),
            min_speckle: (config.speckle_size >> (2 * config.level)) as u32,
            n_lines: lines.len() as u32,
            cap: config.pre_filter_cap as f32,
            ndp: ndp as u32,
        };
        let storage = |label: &str, bytes: usize| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n = w * h;
        let buf = Buffers {
            params: gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sgbm params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
            spans: storage("sgbm spans", 2 * n * 24),
            cost: storage("sgbm costs", volume),
            tmp: storage("sgbm row sums", volume),
            total: storage("sgbm path sums", volume),
            scanlines: gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sgbm scanlines"),
                    contents: bytemuck::cast_slice(&lines),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            disp: storage("sgbm disparity", 4 * n),
            best: storage("sgbm best", 4 * n),
            right_best: storage("sgbm right best", 4 * n),
            valid: storage("sgbm valid", 4 * n),
            parent: storage("sgbm parents", 4 * n),
            region: storage("sgbm regions", 4 * n),
            out: storage("sgbm out", 4 * n),
            readback: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sgbm readback"),
                size: (4 * n) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let pyramid = GpuPyramidPipeline::new(gpu);
        let pyr = [0, 1].map(|_| pyramid.allocate(gpu, cpu.w0, cpu.h0, config.level + 1));
        let level = config.level;
        let b = |e: &[(u32, wgpu::BindingResource)], p: &wgpu::ComputePipeline| {
            let entries: Vec<wgpu::BindGroupEntry> = e
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect();
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let r = res;
        let p = &pipelines;
        let bind = BindGroups([
            b(
                &[
                    (0, r(&buf.params)),
                    (1, tex(&pyr[0].levels[level].read_view)),
                    (2, tex(&pyr[1].levels[level].read_view)),
                    (3, r(&buf.spans)),
                ],
                &p.spans,
            ),
            b(
                &[(0, r(&buf.params)), (3, r(&buf.spans)), (4, r(&buf.cost))],
                &p.pixel_costs,
            ),
            b(
                &[(0, r(&buf.params)), (4, r(&buf.cost)), (5, r(&buf.tmp))],
                &p.box_rows,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (4, r(&buf.cost)),
                    (5, r(&buf.tmp)),
                    (7, r(&buf.total)),
                    (11, r(&buf.right_best)),
                ],
                &p.box_cols,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (4, r(&buf.cost)),
                    (7, r(&buf.total)),
                    (8, r(&buf.scanlines)),
                ],
                &p.paths,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (7, r(&buf.total)),
                    (9, r(&buf.disp)),
                    (10, r(&buf.best)),
                    (11, r(&buf.right_best)),
                    (12, r(&buf.valid)),
                ],
                &p.winners,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (9, r(&buf.disp)),
                    (11, r(&buf.right_best)),
                    (12, r(&buf.valid)),
                    (13, r(&buf.parent)),
                    (14, r(&buf.region)),
                ],
                &p.left_right,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (9, r(&buf.disp)),
                    (12, r(&buf.valid)),
                    (13, r(&buf.parent)),
                ],
                &p.unite,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (12, r(&buf.valid)),
                    (13, r(&buf.parent)),
                    (14, r(&buf.region)),
                ],
                &p.count,
            ),
            b(
                &[
                    (0, r(&buf.params)),
                    (9, r(&buf.disp)),
                    (12, r(&buf.valid)),
                    (13, r(&buf.parent)),
                    (14, r(&buf.region)),
                    (15, r(&buf.out)),
                ],
                &p.finish,
            ),
        ]);
        Self {
            log_scale: cpu.log_scale.clone(),
            config,
            w,
            h,
            nd,
            n_lines: lines.len(),
            pyramid,
            pyr,
            pipelines,
            bind,
            buf,
            readback_rx: None,
            disp: Vec::new(),
        }
    }

    /// Stage a rectified pair for the next recorded frame.
    pub fn upload(&self, gpu: &GpuDevice, left: &Image<u8>, right: &Image<u8>) {
        self.pyramid.upload(gpu, left, &self.pyr[0]);
        self.pyramid.upload(gpu, right, &self.pyr[1]);
    }

    fn dispatch(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        p: &wgpu::ComputePipeline,
        i: usize,
        (x, y): (u32, u32),
    ) {
        pass.set_pipeline(p);
        pass.set_bind_group(0, &self.bind.0[i], &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    /// Both pyramids and the spans (steps 1-2).
    fn record_spans(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        for pyr in &self.pyr {
            self.pyramid.record_build_in_pass(gpu, pass, pyr);
        }
        let px = gpu.dispatch_size(self.w as u32, self.h as u32);
        self.dispatch(pass, &self.pipelines.spans, 0, px);
    }

    /// Pixel and block costs (steps 3-4).
    fn record_costs(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        let p = &self.pipelines;
        let px = gpu.dispatch_size(self.w as u32, self.h as u32);
        let elements = grid(((self.w * self.h * self.nd.div_ceil(2)) as u32).div_ceil(64));
        self.dispatch(pass, &p.pixel_costs, 1, px);
        self.dispatch(pass, &p.box_rows, 2, elements);
        self.dispatch(pass, &p.box_cols, 3, elements);
    }

    /// Path costs (step 5).
    fn record_paths(&self, pass: &mut wgpu::ComputePass<'_>) {
        // Four scanlines per workgroup (sgbm.wgsl's LINES).
        let groups = (self.n_lines as u32).div_ceil(4);
        self.dispatch(pass, &self.pipelines.paths, 4, grid(groups));
    }

    /// Winners, left-right check, speckles and the disparity map (steps 6-8).
    fn record_disparity(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        let p = &self.pipelines;
        let px = gpu.dispatch_size(self.w as u32, self.h as u32);
        self.dispatch(pass, &p.winners, 5, px);
        self.dispatch(pass, &p.left_right, 6, px);
        if self.config.speckle_size >> (2 * self.config.level) > 0 {
            self.dispatch(pass, &p.unite, 7, px);
            self.dispatch(pass, &p.count, 8, px);
        }
        self.dispatch(pass, &p.finish, 9, px);
    }

    /// A frame (after `upload`): record it in the command buffer that first
    /// reads the uploaded pair (the Tegra stale-read rule; see
    /// gpu/frontend.rs).
    pub fn record_frame(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        self.record_spans(gpu, pass);
        self.record_costs(gpu, pass);
        self.record_paths(pass);
        self.record_disparity(gpu, pass);
    }

    /// A frame and the copy of its disparity map for the readback, into
    /// `encoder` (after `upload`; submit, then `arm_readback`).
    pub fn record_into(&self, gpu: &GpuDevice, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sgbm"),
                timestamp_writes: None,
            });
            self.record_frame(gpu, &mut pass);
        }
        encoder.copy_buffer_to_buffer(&self.buf.out, 0, &self.buf.readback, 0, self.buf.out.size());
    }

    /// Map the readback buffer: after `queue.submit`, before `device.poll`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.buf
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        self.readback_rx = Some(rx);
    }

    /// The depth map, after `device.poll` (as `Sgbm::compute_into`).
    pub fn collect_into(&mut self, out: &mut DenseDepth) {
        let rx = self
            .readback_rx
            .take()
            .expect("call arm_readback() before collect_into()");
        rx.recv().unwrap().expect("sgbm readback failed");
        {
            let mapped = self
                .buf
                .readback
                .slice(..)
                .get_mapped_range()
                .expect("buffer mapping failed");
            self.disp.clear();
            self.disp.extend_from_slice(bytemuck::cast_slice(&mapped));
        }
        self.buf.readback.unmap();
        depth_from_disparity(
            &self.disp,
            &self.log_scale,
            &self.config,
            self.w,
            self.h,
            out,
        );
    }

    /// The last collected disparity map (pixels of the level; NaN: none).
    pub fn disparity(&self) -> &[f32] {
        &self.disp
    }

    /// The depth of a rectified pair: upload, record, submit, wait and read
    /// back (for standalone use; a pipeline sharing the GPU records
    /// `record_into` into its own command buffer).
    pub fn compute_into(
        &mut self,
        gpu: &GpuDevice,
        left: &Image<u8>,
        right: &Image<u8>,
        out: &mut DenseDepth,
    ) {
        self.upload(gpu, left, right);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.record_into(gpu, &mut enc);
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
    use crate::sgbm::tests::scene;

    fn read<T: bytemuck::Pod>(gpu: &GpuDevice, b: &wgpu::Buffer) -> Vec<T> {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: b.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, b.size());
        gpu.queue.submit(std::iter::once(enc.finish()));
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let v = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        staging.unmap();
        v
    }

    /// A packed 16-bit volume (two disparities per word) as the CPU's.
    fn unpack(words: &[u32], nd: usize) -> Vec<u32> {
        let ndp = nd.div_ceil(2);
        words
            .chunks(ndp)
            .flat_map(|w| (0..nd).map(move |d| (w[d / 2] >> (16 * (d % 2))) & 0xffff))
            .collect()
    }

    fn run(gpu: &GpuDevice, f: impl FnOnce(&mut wgpu::ComputePass<'_>)) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            f(&mut pass);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_matches_cpu() {
        let (rig, left, right) = scene();
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let config = SgbmConfig {
            level: 0,
            min_range: 0.5,
            ..Default::default()
        };
        let mut cpu = Sgbm::new(&rig, config);
        let mut depth = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        cpu.compute_into(&left, &right, &mut depth);
        let mut g = GpuSgbm::new(&gpu, &cpu);

        // From the CPU's spans, stage by stage: the same integers.
        let spans: Vec<f32> = cpu.spans.iter().flatten().flatten().copied().collect();
        gpu.queue
            .write_buffer(&g.buf.spans, 0, bytemuck::cast_slice(&spans));
        run(&gpu, |pass| g.record_costs(&gpu, pass));
        assert!(
            unpack(&read::<u32>(&gpu, &g.buf.cost), g.nd) == cpu.block,
            "block costs differ"
        );
        run(&gpu, |pass| g.record_paths(pass));
        assert!(
            unpack(&read::<u32>(&gpu, &g.buf.total), g.nd) == cpu.total,
            "path sums differ"
        );
        run(&gpu, |pass| g.record_disparity(&gpu, pass));
        let out: Vec<f32> = read(&gpu, &g.buf.out);
        let same = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| !(x.is_nan() && y.is_nan() || (*x - *y).abs() <= 1e-5))
                .count()
        };
        let differ = same(&out, cpu.disparity());
        println!(
            "[test] from the CPU's spans: disparity differs on {differ} of {} pixels",
            out.len()
        );
        assert_eq!(differ, 0, "disparity maps differ");

        // The whole frame, the GPU's pyramids included.
        let mut gdepth = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        g.compute_into(&gpu, &left, &right, &mut gdepth);
        let differ = same(g.disparity(), cpu.disparity());
        let one_side = g
            .disparity()
            .iter()
            .zip(cpu.disparity())
            .filter(|(a, b)| a.is_nan() != b.is_nan())
            .count();
        println!(
            "[test] whole frame: disparity differs on {differ} of {} pixels, on one side only on {one_side}",
            out.len()
        );
        assert!(differ * 200 <= out.len(), "over 0.5 % of the frame differs");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU timing on recorded frames: set DENSE_STEREO_FRAMES"]
    fn inner_time_on_frames() {
        // Wall time per frame on a recorded pair (DENSE_STEREO_FRAMES: a
        // folder with left/, right/ and camchain.yaml): cumulative stages,
        // and whole frames with the readback, for 8 and 4 paths.
        let Some(dir) = std::env::var_os("DENSE_STEREO_FRAMES").map(std::path::PathBuf::from)
        else {
            println!("DENSE_STEREO_FRAMES not set; skipping");
            println!("GPU_TEST_OK");
            return;
        };
        let rig = crate::camera::StereoRig::from_kalibr_camchain(&dir.join("camchain.yaml"))
            .expect("camchain.yaml");
        let mut names: Vec<_> = std::fs::read_dir(dir.join("left"))
            .expect("left/")
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();
        names.sort();
        let load = |p: std::path::PathBuf| {
            let img = image::open(&p).expect("image").into_luma8();
            Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
        };
        let left = load(dir.join("left").join(&names[0]));
        let right = load(dir.join("right").join(&names[0]));
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let level: usize = std::env::var("SGBM_LEVEL").map_or(3, |v| v.parse().unwrap());
        let source = include_str!("../shaders/sgbm.wgsl");
        // Diagnostic variants of the path costs (wrong results, telling
        // timings): stores instead of atomic adds, no barriers, neither.
        let stores = source.replace(
            "atomicAdd(&total[i * P.nd + d], cur[j]);",
            "atomicStore(&total[i * P.nd + d], cur[j]);",
        );
        let variant = |name: &str| match name {
            "stores" => stores.clone(),
            "no barriers" => source.replace("workgroupBarrier();", ""),
            "stores, no barriers" => stores.replace("workgroupBarrier();", ""),
            _ => source.to_string(),
        };
        let variants: Vec<&str> = match std::env::var("SGBM_VARIANTS") {
            Ok(_) => vec!["as is", "stores", "no barriers", "stores, no barriers"],
            Err(_) => vec!["as is"],
        };
        for (paths, name) in [8, 4]
            .into_iter()
            .flat_map(|p| variants.iter().map(move |v| (p, *v)))
        {
            let cpu = Sgbm::new(
                &rig,
                SgbmConfig {
                    level,
                    paths,
                    ..Default::default()
                },
            );
            let mut g = GpuSgbm::with_shader(&gpu, &cpu, &variant(name));
            // Stages up to: 0 pyramids and spans, 1 pixel costs, 2 block
            // costs, 3 path costs, 4 winners, 5 left-right check, 6 speckles
            // and the disparity map.
            let frame = |g: &GpuSgbm, upto: usize| {
                g.upload(&gpu, &left, &right);
                run(&gpu, |pass| {
                    g.record_spans(&gpu, pass);
                    let px = gpu.dispatch_size(g.w as u32, g.h as u32);
                    let e = grid(((g.w * g.h * g.nd.div_ceil(2)) as u32).div_ceil(64));
                    if upto >= 1 {
                        g.dispatch(pass, &g.pipelines.pixel_costs, 1, px);
                    }
                    if upto >= 2 {
                        g.dispatch(pass, &g.pipelines.box_rows, 2, e);
                        g.dispatch(pass, &g.pipelines.box_cols, 3, e);
                    }
                    if upto >= 3 {
                        g.record_paths(pass);
                    }
                    if upto >= 4 {
                        g.dispatch(pass, &g.pipelines.winners, 5, px);
                    }
                    if upto >= 5 {
                        g.dispatch(pass, &g.pipelines.left_right, 6, px);
                    }
                    if upto >= 6 {
                        g.dispatch(pass, &g.pipelines.unite, 7, px);
                        g.dispatch(pass, &g.pipelines.count, 8, px);
                        g.dispatch(pass, &g.pipelines.finish, 9, px);
                    }
                });
            };
            // The best of three rounds of 30, after a warm-up long enough
            // for the GPU clock to settle.
            let time = |f: &mut dyn FnMut()| {
                (0..3)
                    .map(|_| {
                        let n = 30;
                        let t0 = std::time::Instant::now();
                        for _ in 0..n {
                            f();
                        }
                        t0.elapsed().as_secs_f64() * 1e3 / n as f64
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            for _ in 0..200 {
                frame(&g, 6);
            }
            let s = [0, 1, 2, 3, 4, 5, 6].map(|u| time(&mut || frame(&g, u)));
            // What "pyramids and spans" holds: the upload with an empty
            // submission, then the pyramids alone.
            let upload = time(&mut || {
                g.upload(&gpu, &left, &right);
                run(&gpu, |_| {});
            });
            let pyramids = time(&mut || {
                g.upload(&gpu, &left, &right);
                run(&gpu, |pass| {
                    for pyr in &g.pyr {
                        g.pyramid.record_build_in_pass(&gpu, pass, pyr);
                    }
                });
            });
            eprintln!(
                "[test]   of which upload and submission {upload:.2}, pyramids {:.2}, spans {:.2}",
                pyramids - upload,
                s[0] - pyramids
            );
            let mut depth = DenseDepth {
                width: 0,
                height: 0,
                level: 0,
                log_range: Vec::new(),
                variance: Vec::new(),
                source_level: Vec::new(),
            };
            let whole = time(&mut || g.compute_into(&gpu, &left, &right, &mut depth));
            eprintln!(
                "[test] {}, {}x{}x{}, {paths} paths ({} scanlines), {name}: ms per frame: pyramids and \
                 spans {:.2}, pixel costs {:.2}, block costs {:.2}, path costs {:.2}, winners \
                 {:.2}, left-right {:.2}, speckles {:.2}; whole frame with readback {whole:.2}",
                gpu.adapter_info.name,
                g.w,
                g.h,
                g.nd,
                g.n_lines,
                s[0],
                s[1] - s[0],
                s[2] - s[1],
                s[3] - s[2],
                s[4] - s[3],
                s[5] - s[4],
                s[6] - s[5],
            );
        }
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_matches_cpu() {
        let out = std::process::Command::new("cargo")
            .args([
                "test",
                "--lib",
                "--",
                "gpu::sgbm::tests::inner_matches_cpu",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .expect("subprocess");
        let out = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        print!("{out}");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
