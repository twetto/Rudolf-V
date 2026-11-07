// gpu/clahe.rs — CLAHE on the GPU, as per-tile LUTs.
//
// The CLAHE counterpart of [`GpuGlobalHistEq`](crate::gpu::histeq::GpuGlobalHistEq)
// and it works the same way: one dispatch recorded into the frame's command
// buffer builds the LUTs, the pyramid's level-0 pass reads through them
// (`GpuPyramidPipeline::allocate_with_tile_lut`), and the LUTs are copied to a
// readback buffer in the same submit so CPU-side consumers that need equalized
// values at a few positions — LBP verification — can apply them to the raw
// image and get exactly the values the GPU used.
//
// The difference from the global case is that there is one LUT per tile rather
// than one for the frame, and the mapping a pixel sees is the bilinear blend of
// the four tile LUTs around it. That blend lives in the level-0 shaders
// (pyramid_convert_clahe.wgsl, pyramid_l0l1_clahe.wgsl) and, on the CPU side,
// in `frontend::compute_lbp_at_tile_lut`.
//
// WHY IT IS WORTH MOVING
// ───────────────────────
// CLAHE was the last frontend stage on the CPU in the GPU path: on an Orin NX
// it was ~2.3 ms, the largest CPU-side cost once KLT and FAST had moved, and
// 4.9 ms of a 10.5 ms single-core frontend on an x86 laptop.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramid;

const BINS: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ClaheParams {
    pub img_width: u32,
    pub img_height: u32,
    pub tile_size: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub clip_limit: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Where a pixel sits between tile centres along one axis: the two tiles it
/// blends and its weight toward the second.
#[derive(Clone, Copy, Debug)]
struct AxisBlend {
    t0: u32,
    t1: u32,
    a: f32,
}

/// The blend for every coordinate along one axis. It depends only on the image
/// size and tile size, so `GpuClahe` builds it once and every frame's
/// `TileLuts` shares it: `apply` then does no division, floor or clamp, which
/// made up most of the CPU LBP cost under CLAHE. Built from
/// `histeq::tile_blend`, the level-0 shaders' expression, so the results stay
/// bit-identical.
fn axis_blend(n_px: usize, n_tiles: usize, tile_size: usize) -> Arc<[AxisBlend]> {
    (0..n_px)
        .map(|p| {
            let (t0, t1, a) = crate::histeq::tile_blend(p, tile_size, n_tiles);
            AxisBlend {
                t0: t0 as u32,
                t1: t1 as u32,
                a,
            }
        })
        .collect()
}

/// The frame's per-tile LUTs, read back for CPU-side consumers.
#[derive(Clone, Debug)]
pub struct TileLuts {
    /// `tile_cols * tile_rows * 256` entries, tile-major.
    pub luts: Vec<u8>,
    pub tile_size: usize,
    pub tile_cols: usize,
    pub tile_rows: usize,
    cols: Arc<[AxisBlend]>,
    rows: Arc<[AxisBlend]>,
}

impl TileLuts {
    /// The equalized value of raw pixel `v` at `(x, y)`: the bilinear blend of
    /// the four surrounding tile LUTs.
    ///
    /// Same blend as `histeq::ClaheMap::remap_row` on the CPU path, in point
    /// form rather than per row; both index with `histeq::tile_blend` (here
    /// through the tables `axis_blend` builds). The authority is `shaders/pyramid_convert_clahe.wgsl`: this
    /// exists so a CPU consumer gets exactly the value the GPU produced, which
    /// is why `tile_blend` carries the shader's form.
    #[inline]
    pub fn apply(&self, x: usize, y: usize, v: u8) -> u8 {
        let bx = self.cols[x];
        let by = self.rows[y];
        let at = |t: usize| self.luts[t * BINS + v as usize] as f32;
        let row0 = by.t0 as usize * self.tile_cols;
        let row1 = by.t1 as usize * self.tile_cols;
        let (tx0, tx1) = (bx.t0 as usize, bx.t1 as usize);
        let top = at(row0 + tx0) + bx.a * (at(row0 + tx1) - at(row0 + tx0));
        let bot = at(row1 + tx0) + bx.a * (at(row1 + tx1) - at(row1 + tx0));
        (top * (1.0 - by.a) + bot * by.a).round().clamp(0.0, 255.0) as u8
    }
}

/// GPU CLAHE: per-tile LUTs for one image size, tile size and clip limit.
pub struct GpuClahe {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    luts_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    rb_buf: wgpu::Buffer,
    bg_cache: Vec<(wgpu::Texture, wgpu::BindGroup)>,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    tile_size: usize,
    tile_cols: usize,
    tile_rows: usize,
    cols: Arc<[AxisBlend]>,
    rows: Arc<[AxisBlend]>,
}

impl GpuClahe {
    pub fn new(
        gpu: &GpuDevice,
        width: usize,
        height: usize,
        tile_size: usize,
        clip_limit: f32,
    ) -> Self {
        assert!(tile_size > 0, "GpuClahe: tile_size must be non-zero");
        let tile_cols = width.div_ceil(tile_size);
        let tile_rows = height.div_ceil(tile_size);
        let n_luts = tile_cols * tile_rows;

        let module = gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
            label: Some("clahe_lut.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/clahe_lut.wgsl").into()),
        });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuClahe BGL"),
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

        let pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("clahe_luts"),
                    layout: Some(&gpu.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("GpuClahe layout"),
                            bind_group_layouts: &[Some(&bgl)],
                            immediate_size: 0,
                        },
                    )),
                    module: &module,
                    entry_point: Some("clahe_luts"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let params = ClaheParams {
            img_width: width as u32,
            img_height: height as u32,
            tile_size: tile_size as u32,
            tile_cols: tile_cols as u32,
            tile_rows: tile_rows as u32,
            clip_limit,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuClahe params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        // One byte per entry, packed four per u32 by clahe_lut.wgsl.
        let bytes = (n_luts * BINS) as u64;
        let luts_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuClahe tile LUTs"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuClahe LUT readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            luts_buf,
            params_buf,
            rb_buf,
            bg_cache: Vec::new(),
            readback_rx: None,
            tile_size,
            tile_cols,
            tile_rows,
            cols: axis_blend(width, tile_cols, tile_size),
            rows: axis_blend(height, tile_rows, tile_size),
        }
    }

    /// The tile-LUT buffer, for `allocate_with_tile_lut`.
    pub fn luts_buffer(&self) -> &wgpu::Buffer {
        &self.luts_buf
    }

    /// The tile-grid uniform, for `allocate_with_tile_lut`.
    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buf
    }

    /// LUT dispatch into an existing compute pass, reading the pyramid's raw
    /// texture. Pair with [`record_readback`](GpuClahe::record_readback).
    pub fn record_dispatches(
        &mut self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        pyr: &GpuPyramid,
    ) {
        let raw_view = pyr
            .raw_view()
            .expect("GpuClahe needs a pyramid from GpuPyramidPipeline::allocate*");
        let key = pyr.levels[0].texture.clone();
        let idx = match self.bg_cache.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GpuClahe BG"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(raw_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.luts_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.params_buf.as_entire_binding(),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg_cache[idx].1, &[]);
        pass.dispatch_workgroups((self.tile_cols * self.tile_rows) as u32, 1, 1);
    }

    /// Record the dispatch in its own pass, then the readback.
    pub fn record(
        &mut self,
        gpu: &GpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        pyr: &GpuPyramid,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clahe"),
                timestamp_writes: None,
            });
            self.record_dispatches(gpu, &mut pass, pyr);
        }
        self.record_readback(encoder);
    }

    pub fn record_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.luts_buf, 0, &self.rb_buf, 0, self.rb_buf.size());
    }

    /// Map the LUT readback. Call after `queue.submit()`, before `poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rb_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.readback_rx = Some(rx);
    }

    /// The frame's tile LUTs. Call after `device.poll(Wait)`.
    pub fn collect_luts(&mut self) -> TileLuts {
        let rx = self
            .readback_rx
            .take()
            .expect("call arm_readback() before collect_luts()");
        rx.recv().unwrap().expect("CLAHE LUT readback failed");
        let mapped = self
            .rb_buf
            .slice(..)
            .get_mapped_range()
            .expect("buffer mapping failed");
        // Entry 4k+j sits in byte j of word k, so on a little-endian host the
        // mapped bytes are the u8 LUTs in order.
        #[cfg(target_endian = "little")]
        let luts: Vec<u8> = mapped.to_vec();
        #[cfg(target_endian = "big")]
        let luts: Vec<u8> = bytemuck::cast_slice::<u8, u32>(&mapped)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        drop(mapped);
        self.rb_buf.unmap();
        TileLuts {
            luts,
            tile_size: self.tile_size,
            tile_cols: self.tile_cols,
            tile_rows: self.tile_rows,
            cols: Arc::clone(&self.cols),
            rows: Arc::clone(&self.rows),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::histeq;
    use crate::image::Image;

    pub(super) fn scene_pub(w: usize, h: usize) -> Image<u8> {
        scene(w, h)
    }

    fn scene(w: usize, h: usize) -> Image<u8> {
        let mut img = Image::new(w, h);
        let stride = img.stride();
        let px = img.as_mut_slice();
        let mut seed: u32 = 0x5bf0_3635;
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = ((seed >> 25) & 0x3f) as f32 - 32.0;
                let base = 70.0
                    + 90.0
                        * ((x as f32 / w as f32) * 7.0).sin()
                        * ((y as f32 / h as f32) * 4.0).cos();
                px[y * stride + x] = (base + noise).clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Level 0 built through the GPU tile LUTs must equal the CPU's
    /// `equalize_clahe_into` pixel for pixel, for both the fused l0/l1 kernel
    /// (>= 2 levels) and the standalone convert pass (1 level).
    #[test]
    fn gpu_clahe_level0_matches_cpu() {
        let Ok(gpu) = GpuDevice::new() else {
            eprintln!("no GPU adapter; skipping gpu_clahe_level0_matches_cpu");
            return;
        };
        let pipeline = GpuPyramidPipeline::new(&gpu);
        // Odd sizes exercise the partial tiles at the right/bottom edge.
        for &(w, h) in &[(640usize, 400usize), (501usize, 317usize)] {
            for &(tile, clip) in &[(256usize, 4.0f32), (64, 2.0), (128, 0.0)] {
                for &levels in &[1usize, 4] {
                    let img = scene(w, h);
                    let mut clahe = GpuClahe::new(&gpu, w, h, tile, clip);
                    let pyr = pipeline.allocate_with_tile_lut(
                        &gpu,
                        w,
                        h,
                        levels,
                        clahe.luts_buffer(),
                        clahe.params_buffer(),
                    );
                    pipeline.upload(&gpu, &img, &pyr);
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    {
                        let mut pass =
                            enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                        clahe.record_dispatches(&gpu, &mut pass, &pyr);
                        pipeline.record_build_in_pass(&gpu, &mut pass, &pyr);
                    }
                    clahe.record_readback(&mut enc);
                    gpu.queue.submit(std::iter::once(enc.finish()));
                    clahe.arm_readback();
                    gpu.device
                        .poll(wgpu::PollType::wait_indefinitely())
                        .expect("poll failed");
                    let tiles = clahe.collect_luts();

                    let mut want = Image::new(w, h);
                    histeq::equalize_clahe_into(&img, tile, clip, &mut want);
                    let got = pyr.readback_level(&gpu, 0);

                    let ws = want.stride();
                    let wp = want.as_slice();
                    let mut bad = 0usize;
                    for y in 0..h {
                        for x in 0..w {
                            let g = got[y * w + x];
                            let c = wp[y * ws + x] as f32;
                            if g != c {
                                bad += 1;
                                assert!(
                                    bad > 3,
                                    "{w}x{h} tile {tile} clip {clip} levels {levels}: \
                                     level0 differs at ({x},{y}): gpu {g} cpu {c}"
                                );
                            }
                            // The CPU-side tile-LUT application must agree too:
                            // it is what LBP verification uses.
                            let raw = img.as_slice()[y * img.stride() + x];
                            assert_eq!(
                                tiles.apply(x, y, raw) as f32,
                                c,
                                "{w}x{h} tile {tile}: TileLuts::apply differs at ({x},{y})"
                            );
                        }
                    }
                    assert_eq!(bad, 0, "{bad} pixels differ");
                }
            }
        }
    }
}

#[cfg(test)]
mod gpu_share {
    use super::*;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::image::Image;
    use std::time::Instant;

    /// How much of a frame the GPU is actually busy for.
    /// `cargo test --release --lib gpu_busy_share -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn gpu_busy_share() {
        let Ok(gpu) = GpuDevice::new() else { return };
        println!(
            "  adapter: {} ({:?})",
            gpu.adapter_info.name, gpu.adapter_info.backend
        );
        let (w, h) = (1280usize, 800usize);
        let img = super::tests::scene_pub(w, h);
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let mut clahe = GpuClahe::new(&gpu, w, h, 256, 4.0);
        let pyr = pipeline.allocate_with_tile_lut(
            &gpu,
            w,
            h,
            4,
            clahe.luts_buffer(),
            clahe.params_buffer(),
        );
        pipeline.upload(&gpu, &img, &pyr);

        let mut once = |clahe_on: bool| -> f64 {
            let t0 = Instant::now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                if clahe_on {
                    clahe.record_dispatches(&gpu, &mut pass, &pyr);
                }
                pipeline.record_build_in_pass(&gpu, &mut pass, &pyr);
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..20 {
            once(false);
            once(true);
        }
        // Interleaved, alternating order: the clock drifts, the ratio does not.
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for i in 0..80 {
            if i % 2 == 0 {
                a.push(once(false));
                b.push(once(true));
            } else {
                b.push(once(true));
                a.push(once(false));
            }
        }
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let without = a[a.len() / 2];
        let with = b[b.len() / 2];
        println!("  pyramid only          : {without:.3} ms (submit + wait)");
        println!("  pyramid + CLAHE LUTs  : {with:.3} ms (submit + wait)");
        println!(
            "  => CLAHE dispatch costs {:.3} ms of GPU time",
            with - without
        );
    }

    /// Where the GPU CLAHE time goes, stage by stage, at EuRoC size.
    /// `RUDOLF_PROFILE_IMG=<png> cargo test --release --lib clahe_stage_profile -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn clahe_stage_profile() {
        let Ok(gpu) = GpuDevice::new() else { return };
        println!("  adapter: {}", gpu.adapter_info.name);
        let img = match std::env::var("RUDOLF_PROFILE_IMG") {
            Ok(p) => {
                let g = image::open(&p).expect("load RUDOLF_PROFILE_IMG").to_luma8();
                let (w, h) = (g.width() as usize, g.height() as usize);
                let mut img = Image::new(w, h);
                let stride = img.stride();
                let px = img.as_mut_slice();
                for y in 0..h {
                    px[y * stride..y * stride + w].copy_from_slice(&g.as_raw()[y * w..(y + 1) * w]);
                }
                img
            }
            Err(_) => super::tests::scene_pub(752, 480),
        };
        let (w, h) = (img.width(), img.height());
        let (tile, clip, levels) = (32usize, 2.0f32, 3usize);
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let mut clahe = GpuClahe::new(&gpu, w, h, tile, clip);
        let plain = pipeline.allocate(&gpu, w, h, levels);
        let tiled = pipeline.allocate_with_tile_lut(
            &gpu,
            w,
            h,
            levels,
            clahe.luts_buffer(),
            clahe.params_buffer(),
        );
        pipeline.upload(&gpu, &img, &plain);
        pipeline.upload(&gpu, &img, &tiled);

        // 0: plain pyramid, 1: CLAHE pyramid (blend only), 2: + LUT build,
        // 3: + LUT readback copy and map.
        let mut once = |variant: usize| -> f64 {
            let t0 = Instant::now();
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                if variant >= 2 {
                    clahe.record_dispatches(&gpu, &mut pass, &tiled);
                }
                let pyr = if variant == 0 { &plain } else { &tiled };
                pipeline.record_build_in_pass(&gpu, &mut pass, pyr);
            }
            if variant == 3 {
                clahe.record_readback(&mut enc);
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            if variant == 3 {
                clahe.arm_readback();
            }
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            if variant == 3 {
                let _ = clahe.collect_luts();
            }
            t0.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..50 {
            for v in 0..4 {
                once(v);
            }
        }
        let mut t: [Vec<f64>; 4] = Default::default();
        for i in 0..400 {
            for k in 0..4 {
                let v = (i + k) % 4; // rotate the order so clock drift cancels
                t[v].push(once(v));
            }
        }
        let med: Vec<f64> = t
            .iter_mut()
            .map(|v| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            })
            .collect();
        println!("  {w}x{h}, tile {tile}, clip {clip}, {levels} levels (median submit + wait)");
        println!("  plain pyramid            : {:.3} ms", med[0]);
        println!(
            "  CLAHE pyramid (blend)    : {:.3} ms  (+{:.3})",
            med[1],
            med[1] - med[0]
        );
        println!(
            "  + LUT build              : {:.3} ms  (+{:.3})",
            med[2],
            med[2] - med[1]
        );
        println!(
            "  + LUT readback           : {:.3} ms  (+{:.3})",
            med[3],
            med[3] - med[2]
        );

        // CPU side: collect_luts alone, and LBP at 200 features through the tile LUTs.
        let mut collect = Vec::new();
        let mut tiles = None;
        for _ in 0..200 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            clahe.record_readback(&mut enc);
            gpu.queue.submit(std::iter::once(enc.finish()));
            clahe.arm_readback();
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let t0 = Instant::now();
            tiles = Some(clahe.collect_luts());
            collect.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        collect.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tiles = tiles.unwrap();
        let mut seed = 12345u32;
        let feats: Vec<(f32, f32)> = (0..200)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = 8.0 + (seed >> 8) as f32 / (1u32 << 24) as f32 * (w as f32 - 16.0);
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let y = 8.0 + (seed >> 8) as f32 / (1u32 << 24) as f32 * (h as f32 - 16.0);
                (x, y)
            })
            .collect();
        let mut lbp = Vec::new();
        let mut sink = 0u32;
        for _ in 0..500 {
            let t0 = Instant::now();
            for &(x, y) in &feats {
                sink = sink.wrapping_add(
                    crate::frontend::compute_lbp_at_tile_lut(&img, &tiles, x, y).unwrap_or(0)
                        as u32,
                );
            }
            lbp.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        lbp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  CPU collect_luts         : {:.3} ms",
            collect[collect.len() / 2]
        );
        println!(
            "  CPU LBP x200 (tile LUTs) : {:.3} ms  (sink {sink})",
            lbp[lbp.len() / 2]
        );
    }
}
