// stereo_net.rs — Depth from a stereo network's TensorRT engine
// (tensorrt.rs), as a DenseDepth on the same grid DenseStereo writes, so the
// two are interchangeable downstream.
//
// Two networks, told apart by their tensor names:
//   LAS2 (LiteAnyStereo V2, any fixed input size): inputs `left`, `right`,
//     1×3×H×W RGB in 0..255 (it normalizes internally); output `disparity`.
//     The image is resized to the input's width keeping its aspect ratio
//     (1280×720 → 320×180 for a 320×192 input), centred, its first and last
//     rows repeated into the remaining rows (the LAS repository's
//     InputPadder).
//   Light ESS (NVIDIA Isaac ROS, 480×288): inputs `input_left`,
//     `input_right`, RGB scaled to [−1, 1]; outputs `output_left`
//     (disparity) and `output_conf` (confidence, 0..1). The image is
//     stretched to the input size, as Isaac ROS does.
// Both take the grey image in all three channels, resized by area averaging
// (OpenCV's INTER_AREA), and give disparity in their input's pixels. The
// resizing and scaling run on the GPU where the TensorRT library can
// (TrtEngine::gpu_input), with the CPU's arithmetic: the same inputs, bit for
// bit, and only the images' bytes to copy. With an older library they run on
// the CPU.
//
// The pair must be rectified (a pinhole rig, cameras side by side, the right
// one at +x): disparity d then puts the left pixel's point at range
// f·B / d · |K⁻¹(u, v, 1)| with d in full-resolution pixels. Each depth map
// pixel takes the nearest network pixel's disparity. There is no per-pixel
// uncertainty: the variance assumes `disparity_sigma` network pixels.
//
// Check a network's terms before shipping it: LAS2's weights are MIT, but
// part of its training used research-only FoundationStereo labels; Light
// ESS's EULA allows commercial use on NVIDIA GPUs.

use crate::camera::StereoRig;
use crate::dense_stereo::{DenseDepth, NO_SOURCE, rectified_sign};
use crate::image::Image;
use crate::tensorrt::{GpuResize, TrtEngine};

/// Which network an engine holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoNetKind {
    Las2,
    LightEss,
}

impl StereoNetKind {
    /// The network an engine's tensor names belong to.
    pub fn detect(engine: &TrtEngine) -> Option<Self> {
        let has = |n: &str| engine.find(n).is_some();
        if has("input_left") && has("input_right") && has("output_left") {
            Some(Self::LightEss)
        } else if has("left") && has("right") && has("disparity") {
            Some(Self::Las2)
        } else {
            None
        }
    }

    /// v·scale + offset per resized pixel: the network's input range.
    fn scaling(self) -> (f32, f32) {
        match self {
            Self::Las2 => (1.0, 0.0),
            Self::LightEss => (1.0 / 127.5, -1.0),
        }
    }

    fn names(
        self,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        Option<&'static str>,
    ) {
        match self {
            Self::Las2 => ("left", "right", "disparity", None),
            Self::LightEss => (
                "input_left",
                "input_right",
                "output_left",
                Some("output_conf"),
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StereoNetConfig {
    /// Pyramid level of the depth map: its pixel (x, y) is image pixel
    /// (x·2^level, y·2^level), as DenseStereoConfig::finest_level.
    pub level: usize,
    /// Nearest and farthest range kept (metres).
    pub min_range: f64,
    pub max_range: f64,
    /// Pixels the network is less sure of than this have no depth (Light
    /// ESS's confidence, 0..1; Isaac ROS drops under 0.4). Networks without
    /// a confidence keep every pixel.
    pub min_confidence: f32,
    /// Disparity standard deviation, in the network's pixels, behind the
    /// variance of each pixel's log-range.
    pub disparity_sigma: f32,
}

impl Default for StereoNetConfig {
    fn default() -> Self {
        Self {
            level: 3,
            min_range: 0.25,
            max_range: 30.0,
            min_confidence: 0.0,
            disparity_sigma: 0.5,
        }
    }
}

/// Where the image goes in the network's input: `cw`×`ch` pixels from row
/// `top` of an `nw`×`nh` input, from a `w0`×`h0` image.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Layout {
    w0: usize,
    h0: usize,
    nw: usize,
    nh: usize,
    cw: usize,
    ch: usize,
    top: usize,
}

impl Layout {
    fn new(
        kind: StereoNetKind,
        w0: usize,
        h0: usize,
        nw: usize,
        nh: usize,
    ) -> Result<Self, String> {
        let (cw, ch) = match kind {
            StereoNetKind::Las2 => (nw, ((h0 * nw) as f64 / w0 as f64).round() as usize),
            StereoNetKind::LightEss => (nw, nh),
        };
        if ch == 0 || ch > nh {
            return Err(format!(
                "a {w0}x{h0} image does not fit the network's {nw}x{nh} input"
            ));
        }
        Ok(Self {
            w0,
            h0,
            nw,
            nh,
            cw,
            ch,
            top: (nh - ch) / 2,
        })
    }

    /// The network pixel nearest image pixel (x, y): its index in the
    /// network's output.
    fn nearest(&self, x: usize, y: usize) -> usize {
        let map = |p: usize, content: usize, full: usize| {
            ((((p as f64 + 0.5) * content as f64 / full as f64) - 0.5).round() as isize)
                .clamp(0, content as isize - 1) as usize
        };
        (map(y, self.ch, self.h0) + self.top) * self.nw + map(x, self.cw, self.w0)
    }
}

/// Per output pixel: the first input pixel it covers and the weights of the
/// input pixels it covers (area averaging, as OpenCV's INTER_AREA when
/// shrinking).
fn area_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|i| {
            let (a, b) = (i as f64 * scale, (i + 1) as f64 * scale);
            let first = a.floor() as usize;
            let last = (b.ceil() as usize).min(src);
            let w = (first..last)
                .map(|j| (((j + 1) as f64).min(b) - (j as f64).max(a)) / scale)
                .map(|w| w as f32)
                .collect();
            (first, w)
        })
        .collect()
}

/// `img` shrunk to `cols.len()`×`rows.len()` by area averaging, row-major
/// into `out`. Rows first: each output row sums its input rows into `acc`
/// (whole rows, which vectorize), then its columns from there.
fn area_resize(
    img: &Image<u8>,
    cols: &[(usize, Vec<f32>)],
    rows: &[(usize, Vec<f32>)],
    acc: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    let (w, dw) = (img.width(), cols.len());
    let (data, stride) = (img.as_slice(), img.stride());
    acc.resize(w, 0.0);
    out.resize(rows.len() * dw, 0.0);
    for (row, (first, wr)) in out.chunks_mut(dw).zip(rows) {
        acc.fill(0.0);
        for (k, &wk) in wr.iter().enumerate() {
            let src = &data[(first + k) * stride..][..w];
            for (a, &v) in acc.iter_mut().zip(src) {
                *a += wk * v as f32;
            }
        }
        for (o, (first, wc)) in row.iter_mut().zip(cols) {
            *o = wc.iter().zip(&acc[*first..]).map(|(w, a)| w * a).sum();
        }
    }
}

/// Depth from a stereo network (header).
pub struct StereoNet {
    engine: TrtEngine,
    kind: StereoNetKind,
    config: StereoNetConfig,
    layout: Layout,
    /// Engine tensors: left and right inputs, disparity, confidence.
    io: [usize; 3],
    conf: Option<usize>,
    cols: Vec<(usize, Vec<f32>)>,
    rows: Vec<(usize, Vec<f32>)>,
    /// Per depth map pixel: its network pixel, and ln(f·B·|K⁻¹(u, v, 1)|)
    /// less ln(full-resolution pixels per network pixel), so that its
    /// log-range is this minus ln(disparity in network pixels).
    pixel: Vec<usize>,
    log_scale: Vec<f32>,
    /// ln(disparity_sigma) in network pixels, for the variance.
    log_sigma: f32,
    /// The inputs are made on the GPU from the images.
    gpu_inputs: bool,
    /// fill's scratch: area_resize's row sums and output, and one channel.
    acc: Vec<f32>,
    small: Vec<f32>,
    plane: Vec<f32>,
}

impl StereoNet {
    /// The network in `engine` (`kind`, or told from its tensor names) for a
    /// rectified rig.
    pub fn new(
        mut engine: TrtEngine,
        kind: Option<StereoNetKind>,
        rig: &StereoRig,
        config: StereoNetConfig,
    ) -> Result<Self, String> {
        let kind = match kind.or_else(|| StereoNetKind::detect(&engine)) {
            Some(k) => k,
            None => return Err("unknown network: expected LAS2's or Light ESS's tensors".into()),
        };
        if rectified_sign(rig) != Some(-1.0) {
            return Err("the network needs a rectified pair with the right camera at +x".into());
        }
        let (l, r, d, c) = kind.names();
        let find = |n: &str| {
            engine
                .find(n)
                .ok_or_else(|| format!("the engine has no tensor {n}"))
        };
        let io = [find(l)?, find(r)?, find(d)?];
        let conf = c.map(find).transpose()?;
        let dims = &engine.tensors()[io[0]].dims;
        if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
            return Err(format!("expected a 1x3xHxW input, the engine has {dims:?}"));
        }
        let (nw, nh) = (dims[3], dims[2]);
        for &i in io[2..].iter().chain(conf.iter()) {
            if engine.tensors()[i].dims.iter().product::<usize>() != nw * nh {
                return Err(format!(
                    "output {} is not {nw}x{nh}",
                    engine.tensors()[i].name
                ));
            }
        }
        let [w0, h0] = rig.cam0.resolution;
        let layout = Layout::new(kind, w0, h0, nw, nh)?;

        let c0 = &rig.cam0;
        let fb = c0.fx * rig.baseline_meters();
        let s = 1usize << config.level;
        let (w, h) = (w0 >> config.level, h0 >> config.level);
        let mut pixel = Vec::with_capacity(w * h);
        let mut log_scale = Vec::with_capacity(w * h);
        let per_net_pixel = w0 as f64 / layout.cw as f64;
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x * s) as f64, (y * s) as f64);
                let (a, b) = ((u - c0.cx) / c0.fx, (v - c0.cy) / c0.fy);
                let norm = (a * a + b * b + 1.0).sqrt();
                pixel.push(layout.nearest(x * s, y * s));
                log_scale.push((fb * norm / per_net_pixel).ln() as f32);
            }
        }
        let (cols, rows) = (area_weights(w0, layout.cw), area_weights(h0, layout.ch));
        let (scale, offset) = kind.scaling();
        let resize = GpuResize {
            width: w0,
            height: h0,
            cols: &cols,
            rows: &rows,
            top: layout.top,
            scale,
            offset,
        };
        let mut gpu_inputs = true;
        for &t in &io[..2] {
            gpu_inputs &= engine.gpu_input(t, &resize)?;
        }
        Ok(Self {
            engine,
            kind,
            log_sigma: config.disparity_sigma.ln(),
            config,
            layout,
            io,
            conf,
            cols,
            rows,
            gpu_inputs,
            pixel,
            log_scale,
            acc: Vec::new(),
            small: Vec::new(),
            plane: Vec::new(),
        })
    }

    pub fn kind(&self) -> StereoNetKind {
        self.kind
    }

    /// The network's input size, width and height.
    pub fn input_size(&self) -> (usize, usize) {
        (self.layout.nw, self.layout.nh)
    }

    /// One eye's input plane on the CPU, into `plane`.
    fn cpu_plane(&mut self, img: &Image<u8>) {
        area_resize(img, &self.cols, &self.rows, &mut self.acc, &mut self.small);
        let Layout {
            nw, nh, ch, top, ..
        } = self.layout;
        let (scale, offset) = self.kind.scaling();
        self.plane.resize(nw * nh, 0.0);
        for (r, row) in self.plane.chunks_mut(nw).enumerate() {
            let src = r.saturating_sub(top).min(ch - 1);
            for (o, &v) in row.iter_mut().zip(&self.small[src * nw..(src + 1) * nw]) {
                *o = v * scale + offset;
            }
        }
    }

    /// One eye into input tensor `t`: on the GPU, only the image into its
    /// pinned buffer; else the plane built on the CPU, written to the
    /// tensor's pinned buffer three times (one per channel), never read back
    /// from it.
    fn fill(&mut self, img: &Image<u8>, t: usize) {
        if let Some(buf) = self.engine.image_mut(t) {
            for (y, dst) in buf.chunks_exact_mut(img.width()).enumerate() {
                dst.copy_from_slice(img.row(y));
            }
            return;
        }
        self.cpu_plane(img);
        let plane = &self.plane;
        for channel in self.engine.host_mut(t).chunks_mut(plane.len()) {
            channel.copy_from_slice(plane);
        }
    }

    /// Whether the inputs are made on the GPU (TrtEngine::gpu_input).
    pub fn preprocesses_on_gpu(&self) -> bool {
        self.gpu_inputs
    }

    /// The depth of a rectified pair into `out`.
    pub fn compute_into(
        &mut self,
        left: &Image<u8>,
        right: &Image<u8>,
        out: &mut DenseDepth,
    ) -> Result<(), String> {
        let (w0, h0) = (self.layout.w0, self.layout.h0);
        for img in [left, right] {
            if (img.width(), img.height()) != (w0, h0) {
                return Err(format!(
                    "expected {w0}x{h0} images, got {}x{}",
                    img.width(),
                    img.height()
                ));
            }
        }
        self.fill(left, self.io[0]);
        self.fill(right, self.io[1]);
        self.engine.run()?;

        let level = self.config.level;
        let n = self.pixel.len();
        out.width = w0 >> level;
        out.height = h0 >> level;
        out.level = level;
        out.log_range.resize(n, f32::NAN);
        out.variance.resize(n, f32::NAN);
        out.source_level.resize(n, NO_SOURCE);
        let disp = self.engine.host(self.io[2]);
        let conf = self.conf.map(|c| self.engine.host(c));
        let (lo, hi) = (
            self.config.min_range.ln() as f32,
            self.config.max_range.ln() as f32,
        );
        for i in 0..n {
            let p = self.pixel[i];
            let d = disp[p];
            let sure = conf.is_none_or(|c| c[p] >= self.config.min_confidence);
            let log_range = self.log_scale[i] - d.ln();
            (out.log_range[i], out.variance[i], out.source_level[i]) =
                if d > 0.0 && sure && log_range >= lo && log_range <= hi {
                    let rel = (self.log_sigma - d.ln()).exp();
                    (log_range, rel * rel, level as u8)
                } else {
                    (f32::NAN, f32::NAN, NO_SOURCE)
                };
        }
        Ok(())
    }

    /// The last pair's disparity in the network's pixels, `input_size()`
    /// row-major (rows outside the image's are padding).
    pub fn disparity(&self) -> &[f32] {
        self.engine.host(self.io[2])
    }

    /// The last pair's confidence (0..1) at each depth map pixel, from the
    /// network pixel its depth came from; None for a network without one
    /// (LAS2).
    pub fn confidence(&self) -> Option<Vec<f32>> {
        let conf = self.engine.host(self.conf?);
        Some(self.pixel.iter().map(|&p| conf[p]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn las2_keeps_the_aspect_and_centres() {
        let l = Layout::new(StereoNetKind::Las2, 1280, 720, 320, 192).unwrap();
        assert_eq!((l.cw, l.ch, l.top), (320, 180, 6));
        assert!(Layout::new(StereoNetKind::Las2, 1280, 720, 320, 160).is_err());
        let e = Layout::new(StereoNetKind::LightEss, 1280, 720, 480, 288).unwrap();
        assert_eq!((e.cw, e.ch, e.top), (480, 288, 0));
    }

    #[test]
    fn nearest_network_pixel() {
        let l = Layout::new(StereoNetKind::Las2, 1280, 720, 320, 192).unwrap();
        // Image pixels 0..=3 average into network pixel 0, whose centre is 1.5.
        assert_eq!(l.nearest(0, 0), 6 * 320);
        assert_eq!(l.nearest(8, 8), (2 + 6) * 320 + 2);
        assert_eq!(l.nearest(1279, 719), (179 + 6) * 320 + 319);
    }

    fn resize(img: &Image<u8>, dw: usize, dh: usize) -> Vec<f32> {
        let (cols, rows) = (
            area_weights(img.width(), dw),
            area_weights(img.height(), dh),
        );
        let (mut acc, mut out) = (Vec::new(), Vec::new());
        area_resize(img, &cols, &rows, &mut acc, &mut out);
        out
    }

    #[test]
    fn area_resize_is_the_box_mean_at_whole_factors() {
        let (w, h) = (16, 8);
        let img = Image::from_vec(w, h, (0..w * h).map(|i| (i * 7 % 251) as u8).collect());
        let out = resize(&img, 4, 2);
        for oy in 0..2 {
            for ox in 0..4 {
                let mean = (0..4)
                    .flat_map(|dy| (0..4).map(move |dx| (dx, dy)))
                    .map(|(dx, dy)| img.get(ox * 4 + dx, oy * 4 + dy) as f32)
                    .sum::<f32>()
                    / 16.0;
                assert!((out[oy * 4 + ox] - mean).abs() < 1e-3);
            }
        }
    }

    /// The engine named by RUDOLF_V_TRT_ENGINE, through the library named by
    /// RUDOLF_V_TRT_LIB (plugin libraries: RUDOLF_V_TRT_PLUGINS,
    /// colon-separated); None without them, and the test is skipped.
    fn engine_from_env() -> Option<TrtEngine> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let (Some(engine), Some(lib)) = (var("RUDOLF_V_TRT_ENGINE"), var("RUDOLF_V_TRT_LIB"))
        else {
            eprintln!("skipped: no TensorRT engine (RUDOLF_V_TRT_ENGINE, RUDOLF_V_TRT_LIB)");
            return None;
        };
        let plugins = var("RUDOLF_V_TRT_PLUGINS").unwrap_or_default();
        let plugins: Vec<&std::path::Path> = plugins
            .split(':')
            .filter(|p| !p.is_empty())
            .map(std::path::Path::new)
            .collect();
        Some(TrtEngine::load(lib.as_ref(), engine.as_ref(), &plugins).unwrap())
    }

    /// A 1280×720 texture in rows of `stride` bytes.
    fn texture(stride: usize, seed: u32) -> Image<u8> {
        let data = (0..stride * 720)
            .map(|i| ((i as u32 ^ seed).wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        Image::from_vec_with_stride(1280, 720, stride, data)
    }

    #[test]
    fn gpu_inputs_are_the_cpus() {
        let Some(engine) = engine_from_env() else {
            return;
        };
        let cam = crate::camera::CameraIntrinsics::new(951.4, 951.4, 658.9, 356.7, 1280, 720);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rig = StereoRig::new(cam.clone(), cam, identity, [-0.12, 0.0, 0.0]);
        let mut net = StereoNet::new(engine, None, &rig, StereoNetConfig::default()).unwrap();
        assert!(net.preprocesses_on_gpu());
        let mut out = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        // Rows wider than the image (the copy must take only the pixels);
        // the first run is eager, the later ones replay its CUDA graph.
        let a = [texture(1296, 1), texture(1296, 2)];
        let b = [texture(1280, 3), texture(1280, 4)];
        let mut first = Vec::new();
        for (run, pair) in [&a, &b, &a].into_iter().enumerate() {
            net.compute_into(&pair[0], &pair[1], &mut out).unwrap();
            for (eye, img) in pair.iter().enumerate() {
                let t = net.io[eye];
                net.engine.input_to_host(t).unwrap();
                net.cpu_plane(img);
                for channel in net.engine.host(t).chunks(net.plane.len()) {
                    let same = channel
                        .iter()
                        .zip(&net.plane)
                        .all(|(g, c)| g.to_bits() == c.to_bits());
                    assert!(same, "run {run}, eye {eye}: the GPU's input differs");
                }
            }
            match run {
                0 => first = net.disparity().to_vec(),
                2 => assert!(net.disparity() == first, "the replay differs"),
                _ => {}
            }
        }
    }

    #[test]
    fn area_resize_keeps_constants_and_the_mean() {
        let flat = Image::from_vec(1280, 720, vec![200u8; 1280 * 720]);
        assert!(
            resize(&flat, 480, 288)
                .iter()
                .all(|&v| (v - 200.0).abs() < 1e-3)
        );
        let ramp = Image::from_vec(
            1280,
            4,
            (0..1280 * 4).map(|i| (i % 1280 / 5) as u8).collect(),
        );
        let out = resize(&ramp, 480, 3);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let src: Vec<f32> = ramp.as_slice().iter().map(|&v| v as f32).collect();
        assert!((mean(&out) - mean(&src)).abs() < 0.01);
    }
}
