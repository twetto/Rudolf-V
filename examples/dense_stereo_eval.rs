// examples/dense_stereo_eval.rs — Dense stereo depth on a sequence of stereo pairs.
//
//   cargo run --release --example dense_stereo_eval -- <dir> [options] > frames.jsonl
//
// <dir> is a EuRoC sequence (mav0/cam0, mav0/cam1 with sensor.yaml) or a folder with
// left/*.png, right/*.png (same file names) and camchain.yaml (Kalibr, cam0 and cam1).
//
// Options:
//   --frames a:b:s   frames a..b, every s-th            (default: all)
//   --level L        finest pyramid level                (default: 1)
//   --vr N           variational fixed-point iterations  (default: 5)
//   --sor N          SOR iterations per fixed-point one  (default: 5)
//   --gd N           Gauss-Newton iterations per patch and level (default: 16)
//   --sweeps N       red-black propagation sweeps        (default: 2)
//   --edge S         edge-aware smoothness, grey/px      (default: 0, off)
//   --speckle N      drop depth regions under N full-resolution pixels (default: 400)
//   --fallback N     coarser levels the fallback may fill from, 0 = off (default: 2)
//   --min-texture T  patch texture threshold, grey levels per pixel        (default: 0.5)
//   --bright B       largest mean grey-level difference between the eyes   (default: 30)
//   --sgm P1,P2[,U[,PATHS]]  semi-global fill at the finest level (SemiGlobal:
//                    penalties, uniqueness, 4 or 8 paths)
//   --rectify F,CX,CY  rectify the (raw) images into the pinhole camera F, (CX, CY)
//                    first; the depth is then on the rectified left grid
//   --rectified DIR  with --rectify, also write the rectified images to
//                    DIR/left and DIR/right (rounded to u8)
//   --gpu            match on the GPU (gpu/dense_stereo.rs; a rectified pair,
//                    or --rectify)
//   --trt ENGINE     depth from a stereo network's TensorRT engine instead
//                    (stereo_net.rs: LAS2 or Light ESS; a rectified pair), on
//                    --level's grid; with --trt-lib LIB (librv_trt.so, from
//                    native/tensorrt/build.sh; default: the loader's search),
//                    --trt-plugin LIB (Light ESS: ess_plugins.so) and
//                    --conf C (Light ESS: drop pixels under confidence C)
//   --out DIR        write <stem>.logr (f32 log-range, little-endian, row-major),
//                    <stem>.src (u8 source level per pixel, 255 = none) and
//                    <stem>.png (left image | range, 0.3 m red .. 6 m blue, none black)
//
// One JSON line per frame (file stem, milliseconds, coverage), a summary on stderr.
// All images are decoded before the first frame, so a profiler sees the matcher.

use rudolf_v::camera::StereoRig;
use rudolf_v::dense_stereo::{DenseDepth, DenseStereo, DenseStereoConfig, SemiGlobal};
use rudolf_v::gpu::dense_stereo::GpuDenseStereo;
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::image::Image;
use rudolf_v::rectify::StereoRectification;
use rudolf_v::stereo_net::{StereoNet, StereoNetConfig};
use rudolf_v::tensorrt::TrtEngine;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e))
        .into_luma8();
    Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
}

fn pngs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    files.sort();
    files
}

/// The rig and the left/right image lists of a sequence folder.
fn open_sequence(dir: &Path) -> (StereoRig, Vec<PathBuf>, PathBuf) {
    let mav0 = dir.join("mav0");
    if mav0.is_dir() {
        let rig = StereoRig::from_euroc(
            &mav0.join("cam0/sensor.yaml"),
            &mav0.join("cam1/sensor.yaml"),
        )
        .expect("EuRoC calibration");
        (rig, pngs(&mav0.join("cam0/data")), mav0.join("cam1/data"))
    } else {
        let rig =
            StereoRig::from_kalibr_camchain(&dir.join("camchain.yaml")).expect("camchain.yaml");
        (rig, pngs(&dir.join("left")), dir.join("right"))
    }
}

/// Range in metres → colour, near red to far blue.
fn colour(log_range: f32) -> [u8; 3] {
    if !log_range.is_finite() {
        return [0, 0, 0];
    }
    let (near, far) = (0.3f32.ln(), 6.0f32.ln());
    let t = ((log_range - near) / (far - near)).clamp(0.0, 1.0);
    let r = (255.0 * (1.0 - t)) as u8;
    let g = (255.0 * (1.0 - (2.0 * t - 1.0).abs())) as u8;
    let b = (255.0 * t) as u8;
    [r, g, b]
}

/// Eye `eye` of a raw pair through the matcher's rectification, rounded to u8.
fn rectified(rect: &StereoRectification, eye: usize, raw: &Image<u8>) -> Image<u8> {
    let mut f = Image::new(1, 1);
    rect.remap(eye, raw, &mut f);
    let data = f
        .as_slice()
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    Image::from_vec(f.width(), f.height(), data)
}

fn save_grey(img: &Image<u8>, path: &Path) {
    image::GrayImage::from_raw(
        img.width() as u32,
        img.height() as u32,
        img.as_slice().to_vec(),
    )
    .expect("image size")
    .save(path)
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

fn write_outputs(out: &Path, stem: &str, left: &Image<u8>, depth: &DenseDepth) {
    let bytes: Vec<u8> = depth
        .log_range
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out.join(format!("{stem}.logr")), bytes).expect("write .logr");
    std::fs::write(out.join(format!("{stem}.src")), &depth.source_level).expect("write .src");

    let (w, h) = (depth.width, depth.height);
    let s = 1usize << depth.level;
    let mut rgb = image::RgbImage::new(2 * w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let g = left.get(x * s, y * s);
            rgb.put_pixel(x as u32, y as u32, image::Rgb([g, g, g]));
            rgb.put_pixel(
                (w + x) as u32,
                y as u32,
                image::Rgb(colour(depth.log_range_at(x, y))),
            );
        }
    }
    rgb.save(out.join(format!("{stem}.png")))
        .expect("write png");
}

/// One pair's depth: from the network with `net`, else on the GPU with
/// `gpu`, else on the CPU.
fn compute(
    stereo: &mut DenseStereo,
    gpu: &mut Option<(GpuDevice, GpuDenseStereo)>,
    net: &mut Option<StereoNet>,
    left: &Image<u8>,
    right: &Image<u8>,
    depth: &mut DenseDepth,
) {
    match (net, gpu) {
        (Some(net), _) => net
            .compute_into(left, right, depth)
            .unwrap_or_else(|e| panic!("--trt: {e}")),
        (None, Some((dev, matcher))) => matcher.compute_into(dev, left, right, depth),
        (None, None) => stereo.compute_into(left, right, depth),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let use_gpu = match args.iter().position(|a| a == "--gpu") {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        args.remove(i);
        Some(args.remove(i))
    };
    let frames = take("--frames");
    let level: usize = take("--level").map_or(1, |v| v.parse().expect("--level"));
    let vr: usize = take("--vr").map_or(5, |v| v.parse().expect("--vr"));
    let sor: Option<usize> = take("--sor").map(|v| v.parse().expect("--sor"));
    let gd: Option<usize> = take("--gd").map(|v| v.parse().expect("--gd"));
    let sweeps: Option<usize> = take("--sweeps").map(|v| v.parse().expect("--sweeps"));
    let edge: f32 = take("--edge").map_or(0.0, |v| v.parse().expect("--edge"));
    let speckle: Option<usize> = take("--speckle").map(|v| v.parse().expect("--speckle"));
    let fallback: Option<usize> = take("--fallback").map(|v| v.parse().expect("--fallback"));
    let min_texture: Option<f32> = take("--min-texture").map(|v| v.parse().expect("--min-texture"));
    let bright: Option<f32> = take("--bright").map(|v| v.parse().expect("--bright"));
    let sgm = take("--sgm").map(|v| {
        let p: Vec<&str> = v.split(',').collect();
        let num = |i: usize| p.get(i).map(|x| x.parse::<f32>().expect("--sgm"));
        let d = SemiGlobal::default();
        SemiGlobal {
            p1: num(0).unwrap_or(d.p1),
            p2: num(1).unwrap_or(d.p2),
            uniqueness: num(2).unwrap_or(d.uniqueness),
            paths: num(3).map_or(d.paths, |x| x as usize),
        }
    });
    let rectify: Option<Vec<f64>> = take("--rectify").map(|v| {
        let p: Vec<f64> = v
            .split(',')
            .map(|x| x.parse().expect("--rectify F,CX,CY"))
            .collect();
        assert_eq!(p.len(), 3, "--rectify F,CX,CY");
        p
    });
    let rectified_out = take("--rectified").map(PathBuf::from);
    let trt = take("--trt").map(PathBuf::from);
    let trt_lib = take("--trt-lib").unwrap_or_else(|| "librv_trt.so".into());
    let trt_plugin = take("--trt-plugin").map(PathBuf::from);
    let conf: f32 = take("--conf").map_or(0.0, |v| v.parse().expect("--conf"));
    let out = take("--out").map(PathBuf::from);
    let dir = PathBuf::from(
        args.first()
            .expect("usage: dense_stereo_eval <dir> [options]"),
    );

    let (rig, lefts, right_dir) = open_sequence(&dir);
    let mut net = trt.as_ref().map(|engine| {
        assert!(
            rectify.is_none(),
            "--trt needs a rectified pair, not --rectify"
        );
        let plugins: Vec<&Path> = trt_plugin.iter().map(|p| p.as_path()).collect();
        let engine = TrtEngine::load(Path::new(&trt_lib), engine, &plugins)
            .unwrap_or_else(|e| panic!("--trt: {e}"));
        let config = StereoNetConfig {
            level,
            min_confidence: conf,
            ..Default::default()
        };
        StereoNet::new(engine, None, &rig, config).unwrap_or_else(|e| panic!("--trt: {e}"))
    });
    let (a, b, step) = match frames {
        Some(f) => {
            let p: Vec<usize> = f
                .split(':')
                .map(|v| v.parse().expect("--frames a:b:s"))
                .collect();
            (
                p[0],
                p[1].min(lefts.len()),
                p.get(2).copied().unwrap_or(1).max(1),
            )
        }
        None => (0, lefts.len(), 1),
    };
    let selected: Vec<&PathBuf> = lefts[a..b].iter().step_by(step).collect();
    let pairs: Vec<(String, Image<u8>, Image<u8>)> = selected
        .iter()
        .map(|l| {
            let name = l.file_name().unwrap();
            let stem = l.file_stem().unwrap().to_string_lossy().into_owned();
            (
                stem,
                load_grayscale(l),
                load_grayscale(&right_dir.join(name)),
            )
        })
        .collect();
    if let Some(out) = &out {
        std::fs::create_dir_all(out).expect("--out");
    }

    let config = DenseStereoConfig {
        finest_level: level,
        variational_iterations: vr,
        edge_sigma: edge,
        ..Default::default()
    };
    let config = DenseStereoConfig {
        speckle_size: speckle.unwrap_or(config.speckle_size),
        coarse_fallback: fallback.unwrap_or(config.coarse_fallback),
        sor_iterations: sor.unwrap_or(config.sor_iterations),
        grad_descent_iterations: gd.unwrap_or(config.grad_descent_iterations),
        propagation_sweeps: sweeps.unwrap_or(config.propagation_sweeps),
        min_texture: min_texture.unwrap_or(config.min_texture),
        max_brightness_offset: bright.unwrap_or(config.max_brightness_offset),
        semi_global: sgm,
        ..config
    };
    let mut stereo = match &rectify {
        Some(k) => {
            DenseStereo::new_rectifying(StereoRectification::new(&rig, k[0], k[1], k[2]), config)
        }
        None => DenseStereo::new(rig, config),
    };
    if let Some(dir) = &rectified_out {
        assert!(rectify.is_some(), "--rectified needs --rectify");
        for eye in ["left", "right"] {
            std::fs::create_dir_all(dir.join(eye)).expect("--rectified");
        }
    }
    let mut gpu = use_gpu.then(|| {
        let dev = GpuDevice::new().expect("--gpu needs a Vulkan GPU");
        let matcher = GpuDenseStereo::new(&dev, &stereo);
        (dev, matcher)
    });
    eprintln!(
        "{} pairs, {}x{}, levels {}..={}, {}, on {}",
        pairs.len(),
        pairs[0].1.width(),
        pairs[0].1.height(),
        level,
        stereo.coarsest_level(),
        if stereo.rectification().is_some() {
            "raw, rectified first"
        } else if stereo.is_rectified() {
            "rectified"
        } else {
            "raw (epipolar curves)"
        },
        match (&net, &gpu) {
            (Some(net), _) => format!("TensorRT ({:?}, {:?} input)", net.kind(), net.input_size()),
            (None, Some((dev, _))) => dev.adapter_info.name.clone(),
            (None, None) => "the CPU".to_string(),
        }
    );

    let mut depth = DenseDepth {
        width: 0,
        height: 0,
        level: 0,
        log_range: Vec::new(),
        variance: Vec::new(),
        source_level: Vec::new(),
    };
    compute(
        &mut stereo,
        &mut gpu,
        &mut net,
        &pairs[0].1,
        &pairs[0].2,
        &mut depth,
    ); // warm-up
    let mut ms = Vec::with_capacity(pairs.len());
    let mut coverage = 0.0f64;
    for (stem, left, right) in &pairs {
        let t = Instant::now();
        compute(&mut stereo, &mut gpu, &mut net, left, right, &mut depth);
        let dt = t.elapsed().as_secs_f64() * 1e3;
        ms.push(dt);
        coverage += depth.coverage() as f64;
        println!(
            "{{\"stem\":\"{stem}\",\"ms\":{dt:.3},\"coverage\":{:.4},\"width\":{},\"height\":{},\"level\":{}}}",
            depth.coverage(),
            depth.width,
            depth.height,
            depth.level
        );
        let rect_left = stereo.rectification().map(|r| rectified(r, 0, left));
        if let (Some(dir), Some(r)) = (&rectified_out, stereo.rectification()) {
            save_grey(
                rect_left.as_ref().unwrap(),
                &dir.join("left").join(format!("{stem}.png")),
            );
            save_grey(
                &rectified(r, 1, right),
                &dir.join("right").join(format!("{stem}.png")),
            );
        }
        if let Some(out) = &out {
            write_outputs(out, stem, rect_left.as_ref().unwrap_or(left), &depth);
        }
    }
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "ms per pair: median {:.2}, mean {:.2}, max {:.2}; coverage {:.3}",
        sorted[sorted.len() / 2],
        ms.iter().sum::<f64>() / ms.len() as f64,
        sorted[sorted.len() - 1],
        coverage / pairs.len() as f64
    );
}
