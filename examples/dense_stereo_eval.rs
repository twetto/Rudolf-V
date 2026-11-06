// examples/dense_stereo_eval.rs — Dense stereo depth on a sequence of stereo pairs.
//
//   cargo run --release --example dense_stereo_eval -- <dir> [options] > frames.jsonl
//
// <dir> is a EuRoC sequence (mav0/cam0, mav0/cam1 with sensor.yaml) or a folder with
// left/*.png, right/*.png (same file names) and camchain.yaml (Kalibr, cam0 and cam1).
//
// Options:
//   --frames a:b:s   frames a..b, every s-th            (default: all)
//   --level L        finest pyramid level                (default: 0.5)
//   --vr N           variational fixed-point iterations  (default: 5)
//   --edge S         edge-aware smoothness, grey/px      (default: 0, off)
//   --speckle N      drop depth regions under N pixels   (default: 100)
//   --fallback N     coarser levels the fallback may fill from, 0 = off (default: 2)
//   --min-texture T  patch texture threshold, grey levels per pixel        (default: 0.5)
//   --out DIR        write <stem>.logr (f32 log-range, little-endian, row-major),
//                    <stem>.src (u8 source level per pixel, 255 = none) and
//                    <stem>.png (left image | range, 0.3 m red .. 6 m blue, none black)
//
// One JSON line per frame (file stem, milliseconds, coverage), a summary on stderr.
// All images are decoded before the first frame, so a profiler sees the matcher.

use rudolf_v::camera::StereoRig;
use rudolf_v::dense_stereo::{DenseDepth, DenseStereo, DenseStereoConfig};
use rudolf_v::image::Image;
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

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        args.remove(i);
        Some(args.remove(i))
    };
    let frames = take("--frames");
    let level: usize = take("--level").map_or(1, |v| v.parse().expect("--level"));
    let vr: usize = take("--vr").map_or(5, |v| v.parse().expect("--vr"));
    let edge: f32 = take("--edge").map_or(0.0, |v| v.parse().expect("--edge"));
    let speckle: Option<usize> = take("--speckle").map(|v| v.parse().expect("--speckle"));
    let fallback: Option<usize> = take("--fallback").map(|v| v.parse().expect("--fallback"));
    let min_texture: Option<f32> = take("--min-texture").map(|v| v.parse().expect("--min-texture"));
    let out = take("--out").map(PathBuf::from);
    let dir = PathBuf::from(
        args.first()
            .expect("usage: dense_stereo_eval <dir> [options]"),
    );

    let (rig, lefts, right_dir) = open_sequence(&dir);
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
        min_texture: min_texture.unwrap_or(config.min_texture),
        ..config
    };
    let mut stereo = DenseStereo::new(rig, config);
    eprintln!(
        "{} pairs, {}x{}, levels {}..={}, {}",
        pairs.len(),
        pairs[0].1.width(),
        pairs[0].1.height(),
        level,
        stereo.coarsest_level(),
        if stereo.is_rectified() {
            "rectified"
        } else {
            "raw (epipolar curves)"
        }
    );

    let mut depth = stereo.compute(&pairs[0].1, &pairs[0].2); // warm-up
    let mut ms = Vec::with_capacity(pairs.len());
    let mut coverage = 0.0f64;
    for (stem, left, right) in &pairs {
        let t = Instant::now();
        stereo.compute_into(left, right, &mut depth);
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
        if let Some(out) = &out {
            write_outputs(out, stem, left, &depth);
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
