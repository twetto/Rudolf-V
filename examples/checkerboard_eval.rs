// examples/checkerboard_eval.rs — Run the checkerboard detector on a folder of PNGs.
//
//   cargo run --release --example checkerboard_eval -- <png dir> [cols rows] [--gpu] > results.jsonl
//
// One JSON line per image (file stem, found, corners, milliseconds), a summary on stderr.
// --gpu: the ChESS responses on the GPU (GpuCheckerboardDetector), the same corners.
// For comparing against another detector's results on the same frames. All images are
// decoded before the first detection, so a profiler sees the detector, not PNG decoding.

use std::path::Path;
use std::time::Instant;

use rudolf_v::checkerboard::{CheckerboardConfig, CheckerboardDetector};
use rudolf_v::gpu::checkerboard::GpuCheckerboardDetector;
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::image::Image;

fn load_grayscale(path: &Path) -> Image<u8> {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let gpu = args.iter().position(|a| a == "--gpu").map(|i| {
        args.remove(i);
        GpuDevice::new().expect("--gpu: no Vulkan GPU")
    });
    let dir = args
        .get(1)
        .expect("usage: checkerboard_eval <png dir> [cols rows] [--gpu]");
    let cols = args.get(2).map_or(9, |s| s.parse().expect("cols"));
    let rows = args.get(3).map_or(6, |s| s.parse().expect("rows"));
    let config = CheckerboardConfig {
        cols,
        rows,
        ..Default::default()
    };
    let mut gpu_detector = gpu
        .as_ref()
        .map(|g| GpuCheckerboardDetector::new(g, config.clone()));
    let detector = CheckerboardDetector::new(config);

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("png dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();

    let images: Vec<Image<u8>> = files.iter().map(|p| load_grayscale(p)).collect();

    let (mut found, mut times) = (0, Vec::new());
    for (path, image) in files.iter().zip(&images) {
        let t0 = Instant::now();
        let corners = match (&gpu, &mut gpu_detector) {
            (Some(g), Some(d)) => d.detect(g, image),
            _ => detector.detect(image),
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        let stem = path.file_stem().unwrap().to_string_lossy();
        match corners {
            Some(c) => {
                found += 1;
                let list: Vec<String> = c.iter().map(|[x, y]| format!("[{x:.3},{y:.3}]")).collect();
                println!(
                    "{{\"stamp\":\"{stem}\",\"found\":true,\"ms\":{ms:.2},\"corners\":[{}]}}",
                    list.join(",")
                );
            }
            None => println!("{{\"stamp\":\"{stem}\",\"found\":false,\"ms\":{ms:.2}}}"),
        }
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    eprintln!(
        "{} images, board found in {} ({:.0} %); ms per image: median {:.1}, p90 {:.1}, max {:.1}; total {:.0} ms",
        files.len(),
        found,
        100.0 * found as f64 / files.len().max(1) as f64,
        pct(0.5),
        pct(0.9),
        pct(1.0),
        times.iter().sum::<f64>()
    );
}
