// examples/checkerboard_eval.rs — Run the checkerboard detector on a folder of PNGs.
//
//   cargo run --release --example checkerboard_eval -- <png dir> [cols rows] > results.jsonl
//
// One JSON line per image (file stem, found, corners, milliseconds), a summary on stderr.
// For comparing against another detector's results on the same frames.

use std::path::Path;
use std::time::Instant;

use rudolf_v::checkerboard::{CheckerboardConfig, CheckerboardDetector};
use rudolf_v::image::Image;

fn load_grayscale(path: &Path) -> Image<u8> {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: checkerboard_eval <png dir> [cols rows]");
    let cols = args.get(2).map_or(9, |s| s.parse().expect("cols"));
    let rows = args.get(3).map_or(6, |s| s.parse().expect("rows"));
    let detector = CheckerboardDetector::new(CheckerboardConfig {
        cols,
        rows,
        ..Default::default()
    });

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("png dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();

    let (mut found, mut times) = (0, Vec::new());
    for path in &files {
        let image = load_grayscale(path);
        let t0 = Instant::now();
        let corners = detector.detect(&image);
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
        "{} images, board found in {} ({:.0} %); ms per image: median {:.1}, p90 {:.1}, max {:.1}",
        files.len(),
        found,
        100.0 * found as f64 / files.len().max(1) as f64,
        pct(0.5),
        pct(0.9),
        pct(1.0)
    );
}
