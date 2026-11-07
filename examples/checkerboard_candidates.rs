// examples/checkerboard_candidates.rs — Dump the detector's corner candidates for PNGs (diagnostics).
//
//   cargo run --release --example checkerboard_candidates -- <png> [<png> ...] > candidates.jsonl
//
// One JSON line per image: file stem and its candidates [x, y, response], strongest first.

use std::path::Path;

use rudolf_v::checkerboard::{CheckerboardConfig, chess_response, find_candidates};
use rudolf_v::image::Image;

fn main() {
    let config = CheckerboardConfig::default();
    for arg in std::env::args().skip(1) {
        let path = Path::new(&arg);
        let gray = image::open(path).expect("png").to_luma8();
        let (w, h) = gray.dimensions();
        let img = Image::from_vec(w as usize, h as usize, gray.into_raw());
        let c = find_candidates(&chess_response(&img), &config);
        let list: Vec<String> = c
            .iter()
            .map(|q| format!("[{:.2},{:.2},{:.0}]", q.x, q.y, q.response))
            .collect();
        println!(
            "{{\"stamp\":\"{}\",\"candidates\":[{}]}}",
            path.file_stem().unwrap().to_string_lossy(),
            list.join(",")
        );
    }
}
