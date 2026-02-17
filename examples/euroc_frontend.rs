// examples/euroc_frontend.rs
//
// Run the Rudolf-V frontend on a EuRoC MAV dataset (ASL format).
//
// Usage:
//   cargo run --example euroc_frontend --release -- /path/to/MH_01_easy
//
// The EuRoC ASL format:
//   <dataset>/mav0/cam0/data/          — grayscale PNG images (752×480)
//   <dataset>/mav0/cam0/data.csv       — timestamp,filename
//
// Download from: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
//
// Output:
//   vis_output/euroc_tracks.svg        — feature tracks overlaid on last frame
//   vis_output/euroc_stats.csv         — per-frame statistics
//   stdout                             — per-frame summary

use rudolf_v::frontend::{Frontend, FrontendConfig};
use rudolf_v::image::Image;

use std::env;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <euroc_dataset_path> [max_frames]", args[0]);
        eprintln!("  e.g.: {} /data/EuRoC/MH_01_easy", args[0]);
        eprintln!("  Optional: max_frames limits processing (default: all)");
        std::process::exit(1);
    }

    let dataset_path = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // Locate the image directory and CSV.
    let cam0_dir = dataset_path.join("mav0").join("cam0");
    let data_dir = cam0_dir.join("data");
    let csv_path = cam0_dir.join("data.csv");

    if !data_dir.exists() {
        eprintln!("Error: image directory not found: {}", data_dir.display());
        eprintln!("Expected EuRoC ASL format: <dataset>/mav0/cam0/data/");
        std::process::exit(1);
    }

    // Parse the CSV to get ordered image filenames.
    let image_files = if csv_path.exists() {
        parse_euroc_csv(&csv_path)
    } else {
        // Fallback: just read PNG filenames sorted.
        eprintln!("Warning: data.csv not found, using directory listing.");
        list_png_files(&data_dir)
    };

    if image_files.is_empty() {
        eprintln!("Error: no images found in {}", data_dir.display());
        std::process::exit(1);
    }

    let num_frames = image_files.len().min(max_frames);
    println!("Dataset: {}", dataset_path.display());
    println!("Images: {} (processing {})", image_files.len(), num_frames);

    // Load first image to get dimensions.
    let first_img = load_grayscale(&data_dir.join(&image_files[0]));
    let (img_w, img_h) = (first_img.width(), first_img.height());
    drop(first_img); // Free memory; we'll reload in the loop.
    println!("Resolution: {}×{}", img_w, img_h);

    // Configure frontend.
    let config = FrontendConfig {
        max_features: 150,
        cell_size: 32,
        pyramid_levels: 4,
        klt_window: 11,
        klt_max_iter: 30,
        ..Default::default()
    };
    println!("Config: max_features={}, cell={}px, pyramid={}L, klt_window={}",
        config.max_features, config.cell_size, config.pyramid_levels, config.klt_window);

    let mut frontend = Frontend::new(config, img_w, img_h);

    // Track history for visualization: (id, Vec<(x, y)>).
    let mut track_history: Vec<(u64, Vec<(f32, f32)>)> = Vec::new();

    // Stats CSV.
    let mut stats_csv = String::from("frame,tracked,lost,new,total,occupied_cells\n");

    fs::create_dir_all("vis_output").ok();

    println!("\n{:>5}  {:>7}  {:>4}  {:>3}  {:>5}  {:>5}",
        "frame", "tracked", "lost", "new", "total", "cells");
    println!("{}", "-".repeat(42));

    let mut last_img: Option<Image<u8>> = None;

    for i in 0..num_frames {
        let img = load_grayscale(&data_dir.join(&image_files[i]));

        let (features, stats) = frontend.process(&img);

        println!("{:5}  {:7}  {:4}  {:3}  {:5}  {:5}/{}",
            i, stats.tracked, stats.lost, stats.new_detections,
            stats.total, stats.occupied_cells, stats.total_cells);

        writeln!(stats_csv, "{},{},{},{},{},{}",
            i, stats.tracked, stats.lost, stats.new_detections,
            stats.total, stats.occupied_cells).unwrap();

        // Update track history.
        // Mark all existing tracks as "not seen this frame".
        let mut seen_ids: Vec<u64> = Vec::new();
        for f in features {
            seen_ids.push(f.id);
            if let Some(track) = track_history.iter_mut().find(|(id, _)| *id == f.id) {
                track.1.push((f.x, f.y));
            } else {
                track_history.push((f.id, vec![(f.x, f.y)]));
            }
        }

        last_img = Some(img);
    }

    // Write stats CSV.
    fs::write("vis_output/euroc_stats.csv", &stats_csv).unwrap();
    println!("\nStats saved to vis_output/euroc_stats.csv");

    // Generate track visualization on the last frame.
    let final_img = last_img.expect("no frames processed");
    let svg = render_tracks(&final_img, &track_history, num_frames);
    fs::write("vis_output/euroc_tracks.svg", &svg).unwrap();
    println!("Track visualization saved to vis_output/euroc_tracks.svg");

    // Summary.
    let long_tracks = track_history.iter().filter(|(_, pts)| pts.len() >= 10).count();
    let max_track = track_history.iter().map(|(_, pts)| pts.len()).max().unwrap_or(0);
    println!("\nTrack summary:");
    println!("  Total unique features: {}", track_history.len());
    println!("  Tracks >= 10 frames: {}", long_tracks);
    println!("  Longest track: {} frames", max_track);
}

/// Parse EuRoC data.csv: "#timestamp [ns],filename" → Vec<filename>.
fn parse_euroc_csv(csv_path: &Path) -> Vec<String> {
    let file = fs::File::open(csv_path).expect("failed to open data.csv");
    let reader = BufReader::new(file);
    let mut filenames = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        // Format: "1403636579763555584,1403636579763555584.png"
        // or just: "timestamp,filename"
        if let Some((_ts, fname)) = line.split_once(',') {
            filenames.push(fname.trim().to_string());
        }
    }
    filenames
}

/// Fallback: list .png files in directory, sorted.
fn list_png_files(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = fs::read_dir(dir)
        .expect("failed to read directory")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".png") { Some(name) } else { None }
        })
        .collect();
    files.sort();
    files
}

/// Load a PNG as a grayscale Image<u8>.
fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}

// ---------------------------------------------------------------------------
// SVG track visualization
// ---------------------------------------------------------------------------

/// Render feature tracks overlaid on the last frame.
///
/// Short tracks (< 3 frames) are faded. Long tracks are bright.
/// Track color encodes feature ID for visual distinction.
fn render_tracks(
    img: &Image<u8>,
    tracks: &[(u64, Vec<(f32, f32)>)],
    total_frames: usize,
) -> String {
    // Scale factor: 1px = 1 SVG unit (no zoom — EuRoC is 752×480, plenty large).
    let w = img.width();
    let h = img.height();
    let total_h = h + 50;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        w, total_h, w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #ddd; }}</style>").unwrap();

    // Background: last frame as grayscale rectangles (1px = 1 rect).
    // For large images, use a single <image> with inline base64 would be better,
    // but for simplicity we'll render rows as horizontal lines.
    writeln!(svg, "<g opacity=\"0.6\">").unwrap();
    for y in 0..h {
        // Run-length encode for efficiency.
        let mut x = 0;
        while x < w {
            let v = img.get(x, y);
            let mut run = 1;
            while x + run < w && img.get(x + run, y) == v {
                run += 1;
            }
            if v != 0 {
                writeln!(svg, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"1\" fill=\"rgb({},{},{})\"/>",
                    x, y, run, v, v, v).unwrap();
            }
            x += run;
        }
    }
    writeln!(svg, "</g>").unwrap();

    // Render tracks.
    let long_tracks: Vec<&(u64, Vec<(f32, f32)>)> = tracks.iter()
        .filter(|(_, pts)| pts.len() >= 3)
        .collect();

    for (id, pts) in &long_tracks {
        if pts.len() < 2 {
            continue;
        }

        // Color from ID hash — spread across hue space.
        let hue = ((*id * 137) % 360) as f32;
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 1.0);

        // Opacity based on track length.
        let opacity = if pts.len() >= 10 { 0.9 } else { 0.4 };

        // Polyline for the track.
        write!(svg, "<polyline points=\"").unwrap();
        for (x, y) in pts {
            write!(svg, "{:.1},{:.1} ", x, y).unwrap();
        }
        writeln!(svg, "\" fill=\"none\" stroke=\"rgb({},{},{})\" stroke-width=\"1\" opacity=\"{}\"/>",
            r, g, b, opacity).unwrap();

        // Dot at current (last) position.
        let (lx, ly) = pts.last().unwrap();
        writeln!(svg, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2\" fill=\"rgb({},{},{})\" opacity=\"{}\"/>",
            lx, ly, r, g, b, opacity).unwrap();
    }

    // Legend.
    let ly = h + 10;
    let long_count = long_tracks.len();
    let total_count = tracks.len();
    writeln!(svg, "<rect x=\"0\" y=\"{}\" width=\"{}\" height=\"40\" fill=\"#222\"/>", h, w).unwrap();
    writeln!(svg, "<text x=\"10\" y=\"{}\">Tracks: {} total, {} shown (>=3 frames) | {} frames processed</text>",
        ly + 14, total_count, long_count, total_frames).unwrap();

    writeln!(svg, "</svg>").unwrap();
    svg
}

/// Simple HSV to RGB conversion for track coloring.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h as u32) / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}
