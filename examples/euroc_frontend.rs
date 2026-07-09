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

use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::essential::RansacConfig;
use rudolf_v::frontend::{Frontend, FrontendConfig};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;

use image::{Rgb, RgbImage};
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
    let max_frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

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

    // Load camera intrinsics for geometric verification (optional).
    let sensor_yaml = cam0_dir.join("sensor.yaml");
    let camera = match load_camera(&dataset_path, &sensor_yaml, "cam0") {
        Ok(cam) => {
            println!(
                "Camera: fx={:.1} fy={:.1} cx={:.1} cy={:.1}, {:?}, {} distortion coeffs",
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
                cam.model,
                cam.distortion.len()
            );
            Some(cam)
        }
        Err(e) => {
            eprintln!(
                "Warning: no sensor.yaml ({}), geometric verification disabled",
                e
            );
            None
        }
    };

    // Configure frontend.
    let config = FrontendConfig {
        // Keep this batch/video example aligned with euroc_live.
        max_features: 200,
        cell_size: 32,
        pyramid_levels: 4,
        klt_window: 11,
        klt_max_iter: 30,
        klt_method: LkMethod::InverseCompositional,
        histeq: HistEqMethod::Global,
        camera,
        ransac: RansacConfig {
            threshold: 1e-5,
            max_iterations: 200,
            confidence: 0.99,
        },
        ..Default::default()
    };
    println!(
        "Config: max_features={}, cell={}px, pyramid={}L, klt_window={}, histeq=global",
        config.max_features, config.cell_size, config.pyramid_levels, config.klt_window
    );

    let mut frontend = Frontend::new(config, img_w, img_h);

    // Track history for visualization: (id, Vec<(x, y)>).
    let mut track_history: Vec<(u64, Vec<(f32, f32)>)> = Vec::new();

    // Stats CSV.
    let mut stats_csv = String::from("frame,tracked,lost,rejected,new,total,occupied_cells\n");
    let expected_flow = match (
        env::var("RUDOLF_EXPECT_FLOW_DX"),
        env::var("RUDOLF_EXPECT_FLOW_DY"),
    ) {
        (Ok(dx), Ok(dy)) => Some((
            dx.parse::<f32>()
                .expect("RUDOLF_EXPECT_FLOW_DX must be a float"),
            dy.parse::<f32>()
                .expect("RUDOLF_EXPECT_FLOW_DY must be a float"),
        )),
        _ => None,
    };
    let mut flow_csv =
        expected_flow.map(|_| String::from("frame,id,prev_x,prev_y,x,y,dx,dy,gt_dx,gt_dy,error\n"));

    fs::create_dir_all("vis_output").ok();
    let export_frames = env::var_os("RUDOLF_EXPORT_FRAMES").is_some();
    if export_frames {
        fs::create_dir_all("vis_output/frames").expect("failed to create frame export directory");
        println!("Frame export enabled: vis_output/frames/frame_*.png");
    }

    println!(
        "\n{:>5}  {:>7}  {:>4}  {:>3}  {:>3}  {:>5}  {:>5}",
        "frame", "tracked", "lost", "rej", "new", "total", "cells"
    );
    println!("{}", "-".repeat(48));

    let mut last_img: Option<Image<u8>> = None;

    for i in 0..num_frames {
        let img = load_grayscale(&data_dir.join(&image_files[i]));

        let (features, stats) = frontend.process(&img);

        println!(
            "{:5}  {:7}  {:4}  {:3}  {:3}  {:5}  {:5}/{}",
            i,
            stats.tracked,
            stats.lost,
            stats.rejected,
            stats.new_detections,
            stats.total,
            stats.occupied_cells,
            stats.total_cells
        );

        writeln!(
            stats_csv,
            "{},{},{},{},{},{},{}",
            i,
            stats.tracked,
            stats.lost,
            stats.rejected,
            stats.new_detections,
            stats.total,
            stats.occupied_cells
        )
        .unwrap();

        // Update track history.
        // Mark all existing tracks as "not seen this frame".
        let mut seen_ids: Vec<u64> = Vec::new();
        for f in features {
            seen_ids.push(f.id);
            if let Some(track) = track_history.iter_mut().find(|(id, _)| *id == f.id) {
                if let (Some(flow_csv), Some((gt_dx, gt_dy)), Some((prev_x, prev_y))) =
                    (&mut flow_csv, expected_flow, track.1.last().copied())
                {
                    let dx = f.x - prev_x;
                    let dy = f.y - prev_y;
                    let err = ((dx - gt_dx).powi(2) + (dy - gt_dy).powi(2)).sqrt();
                    writeln!(
                        flow_csv,
                        "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                        i, f.id, prev_x, prev_y, f.x, f.y, dx, dy, gt_dx, gt_dy, err
                    )
                    .unwrap();
                }
                track.1.push((f.x, f.y));
            } else {
                track_history.push((f.id, vec![(f.x, f.y)]));
            }
        }

        if export_frames {
            let frame = render_frame_overlay(
                &img,
                &track_history,
                &seen_ids,
                i,
                stats.tracked,
                stats.total,
            );
            frame
                .save(format!("vis_output/frames/frame_{i:06}.png"))
                .expect("failed to save frame overlay");
        }

        last_img = Some(img);
    }

    // Write stats CSV.
    fs::write("vis_output/euroc_stats.csv", &stats_csv).unwrap();
    println!("\nStats saved to vis_output/euroc_stats.csv");
    if let Some(flow_csv) = flow_csv {
        fs::write("vis_output/euroc_flow.csv", &flow_csv).unwrap();
        println!("Flow validation data saved to vis_output/euroc_flow.csv");
    }

    // Generate track visualization on the last frame.
    let final_img = last_img.expect("no frames processed");
    let svg = render_tracks(&final_img, &track_history, num_frames);
    fs::write("vis_output/euroc_tracks.svg", &svg).unwrap();
    println!("Track visualization saved to vis_output/euroc_tracks.svg");

    // Summary.
    let long_tracks = track_history
        .iter()
        .filter(|(_, pts)| pts.len() >= 10)
        .count();
    let max_track = track_history
        .iter()
        .map(|(_, pts)| pts.len())
        .max()
        .unwrap_or(0);
    println!("\nTrack summary:");
    println!("  Total unique features: {}", track_history.len());
    println!("  Tracks >= 10 frames: {}", long_tracks);
    println!("  Longest track: {} frames", max_track);
}

fn load_camera(
    dataset_path: &Path,
    sensor_yaml: &Path,
    camera: &str,
) -> Result<CameraIntrinsics, String> {
    if sensor_yaml.exists() {
        return CameraIntrinsics::from_euroc_yaml(sensor_yaml);
    }
    let camchain = dataset_path.join("dso").join("camchain.yaml");
    if camchain.exists() {
        return CameraIntrinsics::from_kalibr_camchain(&camchain, camera);
    }
    CameraIntrinsics::from_euroc_yaml(sensor_yaml)
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
            if name.ends_with(".png") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

/// Load a PNG as a grayscale Image<u8>.
fn load_grayscale(path: &Path) -> Image<u8> {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
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
    writeln!(
        svg,
        "<style>text {{ font-family: monospace; font-size: 12px; fill: #ddd; }}</style>"
    )
    .unwrap();

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
                writeln!(
                    svg,
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"1\" fill=\"rgb({},{},{})\"/>",
                    x, y, run, v, v, v
                )
                .unwrap();
            }
            x += run;
        }
    }
    writeln!(svg, "</g>").unwrap();

    // Render tracks.
    let long_tracks: Vec<&(u64, Vec<(f32, f32)>)> =
        tracks.iter().filter(|(_, pts)| pts.len() >= 3).collect();

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
        writeln!(
            svg,
            "\" fill=\"none\" stroke=\"rgb({},{},{})\" stroke-width=\"1\" opacity=\"{}\"/>",
            r, g, b, opacity
        )
        .unwrap();

        // Dot at current (last) position.
        let (lx, ly) = pts.last().unwrap();
        writeln!(
            svg,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2\" fill=\"rgb({},{},{})\" opacity=\"{}\"/>",
            lx, ly, r, g, b, opacity
        )
        .unwrap();
    }

    // Legend.
    let ly = h + 10;
    let long_count = long_tracks.len();
    let total_count = tracks.len();
    writeln!(
        svg,
        "<rect x=\"0\" y=\"{}\" width=\"{}\" height=\"40\" fill=\"#222\"/>",
        h, w
    )
    .unwrap();
    writeln!(svg, "<text x=\"10\" y=\"{}\">Tracks: {} total, {} shown (>=3 frames) | {} frames processed</text>",
        ly + 14, total_count, long_count, total_frames).unwrap();

    writeln!(svg, "</svg>").unwrap();
    svg
}

fn render_frame_overlay(
    img: &Image<u8>,
    track_history: &[(u64, Vec<(f32, f32)>)],
    active_ids: &[u64],
    frame_idx: usize,
    tracked: usize,
    total: usize,
) -> RgbImage {
    let w = img.width() as u32;
    let h = img.height() as u32;
    let footer_h = 28u32;
    let mut out = RgbImage::new(w, h + footer_h);

    for y in 0..h {
        for x in 0..w {
            let v = img[(x as usize, y as usize)];
            out.put_pixel(x, y, Rgb([v, v, v]));
        }
    }

    for (id, points) in track_history {
        if !active_ids.contains(id) {
            continue;
        }
        if points.len() < 2 {
            continue;
        }
        let recent = points.iter().rev().take(8).copied().collect::<Vec<_>>();
        for pair in recent.windows(2) {
            let (x0, y0) = pair[1];
            let (x1, y1) = pair[0];
            draw_rgb_line(&mut out, x0, y0, x1, y1, Rgb([80, 220, 255]));
        }
        if let Some(&(x, y)) = points.last() {
            draw_rgb_circle(&mut out, x, y, 3, Rgb([40, 255, 80]));
        }
    }

    for y in h..h + footer_h {
        for x in 0..w {
            out.put_pixel(x, y, Rgb([18, 18, 18]));
        }
    }
    draw_tiny_text(
        &mut out,
        8,
        h + 8,
        &format!("frame {frame_idx:04} tracked {tracked:03} total {total:03}"),
        Rgb([230, 230, 230]),
    );

    out
}

fn draw_rgb_circle(img: &mut RgbImage, cx: f32, cy: f32, radius: i32, color: Rgb<u8>) {
    let cx = cx.round() as i32;
    let cy = cy.round() as i32;
    let r2 = radius * radius;
    for y in cy - radius..=cy + radius {
        for x in cx - radius..=cx + radius {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                put_rgb_checked(img, x, y, color);
            }
        }
    }
}

fn draw_rgb_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgb<u8>) {
    let mut x0 = x0.round() as i32;
    let mut y0 = y0.round() as i32;
    let x1 = x1.round() as i32;
    let y1 = y1.round() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        put_rgb_checked(img, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put_rgb_checked(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, color);
    }
}

fn draw_tiny_text(img: &mut RgbImage, x: u32, y: u32, text: &str, color: Rgb<u8>) {
    let mut cursor = x;
    for ch in text.chars() {
        draw_tiny_char(img, cursor, y, ch, color);
        cursor += 6;
    }
}

fn draw_tiny_char(img: &mut RgbImage, x: u32, y: u32, ch: char, color: Rgb<u8>) {
    let glyph = tiny_glyph(ch);
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..5 {
            if bits & (1 << (4 - col)) != 0 {
                put_rgb_checked(img, x as i32 + col, y as i32 + row as i32, color);
            }
        }
    }
}

fn tiny_glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        'a' => [
            0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111,
        ],
        'c' => [
            0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'd' => [
            0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10011, 0b01101,
        ],
        'e' => [
            0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110,
        ],
        'f' => [
            0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000,
        ],
        'k' => [
            0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010,
        ],
        'm' => [
            0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101,
        ],
        'o' => [
            0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'r' => [
            0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000,
        ],
        't' => [
            0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110,
        ],
        'l' => [
            0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        ' ' => [0; 7],
        _ => [
            0b11111, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
    }
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
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
