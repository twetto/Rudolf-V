// examples/euroc_live.rs
//
// Live visualization of the Rudolf-V frontend on EuRoC/TUM-VI-style data.
// Shows a window with the camera image, tracked features, flow arrows,
// and fading track trails — like your C tracker's SDL window but in Rust.
//
// Usage:
//   cargo run --example euroc_live --release -- /path/to/MH_01_easy
//   cargo run --example euroc_live --release -- /path/to/MH_01_easy 500
//   cargo run --example tumvi_live --release
//   cargo run --example tumvi_live --release -- /path/to/tumvi/dataset 500
//
// Controls:
//   Space  — pause/resume
//   S      — step one frame (while paused)
//   Q/Esc  — quit
//   +/-    — speed up/slow down
//   T      — toggle track trails
//   F      — toggle flow arrows
//   D      — toggle FAST / Shi-Tomasi detector

use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::essential::RansacConfig;
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, TrackMeta};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;

use minifb::{Key, Window, WindowOptions};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// How many past positions to keep per track for trail rendering.
const TRAIL_LENGTH: usize = 20;

/// Track state for a single feature across frames.
struct Track {
    /// Ring buffer of past positions (most recent last).
    positions: Vec<(f32, f32)>,
    /// Color (packed RGB).
    color: u32,
    /// Frames since last seen (for fade-out).
    age: usize,
}

#[derive(Clone, Copy)]
enum MetaView {
    Survival,
    ReservoirScore,
    KltQuality,
    LbpDistance,
}

impl MetaView {
    fn next(self) -> Self {
        match self {
            MetaView::Survival => MetaView::ReservoirScore,
            MetaView::ReservoirScore => MetaView::KltQuality,
            MetaView::KltQuality => MetaView::LbpDistance,
            MetaView::LbpDistance => MetaView::Survival,
        }
    }

    fn label(self) -> &'static str {
        match self {
            MetaView::Survival => "survival",
            MetaView::ReservoirScore => "reservoir-score",
            MetaView::KltQuality => "klt-quality",
            MetaView::LbpDistance => "lbp-distance",
        }
    }
}

fn detector_label(detector: DetectorType) -> &'static str {
    match detector {
        DetectorType::Fast => "FAST",
        DetectorType::ShiTomasi => "Shi-Tomasi",
        DetectorType::Harris => "Harris",
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let invoked_as_tumvi = args
        .first()
        .and_then(|arg| Path::new(arg).file_stem())
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.contains("tumvi"));

    if args.len() < 2 && !invoked_as_tumvi {
        eprintln!("Usage: {} <dataset_path> [max_frames]", args[0]);
        eprintln!("  EuRoC:  cargo run --example euroc_live --release -- /path/to/MH_01_easy");
        eprintln!("  TUM-VI: cargo run --example tumvi_live --release");
        std::process::exit(1);
    }

    let dataset_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target")
            .join("tumvi")
            .join("dataset-calib-cam1_512_16")
    });
    let max_frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let cam0_dir = dataset_path.join("mav0").join("cam0");
    let data_dir = cam0_dir.join("data");
    let csv_path = cam0_dir.join("data.csv");

    if !data_dir.exists() {
        eprintln!("Error: {}", data_dir.display());
        eprintln!("Expected EuRoC ASL format: <dataset>/mav0/cam0/data/");
        std::process::exit(1);
    }

    let image_files = if csv_path.exists() {
        parse_euroc_csv(&csv_path)
    } else {
        list_png_files(&data_dir)
    };
    let num_frames = image_files.len().min(max_frames);
    println!(
        "Dataset: {} ({} frames)",
        dataset_path.display(),
        num_frames
    );

    // Load first image for dimensions.
    let first = load_grayscale(&data_dir.join(&image_files[0]));
    let (img_w, img_h) = (first.width(), first.height());
    drop(first);
    println!("Resolution: {}×{}", img_w, img_h);

    // Window at 2× scale for small images, 1× for large.
    let scale = if img_w <= 400 { 2 } else { 1 };
    let win_w = img_w * scale;
    let win_h = img_h * scale;

    let mut window = Window::new(
        "Rudolf-V — EuRoC Live",
        win_w,
        win_h,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");

    // Frame rate limiting — EuRoC cam0 is 20 Hz.
    window.set_target_fps(60);

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

    let min_reservoir_score = env::var("RUDOLF_MIN_RESERVOIR_SCORE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(f32::NEG_INFINITY);

    // Frontend.
    let config = FrontendConfig {
        //max_features: 40,
        //cell_size: 128,
        max_features: 200,
        cell_size: 32,
        // max_features: 100,
        // cell_size: 96,
        // max_features: 1410,
        // cell_size: 16,
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
        min_reservoir_score,
        // lbp_policy: LbpPolicy::HardReject,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, img_w, img_h);
    if min_reservoir_score.is_finite() {
        println!("Reservoir pruning: score >= {:.2}", min_reservoir_score);
    } else {
        println!("Reservoir pruning: off");
    }

    // Framebuffer (ARGB packed u32).
    let mut fb = vec![0u32; win_w * win_h];

    // Track history: feature ID → Track.
    let mut tracks: HashMap<u64, Track> = HashMap::new();
    // Previous frame feature positions for flow arrows.
    let mut prev_positions: HashMap<u64, (f32, f32)> = HashMap::new();

    // State.
    let mut frame_idx = 0;
    let mut paused = false;
    let mut show_trails = true;
    let mut show_flow = true;
    let mut meta_view = MetaView::Survival;
    let mut frame_delay_ms: u64 = 30; // ms between frames
    let mut last_frame_time = Instant::now();

    println!(
        "\nControls: Space=pause, S=step, Q/Esc=quit, +/-=speed, T=trails, F=flow, H=histeq, D=detector, M=metadata\n"
    );

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // Handle input.
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            paused = !paused;
            println!("{}", if paused { "Paused" } else { "Resumed" });
        }

        let step = window.is_key_pressed(Key::S, minifb::KeyRepeat::No);

        if window.is_key_pressed(Key::T, minifb::KeyRepeat::No) {
            show_trails = !show_trails;
            println!("Trails: {}", if show_trails { "on" } else { "off" });
        }
        if window.is_key_pressed(Key::F, minifb::KeyRepeat::No) {
            show_flow = !show_flow;
            println!("Flow: {}", if show_flow { "on" } else { "off" });
        }
        if window.is_key_pressed(Key::M, minifb::KeyRepeat::No) {
            meta_view = meta_view.next();
            println!("Metadata view: {}", meta_view.label());
        }
        if window.is_key_pressed(Key::H, minifb::KeyRepeat::No) {
            let next = match frontend.histeq() {
                HistEqMethod::None => HistEqMethod::Global,
                HistEqMethod::Global => HistEqMethod::Clahe {
                    // tile_size: 64,
                    tile_size: 256,
                    clip_limit: 4.0,
                },
                HistEqMethod::Clahe { .. } => HistEqMethod::None,
            };
            frontend.set_histeq(next);
            let label = match next {
                HistEqMethod::None => "off",
                HistEqMethod::Global => "global",
                HistEqMethod::Clahe { .. } => "CLAHE",
            };
            println!("HistEq: {}", label);
        }
        if window.is_key_pressed(Key::D, minifb::KeyRepeat::No) {
            let next = match frontend.detector() {
                DetectorType::Fast => DetectorType::ShiTomasi,
                // DetectorType::ShiTomasi => DetectorType::Fast,
                DetectorType::ShiTomasi => DetectorType::Harris,
                DetectorType::Harris => DetectorType::Fast,
            };
            frontend.set_detector(next);
            println!("Detector: {}", detector_label(next));
        }
        if window.is_key_pressed(Key::Equal, minifb::KeyRepeat::No)
            || window.is_key_pressed(Key::NumPadPlus, minifb::KeyRepeat::No)
        {
            frame_delay_ms = frame_delay_ms.saturating_sub(10);
            println!("Delay: {}ms", frame_delay_ms);
        }
        if window.is_key_pressed(Key::Minus, minifb::KeyRepeat::No)
            || window.is_key_pressed(Key::NumPadMinus, minifb::KeyRepeat::No)
        {
            frame_delay_ms = (frame_delay_ms + 10).min(500);
            println!("Delay: {}ms", frame_delay_ms);
        }

        // Advance frame?
        let should_advance = (!paused || step)
            && frame_idx < num_frames
            && last_frame_time.elapsed().as_millis() >= frame_delay_ms as u128;

        if should_advance {
            let img = load_grayscale(&data_dir.join(&image_files[frame_idx]));
            let (features, stats, meta) = {
                let (features, stats) = frontend.process(&img);
                (features.to_vec(), stats, frontend.track_meta().to_vec())
            };
            let meta_by_id: HashMap<u64, TrackMeta> =
                meta.iter().map(|m| (m.id, m.clone())).collect();
            let meta_summary = summarize_meta(&meta);

            // Render the same preprocessed image used by the frontend.
            let display_img = frontend.preprocessed_image().unwrap_or(&img);
            render_grayscale(display_img, &mut fb, img_w, img_h, scale);

            // Update tracks.
            let curr_positions: HashMap<u64, (f32, f32)> =
                features.iter().map(|f| (f.id, (f.x, f.y))).collect();

            // Age all tracks, remove very old ones.
            tracks.values_mut().for_each(|t| t.age += 1);
            tracks.retain(|_, t| t.age < TRAIL_LENGTH);

            // Update/create tracks for current features.
            for f in &features {
                let track = tracks.entry(f.id).or_insert_with(|| Track {
                    positions: Vec::with_capacity(TRAIL_LENGTH),
                    color: id_to_color(f.id),
                    age: 0,
                });
                track.age = 0;
                track.positions.push((f.x, f.y));
                if track.positions.len() > TRAIL_LENGTH {
                    track.positions.remove(0);
                }
            }

            // Draw trails.
            if show_trails {
                for track in tracks.values() {
                    if track.positions.len() < 2 {
                        continue;
                    }
                    for i in 1..track.positions.len() {
                        let alpha = i as f32 / track.positions.len() as f32;
                        let color = fade_color(track.color, alpha * 0.6);
                        draw_line(
                            &mut fb,
                            win_w,
                            win_h,
                            (track.positions[i - 1].0 * scale as f32) as i32,
                            (track.positions[i - 1].1 * scale as f32) as i32,
                            (track.positions[i].0 * scale as f32) as i32,
                            (track.positions[i].1 * scale as f32) as i32,
                            color,
                        );
                    }
                }
            }

            // Draw flow arrows (from prev frame position to current).
            if show_flow {
                for f in &features {
                    if let Some(&(px, py)) = prev_positions.get(&f.id) {
                        let dx = f.x - px;
                        let dy = f.y - py;
                        let mag = (dx * dx + dy * dy).sqrt();
                        if mag > 0.5 {
                            draw_arrow(
                                &mut fb,
                                win_w,
                                win_h,
                                (px * scale as f32) as i32,
                                (py * scale as f32) as i32,
                                (f.x * scale as f32) as i32,
                                (f.y * scale as f32) as i32,
                                0x00FF88,
                            );
                        }
                    }
                }
            }

            // Draw feature points on top.
            for f in &features {
                let track_len = tracks.get(&f.id).map(|t| t.positions.len()).unwrap_or(0);
                let meta = meta_by_id.get(&f.id);
                let color = feature_color(meta_view, track_len, meta);

                draw_circle(
                    &mut fb,
                    win_w,
                    win_h,
                    (f.x * scale as f32) as i32,
                    (f.y * scale as f32) as i32,
                    2 * scale as i32,
                    color,
                );
            }

            // HUD text (top-left info bar).
            draw_rect(&mut fb, win_w, win_h, 0, 0, win_w, 14, 0x222222);
            // Simple: just print to stdout since bitmap text is painful.
            print!(
                "\r{:5}: det={} trk={:<3} lost={:<3} rej={:<3} new={:<3} tot={:<3} | score={:.2} klt={:.2} lbp={:.1}/{:<2} view={} | {}  ",
                frame_idx,
                detector_label(frontend.detector()),
                stats.tracked,
                stats.lost,
                stats.rejected,
                stats.new_detections,
                stats.total,
                meta_summary.avg_score,
                meta_summary.avg_klt_quality,
                meta_summary.avg_lbp_distance,
                meta_summary.max_lbp_distance,
                meta_view.label(),
                stats.timing
            );

            prev_positions = curr_positions;
            frame_idx += 1;
            last_frame_time = Instant::now();
        }

        window.update_with_buffer(&fb, win_w, win_h).unwrap();
    }

    println!("\n\nProcessed {} frames.", frame_idx);
}

// ---------------------------------------------------------------------------
// EuRoC I/O
// ---------------------------------------------------------------------------

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
        if let Some((_ts, fname)) = line.split_once(',') {
            filenames.push(fname.trim().to_string());
        }
    }
    filenames
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

fn load_grayscale(path: &Path) -> Image<u8> {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}

// ---------------------------------------------------------------------------
// Framebuffer rendering
// ---------------------------------------------------------------------------

/// Blit grayscale image to u32 framebuffer (with optional integer scale).
fn render_grayscale(img: &Image<u8>, fb: &mut [u32], img_w: usize, img_h: usize, scale: usize) {
    for y in 0..img_h {
        for x in 0..img_w {
            let v = img.get(x, y) as u32;
            let pixel = (v << 16) | (v << 8) | v; // 0x00RRGGBB

            // Fill the scaled block.
            for sy in 0..scale {
                for sx in 0..scale {
                    let fx = x * scale + sx;
                    let fy = y * scale + sy;
                    fb[fy * (img_w * scale) + fx] = pixel;
                }
            }
        }
    }
}

/// Draw a filled circle.
fn draw_circle(fb: &mut [u32], w: usize, h: usize, cx: i32, cy: i32, r: i32, color: u32) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                    fb[py as usize * w + px as usize] = color;
                }
            }
        }
    }
}

/// Draw a line using Bresenham's algorithm.
fn draw_line(fb: &mut [u32], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            fb[y as usize * w + x as usize] = color;
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a line with an arrowhead.
fn draw_arrow(fb: &mut [u32], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    draw_line(fb, w, h, x0, y0, x1, y1, color);

    // Arrowhead.
    let dx = (x1 - x0) as f32;
    let dy = (y1 - y0) as f32;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 2.0 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    let head_len = 5.0f32.min(len * 0.4);

    // Two sides of the arrowhead.
    let ax = x1 as f32 - head_len * (ux + 0.4 * uy);
    let ay = y1 as f32 - head_len * (uy - 0.4 * ux);
    let bx = x1 as f32 - head_len * (ux - 0.4 * uy);
    let by = y1 as f32 - head_len * (uy + 0.4 * ux);

    draw_line(fb, w, h, x1, y1, ax as i32, ay as i32, color);
    draw_line(fb, w, h, x1, y1, bx as i32, by as i32, color);
}

/// Draw a filled rectangle.
fn draw_rect(
    fb: &mut [u32],
    w: usize,
    h: usize,
    rx: usize,
    ry: usize,
    rw: usize,
    rh: usize,
    color: u32,
) {
    for y in ry..(ry + rh).min(h) {
        for x in rx..(rx + rw).min(w) {
            fb[y * w + x] = color;
        }
    }
}

/// Generate a distinct color from a feature ID.
fn id_to_color(id: u64) -> u32 {
    let hue = ((id * 137 + 43) % 360) as f32;
    let (r, g, b) = hsv_to_rgb(hue, 0.85, 1.0);
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Fade a color by multiplying RGB by alpha (0.0–1.0).
fn fade_color(color: u32, alpha: f32) -> u32 {
    let r = ((color >> 16) & 0xFF) as f32 * alpha;
    let g = ((color >> 8) & 0xFF) as f32 * alpha;
    let b = (color & 0xFF) as f32 * alpha;
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[derive(Default)]
struct MetaSummary {
    avg_score: f32,
    avg_klt_quality: f32,
    avg_lbp_distance: f32,
    max_lbp_distance: u16,
}

fn summarize_meta(meta: &[TrackMeta]) -> MetaSummary {
    if meta.is_empty() {
        return MetaSummary::default();
    }

    let mut total_score = 0.0;
    let mut total_klt_quality = 0.0;
    let mut total_lbp_distance = 0u32;
    let mut max_lbp_distance = 0u16;

    for m in meta {
        total_score += m.reservoir_score;
        total_klt_quality += m.klt_quality;
        total_lbp_distance += m.lbp_distance as u32;
        max_lbp_distance = max_lbp_distance.max(m.lbp_distance);
    }

    let count = meta.len() as f32;
    MetaSummary {
        avg_score: total_score / count,
        avg_klt_quality: total_klt_quality / count,
        avg_lbp_distance: total_lbp_distance as f32 / count,
        max_lbp_distance,
    }
}

fn feature_color(view: MetaView, track_len: usize, meta: Option<&TrackMeta>) -> u32 {
    match view {
        MetaView::Survival => survival_color(track_len, TRAIL_LENGTH),
        MetaView::ReservoirScore => {
            let quality = meta
                .map(|m| ((m.reservoir_score + 1.0) / 4.0).clamp(0.0, 1.0))
                .unwrap_or(0.0);
            quality_color(quality)
        }
        MetaView::KltQuality => quality_color(meta.map(|m| m.klt_quality).unwrap_or(0.0)),
        MetaView::LbpDistance => {
            let quality = meta
                .map(|m| 1.0 - (m.lbp_distance as f32 / 16.0).min(1.0))
                .unwrap_or(0.0);
            quality_color(quality)
        }
    }
}

fn quality_color(quality: f32) -> u32 {
    let hue = quality.clamp(0.0, 1.0) * 120.0;
    let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

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

/// Compute a color based on track length (Red for short, Green for long).
fn survival_color(len: usize, max_len: usize) -> u32 {
    let t = (len as f32 / max_len as f32).min(1.0);
    // Red (0, 1.0, 1.0) -> Green (120, 1.0, 1.0) in HSV.
    let hue = t * 120.0;
    let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}
