// examples/gpu_fast.rs — CPU vs GPU FAST corner detector visualiser.
//
// For each test scene, produces an SVG with three panels:
//
//   ┌─────────────────┬─────────────────┬─────────────────┐
//   │  CPU raw        │  CPU + NMS      │  GPU + NMS      │
//   │  (red circles)  │  (green + grid) │  (cyan + grid)  │
//   └─────────────────┴─────────────────┴─────────────────┘
//
// GPU NMS is now performed on-GPU in the same encoder as FAST detection,
// so detect() returns already-suppressed winners directly.
//
// USAGE
//   cargo run --example gpu_fast
//   cargo run --example gpu_fast -- path/to/image.png

use std::fmt::Write;
use std::fs;
use std::time::Instant;

use rudolf_v::fast::{FastDetector, Feature};
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::fast::GpuFastDetector;
use rudolf_v::gpu::pyramid::GpuPyramidPipeline;
use rudolf_v::image::Image;
use rudolf_v::nms::OccupancyNms;

const THRESHOLD:  u8    = 20;
const ARC_LENGTH: usize = 9;
const NMS_CELL:   usize = 16;
const SCALE:      usize = 4;

/// Max image dimension before skipping per-pixel SVG rendering.
const MAX_RENDER_DIM: usize = 256;

fn main() {
    fs::create_dir_all("vis_output").expect("failed to create vis_output/");

    println!("Initialising GPU...");
    let gpu = GpuDevice::new().expect("failed to initialise a Vulkan GPU");
    println!("GPU: {}", gpu.adapter_info);

    let gpu_pipeline = GpuPyramidPipeline::new(&gpu);
    let cpu_det = FastDetector::new(THRESHOLD, ARC_LENGTH);
    let cpu_nms = OccupancyNms::new(NMS_CELL);

    let mut scenes: Vec<(String, Image<u8>)> = vec![
        ("rectangles".into(),    make_multi_rect()),
        ("circle_blobs".into(),  make_blobs()),
        ("gradient_step".into(), make_gradient_step()),
    ];

    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = args.get(1) {
        let img = load_image(path);
        println!("Loaded: {}×{}", img.width(), img.height());
        scenes.push((
            std::path::Path::new(path)
                .file_stem().unwrap_or_default()
                .to_string_lossy().into_owned(),
            img,
        ));
    }

    for (name, img) in &scenes {
        println!("\n=== Scene: {name} ({}×{}) ===", img.width(), img.height());

        // CPU: raw detection + manual NMS for comparison.
        let t0 = Instant::now();
        let mut cpu_raw = cpu_det.detect(img);
        cpu_raw.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let cpu_time = t0.elapsed();
        let cpu_suppressed = cpu_nms.suppress(&cpu_raw, img.width(), img.height());
        println!("  CPU: {} raw → {} NMS  ({:.2} ms)",
            cpu_raw.len(), cpu_suppressed.len(), cpu_time.as_secs_f64() * 1000.0);

        // GPU: FAST + NMS chained in one encoder, one round-trip.
        // Detector is created per-scene sized to the image dimensions.
        let mut gpu_det = GpuFastDetector::new(
            &gpu, THRESHOLD, ARC_LENGTH,
            img.width(), img.height(), NMS_CELL,
        );
        let t0 = Instant::now();
        let gpu_pyr = gpu_pipeline.build(&gpu, img, 1, 1.0);
        let gpu_suppressed = gpu_det.detect(&gpu, &gpu_pyr.levels[0], 0);
        let gpu_time = t0.elapsed();
        println!("  GPU: {} corners  ({:.2} ms, FAST+NMS+readback)",
            gpu_suppressed.len(), gpu_time.as_secs_f64() * 1000.0);

        let (match_pct, max_err) = position_agreement(&cpu_suppressed, &gpu_suppressed);
        println!("  NMS agreement: {match_pct:.1}%  (max pos error: {max_err:.1} px)");

        let svg = render_panels(img, &cpu_raw, &cpu_suppressed, &gpu_suppressed, name);
        let path = format!("vis_output/gpu_{name}.svg");
        fs::write(&path, svg).expect("write failed");
        println!("  → {path}");
    }

    // Multi-level: one detector sized for the largest level (level 0).
    println!("\n=== Multi-level GPU FAST ===");
    let img = make_multi_rect();
    let mut gpu_det_ml = GpuFastDetector::new(
        &gpu, THRESHOLD, ARC_LENGTH,
        img.width(), img.height(), NMS_CELL,
    );
    let gpu_pyr = gpu_pipeline.build(&gpu, &img, 4, 1.0);
    let svg = render_multilevel(&img, &gpu, &mut gpu_det_ml, &gpu_pyr, 4);
    fs::write("vis_output/gpu_multilevel_fast.svg", svg).unwrap();
    println!("  → vis_output/gpu_multilevel_fast.svg");

    println!("\nDone. Open vis_output/gpu_*.svg in your browser.");
}

// ---------------------------------------------------------------------------
// Agreement metric
// ---------------------------------------------------------------------------

fn position_agreement(cpu: &[Feature], gpu: &[Feature]) -> (f32, f32) {
    if cpu.is_empty() { return (100.0, 0.0); }
    let mut matched = 0usize;
    let mut max_err = 0.0f32;
    for c in cpu {
        if let Some(g) = gpu.iter().find(|g| {
            (g.x - c.x).abs() < (NMS_CELL as f32) && (g.y - c.y).abs() < (NMS_CELL as f32)
        }) {
            matched += 1;
            let err = ((g.x - c.x).powi(2) + (g.y - c.y).powi(2)).sqrt();
            max_err = max_err.max(err);
        }
    }
    (matched as f32 / cpu.len() as f32 * 100.0, max_err)
}

// ---------------------------------------------------------------------------
// SVG rendering
// ---------------------------------------------------------------------------

fn render_panels(
    img: &Image<u8>,
    cpu_raw: &[Feature],
    cpu_nms: &[Feature],
    gpu_nms: &[Feature],
    name: &str,
) -> String {
    let sw = img.width() * SCALE;
    let sh = img.height() * SCALE;
    let gap = 20;
    let top_pad = 50;
    let bot_pad = 20;
    let total_w = sw * 3 + gap * 2;
    let total_h = sh + top_pad + bot_pad;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" \
        viewBox=\"0 0 {total_w} {total_h}\" width=\"{total_w}\" height=\"{total_h}\">").unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 11px; fill: #222; }}</style>").unwrap();
    writeln!(svg, "<rect width=\"{total_w}\" height=\"{total_h}\" fill=\"#f8f8f8\"/>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">\
        {name} — CPU vs GPU FAST-{ARC_LENGTH} (thr={THRESHOLD})</text>").unwrap();

    let panels: &[(usize, &[Feature], bool, &str, String)] = &[
        (0,          cpu_raw, false, "ff3333", format!("CPU raw ({} corners)", cpu_raw.len())),
        (sw+gap,     cpu_nms, true,  "22cc44", format!("CPU+NMS ({} corners)", cpu_nms.len())),
        (2*(sw+gap), gpu_nms, true,  "00cccc", format!("GPU+NMS ({} corners)", gpu_nms.len())),
    ];

    for (x_off, features, show_grid, color, label) in panels {
        writeln!(svg, "<g transform=\"translate({x_off}, {top_pad})\">").unwrap();
        writeln!(svg, "<text x=\"2\" y=\"-5\">{label}</text>").unwrap();
        write_image_rects(&mut svg, img);
        if *show_grid { write_grid_overlay(&mut svg, img.width(), img.height(), NMS_CELL); }
        write_feature_circles(&mut svg, features, color);
        writeln!(svg, "</g>").unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

fn render_multilevel(
    img: &Image<u8>,
    gpu: &GpuDevice,
    det: &mut GpuFastDetector,
    pyr: &rudolf_v::gpu::pyramid::GpuPyramid,
    num_levels: usize,
) -> String {
    let colors = ["ff3333", "ff8800", "22cc44", "2266ff"];
    let gap = 10;
    let total_w: usize = (0..num_levels)
        .map(|l| pyr.levels[l].width as usize * SCALE + gap)
        .sum::<usize>().saturating_sub(gap);
    let total_h = pyr.levels[0].height as usize * SCALE + 60;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" \
        viewBox=\"0 0 {total_w} {total_h}\" width=\"{total_w}\" height=\"{total_h}\">").unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 11px; fill: #222; }}</style>").unwrap();
    writeln!(svg, "<rect width=\"{total_w}\" height=\"{total_h}\" fill=\"#f8f8f8\"/>").unwrap();
    writeln!(svg, "<text x=\"6\" y=\"16\" font-weight=\"bold\" font-size=\"13\">\
        Multi-level GPU FAST-{ARC_LENGTH} (thr={THRESHOLD})</text>").unwrap();

    let mut x_off = 0usize;
    for lvl in 0..num_levels {
        let level = &pyr.levels[lvl];
        let lw = level.width as usize;
        let lh = level.height as usize;
        let gpu_pixels = pyr.readback_level(gpu, lvl);
        let features = det.detect(gpu, level, lvl);
        println!("  GPU level {lvl} ({}×{}): {} corners", lw, lh, features.len());

        let sw = lw * SCALE;
        let sh = lh * SCALE;
        let color = colors[lvl % colors.len()];

        writeln!(svg, "<g transform=\"translate({x_off}, 30)\">").unwrap();
        writeln!(svg, "<text x=\"2\" y=\"-5\">L{lvl} {}×{} — {} corners</text>",
            lw, lh, features.len()).unwrap();

        for y in 0..lh {
            for x in 0..lw {
                let v = gpu_pixels[y * lw + x].clamp(0.0, 255.0) as u8;
                let px = x * SCALE;
                let py = y * SCALE;
                writeln!(svg, "  <rect x=\"{px}\" y=\"{py}\" width=\"{SCALE}\" height=\"{SCALE}\" \
                    fill=\"rgb({v},{v},{v})\"/>").unwrap();
            }
        }

        let r = SCALE + 1;
        for f in &features {
            let cx = f.x as usize * SCALE + SCALE / 2;
            let cy = f.y as usize * SCALE + SCALE / 2;
            writeln!(svg, "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" \
                fill=\"none\" stroke=\"#{color}\" stroke-width=\"1.5\" opacity=\"0.9\"/>").unwrap();
            writeln!(svg, "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"1.5\" fill=\"#{color}\"/>").unwrap();
        }

        writeln!(svg, "  <rect x=\"0\" y=\"0\" width=\"{sw}\" height=\"{sh}\" \
            fill=\"none\" stroke=\"#aaa\" stroke-width=\"0.5\"/>").unwrap();
        writeln!(svg, "</g>").unwrap();
        x_off += lw * SCALE + gap;
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

fn write_image_rects(svg: &mut String, img: &Image<u8>) {
    if img.width() > MAX_RENDER_DIM || img.height() > MAX_RENDER_DIM {
        let sw = img.width() * SCALE;
        let sh = img.height() * SCALE;
        writeln!(svg, "  <rect x=\"0\" y=\"0\" width=\"{sw}\" height=\"{sh}\" fill=\"#555\"/>").unwrap();
        writeln!(svg, "  <text x=\"6\" y=\"16\" fill=\"white\" font-size=\"11\">\
            {}×{} (pixel rendering skipped — too large)</text>",
            img.width(), img.height()).unwrap();
        return;
    }
    for y in 0..img.height() {
        for x in 0..img.width() {
            let v = img.get(x, y);
            let px = x * SCALE;
            let py = y * SCALE;
            writeln!(svg, "  <rect x=\"{px}\" y=\"{py}\" width=\"{SCALE}\" height=\"{SCALE}\" \
                fill=\"rgb({v},{v},{v})\"/>").unwrap();
        }
    }
}

fn write_feature_circles(svg: &mut String, features: &[Feature], color: &str) {
    let r = SCALE + 1;
    for f in features {
        let cx = f.x as usize * SCALE + SCALE / 2;
        let cy = f.y as usize * SCALE + SCALE / 2;
        writeln!(svg, "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" \
            fill=\"none\" stroke=\"#{color}\" stroke-width=\"1.5\" opacity=\"0.9\"/>").unwrap();
        writeln!(svg, "  <circle cx=\"{cx}\" cy=\"{cy}\" r=\"1.5\" fill=\"#{color}\"/>").unwrap();
    }
}

fn write_grid_overlay(svg: &mut String, img_w: usize, img_h: usize, cell_size: usize) {
    let sw = img_w * SCALE;
    let sh = img_h * SCALE;
    let cell_px = cell_size * SCALE;
    let mut x = cell_px;
    while x < sw {
        writeln!(svg, "  <line x1=\"{x}\" y1=\"0\" x2=\"{x}\" y2=\"{sh}\" \
            stroke=\"#ffff00\" stroke-width=\"0.5\" opacity=\"0.4\"/>").unwrap();
        x += cell_px;
    }
    let mut y = cell_px;
    while y < sh {
        writeln!(svg, "  <line x1=\"0\" y1=\"{y}\" x2=\"{sw}\" y2=\"{y}\" \
            stroke=\"#ffff00\" stroke-width=\"0.5\" opacity=\"0.4\"/>").unwrap();
        y += cell_px;
    }
}

// ---------------------------------------------------------------------------
// Test scenes
// ---------------------------------------------------------------------------

fn make_multi_rect() -> Image<u8> {
    let mut img = Image::from_vec(120, 120, vec![20u8; 120 * 120]);
    for &(rx, ry, rw, rh) in &[(10,10,25,25),(50,8,35,20),(15,55,20,40),(60,45,40,30),(75,85,30,25)] {
        for y in ry..(ry+rh).min(120) {
            for x in rx..(rx+rw).min(120) { img.set(x, y, 220); }
        }
    }
    img
}

fn make_blobs() -> Image<u8> {
    let mut img = Image::from_vec(100, 100, vec![40u8; 10000]);
    for &(cx, cy, r) in &[(25.0f32,25.0,8.0),(70.0,20.0,6.0),(50.0,55.0,10.0),(20.0,75.0,7.0),(80.0,75.0,9.0)] {
        for y in 0..100usize {
            for x in 0..100usize {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx*dx + dy*dy <= r*r { img.set(x, y, 210); }
            }
        }
    }
    img
}

fn make_gradient_step() -> Image<u8> {
    let mut img = Image::new(100, 100);
    for y in 0..100 {
        for x in 0..100 {
            let base: u8 = if x < 50 { 30 } else { 200 };
            img.set(x, y, base.saturating_add((y as f32 * 0.3) as u8));
        }
    }
    img
}

fn load_image(path: &str) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .to_luma8();
    let (w, h) = img.dimensions();
    Image::<u8>::from_vec(w as usize, h as usize, img.into_raw())
}
