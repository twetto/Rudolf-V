// examples/visualize_fast.rs
//
// Generate SVG visualizations of FAST corner detection and NMS.
// Run with: cargo run --example visualize_fast
// Output: vis_output/*.svg — open in any browser.

use rudolf_v::fast::FastDetector;
use rudolf_v::image::Image;
use rudolf_v::nms::OccupancyNms;
use rudolf_v::pyramid::Pyramid;

use std::fmt::Write;
use std::fs;

fn main() {
    fs::create_dir_all("vis_output").expect("failed to create vis_output/");

    let scenes: Vec<(&str, Image<u8>)> = vec![
        ("rectangles", make_multi_rect()),
        ("circle_blobs", make_blobs()),
        ("gradient_step", make_gradient_step()),
    ];

    for (name, img) in &scenes {
        println!("=== Scene: {name} ({}x{}) ===", img.width(), img.height());

        let det = FastDetector::new(20, 9);
        let raw = det.detect(img);
        println!("  Raw FAST corners: {}", raw.len());

        let nms = OccupancyNms::new(16);
        let suppressed = nms.suppress(&raw, img.width(), img.height());
        println!("  After NMS (cell=16): {}", suppressed.len());

        let title_raw = format!("{name} - FAST raw ({} pts)", raw.len());
        let svg_raw = render_svg(img, &raw, None, &title_raw);
        fs::write(format!("vis_output/{name}_fast_raw.svg"), &svg_raw).unwrap();

        let title_nms = format!("{name} - after NMS ({} pts)", suppressed.len());
        let svg_nms = render_svg(img, &suppressed, Some(16), &title_nms);
        fs::write(format!("vis_output/{name}_fast_nms.svg"), &svg_nms).unwrap();

        let svg_compare = render_comparison(img, &raw, &suppressed, 16, name);
        fs::write(format!("vis_output/{name}_compare.svg"), &svg_compare).unwrap();
    }

    // Pyramid visualization
    println!("\n=== Pyramid ===");
    let img = make_multi_rect();
    let pyr = Pyramid::build(&img, 4, 1.0);
    let svg_pyr = render_pyramid(&pyr);
    fs::write("vis_output/pyramid.svg", &svg_pyr).unwrap();

    // Multi-level FAST
    let svg_ml = render_multilevel_fast(&pyr);
    fs::write("vis_output/multilevel_fast.svg", &svg_ml).unwrap();

    println!("\nDone! Open vis_output/*.svg in your browser.");
}

// ---------------------------------------------------------------------------
// Test scenes
// ---------------------------------------------------------------------------

fn make_multi_rect() -> Image<u8> {
    let mut img = Image::from_vec(120, 120, vec![20u8; 120 * 120]);
    let rects = [
        (10, 10, 25, 25),
        (50, 8, 35, 20),
        (15, 55, 20, 40),
        (60, 45, 40, 30),
        (75, 85, 30, 25),
    ];
    for &(rx, ry, rw, rh) in &rects {
        for y in ry..(ry + rh).min(120) {
            for x in rx..(rx + rw).min(120) {
                img.set(x, y, 220);
            }
        }
    }
    img
}

fn make_blobs() -> Image<u8> {
    let mut img = Image::from_vec(100, 100, vec![40u8; 10000]);
    let blobs: [(f32, f32, f32); 5] = [
        (25.0, 25.0, 8.0),
        (70.0, 20.0, 6.0),
        (50.0, 55.0, 10.0),
        (20.0, 75.0, 7.0),
        (80.0, 75.0, 9.0),
    ];
    for &(cx, cy, r) in &blobs {
        for y in 0..100 {
            for x in 0..100 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r * r {
                    img.set(x, y, 210);
                }
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
            let grad = (y as f32 * 0.3) as u8;
            img.set(x, y, base.saturating_add(grad));
        }
    }
    img
}

// ---------------------------------------------------------------------------
// SVG rendering
// ---------------------------------------------------------------------------

const SCALE: usize = 4;

fn render_svg(
    img: &Image<u8>,
    features: &[rudolf_v::fast::Feature],
    grid_cell: Option<usize>,
    title: &str,
) -> String {
    let sw = img.width() * SCALE;
    let sh = img.height() * SCALE;
    let total_h = sh + 40;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        sw, total_h, sw, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 14px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"20\" font-weight=\"bold\">{}</text>", title).unwrap();

    writeln!(svg, "<g transform=\"translate(0, 30)\">").unwrap();
    write_image_rects(&mut svg, img);

    if let Some(cell) = grid_cell {
        write_grid_overlay(&mut svg, img.width(), img.height(), cell);
    }

    write_feature_circles(&mut svg, features, "ff2222");

    writeln!(svg, "</g>").unwrap();
    writeln!(svg, "</svg>").unwrap();
    svg
}

fn render_comparison(
    img: &Image<u8>,
    raw: &[rudolf_v::fast::Feature],
    suppressed: &[rudolf_v::fast::Feature],
    cell_size: usize,
    name: &str,
) -> String {
    let sw = img.width() * SCALE;
    let sh = img.height() * SCALE;
    let gap = 30;
    let total_w = sw * 2 + gap;
    let total_h = sh + 60;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        total_w, total_h, total_w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">{} - FAST-9 (threshold=20)</text>", name).unwrap();

    // Left: raw
    writeln!(svg, "<g transform=\"translate(0, 40)\">").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">Raw: {} features</text>", raw.len()).unwrap();
    write_image_rects(&mut svg, img);
    write_feature_circles(&mut svg, raw, "ff2222");
    writeln!(svg, "</g>").unwrap();

    // Right: NMS
    let offset = sw + gap;
    writeln!(svg, "<g transform=\"translate({}, 40)\">", offset).unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">NMS (cell={}): {} features</text>", cell_size, suppressed.len()).unwrap();
    write_image_rects(&mut svg, img);
    write_grid_overlay(&mut svg, img.width(), img.height(), cell_size);
    write_feature_circles(&mut svg, suppressed, "22cc44");
    writeln!(svg, "</g>").unwrap();

    writeln!(svg, "</svg>").unwrap();
    svg
}

fn render_pyramid(pyr: &Pyramid) -> String {
    let mut total_w = 0;
    let mut max_h = 0;
    for level in &pyr.levels {
        total_w += level.width() * SCALE + 10;
        max_h = max_h.max(level.height() * SCALE);
    }
    let total_h = max_h + 50;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        total_w, total_h, total_w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">Gaussian Pyramid (sigma=1.0, {} levels)</text>",
        pyr.num_levels()).unwrap();

    let mut x_off = 0;
    for (lvl, level) in pyr.levels.iter().enumerate() {
        writeln!(svg, "<g transform=\"translate({}, 40)\">", x_off).unwrap();
        writeln!(svg, "<text x=\"2\" y=\"-5\">L{} ({}x{})</text>", lvl, level.width(), level.height()).unwrap();
        write_f32_image_rects(&mut svg, level);
        writeln!(svg, "</g>").unwrap();
        x_off += level.width() * SCALE + 10;
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

fn render_multilevel_fast(pyr: &Pyramid) -> String {
    let det = FastDetector::new(20, 9);
    let colors = ["ff2222", "22cc44", "2266ff", "ff8800"];

    let mut total_w = 0;
    let mut max_h = 0;
    for level in &pyr.levels {
        total_w += level.width() * SCALE + 10;
        max_h = max_h.max(level.height() * SCALE);
    }
    let total_h = max_h + 50;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        total_w, total_h, total_w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">Multi-level FAST-9 (threshold=20)</text>").unwrap();

    let mut x_off = 0;
    for (lvl, level) in pyr.levels.iter().enumerate() {
        // Convert f32 pyramid level to u8 for FAST
        let mut u8_img = Image::new(level.width(), level.height());
        for (x, y, v) in level.pixels() {
            u8_img.set(x, y, v.clamp(0.0, 255.0).round() as u8);
        }

        let features = det.detect_at_level(&u8_img, lvl);
        let color = colors[lvl % colors.len()];

        writeln!(svg, "<g transform=\"translate({}, 40)\">", x_off).unwrap();
        writeln!(svg, "<text x=\"2\" y=\"-5\">L{}: {} pts</text>", lvl, features.len()).unwrap();
        write_f32_image_rects(&mut svg, level);
        write_feature_circles(&mut svg, &features, color);
        writeln!(svg, "</g>").unwrap();

        x_off += level.width() * SCALE + 10;
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

fn write_image_rects(svg: &mut String, img: &Image<u8>) {
    for y in 0..img.height() {
        for x in 0..img.width() {
            let v = img.get(x, y);
            let px = x * SCALE;
            let py = y * SCALE;
            writeln!(svg, "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>",
                px, py, SCALE, SCALE, v, v, v).unwrap();
        }
    }
}

fn write_f32_image_rects(svg: &mut String, img: &Image<f32>) {
    for y in 0..img.height() {
        for x in 0..img.width() {
            let v = img.get(x, y).clamp(0.0, 255.0) as u8;
            let px = x * SCALE;
            let py = y * SCALE;
            writeln!(svg, "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>",
                px, py, SCALE, SCALE, v, v, v).unwrap();
        }
    }
}

fn write_feature_circles(svg: &mut String, features: &[rudolf_v::fast::Feature], color: &str) {
    let r = SCALE + 1;
    for f in features {
        let cx = f.x as usize * SCALE + SCALE / 2;
        let cy = f.y as usize * SCALE + SCALE / 2;
        writeln!(svg,
            "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"#{}\" stroke-width=\"1.5\" opacity=\"0.9\"/>",
            cx, cy, r, color).unwrap();
        writeln!(svg,
            "  <circle cx=\"{}\" cy=\"{}\" r=\"1.5\" fill=\"#{}\"/>",
            cx, cy, color).unwrap();
    }
}

fn write_grid_overlay(svg: &mut String, img_w: usize, img_h: usize, cell_size: usize) {
    let sw = img_w * SCALE;
    let sh = img_h * SCALE;
    let cell_scaled = cell_size * SCALE;

    let mut x = cell_scaled;
    while x < sw {
        writeln!(svg,
            "  <line x1=\"{}\" y1=\"0\" x2=\"{}\" y2=\"{}\" stroke=\"#ffff00\" stroke-width=\"0.5\" opacity=\"0.4\"/>",
            x, x, sh).unwrap();
        x += cell_scaled;
    }
    let mut y = cell_scaled;
    while y < sh {
        writeln!(svg,
            "  <line x1=\"0\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ffff00\" stroke-width=\"0.5\" opacity=\"0.4\"/>",
            y, sw, y).unwrap();
        y += cell_scaled;
    }
}
