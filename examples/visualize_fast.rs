// examples/visualize_fast.rs
//
// Generate SVG visualizations of FAST corner detection and NMS.
// Run with: cargo run --example visualize_fast
// Output: vis_output/*.svg — open in any browser.

use rudolf_v::fast::FastDetector;
use rudolf_v::harris::HarrisDetector;
use rudolf_v::image::Image;
use rudolf_v::klt::{KltTracker, LkMethod, TrackStatus};
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

    // Harris on chessboard — the test FAST can't pass
    println!("\n=== Harris ===");
    let chessboard = make_chessboard(80, 10, 20, 230);
    let harris = HarrisDetector::new(0.04, 1e6, 2);
    let harris_raw = harris.detect(&chessboard);
    println!("  Chessboard Harris raw: {}", harris_raw.len());

    let nms = OccupancyNms::new(12);
    let harris_nms = nms.suppress(&harris_raw, chessboard.width(), chessboard.height());
    println!("  After NMS (cell=12): {}", harris_nms.len());

    let svg = render_comparison(&chessboard, &harris_raw, &harris_nms, 12, "chessboard Harris");
    fs::write("vis_output/harris_chessboard.svg", &svg).unwrap();

    // Harris response heatmap
    let response = harris.corner_response(&chessboard);
    let svg_resp = render_response_heatmap(&response, "Harris response (chessboard)");
    fs::write("vis_output/harris_response.svg", &svg_resp).unwrap();

    // FAST vs Harris side-by-side on the same image
    let fast_det = FastDetector::new(20, 9);
    let fast_chess = fast_det.detect(&chessboard);
    println!("  FAST on chessboard: {} (expect ~0)", fast_chess.len());

    let svg_vs = render_fast_vs_harris(&chessboard, &fast_chess, &harris_nms);
    fs::write("vis_output/fast_vs_harris.svg", &svg_vs).unwrap();

    // ===== KLT Tracking =====
    println!("\n=== KLT Tracking ===");

    // Scene: multiple squares shifted by known amounts.
    let klt_img1 = make_klt_scene(0, 0);
    let klt_img2 = make_klt_scene(4, 2);
    let klt_img3 = make_klt_scene(8, 4); // cumulative shift for 3-frame sequence

    let klt_pyr1 = Pyramid::build(&klt_img1, 3, 1.0);
    let klt_pyr2 = Pyramid::build(&klt_img2, 3, 1.0);
    let klt_pyr3 = Pyramid::build(&klt_img3, 3, 1.0);

    // Detect in frame 1.
    let klt_det = FastDetector::new(20, 9);
    let mut u8_l0 = Image::new(klt_pyr1.levels[0].width(), klt_pyr1.levels[0].height());
    for (x, y, v) in klt_pyr1.levels[0].pixels() {
        u8_l0.set(x, y, v.clamp(0.0, 255.0).round() as u8);
    }
    let detected = klt_det.detect(&u8_l0);
    let nms_klt = OccupancyNms::new(12);
    let features = nms_klt.suppress(&detected, klt_img1.width(), klt_img1.height());
    println!("  Detected features (frame 1): {}", features.len());

    // Track frame 1 → 2.
    let tracker = KltTracker::new(7, 30, 0.01, 3);
    let results_12 = tracker.track(&klt_pyr1, &klt_pyr2, &features);
    let tracked_12 = results_12.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    println!("  Tracked frame 1→2: {}", tracked_12);

    // Single-step flow arrows.
    let svg_klt1 = render_klt_flow(&klt_img1, &klt_img2, &features, &results_12,
        "KLT: frame 1 -> 2 (shift +4, +2)");
    fs::write("vis_output/klt_flow_1to2.svg", &svg_klt1).unwrap();

    // Track frame 2 → 3 using tracked positions from step 1.
    let features_f2: Vec<_> = results_12.iter()
        .filter(|r| r.status == TrackStatus::Tracked)
        .map(|r| r.feature.clone())
        .collect();
    let results_23 = tracker.track(&klt_pyr2, &klt_pyr3, &features_f2);

    let svg_klt2 = render_klt_flow(&klt_img2, &klt_img3, &features_f2, &results_23,
        "KLT: frame 2 -> 3 (shift +4, +2)");
    fs::write("vis_output/klt_flow_2to3.svg", &svg_klt2).unwrap();

    // 3-frame track visualization: frame1 positions → frame2 → frame3 overlaid.
    let svg_multi = render_klt_multiframe(
        &klt_img3, &features, &results_12, &results_23,
        "KLT: 3-frame tracking (cumulative +8, +4)");
    fs::write("vis_output/klt_multiframe.svg", &svg_multi).unwrap();

    // Gaussian blob sub-pixel tracking demo.
    let (blob1, blob2) = make_blob_pair(1.7, 0.8);
    let blob_pyr1 = Pyramid::build(&blob1, 3, 1.0);
    let blob_pyr2 = Pyramid::build(&blob2, 3, 1.0);

    let blob_features = vec![rudolf_v::fast::Feature {
        x: 50.0, y: 50.0, score: 100.0, level: 0, id: 1,
    }];
    let blob_results = tracker.track(&blob_pyr1, &blob_pyr2, &blob_features);
    if let Some(r) = blob_results.first() {
        let dx = r.feature.x - 50.0;
        let dy = r.feature.y - 50.0;
        println!("  Blob sub-pixel: tracked ({dx:.3}, {dy:.3}), expected (1.700, 0.800)");
    }

    let svg_blob = render_klt_flow(&blob1, &blob2, &blob_features, &blob_results,
        &format!("KLT sub-pixel (FA): shift (1.7, 0.8), recovered ({:.2}, {:.2})",
            blob_results[0].feature.x - 50.0, blob_results[0].feature.y - 50.0));
    fs::write("vis_output/klt_subpixel.svg", &svg_blob).unwrap();

    // ===== FA vs IC comparison =====
    println!("\n=== FA vs IC Comparison ===");
    let cmp_img1 = make_klt_scene(0, 0);
    let cmp_img2 = make_klt_scene(4, 2);
    let cmp_pyr1 = Pyramid::build(&cmp_img1, 3, 1.0);
    let cmp_pyr2 = Pyramid::build(&cmp_img2, 3, 1.0);

    // Detect features.
    let mut cmp_u8 = Image::new(cmp_pyr1.levels[0].width(), cmp_pyr1.levels[0].height());
    for (x, y, v) in cmp_pyr1.levels[0].pixels() {
        cmp_u8.set(x, y, v.clamp(0.0, 255.0).round() as u8);
    }
    let cmp_features_raw = klt_det.detect(&cmp_u8);
    let cmp_features = nms_klt.suppress(&cmp_features_raw, cmp_img1.width(), cmp_img1.height());

    let fa_tracker = KltTracker::with_method(7, 30, 0.01, 3, LkMethod::ForwardAdditive);
    let ic_tracker = KltTracker::with_method(7, 30, 0.01, 3, LkMethod::InverseCompositional);

    let fa_results = fa_tracker.track(&cmp_pyr1, &cmp_pyr2, &cmp_features);
    let ic_results = ic_tracker.track(&cmp_pyr1, &cmp_pyr2, &cmp_features);

    let fa_tracked = fa_results.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    let ic_tracked = ic_results.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    println!("  FA tracked: {}/{}", fa_tracked, cmp_features.len());
    println!("  IC tracked: {}/{}", ic_tracked, cmp_features.len());

    let svg_fa_vs_ic = render_fa_vs_ic(&cmp_img2, &cmp_features, &fa_results, &ic_results);
    fs::write("vis_output/klt_fa_vs_ic.svg", &svg_fa_vs_ic).unwrap();

    // Sub-pixel comparison.
    let ic_blob_tracker = KltTracker::with_method(7, 30, 0.01, 3, LkMethod::InverseCompositional);
    let ic_blob_results = ic_blob_tracker.track(&blob_pyr1, &blob_pyr2, &blob_features);
    if let Some(r) = ic_blob_results.first() {
        let dx = r.feature.x - 50.0;
        let dy = r.feature.y - 50.0;
        println!("  IC blob sub-pixel: ({dx:.3}, {dy:.3}), expected (1.700, 0.800)");
    }
    let svg_ic_blob = render_klt_flow(&blob1, &blob2, &blob_features, &ic_blob_results,
        &format!("KLT sub-pixel (IC): shift (1.7, 0.8), recovered ({:.2}, {:.2})",
            ic_blob_results[0].feature.x - 50.0, ic_blob_results[0].feature.y - 50.0));
    fs::write("vis_output/klt_subpixel_ic.svg", &svg_ic_blob).unwrap();

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

fn make_chessboard(img_size: usize, cell_size: usize, lo: u8, hi: u8) -> Image<u8> {
    let mut img = Image::new(img_size, img_size);
    for y in 0..img_size {
        for x in 0..img_size {
            let cx = x / cell_size;
            let cy = y / cell_size;
            let val = if (cx + cy) % 2 == 0 { lo } else { hi };
            img.set(x, y, val);
        }
    }
    img
}

/// Scene for KLT demos: multiple bright rectangles on dark background,
/// shifted by (shift_x, shift_y) to simulate inter-frame motion.
fn make_klt_scene(shift_x: usize, shift_y: usize) -> Image<u8> {
    let w = 160;
    let h = 120;
    let mut img = Image::from_vec(w, h, vec![25u8; w * h]);

    let rects: [(usize, usize, usize, usize, u8); 6] = [
        (30, 25, 20, 20, 200),
        (70, 20, 25, 15, 180),
        (110, 30, 18, 22, 210),
        (25, 65, 22, 25, 190),
        (75, 60, 30, 20, 170),
        (115, 70, 20, 18, 205),
    ];

    for &(rx, ry, rw, rh, val) in &rects {
        let rx = rx + shift_x;
        let ry = ry + shift_y;
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                img.set(x, y, val);
            }
        }
    }
    img
}

/// A pair of Gaussian blob images for sub-pixel tracking demo.
fn make_blob_pair(dx: f32, dy: f32) -> (Image<u8>, Image<u8>) {
    let w = 100;
    let h = 100;
    let cx = 50.0f32;
    let cy = 50.0f32;
    let sigma2 = 80.0;

    let mut img1 = Image::from_vec(w, h, vec![30u8; w * h]);
    let mut img2 = Image::from_vec(w, h, vec![30u8; w * h]);

    for y in 0..h {
        for x in 0..w {
            let d1 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
            let v1 = 30.0 + 200.0 * (-d1 / sigma2).exp();
            img1.set(x, y, v1.min(255.0) as u8);

            let d2 = (x as f32 - cx - dx).powi(2) + (y as f32 - cy - dy).powi(2);
            let v2 = 30.0 + 200.0 * (-d2 / sigma2).exp();
            img2.set(x, y, v2.min(255.0) as u8);
        }
    }
    (img1, img2)
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
// KLT rendering
// ---------------------------------------------------------------------------

/// Render KLT optical flow: left = frame1 with features, right = frame2 with
/// arrows from original to tracked positions.
fn render_klt_flow(
    img1: &Image<u8>,
    img2: &Image<u8>,
    features: &[rudolf_v::fast::Feature],
    results: &[rudolf_v::klt::TrackedFeature],
    title: &str,
) -> String {
    let sw = img1.width() * SCALE;
    let sh = img1.height() * SCALE;
    let gap = 30;
    let total_w = sw * 2 + gap;
    let total_h = sh + 60;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        total_w, total_h, total_w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<defs><marker id=\"ah\" markerWidth=\"8\" markerHeight=\"6\" refX=\"8\" refY=\"3\" orient=\"auto\">\
        <path d=\"M0,0 L8,3 L0,6\" fill=\"#22dd44\"/></marker></defs>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">{}</text>", title).unwrap();

    // Left: frame 1 with detected features.
    writeln!(svg, "<g transform=\"translate(0, 40)\">").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">Frame 1: {} features</text>", features.len()).unwrap();
    write_image_rects(&mut svg, img1);
    write_feature_circles(&mut svg, features, "4488ff");
    writeln!(svg, "</g>").unwrap();

    // Right: frame 2 with flow arrows.
    let offset = sw + gap;
    let tracked_count = results.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    let lost_count = results.len() - tracked_count;
    writeln!(svg, "<g transform=\"translate({}, 40)\">", offset).unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">Frame 2: {} tracked, {} lost</text>",
        tracked_count, lost_count).unwrap();
    write_image_rects(&mut svg, img2);

    for (feat, result) in features.iter().zip(results.iter()) {
        let x0 = feat.x as usize * SCALE + SCALE / 2;
        let y0 = feat.y as usize * SCALE + SCALE / 2;

        match result.status {
            TrackStatus::Tracked => {
                let x1 = (result.feature.x * SCALE as f32) as usize + SCALE / 2;
                let y1 = (result.feature.y * SCALE as f32) as usize + SCALE / 2;

                // Origin dot (blue).
                writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"#4488ff\" opacity=\"0.6\"/>",
                    x0, y0).unwrap();
                // Arrow from origin to tracked position.
                let arrow_dx = x1 as f32 - x0 as f32;
                let arrow_dy = y1 as f32 - y0 as f32;
                let arrow_len = (arrow_dx * arrow_dx + arrow_dy * arrow_dy).sqrt();
                if arrow_len > 1.0 {
                    writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                        stroke=\"#22dd44\" stroke-width=\"1.5\" marker-end=\"url(#ah)\"/>",
                        x0, y0, x1, y1).unwrap();
                }
                // Tracked position dot (green).
                writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"#22dd44\"/>",
                    x1, y1).unwrap();
            }
            _ => {
                // Lost/OOB: red X.
                let s = 3;
                writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                    x0 - s, y0 - s, x0 + s, y0 + s).unwrap();
                writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                    x0 + s, y0 - s, x0 - s, y0 + s).unwrap();
            }
        }
    }

    writeln!(svg, "</g>").unwrap();
    writeln!(svg, "</svg>").unwrap();
    svg
}

/// Render a 3-frame tracking sequence: trails from frame1 → frame2 → frame3
/// overlaid on the final frame image.
fn render_klt_multiframe(
    img3: &Image<u8>,
    features_f1: &[rudolf_v::fast::Feature],
    results_12: &[rudolf_v::klt::TrackedFeature],
    results_23: &[rudolf_v::klt::TrackedFeature],
    title: &str,
) -> String {
    let sw = img3.width() * SCALE;
    let sh = img3.height() * SCALE;
    let total_h = sh + 60;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        sw, total_h, sw, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<defs><marker id=\"ah2\" markerWidth=\"6\" markerHeight=\"4\" refX=\"6\" refY=\"2\" orient=\"auto\">\
        <path d=\"M0,0 L6,2 L0,4\" fill=\"#22dd44\"/></marker></defs>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">{}</text>", title).unwrap();

    writeln!(svg, "<g transform=\"translate(0, 40)\">").unwrap();
    write_image_rects(&mut svg, img3);

    // Build a map from frame2 features to frame3 results.
    // results_12[i] → features_f2[j] where j indexes only tracked ones.
    let mut f2_idx = 0;
    for (i, r12) in results_12.iter().enumerate() {
        if r12.status != TrackStatus::Tracked {
            continue;
        }

        let p1x = features_f1[i].x;
        let p1y = features_f1[i].y;
        let p2x = r12.feature.x;
        let p2y = r12.feature.y;

        // Check if this feature survived to frame 3.
        let (p3x, p3y, survived) = if f2_idx < results_23.len()
            && results_23[f2_idx].status == TrackStatus::Tracked
        {
            (results_23[f2_idx].feature.x, results_23[f2_idx].feature.y, true)
        } else {
            (p2x, p2y, false)
        };
        f2_idx += 1;

        // Scale to SVG coordinates.
        let sx = |v: f32| (v * SCALE as f32) as usize + SCALE / 2;

        // Frame 1 position (blue dot).
        writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"2.5\" fill=\"#4488ff\" opacity=\"0.5\"/>",
            sx(p1x), sx(p1y)).unwrap();

        // Trail: frame1 → frame2 (faded).
        writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
            stroke=\"#88aaff\" stroke-width=\"1\" opacity=\"0.5\"/>",
            sx(p1x), sx(p1y), sx(p2x), sx(p2y)).unwrap();

        if survived {
            // Trail: frame2 → frame3 (bright green arrow).
            writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"#88aaff\" opacity=\"0.5\"/>",
                sx(p2x), sx(p2y)).unwrap();
            writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                stroke=\"#22dd44\" stroke-width=\"1.5\" marker-end=\"url(#ah2)\"/>",
                sx(p2x), sx(p2y), sx(p3x), sx(p3y)).unwrap();
            // Final position (solid green).
            writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"3.5\" fill=\"none\" stroke=\"#22dd44\" stroke-width=\"1.5\"/>",
                sx(p3x), sx(p3y)).unwrap();
        } else {
            // Lost at frame 3: red X at frame 2 position.
            let cx = sx(p2x);
            let cy = sx(p2y);
            writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                cx - 3, cy - 3, cx + 3, cy + 3).unwrap();
            writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                cx + 3, cy - 3, cx - 3, cy + 3).unwrap();
        }
    }

    // Legend.
    let ly = sh + 5;
    writeln!(svg, "  <circle cx=\"10\" cy=\"{}\" r=\"3\" fill=\"#4488ff\"/>", ly).unwrap();
    writeln!(svg, "  <text x=\"18\" y=\"{}\">Frame 1</text>", ly + 4).unwrap();
    writeln!(svg, "  <circle cx=\"80\" cy=\"{}\" r=\"3\" fill=\"#88aaff\"/>", ly).unwrap();
    writeln!(svg, "  <text x=\"88\" y=\"{}\">Frame 2</text>", ly + 4).unwrap();
    writeln!(svg, "  <circle cx=\"150\" cy=\"{}\" r=\"3\" fill=\"#22dd44\"/>", ly).unwrap();
    writeln!(svg, "  <text x=\"158\" y=\"{}\">Frame 3</text>", ly + 4).unwrap();

    writeln!(svg, "</g>").unwrap();
    writeln!(svg, "</svg>").unwrap();
    svg
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

/// Forward Additive vs Inverse Compositional side-by-side.
/// Left = FA arrows, Right = IC arrows, both on the same target frame.
fn render_fa_vs_ic(
    img: &Image<u8>,
    features: &[rudolf_v::fast::Feature],
    fa_results: &[rudolf_v::klt::TrackedFeature],
    ic_results: &[rudolf_v::klt::TrackedFeature],
) -> String {
    let sw = img.width() * SCALE;
    let sh = img.height() * SCALE;
    let gap = 30;
    let total_w = sw * 2 + gap;
    let total_h = sh + 80;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        total_w, total_h, total_w, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<defs>\
        <marker id=\"ahfa\" markerWidth=\"8\" markerHeight=\"6\" refX=\"8\" refY=\"3\" orient=\"auto\">\
        <path d=\"M0,0 L8,3 L0,6\" fill=\"#22dd44\"/></marker>\
        <marker id=\"ahic\" markerWidth=\"8\" markerHeight=\"6\" refX=\"8\" refY=\"3\" orient=\"auto\">\
        <path d=\"M0,0 L8,3 L0,6\" fill=\"#dd8822\"/></marker>\
        </defs>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">Forward Additive vs Inverse Compositional</text>").unwrap();

    // Render a side (FA or IC).
    let render_side = |svg: &mut String, x_off: usize, label: &str, results: &[rudolf_v::klt::TrackedFeature], color: &str, marker: &str| {
        let tracked = results.iter().filter(|r| r.status == TrackStatus::Tracked).count();
        writeln!(svg, "<g transform=\"translate({}, 40)\">", x_off).unwrap();
        writeln!(svg, "<text x=\"10\" y=\"-5\">{}: {} tracked</text>", label, tracked).unwrap();
        write_image_rects(svg, img);

        for (feat, result) in features.iter().zip(results.iter()) {
            let x0 = feat.x as usize * SCALE + SCALE / 2;
            let y0 = feat.y as usize * SCALE + SCALE / 2;

            match result.status {
                TrackStatus::Tracked => {
                    let x1 = (result.feature.x * SCALE as f32) as usize + SCALE / 2;
                    let y1 = (result.feature.y * SCALE as f32) as usize + SCALE / 2;
                    writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"2\" fill=\"#4488ff\" opacity=\"0.5\"/>",
                        x0, y0).unwrap();
                    let arrow_len = ((x1 as f32 - x0 as f32).powi(2) + (y1 as f32 - y0 as f32).powi(2)).sqrt();
                    if arrow_len > 1.0 {
                        writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                            stroke=\"#{}\" stroke-width=\"1.5\" marker-end=\"url(#{})\"/>",
                            x0, y0, x1, y1, color, marker).unwrap();
                    }
                    writeln!(svg, "  <circle cx=\"{}\" cy=\"{}\" r=\"3\" fill=\"#{}\"/>",
                        x1, y1, color).unwrap();
                }
                _ => {
                    let s = 3;
                    writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                        x0 - s, y0 - s, x0 + s, y0 + s).unwrap();
                    writeln!(svg, "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ff3333\" stroke-width=\"1.5\"/>",
                        x0 + s, y0 - s, x0 - s, y0 + s).unwrap();
                }
            }
        }
        writeln!(svg, "</g>").unwrap();
    };

    render_side(&mut svg, 0, "Forward Additive", fa_results, "22dd44", "ahfa");
    render_side(&mut svg, sw + gap, "Inverse Compositional", ic_results, "dd8822", "ahic");

    // Displacement stats at bottom.
    let mut fa_dx_sum = 0.0f32;
    let mut fa_dy_sum = 0.0f32;
    let mut ic_dx_sum = 0.0f32;
    let mut ic_dy_sum = 0.0f32;
    let mut fa_n = 0;
    let mut ic_n = 0;
    for (feat, (fa, ic)) in features.iter().zip(fa_results.iter().zip(ic_results.iter())) {
        if fa.status == TrackStatus::Tracked {
            fa_dx_sum += fa.feature.x - feat.x;
            fa_dy_sum += fa.feature.y - feat.y;
            fa_n += 1;
        }
        if ic.status == TrackStatus::Tracked {
            ic_dx_sum += ic.feature.x - feat.x;
            ic_dy_sum += ic.feature.y - feat.y;
            ic_n += 1;
        }
    }
    let ly = sh + 50;
    if fa_n > 0 && ic_n > 0 {
        writeln!(svg, "<text x=\"10\" y=\"{}\">FA mean displacement: ({:.2}, {:.2})  |  IC mean displacement: ({:.2}, {:.2})  |  Expected: (4.0, 2.0)</text>",
            ly, fa_dx_sum / fa_n as f32, fa_dy_sum / fa_n as f32,
            ic_dx_sum / ic_n as f32, ic_dy_sum / ic_n as f32).unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    svg
}

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

/// Render Harris response as a heatmap: blue (negative/edge) → black (zero) → red (positive/corner).
fn render_response_heatmap(response: &Image<f32>, title: &str) -> String {
    let sw = response.width() * SCALE;
    let sh = response.height() * SCALE;
    let total_h = sh + 40;

    // Find min/max for normalization.
    let mut min_r = f32::MAX;
    let mut max_r = f32::MIN;
    for (_, _, v) in response.pixels() {
        if v < min_r { min_r = v; }
        if v > max_r { max_r = v; }
    }

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
        sw, total_h, sw, total_h).unwrap();
    writeln!(svg, "<style>text {{ font-family: monospace; font-size: 12px; fill: #333; }}</style>").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"20\" font-weight=\"bold\">{}</text>", title).unwrap();

    writeln!(svg, "<g transform=\"translate(0, 30)\">").unwrap();
    for y in 0..response.height() {
        for x in 0..response.width() {
            let v = response.get(x, y);
            let (r, g, b) = if v > 0.0 {
                // Positive → red (corner)
                let t = (v / max_r).sqrt().min(1.0); // sqrt for better visibility
                ((t * 255.0) as u8, 0u8, 0u8)
            } else if v < 0.0 {
                // Negative → blue (edge)
                let t = (v / min_r).sqrt().min(1.0);
                (0u8, 0u8, (t * 255.0) as u8)
            } else {
                (0, 0, 0)
            };
            let px = x * SCALE;
            let py = y * SCALE;
            writeln!(svg, "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>",
                px, py, SCALE, SCALE, r, g, b).unwrap();
        }
    }
    writeln!(svg, "</g>").unwrap();
    writeln!(svg, "</svg>").unwrap();
    svg
}

/// FAST vs Harris side-by-side on the same image.
fn render_fast_vs_harris(
    img: &Image<u8>,
    fast_features: &[rudolf_v::fast::Feature],
    harris_features: &[rudolf_v::fast::Feature],
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
    writeln!(svg, "<text x=\"10\" y=\"18\" font-weight=\"bold\" font-size=\"14\">FAST vs Harris on Chessboard</text>").unwrap();

    // Left: FAST
    writeln!(svg, "<g transform=\"translate(0, 40)\">").unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">FAST-9: {} features</text>", fast_features.len()).unwrap();
    write_image_rects(&mut svg, img);
    write_feature_circles(&mut svg, fast_features, "ff2222");
    writeln!(svg, "</g>").unwrap();

    // Right: Harris
    let offset = sw + gap;
    writeln!(svg, "<g transform=\"translate({}, 40)\">", offset).unwrap();
    writeln!(svg, "<text x=\"10\" y=\"-5\">Harris: {} features</text>", harris_features.len()).unwrap();
    write_image_rects(&mut svg, img);
    write_feature_circles(&mut svg, harris_features, "22cc44");
    writeln!(svg, "</g>").unwrap();

    writeln!(svg, "</svg>").unwrap();
    svg
}
