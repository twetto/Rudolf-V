// tests/test_fast.rs — Integration tests for FAST detector + NMS.
//
// IMPORTANT: FAST detects "segment" corners — bright (or dark) spots where
// a contiguous arc on the Bresenham circle is all brighter/darker than the
// center. It does NOT detect chessboard junction corners (where four regions
// meet) — that's what Harris is for. At a chessboard junction the circle
// spans all four quadrants, giving alternating runs of ~4 bright and ~4 dark,
// never reaching 9 contiguous.
//
// Good FAST test patterns: bright rectangles on dark background. The corners
// of a bright rectangle have a large contiguous dark arc on the circle.

use rudolf_v::fast::FastDetector;
use rudolf_v::image::Image;
use rudolf_v::nms::OccupancyNms;

/// Place a bright rectangle on a dark background.
fn make_rectangle_image(
    img_w: usize,
    img_h: usize,
    rect_x: usize,
    rect_y: usize,
    rect_w: usize,
    rect_h: usize,
    bg: u8,
    fg: u8,
) -> Image<u8> {
    let mut img = Image::from_vec(img_w, img_h, vec![bg; img_w * img_h]);
    for y in rect_y..(rect_y + rect_h) {
        for x in rect_x..(rect_x + rect_w) {
            img.set(x, y, fg);
        }
    }
    img
}

/// Place multiple bright squares on a dark background to get many corners.
fn make_multi_square_image() -> Image<u8> {
    let mut img = Image::from_vec(100, 100, vec![20u8; 10000]);
    let squares = [(10, 10), (60, 10), (10, 60), (60, 60)];
    for &(sx, sy) in &squares {
        for y in sy..sy + 20 {
            for x in sx..sx + 20 {
                img.set(x, y, 220);
            }
        }
    }
    img
}

// ===== Basic detection =====

#[test]
fn rectangle_corners_detected() {
    let img = make_rectangle_image(60, 60, 15, 15, 30, 30, 20, 220);
    let det = FastDetector::new(30, 9);
    let features = det.detect(&img);

    assert!(
        features.len() >= 4,
        "expected at least 4 corners from a rectangle, got {}",
        features.len()
    );
}

#[test]
fn multi_square_detects_many_corners() {
    let img = make_multi_square_image();
    let det = FastDetector::new(30, 9);
    let features = det.detect(&img);

    assert!(
        features.len() >= 10,
        "expected many corners from 4 squares, got {}",
        features.len()
    );
}

#[test]
fn corners_near_rectangle_edges() {
    let rx = 20;
    let ry = 20;
    let rw = 25;
    let rh = 25;
    let img = make_rectangle_image(70, 70, rx, ry, rw, rh, 20, 220);
    let det = FastDetector::new(30, 9);
    let features = det.detect(&img);

    let tolerance = 5.0;
    for f in &features {
        let dx_to_edge = (f.x - rx as f32)
            .abs()
            .min((f.x - (rx + rw) as f32).abs());
        let dy_to_edge = (f.y - ry as f32)
            .abs()
            .min((f.y - (ry + rh) as f32).abs());
        let near_edge = dx_to_edge <= tolerance || dy_to_edge <= tolerance;
        assert!(
            near_edge,
            "feature at ({:.0}, {:.0}) is far from rectangle edges",
            f.x, f.y,
        );
    }
}

// ===== NMS integration =====

#[test]
fn nms_reduces_rectangle_features() {
    let img = make_multi_square_image();
    let det = FastDetector::new(30, 9);
    let raw = det.detect(&img);
    assert!(!raw.is_empty(), "need features to test NMS");

    let nms = OccupancyNms::new(16);
    let suppressed = nms.suppress(&raw, img.width(), img.height());

    assert!(
        suppressed.len() <= raw.len(),
        "NMS should not increase count: {} → {}",
        raw.len(),
        suppressed.len(),
    );
    assert!(!suppressed.is_empty(), "NMS should keep some features");
}

#[test]
fn nms_spatial_separation() {
    let img = make_multi_square_image();
    let det = FastDetector::new(30, 9);
    let raw = det.detect(&img);
    let cell_size = 16;
    let nms = OccupancyNms::new(cell_size);
    let suppressed = nms.suppress(&raw, img.width(), img.height());

    for i in 0..suppressed.len() {
        for j in (i + 1)..suppressed.len() {
            let ci = (
                suppressed[i].x as usize / cell_size,
                suppressed[i].y as usize / cell_size,
            );
            let cj = (
                suppressed[j].x as usize / cell_size,
                suppressed[j].y as usize / cell_size,
            );
            assert_ne!(
                ci, cj,
                "two suppressed features in same cell: ({:.0},{:.0}) and ({:.0},{:.0})",
                suppressed[i].x, suppressed[i].y, suppressed[j].x, suppressed[j].y,
            );
        }
    }
}

// ===== FAST on pyramid levels =====

#[test]
fn fast_on_pyramid_levels() {
    let img = make_multi_square_image();
    let pyr = rudolf_v::pyramid::Pyramid::build(&img, 3, 1.0);
    let det = FastDetector::new(20, 9);

    for lvl in 0..pyr.num_levels() {
        let level_img = &pyr.levels[lvl];
        let mut u8_img = Image::new(level_img.width(), level_img.height());
        for (x, y, v) in level_img.pixels() {
            u8_img.set(x, y, v.clamp(0.0, 255.0).round() as u8);
        }

        let features = det.detect_at_level(&u8_img, lvl);
        for f in &features {
            assert_eq!(f.level, lvl, "feature level tag mismatch");
        }
    }
}

// ===== Edge cases =====

#[test]
fn detect_on_gradient_image() {
    let mut img = Image::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            img.set(x, y, ((x * 4).min(255)) as u8);
        }
    }
    let det = FastDetector::new(20, 9);
    let features = det.detect(&img);
    assert!(
        features.len() < 10,
        "smooth gradient produced too many corners: {}",
        features.len()
    );
}
