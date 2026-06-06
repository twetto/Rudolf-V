// tests/test_shi_tomasi.rs — Integration tests for Shi-Tomasi detection.

use rudolf_v::image::Image;
use rudolf_v::nms::OccupancyNms;
use rudolf_v::shi_tomasi::ShiTomasiDetector;

fn make_checkerboard(img_size: usize, cell_size: usize, lo: u8, hi: u8) -> Image<u8> {
    let mut img = Image::new(img_size, img_size);
    for y in 0..img_size {
        for x in 0..img_size {
            let cx = x / cell_size;
            let cy = y / cell_size;
            img.set(x, y, if (cx + cy) % 2 == 0 { lo } else { hi });
        }
    }
    img
}

#[test]
fn detects_checkerboard_junctions() {
    let img = make_checkerboard(100, 10, 20, 230);
    let det = ShiTomasiDetector::new(1e5, 2);
    let features = det.detect(&img);

    assert!(
        features.len() >= 20,
        "expected many Shi-Tomasi corners, got {}",
        features.len()
    );
}

#[test]
fn score_is_stronger_at_corner_than_edge() {
    let mut img = Image::from_vec(40, 40, vec![30u8; 1600]);
    for y in 15..25 {
        for x in 5..30 {
            img.set(x, y, 220);
        }
    }
    for y in 5..30 {
        for x in 15..25 {
            img.set(x, y, 220);
        }
    }

    let det = ShiTomasiDetector::new(0.0, 2);
    let response = det.corner_response(&img);

    let corner_response = response.get(15, 15);
    let edge_response = response.get(20, 15);
    assert!(
        corner_response > edge_response,
        "corner response ({corner_response}) should exceed edge response ({edge_response})"
    );
}

#[test]
fn nms_keeps_best_per_cell() {
    let img = make_checkerboard(100, 10, 20, 230);
    let det = ShiTomasiDetector::new(1e5, 2);
    let raw = det.detect(&img);

    let nms = OccupancyNms::new(12);
    let suppressed = nms.suppress(&raw, img.width(), img.height());

    assert!(suppressed.len() <= raw.len());
    assert!(!suppressed.is_empty());
}
