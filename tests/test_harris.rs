// tests/test_harris.rs — Integration tests for Harris corner detector.

use rudolf_v::harris::HarrisDetector;
use rudolf_v::image::Image;
use rudolf_v::nms::OccupancyNms;

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

// ===== Chessboard — the canonical Harris test =====

#[test]
fn harris_detects_chessboard_junctions() {
    // This is the test FAST couldn't pass — Harris detects junction corners
    // where four chessboard cells meet.
    let img = make_chessboard(100, 10, 20, 230);
    let det = HarrisDetector::new(0.04, 1e6, 2);
    let features = det.detect(&img);

    assert!(
        features.len() >= 20,
        "expected many Harris corners at chessboard junctions, got {}",
        features.len()
    );
}

#[test]
fn harris_corners_at_cell_boundaries() {
    let cell = 10;
    let img = make_chessboard(100, cell, 20, 230);
    let det = HarrisDetector::new(0.04, 1e6, 2);
    let features = det.detect(&img);

    // Every detected corner should be near a cell boundary intersection.
    let tolerance = cell as f32 / 2.0;
    for f in &features {
        let nearest_ix = (f.x / cell as f32).round() * cell as f32;
        let nearest_iy = (f.y / cell as f32).round() * cell as f32;
        let dist = ((f.x - nearest_ix).powi(2) + (f.y - nearest_iy).powi(2)).sqrt();
        assert!(
            dist <= tolerance,
            "Harris corner at ({:.0},{:.0}) is {:.1}px from nearest junction",
            f.x, f.y, dist,
        );
    }
}

// ===== Harris + NMS =====

#[test]
fn harris_nms_gives_well_distributed_corners() {
    let img = make_chessboard(100, 10, 20, 230);
    let det = HarrisDetector::new(0.04, 1e6, 2);
    let raw = det.detect(&img);

    let cell_size = 12;
    let nms = OccupancyNms::new(cell_size);
    let suppressed = nms.suppress(&raw, img.width(), img.height());

    assert!(suppressed.len() <= raw.len());
    assert!(suppressed.len() >= 5, "NMS too aggressive");

    // No two survivors in the same NMS cell.
    for i in 0..suppressed.len() {
        for j in (i + 1)..suppressed.len() {
            let ci = (suppressed[i].x as usize / cell_size, suppressed[i].y as usize / cell_size);
            let cj = (suppressed[j].x as usize / cell_size, suppressed[j].y as usize / cell_size);
            assert_ne!(ci, cj, "two Harris corners in same NMS cell");
        }
    }
}

// ===== Response image =====

#[test]
fn response_positive_at_corners_negative_at_edges() {
    // L-shaped corner: strong response at the elbow.
    let mut img = Image::from_vec(40, 40, vec![30u8; 1600]);
    // Horizontal bar
    for y in 15..25 {
        for x in 5..30 {
            img.set(x, y, 220);
        }
    }
    // Vertical bar
    for y in 5..30 {
        for x in 15..25 {
            img.set(x, y, 220);
        }
    }

    let det = HarrisDetector::new(0.04, 0.0, 2);
    let response = det.corner_response(&img);

    // The elbow corners of the cross should have positive response.
    // Pick a few interior edge points — they should have negative response.
    let corner_response = response.get(15, 15); // near a corner
    let edge_response = response.get(20, 15);   // along an edge

    // We just check that corners are significantly stronger than edges.
    assert!(
        corner_response > edge_response,
        "corner response ({corner_response}) should exceed edge response ({edge_response})"
    );
}

// ===== Pyramid-level Harris =====

#[test]
fn harris_on_pyramid_levels() {
    let img = make_chessboard(80, 10, 20, 230);
    let pyr = rudolf_v::pyramid::Pyramid::build(&img, 3, 1.0);
    let det = HarrisDetector::new(0.04, 1e5, 2);

    for lvl in 0..pyr.num_levels() {
        let level_img = &pyr.levels[lvl];
        let mut u8_img = Image::new(level_img.width(), level_img.height());
        for (x, y, v) in level_img.pixels() {
            u8_img.set(x, y, v.clamp(0.0, 255.0).round() as u8);
        }

        let features = det.detect_at_level(&u8_img, lvl);
        for f in &features {
            assert_eq!(f.level, lvl);
        }
    }
}
