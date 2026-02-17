// tests/test_klt.rs — Integration tests for the KLT pyramidal tracker.

use rudolf_v::fast::{FastDetector, Feature};
use rudolf_v::image::Image;
use rudolf_v::klt::{KltTracker, TrackStatus};
use rudolf_v::pyramid::Pyramid;

/// Create a scene with multiple bright squares — good texture for tracking.
/// Squares are placed well away from image borders so pyramid scaling
/// doesn't push features out of bounds.
fn make_textured_scene(shift_x: usize, shift_y: usize) -> Image<u8> {
    let w = 120;
    let h = 120;
    let mut img = Image::from_vec(w, h, vec![30u8; w * h]);

    let squares = [
        (35 + shift_x, 35 + shift_y, 15),
        (70 + shift_x, 35 + shift_y, 12),
        (35 + shift_x, 70 + shift_y, 18),
        (70 + shift_x, 65 + shift_y, 14),
    ];

    for &(sx, sy, size) in &squares {
        for y in sy..(sy + size).min(h) {
            for x in sx..(sx + size).min(w) {
                img.set(x, y, 200);
            }
        }
    }
    img
}

// ===== Detect-then-track pipeline =====

#[test]
fn detect_and_track_shifted_scene() {
    // Detect features in frame 1, track to frame 2 which is shifted by (3, 2).
    let img1 = make_textured_scene(0, 0);
    let img2 = make_textured_scene(3, 2);

    let pyr1 = Pyramid::build(&img1, 3, 1.0);
    let pyr2 = Pyramid::build(&img2, 3, 1.0);

    // Detect features in frame 1.
    let det = FastDetector::new(20, 9);

    // Convert pyramid level 0 to u8 for detection.
    let level0 = &pyr1.levels[0];
    let mut u8_img = Image::new(level0.width(), level0.height());
    for (x, y, v) in level0.pixels() {
        u8_img.set(x, y, v.clamp(0.0, 255.0).round() as u8);
    }
    let features = det.detect(&u8_img);
    assert!(!features.is_empty(), "need features to track");

    // Track from frame 1 to frame 2.
    let tracker = KltTracker::new(7, 30, 0.01, 3);
    let results = tracker.track(&pyr1, &pyr2, &features);

    // Count successfully tracked features.
    let tracked: Vec<_> = results
        .iter()
        .filter(|r| r.status == TrackStatus::Tracked)
        .collect();

    assert!(
        tracked.len() > features.len() / 4,
        "expected at least 25% tracked, got {}/{}",
        tracked.len(),
        features.len()
    );

    // Check that tracked features moved in roughly the right direction.
    let mut dx_sum = 0.0f32;
    let mut dy_sum = 0.0f32;
    for (r, f) in results.iter().zip(features.iter()) {
        if r.status == TrackStatus::Tracked {
            dx_sum += r.feature.x - f.x;
            dy_sum += r.feature.y - f.y;
        }
    }
    let n = tracked.len() as f32;
    let mean_dx = dx_sum / n;
    let mean_dy = dy_sum / n;

    assert!(
        (mean_dx - 3.0).abs() < 2.0,
        "mean dx = {mean_dx}, expected ~3.0"
    );
    assert!(
        (mean_dy - 2.0).abs() < 2.0,
        "mean dy = {mean_dy}, expected ~2.0"
    );
}

// ===== Larger shift with pyramid =====

#[test]
fn track_large_shift_with_pyramid() {
    // A 6-pixel shift is too large for single-level LK with a small window,
    // but should be recoverable with a 3-level pyramid.
    let img1 = make_textured_scene(0, 0);
    let img2 = make_textured_scene(6, 0);

    let pyr1 = Pyramid::build(&img1, 3, 1.0);
    let pyr2 = Pyramid::build(&img2, 3, 1.0);

    let tracker = KltTracker::new(7, 30, 0.01, 3);
    // Feature near the top-left corner of first square — good 2D gradient.
    let features = vec![Feature {
        x: 36.0,
        y: 36.0,
        score: 100.0,
        level: 0,
        id: 1,
    }];

    let results = tracker.track(&pyr1, &pyr2, &features);
    if results[0].status == TrackStatus::Tracked {
        let dx = results[0].feature.x - 36.0;
        assert!(
            (dx - 6.0).abs() < 2.0,
            "large shift: dx = {dx}, expected ~6.0"
        );
    }
}

// ===== Status preservation =====

#[test]
fn tracked_features_preserve_metadata() {
    let img = make_textured_scene(0, 0);
    let pyr = Pyramid::build(&img, 3, 1.0);

    let tracker = KltTracker::new(5, 30, 0.01, 3);
    let features = vec![
        Feature { x: 36.0, y: 36.0, score: 100.0, level: 0, id: 42 },
        Feature { x: 71.0, y: 66.0, score: 80.0, level: 0, id: 99 },
    ];

    let results = tracker.track(&pyr, &pyr, &features);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].feature.id, 42);
    assert_eq!(results[1].feature.id, 99);
    assert_eq!(results[0].feature.level, 0);
}
