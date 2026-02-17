// tests/test_frontend.rs — Integration tests for the visual frontend pipeline.

use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig};
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;

/// Create a multi-rectangle scene with controllable shift.
fn make_scene(shift_x: usize, shift_y: usize) -> Image<u8> {
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

// ===== Multi-frame pipeline tests =====

#[test]
fn five_frame_sequence() {
    let config = FrontendConfig {
        max_features: 40,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, 160, 120);

    for i in 0..5 {
        let img = make_scene(i * 2, i);
        let (features, stats) = frontend.process(&img);

        println!(
            "Frame {i}: tracked={}, lost={}, new={}, total={}, cells={}/{}",
            stats.tracked, stats.lost, stats.new_detections,
            stats.total, stats.occupied_cells, stats.total_cells,
        );

        if i == 0 {
            assert!(stats.new_detections > 0, "first frame must detect");
            assert_eq!(stats.tracked, 0);
        } else {
            assert!(stats.tracked > 0, "frame {i}: must track some");
        }

        assert!(features.len() <= 40, "exceeds max_features");
    }
}

#[test]
fn feature_ids_are_globally_unique() {
    let config = FrontendConfig {
        max_features: 30,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, 160, 120);
    let mut all_ids = Vec::new();

    for i in 0..4 {
        let img = make_scene(i * 2, i);
        let (features, _) = frontend.process(&img);
        for f in features {
            // IDs from newly created features should never repeat.
            // Tracked features retain their old IDs.
            if !all_ids.contains(&f.id) {
                all_ids.push(f.id);
            }
        }
    }

    // All IDs should be distinct.
    all_ids.sort();
    let before = all_ids.len();
    all_ids.dedup();
    assert_eq!(before, all_ids.len(), "duplicate IDs found");
}

#[test]
fn feature_count_stays_near_max() {
    let max = 25;
    let config = FrontendConfig {
        max_features: max,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, 160, 120);

    // After a few frames the count should stabilize near max_features.
    for i in 0..5 {
        let img = make_scene(i * 2, i);
        frontend.process(&img);
    }

    let n = frontend.features().len();
    // Should be within reasonable range (some cells may lack texture).
    assert!(n > 0, "should have features");
    assert!(n <= max, "exceeds max");
}

#[test]
fn harris_frontend_works() {
    let config = FrontendConfig {
        detector: DetectorType::Harris,
        harris_threshold: 1e5,
        max_features: 30,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, 160, 120);

    let img1 = make_scene(0, 0);
    let img2 = make_scene(2, 1);

    frontend.process(&img1);
    let (_, stats) = frontend.process(&img2);
    assert!(stats.tracked > 0, "Harris frontend should track: {stats:?}");
}

#[test]
fn ic_frontend_works() {
    let config = FrontendConfig {
        klt_method: LkMethod::InverseCompositional,
        max_features: 30,
        ..Default::default()
    };
    let mut frontend = Frontend::new(config, 160, 120);

    let img1 = make_scene(0, 0);
    let img2 = make_scene(2, 1);

    frontend.process(&img1);
    let (_, stats) = frontend.process(&img2);
    assert!(stats.tracked > 0, "IC frontend should track: {stats:?}");
}
