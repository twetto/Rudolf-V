// tests/test_klt.rs — Integration tests for the KLT pyramidal tracker.

use std::fs;
use std::path::Path;

use rudolf_v::fast::{FastDetector, Feature};
use rudolf_v::image::{Image, interpolate_bilinear};
use rudolf_v::klt::{KltTracker, LkMethod, TrackStatus};
use rudolf_v::klt_reference::{ReferenceKltTracker, ReferenceKltWarp};
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
        descriptor: 0,
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
        Feature {
            x: 36.0,
            y: 36.0,
            score: 100.0,
            level: 0,
            id: 42,
            descriptor: 0,
        },
        Feature {
            x: 71.0,
            y: 66.0,
            score: 80.0,
            level: 0,
            id: 99,
            descriptor: 0,
        },
    ];

    let results = tracker.track(&pyr, &pyr, &features);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].feature.id, 42);
    assert_eq!(results[1].feature.id, 99);
    assert_eq!(results[0].feature.level, 0);
}

fn make_affine_texture(width: usize, height: usize) -> Image<f32> {
    let mut img = Image::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let xf = x as f32;
            let yf = y as f32;
            let value = 95.0
                + 34.0 * (0.083 * xf + 0.017 * yf).sin()
                + 29.0 * (0.031 * xf - 0.071 * yf).cos()
                + 18.0 * (0.113 * xf + 0.097 * yf).sin()
                + ((x * 17 + y * 29 + x * y) % 41) as f32;
            img.set(x, y, value.clamp(0.0, 255.0));
        }
    }
    img
}

fn warp_affine_image(
    src: &Image<f32>,
    a00: f32,
    a01: f32,
    a10: f32,
    a11: f32,
    tx: f32,
    ty: f32,
) -> Image<f32> {
    let det = a00 * a11 - a01 * a10;
    assert!(det.abs() > 1e-6);
    let ia00 = a11 / det;
    let ia01 = -a01 / det;
    let ia10 = -a10 / det;
    let ia11 = a00 / det;

    let mut dst = Image::new(src.width(), src.height());
    for y in 0..dst.height() {
        for x in 0..dst.width() {
            let qx = x as f32 - tx;
            let qy = y as f32 - ty;
            let sx = ia00 * qx + ia01 * qy;
            let sy = ia10 * qx + ia11 * qy;
            dst.set(x, y, interpolate_bilinear(src, sx, sy));
        }
    }
    dst
}

#[test]
fn reference_affine_klt_matches_known_affine_optical_flow() {
    let img0 = make_affine_texture(128, 128);

    let a00 = 1.015;
    let a01 = -0.045;
    let a10 = 0.035;
    let a11 = 0.985;
    let tx = 3.4;
    let ty = -2.1;
    let img1 = warp_affine_image(&img0, a00, a01, a10, a11, tx, ty);

    let pyr0 = Pyramid::build(&img0, 3, 1.0);
    let pyr1 = Pyramid::build(&img1, 3, 1.0);
    let tracker = ReferenceKltTracker::new(10, 40, 0.001, 3, true, ReferenceKltWarp::Affine);

    let features = [
        Feature {
            x: 32.0,
            y: 32.0,
            score: 100.0,
            level: 0,
            id: 1,
            descriptor: 0,
        },
        Feature {
            x: 82.0,
            y: 34.0,
            score: 100.0,
            level: 0,
            id: 2,
            descriptor: 0,
        },
        Feature {
            x: 39.0,
            y: 88.0,
            score: 100.0,
            level: 0,
            id: 3,
            descriptor: 0,
        },
        Feature {
            x: 91.0,
            y: 91.0,
            score: 100.0,
            level: 0,
            id: 4,
            descriptor: 0,
        },
    ];

    for feature in features {
        let reference = tracker
            .make_track(&feature, &pyr0)
            .expect("synthetic feature should have a valid reference patch");
        let tracked = tracker.track(&reference, &pyr1, &feature);

        assert_eq!(
            tracked.status,
            TrackStatus::Tracked,
            "feature {} should track",
            feature.id
        );

        let expected_x = a00 * feature.x + a01 * feature.y + tx;
        let expected_y = a10 * feature.x + a11 * feature.y + ty;
        let err = ((tracked.feature.x - expected_x).powi(2)
            + (tracked.feature.y - expected_y).powi(2))
        .sqrt();

        assert!(
            err < 0.35,
            "feature {} flow mismatch: got ({:.3}, {:.3}), expected ({:.3}, {:.3}), err={:.3}",
            feature.id,
            tracked.feature.x,
            tracked.feature.y,
            expected_x,
            expected_y,
            err
        );
    }
}

#[test]
fn previous_frame_affine_klt_matches_known_affine_optical_flow() {
    let img0 = make_affine_texture(128, 128);

    let a00 = 1.015;
    let a01 = -0.045;
    let a10 = 0.035;
    let a11 = 0.985;
    let tx = 3.4;
    let ty = -2.1;
    let img1 = warp_affine_image(&img0, a00, a01, a10, a11, tx, ty);

    let pyr0 = Pyramid::build(&img0, 3, 1.0);
    let pyr1 = Pyramid::build(&img1, 3, 1.0);
    let tracker = KltTracker::with_method(10, 40, 0.001, 3, LkMethod::InverseCompositionalAffine);

    let features = [
        Feature {
            x: 32.0,
            y: 32.0,
            score: 100.0,
            level: 0,
            id: 1,
            descriptor: 0,
        },
        Feature {
            x: 82.0,
            y: 34.0,
            score: 100.0,
            level: 0,
            id: 2,
            descriptor: 0,
        },
        Feature {
            x: 39.0,
            y: 88.0,
            score: 100.0,
            level: 0,
            id: 3,
            descriptor: 0,
        },
        Feature {
            x: 91.0,
            y: 91.0,
            score: 100.0,
            level: 0,
            id: 4,
            descriptor: 0,
        },
    ];

    let tracked = tracker.track(&pyr0, &pyr1, &features);
    for (feature, result) in features.iter().zip(&tracked) {
        assert_eq!(
            result.status,
            TrackStatus::Tracked,
            "feature {} should track",
            feature.id
        );

        let expected_x = a00 * feature.x + a01 * feature.y + tx;
        let expected_y = a10 * feature.x + a11 * feature.y + ty;
        let err = ((result.feature.x - expected_x).powi(2)
            + (result.feature.y - expected_y).powi(2))
        .sqrt();

        assert!(
            err < 0.35,
            "feature {} flow mismatch: got ({:.3}, {:.3}), expected ({:.3}, {:.3}), err={:.3}",
            feature.id,
            result.feature.x,
            result.feature.y,
            expected_x,
            expected_y,
            err
        );
    }
}

fn affine_patch_sample(
    image: &Image<f32>,
    feature: &Feature,
    warp: [f32; 6],
    radius: usize,
) -> Image<f32> {
    let side = 2 * radius + 1;
    let mut patch = Image::new(side, side);
    for py in 0..side {
        let oy = py as f32 - radius as f32;
        for px in 0..side {
            let ox = px as f32 - radius as f32;
            let x = feature.x + warp[4] + warp[0] * ox + warp[1] * oy;
            let y = feature.y + warp[5] + warp[2] * ox + warp[3] * oy;
            patch.set(px, py, interpolate_bilinear(image, x, y));
        }
    }
    patch
}

fn write_pgm(path: &Path, image: &Image<f32>, scale: usize) {
    let width = image.width() * scale;
    let height = image.height() * scale;
    let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();

    for y in 0..height {
        for x in 0..width {
            let src_x = x / scale;
            let src_y = y / scale;
            bytes.push(image.get(src_x, src_y).clamp(0.0, 255.0).round() as u8);
        }
    }

    fs::write(path, bytes).expect("write diagnostic pgm");
}

fn write_pgm_with_patch_box(
    path: &Path,
    image: &Image<f32>,
    center_x: f32,
    center_y: f32,
    radius: usize,
    scale: usize,
) {
    let width = image.width() * scale;
    let height = image.height() * scale;
    let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();

    let cx = (center_x * scale as f32).round() as isize;
    let cy = (center_y * scale as f32).round() as isize;
    let r = (radius * scale) as isize;

    for y in 0..height {
        for x in 0..width {
            let xi = x as isize;
            let yi = y as isize;
            let on_box = (xi >= cx - r && xi <= cx + r && (yi == cy - r || yi == cy + r))
                || (yi >= cy - r && yi <= cy + r && (xi == cx - r || xi == cx + r));
            let on_cross = (xi == cx && yi >= cy - 4 && yi <= cy + 4)
                || (yi == cy && xi >= cx - 4 && xi <= cx + 4);
            if on_box || on_cross {
                bytes.push(255);
            } else {
                let src_x = x / scale;
                let src_y = y / scale;
                bytes.push(image.get(src_x, src_y).clamp(0.0, 220.0).round() as u8);
            }
        }
    }

    fs::write(path, bytes).expect("write marked diagnostic pgm");
}

fn write_patch_triplet(
    path: &Path,
    reference: &Image<f32>,
    groundtruth: &Image<f32>,
    estimated: &Image<f32>,
    scale: usize,
) {
    let gap = 2;
    let width = reference.width() * 3 + gap * 2;
    let height = reference.height();
    let mut montage = Image::from_vec(width, height, vec![0.0f32; width * height]);

    for y in 0..height {
        for x in 0..reference.width() {
            montage.set(x, y, reference.get(x, y));
            montage.set(x + reference.width() + gap, y, groundtruth.get(x, y));
            montage.set(x + 2 * (reference.width() + gap), y, estimated.get(x, y));
        }
    }

    write_pgm(path, &montage, scale);
}

#[test]
#[ignore = "writes affine KLT diagnostic images under target/affine_klt_diagnostics"]
fn reference_affine_klt_writes_warp_debug_images() {
    let img0 = make_affine_texture(128, 128);

    let a00 = 1.015;
    let a01 = -0.045;
    let a10 = 0.035;
    let a11 = 0.985;
    let tx = 3.4;
    let ty = -2.1;
    let img1 = warp_affine_image(&img0, a00, a01, a10, a11, tx, ty);

    let pyr0 = Pyramid::build(&img0, 3, 1.0);
    let pyr1 = Pyramid::build(&img1, 3, 1.0);
    let tracker = ReferenceKltTracker::new(10, 40, 0.001, 3, true, ReferenceKltWarp::Affine);
    let out_dir = Path::new("target").join("affine_klt_diagnostics");
    fs::create_dir_all(&out_dir).expect("create diagnostic output dir");

    write_pgm(&out_dir.join("reference_full.pgm"), &img0, 1);
    write_pgm(&out_dir.join("groundtruth_affine_full.pgm"), &img1, 1);

    let features = [
        Feature {
            x: 32.0,
            y: 32.0,
            score: 100.0,
            level: 0,
            id: 1,
            descriptor: 0,
        },
        Feature {
            x: 82.0,
            y: 34.0,
            score: 100.0,
            level: 0,
            id: 2,
            descriptor: 0,
        },
        Feature {
            x: 39.0,
            y: 88.0,
            score: 100.0,
            level: 0,
            id: 3,
            descriptor: 0,
        },
        Feature {
            x: 91.0,
            y: 91.0,
            score: 100.0,
            level: 0,
            id: 4,
            descriptor: 0,
        },
    ];

    let mut report = String::from(
        "feature,component,estimated,expected,abs_error\n\
         # components are a00,a01,a10,a11,center_dx,center_dy\n",
    );

    for feature in features {
        let reference_track = tracker
            .make_track(&feature, &pyr0)
            .expect("synthetic feature should have a valid reference patch");
        let estimated_warp = tracker
            .track_affine_warp(&reference_track, &pyr1, &feature)
            .expect("synthetic affine warp should be estimated");

        let expected_x = a00 * feature.x + a01 * feature.y + tx;
        let expected_y = a10 * feature.x + a11 * feature.y + ty;
        let groundtruth_warp = [
            a00,
            a01,
            a10,
            a11,
            expected_x - feature.x,
            expected_y - feature.y,
        ];

        let reference_patch =
            affine_patch_sample(&img0, &feature, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10);
        let groundtruth_patch = affine_patch_sample(&img1, &feature, groundtruth_warp, 10);
        let estimated_patch = affine_patch_sample(&img1, &feature, estimated_warp, 10);
        let estimated_global_tx = feature.x + estimated_warp[4]
            - estimated_warp[0] * feature.x
            - estimated_warp[1] * feature.y;
        let estimated_global_ty = feature.y + estimated_warp[5]
            - estimated_warp[2] * feature.x
            - estimated_warp[3] * feature.y;
        let estimated_full = warp_affine_image(
            &img0,
            estimated_warp[0],
            estimated_warp[1],
            estimated_warp[2],
            estimated_warp[3],
            estimated_global_tx,
            estimated_global_ty,
        );

        write_patch_triplet(
            &out_dir.join(format!("feature_{}_ref_gt_estimated.pgm", feature.id)),
            &reference_patch,
            &groundtruth_patch,
            &estimated_patch,
            10,
        );
        write_pgm(
            &out_dir.join(format!("feature_{}_estimated_affine_full.pgm", feature.id)),
            &estimated_full,
            1,
        );
        write_pgm_with_patch_box(
            &out_dir.join(format!("feature_{}_reference_full_marked.pgm", feature.id)),
            &img0,
            feature.x,
            feature.y,
            10,
            2,
        );
        write_pgm_with_patch_box(
            &out_dir.join(format!(
                "feature_{}_groundtruth_full_marked.pgm",
                feature.id
            )),
            &img1,
            expected_x,
            expected_y,
            10,
            2,
        );
        write_pgm_with_patch_box(
            &out_dir.join(format!("feature_{}_estimated_full_marked.pgm", feature.id)),
            &estimated_full,
            feature.x + estimated_warp[4],
            feature.y + estimated_warp[5],
            10,
            2,
        );
        write_pgm(
            &out_dir.join(format!("feature_{}_reference_patch.pgm", feature.id)),
            &reference_patch,
            10,
        );
        write_pgm(
            &out_dir.join(format!("feature_{}_groundtruth_patch.pgm", feature.id)),
            &groundtruth_patch,
            10,
        );
        write_pgm(
            &out_dir.join(format!("feature_{}_estimated_patch.pgm", feature.id)),
            &estimated_patch,
            10,
        );

        for (idx, (estimated, expected)) in estimated_warp.iter().zip(groundtruth_warp).enumerate()
        {
            report.push_str(&format!(
                "{},{},{:.6},{:.6},{:.6}\n",
                feature.id,
                idx,
                estimated,
                expected,
                (estimated - expected).abs()
            ));
        }
    }

    fs::write(out_dir.join("affine_components.csv"), report).expect("write component report");
    println!("wrote {}", out_dir.display());
}
