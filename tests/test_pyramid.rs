// tests/test_pyramid.rs — Integration tests for convolution and pyramid.

use rudolf_v::convolution::{convolve_separable, gaussian_kernel_1d};
use rudolf_v::image::Image;
use rudolf_v::pyramid::Pyramid;

// ===== Convolution =====

#[test]
fn separable_gaussian_preserves_mean() {
    // Blurring should preserve the average intensity of the image
    // (since the kernel sums to 1), at least for interior pixels.
    // With clamp borders, edge pixels pull toward the edge value,
    // but the overall mean should be very close.
    let mut img: Image<u8> = Image::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            img.set(x, y, ((x * 7 + y * 13) % 256) as u8);
        }
    }

    let n = (img.width() * img.height()) as f32;
    let mean_before: f32 = img.pixels().map(|(_, _, v)| v as f32).sum::<f32>() / n;

    let k = gaussian_kernel_1d(2, 1.0);
    let blurred = convolve_separable(&img, &k, &k);

    let mean_after: f32 = blurred.pixels().map(|(_, _, v)| v).sum::<f32>() / n;

    assert!(
        (mean_before - mean_after).abs() < 2.0,
        "mean shifted too much: {mean_before} → {mean_after}"
    );
}

#[test]
fn horizontal_gradient_survives_vertical_blur() {
    // A pure horizontal gradient should be largely unaffected by
    // vertical-only blur (kernel_row = identity, kernel_col = gaussian).
    let mut img = Image::<f32>::new(20, 20);
    for y in 0..20 {
        for x in 0..20 {
            img.set(x, y, x as f32 * 10.0); // gradient along x only
        }
    }

    let identity = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let gauss = gaussian_kernel_1d(2, 1.0);
    let out = convolve_separable(&img, &identity, &gauss);

    // Interior pixels should match original (horizontal structure preserved).
    for y in 3..17 {
        for x in 3..17 {
            assert!(
                (out.get(x, y) - img.get(x, y)).abs() < 1e-3,
                "horizontal gradient damaged at ({x},{y})"
            );
        }
    }
}

// ===== Pyramid =====

#[test]
fn pyramid_from_u8_euroc_size() {
    // EuRoC camera: 752×480. Typical 4-level pyramid.
    let img: Image<u8> = Image::new(752, 480);
    let pyr = Pyramid::build(&img, 4, 1.0);

    assert_eq!(pyr.num_levels(), 4);
    assert_eq!(pyr.level(0).width(), 752);
    assert_eq!(pyr.level(0).height(), 480);
    assert_eq!(pyr.level(1).width(), 376);
    assert_eq!(pyr.level(1).height(), 240);
    assert_eq!(pyr.level(2).width(), 188);
    assert_eq!(pyr.level(2).height(), 120);
    assert_eq!(pyr.level(3).width(), 94);
    assert_eq!(pyr.level(3).height(), 60);
}

#[test]
fn pyramid_gradient_gets_smoother() {
    // Build a sharp-edged image: left half = 0, right half = 255.
    // At each pyramid level, the edge should be smoother (measured by
    // max absolute difference between adjacent pixels along a row).
    let mut img: Image<u8> = Image::new(128, 64);
    for y in 0..64 {
        for x in 0..128 {
            img.set(x, y, if x < 64 { 0 } else { 255 });
        }
    }

    let pyr = Pyramid::build(&img, 4, 1.0);

    let max_adjacent_diff = |img: &Image<f32>| -> f32 {
        let mut max_diff = 0.0f32;
        let mid_y = img.height() / 2;
        for x in 1..img.width() {
            let diff = (img.get(x, mid_y) - img.get(x - 1, mid_y)).abs();
            max_diff = max_diff.max(diff);
        }
        max_diff
    };

    let mut prev_diff = max_adjacent_diff(&pyr.levels[0]);
    for lvl in 1..pyr.num_levels() {
        let diff = max_adjacent_diff(&pyr.levels[lvl]);
        assert!(
            diff <= prev_diff + 1.0,
            "edge got sharper from level {} to {lvl}: {prev_diff} → {diff}",
            lvl - 1
        );
        prev_diff = diff;
    }
}

#[test]
fn pyramid_level_accessor() {
    let img: Image<u8> = Image::new(16, 16);
    let pyr = Pyramid::build(&img, 3, 1.0);
    // .level() should return the same as direct indexing.
    assert_eq!(pyr.level(0).width(), pyr.levels[0].width());
    assert_eq!(pyr.level(2).width(), pyr.levels[2].width());
}
