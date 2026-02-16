// tests/test_image.rs — Integration tests for Image<T>, ImageView, and conversions.
//
// These run with `cargo test --test test_image`.
// Unlike unit tests (inside #[cfg(test)] mod tests {}), integration tests
// live in tests/ and can only access the crate's public API — a good check
// that the public surface is usable.

use rudolf_v::image::{Image, interpolate_bilinear};
use rudolf_v::convert;

// ===== Image construction & basic access =====

#[test]
fn image_new_zero_initialized() {
    let img: Image<u8> = Image::new(100, 50);
    assert_eq!(img.width(), 100);
    assert_eq!(img.height(), 50);
    assert_eq!(img.get(0, 0), 0);
    assert_eq!(img.get(99, 49), 0);
}

#[test]
fn image_set_get_consistency() {
    let mut img: Image<u8> = Image::new(10, 10);
    // Write a checkerboard pattern.
    for y in 0..10 {
        for x in 0..10 {
            let val = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
            img.set(x, y, val);
        }
    }
    // Verify the pattern.
    for y in 0..10 {
        for x in 0..10 {
            let expected = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
            assert_eq!(img.get(x, y), expected, "mismatch at ({x}, {y})");
        }
    }
}

#[test]
fn image_from_vec_layout() {
    // 3×2 image, row-major:
    //  [10, 20, 30]
    //  [40, 50, 60]
    let data = vec![10u8, 20, 30, 40, 50, 60];
    let img = Image::from_vec(3, 2, data);
    assert_eq!(img.get(0, 0), 10);
    assert_eq!(img.get(2, 0), 30);
    assert_eq!(img.get(0, 1), 40);
    assert_eq!(img.get(2, 1), 60);
}

// ===== Stride =====

#[test]
fn image_stride_does_not_affect_pixel_access() {
    // Width 3, stride 8 — lots of padding.
    let mut img: Image<u8> = Image::new_with_stride(3, 2, 8);
    img.set(0, 0, 1);
    img.set(2, 0, 2);
    img.set(0, 1, 3);
    img.set(2, 1, 4);

    assert_eq!(img.get(0, 0), 1);
    assert_eq!(img.get(2, 0), 2);
    assert_eq!(img.get(0, 1), 3);
    assert_eq!(img.get(2, 1), 4);

    // Row slices should only be `width` long, not `stride` long.
    assert_eq!(img.row(0).len(), 3);
    assert_eq!(img.row(0), &[1, 0, 2]);
}

// ===== Sub-image views =====

#[test]
fn sub_image_coordinates() {
    // 5×5 image with pixel value = x * 10 + y
    let mut img: Image<u8> = Image::new(5, 5);
    for y in 0..5u8 {
        for x in 0..5u8 {
            img.set(x as usize, y as usize, x * 10 + y);
        }
    }

    // 3×3 view starting at (1, 2)
    let view = img.sub_image(1, 2, 3, 3);
    assert_eq!(view.width(), 3);
    assert_eq!(view.height(), 3);

    // view(0,0) should be img(1,2) = 1*10 + 2 = 12
    assert_eq!(view.get(0, 0), 12);
    // view(2,2) should be img(3,4) = 3*10 + 4 = 34
    assert_eq!(view.get(2, 2), 34);
}

#[test]
fn sub_image_with_stride() {
    // Verify sub_image works correctly when the parent has stride != width.
    let mut img: Image<u8> = Image::new_with_stride(4, 4, 8);
    for y in 0..4 {
        for x in 0..4 {
            img.set(x, y, (y * 4 + x) as u8);
        }
    }
    // Image:
    //  0  1  2  3 [pad pad pad pad]
    //  4  5  6  7 [pad pad pad pad]
    //  8  9 10 11 [pad pad pad pad]
    // 12 13 14 15 [pad pad pad pad]

    let view = img.sub_image(1, 1, 2, 2);
    assert_eq!(view.get(0, 0), 5);   // img(1,1)
    assert_eq!(view.get(1, 0), 6);   // img(2,1)
    assert_eq!(view.get(0, 1), 9);   // img(1,2)
    assert_eq!(view.get(1, 1), 10);  // img(2,2)
}

#[test]
fn sub_image_to_owned_decoupled() {
    let data: Vec<u8> = (0..16).collect();
    let img = Image::from_vec(4, 4, data);
    let view = img.sub_image(1, 1, 2, 2);
    let owned = view.to_owned_image();

    // Owned image should have its own independent data.
    assert_eq!(owned.width(), 2);
    assert_eq!(owned.height(), 2);
    assert_eq!(owned.stride(), 2); // No padding in owned copy
    assert_eq!(owned.get(0, 0), 5);
    assert_eq!(owned.get(1, 1), 10);
}

// ===== Iterator =====

#[test]
fn pixels_iterator_count() {
    let img: Image<u8> = Image::new(7, 3);
    let count = img.pixels().count();
    assert_eq!(count, 7 * 3);
}

#[test]
fn pixels_iterator_with_stride() {
    // Stride padding should NOT appear in the iterator.
    let mut img: Image<u8> = Image::new_with_stride(2, 2, 4);
    img.set(0, 0, 1);
    img.set(1, 0, 2);
    img.set(0, 1, 3);
    img.set(1, 1, 4);

    let pixels: Vec<_> = img.pixels().collect();
    assert_eq!(pixels.len(), 4); // NOT 8 (stride * height)
    assert_eq!(pixels[0], (0, 0, 1));
    assert_eq!(pixels[1], (1, 0, 2));
    assert_eq!(pixels[2], (0, 1, 3));
    assert_eq!(pixels[3], (1, 1, 4));
}

// ===== Conversions =====

#[test]
fn normalized_roundtrip_preserves_extremes() {
    let data = vec![0u8, 1, 127, 128, 254, 255];
    let img = Image::from_vec(6, 1, data.clone());
    let f = convert::u8_to_f32_normalized(&img);
    let back = convert::f32_normalized_to_u8(&f);
    for i in 0..6 {
        assert_eq!(
            back.get(i, 0),
            data[i],
            "roundtrip mismatch at pixel {i}"
        );
    }
}

#[test]
fn raw_conversion_preserves_values() {
    let data = vec![0u8, 42, 128, 255];
    let img = Image::from_vec(2, 2, data.clone());
    let f = convert::u8_to_f32_raw(&img);
    for y in 0..2 {
        for x in 0..2 {
            let expected = data[y * 2 + x] as f32;
            assert!(
                (f.get(x, y) - expected).abs() < 1e-6,
                "raw conversion mismatch at ({x}, {y})"
            );
        }
    }
}

// ===== Bilinear interpolation =====

#[test]
fn bilinear_exact_at_corners() {
    // 3×3 image with a gradient: value = x + y * 10
    let mut img: Image<f32> = Image::new(3, 3);
    for y in 0..3 {
        for x in 0..3 {
            img.set(x, y, x as f32 + y as f32 * 10.0);
        }
    }
    // At integer coords, should return exact values.
    assert!((interpolate_bilinear(&img, 0.0, 0.0) - 0.0).abs() < 1e-6);
    assert!((interpolate_bilinear(&img, 1.0, 0.0) - 1.0).abs() < 1e-6);
    assert!((interpolate_bilinear(&img, 0.0, 1.0) - 10.0).abs() < 1e-6);
    assert!((interpolate_bilinear(&img, 1.0, 1.0) - 11.0).abs() < 1e-6);
}

#[test]
fn bilinear_linear_gradient() {
    // On a linear gradient, bilinear interpolation should be exact.
    let mut img: Image<f32> = Image::new(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            img.set(x, y, x as f32 * 3.0 + y as f32 * 7.0);
        }
    }
    // Test several sub-pixel points.
    let test_points = [(0.5, 0.5), (2.3, 4.7), (7.9, 1.1), (0.0, 8.0)];
    for (px, py) in test_points {
        let expected = px * 3.0 + py * 7.0;
        let actual = interpolate_bilinear(&img, px, py);
        assert!(
            (actual - expected).abs() < 1e-4,
            "bilinear({px}, {py}): expected {expected}, got {actual}"
        );
    }
}

// ===== Clone =====

#[test]
fn clone_is_independent() {
    let mut img: Image<u8> = Image::new(4, 4);
    img.set(0, 0, 42);
    let img2 = img.clone();
    img.set(0, 0, 99);
    // Clone should not be affected by mutation of original.
    assert_eq!(img2.get(0, 0), 42);
    assert_eq!(img.get(0, 0), 99);
}

// ===== Edge cases =====

#[test]
fn empty_dimension_image() {
    let img: Image<u8> = Image::new(0, 0);
    assert_eq!(img.width(), 0);
    assert_eq!(img.height(), 0);
    assert_eq!(img.pixels().count(), 0);
}

#[test]
fn single_pixel_image() {
    let mut img: Image<u8> = Image::new(1, 1);
    img.set(0, 0, 123);
    assert_eq!(img.get(0, 0), 123);
    assert_eq!(img.row(0), &[123]);

    let view = img.sub_image(0, 0, 1, 1);
    assert_eq!(view.get(0, 0), 123);
}
