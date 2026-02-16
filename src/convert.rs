// convert.rs — Pixel type conversions between Image<T> types.
//
// This module provides conversions between different pixel types:
//   u8  → f32 (normalized to [0, 1])
//   f32 → u8  (denormalized from [0, 1] to [0, 255])
//   u8  → f32 (raw, preserving integer values as floats — for algorithms)
//
// Note: The Pixel trait's to_f32/from_f32 methods do RAW conversion
// (u8 42 → f32 42.0). The functions here do NORMALIZED conversion
// (u8 42 → f32 0.1647...) which is sometimes needed for display or
// certain algorithm formulations.

use crate::image::{Image, Pixel};

/// Convert an Image<u8> to Image<f32> with normalized values in [0.0, 1.0].
/// u8 0 → 0.0, u8 255 → 1.0.
pub fn u8_to_f32_normalized(src: &Image<u8>) -> Image<f32> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, src.get(x, y) as f32 / 255.0);
        }
    }
    dst
}

/// Convert an Image<f32> (assumed [0.0, 1.0]) to Image<u8>.
/// Values are clamped to [0, 255] and rounded.
pub fn f32_normalized_to_u8(src: &Image<f32>) -> Image<u8> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            let v = (src.get(x, y) * 255.0).clamp(0.0, 255.0).round() as u8;
            dst.set(x, y, v);
        }
    }
    dst
}

/// Convert an Image<u8> to Image<f32> preserving raw values.
/// u8 42 → f32 42.0. This is what most CV algorithms expect.
pub fn u8_to_f32_raw(src: &Image<u8>) -> Image<f32> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, src.get(x, y) as f32);
        }
    }
    dst
}

/// Convert an Image<f32> with raw intensity values to Image<u8>.
/// Clamps to [0, 255] and rounds.
pub fn f32_raw_to_u8(src: &Image<f32>) -> Image<u8> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, u8::from_f32(src.get(x, y)));
        }
    }
    dst
}

/// Generic conversion between any two Pixel types via f32 as intermediate.
/// Goes through the raw to_f32 / from_f32 path defined on each Pixel impl.
///
/// NOTE ON GENERICS:
/// This function is generic over TWO type parameters, S (source pixel) and
/// D (destination pixel). Both must implement the Pixel trait. The compiler
/// generates specialized code for each concrete (S, D) pair you use — this
/// is monomorphization, and it means zero runtime overhead vs. hand-written
/// u8→f32 or f32→u8 functions.
pub fn convert_image<S: Pixel, D: Pixel>(src: &Image<S>) -> Image<D> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, D::from_f32(src.get(x, y).to_f32()));
        }
    }
    dst
}

/// Grayscale from RGB stored as 3 separate channel images.
/// Uses ITU-R BT.601 luma coefficients: Y = 0.299*R + 0.587*G + 0.114*B
///
/// If you later want to support interleaved RGB, you'd work with a
/// hypothetical `Image<Rgb<u8>>` or similar — but for vilib's VIO pipeline,
/// input is already grayscale.
pub fn rgb_to_grayscale(r: &Image<f32>, g: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    assert_eq!(r.width(), g.width());
    assert_eq!(r.width(), b.width());
    assert_eq!(r.height(), g.height());
    assert_eq!(r.height(), b.height());

    let mut gray = Image::new(r.width(), r.height());
    for y in 0..r.height() {
        for x in 0..r.width() {
            let luma = 0.299 * r.get(x, y) + 0.587 * g.get(x, y) + 0.114 * b.get(x, y);
            gray.set(x, y, luma);
        }
    }
    gray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_to_f32_normalized_roundtrip() {
        let data: Vec<u8> = vec![0, 128, 255, 42];
        let img = Image::from_vec(2, 2, data);

        let f = u8_to_f32_normalized(&img);
        assert!((f.get(0, 0) - 0.0).abs() < 1e-6);
        assert!((f.get(1, 0) - 128.0 / 255.0).abs() < 1e-6);
        assert!((f.get(0, 1) - 1.0).abs() < 1e-6);

        // Round-trip back to u8.
        let back = f32_normalized_to_u8(&f);
        assert_eq!(back.get(0, 0), 0);
        assert_eq!(back.get(1, 0), 128);
        assert_eq!(back.get(0, 1), 255);
        assert_eq!(back.get(1, 1), 42);
    }

    #[test]
    fn test_u8_to_f32_raw_roundtrip() {
        let data: Vec<u8> = vec![0, 100, 200, 255];
        let img = Image::from_vec(2, 2, data);

        let f = u8_to_f32_raw(&img);
        assert!((f.get(0, 0) - 0.0).abs() < 1e-6);
        assert!((f.get(1, 0) - 100.0).abs() < 1e-6);
        assert!((f.get(0, 1) - 200.0).abs() < 1e-6);
        assert!((f.get(1, 1) - 255.0).abs() < 1e-6);

        let back = f32_raw_to_u8(&f);
        assert_eq!(back.get(0, 0), 0);
        assert_eq!(back.get(1, 0), 100);
        assert_eq!(back.get(0, 1), 200);
        assert_eq!(back.get(1, 1), 255);
    }

    #[test]
    fn test_f32_to_u8_clamping() {
        let data: Vec<f32> = vec![-10.0, 0.0, 300.0, 127.6];
        let img = Image::from_vec(2, 2, data);
        let out = f32_raw_to_u8(&img);
        assert_eq!(out.get(0, 0), 0);   // clamped from -10
        assert_eq!(out.get(1, 0), 0);
        assert_eq!(out.get(0, 1), 255); // clamped from 300
        assert_eq!(out.get(1, 1), 128); // 127.6 rounds to 128
    }

    #[test]
    fn test_generic_convert() {
        let data: Vec<u8> = vec![0, 128, 255, 42];
        let img = Image::from_vec(2, 2, data);

        // u8 → f32 via generic (raw path)
        let f: Image<f32> = convert_image(&img);
        assert!((f.get(0, 0) - 0.0).abs() < 1e-6);
        assert!((f.get(1, 0) - 128.0).abs() < 1e-6);

        // f32 → u8 via generic (raw path)
        let back: Image<u8> = convert_image(&f);
        assert_eq!(back.get(0, 0), 0);
        assert_eq!(back.get(1, 0), 128);
    }

    #[test]
    fn test_rgb_to_grayscale() {
        // Pure red → luma should be 0.299
        let r = Image::from_vec(1, 1, vec![1.0f32]);
        let g = Image::from_vec(1, 1, vec![0.0f32]);
        let b = Image::from_vec(1, 1, vec![0.0f32]);
        let gray = rgb_to_grayscale(&r, &g, &b);
        assert!((gray.get(0, 0) - 0.299).abs() < 1e-6);

        // Equal channels → luma = 0.299 + 0.587 + 0.114 = 1.0
        let r = Image::from_vec(1, 1, vec![1.0f32]);
        let g = Image::from_vec(1, 1, vec![1.0f32]);
        let b = Image::from_vec(1, 1, vec![1.0f32]);
        let gray = rgb_to_grayscale(&r, &g, &b);
        assert!((gray.get(0, 0) - 1.0).abs() < 1e-6);
    }
}
