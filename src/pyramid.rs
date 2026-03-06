// pyramid.rs — Gaussian image pyramid (fused blur + downsample).
//
// Mirrors vilib's pyramid_gpu.cu: Gaussian blur then 2× downsample at each
// level. The pyramid is the backbone of multi-scale detection and tracking —
// FAST runs on each level, and KLT does coarse-to-fine refinement.
//
// OPTIMIZATION: Fused blur + downsample with hardcoded [1,4,6,4,1]/16 kernel.
//
// The standard pyramid kernel [1,4,6,4,1]/16 is the same one OpenCV uses in
// pyrDown(). It's a binomial approximation to a Gaussian with sigma ≈ 1.0.
//
// INTEGER ARITHMETIC PATH (build_reuse):
//   Horizontal: u8 input → u16 intermediate (max 255×16 = 4080, fits u16)
//   Vertical:   u16 intermediate → u32 accumulator (max 4080×16 = 65280)
//   Output:     u32 >> 8 → u8 (for next level), u32 * (1/256) → f32 (for pyramid)
//
//   This is 4× less memory bandwidth than the f32 path:
//     f32 path: 752×480×4 = 1.44 MB per read/write pass
//     u8 path:  752×480×1 = 0.36 MB per read/write pass
//
//   When memory-bound (as measured on 8845H), this is the dominant factor.
//   Integer multiply-accumulate also benefits from u8/u16 SIMD (future).
//
// f32 PATH (build):
//   Kept for generic Pixel input (f32 images) and tests. Uses the fused
//   [K0,K1,K2,K1,K0] f32 convolution with AVX2 acceleration.
//
// GPU MAPPING: The fused single-pass structure maps directly to a compute
// shader with output-size dispatch — each thread computes one output pixel
// by gathering from the source texture with hardware sampling.

use crate::image::{Image, Pixel};

// Re-export convolution utilities for other modules that may still need them.
#[allow(unused_imports)]
use crate::convolution::{gaussian_kernel_1d, ConvolveScratch};

/// A Gaussian image pyramid.
pub struct Pyramid {
    /// Pyramid levels, from finest (index 0) to coarsest.
    pub levels: Vec<Image<f32>>,
}

/// Pre-allocated scratch buffers for pyramid construction.
///
/// Contains integer intermediate buffers for the hot `build_reuse` path,
/// plus ping-pong u8 buffers for cascading levels.
pub struct PyramidScratch {
    /// Horizontal-pass intermediate (u16, un-normalized, max 4080).
    /// Sized for the largest level (level 0 dimensions).
    h_buf: Vec<u16>,
    /// Ping-pong u8 buffers for feeding integer pyrdown across levels.
    /// Level 0→1 reads from the original u8 input, writes u8_ping.
    /// Level 1→2 reads u8_ping, writes u8_pong.
    /// Level 2→3 reads u8_pong, writes u8_ping. Etc.
    /// Sized for level 1 output (w/2 × h/2) — the largest output level.
    u8_ping: Vec<u8>,
    u8_pong: Vec<u8>,
}

impl PyramidScratch {
    /// Create scratch buffers for the given image dimensions.
    ///
    /// The `sigma` parameter is accepted for API compatibility but ignored.
    pub fn new(width: usize, height: usize, _sigma: f32) -> Self {
        let level1_size = (width / 2) * (height / 2);
        PyramidScratch {
            h_buf: vec![0u16; width * height],
            u8_ping: vec![0u8; level1_size],
            u8_pong: vec![0u8; level1_size],
        }
    }
}

impl Pyramid {
    /// Build a Gaussian pyramid from an input image (allocating).
    ///
    /// Uses the f32 path. For per-frame use, prefer `build_reuse`.
    pub fn build<T: Pixel>(src: &Image<T>, num_levels: usize, _sigma: f32) -> Self {
        assert!(num_levels >= 1, "pyramid must have at least 1 level");

        let mut levels = Vec::with_capacity(num_levels);

        // Level 0: convert source to f32 (no blur on the original).
        let level0 = to_f32_image(src);
        levels.push(level0);

        // Scratch for horizontal pass — sized for level 0 (largest).
        let mut h_buf = vec![0.0f32; src.width() * src.height()];

        // Each subsequent level: fused blur + 2× downsample.
        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let mut down = Image::new(prev.width() / 2, prev.height() / 2);
            pyrdown_fused(prev, &mut down, &mut h_buf);
            levels.push(down);
        }

        Pyramid { levels }
    }

    /// Build a Gaussian pyramid from a u8 image, reusing buffers.
    ///
    /// Uses integer arithmetic (u8→u16→u32) for 4× less memory bandwidth
    /// than the f32 path. Each level's u8 result feeds the next level via
    /// ping-pong buffers, with the f32 output written to self.levels.
    pub fn build_reuse(
        &mut self,
        src: &Image<u8>,
        num_levels: usize,
        scratch: &mut PyramidScratch,
    ) {
        assert!(num_levels >= 1);

        // Ensure we have enough level slots.
        while self.levels.len() < num_levels {
            self.levels.push(Image::new(1, 1));
        }
        self.levels.truncate(num_levels);

        // Level 0: u8 → f32 (no blur).
        to_f32_image_u8_into(src, &mut self.levels[0]);

        if num_levels < 2 {
            return;
        }

        // Destructure scratch for independent borrows.
        let PyramidScratch { ref mut h_buf, ref mut u8_ping, ref mut u8_pong } = *scratch;

        // Pre-extract raw pointers for ping-pong to avoid borrow conflicts
        // in the loop. Safety: we alternate read/write — when i is odd we
        // write ping and read pong (or src), when i is even vice versa.
        // They never alias within a single pyrdown_int call.
        let ping_ptr = u8_ping.as_mut_ptr();
        let pong_ptr = u8_pong.as_mut_ptr();
        let ping_cap = u8_ping.len();
        let pong_cap = u8_pong.len();

        let mut prev_w = src.width();
        let mut prev_h = src.height();

        for i in 1..num_levels {
            let dw = prev_w / 2;
            let dh = prev_h / 2;

            // Select read source: level 1 reads from original image,
            // subsequent levels read from the previous pyrdown's u8 output.
            let (read_ptr, read_len, read_stride) = if i == 1 {
                (src.as_slice().as_ptr(), src.as_slice().len(), src.stride())
            } else if i % 2 == 0 {
                (ping_ptr as *const u8, prev_w * prev_h, prev_w)
            } else {
                (pong_ptr as *const u8, prev_w * prev_h, prev_w)
            };

            // Select write destination: alternate ping/pong.
            let (write_ptr, write_cap) = if i % 2 == 1 {
                (ping_ptr, ping_cap)
            } else {
                (pong_ptr, pong_cap)
            };

            let write_len = dw * dh;
            debug_assert!(write_len <= write_cap,
                "u8 buffer too small: need {write_len}, have {write_cap}");

            // SAFETY: read and write slices point to different buffers
            // (or to the original src which is immutably borrowed).
            // h_buf is used only within pyrdown_int and doesn't alias either.
            unsafe {
                let read_slice = std::slice::from_raw_parts(read_ptr, read_len);
                let write_slice = std::slice::from_raw_parts_mut(write_ptr, write_len);

                pyrdown_int(
                    read_slice, prev_w, prev_h, read_stride,
                    &mut self.levels[i],
                    write_slice,
                    h_buf,
                );
            }

            prev_w = dw;
            prev_h = dh;
        }
    }

    /// Number of pyramid levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get a reference to a specific level.
    pub fn level(&self, level: usize) -> &Image<f32> {
        &self.levels[level]
    }
}

// =============================================================================
// Integer-arithmetic fused blur + downsample (hot path)
// =============================================================================

/// Integer fused Gaussian blur + 2× downsample for one pyramid level.
///
/// Horizontal: u8 → u16 (multiply-accumulate with integer [1,4,6,4,1])
/// Vertical:   u16 → u32 → f32 output + u8 output
///
/// Max horizontal: 255 × 16 = 4080 (fits u16)
/// Max vertical:   4080 × 16 = 65280 (fits u16, but u32 for clarity)
/// Final division: × (1/256) for f32, >> 8 for u8.
fn pyrdown_int(
    src: &[u8],
    sw: usize,
    sh: usize,
    src_stride: usize,
    dst_f32: &mut Image<f32>,
    dst_u8: &mut [u8],
    h_buf: &mut [u16],
) {
    let dw = sw / 2;
    let dh = sh / 2;
    dst_f32.clear_resize(dw, dh);

    if dw == 0 || dh == 0 {
        return;
    }

    // Ensure h_buf is large enough.
    debug_assert!(h_buf.len() >= sw * sh);

    // ── Phase 1: Horizontal blur (u8 → u16) ──────────────────────────
    // Kernel [1, 4, 6, 4, 1], no division. Max output 4080.

    for y in 0..sh {
        let row_off = y * src_stride;
        let hrow_off = y * sw;

        // Left border: x = 0..min(2, sw)
        for x in 0..2.min(sw) {
            let s = |dx: isize| -> u16 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src[row_off + idx] as u16
            };
            h_buf[hrow_off + x] = s(-2) + 4 * s(-1) + 6 * s(0) + 4 * s(1) + s(2);
        }

        // Interior: x in [2, sw-2) — no clamping needed.
        if sw > 4 {
            unsafe {
                let src_row = src.as_ptr().add(row_off);
                let h_row = h_buf.as_mut_ptr().add(hrow_off);
                for x in 2..sw - 2 {
                    let p = |off: usize| -> u16 { *src_row.add(off) as u16 };
                    *h_row.add(x) =
                          p(x - 2)
                        + 4 * p(x - 1)
                        + 6 * p(x)
                        + 4 * p(x + 1)
                        + p(x + 2);
                }
            }
        }

        // Right border
        let right_start = if sw > 2 { sw - 2 } else { 2.min(sw) };
        for x in right_start..sw {
            let s = |dx: isize| -> u16 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src[row_off + idx] as u16
            };
            h_buf[hrow_off + x] = s(-2) + 4 * s(-1) + 6 * s(0) + 4 * s(1) + s(2);
        }
    }

    // ── Phase 2: Vertical blur + downsample (u16 → u32 → f32/u8) ────
    // Kernel [1, 4, 6, 4, 1]. Accumulate in u32, then:
    //   f32 output = acc × (1/256)
    //   u8 output  = (acc + 128) >> 8  (rounded)

    let dst_f32_stride = dst_f32.stride();
    let dst_f32_slice = dst_f32.as_mut_slice();

    const INV256: f32 = 1.0 / 256.0;

    for oy in 0..dh {
        let sy = oy * 2;

        let clamp_row = |dy: isize| -> usize {
            (sy as isize + dy).clamp(0, (sh - 1) as isize) as usize * sw
        };
        let r0 = clamp_row(-2);
        let r1 = clamp_row(-1);
        let r2 = sy * sw;
        let r3 = clamp_row(1);
        let r4 = clamp_row(2);

        let f32_off = oy * dst_f32_stride;
        let u8_off = oy * dw;

        unsafe {
            for ox in 0..dw {
                let sx = ox * 2;
                let acc =
                      *h_buf.get_unchecked(r0 + sx) as u32
                    + 4 * *h_buf.get_unchecked(r1 + sx) as u32
                    + 6 * *h_buf.get_unchecked(r2 + sx) as u32
                    + 4 * *h_buf.get_unchecked(r3 + sx) as u32
                    + *h_buf.get_unchecked(r4 + sx) as u32;

                *dst_f32_slice.get_unchecked_mut(f32_off + ox) = acc as f32 * INV256;
                *dst_u8.get_unchecked_mut(u8_off + ox) = ((acc + 128) >> 8) as u8;
            }
        }
    }
}

// =============================================================================
// f32 fused blur + downsample (generic path, kept for build() and tests)
// =============================================================================

const K0: f32 = 1.0 / 16.0;
const K1: f32 = 4.0 / 16.0;
const K2: f32 = 6.0 / 16.0;

/// f32 fused Gaussian blur + 2× downsample. Used by `build()`.
fn pyrdown_fused(
    src: &Image<f32>,
    dst: &mut Image<f32>,
    h_buf: &mut Vec<f32>,
) {
    let sw = src.width();
    let sh = src.height();
    let dw = sw / 2;
    let dh = sh / 2;
    dst.clear_resize(dw, dh);

    if dw == 0 || dh == 0 {
        return;
    }

    let h_needed = sw * sh;
    if h_buf.len() < h_needed {
        h_buf.resize(h_needed, 0.0);
    }

    let src_slice = src.as_slice();
    let src_stride = src.stride();

    // Phase 1: Horizontal blur.
    for y in 0..sh {
        let row_off = y * src_stride;
        let hrow_off = y * sw;

        for x in 0..2.min(sw) {
            let sx = |dx: isize| -> f32 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src_slice[row_off + idx]
            };
            h_buf[hrow_off + x] = K0 * sx(-2) + K1 * sx(-1) + K2 * sx(0)
                                 + K1 * sx(1)  + K0 * sx(2);
        }

        if sw > 4 {
            let src_row = &src_slice[row_off..row_off + sw];
            let h_row = &mut h_buf[hrow_off..hrow_off + sw];
            hblur_row_interior(src_row, h_row, sw);
        }

        let right_start = if sw > 2 { sw - 2 } else { 2.min(sw) };
        for x in right_start..sw {
            let sx = |dx: isize| -> f32 {
                let idx = (x as isize + dx).clamp(0, (sw - 1) as isize) as usize;
                src_slice[row_off + idx]
            };
            h_buf[hrow_off + x] = K0 * sx(-2) + K1 * sx(-1) + K2 * sx(0)
                                 + K1 * sx(1)  + K0 * sx(2);
        }
    }

    // Phase 2: Vertical blur + downsample.
    let dst_stride = dst.stride();
    let dst_slice = dst.as_mut_slice();

    for oy in 0..dh {
        let sy = oy * 2;

        let clamp_row = |dy: isize| -> usize {
            (sy as isize + dy).clamp(0, (sh - 1) as isize) as usize * sw
        };
        let r0 = clamp_row(-2);
        let r1 = clamp_row(-1);
        let r2 = sy * sw;
        let r3 = clamp_row(1);
        let r4 = clamp_row(2);

        let out_off = oy * dst_stride;

        unsafe {
            for ox in 0..dw {
                let sx = ox * 2;
                let v = K0 * *h_buf.get_unchecked(r0 + sx)
                      + K1 * *h_buf.get_unchecked(r1 + sx)
                      + K2 * *h_buf.get_unchecked(r2 + sx)
                      + K1 * *h_buf.get_unchecked(r3 + sx)
                      + K0 * *h_buf.get_unchecked(r4 + sx);
                *dst_slice.get_unchecked_mut(out_off + ox) = v;
            }
        }
    }
}

// =============================================================================
// SIMD-accelerated horizontal blur (f32 path only)
// =============================================================================

#[inline]
fn hblur_row_interior(src: &[f32], dst: &mut [f32], width: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { hblur_row_avx2(src, dst, width); }
            return;
        }
    }
    hblur_row_scalar(src, dst, width);
}

#[inline]
fn hblur_row_scalar(src: &[f32], dst: &mut [f32], width: usize) {
    unsafe {
        for x in 2..width - 2 {
            *dst.get_unchecked_mut(x) =
                  K0 * *src.get_unchecked(x - 2)
                + K1 * *src.get_unchecked(x - 1)
                + K2 * *src.get_unchecked(x)
                + K1 * *src.get_unchecked(x + 1)
                + K0 * *src.get_unchecked(x + 2);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hblur_row_avx2(src: &[f32], dst: &mut [f32], width: usize) {
    use std::arch::x86_64::*;

    let vk0 = _mm256_set1_ps(K0);
    let vk1 = _mm256_set1_ps(K1);
    let vk2 = _mm256_set1_ps(K2);

    let interior_start = 2usize;
    let interior_end = width - 2;
    let interior_len = interior_end - interior_start;
    let chunks = interior_len / 8;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let x = interior_start + i * 8;
        let vm2 = _mm256_loadu_ps(src_ptr.add(x - 2));
        let vm1 = _mm256_loadu_ps(src_ptr.add(x - 1));
        let v0  = _mm256_loadu_ps(src_ptr.add(x));
        let vp1 = _mm256_loadu_ps(src_ptr.add(x + 1));
        let vp2 = _mm256_loadu_ps(src_ptr.add(x + 2));

        let mut acc = _mm256_mul_ps(vk0, vm2);
        acc = _mm256_fmadd_ps(vk1, vm1, acc);
        acc = _mm256_fmadd_ps(vk2, v0, acc);
        acc = _mm256_fmadd_ps(vk1, vp1, acc);
        acc = _mm256_fmadd_ps(vk0, vp2, acc);

        _mm256_storeu_ps(dst_ptr.add(x), acc);
    }

    for x in (interior_start + chunks * 8)..interior_end {
        *dst_ptr.add(x) =
              K0 * *src_ptr.add(x - 2)
            + K1 * *src_ptr.add(x - 1)
            + K2 * *src_ptr.add(x)
            + K1 * *src_ptr.add(x + 1)
            + K0 * *src_ptr.add(x + 2);
    }
}

// =============================================================================
// Reference downsample (kept for standalone use)
// =============================================================================

#[allow(dead_code)]
fn downsample_2x(src: &Image<f32>) -> Image<f32> {
    let new_w = src.width() / 2;
    let new_h = src.height() / 2;
    let mut dst = Image::new(new_w, new_h);
    for y in 0..new_h {
        for x in 0..new_w {
            unsafe { dst.set_unchecked(x, y, src.get_unchecked(x * 2, y * 2)); }
        }
    }
    dst
}

#[allow(dead_code)]
fn downsample_2x_into(src: &Image<f32>, dst: &mut Image<f32>) {
    let new_w = src.width() / 2;
    let new_h = src.height() / 2;
    dst.clear_resize(new_w, new_h);
    for y in 0..new_h {
        for x in 0..new_w {
            unsafe { dst.set_unchecked(x, y, src.get_unchecked(x * 2, y * 2)); }
        }
    }
}

// =============================================================================
// u8 → f32 conversion helpers
// =============================================================================

fn to_f32_image<T: Pixel>(src: &Image<T>) -> Image<f32> {
    let mut dst = Image::new(src.width(), src.height());
    for y in 0..src.height() {
        for x in 0..src.width() {
            dst.set(x, y, src.get(x, y).to_f32());
        }
    }
    dst
}

/// Fast u8 → f32 conversion into pre-allocated buffer.
fn to_f32_image_u8_into(src: &Image<u8>, dst: &mut Image<f32>) {
    dst.clear_resize(src.width(), src.height());
    let src_stride = src.stride();
    let dst_stride = dst.stride();
    let src_slice = src.as_slice();
    let dst_slice = dst.as_mut_slice();
    let w = src.width();

    for y in 0..src.height() {
        let src_off = y * src_stride;
        let dst_off = y * dst_stride;
        unsafe {
            for x in 0..w {
                *dst_slice.get_unchecked_mut(dst_off + x) =
                    *src_slice.get_unchecked(src_off + x) as f32;
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_dimensions() {
        let img = Image::<f32>::new(100, 80);
        let down = downsample_2x(&img);
        assert_eq!(down.width(), 50);
        assert_eq!(down.height(), 40);
    }

    #[test]
    fn test_downsample_odd_dimensions() {
        let img = Image::<f32>::new(7, 5);
        let down = downsample_2x(&img);
        assert_eq!(down.width(), 3);
        assert_eq!(down.height(), 2);
    }

    #[test]
    fn test_downsample_preserves_values() {
        let mut img = Image::<f32>::new(4, 4);
        img.set(0, 0, 1.0);
        img.set(2, 0, 2.0);
        img.set(0, 2, 3.0);
        img.set(2, 2, 4.0);

        let down = downsample_2x(&img);
        assert_eq!(down.width(), 2);
        assert_eq!(down.height(), 2);
        assert!((down.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((down.get(1, 0) - 2.0).abs() < 1e-6);
        assert!((down.get(0, 1) - 3.0).abs() < 1e-6);
        assert!((down.get(1, 1) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_level_dimensions() {
        let img: Image<u8> = Image::new(640, 480);
        let pyr = Pyramid::build(&img, 5, 1.0);

        assert_eq!(pyr.num_levels(), 5);
        assert_eq!(pyr.levels[0].width(), 640);
        assert_eq!(pyr.levels[0].height(), 480);
        assert_eq!(pyr.levels[1].width(), 320);
        assert_eq!(pyr.levels[1].height(), 240);
        assert_eq!(pyr.levels[2].width(), 160);
        assert_eq!(pyr.levels[2].height(), 120);
        assert_eq!(pyr.levels[3].width(), 80);
        assert_eq!(pyr.levels[3].height(), 60);
        assert_eq!(pyr.levels[4].width(), 40);
        assert_eq!(pyr.levels[4].height(), 30);
    }

    #[test]
    fn test_pyramid_single_level() {
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let img = Image::from_vec(2, 2, data);
        let pyr = Pyramid::build(&img, 1, 1.0);

        assert_eq!(pyr.num_levels(), 1);
        assert!((pyr.levels[0].get(0, 0) - 10.0).abs() < 1e-6);
        assert!((pyr.levels[0].get(1, 1) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_constant_image() {
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]);
        let pyr = Pyramid::build(&img, 4, 1.0);

        for (lvl, level) in pyr.levels.iter().enumerate() {
            for (x, y, v) in level.pixels() {
                assert!(
                    (v - 128.0).abs() < 0.5,
                    "level {lvl} pixel ({x},{y}) = {v}, expected 128.0"
                );
            }
        }
    }

    #[test]
    fn test_pyramid_decreasing_variance() {
        let mut data = vec![0u8; 128 * 128];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if (i / 128 + i % 128) % 2 == 0 { 255 } else { 0 };
        }
        let img = Image::from_vec(128, 128, data);
        let pyr = Pyramid::build(&img, 5, 1.0);

        let variance = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        let mut prev_var = variance(&pyr.levels[0]);
        for lvl in 1..pyr.num_levels() {
            let var = variance(&pyr.levels[lvl]);
            assert!(
                var <= prev_var * 1.2 + 1e-3,
                "variance increased too much from level {} ({prev_var}) to level {lvl} ({var})",
                lvl - 1
            );
            prev_var = var;
        }
    }

    #[test]
    fn test_pyramid_f32_input() {
        let img = Image::from_vec(32, 32, vec![50.0f32; 32 * 32]);
        let pyr = Pyramid::build(&img, 3, 1.0);
        assert_eq!(pyr.num_levels(), 3);
        assert!((pyr.levels[0].get(0, 0) - 50.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn test_pyramid_zero_levels_panics() {
        let img: Image<u8> = Image::new(10, 10);
        Pyramid::build(&img, 0, 1.0);
    }

    // ===== Fused pyrdown f32 tests =====

    #[test]
    fn test_fused_constant_passthrough() {
        let src = Image::from_vec(20, 20, vec![42.0f32; 400]);
        let mut dst = Image::new(10, 10);
        let mut h_buf = vec![0.0f32; 400];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        for (x, y, v) in dst.pixels() {
            assert!(
                (v - 42.0).abs() < 1e-4,
                "fused constant: ({x},{y}) = {v}, expected 42.0"
            );
        }
    }

    #[test]
    fn test_fused_dimensions() {
        let src = Image::<f32>::new(100, 80);
        let mut dst = Image::new(1, 1);
        let mut h_buf = vec![0.0f32; 8000];
        pyrdown_fused(&src, &mut dst, &mut h_buf);
        assert_eq!(dst.width(), 50);
        assert_eq!(dst.height(), 40);
    }

    #[test]
    fn test_fused_reduces_variance() {
        let mut data = vec![0.0f32; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                data[y * 64 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
            }
        }
        let src = Image::from_vec(64, 64, data);
        let mut dst = Image::new(32, 32);
        let mut h_buf = vec![0.0f32; 64 * 64];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        let var = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        let var_src = var(&src);
        let var_dst = var(&dst);
        assert!(var_dst < var_src,
            "fused pyrdown should reduce variance: src={var_src}, dst={var_dst}");
    }

    #[test]
    fn test_fused_symmetry() {
        let mut data = vec![0.0f32; 100 * 100];
        for y in 0..100 {
            for x in 0..100 {
                data[y * 100 + x] = x as f32 * 2.55;
            }
        }
        let src = Image::from_vec(100, 100, data);
        let mut dst = Image::new(50, 50);
        let mut h_buf = vec![0.0f32; 10000];
        pyrdown_fused(&src, &mut dst, &mut h_buf);

        for y in 2..48 {
            for x in 2..48 {
                assert!(
                    dst.get(x, y) >= dst.get(x - 1, y) - 0.1,
                    "monotonicity violated at ({x},{y}): {} < {}",
                    dst.get(x, y), dst.get(x - 1, y)
                );
            }
        }
    }

    // ===== Integer pyrdown tests =====

    #[test]
    fn test_int_constant_passthrough() {
        // Constant 128 image: blurred result should be 128.0
        let src = vec![128u8; 20 * 20];
        let mut dst_f32 = Image::new(10, 10);
        let mut dst_u8 = vec![0u8; 100];
        let mut h_buf = vec![0u16; 400];

        pyrdown_int(&src, 20, 20, 20, &mut dst_f32, &mut dst_u8, &mut h_buf);

        for (x, y, v) in dst_f32.pixels() {
            assert!(
                (v - 128.0).abs() < 1.0,
                "int constant: ({x},{y}) = {v}, expected ~128.0"
            );
        }
        for &v in &dst_u8[..100] {
            assert!(
                (v as i16 - 128).abs() <= 1,
                "int u8 constant: {v}, expected ~128"
            );
        }
    }

    #[test]
    fn test_int_dimensions() {
        let src = vec![0u8; 100 * 80];
        let mut dst_f32 = Image::new(1, 1);
        let mut dst_u8 = vec![0u8; 50 * 40];
        let mut h_buf = vec![0u16; 8000];

        pyrdown_int(&src, 100, 80, 100, &mut dst_f32, &mut dst_u8, &mut h_buf);
        assert_eq!(dst_f32.width(), 50);
        assert_eq!(dst_f32.height(), 40);
    }

    #[test]
    fn test_build_reuse_constant() {
        // Constant image: build_reuse should preserve value at all levels.
        let img = Image::from_vec(64, 64, vec![128u8; 64 * 64]);
        let mut pyr = Pyramid { levels: Vec::new() };
        let mut scratch = PyramidScratch::new(64, 64, 1.0);
        pyr.build_reuse(&img, 4, &mut scratch);

        for (lvl, level) in pyr.levels.iter().enumerate() {
            for (x, y, v) in level.pixels() {
                assert!(
                    (v - 128.0).abs() < 1.0,
                    "build_reuse constant: level {lvl} ({x},{y}) = {v}, expected ~128.0"
                );
            }
        }
    }

    #[test]
    fn test_build_reuse_dimensions() {
        let img: Image<u8> = Image::new(640, 480);
        let mut pyr = Pyramid { levels: Vec::new() };
        let mut scratch = PyramidScratch::new(640, 480, 1.0);
        pyr.build_reuse(&img, 5, &mut scratch);

        assert_eq!(pyr.levels[0].width(), 640);
        assert_eq!(pyr.levels[0].height(), 480);
        assert_eq!(pyr.levels[1].width(), 320);
        assert_eq!(pyr.levels[1].height(), 240);
        assert_eq!(pyr.levels[2].width(), 160);
        assert_eq!(pyr.levels[2].height(), 120);
        assert_eq!(pyr.levels[3].width(), 80);
        assert_eq!(pyr.levels[3].height(), 60);
        assert_eq!(pyr.levels[4].width(), 40);
        assert_eq!(pyr.levels[4].height(), 30);
    }

    #[test]
    fn test_build_reuse_close_to_build() {
        // Integer and f32 paths should agree within rounding tolerance.
        let data: Vec<u8> = (0..100).cycle().take(640 * 480).collect();
        let img = Image::from_vec(640, 480, data);

        let pyr_f32 = Pyramid::build(&img, 4, 1.0);

        let mut pyr_int = Pyramid { levels: Vec::new() };
        let mut scratch = PyramidScratch::new(640, 480, 1.0);
        pyr_int.build_reuse(&img, 4, &mut scratch);

        for lvl in 0..4 {
            let a = &pyr_f32.levels[lvl];
            let b = &pyr_int.levels[lvl];
            assert_eq!(a.width(), b.width(), "level {lvl} width mismatch");
            assert_eq!(a.height(), b.height(), "level {lvl} height mismatch");

            // Level 0 is exact (both are just u8→f32).
            // Levels 1+ may differ by up to ~1.0 due to integer rounding.
            let tol = if lvl == 0 { 1e-4 } else { 1.5 };
            for (x, y, va) in a.pixels() {
                let vb = b.get(x, y);
                assert!(
                    (va - vb).abs() < tol,
                    "level {lvl} ({x},{y}): f32={va}, int={vb}, diff={:.4}",
                    (va - vb).abs()
                );
            }
        }
    }

    #[test]
    fn test_build_reuse_decreasing_variance() {
        let mut data = vec![0u8; 128 * 128];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if (i / 128 + i % 128) % 2 == 0 { 255 } else { 0 };
        }
        let img = Image::from_vec(128, 128, data);

        let mut pyr = Pyramid { levels: Vec::new() };
        let mut scratch = PyramidScratch::new(128, 128, 1.0);
        pyr.build_reuse(&img, 5, &mut scratch);

        let variance = |img: &Image<f32>| {
            let n = (img.width() * img.height()) as f32;
            let mean: f32 = img.pixels().map(|(_, _, v)| v).sum::<f32>() / n;
            img.pixels().map(|(_, _, v)| (v - mean) * (v - mean)).sum::<f32>() / n
        };

        let mut prev_var = variance(&pyr.levels[0]);
        for lvl in 1..pyr.num_levels() {
            let var = variance(&pyr.levels[lvl]);
            assert!(
                var <= prev_var * 1.2 + 1e-3,
                "variance increased: level {} ({prev_var}) → level {lvl} ({var})",
                lvl - 1
            );
            prev_var = var;
        }
    }
}
