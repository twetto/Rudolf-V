// histeq.rs — Histogram equalization for brightness normalization.
//
// Auto-exposure on cameras like EuRoC's MT9V034 causes sudden brightness
// jumps between frames. The KLT tracker matches templates by SSD — if
// the overall brightness shifts, the error landscape changes and features
// either jump to wrong locations or get lost.
//
// Histogram equalization normalizes the brightness distribution to be
// approximately uniform, making the image intensity statistics consistent
// frame-to-frame regardless of exposure changes.
//
// Two variants:
//
// 1. GLOBAL histogram equalization: compute one histogram over the entire
//    image, build the CDF, and remap every pixel. Simple, fast, and
//    effective when auto-exposure is the main problem.
//
// 2. CLAHE (Contrast Limited Adaptive Histogram Equalization): divide the
//    image into tiles, equalize each tile independently with a clip limit,
//    then bilinearly interpolate between tile CDFs for smooth transitions.
//
// OPTIMIZATIONS:
// - Direct slice access (no per-pixel method call overhead)
// - equalize_histogram_into() avoids allocation by writing into caller's buffer
// - Scratch buffer for histeq output can be held in Frontend (reused per frame)
// - RAYON PARALLELISM (feature-gated): histogram via fold/reduce with per-thread
//   local histograms (no atomics), remap via parallel row iteration.
//
// GPU NOTES:
// - Global histEq: atomic histogram in shared memory, prefix sum for CDF,
//   per-pixel LUT remap. Three dispatches.
// - CLAHE: one workgroup per tile, then per-pixel interpolation pass.

use crate::image::Image;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Histogram equalization method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistEqMethod {
    /// No preprocessing.
    None,
    /// Global histogram equalization.
    Global,
    /// CLAHE with the given tile size and clip limit.
    Clahe { tile_size: usize, clip_limit: f32 },
}

// ============================================================
// Global histogram equalization
// ============================================================

/// Apply global histogram equalization, allocating output.
pub fn equalize_histogram(image: &Image<u8>) -> Image<u8> {
    let mut out = Image::new(image.width(), image.height());
    equalize_histogram_into(image, &mut out);
    out
}

/// Compute the histogram equalization LUT without applying it.
///
/// Returns a 256-entry lookup table that maps input pixel values to
/// equalized output values. The caller can apply this LUT during another
/// pass (e.g., fused into pyramid construction) to avoid a separate remap.
pub fn equalize_histogram_lut(image: &Image<u8>) -> [u8; 256] {
    let w = image.width();
    let h = image.height();
    let n = w * h;

    if n == 0 {
        return [0u8; 256];
    }

    let src = image.as_slice();
    let src_stride = image.stride();

    // Four interleaved sub-histograms: consecutive pixels of equal value
    // (common in smooth regions) would otherwise serialize on one counter's
    // load→increment→store chain. ~1.35× faster on Cortex-A78AE; the summed
    // result is identical.
    let mut sub = [[0u32; 256]; 4];
    for y in 0..h {
        let row = &src[y * src_stride..y * src_stride + w];
        let mut quads = row.chunks_exact(4);
        for q in &mut quads {
            unsafe {
                *sub[0].get_unchecked_mut(q[0] as usize) += 1;
                *sub[1].get_unchecked_mut(q[1] as usize) += 1;
                *sub[2].get_unchecked_mut(q[2] as usize) += 1;
                *sub[3].get_unchecked_mut(q[3] as usize) += 1;
            }
        }
        for &v in quads.remainder() {
            sub[0][v as usize] += 1;
        }
    }
    let mut hist = [0u32; 256];
    for (i, bin) in hist.iter_mut().enumerate() {
        *bin = sub[0][i] + sub[1][i] + sub[2][i] + sub[3][i];
    }

    build_lut(&hist, n)
}

/// Remap `src` through a 256-entry LUT into `dst` (same length).
///
/// On aarch64 this uses NEON `tbl` lookups (four 64-byte table slices per 16
/// pixels, ~2× faster than the scalar loop on Cortex-A78AE); other targets use
/// a scalar loop. Output is identical either way.
pub(crate) fn remap_lut(src: &[u8], dst: &mut [u8], lut: &[u8; 256]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = src.len() / 16 * 16;
        // SAFETY: NEON is mandatory on aarch64; all loads/stores are within
        // `src[..n]` / `dst[..n]` and the 256-byte LUT.
        unsafe {
            let t0 = vld1q_u8_x4(lut.as_ptr());
            let t1 = vld1q_u8_x4(lut.as_ptr().add(64));
            let t2 = vld1q_u8_x4(lut.as_ptr().add(128));
            let t3 = vld1q_u8_x4(lut.as_ptr().add(192));
            let c64 = vdupq_n_u8(64);
            let mut i = 0;
            while i < n {
                // vqtbl4q yields 0 for indices >= 64, so each 64-entry slice
                // contributes only for its own index range; OR them together.
                let x0 = vld1q_u8(src.as_ptr().add(i));
                let x1 = vsubq_u8(x0, c64);
                let x2 = vsubq_u8(x1, c64);
                let x3 = vsubq_u8(x2, c64);
                let r = vorrq_u8(
                    vorrq_u8(vqtbl4q_u8(t0, x0), vqtbl4q_u8(t1, x1)),
                    vorrq_u8(vqtbl4q_u8(t2, x2), vqtbl4q_u8(t3, x3)),
                );
                vst1q_u8(dst.as_mut_ptr().add(i), r);
                i += 16;
            }
        }
        for j in n..src.len() {
            dst[j] = lut[src[j] as usize];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = lut[v as usize];
    }
}

/// Apply global histogram equalization into a pre-allocated buffer.
///
/// Avoids per-frame allocation when called repeatedly (e.g., from Frontend).
/// Uses direct slice access for both histogram and remap passes.
/// When the `parallel` feature is enabled, both passes use rayon:
/// - Histogram: fold/reduce with per-thread local histograms (no atomics).
/// - Remap: parallel row iteration with LUT lookup.
pub fn equalize_histogram_into(image: &Image<u8>, out: &mut Image<u8>) {
    let w = image.width();
    let h = image.height();
    let n = w * h;

    out.clear_resize(w, h);

    if n == 0 {
        return;
    }

    let src = image.as_slice();
    let src_stride = image.stride();

    // Step 1: Histogram.
    #[cfg(feature = "parallel")]
    let hist = {
        // Each thread builds a local [u32; 256], then we reduce by summing.
        // No atomics, no contention on the 256 bins.
        (0..h)
            .into_par_iter()
            .fold(
                || [0u32; 256],
                |mut local_hist, y| {
                    let row = y * src_stride;
                    unsafe {
                        for x in 0..w {
                            let v = *src.get_unchecked(row + x) as usize;
                            *local_hist.get_unchecked_mut(v) += 1;
                        }
                    }
                    local_hist
                },
            )
            .reduce(
                || [0u32; 256],
                |mut a, b| {
                    for i in 0..256 {
                        a[i] += b[i];
                    }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let hist = {
        let mut hist = [0u32; 256];
        for y in 0..h {
            let row = y * src_stride;
            unsafe {
                for x in 0..w {
                    let v = *src.get_unchecked(row + x) as usize;
                    *hist.get_unchecked_mut(v) += 1;
                }
            }
        }
        hist
    };

    // Step 2: CDF → LUT (256 elements, trivially fast, always sequential).
    let lut = build_lut(&hist, n);

    // Step 3: Remap.
    let dst_stride = out.stride();
    let dst = out.as_mut_slice();

    #[cfg(feature = "parallel")]
    {
        let dst_base = dst.as_mut_ptr() as usize;
        (0..h).into_par_iter().for_each(|y| {
            let src_off = y * src_stride;
            let dst_off = y * dst_stride;
            unsafe {
                let dp = dst_base as *mut u8;
                for x in 0..w {
                    let v = *src.get_unchecked(src_off + x) as usize;
                    *dp.add(dst_off + x) = *lut.get_unchecked(v);
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..h {
            let src_off = y * src_stride;
            let dst_off = y * dst_stride;
            unsafe {
                for x in 0..w {
                    let v = *src.get_unchecked(src_off + x) as usize;
                    *dst.get_unchecked_mut(dst_off + x) = *lut.get_unchecked(v);
                }
            }
        }
    }
}

/// Wrapper to send a raw pointer across threads.
/// SAFETY: The caller must guarantee non-overlapping writes.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}

/// Build a 256-entry lookup table from a histogram and total pixel count.
fn build_lut(hist: &[u32; 256], total: usize) -> [u8; 256] {
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Find the first non-zero CDF value (skip fully black bins).
    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);

    let mut lut = [0u8; 256];
    let denom = total as f32 - cdf_min as f32;
    if denom <= 0.0 {
        return lut;
    }

    for i in 0..256 {
        let val = (cdf[i] as f32 - cdf_min as f32) / denom * 255.0;
        lut[i] = val.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

// ============================================================
// CLAHE
// ============================================================

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
/// The spatially-varying map CLAHE applies: per-tile LUTs plus the x-direction
/// tile indices and blend weights, which are identical for every row.
///
/// Exists so the remap can be FUSED into pyramid level 0 rather than
/// materialising a whole equalized image first. Global histeq has always been
/// fused: it needs only a single `[u8; 256]`, which the pyramid applies while it
/// copies src into `u8_levels[0]`. CLAHE could not be, because a spatially
/// varying map is not a LUT -- this type is that map. Measured at 320x200, the
/// separate pass was 0.651 ms of CLAHE's 0.973 ms even with the NEON kernel, and
/// all of it duplicates a read and a write the pyramid performs anyway.
pub struct ClaheMap {
    tile_luts: Vec<[u8; 256]>,
    col_tx0: Vec<usize>,
    col_tx1: Vec<usize>,
    col_ax: Vec<f32>,
    rows: usize,
    cols: usize,
    tile_size: usize,
    width: usize,
    height: usize,
}

impl ClaheMap {
    /// Build the per-tile LUTs and the x-direction tables. This is the
    /// histogram half of CLAHE (0.321 ms at 320x200, tile 32); the per-pixel
    /// half is `remap_row`.
    pub fn build(image: &Image<u8>, tile_size: usize, clip_limit: f32) -> Self {
        let w = image.width();
        let h = image.height();
        let cols = if tile_size == 0 { 0 } else { (w + tile_size - 1) / tile_size };
        let rows = if tile_size == 0 { 0 } else { (h + tile_size - 1) / tile_size };
        if w == 0 || h == 0 || tile_size == 0 {
            return Self {
                tile_luts: Vec::new(),
                col_tx0: Vec::new(),
                col_tx1: Vec::new(),
                col_ax: Vec::new(),
                rows,
                cols,
                tile_size,
                width: w,
                height: h,
            };
        }

        let src = image.as_slice();
        let src_stride = image.stride();

        #[cfg(feature = "parallel")]
        let tile_luts: Vec<[u8; 256]> = (0..rows * cols)
            .into_par_iter()
            .map(|idx| {
                clahe_tile_lut(src, src_stride, w, h, tile_size, idx % cols, idx / cols, clip_limit)
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let tile_luts: Vec<[u8; 256]> = (0..rows * cols)
            .map(|idx| {
                clahe_tile_lut(src, src_stride, w, h, tile_size, idx % cols, idx / cols, clip_limit)
            })
            .collect();

        let inv_tile = 1.0 / tile_size as f32;
        let tile_cx = |tx: usize| -> f32 { (tx as f32 + 0.5) * tile_size as f32 };

        let col_tx0: Vec<usize> = (0..w)
            .map(|x| (((x as f32) * inv_tile - 0.5).floor() as isize).max(0) as usize)
            .collect();
        let col_tx1: Vec<usize> = col_tx0.iter().map(|&t| (t + 1).min(cols - 1)).collect();
        let col_ax: Vec<f32> = (0..w)
            .map(|x| {
                let tx0 = col_tx0[x];
                let tx1 = col_tx1[x];
                if tx0 == tx1 {
                    0.0
                } else {
                    ((x as f32 - tile_cx(tx0)) / (tile_cx(tx1) - tile_cx(tx0))).clamp(0.0, 1.0)
                }
            })
            .collect();

        Self { tile_luts, col_tx0, col_tx1, col_ax, rows, cols, tile_size, width: w, height: h }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// True when the map has nothing to apply (degenerate image or tile size).
    pub fn is_empty(&self) -> bool {
        self.tile_luts.is_empty()
    }

    /// Remap one row. `src_row` and `dst_row` must both be at least
    /// `self.width()` long. Bit-identical to the two-pass path: same tables,
    /// same kernel.
    pub fn remap_row(&self, src_row: &[u8], dst_row: &mut [u8], y: usize) {
        if self.tile_luts.is_empty() {
            return;
        }
        let w = self.width;
        let inv_tile = 1.0 / self.tile_size as f32;
        let tile_cy = |ty: usize| -> f32 { (ty as f32 + 0.5) * self.tile_size as f32 };
        let py = y as f32;
        let fy = py * inv_tile - 0.5;
        let ty0 = (fy.floor() as isize).max(0) as usize;
        let ty1 = (ty0 + 1).min(self.rows - 1);
        let ay = if ty0 == ty1 {
            0.0
        } else {
            ((py - tile_cy(ty0)) / (tile_cy(ty1) - tile_cy(ty0))).clamp(0.0, 1.0)
        };
        let w_top = 1.0 - ay;
        let lut_row0 = &self.tile_luts[ty0 * self.cols..ty0 * self.cols + self.cols];
        let lut_row1 = &self.tile_luts[ty1 * self.cols..ty1 * self.cols + self.cols];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            remap_row_neon(
                src_row, dst_row, w, lut_row0, lut_row1,
                &self.col_tx0, &self.col_tx1, &self.col_ax, ay, w_top,
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        remap_row_scalar(
            src_row, dst_row, w, lut_row0, lut_row1,
            &self.col_tx0, &self.col_tx1, &self.col_ax, ay, w_top,
        );
    }
}

pub fn equalize_clahe(image: &Image<u8>, tile_size: usize, clip_limit: f32) -> Image<u8> {
    let mut out = Image::new(image.width(), image.height());
    equalize_clahe_into(image, tile_size, clip_limit, &mut out);
    out
}

/// Apply CLAHE into a pre-allocated buffer.
///
/// When the `parallel` feature is enabled, both the per-tile LUT computation
/// and the bilinear remap pass are parallelized with rayon.
pub fn equalize_clahe_into(
    image: &Image<u8>,
    tile_size: usize,
    clip_limit: f32,
    out: &mut Image<u8>,
) {
    let w = image.width();
    let h = image.height();

    out.clear_resize(w, h);

    if w == 0 || h == 0 || tile_size == 0 {
        return;
    }

    // Same map the fused pyramid path uses; this standalone form remains for
    // callers that genuinely want a materialised equalized image.
    let map = ClaheMap::build(image, tile_size, clip_limit);
    let src = image.as_slice();
    let src_stride = image.stride();
    let dst_stride = out.stride();
    let dst = out.as_mut_slice();

    #[cfg(feature = "parallel")]
    {
        let dst_base = dst.as_mut_ptr() as usize;
        (0..h).into_par_iter().for_each(|y| {
            let row = unsafe {
                std::slice::from_raw_parts_mut((dst_base as *mut u8).add(y * dst_stride), w)
            };
            map.remap_row(&src[y * src_stride..y * src_stride + w], row, y);
        });
    }

    #[cfg(not(feature = "parallel"))]
    for y in 0..h {
        let (_, rest) = dst.split_at_mut(y * dst_stride);
        map.remap_row(
            &src[y * src_stride..y * src_stride + w],
            &mut rest[..w],
            y,
        );
    }
}

/// Scalar reference for the CLAHE bilinear remap of one row.
///
/// Always compiled: it is the non-aarch64 implementation AND the reference the
/// NEON path is asserted bit-equal against, so it cannot rot unnoticed.
#[inline]
fn remap_row_scalar(
    src_row: &[u8],
    dst_row: &mut [u8],
    w: usize,
    lut_row0: &[[u8; 256]],
    lut_row1: &[[u8; 256]],
    col_tx0: &[usize],
    col_tx1: &[usize],
    col_ax: &[f32],
    ay: f32,
    w_top: f32,
) {
    for x in 0..w {
        let tx0 = unsafe { *col_tx0.get_unchecked(x) };
        let tx1 = unsafe { *col_tx1.get_unchecked(x) };
        let ax = unsafe { *col_ax.get_unchecked(x) };
        let v = unsafe { *src_row.get_unchecked(x) as usize };

        let v00 = unsafe { *lut_row0.get_unchecked(tx0).get_unchecked(v) } as f32;
        let v10 = unsafe { *lut_row0.get_unchecked(tx1).get_unchecked(v) } as f32;
        let v01 = unsafe { *lut_row1.get_unchecked(tx0).get_unchecked(v) } as f32;
        let v11 = unsafe { *lut_row1.get_unchecked(tx1).get_unchecked(v) } as f32;

        let top = v00 + ax * (v10 - v00);
        let bot = v01 + ax * (v11 - v01);
        let val = top * w_top + bot * ay;

        unsafe {
            *dst_row.get_unchecked_mut(x) = val.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Bilinear CLAHE remap for one row, four pixels at a time.
///
/// The LUT gathers stay scalar: NEON has no gather, and `tbl` would need four
/// 64-byte slices per table times four tables = 16 vector registers held across
/// the inner loop, which does not fit alongside the data on a 32-register core.
/// The four 256-byte tables are L1-resident, so the win here is the arithmetic --
/// three lerps, a round and a clamp per pixel -- which vectorises cleanly.
///
/// BIT-IDENTICAL to the scalar path, and deliberately so: the replay harness
/// compares trajectories bit-for-bit, so a different rounding here would
/// silently void every A/B against an earlier run. Two things that matters for:
///   * `vmulq`+`vaddq`, NOT `vfmaq`. Rust does not fuse `a + b * c` without
///     fast-math, so the scalar path keeps a rounded intermediate; an FMA here
///     would carry full precision and differ in the last bit.
///   * `vrndaq_f32` is round-half-away-from-zero, which is what `f32::round`
///     does; round-to-even (`vrndnq_f32`) would differ on exact .5 values.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn remap_row_neon(
    src_row: &[u8],
    dst_row: &mut [u8],
    w: usize,
    lut_row0: &[[u8; 256]],
    lut_row1: &[[u8; 256]],
    col_tx0: &[usize],
    col_tx1: &[usize],
    col_ax: &[f32],
    ay: f32,
    w_top: f32,
) {
    use std::arch::aarch64::*;
    let v_ay = vdupq_n_f32(ay);
    let v_wt = vdupq_n_f32(w_top);
    let v_zero = vdupq_n_f32(0.0);
    let v_max = vdupq_n_f32(255.0);

    let n4 = w / 4 * 4;
    let mut x = 0usize;
    while x < n4 {
        let mut g00 = [0f32; 4];
        let mut g10 = [0f32; 4];
        let mut g01 = [0f32; 4];
        let mut g11 = [0f32; 4];
        for i in 0..4 {
            let xi = x + i;
            let t0 = *col_tx0.get_unchecked(xi);
            let t1 = *col_tx1.get_unchecked(xi);
            let v = *src_row.get_unchecked(xi) as usize;
            g00[i] = *lut_row0.get_unchecked(t0).get_unchecked(v) as f32;
            g10[i] = *lut_row0.get_unchecked(t1).get_unchecked(v) as f32;
            g01[i] = *lut_row1.get_unchecked(t0).get_unchecked(v) as f32;
            g11[i] = *lut_row1.get_unchecked(t1).get_unchecked(v) as f32;
        }
        let v00 = vld1q_f32(g00.as_ptr());
        let v10 = vld1q_f32(g10.as_ptr());
        let v01 = vld1q_f32(g01.as_ptr());
        let v11 = vld1q_f32(g11.as_ptr());
        let v_ax = vld1q_f32(col_ax.as_ptr().add(x));

        // same association as the scalar: v00 + ax * (v10 - v00)
        let top = vaddq_f32(v00, vmulq_f32(v_ax, vsubq_f32(v10, v00)));
        let bot = vaddq_f32(v01, vmulq_f32(v_ax, vsubq_f32(v11, v01)));
        let val = vaddq_f32(vmulq_f32(top, v_wt), vmulq_f32(bot, v_ay));

        let r = vminq_f32(vmaxq_f32(vrndaq_f32(val), v_zero), v_max);
        let u = vcvtq_u32_f32(r);
        let h = vmovn_u32(u);
        let b = vmovn_u16(vcombine_u16(h, h));
        let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(b));
        std::ptr::copy_nonoverlapping(
            &packed as *const u32 as *const u8,
            dst_row.as_mut_ptr().add(x),
            4,
        );
        x += 4;
    }
    // scalar tail: w is not required to be a multiple of 4
    while x < w {
        let t0 = *col_tx0.get_unchecked(x);
        let t1 = *col_tx1.get_unchecked(x);
        let ax = *col_ax.get_unchecked(x);
        let v = *src_row.get_unchecked(x) as usize;
        let a = *lut_row0.get_unchecked(t0).get_unchecked(v) as f32;
        let b = *lut_row0.get_unchecked(t1).get_unchecked(v) as f32;
        let c = *lut_row1.get_unchecked(t0).get_unchecked(v) as f32;
        let d = *lut_row1.get_unchecked(t1).get_unchecked(v) as f32;
        let top = a + ax * (b - a);
        let bot = c + ax * (d - c);
        let val = top * w_top + bot * ay;
        *dst_row.get_unchecked_mut(x) = val.round().clamp(0.0, 255.0) as u8;
        x += 1;
    }
}

fn clahe_tile_lut(
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    tile_size: usize,
    tx: usize,
    ty: usize,
    clip_limit: f32,
) -> [u8; 256] {
    let x0 = tx * tile_size;
    let y0 = ty * tile_size;
    let x1 = (x0 + tile_size).min(w);
    let y1 = (y0 + tile_size).min(h);
    let tile_pixels = (x1 - x0) * (y1 - y0);

    let mut hist = [0u32; 256];
    for y in y0..y1 {
        let row = y * src_stride;
        unsafe {
            for x in x0..x1 {
                let v = *src.get_unchecked(row + x) as usize;
                *hist.get_unchecked_mut(v) += 1;
            }
        }
    }

    if clip_limit > 0.0 {
        clip_histogram(&mut hist, tile_pixels, clip_limit);
    }

    build_lut(&hist, tile_pixels)
}

/// Clip histogram bins and redistribute excess counts.
fn clip_histogram(hist: &mut [u32; 256], total_pixels: usize, clip_multiplier: f32) {
    let clip_val = ((total_pixels as f32 / 256.0) * clip_multiplier).ceil() as u32;

    let mut excess = 0u32;
    for bin in hist.iter_mut() {
        if *bin > clip_val {
            excess += *bin - clip_val;
            *bin = clip_val;
        }
    }

    let per_bin = excess / 256;
    let remainder = (excess % 256) as usize;
    for (i, bin) in hist.iter_mut().enumerate() {
        *bin += per_bin;
        if i < remainder {
            *bin += 1;
        }
    }
}

/// Convenience function: apply the specified equalization method.
pub fn apply_histeq(image: &Image<u8>, method: HistEqMethod) -> Image<u8> {
    match method {
        HistEqMethod::None => image.clone(),
        HistEqMethod::Global => equalize_histogram(image),
        HistEqMethod::Clahe {
            tile_size,
            clip_limit,
        } => equalize_clahe(image, tile_size, clip_limit),
    }
}

/// Apply histogram equalization into a pre-allocated buffer.
///
/// Use this from Frontend to avoid per-frame allocation.
pub fn apply_histeq_into(image: &Image<u8>, method: HistEqMethod, out: &mut Image<u8>) {
    match method {
        HistEqMethod::None => {
            // Copy instead of clone — reuses out's allocation.
            out.clear_resize(image.width(), image.height());
            let w = image.width();
            let src_stride = image.stride();
            let dst_stride = out.stride();
            let src = image.as_slice();
            let dst = out.as_mut_slice();
            for y in 0..image.height() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(y * src_stride),
                        dst.as_mut_ptr().add(y * dst_stride),
                        w,
                    );
                }
            }
        }
        HistEqMethod::Global => equalize_histogram_into(image, out),
        HistEqMethod::Clahe {
            tile_size,
            clip_limit,
        } => equalize_clahe_into(image, tile_size, clip_limit, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decompose the CLAHE-vs-global cost gap into its two components at the
    /// rig's actual resolution. Run with:
    ///   cargo test --release global_vs_clahe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn global_vs_clahe_decomposition() {
        use std::time::Instant;
        const W: usize = 320;
        const H: usize = 200;
        let Some(p) = std::env::var_os("RUDOLF_BENCH_FRAMES") else {
            eprintln!("set RUDOLF_BENCH_FRAMES to a raw 320x200 u8 frame cache to run this");
            return;
        };
        let all = std::fs::read(p).expect("RUDOLF_BENCH_FRAMES");
        let off = 15000 * W * H;
        let img = Image::from_vec(W, H, all[off..off + W * H].to_vec());
        let reps = 400;

        // --- global ---
        let t = Instant::now();
        let mut sink = 0u64;
        for _ in 0..reps {
            sink += equalize_histogram_lut(&img)[128] as u64;
        }
        let g_lut = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let mut out = Image::new(W, H);
        let t = Instant::now();
        for _ in 0..reps {
            equalize_histogram_into(&img, &mut out);
        }
        let g_full = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        println!(
            "global      hist+lut={:.3} ms  remap={:.3} ms  full={:.3} ms   (1 lookup/px, no interp)",
            g_lut, g_full - g_lut, g_full
        );

        // --- clahe, tile 32 ---
        let tile = 32usize;
        let cols = (W + tile - 1) / tile;
        let rows = (H + tile - 1) / tile;
        let src = img.as_slice();
        let stride = img.stride();
        let t = Instant::now();
        for _ in 0..reps {
            for idx in 0..rows * cols {
                sink += clahe_tile_lut(src, stride, W, H, tile, idx % cols, idx / cols, 4.0)[128]
                    as u64;
            }
        }
        let c_lut = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        let t = Instant::now();
        for _ in 0..reps {
            equalize_clahe_into(&img, tile, 4.0, &mut out);
        }
        let c_full = t.elapsed().as_secs_f64() / reps as f64 * 1e3;
        println!(
            "clahe t=32  hist+lut={:.3} ms  remap={:.3} ms  full={:.3} ms   ({} tiles, 4 lookups/px + 3 lerps, NEON)",
            c_lut, c_full - c_lut, c_full, rows * cols
        );
        println!(
            "ratio       hist+lut={:.1}x        remap={:.1}x        full={:.1}x   sink={}",
            c_lut / g_lut,
            (c_full - c_lut) / (g_full - g_lut),
            c_full / g_full,
            sink & 0xff
        );
    }

    /// The NEON CLAHE remap must be BIT-identical to the scalar reference, not
    /// merely close: the replay harness compares trajectories bit-for-bit, so a
    /// one-LSB difference here would silently invalidate every A/B against a
    /// run made with the old code. Covers widths that are not multiples of 4
    /// (the vector tail), tile sizes that do not divide the image, and a real
    /// frame as well as synthetic extremes.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_clahe_neon_bit_identical_to_scalar() {
        fn scalar_clahe(img: &Image<u8>, tile: usize, clip: f32) -> Image<u8> {
            let w = img.width();
            let h = img.height();
            let cols = (w + tile - 1) / tile;
            let rows = (h + tile - 1) / tile;
            let src = img.as_slice();
            let stride = img.stride();
            let luts: Vec<[u8; 256]> = (0..rows * cols)
                .map(|i| clahe_tile_lut(src, stride, w, h, tile, i % cols, i / cols, clip))
                .collect();
            let inv_tile = 1.0 / tile as f32;
            let tcx = |tx: usize| (tx as f32 + 0.5) * tile as f32;
            let tcy = |ty: usize| (ty as f32 + 0.5) * tile as f32;
            let tx0: Vec<usize> = (0..w)
                .map(|x| (((x as f32) * inv_tile - 0.5).floor() as isize).max(0) as usize)
                .collect();
            let tx1: Vec<usize> = tx0.iter().map(|&t| (t + 1).min(cols - 1)).collect();
            let ax: Vec<f32> = (0..w)
                .map(|x| {
                    if tx0[x] == tx1[x] {
                        0.0
                    } else {
                        ((x as f32 - tcx(tx0[x])) / (tcx(tx1[x]) - tcx(tx0[x]))).clamp(0.0, 1.0)
                    }
                })
                .collect();
            let mut out = Image::new(w, h);
            let dstride = out.stride();
            let dst = out.as_mut_slice();
            for y in 0..h {
                let py = y as f32;
                let fy = py * inv_tile - 0.5;
                let ty0 = (fy.floor() as isize).max(0) as usize;
                let ty1 = (ty0 + 1).min(rows - 1);
                let ayv = if ty0 == ty1 {
                    0.0
                } else {
                    ((py - tcy(ty0)) / (tcy(ty1) - tcy(ty0))).clamp(0.0, 1.0)
                };
                let (_, rest) = dst.split_at_mut(y * dstride);
                remap_row_scalar(
                    &src[y * stride..y * stride + w],
                    &mut rest[..w],
                    w,
                    &luts[ty0 * cols..ty0 * cols + cols],
                    &luts[ty1 * cols..ty1 * cols + cols],
                    &tx0,
                    &tx1,
                    &ax,
                    ayv,
                    1.0 - ayv,
                );
            }
            out
        }

        let mut cases: Vec<(String, Image<u8>)> = Vec::new();
        // synthetic: widths 317/319/320/321 exercise every vector-tail remainder
        let mut seed = 0x12345678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 24) as u8
        };
        for &(w, h) in &[(317usize, 100usize), (319, 61), (320, 200), (321, 37), (7, 5)] {
            let n = w * h;
            cases.push((
                format!("random {w}x{h}"),
                Image::from_vec(w, h, (0..n).map(|_| rnd()).collect()),
            ));
            cases.push((
                format!("constant {w}x{h}"),
                Image::from_vec(w, h, vec![128u8; n]),
            ));
            cases.push((
                format!("gradient {w}x{h}"),
                Image::from_vec(w, h, (0..n).map(|i| (i % 256) as u8).collect()),
            ));
            // extremes: all-0 / all-255 drive the clamp branches
            cases.push((format!("zeros {w}x{h}"), Image::from_vec(w, h, vec![0u8; n])));
            cases.push((format!("sat {w}x{h}"), Image::from_vec(w, h, vec![255u8; n])));
        }
        // a real frame, if the bike cache is present on this machine
        if let Ok(all) = std::env::var_os("RUDOLF_BENCH_FRAMES")
            .map(std::fs::read)
            .unwrap_or_else(|| Ok(Vec::new()))
        {
            for &k in &[0usize, 15000, 30000] {
                let off = k * 320 * 200;
                if off + 320 * 200 <= all.len() {
                    cases.push((
                        format!("bike frame {k}"),
                        Image::from_vec(320, 200, all[off..off + 320 * 200].to_vec()),
                    ));
                }
            }
        }

        let mut checked = 0;
        for (name, img) in &cases {
            for &tile in &[7usize, 16, 32, 64, 256] {
                for &clip in &[1.0f32, 4.0, 40.0] {
                    let want = scalar_clahe(img, tile, clip);
                    let mut got = Image::new(img.width(), img.height());
                    equalize_clahe_into(img, tile, clip, &mut got);
                    assert_eq!(
                        want.as_slice(),
                        got.as_slice(),
                        "NEON != scalar for {name}, tile={tile}, clip={clip}"
                    );
                    checked += 1;
                }
            }
        }
        println!("bit-identical over {checked} (image, tile, clip) combinations");
    }

    /// Where does CLAHE's time go: per-tile LUT construction, or the per-pixel
    /// bilinear remap? Run with:
    ///   cargo test --release clahe_split -- --ignored --nocapture
    /// Uses a real 320x200 frame from the bike cache so the histogram (and so
    /// the clipping work) is representative rather than synthetic.
    #[test]
    #[ignore]
    fn clahe_split_profile() {
        use std::time::Instant;
        const W: usize = 320;
        const H: usize = 200;
        let Some(path) = std::env::var_os("RUDOLF_BENCH_FRAMES") else {
            eprintln!("set RUDOLF_BENCH_FRAMES to a raw 320x200 u8 frame cache to run this");
            return;
        };
        let all = std::fs::read(path).expect("RUDOLF_BENCH_FRAMES");
        // take a mid-bag frame, not frame 0 (which is the static checkerboard)
        let off = 15000 * W * H;
        let img = Image::from_vec(W, H, all[off..off + W * H].to_vec());
        let src = img.as_slice();
        let stride = img.stride();

        for &tile in &[32usize, 64, 256] {
            let cols = (W + tile - 1) / tile;
            let rows = (H + tile - 1) / tile;
            let reps = 200;

            // (a) LUT construction only
            let t0 = Instant::now();
            let mut sink = 0u64;
            for _ in 0..reps {
                for idx in 0..rows * cols {
                    let lut = clahe_tile_lut(src, stride, W, H, tile, idx % cols, idx / cols, 4.0);
                    sink += lut[128] as u64;
                }
            }
            let t_lut = t0.elapsed().as_secs_f64() / reps as f64 * 1e3;

            // (b) full stage
            let mut out = Image::new(W, H);
            let t1 = Instant::now();
            for _ in 0..reps {
                equalize_clahe_into(&img, tile, 4.0, &mut out);
            }
            let t_full = t1.elapsed().as_secs_f64() / reps as f64 * 1e3;

            println!(
                "tile={:3} ({}x{}={:3} tiles)  lut={:.3} ms  full={:.3} ms  remap={:.3} ms  ({:.0}% remap)  sink={}",
                tile, cols, rows, cols * rows, t_lut, t_full, t_full - t_lut,
                100.0 * (t_full - t_lut) / t_full, sink & 0xff
            );
        }
    }

    #[test]
    fn test_global_uniform_input() {
        let mut img = Image::new(256, 1);
        for x in 0..256 {
            img.set(x, 0, x as u8);
        }
        let out = equalize_histogram(&img);
        for x in 0..256 {
            let diff = (out.get(x, 0) as i32 - x as i32).abs();
            assert!(diff <= 1, "pixel {x}: expected ~{x}, got {}", out.get(x, 0));
        }
    }

    #[test]
    fn test_global_constant_image() {
        let img = Image::from_vec(10, 10, vec![128u8; 100]);
        let out = equalize_histogram(&img);
        let v = out.get(0, 0);
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(out.get(x, y), v);
            }
        }
    }

    #[test]
    fn test_global_low_contrast() {
        let w = 110;
        let h = 1;
        let mut img = Image::new(w, h);
        for x in 0..w {
            img.set(x, 0, (100 + x % 11) as u8);
        }
        let out = equalize_histogram(&img);
        let min_val = (0..w).map(|x| out.get(x, 0)).min().unwrap();
        let max_val = (0..w).map(|x| out.get(x, 0)).max().unwrap();
        assert!(
            max_val - min_val > 100,
            "range {min_val}..{max_val} not expanded enough"
        );
    }

    #[test]
    fn test_global_preserves_ordering() {
        let img = Image::from_vec(5, 1, vec![10, 50, 100, 150, 200]);
        let out = equalize_histogram(&img);
        for i in 1..5 {
            assert!(
                out.get(i, 0) >= out.get(i - 1, 0),
                "monotonicity violated at {i}"
            );
        }
    }

    #[test]
    fn test_global_output_range() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }
        let out = equalize_histogram(&img);
        assert_eq!(out.width(), 50);
        assert_eq!(out.height(), 50);
    }

    #[test]
    fn test_into_matches_alloc() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }

        let alloc = equalize_histogram(&img);
        let mut into = Image::new(1, 1);
        equalize_histogram_into(&img, &mut into);

        assert_eq!(alloc.width(), into.width());
        assert_eq!(alloc.height(), into.height());
        for y in 0..50 {
            for x in 0..50 {
                assert_eq!(alloc.get(x, y), into.get(x, y), "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_clahe_basic() {
        let w = 64;
        let h = 64;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, ((x * 4) % 256) as u8);
            }
        }
        let out = equalize_clahe(&img, 16, 2.0);
        assert_eq!(out.width(), w);
        assert_eq!(out.height(), h);
    }

    #[test]
    fn test_clahe_vs_global_on_bimodal() {
        let w = 64;
        let h = 32;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base = if x < w / 2 { 30 } else { 200 };
                let noise = ((x + y * 7) % 20) as u8;
                img.set(x, y, base + noise);
            }
        }

        let global = equalize_histogram(&img);
        let clahe = equalize_clahe(&img, 16, 2.0);

        let g_range = range(&global);
        let c_range = range(&clahe);
        assert!(g_range > 50, "global range too small: {g_range}");
        assert!(c_range > 50, "clahe range too small: {c_range}");
    }

    #[test]
    fn test_clahe_non_divisible() {
        let img = Image::from_vec(100, 75, vec![128u8; 100 * 75]);
        let out = equalize_clahe(&img, 16, 3.0);
        assert_eq!(out.width(), 100);
        assert_eq!(out.height(), 75);
    }

    #[test]
    fn test_apply_histeq_none() {
        let img = Image::from_vec(4, 4, vec![42u8; 16]);
        let out = apply_histeq(&img, HistEqMethod::None);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.get(x, y), 42);
            }
        }
    }

    #[test]
    fn test_apply_histeq_into_global() {
        let mut img = Image::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.set(x, y, ((x * 3 + y * 7) % 256) as u8);
            }
        }

        let alloc = apply_histeq(&img, HistEqMethod::Global);
        let mut into = Image::new(1, 1);
        apply_histeq_into(&img, HistEqMethod::Global, &mut into);

        for y in 0..50 {
            for x in 0..50 {
                assert_eq!(alloc.get(x, y), into.get(x, y));
            }
        }
    }

    fn range(img: &Image<u8>) -> u8 {
        let mut lo = 255u8;
        let mut hi = 0u8;
        for y in 0..img.height() {
            for x in 0..img.width() {
                let v = img.get(x, y);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi - lo
    }
}
