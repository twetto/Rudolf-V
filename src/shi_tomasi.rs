// shi_tomasi.rs — Shi-Tomasi corner detector.
//
// This is the detector behind "Good Features to Track" for KLT/LK tracking.
// It scores each pixel by the smaller eigenvalue of the local structure
// tensor, which is the weakest gradient direction in the patch. A high score
// means the LK Hessian is well-conditioned; a low but positive score can still
// be selected in weak texture regions by downstream top-N/grid logic.
//
// GPU-friendly dataflow:
//   1. Sobel gradients: Ix, Iy
//   2. Tensor products: Ix², Iy², Ix·Iy
//   3. Separable blur of each product
//   4. Per-pixel score: min eigenvalue of [[Sxx, Sxy], [Sxy, Syy]]
//   5. Threshold floor; NMS/top-N are handled by the frontend

use crate::convolution::gaussian_kernel_1d;
use crate::fast::Feature;
use crate::image::Image;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Shi-Tomasi corner detector.
pub struct ShiTomasiDetector {
    /// Minimum accepted Shi-Tomasi score.
    ///
    /// Treat this as a numerical floor, not the main selection policy.
    /// Spatial NMS and tile-deficit top-N selection should decide which weak
    /// regions get represented.
    pub threshold: f32,
    /// Half-size of the Gaussian window used for the structure tensor.
    /// block_size=1 -> 3x3 window, block_size=2 -> 5x5 window.
    pub block_size: usize,
}

impl ShiTomasiDetector {
    pub fn new(threshold: f32, block_size: usize) -> Self {
        ShiTomasiDetector {
            threshold,
            block_size,
        }
    }

    /// Compute the dense Shi-Tomasi response image.
    pub fn corner_response(&self, image: &Image<u8>) -> Image<f32> {
        let mut scratch = ShiTomasiScratch::new(image.width(), image.height());
        self.corner_response_with_scratch(image, &mut scratch)
    }

    pub fn detect(&self, image: &Image<u8>) -> Vec<Feature> {
        let mut scratch = ShiTomasiScratch::new(image.width(), image.height());
        self.detect_at_level_with_scratch(image, 0, &mut scratch)
    }

    pub fn detect_at_level(&self, image: &Image<u8>, level: usize) -> Vec<Feature> {
        let mut scratch = ShiTomasiScratch::new(image.width(), image.height());
        self.detect_at_level_with_scratch(image, level, &mut scratch)
    }

    pub fn detect_with_scratch(
        &self,
        image: &Image<u8>,
        scratch: &mut ShiTomasiScratch,
    ) -> Vec<Feature> {
        self.detect_at_level_with_scratch(image, 0, scratch)
    }

    pub fn detect_unoccupied_cell_nms(
        &self,
        image: &Image<u8>,
        grid: &[bool],
        grid_cols: usize,
        cell_size: usize,
        scratch: &mut ShiTomasiScratch,
    ) -> Vec<Feature> {
        let w = image.width();
        let h = image.height();

        let border = self.block_size + 2;
        if w <= 2 * border || h <= 2 * border {
            return Vec::new();
        }

        scratch.compute_cell_winners(
            image,
            self.block_size,
            border,
            self.threshold,
            grid,
            grid_cols,
            cell_size,
        )
    }

    pub fn detect_at_level_with_scratch(
        &self,
        image: &Image<u8>,
        level: usize,
        scratch: &mut ShiTomasiScratch,
    ) -> Vec<Feature> {
        let w = image.width();
        let h = image.height();

        let border = self.block_size + 2;
        if w <= 2 * border || h <= 2 * border {
            return Vec::new();
        }

        scratch.compute_response(image, self.block_size);
        let response = &scratch.response;
        let mut features = Vec::new();
        for y in border..(h - border) {
            for x in border..(w - border) {
                let score = response[y * w + x];
                if score > self.threshold {
                    features.push(Feature {
                        x: x as f32,
                        y: y as f32,
                        score,
                        level,
                        id: 0,
                        descriptor: 0,
                    });
                }
            }
        }

        features.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.x.total_cmp(&b.x))
        });
        features
    }

    pub fn corner_response_with_scratch(
        &self,
        image: &Image<u8>,
        scratch: &mut ShiTomasiScratch,
    ) -> Image<f32> {
        scratch.compute_response(image, self.block_size);
        Image::from_vec(image.width(), image.height(), scratch.response.clone())
    }
}

pub struct ShiTomasiScratch {
    width: usize,
    height: usize,
    ix2: Vec<f32>,
    iy2: Vec<f32>,
    ixiy: Vec<f32>,
    tmp_xx: Vec<f32>,
    tmp_yy: Vec<f32>,
    tmp_xy: Vec<f32>,
    response: Vec<f32>,
    cell_winners: Vec<Option<Feature>>,
}

impl ShiTomasiScratch {
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height;
        ShiTomasiScratch {
            width,
            height,
            ix2: vec![0.0; len],
            iy2: vec![0.0; len],
            ixiy: vec![0.0; len],
            tmp_xx: vec![0.0; len],
            tmp_yy: vec![0.0; len],
            tmp_xy: vec![0.0; len],
            response: vec![0.0; len],
            cell_winners: Vec::new(),
        }
    }

    fn ensure_size(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        let len = width * height;
        self.width = width;
        self.height = height;
        self.ix2.resize(len, 0.0);
        self.iy2.resize(len, 0.0);
        self.ixiy.resize(len, 0.0);
        self.tmp_xx.resize(len, 0.0);
        self.tmp_yy.resize(len, 0.0);
        self.tmp_xy.resize(len, 0.0);
        self.response.resize(len, 0.0);
    }

    fn compute_response(&mut self, image: &Image<u8>, block_size: usize) {
        let w = image.width();
        let h = image.height();
        self.ensure_size(w, h);
        if w == 0 || h == 0 {
            return;
        }

        let sigma = block_size as f32 * 0.5 + 0.5;
        let kernel = gaussian_kernel_1d(block_size, sigma);
        compute_tensor_products(image, &mut self.ix2, &mut self.iy2, &mut self.ixiy);
        blur_tensor_rows(
            w,
            h,
            &kernel,
            &self.ix2,
            &self.iy2,
            &self.ixiy,
            &mut self.tmp_xx,
            &mut self.tmp_yy,
            &mut self.tmp_xy,
        );
        blur_tensor_cols_score(
            w,
            h,
            &kernel,
            &self.tmp_xx,
            &self.tmp_yy,
            &self.tmp_xy,
            &mut self.response,
        );
    }

    fn compute_cell_winners(
        &mut self,
        image: &Image<u8>,
        block_size: usize,
        border: usize,
        threshold: f32,
        grid: &[bool],
        grid_cols: usize,
        cell_size: usize,
    ) -> Vec<Feature> {
        let w = image.width();
        let h = image.height();
        self.ensure_size(w, h);
        if w == 0 || h == 0 {
            return Vec::new();
        }

        let cell = cell_size.max(1);
        let cols = w.div_ceil(cell);
        let rows = h.div_ceil(cell);
        let n_cells = cols * rows;

        compute_tensor_products(image, &mut self.ix2, &mut self.iy2, &mut self.ixiy);
        let sigma = block_size as f32 * 0.5 + 0.5;
        let kernel = gaussian_kernel_1d(block_size, sigma);
        blur_tensor_rows(
            w,
            h,
            &kernel,
            &self.ix2,
            &self.iy2,
            &self.ixiy,
            &mut self.tmp_xx,
            &mut self.tmp_yy,
            &mut self.tmp_xy,
        );
        #[cfg(feature = "parallel")]
        {
            if self.cell_winners.len() != n_cells {
                self.cell_winners.clear();
                self.cell_winners.resize(n_cells, None);
            } else {
                for winner in &mut self.cell_winners {
                    *winner = None;
                }
            }

            blur_tensor_cols_collect(
                w,
                h,
                border,
                &kernel,
                &self.tmp_xx,
                &self.tmp_yy,
                &self.tmp_xy,
                threshold,
                grid,
                grid_cols,
                cell,
                cols,
                &mut self.cell_winners,
            );

            return self.cell_winners.iter().filter_map(Clone::clone).collect();
        }

        #[cfg(not(feature = "parallel"))]
        {
            if self.cell_winners.len() != n_cells {
                self.cell_winners.clear();
                self.cell_winners.resize(n_cells, None);
            } else {
                for winner in &mut self.cell_winners {
                    *winner = None;
                }
            }

            blur_tensor_cols_collect(
                w,
                h,
                border,
                &kernel,
                &self.tmp_xx,
                &self.tmp_yy,
                &self.tmp_xy,
                threshold,
                grid,
                grid_cols,
                cell,
                cols,
                &mut self.cell_winners,
            );

            self.cell_winners.iter().filter_map(Clone::clone).collect()
        }
    }
}

fn compute_tensor_products(image: &Image<u8>, ix2: &mut [f32], iy2: &mut [f32], ixiy: &mut [f32]) {
    let w = image.width();
    let h = image.height();
    let src = image.as_slice();
    let stride = image.stride();

    #[cfg(not(feature = "parallel"))]
    {
        compute_tensor_products_scalar(src, stride, w, h, ix2, iy2, ixiy);
    }

    #[cfg(feature = "parallel")]
    {
        ix2.par_chunks_mut(w)
            .zip(iy2.par_chunks_mut(w))
            .zip(ixiy.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((ix2_row, iy2_row), ixiy_row))| {
                compute_tensor_row(src, stride, w, h, y, ix2_row, iy2_row, ixiy_row);
            });
    }
}

#[cfg(not(feature = "parallel"))]
fn compute_tensor_products_scalar(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    ix2: &mut [f32],
    iy2: &mut [f32],
    ixiy: &mut [f32],
) {
    if w < 3 || h < 3 {
        for y in 0..h {
            for x in 0..w {
                tensor_at_clamped(src, stride, w, h, x, y, ix2, iy2, ixiy);
            }
        }
        return;
    }

    for x in 0..w {
        tensor_at_clamped(src, stride, w, h, x, 0, ix2, iy2, ixiy);
        tensor_at_clamped(src, stride, w, h, x, h - 1, ix2, iy2, ixiy);
    }
    for y in 1..(h - 1) {
        tensor_at_clamped(src, stride, w, h, 0, y, ix2, iy2, ixiy);
        tensor_at_clamped(src, stride, w, h, w - 1, y, ix2, iy2, ixiy);
    }

    for y in 1..(h - 1) {
        let row_m = (y - 1) * stride;
        let row_0 = y * stride;
        let row_p = (y + 1) * stride;
        let dst_row = y * w;

        for x in 1..(w - 1) {
            let p00 = src[row_m + x - 1] as f32;
            let p01 = src[row_m + x] as f32;
            let p02 = src[row_m + x + 1] as f32;
            let p10 = src[row_0 + x - 1] as f32;
            let p12 = src[row_0 + x + 1] as f32;
            let p20 = src[row_p + x - 1] as f32;
            let p21 = src[row_p + x] as f32;
            let p22 = src[row_p + x + 1] as f32;

            let gx = (p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20);
            let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
            let idx = dst_row + x;
            ix2[idx] = gx * gx;
            iy2[idx] = gy * gy;
            ixiy[idx] = gx * gy;
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn tensor_at_clamped(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    ix2: &mut [f32],
    iy2: &mut [f32],
    ixiy: &mut [f32],
) {
    let (gx, gy) = sobel_at_clamped(src, stride, w, h, x, y);
    let idx = y * w + x;
    ix2[idx] = gx * gx;
    iy2[idx] = gy * gy;
    ixiy[idx] = gx * gy;
}

#[cfg(feature = "parallel")]
fn compute_tensor_row(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    y: usize,
    ix2_row: &mut [f32],
    iy2_row: &mut [f32],
    ixiy_row: &mut [f32],
) {
    if w < 3 || h < 3 {
        for x in 0..w {
            let (gx, gy) = sobel_at_clamped(src, stride, w, h, x, y);
            ix2_row[x] = gx * gx;
            iy2_row[x] = gy * gy;
            ixiy_row[x] = gx * gy;
        }
        return;
    }

    if y == 0 || y == h - 1 {
        for x in 0..w {
            let (gx, gy) = sobel_at_clamped(src, stride, w, h, x, y);
            ix2_row[x] = gx * gx;
            iy2_row[x] = gy * gy;
            ixiy_row[x] = gx * gy;
        }
        return;
    }

    let (gx0, gy0) = sobel_at_clamped(src, stride, w, h, 0, y);
    ix2_row[0] = gx0 * gx0;
    iy2_row[0] = gy0 * gy0;
    ixiy_row[0] = gx0 * gy0;

    let row_m = (y - 1) * stride;
    let row_0 = y * stride;
    let row_p = (y + 1) * stride;

    for x in 1..(w - 1) {
        let p00 = src[row_m + x - 1] as f32;
        let p01 = src[row_m + x] as f32;
        let p02 = src[row_m + x + 1] as f32;
        let p10 = src[row_0 + x - 1] as f32;
        let p12 = src[row_0 + x + 1] as f32;
        let p20 = src[row_p + x - 1] as f32;
        let p21 = src[row_p + x] as f32;
        let p22 = src[row_p + x + 1] as f32;

        let gx = (p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20);
        let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
        ix2_row[x] = gx * gx;
        iy2_row[x] = gy * gy;
        ixiy_row[x] = gx * gy;
    }

    let last = w - 1;
    let (gxl, gyl) = sobel_at_clamped(src, stride, w, h, last, y);
    ix2_row[last] = gxl * gxl;
    iy2_row[last] = gyl * gyl;
    ixiy_row[last] = gxl * gyl;
}

fn sobel_at_clamped(
    src: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    x: usize,
    y: usize,
) -> (f32, f32) {
    let ym = y.saturating_sub(1);
    let yp = (y + 1).min(h - 1);
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(w - 1);

    let row_m = ym * stride;
    let row_0 = y * stride;
    let row_p = yp * stride;

    let p00 = src[row_m + xm] as f32;
    let p01 = src[row_m + x] as f32;
    let p02 = src[row_m + xp] as f32;
    let p10 = src[row_0 + xm] as f32;
    let p12 = src[row_0 + xp] as f32;
    let p20 = src[row_p + xm] as f32;
    let p21 = src[row_p + x] as f32;
    let p22 = src[row_p + xp] as f32;

    let gx = (p02 - p00) + 2.0 * (p12 - p10) + (p22 - p20);
    let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);
    (gx, gy)
}

fn blur_tensor_rows(
    w: usize,
    h: usize,
    kernel: &[f32],
    ix2: &[f32],
    iy2: &[f32],
    ixiy: &[f32],
    tmp_xx: &mut [f32],
    tmp_yy: &mut [f32],
    tmp_xy: &mut [f32],
) {
    if kernel.len() == 5 {
        blur_tensor_rows_5(w, h, kernel, ix2, iy2, ixiy, tmp_xx, tmp_yy, tmp_xy);
        return;
    }

    let half = kernel.len() / 2;
    for y in 0..h {
        let row = y * w;
        for x in 0..half.min(w) {
            let mut xx = 0.0;
            let mut yy = 0.0;
            let mut xy = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = ((x as isize) + (ki as isize) - (half as isize)).clamp(0, (w - 1) as isize)
                    as usize;
                let idx = row + sx;
                xx += ix2[idx] * kv;
                yy += iy2[idx] * kv;
                xy += ixiy[idx] * kv;
            }
            let idx = row + x;
            tmp_xx[idx] = xx;
            tmp_yy[idx] = yy;
            tmp_xy[idx] = xy;
        }

        if w > 2 * half {
            for x in half..(w - half) {
                let mut xx = 0.0;
                let mut yy = 0.0;
                let mut xy = 0.0;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let idx = row + x + ki - half;
                    xx += ix2[idx] * kv;
                    yy += iy2[idx] * kv;
                    xy += ixiy[idx] * kv;
                }
                let idx = row + x;
                tmp_xx[idx] = xx;
                tmp_yy[idx] = yy;
                tmp_xy[idx] = xy;
            }
        }

        let right_start = if w > half { w - half } else { half.min(w) };
        for x in right_start..w {
            let mut xx = 0.0;
            let mut yy = 0.0;
            let mut xy = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = ((x as isize) + (ki as isize) - (half as isize)).clamp(0, (w - 1) as isize)
                    as usize;
                let idx = row + sx;
                xx += ix2[idx] * kv;
                yy += iy2[idx] * kv;
                xy += ixiy[idx] * kv;
            }
            let idx = row + x;
            tmp_xx[idx] = xx;
            tmp_yy[idx] = yy;
            tmp_xy[idx] = xy;
        }
    }
}

fn blur_tensor_rows_5(
    w: usize,
    h: usize,
    kernel: &[f32],
    ix2: &[f32],
    iy2: &[f32],
    ixiy: &[f32],
    tmp_xx: &mut [f32],
    tmp_yy: &mut [f32],
    tmp_xy: &mut [f32],
) {
    if w == 0 {
        return;
    }

    debug_assert_eq!(tmp_xx.len(), h * w);
    debug_assert_eq!(tmp_yy.len(), h * w);
    debug_assert_eq!(tmp_xy.len(), h * w);

    let k0 = kernel[0];
    let k1 = kernel[1];
    let k2 = kernel[2];
    let k3 = kernel[3];
    let k4 = kernel[4];

    #[cfg(feature = "parallel")]
    {
        tmp_xx
            .par_chunks_mut(w)
            .zip(tmp_yy.par_chunks_mut(w))
            .zip(tmp_xy.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((tmp_xx_row, tmp_yy_row), tmp_xy_row))| {
                let row = y * w;
                blur_tensor_row_5(
                    w, row, k0, k1, k2, k3, k4, ix2, iy2, ixiy, tmp_xx_row, tmp_yy_row, tmp_xy_row,
                );
            });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..h {
            let row = y * w;

            for x in 0..2.min(w) {
                let x0 = x.saturating_sub(2);
                let x1 = x.saturating_sub(1);
                let x3 = (x + 1).min(w - 1);
                let x4 = (x + 2).min(w - 1);
                let idx = row + x;
                let i0 = row + x0;
                let i1 = row + x1;
                let i2 = row + x;
                let i3 = row + x3;
                let i4 = row + x4;
                tmp_xx[idx] =
                    ix2[i0] * k0 + ix2[i1] * k1 + ix2[i2] * k2 + ix2[i3] * k3 + ix2[i4] * k4;
                tmp_yy[idx] =
                    iy2[i0] * k0 + iy2[i1] * k1 + iy2[i2] * k2 + iy2[i3] * k3 + iy2[i4] * k4;
                tmp_xy[idx] =
                    ixiy[i0] * k0 + ixiy[i1] * k1 + ixiy[i2] * k2 + ixiy[i3] * k3 + ixiy[i4] * k4;
            }

            if w > 4 {
                for x in 2..(w - 2) {
                    let idx = row + x;
                    tmp_xx[idx] = ix2[idx - 2] * k0
                        + ix2[idx - 1] * k1
                        + ix2[idx] * k2
                        + ix2[idx + 1] * k3
                        + ix2[idx + 2] * k4;
                    tmp_yy[idx] = iy2[idx - 2] * k0
                        + iy2[idx - 1] * k1
                        + iy2[idx] * k2
                        + iy2[idx + 1] * k3
                        + iy2[idx + 2] * k4;
                    tmp_xy[idx] = ixiy[idx - 2] * k0
                        + ixiy[idx - 1] * k1
                        + ixiy[idx] * k2
                        + ixiy[idx + 1] * k3
                        + ixiy[idx + 2] * k4;
                }
            }

            let right_start = if w > 2 { w - 2 } else { 2.min(w) };
            for x in right_start..w {
                let x0 = x.saturating_sub(2);
                let x1 = x.saturating_sub(1);
                let x3 = (x + 1).min(w - 1);
                let x4 = (x + 2).min(w - 1);
                let idx = row + x;
                let i0 = row + x0;
                let i1 = row + x1;
                let i2 = row + x;
                let i3 = row + x3;
                let i4 = row + x4;
                tmp_xx[idx] =
                    ix2[i0] * k0 + ix2[i1] * k1 + ix2[i2] * k2 + ix2[i3] * k3 + ix2[i4] * k4;
                tmp_yy[idx] =
                    iy2[i0] * k0 + iy2[i1] * k1 + iy2[i2] * k2 + iy2[i3] * k3 + iy2[i4] * k4;
                tmp_xy[idx] =
                    ixiy[i0] * k0 + ixiy[i1] * k1 + ixiy[i2] * k2 + ixiy[i3] * k3 + ixiy[i4] * k4;
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn blur_tensor_row_5(
    w: usize,
    row: usize,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    ix2: &[f32],
    iy2: &[f32],
    ixiy: &[f32],
    tmp_xx: &mut [f32],
    tmp_yy: &mut [f32],
    tmp_xy: &mut [f32],
) {
    for x in 0..2.min(w) {
        let x0 = x.saturating_sub(2);
        let x1 = x.saturating_sub(1);
        let x3 = (x + 1).min(w - 1);
        let x4 = (x + 2).min(w - 1);
        let i0 = row + x0;
        let i1 = row + x1;
        let i2 = row + x;
        let i3 = row + x3;
        let i4 = row + x4;
        tmp_xx[x] = ix2[i0] * k0 + ix2[i1] * k1 + ix2[i2] * k2 + ix2[i3] * k3 + ix2[i4] * k4;
        tmp_yy[x] = iy2[i0] * k0 + iy2[i1] * k1 + iy2[i2] * k2 + iy2[i3] * k3 + iy2[i4] * k4;
        tmp_xy[x] = ixiy[i0] * k0 + ixiy[i1] * k1 + ixiy[i2] * k2 + ixiy[i3] * k3 + ixiy[i4] * k4;
    }

    if w > 4 {
        for x in 2..(w - 2) {
            let idx = row + x;
            tmp_xx[x] = ix2[idx - 2] * k0
                + ix2[idx - 1] * k1
                + ix2[idx] * k2
                + ix2[idx + 1] * k3
                + ix2[idx + 2] * k4;
            tmp_yy[x] = iy2[idx - 2] * k0
                + iy2[idx - 1] * k1
                + iy2[idx] * k2
                + iy2[idx + 1] * k3
                + iy2[idx + 2] * k4;
            tmp_xy[x] = ixiy[idx - 2] * k0
                + ixiy[idx - 1] * k1
                + ixiy[idx] * k2
                + ixiy[idx + 1] * k3
                + ixiy[idx + 2] * k4;
        }
    }

    let right_start = if w > 2 { w - 2 } else { 2.min(w) };
    for x in right_start..w {
        let x0 = x.saturating_sub(2);
        let x1 = x.saturating_sub(1);
        let x3 = (x + 1).min(w - 1);
        let x4 = (x + 2).min(w - 1);
        let i0 = row + x0;
        let i1 = row + x1;
        let i2 = row + x;
        let i3 = row + x3;
        let i4 = row + x4;
        tmp_xx[x] = ix2[i0] * k0 + ix2[i1] * k1 + ix2[i2] * k2 + ix2[i3] * k3 + ix2[i4] * k4;
        tmp_yy[x] = iy2[i0] * k0 + iy2[i1] * k1 + iy2[i2] * k2 + iy2[i3] * k3 + iy2[i4] * k4;
        tmp_xy[x] = ixiy[i0] * k0 + ixiy[i1] * k1 + ixiy[i2] * k2 + ixiy[i3] * k3 + ixiy[i4] * k4;
    }
}

fn blur_tensor_cols_score(
    w: usize,
    h: usize,
    kernel: &[f32],
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    response: &mut [f32],
) {
    let half = kernel.len() / 2;
    for y in 0..h {
        for x in 0..w {
            let mut xx = 0.0;
            let mut yy = 0.0;
            let mut xy = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = ((y as isize) + (ki as isize) - (half as isize)).clamp(0, (h - 1) as isize)
                    as usize;
                let idx = sy * w + x;
                xx += tmp_xx[idx] * kv;
                yy += tmp_yy[idx] * kv;
                xy += tmp_xy[idx] * kv;
            }
            response[y * w + x] = min_eigenvalue_2x2(xx, yy, xy);
        }
    }
}

fn blur_tensor_cols_collect_5(
    w: usize,
    h: usize,
    border: usize,
    kernel: &[f32],
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    threshold: f32,
    grid: &[bool],
    grid_cols: usize,
    cell_size: usize,
    cell_cols: usize,
    cell_winners: &mut [Option<Feature>],
) {
    #[cfg(feature = "parallel")]
    {
        cell_winners
            .par_chunks_mut(cell_cols)
            .enumerate()
            .for_each(|(cell_row, row_winners)| {
                let y_start = border.max(cell_row * cell_size);
                let y_end = (h - border).min((cell_row + 1) * cell_size);
                for y in y_start..y_end {
                    collect_row_5_cell_row(
                        w,
                        y,
                        border,
                        kernel,
                        tmp_xx,
                        tmp_yy,
                        tmp_xy,
                        threshold,
                        grid,
                        grid_cols,
                        cell_size,
                        cell_row,
                        row_winners,
                    );
                }
            });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    {
        let cell_rows = cell_winners.len() / cell_cols;
        for y in border..(h - border) {
            collect_row_5(
                w,
                y,
                border,
                kernel,
                tmp_xx,
                tmp_yy,
                tmp_xy,
                threshold,
                grid,
                grid_cols,
                cell_size,
                cell_cols,
                cell_rows,
                cell_winners,
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn collect_row_5_cell_row(
    w: usize,
    y: usize,
    border: usize,
    kernel: &[f32],
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    threshold: f32,
    grid: &[bool],
    grid_cols: usize,
    cell_size: usize,
    cell_row: usize,
    row_winners: &mut [Option<Feature>],
) {
    let k0 = kernel[0];
    let k1 = kernel[1];
    let k2 = kernel[2];
    let k3 = kernel[3];
    let k4 = kernel[4];

    let grid_row = if grid.is_empty() {
        0
    } else {
        cell_row * grid_cols
    };

    let row0 = (y - 2) * w;
    let row1 = (y - 1) * w;
    let row2 = y * w;
    let row3 = (y + 1) * w;
    let row4 = (y + 2) * w;

    let x_limit = w - border;
    let first_cell_col = border / cell_size;
    let last_cell_col = x_limit.div_ceil(cell_size).min(row_winners.len());
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2");

    for cell_col in first_cell_col..last_cell_col {
        if !grid.is_empty() && cell_col < grid_cols && grid[grid_row + cell_col] {
            continue;
        }

        let x_start = border.max(cell_col * cell_size);
        let x_end = x_limit.min((cell_col + 1) * cell_size);
        let winner = &mut row_winners[cell_col];

        #[cfg(target_arch = "x86_64")]
        {
            let min_score = winner
                .as_ref()
                .map_or(threshold, |prev| prev.score.max(threshold));
            if avx2 && x_start + 8 <= x_end {
                if let Some((score, x)) = unsafe {
                    collect_cell_5_avx2(
                        x_start, x_end, row0, row1, row2, row3, row4, k0, k1, k2, k3, k4, tmp_xx,
                        tmp_yy, tmp_xy, min_score,
                    )
                } {
                    *winner = Some(Feature {
                        x: x as f32,
                        y: y as f32,
                        score,
                        level: 0,
                        id: 0,
                        descriptor: 0,
                    });
                }
                continue;
            }
        }

        for x in x_start..x_end {
            let i0 = row0 + x;
            let i1 = row1 + x;
            let i2 = row2 + x;
            let i3 = row3 + x;
            let i4 = row4 + x;
            let xx = tmp_xx[i0] * k0
                + tmp_xx[i1] * k1
                + tmp_xx[i2] * k2
                + tmp_xx[i3] * k3
                + tmp_xx[i4] * k4;
            let yy = tmp_yy[i0] * k0
                + tmp_yy[i1] * k1
                + tmp_yy[i2] * k2
                + tmp_yy[i3] * k3
                + tmp_yy[i4] * k4;
            let xy = tmp_xy[i0] * k0
                + tmp_xy[i1] * k1
                + tmp_xy[i2] * k2
                + tmp_xy[i3] * k3
                + tmp_xy[i4] * k4;

            let score = min_eigenvalue_2x2(xx, yy, xy);
            if score <= threshold {
                continue;
            }

            if winner.as_ref().map_or(true, |prev| score > prev.score) {
                *winner = Some(Feature {
                    x: x as f32,
                    y: y as f32,
                    score,
                    level: 0,
                    id: 0,
                    descriptor: 0,
                });
            }
        }
    }
}

#[cfg(not(feature = "parallel"))]
#[allow(clippy::too_many_arguments)]
fn collect_row_5(
    w: usize,
    y: usize,
    border: usize,
    kernel: &[f32],
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    threshold: f32,
    grid: &[bool],
    grid_cols: usize,
    cell_size: usize,
    cell_cols: usize,
    cell_rows: usize,
    cell_winners: &mut [Option<Feature>],
) {
    let k0 = kernel[0];
    let k1 = kernel[1];
    let k2 = kernel[2];
    let k3 = kernel[3];
    let k4 = kernel[4];

    let cell_row = y / cell_size;
    if cell_row >= cell_rows {
        return;
    }
    let grid_row = if grid.is_empty() {
        0
    } else {
        cell_row * grid_cols
    };

    let row0 = (y - 2) * w;
    let row1 = (y - 1) * w;
    let row2 = y * w;
    let row3 = (y + 1) * w;
    let row4 = (y + 2) * w;

    let x_limit = w - border;
    let first_cell_col = border / cell_size;
    let last_cell_col = x_limit.div_ceil(cell_size).min(cell_cols);
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::arch::is_x86_feature_detected!("avx2");

    for cell_col in first_cell_col..last_cell_col {
        if !grid.is_empty() && cell_col < grid_cols && grid[grid_row + cell_col] {
            continue;
        }

        let x_start = border.max(cell_col * cell_size);
        let x_end = x_limit.min((cell_col + 1) * cell_size);
        let cell_idx = cell_row * cell_cols + cell_col;
        debug_assert!(cell_idx < cell_winners.len());
        let winner = unsafe { cell_winners.get_unchecked_mut(cell_idx) };

        #[cfg(target_arch = "x86_64")]
        {
            let min_score = winner
                .as_ref()
                .map_or(threshold, |prev| prev.score.max(threshold));
            if avx2 && x_start + 8 <= x_end {
                if let Some((score, x)) = unsafe {
                    collect_cell_5_avx2(
                        x_start, x_end, row0, row1, row2, row3, row4, k0, k1, k2, k3, k4, tmp_xx,
                        tmp_yy, tmp_xy, min_score,
                    )
                } {
                    *winner = Some(Feature {
                        x: x as f32,
                        y: y as f32,
                        score,
                        level: 0,
                        id: 0,
                        descriptor: 0,
                    });
                }
                continue;
            }
        }

        for x in x_start..x_end {
            let i0 = row0 + x;
            let i1 = row1 + x;
            let i2 = row2 + x;
            let i3 = row3 + x;
            let i4 = row4 + x;
            debug_assert!(i4 < tmp_xx.len());
            debug_assert!(i4 < tmp_yy.len());
            debug_assert!(i4 < tmp_xy.len());
            let xx = unsafe {
                *tmp_xx.get_unchecked(i0) * k0
                    + *tmp_xx.get_unchecked(i1) * k1
                    + *tmp_xx.get_unchecked(i2) * k2
                    + *tmp_xx.get_unchecked(i3) * k3
                    + *tmp_xx.get_unchecked(i4) * k4
            };
            let yy = unsafe {
                *tmp_yy.get_unchecked(i0) * k0
                    + *tmp_yy.get_unchecked(i1) * k1
                    + *tmp_yy.get_unchecked(i2) * k2
                    + *tmp_yy.get_unchecked(i3) * k3
                    + *tmp_yy.get_unchecked(i4) * k4
            };
            let xy = unsafe {
                *tmp_xy.get_unchecked(i0) * k0
                    + *tmp_xy.get_unchecked(i1) * k1
                    + *tmp_xy.get_unchecked(i2) * k2
                    + *tmp_xy.get_unchecked(i3) * k3
                    + *tmp_xy.get_unchecked(i4) * k4
            };

            let score = min_eigenvalue_2x2(xx, yy, xy);
            if score <= threshold {
                continue;
            }

            if winner.as_ref().map_or(true, |prev| score > prev.score) {
                *winner = Some(Feature {
                    x: x as f32,
                    y: y as f32,
                    score,
                    level: 0,
                    id: 0,
                    descriptor: 0,
                });
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn collect_cell_5_avx2(
    x_start: usize,
    x_end: usize,
    row0: usize,
    row1: usize,
    row2: usize,
    row3: usize,
    row4: usize,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
    k4: f32,
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    min_score: f32,
) -> Option<(f32, usize)> {
    use std::arch::x86_64::*;

    let xx_ptr = tmp_xx.as_ptr();
    let yy_ptr = tmp_yy.as_ptr();
    let xy_ptr = tmp_xy.as_ptr();

    let k0v = _mm256_set1_ps(k0);
    let k1v = _mm256_set1_ps(k1);
    let k2v = _mm256_set1_ps(k2);
    let k3v = _mm256_set1_ps(k3);
    let k4v = _mm256_set1_ps(k4);
    let four = _mm256_set1_ps(4.0);
    let half = _mm256_set1_ps(0.5);
    let zero = _mm256_setzero_ps();

    let mut best_score = _mm256_set1_ps(min_score);
    let mut best_x = _mm256_set1_epi32(-1);
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    let mut x = x_start;
    while x + 8 <= x_end {
        let base_x = _mm256_add_epi32(_mm256_set1_epi32(x as i32), lane);

        let xx = {
            let a0 = _mm256_loadu_ps(xx_ptr.add(row0 + x));
            let a1 = _mm256_loadu_ps(xx_ptr.add(row1 + x));
            let a2 = _mm256_loadu_ps(xx_ptr.add(row2 + x));
            let a3 = _mm256_loadu_ps(xx_ptr.add(row3 + x));
            let a4 = _mm256_loadu_ps(xx_ptr.add(row4 + x));
            _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(a0, k0v), _mm256_mul_ps(a1, k1v)),
                _mm256_add_ps(
                    _mm256_mul_ps(a2, k2v),
                    _mm256_add_ps(_mm256_mul_ps(a3, k3v), _mm256_mul_ps(a4, k4v)),
                ),
            )
        };
        let yy = {
            let a0 = _mm256_loadu_ps(yy_ptr.add(row0 + x));
            let a1 = _mm256_loadu_ps(yy_ptr.add(row1 + x));
            let a2 = _mm256_loadu_ps(yy_ptr.add(row2 + x));
            let a3 = _mm256_loadu_ps(yy_ptr.add(row3 + x));
            let a4 = _mm256_loadu_ps(yy_ptr.add(row4 + x));
            _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(a0, k0v), _mm256_mul_ps(a1, k1v)),
                _mm256_add_ps(
                    _mm256_mul_ps(a2, k2v),
                    _mm256_add_ps(_mm256_mul_ps(a3, k3v), _mm256_mul_ps(a4, k4v)),
                ),
            )
        };
        let xy = {
            let a0 = _mm256_loadu_ps(xy_ptr.add(row0 + x));
            let a1 = _mm256_loadu_ps(xy_ptr.add(row1 + x));
            let a2 = _mm256_loadu_ps(xy_ptr.add(row2 + x));
            let a3 = _mm256_loadu_ps(xy_ptr.add(row3 + x));
            let a4 = _mm256_loadu_ps(xy_ptr.add(row4 + x));
            _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(a0, k0v), _mm256_mul_ps(a1, k1v)),
                _mm256_add_ps(
                    _mm256_mul_ps(a2, k2v),
                    _mm256_add_ps(_mm256_mul_ps(a3, k3v), _mm256_mul_ps(a4, k4v)),
                ),
            )
        };

        let trace = _mm256_add_ps(xx, yy);
        let diff = _mm256_sub_ps(xx, yy);
        let discr = _mm256_sqrt_ps(_mm256_max_ps(
            _mm256_add_ps(
                _mm256_mul_ps(diff, diff),
                _mm256_mul_ps(four, _mm256_mul_ps(xy, xy)),
            ),
            zero,
        ));
        let score = _mm256_mul_ps(half, _mm256_sub_ps(trace, discr));
        let replace = _mm256_cmp_ps(score, best_score, _CMP_GT_OQ);

        best_score = _mm256_blendv_ps(best_score, score, replace);
        best_x = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(best_x),
            _mm256_castsi256_ps(base_x),
            replace,
        ));

        x += 8;
    }

    let mut scores = [0.0f32; 8];
    let mut xs = [0i32; 8];
    _mm256_storeu_ps(scores.as_mut_ptr(), best_score);
    _mm256_storeu_si256(xs.as_mut_ptr() as *mut __m256i, best_x);

    let mut out_score = min_score;
    let mut out_x = usize::MAX;
    for lane in 0..8 {
        if xs[lane] >= 0 && scores[lane] > out_score {
            out_score = scores[lane];
            out_x = xs[lane] as usize;
        }
    }

    while x < x_end {
        let i0 = row0 + x;
        let i1 = row1 + x;
        let i2 = row2 + x;
        let i3 = row3 + x;
        let i4 = row4 + x;
        let xx = *tmp_xx.get_unchecked(i0) * k0
            + *tmp_xx.get_unchecked(i1) * k1
            + *tmp_xx.get_unchecked(i2) * k2
            + *tmp_xx.get_unchecked(i3) * k3
            + *tmp_xx.get_unchecked(i4) * k4;
        let yy = *tmp_yy.get_unchecked(i0) * k0
            + *tmp_yy.get_unchecked(i1) * k1
            + *tmp_yy.get_unchecked(i2) * k2
            + *tmp_yy.get_unchecked(i3) * k3
            + *tmp_yy.get_unchecked(i4) * k4;
        let xy = *tmp_xy.get_unchecked(i0) * k0
            + *tmp_xy.get_unchecked(i1) * k1
            + *tmp_xy.get_unchecked(i2) * k2
            + *tmp_xy.get_unchecked(i3) * k3
            + *tmp_xy.get_unchecked(i4) * k4;
        let score = min_eigenvalue_2x2(xx, yy, xy);
        if score > out_score {
            out_score = score;
            out_x = x;
        }
        x += 1;
    }

    (out_x != usize::MAX).then_some((out_score, out_x))
}

fn blur_tensor_cols_collect(
    w: usize,
    h: usize,
    border: usize,
    kernel: &[f32],
    tmp_xx: &[f32],
    tmp_yy: &[f32],
    tmp_xy: &[f32],
    threshold: f32,
    grid: &[bool],
    grid_cols: usize,
    cell_size: usize,
    cell_cols: usize,
    cell_winners: &mut [Option<Feature>],
) {
    if kernel.len() == 5 {
        blur_tensor_cols_collect_5(
            w,
            h,
            border,
            kernel,
            tmp_xx,
            tmp_yy,
            tmp_xy,
            threshold,
            grid,
            grid_cols,
            cell_size,
            cell_cols,
            cell_winners,
        );
        return;
    }

    let half = kernel.len() / 2;
    for y in border..(h - border) {
        let cell_row = y / cell_size;
        let grid_row = if grid.is_empty() {
            0
        } else {
            cell_row * grid_cols
        };

        for x in border..(w - border) {
            let cell_col = x / cell_size;
            if !grid.is_empty() && cell_col < grid_cols && grid[grid_row + cell_col] {
                continue;
            }

            let mut xx = 0.0;
            let mut yy = 0.0;
            let mut xy = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = y + ki - half;
                let idx = sy * w + x;
                xx += tmp_xx[idx] * kv;
                yy += tmp_yy[idx] * kv;
                xy += tmp_xy[idx] * kv;
            }

            let score = min_eigenvalue_2x2(xx, yy, xy);
            if score <= threshold {
                continue;
            }

            let cell_idx = cell_row * cell_cols + cell_col;
            let replace = match &cell_winners[cell_idx] {
                None => true,
                Some(prev) => {
                    score > prev.score
                        || (score == prev.score && (y as f32, x as f32) < (prev.y, prev.x))
                }
            };
            if replace {
                cell_winners[cell_idx] = Some(Feature {
                    x: x as f32,
                    y: y as f32,
                    score,
                    level: 0,
                    id: 0,
                    descriptor: 0,
                });
            }
        }
    }
}

#[inline]
fn min_eigenvalue_2x2(a: f32, b: f32, c: f32) -> f32 {
    let trace = a + b;
    let diff = a - b;
    let discr = (diff * diff + 4.0 * c * c).max(0.0).sqrt();
    0.5 * (trace - discr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nms::OccupancyNms;

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
    fn rejects_straight_edge() {
        let mut img = Image::new(60, 60);
        for y in 0..60 {
            for x in 0..60 {
                img.set(x, y, if x < 30 { 40 } else { 220 });
            }
        }

        let det = ShiTomasiDetector::new(1e5, 2);
        let features = det.detect(&img);
        assert!(
            features.len() < 5,
            "straight edge produced too many Shi-Tomasi features: {}",
            features.len()
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
}
