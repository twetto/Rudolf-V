// sgbm.rs — Semi-global block matching on one pyramid level of a rectified
// pair, as OpenCV's StereoSGBM computes it in MODE_HH (Hirschmüller 2008,
// eight paths): the CPU reference for gpu/sgbm.rs, which computes the same
// integers.
//
//   1. Both images' f32 pyramids (as DenseStereo's), down to `level`.
//   2. Prefilter: the x-derivative (3×3 Sobel), clamped to ±pre_filter_cap
//      and offset by it (OpenCV's prefilterXSobel); the first and last
//      columns read no derivative, the first and last rows repeat.
//   3. Pixel cost of disparity d at (x, y): the Birchfield–Tomasi
//      dissimilarity of left (x, y) and right (x − d, y) (each against the
//      other's half-pixel neighbourhood), on the prefiltered images plus a
//      quarter of it on the images themselves (OpenCV's calcPixelCostBT),
//      rounded. A disparity that puts x − d left of the right image costs
//      halfway between the pixel's least and mean cost inside it. Near the
//      left border a near surface's disparity is there; priced at the mean
//      (or with OpenCV's padding), a textureless near surface lost to some
//      far match inside and read too far; priced at the least, every pixel
//      within the disparity range of the border tied with it and lost its
//      depth. On a ZED walk (160x90) halfway kept 87 % of the SDK's pixels
//      (mean: 89 %) and cut errors over 25 % at 0.3-0.5 m from 29 % to 10 %.
//      (OpenCV has no cost there at all.)
//   4. Block cost: the pixel costs summed over a block_size² window, the
//      border repeated.
//   5. Path costs along `paths` directions (rows and columns; with 8 the
//      diagonals too):
//        L(p, d) = C(p, d) + min(L(q, d), L(q, d ± 1) + P1, min L(q) + P2)
//                  − min L(q),
//      q the pixel before p on the path; S(p, d) their sum.
//   6. Each pixel's disparity is S's minimum if every disparity more than one
//      from it sums at least 100 / (100 − uniqueness) times as much, refined
//      by a parabola through its neighbours, and inside the right image.
//   7. Left-right check: each right pixel keeps the left pixel that matches it
//      best (least S, then least disparity); a left pixel whose right pixels
//      on both sides of its disparity have best matches more than
//      disp12_max_diff from it has no depth (OpenCV's test).
//   8. Speckles: connected regions whose disparities step by at most a pixel,
//      smaller than speckle_size full-resolution pixels, have no depth (as
//      DenseStereo's).
//   9. A DenseDepth on the level's grid: range f·B / (d·2^level) · |K⁻¹(u, v, 1)|.
//
// Costs are integers: S stays under 2^24 (block costs under 25 · 190 plus P2,
// eight paths), so each pixel's best (S, d) packs into a u32 for the
// left-right check, and the GPU's sums are exact.

use crate::camera::StereoRig;
use crate::dense_stereo::{DenseDepth, NO_SOURCE, SpeckleScratch, rectified_sign, remove_speckles};
use crate::image::Image;
use crate::pyramid::Pyramid;

#[derive(Debug, Clone)]
pub struct SgbmConfig {
    /// Pyramid level matched: the depth map's pixel (x, y) is image pixel
    /// (x·2^level, y·2^level).
    pub level: usize,
    /// Nearest range searched (metres): sets the number of disparities.
    pub min_range: f64,
    /// Farthest range kept (metres).
    pub max_range: f64,
    /// Side of the block the pixel costs are summed over (odd).
    pub block_size: usize,
    /// Penalties for the disparity changing by one, and by more, between
    /// neighbours on a path (OpenCV's stereo_match sample: 8 and 32 ×
    /// block_size²).
    pub p1: u32,
    pub p2: u32,
    /// The prefiltered derivative is clamped to ±this.
    pub pre_filter_cap: u32,
    /// Margin (percent) by which the best disparity must beat every other
    /// more than one away.
    pub uniqueness: u32,
    /// Left-right check tolerance (pixels of the level); None skips the check.
    pub disp12_max_diff: Option<u32>,
    /// Drop regions of consistent disparity smaller than this many
    /// full-resolution pixels (as DenseStereoConfig::speckle_size); 0 keeps
    /// them.
    pub speckle_size: usize,
    /// Path directions: 4 (rows and columns) or 8 (and diagonals).
    pub paths: usize,
    /// Disparity standard deviation (pixels of the level) behind each
    /// pixel's log-range variance.
    pub disparity_sigma: f32,
}

impl Default for SgbmConfig {
    fn default() -> Self {
        let block_size = 5;
        Self {
            level: 3,
            min_range: 0.25,
            max_range: 30.0,
            block_size,
            p1: 8 * (block_size * block_size) as u32,
            p2: 32 * (block_size * block_size) as u32,
            pre_filter_cap: 63,
            uniqueness: 10,
            disp12_max_diff: Some(1),
            speckle_size: 400,
            paths: 8,
            disparity_sigma: 0.5,
        }
    }
}

/// Path directions (dx, dy): rows, columns, then diagonals.
pub(crate) const DIRS: [(isize, isize); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
];

/// A disparity map pixel without a disparity.
pub(crate) const NO_MATCH: u32 = u32::MAX;

/// Per pixel and image (prefiltered, then grey): its value and the least
/// and greatest of it and its half-pixel neighbours along the row.
pub(crate) type PixelSpan = [f32; 6];

/// One level's half-pixel spans of an image (step 2 and 3's inputs).
pub(crate) fn spans(img: &Image<f32>, cap: f32, out: &mut Vec<PixelSpan>) {
    let (w, h) = (img.width(), img.height());
    out.clear();
    out.resize(w * h, [0.0; 6]);
    let at = |x: usize, y: usize| img.get(x, y);
    let mut pref = vec![0.0f32; w];
    let mut grey = vec![0.0f32; w];
    for y in 0..h {
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            grey[x] = at(x, y);
            pref[x] = if x == 0 || x + 1 == w {
                cap
            } else {
                let sobel = 2.0 * (at(x + 1, y) - at(x - 1, y))
                    + (at(x + 1, yu) - at(x - 1, yu))
                    + (at(x + 1, yd) - at(x - 1, yd));
                sobel.clamp(-cap, cap) + cap
            };
        }
        for x in 0..w {
            let span = |v: &[f32]| {
                let c = v[x];
                let l = if x > 0 { 0.5 * (c + v[x - 1]) } else { c };
                let r = if x + 1 < w { 0.5 * (c + v[x + 1]) } else { c };
                (c, c.min(l).min(r), c.max(l).max(r))
            };
            let (p, pl, ph) = span(&pref);
            let (g, gl, gh) = span(&grey);
            out[y * w + x] = [p, pl, ph, g, gl, gh];
        }
    }
}

/// Birchfield–Tomasi dissimilarity of value `u` (with its neighbourhood's
/// least and greatest, `ul` and `uh`) and `v` (`vl`, `vh`).
#[inline]
pub(crate) fn bt(u: f32, ul: f32, uh: f32, v: f32, vl: f32, vh: f32) -> f32 {
    let c0 = (u - vh).max(vl - u).max(0.0);
    let c1 = (v - uh).max(ul - v).max(0.0);
    c0.min(c1)
}

/// The pixel cost of a left and a right span (step 3), rounded half up.
#[inline]
pub(crate) fn pixel_cost(l: &PixelSpan, r: &PixelSpan) -> u32 {
    let p = bt(l[0], l[1], l[2], r[0], r[1], r[2]);
    let g = bt(l[3], l[4], l[5], r[3], r[4], r[5]);
    (p + 0.25 * g + 0.5).floor() as u32
}

/// The cost of a disparity outside the right image, from the pixel's least
/// cost inside it, their sum and number: halfway between the least and the
/// mean (header, step 3).
#[inline]
pub(crate) fn outside_cost(least: u32, sum: u32, inside: u32) -> u32 {
    (least + sum / inside) / 2
}

/// The number of disparities searched at `level` for ranges from `min_range`.
pub(crate) fn num_disparities(rig: &StereoRig, level: usize, min_range: f64) -> usize {
    let s = rig.cam0.fx * rig.baseline_meters() / (1usize << level) as f64;
    (s / min_range).ceil() as usize + 1
}

/// Semi-global block matching (header).
pub struct Sgbm {
    pub(crate) config: SgbmConfig,
    pub(crate) w0: usize,
    pub(crate) h0: usize,
    /// The matched level's size and the number of disparities.
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) nd: usize,
    pyr0: Pyramid,
    pyr1: Pyramid,
    pyr_scratch: [Vec<f32>; 2],
    pub(crate) spans: [Vec<PixelSpan>; 2],
    /// Pixel costs, then block costs, a path's costs and their sum: per pixel
    /// `nd` disparities.
    pub(crate) cost: Vec<u32>,
    tmp: Vec<u32>,
    pub(crate) block: Vec<u32>,
    path: Vec<u32>,
    pub(crate) total: Vec<u32>,
    /// Per pixel: disparity (pixels of the level), its best (S << 8 | d) for
    /// the left-right check, per right pixel its best match.
    disp: Vec<f32>,
    best: Vec<u32>,
    right_best: Vec<u32>,
    valid: Vec<bool>,
    speckle: SpeckleScratch,
    /// Per pixel: ln(f·B·|K⁻¹(u, v, 1)| / 2^level), so its log-range is this
    /// minus ln(disparity).
    pub(crate) log_scale: Vec<f32>,
}

impl Sgbm {
    /// For a rectified pinhole rig, the right camera at +x; images of the
    /// left camera's resolution.
    pub fn new(rig: &StereoRig, config: SgbmConfig) -> Self {
        assert_eq!(
            rectified_sign(rig),
            Some(-1.0),
            "SGBM needs a rectified pair with the right camera at +x"
        );
        assert!(config.block_size % 2 == 1, "the block size must be odd");
        assert!(matches!(config.paths, 4 | 8), "4 or 8 paths");
        let [w0, h0] = rig.cam0.resolution;
        let (w, h) = (w0 >> config.level, h0 >> config.level);
        assert!(
            w > config.block_size && h > config.block_size,
            "image too small"
        );
        let nd = num_disparities(rig, config.level, config.min_range).min(w);
        assert!(
            nd <= 256,
            "at most 256 disparities (a coarser level, or a farther min_range)"
        );
        let c0 = &rig.cam0;
        let s = 1usize << config.level;
        let fb = c0.fx * rig.baseline_meters() / s as f64;
        let mut log_scale = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let a = ((x * s) as f64 - c0.cx) / c0.fx;
                let b = ((y * s) as f64 - c0.cy) / c0.fy;
                log_scale.push((fb * (a * a + b * b + 1.0).sqrt()).ln() as f32);
            }
        }
        let blank = Image::<u8>::new(w0, h0);
        Self {
            w0,
            h0,
            w,
            h,
            nd,
            pyr0: Pyramid::build(&blank, config.level + 1, 1.0),
            pyr1: Pyramid::build(&blank, config.level + 1, 1.0),
            pyr_scratch: [Vec::new(), Vec::new()],
            spans: [Vec::new(), Vec::new()],
            cost: Vec::new(),
            tmp: Vec::new(),
            block: Vec::new(),
            path: Vec::new(),
            total: Vec::new(),
            disp: Vec::new(),
            best: Vec::new(),
            right_best: Vec::new(),
            valid: Vec::new(),
            speckle: SpeckleScratch::default(),
            log_scale,
            config,
        }
    }

    pub fn config(&self) -> &SgbmConfig {
        &self.config
    }

    /// Disparities searched: 0..num_disparities().
    pub fn num_disparities(&self) -> usize {
        self.nd
    }

    /// The matched level's size.
    pub fn level_size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// The last pair's disparity map (pixels of the level; NaN: none).
    pub fn disparity(&self) -> &[f32] {
        &self.disp
    }

    /// The depth of a rectified pair.
    pub fn compute_into(&mut self, left: &Image<u8>, right: &Image<u8>, out: &mut DenseDepth) {
        assert_eq!((left.width(), left.height()), (self.w0, self.h0));
        assert_eq!((right.width(), right.height()), (self.w0, self.h0));
        let n = self.config.level + 1;
        let [s0, s1] = &mut self.pyr_scratch;
        self.pyr0.build_f32_reuse(left, n, s0);
        self.pyr1.build_f32_reuse(right, n, s1);
        let cap = self.config.pre_filter_cap as f32;
        spans(self.pyr0.level(self.config.level), cap, &mut self.spans[0]);
        spans(self.pyr1.level(self.config.level), cap, &mut self.spans[1]);
        self.match_spans();
        self.depth(out);
    }

    /// Steps 3–8 from both images' spans.
    pub(crate) fn match_spans(&mut self) {
        self.pixel_costs();
        self.block_costs();
        self.path_costs();
        self.winners();
        self.left_right_check();
        let size = self.config.speckle_size >> (2 * self.config.level);
        if size > 0 {
            remove_speckles(
                &mut self.valid,
                &self.disp,
                self.w,
                self.h,
                size,
                &mut self.speckle,
            );
        }
        for (d, &v) in self.disp.iter_mut().zip(&self.valid) {
            if !v {
                *d = f32::NAN;
            }
        }
    }

    fn pixel_costs(&mut self) {
        let (w, h, nd) = (self.w, self.h, self.nd);
        let [l, r] = &self.spans;
        self.cost.resize(w * h * nd, 0);
        for y in 0..h {
            for x in 0..w {
                let c = &mut self.cost[(y * w + x) * nd..][..nd];
                let inside = nd.min(x + 1);
                let lp = &l[y * w + x];
                let mut least = u32::MAX;
                let mut sum = 0u32;
                for (d, c) in c[..inside].iter_mut().enumerate() {
                    *c = pixel_cost(lp, &r[y * w + x - d]);
                    least = least.min(*c);
                    sum += *c;
                }
                c[inside..].fill(outside_cost(least, sum, inside as u32));
            }
        }
    }

    fn block_costs(&mut self) {
        let (w, h, nd) = (self.w, self.h, self.nd);
        let half = self.config.block_size as isize / 2;
        self.tmp.resize(w * h * nd, 0);
        self.block.resize(w * h * nd, 0);
        let clamp = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
        for y in 0..h {
            for x in 0..w {
                let t = &mut self.tmp[(y * w + x) * nd..][..nd];
                t.fill(0);
                for k in -half..=half {
                    let src = &self.cost[(y * w + clamp(x as isize + k, w)) * nd..][..nd];
                    for (t, &c) in t.iter_mut().zip(src) {
                        *t += c;
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let b = &mut self.block[(y * w + x) * nd..][..nd];
                b.fill(0);
                for k in -half..=half {
                    let src = &self.tmp[(clamp(y as isize + k, h) * w + x) * nd..][..nd];
                    for (b, &c) in b.iter_mut().zip(src) {
                        *b += c;
                    }
                }
            }
        }
    }

    fn path_costs(&mut self) {
        let (w, h, nd) = (self.w, self.h, self.nd);
        let (p1, p2) = (self.config.p1, self.config.p2);
        self.total.clear();
        self.total.resize(w * h * nd, 0);
        self.path.resize(w * h * nd, 0);
        for &(dx, dy) in &DIRS[..self.config.paths] {
            for i in 0..h {
                let y = if dy < 0 { h - 1 - i } else { i };
                for j in 0..w {
                    let x = if dx < 0 { w - 1 - j } else { j };
                    let p = y * w + x;
                    let c = &self.block[p * nd..][..nd];
                    let (qx, qy) = (x as isize - dx, y as isize - dy);
                    if qx < 0 || qy < 0 || qx >= w as isize || qy >= h as isize {
                        self.path[p * nd..][..nd].copy_from_slice(c);
                    } else {
                        let q = qy as usize * w + qx as usize;
                        // The pixel before p was visited earlier: below it in
                        // memory when the path runs forward, above otherwise.
                        let (prev, cur) = if q < p {
                            let (a, b) = self.path.split_at_mut(p * nd);
                            (&a[q * nd..][..nd], &mut b[..nd])
                        } else {
                            let (a, b) = self.path.split_at_mut(q * nd);
                            (&b[..nd], &mut a[p * nd..][..nd])
                        };
                        let m = *prev.iter().min().unwrap();
                        for d in 0..nd {
                            let mut best = prev[d].min(m + p2);
                            if d > 0 {
                                best = best.min(prev[d - 1] + p1);
                            }
                            if d + 1 < nd {
                                best = best.min(prev[d + 1] + p1);
                            }
                            cur[d] = c[d] + best - m;
                        }
                    }
                    for (t, &l) in self.total[p * nd..][..nd]
                        .iter_mut()
                        .zip(&self.path[p * nd..][..nd])
                    {
                        *t += l;
                    }
                }
            }
        }
    }

    fn winners(&mut self) {
        let (w, h, nd) = (self.w, self.h, self.nd);
        let u = self.config.uniqueness;
        self.disp.resize(w * h, 0.0);
        self.best.resize(w * h, NO_MATCH);
        self.valid.resize(w * h, false);
        for p in 0..w * h {
            let s = &self.total[p * nd..][..nd];
            let (db, sb) = winner(s);
            let unique = s
                .iter()
                .enumerate()
                .all(|(d, &v)| d.abs_diff(db) <= 1 || v * (100 - u) >= sb * 100);
            let d = subpixel(s, db);
            let x = p % w;
            self.valid[p] = unique && x as f32 - d >= 0.0;
            // A right pixel's matches: only the unique ones inside the image.
            self.best[p] = if unique && db <= x {
                (sb << 8) | db as u32
            } else {
                NO_MATCH
            };
            self.disp[p] = d;
        }
    }

    fn left_right_check(&mut self) {
        let Some(max_diff) = self.config.disp12_max_diff else {
            return;
        };
        let (w, h) = (self.w, self.h);
        self.right_best.clear();
        self.right_best.resize(w * h, NO_MATCH);
        for p in 0..w * h {
            let b = self.best[p];
            if b != NO_MATCH {
                let q = p - (b & 0xff) as usize;
                self.right_best[q] = self.right_best[q].min(b);
            }
        }
        for p in 0..w * h {
            if self.valid[p] {
                let x = p % w;
                self.valid[p] =
                    consistent(&self.right_best[p - x..][..w], x, self.disp[p], max_diff);
            }
        }
    }

    fn depth(&self, out: &mut DenseDepth) {
        depth_from_disparity(
            &self.disp,
            &self.log_scale,
            &self.config,
            self.w,
            self.h,
            out,
        );
    }
}

/// Step 9: the depth map from a disparity map (NaN: none) and the
/// per-pixel log scale.
pub(crate) fn depth_from_disparity(
    disp: &[f32],
    log_scale: &[f32],
    config: &SgbmConfig,
    w: usize,
    h: usize,
    out: &mut DenseDepth,
) {
    let level = config.level;
    let n = w * h;
    out.width = w;
    out.height = h;
    out.level = level;
    out.log_range.resize(n, f32::NAN);
    out.variance.resize(n, f32::NAN);
    out.source_level.resize(n, NO_SOURCE);
    let (lo, hi) = (config.min_range.ln() as f32, config.max_range.ln() as f32);
    let sigma = config.disparity_sigma;
    for i in 0..n {
        let d = disp[i];
        let log_range = log_scale[i] - d.ln();
        (out.log_range[i], out.variance[i], out.source_level[i]) =
            if d > 0.0 && log_range >= lo && log_range <= hi {
                let rel = sigma / d;
                (log_range, rel * rel, level as u8)
            } else {
                (f32::NAN, f32::NAN, NO_SOURCE)
            };
    }
}

/// The least sum and its disparity (the first of equals).
#[inline]
pub(crate) fn winner(s: &[u32]) -> (usize, u32) {
    s.iter()
        .copied()
        .enumerate()
        .fold((0, u32::MAX), |b, (d, v)| if v < b.1 { (d, v) } else { b })
}

/// Disparity `d` refined by a parabola through its neighbours' sums.
#[inline]
pub(crate) fn subpixel(s: &[u32], d: usize) -> f32 {
    if d == 0 || d + 1 >= s.len() {
        return d as f32;
    }
    let (a, b, c) = (s[d - 1] as i64, s[d] as i64, s[d + 1] as i64);
    let denom = (a + c - 2 * b).max(1);
    d as f32 + (a - c) as f32 / (2 * denom) as f32
}

/// OpenCV's left-right test of left pixel x (disparity `d`) against its
/// row's best matches per right pixel: inconsistent only if the right pixels
/// at both ⌊d⌋ and ⌈d⌉ have best matches more than `max_diff` away.
#[inline]
pub(crate) fn consistent(row_best: &[u32], x: usize, d: f32, max_diff: u32) -> bool {
    let off = |dd: u32| -> bool {
        let Some(xr) = x.checked_sub(dd as usize) else {
            return false;
        };
        let b = row_best[xr];
        b != NO_MATCH && (b & 0xff).abs_diff(dd) > max_diff
    };
    !(off(d.floor() as u32) && off(d.ceil() as u32))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::camera::CameraIntrinsics;

    #[test]
    fn birchfield_tomasi() {
        // Equal values match; a value inside the other's half-pixel range
        // costs nothing; otherwise the gap to the nearer range.
        assert_eq!(bt(10.0, 9.0, 11.0, 10.0, 10.0, 10.0), 0.0);
        assert_eq!(bt(10.0, 10.0, 10.0, 12.0, 9.5, 12.0), 0.0);
        assert_eq!(bt(10.0, 10.0, 10.0, 14.0, 13.0, 15.0), 3.0);
        assert_eq!(bt(20.0, 18.0, 20.0, 14.0, 13.0, 15.0), 4.0);
    }

    #[test]
    fn subpixel_and_winner() {
        assert_eq!(winner(&[5, 3, 3, 9]), (1, 3));
        // A parabola through 10, 4, 6 has its vertex 0.25 right of the middle.
        assert!((subpixel(&[10, 4, 6], 1) - 1.25).abs() < 1e-6);
        assert_eq!(subpixel(&[1, 4, 6], 0), 0.0);
    }

    #[test]
    fn left_right_test_needs_both_sides() {
        // Right pixel 7 is best matched at disparity 3, right pixel 6 at 2.
        let mut row = vec![NO_MATCH; 16];
        row[7] = (100 << 8) | 3;
        row[6] = (100 << 8) | 2;
        assert!(consistent(&row, 10, 3.0, 1)); // pixel 7 agrees
        assert!(consistent(&row, 10, 2.6, 1)); // pixel 8 has no match: kept
        assert!(!consistent(&row, 15, 8.5, 1)); // pixels 7 (3 vs 8) and 6 (2 vs 9)
        assert!(consistent(&row, 15, 9.5, 1)); // pixel 5 has no match
    }

    /// Value noise in [0, 1), cells of side `cell`.
    fn noise(x: f64, y: f64, cell: f64, seed: u64) -> f64 {
        let hash = |i: i64, j: i64| {
            let mut v = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
                ^ seed.wrapping_mul(0x165667B19E3779F9);
            v ^= v >> 31;
            v = v.wrapping_mul(0xD6E8FEB86659FD93);
            v ^= v >> 32;
            (v >> 11) as f64 / (1u64 << 53) as f64
        };
        let (u, v) = (x / cell, y / cell);
        let (i, j) = (u.floor(), v.floor());
        let s = |t: f64| t * t * (3.0 - 2.0 * t);
        let (fu, fv) = (s(u - i), s(v - j));
        let (i, j) = (i as i64, j as i64);
        let a = hash(i, j) * (1.0 - fu) + hash(i + 1, j) * fu;
        let b = hash(i, j + 1) * (1.0 - fu) + hash(i + 1, j + 1) * fu;
        a * (1.0 - fv) + b * fv
    }

    /// A 320×240 rectified pair (f 300, baseline 0.1 m): a textured wall at
    /// 2 m (disparity 15) and a textured box at 0.75 m (disparity 40) over
    /// x 120..200, y 80..160 of the left image.
    pub(crate) fn scene() -> (StereoRig, Image<u8>, Image<u8>) {
        let (w, h, f, b) = (320usize, 240usize, 300.0, 0.1);
        let cam = CameraIntrinsics::new(f, f, 159.5, 119.5, w, h);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rig = StereoRig::new(cam.clone(), cam, identity, [-b, 0.0, 0.0]);
        let (d_wall, d_box) = (f * b / 2.0, f * b / 0.75);
        let in_box = |x: f64, y: f64| (120.0..200.0).contains(&x) && (80.0..160.0).contains(&y);
        // The texture is painted on the surfaces: a right-image pixel shows
        // the box where the box's projection (x + d_box in the left image)
        // falls inside it, else the wall.
        let shade =
            |xl: f64, y: f64, seed: u64| (40.0 + 180.0 * noise(xl, y, 3.0, seed)).round() as u8;
        let left = Image::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    if in_box(x, y) {
                        shade(x, y, 2)
                    } else {
                        shade(x, y, 1)
                    }
                })
                .collect(),
        );
        let right = Image::from_vec(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    if in_box(x + d_box, y) {
                        shade(x + d_box, y, 2)
                    } else {
                        shade(x + d_wall, y, 1)
                    }
                })
                .collect(),
        );
        (rig, left, right)
    }

    #[test]
    fn wall_and_box_at_their_disparities() {
        let (rig, left, right) = scene();
        let mut sgbm = Sgbm::new(
            &rig,
            SgbmConfig {
                level: 0,
                min_range: 0.5,
                ..Default::default()
            },
        );
        let mut depth = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        sgbm.compute_into(&left, &right, &mut depth);
        let d = sgbm.disparity();
        let w = 320;
        let near = |x: usize, y: usize, want: f32| (d[y * w + x] - want).abs() < 0.3;
        // Inside the box and on the wall away from the box, the borders and
        // the strip the box hides from the right camera.
        assert!(near(160, 120, 40.0) && near(130, 90, 40.0) && near(190, 150, 40.0));
        assert!(near(260, 60, 15.0) && near(60, 200, 15.0) && near(240, 200, 15.0));
        // Occluded: wall pixels just left of the box, which the right camera
        // sees behind the box: no depth (left-right check).
        let occluded = (95..115).filter(|&x| d[120 * w + x].is_finite()).count();
        assert!(occluded <= 4, "{occluded} occluded pixels kept a disparity");
        // The left border, where the wall's match is outside the right image.
        assert!((0..10).all(|x| !d[120 * w + x].is_finite()));
        // Range from disparity: the box's centre at 0.75 m.
        let r = depth.range_at(160, 120).unwrap();
        assert!(
            (r - 0.75 * (1.0 + (0.5f32 / 300.0).powi(2)).sqrt()).abs() < 0.01,
            "{r}"
        );
        let coverage = d.iter().filter(|v| v.is_finite()).count() as f32 / d.len() as f32;
        assert!(coverage > 0.85, "coverage {coverage}");
    }
}
