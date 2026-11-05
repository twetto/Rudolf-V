// checkerboard.rs — Checkerboard (chessboard) inner-corner detector.
//
// Reference: Bennett & Lasenby, "ChESS – Quick and robust detection of
// chess-board features" (CVIU 2014), for the corner response.
//
// Not a port of OpenCV's quad-based findChessboardCorners (adaptive threshold,
// dilation, quad linking). That approach is slow, and slowest when a board is
// half visible or the scene is cluttered. Here the per-pixel work is one
// X-corner response, which maps onto the GPU like FAST does, and the rest is a
// small search over a few hundred candidates.
//
// Algorithm:
//   1. ChESS response at every pixel, from 16 samples I_0..I_15 on a radius-5
//      ring (I_n and I_{n+8} opposite, I_{n+4} a quarter turn on):
//        sum  = Σ_{n=0..3} |(I_n + I_{n+8}) − (I_{n+4} + I_{n+12})|
//        diff = Σ_{n=0..7} |I_n − I_{n+8}|
//        mean = |mean of the ring − mean of the centre and its 4 neighbours|
//        R    = sum − diff − 16·mean
//      At an X-corner the four quadrants alternate, so opposite samples match
//      (diff ≈ 0) and quarter turns differ (sum large): R > 0. On an edge
//      opposite samples differ (diff large): R < 0.
//   2. Candidates: local maxima of R over a 5×5 window, sub-pixel by a 3×3
//      quadratic fit. Strong ones (above an absolute floor and a fraction of
//      the image's strongest response) seed lattices; weaker ones (blur, glare,
//      a corner at the board's edge) may only extend one.
//   3. Lattice: a seed quad from a strong candidate, its nearest neighbour and
//      the most perpendicular other neighbour, if it is a cell (darker or
//      lighter than its four neighbours), grown by whole rows and columns.
//      Each new corner is predicted from the last two on its line (linearly;
//      from three, quadratically, if that misses), so the lattice follows
//      perspective and lens distortion (raw, unrectified images work); the
//      nearest candidate within a fraction of the local spacing is taken. A
//      full cols×rows lattice (either orientation) whose cells alternate
//      dark/light and whose rows and columns bend smoothly is the board, if
//      it ends there: a partial board is no board, and neither is a bigger
//      one (its next row or column would match too).
//   4. Sub-pixel refinement of each corner as it joins the lattice: the
//      gradient-orthogonality solve of OpenCV's cornerSubPix (at a saddle,
//      every gradient in the window is orthogonal to the vector from the
//      corner). The predictions use these positions: a candidate's raw peak is
//      only good to ~0.5 px, and extrapolation amplifies that.
//   5. If no board was found, steps 1–4 again on the image at half size, the
//      corners refined at full size. Blur about as long as the ring is wide
//      (fast motion, defocus) turns the response at the corners negative, and
//      squares many times the ring's size weaken it; halving the image halves
//      both.
//   6. Order: row-major, rows along the `cols` axis pointing +x, rows ordered
//      so the numbering keeps the image's handedness (+y for rows along +x).
//      Never a mirror image, which would put the board's front behind it; a
//      180° turn of the board, which the pattern cannot tell, gives the same
//      order flipped.
//
// The output is in the input image's pixels: on a raw image, raw pixels. Undo
// lens distortion on these points (camera-geometry), not on the image.
//
// NEW RUST CONCEPTS:
// - core::array::from_fn to build a fixed-size array from a closure (the ring
//   offsets for the image's stride).
// - A Vec<Vec<usize>> lattice grown at either end (insert(0, ...) / push).

use std::sync::Mutex;

use crate::image::{Image, Pixel};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// ChESS ring: radius 5, 16 samples 22.5° apart, rounded to pixels.
/// RING[n] and RING[n + 8] are opposite.
const RING: [(isize, isize); 16] = [
    (5, 0),
    (5, 2),
    (4, 4),
    (2, 5),
    (0, 5),
    (-2, 5),
    (-4, 4),
    (-5, 2),
    (-5, 0),
    (-5, -2),
    (-4, -4),
    (-2, -5),
    (0, -5),
    (2, -5),
    (4, -4),
    (5, -2),
];

/// Border without a response: the ring's radius.
const RADIUS: usize = 5;

/// The 5×5 window around a pixel, nearest first: a pixel on a peak's slope
/// usually meets a higher neighbour among the first few.
const WINDOW: [(isize, isize); 24] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
    (2, 2),
    (2, -2),
    (-2, 2),
    (-2, -2),
];

/// Nearest neighbours considered when forming a seed quad.
const SEED_NEIGHBOURS: usize = 8;

/// Largest second difference along a row or column (a corner's offset from
/// its two neighbours' midpoint, ×2), as a fraction of the local spacing.
/// Perspective and lens distortion stay far below it; corners that heavy blur
/// scattered, or merged into one, do not.
const MAX_BEND: f32 = 0.5;

/// Seed quads at least this large (spacing, pixels) must be a cell before they
/// are grown. Smaller ones skip the test: in a small or blurred board's cells
/// its samples sit too near the edges, and a later seed then found the board
/// with other corners. Clutter's chance quads were mostly larger.
const CELL_TEST_MIN_SPACING: f32 = 20.0;

/// Detector settings.
#[derive(Debug, Clone)]
pub struct CheckerboardConfig {
    /// Inner corners along a row (9 for a 9×6 board).
    pub cols: usize,
    /// Rows of inner corners (6 for a 9×6 board).
    pub rows: usize,
    /// Minimum ChESS response of a candidate. An ideal X-corner gives about
    /// 8 × its dark/light contrast, so 160 asks for ~20 grey levels.
    pub min_response: f32,
    /// Candidates must also reach this fraction of the strongest response.
    pub relative_threshold: f32,
    /// Weaker candidates, down to this response, may extend a lattice but
    /// not seed one: the lattice's geometry vouches for them.
    pub grow_response: f32,
    /// Strongest candidates kept for the lattice search.
    pub max_candidates: usize,
    /// Strongest candidates tried as seeds before giving up (per scale). Every
    /// seed is tried when there is no board, so this bounds the cost of a look
    /// at an empty scene: 20 found the same boards as 40 on the ZED and fisheye
    /// test sets but 2 of 315 raw ZED frames, at three quarters of the cost.
    pub max_seeds: usize,
    /// A predicted corner matches a candidate within this fraction of the
    /// local corner spacing.
    pub match_tolerance: f32,
    /// The dark and light cells must differ by at least this many grey levels.
    pub min_cell_contrast: f32,
}

impl Default for CheckerboardConfig {
    fn default() -> Self {
        Self {
            cols: 9,
            rows: 6,
            min_response: 160.0,
            relative_threshold: 0.1,
            grow_response: 40.0,
            max_candidates: 1000,
            max_seeds: 20,
            match_tolerance: 0.35,
            min_cell_contrast: 20.0,
        }
    }
}

/// A corner candidate: a local maximum of the ChESS response.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub x: f32,
    pub y: f32,
    pub response: f32,
}

/// ChESS response of every pixel; zero within RADIUS of the border.
pub fn chess_response(image: &Image<u8>) -> Image<f32> {
    let mut out = Image::new(image.width(), image.height());
    chess_response_into(image, &mut out);
    out
}

/// As `chess_response`, into `out` (the image's size, stride = width): every
/// pixel inside the border is written and the border is left as it is, so a
/// buffer that started zeroed can be reused without clearing.
pub fn chess_response_into(image: &Image<u8>, out: &mut Image<f32>) {
    fill_response(image, out);
}

#[cfg(test)]
/// The response as the detector stores it, 5R in i16 (for checking a GPU's).
pub(crate) fn chess_response_r5(image: &Image<u8>) -> Image<i16> {
    let mut out = Image::new(image.width(), image.height());
    fill_response(image, &mut out);
    out
}

/// `image` at half size, each pixel the rounded mean of a 2×2 block.
pub(crate) fn half_size(image: &Image<u8>) -> Image<u8> {
    let mut out = Image::new(image.width() / 2, image.height() / 2);
    half_size_into(image, &mut out);
    out
}

/// A response as stored: f32, the response itself; or i16, five times it as
/// computed (5R), which the detector keeps: half the bytes to write and to scan
/// for candidates, no conversion. 5R ↦ 0.2·5R keeps every order and equality
/// (distinct integers stay distinct in f32 at this range), so both find the same
/// candidates. A GPU kernel would store 5R as i32.
trait ResponseValue: Pixel {
    fn from_r5(r5: i16) -> Self;
    fn response(self) -> f32;
}

impl ResponseValue for f32 {
    #[inline(always)]
    fn from_r5(r5: i16) -> f32 {
        r5 as f32 * 0.2
    }
    #[inline(always)]
    fn response(self) -> f32 {
        self
    }
}

impl ResponseValue for i16 {
    #[inline(always)]
    fn from_r5(r5: i16) -> i16 {
        r5
    }
    #[inline(always)]
    fn response(self) -> f32 {
        self as f32 * 0.2
    }
}

/// The smallest 5R whose response reaches `t`: `5R >= r5_at_least(t)` exactly
/// when `0.2·5R >= t`, as the f32 comparison would decide it.
fn r5_at_least(t: f32) -> i16 {
    let mut k = ((t / 0.2).ceil() as i32).clamp(i16::MIN as i32, i16::MAX as i32);
    while k > i16::MIN as i32 && (k - 1) as f32 * 0.2 >= t {
        k -= 1;
    }
    while k < i16::MAX as i32 && (k as f32 * 0.2) < t {
        k += 1;
    }
    k as i16
}

fn fill_response<O: ResponseValue>(image: &Image<u8>, out: &mut Image<O>) {
    let (w, h) = (image.width(), image.height());
    assert!(
        out.width() == w && out.height() == h && out.stride() == w,
        "response buffer {}x{} (stride {}) for a {w}x{h} image",
        out.width(),
        out.height(),
        out.stride()
    );
    if w <= 2 * RADIUS || h <= 2 * RADIUS {
        return;
    }
    let fill_row =
        |y: usize, out_row: &mut [O]| response_row(image, y, &mut out_row[RADIUS..w - RADIUS]);

    #[cfg(feature = "parallel")]
    out.as_mut_slice()
        .par_chunks_mut(w)
        .enumerate()
        .filter(|(y, _)| *y >= RADIUS && *y < h - RADIUS)
        .for_each(|(y, row)| fill_row(y, row));
    #[cfg(not(feature = "parallel"))]
    for y in RADIUS..h - RADIUS {
        fill_row(y, out.row_mut(y));
    }
}

/// The response of row `y` for x in RADIUS..width − RADIUS, into `out`.
fn response_row<O: ResponseValue>(image: &Image<u8>, y: usize, out: &mut [O]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2.
        return unsafe { response_row_avx2(image, y, out) };
    }
    response_row_body(image, y, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn response_row_avx2<O: ResponseValue>(image: &Image<u8>, y: usize, out: &mut [O]) {
    response_row_body(image, y, out)
}

/// 5R in i16, which needs no division and holds every value (−30600..10200):
///   5R = 5·(sum − diff) − |5·(ring sum) − 16·(sum of the centre and its 4 neighbours)|
/// Each ring sample and each centre pixel is one row slice, and the slices are
/// walked in step, so the compiler vectorizes across pixels (16 per instruction
/// with AVX2, 8 with NEON).
#[inline(always)]
fn response_row_body<O: ResponseValue>(image: &Image<u8>, y: usize, out: &mut [O]) {
    let (data, stride, n) = (image.as_slice(), image.stride() as isize, out.len());
    // The n pixels from (RADIUS + dx, y + dy) on.
    let at = |dx: isize, dy: isize| {
        let start = ((y as isize + dy) * stride + RADIUS as isize + dx) as usize;
        &data[start..start + n]
    };
    let ring: [&[u8]; 16] = core::array::from_fn(|k| at(RING[k].0, RING[k].1));
    let centre = [at(0, 0), at(-1, 0), at(1, 0), at(0, -1), at(0, 1)];
    for x in 0..n {
        // In four groups of samples a quarter turn apart (k, k+4, k+8, k+12),
        // each feeding all three sums, so few values are live at once.
        let (mut sum, mut diff, mut ring_sum) = (0i16, 0i16, 0i16);
        for k in 0..4 {
            let (a, b) = (ring[k][x] as i16, ring[k + 4][x] as i16);
            let (c, d) = (ring[k + 8][x] as i16, ring[k + 12][x] as i16);
            sum += ((a + c) - (b + d)).abs();
            diff += (a - c).abs() + (b - d).abs();
            ring_sum += a + b + c + d;
        }
        let centre_sum: i16 = centre.iter().map(|row| row[x] as i16).sum();
        let r5 = 5 * (sum - diff) - (5 * ring_sum - 16 * centre_sum).abs();
        out[x] = O::from_r5(r5);
    }
}

/// Local maxima of the response (5×5) down to `grow_response`, strongest
/// first, sub-pixel.
pub fn find_candidates(response: &Image<f32>, config: &CheckerboardConfig) -> Vec<Candidate> {
    candidates_in(response, config.grow_response, config)
}

/// `find_candidates` on a stored response (f32 or 5R), `threshold` in its units.
fn candidates_in<T: ResponseValue>(
    response: &Image<T>,
    threshold: T,
    config: &CheckerboardConfig,
) -> Vec<Candidate> {
    let (w, h) = (response.width(), response.height());
    let m = RADIUS + 2;
    if w <= 2 * m || h <= 2 * m {
        return Vec::new();
    }
    let mut out = Vec::new();
    for y in m..h - m {
        let rows: [&[T]; 5] = core::array::from_fn(|k| response.row(y + k - 2));
        // Most pixels are below the threshold: test 16 at once (an OR without
        // short-circuit, which vectorizes) and look closer where one passes.
        for (i, chunk) in rows[2][m..w - m].chunks(16).enumerate() {
            if !chunk.iter().fold(false, |any, &v| any | (v >= threshold)) {
                continue;
            }
            for (j, &v) in chunk.iter().enumerate() {
                if v < threshold {
                    continue;
                }
                let x = m + 16 * i + j;
                // Strict maximum over earlier pixels, at least equal to later
                // ones, so a plateau yields exactly one candidate.
                let is_max = WINDOW.iter().all(|&(dx, dy)| {
                    let u = rows[(2 + dy) as usize][(x as isize + dx) as usize];
                    let earlier = dy < 0 || (dy == 0 && dx < 0);
                    !(u > v || (earlier && u == v))
                });
                if !is_max {
                    continue;
                }
                let near = [rows[2][x - 1], rows[2][x + 1], rows[1][x], rows[3][x]];
                out.push(candidate_at(x, y, v, near));
            }
        }
    }
    out.sort_by(|a, b| b.response.total_cmp(&a.response));
    out.truncate(config.max_candidates);
    out
}

/// The candidate at the maximum (x, y) of response `v`, `near` its left, right,
/// upper and lower neighbours' responses: sub-pixel by a parabola each way.
fn candidate_at<T: ResponseValue>(x: usize, y: usize, v: T, near: [T; 4]) -> Candidate {
    let [l, r, u, d] = near.map(|n| n.response());
    let v = v.response();
    Candidate {
        x: x as f32 + parabola_peak(l, v, r),
        y: y as f32 + parabola_peak(u, v, d),
        response: v,
    }
}

/// A local maximum as a GPU finds it: position, 5R, and its left, right, upper
/// and lower neighbours' 5R.
#[derive(Clone, Copy)]
pub(crate) struct Maximum {
    pub x: u16,
    pub y: u16,
    pub r5: i16,
    pub near: [i16; 4],
}

/// `candidates_in`'s result from its maxima, found in any order: strongest
/// first, ties in row-major order (as its stable sort of a row-major scan).
pub(crate) fn candidates_from(
    maxima: &mut [Maximum],
    config: &CheckerboardConfig,
) -> Vec<Candidate> {
    maxima.sort_unstable_by_key(|m| (std::cmp::Reverse(m.r5), m.y, m.x));
    maxima
        .iter()
        .take(config.max_candidates)
        .map(|m| candidate_at(m.x as usize, m.y as usize, m.r5, m.near))
        .collect()
}

/// `candidates_in` on a response computed here: for a GPU's frame with more
/// maxima than its buffer holds.
pub(crate) fn candidates_of(image: &Image<u8>, config: &CheckerboardConfig) -> Vec<Candidate> {
    let mut response = Image::<i16>::new(image.width(), image.height());
    fill_response(image, &mut response);
    candidates_in(&response, r5_at_least(config.grow_response), config)
}

/// The threshold `candidates_in` applies, in 5R.
pub(crate) fn grow_r5(config: &CheckerboardConfig) -> i16 {
    r5_at_least(config.grow_response)
}

/// Offset of a parabola's peak through (-1, a), (0, b), (1, c), within ±0.5.
fn parabola_peak(a: f32, b: f32, c: f32) -> f32 {
    let den = 2.0 * (2.0 * b - a - c);
    if den.abs() < 1e-6 {
        0.0
    } else {
        ((c - a) / den).clamp(-0.5, 0.5)
    }
}

/// Checkerboard detector.
pub struct CheckerboardDetector {
    pub config: CheckerboardConfig,
    scratch: Mutex<Scratch>,
}

/// Buffers a detector reuses from one image to the next (a GPU version keeps
/// them on the device): the responses at both scales and the half-size image.
/// Allocating and zeroing a 1280×720 response twice per look was ~15 % of a
/// look at an empty scene on the Orin.
#[derive(Default)]
struct Scratch {
    response: Option<Image<i16>>,
    response_half: Option<Image<i16>>,
    half: Option<Image<u8>>,
}

/// The image in `slot`, allocated (zeroed) only when there is none of this size.
fn reuse<T: Pixel>(slot: &mut Option<Image<T>>, w: usize, h: usize) -> &mut Image<T> {
    if !slot
        .as_ref()
        .is_some_and(|i| i.width() == w && i.height() == h)
    {
        *slot = Some(Image::new(w, h));
    }
    slot.as_mut().unwrap()
}

impl CheckerboardDetector {
    pub fn new(config: CheckerboardConfig) -> Self {
        Self {
            config,
            scratch: Mutex::new(Scratch::default()),
        }
    }

    /// The board's `cols × rows` inner corners in pixels (row-major), or None
    /// when no complete board is found.
    pub fn detect(&self, image: &Image<u8>) -> Option<Vec<[f32; 2]>> {
        self.with_scratch(|sc| {
            if let Some((corners, _)) = self.detect_at_scale(image, &mut sc.response) {
                return Some(self.ordered(corners));
            }
            let small = reuse(&mut sc.half, image.width() / 2, image.height() / 2);
            half_size_into(image, small);
            let found = self.detect_at_scale(small, &mut sc.response_half)?;
            self.from_half_size(image, found)
        })
    }

    /// As `detect`, with the candidates found elsewhere (on a GPU): `full` in
    /// `image`, `half` in its half-size image, as `candidates_in` gives them.
    pub(crate) fn detect_with_candidates(
        &self,
        image: &Image<u8>,
        full: Vec<Candidate>,
        half: Vec<Candidate>,
    ) -> Option<Vec<[f32; 2]>> {
        if let Some((corners, _)) = self.search(image, &full) {
            return Some(self.ordered(corners));
        }
        self.with_scratch(|sc| {
            let small = reuse(&mut sc.half, image.width() / 2, image.height() / 2);
            half_size_into(image, small);
            let found = self.search(small, &half)?;
            self.from_half_size(image, found)
        })
    }

    /// `f` with this detector's buffers; while another thread detects with
    /// this detector, with buffers of its own rather than waiting.
    fn with_scratch<T>(&self, f: impl FnOnce(&mut Scratch) -> T) -> T {
        let mut guard = self.scratch.try_lock().ok();
        let mut own = Scratch::default();
        f(match guard.as_deref_mut() {
            Some(sc) => sc,
            None => &mut own,
        })
    }

    /// A lattice found in the half-size image, refined in `image` and ordered,
    /// if its lines are smooth there too.
    fn from_half_size(
        &self,
        image: &Image<u8>,
        (corners, half): (Vec<Vec<[f32; 2]>>, usize),
    ) -> Option<Vec<[f32; 2]>> {
        // A half-size pixel (u, v) covers full-size pixels 2u..2u+1.
        let weights = gaussian_weights(2 * half);
        let corners: Vec<Vec<[f32; 2]>> = corners
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|[u, v]| {
                        refine_corner_with(
                            image,
                            [2.0 * u + 0.5, 2.0 * v + 0.5],
                            2 * half,
                            &weights,
                        )
                    })
                    .collect()
            })
            .collect();
        lines_smooth(&corners).then(|| self.ordered(corners))
    }

    /// As `detect`, within the region (x, y, w, h) of `image`; corners in the
    /// whole image's pixels.
    pub fn detect_region(
        &self,
        image: &Image<u8>,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Option<Vec<[f32; 2]>> {
        let w = w.min(image.width().saturating_sub(x));
        let h = h.min(image.height().saturating_sub(y));
        let crop = image.sub_image(x, y, w, h).to_owned_image();
        let corners = self.detect(&crop)?;
        Some(
            corners
                .into_iter()
                .map(|[u, v]| [u + x as f32, v + y as f32])
                .collect(),
        )
    }

    /// Steps 1–4 on one image: the lattice's refined corners, unordered, and
    /// the refinement half-window used.
    fn detect_at_scale(
        &self,
        image: &Image<u8>,
        response: &mut Option<Image<i16>>,
    ) -> Option<(Vec<Vec<[f32; 2]>>, usize)> {
        let response = reuse(response, image.width(), image.height());
        fill_response(image, response);
        self.search_at_scale(image, response)
    }

    /// Steps 2–4 on one image whose response (5R) is `response`.
    fn search_at_scale(
        &self,
        image: &Image<u8>,
        response: &Image<i16>,
    ) -> Option<(Vec<Vec<[f32; 2]>>, usize)> {
        let candidates = candidates_in(
            response,
            r5_at_least(self.config.grow_response),
            &self.config,
        );
        self.search(image, &candidates)
    }

    /// Steps 3–4: the lattice among `candidates` (strongest first) in `image`.
    fn search(
        &self,
        image: &Image<u8>,
        candidates: &[Candidate],
    ) -> Option<(Vec<Vec<[f32; 2]>>, usize)> {
        if candidates.len() < self.config.cols * self.config.rows {
            return None;
        }
        let strong = self
            .config
            .min_response
            .max(self.config.relative_threshold * candidates[0].response);
        let n_strong = candidates
            .iter()
            .take_while(|c| c.response >= strong)
            .count();
        let mut search = Search::new(image, candidates, n_strong);
        let (lattice, half) = self.find_lattice(&mut search)?;
        Some((search.positions(&lattice), half))
    }

    /// Seed quads of strong candidates, strongest first, each grown into a
    /// lattice; the first complete board whose cells alternate dark/light and
    /// whose lines are smooth, with its refinement half-window.
    fn find_lattice(&self, s: &mut Search) -> Option<(Vec<Vec<usize>>, usize)> {
        for seed in 0..s.n_strong.min(self.config.max_seeds) {
            let near = s.seed_neighbours(seed, SEED_NEIGHBOURS);
            for &a in near.iter().take(4) {
                // Second axis: the most perpendicular neighbours first, so a
                // diagonal (also a lattice vector) is tried last.
                let mut others = [(0.0f32, 0usize); SEED_NEIGHBOURS];
                let mut n_others = 0;
                for &b in near.iter().filter(|&&b| b != a) {
                    others[n_others] = (cos_at(s.c, seed, a, b).abs(), b);
                    n_others += 1;
                }
                others[..n_others].sort_by(|x, y| x.0.total_cmp(&y.0));
                for &(_, b) in &others[..n_others] {
                    let Some((quad, half)) = self.seed_quad(s, seed, a, b) else {
                        continue;
                    };
                    let Some(lattice) = self.grow(s, quad, half) else {
                        continue;
                    };
                    if self.cells_alternate(s, &lattice) && lines_smooth(&s.positions(&lattice)) {
                        return Some((lattice, half));
                    }
                }
            }
        }
        None
    }

    /// A quad [[seed, a], [b, d]] when a and b leave the seed at a plausible
    /// angle and a candidate sits where the fourth corner should be; also the
    /// refinement half-window for this board's spacing.
    fn seed_quad(
        &self,
        s: &mut Search,
        seed: usize,
        a: usize,
        b: usize,
    ) -> Option<(Vec<Vec<usize>>, usize)> {
        let (c0, ca, cb) = (&s.c[seed], &s.c[a], &s.c[b]);
        let (n1, n2) = (
            norm([ca.x - c0.x, ca.y - c0.y]),
            norm([cb.x - c0.x, cb.y - c0.y]),
        );
        if n1 < 4.0
            || n2 < 4.0
            || n1 / n2 > 2.0
            || n2 / n1 > 2.0
            || cos_at(s.c, seed, a, b).abs() > 0.77
        {
            return None; // outside 40°..140°
        }
        let half = ((0.3 * n1.min(n2)).round() as usize).clamp(2, 6);
        let (p0, pa, pb) = (s.pos(seed, half), s.pos(a, half), s.pos(b, half));
        let p3 = [pa[0] + pb[0] - p0[0], pa[1] + pb[1] - p0[1]];
        let d = s
            .index
            .nearest(p3, self.config.match_tolerance * n1.min(n2), &[seed, a, b])?;
        let pd = s.pos(d, half);
        // In clutter a fourth corner turns up by chance for most seeds, and
        // growing each such quad until it fails was most of the search.
        if n1.min(n2) >= CELL_TEST_MIN_SPACING && !self.is_a_cell(s.image, [p0, pa, pd, pb]) {
            return None;
        }
        Some((vec![vec![seed, a], vec![b, d]], half))
    }

    /// The quad (corners in order around it) is a checkerboard cell: across
    /// each of its edges, the cell and its neighbour differ, and the cell is
    /// on the same side every time (darker than all four neighbours, or
    /// lighter). Sampled a quarter of the edge's length either side of its
    /// midpoint; half the final check's contrast, for blurred boards.
    fn is_a_cell(&self, image: &Image<u8>, q: [[f32; 2]; 4]) -> bool {
        let centre = [
            q.iter().map(|p| p[0]).sum::<f32>() / 4.0,
            q.iter().map(|p| p[1]).sum::<f32>() / 4.0,
        ];
        let min_step = 0.5 * self.config.min_cell_contrast;
        let mut sign = 0.0f32;
        for k in 0..4 {
            let (p, r) = (q[k], q[(k + 1) % 4]);
            let m = [0.5 * (p[0] + r[0]), 0.5 * (p[1] + r[1])];
            let e = [r[0] - p[0], r[1] - p[1]];
            let mut n = [-e[1] * 0.25, e[0] * 0.25]; // a quarter edge length, across it
            if n[0] * (centre[0] - m[0]) + n[1] * (centre[1] - m[1]) < 0.0 {
                n = [-n[0], -n[1]]; // pointing into the cell
            }
            let step = mean_3x3(image, m[0] + n[0], m[1] + n[1])
                - mean_3x3(image, m[0] - n[0], m[1] - n[1]);
            if step.abs() < min_step || step * sign < 0.0 {
                return false;
            }
            sign = step;
        }
        true
    }

    /// Grow the lattice by whole rows and columns until nothing more fits;
    /// Some only for a complete cols×rows (or rows×cols) board that ends there.
    /// A bigger board would grow on, so it is no board either, as half a board
    /// is not. Predictions use refined positions: the candidates' raw peaks are
    /// only good to ~0.5 px, which extrapolation would amplify.
    fn grow(&self, s: &mut Search, mut g: Vec<Vec<usize>>, half: usize) -> Option<Vec<Vec<usize>>> {
        let (cols, rows) = (self.config.cols, self.config.rows);
        let (long, short) = (cols.max(rows), cols.min(rows));
        let fits = |ny: usize, nx: usize| ny.max(nx) <= long && ny.min(nx) <= short;
        loop {
            let mut grew = false;
            for side in 0..4 {
                // 0: right, 1: left, 2: down, 3: up
                let (ny, nx) = (g.len(), g[0].len());
                let along_x = side < 2;
                if (along_x && !fits(ny, nx + 1)) || (!along_x && !fits(ny + 1, nx)) {
                    continue;
                }
                if !self.next_line(s, &g, side, half) {
                    continue;
                }
                let new_line = &s.used[ny * nx..];
                match side {
                    0 => g.iter_mut().zip(new_line).for_each(|(row, &m)| row.push(m)),
                    1 => g
                        .iter_mut()
                        .zip(new_line)
                        .for_each(|(row, &m)| row.insert(0, m)),
                    2 => g.push(new_line.to_vec()),
                    _ => g.insert(0, new_line.to_vec()),
                }
                grew = true;
            }
            if !grew {
                break;
            }
        }
        let (ny, nx) = (g.len(), g[0].len());
        if (ny, nx) != (rows, cols) && (ny, nx) != (cols, rows) {
            return None;
        }
        // The board must end here: a bigger one has its next row or column too.
        if (0..4).any(|side| self.next_line(s, &g, side, half)) {
            return None;
        }
        Some(g)
    }

    /// A new row or column beyond `side` of the lattice (0: right, 1: left,
    /// 2: down, 3: up): each line's continuation, left in `s.used` after the
    /// lattice's own candidates; false unless every line has one.
    fn next_line(&self, s: &mut Search, g: &[Vec<usize>], side: usize, half: usize) -> bool {
        let (ny, nx) = (g.len(), g[0].len());
        let lines = if side < 2 { ny } else { nx };
        let mut used = std::mem::take(&mut s.used);
        used.clear();
        used.extend(g.iter().flatten().copied());
        let mut found = true;
        for k in 0..lines {
            // The line's points from the end being extended, inwards.
            let mut pts = [[0.0f32; 2]; 3];
            let mut n = 0;
            for t in 0..3isize {
                let (j, i) = match side {
                    0 => (k as isize, nx as isize - 1 - t),
                    1 => (k as isize, t),
                    2 => (ny as isize - 1 - t, k as isize),
                    _ => (t, k as isize),
                };
                if j >= 0 && i >= 0 && (j as usize) < ny && (i as usize) < nx {
                    pts[n] = s.refined[g[j as usize][i as usize]].unwrap();
                    n += 1;
                }
            }
            let Some(m) = self.next_on_line(s, &pts[..n], &used) else {
                found = false;
                break;
            };
            s.pos(m, half);
            used.push(m);
        }
        s.used = used;
        found
    }

    /// The candidate continuing a line whose last points are `p` (nearest the
    /// end first): at the linear prediction, else the quadratic one.
    fn next_on_line(&self, s: &Search, p: &[[f32; 2]], used: &[usize]) -> Option<usize> {
        let d1 = [p[0][0] - p[1][0], p[0][1] - p[1][1]];
        let radius = self.config.match_tolerance * norm(d1);
        let linear = [p[0][0] + d1[0], p[0][1] + d1[1]];
        if let Some(m) = s.index.nearest(linear, radius, used) {
            return Some(m);
        }
        if p.len() < 3 {
            return None;
        }
        let d2 = [p[1][0] - p[2][0], p[1][1] - p[2][1]];
        let quadratic = [linear[0] + d1[0] - d2[0], linear[1] + d1[1] - d2[1]];
        s.index.nearest(quadratic, radius, used)
    }

    /// The cells between the corners alternate dark/light, with enough contrast.
    fn cells_alternate(&self, s: &Search, g: &[Vec<usize>]) -> bool {
        let mut cells = Vec::new();
        for j in 0..g.len() - 1 {
            for i in 0..g[0].len() - 1 {
                let k = [g[j][i], g[j][i + 1], g[j + 1][i], g[j + 1][i + 1]];
                let p: Vec<[f32; 2]> = k.iter().map(|&q| s.refined[q].unwrap()).collect();
                let cx = p.iter().map(|q| q[0]).sum::<f32>() / 4.0;
                let cy = p.iter().map(|q| q[1]).sum::<f32>() / 4.0;
                cells.push(((i + j) % 2, mean_3x3(s.image, cx, cy)));
            }
        }
        let mean_of = |parity: usize| {
            let v: Vec<f32> = cells
                .iter()
                .filter(|(p, _)| *p == parity)
                .map(|(_, m)| *m)
                .collect();
            v.iter().sum::<f32>() / v.len().max(1) as f32
        };
        let (m0, m1) = (mean_of(0), mean_of(1));
        if (m0 - m1).abs() < self.config.min_cell_contrast {
            return false;
        }
        let split = 0.5 * (m0 + m1);
        let light0 = m0 > m1;
        let agree = cells
            .iter()
            .filter(|(p, m)| ((*m > split) == light0) == (*p == 0))
            .count();
        agree as f32 >= 0.9 * cells.len() as f32
    }

    /// Row-major: rows along the `cols` axis pointing +x, rows ordered to
    /// keep the image's handedness (never a mirror image).
    fn ordered(&self, mut g: Vec<Vec<[f32; 2]>>) -> Vec<[f32; 2]> {
        if g[0].len() != self.config.cols {
            g = (0..g[0].len())
                .map(|i| g.iter().map(|row| row[i]).collect())
                .collect();
        }
        let last = g[0].len() - 1;
        if g[0][last][0] < g[0][0][0] {
            g.iter_mut().for_each(|row| row.reverse());
        }
        // Rows ordered by the side the columns run to, not by y alone: a board
        // turned near 90° has its rows' first corners at about the same y.
        let along = [g[0][last][0] - g[0][0][0], g[0][last][1] - g[0][0][1]];
        let down = [
            g[g.len() - 1][0][0] - g[0][0][0],
            g[g.len() - 1][0][1] - g[0][0][1],
        ];
        if along[0] * down[1] - along[1] * down[0] < 0.0 {
            g.reverse();
        }
        g.into_iter().flatten().collect()
    }
}

/// The search state of one image: candidates (the first `n_strong` may seed),
/// their index, and the refined positions of those used so far.
struct Search<'a> {
    image: &'a Image<u8>,
    c: &'a [Candidate],
    n_strong: usize,
    index: SpatialIndex,
    refined: Vec<Option<[f32; 2]>>,
    /// Refinement weights by window half-size, built on first use.
    weights: Vec<Vec<f64>>,
    /// Reused buffers: seed-neighbour distances; a growing line's exclusions,
    /// the lattice's candidates then the new line's.
    knn: Vec<(f32, usize)>,
    used: Vec<usize>,
}

impl<'a> Search<'a> {
    fn new(image: &'a Image<u8>, c: &'a [Candidate], n_strong: usize) -> Self {
        let index = SpatialIndex::new(c, image.width(), image.height());
        Self {
            image,
            c,
            n_strong,
            index,
            refined: vec![None; c.len()],
            weights: Vec::new(),
            knn: Vec::new(),
            used: Vec::new(),
        }
    }

    /// The k strong candidates nearest to strong candidate `i`, nearest first
    /// (ties: lower index first).
    fn seed_neighbours(&mut self, i: usize, k: usize) -> Vec<usize> {
        let (c, ci) = (&self.c[..self.n_strong], self.c[i]);
        self.knn.clear();
        self.knn.extend(
            c.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, q)| ((q.x - ci.x).powi(2) + (q.y - ci.y).powi(2), j)),
        );
        let by = |a: &(f32, usize), b: &(f32, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        let k = k.min(self.knn.len());
        if k < self.knn.len() && k > 0 {
            self.knn.select_nth_unstable_by(k - 1, by);
        }
        self.knn[..k].sort_unstable_by(by);
        self.knn[..k].iter().map(|&(_, j)| j).collect()
    }

    /// Candidate k's refined position (computed once).
    fn pos(&mut self, k: usize, half: usize) -> [f32; 2] {
        if let Some(p) = self.refined[k] {
            return p;
        }
        if self.weights.len() <= half {
            self.weights.resize(half + 1, Vec::new());
        }
        if self.weights[half].is_empty() {
            self.weights[half] = gaussian_weights(half);
        }
        let p = refine_corner_with(
            self.image,
            [self.c[k].x, self.c[k].y],
            half,
            &self.weights[half],
        );
        self.refined[k] = Some(p);
        p
    }

    /// The refined positions of a lattice's corners.
    fn positions(&self, g: &[Vec<usize>]) -> Vec<Vec<[f32; 2]>> {
        g.iter()
            .map(|row| row.iter().map(|&k| self.refined[k].unwrap()).collect())
            .collect()
    }
}

/// No row or column bends by more than MAX_BEND at any corner.
fn lines_smooth(g: &[Vec<[f32; 2]>]) -> bool {
    let bent = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        let d2 = norm([a[0] - 2.0 * b[0] + c[0], a[1] - 2.0 * b[1] + c[1]]);
        let spacing = 0.5 * (norm([b[0] - a[0], b[1] - a[1]]) + norm([c[0] - b[0], c[1] - b[1]]));
        d2 > MAX_BEND * spacing
    };
    let (ny, nx) = (g.len(), g[0].len());
    let rows_bent = (0..ny).any(|j| (1..nx - 1).any(|i| bent(g[j][i - 1], g[j][i], g[j][i + 1])));
    let cols_bent = (0..nx).any(|i| (1..ny - 1).any(|j| bent(g[j - 1][i], g[j][i], g[j + 1][i])));
    !rows_bent && !cols_bent
}

/// Cosine of the angle at candidate `o` between `a` and `b`.
fn cos_at(c: &[Candidate], o: usize, a: usize, b: usize) -> f32 {
    let v1 = [c[a].x - c[o].x, c[a].y - c[o].y];
    let v2 = [c[b].x - c[o].x, c[b].y - c[o].y];
    (v1[0] * v2[0] + v1[1] * v2[1]) / (norm(v1) * norm(v2)).max(1e-6)
}

fn norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// `out` (half the image's size): the image at half size, each pixel the mean
/// of a 2×2 block. Every pixel is written.
fn half_size_into(image: &Image<u8>, out: &mut Image<u8>) {
    let (w, h) = (image.width() / 2, image.height() / 2);
    debug_assert!(out.width() == w && out.height() == h);
    for y in 0..h {
        let (r0, r1) = (image.row(2 * y), image.row(2 * y + 1));
        // Pairs of pixels from both rows: no indexing, so it vectorizes.
        let pairs = r0.chunks_exact(2).zip(r1.chunks_exact(2));
        for (o, (a, b)) in out.row_mut(y).iter_mut().zip(pairs) {
            let sum = a[0] as u16 + a[1] as u16 + b[0] as u16 + b[1] as u16;
            *o = ((sum + 2) / 4) as u8;
        }
    }
}

/// Mean of the 3×3 pixels around (x, y).
fn mean_3x3(image: &Image<u8>, x: f32, y: f32) -> f32 {
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    let (w, h) = (image.width() as isize, image.height() as isize);
    let mut sum = 0.0;
    let mut n = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (u, v) = (cx + dx, cy + dy);
            if u >= 0 && v >= 0 && u < w && v < h {
                sum += image.get(u as usize, v as usize) as f32;
                n += 1;
            }
        }
    }
    sum / n.max(1) as f32
}

/// cornerSubPix-style refinement: q solves Σ g gᵀ (p − q) = 0 over the window,
/// gradients Gaussian-weighted. Keeps `p` if the solve is ill-conditioned or
/// leaves the window.
pub fn refine_corner(image: &Image<u8>, p: [f32; 2], half: usize) -> [f32; 2] {
    refine_corner_with(image, p, half, &gaussian_weights(half))
}

/// The refinement window's Gaussian weights, row-major over (2·half + 1)².
fn gaussian_weights(half: usize) -> Vec<f64> {
    let half = half as isize;
    let sigma2 = (0.5 * half as f64 + 0.5).powi(2);
    let n = 2 * half + 1;
    (0..n * n)
        .map(|i| {
            let (dx, dy) = (i % n - half, i / n - half);
            (-((dx * dx + dy * dy) as f64) / (2.0 * sigma2)).exp()
        })
        .collect()
}

/// `refine_corner` with the window's weights given (`gaussian_weights(half)`).
fn refine_corner_with(image: &Image<u8>, p: [f32; 2], half: usize, weight: &[f64]) -> [f32; 2] {
    let (w, h) = (image.width() as isize, image.height() as isize);
    let half = half as isize;
    let n = 2 * half + 1;
    let mut q = [p[0] as f64, p[1] as f64];
    let mut last_centre = None;
    for _ in 0..10 {
        let (cx, cy) = (q[0].round() as isize, q[1].round() as isize);
        // The same window again sums to the same point: q would not move.
        if last_centre == Some((cx, cy)) {
            break;
        }
        last_centre = Some((cx, cy));
        if cx - half < 1 || cy - half < 1 || cx + half >= w - 1 || cy + half >= h - 1 {
            return p;
        }
        let (mut a, mut b, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        let (x0, len) = ((cx - half) as usize, n as usize);
        for dy in -half..=half {
            let y = (cy + dy) as usize;
            // The window's row, one pixel wider each side, and the rows above
            // and below: slices, so the loop below needs no bounds checks.
            let row = &image.row(y)[x0 - 1..x0 + len + 1];
            let (up, down) = (
                &image.row(y - 1)[x0..x0 + len],
                &image.row(y + 1)[x0..x0 + len],
            );
            let weights = &weight[(dy + half) as usize * len..][..len];
            let pixels = row.windows(3).zip(up.iter().zip(down)).zip(weights);
            for (i, ((lr, (&u, &d)), &wgt)) in pixels.enumerate() {
                let x = x0 + i;
                let gx = 0.5 * (lr[2] as f64 - lr[0] as f64);
                let gy = 0.5 * (d as f64 - u as f64);
                let (gxx, gyy, gxy) = (gx * gx * wgt, gy * gy * wgt, gx * gy * wgt);
                a += gxx;
                b += gyy;
                cxy += gxy;
                bx += gxx * x as f64 + gxy * y as f64;
                by += gxy * x as f64 + gyy * y as f64;
            }
        }
        let det = a * b - cxy * cxy;
        if det.abs() < 1e-6 {
            return p;
        }
        let nq = [(b * bx - cxy * by) / det, (a * by - cxy * bx) / det];
        let moved = ((nq[0] - q[0]).powi(2) + (nq[1] - q[1]).powi(2)).sqrt();
        q = nq;
        if moved < 0.01 {
            break;
        }
    }
    if (q[0] - p[0] as f64).abs() > half as f64 || (q[1] - p[1] as f64).abs() > half as f64 {
        return p;
    }
    [q[0] as f32, q[1] as f32]
}

/// Candidates bucketed on a grid, for nearest-candidate queries near a point.
/// One array sorted by bucket, holding the positions themselves: the buckets
/// of one grid row are then one contiguous slice, read without following
/// candidate numbers elsewhere in memory.
struct SpatialIndex {
    /// 1 / the bucket size. The size is a power of two, so multiplying by
    /// this is exactly dividing by the size, without a division's latency.
    per_cell: f32,
    nx: usize,
    ny: usize,
    /// Bucket b holds entries[start[b]..start[b + 1]].
    start: Vec<usize>,
    /// (x, y, candidate), in candidate order within each bucket.
    entries: Vec<(f32, f32, usize)>,
}

impl SpatialIndex {
    fn new(c: &[Candidate], width: usize, height: usize) -> Self {
        let cell = 16.0;
        let nx = (width as f32 / cell).ceil() as usize + 1;
        let ny = (height as f32 / cell).ceil() as usize + 1;
        let bucket = |q: &Candidate| {
            ((q.y / cell) as usize).min(ny - 1) * nx + ((q.x / cell) as usize).min(nx - 1)
        };
        // Counting sort: bucket sizes, their running sum, then each entry into place.
        let mut start = vec![0; nx * ny + 1];
        for q in c {
            start[bucket(q) + 1] += 1;
        }
        for b in 0..nx * ny {
            start[b + 1] += start[b];
        }
        let mut next = start.clone();
        let mut entries = vec![(0.0, 0.0, 0); c.len()];
        for (k, q) in c.iter().enumerate() {
            let b = bucket(q);
            entries[next[b]] = (q.x, q.y, k);
            next[b] += 1;
        }
        Self {
            per_cell: 1.0 / cell,
            nx,
            ny,
            start,
            entries,
        }
    }

    /// The candidate nearest to `p` within `radius`, not in `exclude`.
    fn nearest(&self, p: [f32; 2], radius: f32, exclude: &[usize]) -> Option<usize> {
        // The buckets under the circle's bounding box (padded against rounding).
        let reach = radius + 0.01;
        let bucket = |v: f32| (v * self.per_cell).floor() as isize;
        let (x0, x1) = (
            bucket(p[0] - reach).max(0),
            bucket(p[0] + reach).min(self.nx as isize - 1),
        );
        if x1 < x0 {
            return None;
        }
        // Within the radius, then only strictly nearer: one comparison, against a limit that
        // starts just above radius² (d < next_up(r²) is d <= r²) and then is the best distance.
        let mut limit = (radius * radius).next_up();
        let mut best = None;
        for y in bucket(p[1] - reach).max(0)..=bucket(p[1] + reach).min(self.ny as isize - 1) {
            let row = y as usize * self.nx;
            let span = self.start[row + x0 as usize]..self.start[row + x1 as usize + 1];
            for &(x, y, k) in &self.entries[span] {
                let d = (x - p[0]).powi(2) + (y - p[1]).powi(2);
                if d < limit && !exclude.contains(&k) {
                    (limit, best) = (d, Some(k));
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r5_threshold_is_the_f32_comparison() {
        for t in [
            0.0f32, 0.1, 0.2, 39.9, 40.0, 40.1, 159.99, 160.0, 160.2, 1234.5, -3.0,
        ] {
            let k = r5_at_least(t);
            for r5 in (k as i32 - 3)..=(k as i32 + 3) {
                let r5 = r5 as i16;
                assert_eq!(r5 >= k, r5 as f32 * 0.2 >= t, "t {t}, 5R {r5}");
            }
        }
    }

    #[test]
    fn stored_5r_finds_the_same_candidates_as_f32() {
        // a smooth random texture with some sharp blocks: plenty of candidates
        let (w, h) = (320, 240);
        let mut s = 12345u64;
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let block = ((x / 17 + y / 13) % 2) as u64 * 120;
                img.set(x, y, ((block + (s >> 58)) % 256) as u8);
            }
        }
        let config = CheckerboardConfig::default();
        let f = find_candidates(&chess_response(&img), &config);
        let mut r5 = Image::new(w, h);
        fill_response(&img, &mut r5);
        let i = candidates_in(&r5, r5_at_least(config.grow_response), &config);
        assert!(f.len() > 50);
        assert_eq!(f.len(), i.len());
        for (a, b) in f.iter().zip(&i) {
            assert_eq!((a.x, a.y, a.response), (b.x, b.y, b.response));
        }
    }
}
