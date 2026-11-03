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
//      the most perpendicular other neighbour, grown by whole rows and columns.
//      Each new corner is predicted from the last two on its line (linearly;
//      from three, quadratically, if that misses), so the lattice follows
//      perspective and lens distortion (raw, unrectified images work); the
//      nearest candidate within a fraction of the local spacing is taken. A
//      full cols×rows lattice (either orientation) whose cells alternate
//      dark/light and whose rows and columns bend smoothly is the board; a
//      partial board is no board.
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
//      +y. A 180° turn of the board gives the same order flipped; the board's
//      centre and distance are the same either way.
//
// The output is in the input image's pixels: on a raw image, raw pixels. Undo
// lens distortion on these points (camera-geometry), not on the image.
//
// NEW RUST CONCEPTS:
// - core::array::from_fn to build a fixed-size array from a closure (the ring
//   offsets for the image's stride).
// - A Vec<Vec<usize>> lattice grown at either end (insert(0, ...) / push).

use crate::image::Image;

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

/// Nearest neighbours considered when forming a seed quad.
const SEED_NEIGHBOURS: usize = 8;

/// Largest second difference along a row or column (a corner's offset from
/// its two neighbours' midpoint, ×2), as a fraction of the local spacing.
/// Perspective and lens distortion stay far below it; corners that heavy blur
/// scattered, or merged into one, do not.
const MAX_BEND: f32 = 0.5;

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
    /// Strongest candidates tried as seeds before giving up.
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
            max_seeds: 40,
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
    let (w, h) = (image.width(), image.height());
    let mut out = Image::new(w, h);
    if w <= 2 * RADIUS || h <= 2 * RADIUS {
        return out;
    }
    let stride = image.stride();
    let data = image.as_slice();
    let ring: [isize; 16] = core::array::from_fn(|k| RING[k].1 * stride as isize + RING[k].0);

    let fill_row = |y: usize, out_row: &mut [f32]| {
        for x in RADIUS..w - RADIUS {
            let c = (y * stride + x) as isize;
            let s: [i32; 16] = core::array::from_fn(|k| data[(c + ring[k]) as usize] as i32);
            let sum: i32 = (0..4)
                .map(|n| ((s[n] + s[n + 8]) - (s[n + 4] + s[n + 12])).abs())
                .sum();
            let diff: i32 = (0..8).map(|n| (s[n] - s[n + 8]).abs()).sum();
            let ring_mean = s.iter().sum::<i32>() as f32 / 16.0;
            let local = (data[c as usize] as i32
                + data[(c - 1) as usize] as i32
                + data[(c + 1) as usize] as i32
                + data[(c - stride as isize) as usize] as i32
                + data[(c + stride as isize) as usize] as i32) as f32
                / 5.0;
            out_row[x] = (sum - diff) as f32 - 16.0 * (ring_mean - local).abs();
        }
    };

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
    out
}

/// Local maxima of the response (5×5) down to `grow_response`, strongest
/// first, sub-pixel.
pub fn find_candidates(response: &Image<f32>, config: &CheckerboardConfig) -> Vec<Candidate> {
    let (w, h) = (response.width(), response.height());
    let m = RADIUS + 2;
    if w <= 2 * m || h <= 2 * m {
        return Vec::new();
    }
    let threshold = config.grow_response;
    let mut out = Vec::new();
    for y in m..h - m {
        let row = response.row(y);
        for x in m..w - m {
            let v = row[x];
            if v < threshold {
                continue;
            }
            // Strict maximum over earlier pixels, at least equal to later ones,
            // so a plateau yields exactly one candidate.
            let mut is_max = true;
            'win: for dy in -2isize..=2 {
                let r = response.row((y as isize + dy) as usize);
                for dx in -2isize..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let u = r[(x as isize + dx) as usize];
                    let earlier = dy < 0 || (dy == 0 && dx < 0);
                    if u > v || (earlier && u == v) {
                        is_max = false;
                        break 'win;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let (l, r) = (row[x - 1], row[x + 1]);
            let (u, d) = (response.row(y - 1)[x], response.row(y + 1)[x]);
            out.push(Candidate {
                x: x as f32 + parabola_peak(l, v, r),
                y: y as f32 + parabola_peak(u, v, d),
                response: v,
            });
        }
    }
    out.sort_by(|a, b| b.response.total_cmp(&a.response));
    out.truncate(config.max_candidates);
    out
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
}

impl CheckerboardDetector {
    pub fn new(config: CheckerboardConfig) -> Self {
        Self { config }
    }

    /// The board's `cols × rows` inner corners in pixels (row-major), or None
    /// when no complete board is found.
    pub fn detect(&self, image: &Image<u8>) -> Option<Vec<[f32; 2]>> {
        if let Some((corners, _)) = self.detect_at_scale(image) {
            return Some(self.ordered(corners));
        }
        let (corners, half) = self.detect_at_scale(&half_size(image))?;
        // A half-size pixel (u, v) covers full-size pixels 2u..2u+1.
        let corners: Vec<Vec<[f32; 2]>> = corners
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|[u, v]| refine_corner(image, [2.0 * u + 0.5, 2.0 * v + 0.5], 2 * half))
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
    fn detect_at_scale(&self, image: &Image<u8>) -> Option<(Vec<Vec<[f32; 2]>>, usize)> {
        let candidates = find_candidates(&chess_response(image), &self.config);
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
        let mut search = Search::new(image, &candidates, n_strong);
        let (lattice, half) = self.find_lattice(&mut search)?;
        Some((search.positions(&lattice), half))
    }

    /// Seed quads of strong candidates, strongest first, each grown into a
    /// lattice; the first complete board whose cells alternate dark/light and
    /// whose lines are smooth, with its refinement half-window.
    fn find_lattice(&self, s: &mut Search) -> Option<(Vec<Vec<usize>>, usize)> {
        for seed in 0..s.n_strong.min(self.config.max_seeds) {
            let near = nearest(&s.c[..s.n_strong], seed, SEED_NEIGHBOURS);
            for &a in near.iter().take(4) {
                // Second axis: the most perpendicular neighbours first, so a
                // diagonal (also a lattice vector) is tried last.
                let mut others: Vec<(f32, usize)> = near
                    .iter()
                    .filter(|&&b| b != a)
                    .map(|&b| (cos_at(s.c, seed, a, b).abs(), b))
                    .collect();
                others.sort_by(|x, y| x.0.total_cmp(&y.0));
                for (_, b) in others {
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
        let d = s.index.nearest(
            s.c,
            p3,
            self.config.match_tolerance * n1.min(n2),
            &[seed, a, b],
        )?;
        s.pos(d, half);
        Some((vec![vec![seed, a], vec![b, d]], half))
    }

    /// Grow the lattice by whole rows and columns until nothing more fits;
    /// Some only for a complete cols×rows (or rows×cols) board. Predictions use
    /// refined positions: the candidates' raw peaks are only good to ~0.5 px,
    /// which extrapolation would amplify.
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
                let lines = if along_x { ny } else { nx };
                let mut used: Vec<usize> = g.iter().flatten().copied().collect();
                let mut new_line = Vec::with_capacity(lines);
                for k in 0..lines {
                    // The line's points from the end being extended, inwards.
                    let pts: Vec<[f32; 2]> = (0..3isize)
                        .filter_map(|t| {
                            let (j, i) = match side {
                                0 => (k as isize, nx as isize - 1 - t),
                                1 => (k as isize, t),
                                2 => (ny as isize - 1 - t, k as isize),
                                _ => (t, k as isize),
                            };
                            (j >= 0 && i >= 0 && (j as usize) < ny && (i as usize) < nx)
                                .then(|| s.refined[g[j as usize][i as usize]].unwrap())
                        })
                        .collect();
                    let Some(m) = self.next_on_line(s, &pts, &used) else {
                        break;
                    };
                    s.pos(m, half);
                    used.push(m);
                    new_line.push(m);
                }
                if new_line.len() < lines {
                    continue;
                }
                match side {
                    0 => g
                        .iter_mut()
                        .zip(&new_line)
                        .for_each(|(row, &m)| row.push(m)),
                    1 => g
                        .iter_mut()
                        .zip(&new_line)
                        .for_each(|(row, &m)| row.insert(0, m)),
                    2 => g.push(new_line),
                    _ => g.insert(0, new_line),
                }
                grew = true;
            }
            if !grew {
                break;
            }
        }
        let (ny, nx) = (g.len(), g[0].len());
        ((ny, nx) == (rows, cols) || (ny, nx) == (cols, rows)).then_some(g)
    }

    /// The candidate continuing a line whose last points are `p` (nearest the
    /// end first): at the linear prediction, else the quadratic one.
    fn next_on_line(&self, s: &Search, p: &[[f32; 2]], used: &[usize]) -> Option<usize> {
        let d1 = [p[0][0] - p[1][0], p[0][1] - p[1][1]];
        let radius = self.config.match_tolerance * norm(d1);
        let linear = [p[0][0] + d1[0], p[0][1] + d1[1]];
        if let Some(m) = s.index.nearest(s.c, linear, radius, used) {
            return Some(m);
        }
        if p.len() < 3 {
            return None;
        }
        let d2 = [p[1][0] - p[2][0], p[1][1] - p[2][1]];
        let quadratic = [linear[0] + d1[0] - d2[0], linear[1] + d1[1] - d2[1]];
        s.index.nearest(s.c, quadratic, radius, used)
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

    /// Row-major: rows along the `cols` axis pointing +x, rows ordered +y.
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
        if g[g.len() - 1][0][1] < g[0][0][1] {
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
        }
    }

    /// Candidate k's refined position (computed once).
    fn pos(&mut self, k: usize, half: usize) -> [f32; 2] {
        if let Some(p) = self.refined[k] {
            return p;
        }
        let p = refine_corner(self.image, [self.c[k].x, self.c[k].y], half);
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

/// The k nearest candidates to candidate `i`, nearest first.
fn nearest(c: &[Candidate], i: usize, k: usize) -> Vec<usize> {
    let mut d: Vec<(f32, usize)> = c
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(j, q)| ((q.x - c[i].x).powi(2) + (q.y - c[i].y).powi(2), j))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.into_iter().take(k).map(|(_, j)| j).collect()
}

/// The image at half size, each pixel the mean of a 2×2 block.
fn half_size(image: &Image<u8>) -> Image<u8> {
    let (w, h) = (image.width() / 2, image.height() / 2);
    let mut out = Image::new(w, h);
    for y in 0..h {
        let (r0, r1) = (image.row(2 * y), image.row(2 * y + 1));
        for (x, o) in out.row_mut(y).iter_mut().enumerate() {
            let sum =
                r0[2 * x] as u16 + r0[2 * x + 1] as u16 + r1[2 * x] as u16 + r1[2 * x + 1] as u16;
            *o = ((sum + 2) / 4) as u8;
        }
    }
    out
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
    let (w, h) = (image.width() as isize, image.height() as isize);
    let half = half as isize;
    let sigma2 = (0.5 * half as f64 + 0.5).powi(2);
    let mut q = [p[0] as f64, p[1] as f64];
    for _ in 0..10 {
        let (cx, cy) = (q[0].round() as isize, q[1].round() as isize);
        if cx - half < 1 || cy - half < 1 || cx + half >= w - 1 || cy + half >= h - 1 {
            return p;
        }
        let (mut a, mut b, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut bx, mut by) = (0.0f64, 0.0f64);
        for dy in -half..=half {
            for dx in -half..=half {
                let (x, y) = ((cx + dx) as usize, (cy + dy) as usize);
                let gx = 0.5 * (image.get(x + 1, y) as f64 - image.get(x - 1, y) as f64);
                let gy = 0.5 * (image.get(x, y + 1) as f64 - image.get(x, y - 1) as f64);
                let wgt = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma2)).exp();
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
struct SpatialIndex {
    cell: f32,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
}

impl SpatialIndex {
    fn new(c: &[Candidate], width: usize, height: usize) -> Self {
        let cell = 16.0;
        let nx = (width as f32 / cell).ceil() as usize + 1;
        let ny = (height as f32 / cell).ceil() as usize + 1;
        let mut buckets = vec![Vec::new(); nx * ny];
        for (k, q) in c.iter().enumerate() {
            let (bx, by) = ((q.x / cell) as usize, (q.y / cell) as usize);
            buckets[by.min(ny - 1) * nx + bx.min(nx - 1)].push(k);
        }
        Self {
            cell,
            nx,
            ny,
            buckets,
        }
    }

    /// The candidate nearest to `p` within `radius`, not in `exclude`.
    fn nearest(
        &self,
        c: &[Candidate],
        p: [f32; 2],
        radius: f32,
        exclude: &[usize],
    ) -> Option<usize> {
        let r = (radius / self.cell).ceil() as isize;
        let (bx, by) = (
            (p[0] / self.cell).floor() as isize,
            (p[1] / self.cell).floor() as isize,
        );
        let mut best: Option<(f32, usize)> = None;
        for y in (by - r).max(0)..=(by + r).min(self.ny as isize - 1) {
            for x in (bx - r).max(0)..=(bx + r).min(self.nx as isize - 1) {
                for &k in &self.buckets[y as usize * self.nx + x as usize] {
                    let d = (c[k].x - p[0]).powi(2) + (c[k].y - p[1]).powi(2);
                    if d <= radius * radius
                        && best.is_none_or(|(bd, _)| d < bd)
                        && !exclude.contains(&k)
                    {
                        best = Some((d, k));
                    }
                }
            }
        }
        best.map(|(_, k)| k)
    }
}
