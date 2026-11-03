// tests/test_checkerboard.rs — Integration tests for the checkerboard detector.
//
// Boards are rendered through a homography (four image points for the board's
// outer square corners), anti-aliased by 4×4 supersampling, with optional
// barrel distortion, so every test knows the true corner positions.

use rudolf_v::checkerboard::{chess_response, CheckerboardConfig, CheckerboardDetector};
use rudolf_v::image::Image;

const COLS: usize = 9;
const ROWS: usize = 6;
const DARK: f64 = 40.0;
const LIGHT: f64 = 210.0;
const BACKGROUND: f64 = 128.0;

/// Homography mapping the unit square's corners (0,0),(1,0),(1,1),(0,1) to `q`.
fn homography(q: [[f64; 2]; 4]) -> [[f64; 3]; 3] {
    let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut a = [[0.0f64; 9]; 8];
    for k in 0..4 {
        let ([x, y], [u, v]) = (src[k], q[k]);
        a[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gauss-Jordan on the 8×8 system (last column = right-hand side).
    for col in 0..8 {
        let piv = (col..8)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, piv);
        for r in 0..8 {
            if r != col {
                let f = a[r][col] / a[col][col];
                for cc in col..9 {
                    a[r][cc] -= f * a[col][cc];
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
}

fn apply(h: &[[f64; 3]; 3], x: f64, y: f64) -> [f64; 2] {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    [
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    ]
}

fn inverse(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let m = h;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 2, 1, 2) / det,
            -c(0, 2, 1, 2) / det,
            c(0, 1, 1, 2) / det,
        ],
        [
            -c(1, 2, 0, 2) / det,
            c(0, 2, 0, 2) / det,
            -c(0, 1, 0, 2) / det,
        ],
        [
            c(1, 2, 0, 1) / det,
            -c(0, 2, 0, 1) / det,
            c(0, 1, 0, 1) / det,
        ],
    ]
}

/// A board of (COLS+1)×(ROWS+1) squares with a one-square white margin; `q` are
/// the image points of the squares' outer corners (top-left, top-right,
/// bottom-right, bottom-left). `k`: barrel distortion, applied as
/// undistorted = c + (distorted − c)(1 + k r²) around the image centre.
/// Returns the image and the true inner corners, row-major.
fn render(w: usize, h: usize, q: [[f64; 2]; 4], k: f64, seed: u64) -> (Image<u8>, Vec<[f32; 2]>) {
    let (sx, sy) = ((COLS + 1) as f64, (ROWS + 1) as f64);
    let hm = homography(q);
    let inv = inverse(&hm);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let undistort = |x: f64, y: f64| {
        let r2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (cx * cx);
        [
            cx + (x - cx) * (1.0 + k * r2),
            cy + (y - cy) * (1.0 + k * r2),
        ]
    };
    let mut rng = seed;
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for sy_ in 0..4 {
                for sx_ in 0..4 {
                    let [ux, uy] = undistort(
                        x as f64 + (sx_ as f64 + 0.5) / 4.0 - 0.5,
                        y as f64 + (sy_ as f64 + 0.5) / 4.0 - 0.5,
                    );
                    let [bu, bv] = apply(&inv, ux, uy);
                    let (u, v) = (bu * sx, bv * sy);
                    acc += if u >= 0.0 && u < sx && v >= 0.0 && v < sy {
                        if (u.floor() as i64 + v.floor() as i64) % 2 == 0 {
                            DARK
                        } else {
                            LIGHT
                        }
                    } else if u >= -1.0 && u < sx + 1.0 && v >= -1.0 && v < sy + 1.0 {
                        LIGHT
                    } else {
                        BACKGROUND
                    };
                }
            }
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng >> 33) % 9) as f64 - 4.0;
            img.set(x, y, (acc / 16.0 + noise).clamp(0.0, 255.0) as u8);
        }
    }
    let corners = (0..ROWS)
        .flat_map(|j| (0..COLS).map(move |i| (i, j)))
        .map(|(i, j)| {
            let [ux, uy] = apply(&hm, (i + 1) as f64 / sx, (j + 1) as f64 / sy);
            // distorted point: solve undistort(d) = (ux, uy) by fixed-point iteration
            let mut d = [ux, uy];
            for _ in 0..50 {
                let r2 = ((d[0] - cx).powi(2) + (d[1] - cy).powi(2)) / (cx * cx);
                d = [
                    cx + (ux - cx) / (1.0 + k * r2),
                    cy + (uy - cy) / (1.0 + k * r2),
                ];
            }
            [d[0] as f32, d[1] as f32]
        })
        .collect();
    (img, corners)
}

/// Each pixel the mean of `len` pixels along its row: the camera panning
/// during the exposure. Symmetric, so the corners stay where they were, but
/// on a turned board they are poorly determined along the smear.
fn motion_blur(img: &Image<u8>, len: usize) -> Image<u8> {
    let (w, h) = (img.width(), img.height());
    let mut out = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(len / 2), (x + len / 2).min(w - 1));
            let sum: u32 = (lo..=hi).map(|u| img.get(u, y) as u32).sum();
            out.set(x, y, (sum / (hi - lo + 1) as u32) as u8);
        }
    }
    out
}

/// Contrast fading from full at x0 to `last` (a fraction) at x1: glare.
fn fade(img: &mut Image<u8>, x0: f64, x1: f64, last: f64) {
    for y in 0..img.height() {
        for x in 0..img.width() {
            let t = ((x as f64 - x0) / (x1 - x0)).clamp(0.0, 1.0);
            let v = BACKGROUND + (img.get(x, y) as f64 - BACKGROUND) * (1.0 - t * (1.0 - last));
            img.set(x, y, v.round() as u8);
        }
    }
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> [[f64; 2]; 4] {
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

/// A w×h rectangle centred on (cx, cy), turned by `deg`.
fn turned(cx: f64, cy: f64, w: f64, h: f64, deg: f64) -> [[f64; 2]; 4] {
    let (s, c) = deg.to_radians().sin_cos();
    [[-w, -h], [w, -h], [w, h], [-w, h]]
        .map(|[x, y]| [cx + 0.5 * (c * x - s * y), cy + 0.5 * (s * x + c * y)])
}

/// Largest distance between found and true corners, matched in order or with
/// the 180° flip (both describe the same board).
fn max_error(found: &[[f32; 2]], truth: &[[f32; 2]]) -> f32 {
    let err = |order: &mut dyn Iterator<Item = &[f32; 2]>| {
        found
            .iter()
            .zip(order)
            .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
            .fold(0.0, f32::max)
    };
    err(&mut truth.iter()).min(err(&mut truth.iter().rev()))
}

fn detector() -> CheckerboardDetector {
    CheckerboardDetector::new(CheckerboardConfig::default())
}

#[test]
fn response_positive_at_an_x_corner_negative_on_an_edge() {
    let (img, truth) = render(320, 240, rect(60.0, 40.0, 200.0, 140.0), 0.0, 1);
    let r = chess_response(&img);
    let [x, y] = truth[0];
    assert!(r.get(x.round() as usize, y.round() as usize) > 400.0);
    // midway along the edge between the first two corners (an edge between two squares)
    let [x1, y1] = truth[1];
    let (ex, ey) = (((x + x1) / 2.0).round() as usize, y.round() as usize);
    assert!(r.get(ex, ey) < 0.0, "edge response {}", r.get(ex, ey));
    let _ = y1;
}

#[test]
fn upright_board_with_subpixel_corners() {
    let (img, truth) = render(400, 300, rect(80.5, 60.25, 240.0, 168.0), 0.0, 2);
    let found = detector().detect(&img).expect("board");
    assert_eq!(found.len(), COLS * ROWS);
    assert!(
        max_error(&found, &truth) < 0.15,
        "error {}",
        max_error(&found, &truth)
    );
    // row-major, first corner top-left, rows along +x
    assert!(found[0][0] < found[1][0] && found[0][1] < found[COLS][1]);
    assert!((found[0][0] - truth[0][0]).abs() < 0.2 && (found[0][1] - truth[0][1]).abs() < 0.2);
}

#[test]
fn rotated_and_in_perspective() {
    let q = [[120.0, 70.0], [330.0, 110.0], [300.0, 250.0], [70.0, 190.0]];
    let (img, truth) = render(400, 300, q, 0.0, 3);
    let found = detector().detect(&img).expect("board");
    assert!(
        max_error(&found, &truth) < 0.3,
        "error {}",
        max_error(&found, &truth)
    );
}

#[test]
fn turned_near_upright_never_mirrored() {
    // Rows near vertical: ordering them by y alone could number the board as
    // its mirror image. max_error accepts the true order or its 180° flip only.
    for deg in [80.0, 88.0, 90.0, 92.0, 100.0, -90.0] {
        let (img, truth) = render(400, 400, turned(200.3, 200.7, 240.0, 168.0, deg), 0.0, 11);
        let found = detector().detect(&img).expect("board");
        assert!(
            max_error(&found, &truth) < 0.3,
            "{deg}°: error {}",
            max_error(&found, &truth)
        );
    }
}

#[test]
fn small_squares_below_opencvs_limit() {
    // 9 px squares; OpenCV's findChessboardCorners needs about 13
    let (img, truth) = render(200, 150, rect(50.3, 40.7, 90.0, 63.0), 0.0, 4);
    let found = detector().detect(&img).expect("board");
    assert!(
        max_error(&found, &truth) < 0.3,
        "error {}",
        max_error(&found, &truth)
    );
}

#[test]
fn barrel_distorted_raw_image() {
    let (img, truth) = render(480, 360, rect(40.0, 40.0, 400.0, 280.0), 0.15, 5);
    let found = detector().detect(&img).expect("board");
    assert!(
        max_error(&found, &truth) < 0.3,
        "error {}",
        max_error(&found, &truth)
    );
}

#[test]
fn motion_blur_across_a_turned_board() {
    // 9 px of horizontal smear across 22 px squares turned 35°: no board at
    // full size, found at half size
    let (img, truth) = render(400, 300, turned(200.3, 150.7, 220.0, 154.0, 35.0), 0.0, 9);
    let found = detector().detect(&motion_blur(&img, 9)).expect("board");
    assert!(
        max_error(&found, &truth) < 0.6,
        "error {}",
        max_error(&found, &truth)
    );
}

#[test]
fn weak_corners_in_glare() {
    // contrast fades to a tenth across the board: the far corners are too
    // weak to seed a lattice but can extend one
    let (mut img, truth) = render(400, 300, rect(60.3, 50.7, 280.0, 196.0), 0.0, 10);
    fade(&mut img, 80.0, 330.0, 0.1);
    let found = detector().detect(&img).expect("board");
    assert!(
        max_error(&found, &truth) < 0.3,
        "error {}",
        max_error(&found, &truth)
    );
}

#[test]
fn half_a_board_is_no_board() {
    // the right two columns of squares fall outside the image
    let (img, _) = render(240, 240, rect(40.0, 40.0, 250.0, 175.0), 0.0, 6);
    assert!(detector().detect(&img).is_none());
}

#[test]
fn a_bigger_board_is_no_board() {
    // a 9x6 board holds 7x6 and 9x5 lattices; they are not those boards
    let (img, _) = render(400, 300, rect(80.5, 60.25, 240.0, 168.0), 0.0, 12);
    for (cols, rows) in [(7, 6), (9, 5), (8, 5)] {
        let d = CheckerboardDetector::new(CheckerboardConfig {
            cols,
            rows,
            ..Default::default()
        });
        assert!(
            d.detect(&img).is_none(),
            "{cols}x{rows} found in a 9x6 board"
        );
    }
}

#[test]
fn clutter_is_no_board() {
    let (w, h) = (320, 240);
    let mut img = Image::from_vec(w, h, vec![BACKGROUND as u8; w * h]);
    let mut rng = 7u64;
    let mut next = |n: u64| {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) % n
    };
    for _ in 0..60 {
        let (x0, y0, rw, rh, v) = (
            next(300),
            next(220),
            5 + next(40),
            5 + next(40),
            next(256) as u8,
        );
        for y in y0..(y0 + rh).min(h as u64) {
            for x in x0..(x0 + rw).min(w as u64) {
                img.set(x as usize, y as usize, v);
            }
        }
    }
    assert!(detector().detect(&img).is_none());
}

#[test]
fn region_gives_whole_image_pixels() {
    let (img, truth) = render(400, 300, rect(150.0, 100.0, 180.0, 126.0), 0.0, 8);
    let found = detector()
        .detect_region(&img, 120, 70, 240, 190)
        .expect("board");
    assert!(
        max_error(&found, &truth) < 0.15,
        "error {}",
        max_error(&found, &truth)
    );
    assert!(detector().detect_region(&img, 0, 0, 140, 300).is_none());
}
