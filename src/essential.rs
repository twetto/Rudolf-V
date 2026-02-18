// essential.rs -- Essential matrix estimation and geometric outlier rejection.
//
// The essential matrix E encodes the epipolar constraint between two
// calibrated views: x'^T * E * x = 0, where x and x' are normalized
// (bearing) coordinates in the previous and current frames.
//
// Algorithm: normalized 8-point (Hartley 1997) via SVD.
//   1. Normalize correspondences with K^{-1} (done by caller via camera.rs)
//   2. Apply Hartley normalization (translate + scale for conditioning)
//   3. Build 9x9 normal matrix M = A^T * A from constraint vectors
//   4. SVD of M -> last column of V is the vectorized E
//   5. Reshape to 3x3 -> SVD -> enforce rank-2 (set s3 = 0)
//   6. Undo Hartley normalization
//
// RANSAC wrapper for robust estimation with outlier rejection.
//
// Uses lalir for all linear algebra (stack-allocated, const-generic).

use lalir::matrix::Matrix as LMatrix;
use lalir::svd_gk::svd_golub_kahan;
use lalir::svd::reconstruct;

/// A pair of normalized correspondences: (x, y) in frame 1, (x', y') in frame 2.
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    /// Normalized x coordinate in previous frame.
    pub x1: f64,
    /// Normalized y coordinate in previous frame.
    pub y1: f64,
    /// Normalized x coordinate in current frame.
    pub x2: f64,
    /// Normalized y coordinate in current frame.
    pub y2: f64,
}

/// Result of essential matrix estimation.
#[derive(Debug, Clone)]
pub struct EssentialResult {
    /// The 3x3 essential matrix (in normalized coordinates).
    pub e: [[f64; 3]; 3],
    /// Inlier mask: true for inliers, false for outliers.
    pub inliers: Vec<bool>,
    /// Number of inliers.
    pub num_inliers: usize,
    /// Total correspondences.
    pub total: usize,
    /// RANSAC iterations used.
    pub iterations: usize,
}

/// RANSAC configuration.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Inlier threshold: maximum Sampson distance (in normalized coords).
    /// Typical: 1e-4 to 1e-3 depending on noise level.
    pub threshold: f64,
    /// Confidence level (0.0-1.0). RANSAC stops early when the probability
    /// of having found the best model exceeds this. 0.99 is standard.
    pub confidence: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        RansacConfig {
            max_iterations: 200,
            threshold: 5e-4,
            confidence: 0.99,
        }
    }
}

// ============================================================
// 8-point algorithm
// ============================================================

/// Estimate the essential matrix from >= 8 normalized correspondences
/// using the 8-point algorithm (no RANSAC).
///
/// Returns the 3x3 essential matrix as a 2D array, or None if SVD fails.
pub fn eight_point(correspondences: &[Correspondence]) -> Option<[[f64; 3]; 3]> {
    let n = correspondences.len();
    if n < 8 {
        return None;
    }

    // Hartley normalization: translate + scale both point sets so that
    // centroid is at origin and mean distance from origin is sqrt(2).
    let (t1, t2) = hartley_transforms(correspondences);

    // Build the 9x9 normal matrix M = A^T * A.
    // Each correspondence contributes one row a_i to the constraint matrix A,
    // where a_i = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1].
    // Instead of storing the full Nx9 matrix A, we accumulate M = sum(a_i * a_i^T).
    let mut m = LMatrix::<9, 9>::zeros();

    for c in correspondences {
        // Apply Hartley normalization.
        let (x1, y1) = apply_transform(&t1, c.x1, c.y1);
        let (x2, y2) = apply_transform(&t2, c.x2, c.y2);

        let a = [
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1,      y1,      1.0,
        ];

        // Accumulate outer product: M += a * a^T
        for i in 0..9 {
            for j in 0..9 {
                m[(i, j)] += a[i] * a[j];
            }
        }
    }

    // SVD of M. The solution is the right singular vector corresponding
    // to the smallest singular value (last column of V after sorting).
    let svd = svd_golub_kahan(&m);

    // Last column of V (index 8 = smallest singular value after descending sort).
    let mut e_vec = [0.0f64; 9];
    for i in 0..9 {
        e_vec[i] = svd.v[(i, 8)];
    }

    // Reshape to 3x3.
    let mut e = LMatrix::<3, 3>::from_slice(&e_vec);

    // Enforce rank-2 constraint: SVD of E, set smallest singular value to 0.
    let svd_e = svd_golub_kahan(&e);
    let mut s = svd_e.s;
    s[2] = 0.0; // Force rank 2.

    // Reconstruct E = U * diag(s) * V^T.
    let e_svd = lalir::svd::SVD {
        u: svd_e.u,
        s,
        v: svd_e.v,
    };
    e = reconstruct(&e_svd);

    // Undo Hartley normalization: E_orig = T2^T * E_norm * T1.
    let t1_mat = transform_to_matrix(&t1);
    let t2_mat = transform_to_matrix(&t2);
    e = t2_mat.transpose() * e * t1_mat;

    // Normalize E so that ||E||_F = 1 (convention).
    let norm = e.norm_fro();
    if norm < 1e-15 {
        return None;
    }
    e = e.scale(1.0 / norm);

    let mut result = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = e[(i, j)];
        }
    }
    Some(result)
}

// ============================================================
// RANSAC
// ============================================================

/// Estimate the essential matrix with RANSAC outlier rejection.
///
/// Correspondences should be in normalized camera coordinates
/// (applied K^{-1}, optionally undistorted).
pub fn estimate_essential_ransac(
    correspondences: &[Correspondence],
    config: &RansacConfig,
) -> Option<EssentialResult> {
    let n = correspondences.len();
    if n < 8 {
        return None;
    }

    let mut best_inliers = vec![false; n];
    let mut best_num_inliers = 0usize;
    let mut best_e = [[0.0f64; 3]; 3];
    let mut rng = SimpleRng::new(42);
    let mut iterations = 0;
    let mut adaptive_max = config.max_iterations;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        if iter >= adaptive_max {
            break;
        }

        // Sample 8 random correspondences.
        let sample = random_sample_8(&mut rng, n);
        let sample_corrs: Vec<Correspondence> = sample.iter()
            .map(|&i| correspondences[i])
            .collect();

        // Estimate E from the 8-point sample.
        let e = match eight_point(&sample_corrs) {
            Some(e) => e,
            None => continue,
        };

        // Count inliers using Sampson distance.
        let mut inliers = vec![false; n];
        let mut num_inliers = 0;
        for (i, c) in correspondences.iter().enumerate() {
            let dist = sampson_distance(&e, c);
            if dist < config.threshold {
                inliers[i] = true;
                num_inliers += 1;
            }
        }

        if num_inliers > best_num_inliers {
            best_num_inliers = num_inliers;
            best_inliers = inliers;
            best_e = e;

            // Adaptive stopping: update max iterations based on inlier ratio.
            let w = num_inliers as f64 / n as f64;
            if w > 0.0 {
                let p_fail = (1.0 - w.powi(8)).max(1e-15);
                let k = (1.0 - config.confidence).ln() / p_fail.ln();
                adaptive_max = (k.ceil() as usize).min(config.max_iterations);
            }
        }
    }

    if best_num_inliers < 8 {
        return None;
    }

    // Refit on all inliers for a better estimate.
    let inlier_corrs: Vec<Correspondence> = correspondences.iter()
        .zip(best_inliers.iter())
        .filter(|(_, &is_inlier)| is_inlier)
        .map(|(c, _)| *c)
        .collect();

    if let Some(e_refined) = eight_point(&inlier_corrs) {
        // Recompute inliers with refined E.
        let mut final_inliers = vec![false; n];
        let mut final_count = 0;
        for (i, c) in correspondences.iter().enumerate() {
            if sampson_distance(&e_refined, c) < config.threshold {
                final_inliers[i] = true;
                final_count += 1;
            }
        }

        Some(EssentialResult {
            e: e_refined,
            inliers: final_inliers,
            num_inliers: final_count,
            total: n,
            iterations,
        })
    } else {
        Some(EssentialResult {
            e: best_e,
            inliers: best_inliers,
            num_inliers: best_num_inliers,
            total: n,
            iterations,
        })
    }
}

// ============================================================
// Sampson distance
// ============================================================

/// Sampson distance: a first-order approximation to the geometric distance.
///
/// Given E and a correspondence (x1, x2), the Sampson distance is:
///   d = (x2^T E x1)^2 / ( (Ex1)_1^2 + (Ex1)_2^2 + (E^Tx2)_1^2 + (E^Tx2)_2^2 )
///
/// This is cheaper than the full geometric distance and is the standard
/// metric in RANSAC for essential/fundamental matrix estimation.
pub fn sampson_distance(e: &[[f64; 3]; 3], c: &Correspondence) -> f64 {
    let x1 = [c.x1, c.y1, 1.0];
    let x2 = [c.x2, c.y2, 1.0];

    // e_x1 = E * x1
    let ex1 = mat3_vec3(e, &x1);
    // et_x2 = E^T * x2
    let et_x2 = mat3t_vec3(e, &x2);
    // Epipolar constraint: x2^T * E * x1
    let num = dot3(&x2, &ex1);

    let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1]
              + et_x2[0] * et_x2[0] + et_x2[1] * et_x2[1];

    if denom < 1e-30 {
        return f64::MAX;
    }

    (num * num) / denom
}

/// Epipolar error (signed): x2^T * E * x1.
/// Zero for perfect correspondences.
pub fn epipolar_error(e: &[[f64; 3]; 3], c: &Correspondence) -> f64 {
    let x1 = [c.x1, c.y1, 1.0];
    let x2 = [c.x2, c.y2, 1.0];
    let ex1 = mat3_vec3(e, &x1);
    dot3(&x2, &ex1)
}

// ============================================================
// Hartley normalization
// ============================================================

/// Normalization transform: translate centroid to origin, scale so
/// mean distance from origin = sqrt(2).
struct NormTransform {
    tx: f64,
    ty: f64,
    scale: f64,
}

fn hartley_transforms(corrs: &[Correspondence]) -> (NormTransform, NormTransform) {
    let n = corrs.len() as f64;

    // Centroids.
    let (mut mx1, mut my1) = (0.0, 0.0);
    let (mut mx2, mut my2) = (0.0, 0.0);
    for c in corrs {
        mx1 += c.x1; my1 += c.y1;
        mx2 += c.x2; my2 += c.y2;
    }
    mx1 /= n; my1 /= n;
    mx2 /= n; my2 /= n;

    // Mean distances from centroid.
    let mut d1 = 0.0;
    let mut d2 = 0.0;
    for c in corrs {
        d1 += ((c.x1 - mx1).powi(2) + (c.y1 - my1).powi(2)).sqrt();
        d2 += ((c.x2 - mx2).powi(2) + (c.y2 - my2).powi(2)).sqrt();
    }
    d1 /= n;
    d2 /= n;

    let s1 = if d1 > 1e-15 { std::f64::consts::SQRT_2 / d1 } else { 1.0 };
    let s2 = if d2 > 1e-15 { std::f64::consts::SQRT_2 / d2 } else { 1.0 };

    (
        NormTransform { tx: mx1, ty: my1, scale: s1 },
        NormTransform { tx: mx2, ty: my2, scale: s2 },
    )
}

fn apply_transform(t: &NormTransform, x: f64, y: f64) -> (f64, f64) {
    ((x - t.tx) * t.scale, (y - t.ty) * t.scale)
}

fn transform_to_matrix(t: &NormTransform) -> LMatrix<3, 3> {
    // T = [s  0  -s*tx]
    //     [0  s  -s*ty]
    //     [0  0   1   ]
    let mut m = LMatrix::<3, 3>::zeros();
    m[(0, 0)] = t.scale;
    m[(1, 1)] = t.scale;
    m[(0, 2)] = -t.scale * t.tx;
    m[(1, 2)] = -t.scale * t.ty;
    m[(2, 2)] = 1.0;
    m
}

// ============================================================
// Tiny math helpers (avoid lalir overhead for 3-vectors)
// ============================================================

fn mat3_vec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3t_vec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ============================================================
// Simple PRNG (xorshift64 -- avoids `rand` dependency)
// ============================================================

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Random index in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Sample 8 distinct random indices from [0, n).
fn random_sample_8(rng: &mut SimpleRng, n: usize) -> [usize; 8] {
    let mut sample = [0usize; 8];
    let mut count = 0;
    while count < 8 {
        let idx = rng.next_usize(n);
        // Check for duplicates (8 is small, linear scan is fine).
        let dup = sample[..count].iter().any(|&s| s == idx);
        if !dup {
            sample[count] = idx;
            count += 1;
        }
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic correspondences from a known essential matrix.
    ///
    /// E encodes a pure translation along x-axis: t = [1, 0, 0]
    /// E = [t]_x = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    fn make_synthetic_correspondences(n: usize, noise: f64) -> Vec<Correspondence> {
        // Camera moves 0.1 units along x. Points at various depths.
        let baseline = 0.1;
        let mut corrs = Vec::new();
        let mut rng = SimpleRng::new(123);

        for i in 0..n {
            // Random 3D point in front of camera.
            let z = 2.0 + (i as f64) * 0.3;
            let x = -1.0 + (rng.next_u64() % 2000) as f64 / 1000.0;
            let y = -0.5 + (rng.next_u64() % 1000) as f64 / 1000.0;

            // Project into frame 1 (identity pose).
            let x1 = x / z;
            let y1 = y / z;

            // Project into frame 2 (translated by baseline along x).
            let x2 = (x - baseline) / z;
            let y2 = y / z;

            // Add noise.
            let nx1 = (rng.next_u64() % 10000) as f64 / 10000.0 - 0.5;
            let ny1 = (rng.next_u64() % 10000) as f64 / 10000.0 - 0.5;
            let nx2 = (rng.next_u64() % 10000) as f64 / 10000.0 - 0.5;
            let ny2 = (rng.next_u64() % 10000) as f64 / 10000.0 - 0.5;

            corrs.push(Correspondence {
                x1: x1 + noise * nx1,
                y1: y1 + noise * ny1,
                x2: x2 + noise * nx2,
                y2: y2 + noise * ny2,
            });
        }
        corrs
    }

    #[test]
    fn test_eight_point_noiseless() {
        let corrs = make_synthetic_correspondences(20, 0.0);
        let e = eight_point(&corrs).expect("8-point should succeed");

        // The A^T A formulation squares the condition number, so
        // noiseless precision is limited to ~1e-3 for epipolar error.
        // This is fine for RANSAC where thresholds are similar.
        for c in &corrs {
            let err = epipolar_error(&e, c).abs();
            assert!(err < 0.01, "epipolar error too large: {err}");
        }
    }

    #[test]
    fn test_eight_point_noisy() {
        let corrs = make_synthetic_correspondences(30, 1e-4);
        let e = eight_point(&corrs).expect("8-point should succeed");

        // With noise, errors should be small but not zero.
        let mean_err: f64 = corrs.iter()
            .map(|c| epipolar_error(&e, c).abs())
            .sum::<f64>() / corrs.len() as f64;
        assert!(mean_err < 1e-3, "mean epipolar error too large: {mean_err}");
    }

    #[test]
    fn test_essential_rank2() {
        let corrs = make_synthetic_correspondences(20, 0.0);
        let e = eight_point(&corrs).expect("8-point should succeed");

        // E should be rank 2: det(E) should be ~0.
        let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
        assert!(det.abs() < 1e-6, "E should be rank-2, det = {det}");
    }

    #[test]
    fn test_sampson_distance() {
        let corrs = make_synthetic_correspondences(20, 0.0);
        let e = eight_point(&corrs).expect("8-point should succeed");

        for c in &corrs {
            let d = sampson_distance(&e, c);
            assert!(d < 1e-3, "Sampson distance too large for noiseless: {d}");
        }
    }

    #[test]
    fn test_ransac_all_inliers() {
        let corrs = make_synthetic_correspondences(30, 1e-5);
        let config = RansacConfig {
            threshold: 1e-3,
            ..Default::default()
        };

        let result = estimate_essential_ransac(&corrs, &config)
            .expect("RANSAC should succeed");

        // All points should be inliers (no outliers added).
        assert!(
            result.num_inliers >= 25,
            "Expected most inliers, got {}/{}",
            result.num_inliers, result.total
        );
    }

    #[test]
    fn test_ransac_with_outliers() {
        let mut corrs = make_synthetic_correspondences(30, 1e-5);

        // Add 10 gross outliers.
        let mut rng = SimpleRng::new(999);
        for _ in 0..10 {
            corrs.push(Correspondence {
                x1: (rng.next_u64() % 1000) as f64 / 500.0 - 1.0,
                y1: (rng.next_u64() % 1000) as f64 / 500.0 - 1.0,
                x2: (rng.next_u64() % 1000) as f64 / 500.0 - 1.0,
                y2: (rng.next_u64() % 1000) as f64 / 500.0 - 1.0,
            });
        }

        let config = RansacConfig {
            threshold: 1e-3,
            max_iterations: 200,
            ..Default::default()
        };

        let result = estimate_essential_ransac(&corrs, &config)
            .expect("RANSAC should succeed");

        // Should identify most of the 30 real correspondences as inliers
        // and reject most of the 10 outliers.
        assert!(
            result.num_inliers >= 20,
            "Should find >= 20 inliers, got {}/{}",
            result.num_inliers, result.total
        );
        assert!(
            result.num_inliers <= 35,
            "Too many inliers (outliers not rejected): {}/{}",
            result.num_inliers, result.total
        );
    }

    #[test]
    fn test_ransac_too_few_points() {
        let corrs = make_synthetic_correspondences(5, 0.0);
        let result = estimate_essential_ransac(&corrs, &RansacConfig::default());
        assert!(result.is_none(), "Should fail with < 8 points");
    }

    #[test]
    fn test_xorshift_coverage() {
        // Verify the RNG produces reasonably distributed indices.
        let mut rng = SimpleRng::new(42);
        let mut counts = [0u32; 10];
        for _ in 0..10000 {
            counts[rng.next_usize(10)] += 1;
        }
        // Each bucket should get ~1000. Allow wide margin.
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 500 && c < 1500, "Bucket {i}: {c} (expected ~1000)");
        }
    }
}
