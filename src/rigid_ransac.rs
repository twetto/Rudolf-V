// rigid_ransac.rs -- Rigid 3D-3D motion estimation via Horn + RANSAC.
//
// Given pairs of 3D points (p1_i, p2_i) observed in the same frame across
// two timesteps, fits R, t minimizing sum ||p2_i - (R p1_i + t)||^2 via
// Horn's closed-form SVD method (Arun 1987 / Horn 1987), then wraps in
// RANSAC for outlier rejection.
//
// Used by the stereo VIO frontend to reject feature tracks that don't fit
// the dominant rigid ego-motion — typically moving objects or stereo
// mismatches that survived the 1D inverse-depth GN refinement.

use nalgebra::SMatrix;

#[derive(Debug, Clone, Copy)]
pub struct Correspondence3d {
    pub p1: [f64; 3],
    pub p2: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Rigid3dResult {
    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
    pub inliers: Vec<bool>,
    pub num_inliers: usize,
    pub total: usize,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct Rigid3dRansacConfig {
    pub max_iterations: usize,
    pub threshold_meters: f64,
    pub confidence: f64,
}

impl Default for Rigid3dRansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            threshold_meters: 0.05,
            confidence: 0.99,
        }
    }
}

/// Horn's method: fit (R, t) minimizing sum ||p2 - (R p1 + t)||^2.
/// Returns None on degenerate input (< 3 points or rank-deficient covariance).
pub fn horn(corrs: &[Correspondence3d]) -> Option<([[f64; 3]; 3], [f64; 3])> {
    let n = corrs.len();
    if n < 3 {
        return None;
    }
    let inv_n = 1.0 / n as f64;

    let mut mu1 = [0.0f64; 3];
    let mut mu2 = [0.0f64; 3];
    for c in corrs {
        for i in 0..3 {
            mu1[i] += c.p1[i];
            mu2[i] += c.p2[i];
        }
    }
    for i in 0..3 {
        mu1[i] *= inv_n;
        mu2[i] *= inv_n;
    }

    // Cross-covariance H = sum (p1 - mu1) (p2 - mu2)^T.
    let mut h = SMatrix::<f64, 3, 3>::zeros();
    for c in corrs {
        let q1 = [c.p1[0] - mu1[0], c.p1[1] - mu1[1], c.p1[2] - mu1[2]];
        let q2 = [c.p2[0] - mu2[0], c.p2[1] - mu2[1], c.p2[2] - mu2[2]];
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += q1[i] * q2[j];
            }
        }
    }

    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let v = v_t.transpose();

    // R = V * diag(1, 1, det(V U^T)) * U^T to handle reflection cases.
    let det = (v * u.transpose()).determinant();
    let mut diag = SMatrix::<f64, 3, 3>::identity();
    if det < 0.0 {
        diag[(2, 2)] = -1.0;
    }
    let r_mat = v * diag * u.transpose();

    let mu1_v = nalgebra::Vector3::new(mu1[0], mu1[1], mu1[2]);
    let mu2_v = nalgebra::Vector3::new(mu2[0], mu2[1], mu2[2]);
    let t_v = mu2_v - r_mat * mu1_v;

    let mut r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = r_mat[(i, j)];
        }
    }
    Some((r, [t_v[0], t_v[1], t_v[2]]))
}

#[inline]
pub fn residual_sq(r: &[[f64; 3]; 3], t: &[f64; 3], c: &Correspondence3d) -> f64 {
    let rx = r[0][0] * c.p1[0] + r[0][1] * c.p1[1] + r[0][2] * c.p1[2] + t[0];
    let ry = r[1][0] * c.p1[0] + r[1][1] * c.p1[1] + r[1][2] * c.p1[2] + t[1];
    let rz = r[2][0] * c.p1[0] + r[2][1] * c.p1[1] + r[2][2] * c.p1[2] + t[2];
    let dx = c.p2[0] - rx;
    let dy = c.p2[1] - ry;
    let dz = c.p2[2] - rz;
    dx * dx + dy * dy + dz * dz
}

pub fn estimate_rigid_ransac(
    corrs: &[Correspondence3d],
    cfg: &Rigid3dRansacConfig,
) -> Option<Rigid3dResult> {
    let n = corrs.len();
    if n < 3 {
        return None;
    }
    let thresh_sq = cfg.threshold_meters * cfg.threshold_meters;

    let mut best_inliers = vec![false; n];
    let mut best_count = 0usize;
    let mut best_r = [[0.0f64; 3]; 3];
    let mut best_t = [0.0f64; 3];
    let mut rng = SimpleRng::new(42);
    let mut iterations = 0usize;
    let mut adaptive_max = cfg.max_iterations;

    let mut sample_corrs = [Correspondence3d {
        p1: [0.0; 3],
        p2: [0.0; 3],
    }; 3];

    for iter in 0..cfg.max_iterations {
        iterations = iter + 1;
        if iter >= adaptive_max {
            break;
        }

        let sample = random_sample_3(&mut rng, n);
        for i in 0..3 {
            sample_corrs[i] = corrs[sample[i]];
        }
        let (r, t) = match horn(&sample_corrs) {
            Some(v) => v,
            None => continue,
        };

        let mut inliers = vec![false; n];
        let mut count = 0;
        for (i, c) in corrs.iter().enumerate() {
            if residual_sq(&r, &t, c) < thresh_sq {
                inliers[i] = true;
                count += 1;
            }
        }

        if count > best_count {
            best_count = count;
            best_inliers = inliers;
            best_r = r;
            best_t = t;

            let w = count as f64 / n as f64;
            if w > 0.0 {
                let p_fail = (1.0 - w.powi(3)).max(1e-15);
                let k = (1.0 - cfg.confidence).ln() / p_fail.ln();
                adaptive_max = (k.ceil() as usize).min(cfg.max_iterations);
            }
        }
    }

    if best_count < 3 {
        return None;
    }

    let inlier_corrs: Vec<Correspondence3d> = corrs
        .iter()
        .zip(best_inliers.iter())
        .filter(|(_, &b)| b)
        .map(|(c, _)| *c)
        .collect();

    if let Some((r_refined, t_refined)) = horn(&inlier_corrs) {
        let mut final_inliers = vec![false; n];
        let mut final_count = 0;
        for (i, c) in corrs.iter().enumerate() {
            if residual_sq(&r_refined, &t_refined, c) < thresh_sq {
                final_inliers[i] = true;
                final_count += 1;
            }
        }
        Some(Rigid3dResult {
            r: r_refined,
            t: t_refined,
            inliers: final_inliers,
            num_inliers: final_count,
            total: n,
            iterations,
        })
    } else {
        Some(Rigid3dResult {
            r: best_r,
            t: best_t,
            inliers: best_inliers,
            num_inliers: best_count,
            total: n,
            iterations,
        })
    }
}

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
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn random_sample_3(rng: &mut SimpleRng, n: usize) -> [usize; 3] {
    let mut sample = [0usize; 3];
    let mut count = 0;
    while count < 3 {
        let idx = rng.next_usize(n);
        if !sample[..count].iter().any(|&s| s == idx) {
            sample[count] = idx;
            count += 1;
        }
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_motion() -> ([[f64; 3]; 3], [f64; 3]) {
        // 10 degrees about Y, 5 cm forward.
        let a = 10.0f64.to_radians();
        let r = [
            [a.cos(), 0.0, a.sin()],
            [0.0, 1.0, 0.0],
            [-a.sin(), 0.0, a.cos()],
        ];
        let t = [0.0, 0.0, 0.05];
        (r, t)
    }

    fn apply(r: &[[f64; 3]; 3], t: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    }

    fn make_corrs(n: usize, seed: u64) -> Vec<Correspondence3d> {
        let (r, t) = synthetic_motion();
        let mut rng = SimpleRng::new(seed);
        let mut corrs = Vec::with_capacity(n);
        for _ in 0..n {
            let p1 = [
                ((rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 4.0,
                ((rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 4.0,
                ((rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64) * 8.0 + 1.0,
            ];
            let p2 = apply(&r, &t, &p1);
            corrs.push(Correspondence3d { p1, p2 });
        }
        corrs
    }

    #[test]
    fn horn_recovers_known_motion() {
        let corrs = make_corrs(30, 11);
        let (r_est, t_est) = horn(&corrs).expect("horn should succeed");
        let (r_true, t_true) = synthetic_motion();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r_est[i][j] - r_true[i][j]).abs() < 1e-9,
                    "R mismatch [{i},{j}]: got {} expected {}",
                    r_est[i][j],
                    r_true[i][j]
                );
            }
            assert!(
                (t_est[i] - t_true[i]).abs() < 1e-9,
                "t mismatch [{i}]: got {} expected {}",
                t_est[i],
                t_true[i]
            );
        }
    }

    #[test]
    fn ransac_rejects_outliers() {
        let mut corrs = make_corrs(50, 7);
        // Replace 15 with random outliers.
        let mut rng = SimpleRng::new(99);
        for i in 0..15 {
            corrs[i].p2 = [
                (rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64 * 10.0,
                (rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64 * 10.0,
                (rng.next_u64() >> 33) as f64 / (1u64 << 31) as f64 * 10.0,
            ];
        }
        let cfg = Rigid3dRansacConfig {
            max_iterations: 200,
            threshold_meters: 0.01,
            confidence: 0.99,
        };
        let result = estimate_rigid_ransac(&corrs, &cfg).expect("RANSAC should succeed");
        assert!(
            result.num_inliers >= 33,
            "expected ~35 inliers, got {}",
            result.num_inliers
        );
        // Inliers must be from the un-corrupted set [15..50).
        for (i, &is_inlier) in result.inliers.iter().enumerate() {
            if is_inlier {
                assert!(i >= 15, "outlier {i} marked as inlier");
            }
        }
    }

    #[test]
    fn ransac_fails_below_minimum() {
        let corrs = make_corrs(2, 1);
        assert!(estimate_rigid_ransac(&corrs, &Rigid3dRansacConfig::default()).is_none());
    }
}
