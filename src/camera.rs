// camera.rs -- Pinhole camera model with radial-tangential distortion.
//
// Handles:
// - Parsing camera intrinsics from EuRoC ASL sensor.yaml
// - Pixel <-> normalized (bearing) coordinate conversion
// - K matrix construction for geometric verification
//
// The EuRoC sensor.yaml format is simple enough that we parse it
// manually, avoiding a serde_yaml dependency.

use std::fs;
use std::path::Path;

/// Pinhole camera intrinsics with optional radial-tangential distortion.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Focal length in pixels (x-axis).
    pub fx: f64,
    /// Focal length in pixels (y-axis).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Image resolution [width, height].
    pub resolution: [usize; 2],
    /// Radial-tangential distortion coefficients [k1, k2, p1, p2].
    /// Empty if no distortion model.
    pub distortion: Vec<f64>,
}

impl CameraIntrinsics {
    /// Construct from explicit parameters (no distortion).
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: usize, height: usize) -> Self {
        CameraIntrinsics {
            fx,
            fy,
            cx,
            cy,
            resolution: [width, height],
            distortion: Vec::new(),
        }
    }

    /// Parse from an EuRoC ASL sensor.yaml file.
    ///
    /// Expected format (relevant lines):
    /// ```text
    /// resolution: [752, 480]
    /// intrinsics: [458.654, 457.296, 367.215, 248.375]
    /// distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
    /// ```
    pub fn from_euroc_yaml(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let intrinsics = parse_bracket_values(&content, "intrinsics:")
            .ok_or_else(|| "intrinsics: line not found".to_string())?;
        if intrinsics.len() != 4 {
            return Err(format!("Expected 4 intrinsics, got {}", intrinsics.len()));
        }

        let resolution = parse_bracket_values(&content, "resolution:")
            .ok_or_else(|| "resolution: line not found".to_string())?;
        if resolution.len() != 2 {
            return Err(format!(
                "Expected 2 resolution values, got {}",
                resolution.len()
            ));
        }

        let distortion =
            parse_bracket_values(&content, "distortion_coefficients:").unwrap_or_default();

        Ok(CameraIntrinsics {
            fx: intrinsics[0],
            fy: intrinsics[1],
            cx: intrinsics[2],
            cy: intrinsics[3],
            resolution: [resolution[0] as usize, resolution[1] as usize],
            distortion,
        })
    }

    /// Convert pixel coordinates (u, v) to normalized (bearing) coordinates.
    ///
    /// x_n = (u - cx) / fx
    /// y_n = (v - cy) / fy
    ///
    /// This applies K^{-1} to the homogeneous pixel coordinate [u, v, 1]^T.
    /// Does NOT undistort -- call undistort_point first if needed.
    pub fn normalize(&self, u: f64, v: f64) -> (f64, f64) {
        ((u - self.cx) / self.fx, (v - self.cy) / self.fy)
    }

    /// Convert normalized coordinates back to pixel coordinates.
    pub fn denormalize(&self, x_n: f64, y_n: f64) -> (f64, f64) {
        (x_n * self.fx + self.cx, y_n * self.fy + self.cy)
    }

    /// Undistort a pixel point using the radial-tangential model.
    ///
    /// Uses iterative refinement (fixed-point iteration on the distortion
    /// equations). Converges in 5-10 iterations for typical lens distortion.
    ///
    /// Returns the undistorted pixel coordinates.
    pub fn undistort_point(&self, u: f64, v: f64) -> (f64, f64) {
        if self.distortion.is_empty() {
            return (u, v);
        }

        let k1 = self.distortion.get(0).copied().unwrap_or(0.0);
        let k2 = self.distortion.get(1).copied().unwrap_or(0.0);
        let p1 = self.distortion.get(2).copied().unwrap_or(0.0);
        let p2 = self.distortion.get(3).copied().unwrap_or(0.0);

        // Work in normalized coordinates.
        let x0 = (u - self.cx) / self.fx;
        let y0 = (v - self.cy) / self.fy;

        // Iterative undistortion: solve the forward distortion model
        // x_d = x(1 + k1*r^2 + k2*r^4) + 2*p1*x*y + p2*(r^2 + 2*x^2)
        // y_d = y(1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y^2) + 2*p2*x*y
        // for (x, y) given (x_d, y_d) = (x0, y0).
        let mut x = x0;
        let mut y = y0;

        for _ in 0..20 {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let radial = 1.0 + k1 * r2 + k2 * r4;
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

            x = (x0 - dx) / radial;
            y = (y0 - dy) / radial;
        }

        self.denormalize(x, y)
    }

    /// Normalize a pixel point with undistortion: pixel -> undistorted normalized.
    pub fn normalize_undistorted(&self, u: f64, v: f64) -> (f64, f64) {
        let (u_ud, v_ud) = self.undistort_point(u, v);
        self.normalize(u_ud, v_ud)
    }

    /// Apply forward radial-tangential distortion to normalized coordinates.
    ///
    /// Given undistorted normalized (x, y), returns distorted normalized (x_d, y_d).
    pub fn distort_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        if self.distortion.is_empty() {
            return (x, y);
        }
        let k1 = self.distortion.get(0).copied().unwrap_or(0.0);
        let k2 = self.distortion.get(1).copied().unwrap_or(0.0);
        let p1 = self.distortion.get(2).copied().unwrap_or(0.0);
        let p2 = self.distortion.get(3).copied().unwrap_or(0.0);

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4;
        let x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        (x_d, y_d)
    }

    /// Project a 3D point (in camera frame) to distorted pixel coordinates.
    ///
    /// Returns `None` if the point is behind or too close to the camera (Z ≤ 1e-6).
    pub fn project_point(&self, p: [f64; 3]) -> Option<(f64, f64)> {
        if p[2] <= 1e-6 {
            return None;
        }
        let z_inv = 1.0 / p[2];
        let x_n = p[0] * z_inv;
        let y_n = p[1] * z_inv;
        let (x_d, y_d) = self.distort_normalized(x_n, y_n);
        Some(self.denormalize(x_d, y_d))
    }

    /// 2×3 Jacobian d(u,v)/d(X,Y,Z) of the full projection including distortion.
    ///
    /// Returns `[[du/dX, du/dY, du/dZ], [dv/dX, dv/dY, dv/dZ]]`.
    pub fn projection_jacobian(&self, p: [f64; 3]) -> [[f64; 3]; 2] {
        let z_inv = 1.0 / p[2];
        let x = p[0] * z_inv;
        let y = p[1] * z_inv;

        // d(x,y)/d(X,Y,Z) — pinhole normalization Jacobian
        let dx_dp = [z_inv, 0.0, -x * z_inv];
        let dy_dp = [0.0, z_inv, -y * z_inv];

        if self.distortion.is_empty() {
            return [
                [self.fx * dx_dp[0], self.fx * dx_dp[1], self.fx * dx_dp[2]],
                [self.fy * dy_dp[0], self.fy * dy_dp[1], self.fy * dy_dp[2]],
            ];
        }

        let k1 = self.distortion.get(0).copied().unwrap_or(0.0);
        let k2 = self.distortion.get(1).copied().unwrap_or(0.0);
        let p1 = self.distortion.get(2).copied().unwrap_or(0.0);
        let p2 = self.distortion.get(3).copied().unwrap_or(0.0);

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4;
        let d_radial_dr2 = k1 + 2.0 * k2 * r2;

        // d(x_d)/d(x) and d(x_d)/d(y)
        let dxd_dx = radial + 2.0 * x * x * d_radial_dr2 + 2.0 * p1 * y + 6.0 * p2 * x;
        let dxd_dy = 2.0 * x * y * d_radial_dr2 + 2.0 * p1 * x + 2.0 * p2 * y;
        let dyd_dx = 2.0 * x * y * d_radial_dr2 + 2.0 * p1 * x + 2.0 * p2 * y;
        let dyd_dy = radial + 2.0 * y * y * d_radial_dr2 + 6.0 * p1 * y + 2.0 * p2 * x;

        // Chain: d(u,v)/d(X,Y,Z) = diag(fx,fy) * d(xd,yd)/d(x,y) * d(x,y)/d(X,Y,Z)
        let mut row_u = [0.0; 3];
        let mut row_v = [0.0; 3];
        for i in 0..3 {
            let dxi = dx_dp[i];
            let dyi = dy_dp[i];
            row_u[i] = self.fx * (dxd_dx * dxi + dxd_dy * dyi);
            row_v[i] = self.fy * (dyd_dx * dxi + dyd_dy * dyi);
        }
        [row_u, row_v]
    }
}

// ---------------------------------------------------------------------------
// Stereo rig (two cameras with known extrinsics)
// ---------------------------------------------------------------------------

/// Stereo camera rig with calibrated extrinsics.
///
/// `r_10` and `t_10` describe the rigid transform from cam0 to cam1:
/// `p_cam1 = R * p_cam0 + t`.
#[derive(Debug, Clone)]
pub struct StereoRig {
    pub cam0: CameraIntrinsics,
    pub cam1: CameraIntrinsics,
    pub r_10: [[f64; 3]; 3],
    pub t_10: [f64; 3],
}

impl StereoRig {
    /// Parse a stereo rig from two EuRoC sensor.yaml files.
    ///
    /// Each file contains `T_BS` (sensor → body). The cam0→cam1 transform is:
    /// `T_10 = inv(T_BS1) * T_BS0`.
    pub fn from_euroc(cam0_yaml: &Path, cam1_yaml: &Path) -> Result<Self, String> {
        let cam0 = CameraIntrinsics::from_euroc_yaml(cam0_yaml)?;
        let cam1 = CameraIntrinsics::from_euroc_yaml(cam1_yaml)?;
        let t_bs0 = parse_t_bs(cam0_yaml)?;
        let t_bs1 = parse_t_bs(cam1_yaml)?;
        let t_bs1_inv = invert_se3(&t_bs1);
        let t_10 = multiply_se3(&t_bs1_inv, &t_bs0);
        let r_10 = [
            [t_10[0][0], t_10[0][1], t_10[0][2]],
            [t_10[1][0], t_10[1][1], t_10[1][2]],
            [t_10[2][0], t_10[2][1], t_10[2][2]],
        ];
        let t = [t_10[0][3], t_10[1][3], t_10[2][3]];
        Ok(StereoRig {
            cam0,
            cam1,
            r_10,
            t_10: t,
        })
    }

    /// Transform a 3D point from cam0 frame to cam1 frame.
    #[inline]
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.r_10;
        let t = &self.t_10;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    }

    /// Multiply rotation matrix R by a 3-vector.
    #[inline]
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.r_10;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }

    /// Stereo baseline length in meters.
    pub fn baseline_meters(&self) -> f64 {
        let t = &self.t_10;
        (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()
    }
}

/// Parse the T_BS 4×4 matrix from a EuRoC sensor.yaml file.
fn parse_t_bs(path: &Path) -> Result<[[f64; 4]; 4], String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let vals = parse_bracket_values(&content, "data:")
        .ok_or_else(|| format!("T_BS data: not found in {}", path.display()))?;
    if vals.len() != 16 {
        return Err(format!("Expected 16 T_BS values, got {}", vals.len()));
    }
    Ok([
        [vals[0], vals[1], vals[2], vals[3]],
        [vals[4], vals[5], vals[6], vals[7]],
        [vals[8], vals[9], vals[10], vals[11]],
        [vals[12], vals[13], vals[14], vals[15]],
    ])
}

fn invert_se3(t: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    // inv([R t; 0 1]) = [R^T  -R^T*t; 0 1]
    let mut out = [[0.0; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = t[j][i]; // R^T
        }
    }
    for i in 0..3 {
        out[i][3] = -(out[i][0] * t[0][3] + out[i][1] * t[1][3] + out[i][2] * t[2][3]);
    }
    out[3][3] = 1.0;
    out
}

fn multiply_se3(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Parse a line of the form "key: [v1, v2, ...]" and return the values.
fn parse_bracket_values(content: &str, key: &str) -> Option<Vec<f64>> {
    let key_pos = content.find(key)?;
    let after_key = &content[key_pos + key.len()..];
    let open = after_key.find('[')?;
    let close = after_key.find(']')?;
    let inner = &after_key[open + 1..close];
    let vals: Vec<f64> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    Some(vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_denormalize() {
        let cam = CameraIntrinsics::new(458.654, 457.296, 367.215, 248.375, 752, 480);
        let (xn, yn) = cam.normalize(367.215, 248.375);
        assert!(
            (xn).abs() < 1e-10,
            "principal point should normalize to (0, 0)"
        );
        assert!((yn).abs() < 1e-10);

        let (u, v) = cam.denormalize(xn, yn);
        assert!((u - 367.215).abs() < 1e-10);
        assert!((v - 248.375).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_corner() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let (xn, yn) = cam.normalize(0.0, 0.0);
        assert!((xn - (-0.64)).abs() < 1e-10);
        assert!((yn - (-0.48)).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        let cam = CameraIntrinsics::new(458.654, 457.296, 367.215, 248.375, 752, 480);
        let u = 123.456;
        let v = 321.654;
        let (xn, yn) = cam.normalize(u, v);
        let (u2, v2) = cam.denormalize(xn, yn);
        assert!((u - u2).abs() < 1e-10);
        assert!((v - v2).abs() < 1e-10);
    }

    #[test]
    fn test_undistort_no_distortion() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let (u, v) = cam.undistort_point(100.0, 200.0);
        assert!((u - 100.0).abs() < 1e-10);
        assert!((v - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_undistort_with_distortion() {
        let mut cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cam.distortion = vec![-0.28, 0.07, 0.0, 0.0];
        // Principal point should be unchanged regardless of distortion.
        let (u, v) = cam.undistort_point(320.0, 240.0);
        assert!(
            (u - 320.0).abs() < 1e-6,
            "principal point undistorted: ({u}, {v})"
        );
        assert!((v - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_bracket_values() {
        let yaml = "intrinsics: [458.654, 457.296, 367.215, 248.375]\n\
                     resolution: [752, 480]\n";
        let vals = parse_bracket_values(yaml, "intrinsics:").unwrap();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 458.654).abs() < 1e-6);

        let res = parse_bracket_values(yaml, "resolution:").unwrap();
        assert_eq!(res.len(), 2);
        assert!((res[0] - 752.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_scientific_notation() {
        let yaml = "distortion_coefficients: [-0.283, 0.074, 0.000194, 1.76e-05]\n";
        let vals = parse_bracket_values(yaml, "distortion_coefficients:").unwrap();
        assert_eq!(vals.len(), 4);
        assert!((vals[3] - 1.76e-05).abs() < 1e-10);
    }

    #[test]
    fn test_distort_no_distortion() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let (xd, yd) = cam.distort_normalized(0.1, 0.2);
        assert!((xd - 0.1).abs() < 1e-10);
        assert!((yd - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_distort_undistort_roundtrip() {
        let mut cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cam.distortion = vec![-0.28, 0.07, 0.0001, 0.00002];
        let x = 0.15;
        let y = -0.1;
        let (xd, yd) = cam.distort_normalized(x, y);
        let (ud, vd) = cam.denormalize(xd, yd);
        let (uu, vu) = cam.undistort_point(ud, vd);
        let (xu, yu) = cam.normalize(uu, vu);
        assert!((xu - x).abs() < 1e-6, "roundtrip x: {xu} vs {x}");
        assert!((yu - y).abs() < 1e-6, "roundtrip y: {yu} vs {y}");
    }

    #[test]
    fn test_project_point_pinhole() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let (u, v) = cam.project_point([0.0, 0.0, 1.0]).unwrap();
        assert!((u - 320.0).abs() < 1e-10);
        assert!((v - 240.0).abs() < 1e-10);

        let (u, v) = cam.project_point([0.1, -0.2, 2.0]).unwrap();
        assert!((u - (500.0 * 0.05 + 320.0)).abs() < 1e-10);
        assert!((v - (500.0 * -0.1 + 240.0)).abs() < 1e-10);
    }

    #[test]
    fn test_project_behind_camera() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        assert!(cam.project_point([0.0, 0.0, -1.0]).is_none());
        assert!(cam.project_point([0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_projection_jacobian_pinhole() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let p = [0.3, -0.2, 2.0];
        let j = cam.projection_jacobian(p);
        let eps = 1e-6;
        for i in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[i] += eps;
            pm[i] -= eps;
            let (up, vp) = cam.project_point(pp).unwrap();
            let (um, vm) = cam.project_point(pm).unwrap();
            let du = (up - um) / (2.0 * eps);
            let dv = (vp - vm) / (2.0 * eps);
            assert!(
                (j[0][i] - du).abs() < 1e-3,
                "du/d{i}: analytical={} numerical={}",
                j[0][i],
                du
            );
            assert!(
                (j[1][i] - dv).abs() < 1e-3,
                "dv/d{i}: analytical={} numerical={}",
                j[1][i],
                dv
            );
        }
    }

    #[test]
    fn test_projection_jacobian_with_distortion() {
        let mut cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cam.distortion = vec![-0.28, 0.07, 0.0001, 0.00002];
        let p = [0.3, -0.2, 2.0];
        let j = cam.projection_jacobian(p);
        let eps = 1e-6;
        for i in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[i] += eps;
            pm[i] -= eps;
            let (up, vp) = cam.project_point(pp).unwrap();
            let (um, vm) = cam.project_point(pm).unwrap();
            let du = (up - um) / (2.0 * eps);
            let dv = (vp - vm) / (2.0 * eps);
            assert!(
                (j[0][i] - du).abs() < 1e-3,
                "du/d{i}: analytical={} numerical={}",
                j[0][i],
                du
            );
            assert!(
                (j[1][i] - dv).abs() < 1e-3,
                "dv/d{i}: analytical={} numerical={}",
                j[1][i],
                dv
            );
        }
    }

    #[test]
    fn test_se3_invert() {
        let t = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = super::invert_se3(&t);
        let product = super::multiply_se3(&t, &inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-10,
                    "T*T^-1 [{i}][{j}] = {}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_euroc_cam0_to_cam1_direction() {
        let t_bs0 = [
            [
                0.0148655429818,
                -0.999880929698,
                0.00414029679422,
                -0.0216401454975,
            ],
            [
                0.999557249008,
                0.0149672133247,
                0.025715529948,
                -0.064676986768,
            ],
            [
                -0.0257744366974,
                0.00375618835797,
                0.999660727178,
                0.00981073058949,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let t_bs1 = [
            [
                0.0125552670891,
                -0.999755099723,
                0.0182237714554,
                -0.0198435579556,
            ],
            [
                0.999598781151,
                0.0130119051815,
                0.0251588363115,
                0.0453689425024,
            ],
            [
                -0.0253898008918,
                0.0179005838253,
                0.999517347078,
                0.00786212447038,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let t_10 = super::multiply_se3(&super::invert_se3(&t_bs1), &t_bs0);
        assert!(
            t_10[0][3] < -0.10,
            "cam1 should be to the right of cam0, so cam0 points transform with negative x: {}",
            t_10[0][3]
        );
        assert!(
            t_10[1][3].abs() < 0.01,
            "EuRoC stereo baseline should be mostly horizontal, got y={}",
            t_10[1][3]
        );
    }
}
