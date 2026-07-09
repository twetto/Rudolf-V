// camera.rs -- Rudolf-V camera calibration compatibility layer.
//
// Handles:
// - Parsing camera intrinsics from EuRoC ASL sensor.yaml
// - Pixel <-> normalized (bearing) coordinate conversion
// - K matrix construction for geometric verification
//
// The EuRoC sensor.yaml format is simple enough that we parse it
// manually, avoiding a serde_yaml dependency.

use camera_geometry::{CameraModel, CameraProjection, Pixel, UnitBearing};
use nalgebra::Vector3;
use std::{fs, path::Path};

/// Projection model represented by this calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModel {
    None,
    RadTan,
    Equidistant,
}

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
    /// Projection/distortion interpretation for `distortion`.
    pub model: DistortionModel,
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
            model: DistortionModel::None,
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
        Self::from_yaml_content(&content)
    }

    /// Parse one camera from a Kalibr camchain file.
    pub fn from_kalibr_camchain(path: &Path, camera: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let section = yaml_section(&content, camera)
            .ok_or_else(|| format!("camera section {camera}: not found in {}", path.display()))?;
        Self::from_yaml_content(section)
    }

    fn from_yaml_content(content: &str) -> Result<Self, String> {
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

        let distortion = parse_bracket_values(&content, "distortion_coefficients:")
            .or_else(|| parse_bracket_values(&content, "distortion_coeffs:"))
            .unwrap_or_default();
        let model = parse_projection_model(&content, &distortion)?;

        Ok(CameraIntrinsics {
            fx: intrinsics[0],
            fy: intrinsics[1],
            cx: intrinsics[2],
            cy: intrinsics[3],
            resolution: [resolution[0] as usize, resolution[1] as usize],
            distortion,
            model,
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
        let (x, y) = self.pixel_to_normalized_legacy(u, v);
        self.denormalize(x, y)
    }

    /// Normalize a pixel point with undistortion: pixel -> undistorted normalized.
    pub fn normalize_undistorted(&self, u: f64, v: f64) -> (f64, f64) {
        self.pixel_to_normalized_legacy(u, v)
    }

    /// Convert a pixel to a unit-sphere bearing.
    ///
    /// This is the geometry-first API. It preserves fisheye rays that cannot be
    /// represented safely as pinhole-normalized `(x/z, y/z)` coordinates.
    pub fn pixel_to_bearing(&self, u: f64, v: f64) -> Option<UnitBearing> {
        self.projection().unproject(Pixel::new(u, v))
    }

    /// Convert a unit bearing to legacy pinhole-normalized coordinates.
    ///
    /// Transitional adapter for old call sites. New geometric code should use
    /// `UnitBearing` directly.
    pub fn bearing_to_normalized_legacy(bearing: UnitBearing) -> Option<(f64, f64)> {
        let b = bearing.vector();
        (b.z.abs() > 1.0e-12).then_some((b.x / b.z, b.y / b.z))
    }

    /// Legacy pixel -> `(x/z, y/z)` adapter.
    pub fn pixel_to_normalized_legacy(&self, u: f64, v: f64) -> (f64, f64) {
        self.pixel_to_bearing(u, v)
            .and_then(Self::bearing_to_normalized_legacy)
            .unwrap_or((f64::NAN, f64::NAN))
    }

    /// Apply forward radial-tangential distortion to normalized coordinates.
    ///
    /// Given undistorted normalized (x, y), returns distorted normalized (x_d, y_d).
    pub fn distort_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        let Some(pixel) = self.projection().project(Vector3::new(x, y, 1.0)) else {
            return (f64::NAN, f64::NAN);
        };
        self.normalize(pixel.x(), pixel.y())
    }

    /// Project a 3D point (in camera frame) to distorted pixel coordinates.
    ///
    /// Returns `None` if the point is behind or too close to the camera (Z ≤ 1e-6).
    pub fn project_point(&self, p: [f64; 3]) -> Option<(f64, f64)> {
        let pixel = self.projection().project(Vector3::new(p[0], p[1], p[2]))?;
        Some((pixel.x(), pixel.y()))
    }

    /// 2×3 Jacobian d(u,v)/d(X,Y,Z) of the full projection including distortion.
    ///
    /// Returns `[[du/dX, du/dY, du/dZ], [dv/dX, dv/dY, dv/dZ]]`.
    pub fn projection_jacobian(&self, p: [f64; 3]) -> [[f64; 3]; 2] {
        let Some(j) = self
            .projection()
            .project_jacobian(Vector3::new(p[0], p[1], p[2]))
        else {
            return [[f64::NAN; 3]; 2];
        };
        [
            [j[(0, 0)], j[(0, 1)], j[(0, 2)]],
            [j[(1, 0)], j[(1, 1)], j[(1, 2)]],
        ]
    }

    /// Concrete camera-geometry projection model for this calibration.
    pub fn projection(&self) -> CameraProjection {
        let intrinsics = [self.fx, self.fy, self.cx, self.cy];
        match self.effective_model() {
            DistortionModel::None => CameraProjection::pinhole(intrinsics, self.resolution),
            DistortionModel::RadTan => CameraProjection::pinhole_radtan(
                intrinsics,
                distortion4(&self.distortion),
                self.resolution,
            ),
            DistortionModel::Equidistant => CameraProjection::pinhole_equidistant(
                intrinsics,
                distortion4(&self.distortion),
                self.resolution,
            ),
        }
    }

    fn effective_model(&self) -> DistortionModel {
        match (self.model, self.distortion.is_empty()) {
            (DistortionModel::None, false) => DistortionModel::RadTan,
            (model, _) => model,
        }
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

    /// Parse a stereo rig from a Kalibr camchain file.
    ///
    /// Kalibr stores `T_cn_cnm1` under `cam1` as the transform from the
    /// previous camera (`cam0`) to the current camera (`cam1`), matching this
    /// type's `p_cam1 = R * p_cam0 + t` convention.
    pub fn from_kalibr_camchain(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let cam0 = CameraIntrinsics::from_kalibr_camchain(path, "cam0")?;
        let cam1 = CameraIntrinsics::from_kalibr_camchain(path, "cam1")?;
        let cam1_section = yaml_section(&content, "cam1")
            .ok_or_else(|| format!("camera section cam1: not found in {}", path.display()))?;
        let t_10 = parse_matrix4_after_key(cam1_section, "T_cn_cnm1:")
            .ok_or_else(|| format!("T_cn_cnm1: not found in {}", path.display()))?;
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

fn parse_matrix4_after_key(content: &str, key: &str) -> Option<[[f64; 4]; 4]> {
    let mut lines = content.lines().skip_while(|line| line.trim() != key);
    lines.next()?;
    let mut rows = [[0.0; 4]; 4];
    for row in &mut rows {
        let line = lines.next()?.trim();
        let open = line.find('[')?;
        let close = line.rfind(']')?;
        let vals: Vec<f64> = line[open + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        if vals.len() != 4 {
            return None;
        }
        row.copy_from_slice(&vals);
    }
    Some(rows)
}

fn yaml_section<'a>(content: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{name}:");
    let start_line = content.lines().position(|line| line.trim_end() == marker)?;
    let start_byte = content
        .lines()
        .take(start_line + 1)
        .map(|line| line.len() + 1)
        .sum::<usize>();
    let tail = &content[start_byte..];
    let end = tail.find("\ncam").unwrap_or(tail.len());
    Some(&tail[..end])
}

fn parse_projection_model(content: &str, distortion: &[f64]) -> Result<DistortionModel, String> {
    let camera_model = parse_scalar_value(content, "camera_model:").unwrap_or("pinhole");
    let distortion_model = parse_scalar_value(content, "distortion_model:").unwrap_or("");
    match (camera_model, distortion_model) {
        ("pinhole", "" | "none") if distortion.is_empty() => Ok(DistortionModel::None),
        ("pinhole", "radtan" | "radial-tangential") => {
            validate_distortion_len("radtan", distortion)?;
            Ok(DistortionModel::RadTan)
        }
        ("pinhole", "equidistant") => {
            validate_distortion_len("equidistant", distortion)?;
            Ok(DistortionModel::Equidistant)
        }
        ("pinhole", "") if !distortion.is_empty() => {
            validate_distortion_len("implicit radtan", distortion)?;
            Ok(DistortionModel::RadTan)
        }
        _ => Err(format!(
            "unsupported camera projection camera_model={camera_model}, distortion_model={distortion_model}"
        )),
    }
}

fn parse_scalar_value<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    content
        .lines()
        .find_map(|line| line.trim().strip_prefix(key).map(str::trim))
}

fn validate_distortion_len(name: &str, distortion: &[f64]) -> Result<(), String> {
    if distortion.len() == 4 {
        Ok(())
    } else {
        Err(format!(
            "{name} distortion expected 4 values, got {}",
            distortion.len()
        ))
    }
}

fn distortion4(distortion: &[f64]) -> [f64; 4] {
    [
        distortion.first().copied().unwrap_or(0.0),
        distortion.get(1).copied().unwrap_or(0.0),
        distortion.get(2).copied().unwrap_or(0.0),
        distortion.get(3).copied().unwrap_or(0.0),
    ]
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

    #[test]
    fn test_kalibr_camchain_stereo_rig() {
        let camchain = r#"
cam0:
  T_cam_imu:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [0.01, 0.02, 0.03, 0.04]
  distortion_model: equidistant
  intrinsics: [190.0, 191.0, 254.0, 255.0]
  resolution: [512, 512]
cam1:
  T_cam_imu:
  - [1.0, 0.0, 0.0, -0.1]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, -0.101]
  - [0.0, 1.0, 0.0, -0.002]
  - [0.0, 0.0, 1.0, -0.001]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [0.05, 0.06, 0.07, 0.08]
  distortion_model: equidistant
  intrinsics: [192.0, 193.0, 252.0, 253.0]
  resolution: [512, 512]
"#;
        let path = std::env::temp_dir().join(format!(
            "rudolf_v_test_camchain_{}_{}.yaml",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, camchain).unwrap();

        let rig = StereoRig::from_kalibr_camchain(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(rig.cam0.model, DistortionModel::Equidistant);
        assert_eq!(rig.cam1.model, DistortionModel::Equidistant);
        assert!((rig.cam0.fx - 190.0).abs() < 1e-12);
        assert!((rig.cam1.cx - 252.0).abs() < 1e-12);
        assert!((rig.t_10[0] + 0.101).abs() < 1e-12);
        assert!((rig.t_10[1] + 0.002).abs() < 1e-12);
        assert!((rig.t_10[2] + 0.001).abs() < 1e-12);
        assert!(
            (rig.baseline_meters() - (0.101f64 * 0.101 + 0.002 * 0.002 + 0.001 * 0.001).sqrt())
                .abs()
                < 1e-12
        );
    }
}
