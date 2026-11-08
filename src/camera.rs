// camera.rs -- Rudolf-V camera calibration compatibility layer.
//
// Handles:
// - Parsing camera intrinsics from EuRoC ASL sensor.yaml
// - Pixel <-> unit-bearing conversion
// - Camera-geometry projection model construction
//
// The EuRoC sensor.yaml format is simple enough that we parse it
// manually, avoiding a serde_yaml dependency.

use camera_geometry::{CameraModel, CameraProjection, Pixel, UnitBearing};
use std::{fs, path::Path};

/// Projection model represented by this calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModel {
    None,
    RadTan,
    Equidistant,
    /// OpenCV's rational model, 8 coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
    /// (ZED SDK factory calibrations).
    Rational,
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
    /// Distortion coefficients for `model`: [k1, k2, p1, p2] (radtan),
    /// [k1, k2, k3, k4] (equidistant) or [k1, k2, p1, p2, k3, k4, k5, k6]
    /// (rational). Empty if no distortion model.
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
        let intrinsics = parse_bracket_values(content, "intrinsics:")
            .ok_or_else(|| "intrinsics: line not found".to_string())?;
        if intrinsics.len() != 4 {
            return Err(format!("Expected 4 intrinsics, got {}", intrinsics.len()));
        }

        let resolution = parse_bracket_values(content, "resolution:")
            .ok_or_else(|| "resolution: line not found".to_string())?;
        if resolution.len() != 2 {
            return Err(format!(
                "Expected 2 resolution values, got {}",
                resolution.len()
            ));
        }

        let distortion = parse_bracket_values(content, "distortion_coefficients:")
            .or_else(|| parse_bracket_values(content, "distortion_coeffs:"))
            .unwrap_or_default();
        let model = parse_projection_model(content, &distortion)?;
        let distortion = if model == DistortionModel::None {
            Vec::new()
        } else {
            distortion
        };

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

    /// Convert a pixel to a unit-sphere bearing.
    ///
    /// This is the geometry-first API. It preserves fisheye rays that cannot be
    /// represented safely as pinhole-normalized `(x/z, y/z)` coordinates.
    pub fn pixel_to_bearing(&self, u: f64, v: f64) -> Option<UnitBearing> {
        self.projection().unproject(Pixel::new(u, v))
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
            DistortionModel::Rational => CameraProjection::pinhole_rational(
                intrinsics,
                distortion8(&self.distortion),
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
    pub(crate) cam0_projection: CameraProjection,
    pub(crate) cam1_projection: CameraProjection,
    pub r_10: [[f64; 3]; 3],
    pub t_10: [f64; 3],
}

impl StereoRig {
    /// Construct a stereo rig and cache concrete projection models.
    ///
    /// The public `cam0`/`cam1` calibration structs are retained for existing
    /// callers that inspect calibration values. Matching/projection code should
    /// use the cached projection fields inside this crate, so hot paths do not
    /// rebuild `CameraProjection` from intrinsics and distortion coefficients.
    pub fn new(
        cam0: CameraIntrinsics,
        cam1: CameraIntrinsics,
        r_10: [[f64; 3]; 3],
        t_10: [f64; 3],
    ) -> Self {
        let cam0_projection = cam0.projection();
        let cam1_projection = cam1.projection();
        Self {
            cam0,
            cam1,
            cam0_projection,
            cam1_projection,
            r_10,
            t_10,
        }
    }

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
        Ok(StereoRig::new(cam0, cam1, r_10, t))
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
        Ok(StereoRig::new(cam0, cam1, r_10, t))
    }

    /// Parse a stereo rig from a ZED camera's factory calibration
    /// (`/usr/local/zed/settings/SN<serial>.conf`, the ZED SDK's INI file),
    /// for raw (unrectified) images at `resolution`: "2K", "FHD", "HD" or
    /// "VGA".
    ///
    /// `model` picks the distortion: `Rational` reads `[LEFT_DISTO]` and
    /// `[RIGHT_DISTO]` (k1..k6, p1, p2: the model the SDK's own export calls
    /// rational_polynomial); `RadTan` reads the resolution's k1, k2, p1, p2,
    /// k3, as a rational model with no denominator.
    ///
    /// The extrinsic follows zed-open-capture: R = Rodrigues(RX, CV, RZ)
    /// maps left-camera points into the right camera, and the right camera
    /// sits at t = (-Baseline, TY, TZ), in millimetres in the file.
    pub fn from_zed_conf(
        path: &Path,
        resolution: &str,
        model: DistortionModel,
    ) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let size = match resolution {
            "2K" => [2208, 1242],
            "FHD" => [1920, 1080],
            "HD" => [1280, 720],
            "VGA" => [672, 376],
            _ => {
                return Err(format!(
                    "unsupported ZED resolution '{resolution}'; expected 2K, FHD, HD or VGA"
                ));
            }
        };
        let get = |section: &str, key: &str| {
            ini_value(&content, section, key)
                .ok_or_else(|| format!("[{section}] {key}: not found in {}", path.display()))
        };
        let camera = |eye: &str| -> Result<CameraIntrinsics, String> {
            let cam = format!("{eye}_CAM_{resolution}");
            let mut c = CameraIntrinsics::new(
                get(&cam, "fx")?,
                get(&cam, "fy")?,
                get(&cam, "cx")?,
                get(&cam, "cy")?,
                size[0],
                size[1],
            );
            let disto = format!("{eye}_DISTO");
            let (keys, section): (&[&str], &str) = match model {
                DistortionModel::Rational => (
                    &["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
                    disto.as_str(),
                ),
                DistortionModel::RadTan => (&["k1", "k2", "p1", "p2", "k3"], cam.as_str()),
                _ => {
                    return Err(format!(
                        "ZED calibrations are rational or radtan, not {model:?}"
                    ));
                }
            };
            c.distortion = keys
                .iter()
                .map(|k| get(section, k))
                .collect::<Result<_, _>>()?;
            c.distortion.resize(8, 0.0);
            c.model = DistortionModel::Rational;
            Ok(c)
        };
        let rotation = rodrigues([
            get("STEREO", &format!("RX_{resolution}"))?,
            get("STEREO", &format!("CV_{resolution}"))?,
            get("STEREO", &format!("RZ_{resolution}"))?,
        ]);
        let t = [
            -get("STEREO", "Baseline")? * 1e-3,
            get("STEREO", "TY")? * 1e-3,
            get("STEREO", "TZ")? * 1e-3,
        ];
        Ok(StereoRig::new(
            camera("LEFT")?,
            camera("RIGHT")?,
            rotation,
            t,
        ))
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

/// `key=value` in `[section]` of an INI file, as a number.
fn ini_value(content: &str, section: &str, key: &str) -> Option<f64> {
    let header = format!("[{section}]");
    let mut lines = content.lines().map(str::trim);
    lines.find(|l| *l == header)?;
    lines
        .take_while(|l| !l.starts_with('['))
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim() == key)
        .and_then(|(_, v)| v.trim().parse().ok())
}

/// The rotation matrix of rotation vector `r` (axis times angle, radians).
fn rodrigues(r: [f64; 3]) -> [[f64; 3]; 3] {
    let theta = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if theta < 1e-15 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let k = [r[0] / theta, r[1] / theta, r[2] / theta];
    let (s, c) = theta.sin_cos();
    let skew = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let identity = if i == j { c } else { 0.0 };
            out[i][j] = identity + (1.0 - c) * k[i] * k[j] + s * skew[i][j];
        }
    }
    out
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
    for row in out.iter_mut().take(3) {
        row[3] = -(row[0] * t[0][3] + row[1] * t[1][3] + row[2] * t[2][3]);
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
    // Line by line with their endings, "\n" or "\r\n", to find the section's
    // first byte.
    let mut start_byte = 0;
    for line in content.split_inclusive('\n') {
        start_byte += line.len();
        if line.trim_end() == marker {
            let tail = &content[start_byte..];
            let end = tail.find("\ncam").unwrap_or(tail.len());
            return Some(&tail[..end]);
        }
    }
    None
}

fn parse_projection_model(content: &str, distortion: &[f64]) -> Result<DistortionModel, String> {
    let camera_model = parse_scalar_value(content, "camera_model:").unwrap_or("pinhole");
    let distortion_model = parse_scalar_value(content, "distortion_model:").unwrap_or("");
    match (camera_model, distortion_model) {
        // Rectified calibrations often list `none` with zero coefficients.
        ("pinhole", "none") if distortion.iter().all(|&k| k == 0.0) => Ok(DistortionModel::None),
        ("pinhole", "") if distortion.is_empty() => Ok(DistortionModel::None),
        ("pinhole", "radtan" | "radial-tangential") => {
            validate_distortion_len("radtan", distortion, 4)?;
            Ok(DistortionModel::RadTan)
        }
        ("pinhole", "equidistant") => {
            validate_distortion_len("equidistant", distortion, 4)?;
            Ok(DistortionModel::Equidistant)
        }
        ("pinhole", "rational_polynomial" | "rational") => {
            validate_distortion_len("rational", distortion, 8)?;
            Ok(DistortionModel::Rational)
        }
        ("pinhole", "") if !distortion.is_empty() => {
            validate_distortion_len("implicit radtan", distortion, 4)?;
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

fn validate_distortion_len(name: &str, distortion: &[f64], expected: usize) -> Result<(), String> {
    if distortion.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} distortion expected {expected} values, got {}",
            distortion.len()
        ))
    }
}

fn distortion8(distortion: &[f64]) -> [f64; 8] {
    std::array::from_fn(|i| distortion.get(i).copied().unwrap_or(0.0))
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
    use nalgebra::Vector3;

    #[test]
    fn test_pixel_to_bearing_no_distortion() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let b = cam.pixel_to_bearing(320.0, 240.0).unwrap().vector();
        assert!((b.x).abs() < 1e-12);
        assert!((b.y).abs() < 1e-12);
        assert!((b.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pixel_to_bearing_with_distortion() {
        let mut cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cam.distortion = vec![-0.28, 0.07, 0.0, 0.0];
        let b = cam.pixel_to_bearing(320.0, 240.0).unwrap().vector();
        assert!((b.x).abs() < 1e-12);
        assert!((b.y).abs() < 1e-12);
        assert!((b.z - 1.0).abs() < 1e-12);
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
    fn test_projection_pinhole() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let projection = cam.projection();
        let pixel = projection.project(Vector3::new(0.0, 0.0, 1.0)).unwrap();
        let (u, v) = (pixel.x(), pixel.y());
        assert!((u - 320.0).abs() < 1e-10);
        assert!((v - 240.0).abs() < 1e-10);

        let pixel = projection.project(Vector3::new(0.1, -0.2, 2.0)).unwrap();
        let (u, v) = (pixel.x(), pixel.y());
        assert!((u - (500.0 * 0.05 + 320.0)).abs() < 1e-10);
        assert!((v - (500.0 * -0.1 + 240.0)).abs() < 1e-10);
    }

    #[test]
    fn test_project_behind_camera() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let projection = cam.projection();
        assert!(projection.project(Vector3::new(0.0, 0.0, -1.0)).is_none());
        assert!(projection.project(Vector3::new(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn test_projection_jacobian_pinhole() {
        let cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let projection = cam.projection();
        let p = Vector3::new(0.3, -0.2, 2.0);
        let j = projection.project_jacobian(p).unwrap();
        let eps = 1e-6;
        for i in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[i] += eps;
            pm[i] -= eps;
            let plus = projection.project(pp).unwrap();
            let minus = projection.project(pm).unwrap();
            let (up, vp) = (plus.x(), plus.y());
            let (um, vm) = (minus.x(), minus.y());
            let du = (up - um) / (2.0 * eps);
            let dv = (vp - vm) / (2.0 * eps);
            assert!(
                (j[(0, i)] - du).abs() < 1e-3,
                "du/d{i}: analytical={} numerical={}",
                j[(0, i)],
                du
            );
            assert!(
                (j[(1, i)] - dv).abs() < 1e-3,
                "dv/d{i}: analytical={} numerical={}",
                j[(1, i)],
                dv
            );
        }
    }

    #[test]
    fn test_projection_jacobian_with_distortion() {
        let mut cam = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        cam.distortion = vec![-0.28, 0.07, 0.0001, 0.00002];
        let projection = cam.projection();
        let p = Vector3::new(0.3, -0.2, 2.0);
        let j = projection.project_jacobian(p).unwrap();
        let eps = 1e-6;
        for i in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[i] += eps;
            pm[i] -= eps;
            let plus = projection.project(pp).unwrap();
            let minus = projection.project(pm).unwrap();
            let (up, vp) = (plus.x(), plus.y());
            let (um, vm) = (minus.x(), minus.y());
            let du = (up - um) / (2.0 * eps);
            let dv = (vp - vm) / (2.0 * eps);
            assert!(
                (j[(0, i)] - du).abs() < 1e-3,
                "du/d{i}: analytical={} numerical={}",
                j[(0, i)],
                du
            );
            assert!(
                (j[(1, i)] - dv).abs() < 1e-3,
                "dv/d{i}: analytical={} numerical={}",
                j[(1, i)],
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
        for (i, row) in product.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-10, "T*T^-1 [{i}][{j}] = {v}");
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
        // The same file with Windows line endings.
        std::fs::write(&path, camchain.replace('\n', "\r\n")).unwrap();
        let crlf = StereoRig::from_kalibr_camchain(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(format!("{crlf:?}"), format!("{rig:?}"));

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

    #[test]
    fn test_kalibr_camchain_rational_zed() {
        // ZED 2i factory calibration as exported to Kalibr (raw images).
        let camchain = r#"cam0:
  camera_model: pinhole
  intrinsics: [958.0499877929688, 958.364990234375, 643.2249755859375, 362.3590087890625]
  distortion_model: rational_polynomial
  distortion_coeffs: [-0.5764279961585999, 28.025299072265625, -3.6843000998487696e-05,
    -0.00027499301359057426, -11.46560001373291, -0.4730679988861084, 27.756500244140625,
    -8.833189964294434]
  resolution: [1280, 720]
cam1:
  camera_model: pinhole
  intrinsics: [955.2849731445312, 955.5399780273438, 644.780029296875, 367.343505859375]
  distortion_model: rational_polynomial
  distortion_coeffs: [-4.018050193786621, 19.540700912475586, -2.1755000489065424e-05,
    -0.0001793120027286932, -8.16450023651123, -3.937230110168457, 19.188400268554688,
    -6.666619777679443]
  resolution: [1280, 720]
  T_cn_cnm1:
  - [0.999984351, 0.003146175, -0.004622436, -0.119868005]
  - [-0.003142531, 0.999994737, 0.000795437, 0.000731885]
  - [0.004624914, -0.000780899, 0.999989027, -0.001205486]
  - [0.0, 0.0, 0.0, 1.0]
"#;
        let path = std::env::temp_dir().join(format!(
            "rudolf_v_test_camchain_{}_{}.yaml",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, camchain).unwrap();
        let rig = StereoRig::from_kalibr_camchain(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(rig.cam0.model, DistortionModel::Rational);
        assert_eq!(rig.cam1.distortion.len(), 8);
        // cv2.projectPoints with cam0's calibration.
        let pixel = rig
            .cam0
            .projection()
            .project(nalgebra::Vector3::new(0.3, -0.2, 2.0))
            .unwrap();
        assert!((pixel.x() - 786.464309215).abs() < 1e-6);
        assert!((pixel.y() - 266.827864422).abs() < 1e-6);
        assert!((rig.baseline_meters() - 0.11988).abs() < 1e-4);
    }

    /// The factory file of the lab Orin's ZED 2i (S/N 36835247).
    const ZED_CONF: &str = "[LEFT_CAM_HD]
fx=958.245
fy=958.715
cx=651.625
cy=353.5555
k1=-0.0720847
k2=-0.00535637
p1=0.00042353
p2=2.62033e-05
k3=-0.0735599

[RIGHT_CAM_HD]
fx=956.08
fy=956.605
cx=653.05
cy=359.875
k1=-0.0701151
k2=-0.0175723
p1=0.000313971
p2=4.32652e-05
k3=-0.0535535

[LEFT_DISTO]
k1=-4.26217
k2=46.5049
k3=-23.2569
k4=-4.16774
k5=46.0752
k6=-19.7633
p1=0.00042056
p2=9.13147e-06

[RIGHT_DISTO]
k1=-5.1226
k2=47.702
k3=-22.8344
k4=-5.03039
k5=47.2158
k6=-19.2661
p1=0.00032012
p2=2.33182e-05

[STEREO]
Baseline=120.042
TY=-0.252054
TZ=0.432304
CV_HD=0.00391325
RX_HD=0.00305798
RZ_HD=0.00347634

[MISC]
Sensor_ID=0
";

    #[test]
    fn test_zed_conf_matches_the_sdk() {
        let path = std::env::temp_dir().join(format!(
            "rudolf_v_test_zed_{}_{}.conf",
            std::process::id(),
            line!()
        ));
        std::fs::write(&path, ZED_CONF).unwrap();
        let rig = StereoRig::from_zed_conf(&path, "HD", DistortionModel::Rational).unwrap();
        let radtan = StereoRig::from_zed_conf(&path, "HD", DistortionModel::RadTan).unwrap();
        let missing = StereoRig::from_zed_conf(&path, "FHD", DistortionModel::Rational);
        let _ = std::fs::remove_file(&path);

        // The ZED SDK 5.5's raw calibration of the same camera, as
        // tools/dense_stereo/zed_record.py exported it (float32 there).
        let sdk_r = [
            [0.999986291, -0.003470335, 0.003918541],
            [0.003482302, 0.999989271, -0.003051159],
            [-0.003907911, 0.003064763, 0.999987662],
        ];
        let sdk_t = [-0.120042004, -0.000252054, 0.000432304];
        for (i, (row, sdk_row)) in rig.r_10.iter().zip(sdk_r).enumerate() {
            for (j, (a, b)) in row.iter().zip(sdk_row).enumerate() {
                assert!((a - b).abs() < 1e-7, "R[{i}][{j}]");
            }
        }
        for (i, (a, b)) in rig.t_10.iter().zip(sdk_t).enumerate() {
            assert!((a - b).abs() < 1e-8, "t[{i}]");
        }
        let sdk_left = [
            -4.26216983795166,
            46.50490188598633,
            0.00042056001257151365,
            9.131469596468378e-06,
            -23.256900787353516,
            -4.1677398681640625,
            46.075199127197266,
            -19.7632999420166,
        ];
        assert_eq!(rig.cam0.model, DistortionModel::Rational);
        for (a, b) in rig.cam0.distortion.iter().zip(sdk_left) {
            assert!((a - b).abs() < 1e-5 * b.abs().max(1e-3), "{a} vs {b}");
        }
        assert_eq!(rig.cam1.resolution, [1280, 720]);
        assert!((rig.cam1.fx - 956.0800170898438).abs() < 1e-4);
        assert!((rig.baseline_meters() - 0.120043054).abs() < 1e-7);

        assert_eq!(
            radtan.cam0.distortion,
            [
                -0.0720847,
                -0.00535637,
                0.00042353,
                2.62033e-05,
                -0.0735599,
                0.0,
                0.0,
                0.0
            ]
        );
        let e = missing.unwrap_err();
        assert!(e.contains("_FHD") && e.contains("not found"), "{e}");
    }
}
