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
            fx, fy, cx, cy,
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
            return Err(format!("Expected 2 resolution values, got {}", resolution.len()));
        }

        let distortion = parse_bracket_values(&content, "distortion_coefficients:")
            .unwrap_or_default();

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
}

/// Parse a line of the form "key: [v1, v2, ...]" and return the values.
fn parse_bracket_values(content: &str, key: &str) -> Option<Vec<f64>> {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(key) {
            // Find the bracket contents.
            let rest = &trimmed[key.len()..];
            let open = rest.find('[')?;
            let close = rest.find(']')?;
            let inner = &rest[open + 1..close];
            let vals: Vec<f64> = inner
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();
            return Some(vals);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_denormalize() {
        let cam = CameraIntrinsics::new(458.654, 457.296, 367.215, 248.375, 752, 480);
        let (xn, yn) = cam.normalize(367.215, 248.375);
        assert!((xn).abs() < 1e-10, "principal point should normalize to (0, 0)");
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
        assert!((u - 320.0).abs() < 1e-6, "principal point undistorted: ({u}, {v})");
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
}
