//! Calibrated central-camera projection and unit-bearing geometry.

use nalgebra::{Matrix2, Matrix2x3, Matrix3x2, Vector2, Vector3};
use std::{fs, path::Path};

const MIN_Z: f64 = 1.0e-9;
const MIN_NORM: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel(pub Vector2<f64>);

impl Pixel {
    pub fn new(x: f64, y: f64) -> Self {
        Self(Vector2::new(x, y))
    }
    pub fn x(self) -> f64 {
        self.0.x
    }
    pub fn y(self) -> f64 {
        self.0.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitBearing(Vector3<f64>);

impl UnitBearing {
    pub fn try_new(vector: Vector3<f64>) -> Option<Self> {
        let norm = vector.norm();
        (norm.is_finite() && norm > MIN_NORM).then(|| Self(vector / norm))
    }
    pub fn vector(self) -> Vector3<f64> {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingTangent(pub Vector2<f64>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPinhole(pub Vector2<f64>);

pub trait CameraModel: Send + Sync {
    fn unproject(&self, pixel: Pixel) -> Option<UnitBearing>;
    fn project(&self, point: Vector3<f64>) -> Option<Pixel>;
    fn project_jacobian(&self, point: Vector3<f64>) -> Option<Matrix2x3<f64>>;
    fn image_size(&self) -> [usize; 2];

    fn unproject_tangent_jacobian(&self, pixel: Pixel) -> Option<Matrix2<f64>> {
        let bearing = self.unproject(pixel)?;
        let basis = tangent_basis(bearing);
        let projection = self.project_jacobian(bearing.vector())?;
        (projection * basis).try_inverse()
    }
}

fn tangent_basis(bearing: UnitBearing) -> Matrix3x2<f64> {
    let b = bearing.vector();
    let seed = if b.z.abs() < 0.9 {
        Vector3::z_axis().into_inner()
    } else {
        Vector3::x_axis().into_inner()
    };
    let e1 = seed.cross(&b).normalize();
    let e2 = b.cross(&e1);
    Matrix3x2::from_columns(&[e1, e2])
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pinhole {
    pub intrinsics: [f64; 4],
    pub resolution: [usize; 2],
}

impl Pinhole {
    pub fn new(intrinsics: [f64; 4], resolution: [usize; 2]) -> Self {
        Self {
            intrinsics,
            resolution,
        }
    }
}

impl CameraModel for Pinhole {
    fn unproject(&self, pixel: Pixel) -> Option<UnitBearing> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        UnitBearing::try_new(Vector3::new(
            (pixel.x() - cx) / fx,
            (pixel.y() - cy) / fy,
            1.0,
        ))
    }

    fn project(&self, point: Vector3<f64>) -> Option<Pixel> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_point(&point)?;
        Some(Pixel::new(
            fx * point.x / point.z + cx,
            fy * point.y / point.z + cy,
        ))
    }

    fn project_jacobian(&self, point: Vector3<f64>) -> Option<Matrix2x3<f64>> {
        let [fx, fy, _, _] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_point(&point)?;
        let iz = 1.0 / point.z;
        Some(Matrix2x3::new(
            fx * iz,
            0.0,
            -fx * point.x * iz * iz,
            0.0,
            fy * iz,
            -fy * point.y * iz * iz,
        ))
    }

    fn image_size(&self) -> [usize; 2] {
        self.resolution
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PinholeRadTan {
    pub intrinsics: [f64; 4],
    pub distortion: [f64; 4],
    pub resolution: [usize; 2],
}

impl PinholeRadTan {
    pub fn new(intrinsics: [f64; 4], distortion: [f64; 4], resolution: [usize; 2]) -> Self {
        Self {
            intrinsics,
            distortion,
            resolution,
        }
    }

    fn distort_with_jacobian(&self, p: Vector2<f64>) -> (Vector2<f64>, Matrix2<f64>) {
        let [k1, k2, p1, p2] = self.distortion;
        let x = p.x;
        let y = p.y;
        let r2 = x * x + y * y;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        let dr = k1 + 2.0 * k2 * r2;
        let distorted = Vector2::new(
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        );
        let jacobian = Matrix2::new(
            radial + 2.0 * x * x * dr + 2.0 * p1 * y + 6.0 * p2 * x,
            2.0 * x * y * dr + 2.0 * p1 * x + 2.0 * p2 * y,
            2.0 * x * y * dr + 2.0 * p1 * x + 2.0 * p2 * y,
            radial + 2.0 * y * y * dr + 6.0 * p1 * y + 2.0 * p2 * x,
        );
        (distorted, jacobian)
    }
}

impl CameraModel for PinholeRadTan {
    fn unproject(&self, pixel: Pixel) -> Option<UnitBearing> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        let target = Vector2::new((pixel.x() - cx) / fx, (pixel.y() - cy) / fy);
        let mut undistorted = target;
        for _ in 0..12 {
            let (estimate, jacobian) = self.distort_with_jacobian(undistorted);
            let step = jacobian.lu().solve(&(estimate - target))?;
            undistorted -= step;
            if step.norm_squared() < 1.0e-24 {
                break;
            }
        }
        UnitBearing::try_new(Vector3::new(undistorted.x, undistorted.y, 1.0))
    }

    fn project(&self, point: Vector3<f64>) -> Option<Pixel> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_point(&point)?;
        let (p, _) = self.distort_with_jacobian(Vector2::new(point.x / point.z, point.y / point.z));
        Some(Pixel::new(fx * p.x + cx, fy * p.y + cy))
    }

    fn project_jacobian(&self, point: Vector3<f64>) -> Option<Matrix2x3<f64>> {
        let [fx, fy, _, _] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_point(&point)?;
        let iz = 1.0 / point.z;
        let normalized = Vector2::new(point.x * iz, point.y * iz);
        let (_, distortion) = self.distort_with_jacobian(normalized);
        let pinhole = Matrix2x3::new(iz, 0.0, -point.x * iz * iz, 0.0, iz, -point.y * iz * iz);
        Some(Matrix2::new(fx, 0.0, 0.0, fy) * distortion * pinhole)
    }

    fn image_size(&self) -> [usize; 2] {
        self.resolution
    }
}

/// Kalibr/OpenCV equidistant fisheye model used by TUM-VI.
#[derive(Debug, Clone, PartialEq)]
pub struct EquidistantFisheye {
    pub intrinsics: [f64; 4],
    pub distortion: [f64; 4],
    pub resolution: [usize; 2],
}

impl EquidistantFisheye {
    pub fn new(intrinsics: [f64; 4], distortion: [f64; 4], resolution: [usize; 2]) -> Self {
        Self {
            intrinsics,
            distortion,
            resolution,
        }
    }

    /// Parse one camera entry from a Kalibr camchain.
    pub fn from_kalibr_camchain(path: &Path, camera: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let section = yaml_section(&text, camera)
            .ok_or_else(|| format!("camera section {camera}: not found"))?;
        let camera_model = scalar_value(section, "camera_model:")
            .ok_or_else(|| "camera_model: not found".to_string())?;
        let distortion_model = scalar_value(section, "distortion_model:")
            .ok_or_else(|| "distortion_model: not found".to_string())?;
        if camera_model != "pinhole" || distortion_model != "equidistant" {
            return Err(format!(
                "unsupported Kalibr model camera_model={camera_model}, distortion_model={distortion_model}"
            ));
        }
        let intrinsics = array4(section, "intrinsics:")?;
        let distortion = array4(section, "distortion_coeffs:")?;
        let size = array2(section, "resolution:")?;
        Ok(Self::new(
            intrinsics,
            distortion,
            [size[0] as usize, size[1] as usize],
        ))
    }

    fn theta_distortion(&self, theta: f64) -> (f64, f64) {
        let [k1, k2, k3, k4] = self.distortion;
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        (
            theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8),
            1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8,
        )
    }
}

impl CameraModel for EquidistantFisheye {
    fn unproject(&self, pixel: Pixel) -> Option<UnitBearing> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        let distorted = Vector2::new((pixel.x() - cx) / fx, (pixel.y() - cy) / fy);
        let theta_d = distorted.norm();
        if theta_d < MIN_NORM {
            return UnitBearing::try_new(Vector3::z());
        }
        let mut theta = theta_d;
        for _ in 0..12 {
            let (value, derivative) = self.theta_distortion(theta);
            if derivative.abs() < MIN_NORM {
                return None;
            }
            let step = (value - theta_d) / derivative;
            theta -= step;
            if step.abs() < 1.0e-12 {
                break;
            }
        }
        if !theta.is_finite() || !(0.0..std::f64::consts::PI).contains(&theta) {
            return None;
        }
        let radial_direction = distorted / theta_d;
        let sin_theta = theta.sin();
        UnitBearing::try_new(Vector3::new(
            radial_direction.x * sin_theta,
            radial_direction.y * sin_theta,
            theta.cos(),
        ))
    }

    fn project(&self, point: Vector3<f64>) -> Option<Pixel> {
        let [fx, fy, cx, cy] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_fisheye_point(&point)?;
        let radius = point.xy().norm();
        if radius < MIN_NORM {
            return Some(Pixel::new(cx, cy));
        }
        let theta = radius.atan2(point.z);
        let (theta_d, _) = self.theta_distortion(theta);
        let distorted = point.xy() * (theta_d / radius);
        Some(Pixel::new(fx * distorted.x + cx, fy * distorted.y + cy))
    }

    fn project_jacobian(&self, point: Vector3<f64>) -> Option<Matrix2x3<f64>> {
        let [fx, fy, _, _] = self.intrinsics;
        valid_intrinsics(fx, fy)?;
        valid_fisheye_point(&point)?;
        let x = point.x;
        let y = point.y;
        let z = point.z;
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        if r < 1.0e-8 {
            return Some(Matrix2x3::new(fx / z, 0.0, 0.0, 0.0, fy / z, 0.0));
        }
        let theta = r.atan2(z);
        let (theta_d, dtheta_d) = self.theta_distortion(theta);
        let denom = r2 + z * z;
        let dtheta_dr = z / denom;
        let dtheta_dz = -r / denom;
        let scale = theta_d / r;
        let dscale_dr = (dtheta_d * dtheta_dr * r - theta_d) / r2;
        let dscale_dx = dscale_dr * x / r;
        let dscale_dy = dscale_dr * y / r;
        let dscale_dz = dtheta_d * dtheta_dz / r;
        Some(Matrix2x3::new(
            fx * (scale + x * dscale_dx),
            fx * x * dscale_dy,
            fx * x * dscale_dz,
            fy * y * dscale_dx,
            fy * (scale + y * dscale_dy),
            fy * y * dscale_dz,
        ))
    }

    fn image_size(&self) -> [usize; 2] {
        self.resolution
    }
}

fn valid_intrinsics(fx: f64, fy: f64) -> Option<()> {
    (fx.is_finite() && fy.is_finite() && fx > 0.0 && fy > 0.0).then_some(())
}

fn valid_point(point: &Vector3<f64>) -> Option<()> {
    (point.iter().all(|v| v.is_finite()) && point.z > MIN_Z).then_some(())
}

fn valid_fisheye_point(point: &Vector3<f64>) -> Option<()> {
    (point.iter().all(|v| v.is_finite()) && point.norm_squared() > MIN_NORM * MIN_NORM)
        .then_some(())
}

fn yaml_section<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{name}:");
    let start = text.lines().position(|line| line.trim_end() == marker)?;
    let start_byte = text
        .lines()
        .take(start + 1)
        .map(|line| line.len() + 1)
        .sum::<usize>();
    let tail = &text[start_byte..];
    let end = tail.find("\ncam").unwrap_or(tail.len());
    Some(&tail[..end])
}

fn scalar_value<'a>(section: &'a str, key: &str) -> Option<&'a str> {
    section
        .lines()
        .find_map(|line| line.trim().strip_prefix(key).map(str::trim))
}

fn bracket_values(section: &str, key: &str) -> Result<Vec<f64>, String> {
    let start = section
        .find(key)
        .ok_or_else(|| format!("{key} not found"))?;
    let after_key = &section[start + key.len()..];
    let after_open = after_key
        .split_once('[')
        .map(|(_, rest)| rest)
        .ok_or_else(|| format!("invalid {key} array"))?;
    let inner = after_open
        .split_once(']')
        .map(|(inner, _)| inner)
        .ok_or_else(|| format!("invalid {key} array"))?;
    inner
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("invalid {key} value: {e}"))
        })
        .collect()
}

fn array4(section: &str, key: &str) -> Result<[f64; 4], String> {
    bracket_values(section, key)?
        .try_into()
        .map_err(|v: Vec<f64>| format!("{key} expected 4 values, got {}", v.len()))
}

fn array2(section: &str, key: &str) -> Result<[f64; 2], String> {
    bracket_values(section, key)?
        .try_into()
        .map_err(|v: Vec<f64>| format!("{key} expected 2 values, got {}", v.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_roundtrip(camera: &dyn CameraModel, pixels: &[Pixel], tolerance: f64) {
        for &pixel in pixels {
            let bearing = camera.unproject(pixel).unwrap();
            let projected = camera.project(bearing.vector()).unwrap();
            assert_relative_eq!(projected.0, pixel.0, epsilon = tolerance);
        }
    }

    fn assert_projection_jacobian(camera: &dyn CameraModel, point: Vector3<f64>) {
        let analytic = camera.project_jacobian(point).unwrap();
        let eps = 1.0e-7;
        for column in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[column] += eps;
            minus[column] -= eps;
            let numeric =
                (camera.project(plus).unwrap().0 - camera.project(minus).unwrap().0) / (2.0 * eps);
            assert_relative_eq!(analytic[(0, column)], numeric[0], epsilon = 2.0e-5);
            assert_relative_eq!(analytic[(1, column)], numeric[1], epsilon = 2.0e-5);
        }
    }

    fn assert_tangent_jacobian(camera: &dyn CameraModel, pixel: Pixel) {
        let analytic = camera.unproject_tangent_jacobian(pixel).unwrap();
        let bearing = camera.unproject(pixel).unwrap();
        let basis = tangent_basis(bearing);
        let eps = 1.0e-5;
        for column in 0..2 {
            let mut plus = pixel;
            let mut minus = pixel;
            plus.0[column] += eps;
            minus.0[column] -= eps;
            let bp = camera.unproject(plus).unwrap().vector();
            let bm = camera.unproject(minus).unwrap().vector();
            let numeric = basis.transpose() * ((bp - bm) / (2.0 * eps));
            assert_relative_eq!(analytic[(0, column)], numeric[0], epsilon = 2.0e-6);
            assert_relative_eq!(analytic[(1, column)], numeric[1], epsilon = 2.0e-6);
        }
    }

    #[test]
    fn pinhole_contract() {
        let camera = Pinhole::new([500.0, 510.0, 320.0, 240.0], [640, 480]);
        assert_roundtrip(
            &camera,
            &[Pixel::new(320.0, 240.0), Pixel::new(1.0, 1.0)],
            1.0e-10,
        );
        assert_projection_jacobian(&camera, Vector3::new(0.3, -0.2, 2.0));
        assert_tangent_jacobian(&camera, Pixel::new(12.0, 450.0));
        assert!(camera.project(Vector3::new(0.0, 0.0, -1.0)).is_none());
    }

    #[test]
    fn radtan_contract() {
        let camera = PinholeRadTan::new(
            [458.654, 457.296, 367.215, 248.375],
            [-0.28340811, 0.07395907, 0.00019359, 0.0000176187],
            [752, 480],
        );
        assert_roundtrip(
            &camera,
            &[Pixel::new(367.215, 248.375), Pixel::new(5.0, 5.0)],
            1.0e-8,
        );
        assert_projection_jacobian(&camera, Vector3::new(0.3, -0.2, 2.0));
        assert_tangent_jacobian(&camera, Pixel::new(20.0, 450.0));
    }

    #[test]
    fn tum_vi_equidistant_contract() {
        let camera = EquidistantFisheye::new(
            [
                190.97847715128717,
                190.9733070521226,
                254.93170605935475,
                256.8974428996504,
            ],
            [
                0.0034823894022493434,
                0.0007150348452162257,
                -0.0020532361418706202,
                0.00020293673591811182,
            ],
            [512, 512],
        );
        assert_roundtrip(
            &camera,
            &[
                Pixel::new(254.931706, 256.897443),
                Pixel::new(1.0, 1.0),
                Pixel::new(510.0, 510.0),
            ],
            1.0e-8,
        );
        assert_projection_jacobian(&camera, Vector3::new(0.8, -0.6, 1.0));
        assert_tangent_jacobian(&camera, Pixel::new(10.0, 500.0));
    }

    #[test]
    fn parser_rejects_model_mismatch() {
        let section = "  camera_model: pinhole\n  distortion_model: radtan\n";
        assert_eq!(scalar_value(section, "distortion_model:"), Some("radtan"));
    }

    #[test]
    fn parser_accepts_multiline_bracket_values() {
        let section = "  distortion_coeffs: [0.0034823894022493434, 0.0007150348452162257,\n    -0.0020532361418706202, 0.00020293673591811182]\n";
        assert_eq!(
            array4(section, "distortion_coeffs:").unwrap(),
            [
                0.0034823894022493434,
                0.0007150348452162257,
                -0.0020532361418706202,
                0.00020293673591811182
            ]
        );
    }
}
