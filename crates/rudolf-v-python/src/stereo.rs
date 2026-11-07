use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::path::PathBuf;

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::config::StereoSection;
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::stereo::{StereoConfig as RudolfStereoConfig, StereoMatcher};

use crate::frontend::PyFrontend;
use crate::{config_error, read_section};

/// Stereo matcher wrapping `rudolf_v::stereo::StereoMatcher`.
///
/// The matcher owns cam0/cam1 intrinsics + rig and a cam1 image pyramid. It
/// pairs cam0 features (from the Python `Frontend`) with cam1 patches and
/// returns per-feature inverse-depth + residual. Use `range_priors()` to
/// convert matches into `(range, range_var)` priors, the form ECHO-LI's
/// `VIOFilter.process_vision_with_depth_priors` takes.
#[pyclass(name = "Stereo")]
pub struct PyStereo {
    inner: StereoMatcher,
    img_w: usize,
    img_h: usize,
}

#[pymethods]
impl PyStereo {
    /// Build a stereo matcher from a pair of EuRoC `sensor.yaml` files.
    ///
    /// When `vio_config` is given, the `Stereo:` section of that YAML
    /// overrides the defaults (pyramid levels, patch size, residual gate,
    /// histeq, etc.), read as `rudolf_v::config::StereoSection` reads it.
    /// Pass the same config file the rest of the pipeline reads, so the
    /// matcher settings stay in sync with it.
    #[staticmethod]
    #[pyo3(signature = (cam0_yaml, cam1_yaml, img_w, img_h, vio_config=None))]
    fn from_euroc(
        cam0_yaml: &str,
        cam1_yaml: &str,
        img_w: usize,
        img_h: usize,
        vio_config: Option<&str>,
    ) -> PyResult<Self> {
        let rig = StereoRig::from_euroc(&PathBuf::from(cam0_yaml), &PathBuf::from(cam1_yaml))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build StereoRig: {e}"))
            })?;
        let cfg = stereo_config_from_vio(vio_config)?;
        Ok(Self {
            inner: StereoMatcher::new(rig, cfg, img_w, img_h),
            img_w,
            img_h,
        })
    }

    /// Build a rectified pinhole stereo matcher from explicit calibration.
    ///
    /// `t_10` is the cam0->cam1 translation in cam1 coordinates under
    /// Rudolf-V's convention `p_cam1 = R_10 p_cam0 + t_10`. For a standard
    /// rectified left/right pair where cam1 is to the right of cam0, use
    /// `t_10 = (-baseline, 0, 0)`.
    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, img_w, img_h, t_10, r_10=None, vio_config=None))]
    fn from_pinhole(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        img_w: usize,
        img_h: usize,
        t_10: [f64; 3],
        r_10: Option<[[f64; 3]; 3]>,
        vio_config: Option<&str>,
    ) -> PyResult<Self> {
        let cam0 = CameraIntrinsics::new(fx, fy, cx, cy, img_w, img_h);
        let cam1 = CameraIntrinsics::new(fx, fy, cx, cy, img_w, img_h);
        let rig = StereoRig::new(
            cam0,
            cam1,
            r_10.unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            t_10,
        );
        let cfg = stereo_config_from_vio(vio_config)?;
        Ok(Self {
            inner: StereoMatcher::new(rig, cfg, img_w, img_h),
            img_w,
            img_h,
        })
    }

    #[getter]
    fn baseline_meters(&self) -> f64 {
        self.inner.rig().baseline_meters()
    }

    /// Match the current frontend features into `cam1_image` and return
    /// `(range, range_var)` priors, ready for
    /// `VIOFilter.process_vision_with_depth_priors`.
    ///
    /// `range` is the euclidean distance from cam0 origin to the
    /// triangulated point. `range_var` follows the CLI's heuristic:
    /// `(residual_px / sigma_pixel_scale)² · range² / baseline²`, clamped
    /// to `>= 1e-4`.
    #[pyo3(signature = (cam1_image, frontend, sigma_pixel_scale=20.0))]
    fn range_priors<'py>(
        &mut self,
        py: Python<'py>,
        cam1_image: PyReadonlyArray2<'py, u8>,
        frontend: PyRef<'py, PyFrontend>,
        sigma_pixel_scale: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let matches = self.match_into(&cam1_image, &frontend)?;
        let rig = self.inner.rig();
        let baseline = rig.baseline_meters().max(1e-9);
        let features = frontend.inner().features();
        let dict = pyo3::types::PyDict::new(py);
        for (feat, m) in features.iter().zip(matches.iter()) {
            let Some(p) = m.point_cam0(rig, feat) else {
                continue;
            };
            let range = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if range <= 0.0 {
                continue;
            }
            let range_var = (m.residual as f64 / sigma_pixel_scale).powi(2) * range * range
                / (baseline * baseline);
            dict.set_item(feat.id, (range, range_var.max(1e-4)))?;
        }
        Ok(dict)
    }

    /// Lower-level access: return a list of per-feature stereo matches with
    /// all the matcher's intermediates (cam1 pixel, inverse depth, residual,
    /// success flag). Useful when you want to do your own RANSAC validation
    /// before promoting matches to depth priors.
    fn match_features<'py>(
        &mut self,
        py: Python<'py>,
        cam1_image: PyReadonlyArray2<'py, u8>,
        frontend: PyRef<'py, PyFrontend>,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let matches = self.match_into(&cam1_image, &frontend)?;
        let features = frontend.inner().features();
        let rig = self.inner.rig();
        let list = pyo3::types::PyList::empty(py);
        for (feat, m) in features.iter().zip(matches.iter()) {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", feat.id)?;
            dict.set_item("u1", m.u1)?;
            dict.set_item("v1", m.v1)?;
            dict.set_item("inv_depth", m.inv_depth)?;
            dict.set_item("residual", m.residual)?;
            dict.set_item("matched", m.matched)?;
            if let Some(p) = m.point_cam0(rig, feat) {
                dict.set_item("point_cam0", (p[0], p[1], p[2]))?;
            } else {
                dict.set_item("point_cam0", py.None())?;
            }
            list.append(dict)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "Stereo(img={}x{}, baseline={:.4}m)",
            self.img_w,
            self.img_h,
            self.inner.rig().baseline_meters()
        )
    }
}

impl PyStereo {
    fn match_into<'py>(
        &mut self,
        cam1_image: &PyReadonlyArray2<'py, u8>,
        frontend: &PyRef<'py, PyFrontend>,
    ) -> PyResult<Vec<rudolf_v::stereo::StereoMatch>> {
        let shape = cam1_image.shape();
        let (w, h) = frontend.dimensions();
        if shape[0] != h || shape[1] != w {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cam1 shape ({}, {}) doesn't match frontend ({}, {})",
                shape[0], shape[1], h, w
            )));
        }
        let data = cam1_image.as_slice()?.to_vec();
        let img = RudolfImage::from_vec(w, h, data);
        let fe = frontend.inner();
        Ok(self
            .inner
            .match_features(&img, fe.features(), fe.current_pyramid()))
    }
}

/// The `Stereo:` section of the YAML file at `vio_config`; the defaults when
/// there is no file or no section.
fn stereo_config_from_vio(vio_config: Option<&str>) -> PyResult<RudolfStereoConfig> {
    let section: Option<StereoSection> = match vio_config {
        Some(path) => read_section(path, "Stereo")?,
        None => None,
    };
    section
        .unwrap_or_default()
        .to_stereo_config()
        .map_err(config_error)
}
