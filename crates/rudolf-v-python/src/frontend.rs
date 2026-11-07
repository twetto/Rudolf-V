use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rudolf_v::camera::{CameraIntrinsics, DistortionModel};
use rudolf_v::config::{self, ConfigError, FrontendSection};
use rudolf_v::frontend::{self, DetectorType, Frontend, LbpPolicy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image as RudolfImage;

use crate::{config_error, read_section};

/// The frontend's settings. The defaults are this binding's own, not
/// `rudolf_v::frontend::FrontendConfig`'s; `from_yaml` reads them from a
/// config file instead.
#[pyclass]
#[derive(Clone)]
pub struct FrontendConfig {
    #[pyo3(get, set)]
    pub max_features: usize,
    /// "fast", "harris" or "shi_tomasi".
    #[pyo3(get, set)]
    pub detector: String,
    #[pyo3(get, set)]
    pub fast_threshold: u8,
    #[pyo3(get, set)]
    pub shi_tomasi_threshold: f32,
    #[pyo3(get, set)]
    pub shi_tomasi_block_size: usize,
    #[pyo3(get, set)]
    pub pyramid_levels: usize,
    #[pyo3(get, set)]
    pub cell_size: usize,
    #[pyo3(get, set)]
    pub klt_window: usize,
    #[pyo3(get, set)]
    pub klt_max_iter: usize,
    /// Only "translation".
    #[pyo3(get, set)]
    pub klt_warp: String,
    #[pyo3(get, set)]
    pub klt_residual: bool,
    #[pyo3(get, set)]
    pub klt_fb_threshold_px: f32,
    #[pyo3(get, set)]
    pub enable_ransac: bool,
    #[pyo3(get, set)]
    pub epipolar_gate_threshold: f64,
    #[pyo3(get, set)]
    pub epipolar_refine: bool,
    #[pyo3(get, set)]
    pub epipolar_min_baseline: f64,
    #[pyo3(get, set)]
    pub epipolar_max_reject_frac: f64,
    #[pyo3(get, set)]
    pub lbp_verification: bool,
    /// "soft" or "hard" (also "SoftPenalty" and "HardReject").
    #[pyo3(get, set)]
    pub lbp_policy: String,
    /// "none", "global" or "clahe".
    #[pyo3(get, set)]
    pub histeq: String,
    #[pyo3(get, set)]
    pub clahe_tile_size: usize,
    #[pyo3(get, set)]
    pub clahe_clip_limit: f32,
    camera: Option<CameraIntrinsics>,
}

#[pymethods]
impl FrontendConfig {
    #[new]
    #[pyo3(signature = (
        max_features = 200,
        fast_threshold = 20,
        pyramid_levels = 3,
        cell_size = 64,
        klt_window = 21,
        klt_max_iter = 30,
        klt_warp = "translation".to_string(),
        klt_residual = false,
        klt_fb_threshold_px = 0.0,
        enable_ransac = true,
        epipolar_gate_threshold = 0.0,
        epipolar_refine = false,
        epipolar_min_baseline = 1e-3,
        epipolar_max_reject_frac = 0.5,
        lbp_verification = true,
        lbp_policy = "soft".to_string(),
        histeq = "global".to_string(),
        clahe_tile_size = 256,
        clahe_clip_limit = 4.0,
        detector = "fast".to_string(),
        shi_tomasi_threshold = 0.0,
        shi_tomasi_block_size = 2,
    ))]
    fn new(
        max_features: usize,
        fast_threshold: u8,
        pyramid_levels: usize,
        cell_size: usize,
        klt_window: usize,
        klt_max_iter: usize,
        klt_warp: String,
        klt_residual: bool,
        klt_fb_threshold_px: f32,
        enable_ransac: bool,
        epipolar_gate_threshold: f64,
        epipolar_refine: bool,
        epipolar_min_baseline: f64,
        epipolar_max_reject_frac: f64,
        lbp_verification: bool,
        lbp_policy: String,
        histeq: String,
        clahe_tile_size: usize,
        clahe_clip_limit: f32,
        detector: String,
        shi_tomasi_threshold: f32,
        shi_tomasi_block_size: usize,
    ) -> Self {
        Self {
            max_features,
            detector,
            fast_threshold,
            shi_tomasi_threshold,
            shi_tomasi_block_size,
            pyramid_levels,
            cell_size,
            klt_window,
            klt_max_iter,
            klt_warp,
            klt_residual,
            klt_fb_threshold_px,
            enable_ransac,
            epipolar_gate_threshold,
            epipolar_refine,
            epipolar_min_baseline,
            epipolar_max_reject_frac,
            lbp_verification,
            lbp_policy,
            histeq,
            clahe_tile_size,
            clahe_clip_limit,
            camera: None,
        }
    }

    /// The `RudolfV:` section of a YAML config file (an ECHO-LI one, say),
    /// read as `rudolf_v::config::FrontendSection` reads it.
    #[staticmethod]
    fn from_yaml(path: &str) -> PyResult<Self> {
        let section: FrontendSection = read_section(path, "RudolfV")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("{path} has no RudolfV section"))
        })?;
        let config = section.to_frontend_config().map_err(config_error)?;
        Ok(Self::from_rust(&config))
    }

    #[pyo3(signature = (
        fx, fy, cx, cy, width, height, distortion,
        distortion_model = "radtan"
    ))]
    fn set_camera(
        &mut self,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
        distortion: Vec<f64>,
        distortion_model: &str,
    ) -> PyResult<()> {
        let mut camera = CameraIntrinsics::new(fx, fy, cx, cy, width, height);
        camera.distortion = distortion;
        camera.model = parse_distortion_model(distortion_model).map_err(config_error)?;
        self.camera = Some(camera);
        Ok(())
    }

    fn __repr__(&self) -> String {
        let cam = if self.camera.is_some() {
            "camera=set"
        } else {
            "camera=none"
        };
        format!(
            "FrontendConfig(max_features={}, detector={}, fast_threshold={}, pyramid_levels={}, cell_size={}, lbp={}, histeq={}, {})",
            self.max_features,
            self.detector,
            self.fast_threshold,
            self.pyramid_levels,
            self.cell_size,
            self.lbp_policy,
            self.histeq,
            cam
        )
    }
}

impl FrontendConfig {
    fn from_rust(cfg: &frontend::FrontendConfig) -> Self {
        let (histeq, clahe_tile_size, clahe_clip_limit) = match cfg.histeq {
            HistEqMethod::None => ("none", 256, 4.0),
            HistEqMethod::Global => ("global", 256, 4.0),
            HistEqMethod::Clahe {
                tile_size,
                clip_limit,
            } => ("clahe", tile_size, clip_limit),
        };
        Self {
            max_features: cfg.max_features,
            detector: match cfg.detector {
                DetectorType::Fast => "fast",
                DetectorType::Harris => "harris",
                DetectorType::ShiTomasi => "shi_tomasi",
            }
            .to_string(),
            fast_threshold: cfg.fast_threshold,
            shi_tomasi_threshold: cfg.shi_tomasi_threshold,
            shi_tomasi_block_size: cfg.shi_tomasi_block_size,
            pyramid_levels: cfg.pyramid_levels,
            cell_size: cfg.cell_size,
            klt_window: cfg.klt_window,
            klt_max_iter: cfg.klt_max_iter,
            klt_warp: "translation".to_string(),
            klt_residual: cfg.klt_residual_enabled,
            klt_fb_threshold_px: cfg.klt_fb_threshold_px,
            enable_ransac: cfg.enable_internal_ransac,
            epipolar_gate_threshold: cfg.epipolar_gate_threshold,
            epipolar_refine: cfg.epipolar_refine,
            epipolar_min_baseline: cfg.epipolar_min_baseline,
            epipolar_max_reject_frac: cfg.epipolar_max_reject_frac,
            lbp_verification: cfg.lbp_verification_enabled,
            lbp_policy: match cfg.lbp_policy {
                LbpPolicy::SoftPenalty => "soft",
                LbpPolicy::HardReject => "hard",
            }
            .to_string(),
            histeq: histeq.to_string(),
            clahe_tile_size,
            clahe_clip_limit,
            camera: None,
        }
    }

    /// The Rust config, with inverse-compositional KLT; an error when a name
    /// is not one of the supported ones. No Python in here, so the tests link
    /// without libpython.
    fn to_rust(&self) -> Result<frontend::FrontendConfig, ConfigError> {
        if !self.klt_warp.eq_ignore_ascii_case("translation") {
            return Err(ConfigError {
                key: "klt_warp",
                value: self.klt_warp.clone(),
                expected: "translation",
            });
        }
        Ok(frontend::FrontendConfig {
            max_features: self.max_features,
            detector: config::parse_detector(&self.detector)?,
            fast_threshold: self.fast_threshold,
            shi_tomasi_threshold: self.shi_tomasi_threshold,
            shi_tomasi_block_size: self.shi_tomasi_block_size,
            pyramid_levels: self.pyramid_levels,
            cell_size: self.cell_size,
            klt_window: self.klt_window,
            klt_max_iter: self.klt_max_iter,
            klt_method: rudolf_v::klt::LkMethod::InverseCompositional,
            klt_residual_enabled: self.klt_residual,
            klt_fb_threshold_px: self.klt_fb_threshold_px,
            enable_internal_ransac: self.enable_ransac,
            epipolar_gate_threshold: self.epipolar_gate_threshold,
            epipolar_refine: self.epipolar_refine,
            epipolar_min_baseline: self.epipolar_min_baseline,
            epipolar_max_reject_frac: self.epipolar_max_reject_frac,
            lbp_verification_enabled: self.lbp_verification,
            lbp_policy: config::parse_lbp_policy(&self.lbp_policy)?,
            histeq: config::parse_histeq(
                &self.histeq,
                self.clahe_tile_size,
                self.clahe_clip_limit,
            )?,
            camera: self.camera.clone(),
            ..frontend::FrontendConfig::default()
        })
    }
}

fn parse_distortion_model(name: &str) -> Result<DistortionModel, ConfigError> {
    match name.to_ascii_lowercase().as_str() {
        "none" | "pinhole" => Ok(DistortionModel::None),
        "radtan" | "radial-tangential" => Ok(DistortionModel::RadTan),
        "equidistant" | "fisheye" => Ok(DistortionModel::Equidistant),
        _ => Err(ConfigError {
            key: "distortion_model",
            value: name.to_string(),
            expected: "none, radtan or equidistant",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> FrontendConfig {
        FrontendConfig::new(
            200,
            20,
            3,
            64,
            21,
            30,
            "translation".to_string(),
            false,
            0.0,
            true,
            0.0,
            false,
            1e-3,
            0.5,
            true,
            "soft".to_string(),
            "global".to_string(),
            256,
            4.0,
            "fast".to_string(),
            0.0,
            2,
        )
    }

    #[test]
    fn python_frontend_preserves_equidistant_model() {
        let mut config = defaults();
        let mut camera = CameraIntrinsics::new(462.4, 462.5, 670.4, 398.4, 1280, 800);
        camera.distortion = vec![0.068, 0.0022, 0.0046, -0.0033];
        camera.model = parse_distortion_model("equidistant").unwrap();
        config.camera = Some(camera);

        let camera = config.to_rust().unwrap().camera.unwrap();
        assert_eq!(camera.model, DistortionModel::Equidistant);
        assert_eq!(camera.distortion.len(), 4);
        assert!(parse_distortion_model("kannala").is_err());
    }

    #[test]
    fn names_round_trip_through_the_rust_config() {
        let mut config = defaults();
        config.detector = "Shi-Tomasi".to_string();
        config.lbp_policy = "HardReject".to_string();
        config.histeq = "clahe".to_string();
        config.clahe_tile_size = 32;
        let rust = config.to_rust().unwrap();
        let back = FrontendConfig::from_rust(&rust);
        assert_eq!(back.detector, "shi_tomasi");
        assert_eq!(back.lbp_policy, "hard");
        assert_eq!((back.histeq.as_str(), back.clahe_tile_size), ("clahe", 32));
        assert_eq!(back.cell_size, 64);
        assert_eq!(back.klt_window, 21);
        config.klt_warp = "affine".to_string();
        assert_eq!(config.to_rust().err().unwrap().key, "klt_warp");
    }
}

#[pyclass(name = "Frontend")]
pub struct PyFrontend {
    inner: Frontend,
    width: usize,
    height: usize,
}

impl PyFrontend {
    pub(crate) fn inner(&self) -> &Frontend {
        &self.inner
    }

    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

#[pymethods]
impl PyFrontend {
    #[new]
    fn new(config: &FrontendConfig, width: usize, height: usize) -> PyResult<Self> {
        let inner = Frontend::new(config.to_rust().map_err(config_error)?, width, height);
        Ok(Self {
            inner,
            width,
            height,
        })
    }

    fn process<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<(Bound<'py, pyo3::types::PyList>, PyObject)> {
        let shape = image.shape();
        if shape[0] != self.height || shape[1] != self.width {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Image shape ({}, {}) doesn't match frontend ({}, {})",
                shape[0], shape[1], self.height, self.width
            )));
        }
        let data = image.as_slice()?;
        let rudolf_img = RudolfImage::from_vec(self.width, self.height, data.to_vec());
        let (features, stats) = py.allow_threads(|| self.inner.process(&rudolf_img));

        let feat_list = pyo3::types::PyList::empty(py);
        for f in features {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", f.id)?;
            dict.set_item("x", f.x)?;
            dict.set_item("y", f.y)?;
            dict.set_item("score", f.score)?;
            dict.set_item("level", f.level)?;
            feat_list.append(dict)?;
        }

        let stats_dict = pyo3::types::PyDict::new(py);
        stats_dict.set_item("tracked", stats.tracked)?;
        stats_dict.set_item("lost", stats.lost)?;
        stats_dict.set_item("rejected", stats.rejected)?;
        stats_dict.set_item("new_detections", stats.new_detections)?;
        stats_dict.set_item("total", stats.total)?;
        stats_dict.set_item("timing_ms", stats.timing.total_ms())?;
        stats_dict.set_item("histeq_ms", stats.timing.histeq_ms())?;
        stats_dict.set_item("pyramid_ms", stats.timing.pyramid_ms())?;
        stats_dict.set_item("klt_ms", stats.timing.klt_ms())?;
        stats_dict.set_item("ransac_ms", stats.timing.ransac_ms())?;
        stats_dict.set_item("detect_ms", stats.timing.detect_ms())?;

        Ok((feat_list, stats_dict.into_any().unbind()))
    }

    fn drop_tracks(&mut self, ids: Vec<u64>) -> usize {
        self.inner.drop_tracks(&ids)
    }

    /// Relative-pose prior for the NEXT process() call: 4x4 T (prev camera ->
    /// current camera, i.e. x_curr ~ R x_prev + t). Enables the epipolar gate
    /// when epipolar_gate_threshold > 0.
    fn set_pose_prior(&mut self, t_rel: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let a = t_rel.as_array();
        if a.shape() != [4, 4] {
            return Err(pyo3::exceptions::PyValueError::new_err("t_rel must be 4x4"));
        }
        let mut r = [[0.0f64; 3]; 3];
        let mut t = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[[i, j]];
            }
            t[i] = a[[i, 3]];
        }
        self.inner.set_pose_prior(r, t);
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let feat_list = pyo3::types::PyList::empty(py);
        for f in self.inner.features() {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", f.id)?;
            dict.set_item("x", f.x)?;
            dict.set_item("y", f.y)?;
            dict.set_item("score", f.score)?;
            dict.set_item("level", f.level)?;
            feat_list.append(dict)?;
        }
        Ok(feat_list)
    }

    fn track_meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let meta_list = pyo3::types::PyList::empty(py);
        for m in self.inner.track_meta() {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", m.id)?;
            dict.set_item("age", m.age)?;
            dict.set_item("klt_quality", m.klt_quality)?;
            dict.set_item("lbp_distance", m.lbp_distance)?;
            dict.set_item("reservoir_score", m.reservoir_score)?;
            dict.set_item("is_ekf_landmark", m.is_ekf_landmark)?;
            meta_list.append(dict)?;
        }
        Ok(meta_list)
    }

    fn preprocessed_image<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<u8>>>> {
        let Some(img) = self.inner.preprocessed_image() else {
            return Ok(None);
        };

        let width = img.width();
        let height = img.height();
        let stride = img.stride();
        let src = img.as_slice();
        let data = if stride == width {
            src[..width * height].to_vec()
        } else {
            let mut compact = Vec::with_capacity(width * height);
            for y in 0..height {
                let row = &src[y * stride..y * stride + width];
                compact.extend_from_slice(row);
            }
            compact
        };
        let arr = Array2::from_shape_vec((height, width), data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Some(PyArray2::from_owned_array(py, arr)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Frontend({}x{}, {} features)",
            self.width,
            self.height,
            self.inner.features().len()
        )
    }
}
