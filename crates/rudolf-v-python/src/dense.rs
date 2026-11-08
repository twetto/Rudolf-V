use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rudolf_v::camera::{CameraIntrinsics, DistortionModel, StereoRig};
use rudolf_v::dense_stereo::DenseDepth;
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::rectify::GpuRectification;
use rudolf_v::gpu::sgbm::GpuSgbm;
use rudolf_v::image::Image as RudolfImage;
use rudolf_v::rectify::StereoRectification;
use rudolf_v::sgbm::{Sgbm, SgbmConfig};
#[cfg(unix)]
use rudolf_v::stereo_net::{StereoNet, StereoNetConfig, StereoNetKind};
#[cfg(unix)]
use rudolf_v::tensorrt::TrtEngine;

use crate::checkerboard::to_image;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// `f`, with a panic (a failed assertion on the settings) as a ValueError.
fn checked<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("invalid settings");
        value_error(msg)
    })
}

/// A rectified pinhole pair: both eyes the same camera, the right one
/// `baseline` metres along +x.
fn rectified_rig(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    baseline: f64,
    width: usize,
    height: usize,
) -> PyResult<StereoRig> {
    if !(baseline > 0.0 && fx > 0.0 && fy > 0.0) {
        return Err(value_error(
            "fx, fy and baseline must be positive (the right camera at +x)",
        ));
    }
    let cam = CameraIntrinsics::new(fx, fy, cx, cy, width, height);
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    Ok(StereoRig::new(
        cam.clone(),
        cam,
        identity,
        [-baseline, 0.0, 0.0],
    ))
}

fn check_shape(image: &PyReadonlyArray2<'_, u8>, width: usize, height: usize) -> PyResult<()> {
    let shape = image.shape();
    if shape != [height, width] {
        return Err(value_error(format!(
            "expected a ({height}, {width}) image, got {shape:?}"
        )));
    }
    Ok(())
}

fn array<T: numpy::Element>(
    py: Python<'_>,
    width: usize,
    height: usize,
    data: Vec<T>,
) -> Bound<'_, PyArray2<T>> {
    PyArray2::from_owned_array(py, Array2::from_shape_vec((height, width), data).unwrap())
}

/// A uint8 (height, width) image.
type Grey<'py> = Bound<'py, PyArray2<u8>>;

enum Matcher {
    Sgbm(Box<Sgbm>),
    GpuSgbm(Box<(GpuDevice, GpuSgbm)>),
    #[cfg(unix)]
    Net(Box<StereoNet>),
}

/// Dense depth of a rectified stereo pair, from semi-global block matching
/// (`sgbm`) or a stereo network's TensorRT engine (`tensorrt`).
///
/// `compute(left, right)` gives depth on the left image's pyramid level
/// `level` (1280×720 at level 3: 160×90), map pixel (x, y) at image pixel
/// (x·2^level, y·2^level): Z along the optical axis in metres, NaN where
/// there is none. The GIL is released while it runs; one pair at a time per
/// object.
#[pyclass(name = "StereoDepth")]
pub struct PyStereoDepth {
    /// In a mutex only because pyo3 needs the class to be Sync; the methods
    /// that use it take &mut self.
    matcher: Mutex<Matcher>,
    kind: &'static str,
    depth: DenseDepth,
    /// 1/|K⁻¹(u, v, 1)| per map pixel: Z = range · this.
    inv_ray: Vec<f32>,
    width: usize,
    height: usize,
    confidence: Option<Vec<f32>>,
}

impl PyStereoDepth {
    fn new(matcher: Matcher, rig: &StereoRig, level: usize) -> Self {
        let kind = match &matcher {
            Matcher::Sgbm(_) => "sgbm",
            Matcher::GpuSgbm(_) => "sgbm-gpu",
            #[cfg(unix)]
            Matcher::Net(m) => match m.kind() {
                StereoNetKind::LightEss => "light-ess",
                StereoNetKind::Las2 => "las2",
            },
        };
        let [width, height] = rig.cam0.resolution;
        let c = &rig.cam0;
        let s = 1usize << level;
        let inv_ray = (0..height >> level)
            .flat_map(|y| (0..width >> level).map(move |x| (x, y)))
            .map(|(x, y)| {
                let a = ((x * s) as f64 - c.cx) / c.fx;
                let b = ((y * s) as f64 - c.cy) / c.fy;
                (1.0 / (a * a + b * b + 1.0).sqrt()) as f32
            })
            .collect();
        Self {
            matcher: Mutex::new(matcher),
            kind,
            depth: DenseDepth {
                width: 0,
                height: 0,
                level,
                log_range: Vec::new(),
                variance: Vec::new(),
                source_level: Vec::new(),
            },
            inv_ray,
            width,
            height,
            confidence: None,
        }
    }
}

#[pymethods]
impl PyStereoDepth {
    /// Semi-global block matching (OpenCV's MODE_HH: 8 paths, 5×5 blocks)
    /// on the GPU (`gpu=True`; RuntimeError without a Vulkan GPU) or the CPU.
    /// The rectified camera: fx, fy, cx, cy in pixels of the `width`×`height`
    /// images, `baseline` in metres.
    #[staticmethod]
    #[pyo3(signature = (
        fx, fy, cx, cy, baseline, width, height,
        level=3, min_range=0.25, max_range=30.0, gpu=true, paths=8,
    ))]
    fn sgbm(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        baseline: f64,
        width: usize,
        height: usize,
        level: usize,
        min_range: f64,
        max_range: f64,
        gpu: bool,
        paths: usize,
    ) -> PyResult<Self> {
        let rig = rectified_rig(fx, fy, cx, cy, baseline, width, height)?;
        let config = SgbmConfig {
            level,
            min_range,
            max_range,
            paths,
            ..SgbmConfig::default()
        };
        let cpu = checked(|| Sgbm::new(&rig, config))?;
        let matcher = if gpu {
            let device = GpuDevice::new().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("gpu=True: no Vulkan GPU ({e})"))
            })?;
            let m = GpuSgbm::new(&device, &cpu);
            Matcher::GpuSgbm(Box::new((device, m)))
        } else {
            Matcher::Sgbm(Box::new(cpu))
        };
        Ok(Self::new(matcher, &rig, level))
    }

    /// A stereo network's TensorRT engine: NVIDIA's Light ESS (its EULA
    /// allows commercial use on NVIDIA GPUs) or LAS2 (MIT weights, but
    /// trained partly on research-only labels: check its terms first), told
    /// apart by their tensor names. `library` is Rudolf-V's librv_trt.so
    /// (native/tensorrt, built on the Jetson), `plugins` the engine's plugin
    /// libraries (Light ESS: ess_plugins.so). Pixels the network is less
    /// sure of than `min_confidence` (Light ESS only) have no depth.
    #[cfg(unix)]
    #[staticmethod]
    #[pyo3(signature = (
        engine, fx, fy, cx, cy, baseline, width, height, library,
        plugins=Vec::new(), level=3, min_range=0.25, max_range=30.0, min_confidence=0.0,
    ))]
    fn tensorrt(
        py: Python<'_>,
        engine: std::path::PathBuf,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        baseline: f64,
        width: usize,
        height: usize,
        library: std::path::PathBuf,
        plugins: Vec<std::path::PathBuf>,
        level: usize,
        min_range: f64,
        max_range: f64,
        min_confidence: f32,
    ) -> PyResult<Self> {
        let rig = rectified_rig(fx, fy, cx, cy, baseline, width, height)?;
        let config = StereoNetConfig {
            level,
            min_range,
            max_range,
            min_confidence,
            ..StereoNetConfig::default()
        };
        let net = py.allow_threads(|| {
            let plugins: Vec<&Path> = plugins.iter().map(|p| p.as_path()).collect();
            let engine = TrtEngine::load(&library, &engine, &plugins)?;
            StereoNet::new(engine, None, &rig, config)
        });
        let net = net.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self::new(Matcher::Net(Box::new(net)), &rig, level))
    }

    /// Depth of a rectified pair, uint8 (height, width) each: Z in metres,
    /// float32 on the level's grid, NaN where there is none.
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        left: PyReadonlyArray2<'py, u8>,
        right: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        check_shape(&left, self.width, self.height)?;
        check_shape(&right, self.width, self.height)?;
        let (left, right) = (to_image(&left), to_image(&right));
        let Self {
            matcher,
            depth,
            inv_ray,
            confidence,
            ..
        } = self;
        let matcher = matcher.get_mut().unwrap_or_else(|e| e.into_inner());
        let z = py.allow_threads(|| -> Result<Vec<f32>, String> {
            match matcher {
                Matcher::Sgbm(m) => m.compute_into(&left, &right, depth),
                Matcher::GpuSgbm(m) => {
                    let (device, m) = &mut **m;
                    m.compute_into(device, &left, &right, depth)
                }
                #[cfg(unix)]
                Matcher::Net(m) => {
                    m.compute_into(&left, &right, depth)?;
                    *confidence = m.confidence();
                }
            }
            Ok(depth
                .log_range
                .iter()
                .zip(inv_ray.iter())
                .map(|(&l, &r)| if l.is_finite() { l.exp() * r } else { f32::NAN })
                .collect())
        });
        let z = z.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(array(py, self.depth.width, self.depth.height, z))
    }

    /// The last pair's variance of ln(depth), on the depth's grid; NaN where
    /// there is no depth.
    fn variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let d = &self.depth;
        array(py, d.width, d.height, d.variance.clone())
    }

    /// The last pair's confidence (0..1) on the depth's grid, for a network
    /// that has one (Light ESS); None otherwise.
    fn confidence<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        let d = &self.depth;
        let c = self.confidence.clone()?;
        Some(array(py, d.width, d.height, c))
    }

    /// "sgbm", "sgbm-gpu", or the network: "light-ess", "las2".
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn level(&self) -> usize {
        self.depth.level
    }

    fn __repr__(&self) -> String {
        format!(
            "StereoDepth({}, {}x{} at level {})",
            self.kind(),
            self.width,
            self.height,
            self.depth.level
        )
    }
}

/// Rectification of a raw stereo pair into a shared pinhole camera with
/// focal length `fx` (= fy) and principal point (cx, cy), at the left
/// camera's resolution (Bouguet's rotations, as OpenCV's stereoRectify with
/// CALIB_ZERO_DISPARITY). Pass the rectified camera on to `StereoDepth`:
/// `fx`, `fy`, `cx`, `cy`, `baseline`, `width`, `height`.
///
/// `gpu=True` rectifies on the GPU (Vulkan; RuntimeError without one): the
/// same images but for one grey level on a few pixels in a million (the
/// GPU's fused multiply-adds), and the CPU's cores free.
#[pyclass(name = "StereoRectification")]
pub struct PyStereoRectification {
    inner: StereoRectification,
    scratch: [RudolfImage<f32>; 2],
    /// In a mutex only because pyo3 needs the class to be Sync.
    gpu: Option<Mutex<(GpuDevice, GpuRectification)>>,
    rectified: [RudolfImage<u8>; 2],
}

impl PyStereoRectification {
    fn new(rig: &StereoRig, fx: f64, cx: f64, cy: f64, gpu: bool) -> PyResult<Self> {
        let inner = checked(|| StereoRectification::new(rig, fx, cx, cy))?;
        let gpu = if gpu {
            let device = GpuDevice::new().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("gpu=True: no Vulkan GPU ({e})"))
            })?;
            let on_gpu = GpuRectification::new(&device, &inner);
            Some(Mutex::new((device, on_gpu)))
        } else {
            None
        };
        Ok(Self {
            inner,
            scratch: [RudolfImage::new(1, 1), RudolfImage::new(1, 1)],
            gpu,
            rectified: [RudolfImage::new(1, 1), RudolfImage::new(1, 1)],
        })
    }
}

#[pymethods]
impl PyStereoRectification {
    /// From a ZED camera's factory calibration
    /// (/usr/local/zed/settings/SN<serial>.conf) at `resolution` ("2K",
    /// "FHD", "HD", "VGA"), with its `distortion`: "rational" (the
    /// [LEFT_DISTO]/[RIGHT_DISTO] model) or "radtan".
    #[staticmethod]
    #[pyo3(signature = (path, fx, cx, cy, resolution="HD", distortion="rational", gpu=false))]
    fn from_zed_conf(
        path: std::path::PathBuf,
        fx: f64,
        cx: f64,
        cy: f64,
        resolution: &str,
        distortion: &str,
        gpu: bool,
    ) -> PyResult<Self> {
        let model = match distortion.to_ascii_lowercase().as_str() {
            "rational" => DistortionModel::Rational,
            "radtan" => DistortionModel::RadTan,
            other => {
                return Err(value_error(format!(
                    "unsupported distortion '{other}'; expected rational or radtan"
                )));
            }
        };
        let rig = StereoRig::from_zed_conf(&path, resolution, model).map_err(value_error)?;
        Self::new(&rig, fx, cx, cy, gpu)
    }

    /// From a Kalibr camchain.yaml of the raw pair.
    #[staticmethod]
    #[pyo3(signature = (path, fx, cx, cy, gpu=false))]
    fn from_kalibr(
        path: std::path::PathBuf,
        fx: f64,
        cx: f64,
        cy: f64,
        gpu: bool,
    ) -> PyResult<Self> {
        let rig = StereoRig::from_kalibr_camchain(&path).map_err(value_error)?;
        Self::new(&rig, fx, cx, cy, gpu)
    }

    /// Both eyes of a raw pair, uint8 (height, width) each, rectified:
    /// bilinear, rounded to uint8, 0 where a rectified pixel sees outside
    /// the raw image.
    fn rectify<'py>(
        &mut self,
        py: Python<'py>,
        left: PyReadonlyArray2<'py, u8>,
        right: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<(Grey<'py>, Grey<'py>)> {
        let (w, h) = (self.inner.width(), self.inner.height());
        check_shape(&left, w, h)?;
        check_shape(&right, w, h)?;
        let raw = [to_image(&left), to_image(&right)];
        let Self {
            inner,
            scratch,
            gpu,
            rectified,
        } = self;
        let [l, r] = py.allow_threads(|| match gpu {
            Some(g) => {
                let (device, on_gpu) = g.get_mut().unwrap_or_else(|e| e.into_inner());
                on_gpu.rectify_into(device, &raw[0], &raw[1], rectified);
                [0, 1].map(|eye| rectified[eye].as_slice()[..w * h].to_vec())
            }
            None => [0, 1].map(|eye| {
                inner.remap(eye, &raw[eye], &mut scratch[eye]);
                scratch[eye]
                    .as_slice()
                    .iter()
                    .map(|v| v.round().clamp(0.0, 255.0) as u8)
                    .collect::<Vec<u8>>()
            }),
        });
        Ok((array(py, w, h, l), array(py, w, h, r)))
    }

    #[getter]
    fn fx(&self) -> f64 {
        self.inner.camera().0
    }

    #[getter]
    fn fy(&self) -> f64 {
        self.inner.camera().0
    }

    #[getter]
    fn cx(&self) -> f64 {
        self.inner.camera().1
    }

    #[getter]
    fn cy(&self) -> f64 {
        self.inner.camera().2
    }

    /// Metres between the rectified eyes.
    #[getter]
    fn baseline(&self) -> f64 {
        self.inner.rig().baseline_meters()
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Whether `rectify` runs on the GPU.
    #[getter]
    fn gpu(&self) -> bool {
        self.gpu.is_some()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    fn __repr__(&self) -> String {
        let (fx, cx, cy) = self.inner.camera();
        format!(
            "StereoRectification({}x{}, fx={fx:.2}, cx={cx:.2}, cy={cy:.2}, baseline={:.6} m, gpu={})",
            self.inner.width(),
            self.inner.height(),
            self.baseline(),
            if self.gpu.is_some() { "True" } else { "False" },
        )
    }
}
