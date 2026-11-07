use std::sync::Mutex;

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use rudolf_v::checkerboard::{CheckerboardConfig, CheckerboardDetector};
use rudolf_v::gpu::checkerboard::GpuCheckerboardDetector;
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::image::Image as RudolfImage;

/// Checkerboard inner-corner detector wrapping `rudolf_v::checkerboard`.
///
/// ChESS corner responses grown into a `cols x rows` lattice. Works on raw
/// (distorted) images, fisheye included; the corners come back in the input
/// image's pixels, so undistort the points, not the image. Order: row-major,
/// rows along the `cols` axis pointing +x, numbered to keep the image's
/// handedness (never a mirror image). A board turned 180° gives the same order
/// reversed, which the pattern itself cannot tell.
///
/// `threads`: run on a pool of its own with this many threads (the response
/// image is computed in parallel rows). None uses rayon's global pool, which
/// has a thread per core.
///
/// `gpu`: the response and its candidates on the GPU (Vulkan), the lattice
/// search on the CPU; the same corners, bit for bit. Raises RuntimeError when
/// there is no Vulkan GPU. One look at a time per detector.
///
/// The other keyword arguments override `CheckerboardConfig`'s defaults.
#[pyclass(name = "CheckerboardDetector")]
pub struct PyCheckerboardDetector {
    inner: CheckerboardDetector,
    gpu: Option<Mutex<OnGpu>>,
    pool: Option<rayon::ThreadPool>,
    threads: Option<usize>,
}

/// A GPU detector with its device.
struct OnGpu {
    device: GpuDevice,
    detector: GpuCheckerboardDetector,
}

#[pymethods]
impl PyCheckerboardDetector {
    #[new]
    #[pyo3(signature = (
        cols=9, rows=6, threads=None, min_response=None, relative_threshold=None,
        grow_response=None, max_candidates=None, max_seeds=None, match_tolerance=None,
        min_cell_contrast=None, gpu=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cols: usize,
        rows: usize,
        threads: Option<usize>,
        min_response: Option<f32>,
        relative_threshold: Option<f32>,
        grow_response: Option<f32>,
        max_candidates: Option<usize>,
        max_seeds: Option<usize>,
        match_tolerance: Option<f32>,
        min_cell_contrast: Option<f32>,
        gpu: bool,
    ) -> PyResult<Self> {
        if cols < 2 || rows < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "a board needs at least 2x2 inner corners, got {cols}x{rows}"
            )));
        }
        let d = CheckerboardConfig::default();
        let config = CheckerboardConfig {
            cols,
            rows,
            min_response: min_response.unwrap_or(d.min_response),
            relative_threshold: relative_threshold.unwrap_or(d.relative_threshold),
            grow_response: grow_response.unwrap_or(d.grow_response),
            max_candidates: max_candidates.unwrap_or(d.max_candidates),
            max_seeds: max_seeds.unwrap_or(d.max_seeds),
            match_tolerance: match_tolerance.unwrap_or(d.match_tolerance),
            min_cell_contrast: min_cell_contrast.unwrap_or(d.min_cell_contrast),
        };
        let pool = match threads {
            Some(0) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "threads must be at least 1",
                ));
            }
            Some(n) => Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to start the thread pool: {e}"
                        ))
                    })?,
            ),
            None => None,
        };
        let gpu = if gpu {
            let device = GpuDevice::new().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("gpu=True: no Vulkan GPU ({e})"))
            })?;
            let detector = GpuCheckerboardDetector::new(&device, config.clone());
            Some(Mutex::new(OnGpu { device, detector }))
        } else {
            None
        };
        Ok(Self {
            inner: CheckerboardDetector::new(config),
            gpu,
            pool,
            threads,
        })
    }

    /// The board's inner corners as a (cols*rows, 2) float32 array of (x, y)
    /// pixels, or None when no complete board is in the image. `image`: 8-bit
    /// grey, (height, width).
    fn detect<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
    ) -> Option<Bound<'py, PyArray2<f32>>> {
        let img = to_image(&image);
        let corners = py.allow_threads(|| self.detect_image(&img))?;
        Some(corners_array(py, corners))
    }

    /// As `detect`, within the region (x, y, w, h) of `image`, clipped to the
    /// image; corners in the whole image's pixels. For looking again near
    /// where the board was last seen.
    fn detect_region<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
        x: i64,
        y: i64,
        w: i64,
        h: i64,
    ) -> Option<Bound<'py, PyArray2<f32>>> {
        let shape = image.shape();
        let (img_w, img_h) = (shape[1] as i64, shape[0] as i64);
        let (x0, y0) = (x.clamp(0, img_w), y.clamp(0, img_h));
        let (x1, y1) = ((x + w).clamp(0, img_w), (y + h).clamp(0, img_h));
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let img = to_image(&image);
        let (rx, ry, rw, rh) = (
            x0 as usize,
            y0 as usize,
            (x1 - x0) as usize,
            (y1 - y0) as usize,
        );
        let corners = py.allow_threads(|| match &self.gpu {
            Some(g) => {
                let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
                let OnGpu { device, detector } = &mut *g;
                self.in_pool(|| detector.detect_region(device, &img, rx, ry, rw, rh))
            }
            None => self.run(|d| d.detect_region(&img, rx, ry, rw, rh)),
        })?;
        Some(corners_array(py, corners))
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.config.cols
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.config.rows
    }

    fn __repr__(&self) -> String {
        let c = &self.inner.config;
        format!(
            "CheckerboardDetector(cols={}, rows={}, threads={}, min_response={}, relative_threshold={}, \
             grow_response={}, max_candidates={}, max_seeds={}, match_tolerance={}, min_cell_contrast={}, \
             gpu={})",
            c.cols,
            c.rows,
            self.threads.map_or("None".to_string(), |n| n.to_string()),
            c.min_response,
            c.relative_threshold,
            c.grow_response,
            c.max_candidates,
            c.max_seeds,
            c.match_tolerance,
            c.min_cell_contrast,
            if self.gpu.is_some() { "True" } else { "False" },
        )
    }

    /// Whether the response and candidates run on the GPU.
    #[getter]
    fn gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

impl PyCheckerboardDetector {
    /// `f` on the detector, in this detector's pool when it has one.
    fn run<T: Send>(&self, f: impl FnOnce(&CheckerboardDetector) -> T + Send) -> T {
        self.in_pool(|| f(&self.inner))
    }

    /// `f` in this detector's pool when it has one.
    fn in_pool<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// The board in `img`, on the GPU or the CPU.
    fn detect_image(&self, img: &RudolfImage<u8>) -> Option<Vec<[f32; 2]>> {
        match &self.gpu {
            Some(g) => {
                let mut g = g.lock().unwrap_or_else(|e| e.into_inner());
                let OnGpu { device, detector } = &mut *g;
                self.in_pool(|| detector.detect(device, img))
            }
            None => self.run(|d| d.detect(img)),
        }
    }
}

/// A numpy (height, width) u8 array as a Rudolf-V image; a strided view (a
/// crop, say) is copied row by row.
fn to_image(image: &PyReadonlyArray2<'_, u8>) -> RudolfImage<u8> {
    let shape = image.shape();
    let data = match image.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => image.as_array().iter().copied().collect(),
    };
    RudolfImage::from_vec(shape[1], shape[0], data)
}

fn corners_array(py: Python<'_>, corners: Vec<[f32; 2]>) -> Bound<'_, PyArray2<f32>> {
    let n = corners.len();
    let flat: Vec<f32> = corners.into_iter().flatten().collect();
    PyArray2::from_owned_array(py, Array2::from_shape_vec((n, 2), flat).unwrap())
}
