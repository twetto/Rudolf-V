use camera_geometry::{CameraModel, EquidistantFisheye, Pixel};
use nalgebra::Vector3;
use pyo3::prelude::*;

#[pyclass(name = "EquidistantFisheye")]
struct PyEquidistantFisheye {
    inner: EquidistantFisheye,
}

#[pymethods]
impl PyEquidistantFisheye {
    #[new]
    fn new(intrinsics: [f64; 4], distortion: [f64; 4], resolution: [usize; 2]) -> Self {
        Self {
            inner: EquidistantFisheye::new(intrinsics, distortion, resolution),
        }
    }

    #[getter]
    fn image_size(&self) -> [usize; 2] {
        self.inner.image_size()
    }

    fn unproject(&self, pixel: [f64; 2]) -> Option<[f64; 3]> {
        self.inner
            .unproject(Pixel::new(pixel[0], pixel[1]))
            .map(|bearing| bearing.vector().into())
    }

    fn project(&self, point: [f64; 3]) -> Option<[f64; 2]> {
        self.inner
            .project(Vector3::from(point))
            .map(|pixel| pixel.0.into())
    }

    fn unproject_tangent_jacobian(&self, pixel: [f64; 2]) -> Option<[[f64; 2]; 2]> {
        self.inner
            .unproject_tangent_jacobian(Pixel::new(pixel[0], pixel[1]))
            .map(|j| [[j[(0, 0)], j[(0, 1)]], [j[(1, 0)], j[(1, 1)]]])
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEquidistantFisheye>()?;
    Ok(())
}
