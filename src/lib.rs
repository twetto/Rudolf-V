//! Rudolf-V: **Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend
//! **V**ision.
//!
//! A visual frontend for VIO (feature detection, pyramidal KLT tracking,
//! geometric verification) and stereo depth, on the CPU and, with the `gpu`
//! feature, on the GPU through wgpu. It started as a port of vilib: Nagy,
//! Foehn, Scaramuzza, "Faster than FAST: GPU-Accelerated Frontend for
//! High-Speed VIO" (IROS 2020).
//!
//! # Depending on it
//!
//! There are no crates.io releases: depend on a git revision, and move the
//! pin on purpose.
//!
//! ```toml
//! rudolf-v = { git = "https://github.com/twetto/Rudolf-V.git", rev = "..." }
//! ```
//!
//! Features: `gpu` (default) builds `gpu`, on wgpu; without it the CPU
//! library needs neither wgpu nor a GPU. `parallel` spreads the CPU work over
//! threads with rayon. `serde` adds `config`, the settings as a YAML config
//! file spells them. `stereo_net` and `tensorrt` are Unix only.
//!
//! # The supported API
//!
//! What downstream code (ECHO-LI, iq-neuron, mr-neuro-nav through the Python
//! wheel in `crates/rudolf-v-python`) is meant to use, and what changes with
//! care:
//!
//! - The frontend: `frontend::Frontend` and `FrontendConfig`; on the GPU,
//!   `gpu::frontend::GpuFrontend` (a drop-in, with `GpuFrontendConfig`).
//! - Sparse stereo: `stereo::StereoMatcher` and `StereoConfig`.
//! - Dense depth, all to a `dense_stereo::DenseDepth`: `sgbm::Sgbm` and
//!   `gpu::sgbm::GpuSgbm`; `dense_stereo::DenseStereo` and
//!   `gpu::dense_stereo::GpuDenseStereo`; `stereo_net::StereoNet` over a
//!   `tensorrt::TrtEngine`. Raw pairs first through
//!   `rectify::StereoRectification` or `gpu::rectify::GpuRectification`.
//! - Checkerboards: `checkerboard::CheckerboardDetector` and
//!   `gpu::checkerboard::GpuCheckerboardDetector`.
//! - Cameras: `camera::CameraIntrinsics` and `camera::StereoRig`, and the
//!   re-exported `camera_geometry` crate.
//! - Images: `image::Image` and `pyramid::Pyramid` (`Pyramid::empty` for a
//!   pyramid to build into).
//! - Config files (`serde`): `config::FrontendSection`, `StereoSection`.
//!
//! The other modules are public too, for the tests, benchmarks and examples
//! and for whoever wants the building blocks (FAST, Shi-Tomasi, Harris, KLT,
//! NMS, RANSAC, the GPU kernels' pipelines), but they follow the frontend's
//! needs and may change from one revision to the next.

pub mod camera;
pub mod checkerboard;
#[cfg(feature = "serde")]
pub mod config;
pub mod convert;
pub mod convolution;
pub mod dense_stereo;
pub mod essential;
pub mod fast;
pub mod frontend;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod gradient;
pub mod harris;
pub mod histeq;
pub mod image;
pub mod klt;
pub mod nms;
pub mod occupancy;
pub mod pyramid;
pub mod rectify;
pub mod rigid_ransac;
pub mod sgbm;
pub mod shi_tomasi;
pub mod stereo;
#[cfg(unix)]
pub mod stereo_net;
#[cfg(unix)]
pub mod tensorrt;

pub use camera_geometry;
