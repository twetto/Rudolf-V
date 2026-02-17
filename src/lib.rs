// Rudolf-V: RUst Device-Optimized Library for Frontend Vision
// CPU reference implementation of the vilib visual frontend
//
// Reference: Nagy, Foehn, Scaramuzza — "Faster than FAST: GPU-Accelerated
// Frontend for High-Speed VIO" (IROS 2020)

pub mod image;
pub mod convert;
pub mod convolution;
pub mod fast;
pub mod gradient;
pub mod harris;
pub mod klt;
pub mod nms;
pub mod pyramid;

// Future steps (uncomment as implemented):
// pub mod occupancy;     // Step 6: occupancy grid
// pub mod frontend;      // Step 6b: top-level pipeline
