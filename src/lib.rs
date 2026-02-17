// Rudolf-V: RUst Device-Optimized Library for Frontend Vision
// CPU reference implementation of the vilib visual frontend
//
// Reference: Nagy, Foehn, Scaramuzza — "Faster than FAST: GPU-Accelerated
// Frontend for High-Speed VIO" (IROS 2020)

pub mod image;
pub mod convert;
pub mod convolution;
pub mod pyramid;

// Future steps (uncomment as implemented):
// pub mod fast;          // Step 3: FAST corner detector
// pub mod nms;           // Step 3b: non-maximum suppression
// pub mod gradient;      // Step 4 dep: Sobel gradients
// pub mod harris;        // Step 4: Harris corner detector
// pub mod klt;           // Step 5: KLT / Lucas-Kanade tracker
// pub mod occupancy;     // Step 6: occupancy grid
// pub mod frontend;      // Step 6b: top-level pipeline
