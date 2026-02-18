// gpu/mod.rs — GPU acceleration layer (Phase 2).
//
// This module provides wgpu-based compute kernels that mirror the CPU
// algorithms in the parent crate. The CPU implementations in each sibling
// module remain the authoritative reference — every GPU kernel is validated
// against them pixel-for-pixel.
//
// Architecture: Hybrid CPU/GPU model.
//
//   GPU handles all heavy compute WITHIN a frame:
//     image upload → pyramid construction → FAST/Harris → KLT iterations
//
//   CPU handles orchestration BETWEEN frames:
//     RANSAC outlier rejection → occupancy grid → replenishment decisions
//
// The boundary is a small readback of tracked (x, y) positions after KLT.
// RANSAC requires feature positions on CPU every frame, making a fully
// stateful GPU design pointless — the readback is mandatory regardless.
//
// See README.md Phase 2 for the full architecture diagram.

pub mod device;
