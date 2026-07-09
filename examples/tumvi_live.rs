// examples/tumvi_live.rs
//
// TUM-VI live frontend viewer.
//
// Usage:
//   cargo run --release --example tumvi_live
//   cargo run --release --example tumvi_live -- target/tumvi/dataset-calib-cam1_512_16 200
//
// This reuses euroc_live.rs. When invoked as `tumvi_live` with no dataset
// argument, euroc_live defaults to the downloaded TUM-VI calibration sequence:
// target/tumvi/dataset-calib-cam1_512_16

include!("euroc_live.rs");
