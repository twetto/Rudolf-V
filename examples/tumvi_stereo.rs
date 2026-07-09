// examples/tumvi_stereo.rs — Stereo matching on TUM-VI/Kalibr camchain datasets.
//
// Defaults to the locally downloaded calibration sequence:
//     target/tumvi/dataset-calib-cam1_512_16
//
// Usage:
//     cargo run --release --example tumvi_stereo
//     cargo run --release --example tumvi_stereo -- /path/to/dataset-calib-cam1_512_16 200

include!("euroc_stereo.rs");
