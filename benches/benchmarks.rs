// benches/benchmarks.rs -- Per-stage and full-pipeline benchmarks.
//
// Synthetic benchmarks (always run):
//   cargo bench
//
// With real EuRoC data (set EUROC_PATH to dataset root):
//   EUROC_PATH=/path/to/MH_01_easy cargo bench
//
// The EuRoC benchmark loads the first 50 frames and runs the full
// frontend pipeline with the euroc_live configuration.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::essential::{self, Correspondence, RansacConfig};
use rudolf_v::fast::FastDetector;
use rudolf_v::frontend::{Frontend, FrontendConfig};
use rudolf_v::histeq::{self, HistEqMethod};
use rudolf_v::image::Image;
use rudolf_v::klt::{KltTracker, LkMethod};
use rudolf_v::nms::OccupancyNms;
use rudolf_v::pyramid::Pyramid;

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ============================================================
// Helpers
// ============================================================

/// Generate a synthetic test image with texture (rectangles + gradients).
fn make_scene(w: usize, h: usize, dx: usize, dy: usize) -> Image<u8> {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = x + dx;
            let sy = y + dy;
            let base = ((sx * 200 / w) + (sy * 55 / h)) as u8;
            img.set(x, y, base);
        }
    }
    for rect in 0..6 {
        let rx = ((50 + rect * 100) as usize).wrapping_add(dx) % w;
        let ry = ((40 + (rect % 3) * 120) as usize).wrapping_add(dy) % h;
        let bright = 180u8.wrapping_add(rect as u8 * 10);
        for y in ry..(ry + 60).min(h) {
            for x in rx..(rx + 80).min(w) {
                img.set(x, y, bright);
            }
        }
    }
    img
}

// ============================================================
// Per-stage benchmarks (synthetic, always runnable)
// ============================================================

fn bench_histeq(c: &mut Criterion) {
    let img = make_scene(752, 480, 0, 0);

    let mut group = c.benchmark_group("histeq");
    group.bench_function("global_752x480", |b| {
        b.iter(|| histeq::equalize_histogram(&img))
    });
    group.bench_function("clahe_752x480_t32", |b| {
        b.iter(|| histeq::equalize_clahe(&img, 32, 2.0))
    });
    group.finish();
}

fn bench_pyramid(c: &mut Criterion) {
    let img = make_scene(752, 480, 0, 0);

    let mut group = c.benchmark_group("pyramid");
    group.bench_function("build_4level_752x480", |b| {
        b.iter(|| Pyramid::build(&img, 4, 1.0))
    });
    group.finish();
}

fn bench_fast(c: &mut Criterion) {
    let img = make_scene(752, 480, 0, 0);
    let det = FastDetector::new(20, 9);

    let mut group = c.benchmark_group("fast");
    group.bench_function("detect_752x480", |b| {
        b.iter(|| det.detect(&img))
    });
    group.finish();
}

fn bench_klt(c: &mut Criterion) {
    let img1 = make_scene(752, 480, 0, 0);
    let img2 = make_scene(752, 480, 3, 2);

    let pyr1 = Pyramid::build(&img1, 4, 1.0);
    let pyr2 = Pyramid::build(&img2, 4, 1.0);

    // Detect features on first image.
    let det = FastDetector::new(20, 9);
    let features = det.detect(&img1);
    let nms = OccupancyNms::new(128);
    let features: Vec<_> = nms.suppress(&features, 752, 480).into_iter().take(40).collect();

    let mut group = c.benchmark_group("klt");
    group.bench_function(
        BenchmarkId::new("FA", format!("{}feat_4pyr", features.len())),
        |b| {
            let tracker = KltTracker::with_method(11, 30, 0.01, 4, LkMethod::ForwardAdditive);
            b.iter(|| tracker.track(&pyr1, &pyr2, &features))
        },
    );
    group.bench_function(
        BenchmarkId::new("IC", format!("{}feat_4pyr", features.len())),
        |b| {
            let tracker = KltTracker::with_method(11, 30, 0.01, 4, LkMethod::InverseCompositional);
            b.iter(|| tracker.track(&pyr1, &pyr2, &features))
        },
    );
    group.finish();
}

fn bench_ransac(c: &mut Criterion) {
    // Synthetic correspondences for RANSAC benchmarking.
    let n = 40;
    let mut corrs = Vec::new();
    let baseline = 0.1;
    for i in 0..n {
        let z = 2.0 + (i as f64) * 0.3;
        let x = -1.0 + (i as f64) * 0.05;
        let y = -0.5 + (i as f64) * 0.02;
        corrs.push(Correspondence {
            x1: x / z,
            y1: y / z,
            x2: (x - baseline) / z,
            y2: y / z,
        });
    }

    let config = RansacConfig {
        threshold: 1e-5,
        max_iterations: 200,
        confidence: 0.99,
    };

    let mut group = c.benchmark_group("ransac");
    group.bench_function("8pt_40corr", |b| {
        b.iter(|| essential::estimate_essential_ransac(&corrs, &config))
    });
    group.finish();
}

fn bench_frontend_synthetic(c: &mut Criterion) {
    let frames: Vec<Image<u8>> = (0..10)
        .map(|i| make_scene(752, 480, i * 3, i * 2))
        .collect();

    let config = FrontendConfig {
        max_features: 40,
        cell_size: 128,
        pyramid_levels: 4,
        klt_window: 11,
        klt_max_iter: 30,
        klt_method: LkMethod::InverseCompositional,
        histeq: HistEqMethod::Global,
        ..Default::default()
    };

    let mut group = c.benchmark_group("frontend");
    group.bench_function("synthetic_752x480_10frames", |b| {
        b.iter(|| {
            let mut frontend = Frontend::new(config.clone(), 752, 480);
            for frame in &frames {
                frontend.process(frame);
            }
        })
    });
    group.finish();
}

// ============================================================
// EuRoC benchmark (optional, needs EUROC_PATH env var)
// ============================================================

fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}

fn parse_euroc_csv(csv_path: &Path) -> Vec<String> {
    let file = fs::File::open(csv_path).expect("failed to open data.csv");
    let reader = BufReader::new(file);
    let mut filenames = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some((_ts, fname)) = trimmed.split_once(',') {
            filenames.push(fname.trim().to_string());
        }
    }
    filenames
}

fn bench_euroc(c: &mut Criterion) {
    let euroc_path = match env::var("EUROC_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("EUROC_PATH not set, skipping EuRoC benchmark.");
            eprintln!("Set it to run: EUROC_PATH=/path/to/MH_01_easy cargo bench");
            return;
        }
    };

    let cam0_dir = euroc_path.join("mav0").join("cam0");
    let data_dir = cam0_dir.join("data");
    let csv_path = cam0_dir.join("data.csv");

    if !data_dir.exists() {
        eprintln!("EuRoC data dir not found: {}", data_dir.display());
        return;
    }

    let image_files = parse_euroc_csv(&csv_path);
    let num_frames = image_files.len().min(50);

    // Preload frames into memory to bench processing only, not I/O.
    eprintln!("Loading {} EuRoC frames...", num_frames);
    let frames: Vec<Image<u8>> = (0..num_frames)
        .map(|i| load_grayscale(&data_dir.join(&image_files[i])))
        .collect();
    let (img_w, img_h) = (frames[0].width(), frames[0].height());
    eprintln!("EuRoC benchmark ready: {}x{}, {} frames", img_w, img_h, num_frames);

    // Load camera intrinsics if available.
    let sensor_yaml = cam0_dir.join("sensor.yaml");
    let camera = CameraIntrinsics::from_euroc_yaml(&sensor_yaml).ok();

    let config = FrontendConfig {
        max_features: 40,
        cell_size: 128,
        pyramid_levels: 4,
        klt_window: 11,
        klt_max_iter: 30,
        klt_method: LkMethod::InverseCompositional,
        histeq: HistEqMethod::Global,
        camera,
        ransac: RansacConfig {
            threshold: 1e-5,
            max_iterations: 200,
            confidence: 0.99,
        },
        ..Default::default()
    };

    let mut group = c.benchmark_group("euroc");
    group.sample_size(10); // Fewer samples since each iteration processes 50 frames.

    group.bench_function(
        format!("pipeline_{}frames", num_frames),
        |b| {
            b.iter(|| {
                let mut frontend = Frontend::new(config.clone(), img_w, img_h);
                for frame in &frames {
                    frontend.process(frame);
                }
            })
        },
    );

    // Per-stage on a real EuRoC frame.
    let frame0 = &frames[0];
    group.bench_function("histeq_single_frame", |b| {
        b.iter(|| histeq::equalize_histogram(frame0))
    });
    group.bench_function("pyramid_single_frame", |b| {
        b.iter(|| Pyramid::build(frame0, 4, 1.0))
    });

    group.finish();
}

// ============================================================
// Register
// ============================================================

criterion_group!(
    benches,
    bench_histeq,
    bench_pyramid,
    bench_fast,
    bench_klt,
    bench_ransac,
    bench_frontend_synthetic,
    bench_euroc,
);
criterion_main!(benches);
