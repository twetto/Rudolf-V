// benches/gpu_benchmarks.rs — GPU pipeline benchmarks.
//
// Mirrors benchmarks.rs structure. Each CPU stage has a corresponding GPU
// benchmark in the same group for direct comparison.
//
// Synthetic benchmarks (always run):
//   cargo bench --bench gpu_benchmarks
//
// With real EuRoC data:
//   EUROC_PATH=/path/to/MH_01_easy cargo bench --bench gpu_benchmarks
//
//
// CRITERION + GPU CAVEATS
// ────────────────────────
// Criterion measures wall time including CPU overhead (buffer writes, bind
// group creation, submit, poll). GPU shader execution is included in poll().
// This is the right metric for VIO: the frontend blocks on tracking results
// before the next stage can run.
//
// Criterion's warmup matters here: the first few iterations pay shader JIT
// costs (wgpu compiles pipelines lazily on some drivers). We set warmup_time
// explicitly to ensure stable measurements.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use rudolf_v::fast::FastDetector;
use rudolf_v::frontend::{Frontend, FrontendConfig};
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::fast::GpuFastDetector;
use rudolf_v::gpu::frontend::{GpuFrontend, GpuFrontendConfig};
use rudolf_v::gpu::klt::GpuKltTracker;
use rudolf_v::gpu::pyramid::GpuPyramidPipeline;
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::{KltTracker, LkMethod};
use rudolf_v::nms::OccupancyNms;
use rudolf_v::pyramid::Pyramid;

// ============================================================
// Shared helpers
// ============================================================

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
// Pyramid: CPU vs GPU
// ============================================================

fn bench_pyramid(c: &mut Criterion) {
    let img = make_scene(752, 480, 0, 0);

    let gpu = GpuDevice::new().expect("no Vulkan GPU");
    let pyr_pipeline = GpuPyramidPipeline::new(&gpu);

    let mut group = c.benchmark_group("pyramid");
    group.warm_up_time(Duration::from_secs(2));

    group.bench_function("cpu_4level_752x480", |b| {
        b.iter(|| Pyramid::build(&img, 4, 1.0))
    });

    group.bench_function("gpu_4level_752x480", |b| {
        b.iter(|| pyr_pipeline.build(&gpu, &img, 4, 1.0))
    });

    group.finish();
}

// ============================================================
// FAST: CPU vs GPU
// ============================================================

fn bench_fast(c: &mut Criterion) {
    let img = make_scene(752, 480, 0, 0);

    let gpu = GpuDevice::new().expect("no Vulkan GPU");
    let pyr_pipeline = GpuPyramidPipeline::new(&gpu);
    let pyr = pyr_pipeline.build(&gpu, &img, 1, 1.0);
    let gpu_fast = GpuFastDetector::new(&gpu, 20, 9);
    let cpu_fast = FastDetector::new(20, 9);

    let mut group = c.benchmark_group("fast");
    group.warm_up_time(Duration::from_secs(2));

    group.bench_function("cpu_detect_752x480", |b| {
        b.iter(|| cpu_fast.detect(&img))
    });

    group.bench_function("gpu_detect_752x480", |b| {
        b.iter(|| gpu_fast.detect(&gpu, &pyr.levels[0]))
    });

    group.finish();
}

// ============================================================
// KLT: CPU vs GPU, varying feature count
// ============================================================

fn bench_klt(c: &mut Criterion) {
    let img1 = make_scene(752, 480, 0, 0);
    let img2 = make_scene(752, 480, 3, 2);

    // CPU pyramids
    let cpu_pyr1 = Pyramid::build(&img1, 4, 1.0);
    let cpu_pyr2 = Pyramid::build(&img2, 4, 1.0);

    // GPU setup
    let gpu = GpuDevice::new().expect("no Vulkan GPU");
    let pyr_pipeline = GpuPyramidPipeline::new(&gpu);
    let gpu_pyr1 = pyr_pipeline.build(&gpu, &img1, 4, 1.0);
    let gpu_pyr2 = pyr_pipeline.build(&gpu, &img2, 4, 1.0);

    // Detect features at different densities
    let cpu_fast = FastDetector::new(20, 9);
    let all_features = cpu_fast.detect(&img1);
    let nms = OccupancyNms::new(128);
    let features_40: Vec<_> = nms.suppress(&all_features, 752, 480)
        .into_iter().take(40).collect();
    let nms2 = OccupancyNms::new(64);
    let features_150: Vec<_> = nms2.suppress(&all_features, 752, 480)
        .into_iter().take(150).collect();

    let mut group = c.benchmark_group("klt");
    group.warm_up_time(Duration::from_secs(2));

    for (label, features) in [("40feat", &features_40), ("150feat", &features_150)] {
        let n = features.len();

        // CPU IC (same method GPU uses)
        group.bench_with_input(
            BenchmarkId::new("cpu_IC_4pyr", format!("{n}feat")),
            features,
            |b, feats| {
                let tracker = KltTracker::with_method(7, 30, 0.01, 4, LkMethod::InverseCompositional);
                b.iter(|| tracker.track(&cpu_pyr1, &cpu_pyr2, feats))
            },
        );

        // GPU IC
        group.bench_with_input(
            BenchmarkId::new("gpu_IC_4pyr", format!("{n}feat")),
            features,
            |b, feats| {
                let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 4, 256);
                b.iter(|| tracker.track(&gpu, &gpu_pyr1, &gpu_pyr2, feats))
            },
        );
    }

    group.finish();
}

// ============================================================
// Full frontend pipeline: CPU vs GPU
// ============================================================

fn bench_frontend(c: &mut Criterion) {
    let frames: Vec<Image<u8>> = (0..10)
        .map(|i| make_scene(752, 480, i * 3, i * 2))
        .collect();

    let gpu = GpuDevice::new().expect("no Vulkan GPU");

    let cpu_config = FrontendConfig {
        max_features:   40,
        cell_size:      128,
        pyramid_levels: 4,
        klt_window:     7,
        klt_max_iter:   30,
        klt_method:     LkMethod::InverseCompositional,
        histeq:         HistEqMethod::Global,
        ..Default::default()
    };

    let gpu_config = GpuFrontendConfig {
        max_features:   40,
        cell_size:      128,
        pyramid_levels: 4,
        klt_window:     7,
        klt_max_iter:   30,
        histeq:         HistEqMethod::Global,
        ..Default::default()
    };

    let mut group = c.benchmark_group("frontend");
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(20);

    group.bench_function("cpu_synthetic_752x480_10frames", |b| {
        b.iter(|| {
            let mut fe = Frontend::new(cpu_config.clone(), 752, 480);
            for frame in &frames { fe.process(frame); }
        })
    });

    group.bench_function("gpu_synthetic_752x480_10frames", |b| {
        b.iter(|| {
            let mut fe = GpuFrontend::new(&gpu, gpu_config.clone(), 752, 480);
            for frame in &frames { fe.process(&gpu, frame); }
        })
    });

    group.finish();
}

// ============================================================
// KLT allocation overhead: new() vs reuse
// ============================================================
// Shows exactly how much the pre-allocated buffer reuse saves vs
// the old approach of constructing a fresh GpuKltTracker each frame.

fn bench_klt_allocation(c: &mut Criterion) {
    let img1 = make_scene(752, 480, 0, 0);
    let img2 = make_scene(752, 480, 3, 2);

    let gpu = GpuDevice::new().expect("no Vulkan GPU");
    let pyr_pipeline = GpuPyramidPipeline::new(&gpu);
    let gpu_pyr1 = pyr_pipeline.build(&gpu, &img1, 4, 1.0);
    let gpu_pyr2 = pyr_pipeline.build(&gpu, &img2, 4, 1.0);

    let cpu_fast = FastDetector::new(20, 9);
    let raw = cpu_fast.detect(&img1);
    let nms = OccupancyNms::new(128);
    let features: Vec<_> = nms.suppress(&raw, 752, 480).into_iter().take(40).collect();

    let mut group = c.benchmark_group("klt_alloc");
    group.warm_up_time(Duration::from_secs(2));

    // Recreate tracker each iteration — pay new() cost every frame (old behaviour)
    group.bench_function("new_per_call", |b| {
        b.iter(|| {
            let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 4, 256);
            tracker.track(&gpu, &gpu_pyr1, &gpu_pyr2, &features)
        })
    });

    // Reuse pre-allocated buffers (current behaviour)
    group.bench_function("reuse_buffers", |b| {
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 4, 256);
        b.iter(|| tracker.track(&gpu, &gpu_pyr1, &gpu_pyr2, &features))
    });

    group.finish();
}

// ============================================================
// EuRoC benchmark (optional)
// ============================================================

fn bench_euroc(c: &mut Criterion) {
    use std::env;
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;

    let euroc_path = match env::var("EUROC_PATH") {
        Ok(p)  => PathBuf::from(p),
        Err(_) => {
            eprintln!("EUROC_PATH not set, skipping EuRoC GPU benchmark.");
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

    // Parse frame list
    let image_files: Vec<String> = {
        let file = fs::File::open(&csv_path).expect("failed to open data.csv");
        BufReader::new(file).lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| l.split_once(',').map(|(_, f)| f.trim().to_string()))
            .collect()
    };
    let num_frames = image_files.len().min(50);

    eprintln!("Loading {} EuRoC frames...", num_frames);
    let frames: Vec<Image<u8>> = (0..num_frames).map(|i| {
        let img = image::open(data_dir.join(&image_files[i])).expect("load failed");
        let gray = img.to_luma8();
        let (w, h) = gray.dimensions();
        Image::from_vec(w as usize, h as usize, gray.into_raw())
    }).collect();

    let (img_w, img_h) = (frames[0].width(), frames[0].height());
    eprintln!("EuRoC GPU benchmark ready: {}×{}, {} frames", img_w, img_h, num_frames);

    let gpu = GpuDevice::new().expect("no Vulkan GPU");

    let cpu_config = FrontendConfig {
        max_features:   40,
        cell_size:      128,
        pyramid_levels: 4,
        klt_window:     7,
        klt_max_iter:   30,
        klt_method:     LkMethod::InverseCompositional,
        histeq:         HistEqMethod::Global,
        ..Default::default()
    };

    let gpu_config = GpuFrontendConfig {
        max_features:   40,
        cell_size:      128,
        pyramid_levels: 4,
        klt_window:     7,
        klt_max_iter:   30,
        histeq:         HistEqMethod::Global,
        ..Default::default()
    };

    let mut group = c.benchmark_group("euroc");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function(format!("cpu_{num_frames}frames"), |b| {
        b.iter(|| {
            let mut fe = Frontend::new(cpu_config.clone(), img_w, img_h);
            for frame in &frames { fe.process(frame); }
        })
    });

    group.bench_function(format!("gpu_{num_frames}frames"), |b| {
        b.iter(|| {
            let mut fe = GpuFrontend::new(&gpu, gpu_config.clone(), img_w, img_h);
            for frame in &frames { fe.process(&gpu, frame); }
        })
    });

    group.finish();
}

// ============================================================
// Register
// ============================================================

criterion_group!(
    benches,
    bench_pyramid,
    bench_fast,
    bench_klt,
    bench_klt_allocation,
    bench_frontend,
    bench_euroc,
);
criterion_main!(benches);
