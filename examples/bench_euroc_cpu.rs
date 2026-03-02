// examples/bench_euroc_cpu.rs — Lightweight CPU frontend benchmark.
//
// Mirrors bench_euroc_gpu.rs but uses the CPU Frontend (ForwardAdditive or
// InverseCompositional KLT, CPU pyramid, CPU FAST+NMS).
//
// Usage:
//     cargo run --example bench_euroc_cpu --release -- /path/to/MH_01_easy 200
//     RUDOLF_FEATURES=50 RUDOLF_KLT=ic cargo run --example bench_euroc_cpu --release -- /path/to/MH_01_easy 200 2>bench.csv
//
// Environment variables (all optional):
//     RUDOLF_FEATURES  — max features          (default: 200)
//     RUDOLF_WINDOW    — KLT half-window       (default: 7)
//     RUDOLF_LEVELS    — pyramid levels        (default: 3)
//     RUDOLF_CELL      — occupancy cell size   (default: 32)
//     RUDOLF_HISTEQ    — "none" / "standard"   (default: none)
//     RUDOLF_KLT       — "fa" / "ic"           (default: ic)

use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::frontend::{Frontend, FrontendConfig, DetectorType, FrameStats};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;

use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// EuRoC image loader (same as GPU benchmark)
// ---------------------------------------------------------------------------

fn load_grayscale(path: &std::path::Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .into_luma8();
    Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
}

fn list_euroc_images(data_dir: &std::path::Path) -> Vec<PathBuf> {
    let cam0 = data_dir.join("mav0/cam0/data");
    if !cam0.is_dir() {
        panic!("Expected EuRoC cam0/data at {}", cam0.display());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&cam0)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "png"))
        .collect();
    files.sort();
    files
}

fn load_camera(data_dir: &std::path::Path) -> Option<CameraIntrinsics> {
    let yaml_path = data_dir.join("mav0/cam0/sensor.yaml");
    if yaml_path.exists() {
        match CameraIntrinsics::from_euroc_yaml(&yaml_path) {
            Ok(cam) => {
                println!("Camera intrinsics loaded from sensor.yaml");
                Some(cam)
            }
            Err(e) => {
                println!("Warning: could not parse sensor.yaml: {e}");
                None
            }
        }
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Env-var config helpers
// ---------------------------------------------------------------------------

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_histeq() -> HistEqMethod {
    match std::env::var("RUDOLF_HISTEQ").as_deref() {
        Ok("standard") => HistEqMethod::Global,
        _ => HistEqMethod::None,
    }
}

fn env_klt_method() -> LkMethod {
    match std::env::var("RUDOLF_KLT").as_deref() {
        Ok("fa") => LkMethod::ForwardAdditive,
        _ => LkMethod::InverseCompositional,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_euroc_cpu <euroc_dataset_path> [num_frames]");
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // Discover and load images.
    let image_files = list_euroc_images(&data_dir);
    let num_frames = image_files.len().min(max_frames);
    println!("Loading {num_frames} frames from {}...", data_dir.display());

    let frames: Vec<Image<u8>> = image_files[..num_frames]
        .iter()
        .map(|f| load_grayscale(f))
        .collect();

    println!("Resolution: {}×{}, {num_frames} frames loaded",
        frames[0].width(), frames[0].height());

    // Config from env vars.
    let max_features = env_usize("RUDOLF_FEATURES", 200);
    let klt_window   = env_usize("RUDOLF_WINDOW", 7);
    let pyr_levels   = env_usize("RUDOLF_LEVELS", 3);
    let cell_size    = env_usize("RUDOLF_CELL", 32);
    let histeq       = env_histeq();
    let klt_method   = env_klt_method();
    let camera       = load_camera(&data_dir);

    let config = FrontendConfig {
        detector: DetectorType::Fast,
        fast_threshold: 40,
        fast_arc_length: 9,
        max_features,
        cell_size,
        pyramid_levels: pyr_levels,
        pyramid_sigma: 1.0,
        klt_window,
        klt_max_iter: 30,
        klt_epsilon: 0.01,
        klt_method,
        histeq,
        camera,
        ..FrontendConfig::default()
    };

    println!("Config: features={max_features} cell={cell_size} levels={pyr_levels} \
              window={klt_window} histeq={histeq:?} klt={klt_method:?}");

    let w = frames[0].width();
    let h = frames[0].height();
    let mut frontend = Frontend::new(config, w, h);

    // Warmup.
    let warmup = 5.min(num_frames);
    println!("Warmup: {warmup} frames");
    for frame in &frames[..warmup] {
        frontend.process(frame);
    }

    // CSV header on stderr.
    eprintln!("frame,tracked,lost,rejected,new,total,histeq_ms,pyramid_ms,klt_ms,ransac_ms,detect_ms,total_ms");

    // Benchmark loop.
    println!("\nRunning {num_frames} frames...\n");
    let mut all_stats: Vec<FrameStats> = Vec::with_capacity(num_frames);
    let t_run = Instant::now();

    // Re-create frontend for clean state.
    let config2 = FrontendConfig {
        detector: DetectorType::Fast,
        fast_threshold: 40,
        fast_arc_length: 9,
        max_features,
        cell_size,
        pyramid_levels: pyr_levels,
        pyramid_sigma: 1.0,
        klt_window,
        klt_max_iter: 30,
        klt_epsilon: 0.01,
        klt_method,
        histeq: env_histeq(),
        camera: load_camera(&data_dir),
        ..FrontendConfig::default()
    };
    frontend = Frontend::new(config2, w, h);

    for (i, frame) in frames.iter().enumerate() {
        let (_features, stats) = frontend.process(frame);

        eprintln!("{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            i, stats.tracked, stats.lost, stats.rejected, stats.new_detections, stats.total,
            stats.timing.histeq_ms(), stats.timing.pyramid_ms(), stats.timing.klt_ms(),
            stats.timing.ransac_ms(), stats.timing.detect_ms(), stats.timing.total_ms());

        all_stats.push(stats);
    }

    let elapsed = t_run.elapsed().as_secs_f64();
    let fps = num_frames as f64 / elapsed;

    // Summary table.
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Rudolf-V CPU Benchmark — {} frames in {:.2}s ({:.1} FPS)",
        num_frames, elapsed, fps);
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Per-stage statistics.
    let stages: [(&str, Box<dyn Fn(&FrameStats) -> f32>); 5] = [
        ("histeq",  Box::new(|s: &FrameStats| s.timing.histeq_ms())),
        ("pyramid", Box::new(|s: &FrameStats| s.timing.pyramid_ms())),
        ("klt",     Box::new(|s: &FrameStats| s.timing.klt_ms())),
        ("ransac",  Box::new(|s: &FrameStats| s.timing.ransac_ms())),
        ("detect",  Box::new(|s: &FrameStats| s.timing.detect_ms())),
    ];

    println!("  {:16} {:>8} {:>8} {:>8} {:>8}", "Stage", "Mean", "Min", "Max", "Median");
    println!("  {:16} {:>8} {:>8} {:>8} {:>8}", "─────", "────", "───", "───", "──────");

    let mut total_mean = 0.0f32;
    let mut total_min  = f32::MAX;
    let mut total_max  = 0.0f32;
    let mut total_vals: Vec<f32> = all_stats.iter().map(|s| s.timing.total_ms()).collect();
    total_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (name, extract) in &stages {
        let mut vals: Vec<f32> = all_stats.iter().map(|s| extract(s)).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        let min = vals[0];
        let max = *vals.last().unwrap();
        let median = vals[vals.len() / 2];
        println!("  {:16} {:>7.2}ms {:>7.2}ms {:>7.2}ms {:>7.2}ms",
            name, mean, min, max, median);

        if *name != "histeq" { // Track non-histeq for total reference.
            total_mean += mean;
        }
    }

    let t_mean: f32 = total_vals.iter().sum::<f32>() / total_vals.len() as f32;
    let t_min  = total_vals[0];
    let t_max  = *total_vals.last().unwrap();
    let t_med  = total_vals[total_vals.len() / 2];

    println!("  {:16} {:>8} {:>8} {:>8} {:>8}", "", "────", "───", "───", "──────");
    println!("  {:16} {:>7.2}ms {:>7.2}ms {:>7.2}ms {:>7.2}ms",
        "TOTAL", t_mean, t_min, t_max, t_med);

    println!();
    println!("  Per-frame CSV written to stderr. Redirect with:");
    println!("    cargo run --example bench_euroc_cpu --release -- <path> 2>bench.csv");
    println!();
}
