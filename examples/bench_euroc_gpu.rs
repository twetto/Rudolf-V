// examples/bench_euroc_gpu.rs
//
// Lightweight GPU frontend benchmark on EuRoC data.
//
// Unlike `cargo bench` (Criterion), this constructs GpuFrontend once and
// runs N frames in a tight loop — no warmup overhead, no repeated shader
// compilation, and no excess RAM usage. Designed to work on RPi 4 with
// 2 GB RAM where Criterion fails.
//
// Usage:
//   cargo run --example bench_euroc_gpu --release -- /path/to/MH_01_easy
//   cargo run --example bench_euroc_gpu --release -- /path/to/MH_01_easy 100
//
// Output: per-stage timing summary (mean, min, max) plus per-frame CSV on
// stderr for further analysis.
//
// Environment variables:
//   RUDOLF_HISTEQ=global|clahe|none   Override histogram equalization (default: none)
//   RUDOLF_FEATURES=200               Override max_features
//   RUDOLF_WINDOW=7                   Override KLT window half-size
//   RUDOLF_LEVELS=3                   Override pyramid levels
//   RUDOLF_CELL=32                    Override cell size

use rudolf_v::camera::CameraIntrinsics;
use rudolf_v::essential::RansacConfig;
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::frontend::{GpuFrontend, GpuFrontendConfig, SubmitStrategy};
use rudolf_v::histeq::HistEqMethod;
use rudolf_v::image::Image;

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <euroc_dataset_path> [max_frames]", args[0]);
        eprintln!("");
        eprintln!("Environment variables:");
        eprintln!("  RUDOLF_HISTEQ=global|clahe|none");
        eprintln!("  RUDOLF_FEATURES=200");
        eprintln!("  RUDOLF_WINDOW=7");
        eprintln!("  RUDOLF_LEVELS=3");
        eprintln!("  RUDOLF_CELL=32");
        std::process::exit(1);
    }

    let dataset_path = PathBuf::from(&args[1]);
    let max_frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // --- Load dataset ---
    let cam0_dir = dataset_path.join("mav0").join("cam0");
    let data_dir = cam0_dir.join("data");
    let csv_path = cam0_dir.join("data.csv");

    if !data_dir.exists() {
        eprintln!("Error: {}", data_dir.display());
        eprintln!("Expected EuRoC ASL format: <dataset>/mav0/cam0/data/");
        std::process::exit(1);
    }

    let image_files = if csv_path.exists() {
        parse_euroc_csv(&csv_path)
    } else {
        list_png_files(&data_dir)
    };
    let num_frames = image_files.len().min(max_frames);

    // Pre-load all frames into memory to exclude I/O from timing.
    println!("Loading {} frames from {}...", num_frames, dataset_path.display());
    let frames: Vec<Image<u8>> = image_files[..num_frames]
        .iter()
        .map(|f| load_grayscale(&data_dir.join(f)))
        .collect();

    let (img_w, img_h) = (frames[0].width(), frames[0].height());
    println!("Resolution: {}×{}, {} frames loaded", img_w, img_h, num_frames);

    // --- GPU init ---
    println!("Initialising GPU...");
    let gpu = GpuDevice::new().expect("no Vulkan GPU found");
    println!("GPU: {}", gpu.adapter_info);

    // --- Config from env ---
    let histeq = match env::var("RUDOLF_HISTEQ").as_deref() {
        Ok("global") => HistEqMethod::Global,
        Ok("clahe")  => HistEqMethod::Clahe { tile_size: 32, clip_limit: 2.0 },
        _            => HistEqMethod::None,
    };
    let max_features: usize = env::var("RUDOLF_FEATURES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let klt_window: usize = env::var("RUDOLF_WINDOW")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let pyramid_levels: usize = env::var("RUDOLF_LEVELS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let cell_size: usize = env::var("RUDOLF_CELL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    let sensor_yaml = cam0_dir.join("sensor.yaml");
    let camera = CameraIntrinsics::from_euroc_yaml(&sensor_yaml).ok();
    if camera.is_some() {
        println!("Camera intrinsics loaded from sensor.yaml");
    } else {
        println!("No sensor.yaml — geometric verification disabled");
    }

    let config = GpuFrontendConfig {
        submit_strategy: SubmitStrategy::Fused,
        max_features,
        cell_size,
        pyramid_levels,
        klt_window,
        klt_max_iter: 30,
        klt_epsilon:  0.01,
        histeq,
        camera,
        ransac: RansacConfig {
            threshold:      1e-5,
            max_iterations: 200,
            confidence:     0.99,
        },
        ..Default::default()
    };

    println!("Config: features={} cell={} levels={} window={} histeq={:?}",
        max_features, cell_size, pyramid_levels, klt_window, histeq);
    println!("Submit: {:?}  NMS: {:?}", config.submit_strategy, config.nms_strategy);

    let mut frontend = GpuFrontend::new(&gpu, config, img_w, img_h);

    // --- Warmup: 5 frames (or fewer if dataset is tiny) ---
    let warmup = 5.min(num_frames);
    for i in 0..warmup {
        frontend.process(&gpu, &frames[i]);
    }
    frontend.reset();
    println!("Warmup: {} frames", warmup);

    // --- Benchmark ---
    println!("\nRunning {} frames...\n", num_frames);

    // Per-frame CSV header on stderr for piping to file.
    eprintln!("frame,tracked,lost,rejected,new,total,histeq_ms,pyramid_ms,klt_ms,ransac_ms,detect_ms,total_ms");

    struct Accum {
        histeq:  Vec<f64>,
        pyramid: Vec<f64>,
        klt:     Vec<f64>,
        ransac:  Vec<f64>,
        detect:  Vec<f64>,
        total:   Vec<f64>,
    }
    let mut acc = Accum {
        histeq:  Vec::with_capacity(num_frames),
        pyramid: Vec::with_capacity(num_frames),
        klt:     Vec::with_capacity(num_frames),
        ransac:  Vec::with_capacity(num_frames),
        detect:  Vec::with_capacity(num_frames),
        total:   Vec::with_capacity(num_frames),
    };

    let bench_start = std::time::Instant::now();

    for (i, frame) in frames.iter().enumerate() {
        let (_features, stats) = frontend.process(&gpu, frame);
        let t = &stats.timing;

        eprintln!("{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            i, stats.tracked, stats.lost, stats.rejected,
            stats.new_detections, stats.total,
            t.histeq_ms(), t.pyramid_ms(), t.klt_ms(),
            t.ransac_ms(), t.detect_ms(), t.total_ms());

        if i == 1 {
            // Frame 1 is the first frame with tracking
            for (j, f) in _features.iter().take(5).enumerate() {
                eprintln!("  feat[{}] id={} pos=({:.2}, {:.2})", j, f.id, f.x, f.y);
            }
        }

        acc.histeq.push(t.histeq);
        acc.pyramid.push(t.pyramid);
        acc.klt.push(t.klt);
        acc.ransac.push(t.ransac);
        acc.detect.push(t.detect);
        acc.total.push(t.total);
    }

    let bench_elapsed = bench_start.elapsed();

    // --- Summary ---
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Rudolf-V GPU Benchmark — {} frames in {:.2}s ({:.1} FPS)",
        num_frames, bench_elapsed.as_secs_f64(),
        num_frames as f64 / bench_elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════════");
    println!("");
    println!("  {:12} {:>8} {:>8} {:>8} {:>8}",
        "Stage", "Mean", "Min", "Max", "Median");
    println!("  {:12} {:>8} {:>8} {:>8} {:>8}",
        "─────", "────", "───", "───", "──────");

    print_stat("histeq",  &acc.histeq);
    print_stat("pyramid", &acc.pyramid);
    print_stat("klt",     &acc.klt);
    print_stat("ransac",  &acc.ransac);
    print_stat("detect",  &acc.detect);
    println!("  {:12} {:>8} {:>8} {:>8} {:>8}",
        "", "────", "───", "───", "──────");
    print_stat("TOTAL",   &acc.total);

    println!("");
    println!("  Per-frame CSV written to stderr. Redirect with:");
    println!("    cargo run --example bench_euroc_gpu --release -- <path> 2>bench.csv");
    println!("");
}

fn print_stat(label: &str, vals: &[f64]) {
    if vals.is_empty() { return; }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean   = vals.iter().sum::<f64>() / vals.len() as f64;
    let min    = sorted[0];
    let max    = sorted[sorted.len() - 1];
    let median = sorted[sorted.len() / 2];
    println!("  {:12} {:>7.2}ms {:>7.2}ms {:>7.2}ms {:>7.2}ms",
        label,
        mean * 1000.0, min * 1000.0, max * 1000.0, median * 1000.0);
}

// ---------------------------------------------------------------------------
// EuRoC I/O (shared with gpu_euroc_live.rs)
// ---------------------------------------------------------------------------

fn parse_euroc_csv(csv_path: &Path) -> Vec<String> {
    let file = fs::File::open(csv_path).expect("failed to open data.csv");
    let reader = BufReader::new(file);
    let mut filenames = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        if let Some((_ts, fname)) = line.split_once(',') {
            filenames.push(fname.trim().to_string());
        }
    }
    filenames
}

fn list_png_files(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = fs::read_dir(dir)
        .expect("failed to read directory")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".png") { Some(name) } else { None }
        })
        .collect();
    files.sort();
    files
}

fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {e}", path.display()));
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    Image::from_vec(w as usize, h as usize, gray.into_raw())
}
