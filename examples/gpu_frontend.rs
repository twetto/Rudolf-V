// examples/gpu_frontend.rs — End-to-end GPU visual frontend pipeline.
//
// Chains the full Rudolf-V GPU frontend in the order a VIO system would run it:
//
//   Frame(t)   → GpuPyramidPipeline → GpuFastDetector → OccupancyNms (CPU)
//   Frame(t+1) → GpuPyramidPipeline → GpuKltTracker → tracked positions
//
// Output: live minifb window with two panels:
//
//   ┌───────────────────────────┬───────────────────────────┐
//   │ Frame t                   │ Frame t+1                 │
//   │ • FAST corners (cyan)     │ → tracked (green arrow)   │
//   │                           │ ✕ lost    (red cross)     │
//   └───────────────────────────┴───────────────────────────┘
//
// MODES
// ─────
//   cargo run --example gpu_frontend
//       Synthetic animated scene: rectangles that drift slowly.
//       Re-detects every N frames to refresh features.
//       Demonstrates the full real-time loop.
//
//   cargo run --example gpu_frontend -- frame1.png frame2.png
//       Load two images from disk. Detects in frame1, tracks into frame2.
//       Static display (press Escape to quit).
//
//   cargo run --example gpu_frontend -- image.png
//       Zero-motion sanity check: uses the same image as both frames.
//       All features should track with ~0 displacement.

use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};

use rudolf_v::fast::{FastDetector, Feature};
use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::fast::GpuFastDetector;
use rudolf_v::gpu::klt::GpuKltTracker;
use rudolf_v::gpu::pyramid::GpuPyramidPipeline;
use rudolf_v::image::Image;
use rudolf_v::klt::TrackStatus;
use rudolf_v::nms::OccupancyNms;

// ---------------------------------------------------------------------------
// Pipeline parameters
// ---------------------------------------------------------------------------

const FAST_THRESHOLD: u8    = 20;
const FAST_ARC:       usize = 9;
const NMS_CELL:       usize = 16;
const PYR_LEVELS:     usize = 3;
const PYR_SIGMA:      f64   = 1.0;
const KLT_WIN:        usize = 7;
const KLT_MAX_ITER:   usize = 30;
const KLT_EPSILON:    f32   = 0.01;

/// Redetect features every this many frames in synthetic mode.
const REDETECT_INTERVAL: usize = 30;

/// Display scale: pixels per image pixel.
const SCALE: usize = 3;

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --- GPU init ---
    eprintln!("[frontend] initialising GPU...");
    let gpu = GpuDevice::new().expect("no Vulkan GPU found");
    eprintln!("[frontend] {}", gpu.adapter_info);

    let pyr_pipeline = GpuPyramidPipeline::new(&gpu);
    let gpu_fast     = GpuFastDetector::new(&gpu, FAST_THRESHOLD, FAST_ARC);
    let mut gpu_klt  = GpuKltTracker::new(&gpu, KLT_WIN, KLT_MAX_ITER, KLT_EPSILON, PYR_LEVELS, 512);
    let nms          = OccupancyNms::new(NMS_CELL);

    match args.len() {
        // ---- Synthetic animated mode ----
        1 => run_synthetic(&gpu, &pyr_pipeline, &gpu_fast, &mut gpu_klt, &nms),

        // ---- Single image: zero-motion sanity check ----
        2 => {
            let img = load_image(&args[1]);
            eprintln!("[frontend] zero-motion mode: {}×{}", img.width(), img.height());
            run_static(&gpu, &pyr_pipeline, &gpu_fast, &mut gpu_klt, &nms, &img, &img);
        }

        // ---- Two images: detect in frame1, track into frame2 ----
        _ => {
            let img1 = load_image(&args[1]);
            let img2 = load_image(&args[2]);
            assert_eq!((img1.width(), img1.height()), (img2.width(), img2.height()),
                "images must be the same size");
            eprintln!("[frontend] two-frame mode: {}×{}", img1.width(), img1.height());
            run_static(&gpu, &pyr_pipeline, &gpu_fast, &mut gpu_klt, &nms, &img1, &img2);
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic animated mode
// ---------------------------------------------------------------------------

fn run_synthetic(
    gpu:          &GpuDevice,
    pyr_pipeline: &GpuPyramidPipeline,
    gpu_fast:     &GpuFastDetector,
    gpu_klt:      &mut GpuKltTracker,
    nms:          &OccupancyNms,
) {
    let w = 240usize;
    let h = 180usize;

    let (win_w, win_h) = (w * SCALE * 2 + 4, h * SCALE + 40);
    let mut window = Window::new(
        "Rudolf-V GPU frontend — synthetic (Esc to quit)",
        win_w, win_h,
        WindowOptions { resize: false, ..Default::default() },
    ).expect("window creation failed");
    window.limit_update_rate(Some(Duration::from_millis(33)));

    let mut fb = vec![0u32; win_w * win_h];
    let mut frame_idx = 0usize;
    let mut features: Vec<Feature> = Vec::new();

    // Keep the previous pyramid so KLT has something to track from.
    let mut prev_pyr = None;

    // Shift state for the animated scene.
    let mut shift_x = 0.0f32;
    let mut shift_y = 0.0f32;
    let vx = 0.7f32; // pixels per frame
    let vy = 0.3f32;

    eprintln!("[frontend] starting synthetic loop — Esc to quit");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t_frame = Instant::now();

        // Generate current and next frames.
        let frame_t   = make_scene(w, h, shift_x,        shift_y);
        let frame_t1  = make_scene(w, h, shift_x + vx,   shift_y + vy);

        // --- GPU pyramids ---
        let t0 = Instant::now();
        let pyr_t  = pyr_pipeline.build(gpu, &frame_t,  PYR_LEVELS, PYR_SIGMA as f32);
        let pyr_t1 = pyr_pipeline.build(gpu, &frame_t1, PYR_LEVELS, PYR_SIGMA as f32);
        let pyr_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // --- Redetect features periodically ---
        let redetect = frame_idx % REDETECT_INTERVAL == 0 || features.is_empty();
        let detect_ms;
        if redetect {
            let t0 = Instant::now();
            let mut raw = gpu_fast.detect(gpu, &pyr_t.levels[0]);
            raw.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            features = nms.suppress(&raw, w, h);
            detect_ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[frontend] frame {frame_idx}: redetected {} corners", features.len());
        } else {
            detect_ms = 0.0;
        }

        // --- GPU KLT ---
        let t0 = Instant::now();
        let tracked = if let Some(ref pp) = prev_pyr {
            gpu_klt.track(gpu, pp, &pyr_t1, &features)
        } else {
            Vec::new()
        };
        let klt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let n_tracked = tracked.iter().filter(|r| r.status == TrackStatus::Tracked).count();

        // --- Render ---
        render_panels(&mut fb, win_w,
            &frame_t, &frame_t1,
            &features, &tracked,
            frame_idx, pyr_ms, detect_ms, klt_ms, n_tracked);

        window.update_with_buffer(&fb, win_w, win_h).unwrap();

        // Advance state.
        // Update features to tracked positions for next frame.
        if !tracked.is_empty() {
            features = tracked.into_iter()
                .filter(|r| r.status == TrackStatus::Tracked)
                .map(|r| r.feature)
                .collect();
        }
        prev_pyr = Some(pyr_t1);
        shift_x += vx;
        shift_y += vy;
        frame_idx += 1;

        let frame_ms = t_frame.elapsed().as_secs_f64() * 1000.0;
        eprint!("\r[frontend] frame {frame_idx:4}  pyr={pyr_ms:.1}ms  \
            detect={detect_ms:.1}ms  klt={klt_ms:.1}ms  total={frame_ms:.1}ms  \
            tracked={n_tracked:3}    ");
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Static two-frame mode
// ---------------------------------------------------------------------------

fn run_static(
    gpu:          &GpuDevice,
    pyr_pipeline: &GpuPyramidPipeline,
    gpu_fast:     &GpuFastDetector,
    gpu_klt:      &mut GpuKltTracker,
    nms:          &OccupancyNms,
    frame1:       &Image<u8>,
    frame2:       &Image<u8>,
) {
    let w = frame1.width();
    let h = frame1.height();

    // --- GPU pyramids ---
    let t0 = Instant::now();
    let pyr1 = pyr_pipeline.build(gpu, frame1, PYR_LEVELS, PYR_SIGMA as f32);
    let pyr2 = pyr_pipeline.build(gpu, frame2, PYR_LEVELS, PYR_SIGMA as f32);
    let pyr_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- FAST + NMS ---
    let t0 = Instant::now();
    let mut raw = gpu_fast.detect(gpu, &pyr1.levels[0]);
    raw.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let features = nms.suppress(&raw, w, h);
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- GPU KLT ---
    let t0 = Instant::now();
    let tracked = gpu_klt.track(gpu, &pyr1, &pyr2, &features);
    let klt_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let n_tracked = tracked.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    let n_lost    = tracked.iter().filter(|r| r.status == TrackStatus::Lost).count();
    let n_oob     = tracked.iter().filter(|r| r.status == TrackStatus::OutOfBounds).count();

    eprintln!("[frontend] pyr={pyr_ms:.1}ms  detect={detect_ms:.1}ms  klt={klt_ms:.1}ms");
    eprintln!("[frontend] {}/{} tracked  {}/{} lost  {}/{} out-of-bounds",
        n_tracked, features.len(), n_lost, features.len(), n_oob, features.len());

    // --- Display ---
    // For large images, cap the scale so the window fits on screen.
    let scale = if w > 640 || h > 480 { 1 } else { SCALE };
    let win_w = w * scale * 2 + 4;
    let win_h = h * scale + 40;
    let mut fb = vec![0u32; win_w * win_h];
    let mut window = Window::new(
        "Rudolf-V GPU frontend (Esc to quit)",
        win_w, win_h,
        WindowOptions { resize: false, ..Default::default() },
    ).expect("window failed");
    window.limit_update_rate(Some(Duration::from_millis(16)));

    // Render once with scale as runtime parameter.
    render_panels_scaled(&mut fb, win_w, scale,
        frame1, frame2, &features, &tracked,
        0, pyr_ms, detect_ms, klt_ms, n_tracked);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&fb, win_w, win_h).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Framebuffer rendering (minifb, ARGB u32)
// ---------------------------------------------------------------------------

fn render_panels(
    fb: &mut Vec<u32>, win_w: usize,
    frame1: &Image<u8>, frame2: &Image<u8>,
    features: &[Feature],
    tracked: &[rudolf_v::klt::TrackedFeature],
    frame_idx: usize,
    pyr_ms: f64, detect_ms: f64, klt_ms: f64, n_tracked: usize,
) {
    render_panels_scaled(fb, win_w, SCALE,
        frame1, frame2, features, tracked,
        frame_idx, pyr_ms, detect_ms, klt_ms, n_tracked);
}

fn render_panels_scaled(
    fb: &mut Vec<u32>, win_w: usize, scale: usize,
    frame1: &Image<u8>, frame2: &Image<u8>,
    features: &[Feature],
    tracked: &[rudolf_v::klt::TrackedFeature],
    _frame_idx: usize,
    _pyr_ms: f64, _detect_ms: f64, _klt_ms: f64, _n_tracked: usize,
) {
    let w = frame1.width();
    let h = frame1.height();
    let sw = w * scale;
    let sh = h * scale;
    let panel_gap = 4usize;
    let top_pad = 20usize;

    // Clear.
    fb.fill(0xFF1A1A2E); // dark navy background

    // --- Left panel: frame 1 + detected corners ---
    for y in 0..h {
        for x in 0..w {
            let v = frame1.get(x, y);
            let px = (top_pad + y * scale) * win_w + x * scale;
            for dy in 0..scale {
                for dx in 0..scale {
                    fb[px + dy * win_w + dx] = grey(v);
                }
            }
        }
    }
    // Detected corners: cyan circles.
    for f in features {
        let cx = (f.x as usize).min(w.saturating_sub(1)) * scale + scale / 2;
        let cy = top_pad + (f.y as usize).min(h.saturating_sub(1)) * scale + scale / 2;
        draw_circle(fb, win_w, cx, cy, scale + 1, 0xFF00FFFF);
        set_pixel(fb, win_w, cx, cy, 0xFF00FFFF);
    }

    // --- Right panel: frame 2 + flow arrows ---
    let x_off = sw + panel_gap;
    for y in 0..h {
        for x in 0..w {
            let v = frame2.get(x, y);
            let px = (top_pad + y * scale) * win_w + x_off + x * scale;
            for dy in 0..scale {
                for dx in 0..scale {
                    fb[px + dy * win_w + dx] = grey(v);
                }
            }
        }
    }
    // Flow arrows.
    for (feat, result) in features.iter().zip(tracked.iter()) {
        let x0 = x_off + (feat.x as usize).min(w.saturating_sub(1)) * scale + scale / 2;
        let y0 = top_pad + (feat.y as usize).min(h.saturating_sub(1)) * scale + scale / 2;

        match result.status {
            TrackStatus::Tracked => {
                let x1 = x_off + (result.feature.x as usize).min(w.saturating_sub(1)) * scale + scale / 2;
                let y1 = top_pad + (result.feature.y as usize).min(h.saturating_sub(1)) * scale + scale / 2;
                draw_line(fb, win_w, x0, y0, x1, y1, 0xFF44FF88);
                draw_circle(fb, win_w, x1, y1, scale.max(2), 0xFF44FF88);
            }
            TrackStatus::Lost | TrackStatus::OutOfBounds => {
                let r = (scale + 1).max(3);
                draw_cross(fb, win_w, x0, y0, r, 0xFFFF4444);
            }
        }
    }

    // Status bar at bottom.
    let n_tracked = tracked.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    let _ = (sw, sh, n_tracked); // suppress unused warnings in non-print path
}

// ---------------------------------------------------------------------------
// Primitive rasterisers (integer Bresenham — no floats, no deps)
// ---------------------------------------------------------------------------

fn grey(v: u8) -> u32 {
    let c = v as u32;
    0xFF000000 | (c << 16) | (c << 8) | c
}

fn set_pixel(fb: &mut Vec<u32>, w: usize, x: usize, y: usize, color: u32) {
    if y * w + x < fb.len() {
        fb[y * w + x] = color;
    }
}

fn draw_circle(fb: &mut Vec<u32>, w: usize, cx: usize, cy: usize, r: usize, color: u32) {
    // Midpoint circle algorithm — outline only.
    let r = r as isize;
    let mut x = 0isize;
    let mut y = r;
    let mut d = 1 - r;
    while x <= y {
        for &(ox, oy) in &[
            ( x,  y), (-x,  y), ( x, -y), (-x, -y),
            ( y,  x), (-y,  x), ( y, -x), (-y, -x),
        ] {
            let px = cx as isize + ox;
            let py = cy as isize + oy;
            if px >= 0 && py >= 0 {
                set_pixel(fb, w, px as usize, py as usize, color);
            }
        }
        x += 1;
        if d < 0 { d += 2 * x + 1; } else { y -= 1; d += 2 * (x - y) + 1; }
    }
}

fn draw_line(fb: &mut Vec<u32>, w: usize, x0: usize, y0: usize, x1: usize, y1: usize, color: u32) {
    // Bresenham line.
    let (mut x0, mut y0) = (x0 as isize, y0 as isize);
    let (x1, y1) = (x1 as isize, y1 as isize);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: isize = if x0 < x1 { 1 } else { -1 };
    let sy: isize = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 { set_pixel(fb, w, x0 as usize, y0 as usize, color); }
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { if x0 == x1 { break; } err += dy; x0 += sx; }
        if e2 <= dx { if y0 == y1 { break; } err += dx; y0 += sy; }
    }
}

fn draw_cross(fb: &mut Vec<u32>, w: usize, cx: usize, cy: usize, r: usize, color: u32) {
    let r = r as isize;
    let cx = cx as isize;
    let cy = cy as isize;
    for d in -r..=r {
        let x = cx + d;
        let y = cy + d;
        let ax = cx - d;
        if x >= 0 && cy >= 0 { set_pixel(fb, w, x as usize, cy as usize, color); }
        if cx >= 0 && y >= 0 { set_pixel(fb, w, cx as usize, y as usize, color); }
        if ax >= 0 && cy >= 0 { set_pixel(fb, w, ax as usize, cy as usize, color); }
    }
}

// ---------------------------------------------------------------------------
// Synthetic scene generator
// ---------------------------------------------------------------------------

/// Generate a frame with several bright rectangles on a dark background,
/// shifted by (shift_x, shift_y) to simulate camera motion.
fn make_scene(w: usize, h: usize, shift_x: f32, shift_y: f32) -> Image<u8> {
    let mut img = Image::from_vec(w, h, vec![18u8; w * h]);

    // Rectangle definitions: (base_x, base_y, width, height, brightness)
    let rects: &[(isize, isize, usize, usize, u8)] = &[
        (20,  15, 30, 25, 210),
        (80,  10, 40, 20, 190),
        (150, 20, 25, 35, 220),
        (18,  70, 20, 30, 200),
        (70,  60, 35, 25, 185),
        (135, 65, 30, 30, 205),
        (40,  120, 45, 20, 195),
        (120, 110, 30, 35, 215),
        (180, 80,  25, 40, 200),
    ];

    for &(bx, by, rw, rh, brightness) in rects {
        let rx = (bx + shift_x.round() as isize).max(0) as usize;
        let ry = (by + shift_y.round() as isize).max(0) as usize;
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                img.set(x, y, brightness);
            }
        }
    }
    img
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

fn load_image(path: &str) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .to_luma8();
    let (w, h) = img.dimensions();
    Image::<u8>::from_vec(w as usize, h as usize, img.into_raw())
}
