// examples/euroc_stereo.rs — Stereo matching on EuRoC datasets.
//
// Runs the CPU frontend on cam0, then matches tracked features into cam1
// using the 1D inverse-depth Gauss-Newton stereo matcher.
//
// Usage:
//     cargo run --example euroc_stereo --release -- /path/to/V2_01_easy
//     cargo run --example euroc_stereo --release -- /path/to/V2_01_easy 200

use rudolf_v::camera::{CameraIntrinsics, StereoRig};
use rudolf_v::fast::Feature;
use rudolf_v::frontend::{DetectorType, Frontend, FrontendConfig, LbpPolicy};
use rudolf_v::histeq::{self, HistEqMethod};
use rudolf_v::image::Image;
use rudolf_v::klt::LkMethod;
use rudolf_v::rigid_ransac::{self, Correspondence3d, Rigid3dRansacConfig};
use rudolf_v::stereo::{StereoConfig, StereoMatch, StereoMatcher};

use minifb::{Key, Window, WindowOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const VIS_NEAR_DEPTH_M: f64 = 0.5;
const VIS_FAR_DEPTH_M: f64 = 5.0;
const PANEL_GAP: usize = 8;

#[derive(Clone, Copy)]
struct DepthStats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
}

fn load_grayscale(path: &Path) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .into_luma8();
    Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
}

fn list_euroc_images(cam_dir: &Path) -> Vec<PathBuf> {
    let data_dir = cam_dir.join("data");
    if !data_dir.is_dir() {
        panic!("Expected data dir at {}", data_dir.display());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&data_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "png"))
        .collect();
    files.sort();
    files
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: euroc_stereo <euroc_dataset_path> [num_frames]");
        std::process::exit(1);
    }
    let data_dir = PathBuf::from(&args[1]);
    let max_frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let cam0_dir = data_dir.join("mav0/cam0");
    let cam1_dir = data_dir.join("mav0/cam1");

    let rig = StereoRig::from_euroc(&cam0_dir.join("sensor.yaml"), &cam1_dir.join("sensor.yaml"))
        .expect("Failed to load stereo rig from EuRoC sensor.yaml files");

    println!("Stereo rig: baseline = {:.4}m", rig.baseline_meters());
    println!(
        "  cam0: fx={:.1} fy={:.1} cx={:.1} cy={:.1} {}×{}",
        rig.cam0.fx,
        rig.cam0.fy,
        rig.cam0.cx,
        rig.cam0.cy,
        rig.cam0.resolution[0],
        rig.cam0.resolution[1]
    );
    println!(
        "  cam1: fx={:.1} fy={:.1} cx={:.1} cy={:.1} {}×{}",
        rig.cam1.fx,
        rig.cam1.fy,
        rig.cam1.cx,
        rig.cam1.cy,
        rig.cam1.resolution[0],
        rig.cam1.resolution[1]
    );

    let cam0_files = list_euroc_images(&cam0_dir);
    let cam1_files = list_euroc_images(&cam1_dir);
    let num_frames = cam0_files.len().min(cam1_files.len()).min(max_frames);
    println!("Loading {num_frames} stereo pairs...");

    let cam0_frames: Vec<Image<u8>> = cam0_files[..num_frames]
        .iter()
        .map(|f| load_grayscale(f))
        .collect();
    let cam1_frames: Vec<Image<u8>> = cam1_files[..num_frames]
        .iter()
        .map(|f| load_grayscale(f))
        .collect();

    let (w, h) = (cam0_frames[0].width(), cam0_frames[0].height());
    println!("Resolution: {w}×{h}, {num_frames} pairs loaded\n");

    let camera = Some(CameraIntrinsics::from_euroc_yaml(&cam0_dir.join("sensor.yaml")).unwrap());

    let frontend_config = FrontendConfig {
        detector: DetectorType::Fast,
        fast_threshold: 20,
        fast_arc_length: 9,
        max_features: 200,
        cell_size: 32,
        pyramid_levels: 3,
        pyramid_sigma: 1.0,
        klt_window: 7,
        klt_max_iter: 30,
        klt_epsilon: 0.01,
        klt_method: LkMethod::InverseCompositional,
        histeq: HistEqMethod::Global,
        camera,
        lbp_policy: LbpPolicy::HardReject,
        // Defer outlier rejection to the joint 3D-3D RANSAC over stereo pairs
        // below — it has more constraint than cam0-only essential-matrix.
        enable_internal_ransac: false,
        ..FrontendConfig::default()
    };

    let mut frontend = Frontend::new(frontend_config, w, h);

    let stereo_config = StereoConfig {
        pyramid_levels: 3,
        patch_half_size: 4,
        max_iterations: 30,
        n_search_candidates: 0,
        histeq: HistEqMethod::Global,
        ..StereoConfig::default()
    };
    let stereo_histeq = stereo_config.histeq;
    let mut matcher = StereoMatcher::new(rig, stereo_config, w, h);

    let scale = if w <= 400 { 2 } else { 1 };
    let panel_w = w * scale;
    let panel_h = h * scale;
    let win_w = panel_w * 2 + PANEL_GAP;
    let win_h = panel_h;
    let mut window = Window::new(
        "Rudolf-V — EuRoC Stereo Depth",
        win_w,
        win_h,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("failed to create window");
    window.set_target_fps(60);

    let mut fb = vec![0u32; win_w * win_h];
    let mut cam1_display_buf = Image::new(w, h);

    eprintln!("frame,features,matched,match_rate,mean_depth,median_depth,min_depth,max_depth,frontend_ms,stereo_ms");
    println!(
        "Depth colors: {:.1}m red/yellow, {:.1}m blue (metric Z-depth). Controls: Q/Esc=quit",
        VIS_NEAR_DEPTH_M, VIS_FAR_DEPTH_M
    );

    let mut total_frontend_ms = 0.0f64;
    let mut total_stereo_ms = 0.0f64;
    let mut processed_frames = 0usize;

    let rigid_cfg = Rigid3dRansacConfig::default();
    let mut prev_3d: HashMap<u64, [f64; 3]> = HashMap::new();

    for i in 0..num_frames {
        if !window.is_open() || window.is_key_down(Key::Escape) || window.is_key_down(Key::Q) {
            break;
        }

        let (features, _stats) = frontend.process(&cam0_frames[i]);
        let features = features.to_vec();
        let frontend_ms = _stats.timing.total_ms() as f64;

        let t0 = Instant::now();
        let cam0_pyramid = frontend.current_pyramid();
        let mut matches = matcher.match_features(&cam1_frames[i], &features, cam0_pyramid);

        // Joint 3D-3D RANSAC over IDs that co-exist in prev and curr stereo matches.
        let rig = matcher.rig();
        let mut curr_3d: HashMap<u64, [f64; 3]> = HashMap::new();
        for (feat, m) in features.iter().zip(matches.iter()) {
            if let Some(p) = m.point_cam0(rig, feat) {
                curr_3d.insert(feat.id, p);
            }
        }
        let mut corrs_with_id: Vec<(u64, Correspondence3d)> = Vec::new();
        for (&id, &p_curr) in curr_3d.iter() {
            if let Some(&p_prev) = prev_3d.get(&id) {
                corrs_with_id.push((
                    id,
                    Correspondence3d {
                        p1: p_prev,
                        p2: p_curr,
                    },
                ));
            }
        }
        let mut outlier_ids: Vec<u64> = Vec::new();
        if corrs_with_id.len() >= 3 {
            let corrs_only: Vec<Correspondence3d> = corrs_with_id.iter().map(|(_, c)| *c).collect();
            if let Some(result) = rigid_ransac::estimate_rigid_ransac(&corrs_only, &rigid_cfg) {
                for ((id, _), &is_in) in corrs_with_id.iter().zip(result.inliers.iter()) {
                    if !is_in {
                        outlier_ids.push(*id);
                    }
                }
            }
        }
        if !outlier_ids.is_empty() {
            frontend.drop_tracks(&outlier_ids);
            for m in matches.iter_mut() {
                if outlier_ids.contains(&m.id) {
                    m.matched = false;
                }
            }
            for id in &outlier_ids {
                curr_3d.remove(id);
            }
        }
        prev_3d = curr_3d;

        let stereo_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let matched: Vec<_> = matches.iter().filter(|m| m.matched).collect();
        let n_matched = matched.len();
        let n_features = features.len();
        let rate = if n_features > 0 {
            n_matched as f64 / n_features as f64 * 100.0
        } else {
            0.0
        };

        let depth_stats = summarize_depths(&matched);

        eprintln!(
            "{i},{n_features},{n_matched},{rate:.1},{:.2},{:.2},{:.2},{:.2},{frontend_ms:.2},{stereo_ms:.2}",
            depth_stats.mean, depth_stats.median, depth_stats.min, depth_stats.max
        );

        total_frontend_ms += frontend_ms;
        total_stereo_ms += stereo_ms;
        processed_frames += 1;

        let cam0_display = frontend.preprocessed_image().unwrap_or(&cam0_frames[i]);
        let cam1_display = if stereo_histeq == HistEqMethod::None {
            &cam1_frames[i]
        } else {
            histeq::apply_histeq_into(&cam1_frames[i], stereo_histeq, &mut cam1_display_buf);
            &cam1_display_buf
        };
        render_stereo_depth(
            &mut fb,
            win_w,
            cam0_display,
            cam1_display,
            &features,
            &matches,
            scale,
        );
        window
            .update_with_buffer(&fb, win_w, win_h)
            .expect("failed to update stereo window");

        if i % 50 == 0 || i == num_frames - 1 {
            println!(
                "Frame {i:4}: {n_matched:3}/{n_features:3} matched ({rate:4.1}%) | depth: mean={:5.2}m median={:5.2}m [{:.2}–{:.2}m] | fe={frontend_ms:.1}ms stereo={stereo_ms:.1}ms",
                depth_stats.mean, depth_stats.median, depth_stats.min, depth_stats.max
            );
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(
        "  Rudolf-V Stereo — {processed_frames}/{num_frames} frames, baseline={:.4}m",
        matcher.rig().baseline_meters()
    );
    println!("═══════════════════════════════════════════════════════════════════");
    if processed_frames > 0 {
        println!(
            "  Frontend:  {:.1}ms/frame ({:.1}ms total)",
            total_frontend_ms / processed_frames as f64,
            total_frontend_ms
        );
        println!(
            "  Stereo:    {:.1}ms/frame ({:.1}ms total)",
            total_stereo_ms / processed_frames as f64,
            total_stereo_ms
        );
        println!(
            "  Combined:  {:.1}ms/frame ({:.0} FPS)",
            (total_frontend_ms + total_stereo_ms) / processed_frames as f64,
            processed_frames as f64 / ((total_frontend_ms + total_stereo_ms) / 1000.0)
        );
    }
    println!();
    println!("  CSV written to stderr. Redirect: ... 2>stereo.csv");
    println!();
}

fn summarize_depths(matches: &[&StereoMatch]) -> DepthStats {
    let mut depths: Vec<f64> = matches.iter().map(|m| 1.0 / m.inv_depth as f64).collect();
    depths.sort_by(|a, b| a.total_cmp(b));

    if depths.is_empty() {
        return DepthStats {
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    let mean = depths.iter().sum::<f64>() / depths.len() as f64;
    DepthStats {
        mean,
        median: depths[depths.len() / 2],
        min: depths[0],
        max: depths[depths.len() - 1],
    }
}

fn render_stereo_depth(
    fb: &mut [u32],
    win_w: usize,
    cam0: &Image<u8>,
    cam1: &Image<u8>,
    features: &[Feature],
    matches: &[StereoMatch],
    scale: usize,
) {
    fb.fill(0xFF202020);

    let img_w = cam0.width();
    let right_x = img_w * scale + PANEL_GAP;

    render_grayscale_panel(fb, win_w, 0, cam0, scale);
    render_grayscale_panel(fb, win_w, right_x, cam1, scale);

    for (feature, stereo_match) in features.iter().zip(matches) {
        if !stereo_match.matched || stereo_match.inv_depth <= 0.0 {
            draw_cross_scaled(fb, win_w, 0, feature.x, feature.y, scale, 0xFFFF3030);
            continue;
        }

        let depth_m = 1.0 / stereo_match.inv_depth as f64;
        let color = color_for_depth(depth_m);

        draw_point_scaled(fb, win_w, 0, feature.x, feature.y, scale, color);
        draw_point_scaled(
            fb,
            win_w,
            right_x,
            stereo_match.u1,
            stereo_match.v1,
            scale,
            color,
        );
    }
}

fn render_grayscale_panel(
    fb: &mut [u32],
    win_w: usize,
    x_offset: usize,
    image: &Image<u8>,
    scale: usize,
) {
    let img_w = image.width();
    let img_h = image.height();
    for y in 0..img_h {
        for x in 0..img_w {
            let color = grayscale_color(image.get(x, y));
            let out_x = x_offset + x * scale;
            let out_y = y * scale;
            for dy in 0..scale {
                for dx in 0..scale {
                    let idx = (out_y + dy) * win_w + out_x + dx;
                    fb[idx] = color;
                }
            }
        }
    }
}

fn draw_point_scaled(
    fb: &mut [u32],
    win_w: usize,
    x_offset: usize,
    x: f32,
    y: f32,
    scale: usize,
    color: u32,
) {
    let cx = x_offset as i32 + (x * scale as f32).round() as i32;
    let cy = (y * scale as f32).round() as i32;
    let radius = (scale as i32 + 1).max(2);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                set_pixel(fb, win_w, cx + dx, cy + dy, color);
            }
        }
    }
}

fn draw_cross_scaled(
    fb: &mut [u32],
    win_w: usize,
    x_offset: usize,
    x: f32,
    y: f32,
    scale: usize,
    color: u32,
) {
    let cx = x_offset as i32 + (x * scale as f32).round() as i32;
    let cy = (y * scale as f32).round() as i32;
    let radius = (scale as i32 + 2).max(3);

    for d in -radius..=radius {
        set_pixel(fb, win_w, cx + d, cy + d, color);
        set_pixel(fb, win_w, cx + d, cy - d, color);
    }
}

fn set_pixel(fb: &mut [u32], win_w: usize, x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    if x >= win_w {
        return;
    }
    let idx = y * win_w + x;
    if idx < fb.len() {
        fb[idx] = color;
    }
}

fn grayscale_color(value: u8) -> u32 {
    let c = value as u32;
    0xFF000000 | (c << 16) | (c << 8) | c
}

fn color_for_depth(depth_m: f64) -> u32 {
    let t = ((VIS_FAR_DEPTH_M - depth_m) / (VIS_FAR_DEPTH_M - VIS_NEAR_DEPTH_M)).clamp(0.0, 1.0);
    jet_color(t)
}

fn jet_color(t: f64) -> u32 {
    const JET: [(f64, u8, u8, u8); 6] = [
        (0.0, 0, 0, 128),
        (0.125, 0, 0, 255),
        (0.375, 0, 255, 255),
        (0.625, 255, 255, 0),
        (0.875, 255, 0, 0),
        (1.0, 128, 0, 0),
    ];

    let mut idx = JET.len() - 2;
    for i in 0..JET.len() - 1 {
        if t <= JET[i + 1].0 {
            idx = i;
            break;
        }
    }

    let (t0, r0, g0, b0) = JET[idx];
    let (t1, r1, g1, b1) = JET[idx + 1];
    let local_t = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    let r = lerp_u8(r0, r1, local_t) as u32;
    let g = lerp_u8(g0, g1, local_t) as u32;
    let b = lerp_u8(b0, b1, local_t) as u32;

    0xFF000000 | (r << 16) | (g << 8) | b
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}
