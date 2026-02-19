// examples/gpu_pyramid.rs — GPU pyramid visualiser.
//
// Builds a Gaussian pyramid from an input image using both the CPU and GPU
// pipelines, then displays all levels in a single minifb window:
//
//   ┌──────────────────────────────────────────────────────────┐
//   │  CPU level 0  │  CPU level 1  │  CPU level 2  │  ...    │
//   ├──────────────────────────────────────────────────────────┤
//   │  GPU level 0  │  GPU level 1  │  GPU level 2  │  ...    │
//   └──────────────────────────────────────────────────────────┘
//
// Per-level max error (GPU vs CPU) is printed to stdout.
//
// USAGE
// ─────
//   cargo run --example gpu_pyramid                    # generated checkerboard
//   cargo run --example gpu_pyramid -- path/to/img.png # any image file
//   cargo run --example gpu_pyramid -- path/to/img.png 5 1.5
//                                                      # 5 levels, sigma=1.5

use rudolf_v::gpu::device::GpuDevice;
use rudolf_v::gpu::pyramid::GpuPyramidPipeline;
use rudolf_v::image::Image;
use rudolf_v::pyramid::Pyramid;

fn main() {
    // --- Parse arguments ---
    let args: Vec<String> = std::env::args().collect();
    let num_levels: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let sigma: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    // --- Load or generate source image ---
    let src: Image<u8> = if let Some(path) = args.get(1) {
        load_image(path)
    } else {
        eprintln!("[gpu_pyramid] no image path given — using generated checkerboard");
        checkerboard(640, 480, 16)
    };

    eprintln!(
        "[gpu_pyramid] source: {}×{}, {} levels, sigma={sigma}",
        src.width(), src.height(), num_levels
    );

    // --- CPU pyramid ---
    eprintln!("[gpu_pyramid] building CPU pyramid...");
    let cpu_pyr = Pyramid::build(&src, num_levels, sigma);

    // --- GPU pyramid ---
    eprintln!("[gpu_pyramid] initialising GPU...");
    let gpu = GpuDevice::new().expect("failed to initialise a Vulkan GPU");
    eprintln!("[gpu_pyramid] GPU: {}", gpu.adapter_info);

    let pipeline = GpuPyramidPipeline::new(&gpu);
    eprintln!("[gpu_pyramid] building GPU pyramid...");
    let gpu_pyr = pipeline.build(&gpu, &src, num_levels, sigma);

    // --- Read back GPU levels and compute error ---
    eprintln!("[gpu_pyramid] reading back GPU levels...");
    let mut gpu_levels: Vec<Vec<f32>> = Vec::new();
    for lvl in 0..num_levels {
        let data = gpu_pyr.readback_level(&gpu, lvl);
        let cpu_data = cpu_pyr.level(lvl).as_slice();

        let max_err = data.iter().zip(cpu_data.iter())
            .map(|(&g, &c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        let mean_err = data.iter().zip(cpu_data.iter())
            .map(|(&g, &c)| (g - c).abs())
            .sum::<f32>() / data.len() as f32;

        eprintln!(
            "[gpu_pyramid] level {lvl}: {}×{}  max_err={max_err:.4}  mean_err={mean_err:.4}",
            gpu_pyr.levels[lvl].width, gpu_pyr.levels[lvl].height,
        );
        gpu_levels.push(data);
    }

    // --- Build display atlas ---
    // Layout: all CPU levels on top row, all GPU levels on bottom row.
    // Levels are scaled to the height of level 0 for easy comparison.
    let target_h = cpu_pyr.level(0).height();
    let target_h_u32 = target_h as u32;

    // Compute the width of each scaled level panel.
    let panel_widths: Vec<usize> = (0..num_levels)
        .map(|lvl| {
            let lw = cpu_pyr.level(lvl).width();
            let lh = cpu_pyr.level(lvl).height();
            // Scale width proportionally so height == target_h.
            ((lw as f32 / lh as f32) * target_h as f32).round() as usize
        })
        .collect();

    let total_w: usize = panel_widths.iter().sum();
    let total_h: usize = target_h * 2 + 4; // +4px separator row
    let mut fb = vec![0u32; total_w * total_h];

    // Draw separator line (white) between CPU and GPU rows.
    let sep_y_start = target_h;
    let sep_y_end = target_h + 4;
    for y in sep_y_start..sep_y_end {
        for x in 0..total_w {
            fb[y * total_w + x] = 0x00FF_FFFFu32; // cyan separator
        }
    }

    // Render each level panel.
    let mut x_offset = 0usize;
    for lvl in 0..num_levels {
        let pw = panel_widths[lvl];
        let src_w = cpu_pyr.level(lvl).width();
        let src_h = cpu_pyr.level(lvl).height();
        let cpu_data = cpu_pyr.level(lvl).as_slice();
        let gpu_data = &gpu_levels[lvl];

        for py in 0..target_h {
            // Map display row → source row (nearest-neighbour scale).
            let sy = (py as f32 * src_h as f32 / target_h as f32) as usize;
            let sy = sy.min(src_h - 1);

            for px in 0..pw {
                let sx = (px as f32 * src_w as f32 / pw as f32) as usize;
                let sx = sx.min(src_w - 1);

                let cpu_val = cpu_data[sy * src_w + sx].clamp(0.0, 255.0) as u8;
                let gpu_val = gpu_data[sy * src_w + sx].clamp(0.0, 255.0) as u8;

                // CPU row (top).
                let cpu_pixel = grey_to_u32(cpu_val);
                fb[py * total_w + x_offset + px] = cpu_pixel;

                // GPU row (bottom, after separator).
                let gpu_pixel = grey_to_u32(gpu_val);
                fb[(sep_y_end + py) * total_w + x_offset + px] = gpu_pixel;
            }
        }

        // Draw a thin vertical separator between levels (dark grey).
        if lvl + 1 < num_levels {
            for row in 0..total_h {
                let col = x_offset + pw;
                if col < total_w {
                    fb[row * total_w + col] = 0x0040_4040u32;
                }
            }
        }

        x_offset += pw;
    }

    // --- Display in minifb window ---
    let title = format!(
        "GPU pyramid — CPU (top) vs GPU (bottom) — {}×{} | {} levels | σ={sigma}",
        src.width(), src.height(), num_levels
    );

    let mut window = minifb::Window::new(
        &title,
        total_w,
        total_h,
        minifb::WindowOptions {
            resize: false,
            ..Default::default()
        },
    )
    .expect("failed to open window");

    window.limit_update_rate(Some(std::time::Duration::from_millis(16)));

    eprintln!("[gpu_pyramid] window open — press Escape or close to exit");

    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        window.update_with_buffer(&fb, total_w, total_h)
            .expect("window update failed");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pack a greyscale u8 value into an ARGB u32 for minifb.
#[inline]
fn grey_to_u32(v: u8) -> u32 {
    let c = v as u32;
    0xFF00_0000 | (c << 16) | (c << 8) | c
}

/// Load any image format supported by the `image` crate, convert to greyscale u8.
fn load_image(path: &str) -> Image<u8> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e}"))
        .to_luma8();
    let (w, h) = img.dimensions();
    let pixels = img.into_raw(); // Vec<u8>, row-major, stride == width
    Image::<u8>::from_vec(w as usize, h as usize, pixels)
}

/// Generate a checkerboard test image.
fn checkerboard(width: usize, height: usize, tile: usize) -> Image<u8> {
    let pixels: Vec<u8> = (0..width * height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            if (x / tile + y / tile) % 2 == 0 { 220 } else { 40 }
        })
        .collect();
    Image::<u8>::from_vec(width, height, pixels)
}
