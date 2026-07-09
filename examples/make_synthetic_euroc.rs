// examples/make_synthetic_euroc.rs
//
// Generate a tiny deterministic EuRoC-ASL-shaped dataset for CI visualization
// artifacts. This is not a benchmark dataset; it exists so CI can exercise the
// existing euroc_frontend example, upload its SVG/MP4 outputs, and assert that
// tracked flow roughly matches the known +1 px/frame translation.

use image::{GrayImage, Luma};

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <output_dataset_path> [frames]", args[0]);
        std::process::exit(1);
    }

    let root = PathBuf::from(&args[1]);
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(90);
    let width = 320u32;
    let height = 240u32;
    let gt_dx = 1i32;
    let gt_dy = 1i32;

    let cam0 = root.join("mav0").join("cam0");
    let data = cam0.join("data");
    fs::create_dir_all(&data).expect("failed to create EuRoC data directory");

    write_sensor_yaml(&cam0, width, height);

    let mut csv = String::from("#timestamp [ns],filename\n");
    for frame in 0..frames {
        let timestamp = 1_400_000_000_000_000_000u64 + frame as u64 * 50_000_000;
        let filename = format!("{timestamp}.png");
        let image = render_frame(width, height, frame as i32);
        image
            .save(data.join(&filename))
            .expect("failed to save synthetic frame");
        csv.push_str(&format!("{timestamp},{filename}\n"));
    }

    fs::write(cam0.join("data.csv"), csv).expect("failed to write data.csv");
    println!(
        "Wrote synthetic EuRoC dataset: {} ({} frames, {}x{}, GT flow +{}, +{})",
        root.display(),
        frames,
        width,
        height,
        gt_dx,
        gt_dy
    );
}

fn write_sensor_yaml(cam0: &Path, width: u32, height: u32) {
    let fx = 220.0;
    let fy = 220.0;
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let yaml = format!(
        "sensor_type: camera\n\
         comment: synthetic CI fixture\n\
         resolution: [{width}, {height}]\n\
         intrinsics: [{fx}, {fy}, {cx}, {cy}]\n\
         distortion_model: radtan\n\
         distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n"
    );
    fs::write(cam0.join("sensor.yaml"), yaml).expect("failed to write sensor.yaml");
}

fn render_frame(width: u32, height: u32, frame: i32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    let dx = frame;
    let dy = frame;

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            image.put_pixel(x as u32, y as u32, Luma([28]));
        }
    }

    for i in 0..42 {
        let seed = i * 97 + 13;
        let cx = 16 + pseudo(seed, 210) + dx;
        let cy = 16 + pseudo(seed * 3 + 7, 130) + dy;
        let angle_bucket = pseudo(seed * 5 + 11, 8);
        let polarity = if i % 2 == 0 { 225 } else { 38 };

        draw_slanted_patch(&mut image, cx, cy, angle_bucket, polarity);
        draw_blob(
            &mut image,
            cx + pseudo(seed * 17 + 3, 15) - 7,
            cy + pseudo(seed * 19 + 5, 15) - 7,
            3 + pseudo(seed * 23 + 1, 4),
            255u8.saturating_sub(polarity / 2),
        );
    }

    for i in 0..18 {
        let seed = i * 211 + 29;
        let x0 = 16 + pseudo(seed, 210);
        let y0 = 16 + pseudo(seed * 2 + 1, 130);
        let x1 = 16 + pseudo(seed * 3 + 5, 210);
        let y1 = 16 + pseudo(seed * 7 + 9, 130);
        draw_line(&mut image, x0 + dx, y0 + dy, x1 + dx, y1 + dy, 190);
    }

    image
}

fn pseudo(seed: i32, modulo: i32) -> i32 {
    let mut x = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    x ^= x >> 16;
    x.rem_euclid(modulo.max(1))
}

fn draw_slanted_patch(image: &mut GrayImage, cx: i32, cy: i32, angle_bucket: i32, value: u8) {
    let strokes = [
        ((-8, -5), (7, 2)),
        ((-6, 7), (9, -7)),
        ((-3, -9), (6, 8)),
        ((-10, 1), (5, 10)),
    ];
    for (idx, (a, b)) in strokes.iter().enumerate() {
        let (x0, y0) = rotate_45_bucket(a.0, a.1, angle_bucket + idx as i32);
        let (x1, y1) = rotate_45_bucket(b.0, b.1, angle_bucket + idx as i32);
        draw_line(image, cx + x0, cy + y0, cx + x1, cy + y1, value);
        draw_line(image, cx + x0 + 1, cy + y0, cx + x1 + 1, cy + y1, value);
    }
}

fn rotate_45_bucket(x: i32, y: i32, bucket: i32) -> (i32, i32) {
    match bucket.rem_euclid(8) {
        0 => (x, y),
        1 => (x - y, x + y),
        2 => (-y, x),
        3 => (-x - y, x - y),
        4 => (-x, -y),
        5 => (-x + y, -x - y),
        6 => (y, -x),
        _ => (x + y, -x + y),
    }
}

fn draw_blob(image: &mut GrayImage, cx: i32, cy: i32, radius: i32, value: u8) {
    let r2 = radius * radius;
    for y in cy - radius..=cy + radius {
        for x in cx - radius..=cx + radius {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                set_pixel_checked(image, x, y, value);
            }
        }
    }
}

fn draw_line(image: &mut GrayImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, value: u8) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        set_pixel_checked(image, x0, y0, value);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn set_pixel_checked(image: &mut GrayImage, x: i32, y: i32, value: u8) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, Luma([value]));
    }
}
