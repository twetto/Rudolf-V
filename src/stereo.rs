use crate::camera::StereoRig;
use crate::fast::Feature;
use crate::histeq::{self, HistEqMethod};
use crate::image::{interpolate_bilinear, Image};
use crate::pyramid::{Pyramid, PyramidScratch};

#[derive(Debug, Clone)]
pub struct StereoConfig {
    pub pyramid_levels: usize,
    pub patch_half_size: usize,
    pub max_iterations: usize,
    pub convergence_eps: f64,
    pub min_inv_depth: f64,
    pub max_inv_depth: f64,
    pub init_inv_depth: f64,
    pub max_residual: f32,
    pub huber_delta: f64,
    pub histeq: HistEqMethod,
    pub n_search_candidates: usize,
    /// Number of image-space nearest neighbors to seed retries with after the
    /// initial pass (PatchMatch-style propagation). Only failed features are
    /// retried; 0 disables the propagation pass.
    pub knn_propagation: usize,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            patch_half_size: 4,
            max_iterations: 5,
            convergence_eps: 1e-3,
            min_inv_depth: 0.01,
            max_inv_depth: 5.0,
            init_inv_depth: 1.0 / 3.0,
            max_residual: 20.0,
            huber_delta: 5.0,
            histeq: HistEqMethod::None,
            n_search_candidates: 7,
            knn_propagation: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StereoMatch {
    pub id: u64,
    pub u1: f32,
    pub v1: f32,
    pub inv_depth: f32,
    pub residual: f32,
    pub matched: bool,
}

impl StereoMatch {
    /// Recover the 3D position of this match in cam0's frame from the
    /// feature's pixel coordinates and the rig's cam0 intrinsics.
    /// Returns None for failed or non-positive-depth matches.
    pub fn point_cam0(&self, rig: &StereoRig, feat: &Feature) -> Option<[f64; 3]> {
        if !self.matched || self.inv_depth <= 0.0 {
            return None;
        }
        let (bx, by) = rig.cam0.normalize_undistorted(feat.x as f64, feat.y as f64);
        let rho = self.inv_depth as f64;
        Some([bx / rho, by / rho, 1.0 / rho])
    }
}

pub struct StereoMatcher {
    rig: StereoRig,
    config: StereoConfig,
    cam1_pyramid: Pyramid,
    pyr_scratch: PyramidScratch,
    histeq_buf: Image<u8>,
    grad_x: Vec<Image<f32>>,
    grad_y: Vec<Image<f32>>,
}

impl StereoMatcher {
    pub fn new(rig: StereoRig, config: StereoConfig, img_w: usize, img_h: usize) -> Self {
        let cam1_pyramid =
            Pyramid::build(&Image::<u8>::new(img_w, img_h), config.pyramid_levels, 1.0);
        let pyr_scratch = PyramidScratch::new(img_w, img_h, 1.0);
        Self {
            rig,
            config,
            cam1_pyramid,
            pyr_scratch,
            histeq_buf: Image::new(img_w, img_h),
            grad_x: Vec::new(),
            grad_y: Vec::new(),
        }
    }

    pub fn rig(&self) -> &StereoRig {
        &self.rig
    }

    pub fn match_features(
        &mut self,
        cam1_image: &Image<u8>,
        cam0_features: &[Feature],
        cam0_pyramid: &Pyramid,
    ) -> Vec<StereoMatch> {
        let src = if self.config.histeq == HistEqMethod::None {
            cam1_image
        } else {
            histeq::apply_histeq_into(cam1_image, self.config.histeq, &mut self.histeq_buf);
            &self.histeq_buf
        };

        self.cam1_pyramid
            .build_reuse(src, self.config.pyramid_levels, &mut self.pyr_scratch);

        self.build_gradients();

        let mut matches: Vec<StereoMatch> = cam0_features
            .iter()
            .map(|feat| self.match_one(feat, cam0_pyramid))
            .collect();

        if self.config.knn_propagation > 0 {
            self.propagate_knn(&mut matches, cam0_features, cam0_pyramid);
        }

        matches
    }

    fn propagate_knn(
        &self,
        matches: &mut [StereoMatch],
        features: &[Feature],
        cam0_pyramid: &Pyramid,
    ) {
        let k = self.config.knn_propagation;
        let valid: Vec<usize> = matches
            .iter()
            .enumerate()
            .filter(|(_, m)| m.matched)
            .map(|(i, _)| i)
            .collect();
        if valid.is_empty() {
            return;
        }

        // Rank neighbors by dist² / score: a strong-response neighbor pulls in
        // slightly farther than a close mediocre one. Score is FAST arc-sum, so
        // always non-negative; clamp to >= 1 to avoid divide-by-zero / domination.
        let mut ranked: Vec<(f32, usize)> = Vec::with_capacity(valid.len());
        for i in 0..features.len() {
            if matches[i].matched {
                continue;
            }
            let feat = &features[i];
            ranked.clear();
            for &j in &valid {
                let dx = features[j].x - feat.x;
                let dy = features[j].y - feat.y;
                let dist_sq = dx * dx + dy * dy;
                let score = features[j].score.max(1.0);
                ranked.push((dist_sq / score, j));
            }
            let kk = k.min(ranked.len());
            if ranked.len() > kk {
                ranked.select_nth_unstable_by(kk, |a, b| a.0.partial_cmp(&b.0).unwrap());
                ranked.truncate(kk);
            }

            let mut best = matches[i].clone();
            for &(_, j) in &ranked {
                let init_rho = matches[j].inv_depth as f64;
                let candidate = self.match_one_with_init(feat, cam0_pyramid, init_rho);
                if candidate.matched && (!best.matched || candidate.residual < best.residual) {
                    best = candidate;
                }
            }
            matches[i] = best;
        }
    }

    fn build_gradients(&mut self) {
        let n = self.cam1_pyramid.num_levels();
        self.grad_x.resize_with(n, || Image::new(1, 1));
        self.grad_y.resize_with(n, || Image::new(1, 1));
        self.grad_x.truncate(n);
        self.grad_y.truncate(n);

        for level in 0..n {
            let img = self.cam1_pyramid.level(level);
            let w = img.width();
            let h = img.height();
            let mut gx = vec![0.0f32; w * h];
            let mut gy = vec![0.0f32; w * h];
            if w >= 3 {
                for y in 0..h {
                    for x in 1..w - 1 {
                        gx[y * w + x] = 0.5 * (img.get(x + 1, y) - img.get(x - 1, y));
                    }
                }
            }
            if h >= 3 {
                for y in 1..h - 1 {
                    for x in 0..w {
                        gy[y * w + x] = 0.5 * (img.get(x, y + 1) - img.get(x, y - 1));
                    }
                }
            }
            self.grad_x[level] = Image::from_vec(w, h, gx);
            self.grad_y[level] = Image::from_vec(w, h, gy);
        }
    }

    fn cam0_level<'a>(&self, cam0_pyramid: &'a Pyramid, level: usize) -> (&'a Image<f32>, f64) {
        if level == 0 {
            return (cam0_pyramid.level(0), 0.0);
        }
        if cam0_pyramid.pad_border > 0 && level < cam0_pyramid.padded_levels.len() {
            (
                &cam0_pyramid.padded_levels[level],
                cam0_pyramid.pad_border as f64,
            )
        } else {
            (cam0_pyramid.level(level), 0.0)
        }
    }

    fn patch_cost(
        &self,
        bearing: [f64; 3],
        rho: f64,
        feat: &Feature,
        cam0_img: &Image<f32>,
        cam0_pad: f64,
        cam1_img: &Image<f32>,
        scale: f64,
    ) -> Option<f64> {
        let half = self.config.patch_half_size as isize;
        let p_cam0 = [bearing[0] / rho, bearing[1] / rho, bearing[2] / rho];
        let p_cam1 = self.rig.transform_point(p_cam0);
        let (u1_full, v1_full) = self.rig.cam1.project_point(p_cam1)?;
        let u1_s = u1_full * scale;
        let v1_s = v1_full * scale;
        let u0_s = feat.x as f64 * scale + cam0_pad;
        let v0_s = feat.y as f64 * scale + cam0_pad;

        if u1_s - half as f64 <= 0.0
            || v1_s - half as f64 <= 0.0
            || u1_s + half as f64 >= (cam1_img.width() - 1) as f64
            || v1_s + half as f64 >= (cam1_img.height() - 1) as f64
        {
            return None;
        }

        let mut cost = 0.0f64;
        for dy in -half..half {
            for dx in -half..half {
                let i0 = interpolate_bilinear(
                    cam0_img,
                    (u0_s + dx as f64) as f32,
                    (v0_s + dy as f64) as f32,
                );
                let i1 = interpolate_bilinear(
                    cam1_img,
                    (u1_s + dx as f64) as f32,
                    (v1_s + dy as f64) as f32,
                );
                let r = (i1 - i0) as f64;
                let ar = r.abs();
                cost += if ar <= self.config.huber_delta {
                    0.5 * r * r
                } else {
                    self.config.huber_delta * (ar - 0.5 * self.config.huber_delta)
                };
            }
        }
        Some(cost)
    }

    fn search_initial_rho(
        &self,
        bearing: [f64; 3],
        feat: &Feature,
        cam0_img: &Image<f32>,
        cam0_pad: f64,
        cam1_img: &Image<f32>,
        scale: f64,
    ) -> f64 {
        let n = self.config.n_search_candidates.max(1);
        let rho_min = self.config.min_inv_depth;
        let rho_max = self.config.max_inv_depth;
        let log_min = rho_min.ln();
        let log_max = rho_max.ln();
        let mut best_rho = self.config.init_inv_depth;
        let mut best_cost = f64::INFINITY;
        for i in 0..n {
            let a = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.5
            };
            let rho = (log_min + (log_max - log_min) * a).exp();
            if let Some(cost) =
                self.patch_cost(bearing, rho, feat, cam0_img, cam0_pad, cam1_img, scale)
            {
                if cost < best_cost {
                    best_cost = cost;
                    best_rho = rho;
                }
            }
        }
        best_rho
    }

    fn match_one(&self, feat: &Feature, cam0_pyramid: &Pyramid) -> StereoMatch {
        let (bx, by) = self
            .rig
            .cam0
            .normalize_undistorted(feat.x as f64, feat.y as f64);
        let bearing = [bx, by, 1.0];
        let (search_img, search_pad) = self.cam0_level(cam0_pyramid, 0);
        let init_rho = self.search_initial_rho(
            bearing,
            feat,
            search_img,
            search_pad,
            self.cam1_pyramid.level(0),
            1.0,
        );
        self.match_one_with_init(feat, cam0_pyramid, init_rho)
    }

    fn match_one_with_init(
        &self,
        feat: &Feature,
        cam0_pyramid: &Pyramid,
        init_rho: f64,
    ) -> StereoMatch {
        let fail = StereoMatch {
            id: feat.id,
            u1: 0.0,
            v1: 0.0,
            inv_depth: 0.0,
            residual: f32::INFINITY,
            matched: false,
        };

        let (bx, by) = self
            .rig
            .cam0
            .normalize_undistorted(feat.x as f64, feat.y as f64);
        let bearing = [bx, by, 1.0];

        let rho_min = self.config.min_inv_depth;
        let rho_max = self.config.max_inv_depth;
        let half = self.config.patch_half_size as isize;

        let n_levels = self
            .config
            .pyramid_levels
            .min(cam0_pyramid.num_levels())
            .min(self.cam1_pyramid.num_levels());

        let cam0_l0 = cam0_pyramid.level(0);
        if feat.x as f64 - half as f64 <= 0.0
            || feat.y as f64 - half as f64 <= 0.0
            || feat.x as f64 + half as f64 >= (cam0_l0.width() - 1) as f64
            || feat.y as f64 + half as f64 >= (cam0_l0.height() - 1) as f64
        {
            return fail;
        }

        let mut rho = init_rho.clamp(rho_min, rho_max);

        for level in (0..n_levels).rev() {
            let scale = 1.0 / (1usize << level) as f64;
            let (cam0_img, cam0_pad) = self.cam0_level(cam0_pyramid, level);
            let cam1_img = self.cam1_pyramid.level(level);
            let gx_img = &self.grad_x[level];
            let gy_img = &self.grad_y[level];

            let u0_s = feat.x as f64 * scale + cam0_pad;
            let v0_s = feat.y as f64 * scale + cam0_pad;

            if u0_s - half as f64 <= 0.0
                || v0_s - half as f64 <= 0.0
                || u0_s + half as f64 >= (cam0_img.width() - 1) as f64
                || v0_s + half as f64 >= (cam0_img.height() - 1) as f64
            {
                return fail;
            }

            for _ in 0..self.config.max_iterations {
                if rho <= 0.0 {
                    return fail;
                }
                let p_cam0 = [bearing[0] / rho, bearing[1] / rho, bearing[2] / rho];
                let p_cam1 = self.rig.transform_point(p_cam0);
                let Some((u1_full, v1_full)) = self.rig.cam1.project_point(p_cam1) else {
                    return fail;
                };
                let u1_s = u1_full * scale;
                let v1_s = v1_full * scale;

                if u1_s - half as f64 <= 0.0
                    || v1_s - half as f64 <= 0.0
                    || u1_s + half as f64 >= (cam1_img.width() - 1) as f64
                    || v1_s + half as f64 >= (cam1_img.height() - 1) as f64
                {
                    return fail;
                }

                let rho_sq = rho * rho;
                let db_drho = self.rig.rotate([
                    -bearing[0] / rho_sq,
                    -bearing[1] / rho_sq,
                    -bearing[2] / rho_sq,
                ]);
                let j_proj = self.rig.cam1.projection_jacobian(p_cam1);
                let du_drho = (j_proj[0][0] * db_drho[0]
                    + j_proj[0][1] * db_drho[1]
                    + j_proj[0][2] * db_drho[2])
                    * scale;
                let dv_drho = (j_proj[1][0] * db_drho[0]
                    + j_proj[1][1] * db_drho[1]
                    + j_proj[1][2] * db_drho[2])
                    * scale;

                let mut grad_sum = 0.0f64;
                let mut hess_sum = 0.0f64;
                let mut n_valid = 0usize;

                for dy in -half..half {
                    for dx in -half..half {
                        let cx = u0_s + dx as f64;
                        let cy = v0_s + dy as f64;
                        let rx = u1_s + dx as f64;
                        let ry = v1_s + dy as f64;

                        let i_cam0 = interpolate_bilinear(cam0_img, cx as f32, cy as f32);
                        let i_cam1 = interpolate_bilinear(cam1_img, rx as f32, ry as f32);
                        let gx = interpolate_bilinear(gx_img, rx as f32, ry as f32);
                        let gy = interpolate_bilinear(gy_img, rx as f32, ry as f32);

                        let jac = gx as f64 * du_drho + gy as f64 * dv_drho;
                        let r = i_cam1 as f64 - i_cam0 as f64;
                        let ar = r.abs();
                        let weight = if ar <= self.config.huber_delta {
                            1.0
                        } else {
                            self.config.huber_delta / ar
                        };
                        grad_sum += weight * jac * r;
                        hess_sum += weight * jac * jac;
                        n_valid += 1;
                    }
                }

                if hess_sum < 1e-12 || n_valid == 0 {
                    return fail;
                }

                let delta_rho = -grad_sum / hess_sum;
                rho = (rho + delta_rho).clamp(rho_min, rho_max);

                if delta_rho.abs() < self.config.convergence_eps {
                    break;
                }
            }
        }

        if rho <= 0.0 {
            return fail;
        }
        let p_cam0 = [bearing[0] / rho, bearing[1] / rho, bearing[2] / rho];
        let p_cam1 = self.rig.transform_point(p_cam0);
        let Some((u1, v1)) = self.rig.cam1.project_point(p_cam1) else {
            return fail;
        };

        let half_f = self.config.patch_half_size as f64;
        let cam0_l0 = cam0_pyramid.level(0);
        let cam1_l0 = self.cam1_pyramid.level(0);
        if u1 - half_f <= 0.0
            || v1 - half_f <= 0.0
            || u1 + half_f >= (cam1_l0.width() - 1) as f64
            || v1 + half_f >= (cam1_l0.height() - 1) as f64
            || feat.x as f64 - half_f <= 0.0
            || feat.y as f64 - half_f <= 0.0
            || feat.x as f64 + half_f >= (cam0_l0.width() - 1) as f64
            || feat.y as f64 + half_f >= (cam0_l0.height() - 1) as f64
        {
            return fail;
        }

        let mut residual_sum = 0.0f64;
        let mut n = 0usize;
        for dy in -half..half {
            for dx in -half..half {
                let i0 = interpolate_bilinear(cam0_l0, feat.x + dx as f32, feat.y + dy as f32);
                let i1 =
                    interpolate_bilinear(cam1_l0, u1 as f32 + dx as f32, v1 as f32 + dy as f32);
                residual_sum += (i1 - i0).abs() as f64;
                n += 1;
            }
        }
        let mean_residual = if n > 0 {
            (residual_sum / n as f64) as f32
        } else {
            f32::INFINITY
        };

        StereoMatch {
            id: feat.id,
            u1: u1 as f32,
            v1: v1 as f32,
            inv_depth: rho as f32,
            residual: mean_residual,
            matched: mean_residual <= self.config.max_residual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraIntrinsics;

    fn make_stereo_rig_simple() -> StereoRig {
        let cam0 = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let cam1 = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        StereoRig {
            cam0,
            cam1,
            r_10: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t_10: [-0.11, 0.0, 0.0],
        }
    }

    fn make_textured_image(w: usize, h: usize) -> Image<u8> {
        let mut noise = vec![0.0f64; w * h];
        let mut rng: u64 = 42;
        for v in noise.iter_mut() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (rng >> 33) as f64 / (1u64 << 31) as f64 * 255.0;
        }
        // Iterative box blur (3 passes ≈ Gaussian σ≈3.5)
        for _ in 0..3 {
            let prev = noise.clone();
            let r = 3isize;
            let inv = 1.0 / ((2 * r + 1) * (2 * r + 1)) as f64;
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                            let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                            sum += prev[sy * w + sx];
                        }
                    }
                    noise[y * w + x] = sum * inv;
                }
            }
        }
        let mut data = vec![0u8; w * h];
        for (d, &n) in data.iter_mut().zip(noise.iter()) {
            *d = n.clamp(0.0, 255.0) as u8;
        }
        Image::from_vec(w, h, data)
    }

    fn make_shifted_image(src: &Image<u8>, dx: f64) -> Image<u8> {
        let w = src.width();
        let h = src.height();
        let src_f32 = {
            let mut img = Image::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    *img.get_mut(x, y) = src.get(x, y) as f32;
                }
            }
            img
        };
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let sx = x as f64 + dx;
                if sx >= 0.0 && sx < (w - 1) as f64 {
                    out[y * w + x] = interpolate_bilinear(&src_f32, sx as f32, y as f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        Image::from_vec(w, h, out)
    }

    #[test]
    fn test_stereo_match_known_depth() {
        let rig = make_stereo_rig_simple();
        let depth = 3.0;
        let disparity = rig.cam1.fx * -rig.t_10[0] / depth;

        let cam0_img = make_textured_image(640, 480);
        let cam1_img = make_shifted_image(&cam0_img, disparity);

        let cam0_pyramid = Pyramid::build(&cam0_img, 3, 1.0);

        let config = StereoConfig {
            pyramid_levels: 3,
            patch_half_size: 4,
            max_iterations: 10,
            n_search_candidates: 15,
            ..StereoConfig::default()
        };
        let mut matcher = StereoMatcher::new(rig, config, 640, 480);

        let feat = Feature {
            x: 320.0,
            y: 240.0,
            score: 100.0,
            level: 0,
            id: 1,
            descriptor: 0,
        };

        let results = matcher.match_features(&cam1_img, &[feat], &cam0_pyramid);
        assert_eq!(results.len(), 1);
        let m = &results[0];
        assert!(m.matched, "should match, residual={}", m.residual);
        let recovered_depth = 1.0 / m.inv_depth as f64;
        assert!(
            (recovered_depth - depth).abs() < 0.5,
            "depth: expected ~{depth}, got {recovered_depth}"
        );
    }

    #[test]
    fn test_stereo_projection_moves_left_for_close_points() {
        let rig = make_stereo_rig_simple();
        let center_u = rig.cam0.cx;

        let near_cam1 = rig.transform_point([0.0, 0.0, 2.0]);
        let far_cam1 = rig.transform_point([0.0, 0.0, 10.0]);
        let (near_u, _) = rig.cam1.project_point(near_cam1).unwrap();
        let (far_u, _) = rig.cam1.project_point(far_cam1).unwrap();

        assert!(
            near_u < far_u,
            "closer points should move farther left in cam1: near_u={near_u}, far_u={far_u}"
        );
        assert!(
            far_u < center_u,
            "right-camera projection should be left of cam0 center for positive depth: far_u={far_u}, center_u={center_u}"
        );
    }
}
