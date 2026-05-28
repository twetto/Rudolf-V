use crate::camera::StereoRig;
use crate::fast::Feature;
use crate::histeq::{self, HistEqMethod};
use crate::image::Image;
use crate::klt::bilerp_ptr;
use crate::pyramid::{Pyramid, PyramidScratch};

/// Precomputed bilinear corner weights and base integer coords for a (x, y)
/// sample whose fractional part is shared across an entire patch sweep.
#[derive(Clone, Copy)]
struct BilerpKey {
    x0: isize,
    y0: isize,
    w00: f32,
    w10: f32,
    w01: f32,
    w11: f32,
}

#[inline(always)]
fn bilerp_key(x: f64, y: f64) -> BilerpKey {
    let x0 = x as isize;
    let y0 = y as isize;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let one_fx = 1.0 - fx;
    let one_fy = 1.0 - fy;
    BilerpKey {
        x0,
        y0,
        w00: one_fx * one_fy,
        w10: fx * one_fy,
        w01: one_fx * fy,
        w11: fx * fy,
    }
}

#[inline(always)]
unsafe fn accumulate_stereo_row_scalar(
    r0: *const f32,
    r1: *const f32,
    ix0: usize,
    count: usize,
    k: BilerpKey,
    t: &[f32],
    tx: &[f32],
    ty: &[f32],
) -> (f64, f64) {
    let mut s_txr = 0.0f64;
    let mut s_tyr = 0.0f64;
    for lx in 0..count {
        let ix = ix0 + lx;
        let i_cam1 = bilerp_ptr(r0, r1, ix, k.w00, k.w10, k.w01, k.w11);
        let r = (i_cam1 - *t.get_unchecked(lx)) as f64;
        s_txr += *tx.get_unchecked(lx) as f64 * r;
        s_tyr += *ty.get_unchecked(lx) as f64 * r;
    }
    (s_txr, s_tyr)
}

#[inline(always)]
unsafe fn residual_stereo_row_scalar(
    c0_r0: *const f32,
    c0_r1: *const f32,
    c1_r0: *const f32,
    c1_r1: *const f32,
    cam0_ix0: usize,
    cam1_ix0: usize,
    count: usize,
    k0: BilerpKey,
    k1: BilerpKey,
) -> f64 {
    let mut sum = 0.0f64;
    for lx in 0..count {
        let i0 = bilerp_ptr(c0_r0, c0_r1, cam0_ix0 + lx, k0.w00, k0.w10, k0.w01, k0.w11);
        let i1 = bilerp_ptr(c1_r0, c1_r1, cam1_ix0 + lx, k1.w00, k1.w10, k1.w01, k1.w11);
        sum += (i1 - i0).abs() as f64;
    }
    sum
}

#[inline(always)]
unsafe fn residual_stereo_row(
    c0_r0: *const f32,
    c0_r1: *const f32,
    c1_r0: *const f32,
    c1_r1: *const f32,
    cam0_ix0: usize,
    cam1_ix0: usize,
    count: usize,
    k0: BilerpKey,
    k1: BilerpKey,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return residual_stereo_row_avx2(
                c0_r0, c0_r1, c1_r0, c1_r1, cam0_ix0, cam1_ix0, count, k0, k1,
            );
        }
    }
    residual_stereo_row_scalar(
        c0_r0, c0_r1, c1_r0, c1_r1, cam0_ix0, cam1_ix0, count, k0, k1,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn residual_stereo_row_avx2(
    c0_r0: *const f32,
    c0_r1: *const f32,
    c1_r0: *const f32,
    c1_r1: *const f32,
    cam0_ix0: usize,
    cam1_ix0: usize,
    count: usize,
    k0: BilerpKey,
    k1: BilerpKey,
) -> f64 {
    use std::arch::x86_64::*;

    let mut vsum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let k0_w00 = _mm256_set1_ps(k0.w00);
    let k0_w10 = _mm256_set1_ps(k0.w10);
    let k0_w01 = _mm256_set1_ps(k0.w01);
    let k0_w11 = _mm256_set1_ps(k0.w11);
    let k1_w00 = _mm256_set1_ps(k1.w00);
    let k1_w10 = _mm256_set1_ps(k1.w10);
    let k1_w01 = _mm256_set1_ps(k1.w01);
    let k1_w11 = _mm256_set1_ps(k1.w11);

    let chunks = count / 8;
    for c in 0..chunks {
        let off = c * 8;
        let c0_ix = cam0_ix0 + off;
        let c1_ix = cam1_ix0 + off;

        let c0_p00 = _mm256_loadu_ps(c0_r0.add(c0_ix));
        let c0_p10 = _mm256_loadu_ps(c0_r0.add(c0_ix + 1));
        let c0_p01 = _mm256_loadu_ps(c0_r1.add(c0_ix));
        let c0_p11 = _mm256_loadu_ps(c0_r1.add(c0_ix + 1));
        let mut i0 = _mm256_mul_ps(c0_p00, k0_w00);
        i0 = _mm256_fmadd_ps(c0_p10, k0_w10, i0);
        i0 = _mm256_fmadd_ps(c0_p01, k0_w01, i0);
        i0 = _mm256_fmadd_ps(c0_p11, k0_w11, i0);

        let c1_p00 = _mm256_loadu_ps(c1_r0.add(c1_ix));
        let c1_p10 = _mm256_loadu_ps(c1_r0.add(c1_ix + 1));
        let c1_p01 = _mm256_loadu_ps(c1_r1.add(c1_ix));
        let c1_p11 = _mm256_loadu_ps(c1_r1.add(c1_ix + 1));
        let mut i1 = _mm256_mul_ps(c1_p00, k1_w00);
        i1 = _mm256_fmadd_ps(c1_p10, k1_w10, i1);
        i1 = _mm256_fmadd_ps(c1_p01, k1_w01, i1);
        i1 = _mm256_fmadd_ps(c1_p11, k1_w11, i1);

        let diff = _mm256_sub_ps(i1, i0);
        vsum = _mm256_add_ps(vsum, _mm256_andnot_ps(sign_mask, diff));
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), vsum);
    let mut sum = lanes.iter().sum::<f32>() as f64;

    for lx in (chunks * 8)..count {
        let i0 = bilerp_ptr(c0_r0, c0_r1, cam0_ix0 + lx, k0.w00, k0.w10, k0.w01, k0.w11);
        let i1 = bilerp_ptr(c1_r0, c1_r1, cam1_ix0 + lx, k1.w00, k1.w10, k1.w01, k1.w11);
        sum += (i1 - i0).abs() as f64;
    }
    sum
}

#[inline(always)]
unsafe fn accumulate_stereo_row(
    r0: *const f32,
    r1: *const f32,
    ix0: usize,
    count: usize,
    k: BilerpKey,
    t: &[f32],
    tx: &[f32],
    ty: &[f32],
) -> (f64, f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return accumulate_stereo_row_avx2(r0, r1, ix0, count, k, t, tx, ty);
        }
    }
    accumulate_stereo_row_scalar(r0, r1, ix0, count, k, t, tx, ty)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_stereo_row_avx2(
    r0: *const f32,
    r1: *const f32,
    ix0: usize,
    count: usize,
    k: BilerpKey,
    t: &[f32],
    tx: &[f32],
    ty: &[f32],
) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut vxr = _mm256_setzero_ps();
    let mut vyr = _mm256_setzero_ps();
    let vw00 = _mm256_set1_ps(k.w00);
    let vw10 = _mm256_set1_ps(k.w10);
    let vw01 = _mm256_set1_ps(k.w01);
    let vw11 = _mm256_set1_ps(k.w11);

    let chunks = count / 8;
    for c in 0..chunks {
        let off = c * 8;
        let ix = ix0 + off;
        let p00 = _mm256_loadu_ps(r0.add(ix));
        let p10 = _mm256_loadu_ps(r0.add(ix + 1));
        let p01 = _mm256_loadu_ps(r1.add(ix));
        let p11 = _mm256_loadu_ps(r1.add(ix + 1));
        let mut w = _mm256_mul_ps(p00, vw00);
        w = _mm256_fmadd_ps(p10, vw10, w);
        w = _mm256_fmadd_ps(p01, vw01, w);
        w = _mm256_fmadd_ps(p11, vw11, w);

        let vt = _mm256_loadu_ps(t.as_ptr().add(off));
        let vr = _mm256_sub_ps(w, vt);
        let vtx = _mm256_loadu_ps(tx.as_ptr().add(off));
        let vty = _mm256_loadu_ps(ty.as_ptr().add(off));
        vxr = _mm256_fmadd_ps(vtx, vr, vxr);
        vyr = _mm256_fmadd_ps(vty, vr, vyr);
    }

    let mut txr_lanes = [0.0f32; 8];
    let mut tyr_lanes = [0.0f32; 8];
    _mm256_storeu_ps(txr_lanes.as_mut_ptr(), vxr);
    _mm256_storeu_ps(tyr_lanes.as_mut_ptr(), vyr);
    let mut s_txr = txr_lanes.iter().sum::<f32>() as f64;
    let mut s_tyr = tyr_lanes.iter().sum::<f32>() as f64;

    for lx in (chunks * 8)..count {
        let ix = ix0 + lx;
        let i_cam1 = bilerp_ptr(r0, r1, ix, k.w00, k.w10, k.w01, k.w11);
        let r = (i_cam1 - *t.get_unchecked(lx)) as f64;
        s_txr += *tx.get_unchecked(lx) as f64 * r;
        s_tyr += *ty.get_unchecked(lx) as f64 * r;
    }
    (s_txr, s_tyr)
}

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

/// Maximum supported patch_half_size in StereoConfig. Template buffers are
/// stack-allocated using these caps so the matcher avoids per-feature heap
/// traffic; bumping these allocates more stack per `match_one_with_init` call.
const MAX_PATCH_HALF: usize = 8;
const MAX_INNER_SIDE: usize = 2 * MAX_PATCH_HALF;
const MAX_INNER_PIXELS: usize = MAX_INNER_SIDE * MAX_INNER_SIDE;
const MAX_EXT_SIDE: usize = MAX_INNER_SIDE + 2;
const MAX_EXT_PIXELS: usize = MAX_EXT_SIDE * MAX_EXT_SIDE;

pub struct StereoMatcher {
    rig: StereoRig,
    config: StereoConfig,
    cam1_pyramid: Pyramid,
    pyr_scratch: PyramidScratch,
    histeq_buf: Image<u8>,
    t_ext: [f32; MAX_EXT_PIXELS],
    t: [f32; MAX_INNER_PIXELS],
    tx: [f32; MAX_INNER_PIXELS],
    ty: [f32; MAX_INNER_PIXELS],
}

impl StereoMatcher {
    pub fn new(rig: StereoRig, config: StereoConfig, img_w: usize, img_h: usize) -> Self {
        assert!(
            config.patch_half_size <= MAX_PATCH_HALF,
            "patch_half_size {} exceeds MAX_PATCH_HALF {}",
            config.patch_half_size,
            MAX_PATCH_HALF,
        );
        let cam1_pyramid =
            Pyramid::build(&Image::<u8>::new(img_w, img_h), config.pyramid_levels, 1.0);
        let pyr_scratch = PyramidScratch::new(img_w, img_h, 1.0);
        Self {
            rig,
            config,
            cam1_pyramid,
            pyr_scratch,
            histeq_buf: Image::new(img_w, img_h),
            t_ext: [0.0; MAX_EXT_PIXELS],
            t: [0.0; MAX_INNER_PIXELS],
            tx: [0.0; MAX_INNER_PIXELS],
            ty: [0.0; MAX_INNER_PIXELS],
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
        &mut self,
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
            || u0_s - half as f64 <= 0.0
            || v0_s - half as f64 <= 0.0
            || u0_s + half as f64 >= (cam0_img.width() - 1) as f64
            || v0_s + half as f64 >= (cam0_img.height() - 1) as f64
        {
            return None;
        }

        let k0 = bilerp_key(u0_s, v0_s);
        let k1 = bilerp_key(u1_s, v1_s);
        let mut cost = 0.0f64;
        for dy in -half..half {
            // SAFETY: u*_s ± half are inside [0, w-1] / [0, h-1] per the bounds check above.
            let cam0_y = (k0.y0 + dy) as usize;
            let cam1_y = (k1.y0 + dy) as usize;
            unsafe {
                let c0_r0 = cam0_img.row_ptr(cam0_y);
                let c0_r1 = cam0_img.row_ptr(cam0_y + 1);
                let c1_r0 = cam1_img.row_ptr(cam1_y);
                let c1_r1 = cam1_img.row_ptr(cam1_y + 1);
                for dx in -half..half {
                    let cam0_ix = (k0.x0 + dx) as usize;
                    let cam1_ix = (k1.x0 + dx) as usize;
                    let i0 = bilerp_ptr(c0_r0, c0_r1, cam0_ix, k0.w00, k0.w10, k0.w01, k0.w11);
                    let i1 = bilerp_ptr(c1_r0, c1_r1, cam1_ix, k1.w00, k1.w10, k1.w01, k1.w11);
                    let r = (i1 - i0) as f64;
                    cost += 0.5 * r * r;
                }
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

    fn match_one(&mut self, feat: &Feature, cam0_pyramid: &Pyramid) -> StereoMatch {
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
        &mut self,
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

        // Inverse-compositional GN: precompute the template (cam0) patch and its
        // gradients once per level, then each iteration only samples cam1 at the
        // warped position. The warp Jacobian (du/dρ, dv/dρ) is still re-evaluated
        // at the current ρ — strict IC for projective warps is approximate, but
        // staying inside the coarse-to-fine basin keeps convergence well-behaved.
        let inner_side = 2 * half as usize;
        let ext_side = inner_side + 2;

        for level in (0..n_levels).rev() {
            let scale = 1.0 / (1usize << level) as f64;
            let (cam0_img, cam0_pad) = self.cam0_level(cam0_pyramid, level);
            let cam1_img = self.cam1_pyramid.level(level);

            let u0_s = feat.x as f64 * scale + cam0_pad;
            let v0_s = feat.y as f64 * scale + cam0_pad;

            // Bounds: we sample a (2*half + 2) × (2*half + 2) extended patch on
            // cam0 to get central-difference gradients for the inner 8×8 patch.
            let half_ext = half as f64 + 1.0;
            if u0_s - half_ext <= 0.0
                || v0_s - half_ext <= 0.0
                || u0_s + half_ext >= (cam0_img.width() - 1) as f64
                || v0_s + half_ext >= (cam0_img.height() - 1) as f64
            {
                return fail;
            }

            // Sample the extended cam0 patch once.
            let k0 = bilerp_key(u0_s, v0_s);
            for ey in 0..ext_side {
                let dy = ey as isize - half - 1;
                let cam0_y = (k0.y0 + dy) as usize;
                // SAFETY: row + 1 in bounds per the (half + 1) check above.
                unsafe {
                    let r0 = cam0_img.row_ptr(cam0_y);
                    let r1 = cam0_img.row_ptr(cam0_y + 1);
                    for ex in 0..ext_side {
                        let dx = ex as isize - half - 1;
                        let ix = (k0.x0 + dx) as usize;
                        self.t_ext[ey * ext_side + ex] =
                            bilerp_ptr(r0, r1, ix, k0.w00, k0.w10, k0.w01, k0.w11);
                    }
                }
            }

            // Extract template + central-difference gradients for the inner patch.
            for iy in 0..inner_side {
                for ix in 0..inner_side {
                    let e = (iy + 1) * ext_side + (ix + 1);
                    let inner_idx = iy * inner_side + ix;
                    self.t[inner_idx] = self.t_ext[e];
                    self.tx[inner_idx] = 0.5 * (self.t_ext[e + 1] - self.t_ext[e - 1]);
                    self.ty[inner_idx] =
                        0.5 * (self.t_ext[e + ext_side] - self.t_ext[e - ext_side]);
                }
            }

            let n_patch = inner_side * inner_side;
            let mut s_tx2 = 0.0f64;
            let mut s_txty = 0.0f64;
            let mut s_ty2 = 0.0f64;
            for i in 0..n_patch {
                let txi = self.tx[i] as f64;
                let tyi = self.ty[i] as f64;
                s_tx2 += txi * txi;
                s_txty += txi * tyi;
                s_ty2 += tyi * tyi;
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

                let k1 = bilerp_key(u1_s, v1_s);
                let mut s_txr = 0.0f64;
                let mut s_tyr = 0.0f64;
                let mut n_valid = 0usize;

                for row in 0..inner_side {
                    let dy = row as isize - half;
                    let cam1_y = (k1.y0 + dy) as usize;
                    // SAFETY: u1_s ± half within cam1 bounds per check above.
                    unsafe {
                        let c1_r0 = cam1_img.row_ptr(cam1_y);
                        let c1_r1 = cam1_img.row_ptr(cam1_y + 1);
                        let row_start = row * inner_side;
                        let ix0 = (k1.x0 - half) as usize;
                        let (row_txr, row_tyr) = accumulate_stereo_row(
                            c1_r0,
                            c1_r1,
                            ix0,
                            inner_side,
                            k1,
                            &self.t[row_start..row_start + inner_side],
                            &self.tx[row_start..row_start + inner_side],
                            &self.ty[row_start..row_start + inner_side],
                        );
                        s_txr += row_txr;
                        s_tyr += row_tyr;
                        n_valid += inner_side;
                    }
                }

                let grad_sum = du_drho * s_txr + dv_drho * s_tyr;
                let hess_sum = du_drho * du_drho * s_tx2
                    + 2.0 * du_drho * dv_drho * s_txty
                    + dv_drho * dv_drho * s_ty2;

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

        let k0 = bilerp_key(feat.x as f64, feat.y as f64);
        let k1 = bilerp_key(u1, v1);
        let mut residual_sum = 0.0f64;
        let inner_side = 2 * half as usize;
        for row in 0..inner_side {
            let dy = row as isize - half;
            let cam0_y = (k0.y0 + dy) as usize;
            let cam1_y = (k1.y0 + dy) as usize;
            // SAFETY: feat ± half and (u1, v1) ± half validated against level-0 dims above.
            unsafe {
                residual_sum += residual_stereo_row(
                    cam0_l0.row_ptr(cam0_y),
                    cam0_l0.row_ptr(cam0_y + 1),
                    cam1_l0.row_ptr(cam1_y),
                    cam1_l0.row_ptr(cam1_y + 1),
                    (k0.x0 - half) as usize,
                    (k1.x0 - half) as usize,
                    inner_side,
                    k0,
                    k1,
                );
            }
        }
        let n_patch = inner_side * inner_side;
        let mean_residual = if n_patch > 0 {
            (residual_sum / n_patch as f64) as f32
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
    use crate::image::interpolate_bilinear;

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
