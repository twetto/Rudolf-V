use crate::fast::Feature;
use crate::image::{Image, interpolate_bilinear};
use crate::klt::{TrackStatus, TrackedFeature};
use crate::pyramid::Pyramid;
use nalgebra::{Matrix2, SMatrix, SVector, Vector2};

type Mat6 = SMatrix<f32, 6, 6>;
type Vec6 = SVector<f32, 6>;

/// KLT template source for temporal tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KltTemplatePolicy {
    /// Use the previous frame as the template.
    PreviousFrame,
    /// Store the first observation's patch and keep matching against it.
    FirstObservation,
}

/// Geometric warp used when matching a stored reference patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKltWarp {
    Translation,
    Affine,
}

#[derive(Debug, Clone)]
struct ReferenceLevelPatch {
    level: usize,
    ref_x: f32,
    ref_y: f32,
    side: usize,
    values: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
    ih00: f32,
    ih01: f32,
    ih11: f32,
    affine_h: Mat6,
}

#[derive(Debug, Clone)]
pub struct ReferenceTrack {
    pub id: u64,
    pub feature: Feature,
    levels: Vec<ReferenceLevelPatch>,
}

impl ReferenceTrack {
    pub fn new(
        feature: &Feature,
        pyramid: &Pyramid,
        window_size: usize,
        max_levels: usize,
    ) -> Option<Self> {
        let num_levels = max_levels.min(pyramid.num_levels());
        let mut levels = Vec::with_capacity(num_levels);

        for level in 0..num_levels {
            let scale = 1.0 / (1u32 << level) as f32;
            if let Some(patch) = ReferenceLevelPatch::new(
                &pyramid.levels[level],
                level,
                feature.x * scale,
                feature.y * scale,
                window_size,
            ) {
                levels.push(patch);
            }
        }

        (!levels.is_empty()).then(|| ReferenceTrack {
            id: feature.id,
            feature: feature.clone(),
            levels,
        })
    }
}

impl ReferenceLevelPatch {
    fn new(
        image: &Image<f32>,
        level: usize,
        ref_x: f32,
        ref_y: f32,
        window_size: usize,
    ) -> Option<Self> {
        let side = 2 * window_size + 1;
        let len = side * side;
        let mut values = vec![0.0f32; len];
        let mut gx = vec![0.0f32; len];
        let mut gy = vec![0.0f32; len];
        let mut h00 = 0.0f32;
        let mut h01 = 0.0f32;
        let mut h11 = 0.0f32;
        let mut affine_h = Mat6::zeros();

        for py in 0..side {
            let oy = py as isize - window_size as isize;
            let oy_f = oy as f32;
            for px in 0..side {
                let ox = px as isize - window_size as isize;
                let ox_f = ox as f32;
                let x = ref_x + ox as f32;
                let y = ref_y + oy as f32;
                let idx = py * side + px;
                let gxi = 0.5
                    * (interpolate_bilinear(image, x + 1.0, y)
                        - interpolate_bilinear(image, x - 1.0, y));
                let gyi = 0.5
                    * (interpolate_bilinear(image, x, y + 1.0)
                        - interpolate_bilinear(image, x, y - 1.0));
                values[idx] = interpolate_bilinear(image, x, y);
                gx[idx] = gxi;
                gy[idx] = gyi;
                h00 += gxi * gxi;
                h01 += gxi * gyi;
                h11 += gyi * gyi;

                let j = Vec6::new(gxi * ox_f, gxi * oy_f, gyi * ox_f, gyi * oy_f, gxi, gyi);
                affine_h += j * j.transpose();
            }
        }

        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-6 {
            return None;
        }
        affine_h.cholesky()?;
        let inv_det = 1.0 / det;
        Some(ReferenceLevelPatch {
            level,
            ref_x,
            ref_y,
            side,
            values,
            gx,
            gy,
            ih00: inv_det * h11,
            ih01: -inv_det * h01,
            ih11: inv_det * h00,
            affine_h,
        })
    }
}

pub struct ReferenceKltTracker {
    pub window_size: usize,
    pub max_iterations: usize,
    pub epsilon: f32,
    pub max_levels: usize,
    pub compute_residual: bool,
    pub warp: ReferenceKltWarp,
}

impl ReferenceKltTracker {
    pub fn new(
        window_size: usize,
        max_iterations: usize,
        epsilon: f32,
        max_levels: usize,
        compute_residual: bool,
        warp: ReferenceKltWarp,
    ) -> Self {
        Self {
            window_size,
            max_iterations,
            epsilon,
            max_levels,
            compute_residual,
            warp,
        }
    }

    pub fn make_track(&self, feature: &Feature, pyramid: &Pyramid) -> Option<ReferenceTrack> {
        ReferenceTrack::new(feature, pyramid, self.window_size, self.max_levels)
    }

    pub fn track(
        &self,
        reference: &ReferenceTrack,
        curr_pyramid: &Pyramid,
        initial: &Feature,
    ) -> TrackedFeature {
        match self.warp {
            ReferenceKltWarp::Translation => {
                self.track_translation(reference, curr_pyramid, initial)
            }
            ReferenceKltWarp::Affine => self.track_affine(reference, curr_pyramid, initial),
        }
    }

    /// Estimate the affine warp as `[a00, a01, a10, a11, tx, ty]`.
    ///
    /// The matrix maps reference-patch offsets into current-patch offsets:
    ///
    /// `current_xy = reference_center + [tx, ty] + A * reference_offset`
    pub fn track_affine_warp(
        &self,
        reference: &ReferenceTrack,
        curr_pyramid: &Pyramid,
        initial: &Feature,
    ) -> Option<[f32; 6]> {
        let p = self.estimate_affine_parameters(reference, curr_pyramid, initial)?;
        Some([1.0 + p[0], p[1], p[2], 1.0 + p[3], p[4], p[5]])
    }

    fn track_translation(
        &self,
        reference: &ReferenceTrack,
        curr_pyramid: &Pyramid,
        initial: &Feature,
    ) -> TrackedFeature {
        let mut dx0 = initial.x - reference.feature.x;
        let mut dy0 = initial.y - reference.feature.y;

        for patch in reference.levels.iter().rev() {
            if patch.level >= curr_pyramid.num_levels() {
                continue;
            }
            let scale = 1.0 / (1u32 << patch.level) as f32;
            let mut dx = dx0 * scale;
            let mut dy = dy0 * scale;
            let image = &curr_pyramid.levels[patch.level];

            for _ in 0..self.max_iterations {
                let mut b0 = 0.0f32;
                let mut b1 = 0.0f32;
                let half = (patch.side / 2) as isize;

                for py in 0..patch.side {
                    let oy = py as isize - half;
                    for px in 0..patch.side {
                        let ox = px as isize - half;
                        let idx = py * patch.side + px;
                        let wx = patch.ref_x + dx + ox as f32;
                        let wy = patch.ref_y + dy + oy as f32;
                        let e = patch.values[idx] - interpolate_bilinear(image, wx, wy);
                        b0 += patch.gx[idx] * e;
                        b1 += patch.gy[idx] * e;
                    }
                }

                let delta_x = patch.ih00 * b0 + patch.ih01 * b1;
                let delta_y = patch.ih01 * b0 + patch.ih11 * b1;
                dx += delta_x;
                dy += delta_y;

                if delta_x * delta_x + delta_y * delta_y < self.epsilon * self.epsilon {
                    break;
                }
            }

            dx0 = dx / scale;
            dy0 = dy / scale;
        }

        let new_x = reference.feature.x + dx0;
        let new_y = reference.feature.y + dy0;
        let w = curr_pyramid.levels[0].width() as f32;
        let h = curr_pyramid.levels[0].height() as f32;
        let status = if new_x >= 0.0 && new_x < w && new_y >= 0.0 && new_y < h {
            TrackStatus::Tracked
        } else {
            TrackStatus::OutOfBounds
        };
        let residual = if status == TrackStatus::Tracked && self.compute_residual {
            level0_mean_abs_residual(reference, &curr_pyramid.levels[0], dx0, dy0)
        } else if status == TrackStatus::Tracked {
            f32::NAN
        } else {
            f32::INFINITY
        };

        TrackedFeature {
            feature: Feature {
                x: new_x,
                y: new_y,
                score: initial.score,
                level: initial.level,
                id: initial.id,
                descriptor: initial.descriptor,
            },
            status,
            residual,
        }
    }

    fn track_affine(
        &self,
        reference: &ReferenceTrack,
        curr_pyramid: &Pyramid,
        initial: &Feature,
    ) -> TrackedFeature {
        let Some(p) = self.estimate_affine_parameters(reference, curr_pyramid, initial) else {
            return TrackedFeature {
                feature: initial.clone(),
                status: TrackStatus::Lost,
                residual: f32::INFINITY,
            };
        };

        let new_x = reference.feature.x + p[4];
        let new_y = reference.feature.y + p[5];
        let w = curr_pyramid.levels[0].width() as f32;
        let h = curr_pyramid.levels[0].height() as f32;
        let status = if new_x >= 0.0 && new_x < w && new_y >= 0.0 && new_y < h {
            TrackStatus::Tracked
        } else {
            TrackStatus::OutOfBounds
        };
        let residual = if status == TrackStatus::Tracked && self.compute_residual {
            level0_mean_abs_residual_affine(reference, &curr_pyramid.levels[0], &p)
        } else if status == TrackStatus::Tracked {
            f32::NAN
        } else {
            f32::INFINITY
        };

        TrackedFeature {
            feature: Feature {
                x: new_x,
                y: new_y,
                score: initial.score,
                level: initial.level,
                id: initial.id,
                descriptor: initial.descriptor,
            },
            status,
            residual,
        }
    }

    fn estimate_affine_parameters(
        &self,
        reference: &ReferenceTrack,
        curr_pyramid: &Pyramid,
        initial: &Feature,
    ) -> Option<Vec6> {
        let mut p = Vec6::new(
            0.0,
            0.0,
            0.0,
            0.0,
            initial.x - reference.feature.x,
            initial.y - reference.feature.y,
        );

        for patch in reference.levels.iter().rev() {
            if patch.level >= curr_pyramid.num_levels() {
                continue;
            }
            let chol = patch.affine_h.cholesky()?;
            let scale = 1.0 / (1u32 << patch.level) as f32;
            let mut lp = Vec6::new(p[0], p[1], p[2], p[3], p[4] * scale, p[5] * scale);
            let image = &curr_pyramid.levels[patch.level];
            let half = (patch.side / 2) as isize;

            for _ in 0..self.max_iterations {
                let mut b = Vec6::zeros();

                for py in 0..patch.side {
                    let oy = py as isize - half;
                    let oy_f = oy as f32;
                    for px in 0..patch.side {
                        let ox = px as isize - half;
                        let ox_f = ox as f32;
                        let idx = py * patch.side + px;
                        let wx = patch.ref_x + lp[4] + (1.0 + lp[0]) * ox_f + lp[1] * oy_f;
                        let wy = patch.ref_y + lp[5] + lp[2] * ox_f + (1.0 + lp[3]) * oy_f;
                        let e = patch.values[idx] - interpolate_bilinear(image, wx, wy);
                        let j = Vec6::new(
                            patch.gx[idx] * ox_f,
                            patch.gx[idx] * oy_f,
                            patch.gy[idx] * ox_f,
                            patch.gy[idx] * oy_f,
                            patch.gx[idx],
                            patch.gy[idx],
                        );
                        b += j * e;
                    }
                }

                let delta = chol.solve(&b);
                let Some(updated) = compose_inverse_affine_update(&lp, &delta) else {
                    break;
                };
                lp = updated;
                if delta.norm_squared() < self.epsilon * self.epsilon {
                    break;
                }
            }

            p = Vec6::new(lp[0], lp[1], lp[2], lp[3], lp[4] / scale, lp[5] / scale);
        }
        Some(p)
    }
}

fn level0_mean_abs_residual(
    reference: &ReferenceTrack,
    image: &Image<f32>,
    dx: f32,
    dy: f32,
) -> f32 {
    let Some(patch) = reference.levels.iter().find(|p| p.level == 0) else {
        return f32::NAN;
    };
    let half = (patch.side / 2) as isize;
    let mut sum = 0.0f32;
    for py in 0..patch.side {
        let oy = py as isize - half;
        for px in 0..patch.side {
            let ox = px as isize - half;
            let idx = py * patch.side + px;
            let wx = patch.ref_x + dx + ox as f32;
            let wy = patch.ref_y + dy + oy as f32;
            sum += (patch.values[idx] - interpolate_bilinear(image, wx, wy)).abs();
        }
    }
    sum / patch.values.len() as f32
}

fn compose_inverse_affine_update(p: &Vec6, delta_ref_minus_cur: &Vec6) -> Option<Vec6> {
    let a = Matrix2::new(1.0 + p[0], p[1], p[2], 1.0 + p[3]);
    let t = Vector2::new(p[4], p[5]);

    // The residual convention in this module is e = T - I(W). Standard IC
    // derivations use I(W) - T, so the incremental warp to invert is negated.
    let delta_a = Matrix2::new(
        1.0 - delta_ref_minus_cur[0],
        -delta_ref_minus_cur[1],
        -delta_ref_minus_cur[2],
        1.0 - delta_ref_minus_cur[3],
    );
    let delta_t = Vector2::new(-delta_ref_minus_cur[4], -delta_ref_minus_cur[5]);

    let delta_a_inv = delta_a.try_inverse()?;
    let new_a = a * delta_a_inv;
    let new_t = t - new_a * delta_t;

    Some(Vec6::new(
        new_a[(0, 0)] - 1.0,
        new_a[(0, 1)],
        new_a[(1, 0)],
        new_a[(1, 1)] - 1.0,
        new_t[0],
        new_t[1],
    ))
}

fn level0_mean_abs_residual_affine(
    reference: &ReferenceTrack,
    image: &Image<f32>,
    p: &Vec6,
) -> f32 {
    let Some(patch) = reference.levels.iter().find(|p| p.level == 0) else {
        return f32::NAN;
    };
    let half = (patch.side / 2) as isize;
    let mut sum = 0.0f32;
    for py in 0..patch.side {
        let oy = py as isize - half;
        let oy_f = oy as f32;
        for px in 0..patch.side {
            let ox = px as isize - half;
            let ox_f = ox as f32;
            let idx = py * patch.side + px;
            let wx = patch.ref_x + p[4] + (1.0 + p[0]) * ox_f + p[1] * oy_f;
            let wy = patch.ref_y + p[5] + p[2] * ox_f + (1.0 + p[3]) * oy_f;
            sum += (patch.values[idx] - interpolate_bilinear(image, wx, wy)).abs();
        }
    }
    sum / patch.values.len() as f32
}
