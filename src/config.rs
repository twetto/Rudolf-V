//! The frontend's and the stereo matcher's settings as a YAML config file
//! spells them, and their mapping to [`FrontendConfig`] and [`StereoConfig`].
//!
//! [`FrontendSection`] is the `RudolfV:` section of an ECHO-LI config file,
//! with EqVIO's camelCase keys; [`StereoSection`] is its `Stereo:` section,
//! with snake_case keys. An application embeds them in its own config type,
//! so every program reading the same file sets up the same frontend.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::frontend::{DetectorType, FrontendConfig, LbpPolicy};
use crate::histeq::HistEqMethod;
use crate::klt::LkMethod;
use crate::stereo::StereoConfig;

/// A setting whose value names nothing this version supports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// The key, with its section: `RudolfV.histeq`, say.
    pub key: &'static str,
    pub value: String,
    /// The values that are supported.
    pub expected: &'static str,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported {} '{}'; expected {}",
            self.key, self.value, self.expected
        )
    }
}

impl std::error::Error for ConfigError {}

/// The `RudolfV:` section: the frontend's settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FrontendSection {
    /// Equalize the histogram before tracking. Used only when `histeq` is
    /// absent: true means global equalization.
    pub equalise_image_histogram: bool,
    /// Histogram equalization: "none", "global" or "clahe".
    #[serde(default)]
    pub histeq: Option<String>,
    /// CLAHE tile size in pixels (used when histeq == "clahe").
    #[serde(default = "default_clahe_tile_size")]
    pub clahe_tile_size: usize,
    /// CLAHE clip limit (used when histeq == "clahe").
    #[serde(default = "default_clahe_clip_limit")]
    pub clahe_clip_limit: f32,
    /// Detection cell size in pixels: one feature per cell at most.
    pub feature_dist: f64,
    /// EqVIO's GIFT setting; read, but nothing uses it.
    #[serde(default)]
    pub feature_search_threshold: Option<f64>,
    #[serde(default)]
    pub fast_threshold: Option<u8>,
    /// Corner detector: "fast" (the default), "harris" or "shi_tomasi".
    #[serde(default)]
    pub detector: Option<String>,
    /// Shi-Tomasi minimum-eigenvalue floor (only used when detector == shi_tomasi).
    #[serde(default)]
    pub shi_tomasi_threshold: Option<f32>,
    /// Shi-Tomasi structure-tensor block size (only used when detector == shi_tomasi).
    #[serde(default)]
    pub shi_tomasi_block_size: Option<usize>,
    /// "SoftPenalty" (the default) or "HardReject".
    #[serde(default)]
    pub lbp_policy: Option<String>,
    /// Compute the level-0 patch residual per track to feed the KLT quality
    /// term in the reservoir score (extra patch pass; off by default).
    #[serde(default)]
    pub klt_residual: bool,
    /// Run the frontend essential-matrix RANSAC geometric verification. On by
    /// default, but it degenerates under rotation-dominant motion (translation
    /// ~0 => E ill-posed) and falsely rejects good tracks.
    #[serde(default = "default_true")]
    pub enable_ransac: bool,
    /// Squared-Sampson threshold (normalized coordinates) for the pose-prior
    /// epipolar gate. Active only on frames where a relative-pose prior is
    /// supplied (`Frontend::set_pose_prior`); replaces RANSAC there. 0 disables.
    #[serde(default)]
    pub epipolar_gate_threshold: f64,
    /// Consensus refit of the prior gate (kept only if it increases inliers).
    #[serde(default)]
    pub epipolar_refine: bool,
    /// Skip the gate below this inter-frame baseline [m]: near-zero translation
    /// makes the prior's E pure noise.
    #[serde(default = "default_epipolar_min_baseline")]
    pub epipolar_min_baseline: f64,
    /// Distrust the prior when it would reject more than this fraction of
    /// tracks (breaks the reject->starve->diverge feedback loop).
    #[serde(default = "default_epipolar_max_reject_frac")]
    pub epipolar_max_reject_frac: f64,
    pub max_features: usize,
    /// Pyramid levels.
    pub max_level: usize,
}

fn default_true() -> bool {
    true
}

fn default_clahe_tile_size() -> usize {
    256
}

fn default_clahe_clip_limit() -> f32 {
    4.0
}

fn default_epipolar_min_baseline() -> f64 {
    1e-3
}

fn default_epipolar_max_reject_frac() -> f64 {
    0.5
}

impl FrontendSection {
    /// The frontend these settings describe, on [`FrontendConfig::default`]
    /// with inverse-compositional KLT. The camera is left unset.
    pub fn to_frontend_config(&self) -> Result<FrontendConfig, ConfigError> {
        let mut config = FrontendConfig {
            max_features: self.max_features,
            pyramid_levels: self.max_level,
            cell_size: self.feature_dist as usize,
            klt_method: LkMethod::InverseCompositional,
            klt_residual_enabled: self.klt_residual,
            enable_internal_ransac: self.enable_ransac,
            epipolar_gate_threshold: self.epipolar_gate_threshold,
            epipolar_refine: self.epipolar_refine,
            epipolar_min_baseline: self.epipolar_min_baseline,
            epipolar_max_reject_frac: self.epipolar_max_reject_frac,
            ..FrontendConfig::default()
        };
        if let Some(t) = self.fast_threshold {
            config.fast_threshold = t;
        }
        if let Some(name) = &self.detector {
            config.detector = parse_detector(name)?;
        }
        if let Some(v) = self.shi_tomasi_threshold {
            config.shi_tomasi_threshold = v;
        }
        if let Some(v) = self.shi_tomasi_block_size {
            config.shi_tomasi_block_size = v;
        }
        if let Some(name) = &self.lbp_policy {
            config.lbp_policy = parse_lbp_policy(name)?;
        }
        config.histeq = match &self.histeq {
            Some(name) => parse_histeq(name, self.clahe_tile_size, self.clahe_clip_limit)?,
            None if self.equalise_image_histogram => HistEqMethod::Global,
            None => HistEqMethod::None,
        };
        Ok(config)
    }
}

/// The `Stereo:` section: the stereo matcher's settings, each one optional
/// over [`StereoConfig::default`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StereoSection {
    /// Whether the application matches stereo at all; the matcher's
    /// settings do not depend on it.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub pyramid_levels: Option<usize>,
    #[serde(default)]
    pub patch_half_size: Option<usize>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub convergence_eps: Option<f64>,
    #[serde(default)]
    pub min_inv_depth: Option<f64>,
    #[serde(default)]
    pub max_inv_depth: Option<f64>,
    #[serde(default)]
    pub init_inv_depth: Option<f64>,
    #[serde(default)]
    pub max_residual: Option<f32>,
    #[serde(default)]
    pub n_search_candidates: Option<usize>,
    #[serde(default)]
    pub knn_propagation: Option<usize>,
    /// Histogram equalization of the cam1 image: "none" or "global".
    #[serde(default)]
    pub histeq: Option<String>,
}

impl StereoSection {
    /// The matcher these settings describe.
    pub fn to_stereo_config(&self) -> Result<StereoConfig, ConfigError> {
        let d = StereoConfig::default();
        Ok(StereoConfig {
            pyramid_levels: self.pyramid_levels.unwrap_or(d.pyramid_levels),
            patch_half_size: self.patch_half_size.unwrap_or(d.patch_half_size),
            max_iterations: self.max_iterations.unwrap_or(d.max_iterations),
            convergence_eps: self.convergence_eps.unwrap_or(d.convergence_eps),
            min_inv_depth: self.min_inv_depth.unwrap_or(d.min_inv_depth),
            max_inv_depth: self.max_inv_depth.unwrap_or(d.max_inv_depth),
            init_inv_depth: self.init_inv_depth.unwrap_or(d.init_inv_depth),
            max_residual: self.max_residual.unwrap_or(d.max_residual),
            n_search_candidates: self.n_search_candidates.unwrap_or(d.n_search_candidates),
            knn_propagation: self.knn_propagation.unwrap_or(d.knn_propagation),
            histeq: match self
                .histeq
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                None | Some("none") => HistEqMethod::None,
                Some("global") => HistEqMethod::Global,
                Some(_) => {
                    return Err(ConfigError {
                        key: "Stereo.histeq",
                        value: self.histeq.clone().unwrap_or_default(),
                        expected: "none or global",
                    });
                }
            },
            ..d
        })
    }
}

/// A detector by name: "fast", "harris" or "shi_tomasi" (also "shitomasi",
/// "shi-tomasi"), in any case.
pub fn parse_detector(name: &str) -> Result<DetectorType, ConfigError> {
    match name.to_ascii_lowercase().as_str() {
        "fast" => Ok(DetectorType::Fast),
        "harris" => Ok(DetectorType::Harris),
        "shi_tomasi" | "shitomasi" | "shi-tomasi" => Ok(DetectorType::ShiTomasi),
        _ => Err(ConfigError {
            key: "RudolfV.detector",
            value: name.to_string(),
            expected: "fast, harris or shi_tomasi",
        }),
    }
}

/// An LBP policy by name: "SoftPenalty" or "HardReject" (also with `_` or
/// `-` between the words, or "soft" and "hard"), in any case.
pub fn parse_lbp_policy(name: &str) -> Result<LbpPolicy, ConfigError> {
    match name.to_ascii_lowercase().as_str() {
        "softpenalty" | "soft_penalty" | "soft-penalty" | "soft" => Ok(LbpPolicy::SoftPenalty),
        "hardreject" | "hard_reject" | "hard-reject" | "hard" => Ok(LbpPolicy::HardReject),
        _ => Err(ConfigError {
            key: "RudolfV.lbpPolicy",
            value: name.to_string(),
            expected: "SoftPenalty or HardReject",
        }),
    }
}

/// Histogram equalization by name: "none", "global" or "clahe", in any case.
/// The CLAHE settings are used only for "clahe".
pub fn parse_histeq(
    name: &str,
    clahe_tile_size: usize,
    clahe_clip_limit: f32,
) -> Result<HistEqMethod, ConfigError> {
    match name.to_ascii_lowercase().as_str() {
        "none" => Ok(HistEqMethod::None),
        "global" => Ok(HistEqMethod::Global),
        "clahe" => Ok(HistEqMethod::Clahe {
            tile_size: clahe_tile_size,
            clip_limit: clahe_clip_limit,
        }),
        _ => Err(ConfigError {
            key: "RudolfV.histeq",
            value: name.to_string(),
            expected: "none, global or clahe",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `RudolfV:` and `Stereo:` sections of echo-li's
    /// configs/eqvio_midair_stereo.yaml, without the comments.
    const MIDAIR: &str = "
RudolfV:
  equaliseImageHistogram: true
  featureDist: 16.0
  featureSearchThreshold: 0.8
  fastThreshold: 20
  detector: shi_tomasi
  maxFeatures: 300
  maxLevel: 3
  lbpPolicy: HardReject
  kltResidual: false
  enableRansac: true
  epipolarGateThreshold: 1.0e-5
  epipolarRefine: false
Stereo:
  enabled: true
  pyramid_levels: 3
  patch_half_size: 4
  max_iterations: 30
  n_search_candidates: 0
  knn_propagation: 5
  max_residual: 20.0
  histeq: global
";

    #[derive(Deserialize)]
    struct File {
        #[serde(rename = "RudolfV")]
        rudolf_v: FrontendSection,
        #[serde(rename = "Stereo")]
        stereo: StereoSection,
    }

    /// A minimal `RudolfV:` section plus `lines`.
    fn frontend(equalise: bool, lines: &str) -> Result<FrontendConfig, ConfigError> {
        let yaml = format!(
            "equaliseImageHistogram: {equalise}\nfeatureDist: 16\nmaxFeatures: 50\nmaxLevel: 3\n{lines}"
        );
        serde_yaml::from_str::<FrontendSection>(&yaml)
            .unwrap()
            .to_frontend_config()
    }

    #[test]
    fn frontend_section_maps_every_setting() {
        let file: File = serde_yaml::from_str(MIDAIR).unwrap();
        let c = file.rudolf_v.to_frontend_config().unwrap();
        let d = FrontendConfig::default();
        assert_eq!(c.max_features, 300);
        assert_eq!(c.pyramid_levels, 3);
        assert_eq!(c.cell_size, 16);
        assert_eq!(c.fast_threshold, 20);
        assert_eq!(c.detector, DetectorType::ShiTomasi);
        assert_eq!(c.shi_tomasi_threshold, d.shi_tomasi_threshold);
        assert_eq!(c.lbp_policy, LbpPolicy::HardReject);
        assert_eq!(c.histeq, HistEqMethod::Global);
        assert!(c.enable_internal_ransac && !c.klt_residual_enabled && !c.epipolar_refine);
        assert_eq!(c.epipolar_gate_threshold, 1.0e-5);
        assert_eq!(c.epipolar_min_baseline, d.epipolar_min_baseline);
        assert_eq!(c.klt_method, LkMethod::InverseCompositional);
        assert!(c.camera.is_none());
    }

    #[test]
    fn histeq_name_wins_over_the_legacy_flag() {
        assert_eq!(frontend(false, "").unwrap().histeq, HistEqMethod::None);
        assert_eq!(frontend(true, "").unwrap().histeq, HistEqMethod::Global);
        assert_eq!(
            frontend(false, "histeq: global").unwrap().histeq,
            HistEqMethod::Global
        );
        assert_eq!(
            frontend(true, "histeq: none").unwrap().histeq,
            HistEqMethod::None
        );
        assert_eq!(
            frontend(
                false,
                "histeq: CLAHE\nclaheTileSize: 32\nclaheClipLimit: 2.0"
            )
            .unwrap()
            .histeq,
            HistEqMethod::Clahe {
                tile_size: 32,
                clip_limit: 2.0
            }
        );
    }

    #[test]
    fn unknown_names_and_keys_are_errors() {
        let e = frontend(false, "histeq: adaptive").err().unwrap();
        assert_eq!(
            e.to_string(),
            "unsupported RudolfV.histeq 'adaptive'; expected none, global or clahe"
        );
        assert_eq!(
            frontend(false, "detector: orb").err().unwrap().key,
            "RudolfV.detector"
        );
        assert_eq!(
            frontend(false, "lbpPolicy: maybe").err().unwrap().key,
            "RudolfV.lbpPolicy"
        );
        assert_eq!(
            frontend(false, "lbpPolicy: hard").unwrap().lbp_policy,
            LbpPolicy::HardReject
        );
        let typo = "equaliseImageHistogram: false\nfeatureDist: 16\nmaxFeatures: 50\nmaxLevel: 3\nmaxFeature: 5";
        assert!(serde_yaml::from_str::<FrontendSection>(typo).is_err());
    }

    #[test]
    fn stereo_section_overrides_defaults() {
        let file: File = serde_yaml::from_str(MIDAIR).unwrap();
        assert!(file.stereo.enabled);
        let c = file.stereo.to_stereo_config().unwrap();
        let d = StereoConfig::default();
        assert_eq!(c.pyramid_levels, 3);
        assert_eq!(c.max_iterations, 30);
        assert_eq!(c.n_search_candidates, 0);
        assert_eq!(c.convergence_eps, d.convergence_eps);
        assert_eq!(c.huber_delta, d.huber_delta);
        assert_eq!(c.histeq, HistEqMethod::Global);
        let clahe = StereoSection {
            histeq: Some("clahe".into()),
            ..Default::default()
        };
        assert_eq!(clahe.to_stereo_config().err().unwrap().key, "Stereo.histeq");
    }
}
