// gpu/klt.rs — GPU pyramidal KLT tracker (Inverse Compositional).
//
// Design overview:
//
//   GpuKltTracker is created once (expensive shader compilation).
//   `track()` is called each frame with two GpuPyramids and a feature list.
//
//   Under the hood, `track()` dispatches one compute pass per pyramid level,
//   coarse → fine. Between passes, the displacement buffer is scaled by ×2
//   on the CPU (cheap: a few bytes, no GPU round-trip needed since the
//   buffer stays on GPU). The final pass (level 0) writes the tracked
//   positions and statuses to the results buffer, which is then read back.
//
//
// SHADER PARAMETERS BAKED AT COMPILE TIME
// ─────────────────────────────────────────
// Like the pyramid and FAST shaders, window_size is baked into the shader
// source via string substitution before compilation:
//   {{HALF}}    = window_size         (e.g. 7)
//   {{SIDE}}    = 2*window_size + 1   (e.g. 15)
//   {{PATCH}}   = SIDE²               (e.g. 225)
//   {{WG_SIZE}} = 1-D workgroup size  (e.g. 64)
//
// The per-feature patch data (template values, gradients) is stored in
// pre-allocated storage buffers indexed by [feat_idx * PATCH + pixel_idx],
// avoiding private-address-space pressure. This is critical for RPi 4
// (VideoCore VI) which silently corrupts spilled private arrays, and
// enables future wavefront-per-patch cooperation (§4).
//
//
// NEW WGPU CONCEPTS
// ─────────────────
// - **Scaling a storage buffer on CPU between dispatches**: rather than
//   reading back and re-uploading the displacement buffer, we call
//   `queue.write_buffer` directly. The GPU has already finished (after
//   `device.poll(Wait)` is replaced by submitting a pipeline barrier via
//   a dummy submit), but for simplicity we use a CPU-side intermediate
//   for the ×2 scale step.
//
//   More precisely: we submit the coarse-level dispatch, then block with
//   `device.poll(Wait)`, read back the 8n bytes, scale on CPU, write back.
//   This adds one round-trip per level but keeps the code simple. A future
//   optimisation would use a tiny "scale" compute shader to avoid the
//   CPU involvement entirely.
//
// - **Reusing a pipeline across multiple bind groups**: the pipeline is
//   compiled once; each level creates a new bind group with the correct
//   texture pair. Bind groups are cheap to create.

use crate::fast::Feature;
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::GpuPyramid;
use crate::klt::{TrackedFeature, TrackStatus};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// 1-D workgroup size for the scalar shader (1 thread = 1 feature).
/// 64 is a good default: enough parallelism, fits in a single wavefront on
/// AMD (64 lanes) and two warps on NVIDIA (32 lanes each).
const WG_SIZE: u32 = 64;

/// Default workgroup size for the warp shader (1 workgroup = 1 feature).
/// 64 is the cross-platform sweet spot:
///   - RPi 4 (VideoCore VI, subgroup=16): 4 subgroups → good latency hiding
///   - AMD RDNA3 (subgroup=64): single wavefront → barriers are free
///   - NVIDIA Turing (subgroup=32): 2 warps → good occupancy
/// Benchmarked on RPi 4, Radeon 780M, GTX 1660Ti.
const WG_WARP: u32 = 64;

/// How the Warp KLT shader interpolates the pyramid textures.
///
/// Measured on Jetson Orin Nano, EuRoC V1_01_easy (200 features, 3 levels):
/// hardware sampling cuts KLT GPU time ~18% (0.63 → 0.52 ms) and the frontend
/// ~3%. Typical position difference vs the CPU IC tracker is 0.0015 px
/// (p99 0.004 px), but ~2× as many ill-conditioned tracks run away (≈0.1% of
/// tracks; RANSAC rejected +0.6% overall). Hardware sampling was also far more
/// sensitive to the Tegra stale-texture issue (see gpu/frontend.rs SUBMISSION)
/// — it is only verified deterministic inside `SubmitStrategy::Fused`, where
/// each pyramid is built and first read in one command buffer. Hence `Manual`
/// is the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KltSampling {
    /// Hardware when the device supports filterable R32Float and the
    /// dispatch is `Warp`, manual otherwise.
    Auto,
    /// 4 `textureLoad`s + f32 weights per sample. Works everywhere. Default.
    #[default]
    Manual,
    /// One `textureSampleLevel` through a linear sampler per sample. The
    /// texture unit interpolates with reduced (~8-bit) fractional precision.
    /// Panics at construction if the device lacks `FLOAT32_FILTERABLE`.
    Hardware,
}

/// Manual bilinear body for klt_warp.wgsl (`{{BILINEAR_FN}}`).
const BILINEAR_MANUAL: &str = r#"fn bilinear(tex: texture_2d<f32>, x: f32, y: f32) -> f32 {
    let dims = textureDimensions(tex);
    let max_x = f32(dims.x) - 1.0;
    let max_y = f32(dims.y) - 1.0;
    let cx = clamp(x, 0.0, max_x);
    let cy = clamp(y, 0.0, max_y);
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let x1 = min(x0 + 1, i32(dims.x) - 1);
    let y1 = min(y0 + 1, i32(dims.y) - 1);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let v00 = textureLoad(tex, vec2<i32>(x0, y0), 0).r;
    let v10 = textureLoad(tex, vec2<i32>(x1, y0), 0).r;
    let v01 = textureLoad(tex, vec2<i32>(x0, y1), 0).r;
    let v11 = textureLoad(tex, vec2<i32>(x1, y1), 0).r;
    return v00 * (1.0 - fx) * (1.0 - fy)
         + v10 * fx          * (1.0 - fy)
         + v01 * (1.0 - fx) * fy
         + v11 * fx          * fy;
}"#;

/// Hardware bilinear body for klt_warp.wgsl (`{{BILINEAR_FN}}`).
/// Texel centres sit at (i + 0.5) / size in normalised coordinates, so pixel
/// coordinate x maps to (x + 0.5) / width. ClampToEdge reproduces the manual
/// version's coordinate clamping at the borders.
const BILINEAR_HARDWARE: &str = r#"@group(0) @binding(10) var lin_sampler: sampler;

fn bilinear(tex: texture_2d<f32>, x: f32, y: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(tex));
    return textureSampleLevel(tex, lin_sampler, (vec2<f32>(x, y) + 0.5) / dims, 0.0).r;
}"#;

/// Maximum cached bind-group sets (one per prev/curr pyramid pair). Two
/// covers a ping-ponged pyramid pair; callers that build a fresh pyramid every
/// frame keep at most this many old pyramids alive.
const BG_CACHE_CAP: usize = 2;

/// Sentinel displacement value indicating a lost feature.
/// Must match LOST_SENTINEL in klt.wgsl.

/// KLT dispatch strategy.
///
/// `Scalar`: original shader — 1 thread per feature, sequential pixel iteration.
/// `Warp`:   §4 wavefront-per-patch — 1 workgroup per feature, cooperative pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KltDispatch {
    /// 1 thread = 1 feature. Simple, works everywhere.
    Scalar,
    /// 1 workgroup = 1 feature. WG_SIZE threads cooperate on patch pixels.
    /// Requires power-of-2 workgroup size for shared memory reduction.
    Warp(u32),  // workgroup size (must be power of 2, e.g. 16, 32, 64)
}

// ---------------------------------------------------------------------------
// GPU-side structs (must match WGSL layout exactly — repr(C))
// ---------------------------------------------------------------------------

/// Feature input layout. Matches GpuFeature in fast.rs / fast.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuKltFeature {
    pub x:     f32,
    pub y:     f32,
    pub score: f32,
    pub _pad:  f32,
}

impl From<&Feature> for GpuKltFeature {
    fn from(f: &Feature) -> Self {
        GpuKltFeature { x: f.x, y: f.y, score: f.score, _pad: 0.0 }
    }
}

/// Tracking result written by the level-0 shader pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTrackResult {
    x:      f32,
    y:      f32,
    status: u32,  // 0 = Tracked, 1 = Lost, 2 = OutOfBounds
    _pad:   u32,
}

/// Uniform parameters for one level pass (must match KltParams in klt.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KltParams {
    n_features:     u32,
    max_iterations: u32,
    epsilon_sq:     f32,
    level:          u32,
    level_scale:    f32,
    img0_width:     u32,
    img0_height:    u32,
    _pad:           u32,
}

// ---------------------------------------------------------------------------
// GpuKltTracker
// ---------------------------------------------------------------------------

/// GPU inverse-compositional KLT tracker.
///
/// Create once with `GpuKltTracker::new()`; call `track()` every frame.
/// The compute pipeline is compiled at construction time.
pub struct GpuKltTracker {
    pipeline:       wgpu::ComputePipeline,
    bgl:            wgpu::BindGroupLayout,
    dispatch:       KltDispatch,
    // Linear sampler at binding 10 when hardware sampling is active.
    sampler:        Option<wgpu::Sampler>,
    pub window_size:    usize,
    pub max_iterations: usize,
    pub epsilon:        f32,
    pub max_levels:     usize,

    // Pre-allocated GPU buffers — reused every frame to avoid VRAM allocation
    // overhead. Sized for max_features at construction.
    max_features:   usize,
    patch_size:     usize,          // SIDE² = (2*window_size+1)²
    feature_buf:    wgpu::Buffer,   // STORAGE | COPY_DST  — feature positions
    disp_buf:       wgpu::Buffer,   // STORAGE | COPY_DST  — displacements (zeroed each frame)
    results_buf:    wgpu::Buffer,   // STORAGE | COPY_SRC  — track results
    rb_buf:         wgpu::Buffer,   // MAP_READ | COPY_DST — CPU readback
    params_bufs:    Vec<wgpu::Buffer>, // one UNIFORM | COPY_DST per level
    // Per-feature patch data in storage buffers (not private shader memory).
    // Each is max_features × PATCH floats. Avoids VideoCore VI private-memory
    // corruption and enables future wavefront-per-patch (§4).
    t_buf:          wgpu::Buffer,   // STORAGE — template pixel values
    gx_buf:         wgpu::Buffer,   // STORAGE — template x-gradients
    gy_buf:         wgpu::Buffer,   // STORAGE — template y-gradients
    h_inv_buf:      wgpu::Buffer,   // STORAGE — per-feature Hessian inverse (vec4)

    // State set by prepare(), consumed by record_into()/arm_readback()/collect_results().
    n_prepared:     usize,
    result_bytes_p: u64,
    disp_bytes_p:   u64,
    workgroups_p:   u32,
    // Per-level bind groups keyed by (prev level-0 texture id, curr level-0
    // texture id, level count). A frontend that ping-pongs two persistent
    // pyramids hits this cache every frame after the first two, so no bind
    // groups are created in steady state. Bounded to BG_CACHE_CAP entries;
    // the oldest entry (and the textures it keeps alive) is evicted first.
    bg_cache:       Vec<((u64, u64, usize), Vec<wgpu::BindGroup>)>,
    bg_sel:         usize,
    readback_rx:    Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuKltTracker {
    /// Create a GPU KLT tracker with the default Warp(64) dispatch.
    ///
    /// `window_size` is the patch half-width W (patch = (2W+1)²).
    /// Typical values: 4 (GAP8 / small images), 7 (vilib / HD cameras).
    pub fn new(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
    ) -> Self {
        Self::new_with_dispatch(
            gpu, window_size, max_iterations, epsilon, max_levels, max_features,
            KltDispatch::Warp(WG_WARP),
        )
    }

    /// Create a GPU KLT tracker with the default Warp(64) dispatch and an
    /// explicit sampling strategy.
    pub fn new_with_sampling(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
        sampling:       KltSampling,
    ) -> Self {
        Self::new_with_options(
            gpu, window_size, max_iterations, epsilon, max_levels, max_features,
            KltDispatch::Warp(WG_WARP), sampling,
        )
    }

    /// Create a GPU KLT tracker with the Scalar dispatch (fallback).
    /// 1 thread = 1 feature, no shared memory, no barriers.
    pub fn new_scalar(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
    ) -> Self {
        Self::new_with_dispatch(
            gpu, window_size, max_iterations, epsilon, max_levels, max_features,
            KltDispatch::Scalar,
        )
    }

    /// Create a GPU KLT tracker with explicit dispatch strategy.
    ///
    /// `dispatch`:
    ///   - `Scalar`: 1 thread per feature (klt.wgsl)
    ///   - `Warp(wg)`: 1 workgroup of `wg` threads per feature (klt_warp.wgsl)
    pub fn new_with_dispatch(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
        dispatch:       KltDispatch,
    ) -> Self {
        Self::new_with_options(
            gpu, window_size, max_iterations, epsilon, max_levels, max_features,
            dispatch, KltSampling::Manual,
        )
    }

    /// Create a GPU KLT tracker with explicit dispatch and sampling strategy.
    ///
    /// Hardware sampling applies to the `Warp` shader only; `Scalar` always
    /// interpolates manually.
    pub fn new_with_options(
        gpu:            &GpuDevice,
        window_size:    usize,
        max_iterations: usize,
        epsilon:        f32,
        max_levels:     usize,
        max_features:   usize,
        dispatch:       KltDispatch,
        sampling:       KltSampling,
    ) -> Self {
        let filterable = gpu.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE);
        let is_warp = matches!(dispatch, KltDispatch::Warp(_));
        let hw_sampling = match sampling {
            KltSampling::Auto     => filterable && is_warp,
            KltSampling::Manual   => false,
            KltSampling::Hardware => {
                assert!(filterable,
                    "KltSampling::Hardware needs FLOAT32_FILTERABLE, which this device lacks");
                assert!(is_warp, "KltSampling::Hardware needs KltDispatch::Warp");
                true
            }
        };

        let side  = 2 * window_size + 1;
        let patch = side * side;

        // Select shader source and workgroup size based on dispatch mode.
        let (shader_template, wg_size) = match dispatch {
            KltDispatch::Scalar => {
                (include_str!("../shaders/klt.wgsl"), WG_SIZE)
            }
            KltDispatch::Warp(wg) => {
                assert!(wg.is_power_of_two(), "Warp WG_SIZE must be power of 2, got {wg}");
                assert!(wg >= 4, "Warp WG_SIZE must be >= 4, got {wg}");
                (include_str!("../shaders/klt_warp.wgsl"), wg)
            }
        };

        // Bake all compile-time constants into the shader source.
        let shader_src = shader_template
            .replace("{{HALF}}",    &window_size.to_string())
            .replace("{{SIDE}}",    &side.to_string())
            .replace("{{PATCH}}",   &patch.to_string())
            .replace("{{WG_SIZE}}", &wg_size.to_string())
            .replace("{{BILINEAR_FN}}",
                if hw_sampling { BILINEAR_HARDWARE } else { BILINEAR_MANUAL });

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some(match dispatch {
                KltDispatch::Scalar => "klt.wgsl",
                KltDispatch::Warp(_) => "klt_warp.wgsl",
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout mirrors @group(0) in klt.wgsl.
        let mut bgl_entries = vec![
                // 0 — prev_tex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: hw_sampling },
                    },
                    count: None,
                },
                // 1 — curr_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: hw_sampling },
                    },
                    count: None,
                },
                // 2 — features (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3 — displacements (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4 — results (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5 — params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6 — t_buf (storage read_write: template pixel values)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7 — gx_buf (storage read_write: template x-gradients)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8 — gy_buf (storage read_write: template y-gradients)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9 — h_inv (storage read_write: per-feature Hessian inverse)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
        ];
        if hw_sampling {
            // 10 — linear sampler (hardware bilinear only)
            bgl_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuKlt BGL"),
            entries: &bgl_entries,
        });
        let sampler = hw_sampling.then(|| gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("GpuKlt linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let pipeline_layout =
            gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuKlt pipeline layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some("track_level"),
                layout:              Some(&pipeline_layout),
                module:              &shader,
                entry_point:         "track_level",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache:               None,
            });

        // Pre-allocate buffers sized for max_features.
        // write_buffer / clear_buffer are used each frame to update content —
        // both go through the queue's staging ring and avoid VRAM re-allocation.
        let feat_bytes   = (max_features * std::mem::size_of::<GpuKltFeature>()) as u64;
        let disp_bytes   = (max_features * 2 * std::mem::size_of::<f32>()) as u64;
        let result_bytes = (max_features * std::mem::size_of::<GpuTrackResult>()) as u64;

        let feature_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt features"), size: feat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let disp_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt displacements"), size: disp_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt results"), size: result_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt readback"), size: result_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One params buffer per level — content updated via write_buffer each frame.
        let params_bufs: Vec<wgpu::Buffer> = (0..max_levels)
            .map(|_| gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuKlt params"), size: std::mem::size_of::<KltParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        // Per-feature patch storage buffers: max_features × PATCH floats each.
        // These replace private-address-space arrays in the shader, avoiding
        // VideoCore VI's private memory corruption on RPi 4.
        let patch_buf_bytes = (max_features * patch * std::mem::size_of::<f32>()) as u64;
        let t_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt t_buf"), size: patch_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gx_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt gx_buf"), size: patch_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gy_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt gy_buf"), size: patch_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Per-feature Hessian inverse: vec4<f32> per feature (16 bytes each).
        let h_inv_bytes = (max_features * 4 * std::mem::size_of::<f32>()) as u64;
        let h_inv_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKlt h_inv"), size: h_inv_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        GpuKltTracker {
            pipeline, bgl, dispatch, sampler,
            window_size, max_iterations, epsilon, max_levels,
            max_features, patch_size: patch,
            feature_buf, disp_buf, results_buf, rb_buf, params_bufs,
            t_buf, gx_buf, gy_buf, h_inv_buf,
            n_prepared: 0, result_bytes_p: 0, disp_bytes_p: 0, workgroups_p: 0,
            bg_cache: Vec::new(), bg_sel: 0,
            readback_rx: None,
        }
    }

    /// Write feature data and params; build bind groups for this frame.
    ///
    /// Returns `false` if `features` is empty — caller can skip encoding entirely.
    /// Must be called before `record_into()`.
    pub fn prepare(
        &mut self,
        gpu:          &GpuDevice,
        features:     &[Feature],
        prev_pyramid: &GpuPyramid,
        curr_pyramid: &GpuPyramid,
    ) -> bool {
        if features.is_empty() {
            self.n_prepared = 0;
            return false;
        }
        let n = features.len();
        assert!(n <= self.max_features,
            "GpuKltTracker: {} features exceeds max_features={}", n, self.max_features);

        let n_u32 = n as u32;
        let num_levels = self.max_levels
            .min(prev_pyramid.levels.len())
            .min(curr_pyramid.levels.len());
        let img0_w = prev_pyramid.levels[0].width;
        let img0_h = prev_pyramid.levels[0].height;

        // Upload features via staging ring (no VRAM alloc).
        let gpu_features: Vec<GpuKltFeature> =
            features.iter().map(GpuKltFeature::from).collect();
        gpu.queue.write_buffer(&self.feature_buf, 0, bytemuck::cast_slice(&gpu_features));

        let result_bytes = (n * std::mem::size_of::<GpuTrackResult>()) as u64;
        let disp_bytes   = (n * 2 * std::mem::size_of::<f32>()) as u64;

        // Write params for each level.
        for (i, level) in (0..num_levels).rev().enumerate() {
            let params = KltParams {
                n_features:     n_u32,
                max_iterations: self.max_iterations as u32,
                epsilon_sq:     self.epsilon * self.epsilon,
                level:          level as u32,
                level_scale:    1.0f32 / (1u32 << level) as f32,
                img0_width:     img0_w,
                img0_height:    img0_h,
                _pad:           0,
            };
            gpu.queue.write_buffer(&self.params_bufs[i], 0, bytemuck::bytes_of(&params));
        }

        // Look up (or build) the per-level bind groups for this texture pair.
        let key = (
            prev_pyramid.levels[0].texture.global_id().inner(),
            curr_pyramid.levels[0].texture.global_id().inner(),
            num_levels,
        );
        self.bg_sel = match self.bg_cache.iter().position(|(k, _)| *k == key) {
            Some(pos) => pos,
            None => {
                let bind_groups: Vec<wgpu::BindGroup> = (0..num_levels).rev()
                    .enumerate()
                    .map(|(i, level)| {
                        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label:  Some("GpuKlt BG"),
                            layout: &self.bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &prev_pyramid.levels[level].read_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &curr_pyramid.levels[level].read_view),
                                },
                                wgpu::BindGroupEntry { binding: 2, resource: self.feature_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: self.disp_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 4, resource: self.results_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 5, resource: self.params_bufs[i].as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 6, resource: self.t_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 7, resource: self.gx_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 8, resource: self.gy_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 9, resource: self.h_inv_buf.as_entire_binding() },
                            ]
                            .into_iter()
                            .chain(self.sampler.as_ref().map(|s| wgpu::BindGroupEntry {
                                binding: 10,
                                resource: wgpu::BindingResource::Sampler(s),
                            }))
                            .collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                if self.bg_cache.len() >= BG_CACHE_CAP {
                    self.bg_cache.remove(0);
                }
                self.bg_cache.push((key, bind_groups));
                self.bg_cache.len() - 1
            }
        };

        self.n_prepared     = n;
        self.result_bytes_p = result_bytes;
        self.disp_bytes_p   = disp_bytes;
        // Scalar: ceil(n / WG_SIZE) workgroups, multiple features per WG.
        // Warp:   n workgroups, one feature per WG.
        self.workgroups_p   = match self.dispatch {
            KltDispatch::Scalar => (n_u32 + WG_SIZE - 1) / WG_SIZE,
            KltDispatch::Warp(_) => n_u32,
        };

        // Explicitly zero displacements — V3DV's clear_buffer may be unreliable.
        let zeros = vec![0u8; disp_bytes as usize];
        gpu.queue.write_buffer(&self.disp_buf, 0, &zeros);

        true
    }

    /// Record KLT passes into `encoder`.
    /// Must be called after `prepare()` returned `true`.
    pub fn record_into(&self, encoder: &mut wgpu::CommandEncoder) {
        assert!(self.n_prepared > 0, "call prepare() before record_into()");
        encoder.clear_buffer(&self.disp_buf, 0, Some(self.disp_bytes_p));
        for bg in &self.bg_cache[self.bg_sel].1 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("track_level"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.workgroups_p, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.results_buf, 0, &self.rb_buf, 0, self.result_bytes_p);
    }

    /// Map the readback buffer asynchronously.
    /// Must be called after `queue.submit()` and before `device.poll(Wait)`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.rb_buf
            .slice(..self.result_bytes_p)
            .map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.readback_rx = Some(rx);
    }

    /// Collect track results. Must be called after `device.poll(Wait)`.
    /// `features` must be the same slice passed to `prepare()`.
    pub fn collect_results(&mut self, features: &[Feature]) -> Vec<TrackedFeature> {
        let rx = self.readback_rx.take()
            .expect("call arm_readback() before collect_results()");
        rx.recv().unwrap().expect("KLT readback failed");

        let n = self.n_prepared;
        let mapped = self.rb_buf.slice(..self.result_bytes_p).get_mapped_range();
        let gpu_results: &[GpuTrackResult] = bytemuck::cast_slice(&mapped);
        let tracked = gpu_results[..n].iter()
            .zip(features.iter())
            .map(|(r, f)| {
                let status = match r.status {
                    0 => TrackStatus::Tracked,
                    1 => TrackStatus::Lost,
                    _ => TrackStatus::OutOfBounds,
                };
                TrackedFeature {
                    feature: Feature { x: r.x, y: r.y, score: f.score, level: f.level, id: f.id, descriptor: f.descriptor },
                    status,
                    residual: f32::NAN,
                }
            })
            .collect();
        drop(mapped);
        self.rb_buf.unmap();
        tracked
    }

    /// Convenience wrapper: prepare + record + submit + poll + collect.
    /// Use for standalone tracking (tests, benchmarks).
    /// For pipeline fusion use `prepare` → `record_into` → `arm_readback` → `collect_results`.
    pub fn track(
        &mut self,
        gpu:          &GpuDevice,
        prev_pyramid: &GpuPyramid,
        curr_pyramid: &GpuPyramid,
        features:     &[Feature],
    ) -> Vec<TrackedFeature> {
        if !self.prepare(gpu, features, prev_pyramid, curr_pyramid) {
            return Vec::new();
        }
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("GpuKlt standalone") });
        self.record_into(&mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        self.arm_readback();
        gpu.device.poll(wgpu::Maintain::Wait);
        self.collect_results(features)
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pyramid::GpuPyramidPipeline;
    use crate::image::Image;
    use crate::pyramid::Pyramid;

    fn run_gpu_test(test_name: &str) -> String {
        let output = std::process::Command::new("cargo")
            .args([
                "test", "--lib", "--",
                test_name, "--exact", "--ignored", "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("subprocess failed: {e}"));
        let out = String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr);
        print!("{out}");
        out
    }

    // ---- helpers -----------------------------------------------------------

    /// Bright square on dark background — the standard test scene.
    fn make_test_image(w: usize, h: usize, sq_x: usize, sq_y: usize, sq_size: usize) -> Image<u8> {
        let mut img = Image::from_vec(w, h, vec![30u8; w * h]);
        for y in sq_y..(sq_y + sq_size).min(h) {
            for x in sq_x..(sq_x + sq_size).min(w) {
                img.set(x, y, 200);
            }
        }
        img
    }

    /// Make a feature at (x, y) for use in track() calls.
    fn feat(x: f32, y: f32) -> Feature {
        Feature { x, y, score: 100.0, level: 0, id: 1, descriptor: 0 }
    }

    // ---- inner GPU tests (subprocess-isolated) ----------------------------

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_zero_motion() {
        let img = make_test_image(120, 120, 40, 40, 30);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &img, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let results = tracker.track(&gpu, &pyr, &pyr, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked);
        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!(dx.abs() < 0.5 && dy.abs() < 0.5,
            "zero motion: ({dx:.3}, {dy:.3}) should be ~0");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_horizontal_shift() {
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 40, 30); // shifted right 3px
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked,
            "status = {:?}", results[0].status);
        let dx = results[0].feature.x - 41.0;
        let dy = results[0].feature.y - 41.0;
        assert!((dx - 3.0).abs() < 1.5, "horizontal shift: dx={dx:.3}, expected ~3");
        assert!(dy.abs() < 1.5,         "horizontal shift: dy={dy:.3}, expected ~0");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_flat_region_is_lost() {
        let img = Image::from_vec(60, 60, vec![128u8; 3600]);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr = pipeline.build(&gpu, &img, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 5, 30, 0.01, 3, 256);

        let features = vec![feat(30.0, 30.0)];
        let results = tracker.track(&gpu, &pyr, &pyr, &features);

        assert_eq!(results[0].status, TrackStatus::Lost,
            "flat region should be Lost");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_matches_cpu() {
        // Both GPU (IC) and CPU (IC) should recover the same displacement on a
        // clean synthetic shift. Tolerance: 0.5 pixels (same as CPU-only test).
        let img1 = make_test_image(120, 120, 40, 40, 30);
        let img2 = make_test_image(120, 120, 43, 42, 30); // +3, +2

        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(41.0, 41.0)];
        let gpu_results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        // CPU reference (IC).
        use crate::klt::{KltTracker, LkMethod};
        let cpu_pyr1 = Pyramid::build(&img1, 3, 1.0);
        let cpu_pyr2 = Pyramid::build(&img2, 3, 1.0);
        let cpu_tracker = KltTracker::with_method(7, 30, 0.01, 3, LkMethod::InverseCompositional);
        let cpu_results = cpu_tracker.track(&cpu_pyr1, &cpu_pyr2, &features);

        eprintln!("[test] GPU: ({:.3}, {:.3}) status={:?}",
            gpu_results[0].feature.x, gpu_results[0].feature.y, gpu_results[0].status);
        eprintln!("[test] CPU: ({:.3}, {:.3}) status={:?}",
            cpu_results[0].feature.x, cpu_results[0].feature.y, cpu_results[0].status);

        assert_eq!(gpu_results[0].status, TrackStatus::Tracked);
        assert_eq!(cpu_results[0].status, TrackStatus::Tracked);

        let gpu_dx = gpu_results[0].feature.x - 41.0;
        let cpu_dx = cpu_results[0].feature.x - 41.0;
        let gpu_dy = gpu_results[0].feature.y - 41.0;
        let cpu_dy = cpu_results[0].feature.y - 41.0;

        assert!((gpu_dx - cpu_dx).abs() < 0.5,
            "dx mismatch: GPU={gpu_dx:.3} CPU={cpu_dx:.3}");
        assert!((gpu_dy - cpu_dy).abs() < 0.5,
            "dy mismatch: GPU={gpu_dy:.3} CPU={cpu_dy:.3}");

        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_subpixel_shift() {
        // Gaussian blob shifted by (1.5, 0.5) — tests sub-pixel accuracy.
        let w = 80usize;
        let h = 80usize;
        let mut d1 = vec![0u8; w * h];
        let mut d2 = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let r1 = (x as f32 - 40.0).powi(2) + (y as f32 - 40.0).powi(2);
                d1[y * w + x] = (255.0 * (-0.005 * r1).exp()) as u8;
                let r2 = (x as f32 - 41.5).powi(2) + (y as f32 - 40.5).powi(2);
                d2[y * w + x] = (255.0 * (-0.005 * r2).exp()) as u8;
            }
        }

        let img1 = Image::from_vec(w, h, d1);
        let img2 = Image::from_vec(w, h, d2);
        let gpu = GpuDevice::new().unwrap();
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let pyr1 = pipeline.build(&gpu, &img1, 3, 1.0);
        let pyr2 = pipeline.build(&gpu, &img2, 3, 1.0);
        let mut tracker = GpuKltTracker::new(&gpu, 7, 30, 0.01, 3, 256);

        let features = vec![feat(40.0, 40.0)];
        let results = tracker.track(&gpu, &pyr1, &pyr2, &features);

        assert_eq!(results[0].status, TrackStatus::Tracked);
        let dx = results[0].feature.x - 40.0;
        let dy = results[0].feature.y - 40.0;
        assert!((dx - 1.5).abs() < 0.5, "subpixel dx={dx:.3}, expected ~1.5");
        assert!((dy - 0.5).abs() < 0.5, "subpixel dy={dy:.3}, expected ~0.5");
        println!("GPU_TEST_OK");
        drop(tracker); drop(pyr1); drop(pyr2); drop(pipeline); drop(gpu);
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_hardware_sampling_matches_manual() {
        // Hardware (texture-unit) bilinear must agree with the manual 4-tap
        // path to well within a pixel on a clean sub-pixel shift. Pyramids are
        // rebuilt and consumed in the same command buffer, as in the Fused
        // frontend (the only structure verified on Tegra).
        let gpu = GpuDevice::new().unwrap();
        if !gpu.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            eprintln!("[test] FLOAT32_FILTERABLE unsupported — skipping");
            println!("GPU_TEST_OK");
            return;
        }
        let (w, h) = (96usize, 96usize);
        let blob = |cx: f32, cy: f32| {
            let mut d = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let r = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                    d[y * w + x] = (40.0 + 200.0 * (-0.004 * r).exp()) as u8;
                }
            }
            Image::from_vec(w, h, d)
        };
        let (img1, img2) = (blob(48.0, 48.0), blob(49.3, 47.6));
        let pipeline = GpuPyramidPipeline::new(&gpu);
        let (p1, p2) = (pipeline.allocate(&gpu, w, h, 3), pipeline.allocate(&gpu, w, h, 3));
        let features = vec![feat(40.0, 44.0), feat(52.0, 50.0), feat(46.0, 55.0)];

        let mut run = |sampling: KltSampling| {
            let mut t = GpuKltTracker::new_with_sampling(&gpu, 7, 30, 0.01, 3, 16, sampling);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            pipeline.record_rebuild(&gpu, &mut enc, &img1, &p1);
            pipeline.record_rebuild(&gpu, &mut enc, &img2, &p2);
            assert!(t.prepare(&gpu, &features, &p1, &p2));
            t.record_into(&mut enc);
            gpu.queue.submit(std::iter::once(enc.finish()));
            t.arm_readback();
            gpu.device.poll(wgpu::Maintain::Wait);
            t.collect_results(&features)
        };
        let manual = run(KltSampling::Manual);
        let hardware = run(KltSampling::Hardware);
        for (m, hw) in manual.iter().zip(&hardware) {
            assert_eq!(m.status, TrackStatus::Tracked);
            assert_eq!(hw.status, TrackStatus::Tracked);
            let (dx, dy) = (hw.feature.x - m.feature.x, hw.feature.y - m.feature.y);
            eprintln!("[test] manual ({:.4}, {:.4})  hardware ({:.4}, {:.4})",
                m.feature.x, m.feature.y, hw.feature.x, hw.feature.y);
            assert!(dx.abs() < 0.05 && dy.abs() < 0.05,
                "hardware vs manual differ by ({dx:.4}, {dy:.4}) px");
        }
        println!("GPU_TEST_OK");
    }

    // ---- outer wrappers (non-GPU CI, spawn subprocess) --------------------

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_hardware_sampling_matches_manual() {
        let out = run_gpu_test("gpu::klt::tests::inner_hardware_sampling_matches_manual");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_zero_motion() {
        let out = run_gpu_test("gpu::klt::tests::inner_zero_motion");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_horizontal_shift() {
        let out = run_gpu_test("gpu::klt::tests::inner_horizontal_shift");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_flat_region_is_lost() {
        let out = run_gpu_test("gpu::klt::tests::inner_flat_region_is_lost");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_matches_cpu() {
        let out = run_gpu_test("gpu::klt::tests::inner_gpu_matches_cpu");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_subpixel_shift() {
        let out = run_gpu_test("gpu::klt::tests::inner_subpixel_shift");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
