// gpu/dense_stereo.rs — Dense stereo on the GPU (WGSL), stage for stage the
// CPU reference's (dense_stereo.rs, which explains the method), on a
// rectified pair: raw frames go through the rectifying GPU pyramid
// (`StereoRectification`, from `DenseStereo::new_rectifying`), rectified ones
// through the plain one.
//
// Per level, coarsest first: preparation (gradients, patch sums), the
// coarsest level's exhaustive search or a finer level's start from the
// coarser D, red-black propagation with Gauss-Newton, densification with
// patch validity, the variational refinement (finest level), and on
// validated levels the clamp, occlusion test and speckle filter. Then the
// coarse fallback, the check of its fills, and the conversion to log-range,
// read back once per frame (`record_into`, `arm_readback`, `collect_into`).
//
// The patch kernels run one patch per thread and read the images from
// storage buffers (dense_stereo_patch.wgsl explains why); per-pixel kernels
// one pixel per thread. Results match the CPU's up to the order of
// floating-point sums, not bit for bit.

use crate::dense_stereo::{
    AMBIGUITY_RATIO, CANDIDATE_TRUNCATION, DISAGREE, DISTINCT_RISE, DISTINCT_STEP, DenseDepth,
    DenseStereo, EPS2, FALLBACK_AGREE, FLAT_END_MARGIN, FLAT_GRADIENT, GN_MIN_STEP, LevelSetup,
    MAX_COVER, NO_SOURCE, NOISE_FLOOR, SAME_CANDIDATE, SPECKLE_STEP, ZETA2,
};
use crate::gpu::device::GpuDevice;
use crate::gpu::pyramid::{GpuPyramid, GpuPyramidPipeline};
use crate::image::Image;
use wgpu::util::DeviceExt;

/// One level's dispatch parameters (dense_stereo_common.wgsl `Params`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    ws: u32,
    hs: u32,
    stride: u32,
    colour: u32,
    per_row: u32,
    iters: u32,
    dmin: f32,
    dmax: f32,
    mean_norm: u32,
    cw: u32,
    ch: u32,
    validate: u32,
    has_init: u32,
    min_speckle: u32,
    min_hess: f32,
    max_ssd: f32,
    max_bright: f32,
    min_rise: f32,
}

/// The variational refinement's parameters (dense_stereo_refine.wgsl
/// `RefineParams`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RefineParams {
    w: u32,
    h: u32,
    hw: u32,
    stride: u32,
    size: u32,
    colour: u32,
    masked: u32,
    _pad0: u32,
    alpha2: f32,
    gamma2: f32,
    delta2: f32,
    omega: f32,
    inv_edge: f32,
    _pad1: [f32; 3],
}

/// The coarse fallback's and the output's parameters
/// (dense_stereo_finish.wgsl `FinishParams`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FinishParams {
    w: u32,
    h: u32,
    cw: u32,
    ch: u32,
    k: u32,
    level: u32,
    finest: u32,
    first: u32,
    reach: u32,
    has_fallback: u32,
    flat_gradient: f32,
    max_residual: f32,
    ln_s: f32,
    _pad: [u32; 3],
}

/// A patch's estimate as the GPU keeps it (dense_stereo_common.wgsl
/// `Patch`): `PatchState` with propagation's moved and settled flags.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuPatch {
    pub d: f32,
    pub ssd: f32,
    pub cost: f32,
    pub hess: f32,
    pub shift_x: f32,
    pub shift_y: f32,
    pub offset: f32,
    pub flags: u32,
}

#[cfg_attr(not(test), allow(dead_code))]
impl GpuPatch {
    pub const OK: u32 = 1;
    pub const INSIDE: u32 = 2;
    pub const MOVED: u32 = 4;
    pub const SETTLED: u32 = 8;
}

/// Patch side the kernels are written for.
const PSZ: usize = 8;

/// One level's buffers and bind groups. (The tests read the intermediate
/// ones.)
#[cfg_attr(not(test), allow(dead_code))]
struct Level {
    setup: LevelSetup,
    _params: [wgpu::Buffer; 2],
    grad: wgpu::Buffer,
    /// The level's images as buffers: the left grey level with gx, and the
    /// right level.
    tpl: wgpu::Buffer,
    right: wgpu::Buffer,
    sums: wgpu::Buffer,
    _patch_n: wgpu::Buffer,
    /// Per pixel, the right image's shift per unit of D.
    pixel_n: wgpu::Buffer,
    patches: wgpu::Buffer,
    _d_init: wgpu::Buffer,
    /// Per patch: inverse variance and validity.
    pinfo: wgpu::Buffer,
    /// The dense D, its validity (u32) and variance.
    dense_d: wgpu::Buffer,
    dense_valid: wgpu::Buffer,
    dense_var: wgpu::Buffer,
    gradients_bg: wgpu::BindGroup,
    pack_bg: wgpu::BindGroup,
    sums_bg: wgpu::BindGroup,
    upsample_bg: Option<wgpu::BindGroup>,
    init_bg: wgpu::BindGroup,
    /// Per propagation colour (the search uses colour 0's).
    patch_bg: [wgpu::BindGroup; 2],
    validity_bg: wgpu::BindGroup,
    densify_bg: wgpu::BindGroup,
    /// The mask's (validated levels).
    mask_bg: Option<wgpu::BindGroup>,
    /// The speckle filter's smallest region (pixels of the level).
    min_speckle: u32,
}

/// One level of the coarse fallback.
struct Fallback {
    level: usize,
    _params: wgpu::Buffer,
    disagree_bg: wgpu::BindGroup,
    dilate_bg: wgpu::BindGroup,
    fill_bg: wgpu::BindGroup,
}

/// The finest level's output: the fills' check, the conversion, and the
/// readback.
struct Output {
    _params: wgpu::Buffer,
    _source: wgpu::Buffer,
    reject_bg: Option<wgpu::BindGroup>,
    finish_bg: wgpu::BindGroup,
    /// log-range bits, variance bits, sources four to a word.
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
}

/// The variational refinement of one level: its buffers (the solver's split
/// by colour) and bind groups.
struct Refine {
    level: usize,
    iterations: usize,
    sor_iterations: usize,
    _params: [wgpu::Buffer; 2],
    _ai: wgpu::Buffer,
    _dif: wgpu::Buffer,
    _terms: wgpu::Buffer,
    _du: wgpu::Buffer,
    _wts: wgpu::Buffer,
    _ac: wgpu::Buffer,
    warp_bg: wgpu::BindGroup,
    diffs_bg: wgpu::BindGroup,
    terms_bg: wgpu::BindGroup,
    update_bg: wgpu::BindGroup,
    /// Per colour.
    weights_bg: [wgpu::BindGroup; 2],
    data_bg: [wgpu::BindGroup; 2],
    sor_bg: [wgpu::BindGroup; 2],
}

struct Pipelines {
    gradients: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    patch_sums: wgpu::ComputePipeline,
    upsample: wgpu::ComputePipeline,
    init_patches: wgpu::ComputePipeline,
    search: wgpu::ComputePipeline,
    propagate: wgpu::ComputePipeline,
    validity: wgpu::ComputePipeline,
    densify: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    diffs: wgpu::ComputePipeline,
    solver_terms: wgpu::ComputePipeline,
    weights: wgpu::ComputePipeline,
    data: wgpu::ComputePipeline,
    sor: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    clamp_range: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    occlude: wgpu::ComputePipeline,
    unite: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    speckles: wgpu::ComputePipeline,
    disagree: wgpu::ComputePipeline,
    dilate: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    reject: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
}

/// Dense stereo on the GPU, for a CPU matcher's configuration and geometry.
pub struct GpuDenseStereo {
    finest: usize,
    coarsest: usize,
    sweeps: usize,
    occlusion: bool,
    /// Indexed by level − finest.
    levels: Vec<Level>,
    /// The finest level's variational refinement (none with 0 iterations).
    refine: Option<Refine>,
    /// The coarse fallback's levels, finest first.
    fallback: Vec<Fallback>,
    output: Output,
    /// The mask's z-buffer and union-find, and the fallback's cell flags,
    /// shared by the levels.
    _scratch: [wgpu::Buffer; 5],
    pyramid: GpuPyramidPipeline,
    pyr: [GpuPyramid; 2],
    pipelines: Pipelines,
    readback_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn res(b: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    b.as_entire_binding()
}

fn tex(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

/// A compute pipeline for `entry` of `module`, with one bind group of
/// `entries`.
fn pipeline(
    gpu: &GpuDevice,
    module: &wgpu::ShaderModule,
    entry: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::ComputePipeline {
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(entry),
            entries,
        });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    gpu.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

/// A shader module from dense_stereo_common.wgsl and `src`, with each
/// `{{TOKEN}}` of `tokens` replaced.
fn module(
    gpu: &GpuDevice,
    label: &str,
    src: &str,
    tokens: &[(&str, String)],
) -> wgpu::ShaderModule {
    let mut wgsl = [include_str!("../shaders/dense_stereo_common.wgsl"), src].concat();
    for (token, value) in tokens {
        wgsl = wgsl.replace(&format!("{{{{{token}}}}}"), value);
    }
    gpu.create_compute_shader(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    })
}

impl GpuDenseStereo {
    /// The GPU matcher for `cpu`'s configuration and geometry: a rectified
    /// rig, or `new_rectifying`'s rectification (the GPU pyramid rectifies
    /// raw frames then). Patches must be 8×8.
    pub fn new(gpu: &GpuDevice, cpu: &DenseStereo) -> Self {
        let cfg = cpu.config();
        assert_eq!(cfg.patch_size, PSZ, "the GPU matcher needs 8×8 patches");
        let setups = cpu.level_setups();
        let (finest, coarsest) = (cfg.finest_level, cpu.coarsest_level());
        let last_validated = (finest + cfg.coarse_fallback).min(coarsest);
        let sweeps = cfg.propagation_sweeps.max(1);
        let iters = (cfg.grad_descent_iterations / sweeps).max(1) as u32;

        let f = |v: f32| format!("{v:?}");
        let common = [
            ("PSZ", PSZ.to_string()),
            ("BORDER", (PSZ + 2).to_string()),
            ("WG_X", gpu.workgroup_size.x.to_string()),
            ("WG_Y", gpu.workgroup_size.y.to_string()),
        ];
        let tokens = |extra: &[(&'static str, String)]| -> Vec<(&'static str, String)> {
            common
                .iter()
                .cloned()
                .chain(extra.iter().cloned())
                .collect()
        };
        let prep = module(
            gpu,
            "dense_stereo_prep.wgsl",
            include_str!("../shaders/dense_stereo_prep.wgsl"),
            &tokens(&[]),
        );
        let patch = module(
            gpu,
            "dense_stereo_patch.wgsl",
            include_str!("../shaders/dense_stereo_patch.wgsl"),
            &tokens(&[
                ("GN_MIN_STEP", f(GN_MIN_STEP)),
                ("SAME_CANDIDATE", f(SAME_CANDIDATE)),
                ("CAP", f(CANDIDATE_TRUNCATION * CANDIDATE_TRUNCATION)),
                ("NOISE_FLOOR", f(NOISE_FLOOR)),
                ("DISTINCT_STEP", f(DISTINCT_STEP)),
            ]),
        );
        let dense = module(
            gpu,
            "dense_stereo_densify.wgsl",
            include_str!("../shaders/dense_stereo_densify.wgsl"),
            &tokens(&[
                ("DISAGREE", f(DISAGREE)),
                ("AMBIGUITY_RATIO", f(AMBIGUITY_RATIO)),
                ("MAX_COVER", MAX_COVER.to_string()),
            ]),
        );
        let refine_module = module(
            gpu,
            "dense_stereo_refine.wgsl",
            include_str!("../shaders/dense_stereo_refine.wgsl"),
            &tokens(&[("ZETA2", f(ZETA2)), ("EPS2", f(EPS2))]),
        );
        let mask = module(
            gpu,
            "dense_stereo_mask.wgsl",
            include_str!("../shaders/dense_stereo_mask.wgsl"),
            &tokens(&[("SPECKLE_STEP", f(SPECKLE_STEP))]),
        );
        let finish = module(
            gpu,
            "dense_stereo_finish.wgsl",
            include_str!("../shaders/dense_stereo_finish.wgsl"),
            &tokens(&[
                ("DISAGREE", f(DISAGREE)),
                ("FALLBACK_AGREE", f(FALLBACK_AGREE)),
                ("FLAT_END_MARGIN", FLAT_END_MARGIN.to_string()),
                ("NO_SOURCE", NO_SOURCE.to_string()),
            ]),
        );
        let u = uniform_entry;
        let rw = |b| storage_entry(b, false);
        let ro = |b| storage_entry(b, true);
        let patch_layout = [u(0), ro(1), ro(2), ro(4), ro(5), rw(6)];
        let mask_layout = [u(0), rw(1), rw(2), rw(3), rw(4), rw(5), rw(6)];
        let pipelines = Pipelines {
            gradients: pipeline(gpu, &prep, "gradients", &[u(0), texture_entry(1), rw(2)]),
            pack: pipeline(
                gpu,
                &prep,
                "pack",
                &[
                    u(0),
                    texture_entry(1),
                    rw(2),
                    texture_entry(7),
                    rw(8),
                    rw(9),
                ],
            ),
            patch_sums: pipeline(gpu, &prep, "patch_sums", &[u(0), rw(2), rw(3)]),
            upsample: pipeline(gpu, &prep, "upsample", &[u(0), rw(4), rw(5)]),
            init_patches: pipeline(gpu, &prep, "init_patches", &[u(0), rw(5), rw(6)]),
            search: pipeline(gpu, &patch, "search", &patch_layout),
            propagate: pipeline(gpu, &patch, "propagate", &patch_layout),
            validity: pipeline(
                gpu,
                &patch,
                "validity",
                &[u(0), ro(1), ro(2), ro(4), ro(5), rw(6), rw(7)],
            ),
            densify: pipeline(
                gpu,
                &dense,
                "densify",
                &[u(0), ro(1), ro(2), ro(3), ro(4), ro(5), rw(6), rw(7), rw(8)],
            ),
            warp: pipeline(
                gpu,
                &refine_module,
                "warp",
                &[u(0), rw(1), rw(2), rw(3), rw(4), rw(5), rw(6)],
            ),
            diffs: pipeline(gpu, &refine_module, "diffs", &[u(0), rw(6), rw(7)]),
            solver_terms: pipeline(
                gpu,
                &refine_module,
                "solver_terms",
                &[
                    u(0),
                    rw(3),
                    rw(4),
                    rw(5),
                    rw(6),
                    rw(7),
                    rw(8),
                    rw(9),
                    rw(10),
                ],
            ),
            weights: pipeline(
                gpu,
                &refine_module,
                "weights",
                &[u(0), rw(9), rw(10), rw(11)],
            ),
            data: pipeline(
                gpu,
                &refine_module,
                "data",
                &[u(0), rw(9), rw(10), rw(11), rw(12)],
            ),
            sor: pipeline(gpu, &refine_module, "sor", &[u(0), rw(10), rw(11), rw(12)]),
            update: pipeline(
                gpu,
                &refine_module,
                "update",
                &[u(0), rw(3), rw(4), rw(10), rw(13)],
            ),
            clamp_range: pipeline(gpu, &mask, "clamp_range", &mask_layout),
            scatter: pipeline(gpu, &mask, "scatter", &mask_layout),
            occlude: pipeline(gpu, &mask, "occlude", &mask_layout),
            unite: pipeline(gpu, &mask, "unite_neighbours", &mask_layout),
            count: pipeline(gpu, &mask, "count", &mask_layout),
            speckles: pipeline(gpu, &mask, "speckles", &mask_layout),
            disagree: pipeline(
                gpu,
                &finish,
                "disagree",
                &[u(0), rw(1), rw(2), rw(3), rw(4), rw(5)],
            ),
            dilate: pipeline(gpu, &finish, "dilate", &[u(0), rw(5), rw(6)]),
            fill: pipeline(
                gpu,
                &finish,
                "fill",
                &[u(0), rw(1), rw(2), rw(3), rw(4), rw(6), rw(7), rw(8), rw(9)],
            ),
            reject: pipeline(
                gpu,
                &finish,
                "reject",
                &[u(0), rw(1), rw(9), rw(10), rw(11), rw(12)],
            ),
            finish: pipeline(
                gpu,
                &finish,
                "finish",
                &[u(0), rw(1), rw(2), rw(7), rw(9), rw(13)],
            ),
        };

        let pyramid = GpuPyramidPipeline::new(gpu);
        let (w0, h0) = cpu.image_size();
        let pyr = [0, 1].map(|eye| match cpu.rectification() {
            Some(r) => pyramid.allocate_rectified(gpu, w0, h0, coarsest + 1, r.map(eye)),
            None => pyramid.allocate(gpu, w0, h0, coarsest + 1),
        });

        let storage = |label: &str, bytes: usize, extra: wgpu::BufferUsages| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | extra,
                mapped_at_creation: false,
            })
        };
        let bind = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect();
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            })
        };
        let n_px = 64; // patch pixels, for the validity thresholds
        let min_hess = cfg.min_texture * cfg.min_texture * n_px as f32;
        assert!(
            !cfg.refine_coarse_levels,
            "the GPU matcher refines the finest level only"
        );
        let none = wgpu::BufferUsages::empty();
        // The mask's z-buffer, union-find parents and region sizes, at the
        // finest (largest) level's size; the fallback's cell flags at the
        // next level's.
        let (fw, fh) = (setups[0].w, setups[0].h);
        let cells = setups.get(1).map_or(1, |c| c.w * c.h);
        let scratch = [
            storage("z-buffer", 4 * fw * fh, none),
            storage("union-find parents", 4 * fw * fh, none),
            storage("region sizes", 4 * fw * fh, none),
            storage("fallback flags", 4 * cells, none),
            storage("fallback flags, dilated", 4 * cells, none),
        ];
        // Coarsest first: each level starts from the coarser one's dense D.
        let mut levels: Vec<Level> = Vec::with_capacity(setups.len());
        for s in setups.into_iter().rev() {
            {
                let level = s.level;
                let coarser = levels
                    .last()
                    .map(|c| (c.setup.w as u32, c.setup.h as u32, c.dense_d.clone()));
                let (cw, ch) = coarser.as_ref().map_or((1, 1), |c| (c.0, c.1));
                let validate = level <= last_validated;
                let min_speckle = (cfg.speckle_size >> (2 * level)) as u32;
                let params = [0u32, 1].map(|colour| {
                    let p = Params {
                        w: s.w as u32,
                        h: s.h as u32,
                        ws: s.ws as u32,
                        hs: s.hs as u32,
                        stride: cfg.patch_stride as u32,
                        colour,
                        per_row: s.ws.div_ceil(2) as u32,
                        iters,
                        dmin: s.dmin,
                        dmax: s.dmax,
                        mean_norm: cfg.mean_normalization as u32,
                        cw,
                        ch,
                        validate: validate as u32,
                        has_init: (level < coarsest) as u32,
                        min_speckle,
                        min_hess,
                        max_ssd: cfg.max_residual * cfg.max_residual * n_px as f32,
                        max_bright: cfg.max_brightness_offset,
                        min_rise: DISTINCT_RISE * min_hess * DISTINCT_STEP * DISTINCT_STEP,
                    };
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("dense stereo params"),
                            contents: bytemuck::bytes_of(&p),
                            usage: wgpu::BufferUsages::UNIFORM,
                        })
                });
                let (n, px) = (s.ws * s.hs, s.w * s.h);
                let grad = storage("grad", 8 * px, none);
                let tpl = storage("left template", 8 * px, none);
                let right = storage("right level", 4 * px, none);
                let sums = storage("patch sums", 20 * n, none);
                let patch_n = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("patch_n"),
                        contents: bytemuck::cast_slice(&s.patch_n),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                let pixel_n = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("pixel_n"),
                        contents: bytemuck::cast_slice(&s.pixel_n),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                let patches = storage("patches", 32 * n, none);
                let d_init = storage("d_init", 4 * px, none);
                let pinfo = storage("patch info", 8 * n, none);
                let dense_d = storage("dense D", 4 * px, none);
                let dense_valid = storage("dense validity", 4 * px, none);
                let dense_var = storage("dense variance", 4 * px, none);
                let (left_view, right_view) = (
                    &pyr[0].levels[level].read_view,
                    &pyr[1].levels[level].read_view,
                );
                let p0 = res(&params[0]);
                let gradients_bg = bind(
                    &pipelines.gradients,
                    &[(0, p0.clone()), (1, tex(left_view)), (2, res(&grad))],
                );
                let pack_bg = bind(
                    &pipelines.pack,
                    &[
                        (0, p0.clone()),
                        (1, tex(left_view)),
                        (2, res(&grad)),
                        (7, tex(right_view)),
                        (8, res(&tpl)),
                        (9, res(&right)),
                    ],
                );
                let sums_bg = bind(
                    &pipelines.patch_sums,
                    &[(0, p0.clone()), (2, res(&grad)), (3, res(&sums))],
                );
                let upsample_bg = coarser.as_ref().map(|c| {
                    bind(
                        &pipelines.upsample,
                        &[(0, p0.clone()), (4, res(&c.2)), (5, res(&d_init))],
                    )
                });
                let init_bg = bind(
                    &pipelines.init_patches,
                    &[(0, p0.clone()), (5, res(&d_init)), (6, res(&patches))],
                );
                let patch_bg = [0, 1].map(|c| {
                    bind(
                        &pipelines.propagate,
                        &[
                            (0, res(&params[c])),
                            (1, res(&tpl)),
                            (2, res(&right)),
                            (4, res(&sums)),
                            (5, res(&patch_n)),
                            (6, res(&patches)),
                        ],
                    )
                });
                let validity_bg = bind(
                    &pipelines.validity,
                    &[
                        (0, p0.clone()),
                        (1, res(&tpl)),
                        (2, res(&right)),
                        (4, res(&sums)),
                        (5, res(&patch_n)),
                        (6, res(&patches)),
                        (7, res(&pinfo)),
                    ],
                );
                let mask_bg = validate.then(|| {
                    bind(
                        &pipelines.clamp_range,
                        &[
                            (0, p0.clone()),
                            (1, res(&dense_d)),
                            (2, res(&dense_valid)),
                            (3, res(&pixel_n)),
                            (4, res(&scratch[0])),
                            (5, res(&scratch[1])),
                            (6, res(&scratch[2])),
                        ],
                    )
                });
                let densify_bg = bind(
                    &pipelines.densify,
                    &[
                        (0, p0),
                        (1, res(&tpl)),
                        (2, res(&right)),
                        (3, res(&patches)),
                        (4, res(&pinfo)),
                        (5, res(&d_init)),
                        (6, res(&dense_d)),
                        (7, res(&dense_valid)),
                        (8, res(&dense_var)),
                    ],
                );
                levels.push(Level {
                    setup: s,
                    _params: params,
                    grad,
                    tpl,
                    right,
                    sums,
                    _patch_n: patch_n,
                    pixel_n,
                    patches,
                    _d_init: d_init,
                    pinfo,
                    dense_d,
                    dense_valid,
                    dense_var,
                    gradients_bg,
                    pack_bg,
                    sums_bg,
                    upsample_bg,
                    init_bg,
                    patch_bg,
                    validity_bg,
                    densify_bg,
                    mask_bg,
                    min_speckle,
                });
            }
        }
        levels.reverse();

        let refine = (cfg.variational_iterations > 0).then(|| {
            let l = &levels[0];
            let (w, h) = (l.setup.w, l.setup.h);
            let hw = w.div_ceil(2);
            let stride = hw + 2;
            let size = stride * (h + 2);
            let inv_edge = if cfg.edge_sigma > 0.0 {
                1.0 / (cfg.edge_sigma * cfg.edge_sigma)
            } else {
                0.0
            };
            let params = [0u32, 1].map(|colour| {
                let p = RefineParams {
                    w: w as u32,
                    h: h as u32,
                    hw: hw as u32,
                    stride: stride as u32,
                    size: size as u32,
                    colour,
                    masked: 1,
                    _pad0: 0,
                    alpha2: cfg.alpha * 0.5,
                    gamma2: cfg.gamma * 0.5,
                    delta2: cfg.delta * 0.5,
                    omega: cfg.sor_omega,
                    inv_edge,
                    _pad1: [0.0; 3],
                };
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("refine params"),
                        contents: bytemuck::bytes_of(&p),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            });
            let ai = storage("mean and difference", 8 * w * h, none);
            let dif = storage("their differences", 16 * w * h, none);
            // Split by colour; zero where nothing writes (borders, past a
            // row's end).
            let terms = storage("solver terms", 32 * 2 * size, none);
            let du = storage("D and dD", 8 * 2 * size, none);
            let wts = storage("edge weights", 8 * 2 * size, none);
            let ac = storage("1/A and c", 8 * 2 * size, none);
            let p0 = res(&params[0]);
            let warp_bg = bind(
                &pipelines.warp,
                &[
                    (0, p0.clone()),
                    (1, res(&l.tpl)),
                    (2, res(&l.right)),
                    (3, res(&l.dense_d)),
                    (4, res(&l.dense_valid)),
                    (5, res(&l.pixel_n)),
                    (6, res(&ai)),
                ],
            );
            let diffs_bg = bind(
                &pipelines.diffs,
                &[(0, p0.clone()), (6, res(&ai)), (7, res(&dif))],
            );
            let terms_bg = bind(
                &pipelines.solver_terms,
                &[
                    (0, p0.clone()),
                    (3, res(&l.dense_d)),
                    (4, res(&l.dense_valid)),
                    (5, res(&l.pixel_n)),
                    (6, res(&ai)),
                    (7, res(&dif)),
                    (8, res(&l.grad)),
                    (9, res(&terms)),
                    (10, res(&du)),
                ],
            );
            let update_bg = bind(
                &pipelines.update,
                &[
                    (0, p0),
                    (3, res(&l.dense_d)),
                    (4, res(&l.dense_valid)),
                    (10, res(&du)),
                    (13, res(&l.dense_var)),
                ],
            );
            let weights_bg = [0, 1].map(|c| {
                bind(
                    &pipelines.weights,
                    &[
                        (0, res(&params[c])),
                        (9, res(&terms)),
                        (10, res(&du)),
                        (11, res(&wts)),
                    ],
                )
            });
            let data_bg = [0, 1].map(|c| {
                bind(
                    &pipelines.data,
                    &[
                        (0, res(&params[c])),
                        (9, res(&terms)),
                        (10, res(&du)),
                        (11, res(&wts)),
                        (12, res(&ac)),
                    ],
                )
            });
            let sor_bg = [0, 1].map(|c| {
                bind(
                    &pipelines.sor,
                    &[
                        (0, res(&params[c])),
                        (10, res(&du)),
                        (11, res(&wts)),
                        (12, res(&ac)),
                    ],
                )
            });
            Refine {
                level: finest,
                iterations: cfg.variational_iterations,
                sor_iterations: cfg.sor_iterations,
                _params: params,
                _ai: ai,
                _dif: dif,
                _terms: terms,
                _du: du,
                _wts: wts,
                _ac: ac,
                warp_bg,
                diffs_bg,
                terms_bg,
                update_bg,
                weights_bg,
                data_bg,
                sor_bg,
            }
        });

        // The coarse fallback, the fills' check and the output, all at the
        // finest level.
        let fine = &levels[0];
        let (w, h) = (fine.setup.w, fine.setup.h);
        assert!(w <= 2048, "the fills' check takes rows up to 2048 pixels");
        let finish_params = |c: Option<(usize, &LevelSetup)>, has_fallback: bool| FinishParams {
            w: w as u32,
            h: h as u32,
            cw: c.map_or(1, |c| c.1.w as u32),
            ch: c.map_or(1, |c| c.1.h as u32),
            k: c.map_or(0, |c| (c.1.level - finest) as u32),
            level: c.map_or(0, |c| c.1.level as u32),
            finest: finest as u32,
            first: c.is_some_and(|c| c.0 == 0) as u32,
            reach: (PSZ - 1) as u32,
            has_fallback: has_fallback as u32,
            flat_gradient: FLAT_GRADIENT * cfg.min_texture,
            max_residual: cfg.max_residual,
            ln_s: fine.setup.ln_s,
            _pad: [0; 3],
        };
        let uniform = |p: &FinishParams| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("finish params"),
                    contents: bytemuck::bytes_of(p),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let source = storage("sources", 4 * w * h, none);
        let fallback: Vec<Fallback> = levels[1..]
            .iter()
            .filter(|c| c.setup.level <= last_validated)
            .enumerate()
            .map(|(i, c)| {
                let params = uniform(&finish_params(Some((i, &c.setup)), true));
                let p = res(&params);
                let disagree_bg = bind(
                    &pipelines.disagree,
                    &[
                        (0, p.clone()),
                        (1, res(&fine.dense_d)),
                        (2, res(&fine.dense_valid)),
                        (3, res(&c.dense_d)),
                        (4, res(&c.dense_valid)),
                        (5, res(&scratch[3])),
                    ],
                );
                let dilate_bg = bind(
                    &pipelines.dilate,
                    &[(0, p.clone()), (5, res(&scratch[3])), (6, res(&scratch[4]))],
                );
                let fill_bg = bind(
                    &pipelines.fill,
                    &[
                        (0, p),
                        (1, res(&fine.dense_d)),
                        (2, res(&fine.dense_valid)),
                        (3, res(&c.dense_d)),
                        (4, res(&c.dense_valid)),
                        (6, res(&scratch[4])),
                        (7, res(&fine.dense_var)),
                        (8, res(&c.dense_var)),
                        (9, res(&source)),
                    ],
                );
                Fallback {
                    level: c.setup.level,
                    _params: params,
                    disagree_bg,
                    dilate_bg,
                    fill_bg,
                }
            })
            .collect();
        let params = uniform(&finish_params(None, !fallback.is_empty()));
        let n = w * h;
        let out_bytes = 4 * (2 * n + n.div_ceil(4));
        let out = storage("output", out_bytes, none);
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output readback"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let reject_bg = (!fallback.is_empty()).then(|| {
            bind(
                &pipelines.reject,
                &[
                    (0, res(&params)),
                    (1, res(&fine.dense_d)),
                    (9, res(&source)),
                    (10, res(&fine.tpl)),
                    (11, res(&fine.right)),
                    (12, res(&fine.pixel_n)),
                ],
            )
        });
        let finish_bg = bind(
            &pipelines.finish,
            &[
                (0, res(&params)),
                (1, res(&fine.dense_d)),
                (2, res(&fine.dense_valid)),
                (7, res(&fine.dense_var)),
                (9, res(&source)),
                (13, res(&out)),
            ],
        );
        let output = Output {
            _params: params,
            _source: source,
            reject_bg,
            finish_bg,
            out,
            readback,
        };
        Self {
            finest,
            coarsest,
            sweeps,
            occlusion: cfg.occlusion_check,
            levels,
            refine,
            fallback,
            output,
            _scratch: scratch,
            pyramid,
            pyr,
            pipelines,
            readback_rx: None,
        }
    }

    fn level(&self, level: usize) -> &Level {
        &self.levels[level - self.finest]
    }

    /// Workgroups of 64 for `n` items.
    fn groups(n: usize) -> u32 {
        let groups = n.div_ceil(64);
        assert!(groups <= 65535, "too many items for one dispatch row");
        groups as u32
    }

    /// Stage a pair (raw or rectified, as the CPU matcher takes them) for the
    /// next recorded frame.
    pub fn upload(&self, gpu: &GpuDevice, left: &Image<u8>, right: &Image<u8>) {
        self.pyramid.upload(gpu, left, &self.pyr[0]);
        self.pyramid.upload(gpu, right, &self.pyr[1]);
    }

    /// Both pyramids from the uploaded pair. Record them in the command
    /// buffer that first reads them (the Tegra stale-read rule; see
    /// gpu/frontend.rs).
    pub(crate) fn record_pyramids(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        for pyr in &self.pyr {
            self.pyramid.record_build_in_pass(gpu, pass, pyr);
        }
    }

    /// A level's gradients, image buffers and patch sums.
    pub(crate) fn record_prepare(
        &self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        level: usize,
    ) {
        let l = self.level(level);
        let (dx, dy) = gpu.dispatch_size(l.setup.w as u32, l.setup.h as u32);
        for (pipeline, bg) in [
            (&self.pipelines.gradients, &l.gradients_bg),
            (&self.pipelines.pack, &l.pack_bg),
        ] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
        }
        pass.set_pipeline(&self.pipelines.patch_sums);
        pass.set_bind_group(0, &l.sums_bg, &[]);
        pass.dispatch_workgroups(Self::groups(l.setup.ws * l.setup.hs), 1, 1);
    }

    /// The coarsest level's patches, by exhaustive search.
    pub(crate) fn record_search(&self, pass: &mut wgpu::ComputePass<'_>) {
        let l = self.level(self.coarsest);
        pass.set_pipeline(&self.pipelines.search);
        pass.set_bind_group(0, &l.patch_bg[0], &[]);
        pass.dispatch_workgroups(Self::groups(l.setup.ws * l.setup.hs), 1, 1);
    }

    /// A finer level's patches, from the coarser level's dense D.
    pub(crate) fn record_upsample(
        &self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        level: usize,
    ) {
        let l = self.level(level);
        pass.set_pipeline(&self.pipelines.upsample);
        pass.set_bind_group(
            0,
            l.upsample_bg.as_ref().expect("not the coarsest level"),
            &[],
        );
        let (dx, dy) = gpu.dispatch_size(l.setup.w as u32, l.setup.h as u32);
        pass.dispatch_workgroups(dx, dy, 1);
        pass.set_pipeline(&self.pipelines.init_patches);
        pass.set_bind_group(0, &l.init_bg, &[]);
        pass.dispatch_workgroups(Self::groups(l.setup.ws * l.setup.hs), 1, 1);
    }

    /// Red-black propagation sweeps, a dispatch per colour.
    pub(crate) fn record_propagate(&self, pass: &mut wgpu::ComputePass<'_>, level: usize) {
        let l = self.level(level);
        pass.set_pipeline(&self.pipelines.propagate);
        let groups = Self::groups(l.setup.hs * l.setup.ws.div_ceil(2));
        for _ in 0..self.sweeps {
            for bg in &l.patch_bg {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    /// The level's dense D from its patches: their validity, then each
    /// pixel's mean.
    pub(crate) fn record_densify(
        &self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        level: usize,
    ) {
        let l = self.level(level);
        pass.set_pipeline(&self.pipelines.validity);
        pass.set_bind_group(0, &l.validity_bg, &[]);
        pass.dispatch_workgroups(Self::groups(l.setup.ws * l.setup.hs), 1, 1);
        pass.set_pipeline(&self.pipelines.densify);
        pass.set_bind_group(0, &l.densify_bg, &[]);
        let (dx, dy) = gpu.dispatch_size(l.setup.w as u32, l.setup.h as u32);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// The variational refinement of the level's dense D (the finest level
    /// only; nothing at others).
    pub(crate) fn record_refine(
        &self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        level: usize,
    ) {
        let Some(r) = self.refine.as_ref().filter(|r| r.level == level) else {
            return;
        };
        let l = self.level(level);
        let (w, h) = (l.setup.w as u32, l.setup.h as u32);
        let px = gpu.dispatch_size(w, h);
        let cells = gpu.dispatch_size(w.div_ceil(2), h);
        let mut run =
            |pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, (x, y): (u32, u32)| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            };
        let p = &self.pipelines;
        run(&p.warp, &r.warp_bg, px);
        run(&p.diffs, &r.diffs_bg, px);
        run(&p.solver_terms, &r.terms_bg, px);
        for _ in 0..r.iterations {
            for bg in &r.weights_bg {
                run(&p.weights, bg, cells);
            }
            for bg in &r.data_bg {
                run(&p.data, bg, cells);
            }
            for _ in 0..r.sor_iterations {
                for bg in &r.sor_bg {
                    run(&p.sor, bg, cells);
                }
            }
        }
        run(&p.update, &r.update_bg, px);
    }

    /// A validated level's mask: D clamped to the searched range, then
    /// occluded pixels and speckles dropped from the validity (nothing at
    /// other levels).
    pub(crate) fn record_mask(
        &self,
        gpu: &GpuDevice,
        pass: &mut wgpu::ComputePass<'_>,
        level: usize,
    ) {
        let l = self.level(level);
        let Some(bg) = &l.mask_bg else {
            return;
        };
        let p = &self.pipelines;
        let mut stages = vec![&p.clamp_range];
        if self.occlusion {
            stages.extend([&p.scatter, &p.occlude]);
        }
        // A region has at least one pixel.
        if l.min_speckle > 1 {
            stages.extend([&p.unite, &p.count, &p.speckles]);
        }
        let (dx, dy) = gpu.dispatch_size(l.setup.w as u32, l.setup.h as u32);
        pass.set_bind_group(0, bg, &[]);
        for pipeline in stages {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(dx, dy, 1);
        }
    }

    /// The coarse fallback: the finest level's pixels without an estimate
    /// from the coarser levels', then the check of those fills.
    pub(crate) fn record_fallback(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        let fine = &self.levels[0].setup;
        let px = gpu.dispatch_size(fine.w as u32, fine.h as u32);
        let p = &self.pipelines;
        for f in &self.fallback {
            let c = &self.level(f.level).setup;
            let cells = gpu.dispatch_size(c.w as u32, c.h as u32);
            for (pipeline, bg, (x, y)) in [
                (&p.disagree, &f.disagree_bg, cells),
                (&p.dilate, &f.dilate_bg, cells),
                (&p.fill, &f.fill_bg, px),
            ] {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
    }

    /// The check of the fills (a workgroup per row).
    pub(crate) fn record_reject(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(bg) = &self.output.reject_bg {
            pass.set_pipeline(&self.pipelines.reject);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.levels[0].setup.h as u32, 1, 1);
        }
    }

    /// The depth map into the output buffer.
    pub(crate) fn record_finish(&self, pass: &mut wgpu::ComputePass<'_>) {
        let fine = &self.levels[0].setup;
        pass.set_pipeline(&self.pipelines.finish);
        pass.set_bind_group(0, &self.output.finish_bg, &[]);
        pass.dispatch_workgroups(Self::groups((fine.w * fine.h).div_ceil(4)), 1, 1);
    }

    /// Every stage of a frame from the uploaded pair, in one pass.
    pub fn record_frame(&self, gpu: &GpuDevice, pass: &mut wgpu::ComputePass<'_>) {
        self.record_pyramids(gpu, pass);
        for level in (self.finest..=self.coarsest).rev() {
            self.record_prepare(gpu, pass, level);
            if level == self.coarsest {
                self.record_search(pass);
            } else {
                self.record_upsample(gpu, pass, level);
            }
            self.record_propagate(pass, level);
            self.record_densify(gpu, pass, level);
            self.record_refine(gpu, pass, level);
            self.record_mask(gpu, pass, level);
        }
        self.record_fallback(gpu, pass);
        self.record_reject(pass);
        self.record_finish(pass);
    }

    /// A frame and the copy of its depth map for the readback, into
    /// `encoder` (after `upload`; submit, then `arm_readback`).
    pub fn record_into(&self, gpu: &GpuDevice, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dense stereo"),
                timestamp_writes: None,
            });
            self.record_frame(gpu, &mut pass);
        }
        let o = &self.output;
        encoder.copy_buffer_to_buffer(&o.out, 0, &o.readback, 0, o.out.size());
    }

    /// Map the readback buffer: after `queue.submit`, before `device.poll`.
    pub fn arm_readback(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.output
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        self.readback_rx = Some(rx);
    }

    /// The depth map, after `device.poll` (as `DenseStereo::compute_into`).
    pub fn collect_into(&mut self, out: &mut DenseDepth) {
        let rx = self
            .readback_rx
            .take()
            .expect("call arm_readback() before collect_into()");
        rx.recv().unwrap().expect("dense stereo readback failed");
        let fine = &self.levels[0].setup;
        let n = fine.w * fine.h;
        out.width = fine.w;
        out.height = fine.h;
        out.level = self.finest;
        let rb = &self.output.readback;
        {
            let mapped = rb
                .slice(..)
                .get_mapped_range()
                .expect("buffer mapping failed");
            let words: &[f32] = bytemuck::cast_slice(&mapped);
            out.log_range.clear();
            out.log_range.extend_from_slice(&words[..n]);
            out.variance.clear();
            out.variance.extend_from_slice(&words[n..2 * n]);
            out.source_level.clear();
            out.source_level
                .extend_from_slice(&mapped[8 * n..8 * n + n]);
        }
        rb.unmap();
    }

    /// Depth of the left image from a pair: upload, record, submit, wait and
    /// read back (for standalone use; a pipeline sharing the GPU records
    /// `record_into` into its own command buffer).
    pub fn compute(&mut self, gpu: &GpuDevice, left: &Image<u8>, right: &Image<u8>) -> DenseDepth {
        let mut out = DenseDepth {
            width: 0,
            height: 0,
            level: 0,
            log_range: Vec::new(),
            variance: Vec::new(),
            source_level: Vec::new(),
        };
        self.compute_into(gpu, left, right, &mut out);
        out
    }

    /// As `compute`, reusing `out`'s buffers.
    pub fn compute_into(
        &mut self,
        gpu: &GpuDevice,
        left: &Image<u8>,
        right: &Image<u8>,
        out: &mut DenseDepth,
    ) {
        self.upload(gpu, left, right);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.record_into(gpu, &mut enc);
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.arm_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        self.collect_into(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraIntrinsics, StereoRig};
    use crate::dense_stereo::DenseStereoConfig;

    /// Value noise in [0, 1) at (u, v), cells of side 1.
    fn noise(u: f64, v: f64, seed: u64) -> f64 {
        let hash = |i: i64, j: i64| {
            let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
                ^ seed.wrapping_mul(0x165667B19E3779F9);
            x ^= x >> 31;
            x = x.wrapping_mul(0xD6E8FEB86659FD93);
            x ^= x >> 32;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let s = |t: f64| t * t * (3.0 - 2.0 * t);
        let (i, j) = (i as i64, j as i64);
        let a = hash(i, j) * (1.0 - s(fu)) + hash(i + 1, j) * s(fu);
        let b = hash(i, j + 1) * (1.0 - s(fu)) + hash(i + 1, j + 1) * s(fu);
        a * (1.0 - s(fv)) + b * s(fv)
    }

    /// A rectified 640×480 pair (f 500, baseline 0.11 m) of a textured wall
    /// at 3 m with a textured box at 1.2 m before its middle.
    fn scene() -> (StereoRig, Image<u8>, Image<u8>) {
        let (w, h, f, b) = (640usize, 480usize, 500.0, 0.11);
        let (cx, cy) = (319.5, 239.5);
        let cam = CameraIntrinsics::new(f, f, cx, cy, w, h);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rig = StereoRig::new(cam.clone(), cam, identity, [-b, 0.0, 0.0]);
        // (z, x range, y range, seed), nearest first.
        let planes = [
            (1.2, (-0.3, 0.25), (-0.2, 0.3), 2u64),
            (3.0, (-9.0, 9.0), (-9.0, 9.0), 1),
        ];
        let render = |ox: f64| {
            let mut data = vec![0u8; w * h];
            for v in 0..h {
                for u in 0..w {
                    let (dx, dy) = ((u as f64 - cx) / f, (v as f64 - cy) / f);
                    for &(z, xr, yr, seed) in &planes {
                        let (x, y) = (ox + dx * z, dy * z);
                        if x >= xr.0 && x <= xr.1 && y >= yr.0 && y <= yr.1 {
                            let mut t = 0.0;
                            let mut amp = 1.0;
                            for o in 0..4 {
                                let cell = 0.08 / (1 << o) as f64;
                                t += amp * noise(x / cell, y / cell, seed + o);
                                amp *= 0.55;
                            }
                            data[v * w + u] = (30.0 + 190.0 * t / 2.0).clamp(0.0, 255.0) as u8;
                            break;
                        }
                    }
                }
            }
            Image::from_vec(w, h, data)
        };
        // The right camera sits at x = +b (x_right = x_left − b).
        (rig, render(0.0), render(b))
    }

    /// Wait for `bufs` (each copied into a MAP_READ twin) and read them.
    fn read<T: bytemuck::Pod>(gpu: &GpuDevice, buf: &wgpu::Buffer) -> Vec<T> {
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        buf.unmap();
        data
    }

    /// How far the GPU's results are from the CPU's on one level.
    struct LevelReport {
        level: usize,
        patches: usize,
        /// Propagated patches whose D differs by over 0.01 px, and those whose
        /// flags differ.
        far: usize,
        flags_differ: usize,
        pixels: usize,
        /// Densified pixels whose validity differs, and valid ones whose D
        /// differs by over 0.01 px.
        valid_differ: usize,
        dense_far: usize,
        /// Pixels whose validity after the mask differs.
        mask_differ: usize,
    }

    /// ... and on the finest level's output.
    struct OutputReport {
        pixels: usize,
        /// Pixels whose source differs after the coarse fallback, and after
        /// the fills' check.
        filled_differ: usize,
        source_differ: usize,
        /// Where both have depth: largest |Δ log-range|, and the variance's
        /// largest relative difference.
        log_range_err: f32,
        var_err: f32,
    }

    fn to_gpu(p: &crate::dense_stereo::PatchTrace) -> GpuPatch {
        let flag = |b: bool, bit: u32| if b { bit } else { 0 };
        GpuPatch {
            d: p.d,
            ssd: if p.ok { p.ssd } else { 3.0e38 },
            cost: 0.0,
            hess: p.hess,
            shift_x: p.shift[0],
            shift_y: p.shift[1],
            offset: if p.ok { p.offset } else { 3.0e38 },
            flags: flag(p.ok, GpuPatch::OK)
                | flag(p.inside, GpuPatch::INSIDE)
                | flag(p.moved, GpuPatch::MOVED)
                | flag(p.settled, GpuPatch::SETTLED),
        }
    }

    /// Every level's stages against the CPU's, each starting from the CPU's
    /// results of the stage before (the coarser level's dense D, the
    /// propagated patches for densification), so that differences do not
    /// carry over.
    fn check_against_cpu(
        gpu: &GpuDevice,
        mut cpu: DenseStereo,
        left: &Image<u8>,
        right: &Image<u8>,
    ) -> (Vec<LevelReport>, OutputReport) {
        let frame = cpu.trace(left, right);
        let trace = &frame.levels;
        let gds = GpuDenseStereo::new(gpu, &cpu);
        gds.upload(gpu, left, right);

        // One command buffer: pyramids, then per level the stages, copying
        // each result out as it is made.
        let staging = |src: &wgpu::Buffer| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: src.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            gds.record_pyramids(gpu, &mut pass);
        }
        let copy = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer| {
            let dst = staging(src);
            enc.copy_buffer_to_buffer(src, 0, &dst, 0, src.size());
            dst
        };
        let upload = |data: &[u8], dst: &wgpu::Buffer, enc: &mut wgpu::CommandEncoder| {
            let src = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CPU data"),
                    contents: data,
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            enc.copy_buffer_to_buffer(&src, 0, dst, 0, src.size());
        };
        let as_u32 = |v: &[bool]| -> Vec<u32> { v.iter().map(|&b| b as u32).collect() };
        let mut out = Vec::new();
        for t in trace {
            let l = gds.level(t.level);
            // A finer level starts from the coarser one's dense D, the CPU's
            // (uploaded at the end of that level).
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                gds.record_prepare(gpu, &mut pass, t.level);
                if t.level == gds.coarsest {
                    gds.record_search(&mut pass);
                } else {
                    gds.record_upsample(gpu, &mut pass, t.level);
                }
            }
            let (g, s, init) = (
                copy(&mut enc, &l.grad),
                copy(&mut enc, &l.sums),
                copy(&mut enc, &l.patches),
            );
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                gds.record_propagate(&mut pass, t.level);
            }
            let fin = copy(&mut enc, &l.patches);
            // Densify the CPU's propagated patches.
            let cpu_patches: Vec<GpuPatch> = t.patches.iter().map(to_gpu).collect();
            let src = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CPU patches"),
                    contents: bytemuck::cast_slice(&cpu_patches),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            enc.copy_buffer_to_buffer(&src, 0, &l.patches, 0, src.size());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                gds.record_densify(gpu, &mut pass, t.level);
            }
            let dense =
                [&l.pinfo, &l.dense_d, &l.dense_valid, &l.dense_var].map(|b| copy(&mut enc, b));
            // Refine the CPU's densified D.
            upload(bytemuck::cast_slice(&t.dense_d), &l.dense_d, &mut enc);
            upload(
                bytemuck::cast_slice(&as_u32(&t.dense_valid)),
                &l.dense_valid,
                &mut enc,
            );
            upload(bytemuck::cast_slice(&t.dense_var), &l.dense_var, &mut enc);
            let refined = t.refined.as_ref().map(|_| {
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    gds.record_refine(gpu, &mut pass, t.level);
                }
                [copy(&mut enc, &l.dense_d), copy(&mut enc, &l.dense_var)]
            });
            // Mask the CPU's refined D.
            let (d, var) = match &t.refined {
                Some((d, var)) => (d, var),
                None => (&t.dense_d, &t.dense_var),
            };
            let masked = t.masked.as_ref().map(|_| {
                upload(bytemuck::cast_slice(d), &l.dense_d, &mut enc);
                upload(bytemuck::cast_slice(var), &l.dense_var, &mut enc);
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    gds.record_mask(gpu, &mut pass, t.level);
                }
                [copy(&mut enc, &l.dense_d), copy(&mut enc, &l.dense_valid)]
            });
            // The CPU's result: the next level's start, the fallback's input.
            let (d, valid) = match &t.masked {
                Some((d, valid)) => (d, valid),
                None => (d, &t.dense_valid),
            };
            upload(bytemuck::cast_slice(d), &l.dense_d, &mut enc);
            upload(
                bytemuck::cast_slice(&as_u32(valid)),
                &l.dense_valid,
                &mut enc,
            );
            upload(bytemuck::cast_slice(var), &l.dense_var, &mut enc);
            out.push((g, s, init, fin, dense, refined, masked));
        }
        // The fallback, the fills' check and the output, from the CPU's
        // levels.
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            gds.record_fallback(gpu, &mut pass);
        }
        let filled = copy(&mut enc, &gds.output._source);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            gds.record_reject(&mut pass);
            gds.record_finish(&mut pass);
        }
        let output = copy(&mut enc, &gds.output.out);
        gpu.queue.submit(std::iter::once(enc.finish()));

        let max_rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        let mut reports = Vec::new();
        for (t, (g, s, init, fin, dense, refined, masked)) in trace.iter().zip(&out) {
            let grad: Vec<[f32; 2]> = read(gpu, g);
            let gerr = grad
                .iter()
                .zip(t.gx.iter().zip(&t.gy))
                .map(|(a, (&x, &y))| (a[0] - x).abs().max((a[1] - y).abs()))
                .fold(0.0f32, f32::max);
            let sums: Vec<[f32; 5]> = read(gpu, s);
            let serr = sums
                .iter()
                .zip(&t.sums)
                .map(|(a, b)| (0..5).map(|i| max_rel(a[i], b[i])).fold(0.0f32, f32::max))
                .fold(0.0f32, f32::max);
            let init: Vec<GpuPatch> = read(gpu, init);
            let ierr = init
                .iter()
                .zip(&t.init)
                .map(|(a, &b)| (a.d - b).abs())
                .fold(0.0f32, f32::max);
            assert!(gerr < 1e-3, "level {}: gradients differ by {gerr}", t.level);
            assert!(
                serr < 1e-3,
                "level {}: patch sums differ by {serr}",
                t.level
            );
            assert!(
                ierr < 1e-3,
                "level {}: starting D differs by {ierr}",
                t.level
            );
            let fin: Vec<GpuPatch> = read(gpu, fin);
            let mut dd: Vec<f32> = fin
                .iter()
                .zip(&t.patches)
                .map(|(a, b)| (a.d - b.d).abs())
                .collect();
            let flag = |a: &GpuPatch, bit: u32| a.flags & bit != 0;
            let flags_differ = fin
                .iter()
                .zip(&t.patches)
                .filter(|(a, b)| {
                    flag(a, GpuPatch::OK) != b.ok
                        || flag(a, GpuPatch::INSIDE) != b.inside
                        || flag(a, GpuPatch::MOVED) != b.moved
                        || flag(a, GpuPatch::SETTLED) != b.settled
                })
                .count();
            // Where D agrees, the rest of the estimate should too.
            let rest = fin
                .iter()
                .zip(&t.patches)
                .filter(|(a, b)| b.ok && (a.d - b.d).abs() < 1e-4)
                .map(|(a, b)| {
                    [
                        max_rel(a.ssd, b.ssd),
                        max_rel(a.hess, b.hess),
                        (a.shift_x - b.shift[0])
                            .abs()
                            .max((a.shift_y - b.shift[1]).abs()),
                        (a.offset - b.offset).abs(),
                    ]
                    .into_iter()
                    .fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            assert!(
                rest < 1e-2,
                "level {}: SSD, Hessian, shift or offset differ by {rest}",
                t.level
            );
            dd.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| dd[((dd.len() - 1) as f64 * f) as usize];
            let far = dd.iter().filter(|&&e| e > 0.01).count();

            // Densification of the same patches.
            let pinfo: Vec<[u32; 2]> = read(gpu, &dense[0]);
            let prec_err = pinfo
                .iter()
                .zip(&t.precision)
                .map(|(a, &b)| max_rel(f32::from_bits(a[0]), b))
                .fold(0.0f32, f32::max);
            let valid_patch_differ = pinfo
                .iter()
                .zip(&t.valid_patch)
                .filter(|(a, &b)| (a[1] != 0) != b)
                .count();
            let dense_d: Vec<f32> = read(gpu, &dense[1]);
            let dense_valid: Vec<u32> = read(gpu, &dense[2]);
            let dense_var: Vec<f32> = read(gpu, &dense[3]);
            let valid_differ = dense_valid
                .iter()
                .zip(&t.dense_valid)
                .filter(|(a, &b)| (**a != 0) != b)
                .count();
            let both = |i: usize| dense_valid[i] != 0 && t.dense_valid[i];
            let dense_far = (0..dense_d.len())
                .filter(|&i| both(i) && (dense_d[i] - t.dense_d[i]).abs() > 0.01)
                .count();
            // Invalid pixels' D (the all-patch mean or the coarser start).
            let other_far = (0..dense_d.len())
                .filter(|&i| !t.dense_valid[i] && dense_valid[i] == 0)
                .filter(|&i| (dense_d[i] - t.dense_d[i]).abs() > 0.01)
                .count();
            // The variance's spread term E[D²] − mean² cancels: on top of a
            // relative error, allow rounding of D² in f32 (both sides carry
            // it). 1 is the tolerance.
            let var_err = (0..dense_d.len())
                .filter(|&i| both(i) && (dense_d[i] - t.dense_d[i]).abs() < 1e-4)
                .map(|i| {
                    let tol = 1e-2 * t.dense_var[i].abs() + 1e-6 * t.dense_d[i] * t.dense_d[i];
                    (dense_var[i] - t.dense_var[i]).abs() / tol
                })
                .fold(0.0f32, f32::max);
            eprintln!(
                "[test] level {}: grad max {gerr:.1e}, sums max rel {serr:.1e}, start D max {ierr:.1e}; \
                 propagated D |GPU − CPU| median {:.1e} p99 {:.1e} max {:.1e}, {far} of {} over 0.01 px, \
                 {flags_differ} with other flags; the rest where D agrees within {rest:.1e}\n\
                 [test]   densify: precision max rel {prec_err:.1e}, validity differs on {valid_patch_differ} \
                 patches; pixels: validity differs on {valid_differ} of {}, valid D over 0.01 px on {dense_far}, \
                 other D on {other_far}, variance where D agrees within {var_err:.2} of its tolerance",
                t.level,
                q(0.5),
                q(0.99),
                q(1.0),
                dd.len(),
                dense_d.len()
            );
            assert!(
                prec_err < 1e-5,
                "level {}: precision differs by {prec_err}",
                t.level
            );
            assert!(
                var_err <= 1.0,
                "level {}: variance differs by {var_err} tolerances",
                t.level
            );
            // The refinement of the CPU's densified D, where the CPU solved.
            let mut refined_far = 0;
            if let (Some([rd, rv]), Some((cd, cv))) = (refined, &t.refined) {
                let rd: Vec<f32> = read(gpu, rd);
                let rv: Vec<f32> = read(gpu, rv);
                let solved: Vec<usize> = (0..rd.len()).filter(|&i| t.dense_valid[i]).collect();
                let mut e: Vec<f32> = solved.iter().map(|&i| (rd[i] - cd[i]).abs()).collect();
                e.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q = |f: f64| e[((e.len() - 1) as f64 * f) as usize];
                refined_far = e.iter().filter(|&&x| x > 0.01).count();
                let var_err = solved
                    .iter()
                    .filter(|&&i| (rd[i] - cd[i]).abs() < 1e-3)
                    .map(|&i| {
                        let tol = 1e-2 * cv[i].abs() + 1e-6 * cd[i] * cd[i];
                        (rv[i] - cv[i]).abs() / tol
                    })
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "[test]   refine: D |GPU − CPU| median {:.1e} p99 {:.1e} max {:.1e}, {refined_far} of {} \
                     over 0.01 px; variance where D agrees within {var_err:.2} of its tolerance",
                    q(0.5),
                    q(0.99),
                    q(1.0),
                    e.len()
                );
            }
            // The mask of the CPU's refined D.
            let mut mask_differ = 0;
            if let (Some([md, mv]), Some((cd, cv))) = (masked, &t.masked) {
                let md: Vec<f32> = read(gpu, md);
                let mv: Vec<u32> = read(gpu, mv);
                let clamp_err = md
                    .iter()
                    .zip(cd)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                mask_differ = mv.iter().zip(cv).filter(|(a, &b)| (**a != 0) != b).count();
                let kept = cv.iter().filter(|&&v| v).count();
                eprintln!(
                    "[test]   mask: clamped D max {clamp_err:.1e}; validity differs on {mask_differ} of {} \
                     ({kept} valid)",
                    mv.len()
                );
                assert!(clamp_err == 0.0, "level {}: clamped D differs", t.level);
            }
            reports.push(LevelReport {
                level: t.level,
                patches: dd.len(),
                far,
                flags_differ,
                pixels: dense_d.len(),
                valid_differ,
                dense_far: dense_far + other_far + refined_far,
                mask_differ,
            });
        }

        let depth = &frame.depth;
        let n = depth.log_range.len();
        let filled: Vec<u32> = read(gpu, &filled);
        let filled_differ = filled
            .iter()
            .zip(&frame.filled)
            .filter(|(a, b)| **a != **b as u32)
            .count();
        let words: Vec<u32> = read(gpu, &output);
        let log_range: Vec<f32> = words[..n].iter().map(|&b| f32::from_bits(b)).collect();
        let variance: Vec<f32> = words[n..2 * n].iter().map(|&b| f32::from_bits(b)).collect();
        let source: Vec<u8> = bytemuck::cast_slice(&words[2 * n..])[..n].to_vec();
        let source_differ = source
            .iter()
            .zip(&depth.source_level)
            .filter(|(a, b)| a != b)
            .count();
        let both: Vec<usize> = (0..n)
            .filter(|&i| log_range[i].is_finite() && depth.log_range[i].is_finite())
            .collect();
        let log_range_err = both
            .iter()
            .map(|&i| (log_range[i] - depth.log_range[i]).abs())
            .fold(0.0f32, f32::max);
        let var_err = both
            .iter()
            .map(|&i| max_rel(variance[i], depth.variance[i]))
            .fold(0.0f32, f32::max);
        let mismatched_nan = (0..n)
            .filter(|&i| log_range[i].is_finite() != (source[i] != NO_SOURCE))
            .count();
        let fills = frame
            .filled
            .iter()
            .filter(|&&s| s != NO_SOURCE && s as usize != gds.finest)
            .count();
        eprintln!(
            "[test] output: sources differ after the fallback on {filled_differ} of {n} pixels ({fills} \
             filled), after the fills' check on {source_differ}; log-range max |Δ| {log_range_err:.1e}, \
             variance max rel {var_err:.1e}"
        );
        assert_eq!(mismatched_nan, 0, "NaN where there is a source or none");
        (
            reports,
            OutputReport {
                pixels: n,
                filled_differ,
                source_differ,
                log_range_err,
                var_err,
            },
        )
    }

    /// The whole GPU frame against `DenseStereo::compute`: (pixels, pixels
    /// with depth on one side only, with depth on both, of which differing by
    /// over 1 % in range, p99 of |Δ log-range| there).
    fn compare_frames(
        gpu: &GpuDevice,
        cpu: &mut DenseStereo,
        gds: &mut GpuDenseStereo,
        left: &Image<u8>,
        right: &Image<u8>,
    ) -> (usize, usize, usize, usize, f32) {
        let a = cpu.compute(left, right);
        let b = gds.compute(gpu, left, right);
        assert_eq!((a.width, a.height, a.level), (b.width, b.height, b.level));
        let n = a.log_range.len();
        let (mut one, mut both) = (0, Vec::new());
        for i in 0..n {
            match (a.log_range[i].is_finite(), b.log_range[i].is_finite()) {
                (true, true) => both.push((a.log_range[i] - b.log_range[i]).abs()),
                (false, false) => {}
                _ => one += 1,
            }
        }
        both.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let far = both.iter().filter(|&&e| e > 0.01).count();
        let p99 = both
            .get((both.len().max(1) - 1) * 99 / 100)
            .copied()
            .unwrap_or(0.0);
        (n, one, both.len(), far, p99)
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_matches_cpu() {
        let (rig, left, right) = scene();
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let cpu = DenseStereo::new(rig.clone(), DenseStereoConfig::default());
        let (levels, o) = check_against_cpu(&gpu, cpu, &left, &right);
        assert_eq!(o.filled_differ, 0, "the coarse fallback's sources differ");
        assert!(
            o.source_differ * 1000 <= o.pixels,
            "the fills' check differs"
        );
        assert!(o.log_range_err < 1e-4 && o.var_err < 1e-4, "output");
        for r in levels {
            assert!(r.mask_differ * 1000 <= r.pixels, "level {}: mask", r.level);
            assert!(
                r.far * 100 <= r.patches,
                "level {}: over 1 % of patches differ",
                r.level
            );
            assert!(
                r.flags_differ * 100 <= r.patches,
                "level {}: flags",
                r.level
            );
            assert!(
                r.valid_differ * 1000 <= r.pixels,
                "level {}: pixel validity",
                r.level
            );
            assert!(r.dense_far * 1000 <= r.pixels, "level {}: dense D", r.level);
        }

        // The whole frame on the GPU.
        let mut cpu = DenseStereo::new(rig, DenseStereoConfig::default());
        let mut gds = GpuDenseStereo::new(&gpu, &cpu);
        let (n, one, both, far, p99) = compare_frames(&gpu, &mut cpu, &mut gds, &left, &right);
        eprintln!(
            "[test] whole frame: depth on one side only on {one} of {n} pixels, on both on {both}, of \
             which {far} differ by over 1 % in range; p99 |Δ log-range| {p99:.1e}"
        );
        assert!(
            one * 100 <= n,
            "over 1 % of pixels have depth on one side only"
        );
        assert!(far * 100 <= both, "over 1 % of depths differ by over 1 %");
        println!("GPU_TEST_OK");
    }

    /// DENSE_STEREO_FRAMES: a folder with camchain.yaml (a rectified pair)
    /// and left/, right/ PNGs, as tools/dense_stereo extracts; its rig, the
    /// sorted names, and a loader. DENSE_STEREO_LEVEL sets the finest level
    /// (default 1).
    fn recorded_frames() -> Option<(
        StereoRig,
        std::path::PathBuf,
        Vec<std::ffi::OsString>,
        DenseStereoConfig,
    )> {
        let Some(dir) = std::env::var_os("DENSE_STEREO_FRAMES") else {
            eprintln!("DENSE_STEREO_FRAMES not set; skipping");
            return None;
        };
        let dir = std::path::PathBuf::from(dir);
        let rig =
            StereoRig::from_kalibr_camchain(&dir.join("camchain.yaml")).expect("camchain.yaml");
        let mut names: Vec<_> = std::fs::read_dir(dir.join("left"))
            .expect("left/")
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect();
        names.sort();
        let level = std::env::var("DENSE_STEREO_LEVEL").map_or(1, |v| v.parse().expect("level"));
        let cfg = DenseStereoConfig {
            finest_level: level,
            ..Default::default()
        };
        Some((rig, dir, names, cfg))
    }

    fn load(p: std::path::PathBuf) -> Image<u8> {
        let img = image::open(&p).expect("png").into_luma8();
        Image::from_vec(img.width() as usize, img.height() as usize, img.into_raw())
    }

    #[test]
    #[ignore = "GPU integration on recorded frames: set DENSE_STEREO_FRAMES"]
    fn inner_matches_cpu_on_frames() {
        // Four pairs, stage by stage and whole frames; the share of results
        // that differ is reported (real frames have near-ties a few rounding
        // steps decide).
        let Some((rig, dir, names, cfg)) = recorded_frames() else {
            println!("GPU_TEST_OK");
            return;
        };
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let (mut n, mut far, mut flags, mut px, mut vdiff, mut dfar) = (0, 0, 0, 0, 0, 0);
        let (mut mdiff, mut out_px, mut filled, mut sources) = (0, 0, 0, 0);
        let (mut frame_px, mut one, mut both, mut range_far) = (0, 0, 0, 0);
        let mut cpu_whole = DenseStereo::new(rig.clone(), cfg.clone());
        let mut gds = GpuDenseStereo::new(&gpu, &cpu_whole);
        for name in names.iter().step_by((names.len() / 4).max(1)).take(4) {
            let left = load(dir.join("left").join(name));
            let right = load(dir.join("right").join(name));
            let cpu = DenseStereo::new(rig.clone(), cfg.clone());
            let (levels, o) = check_against_cpu(&gpu, cpu, &left, &right);
            for r in levels {
                (n, far, flags) = (n + r.patches, far + r.far, flags + r.flags_differ);
                (px, vdiff, dfar) = (px + r.pixels, vdiff + r.valid_differ, dfar + r.dense_far);
                mdiff += r.mask_differ;
            }
            (out_px, filled, sources) = (
                out_px + o.pixels,
                filled + o.filled_differ,
                sources + o.source_differ,
            );
            assert!(o.log_range_err < 1e-4 && o.var_err < 1e-4, "output");
            let (fp, o1, b, f, p99) = compare_frames(&gpu, &mut cpu_whole, &mut gds, &left, &right);
            eprintln!("[test] whole frame: p99 |Δ log-range| {p99:.1e}");
            (frame_px, one, both, range_far) = (frame_px + fp, one + o1, both + b, range_far + f);
        }
        let pct = |a: usize, b: usize| 100.0 * a as f64 / b as f64;
        eprintln!(
            "[test] recorded frames: propagation: {far} of {n} patches over 0.01 px ({:.3} %), {flags} with \
             other flags; densification: validity differs on {vdiff} of {px} pixels ({:.3} %), D on {dfar} \
             ({:.3} %); mask: validity differs on {mdiff} ({:.3} %); output: sources differ after the \
             fallback on {filled} of {out_px}, after the fills' check on {sources} ({:.3} %)\n\
             [test] whole frames: depth on one side only on {one} of {frame_px} pixels ({:.2} %), on both \
             on {both}, of which {range_far} ({:.2} %) differ by over 1 % in range",
            pct(far, n),
            pct(vdiff, px),
            pct(dfar, px),
            pct(mdiff, px),
            pct(sources, out_px),
            pct(one, frame_px),
            pct(range_far, both),
        );
        assert!(far * 100 <= n, "over 1 % of patches differ");
        assert!(
            vdiff * 100 <= px && dfar * 100 <= px && mdiff * 100 <= px,
            "over 1 % of pixels differ"
        );
        assert_eq!(filled, 0, "the coarse fallback's sources differ");
        assert!(sources * 100 <= out_px, "the fills' check differs");
        assert!(
            one * 20 <= frame_px,
            "over 5 % of pixels have depth on one side only"
        );
        assert!(
            range_far * 20 <= both,
            "over 5 % of depths differ by over 1 %"
        );
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU timing on recorded frames: set DENSE_STEREO_FRAMES"]
    fn inner_time_on_frames() {
        // Wall time per frame on the first pair: cumulative stages, each
        // level's share, and whole frames with the readback.
        let Some((rig, dir, names, cfg)) = recorded_frames() else {
            println!("GPU_TEST_OK");
            return;
        };
        let left = load(dir.join("left").join(&names[0]));
        let right = load(dir.join("right").join(&names[0]));
        let cpu = DenseStereo::new(rig, cfg);
        let gpu = GpuDevice::new().expect("need Vulkan GPU");
        let mut gds = GpuDenseStereo::new(&gpu, &cpu);
        let levels: Vec<usize> = (gds.finest..=gds.coarsest).rev().collect();
        // Stages up to: 0 pyramids, 1 preparation and start, 2 propagation,
        // 3 densification, 4 refinement, 5 mask, 6 fallback, fills' check
        // and output; `only`: one level's.
        let frame = |gds: &GpuDenseStereo, upto: usize, only: Option<usize>| {
            gds.upload(&gpu, &left, &right);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                gds.record_pyramids(&gpu, &mut pass);
                for &level in levels.iter().filter(|&&l| only.is_none_or(|o| o == l)) {
                    if upto >= 1 {
                        gds.record_prepare(&gpu, &mut pass, level);
                        if level == gds.coarsest {
                            gds.record_search(&mut pass);
                        } else {
                            gds.record_upsample(&gpu, &mut pass, level);
                        }
                    }
                    if upto >= 2 {
                        gds.record_propagate(&mut pass, level);
                    }
                    if upto >= 3 {
                        gds.record_densify(&gpu, &mut pass, level);
                    }
                    if upto >= 4 {
                        gds.record_refine(&gpu, &mut pass, level);
                    }
                    if upto >= 5 {
                        gds.record_mask(&gpu, &mut pass, level);
                    }
                }
                if upto >= 6 && only.is_none() {
                    gds.record_fallback(&gpu, &mut pass);
                    gds.record_reject(&mut pass);
                    gds.record_finish(&mut pass);
                }
            }
            gpu.queue.submit(std::iter::once(enc.finish()));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        };
        let time = |f: &mut dyn FnMut()| {
            for _ in 0..10 {
                f();
            }
            let n = 30;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            t0.elapsed().as_secs_f64() * 1e3 / n as f64
        };
        let stages = [0, 1, 2, 3, 4, 5, 6].map(|s| time(&mut || frame(&gds, s, None)));
        eprintln!(
            "[test] {}: ms per frame: pyramids {:.2}, + preparation and start {:.2}, + propagation \
             {:.2}, + densification {:.2}, + refinement {:.2}, + mask {:.2}, + fallback, fills' check \
             and output {:.2}",
            gpu.adapter_info.name,
            stages[0],
            stages[1],
            stages[2],
            stages[3],
            stages[4],
            stages[5],
            stages[6]
        );
        for &level in &levels {
            let [a, b, c, d, e] =
                [1, 2, 3, 4, 5].map(|s| time(&mut || frame(&gds, s, Some(level))));
            eprintln!(
                "[test]   level {level}: propagation {:.2} ms, densification {:.2} ms, refinement {:.2} ms, \
                 mask {:.2} ms",
                b - a,
                c - b,
                d - c,
                e - d
            );
        }
        let whole = time(&mut || {
            gds.compute(&gpu, &left, &right);
        });
        eprintln!("[test]   whole frame with the readback: {whole:.2} ms");
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_matches_cpu() {
        let out = std::process::Command::new("cargo")
            .args([
                "test",
                "--lib",
                "--",
                "gpu::dense_stereo::tests::inner_matches_cpu",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .expect("subprocess");
        let out = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        print!("{out}");
        assert!(out.contains("GPU_TEST_OK"), "inner test failed:\n{out}");
    }
}
