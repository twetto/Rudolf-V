// dense_stereo_prep.wgsl — a level's preparation and starting point, as
// dense_stereo.rs:
//   gradients     central differences of the left level (diff_x, diff_y)
//   patch_sums    each patch's gradient sums (patch_sums)
//   upsample      the coarser level's dense D, upsampled (upsample_init)
//   init_patches  each patch's D at its centre of that (upsample_init)
//
// Each kernel binds only what it uses; storage buffers are read_write so
// that one declaration serves the kernel that writes them and the one that
// reads them. {{WG_X}} / {{WG_Y}}: the 2-D workgroup.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var left_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> grad: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> sums: array<Sums>;
@group(0) @binding(4) var<storage, read_write> coarse_d: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_init: array<f32>;
@group(0) @binding(6) var<storage, read_write> patches: array<Patch>;
@group(0) @binding(7) var right_tex: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> tpl: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> right_img: array<f32>;

fn left_at(x: u32, y: u32) -> f32 {
    return textureLoad(left_tex, vec2<u32>(x, y), 0).r;
}

// Central differences, the border replicated: diff_x and diff_y.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= P.w || y >= P.h {
        return;
    }
    let xl = select(x - 1u, 0u, x == 0u);
    let xr = min(x + 1u, P.w - 1u);
    let yu = select(y - 1u, 0u, y == 0u);
    let yd = min(y + 1u, P.h - 1u);
    let gx = 0.5 * (left_at(xr, y) - left_at(xl, y));
    let gy = 0.5 * (left_at(x, yd) - left_at(x, yu));
    grad[y * P.w + x] = vec2<f32>(gx, gy);
}

// The level's images as storage buffers, for the one-thread-per-patch
// kernels: the left grey level with its x-gradient, and the right level.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    tpl[i] = vec2<f32>(left_at(gid.x, gid.y), grad[i].x);
    right_img[i] = textureLoad(right_tex, vec2<u32>(gid.x, gid.y), 0).r;
}

// Gradient sums of one patch per thread, in the CPU's order (row by row).
@compute @workgroup_size(64)
fn patch_sums(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= P.ws * P.hs {
        return;
    }
    let x0 = (k % P.ws) * P.stride;
    let y0 = (k / P.ws) * P.stride;
    var s = Sums(0.0, 0.0, 0.0, 0.0, 0.0);
    for (var y = y0; y < y0 + PSZ; y++) {
        for (var x = x0; x < x0 + PSZ; x++) {
            let g = grad[y * P.w + x];
            s.xx += g.x * g.x;
            s.xy += g.x * g.y;
            s.yy += g.y * g.y;
            s.x += g.x;
            s.y += g.y;
        }
    }
    sums[k] = s;
}

// Bilinear sample of a dense w × h field, clamped to it (sample_field).
fn sample_coarse(x_in: f32, y_in: f32) -> f32 {
    let w = P.cw;
    let h = P.ch;
    let x = clamp(x_in, 0.0, f32(w - 1u));
    let y = clamp(y_in, 0.0, f32(h - 1u));
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    return (1.0 - fy) * ((1.0 - fx) * coarse_d[y0 * w + x0] + fx * coarse_d[y0 * w + x1])
        + fy * ((1.0 - fx) * coarse_d[y1 * w + x0] + fx * coarse_d[y1 * w + x1]);
}

fn sample_init(x_in: f32, y_in: f32) -> f32 {
    let w = P.w;
    let h = P.h;
    let x = clamp(x_in, 0.0, f32(w - 1u));
    let y = clamp(y_in, 0.0, f32(h - 1u));
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    return (1.0 - fy) * ((1.0 - fx) * d_init[y0 * w + x0] + fx * d_init[y0 * w + x1])
        + fy * ((1.0 - fx) * d_init[y1 * w + x0] + fx * d_init[y1 * w + x1]);
}

// The coarser level's D at this level's pixels (D doubles a level finer).
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    d_init[gid.y * P.w + gid.x] = 2.0 * sample_coarse(f32(gid.x) * 0.5, f32(gid.y) * 0.5);
}

// Each patch starts from the upsampled D at its centre, without an estimate,
// moved (propagation's first sweep evaluates every patch).
@compute @workgroup_size(64)
fn init_patches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= P.ws * P.hs {
        return;
    }
    let c = (f32(PSZ) - 1.0) * 0.5;
    let cx = f32((k % P.ws) * P.stride) + c;
    let cy = f32((k / P.ws) * P.stride) + c;
    var p = failed(sample_init(cx, cy));
    p.flags = MOVED;
    patches[k] = p;
}
