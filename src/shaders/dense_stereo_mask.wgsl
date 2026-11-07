// dense_stereo_mask.wgsl — a validated level's depth map, as dense_stereo.rs
// `mask`: D clamped to the searched range, then pixels dropped from the
// validity where the right image sees a nearer point there (occlusion), and
// in connected regions of consistent D smaller than P.min_speckle pixels
// (`remove_speckles`; here by union-find with atomics, whose regions, and so
// the result, do not depend on the order of the unions).
//   clamp_range  clamp D; reset the z-buffer and the union-find
//   scatter      each valid pixel's D into the z-buffer at its right-image
//                pixel, the largest kept (atomicMax on the bits: D > 0, so
//                they order as the floats)
//   occlude      drop pixels that lost there by more than MARGIN
//   unite        join each valid pixel to its left and upper neighbours
//   count        pixels per region, at its root
//   speckles     drop the small regions' pixels
// {{SPECKLE_STEP}}: dense_stereo.rs's constant.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> dense_d: array<f32>;
@group(0) @binding(2) var<storage, read_write> dense_valid: array<u32>;
// Per pixel: the right image's shift per unit of D (0: no ray).
@group(0) @binding(3) var<storage, read_write> pixel_n: array<f32>;
@group(0) @binding(4) var<storage, read_write> zbuf: array<atomic<u32>>;
// Union-find: each pixel's parent (never above it), and per root its
// region's size.
@group(0) @binding(5) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> region: array<atomic<u32>>;

const MARGIN: f32 = 1.0;
const SPECKLE_STEP: f32 = {{SPECKLE_STEP}};
const NO_PIXEL: u32 = 0xffffffffu;

// Rounding half away from zero, as f32::round (WGSL's rounds half to even).
fn round_away(v: f32) -> f32 {
    let t = trunc(v);
    return select(t, t + sign(v), abs(v - t) >= 0.5);
}

// The right-image pixel seeing pixel (x, y) at its D (`Epipolar::locate`,
// rounded), or NO_PIXEL.
fn right_pixel(x: u32, y: u32, i: u32) -> u32 {
    let n = pixel_n[i];
    if n == 0.0 {
        return NO_PIXEL;
    }
    let q = round_away(f32(x) + dense_d[i] * n);
    if q >= 0.0 && q < f32(P.w) {
        return y * P.w + u32(q);
    }
    return NO_PIXEL;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn clamp_range(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    dense_d[i] = clamp(dense_d[i], P.dmin, P.dmax);
    atomicStore(&zbuf[i], 0u);
    atomicStore(&parent[i], i);
    atomicStore(&region[i], 0u);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if dense_valid[i] == 0u {
        return;
    }
    let q = right_pixel(gid.x, gid.y, i);
    if q != NO_PIXEL {
        atomicMax(&zbuf[q], bitcast<u32>(dense_d[i]));
    }
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn occlude(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if dense_valid[i] == 0u {
        return;
    }
    let q = right_pixel(gid.x, gid.y, i);
    if q != NO_PIXEL && bitcast<f32>(atomicLoad(&zbuf[q])) > dense_d[i] + MARGIN {
        dense_valid[i] = 0u;
    }
}

fn find(start: u32) -> u32 {
    var r = start;
    var p = atomicLoad(&parent[r]);
    while p != r {
        r = p;
        p = atomicLoad(&parent[r]);
    }
    return r;
}

// Merge the regions of a and b: the larger root points to the smaller. If
// another thread attached that root meanwhile, its new parent's region is
// merged in turn (the link just replaced).
fn unite(a0: u32, b0: u32) {
    var a = find(a0);
    var b = find(b0);
    loop {
        if a == b {
            return;
        }
        let hi = max(a, b);
        let lo = min(a, b);
        let old = atomicMin(&parent[hi], lo);
        if old == hi {
            return;
        }
        a = find(old);
        b = find(lo);
    }
}

fn joined(i: u32, j: u32) -> bool {
    return dense_valid[j] != 0u && abs(dense_d[i] - dense_d[j]) <= SPECKLE_STEP;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn unite_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if dense_valid[i] == 0u {
        return;
    }
    if gid.x > 0u && joined(i, i - 1u) {
        unite(i, i - 1u);
    }
    if gid.y > 0u && joined(i, i - P.w) {
        unite(i, i - P.w);
    }
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if dense_valid[i] != 0u {
        atomicAdd(&region[find(i)], 1u);
    }
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn speckles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if dense_valid[i] != 0u && atomicLoad(&region[find(i)]) < P.min_speckle {
        dense_valid[i] = 0u;
    }
}
