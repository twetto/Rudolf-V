// dense_stereo_densify.wgsl — each pixel's D from the patches over it, as
// dense_stereo.rs `Densify` (one thread per pixel, the patches in the CPU's
// order): the weighted mean of all of them (weight 1 / max(1, |photometric
// residual|) times the patch's inverse variance), and where valid patches
// cover the pixel, their mean alone with its variance. Where those disagree
// by more than DISAGREE (a depth edge), the patch whose shift explains the
// pixel's 3×3 neighbourhood best wins if clearly better than the other
// surface's best (AMBIGUITY_RATIO), and only patches agreeing with it count;
// else the pixel has no depth.
// {{DISAGREE}}, {{AMBIGUITY_RATIO}}, {{MAX_COVER}}: dense_stereo.rs's constants.

@group(0) @binding(0) var<uniform> P: Params;
// Per pixel: left grey level, gx.
@group(0) @binding(1) var<storage, read> tpl: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> right_img: array<f32>;
@group(0) @binding(3) var<storage, read> patches: array<Patch>;
@group(0) @binding(4) var<storage, read> pinfo: array<PatchInfo>;
// The coarser level's D upsampled, for pixels no patch covers.
@group(0) @binding(5) var<storage, read> d_init: array<f32>;
@group(0) @binding(6) var<storage, read_write> dense_d: array<f32>;
@group(0) @binding(7) var<storage, read_write> dense_valid: array<u32>;
@group(0) @binding(8) var<storage, read_write> dense_var: array<f32>;

const DISAGREE: f32 = {{DISAGREE}};
const AMBIGUITY_RATIO: f32 = {{AMBIGUITY_RATIO}};
const MAX_COVER: u32 = {{MAX_COVER}}u;

fn left_at(x: u32, y: u32) -> f32 {
    return tpl[y * P.w + x].x;
}

// The right image with its border repeated (the CPU's padding).
fn right_at(x: i32, y: i32) -> f32 {
    let w = i32(P.w);
    let h = i32(P.h);
    return right_img[u32(clamp(y, 0, h - 1)) * P.w + u32(clamp(x, 0, w - 1))];
}

// Patch k's sample of the right image for pixel (x, y), as `ShiftSampler`:
// weights from the shift's fraction when every pixel using the patch (its
// footprint, grown by one) samples inside the CPU's padding, else
// `Padded::sample`, which clamps the position into it.
fn sample(k: u32, x: u32, y: u32) -> f32 {
    let p = patches[k];
    let js = k % P.ws;
    let is = k / P.ws;
    var fx1 = js * P.stride + PSZ - 1u;
    if js + 1u == P.ws {
        fx1 = P.w - 1u;
    }
    var fy1 = is * P.stride + PSZ - 1u;
    if is + 1u == P.hs {
        fy1 = P.h - 1u;
    }
    let gx0 = i32(js * P.stride) - 1;
    let gy0 = i32(is * P.stride) - 1;
    let gx1 = i32(min(fx1 + 1u, P.w - 1u));
    let gy1 = i32(min(fy1 + 1u, P.h - 1u));
    let ox = floor(p.shift_x);
    let oy = floor(p.shift_y);
    let b = BORDER;
    let fast = max(gx0, 0) + b + i32(ox) >= 0 && gx1 + b + i32(ox) + 1 < i32(P.w) + 2 * b
        && max(gy0, 0) + b + i32(oy) >= 0 && gy1 + b + i32(oy) + 1 < i32(P.h) + 2 * b;
    if fast {
        let fx = p.shift_x - ox;
        let fy = p.shift_y - oy;
        let xx = i32(x) + i32(ox);
        let yy = i32(y) + i32(oy);
        return (1.0 - fx) * (1.0 - fy) * right_at(xx, yy) + fx * (1.0 - fy) * right_at(xx + 1, yy)
            + (1.0 - fx) * fy * right_at(xx, yy + 1) + fx * fy * right_at(xx + 1, yy + 1);
    }
    let bf = f32(b);
    let xs = clamp(f32(x) + p.shift_x, -bf + 1.0, f32(P.w) + bf - 2.0) + bf;
    let ys = clamp(f32(y) + p.shift_y, -bf + 1.0, f32(P.h) + bf - 2.0) + bf;
    let x0 = floor(xs);
    let y0 = floor(ys);
    let fx = xs - x0;
    let fy = ys - y0;
    let xx = i32(x0) - b;
    let yy = i32(y0) - b;
    return (1.0 - fy) * ((1.0 - fx) * right_at(xx, yy) + fx * right_at(xx + 1, yy))
        + fy * ((1.0 - fx) * right_at(xx, yy + 1) + fx * right_at(xx + 1, yy + 1));
}

// Patches covering coordinate c along one axis: first..=last (`span`).
fn span(c: u32, count: u32) -> vec2<u32> {
    let last = min(c / P.stride, count - 1u);
    var first = 0u;
    if c + 1u >= PSZ {
        first = (c + 1u - PSZ + P.stride - 1u) / P.stride;
    }
    return vec2<u32>(min(first, last), last);
}

// Photometric cost of patch k's shift on the 3×3 pixels around (x, y).
fn local_cost(x: u32, y: u32, k: u32) -> f32 {
    var c = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        let yy = u32(clamp(i32(y) + dy, 0, i32(P.h) - 1));
        for (var dx = -1; dx <= 1; dx++) {
            let xx = u32(clamp(i32(x) + dx, 0, i32(P.w) - 1));
            c += abs(sample(k, xx, yy) - left_at(xx, yy));
        }
    }
    return c;
}

struct Depth {
    ok: bool,
    d: f32,
    var_: f32,
}

// D and variance where the valid patches over (x, y) disagree (`depth_edge`).
fn depth_edge(x: u32, y: u32, is_span: vec2<u32>, js_span: vec2<u32>) -> Depth {
    var ck: array<u32, {{MAX_COVER}}>;
    var cw: array<f32, {{MAX_COVER}}>;
    var nc = 0u;
    let i0 = left_at(x, y);
    for (var is = is_span.x; is <= is_span.y; is++) {
        for (var js = js_span.x; js <= js_span.y; js++) {
            let k = is * P.ws + js;
            let info = pinfo[k];
            if info.prec == 0.0 || info.valid == 0u || nc == MAX_COVER {
                continue;
            }
            let i1 = sample(k, x, y);
            ck[nc] = k;
            cw[nc] = 1.0 / max(abs(i1 - i0), 1.0) * info.prec;
            nc += 1u;
        }
    }
    var lo = BIG;
    for (var c = 0u; c < nc; c++) {
        lo = min(lo, patches[ck[c]].d);
    }
    var cost: array<f32, {{MAX_COVER}}>;
    var win_cost = BIG;
    var win_d = lo;
    for (var c = 0u; c < nc; c++) {
        cost[c] = local_cost(x, y, ck[c]);
        if cost[c] < win_cost {
            win_cost = cost[c];
            win_d = patches[ck[c]].d;
        }
    }
    // The other surface's best; a winner not clearly better leaves no depth.
    var rival = BIG;
    for (var c = 0u; c < nc; c++) {
        if abs(patches[ck[c]].d - win_d) > DISAGREE {
            rival = min(rival, cost[c]);
        }
    }
    var out = Depth(false, 0.0, 0.0);
    if win_cost * AMBIGUITY_RATIO > rival {
        return out;
    }
    var vw = 0.0;
    var vwd = 0.0;
    var vwd2 = 0.0;
    var vwvar = 0.0;
    for (var c = 0u; c < nc; c++) {
        let pd = patches[ck[c]].d;
        if abs(pd - win_d) > DISAGREE {
            continue;
        }
        let wk = cw[c];
        vw += wk;
        vwd += wk * pd;
        vwd2 += wk * pd * pd;
        vwvar += wk / pinfo[ck[c]].prec;
    }
    let mean = vwd / vw;
    out = Depth(true, mean, vwvar / vw + max(vwd2 / vw - mean * mean, 0.0));
    return out;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn densify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= P.w || y >= P.h {
        return;
    }
    let i = y * P.w + x;
    let is_span = span(y, P.hs);
    let js_span = span(x, P.ws);
    let i0 = left_at(x, y);
    var sw = 0.0;
    var swd = 0.0;
    var vw = 0.0;
    var vwd = 0.0;
    var vwd2 = 0.0;
    var vwvar = 0.0;
    var lo = BIG;
    var hi = -BIG;
    var count = 0u;
    for (var is = is_span.x; is <= is_span.y; is++) {
        for (var js = js_span.x; js <= js_span.y; js++) {
            let k = is * P.ws + js;
            let info = pinfo[k];
            if info.prec == 0.0 {
                continue;
            }
            let dk = patches[k].d;
            let wk = 1.0 / max(abs(sample(k, x, y) - i0), 1.0) * info.prec;
            sw += wk;
            swd += wk * dk;
            if P.validate != 0u && info.valid != 0u {
                vw += wk;
                vwd += wk * dk;
                vwd2 += wk * dk * dk;
                vwvar += wk / info.prec;
                lo = min(lo, dk);
                hi = max(hi, dk);
                count += 1u;
            }
        }
    }
    var d = P.dmin;
    if sw > 0.0 {
        d = swd / sw;
    } else if P.has_init != 0u {
        d = d_init[i];
    }
    var valid = 0u;
    var variance = 0.0;
    if count > 0u {
        if hi - lo > DISAGREE {
            let e = depth_edge(x, y, is_span, js_span);
            if e.ok {
                d = e.d;
                variance = e.var_;
                valid = 1u;
            }
        } else {
            let mean = vwd / vw;
            d = mean;
            variance = vwvar / vw + max(vwd2 / vw - mean * mean, 0.0);
            valid = 1u;
        }
    }
    dense_d[i] = d;
    dense_valid[i] = valid;
    dense_var[i] = variance;
}
