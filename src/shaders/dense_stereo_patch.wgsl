// dense_stereo_patch.wgsl — patch inverse search on a rectified pair, as
// dense_stereo.rs (`LevelView::eval`, `refine`, `best_candidate`, `search`,
// `DenseStereo::propagate`, and the patch half of `densify`):
//   search     the coarsest level's exhaustive 1-D search
//   propagate  one red-black colour of a propagation sweep: each patch takes
//              the best of its own and its four neighbours' D, then runs
//              Gauss-Newton from there
//   validity   each patch's inverse variance and validity, with the
//              distinctness test
//
// One patch per thread, walking it row by row; no barriers. On a rectified
// pair a patch's bilinear samples have no vertical weight: each row of them
// is a linear interpolation of 9 texels of one image row. The truncated cost
// (candidates only) needs the mean first, so a walk keeps its residuals in
// workgroup memory. The images are storage buffers, not textures: texture
// fetches bound these kernels on the Orin. The left level is packed with its
// x-gradient, the only one a rectified pair's search needs (its y-shift per
// unit of D is 0).
// {{GN_MIN_STEP}}, {{SAME_CANDIDATE}}, {{CAP}}, {{NOISE_FLOOR}},
// {{DISTINCT_STEP}}: dense_stereo.rs's constants.

@group(0) @binding(0) var<uniform> P: Params;
// Per pixel: left grey level, gx.
@group(0) @binding(1) var<storage, read> tpl: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> right_img: array<f32>;
@group(0) @binding(4) var<storage, read> sums: array<Sums>;
// Per patch, the right image's shift per unit of D; 0: no projection.
@group(0) @binding(5) var<storage, read> patch_n: array<f32>;
@group(0) @binding(6) var<storage, read_write> patches: array<Patch>;
@group(0) @binding(7) var<storage, read_write> pinfo: array<PatchInfo>;

const GN_MIN_STEP: f32 = {{GN_MIN_STEP}};
const SAME_CANDIDATE: f32 = {{SAME_CANDIDATE}};
const CAP: f32 = {{CAP}};
const NOISE_FLOOR: f32 = {{NOISE_FLOOR}};
const DISTINCT_STEP: f32 = {{DISTINCT_STEP}};

// The right image with its border repeated (the CPU's padding).
fn right_at(x: i32, y: i32) -> f32 {
    let w = i32(P.w);
    let h = i32(P.h);
    return right_img[u32(clamp(y, 0, h - 1)) * P.w + u32(clamp(x, 0, w - 1))];
}

struct Eval {
    ok: bool,
    ssd: f32,
    cost: f32,
    grad: f32,
    hess: f32,
    shift_x: f32,
    shift_y: f32,
    offset: f32,
    inside: bool,
}

// Where patch k samples the right image at D d: its top-left texel and
// bilinear weights.
struct Placement {
    j: u32,
    i: u32,
    px: i32,
    py: i32,
    w: vec4<f32>,  // w00, w10, w01, w11
}

// The residuals of each thread's patch at its latest evaluation, the
// thread's i-th at [i · 64 + lid] (a warp's threads in consecutive banks),
// for the truncated cost.
var<workgroup> res_cache: array<f32, 4096>;

// Σ r, Σ r², Σ r·gx over the patch (Σ r·gy only meets ey = 0: 0), keeping
// each r in res_cache with `keep`.
fn walk(q: Placement, keep: bool, lid: u32) -> vec4<f32> {
    var acc = vec4<f32>(0.0);
    if q.w.z == 0.0 && q.w.w == 0.0 {
        // No vertical weight (always, on a rectified pair): a linear
        // interpolation along each row. The CPU adds the zero-weighted
        // terms, which changes nothing.
        for (var row = 0u; row < PSZ; row++) {
            var r9: array<f32, 9>;
            for (var c = 0; c < 9; c++) {
                r9[c] = right_at(q.px + c, q.py + i32(row));
            }
            let y = q.i + row;
            for (var c = 0u; c < PSZ; c++) {
                let v = q.w.x * r9[c] + q.w.y * r9[c + 1u];
                let t = tpl[y * P.w + q.j + c];
                let r = v - t.x;
                acc += vec4<f32>(r, r * r, r * t.y, 0.0);
                if keep {
                    res_cache[(row * PSZ + c) * 64u + lid] = r;
                }
            }
        }
        return acc;
    }
    var top: array<f32, 9>;
    var bot: array<f32, 9>;
    for (var c = 0; c < 9; c++) {
        top[c] = right_at(q.px + c, q.py);
    }
    for (var row = 0u; row < PSZ; row++) {
        for (var c = 0; c < 9; c++) {
            bot[c] = right_at(q.px + c, q.py + i32(row) + 1);
        }
        let y = q.i + row;
        for (var c = 0u; c < PSZ; c++) {
            let v = q.w.x * top[c] + q.w.y * top[c + 1u] + q.w.z * bot[c] + q.w.w * bot[c + 1u];
            let t = tpl[y * P.w + q.j + c];
            let r = v - t.x;
            acc += vec4<f32>(r, r * r, r * t.y, 0.0);
            if keep {
                res_cache[(row * PSZ + c) * 64u + lid] = r;
            }
        }
        top = bot;
    }
    return acc;
}

fn eval(k: u32, d: f32, want_cost: bool, lid: u32) -> Eval {
    var e: Eval;
    let n = patch_n[k];
    if n == 0.0 {
        e.ok = false;
        return e;
    }
    e.ok = true;
    let j = (k % P.ws) * P.stride;
    let i = (k / P.ws) * P.stride;
    let c = (f32(PSZ) - 1.0) * 0.5;
    let cx = f32(j) + c;
    let cy = f32(i) + c;
    let lx = cx + d * n;
    e.shift_x = lx - cx;
    e.shift_y = cy - cy;
    let tlx_raw = f32(j) + e.shift_x;
    let tly_raw = f32(i) + e.shift_y;
    e.inside = tlx_raw >= 0.0 && tly_raw >= 0.0 && tlx_raw <= f32(P.w - PSZ)
        && tly_raw <= f32(P.h - PSZ);
    let lim = f32(PSZ) - 1.0;
    let b = f32(BORDER);
    let tlx = clamp(tlx_raw, -lim, f32(P.w - 1u)) + b;
    let tly = clamp(tly_raw, -lim, f32(P.h - 1u)) + b;
    let x0 = floor(tlx);
    let y0 = floor(tly);
    let fx = tlx - x0;
    let fy = tly - y0;
    let q = Placement(
        j,
        i,
        i32(x0) - BORDER,
        i32(y0) - BORDER,
        vec4<f32>((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy),
    );
    let t = walk(q, want_cost, lid);
    let nn = f32(PSZ * PSZ);
    var mean = 0.0;
    if P.mean_norm != 0u {
        mean = t.x / nn;
    }
    e.cost = 0.0;
    if want_cost {
        for (var p = 0u; p < PSZ * PSZ; p++) {
            let q = res_cache[p * 64u + lid] - mean;
            e.cost += min(q * q, CAP);
        }
    }
    let su = sums[k];
    let ex = n;
    let ey = 0.0;
    var hess = ex * ex * su.xx + 2.0 * ex * ey * su.xy + ey * ey * su.yy;
    if P.mean_norm != 0u {
        let gd = ex * su.x + ey * su.y;
        hess -= gd * gd / nn;
        e.ssd = t.y - t.x * t.x / nn;
        e.grad = ex * (t.z - t.x * su.x / nn) + ey * (t.w - t.x * su.y / nn);
    } else {
        e.ssd = t.y;
        e.grad = ex * t.z + ey * t.w;
    }
    e.hess = hess;
    e.offset = t.x / nn;
    return e;
}

fn state(d: f32, e: Eval) -> Patch {
    return Patch(d, e.ssd, e.cost, e.hess, e.shift_x, e.shift_y, e.offset,
                 OK | select(0u, INSIDE, e.inside));
}

struct Refined {
    p: Patch,
    converged: bool,
}

fn refine(k: u32, start: f32, e0: Eval, lid: u32) -> Refined {
    let first = state(start, e0);
    var d = start;
    var e = e0;
    var converged = false;
    for (var it = 0u; it < P.iters; it++) {
        if !(e.hess > 1e-6) {
            converged = true;
            break;
        }
        let next = clamp(d - e.grad / e.hess, P.dmin, P.dmax);
        if abs(next - d) < GN_MIN_STEP {
            converged = true;
            break;
        }
        let ne = eval(k, next, false, lid);
        if ne.ok && ne.ssd < e.ssd {
            d = next;
            e = ne;
        } else {
            converged = true;
            break;
        }
    }
    if abs(d - start) > f32(PSZ) {
        return Refined(first, false);
    }
    return Refined(state(d, e), converged);
}

@compute @workgroup_size(64)
fn search(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = gid.x;
    if k >= P.ws * P.hs {
        return;
    }
    var best = failed(0.0);
    let steps = max(u32(ceil(P.dmax - P.dmin)), 1u);
    for (var s = 0u; s <= steps; s++) {
        let d = min(P.dmin + f32(s), P.dmax);
        let e = eval(k, d, true, lid);
        if e.ok && e.inside && e.cost < best.cost {
            best = state(d, e);
        }
    }
    if (best.flags & OK) == 0u {
        best.d = P.dmin;
    }
    best.flags |= MOVED;
    patches[k] = best;
}

fn has_moved(k: u32) -> bool {
    return (patches[k].flags & MOVED) != 0u;
}

@compute @workgroup_size(64)
fn propagate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let is = gid.x / P.per_row;
    let js = (is + P.colour) % 2u + 2u * (gid.x % P.per_row);
    if is >= P.hs || js >= P.ws {
        return;
    }
    let ws = P.ws;
    let k = is * ws + js;
    let me = patches[k];
    let neighbours_moved = (js > 0u && has_moved(k - 1u)) || (js + 1u < ws && has_moved(k + 1u))
        || (is > 0u && has_moved(k - ws)) || (is + 1u < P.hs && has_moved(k + ws));
    if (me.flags & SETTLED) != 0u && !neighbours_moved {
        patches[k].flags = me.flags & ~MOVED;
        return;
    }
    let own = me.d;
    var cand = array<f32, 5>(own, 0.0, 0.0, 0.0, 0.0);
    var has = array<bool, 5>(true, js > 0u, js + 1u < ws, is > 0u, is + 1u < P.hs);
    if has[1] {
        cand[1] = patches[k - 1u].d;
    }
    if has[2] {
        cand[2] = patches[k + 1u].d;
    }
    if has[3] {
        cand[3] = patches[k - ws].d;
    }
    if has[4] {
        cand[4] = patches[k + ws].d;
    }
    var tried = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    var n_tried = 0u;
    var found = false;
    var best_d = 0.0;
    var best: Eval;
    for (var c = 0u; c < 5u; c++) {
        if !has[c] {
            continue;
        }
        let d = cand[c];
        var same = false;
        for (var t = 0u; t < n_tried; t++) {
            if abs(d - tried[t]) < SAME_CANDIDATE {
                same = true;
            }
        }
        if same {
            continue;
        }
        tried[n_tried] = d;
        n_tried += 1u;
        let e = eval(k, d, true, lid);
        if e.ok && (!found || e.cost < best.cost) {
            found = true;
            best_d = d;
            best = e;
        }
    }
    var p = failed(own);
    var converged = false;
    if found {
        let r = refine(k, best_d, best, lid);
        p = r.p;
        converged = r.converged;
    }
    if !(abs(p.d - own) < GN_MIN_STEP) {
        p.flags |= MOVED;
    }
    if converged {
        p.flags |= SETTLED;
    }
    patches[k] = p;
}

// How much patch k's SSD rises from its estimate p when D moves to d.
fn rise(k: u32, p: Patch, d: f32, lid: u32) -> f32 {
    let e = eval(k, d, false, lid);
    if e.ok {
        return e.ssd - p.ssd;
    }
    return 0.0;
}

// Each patch's inverse variance (0 without an estimate) and validity: an
// estimate inside the right image, with texture, a small residual and the
// two eyes' brightness close, and on validated levels a distinct match (its
// SSD rises when D moves DISTINCT_STEP either way).
@compute @workgroup_size(64)
fn validity(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = gid.x;
    if k >= P.ws * P.hs {
        return;
    }
    let p = patches[k];
    let nn = f32(PSZ * PSZ);
    var info = PatchInfo(0.0, 0u);
    let ok = (p.flags & OK) != 0u;
    if ok {
        let sigma2 = max(p.ssd / (nn - 1.0), NOISE_FLOOR * NOISE_FLOOR);
        info.prec = max(p.hess, 1e-6) / sigma2;
    }
    if ok && (p.flags & INSIDE) != 0u && p.hess >= P.min_hess && p.ssd <= P.max_ssd
        && abs(p.offset) <= P.max_bright {
        var valid = true;
        if P.validate != 0u {
            let lo = rise(k, p, p.d - DISTINCT_STEP, lid);
            let hi = rise(k, p, p.d + DISTINCT_STEP, lid);
            valid = min(lo, hi) >= P.min_rise;
        }
        info.valid = select(0u, 1u, valid);
    }
    pinfo[k] = info;
}
