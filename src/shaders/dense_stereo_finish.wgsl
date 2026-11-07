// dense_stereo_finish.wgsl — the finest level's depth map, as dense_stereo.rs:
//   disagree  per cell of a coarser level, whether a finest-level measurement
//             whose nearest cell it is disagrees with it (`coarse_fallback`)
//   dilate    whether any cell within `reach` along its row does
//   fill      pixels without an estimate take the coarser level's unless a
//             cell within `reach` along the column is flagged by dilate
//             (with it, the CPU's box test on its integral image); one
//             dispatch per fallback level, finest first
//   reject    drop fills the right image contradicts
//             (`reject_contradicted_fills`; a workgroup per row)
//   finish    log-range, its variance and the source level, packed for the
//             readback (`finish`)
// {{DISAGREE}}, {{FALLBACK_AGREE}}, {{FLAT_END_MARGIN}}, {{NO_SOURCE}}:
// dense_stereo.rs's constants.

struct FinishParams {
    w: u32,              // the finest level
    h: u32,
    cw: u32,             // the fallback's coarser level
    ch: u32,
    k: u32,              // its level − the finest
    level: u32,          // its level
    finest: u32,
    first: u32,          // the first fallback level: sources start from validity
    reach: u32,          // a coarse estimate's support, in cells either way
    has_fallback: u32,   // the coarse fallback ran: `source` holds the sources
    flat_gradient: f32,  // textureless: |gx| under this
    max_residual: f32,
    ln_s: f32,           // log-range = ln_s − ln D
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> F: FinishParams;
// The finest level's dense D and validity.
@group(0) @binding(1) var<storage, read_write> dense_d: array<f32>;
@group(0) @binding(2) var<storage, read_write> dense_valid: array<u32>;
// The coarser level's (fallback).
@group(0) @binding(3) var<storage, read_write> coarse_d: array<f32>;
@group(0) @binding(4) var<storage, read_write> coarse_valid: array<u32>;
// Per coarse cell: disagree's flags, and dilate's.
@group(0) @binding(5) var<storage, read_write> flag: array<u32>;
@group(0) @binding(6) var<storage, read_write> flag_x: array<u32>;
@group(0) @binding(7) var<storage, read_write> dense_var: array<f32>;
@group(0) @binding(8) var<storage, read_write> coarse_var: array<f32>;
// Per finest pixel, the level its estimate comes from (NO_SOURCE: none).
@group(0) @binding(9) var<storage, read_write> source: array<u32>;
// The finest level's left grey level with gx, its right level, and the
// right image's shift per unit of D.
@group(0) @binding(10) var<storage, read_write> tpl: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> right_img: array<f32>;
@group(0) @binding(12) var<storage, read_write> pixel_n: array<f32>;
// log-range bits (w·h), variance bits (w·h), sources four to a word.
@group(0) @binding(13) var<storage, read_write> out: array<u32>;

const DISAGREE: f32 = {{DISAGREE}};
const FALLBACK_AGREE: f32 = {{FALLBACK_AGREE}};
const FLAT_END_MARGIN: u32 = {{FLAT_END_MARGIN}}u;
const NO_SOURCE: u32 = {{NO_SOURCE}}u;
// A quiet NaN's bits, written for pixels without depth (WGSL floats may
// assume there are none).
const NAN_BITS: u32 = 0x7fc00000u;

fn sat_sub(a: u32, b: u32) -> u32 {
    return select(0u, a - b, a > b);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn disagree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    if cx >= F.cw || cy >= F.ch {
        return;
    }
    let j = cy * F.cw + cx;
    var any = 0u;
    if coarse_valid[j] != 0u {
        // The finest pixels whose nearest cell this is.
        let half = 1u << (F.k - 1u);
        let x0 = sat_sub(cx << F.k, half);
        let y0 = sat_sub(cy << F.k, half);
        var x1 = F.w;
        if cx + 1u < F.cw {
            x1 = min(((cx + 1u) << F.k) - half, F.w);
        }
        var y1 = F.h;
        if cy + 1u < F.ch {
            y1 = min(((cy + 1u) << F.k) - half, F.h);
        }
        let dc = f32(1u << F.k) * coarse_d[j];
        for (var y = y0; y < y1 && any == 0u; y++) {
            for (var x = x0; x < x1; x++) {
                let i = y * F.w + x;
                if dense_valid[i] == 0u {
                    continue;
                }
                let df = dense_d[i];
                if abs(df - dc) > max(DISAGREE, FALLBACK_AGREE * df) {
                    any = 1u;
                    break;
                }
            }
        }
    }
    flag[j] = any;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    if cx >= F.cw || cy >= F.ch {
        return;
    }
    let row = cy * F.cw;
    var any = 0u;
    for (var x = sat_sub(cx, F.reach); x < min(cx + F.reach + 1u, F.cw); x++) {
        any |= flag[row + x];
    }
    flag_x[row + cx] = any;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= F.w || y >= F.h {
        return;
    }
    let i = y * F.w + x;
    var s = NO_SOURCE;
    if dense_valid[i] != 0u {
        s = F.finest;
    } else if F.first == 0u {
        s = source[i];
    }
    if s == NO_SOURCE {
        let half = 1u << (F.k - 1u);
        let cx = min((x + half) >> F.k, F.cw - 1u);
        let cy = min((y + half) >> F.k, F.ch - 1u);
        let j = cy * F.cw + cx;
        if coarse_valid[j] != 0u {
            var blocked = 0u;
            for (var yy = sat_sub(cy, F.reach); yy < min(cy + F.reach + 1u, F.ch); yy++) {
                blocked |= flag_x[yy * F.cw + cx];
            }
            if blocked == 0u {
                let f = f32(1u << F.k);
                dense_d[i] = f * coarse_d[j];
                dense_var[i] = f * f * coarse_var[j];
                s = F.level;
            }
        }
    }
    source[i] = s;
}

// The right image with its border repeated (the CPU's padding).
fn right_at(x: i32, y: u32) -> f32 {
    return right_img[y * F.w + u32(clamp(x, 0, i32(F.w) - 1))];
}

// Whether the right image sees pixel (x, y) at D inside it, within
// max_residual of the left grey level (`seen_at`, `agrees`).
fn agrees(x: u32, y: u32, d: f32) -> bool {
    let i = y * F.w + x;
    let n = pixel_n[i];
    if n == 0.0 {
        return false;
    }
    let loc = f32(x) + d * n;
    if !(loc >= 0.0 && loc <= f32(F.w - 1u)) {
        return false;
    }
    // `Padded::sample` on the row: its vertical weight is 0.
    let b = f32(BORDER);
    let xs = clamp(loc, -b + 1.0, f32(F.w) + b - 2.0) + b;
    let x0 = floor(xs);
    let fx = xs - x0;
    let xx = i32(x0) - BORDER;
    let v = (1.0 - fx) * right_at(xx, y) + fx * right_at(xx + 1, y);
    return abs(v - tpl[i].x) <= F.max_residual;
}

// A textureless run's end checked at D give or take `half` (an end at the
// image border meets no texture).
fn end_agrees(xe: u32, border: bool, y: u32, d: f32, half: f32) -> bool {
    return border || agrees(xe, y, d) || agrees(xe, y, d - half) || agrees(xe, y, d + half);
}

fn textured(i: u32) -> bool {
    return !(abs(tpl[i].y) < F.flat_gradient);
}

const ROW_THREADS: u32 = 64u;
// Pixels per thread at most (rows up to 2048 pixels).
const MAX_CHUNK: u32 = 32u;
// Per thread: the last and the first textured pixel of its chunk (−1, w:
// none).
var<workgroup> last_textured: array<i32, ROW_THREADS>;
var<workgroup> first_textured: array<i32, ROW_THREADS>;

// One row per workgroup, a chunk of it per thread. A pixel's run is itself
// if textured, else the textureless pixels between the nearest textured
// ones: the chunks' first and last textured pixels give each thread those
// outside its chunk.
@compute @workgroup_size(64)
fn reject(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    let row = y * F.w;
    let chunk = (F.w + ROW_THREADS - 1u) / ROW_THREADS;
    let c0 = min(lid * chunk, F.w);
    let c1 = min(c0 + chunk, F.w);
    var lt = -1;
    var ft = i32(F.w);
    for (var x = c0; x < c1; x++) {
        if textured(row + x) {
            lt = i32(x);
            ft = min(ft, i32(x));
        }
    }
    last_textured[lid] = lt;
    first_textured[lid] = ft;
    workgroupBarrier();
    var prev = -1;
    for (var t = 0u; t < lid; t++) {
        prev = max(prev, last_textured[t]);
    }
    var next = i32(F.w);
    for (var t = lid + 1u; t < ROW_THREADS; t++) {
        next = min(next, first_textured[t]);
    }
    // The next textured pixel after each of the chunk's.
    var after: array<i32, MAX_CHUNK>;
    for (var x = c1; x > c0; x--) {
        after[x - 1u - c0] = next;
        if textured(row + x - 1u) {
            next = i32(x - 1u);
        }
    }
    for (var x = c0; x < c1; x++) {
        let i = row + x;
        let flat = !textured(i);
        var first = x;
        var last = x;
        if flat {
            first = u32(prev + 1);
            last = u32(after[x - c0] - 1);
        } else {
            prev = i32(x);
        }
        let src = source[i];
        if src == NO_SOURCE || src == F.finest {
            continue;
        }
        let d = dense_d[i];
        let half = 0.5 * f32(1u << (src - F.finest));
        var ok = agrees(x, y, d);
        if ok && flat {
            let m = min(FLAT_END_MARGIN, (last - first) / 2u);
            ok = end_agrees(first + m, first == 0u, y, d, half)
                && end_agrees(last - m, last == F.w - 1u, y, d, half);
        }
        if !ok {
            source[i] = NO_SOURCE;
        }
    }
}

// Four pixels per thread: their log-range and variance (NaN without
// depth), and their sources in one word.
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = F.w * F.h;
    let q = gid.x;
    if q * 4u >= n {
        return;
    }
    var packed = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = q * 4u + j;
        var s = NO_SOURCE;
        if i < n {
            if F.has_fallback != 0u {
                s = source[i];
            } else if dense_valid[i] != 0u {
                s = F.finest;
            }
            if s != NO_SOURCE {
                let d = dense_d[i];
                out[i] = bitcast<u32>(F.ln_s - log(d));
                out[n + i] = bitcast<u32>(dense_var[i] / (d * d));
            } else {
                out[i] = NAN_BITS;
                out[n + i] = NAN_BITS;
            }
        }
        packed |= s << (8u * j);
    }
    out[2u * n + q] = packed;
}
