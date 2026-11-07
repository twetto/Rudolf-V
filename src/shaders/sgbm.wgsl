// sgbm.wgsl — semi-global block matching on one pyramid level, as sgbm.rs
// (its header has the steps); the same integers:
//   spans        both images' half-pixel spans, prefiltered and grey (step 2)
//   pixel_costs  a pixel's cost at every disparity (step 3)
//   box_rows     the costs summed along rows over the block (step 4) ...
//   box_cols     ... then along columns, into the pixel costs' buffer; also
//                clears the path sums and the right pixels' best matches
//   paths        four scanlines of the path directions per workgroup: their
//                path costs, added to the sums (step 5)
//   winners      each pixel's disparity, uniqueness and best match (step 6)
//   left_right   the left-right check (step 7); starts the union-find
//   unite, count the speckles' regions (step 8), as dense_stereo_mask.wgsl
//   finish       drop the small regions; the disparity map, NaN without one
// Each kernel binds only what it uses. {{WG_X}} / {{WG_Y}}: the 2-D
// workgroup; {{SPECKLE_STEP}}: dense_stereo.rs's constant.

struct Params {
    w: u32,
    h: u32,
    nd: u32,           // disparities 0..nd
    half: u32,         // block_size / 2
    p1: u32,
    p2: u32,
    uniqueness: u32,   // percent
    max_diff: u32,     // left-right tolerance; NO_CHECK: no check
    min_speckle: u32,  // smallest region kept (pixels of the level); 0: all
    n_lines: u32,      // path scanlines
    cap: f32,          // pre_filter_cap
    ndp: u32,          // words per pixel in the volumes: ceil(nd / 2)
}

// A pixel's prefiltered and grey value with the least and greatest of it and
// its half-pixel neighbours along the row.
struct Span {
    p: f32,
    pl: f32,
    ph: f32,
    g: f32,
    gl: f32,
    gh: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var left_tex: texture_2d<f32>;
@group(0) @binding(2) var right_tex: texture_2d<f32>;
// The left image's spans, then the right image's.
@group(0) @binding(3) var<storage, read_write> spans: array<Span>;
// The cost volumes hold 16-bit costs, two disparities per word (d = 2k in
// word k's low half, 2k + 1 in its high half; the last word's high half
// unused when nd is odd), as OpenCV's: half the memory traffic, and every
// stage is bound by it. Halves add without carrying into each other while
// they stay under 2^16, which GpuSgbm::new checks for the settings.
// Per pixel, ndp words: pixel costs, then block costs.
@group(0) @binding(4) var<storage, read_write> cost: array<u32>;
// The row sums of box_rows.
@group(0) @binding(5) var<storage, read_write> tmp: array<u32>;
// The path costs' sum S.
@group(0) @binding(7) var<storage, read_write> total: array<atomic<u32>>;
// Per scanline: its first pixel and its step (x, y, dx, dy).
@group(0) @binding(8) var<storage, read> scanlines: array<vec4<i32>>;
@group(0) @binding(9) var<storage, read_write> disp: array<f32>;
// Per pixel, its best (S << 8 | d) if unique and inside; per right pixel the
// least of those matching it.
@group(0) @binding(10) var<storage, read_write> best: array<u32>;
@group(0) @binding(11) var<storage, read_write> right_best: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> valid: array<u32>;
// Union-find: each pixel's parent (never above it), and per root its
// region's size.
@group(0) @binding(13) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> region: array<atomic<u32>>;
// The disparity map: f32 bits, NaN's where there is none.
@group(0) @binding(15) var<storage, read_write> out: array<u32>;

const NO_MATCH: u32 = 0xffffffffu;
const NO_CHECK: u32 = 0xffffffffu;
const NAN_BITS: u32 = 0x7fc00000u;
const SPECKLE_STEP: f32 = {{SPECKLE_STEP}};

// ---------------------------------------------------------------------------
// Steps 2-4
// ---------------------------------------------------------------------------

// The prefiltered value at x of row y (3x3 Sobel along x, clamped and
// offset; none in the first and last columns).
fn prefiltered(t: texture_2d<f32>, x: i32, y: i32) -> f32 {
    if x == 0 || x + 1 == i32(P.w) {
        return P.cap;
    }
    let yu = max(y - 1, 0);
    let yd = min(y + 1, i32(P.h) - 1);
    let sobel = 2.0 * (textureLoad(t, vec2<i32>(x + 1, y), 0).r - textureLoad(t, vec2<i32>(x - 1, y), 0).r)
        + (textureLoad(t, vec2<i32>(x + 1, yu), 0).r - textureLoad(t, vec2<i32>(x - 1, yu), 0).r)
        + (textureLoad(t, vec2<i32>(x + 1, yd), 0).r - textureLoad(t, vec2<i32>(x - 1, yd), 0).r);
    return clamp(sobel, -P.cap, P.cap) + P.cap;
}

// (value, least, greatest) of c and its half-pixel neighbours l, r (the
// pixel's own value where there is no neighbour).
fn span3(c: f32, l: f32, r: f32, x: i32) -> vec3<f32> {
    var vl = c;
    var vr = c;
    if x > 0 {
        vl = 0.5 * (c + l);
    }
    if x + 1 < i32(P.w) {
        vr = 0.5 * (c + r);
    }
    return vec3<f32>(c, min(min(c, vl), vr), max(max(c, vl), vr));
}

fn span_of(t: texture_2d<f32>, x: i32, y: i32) -> Span {
    let xl = max(x - 1, 0);
    let xr = min(x + 1, i32(P.w) - 1);
    let p = span3(prefiltered(t, x, y), prefiltered(t, xl, y), prefiltered(t, xr, y), x);
    let g = span3(
        textureLoad(t, vec2<i32>(x, y), 0).r,
        textureLoad(t, vec2<i32>(xl, y), 0).r,
        textureLoad(t, vec2<i32>(xr, y), 0).r,
        x,
    );
    return Span(p.x, p.y, p.z, g.x, g.y, g.z);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn spans_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    spans[i] = span_of(left_tex, i32(gid.x), i32(gid.y));
    spans[P.w * P.h + i] = span_of(right_tex, i32(gid.x), i32(gid.y));
}

fn bt(u: f32, ul: f32, uh: f32, v: f32, vl: f32, vh: f32) -> f32 {
    let c0 = max(max(u - vh, vl - u), 0.0);
    let c1 = max(max(v - uh, ul - v), 0.0);
    return min(c0, c1);
}

fn pixel_cost(l: Span, r: Span) -> u32 {
    let p = bt(l.p, l.pl, l.ph, r.p, r.pl, r.ph);
    let g = bt(l.g, l.gl, l.gh, r.g, r.gl, r.gh);
    return u32(floor(p + 0.25 * g + 0.5));
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn pixel_costs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    let lp = spans[i];
    let inside = min(P.nd, gid.x + 1u);
    var least = 0xffffffffu;
    var sum = 0u;
    var low = 0u;
    for (var d = 0u; d < inside; d++) {
        let c = pixel_cost(lp, spans[P.w * P.h + i - d]);
        least = min(least, c);
        sum += c;
        if (d & 1u) == 0u {
            low = c;
        } else {
            cost[i * P.ndp + d / 2u] = low | (c << 16u);
        }
    }
    // Outside the right image: halfway between the least and the mean.
    let fill = (least + sum / inside) / 2u;
    for (var d = inside; d < 2u * P.ndp; d++) {
        let c = select(fill, 0u, d >= P.nd);
        if (d & 1u) == 0u {
            low = c;
        } else {
            cost[i * P.ndp + d / 2u] = low | (c << 16u);
        }
    }
}

// The volume word (pixel, disparity pair) of a thread in a flat dispatch, or
// past the end.
fn element(wid: vec3<u32>, nwg: vec3<u32>, lid: u32) -> u32 {
    return (wid.y * nwg.x + wid.x) * 64u + lid;
}

@compute @workgroup_size(64)
fn box_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let e = element(wid, nwg, lid);
    if e >= P.w * P.h * P.ndp {
        return;
    }
    let i = e / P.ndp;
    let d = e % P.ndp;
    let x = i32(i % P.w);
    let row = i - u32(x);
    var s = 0u;
    for (var k = -i32(P.half); k <= i32(P.half); k++) {
        let xx = u32(clamp(x + k, 0, i32(P.w) - 1));
        s += cost[(row + xx) * P.ndp + d];
    }
    tmp[e] = s;
}

@compute @workgroup_size(64)
fn box_cols(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let e = element(wid, nwg, lid);
    if e >= P.w * P.h * P.ndp {
        return;
    }
    let i = e / P.ndp;
    let d = e % P.ndp;
    let x = i % P.w;
    let y = i32(i / P.w);
    var s = 0u;
    for (var k = -i32(P.half); k <= i32(P.half); k++) {
        let yy = u32(clamp(y + k, 0, i32(P.h) - 1));
        s += tmp[(yy * P.w + x) * P.ndp + d];
    }
    cost[e] = s;
    atomicStore(&total[e], 0u);
    if d == 0u {
        atomicStore(&right_best[i], NO_MATCH);
    }
}

// ---------------------------------------------------------------------------
// Step 5: path costs
// ---------------------------------------------------------------------------

// Four scanlines per workgroup, a 32-thread group (one warp on NVIDIA) per
// scanline: thread t of a group holds volume words t, t + 32, ... (up to 4:
// disparities 2t, 2t + 1, 2t + 64, ...; nd <= 256). Few threads per scanline and several scanlines per workgroup
// let more scanlines run at once than one 64-thread workgroup per scanline,
// where the GPU's limit on resident workgroups left most of it idle
// (the Orin Nano: 16 per SM).
const LINES: u32 = 4u;
const LANES: u32 = 32u;
const PER_LANE: u32 = 4u;

// Per scanline: the previous pixel's path costs, and each thread's least of
// the current pixel's (reduced to [0]).
var<workgroup> prev: array<array<u32, 256>, 4>;
var<workgroup> mins: array<array<u32, 32>, 4>;

// The number of pixels on a scanline.
fn line_length(s: vec4<i32>) -> u32 {
    var n = 0x7fffffff;
    if s.z > 0 {
        n = min(n, i32(P.w) - s.x);
    } else if s.z < 0 {
        n = min(n, s.x + 1);
    }
    if s.w > 0 {
        n = min(n, i32(P.h) - s.y);
    } else if s.w < 0 {
        n = min(n, s.y + 1);
    }
    return u32(n);
}

@compute @workgroup_size(128)
fn paths(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let g = lid / LANES;
    let t = lid % LANES;
    let first_line = (wid.y * nwg.x + wid.x) * LINES;
    if first_line >= P.n_lines {
        return;
    }
    // The workgroup steps as long as its longest scanline (uniform, so the
    // barriers are too); a group whose scanline has ended idles through them.
    var steps = 0u;
    for (var k = 0u; k < LINES; k++) {
        if first_line + k < P.n_lines {
            steps = max(steps, line_length(scanlines[first_line + k]));
        }
    }
    let line = first_line + g;
    var s = vec4<i32>(0, 0, 0, 0);
    var length = 0u;
    if line < P.n_lines {
        s = scanlines[line];
        length = line_length(s);
    }
    var x = s.x;
    var y = s.y;
    var m = 0u;
    // The pixel's block costs, loaded one pixel ahead: the next pixel's
    // loads are in flight while this one waits at its barriers.
    var costs: array<u32, 4>;
    if length > 0u {
        for (var j = 0u; j < PER_LANE; j++) {
            let k = t + LANES * j;
            if k < P.ndp {
                costs[j] = cost[(u32(y) * P.w + u32(x)) * P.ndp + k];
            }
        }
    }
    for (var step = 0u; step < steps; step++) {
        let live = step < length;
        let i = u32(y) * P.w + u32(x);
        var next: array<u32, 4>;
        if step + 1u < length {
            let ni = u32(y + s.w) * P.w + u32(x + s.z);
            for (var j = 0u; j < PER_LANE; j++) {
                let k = t + LANES * j;
                if k < P.ndp {
                    next[j] = cost[ni * P.ndp + k];
                }
            }
        }
        // Per word, its two disparities' path costs (0 past nd).
        var cur: array<vec2<u32>, 4>;
        var least = 0xffffffffu;
        if live {
            for (var j = 0u; j < PER_LANE; j++) {
                let k = t + LANES * j;
                if k < P.ndp {
                    for (var h = 0u; h < 2u; h++) {
                        let d = 2u * k + h;
                        if d < P.nd {
                            let c = (costs[j] >> (16u * h)) & 0xffffu;
                            var l = c;
                            if step > 0u {
                                var b = min(prev[g][d], m + P.p2);
                                if d > 0u {
                                    b = min(b, prev[g][d - 1u] + P.p1);
                                }
                                if d + 1u < P.nd {
                                    b = min(b, prev[g][d + 1u] + P.p1);
                                }
                                l = c + b - m;
                            }
                            cur[j][h] = l;
                            least = min(least, l);
                        }
                    }
                }
            }
        }
        // Everyone has read the previous pixel's costs.
        workgroupBarrier();
        if live {
            for (var j = 0u; j < PER_LANE; j++) {
                let k = t + LANES * j;
                if k < P.ndp {
                    prev[g][2u * k] = cur[j].x;
                    prev[g][2u * k + 1u] = cur[j].y;
                    atomicAdd(&total[i * P.ndp + k], cur[j].x | (cur[j].y << 16u));
                }
            }
        }
        mins[g][t] = least;
        workgroupBarrier();
        for (var k = LANES / 2u; k > 0u; k >>= 1u) {
            if t < k {
                mins[g][t] = min(mins[g][t], mins[g][t + k]);
            }
            workgroupBarrier();
        }
        m = mins[g][0];
        costs = next;
        x += s.z;
        y += s.w;
    }
}

// ---------------------------------------------------------------------------
// Steps 6-8
// ---------------------------------------------------------------------------

// S at disparity d of the pixel whose words start at `base`.
fn sum_at(base: u32, d: u32) -> u32 {
    return (atomicLoad(&total[base + d / 2u]) >> (16u * (d & 1u))) & 0xffffu;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn winners(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    let base = i * P.ndp;
    // The least sum, the first of equals.
    var db = 0u;
    var sb = 0xffffffffu;
    for (var d = 0u; d < P.nd; d++) {
        let v = sum_at(base, d);
        if v < sb {
            sb = v;
            db = d;
        }
    }
    var unique = true;
    for (var d = 0u; d < P.nd; d++) {
        let v = sum_at(base, d);
        let far = select(db - d, d - db, d > db) > 1u;
        if far && v * (100u - P.uniqueness) < sb * 100u {
            unique = false;
        }
    }
    var dd = f32(db);
    if db > 0u && db + 1u < P.nd {
        let a = i32(sum_at(base, db - 1u));
        let b = i32(sb);
        let c = i32(sum_at(base, db + 1u));
        let denom = max(a + c - 2 * b, 1);
        dd = f32(db) + f32(a - c) / f32(2 * denom);
    }
    disp[i] = dd;
    valid[i] = select(0u, 1u, unique && f32(gid.x) - dd >= 0.0);
    var b = NO_MATCH;
    if unique && db <= gid.x {
        b = (sb << 8u) | db;
        atomicMin(&right_best[i - db], b);
    }
    best[i] = b;
}

// Whether the right pixel at disparity dd of left pixel x (row `row`) is best
// matched more than max_diff away from dd.
fn off(row: u32, x: u32, dd: u32) -> bool {
    if dd > x {
        return false;
    }
    let b = atomicLoad(&right_best[row + x - dd]);
    let bd = b & 0xffu;
    return b != NO_MATCH && select(dd - bd, bd - dd, bd > dd) > P.max_diff;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn left_right(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    atomicStore(&parent[i], i);
    atomicStore(&region[i], 0u);
    if P.max_diff == NO_CHECK || valid[i] == 0u {
        return;
    }
    let d = disp[i];
    let row = gid.y * P.w;
    if off(row, gid.x, u32(floor(d))) && off(row, gid.x, u32(ceil(d))) {
        valid[i] = 0u;
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
// merged in turn.
fn unite_roots(a0: u32, b0: u32) {
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
    return valid[j] != 0u && abs(disp[i] - disp[j]) <= SPECKLE_STEP;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn unite(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if valid[i] == 0u {
        return;
    }
    if gid.x > 0u && joined(i, i - 1u) {
        unite_roots(i, i - 1u);
    }
    if gid.y > 0u && joined(i, i - P.w) {
        unite_roots(i, i - P.w);
    }
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    if valid[i] != 0u {
        atomicAdd(&region[find(i)], 1u);
    }
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.w || gid.y >= P.h {
        return;
    }
    let i = gid.y * P.w + gid.x;
    var keep = valid[i] != 0u;
    if keep && P.min_speckle > 0u && atomicLoad(&region[find(i)]) < P.min_speckle {
        keep = false;
    }
    out[i] = select(NAN_BITS, bitcast<u32>(disp[i]), keep);
}
