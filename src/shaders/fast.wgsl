// fast.wgsl — FAST-N corner detector compute shader.
//
// OUTPUT PATTERN: DENSE SCORE BUFFER (no atomics)
// ─────────────────────────────────────────────────
// Rather than claiming slots with atomicAdd (which triggers a naga SPIR-V
// memory-semantics bug on strict Vulkan validation), each thread writes
// directly to its own slot in a flat score buffer:
//
//   scores[y * img_width + x] = score   // if corner
//   scores[y * img_width + x] = 0.0     // not a corner (wgpu zero-fills)
//
// No two threads share a slot, so no synchronisation is needed.
// CPU collects all nonzero entries after readback.
// Buffer: img_w × img_h × 4 bytes (≈1.4 MB for EuRoC 752×480).
//
// WORKGROUP SIZE: {{WG_X}} × {{WG_Y}} (substituted at compile time)

@group(0) @binding(0) var input_tex: texture_2d<f32>;

/// Dense score buffer: index = y * img_width + x.
/// 0.0 = not a corner; positive = FAST score.
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;

@group(0) @binding(2) var<uniform> params: FastParams;

struct FastParams {
    img_width:  u32,
    img_height: u32,
    threshold:  f32,
    arc_length: u32,
}

var<private> OFFSETS_X: array<i32, 16> = array<i32, 16>(
     0,  1,  2,  3,  3,  3,  2,  1,
     0, -1, -2, -3, -3, -3, -2, -1
);
var<private> OFFSETS_Y: array<i32, 16> = array<i32, 16>(
    -3, -3, -2, -1,  0,  1,  2,  3,
     3,  3,  2,  1,  0, -1, -2, -3
);

fn load_pixel(x: i32, y: i32) -> f32 {
    let c = vec2<i32>(
        clamp(x, 0, i32(params.img_width)  - 1),
        clamp(y, 0, i32(params.img_height) - 1),
    );
    return textureLoad(input_tex, c, 0).r;
}

fn build_masks(cx: i32, cy: i32, center: f32) -> vec2<u32> {
    var bright: u32 = 0u;
    var dark:   u32 = 0u;
    let hi = center + params.threshold;
    let lo = center - params.threshold;
    for (var i: u32 = 0u; i < 16u; i++) {
        let v = load_pixel(cx + OFFSETS_X[i], cy + OFFSETS_Y[i]);
        if v > hi { bright |= (1u << i); }
        if v < lo { dark   |= (1u << i); }
    }
    return vec2<u32>(bright, dark);
}

fn has_arc(mask: u32, n: u32) -> bool {
    if n == 0u { return true; }
    var acc: u32 = mask | (mask << 16u);
    for (var k: u32 = 1u; k < n; k++) {
        acc &= (acc >> 1u);
    }
    return (acc & 0xFFFFu) != 0u;
}

fn arc_score(cx: i32, cy: i32, center: f32, mask: u32) -> f32 {
    let m32: u32 = mask | (mask << 16u);
    var best_start: u32 = 0u;
    var best_len:   u32 = 0u;
    var i: u32 = 0u;
    loop {
        if i >= 16u { break; }
        if (m32 & (1u << i)) == 0u { i++; continue; }
        let start = i;
        loop {
            if i >= 32u || (m32 & (1u << i)) == 0u { break; }
            i++;
        }
        let run_len = i - start;
        if run_len > best_len { best_len = run_len; best_start = start; }
    }
    var score: f32 = 0.0;
    for (var j: u32 = best_start; j < best_start + best_len; j++) {
        let idx = j % 16u;
        let v = load_pixel(cx + OFFSETS_X[idx], cy + OFFSETS_Y[idx]);
        score += max(abs(v - center) - params.threshold, 0.0);
    }
    return score;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn detect_corners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if x < 3 || y < 3
       || x >= i32(params.img_width)  - 3
       || y >= i32(params.img_height) - 3 { return; }

    let center    = load_pixel(x, y);
    let min_card  = select(2u, 3u, params.arc_length >= 12u);
    let hi = center + params.threshold;
    let lo = center - params.threshold;

    let p0  = load_pixel(x + OFFSETS_X[0],  y + OFFSETS_Y[0]);
    let p4  = load_pixel(x + OFFSETS_X[4],  y + OFFSETS_Y[4]);
    let p8  = load_pixel(x + OFFSETS_X[8],  y + OFFSETS_Y[8]);
    let p12 = load_pixel(x + OFFSETS_X[12], y + OFFSETS_Y[12]);

    let bc = u32(p0>hi) + u32(p4>hi) + u32(p8>hi) + u32(p12>hi);
    let dc = u32(p0<lo) + u32(p4<lo) + u32(p8<lo) + u32(p12<lo);
    if bc < min_card && dc < min_card { return; }

    let masks      = build_masks(x, y, center);
    let bright_arc = has_arc(masks.x, params.arc_length);
    let dark_arc   = has_arc(masks.y, params.arc_length);
    if !bright_arc && !dark_arc { return; }

    var score: f32 = 0.0;
    if bright_arc { score = max(score, arc_score(x, y, center, masks.x)); }
    if dark_arc   { score = max(score, arc_score(x, y, center, masks.y)); }

    // Write directly to this pixel's own slot — zero contention, no atomics.
    scores[u32(y) * params.img_width + u32(x)] = score;
}
