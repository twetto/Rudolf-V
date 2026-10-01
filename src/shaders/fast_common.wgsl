// fast_common.wgsl — FAST-N scoring shared by fast.wgsl (dense score buffer)
// and fast_nms_fused.wgsl (per-cell winners). Concatenated after each
// kernel's header by gpu/fast.rs; expects module-scope `params` with fields
// img_width, img_height, threshold (f32) and arc_length, and a
// `fn load_pixel(x: i32, y: i32) -> f32` returning the clamp-to-edge pixel
// value (from the texture, or from a shared-memory tile holding exactly the
// same values).

var<private> OFFSETS_X: array<i32, 16> = array<i32, 16>(
     0,  1,  2,  3,  3,  3,  2,  1,
     0, -1, -2, -3, -3, -3, -2, -1
);
var<private> OFFSETS_Y: array<i32, 16> = array<i32, 16>(
    -3, -3, -2, -1,  0,  1,  2,  3,
     3,  3,  2,  1,  0, -1, -2, -3
);

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

/// FAST score at (x, y), or 0.0 if it is not a corner (or lies within the
/// 3-pixel border). Every corner scores > 0: each arc pixel differs from the
/// centre by more than the threshold.
fn corner_score(x: i32, y: i32) -> f32 {
    if x < 3 || y < 3
       || x >= i32(params.img_width)  - 3
       || y >= i32(params.img_height) - 3 { return 0.0; }

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
    if bc < min_card && dc < min_card { return 0.0; }

    let masks      = build_masks(x, y, center);
    let bright_arc = has_arc(masks.x, params.arc_length);
    let dark_arc   = has_arc(masks.y, params.arc_length);
    if !bright_arc && !dark_arc { return 0.0; }

    var score: f32 = 0.0;
    if bright_arc { score = max(score, arc_score(x, y, center, masks.x)); }
    if dark_arc   { score = max(score, arc_score(x, y, center, masks.y)); }
    return score;
}
