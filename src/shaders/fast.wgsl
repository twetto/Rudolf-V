// fast.wgsl — FAST-N corner detector compute shader.
//
// Each thread handles one pixel. The algorithm is identical to the CPU
// implementation in fast.rs — same bitmask AND-shift trick, same score
// formula, same circle offsets. Comments focus on WGSL-specific details.
//
// OUTPUT PATTERN: GPU-side dynamic list via atomic counter
// ─────────────────────────────────────────────────────────
// We can't know ahead of time how many corners exist, so we pre-allocate a
// keypoint buffer large enough for the worst case and use an atomic counter
// to claim slots:
//
//   slot = atomicAdd(&counter, 1u)   // claims slot, returns old value
//   if slot < max_features:
//       keypoints[slot] = { x, y, score }
//
// The CPU reads back `counter` first, then only the first `counter` entries
// of the keypoint buffer. Threads that overflow the buffer are silently
// dropped (handled via the max_features guard).
//
// WORKGROUP SIZE: {{WG_X}} × {{WG_Y}} (substituted by fast.rs at compile time)

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var input_tex:        texture_2d<f32>;
// Wrapping atomic<u32> in a struct works around a naga bug where a bare
// storage atomic generates SPIR-V without the required UniformMemory
// semantics bit, causing Vulkan validation errors.
struct AtomicCounter { value: atomic<u32>, }
@group(0) @binding(1) var<storage, read_write> counter: AtomicCounter;
@group(0) @binding(2) var<storage, read_write> keypoints: array<GpuFeature>;
@group(0) @binding(3) var<uniform>            params:     FastParams;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

struct GpuFeature {
    x:     f32,
    y:     f32,
    score: f32,
    _pad:  f32,   // pad to 16 bytes for storage buffer alignment
}

struct FastParams {
    img_width:   u32,
    img_height:  u32,
    threshold:   f32,   // intensity difference threshold (same scale as pixel values)
    arc_length:  u32,   // N in FAST-N (9..=12)
    max_features: u32,  // capacity of the keypoints buffer
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

// ---------------------------------------------------------------------------
// Circle offsets — Bresenham radius-3, 16 points, clockwise from 12 o'clock.
// Matches CIRCLE_OFFSETS in fast.rs exactly.
// ---------------------------------------------------------------------------

var<private> OFFSETS_X: array<i32, 16> = array<i32, 16>(
     0,  1,  2,  3,  3,  3,  2,  1,
     0, -1, -2, -3, -3, -3, -2, -1
);
var<private> OFFSETS_Y: array<i32, 16> = array<i32, 16>(
    -3, -3, -2, -1,  0,  1,  2,  3,
     3,  3,  2,  1,  0, -1, -2, -3
);

// Cardinal indices (0, 4, 8, 12) for the quick rejection test.
// Same as the CPU: p0=top, p4=right, p8=bottom, p12=left.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load one pixel from the texture, clamped to image bounds.
/// Returns the raw f32 value (in [0, 255] range — same as CPU u8 cast).
fn load_pixel(x: i32, y: i32) -> f32 {
    let clamped = vec2<i32>(
        clamp(x, 0, i32(params.img_width)  - 1),
        clamp(y, 0, i32(params.img_height) - 1),
    );
    return textureLoad(input_tex, clamped, 0).r;
}

/// Build bright/dark bitmasks for the 16 circle pixels.
/// Bit i set in bright_mask ↔ circle[i] > center + threshold.
/// Bit i set in dark_mask  ↔ circle[i] < center - threshold.
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

/// Check whether a u16 circular bitmask (packed in low 16 bits of m)
/// contains N contiguous set bits. Uses the AND-shift trick:
///   acc = m | (m << 16)    // double the pattern for wrap-around
///   repeat N-1 times: acc &= acc >> 1
///   nonzero result → run of at least N exists
fn has_arc(mask: u32, n: u32) -> bool {
    if n == 0u { return true; }
    let m32: u32 = mask | (mask << 16u);
    var acc: u32 = m32;
    for (var k: u32 = 1u; k < n; k++) {
        acc &= (acc >> 1u);
    }
    // Only the lower 16 bits of acc are meaningful (the upper 16 are a
    // mirror and would give false positives if we checked them).
    return (acc & 0xFFFFu) != 0u;
}

/// Score the best arc for a given mask.
/// Finds the longest contiguous run in the circular 16-bit mask,
/// then sums (|pixel - center| - threshold) for each pixel in that run.
/// Matches bitmask_best_arc_score() in fast.rs.
fn arc_score(cx: i32, cy: i32, center: f32, mask: u32) -> f32 {
    let m32: u32 = mask | (mask << 16u);

    // Find the longest run in the doubled mask.
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
        if run_len > best_len {
            best_len  = run_len;
            best_start = start;
        }
    }

    // Sum the score over the best arc.
    var score: f32 = 0.0;
    for (var j: u32 = best_start; j < best_start + best_len; j++) {
        let idx = j % 16u;
        let v = load_pixel(cx + OFFSETS_X[idx], cy + OFFSETS_Y[idx]);
        let diff = abs(v - center) - params.threshold;
        score += max(diff, 0.0);
    }
    return score;
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn detect_corners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Skip the 3-pixel border (circle radius = 3) and out-of-bounds threads.
    if x < 3 || y < 3
       || x >= i32(params.img_width)  - 3
       || y >= i32(params.img_height) - 3 {
        return;
    }

    let center = load_pixel(x, y);

    // --- Quick rejection: check 4 cardinal points ---
    // Matches Rosten's high-speed test. Require min_cardinals bright or dark.
    // min_cardinals = 3 if arc_length >= 12, else 2.
    let min_card: u32 = select(2u, 3u, params.arc_length >= 12u);
    let hi = center + params.threshold;
    let lo = center - params.threshold;

    let p0  = load_pixel(x + OFFSETS_X[0],  y + OFFSETS_Y[0]);
    let p4  = load_pixel(x + OFFSETS_X[4],  y + OFFSETS_Y[4]);
    let p8  = load_pixel(x + OFFSETS_X[8],  y + OFFSETS_Y[8]);
    let p12 = load_pixel(x + OFFSETS_X[12], y + OFFSETS_Y[12]);

    let bright_card = u32(p0  > hi) + u32(p4  > hi)
                    + u32(p8  > hi) + u32(p12 > hi);
    let dark_card   = u32(p0  < lo) + u32(p4  < lo)
                    + u32(p8  < lo) + u32(p12 < lo);

    if bright_card < min_card && dark_card < min_card {
        return;
    }

    // --- Full 16-point bitmask test ---
    let masks = build_masks(x, y, center);
    let bright_mask = masks.x;
    let dark_mask   = masks.y;

    let n = params.arc_length;
    let bright_arc = has_arc(bright_mask, n);
    let dark_arc   = has_arc(dark_mask,   n);

    if !bright_arc && !dark_arc {
        return;
    }

    // --- Score (rare path — only confirmed corners reach here) ---
    var score: f32 = 0.0;
    if bright_arc {
        score = max(score, arc_score(x, y, center, bright_mask));
    }
    if dark_arc {
        score = max(score, arc_score(x, y, center, dark_mask));
    }

    // --- Claim a slot in the keypoint buffer ---
    let slot = atomicAdd(&counter.value, 1u);
    if slot >= params.max_features {
        return; // buffer full — silently drop
    }

    keypoints[slot] = GpuFeature(f32(x), f32(y), score, 0.0);
}
