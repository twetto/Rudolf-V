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
// FAST scoring lives in fast_common.wgsl (appended by gpu/fast.rs).

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

fn load_pixel(x: i32, y: i32) -> f32 {
    let c = vec2<i32>(
        clamp(x, 0, i32(params.img_width)  - 1),
        clamp(y, 0, i32(params.img_height) - 1),
    );
    return textureLoad(input_tex, c, 0).r;
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn detect_corners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let score = corner_score(x, y);
    // Write directly to this pixel's own slot — zero contention, no atomics.
    // Non-corners are left untouched (the buffer is zero-initialized).
    if score > 0.0 {
        scores[u32(y) * params.img_width + u32(x)] = score;
    }
}
