// rectify.wgsl — raw u8 image → rectified u8 image, four pixels per thread.
//
// Each rectified pixel samples the raw image bilinearly through its tap (the
// raw pixel at the top left and the two weights; rectify.rs `RemapTap`), with
// the arithmetic of `StereoRectification::remap` on the CPU, then rounds half
// away from zero as f32::round does (WGSL's round() goes to even). A tap
// whose position is 0xffffffff (no projection) gives 0.
//
// The output packs pixels row-major, four to a u32, pixel i in byte i % 4:
// its bytes are the image's.

struct Tap {
    pos: u32,  // x0 | y0 << 16
    wx: f32,
    wy: f32,
}

@group(0) @binding(0) var raw: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> taps: array<Tap>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn texel(x: u32, y: u32) -> f32 {
    return f32(textureLoad(raw, vec2<u32>(x, y), 0).r);
}

fn sample(t: Tap) -> u32 {
    if t.pos == 0xffffffffu {
        return 0u;
    }
    let x0 = t.pos & 0xffffu;
    let y0 = t.pos >> 16u;
    let a = texel(x0, y0);
    let b = texel(x0 + 1u, y0);
    let c = texel(x0, y0 + 1u);
    let d = texel(x0 + 1u, y0 + 1u);
    let v = (1.0 - t.wy) * ((1.0 - t.wx) * a + t.wx * b) + t.wy * ((1.0 - t.wx) * c + t.wx * d);
    let r = floor(v);
    return u32(clamp(select(r, r + 1.0, v - r >= 0.5), 0.0, 255.0));
}

@compute @workgroup_size(64)
fn rectify(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let word = gid.y * groups.x * 64u + gid.x;
    if word >= arrayLength(&out) {
        return;
    }
    let dims = textureDimensions(raw);
    let n = dims.x * dims.y;
    var packed = 0u;
    for (var k = 0u; k < 4u; k++) {
        let i = word * 4u + k;
        if i < n {
            packed |= sample(taps[i]) << (8u * k);
        }
    }
    out[word] = packed;
}
