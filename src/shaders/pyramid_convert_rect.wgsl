// pyramid_convert_rect.wgsl — raw u8 image → rectified R32Float level 0.
//
// Each rectified pixel samples the raw image bilinearly through its tap (the
// raw pixel at the top left and the two weights; rectify.rs `RemapTap`), with
// the arithmetic of `StereoRectification::remap` on the CPU. A tap whose
// position is 0xffffffff (no projection) gives 0.
//
// {{WG_X}} / {{WG_Y}} are substituted by pyramid.rs (same as pyramid.wgsl).

struct Tap {
    pos: u32,  // x0 | y0 << 16
    wx: f32,
    wy: f32,
}

@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> taps: array<Tap>;

fn texel(x: u32, y: u32) -> f32 {
    return f32(textureLoad(src_tex, vec2<u32>(x, y), 0).r);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn convert_rect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let t = taps[gid.y * dims.x + gid.x];
    var v = 0.0;
    if t.pos != 0xffffffffu {
        let x0 = t.pos & 0xffffu;
        let y0 = t.pos >> 16u;
        let a = texel(x0, y0);
        let b = texel(x0 + 1u, y0);
        let c = texel(x0, y0 + 1u);
        let d = texel(x0 + 1u, y0 + 1u);
        v = (1.0 - t.wy) * ((1.0 - t.wx) * a + t.wx * b) + t.wy * ((1.0 - t.wx) * c + t.wx * d);
    }
    textureStore(dst_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(v, 0.0, 0.0, 1.0));
}
