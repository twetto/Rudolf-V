// pyramid_convert.wgsl — u8 source image → R32Float pyramid level 0.
//
// The persistent pyramid path uploads the raw u8 frame into an R8Uint
// texture with `queue.write_texture` (width × height bytes, no staging
// allocation, no CPU-side conversion) and this pass widens it to the
// R32Float level-0 texture that FAST and KLT read. Every pixel goes through
// a 256-entry LUT: the identity (values identical to the CPU `u8 as f32`
// conversion in `upload_f32_level`) or a histogram-equalization LUT built on
// the GPU in the same submit (histeq_lut.wgsl).
//
// {{WG_X}} / {{WG_Y}} are substituted by pyramid.rs (same as pyramid.wgsl).

@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> lut: array<u32, 256>;

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn convert_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let v = textureLoad(src_tex, vec2<u32>(gid.x, gid.y), 0).r;
    textureStore(dst_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(f32(lut[v]), 0.0, 0.0, 1.0));
}
