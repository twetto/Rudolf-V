// pyramid_convert_clahe.wgsl — raw u8 → R32Float level 0, through CLAHE.
//
// The CLAHE form of pyramid_convert.wgsl. Global equalization can be folded
// into that pass as a single 256-entry LUT lookup; CLAHE cannot, because the
// mapping varies across the image. This pass bilinearly blends the four tile
// LUTs around each pixel (clahe_lut.wgsl built them), so the equalized image
// is still never materialized — it goes straight into level 0.
//
// Matches the remap loop of `histeq::equalize_clahe_into`: tile centres sit at
// (t + 0.5) * tile_size, so the weight is clamp((p - centre0) / tile_size, 0, 1),
// and the blended value is rounded half-away-from-zero (floor(x + 0.5)) to the
// u8 the CPU would have produced, then widened to f32 exactly as the identity
// LUT path does.
//
// {{WG_X}} / {{WG_Y}} are substituted by pyramid.rs (same as pyramid.wgsl).

const BINS: u32 = 256u;

struct ClaheParams {
    img_width:  u32,
    img_height: u32,
    tile_size:  u32,
    tile_cols:  u32,
    tile_rows:  u32,
    clip_limit: f32,
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> luts: array<u32>;
@group(0) @binding(3) var<uniform> params: ClaheParams;

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn convert_u8_clahe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let v = textureLoad(src_tex, vec2<u32>(gid.x, gid.y), 0).r;
    let ts = f32(params.tile_size);

    let fy  = f32(gid.y) / ts - 0.5;
    let ty0 = u32(max(floor(fy), 0.0));
    let ty1 = min(ty0 + 1u, params.tile_rows - 1u);
    var ay  = 0.0;
    if (ty0 != ty1) {
        ay = clamp((f32(gid.y) - (f32(ty0) + 0.5) * ts) / ts, 0.0, 1.0);
    }

    let fx  = f32(gid.x) / ts - 0.5;
    let tx0 = u32(max(floor(fx), 0.0));
    let tx1 = min(tx0 + 1u, params.tile_cols - 1u);
    var ax  = 0.0;
    if (tx0 != tx1) {
        ax = clamp((f32(gid.x) - (f32(tx0) + 0.5) * ts) / ts, 0.0, 1.0);
    }

    let row0 = ty0 * params.tile_cols;
    let row1 = ty1 * params.tile_cols;
    let v00 = f32(luts[(row0 + tx0) * BINS + v]);
    let v10 = f32(luts[(row0 + tx1) * BINS + v]);
    let v01 = f32(luts[(row1 + tx0) * BINS + v]);
    let v11 = f32(luts[(row1 + tx1) * BINS + v]);

    let top = v00 + ax * (v10 - v00);
    let bot = v01 + ax * (v11 - v01);
    let out = clamp(floor(top * (1.0 - ay) + bot * ay + 0.5), 0.0, 255.0);

    textureStore(dst_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(out, 0.0, 0.0, 1.0));
}
