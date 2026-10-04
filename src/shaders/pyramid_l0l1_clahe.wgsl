// pyramid_l0l1_clahe.wgsl — the CLAHE form of pyramid_l0l1.wgsl.
//
// Identical to it except for `value()`: instead of one global LUT lookup, the
// pixel goes through the four tile LUTs around it, bilinearly blended
// (clahe_lut.wgsl built them). Level 1 blurs the *equalized* values, which is
// what the CPU does — it equalizes the whole frame and then builds the
// pyramid — and the border clamp picks the equalized value at the clamped
// coordinate, again as on the CPU.
//
// Keep this in step with pyramid_l0l1.wgsl; only `value()` and the extra
// params binding differ.
//
// One invocation per level-1 pixel (ox, oy), dispatched over
// ceil(w0/2) × ceil(h0/2) so every level-0 pixel is covered:
//   - writes level 0 at (2ox + {0,1}, 2oy + {0,1})  (in bounds),
//   - computes level 1 at (ox, oy) from the raw texels directly.
// Level 1 therefore does not wait for level 0 to be written and flushed, and
// reads the 1-byte raw texture instead of the 4-byte level-0 texture.
//
// Bit-identical to pyramid_convert.wgsl followed by pyramid.wgsl: the level-0
// values are f32(lut[raw]) in both, and level 1 sums the same terms in the same
// order with the same (wx * wy) * v expression.
//
// {{WG_X}} / {{WG_Y}} are substituted by pyramid.rs.

@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var level0_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var level1_tex: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<storage, read> luts: array<u32>;
@group(0) @binding(4) var<uniform> params: ClaheParams;

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

const W5 = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

fn value(x: i32, y: i32) -> f32 {
    let v  = textureLoad(src_tex, vec2<i32>(x, y), 0).r;
    let ts = f32(params.tile_size);

    let fy  = f32(y) / ts - 0.5;
    let ty0 = u32(max(floor(fy), 0.0));
    let ty1 = min(ty0 + 1u, params.tile_rows - 1u);
    var ay  = 0.0;
    if (ty0 != ty1) {
        ay = clamp((f32(y) - (f32(ty0) + 0.5) * ts) / ts, 0.0, 1.0);
    }

    let fx  = f32(x) / ts - 0.5;
    let tx0 = u32(max(floor(fx), 0.0));
    let tx1 = min(tx0 + 1u, params.tile_cols - 1u);
    var ax  = 0.0;
    if (tx0 != tx1) {
        ax = clamp((f32(x) - (f32(tx0) + 0.5) * ts) / ts, 0.0, 1.0);
    }

    let row0 = ty0 * params.tile_cols;
    let row1 = ty1 * params.tile_cols;
    let v00 = f32(luts[(row0 + tx0) * BINS + v]);
    let v10 = f32(luts[(row0 + tx1) * BINS + v]);
    let v01 = f32(luts[(row1 + tx0) * BINS + v]);
    let v11 = f32(luts[(row1 + tx1) * BINS + v]);

    let top = v00 + ax * (v10 - v00);
    let bot = v01 + ax * (v11 - v01);
    // Round to the u8 the CPU would have produced, then widen as usual.
    return clamp(floor(top * (1.0 - ay) + bot * ay + 0.5), 0.0, 255.0);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn l0_l1_clahe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d0 = vec2<i32>(textureDimensions(src_tex));
    let d1 = vec2<i32>(textureDimensions(level1_tex));
    let o = vec2<i32>(gid.xy);
    let c = o * 2;
    if c.x >= d0.x || c.y >= d0.y {
        return;
    }

    // Level 0: this invocation's 2×2 block.
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let p = c + vec2<i32>(dx, dy);
            if p.x < d0.x && p.y < d0.y {
                textureStore(level0_tex, p, vec4<f32>(value(p.x, p.y), 0.0, 0.0, 1.0));
            }
        }
    }

    // Level 1 (floor(w0/2) × floor(h0/2)).
    if o.x >= d1.x || o.y >= d1.y {
        return;
    }
    let max_x = d0.x - 1;
    let max_y = d0.y - 1;
    var sum: f32 = 0.0;
    for (var ky: i32 = -2; ky <= 2; ky++) {
        let sy = clamp(c.y + ky, 0, max_y);
        let wy = W5[ky + 2];
        for (var kx: i32 = -2; kx <= 2; kx++) {
            let sx = clamp(c.x + kx, 0, max_x);
            let wx = W5[kx + 2];
            sum += wx * wy * value(sx, sy);
        }
    }
    textureStore(level1_tex, o, vec4<f32>(sum, 0.0, 0.0, 1.0));
}
