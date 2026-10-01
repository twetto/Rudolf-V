// pyramid_l0l1.wgsl — pyramid level 0 (u8 → f32 through the LUT) and level 1
// (5×5 binomial blur + 2× downsample) in one dispatch.
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
@group(0) @binding(3) var<storage, read> lut: array<u32, 256>;

const W5 = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

fn value(x: i32, y: i32) -> f32 {
    return f32(lut[textureLoad(src_tex, vec2<i32>(x, y), 0).r]);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn l0_l1(@builtin(global_invocation_id) gid: vec3<u32>) {
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
