// checkerboard_response.wgsl — the checkerboard detector's ChESS response.
//
// The CPU's response (checkerboard.rs, response_row_body) in i32, so the same
// integers:
//   5R = 5·(sum − diff) − |5·(ring sum) − 16·(the centre and its 4 neighbours)|
// over the 16-sample ring of radius 5, and 0 within RADIUS of the border.
//
// OUTPUT: THE CPU DETECTOR'S i16, TWO TO A u32
// ────────────────────────────────────────────
// 5R fits i16 (−30600..10200), as the CPU stores it. Each row is stored as
// pairs, pixel 2p in the low half of u32 p and 2p + 1 in the high half (0 past
// an odd width): read back, a row's bytes are its i16 values in order, the
// detector's response row; for an even width the whole buffer is its image.
//
// SHARED-MEMORY TILE
// ──────────────────
// A workgroup of {{WG_X}}×{{WG_Y}} computes 2·{{WG_X}}×{{WG_Y}} pixels (a pair
// each) from a tile of those pixels and a RADIUS border, loaded once: each
// texel is read about twice instead of 21 times (on an Orin Nano the per-pixel
// texture loads made the response slower than the CPU's). The tile is loaded a
// 2×2 block per textureGather, four texels per fetch; the texture's clamping
// past the edge only reaches the border, which has no response.
//
// HALF SIZE: params.half = 1
// ──────────────────────────
// The response of the half-size image instead, each of its pixels the rounded
// mean of a 2×2 block, (sum + 2) / 4 as half_size_into, taken from the
// full-size texture as the tile is loaded (one textureGather per half-size
// pixel): no second upload.

@group(0) @binding(0) var input_tex: texture_2d<u32>;   // R8Uint: the frame's bytes

@group(0) @binding(1) var<storage, read_write> response: array<u32>;

@group(0) @binding(2) var<uniform> params: ResponseParams;

@group(0) @binding(3) var nearest: sampler;   // non-filtering, clamp to edge

struct ResponseParams {
    width:  u32,   // the image the response is of (the half-size one when half = 1)
    height: u32,
    half:   u32,
    _pad:   u32,
}

const RADIUS: i32 = 5;
const TILE_W: i32 = {{TILE_W}};   // 2·WG_X + 2·RADIUS

// TILE_W × (WG_Y + 2·RADIUS, rounded up to even: whole 2×2 blocks)
var<workgroup> tile: array<i32, {{TILE_LEN}}>;

/// The full-size texels (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1), in
/// gather order: (x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y).
fn block(x: i32, y: i32) -> vec4<i32> {
    let centre = vec2<f32>(f32(x + 1), f32(y + 1)) / vec2<f32>(textureDimensions(input_tex));
    return vec4<i32>(textureGather(0, input_tex, nearest, centre));
}

/// The tile's pixel at (x, y) in tile coordinates.
fn at(x: i32, y: i32) -> i32 {
    return tile[y * TILE_W + x];
}

/// 5R at image pixel (x, y), tile pixel (tx, ty).
fn chess(x: i32, y: i32, tx: i32, ty: i32) -> i32 {
    if x < RADIUS || y < RADIUS || x >= i32(params.width) - RADIUS
        || y >= i32(params.height) - RADIUS {
        return 0;
    }
    // The ring in four groups a quarter turn apart, as the CPU sums them: a
    // sample o, its quarter turn (−o.y, o.x), and both opposites.
    var first = array<vec2<i32>, 4>(
        vec2<i32>(5, 0), vec2<i32>(5, 2), vec2<i32>(4, 4), vec2<i32>(2, 5),
    );
    var sum = 0;
    var diff = 0;
    var ring_sum = 0;
    for (var k = 0; k < 4; k++) {
        let o = first[k];
        let a = at(tx + o.x, ty + o.y);
        let b = at(tx - o.y, ty + o.x);
        let c = at(tx - o.x, ty - o.y);
        let d = at(tx + o.y, ty - o.x);
        sum += abs((a + c) - (b + d));
        diff += abs(a - c) + abs(b - d);
        ring_sum += a + b + c + d;
    }
    let centre = at(tx, ty) + at(tx - 1, ty) + at(tx + 1, ty)
               + at(tx, ty - 1) + at(tx, ty + 1);
    return 5 * (sum - diff) - abs(5 * ring_sum - 16 * centre);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn chess_response(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let w = i32(params.width);
    let h = i32(params.height);
    // The tile: this workgroup's pixels and a RADIUS border, clamped into the
    // image (border pixels have no response, so the clamped values are unused).
    let x0 = i32(wg.x) * 2 * {{WG_X}} - RADIUS;
    let y0 = i32(wg.y) * {{WG_Y}} - RADIUS;
    if params.half == 0u {
        // 2×2 blocks of the tile (its sides are even), a gather each.
        let bw = TILE_W / 2;
        for (var i = i32(li); i < {{TILE_LEN}} / 4; i += {{WG_X}} * {{WG_Y}}) {
            let tx = 2 * (i % bw);
            let ty = 2 * (i / bw);
            let b = block(x0 + tx, y0 + ty);
            tile[ty * TILE_W + tx] = b.w;
            tile[ty * TILE_W + tx + 1] = b.z;
            tile[(ty + 1) * TILE_W + tx] = b.x;
            tile[(ty + 1) * TILE_W + tx + 1] = b.y;
        }
    } else {
        // A half-size pixel per gather: its full-size block's rounded mean.
        for (var i = i32(li); i < {{TILE_LEN}}; i += {{WG_X}} * {{WG_Y}}) {
            let x = clamp(x0 + i % TILE_W, 0, w - 1);
            let y = clamp(y0 + i / TILE_W, 0, h - 1);
            let b = block(2 * x, 2 * y);
            tile[i] = (b.x + b.y + b.z + b.w + 2) / 4;
        }
    }
    workgroupBarrier();

    let pairs = (w + 1) / 2;
    let p = i32(wg.x) * {{WG_X}} + i32(lid.x);
    let y = i32(wg.y) * {{WG_Y}} + i32(lid.y);
    if p >= pairs || y >= h {
        return;
    }
    let x = 2 * p;
    let tx = 2 * i32(lid.x) + RADIUS;
    let ty = i32(lid.y) + RADIUS;
    var high = 0;
    if x + 1 < w {
        high = chess(x + 1, y, tx + 1, ty);
    }
    response[y * pairs + p] = (bitcast<u32>(chess(x, y, tx, ty)) & 0xffffu)
                            | (bitcast<u32>(high) << 16u);
}
