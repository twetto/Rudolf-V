// checkerboard_candidates.wgsl — the checkerboard detector's corner candidates.
//
// candidates_in (checkerboard.rs) on the GPU, from checkerboard_response.wgsl's
// output: every pixel at least RADIUS + 2 inside the border whose 5R reaches
// the threshold and is a maximum over its 5×5 window, strictly above the
// pixels before it in row-major order and at least equal to those after it, so
// that a plateau gives exactly one.
//
// Each maximum is appended through an atomic counter with its 5R and its four
// neighbours' 5R; the CPU fits the sub-pixel parabolas and sorts them as
// candidates_in does (candidates_from). Read back: the count and the maxima, a
// few KB instead of the whole response. Past `capacity` a maximum is counted
// but not stored, and the CPU finds that frame's candidates itself.
//
// WORKGROUP SIZE: {{WG_X}} × {{WG_Y}}, one invocation per pixel.

@group(0) @binding(0) var<storage, read> response: array<u32>;

@group(0) @binding(1) var<storage, read_write> found: Found;

@group(0) @binding(2) var<uniform> params: CandidateParams;

struct CandidateParams {
    width:     u32,
    height:    u32,
    threshold: i32,   // in 5R
    capacity:  u32,   // maxima the buffer holds
}

/// Each maximum: x | y << 16, then 5R pairs (centre, left), (right, up), (down, 0).
struct Found {
    count: atomic<u32>,
    items: array<vec4<u32>>,
}

const MARGIN: i32 = 7;   // RADIUS + 2: the response's border and the window's reach

/// 5R at (x, y): the low or high i16 of its pair, sign-extended.
fn r5(x: i32, y: i32) -> i32 {
    let word = response[y * ((i32(params.width) + 1) / 2) + x / 2];
    return bitcast<i32>(word << (16u - 16u * u32(x & 1))) >> 16u;
}

fn pack(lo: i32, hi: i32) -> u32 {
    return (bitcast<u32>(lo) & 0xffffu) | (bitcast<u32>(hi) << 16u);
}

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn candidates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if x < MARGIN || y < MARGIN || x >= i32(params.width) - MARGIN
        || y >= i32(params.height) - MARGIN {
        return;
    }
    let v = r5(x, y);
    if v < params.threshold {
        return;
    }
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let u = r5(x + dx, y + dy);
            let earlier = dy < 0 || (dy == 0 && dx < 0);
            if (dx != 0 || dy != 0) && (u > v || (earlier && u == v)) {
                return;
            }
        }
    }
    let slot = atomicAdd(&found.count, 1u);
    if slot < params.capacity {
        found.items[slot] = vec4<u32>(
            u32(x) | (u32(y) << 16u),
            pack(v, r5(x - 1, y)),
            pack(r5(x + 1, y), r5(x, y - 1)),
            pack(r5(x, y + 1), 0),
        );
    }
}
