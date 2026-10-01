// fast_nms_fused.wgsl — FAST detection fused with occupancy-grid NMS.
//
// One workgroup per NMS cell. Each of the 256 invocations scores a strided
// subset of the cell's pixels (fast_common.wgsl), keeping its best
// (score, in-cell row-major index); a shared-memory tree reduction then
// picks the cell winner: highest score, ties broken by the smallest
// row-major index. That is exactly what the dense path produces (FAST into a
// zeroed score buffer, then nms.wgsl scanning the cell row by row keeping the
// first strict maximum), without the full-resolution score buffer, its
// clear, or a second pass.
//
// OUTPUT: one CellWinner per cell; score == 0.0 means no corner in the cell
// (x = y = 0, as nms.wgsl wrote).


@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> winners: array<CellWinner>;
@group(0) @binding(2) var<uniform> params: FusedParams;

struct FusedParams {
    img_width:  u32,
    img_height: u32,
    threshold:  f32,
    arc_length: u32,
    cell_size:  u32,
    n_cells_x:  u32,
    n_cells_y:  u32,
    _pad:       u32,
}

struct CellWinner {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,
}

const WG: u32 = 256u;
const NO_INDEX: u32 = 0xffffffffu;

var<workgroup> sh_score: array<f32, 256>;
var<workgroup> sh_idx:   array<u32, 256>;

@compute @workgroup_size(256)
fn detect_nms(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let cell = wid.x;
    if cell >= params.n_cells_x * params.n_cells_y { return; }  // uniform per workgroup

    let x0 = (cell % params.n_cells_x) * params.cell_size;
    let y0 = (cell / params.n_cells_x) * params.cell_size;
    let cw = min(x0 + params.cell_size, params.img_width) - x0;
    let ch = min(y0 + params.cell_size, params.img_height) - y0;
    let n  = cw * ch;

    // Per-invocation best. k increases, so a strict `>` keeps the smallest
    // index among equal scores.
    var best: f32 = 0.0;
    var best_k: u32 = NO_INDEX;
    for (var k = li; k < n; k += WG) {
        let s = corner_score(i32(x0 + k % cw), i32(y0 + k / cw));
        if s > best {
            best = s;
            best_k = k;
        }
    }
    sh_score[li] = best;
    sh_idx[li]   = best_k;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
        if li < stride {
            let so = sh_score[li + stride];
            let io = sh_idx[li + stride];
            let sm = sh_score[li];
            if so > sm || (so == sm && io < sh_idx[li]) {
                sh_score[li] = so;
                sh_idx[li]   = io;
            }
        }
        workgroupBarrier();
    }

    if li == 0u {
        let s = sh_score[0];
        if s > 0.0 {
            let k = sh_idx[0];
            winners[cell] = CellWinner(f32(x0 + k % cw), f32(y0 + k / cw), s, 0.0);
        } else {
            winners[cell] = CellWinner(0.0, 0.0, 0.0, 0.0);
        }
    }
}
