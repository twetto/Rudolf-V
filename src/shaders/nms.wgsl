// nms.wgsl — GPU occupancy-grid NMS.
//
// Each thread handles one cell of the occupancy grid. It scans every pixel
// in the cell's bounding box within the FAST score buffer and records the
// pixel with the highest score.
//
// OUTPUT: one CellWinner per cell. score == 0.0 means the cell had no
// FAST corners (all score values were 0.0 from the FAST shader).
//
// The CPU reads back this winners buffer (n_cells × 16 bytes, tiny) and
// collects non-zero entries as Features, filtering by the occupancy grid
// to skip cells already occupied by tracked features.
//
// No atomics — each thread owns its cell exclusively.
//
// WORKGROUP SIZE: {{WG_SIZE}} (1-D, substituted at compile time)

@group(0) @binding(0) var<storage, read>       scores:  array<f32>;
@group(0) @binding(1) var<storage, read_write>  winners: array<CellWinner>;
@group(0) @binding(2) var<uniform>              params:  NmsParams;

struct CellWinner {
    x:    f32,
    y:    f32,
    score: f32,
    _pad: f32,  // pad to 16 bytes
}

struct NmsParams {
    img_width:  u32,
    img_height: u32,
    cell_size:  u32,
    n_cells_x:  u32,
    n_cells_y:  u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@compute @workgroup_size({{WG_SIZE}}, 1, 1)
fn nms_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    if cell_idx >= params.n_cells_x * params.n_cells_y { return; }

    let cell_x = cell_idx % params.n_cells_x;
    let cell_y = cell_idx / params.n_cells_x;

    // Pixel bounds for this cell (clamped to image).
    let x0 = cell_x * params.cell_size;
    let y0 = cell_y * params.cell_size;
    let x1 = min(x0 + params.cell_size, params.img_width);
    let y1 = min(y0 + params.cell_size, params.img_height);

    var best_score: f32 = 0.0;
    var best_x:     f32 = 0.0;
    var best_y:     f32 = 0.0;

    for (var y = y0; y < y1; y = y + 1u) {
        for (var x = x0; x < x1; x = x + 1u) {
            let s = scores[y * params.img_width + x];
            if s > best_score {
                best_score = s;
                best_x     = f32(x);
                best_y     = f32(y);
            }
        }
    }

    winners[cell_idx] = CellWinner(best_x, best_y, best_score, 0.0);
}
