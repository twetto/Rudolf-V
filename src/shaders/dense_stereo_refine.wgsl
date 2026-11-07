// dense_stereo_refine.wgsl — the variational refinement, as dense_stereo.rs
// (`refine_variational`, `RedBlack::solve`; the CPU explains the energy):
//   warp     the right image warped by the current D: the mean and the
//            difference of the two images
//   diffs    their central differences
//   terms    each solved pixel's fixed data terms
//   weights, data, sor   one colour of a fixed-point iteration's passes
//   update   D and its variance from the increments
//
// The solver's arrays are split by colour as the CPU's `RedBlack`: colour c
// holds the pixels with (x + y) % 2 == c, pixel x of row y at column x / 2 of
// that colour's row y, each colour an (h + 2) × (hw + 2) grid with a zero
// border. A colour's pass reads only the other colour's values, a thread per
// pixel of the colour. The normalisations 1 / (g² + ζ²) the CPU stores are
// recomputed here (fewer arrays to move).
// {{ZETA2}}, {{EPS2}}: dense_stereo.rs's constants.

struct RefineParams {
    w: u32,
    h: u32,
    hw: u32,          // pixels of one colour per row (columns of its grid)
    stride: u32,      // hw + 2
    size: u32,        // one colour's grid, (h + 2) · stride
    colour: u32,      // the colour this pass updates
    masked: u32,      // only pixels with valid depth are solved
    _pad0: u32,
    alpha2: f32,      // smoothness weight / 2
    gamma2: f32,      // gradient-constancy weight / 2
    delta2: f32,      // colour-constancy weight / 2
    omega: f32,       // SOR relaxation
    inv_edge: f32,    // 1 / edge_sigma² (0: no edge-aware smoothness)
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

// A solved pixel's fixed terms: the image's derivative along the epipolar
// direction (id) and its derivatives (ixd, iyd); the warped difference (iz)
// and its derivatives (ixz, iyz); the smoothness weight's edge factor; and
// flags: 1 solved, 2 its right neighbour solved too, 4 its lower one.
struct Terms {
    id: f32,
    ixd: f32,
    iyd: f32,
    iz: f32,
    ixz: f32,
    iyz: f32,
    edge: f32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> R: RefineParams;
// Per pixel (row-major): left grey level with gx; the right level; the dense
// D and its validity; the right image's shift per unit of D.
@group(0) @binding(1) var<storage, read_write> tpl: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> right_img: array<f32>;
@group(0) @binding(3) var<storage, read_write> dense_d: array<f32>;
@group(0) @binding(4) var<storage, read_write> dense_valid: array<u32>;
@group(0) @binding(5) var<storage, read_write> pixel_n: array<f32>;
// Per pixel: (mean image, difference image), and their ∂x, ∂y.
@group(0) @binding(6) var<storage, read_write> ai: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> dif: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> grad: array<vec2<f32>>;
// Split by colour: the terms; (D, dD); (wr, wd), the weights of each
// pixel's right and lower edges; (1/A, c) of each pixel's equation
// A·dD = c + Σ w·dD_neighbour.
@group(0) @binding(9) var<storage, read_write> terms: array<Terms>;
@group(0) @binding(10) var<storage, read_write> du: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> wts: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read_write> ac: array<vec2<f32>>;
@group(0) @binding(13) var<storage, read_write> dense_var: array<f32>;

const ZETA2: f32 = {{ZETA2}};
const EPS2: f32 = {{EPS2}};

fn solved(i: u32) -> bool {
    return R.masked == 0u || dense_valid[i] != 0u;
}

// Index of pixel (x, y) in the split arrays (`RedBlack::index`).
fn rb_index(x: u32, y: u32) -> u32 {
    return (x + y) % 2u * R.size + (y + 1u) * R.stride + x / 2u + 1u;
}

// The right image with its border repeated, sampled bilinearly as the CPU's
// `Padded::sample` (clamped into its padding).
fn right_at(x: i32, y: i32) -> f32 {
    let w = i32(R.w);
    let h = i32(R.h);
    return right_img[u32(clamp(y, 0, h - 1)) * R.w + u32(clamp(x, 0, w - 1))];
}

fn padded_sample(x_in: f32, y_in: f32) -> f32 {
    let b = f32(BORDER);
    let xs = clamp(x_in, -b + 1.0, f32(R.w) + b - 2.0) + b;
    let ys = clamp(y_in, -b + 1.0, f32(R.h) + b - 2.0) + b;
    let x0 = floor(xs);
    let y0 = floor(ys);
    let fx = xs - x0;
    let fy = ys - y0;
    let xx = i32(x0) - BORDER;
    let yy = i32(y0) - BORDER;
    return (1.0 - fy) * ((1.0 - fx) * right_at(xx, yy) + fx * right_at(xx + 1, yy))
        + fy * ((1.0 - fx) * right_at(xx, yy + 1) + fx * right_at(xx + 1, yy + 1));
}

// Linearise the warp at the current D: the mean and difference of the left
// and warped right images (an unsolved pixel, or one whose ray misses the
// right image, sees its own grey).
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= R.w || y >= R.h {
        return;
    }
    let i = y * R.w + x;
    let i0 = tpl[i].x;
    var v = vec2<f32>(i0, 0.0);
    let n = pixel_n[i];
    if solved(i) && n != 0.0 {
        let warped = padded_sample(f32(x) + dense_d[i] * n, f32(y));
        v = vec2<f32>(0.5 * (i0 + warped), warped - i0);
    }
    ai[i] = v;
}

// Central differences, the border replicated (diff_x, diff_y).
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn diffs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= R.w || y >= R.h {
        return;
    }
    let xl = select(x - 1u, 0u, x == 0u);
    let xr = min(x + 1u, R.w - 1u);
    let yu = select(y - 1u, 0u, y == 0u);
    let yd = min(y + 1u, R.h - 1u);
    let dx = 0.5 * (ai[y * R.w + xr] - ai[y * R.w + xl]);
    let dy = 0.5 * (ai[yd * R.w + x] - ai[yu * R.w + x]);
    dif[y * R.w + x] = vec4<f32>(dx.x, dy.x, dx.y, dy.y);
}

// Each pixel's D into the split arrays, its increment 0, and a solved
// pixel's terms (an unsolved one's are 0: it stays put and constrains
// nothing).
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn solver_terms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= R.w || y >= R.h {
        return;
    }
    let i = y * R.w + x;
    let r = rb_index(x, y);
    du[r] = vec2<f32>(dense_d[i], 0.0);
    if !solved(i) {
        terms[r] = Terms(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0u);
        return;
    }
    // The epipolar direction (ex, 0) of a rectified pair; none where the ray
    // misses the right image.
    let ex = pixel_n[i];
    let ey = 0.0;
    let xm = select(x - 1u, 0u, x == 0u);
    let xp = min(x + 1u, R.w - 1u);
    let ym = select(y - 1u, 0u, y == 0u);
    let yp = min(y + 1u, R.h - 1u);
    let g = dif[i];
    let ixx = 0.5 * (dif[y * R.w + xp].x - dif[y * R.w + xm].x);
    let ixy = 0.5 * (dif[yp * R.w + x].x - dif[ym * R.w + x].x);
    let iyy = 0.5 * (dif[yp * R.w + x].y - dif[ym * R.w + x].y);
    let gr = grad[i];
    var flags = 1u;
    if x + 1u < R.w && solved(i + 1u) {
        flags |= 2u;
    }
    if y + 1u < R.h && solved(i + R.w) {
        flags |= 4u;
    }
    terms[r] = Terms(
        g.x * ex + g.y * ey,
        ixx * ex + ixy * ey,
        ixy * ex + iyy * ey,
        ai[i].y,
        g.z,
        g.w,
        1.0 / (1.0 + (gr.x * gr.x + gr.y * gr.y) * R.inv_edge),
        flags,
    );
}

struct Cell {
    me: u32,      // this pixel in its colour's grid
    ot: u32,      // the same row and column in the other colour's
    s: u32,       // its right neighbour is the other colour's column q + s
    ok: bool,
}

fn cell(gid: vec3<u32>) -> Cell {
    let q = gid.x;
    let y = gid.y;
    let row = (y + 1u) * R.stride + 1u + q;
    return Cell(R.colour * R.size + row, (1u - R.colour) * R.size + row, (y + R.colour) % 2u,
                q < R.hw && y < R.h);
}

// Smoothness weights from the forward differences of D + dD; each pixel's
// weight belongs to its right and lower edges, and only edges between two
// solved pixels have one.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell(gid);
    if !c.ok {
        return;
    }
    let t = terms[c.me];
    let right = select(0.0, 1.0, (t.flags & 2u) != 0u);
    let down = select(0.0, 1.0, (t.flags & 4u) != 0u);
    let own = du[c.me];
    let u = own.x + own.y;
    let rn = du[c.ot + c.s];
    let bn = du[c.ot + R.stride];
    let ux = (rn.x + rn.y - u) * right;
    let uy = (bn.x + bn.y - u) * down;
    let ws = R.alpha2 / sqrt(ux * ux + uy * uy + EPS2) * t.edge;
    wts[c.me] = vec2<f32>(ws * right, ws * down);
}

// Data terms linearised around D + dD, and the parts of each pixel's
// equation that stay fixed during SOR.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn data(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell(gid);
    if !c.ok {
        return;
    }
    let t = terms[c.me];
    let own = du[c.me];
    let ddq = own.y;
    let inv_nd = 1.0 / (t.id * t.id + ZETA2);
    let inv_nx = 1.0 / (t.ixd * t.ixd + ZETA2);
    let inv_ny = 1.0 / (t.iyd * t.iyd + ZETA2);
    let z = t.iz + t.id * ddq;
    let wc = R.delta2 * inv_nd / sqrt(z * z * inv_nd + EPS2);
    var a = wc * t.id * t.id + ZETA2;
    var rhs = -wc * t.iz * t.id;
    let zx = t.ixz + t.ixd * ddq;
    let zy = t.iyz + t.iyd * ddq;
    let wg = R.gamma2 / sqrt(zx * zx * inv_nx + zy * zy * inv_ny + EPS2);
    a += wg * (t.ixd * t.ixd * inv_nx + t.iyd * t.iyd * inv_ny);
    rhs -= wg * (t.ixd * t.ixz * inv_nx + t.iyd * t.iyz * inv_ny);
    let d = own.x;
    let wm = wts[c.me];
    let wl = wts[c.ot + c.s - 1u].x;
    let wu = wts[c.ot - R.stride].y;
    a += wm.x + wl + wm.y + wu;
    rhs += wm.x * (du[c.ot + c.s].x - d) + wl * (du[c.ot + c.s - 1u].x - d)
        + wm.y * (du[c.ot + R.stride].x - d) + wu * (du[c.ot - R.stride].x - d);
    let solved_here = select(0.0, 1.0, (t.flags & 1u) != 0u);
    ac[c.me] = vec2<f32>(solved_here * (1.0 / a), solved_here * rhs);
}

// One colour's SOR update; it reads only the other colour's increments.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn sor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell(gid);
    if !c.ok {
        return;
    }
    let e = ac[c.me];
    let wm = wts[c.me];
    let wl = wts[c.ot + c.s - 1u].x;
    let wu = wts[c.ot - R.stride].y;
    let sum = e.y + wm.x * du[c.ot + c.s].y + wl * du[c.ot + c.s - 1u].y
        + wm.y * du[c.ot + R.stride].y + wu * du[c.ot - R.stride].y;
    let dd = du[c.me].y;
    du[c.me].y = dd + R.omega * (sum * e.x - dd);
}

// The increment onto D; the refinement's own change counts towards the
// variance.
@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= R.w || y >= R.h {
        return;
    }
    let i = y * R.w + x;
    if solved(i) {
        let change = du[rb_index(x, y)].y;
        dense_d[i] += change;
        dense_var[i] += change * change;
    }
}
