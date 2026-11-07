// clahe_lut.wgsl — per-tile CLAHE LUTs from the raw R8Uint frame.
//
// The CLAHE counterpart of histeq_hist + histeq_lut: where global
// equalization needs one 256-entry LUT, CLAHE needs one per tile, and the
// convert pass blends the four around each pixel (pyramid_convert_clahe.wgsl).
//
// One workgroup per tile, 256 threads, one per histogram bin. Mirrors
// `histeq::clahe_tile_lut` + `build_lut` on the CPU exactly:
//   histogram → clip at ceil(tile_pixels/256 * clip) → redistribute the excess
//   (per_bin to all, +1 to the first `remainder` bins) → inclusive CDF →
//   LUT[i] = round((cdf[i]-cdf_min)/(tile_pixels-cdf_min)*255).
//
// `cdf_min` is the first non-zero CDF entry; the CDF is monotone, so that is
// its minimum positive entry and an atomicMin finds it without a serial scan.
//
// ROUNDING: as histeq_lut.wgsl. The CPU divides two exact integers in f32,
// correctly rounded; WGSL only guarantees 2.5 ULP for `/`, and the Orin's
// approximate division flipped .5 cases, so the quotient comes from div_rn.
// Then Rust's `f32::round` (half away from zero): WGSL `round` is half to
// even, and `floor(x + 0.5)` rounds 0.49999997 up, since x + 0.5 rounds to 1.
//
// PACKING: each LUT entry fits in a byte, so the output packs four per u32
// (entry 4k+j in byte j of word k, i.e. plain byte order on a little-endian
// host). That is a quarter of the bytes to read back each frame, and the
// readback is the u8 LUT array as is.

const BINS: u32 = 256u;

// Correctly rounded (round-to-nearest-even) f32 of a / b for integers
// 0 <= a <= b < 2^24, b > 0: what IEEE f32 division of f32(a) / f32(b) gives.
// The same function as histeq_lut.wgsl's (see there).
fn div_rn(a: u32, b: u32) -> f32 {
    if a == 0u {
        return 0.0;
    }
    if a >= b {
        return 1.0;
    }
    var rem = a;
    var e = 0i;
    while rem < b {
        rem = rem << 1u;
        e = e - 1;
    }
    var q = 0u;
    for (var i = 0u; i < 25u; i++) {
        q = q << 1u;
        if rem >= b {
            rem -= b;
            q |= 1u;
        }
        rem = rem << 1u;
    }
    let round_bit = q & 1u;
    var mant = q >> 1u;
    if round_bit == 1u && (rem != 0u || (mant & 1u) == 1u) {
        mant += 1u;
    }
    return ldexp(f32(mant), e - 23);
}

struct ClaheParams {
    img_width:  u32,
    img_height: u32,
    tile_size:  u32,
    tile_cols:  u32,
    tile_rows:  u32,
    clip_limit: f32,   // 0.0 disables clipping, as on the CPU
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0) var                      src:    texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> luts:   array<u32>;
@group(0) @binding(2) var<uniform>             params: ClaheParams;

var<workgroup> hist:      array<atomic<u32>, 256>;
var<workgroup> scan:      array<u32, 256>;
var<workgroup> excess:    atomic<u32>;
var<workgroup> cdf_min_a: atomic<u32>;

@compute @workgroup_size(256)
fn clahe_luts(@builtin(workgroup_id) wg: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile = wg.x;
    let i    = lid.x;
    if (tile >= params.tile_cols * params.tile_rows) {
        return;
    }
    let tx = tile % params.tile_cols;
    let ty = tile / params.tile_cols;

    let x0 = tx * params.tile_size;
    let y0 = ty * params.tile_size;
    let x1 = min(x0 + params.tile_size, params.img_width);
    let y1 = min(y0 + params.tile_size, params.img_height);
    let tile_pixels = (x1 - x0) * (y1 - y0);

    atomicStore(&hist[i], 0u);
    if (i == 0u) {
        atomicStore(&excess, 0u);
        atomicStore(&cdf_min_a, 0xffffffffu);
    }
    workgroupBarrier();

    // Histogram: each thread walks a strided share of the tile's pixels.
    let span = x1 - x0;
    var p    = i;
    loop {
        if (p >= tile_pixels) { break; }
        let px = x0 + (p % span);
        let py = y0 + (p / span);
        let v  = textureLoad(src, vec2<u32>(px, py), 0).r;
        atomicAdd(&hist[v], 1u);
        p = p + 256u;
    }
    workgroupBarrier();

    // Clip, accumulating the excess (CPU: clip_histogram).
    var binv = atomicLoad(&hist[i]);
    if (params.clip_limit > 0.0) {
        let clip_val = u32(ceil((f32(tile_pixels) / 256.0) * params.clip_limit));
        if (binv > clip_val) {
            atomicAdd(&excess, binv - clip_val);
            binv = clip_val;
        }
    }
    workgroupBarrier();

    // Redistribute: per_bin everywhere, +1 to the first `remainder` bins.
    if (params.clip_limit > 0.0) {
        let total_excess = atomicLoad(&excess);
        binv = binv + total_excess / BINS;
        if (i < total_excess % BINS) {
            binv = binv + 1u;
        }
    }
    scan[i] = binv;
    workgroupBarrier();

    // Inclusive prefix sum (Hillis-Steele, 8 steps over 256 bins).
    var offset: u32 = 1u;
    loop {
        if (offset >= BINS) { break; }
        var add: u32 = 0u;
        if (i >= offset) {
            add = scan[i - offset];
        }
        workgroupBarrier();
        scan[i] = scan[i] + add;
        workgroupBarrier();
        offset = offset * 2u;
    }

    let cdf_i = scan[i];
    if (cdf_i > 0u) {
        atomicMin(&cdf_min_a, cdf_i);
    }
    workgroupBarrier();

    let cdf_min = atomicLoad(&cdf_min_a);
    var out_v: u32 = 0u;
    // CPU: (cdf[i] - cdf_min) / (tile_pixels - cdf_min) * 255; both
    // differences are exact. cdf_i < cdf_min only before the first non-zero
    // bin, where the CPU value is negative and clamps to 0.
    if (cdf_min != 0xffffffffu && tile_pixels > cdf_min && cdf_i >= cdf_min) {
        let val = div_rn(cdf_i - cdf_min, tile_pixels - cdf_min) * 255.0;
        // val - floor(val) is exact in f32 for |val| < 2^23.
        let r = floor(val);
        out_v = u32(clamp(select(r, r + 1.0, val - r >= 0.5), 0.0, 255.0));
    }

    // Pack four entries per word (every thread has read its cdf_i above).
    scan[i] = out_v;
    workgroupBarrier();
    if (i < BINS / 4u) {
        let k = i * 4u;
        luts[tile * (BINS / 4u) + i] = scan[k]
            | (scan[k + 1u] << 8u)
            | (scan[k + 2u] << 16u)
            | (scan[k + 3u] << 24u);
    }
}
