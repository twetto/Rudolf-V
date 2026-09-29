// histeq_lut.wgsl — global histogram-equalization LUT from a 256-bin histogram.
//
// One workgroup of 256 invocations (one per bin): inclusive prefix sum (CDF),
// first non-zero CDF value, then the same f32 formula as histeq.rs
// build_lut():
//     lut[i] = round((cdf[i] - cdf_min) / (total - cdf_min) * 255), clamped
// with Rust's f32::round semantics (half away from zero).
//
// Bit-exactness: the CPU divides two exact integers in f32 (IEEE, correctly
// rounded). WGSL only guarantees 2.5 ULP for `/`, which flipped a .5 rounding
// case on 2 of 2912 EuRoC frames, so the quotient is computed exactly here
// (div_rn). The multiply by 255 is correctly rounded on both sides.

struct HistEqParams {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> hist: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> lut: array<u32, 256>;
@group(0) @binding(2) var<uniform> params: HistEqParams;

var<workgroup> scan: array<u32, 256>;

// Correctly rounded (round-to-nearest-even) f32 of a / b for integers
// 0 <= a <= b < 2^24, b > 0 — the value IEEE f32 division of f32(a) / f32(b)
// produces. Long division: normalize a/b into [1, 2), take 24 mantissa bits +
// a round bit, and use the remainder as the sticky bit. rem stays < 2b < 2^25.
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
var<workgroup> cdf_min: u32;

@compute @workgroup_size(256)
fn build_lut(@builtin(local_invocation_index) i: u32) {
    scan[i] = hist[i];
    workgroupBarrier();

    // Hillis–Steele inclusive scan: 8 steps of 256.
    for (var off = 1u; off < 256u; off = off << 1u) {
        var add = 0u;
        if i >= off {
            add = scan[i - off];
        }
        workgroupBarrier();
        scan[i] += add;
        workgroupBarrier();
    }

    if i == 0u {
        var m = 0u;
        for (var k = 0u; k < 256u; k++) {
            if scan[k] > 0u {
                m = scan[k];
                break;
            }
        }
        cdf_min = m;
    }
    workgroupBarrier();

    // CPU: val = (cdf[i] - cdf_min) / (total - cdf_min) * 255 in f32; both
    // differences are exact. cdf[i] < cdf_min only before the first non-zero
    // bin, where the CPU value is negative and clamps to 0.
    var out = 0u;
    if params.total > cdf_min && scan[i] >= cdf_min {
        let val = div_rn(scan[i] - cdf_min, params.total - cdf_min) * 255.0;
        // Round half away from zero (Rust f32::round). val - floor(val) is
        // exact in f32 for |val| < 2^23.
        let r = floor(val);
        let rounded = select(r, r + 1.0, val - r >= 0.5);
        out = u32(clamp(rounded, 0.0, 255.0));
    }
    lut[i] = out;
}
