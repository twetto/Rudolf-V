// pyramid.wgsl — Gaussian blur + 2× downsample for pyramid construction.
//
// Each invocation computes ONE output pixel at (gid.x, gid.y) in the
// destination (half-resolution) texture. It reads a (2k+1)×(2k+1)
// neighbourhood centred on the corresponding source pixel (2*gid.x, 2*gid.y),
// applying the separable Gaussian weights.
//
// This is a FUSED blur+downsample: mathematically equivalent to
//   1. full Gaussian blur at source resolution
//   2. take pixel at (2x, 2y)
// because the 2D Gaussian is separable: G(x,y) = Gx(x)·Gy(y).
//
// Border handling: clamp-to-edge (replicate the outermost pixel).
// This matches the CPU `convolve_separable` which also clamps.
//
// WORKGROUP SIZE
// ──────────────
// {{WG_X}} and {{WG_Y}} are placeholder tokens replaced by pyramid.rs
// before the shader is compiled (naga does not yet support override
// expressions inside @workgroup_size). Typical values: 16×8 on NVIDIA/AMD,
// 8×8 on Raspberry Pi.
//
// WGSL NOTES FOR LEARNERS
// ─────────────────────────
// - `texture_2d<f32>` is the WGSL type for *reading* any float-component
//   texture (R8Unorm, R32Float, etc.).
// - `texture_storage_2d<r32float, write>` is for *writing* to a specific
//   format. The format must match the texture at creation time.
// - `array<vec4<f32>, N>` in a uniform: WGSL requires 16-byte alignment.
//   Packing 4 floats into vec4 avoids wasted padding per element.

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Binding 0 — source texture (R32Float, values in [0, 255]).
@group(0) @binding(0) var input_tex:  texture_2d<f32>;

// Binding 1 — destination texture (R32Float storage, write-only).
@group(0) @binding(1) var output_tex: texture_storage_2d<r32float, write>;

// Binding 2 — kernel parameters (uniform, 80 bytes).
@group(0) @binding(2) var<uniform> params: PyramidParams;

// ---------------------------------------------------------------------------
// Uniform layout
// Must match `PyramidParams` in pyramid.rs exactly (repr(C), same field order).
// ---------------------------------------------------------------------------

struct PyramidParams {
    dst_width:  u32,
    dst_height: u32,
    half_size:  u32,
    _pad:       u32,
    // Gaussian kernel coefficients for indices 0 … half_size (by symmetry).
    // Packed 4-per-vec4: coeffs[i/4][i%4] = kernel[i].
    // Maximum half_size = 15 (31-tap kernel, covering σ ≈ 5.0).
    coeffs: array<vec4<f32>, 4>,
}

// ---------------------------------------------------------------------------
// Kernel coefficient accessor
// ---------------------------------------------------------------------------

fn coeff(i: i32) -> f32 {
    let vi = u32(i) / 4u;
    let ei = u32(i) % 4u;
    let v = params.coeffs[vi];
    if      ei == 0u { return v.x; }
    else if ei == 1u { return v.y; }
    else if ei == 2u { return v.z; }
    else             { return v.w; }
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn blur_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dst_width || gid.y >= params.dst_height {
        return;
    }

    // Output (ox, oy) maps to source centre (2*ox, 2*oy).
    let cx: i32 = i32(gid.x) * 2;
    let cy: i32 = i32(gid.y) * 2;

    let src_dims: vec2<u32> = textureDimensions(input_tex);
    let max_x: i32 = i32(src_dims.x) - 1;
    let max_y: i32 = i32(src_dims.y) - 1;

    let half: i32 = i32(params.half_size);

    var sum: f32 = 0.0;
    for (var ky: i32 = -half; ky <= half; ky++) {
        let sy: i32 = clamp(cy + ky, 0, max_y);
        let wy: f32 = coeff(abs(ky));
        for (var kx: i32 = -half; kx <= half; kx++) {
            let sx: i32 = clamp(cx + kx, 0, max_x);
            let wx: f32 = coeff(abs(kx));
            sum += wx * wy * textureLoad(input_tex, vec2<i32>(sx, sy), 0).r;
        }
    }

    textureStore(output_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(sum, 0.0, 0.0, 1.0));
}
