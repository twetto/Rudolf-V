// histeq_hist.wgsl — global 256-bin histogram of the raw u8 frame.
//
// Each workgroup accumulates into a shared-memory histogram (workgroup
// atomics), then adds its non-zero bins into the storage histogram. The
// storage histogram must be zeroed before the pass (clear_buffer). Integer
// sums are order-independent, so the result is deterministic and identical
// to the CPU histogram in histeq.rs.

@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 256>;

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    atomicStore(&local_hist[li], 0u);
    workgroupBarrier();

    let dims = textureDimensions(src_tex);
    let total = dims.x * dims.y;
    let stride = nwg.x * 256u;
    // Row-major grid-stride loop: neighbouring invocations read neighbouring
    // pixels.
    for (var i = wid.x * 256u + li; i < total; i += stride) {
        let v = textureLoad(src_tex, vec2<u32>(i % dims.x, i / dims.x), 0).r;
        atomicAdd(&local_hist[v], 1u);
    }
    workgroupBarrier();

    let c = atomicLoad(&local_hist[li]);
    if c > 0u {
        atomicAdd(&hist[li], c);
    }
}
