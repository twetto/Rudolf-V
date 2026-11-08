// rv_trt_resize.cu — An input tensor made on the GPU from a u8 image, for
// rv_trt.cpp: the separable resampling and scaling src/stereo_net.rs does on
// the CPU (`area_resize`, then v·scale + offset into all three channels).
//
// Each output takes the same sums in the same order as the CPU, rounded the
// same way: per column it covers, the rows' weighted sum (the CPU's row
// pass), then the columns' weighted sum. Built with -fmad=false and written
// with __fmul_rn / __fadd_rn, so nothing is fused into an FMA: the tensor is
// the CPU's, bit for bit.
#include <cuda_runtime_api.h>
#include <cstdint>

namespace {

// Per output column (or resized row): the first image column (row), how many
// it covers, and where its weights start in the weights array.
struct Span {
    int first;
    int count;
    int offset;
};

__global__ void resize(const uint8_t* img, int stride, const Span* cols, const float* col_w,
                       const Span* rows, const float* row_w, int nw, int nh, int ch, int top,
                       float scale, float offset, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nw || r >= nh) return;
    // Rows outside the resized image's repeat its first or last (LAS2's
    // padding); Light ESS has none.
    int src = min(max(r - top, 0), ch - 1);
    Span c = cols[x];
    Span w = rows[src];
    float o = -0.0f;  // Rust's f32 sum starts at -0.0
    for (int j = 0; j < c.count; j++) {
        const uint8_t* p = img + (size_t)w.first * stride + c.first + j;
        float a = 0.0f;
        for (int k = 0; k < w.count; k++) {
            a = __fadd_rn(a, __fmul_rn(row_w[w.offset + k], (float)p[(size_t)k * stride]));
        }
        o = __fadd_rn(o, __fmul_rn(col_w[c.offset + j], a));
    }
    float v = __fadd_rn(__fmul_rn(o, scale), offset);
    size_t plane = (size_t)nw * nh;
    size_t i = (size_t)r * nw + x;
    out[i] = v;
    out[plane + i] = v;
    out[2 * plane + i] = v;
}

}  // namespace

// `img` (device, `stride` bytes a row) into the 3×nh×nw float tensor `out`
// (device), on `stream`. `cols` and `rows` hold three ints per output column
// and per resized row (first, count, offset into `col_w` / `row_w`); `ch`
// resized rows start at tensor row `top`.
cudaError_t rv_trt_resize(const uint8_t* img, int stride, const int* cols, const float* col_w,
                          const int* rows, const float* row_w, int nw, int nh, int ch, int top,
                          float scale, float offset, float* out, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((nw + block.x - 1) / block.x, (nh + block.y - 1) / block.y);
    resize<<<grid, block, 0, stream>>>(img, stride, reinterpret_cast<const Span*>(cols), col_w,
                                       reinterpret_cast<const Span*>(rows), row_w, nw, nh, ch, top,
                                       scale, offset, out);
    return cudaGetLastError();
}
