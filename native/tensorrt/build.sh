#!/usr/bin/env bash
# Build librv_trt.so (rv_trt.cpp, rv_trt_resize.cu) on the Jetson, against its
# TensorRT and CUDA:
#   native/tensorrt/build.sh [OUT_DIR]        (default: this directory)
# src/tensorrt.rs loads it at run time; pass its path to TrtEngine::load.
# CUDA_ARCH: the GPU's compute capability (default 87, Orin).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
arch="${CUDA_ARCH:-87}"
mkdir -p "$out"
obj="$(mktemp -d)"
trap 'rm -rf "$obj"' EXIT
# -fmad=false: the resampling rounds as the CPU's (no fused multiply-adds).
"$cuda/bin/nvcc" -std=c++17 -O2 -Xcompiler -fPIC -fmad=false \
    -gencode "arch=compute_$arch,code=sm_$arch" \
    -c "$here/rv_trt_resize.cu" -o "$obj/rv_trt_resize.o"
g++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra \
    -I"$cuda/include" "$here/rv_trt.cpp" "$obj/rv_trt_resize.o" -o "$out/librv_trt.so" \
    -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lnvinfer -lnvinfer_plugin -lcudart -ldl
echo "built $out/librv_trt.so"
