#!/usr/bin/env bash
# Build librv_trt.so (rv_trt.cpp) on the Jetson, against its TensorRT and CUDA:
#   native/tensorrt/build.sh [OUT_DIR]        (default: this directory)
# src/tensorrt.rs loads it at run time; pass its path to TrtEngine::load.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p "$out"
g++ -std=c++17 -O2 -shared -fPIC -Wall -Wextra \
    -I"$cuda/include" "$here/rv_trt.cpp" -o "$out/librv_trt.so" \
    -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lnvinfer -lnvinfer_plugin -lcudart -ldl
echo "built $out/librv_trt.so"
