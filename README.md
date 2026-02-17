# Rudolf-V

**Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend **V**ision

> **Status: Phase 1 — CPU Reference Implementation (In Progress)**
>
> Core image infrastructure, Gaussian pyramids, FAST and Harris corner detection are implemented and tested.
> KLT tracking is next.

## About

**Rudolf-V** is an experimental Rust port of the [UZH RPG vilib](https://github.com/uzh-rpg/vilib) (CUDA Visual Library).

While the original `vilib` is a state-of-the-art, CUDA-only library designed strictly for NVIDIA platforms, **Rudolf-V** aims to democratize high-speed visual tracking. By leveraging **Rust** and **wgpu**, this library targets cross-platform compatibility, bringing GPU-accelerated visual frontends to hardware that CUDA leaves behind:

* **Raspberry Pi 4 & 5** (via Vulkan)
* AMD & Intel integrated graphics
* Apple Silicon
* NVIDIA GPUs

## Quick Start

```bash
# Run all tests
cargo test

# Visualize FAST and Harris corner detection
cargo run --example visualize_fast
# -> outputs SVG files in vis_output/:
#     {scene}_fast_raw.svg     — raw FAST detections
#     {scene}_fast_nms.svg     — after grid-based NMS
#     {scene}_compare.svg      — side-by-side comparison
#     pyramid.svg              — Gaussian pyramid levels
#     multilevel_fast.svg      — FAST at each pyramid level
#     harris_chessboard.svg    — Harris raw vs NMS on chessboard
#     harris_response.svg      — Harris response heatmap
#     fast_vs_harris.svg       — FAST vs Harris on same image

# Run benchmarks
cargo bench
```

## Module Overview

| Module | Description | vilib Equivalent |
|---|---|---|
| `image` | `Image<T>`, `ImageView<'a, T>`, `Pixel` trait, bilinear interpolation | — |
| `convert` | Pixel type conversions (u8 <-> f32, RGB -> grayscale) | — |
| `convolution` | Separable 1D convolution (horizontal + vertical) | `conv_filter_row.cu`, `conv_filter_col.cu` |
| `pyramid` | Gaussian image pyramid (blur + 2x downsample) | `pyramid_gpu.cu` |
| `fast` | FAST-N corner detector (N=9,10,11,12) with Rosten's high-speed test | `fast_gpu_cuda_tools.cu` |
| `nms` | Grid-based non-maximum suppression | `detector_base_gpu_cuda_tools.cu` |
| `gradient` | Sobel gradient computation (separable) | `harris_gpu_cuda_tools.cu` |
| `harris` | Harris corner detector (structure tensor) | `harris_gpu_cuda_tools.cu` |

## Engineering Roadmap

We are adopting a verification-first development strategy. The project is divided into three distinct phases:

### Phase 1: Pure Rust CPU Reference (Current Focus)

**Goal:** Implement the core VIO frontend algorithms in safe, single-threaded Rust.

**Purpose:** Since debugging GPU shaders is difficult, this CPU implementation will serve as the **Test Oracle** (Ground Truth). We will not write a single line of shader code until the math is verified here.

* [x] `Image<T>` Dynamic Container
* [x] Gaussian Pyramid Generation
* [x] FAST Corner Detection
* [x] Harris Corner Response
* [ ] KLT (Lucas-Kanade) Feature Tracker

### Phase 2: `wgpu` Acceleration

**Goal:** Port the verified algorithms to WGSL compute shaders.

**Purpose:** Unlock real-time performance on embedded devices.

* [ ] GPU Device/Queue Abstraction
* [ ] WGSL Compute Kernels (verified against Phase 1 CPU output)

### Phase 3: Zero-Copy Optimization

**Goal:** Optimize memory throughput.

**Purpose:** Minimize CPU-GPU bandwidth usage for high-frequency odometry.

## Scientific Basis

This project is a derivative implementation of the algorithms presented in the IROS 2020 paper:

> **Faster than FAST: GPU-Accelerated Frontend for High-Speed VIO**
> *Balazs Nagy, Philipp Foehn, and Davide Scaramuzza*
> Robotics and Perception Group, University of Zurich

If you use concepts from this library in an academic context, please cite the original authors:

```bibtex
@inproceedings{Nagy2020,
  author = {Nagy, Balazs and Foehn, Philipp and Scaramuzza, Davide},
  title = {{Faster than FAST}: {GPU}-Accelerated Frontend for High-Speed {VIO}},
  booktitle = {IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS)},
  year = {2020},
  doi = {10.1109/IROS45743.2020.9340851}
}
```

