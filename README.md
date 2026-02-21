# Rudolf-V

**Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend **V**ision

> **Status: Phase 2 — GPU Acceleration (Complete)**
>
> The full GPU frontend pipeline is implemented and validated against the CPU reference.
> Pyramid construction, FAST detection, KLT tracking, and NMS all run as WGSL compute
> shaders via wgpu. `GpuFrontend::process()` is API-compatible with the CPU `Frontend`.
> Ready for Phase 3: async readback and double-buffering.

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
#     {scene}_fast_raw.svg     -- raw FAST detections
#     {scene}_fast_nms.svg     -- after grid-based NMS
#     {scene}_compare.svg      -- side-by-side comparison
#     pyramid.svg              -- Gaussian pyramid levels
#     multilevel_fast.svg      -- FAST at each pyramid level
#     harris_chessboard.svg    -- Harris raw vs NMS on chessboard
#     harris_response.svg      -- Harris response heatmap
#     fast_vs_harris.svg       -- FAST vs Harris on same image
#     klt_flow_1to2.svg        -- KLT flow arrows (frame 1->2)
#     klt_flow_2to3.svg        -- KLT flow arrows (frame 2->3)
#     klt_multiframe.svg       -- 3-frame tracking trails
#     klt_subpixel.svg         -- sub-pixel blob tracking demo

# Run on EuRoC MAV dataset
cargo run --example euroc_frontend --release -- /path/to/MH_01_easy
# Optional: limit to N frames
cargo run --example euroc_frontend --release -- /path/to/MH_01_easy 100
# -> outputs:
#     vis_output/euroc_tracks.svg   -- feature tracks overlaid on last frame
#     vis_output/euroc_stats.csv    -- per-frame statistics

# Live visualization window (requires minifb)
cargo run --example euroc_live --release -- /path/to/MH_01_easy
# Controls: Space=pause, S=step, Q/Esc=quit, +/-=speed, T=trails, F=flow

# GPU roundtrip example
cargo run --example gpu_pyramid                         # checkerboard
cargo run --example gpu_pyramid -- path/to/image.png    # real image
cargo run --example gpu_pyramid -- image.png 5 1.5      # 5 levels, sigma=1.5

# Run benchmarks
cargo bench
```

## Module Overview

| Module | Description | vilib Equivalent |
|---|---|---|
| `image` | `Image<T>`, `ImageView<'a, T>`, `Pixel` trait, bilinear interpolation | -- |
| `convert` | Pixel type conversions (u8<->f32, RGB->grayscale) | -- |
| `convolution` | Separable 1D convolution (horizontal + vertical) | `conv_filter_row.cu`, `conv_filter_col.cu` |
| `pyramid` | Gaussian image pyramid (blur + 2x downsample) | `pyramid_gpu.cu` |
| `fast` | FAST-N corner detector (N=9,10,11,12) with Rosten's high-speed test | `fast_gpu_cuda_tools.cu` |
| `nms` | Grid-based non-maximum suppression | `detector_base_gpu_cuda_tools.cu` |
| `gradient` | Sobel gradient computation (separable) | `harris_gpu_cuda_tools.cu` |
| `harris` | Harris corner detector (structure tensor) | `harris_gpu_cuda_tools.cu` |
| `klt` | Pyramidal Lucas-Kanade optical flow tracker | `feature_tracker_cuda_tools.cu` |
| `occupancy` | Occupancy grid for spatial feature distribution | `detector_base_gpu_cuda_tools.cu` |
| `histeq` | Histogram equalization (global + CLAHE) for brightness normalization | -- |
| `camera` | Pinhole camera model, EuRoC YAML parser, undistortion | -- |
| `essential` | 8-point essential matrix + RANSAC outlier rejection (uses lalir) | -- |
| `frontend` | Complete detect-track-replenish pipeline with geometric verification | `vilib_ros.cpp` |
| `gpu::device` | `GpuDevice` abstraction (device/queue/adapter selection) | -- |
| `gpu::pyramid` | WGSL Gaussian pyramid kernel | `pyramid_gpu.cu` |
| `gpu::fast` | WGSL FAST kernel with selectable NMS strategy | `fast_gpu_cuda_tools.cu` |
| `gpu::klt` | WGSL inverse-compositional KLT kernel | `feature_tracker_cuda_tools.cu` |
| `gpu::frontend` | `GpuFrontend` — API-compatible GPU replacement for `Frontend` | `vilib_ros.cpp` |

## Engineering Roadmap

We are adopting a verification-first development strategy. The project is divided into three distinct phases:

### Phase 1: Pure Rust CPU Reference (Complete)

**Goal:** Implement the core VIO frontend algorithms in safe, single-threaded Rust.

**Purpose:** Since debugging GPU shaders is difficult, this CPU implementation serves as the **Test Oracle** (Ground Truth). We did not write a single line of shader code until the math was verified here.

* [x] `Image<T>` Dynamic Container
* [x] Gaussian Pyramid Generation
* [x] FAST Corner Detection
* [x] Harris Corner Response
* [x] KLT (Lucas-Kanade) Feature Tracker
* [x] Occupancy Grid + Frontend Pipeline

### Phase 2: `wgpu` Acceleration (Complete)

**Goal:** Port the verified algorithms to WGSL compute shaders, targeting Vulkan
on NVIDIA, AMDGPU, and Raspberry Pi 4 & 5.

**Purpose:** Unlock real-time performance on embedded and cross-platform devices.

**Architecture — Hybrid CPU/GPU Model:**

GPU handles all heavy compute *within* a frame. CPU handles orchestration
*between* frames. The boundary falls at the KLT output readback.

```
┌─────────────────────────────── GPU (per frame) ──────────────────────────────┐
│  Image upload → Pyramid construction → FAST detection → KLT tracking         │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │ readback: tracked (x, y) positions only
                                        ▼
┌─────────────────────────────── CPU (per frame) ──────────────────────────────┐
│  RANSAC outlier rejection → Occupancy grid update → Replenishment decisions  │
└──────────────────────────────────────────────────────────────────────────────┘
```

The readback is mandatory because RANSAC (essential matrix estimation) requires
feature positions on the CPU every frame. This is a small buffer (~200 features
× 8 bytes) so latency is negligible. Pyramid levels, gradient images, and
detection candidates never need to touch the CPU.

**Hardware tuning knobs:**

Two enums on `GpuFrontendConfig` let you match the frontend to your hardware:

| Setting | iGPU / SoC (780M, Jetson, RPi) | Discrete GPU (RTX, RX) |
|---|---|---|
| `SubmitStrategy` | `Separate` (default) | `Fused` |
| `NmsStrategy` | `Cpu` (default) | `Gpu` |

`SubmitStrategy::Fused` records KLT and FAST into a single command encoder,
saving one `poll(Wait)` round-trip per frame (~1–2 ms over PCIe).

`NmsStrategy::Cpu` reads back the full score buffer (~1.4 MB for 752×480) and
runs cell-max NMS on the CPU. On unified memory this readback is essentially
free. `NmsStrategy::Gpu` runs an extra compute pass and reads back only the
per-cell winners (~17 KB), saving ~175 µs of PCIe transfer on discrete hardware.

**Milestones:**

* [x] `GpuDevice` abstraction (device/queue/command encoder wrappers)
* [x] Image upload + staging buffer infrastructure
* [x] Pyramid construction kernel (validated pixel-for-pixel against `Pyramid::build`)
* [x] FAST corner detection kernel
* [x] KLT tracking kernel (inverse compositional — constant Hessian is GPU-friendly)
* [x] `NmsStrategy::{Cpu, Gpu}` — selectable NMS for iGPU and discrete GPU
* [x] `SubmitStrategy::{Separate, Fused}` — selectable command encoder submit strategy
* [x] `GpuFrontend::process()` integration (API-compatible with CPU `Frontend`)
* [ ] Harris corner response kernel

### Phase 3: Zero-Copy Optimization

**Goal:** Optimize memory throughput for high-frequency odometry.

**Purpose:** Minimize CPU-GPU synchronization overhead on platforms where it matters.

* [ ] Async readback with double-buffering (overlap GPU frame N+1 with CPU frame N)
* [ ] Unified memory path for platforms that support it (Apple Silicon, RPi)

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
