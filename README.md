# Rudolf-V

**Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend **V**ision

> **Status: Phase 3 — CPU SIMD Optimization (Active)**
>
> The full GPU frontend pipeline is implemented and validated against the CPU reference.
> Phase 3 investigation revealed that at EuRoC resolution (752×480), wgpu dispatch
> overhead dominates GPU kernel compute — the CPU pipeline is faster on all tested
> hardware. Focus has shifted to SIMD optimization of the CPU pipeline, starting with
> pyramid construction on ARM (NEON) for the Raspberry Pi 4 target.

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

# Run all tests including GPU tests
cargo test -- --include-ignored

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
cargo run --example gpu_euroc_live --release -- /path/to/MH_01_easy # GPU version
# Controls: Space=pause, S=step, Q/Esc=quit, +/-=speed, T=trails, F=flow

# GPU roundtrip example
cargo run --example gpu_pyramid                         # checkerboard
cargo run --example gpu_pyramid -- path/to/image.png    # real image
cargo run --example gpu_pyramid -- image.png 5 1.5      # 5 levels, sigma=1.5

# Run benchmarks
cargo bench

# To enable parallel CPU processing, add <--features parallel> to the above commands.
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
| `essential` | 8-point essential matrix + RANSAC outlier rejection (uses nalgebra) | -- |
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

Two enums on `GpuFrontendConfig` expose the underlying tradeoffs:

| Setting | Options | Default |
|---|---|---|
| `SubmitStrategy` | `Separate`, `Fused` | `Separate` |
| `NmsStrategy` | `Cpu`, `Gpu` | `Cpu` |

`SubmitStrategy::Fused` records KLT and FAST into a single command encoder,
reducing `poll(Wait)` round-trips. `SubmitStrategy::Separate` lets the driver
schedule each stage independently; the CPU work between KLT and FAST (RANSAC,
grid update) acts as a natural gap that keeps the GPU fed without explicit fusion.

`NmsStrategy::Cpu` reads back the full score buffer (~1.4 MB for 752×480) and
runs cell-max NMS on the CPU. `NmsStrategy::Gpu` adds a compute pass and reads
back only per-cell winners (~17 KB), saving PCIe bandwidth on discrete hardware.

**Benchmark results (EuRoC MH_01, 752×480):**

| Platform | CPU Total | GPU Best | GPU Config | Winner |
|---|---|---|---|---|
| i5-12400F + RTX 3060 | 1.39ms | 1.95ms | Fused+Gpu | CPU |
| AMD Radeon 780M (iGPU) | 1.00ms | 1.82ms | Fused+Gpu | CPU |
| Raspberry Pi 4 | ~50ms | ~141ms | — | CPU (2.6×) |

**Why GPU is slower at this resolution:** At 752×480, individual GPU kernels
complete in <0.1ms — well below wgpu's fixed per-dispatch overhead (~50–100μs
for command recording, bind group creation, validation, and submit). The CPU
processes the same work serially with zero dispatch overhead. Analysis of the
reference CUDA implementation (vilib) confirmed that its GPU advantage comes
from CUDA-specific features unavailable in wgpu: mapped pinned memory
(`cudaHostAllocMapped`), ~5μs kernel launch latency, and persistent device
pointers that eliminate bind group creation entirely.

The GPU pipeline becomes advantageous at higher resolutions (1080p+) where
compute dominates dispatch overhead, or on hardware with lower CPU performance
relative to GPU capability.

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

### Phase 3: Optimization (Active)

**Goal:** Close the gap to OpenCV performance through platform-specific optimization.

**Purpose:** Make the CPU pipeline production-viable on embedded targets (RPi 4).

**Phase 3a — GPU overhead investigation (complete):**

Systematic comparison against vilib's CUDA implementation revealed that wgpu's
GPU advantage at EuRoC resolution is blocked by fixed overhead, not kernel
compute. Optimizations attempted and their outcomes:

| Optimization | Target | Result |
|---|---|---|
| Occupancy grid skip in FAST shader | Reduce GPU FAST work | No effect — dispatch overhead dominates, not compute |
| 5-tap kernel (match CPU) in pyramid shader | Reduce GPU pyramid work | No effect — same reason |
| `MAPPABLE_PRIMARY_BUFFERS` (STORAGE\|MAP_READ) | Eliminate readback copy on iGPU | No effect — copy was already free on unified memory |
| Comparison with vilib CUDA architecture | Identify fundamental gaps | Confirmed: mapped pinned memory, ~5μs launch, no bind groups |

Conclusion: at 752×480, the CPU pipeline wins on all current targets. GPU
pipeline is retained for future higher-resolution use cases.

**Phase 3b — CPU SIMD optimization (active):**

The CPU pipeline at 1.39ms (12400F) is already competitive, but the RPi 4 at
~50ms needs significant optimization. Current CPU bottleneck profile (RPi 4):

| Stage | Current | Target | Approach |
|---|---|---|---|
| Pyramid construction | ~24ms | ~3–4ms | Separable NEON convolution |
| KLT tracking | ~15ms | ~5ms | NEON Hessian accumulation |
| FAST detection | ~8ms | ~3ms | NEON ring comparisons |

* [x] KLT inverse-compositional restructuring (1.7× speedup)
* [x] Row-pointer memory access, constant weight precomputation
* [x] `KltScratch` allocation reuse
* [ ] Separable NEON convolution for pyramid construction (highest priority)
* [ ] NEON Hessian accumulation for KLT
* [ ] NEON ring comparisons for FAST
* [ ] Platform-specific CPU/GPU dispatch strategy

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
