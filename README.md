# Rudolf-V

**Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend **V**ision

A visual frontend for VIO in Rust: feature detection, pyramidal KLT tracking and
geometric verification, on the CPU and, through [wgpu](https://wgpu.rs), on the GPU.
It started as a port of the UZH RPG [vilib](https://github.com/uzh-rpg/vilib), which is
CUDA only; Rudolf-V runs wherever wgpu does (Vulkan, Metal, DX12).

> **Status:** the CPU and GPU frontends are complete and validated against each other;
> dense depth (SGBM, a DIS-style matcher, TensorRT stereo networks) and the Python wheel
> followed. New work comes as features are needed. There are no crates.io releases: depend
> on a git revision. The crate docs (`cargo doc --open`) list the supported API.

## Quick Start

```bash
cargo test                                    # CPU tests
cargo test --lib -- --ignored tests::test_    # GPU tests, each in its own process

# The GPU frontend live on a EuRoC sequence. Space: pause, S: step, T: trails,
# F: flow arrows, H: histogram equalization (none, global, CLAHE), Q: quit
cargo run --release --example gpu_euroc_live -- /path/to/V1_01_easy
```

As a dependency:

```toml
rudolf-v = { git = "https://github.com/twetto/Rudolf-V.git" }
```

Features: `gpu` (default) builds the `gpu::` modules on wgpu; with
`default-features = false` the CPU library needs neither wgpu nor a GPU. `parallel`
spreads the CPU work (detection, KLT, stereo matching) over threads with rayon.
`serde` adds `config`: the frontend's and stereo matcher's settings as the
`RudolfV:` and `Stereo:` sections of a YAML config file.

From Python, `crates/rudolf-v-python` (`pip install crates/rudolf-v-python`)
has the frontend, the sparse stereo matcher, the checkerboard finder, and dense
depth from SGBM or a TensorRT stereo network on a pair rectified (on the CPU or
the GPU) from a ZED's factory calibration:

```python
import rudolf_v
frontend = rudolf_v.Frontend(rudolf_v.FrontendConfig.from_yaml("config.yaml"), 752, 480)
features, stats = frontend.process(grey)  # uint8 (480, 752) array

rect = rudolf_v.StereoRectification.from_zed_conf("SN36835247.conf", fx, cx, cy)
depth = rudolf_v.StereoDepth.sgbm(rect.fx, rect.fy, rect.cx, rect.cy, rect.baseline, 1280, 720)
z = depth.compute(*rect.rectify(left, right))  # Z in metres, (90, 160), NaN for none
```

CI builds the wheel for Linux (x86-64 and aarch64) and Windows. For a Jetson,
`maturin build --release --zig --target aarch64-unknown-linux-gnu` in that
directory cross-builds one.

`tools/cross.sh` cross-compiles for a board, copies over ssh and runs tests or an
example there.

## Modules

| Area | Modules |
|---|---|
| Images | `image` (`Image<T>`, views, bilinear sampling), `convert`, `convolution`, `pyramid`, `histeq` (global, CLAHE) |
| Detection | `fast` (FAST-9…12), `shi_tomasi`, `harris`, `nms`, `occupancy` (spatial distribution), `checkerboard` (ChESS) |
| Tracking | `klt` (pyramidal, inverse-compositional), `essential` (8-point + RANSAC), `frontend` (detect, track, replenish, verify) |
| Cameras and stereo | `camera` (models, EuRoC calibration), `rectify`, `stereo` (sparse feature matching), `rigid_ransac` (3D-3D motion) |
| Dense depth | `dense_stereo` (DIS-style matcher), `sgbm` (as OpenCV's StereoSGBM, `MODE_HH`), `stereo_net` + `tensorrt` (LAS2, Light ESS; Unix) |
| GPU | `gpu::frontend` (`GpuFrontend`, a drop-in for `Frontend`) over `gpu::pyramid`, `fast`, `klt`, `histeq`, `clahe`; `gpu::dense_stereo`, `sgbm`, `checkerboard` |

## Architecture

The GPU does the heavy work within a frame; the CPU makes the decisions between
frames. The boundary is the readback of the tracked positions.

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

RANSAC needs the positions on the CPU every frame, but that is ~200 features ×
8 bytes; pyramids, gradients and detection candidates stay on the GPU.

- `GpuFrontend::process()` is `submit()` + `collect()`: CPU work done in between
  overlaps with the GPU.
- `SubmitStrategy` (default `Auto`): `Fused` records the frame into one submit,
  `Pipelined` puts FAST in a second one so the CPU stages overlap it, and `Auto`
  picks per frame. All give identical output. `NmsStrategy::Gpu` reads back only
  per-cell winners instead of the score buffer, for discrete GPUs.
- The GPU modules are tested against the CPU reference, kernel by kernel.

The development history, with the measurements behind these choices, is in
[docs/optimization-log.md](docs/optimization-log.md).

## Performance

Frontend, EuRoC V1_01_easy (752×480), mean time per frame:

| Platform | CPU, 1 thread | CPU, all cores | GPU | GPU config | Winner |
|---|---|---|---|---|---|
| Ryzen 7 8845H + Radeon 780M | 1.37 ms | 0.83 ms | **0.51 ms** | Auto | GPU (1.6×) |
| Jetson Orin Nano 8 GB, 15 W | 8.77 ms | 3.61 ms | **2.04 ms** | Pipelined | GPU (1.8×) |
| Ryzen 7 3700X + RX Vega 56 | 1.70 ms | 1.40 ms | **0.54 ms** | Auto | GPU (2.6×) |

- Measured on 2026-11-07 at `33ded57`; the Vega row at `becf218`, on EuRoC MH_01.
  "All cores" is `--features parallel`, and the winner is against it.
- At this resolution each GPU kernel takes well under 0.1 ms, below wgpu's
  per-dispatch cost, so the comparison is mostly dispatch overhead against CPU
  throughput: which side wins depends on the machine.
- Against OpenCV CUDA running the same pipeline on the Orin Nano (locked clocks,
  no histogram equalization): 1.11–1.13 ms for the GPU frontend, 1.37–1.40 ms
  for OpenCV.
- A Raspberry Pi 4 runs the CPU frontend at 12.7 ms per frame (7.6 ms on four
  cores). Its GPU frontend takes 116 ms.

Dense stereo, Orin Nano at 15 W, one 1280×720 pair to depth on a 160×90 grid,
on the GPU:

| Method | Matches at | Per pair |
|---|---|---|
| `GpuDenseStereo` | 160×90 (pyramid level 3) | 4.7 ms |
| `GpuSgbm` | 160×90 (pyramid level 3) | 4.6 ms |
| LAS2-S (`StereoNet`, TensorRT) | 320×192 input | 11.4 ms |
| Light ESS (`StereoNet`, TensorRT) | 480×288 input | 10.5 ms |

Pairs back to back. At one pair every 33 ms, the GPU's clock governor lowers
its clock between pairs, and the networks take 20.3 ms (LAS2-S) and 18.9 ms
(Light ESS) a pair, still with 1.2 ms of CPU time.

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
