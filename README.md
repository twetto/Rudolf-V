# Rudolf-V

**Ru**st **D**evice **O**ptimized **L**ibrary for **F**rontend **V**ision

> **Status: Project Initialization**
>
> This repository is currently a **White Paper / Skeleton**. No functional code has been implemented yet.
> We are currently laying the groundwork for **Phase 1 (CPU Reference)**.

## About

**Rudolf-V** is an experimental Rust port of the [UZH RPG vilib](https://github.com/uzh-rpg/vilib) (CUDA Visual Library).

While the original `vilib` is a state-of-the-art, CUDA-only library designed strictly for NVIDIA platforms, **Rudolf-V** aims to democratize high-speed visual tracking. By leveraging **Rust** and **wgpu**, this library targets cross-platform compatibility, bringing GPU-accelerated visual frontends to "generous" hardware that CUDA leaves behind:

* **Raspberry Pi 4 & 5** (via Vulkan)
* AMD & Intel integrated graphics
* Apple Silicon
* NVIDIA GPUs

## Engineering Roadmap

We are adopting a verification-first development strategy. The project is divided into three distinct phases:

### Phase 1: Pure Rust CPU Reference (Current Focus)

**Goal:** Implement the core VIO frontend algorithms in safe, single-threaded Rust.

**Purpose:** Since debugging GPU shaders is difficult, this CPU implementation will serve as the **Test Oracle** (Ground Truth). We will not write a single line of shader code until the math is verified here.

* [x] `Image<T>` Dynamic Container
* [x] Gaussian Pyramid Generation
* [x] FAST Corner Detection
* [ ] Harris Corner Response
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

