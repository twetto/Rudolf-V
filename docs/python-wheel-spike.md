# Camera geometry Python packaging spike

## Decision under test

`neuro-dataloaders` owns dataset discovery, timestamp synchronization, image,
IMU, ground-truth, and calibration parsing. Its camera value is hardware
metadata:

- model name;
- intrinsics and distortion coefficients;
- resolution;
- camera/body extrinsics.

It must not become an independent projection authority. Projection,
unprojection, and their Jacobians belong to `camera-geometry`.

Rust consumers use the crate directly. Python consumers, including
`mr-neuro-nav`, use a small PyO3 wheel built from the same crate. Rudolf-V and
ECHO-LI remain consumers; neither owns the Python camera contract.

```text
neuro-dataloaders --calibration metadata--+
                                          |
camera-geometry --------------------------+--> Rudolf-V
        |
        +--> camera-geometry Python wheel ----> mr-neuro-nav
        |
        +-------------------------------------> ECHO-LI
```

Normal users must not need Rust. CI builds binary wheels, installs each wheel
in a clean Python environment, and runs an import plus TUM-VI projection
round-trip test. Rust is required only for maintainers changing the binding or
the geometry implementation.

## Initial support matrix

- CPython 3.10 and newer through PyO3 `abi3-py310`;
- Windows x86-64;
- Linux x86-64 with a manylinux-compatible wheel.

macOS and additional architectures are added only when an actual consumer
requires them.

The extension should remain free of OpenCV and other native dynamic-library
dependencies. Image decoding stays in Python. This keeps wheel portability
separate from dataset and frontend concerns.

## Probe gates

The approach is accepted only if:

1. `camera-geometry` tests pass independently;
2. Windows and Linux CI build wheels;
3. each wheel installs into a clean virtual environment without Rust;
4. Python can parse/use the checked TUM-VI equidistant calibration parameters;
5. center and edge projection round trips pass;
6. unsupported camera models fail explicitly.

Until these gates pass, `neuro-dataloaders` should not take a mandatory
dependency on the extension. Its current Python camera implementation can be
retained temporarily as a compatibility backend, but new fisheye mathematics
must not be added there.

## Developer commands

From `crates/camera-geometry-python`:

```powershell
py -m pip install maturin
py -m maturin build --release
py -m pip install --force-reinstall ..\..\target\wheels\camera_geometry-*.whl
py tests\smoke.py
```

CI is the release authority; locally built wheels are development artifacts.
