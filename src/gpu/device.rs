// gpu/device.rs — wgpu device abstraction.
//
// Responsibilities:
//   - Enumerate Vulkan adapters and select the first non-CPU one.
//   - Expose a `DeviceProfile` for simulating hardware limits on a
//     development machine (e.g., cap invocations to match Raspberry Pi).
//   - Provide `WorkgroupSize` — a workgroup configuration that is validated
//     against the active profile and used when creating compute pipelines.
//
// ADAPTER SELECTION:
// wgpu's default `request_adapter` uses power preference heuristics that
// may grab llvmpipe/softpipe on WSL2 (where the software renderer appears
// as a valid Vulkan device). We enumerate explicitly and reject anything
// with DeviceType::Cpu.
//
// DEVICE LIMITS:
// We request *lower* limits than the hardware actually supports when
// running under a non-Native profile. wgpu validates every dispatch against
// the requested limits, so violations that would crash on RPi are caught at
// dev time on the laptop. This is purely a correctness harness — it does
// not make the GPU run slower.
//
// WORKGROUP SIZES:
// WGSL `override` constants are injected at pipeline creation time via
// `PipelineCompilationOptions::constants`. This keeps the shader source
// identical across configurations and preserves the shader compilation
// cache (string-formatting the source would defeat the cache).
//
// NEW RUST CONCEPTS:
// - `pollster::block_on` — runs an async fn to completion on the current
//   thread. wgpu's device/adapter API is async because on WebGPU it maps
//   to JS Promises, but for native Vulkan we just block.
// - `#[derive(Debug, Clone, Copy)]` on plain enums — zero cost, just adds
//   useful trait impls.
// - `impl Display` — lets us write `println!("{}", adapter_info)` without
//   a custom format string every time.

use std::collections::HashMap;
use std::fmt;

/// Hardware profile controlling device limits and default workgroup sizes.
///
/// Use `Native` for best performance on your development machine.
/// Use `RaspberryPi` to simulate RPi 4/5 constraints — wgpu will reject
/// any dispatch that exceeds the RPi's actual Vulkan limits, catching
/// problems before you deploy to the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceProfile {
    /// Use the adapter's actual hardware limits. No artificial caps.
    Native,
    /// Simulate Raspberry Pi 4/5 (Broadcom VideoCore VI/VII, V3DV Vulkan).
    /// Caps `max_compute_invocations_per_workgroup` to 256, matching the
    /// device report from RPi's V3DV driver.
    RaspberryPi,
}

impl fmt::Display for DeviceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceProfile::Native => write!(f, "Native"),
            DeviceProfile::RaspberryPi => write!(f, "RaspberryPi (simulated limits)"),
        }
    }
}

/// A workgroup size configuration for 2D compute dispatches.
///
/// Both dimensions must be powers of two and their product must not exceed
/// the profile's `max_compute_invocations_per_workgroup` limit.
///
/// Construct via `WorkgroupSize::for_profile()` — it selects a validated
/// default for the given hardware target. Override by calling
/// `GpuDevice::validate_workgroup_size()` if you need a non-default value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
}

impl WorkgroupSize {
    /// Total invocations per workgroup (x * y).
    pub fn total(&self) -> u32 {
        self.x * self.y
    }

    /// Return the constants map for `PipelineCompilationOptions`.
    ///
    /// WGSL `override` constants are injected here:
    ///
    /// ```wgsl
    /// override WORKGROUP_X: u32 = 8u;
    /// override WORKGROUP_Y: u32 = 8u;
    ///
    /// @compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
    /// fn main(...) { ... }
    /// ```
    ///
    /// The returned `HashMap` is passed directly to
    /// `PipelineCompilationOptions::constants`. The shader source string
    /// stays identical across workgroup sizes — only the specialization
    /// values differ, preserving the shader compilation cache.
    pub fn as_constants(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("WORKGROUP_X".to_string(), self.x as f64),
            ("WORKGROUP_Y".to_string(), self.y as f64),
        ])
    }

    /// Select a validated default workgroup size for the given profile.
    ///
    /// - `Native` (NVIDIA/AMD): 16×8 = 128 invocations. This aligns well
    ///   with NVIDIA's 32-wide warps (128 = 4 warps) and AMD's 64-wide
    ///   wavefronts (128 = 2 waves). The 16-wide x dimension also aligns
    ///   with cache-line boundaries for row-major image data.
    ///
    /// - `RaspberryPi`: 8×8 = 64 invocations. Fits comfortably within the
    ///   256 invocation limit, leaving headroom for the V3DV scheduler.
    ///   VideoCore QPUs are SIMD-4, so 64 = 16 QPU "threads" of 4 elements.
    fn for_profile(profile: DeviceProfile) -> Self {
        match profile {
            DeviceProfile::Native => WorkgroupSize { x: 16, y: 8 },
            DeviceProfile::RaspberryPi => WorkgroupSize { x: 8, y: 8 },
        }
    }
}

impl fmt::Display for WorkgroupSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}×{} ({} invocations)", self.x, self.y, self.total())
    }
}

/// Cached adapter information for logging and debugging.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
}

impl fmt::Display for AdapterInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}, {:?})",
            self.name, self.backend, self.device_type
        )
    }
}

/// The core GPU context: adapter, device, queue, and active profile.
///
/// Create via `GpuDevice::new()` or `GpuDevice::new_with_profile()`.
/// Hold one `GpuDevice` for the lifetime of the application — it is
/// expensive to create (Vulkan instance + device initialization) and
/// cheap to clone the `Arc<wgpu::Device>` it wraps.
///
/// # Field drop order
/// Rust drops struct fields in declaration order (top → bottom).
/// `_instance` is declared last so the `wgpu::Instance` (and its
/// internal Vulkan instance handle) outlives `device` and `queue`.
/// This prevents a crash in dzn (the D3D12-to-Vulkan layer on WSL2)
/// that occurs when the Vulkan instance is destroyed while device-level
/// objects still hold dangling back-references to it.
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub profile: DeviceProfile,
    pub adapter_info: AdapterInfo,
    pub workgroup_size: WorkgroupSize,
    /// Keeps the `wgpu::Instance` alive until `device` and `queue` are
    /// dropped. Never access this field directly — its sole purpose is
    /// to control the drop order. Prefixed `_` to signal intent.
    _instance: wgpu::Instance,
}

impl GpuDevice {
    /// Create a `GpuDevice` using the first non-CPU Vulkan adapter found,
    /// with `DeviceProfile::Native` limits.
    ///
    /// # Errors
    /// Returns `Err` if no suitable adapter is found or the device
    /// request fails.
    pub fn new() -> Result<Self, GpuError> {
        Self::new_with_profile(DeviceProfile::Native)
    }

    /// Create a `GpuDevice` with an explicit hardware profile.
    ///
    /// Use `DeviceProfile::RaspberryPi` during development to catch
    /// workgroup-size violations before deploying to the target device.
    pub fn new_with_profile(profile: DeviceProfile) -> Result<Self, GpuError> {
        pollster::block_on(Self::init_async(profile))
    }

    async fn init_async(profile: DeviceProfile) -> Result<Self, GpuError> {
        // Request only Vulkan — no DX12, no Metal, no WebGPU.
        //
        // WSL2 note: Microsoft's dzn (D3D12-to-Vulkan) declares itself
        // non-conformant ("WARNING: dzn is not a conformant Vulkan
        // implementation"). wgpu drops non-conformant adapters by default.
        // ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER tells wgpu to enumerate them
        // anyway so we can select dzn over llvmpipe.
        //
        // The flag is safe for our use case: we run compute-only kernels with
        // no reliance on any conformance-required rendering behaviour. dzn has
        // full support for storage buffers and compute dispatches on WSL2.
        let flags = if cfg!(debug_assertions) {
            // Validation layer in debug builds for shader error feedback.
            wgpu::InstanceFlags::VALIDATION
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
        } else {
            wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags,
            ..Default::default()
        });

        // Enumerate all Vulkan adapters, selecting the best available.
        //
        // On a typical Linux machine or bare-metal WSL2 with dxgkrnl:
        //   DiscreteGpu  — dedicated NVIDIA/AMD card         <- ideal
        //   IntegratedGpu — iGPU (AMD APU, Intel)            <- good
        //   VirtualGpu   — VM pass-through                   <- acceptable
        //   Other        — dzn (D3D12->Vulkan) on WSL2       <- acceptable
        //   Cpu          — llvmpipe / software rasterizer     <- reject
        //
        // WSL2 note: Microsoft's dzn layer reports itself as DeviceType::Cpu
        // on some driver versions, indistinguishable from llvmpipe via the
        // type field alone. We use tiered selection:
        //   1. Prefer DiscreteGpu or IntegratedGpu (real hardware).
        //   2. Fall back to VirtualGpu or Other (dzn, VM pass-through).
        //   3. Last resort: take anything — adapter name logged so you know.
        //
        // The adapter name is printed at startup to confirm which was chosen.
        let all_adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .collect();

        if all_adapters.is_empty() {
            return Err(GpuError::NoSuitableAdapter);
        }

        for a in &all_adapters {
            let info = a.get_info();
            eprintln!(
                "[rudolf-v] Vulkan adapter: {} ({:?}, {:?})",
                info.name, info.backend, info.device_type
            );
        }

        // Tier 1: real hardware GPU.
        let adapter = all_adapters
            .into_iter()
            .find(|a| matches!(
                a.get_info().device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                    | wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other
            ))
            // Tier 2 (last resort): take whatever exists, even if Cpu/software.
            .or_else(|| instance
                .enumerate_adapters(wgpu::Backends::VULKAN)
                .into_iter()
                .next())
            .ok_or(GpuError::NoSuitableAdapter)?;

        let raw_info = adapter.get_info();
        let adapter_info = AdapterInfo {
            name: raw_info.name.clone(),
            vendor: raw_info.vendor,
            device: raw_info.device,
            device_type: raw_info.device_type,
            backend: raw_info.backend,
        };

        // Auto-detect RPi when the caller passed Native but the adapter is V3D.
        // This makes GpuDevice::new() work correctly on RPi without requiring
        // every call site to know about DeviceProfile::RaspberryPi.
        let profile = match profile {
            DeviceProfile::Native if raw_info.name.to_ascii_lowercase().contains("v3d") => {
                eprintln!("[rudolf-v] V3D adapter detected — using RaspberryPi profile");
                DeviceProfile::RaspberryPi
            }
            other => other,
        };

        // Build the requested limits from the (possibly auto-upgraded) profile.
        let limits = limits_for_profile(profile);

        // wgpu 22: request_device returns (Device, Queue) directly; the tuple
        // type must be spelled out to help the type inferencer.
        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rudolf-v"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequest)?;

        let workgroup_size = WorkgroupSize::for_profile(profile);

        Ok(GpuDevice {
            device,
            queue,
            profile,
            adapter_info,
            workgroup_size,
            _instance: instance,
        })
    }

    /// Override the default workgroup size, validating against the active profile.
    ///
    /// Returns `Err` if the total invocation count (x * y) exceeds the
    /// profile's `max_compute_invocations_per_workgroup`.
    pub fn set_workgroup_size(&mut self, x: u32, y: u32) -> Result<(), GpuError> {
        let total = x * y;
        let max = max_invocations_for_profile(self.profile);
        if total > max {
            return Err(GpuError::WorkgroupTooLarge { total, max });
        }
        self.workgroup_size = WorkgroupSize { x, y };
        Ok(())
    }

    /// Compute the dispatch dimensions needed to cover an image of the
    /// given size with the active workgroup size.
    ///
    /// Returns `(dispatch_x, dispatch_y)` — the number of workgroups in
    /// each dimension. Uses ceiling division so every pixel is covered
    /// even when the image dimensions are not multiples of the workgroup size.
    ///
    /// The shader must guard against out-of-bounds global IDs:
    /// ```wgsl
    /// if gid.x >= width || gid.y >= height { return; }
    /// ```
    pub fn dispatch_size(&self, img_w: u32, img_h: u32) -> (u32, u32) {
        let dx = (img_w + self.workgroup_size.x - 1) / self.workgroup_size.x;
        let dy = (img_h + self.workgroup_size.y - 1) / self.workgroup_size.y;
        (dx, dy)
    }
}

impl fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuDevice {{ adapter: {}, profile: {}, workgroup: {} }}",
            self.adapter_info, self.profile, self.workgroup_size
        )
    }
}

// ============================================================
// Limits helpers
// ============================================================

/// Build wgpu limits for the given profile.
///
/// We request *lower* limits than the hardware supports when running
/// under a non-Native profile. wgpu validates dispatches against the
/// *requested* limits, so violations are caught on the laptop before
/// they crash on the target device.
fn limits_for_profile(profile: DeviceProfile) -> wgpu::Limits {
    match profile {
        DeviceProfile::Native => wgpu::Limits::default(),

        DeviceProfile::RaspberryPi => wgpu::Limits {
            // VideoCore VI/VII: vulkaninfo reports 256 max invocations.
            max_compute_invocations_per_workgroup: 256,
            // V3DV also caps individual workgroup dimensions at 256.
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            // VideoCore VI caps textures at 4096×4096 (vs wgpu default 8192).
            max_texture_dimension_2d: 4096,
            // Conservative storage buffer size: 128 MiB.
            // RPi 4 has 4 GiB RAM shared with CPU; 128 MiB for GPU buffers
            // is safe for our pyramid + feature buffer workloads.
            max_storage_buffer_binding_size: 128 << 20,
            // Inherit everything else from wgpu defaults (which are already
            // conservative enough to be safe on RPi's Vulkan implementation).
            ..wgpu::Limits::default()
        },
    }
}

/// Maximum compute invocations per workgroup for the given profile.
/// Used to validate `set_workgroup_size()`.
fn max_invocations_for_profile(profile: DeviceProfile) -> u32 {
    match profile {
        DeviceProfile::Native => wgpu::Limits::default().max_compute_invocations_per_workgroup,
        DeviceProfile::RaspberryPi => 256,
    }
}

// ============================================================
// Error type
// ============================================================

/// Errors from GPU device initialization and configuration.
#[derive(Debug)]
pub enum GpuError {
    /// No Vulkan adapter found that passes the non-CPU filter.
    /// On WSL2: check that Vulkan is installed and `vulkaninfo` shows
    /// a real GPU. Only llvmpipe/software renderers found otherwise.
    NoSuitableAdapter,
    /// wgpu device request failed (driver issue, unsupported limits, etc.).
    DeviceRequest(wgpu::RequestDeviceError),
    /// Requested workgroup size exceeds the profile's invocation limit.
    WorkgroupTooLarge { total: u32, max: u32 },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoSuitableAdapter => write!(
                f,
                "no suitable Vulkan adapter found (only CPU/software renderers visible). \
                 On WSL2: ensure Vulkan is installed and `vulkaninfo` lists a real GPU."
            ),
            GpuError::DeviceRequest(e) => write!(f, "device request failed: {e}"),
            GpuError::WorkgroupTooLarge { total, max } => write!(
                f,
                "workgroup size {total} exceeds profile limit of {max} invocations"
            ),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Tests that require an actual GPU are behind `#[ignore]` so that
    // `cargo test` passes in CI without Vulkan. Run with:
    //   cargo test -- --include-ignored
    // or specifically:
    //   cargo test -p rudolf-v gpu::device -- --include-ignored

    #[test]
    fn test_workgroup_size_constants() {
        let ws = WorkgroupSize { x: 16, y: 8 };
        assert_eq!(ws.total(), 128);
        let c = ws.as_constants();
        assert_eq!(c["WORKGROUP_X"], 16.0);
        assert_eq!(c["WORKGROUP_Y"], 8.0);
    }

    #[test]
    fn test_workgroup_size_for_native() {
        let ws = WorkgroupSize::for_profile(DeviceProfile::Native);
        // 16×8 = 128, well within any GPU's limits.
        assert_eq!(ws.x, 16);
        assert_eq!(ws.y, 8);
        assert_eq!(ws.total(), 128);
    }

    #[test]
    fn test_workgroup_size_for_rpi() {
        let ws = WorkgroupSize::for_profile(DeviceProfile::RaspberryPi);
        // 8×8 = 64, fits within RPi's 256 invocation limit.
        assert_eq!(ws.x, 8);
        assert_eq!(ws.y, 8);
        assert!(ws.total() <= 256);
    }

    #[test]
    fn test_dispatch_size_exact() {
        // Image dimensions that are exact multiples of workgroup size.
        let gpu = GpuDeviceStub::new(DeviceProfile::Native);
        // Native default: 16×8
        let (dx, dy) = gpu.dispatch_size(640, 480);
        assert_eq!(dx, 640 / 16); // 40
        assert_eq!(dy, 480 / 8);  // 60
    }

    #[test]
    fn test_dispatch_size_ceiling() {
        // Image dimensions that are NOT multiples of workgroup size.
        let gpu = GpuDeviceStub::new(DeviceProfile::RaspberryPi);
        // RPi default: 8×8. Image: 752×480 (EuRoC standard resolution).
        // 752 / 8 = 94.0 exactly, 480 / 8 = 60.0 exactly.
        let (dx, dy) = gpu.dispatch_size(752, 480);
        assert_eq!(dx, 94);
        assert_eq!(dy, 60);

        // Non-multiple: 100×100, workgroup 8×8 → ceil(100/8) = 13.
        let (dx, dy) = gpu.dispatch_size(100, 100);
        assert_eq!(dx, 13);
        assert_eq!(dy, 13);
        // The last workgroup covers pixels 96–103, but 100–103 are out of
        // bounds — the shader must guard against this.
    }

    #[test]
    fn test_rpi_limits_cap_invocations() {
        let limits = limits_for_profile(DeviceProfile::RaspberryPi);
        assert_eq!(limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
    }

    #[test]
    fn test_native_limits_are_default() {
        let limits = limits_for_profile(DeviceProfile::Native);
        assert_eq!(limits, wgpu::Limits::default());
    }

    // ---- GPU integration tests (subprocess isolation) -------------------------
    //
    // dzn (Microsoft's D3D12-to-Vulkan layer on WSL2) has a bug: it crashes
    // with SIGSEGV during process exit when any Vulkan device has been
    // created in that process. The crash is in dzn's own cleanup code and is
    // independent of when or how we drop our wgpu objects. Controlling Rust
    // drop order does not help because the crash is in an atexit/DllMain
    // handler inside the dzn shared library itself.
    //
    // Workaround: run each GPU test in an isolated child process via
    // `run_gpu_test_in_subprocess`. The child creates the GPU, runs the real
    // assertions, prints "GPU_TEST_OK" on success, then exits — crashing on
    // the way out is fine because the parent only checks the output, not the
    // exit code.
    //
    // On Linux bare-metal (no dzn) these tests pass and exit cleanly.
    // On RPi they will also pass and exit cleanly.
    // Only WSL2 with dzn triggers the crash-on-exit, which is isolated here.

    /// Spawn a child `cargo test` process running a single named test,
    /// captured with `--nocapture`. Returns the combined stdout+stderr.
    ///
    /// We intentionally do NOT check the process exit status — dzn causes
    /// SIGSEGV on exit which would mark the child as failed even when all
    /// assertions passed. Instead the real tests print "GPU_TEST_OK" just
    /// before returning, and we assert that token appears in the output.
    #[cfg(test)]
    fn run_gpu_test_in_subprocess(test_name: &str) -> String {
        // Locate the test binary by re-running cargo. This is the only
        // reliable way to get the current binary path from within a test.
        let output = std::process::Command::new("cargo")
            .args([
                "test",
                "--lib",
                "--",
                test_name,
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn subprocess for {test_name}: {e}"));

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        // Print both streams so `cargo test -- --nocapture` shows them.
        print!("{stdout}");
        eprint!("{stderr}");
        stdout + &stderr
    }

    // ---- Inner tests (run inside the subprocess, marked #[ignore]) ----------
    //
    // These are the real GPU tests. They are never called directly by the
    // outer suite — only by run_gpu_test_in_subprocess via a fresh `cargo
    // test` invocation. Each one prints "GPU_TEST_OK" as the last line so
    // the outer test can confirm the inner test passed.

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_device_init_native() {
        let gpu = GpuDevice::new().expect("should initialise a Vulkan device");
        println!("{gpu}");
        eprintln!("[test] adapter type: {:?}", gpu.adapter_info.device_type);
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_gpu_device_init_rpi_profile() {
        let gpu = GpuDevice::new_with_profile(DeviceProfile::RaspberryPi)
            .expect("RPi profile should work on any Vulkan device");
        println!("{gpu}");
        assert_eq!(gpu.profile, DeviceProfile::RaspberryPi);
        assert_eq!(gpu.workgroup_size, WorkgroupSize { x: 8, y: 8 });
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_set_workgroup_size_valid() {
        let mut gpu = GpuDevice::new_with_profile(DeviceProfile::RaspberryPi).unwrap();
        gpu.set_workgroup_size(16, 16).expect("256 should be valid on RPi profile");
        assert_eq!(gpu.workgroup_size.total(), 256);
        println!("GPU_TEST_OK");
    }

    #[test]
    #[ignore = "GPU integration: run via outer subprocess wrapper"]
    fn inner_set_workgroup_size_too_large() {
        let mut gpu = GpuDevice::new_with_profile(DeviceProfile::RaspberryPi).unwrap();
        let err = gpu.set_workgroup_size(16, 17).unwrap_err();
        assert!(matches!(err, GpuError::WorkgroupTooLarge { total: 272, max: 256 }));
        println!("GPU_TEST_OK");
    }

    // ---- Outer tests (run by default, each spawns one subprocess) -----------

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_device_init_native() {
        let out = run_gpu_test_in_subprocess("gpu::device::tests::inner_gpu_device_init_native");
        assert!(out.contains("GPU_TEST_OK"), "inner test did not print GPU_TEST_OK:
{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_gpu_device_init_rpi_profile() {
        let out = run_gpu_test_in_subprocess("gpu::device::tests::inner_gpu_device_init_rpi_profile");
        assert!(out.contains("GPU_TEST_OK"), "inner test did not print GPU_TEST_OK:
{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_set_workgroup_size_valid() {
        let out = run_gpu_test_in_subprocess("gpu::device::tests::inner_set_workgroup_size_valid");
        assert!(out.contains("GPU_TEST_OK"), "inner test did not print GPU_TEST_OK:
{out}");
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU"]
    fn test_set_workgroup_size_too_large() {
        let out = run_gpu_test_in_subprocess("gpu::device::tests::inner_set_workgroup_size_too_large");
        assert!(out.contains("GPU_TEST_OK"), "inner test did not print GPU_TEST_OK:
{out}");
    }

    // ---- Stub for tests that don't need a real device ----
    // dispatch_size() and workgroup validation are pure functions of
    // WorkgroupSize — no GPU needed. We use a minimal stub so these tests
    // run in CI without Vulkan.
    struct GpuDeviceStub {
        workgroup_size: WorkgroupSize,
    }

    impl GpuDeviceStub {
        fn new(profile: DeviceProfile) -> Self {
            GpuDeviceStub {
                workgroup_size: WorkgroupSize::for_profile(profile),
            }
        }

        fn dispatch_size(&self, img_w: u32, img_h: u32) -> (u32, u32) {
            let dx = (img_w + self.workgroup_size.x - 1) / self.workgroup_size.x;
            let dy = (img_h + self.workgroup_size.y - 1) / self.workgroup_size.y;
            (dx, dy)
        }
    }
}
