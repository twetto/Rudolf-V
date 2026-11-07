// tensorrt.rs — A TensorRT engine through native/tensorrt/rv_trt.cpp, which
// is loaded at run time with dlopen: Rudolf-V links neither TensorRT nor
// CUDA, so it builds (and cross-compiles) without them, and a machine without
// them fails at `TrtEngine::load` with a message instead of at start-up.
//
// Build the library on the Jetson (native/tensorrt/build.sh) and the engine
// there too (trtexec --onnx=M.onnx --fp16 --saveEngine=M.engine): an engine
// only runs on the GPU and TensorRT version it was built for.
//
// Fixed shapes, float32 tensors. Each tensor has a pinned host buffer: fill
// the inputs' (`host_mut`), `run`, read the outputs' (`host`).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}
const RTLD_NOW: c_int = 2;

/// rv_trt.cpp's opaque handle.
#[repr(C)]
struct RvTrt {
    _private: [u8; 0],
}

type LoadFn = unsafe extern "C" fn(
    *const c_char,
    *const *const c_char,
    c_int,
    *mut c_char,
    usize,
) -> *mut RvTrt;
type NumTensorsFn = unsafe extern "C" fn(*const RvTrt) -> c_int;
type TensorFn = unsafe extern "C" fn(
    *const RvTrt,
    c_int,
    *mut *const c_char,
    *mut c_int,
    *mut i64,
    *mut *mut f32,
) -> c_int;
type RunFn = unsafe extern "C" fn(*mut RvTrt, *mut c_char, usize) -> c_int;
type FreeFn = unsafe extern "C" fn(*mut RvTrt);

/// One input or output of the engine.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub input: bool,
    pub dims: Vec<usize>,
    host: *mut f32,
    len: usize,
}

/// A deserialized engine with its execution context, CUDA stream and buffers.
pub struct TrtEngine {
    lib: *mut c_void,
    handle: *mut RvTrt,
    run: RunFn,
    free: FreeFn,
    tensors: Vec<TensorInfo>,
}

// The handle owns its CUDA stream and buffers; nothing in it is tied to the
// thread that made it.
unsafe impl Send for TrtEngine {}

fn last_dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string.
    let e = unsafe { dlerror() };
    if e.is_null() {
        "unknown dlopen error".into()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

fn message(buf: &[c_char]) -> String {
    // SAFETY: rv_trt.cpp writes a NUL-terminated message into the buffer.
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn c_path(p: &Path) -> Result<CString, String> {
    CString::new(p.as_os_str().as_encoded_bytes())
        .map_err(|_| format!("{}: NUL in path", p.display()))
}

impl TrtEngine {
    /// The engine at `engine` through the shim library `lib`
    /// (librv_trt.so), after loading TensorRT plugin libraries `plugins`
    /// (Light ESS needs its ess_plugins.so).
    pub fn load(lib: &Path, engine: &Path, plugins: &[&Path]) -> Result<Self, String> {
        let lib_c = c_path(lib)?;
        // SAFETY: a plain dlopen; the library's constructors are TensorRT's.
        let handle_lib = unsafe { dlopen(lib_c.as_ptr(), RTLD_NOW) };
        if handle_lib.is_null() {
            return Err(format!("{}: {}", lib.display(), last_dl_error()));
        }
        let sym = |name: &str| -> Result<*mut c_void, String> {
            let n = CString::new(name).unwrap();
            // SAFETY: handle_lib is a live dlopen handle.
            let p = unsafe { dlsym(handle_lib, n.as_ptr()) };
            if p.is_null() {
                Err(format!("{}: no {name}", lib.display()))
            } else {
                Ok(p)
            }
        };
        let fns = (|| {
            // SAFETY: the symbols are rv_trt.cpp's, with these signatures.
            unsafe {
                Ok::<_, String>((
                    std::mem::transmute::<*mut c_void, LoadFn>(sym("rv_trt_load")?),
                    std::mem::transmute::<*mut c_void, NumTensorsFn>(sym("rv_trt_num_tensors")?),
                    std::mem::transmute::<*mut c_void, TensorFn>(sym("rv_trt_tensor")?),
                    std::mem::transmute::<*mut c_void, RunFn>(sym("rv_trt_run")?),
                    std::mem::transmute::<*mut c_void, FreeFn>(sym("rv_trt_free")?),
                ))
            }
        })();
        let (load, num, tensor, run, free) = match fns {
            Ok(f) => f,
            Err(e) => {
                unsafe { dlclose(handle_lib) };
                return Err(e);
            }
        };
        let engine_c = c_path(engine)?;
        let plugins_c = plugins
            .iter()
            .map(|p| c_path(p))
            .collect::<Result<Vec<_>, _>>()?;
        let plugin_ptrs: Vec<*const c_char> = plugins_c.iter().map(|p| p.as_ptr()).collect();
        let mut err = [0 as c_char; 512];
        // SAFETY: valid C strings and an error buffer of the size passed.
        let handle = unsafe {
            load(
                engine_c.as_ptr(),
                plugin_ptrs.as_ptr(),
                plugin_ptrs.len() as c_int,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if handle.is_null() {
            unsafe { dlclose(handle_lib) };
            return Err(message(&err));
        }
        let mut tensors = Vec::new();
        for i in 0..unsafe { num(handle) } {
            let (mut name, mut input, mut dims, mut host) =
                (std::ptr::null(), 0, [0i64; 8], std::ptr::null_mut());
            // SAFETY: i is in range; the outputs point to locals.
            let nd = unsafe {
                tensor(
                    handle,
                    i,
                    &mut name,
                    &mut input,
                    dims.as_mut_ptr(),
                    &mut host,
                )
            };
            let dims: Vec<usize> = dims[..nd.clamp(0, 8) as usize]
                .iter()
                .map(|&d| d as usize)
                .collect();
            tensors.push(TensorInfo {
                name: unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned(),
                input: input != 0,
                len: dims.iter().product(),
                dims,
                host,
            });
        }
        Ok(Self {
            lib: handle_lib,
            handle,
            run,
            free,
            tensors,
        })
    }

    /// The engine's inputs and outputs, in its order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The index of the tensor named `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// Tensor i's host buffer (an input's to fill before `run`).
    pub fn host_mut(&mut self, i: usize) -> &mut [f32] {
        let t = &self.tensors[i];
        // SAFETY: rv_trt.cpp allocated `len` floats, freed only in Drop.
        unsafe { std::slice::from_raw_parts_mut(t.host, t.len) }
    }

    /// Tensor i's host buffer (an output's after `run`).
    pub fn host(&self, i: usize) -> &[f32] {
        let t = &self.tensors[i];
        // SAFETY: as host_mut.
        unsafe { std::slice::from_raw_parts(t.host, t.len) }
    }

    /// Copies the inputs over, runs the engine, copies the outputs back.
    pub fn run(&mut self) -> Result<(), String> {
        let mut err = [0 as c_char; 512];
        // SAFETY: a live handle and an error buffer of the size passed.
        match unsafe { (self.run)(self.handle, err.as_mut_ptr(), err.len()) } {
            0 => Ok(()),
            _ => Err(message(&err)),
        }
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        // SAFETY: the handle and library are live and dropped once.
        unsafe {
            (self.free)(self.handle);
            dlclose(self.lib);
        }
    }
}
