"""Time a TensorRT engine as a 30 fps pipeline would call it: one inference per frame period on its
own CUDA stream, optionally replayed from a CUDA graph; also back to back, as trtexec does.

usage: nn_time.py ENGINE [--plugin LIB] [--period MS] [--frames N]"""
import argparse
import ctypes
import time

import numpy as np
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("--plugin")
ap.add_argument("--period", type=float, default=1000 / 30)
ap.add_argument("--frames", type=int, default=300)
args = ap.parse_args()

cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")


def check(err, what):
    if err != 0:
        raise RuntimeError(f"{what}: CUDA error {err}")


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
if args.plugin:
    trt.get_plugin_registry().load_library(args.plugin)
engine = trt.Runtime(logger).deserialize_cuda_engine(open(args.engine, "rb").read())
ctx = engine.create_execution_context()
stream = ctypes.c_void_p()
check(cuda.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    n = int(np.prod(engine.get_tensor_shape(name))) * np.dtype(trt.nptype(engine.get_tensor_dtype(name))).itemsize
    p = ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
    check(cuda.cudaMemset(p, 0, ctypes.c_size_t(n)), "cudaMemset")
    ctx.set_tensor_address(name, p.value)


def run_plain():
    assert ctx.execute_async_v3(stream.value)


graph_exec = None


def run_graph():
    check(cuda.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")


def measure(run, period):
    ts = []
    for k in range(args.frames):
        t0 = time.perf_counter()
        run()
        check(cuda.cudaStreamSynchronize(stream), "sync")
        dt = time.perf_counter() - t0
        if k >= 10:
            ts.append(dt * 1e3)
        if period:
            time.sleep(max(0.0, period / 1e3 - dt))
    return np.median(ts), np.percentile(ts, 90)


run_plain()
check(cuda.cudaStreamSynchronize(stream), "sync")
modes = [("plain", run_plain)]
try:  # cudaStreamCaptureModeThreadLocal = 1
    g = ctypes.c_void_p()
    check(cuda.cudaStreamBeginCapture(stream, 1), "begin capture")
    run_plain()
    check(cuda.cudaStreamEndCapture(stream, ctypes.byref(g)), "end capture")
    e = ctypes.c_void_p()
    check(cuda.cudaGraphInstantiate(ctypes.byref(e), g, ctypes.c_ulonglong(0)), "instantiate")
    graph_exec = e
    modes.append(("CUDA graph", run_graph))
except (RuntimeError, AssertionError) as err:
    print("no CUDA graph:", err)
for label, run in modes:
    b = measure(run, 0)
    p = measure(run, args.period)
    print(f"{args.engine}: {label:10s} back to back median {b[0]:6.2f} ms (p90 {b[1]:6.2f}); "
          f"one per {args.period:.1f} ms median {p[0]:6.2f} ms (p90 {p[1]:6.2f})")
