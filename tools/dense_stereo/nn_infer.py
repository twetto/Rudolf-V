"""Run a stereo network's TensorRT engine over a recording's rectified pairs; write its disparity
(and confidence, if it has one) at the network's resolution as float16, for zed_eval.py --nn.

usage: nn_infer.py MODEL ENGINE SRC OUT [--plugin LIB]
  MODEL  light_ess (480x288, stretched, RGB scaled to [-1, 1]; outputs output_left, output_conf)
         or las2_WxH (the image resized to W wide keeping its aspect, centred, rows replicated to H;
         RGB 0..255; output disparity)
  SRC    zed_record.py extract output (rect/left, rect/right, grey PNGs)

Device memory and copies through the CUDA runtime via ctypes (no pycuda here)."""
import argparse
import ctypes
import json
import os
import time

import cv2
import numpy as np
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("engine")
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--plugin")
args = ap.parse_args()

cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
H2D, D2H = 1, 2


def check(err, what):
    if err != 0:
        raise RuntimeError(f"{what}: CUDA error {err}")


def malloc(nbytes):
    p = ctypes.c_void_p()
    check(cuda.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return p.value


def copy(dst, src, nbytes, kind):
    check(cuda.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(nbytes), kind), "cudaMemcpy")


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
if args.plugin:
    trt.get_plugin_registry().load_library(args.plugin)
engine = trt.Runtime(logger).deserialize_cuda_engine(open(args.engine, "rb").read())
ctx = engine.create_execution_context()
tensors = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    host = np.zeros(shape, dtype)
    dev = malloc(host.nbytes)
    ctx.set_tensor_address(name, dev)
    tensors[name] = (host, dev, engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
print({n: (h.shape, h.dtype, "in" if i else "out") for n, (h, _, i) in tensors.items()})

inputs = sorted((n for n, t in tensors.items() if t[2]), key=lambda n: "right" in n)  # left first
if args.model == "light_ess":
    W, H = 480, 288
    content = (W, H, 0)  # width, height, top padding

    def prep(grey):
        im = cv2.resize(grey, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        return np.broadcast_to((im - 127.5) / 127.5, (3, H, W))[None]

    disp_name, conf_name = "output_left", "output_conf"
else:
    W, H = map(int, args.model.split("_")[1].split("x"))
    ch = round(720 * W / 1280)
    top = (H - ch) // 2
    content = (W, ch, top)

    def prep(grey):
        im = cv2.resize(grey, (W, ch), interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, top, H - ch - top, 0, 0, cv2.BORDER_REPLICATE).astype(np.float32)
        return np.broadcast_to(im, (3, H, W))[None]

    disp_name, conf_name = [n for n, t in tensors.items() if not t[2]][0], None

os.makedirs(args.out, exist_ok=True)
json.dump({"name": args.model, "width": content[0], "height": content[1], "top": content[2],
           "full_width": 1280, "full_height": 720, "confidence": conf_name is not None},
          open(os.path.join(args.out, "meta.json"), "w"))
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.src, "rect/left")))
ms = []
for stem in stems:
    ims = [cv2.imread(os.path.join(args.src, "rect", eye, stem + ".png"), cv2.IMREAD_GRAYSCALE)
           for eye in ("left", "right")]
    for name, im in zip(inputs, ims):
        host, dev, _ = tensors[name]
        host[...] = prep(im)
        copy(dev, host.ctypes.data, host.nbytes, H2D)
    t0 = time.perf_counter()
    assert ctx.execute_async_v3(0)
    check(cuda.cudaDeviceSynchronize(), "sync")
    ms.append((time.perf_counter() - t0) * 1e3)
    for name, ext in ((disp_name, "disp"), (conf_name, "conf")):
        if name is None:
            continue
        host, dev, _ = tensors[name]
        copy(host.ctypes.data, dev, host.nbytes, D2H)
        a = host.reshape(host.shape[-2:])[content[2]:content[2] + content[1], :content[0]]
        a.astype(np.float16).tofile(os.path.join(args.out, f"{stem}.{ext}"))
print(f"{len(stems)} pairs; execute + sync median {np.median(ms):.2f} ms (default stream, no overlap)")
