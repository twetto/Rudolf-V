// rv_trt.cpp — A C ABI over TensorRT 10's runtime, for src/tensorrt.rs, which
// loads this library at run time (dlopen): Rudolf-V itself links no TensorRT
// or CUDA and cross-compiles as before. Build it on the Jetson, against the
// TensorRT that runs there (build.sh).
//
// One engine per handle, fixed input shapes, float32 inputs and outputs. Each
// I/O tensor has a device buffer and a pinned host buffer: the caller fills
// the inputs' host buffers, rv_trt_run copies them over, runs the engine on
// the handle's own CUDA stream and copies the outputs back.
//
// An input can instead be made on the GPU from a u8 image (rv_trt_gpu_input:
// a separable resampling and a scaling, rv_trt_resize.cu): the caller writes
// the image into a pinned buffer, and only its bytes cross over.
//
// The first rv_trt_run runs eagerly, then records the same work (copies,
// resampling, engine) as a CUDA graph, which later runs replay: one launch
// instead of the engine's hundreds. On an Orin Nano that is 2.7 ms less CPU
// and 0.75 ms less GPU time a run for Light ESS, 8.9 ms less CPU for LAS2-S.
// If the recording fails, runs stay eager.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Logger : nvinfer1::ILogger {
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

// An input made on the GPU from a u8 image (rv_trt_gpu_input).
struct GpuInput {
    int width = 0, height = 0, nw = 0, nh = 0, ch = 0, top = 0;
    float scale = 1.0f, offset = 0.0f;
    uint8_t* host = nullptr;    // pinned, width × height
    uint8_t* device = nullptr;  // the same on the device
    int* spans = nullptr;       // device: nw column spans, then ch row spans
    float* weights = nullptr;   // device: the columns' weights, then the rows'
    size_t row_spans = 0, row_weights = 0;
};

struct Tensor {
    std::string name;
    bool input;
    nvinfer1::Dims dims;
    size_t bytes;
    void* device = nullptr;
    float* host = nullptr;
    std::unique_ptr<GpuInput> gpu;
};

void free_gpu_input(GpuInput& g) {
    if (g.host != nullptr) cudaFreeHost(g.host);
    if (g.device != nullptr) cudaFree(g.device);
    if (g.spans != nullptr) cudaFree(g.spans);
    if (g.weights != nullptr) cudaFree(g.weights);
}

void set_error(char* err, size_t len, std::string const& msg) {
    if (err != nullptr && len > 0) {
        std::snprintf(err, len, "%s", msg.c_str());
    }
}

bool cuda_ok(cudaError_t e, char const* what, char* err, size_t len) {
    if (e != cudaSuccess) {
        set_error(err, len, std::string(what) + ": " + cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace

cudaError_t rv_trt_resize(const uint8_t* img, int stride, const int* cols, const float* col_w,
                          const int* rows, const float* row_w, int nw, int nh, int ch, int top,
                          float scale, float offset, float* out, cudaStream_t stream);

struct RvTrt {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    std::vector<Tensor> tensors;
    // The recorded run, once there is one; `eager` after a failed recording.
    cudaGraphExec_t graph = nullptr;
    bool eager = false;

    void drop_graph() {
        if (graph != nullptr) cudaGraphExecDestroy(graph);
        graph = nullptr;
        eager = false;
    }

    ~RvTrt() {
        drop_graph();
        for (auto& t : tensors) {
            if (t.device != nullptr) cudaFree(t.device);
            if (t.host != nullptr) cudaFreeHost(t.host);
            if (t.gpu) free_gpu_input(*t.gpu);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
        context.reset();
        engine.reset();
        runtime.reset();
    }
};

namespace {

// One run's work on the handle's stream: inputs over (or made on the GPU),
// the engine, outputs back. 0 on success.
int record(RvTrt* t, char* err, size_t err_len) {
    for (auto& x : t->tensors) {
        if (!x.input) continue;
        if (x.gpu) {
            GpuInput const& g = *x.gpu;
            if (!cuda_ok(cudaMemcpyAsync(g.device, g.host, static_cast<size_t>(g.width) * g.height,
                                         cudaMemcpyHostToDevice, t->stream),
                         "image copy", err, err_len) ||
                !cuda_ok(rv_trt_resize(g.device, g.width, g.spans, g.weights, g.spans + g.row_spans,
                                       g.weights + g.row_weights, g.nw, g.nh, g.ch, g.top, g.scale,
                                       g.offset, static_cast<float*>(x.device), t->stream),
                         "resize", err, err_len))
                return 1;
        } else if (!cuda_ok(cudaMemcpyAsync(x.device, x.host, x.bytes, cudaMemcpyHostToDevice, t->stream),
                            "input copy", err, err_len)) {
            return 1;
        }
    }
    if (!t->context->enqueueV3(t->stream)) {
        set_error(err, err_len, "enqueueV3 failed");
        return 1;
    }
    for (auto& x : t->tensors) {
        if (!x.input && !cuda_ok(cudaMemcpyAsync(x.host, x.device, x.bytes, cudaMemcpyDeviceToHost, t->stream),
                                 "output copy", err, err_len))
            return 1;
    }
    return 0;
}

// `record` captured as a graph; on failure, runs stay eager.
void record_graph(RvTrt* t) {
    cudaGraph_t g = nullptr;
    if (cudaStreamBeginCapture(t->stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        t->eager = true;
        cudaGetLastError();
        return;
    }
    char ignored[256];
    int recorded = record(t, ignored, sizeof ignored);
    cudaError_t e = cudaStreamEndCapture(t->stream, &g);
    if (recorded == 0 && e == cudaSuccess) e = cudaGraphInstantiate(&t->graph, g, 0);
    if (g != nullptr) cudaGraphDestroy(g);
    if (recorded != 0 || e != cudaSuccess) {
        t->graph = nullptr;
        t->eager = true;
        cudaGetLastError();
    }
}

}  // namespace

extern "C" {

// The engine at `engine_path` (serialized for this GPU and TensorRT), after
// loading `n_plugins` plugin libraries; nullptr with a message in `err` on
// failure.
RvTrt* rv_trt_load(char const* engine_path, char const* const* plugins, int n_plugins, char* err,
                   size_t err_len) {
    auto t = std::make_unique<RvTrt>();
    initLibNvInferPlugins(&t->logger, "");
    t->runtime.reset(nvinfer1::createInferRuntime(t->logger));
    if (!t->runtime) {
        set_error(err, err_len, "createInferRuntime failed");
        return nullptr;
    }
    // Plugins that register their creators when loaded (REGISTER_TENSORRT_PLUGIN,
    // as Isaac ROS's ess_plugins.so) need only the load, as trtexec's
    // --staticPlugins; libraries that export their creators instead also go
    // through the registry. Never unloaded: the registry keeps their creators.
    for (int i = 0; i < n_plugins; i++) {
        void* lib = dlopen(plugins[i], RTLD_NOW | RTLD_GLOBAL);
        if (lib == nullptr) {
            set_error(err, err_len, std::string("cannot load plugin library ") + plugins[i] + ": " + dlerror());
            return nullptr;
        }
        bool exports = dlsym(lib, "getCreators") != nullptr || dlsym(lib, "getPluginCreators") != nullptr;
        if (exports && t->runtime->getPluginRegistry().loadLibrary(plugins[i]) == nullptr) {
            set_error(err, err_len, std::string("cannot register plugin library ") + plugins[i]);
            return nullptr;
        }
    }
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        set_error(err, err_len, std::string("cannot read ") + engine_path);
        return nullptr;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    t->engine.reset(t->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!t->engine) {
        set_error(err, err_len, std::string("cannot deserialize ") + engine_path +
                                    " (built for another GPU or TensorRT version?)");
        return nullptr;
    }
    t->context.reset(t->engine->createExecutionContext());
    if (!t->context) {
        set_error(err, err_len, "createExecutionContext failed");
        return nullptr;
    }
    if (!cuda_ok(cudaStreamCreate(&t->stream), "cudaStreamCreate", err, err_len)) return nullptr;
    for (int i = 0; i < t->engine->getNbIOTensors(); i++) {
        Tensor x;
        x.name = t->engine->getIOTensorName(i);
        x.input = t->engine->getTensorIOMode(x.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        x.dims = t->engine->getTensorShape(x.name.c_str());
        if (t->engine->getTensorDataType(x.name.c_str()) != nvinfer1::DataType::kFLOAT) {
            set_error(err, err_len, "tensor " + x.name + " is not float32");
            return nullptr;
        }
        size_t n = 1;
        for (int d = 0; d < x.dims.nbDims; d++) {
            if (x.dims.d[d] < 0) {
                set_error(err, err_len, "tensor " + x.name + " has a dynamic shape");
                return nullptr;
            }
            n *= static_cast<size_t>(x.dims.d[d]);
        }
        x.bytes = n * sizeof(float);
        t->tensors.push_back(std::move(x));
        Tensor& y = t->tensors.back();
        if (!cuda_ok(cudaMalloc(&y.device, y.bytes), "cudaMalloc", err, err_len)) return nullptr;
        if (!cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&y.host), y.bytes), "cudaMallocHost", err,
                     err_len))
            return nullptr;
        if (!t->context->setTensorAddress(y.name.c_str(), y.device)) {
            set_error(err, err_len, "setTensorAddress failed for " + y.name);
            return nullptr;
        }
    }
    return t.release();
}

int rv_trt_num_tensors(RvTrt const* t) { return static_cast<int>(t->tensors.size()); }

// Tensor i: its name, whether it is an input, its shape (`dims`, room for 8),
// and its pinned host buffer. Returns the number of dimensions, -1 if i is out
// of range.
int rv_trt_tensor(RvTrt const* t, int i, char const** name, int* is_input, int64_t* dims,
                  float** host) {
    if (i < 0 || i >= static_cast<int>(t->tensors.size())) return -1;
    Tensor const& x = t->tensors[i];
    *name = x.name.c_str();
    *is_input = x.input ? 1 : 0;
    for (int d = 0; d < x.dims.nbDims && d < 8; d++) dims[d] = x.dims.d[d];
    *host = x.host;
    return x.dims.nbDims;
}

// Make tensor `i` (an input, 1×3×nh×nw) on the GPU from a `width`×`height`
// u8 image, as src/stereo_net.rs's area_resize and scaling on the CPU: output
// column x averages image columns cols[3x] .. + cols[3x+1] with weights
// col_w[cols[3x+2] ..], and resized row y likewise from `rows` (`n_rows` of
// them, from tensor row `top`; the rows above and below repeat the first and
// last); then v·scale + offset, into all three channels. `*image` is the
// pinned buffer to write each image into, row-major, before rv_trt_run.
// 0 on success.
int rv_trt_gpu_input(RvTrt* t, int i, int width, int height, int const* cols,
                     float const* col_w, int n_col_w, int const* rows, int n_rows,
                     float const* row_w, int n_row_w, int top, float scale, float offset,
                     uint8_t** image, char* err, size_t err_len) {
    if (i < 0 || i >= static_cast<int>(t->tensors.size()) || !t->tensors[i].input) {
        set_error(err, err_len, "rv_trt_gpu_input: not an input tensor");
        return 1;
    }
    Tensor& x = t->tensors[i];
    if (x.dims.nbDims != 4 || x.dims.d[0] != 1 || x.dims.d[1] != 3) {
        set_error(err, err_len, "rv_trt_gpu_input: " + x.name + " is not 1x3xHxW");
        return 1;
    }
    int nh = static_cast<int>(x.dims.d[2]), nw = static_cast<int>(x.dims.d[3]);
    if (width <= 0 || height <= 0 || n_rows <= 0 || top < 0 || top + n_rows > nh) {
        set_error(err, err_len, "rv_trt_gpu_input: the resized rows do not fit " + x.name);
        return 1;
    }
    // Every span inside the image and its weights: the kernel trusts them.
    auto spans_ok = [](int const* s, int n, int size, int n_w) {
        for (int k = 0; k < n; k++) {
            int first = s[3 * k], count = s[3 * k + 1], off = s[3 * k + 2];
            if (first < 0 || count <= 0 || first + count > size || off < 0 || off + count > n_w)
                return false;
        }
        return true;
    };
    if (!spans_ok(cols, nw, width, n_col_w) || !spans_ok(rows, n_rows, height, n_row_w)) {
        set_error(err, err_len, "rv_trt_gpu_input: a span is outside the image or its weights");
        return 1;
    }
    auto g = std::make_unique<GpuInput>();
    *g = GpuInput{width, height, nw, nh, n_rows, top, scale, offset};
    size_t px = static_cast<size_t>(width) * height;
    size_t n_spans = 3 * static_cast<size_t>(nw + n_rows);
    size_t n_weights = static_cast<size_t>(n_col_w) + n_row_w;
    g->row_spans = 3 * static_cast<size_t>(nw);
    g->row_weights = n_col_w;
    auto alloc = [&](void* p, size_t bytes) {
        return cuda_ok(cudaMalloc(static_cast<void**>(p), bytes), "cudaMalloc", err, err_len);
    };
    auto copy = [&](void* dst, void const* src, size_t bytes) {
        return cuda_ok(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "table copy", err,
                       err_len);
    };
    bool ok = cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&g->host), px), "cudaMallocHost",
                      err, err_len) &&
              alloc(&g->device, px) && alloc(&g->spans, n_spans * sizeof(int)) &&
              alloc(&g->weights, n_weights * sizeof(float)) &&
              copy(g->spans, cols, g->row_spans * sizeof(int)) &&
              copy(g->spans + g->row_spans, rows, 3 * static_cast<size_t>(n_rows) * sizeof(int)) &&
              copy(g->weights, col_w, n_col_w * sizeof(float)) &&
              copy(g->weights + n_col_w, row_w, n_row_w * sizeof(float));
    if (!ok) {
        free_gpu_input(*g);
        return 1;
    }
    std::memset(g->host, 0, px);
    if (x.gpu) free_gpu_input(*x.gpu);
    x.gpu = std::move(g);
    *image = x.gpu->host;
    t->drop_graph();  // recorded with the input as it was
    return 0;
}

// Inputs to the device (or made there), the engine, outputs back; returns
// when they are in their host buffers. 0 on success.
int rv_trt_run(RvTrt* t, char* err, size_t err_len) {
    if (t->graph != nullptr) {
        if (!cuda_ok(cudaGraphLaunch(t->graph, t->stream), "graph launch", err, err_len)) return 1;
    } else {
        if (record(t, err, err_len) != 0) return 1;
    }
    if (!cuda_ok(cudaStreamSynchronize(t->stream), "run", err, err_len)) return 1;
    // After a first eager run (TensorRT's own setup is done by then).
    if (t->graph == nullptr && !t->eager) record_graph(t);
    return 0;
}

// Input tensor i's device buffer into its host buffer (to check an input made
// on the GPU). 0 on success.
int rv_trt_input_to_host(RvTrt* t, int i, char* err, size_t err_len) {
    if (i < 0 || i >= static_cast<int>(t->tensors.size()) || !t->tensors[i].input) {
        set_error(err, err_len, "rv_trt_input_to_host: not an input tensor");
        return 1;
    }
    Tensor& x = t->tensors[i];
    return cuda_ok(cudaMemcpy(x.host, x.device, x.bytes, cudaMemcpyDeviceToHost), "input copy", err,
                   err_len)
               ? 0
               : 1;
}

void rv_trt_free(RvTrt* t) { delete t; }

}  // extern "C"
