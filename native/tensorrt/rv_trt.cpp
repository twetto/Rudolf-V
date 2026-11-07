// rv_trt.cpp — A C ABI over TensorRT 10's runtime, for src/tensorrt.rs, which
// loads this library at run time (dlopen): Rudolf-V itself links no TensorRT
// or CUDA and cross-compiles as before. Build it on the Jetson, against the
// TensorRT that runs there (build.sh).
//
// One engine per handle, fixed input shapes, float32 inputs and outputs. Each
// I/O tensor has a device buffer and a pinned host buffer: the caller fills
// the inputs' host buffers, rv_trt_run copies them over, runs the engine on
// the handle's own CUDA stream and copies the outputs back.

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
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

struct Tensor {
    std::string name;
    bool input;
    nvinfer1::Dims dims;
    size_t bytes;
    void* device = nullptr;
    float* host = nullptr;
};

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

struct RvTrt {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    std::vector<Tensor> tensors;

    ~RvTrt() {
        for (auto& t : tensors) {
            if (t.device != nullptr) cudaFree(t.device);
            if (t.host != nullptr) cudaFreeHost(t.host);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
        context.reset();
        engine.reset();
        runtime.reset();
    }
};

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
        t->tensors.push_back(x);
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

// Inputs' host buffers to the device, the engine, outputs back; returns when
// they are in their host buffers. 0 on success.
int rv_trt_run(RvTrt* t, char* err, size_t err_len) {
    for (auto& x : t->tensors) {
        if (x.input && !cuda_ok(cudaMemcpyAsync(x.device, x.host, x.bytes, cudaMemcpyHostToDevice, t->stream),
                                "input copy", err, err_len))
            return 1;
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
    return cuda_ok(cudaStreamSynchronize(t->stream), "run", err, err_len) ? 0 : 1;
}

void rv_trt_free(RvTrt* t) { delete t; }

}  // extern "C"
