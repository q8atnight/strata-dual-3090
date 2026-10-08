// The decode window's quantized GEMVs (native_mmvq, 1..8 columns) at the shapes a 2x CMP 50HX layer split runs,
// timed per call in a CUDA graph (GPU, synthetic, no model), with a checksum of the outputs, so two builds can be
// compared for speed and for identical results.
//   mmvq_bench [reps=200] [ncols=4]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Shape { const char* name; int type, n_in, n_out; };
// ggml types: Q4_K 12, Q5_K 13, Q6_K 14, IQ4_XS 23
const Shape kShapes[] = {
    {"head Q5_K", 13, 2560, 248320},
    {"Q6_K 2560->10240", 14, 2560, 10240},
    {"Q6_K 2560->12288", 14, 2560, 12288},
    {"Q6_K 6144->2560", 14, 6144, 2560},
    {"IQ4_XS 2560->10240", 23, 2560, 10240},
    {"IQ4_XS 2560->6144", 23, 2560, 6144},
    {"Q4_K 2560->10240", 12, 2560, 10240},
    {"Q5_K 2560->2560", 13, 2560, 2560},
};
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200, ncols = argc > 2 ? std::atoi(argv[2]) : 4;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    std::mt19937 g(777);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const Shape& s : kShapes) {
        if (!k::native_mmvq_supported(s.type)) continue;
        // bytes below 0x3c: every fp16 scale finite and below 1, the quants still varied
        const size_t wb = k::native_mmvq_weight_bytes(s.type, s.n_in, s.n_out);
        std::vector<uint8_t> hw(wb);
        std::uniform_int_distribution<int> byte(0, 0x3b);
        for (auto& b : hw) b = (uint8_t) byte(g);
        std::vector<float> hx((size_t) s.n_in * ncols);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto& x : hx) x = u(g);
        void *w, *xq;
        float *x, *y;
        ck(cudaMalloc(&w, wb), "malloc");
        ck(cudaMalloc(&x, hx.size() * 4), "malloc");
        ck(cudaMalloc(&xq, k::native_q8_1_bytes(s.n_in, ncols)), "malloc");
        ck(cudaMalloc(&y, (size_t) s.n_out * ncols * 4), "malloc");
        ck(cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "upload");
        k::native_quantize_q8_1(x, xq, s.n_in, ncols, st);
        for (int r = 0; r < 50; ++r) k::native_mmvq(s.type, w, xq, y, s.n_in, s.n_out, ncols, st);   // clocks up
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int r = 0; r < 10; ++r) k::native_mmvq(s.type, w, xq, y, s.n_in, s.n_out, ncols, st);
        ck(cudaStreamEndCapture(st, &gr), "capture end");
        ck(cudaGraphInstantiate(&ge, gr, 0), "instantiate");
        ck(cudaGraphLaunch(ge, st), "launch");
        ck(cudaEventRecord(e0, st), "record");
        for (int r = 0; r < reps / 10; ++r) ck(cudaGraphLaunch(ge, st), "launch");
        ck(cudaEventRecord(e1, st), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / (reps / 10 * 10);
        std::vector<float> hy((size_t) s.n_out * ncols);
        ck(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost), "copy back");
        uint64_t sum = 1469598103934665603ull;
        for (float f : hy) {
            uint32_t b;
            std::memcpy(&b, &f, 4);
            sum = (sum ^ b) * 1099511628211ull;
        }
        std::printf("%-20s cols %d: %8.2f us per call, %6.1f GB/s, checksum %016llx\n", s.name, ncols, us,
                    wb / (us * 1e3), (unsigned long long) sum);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
        cudaFree(w); cudaFree(x); cudaFree(xq); cudaFree(y);
    }
    return 0;
}
