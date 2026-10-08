// The decode hyper-connection read (fused_gr_read_multi) at the real geometry, timed per call for 1..8 tokens (GPU,
// synthetic, no model).  Prints the variant this card's check chose, the time per call and a checksum of every
// output, so two builds can be compared for speed and for identical results.
//   hc_read_bench [reps=2000] [tokens=0 (all of 1..8)]
#include "strata/kernels/fused_gr.hpp"

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
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
std::vector<float> rnd(size_t n, float a, std::mt19937& g) {
    std::uniform_real_distribution<float> u(-a, a);
    std::vector<float> v(n);
    for (auto& x : v) x = u(g);
    return v;
}
std::vector<uint16_t> bf16(size_t n, float a, std::mt19937& g) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) {
        const float f = std::uniform_real_distribution<float>(-a, a)(g);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        x = (uint16_t) (b >> 16);
    }
    return v;
}
void mix(uint64_t& s, const float* d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "copy back");
    for (float f : h) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        s = (s ^ b) * 1099511628211ull;
    }
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 2000, only = argc > 2 ? std::atoi(argv[2]) : 0;
    constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, T = k::kFusedGrMaxT;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    k::fused_gr_check();
    std::printf("variant %d\n", k::fused_gr_variant());
    std::mt19937 g(31337);
    const uint16_t* w_down = up(bf16((size_t) LR * D, 0.02f, g));
    const uint16_t* w_up = up(bf16((size_t) D * LR, 0.05f, g));
    const uint16_t* w_inj = up(bf16((size_t) HC * D, 0.02f, g));
    const float* w_norm = up(rnd(D, 1.0f, g));
    float* xn = nullptr;
    ck(cudaMalloc(&xn, (size_t) T * D * 4), "malloc");
    std::vector<k::FusedGrArgs> a(T);
    for (int t = 0; t < T; ++t) {
        k::FusedGrArgs& x = a[t];
        x.R = up(rnd(D, 1.0f, g));
        x.R_out = up(std::vector<float>(D, 0.0f));
        x.apply = true;
        x.bo_prev = up(rnd(N, 1.0f, g));
        x.inj_prev = up(rnd(HC, 1.0f, g));
        x.w_norm = w_norm;
        x.w_down = w_down;
        x.w_up = w_up;
        x.w_inject = w_inj;
        x.lo = up(std::vector<float>(LR, 0.0f));
        x.rs = up(std::vector<float>(HC, 0.0f));
        x.inject_out = up(std::vector<float>(HC, 0.0f));
        x.mixed = up(std::vector<float>(N, 0.0f));
    }
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < 3000; ++r) k::fused_gr_read_multi(a.data(), only ? only : T, xn, st);   // the clocks up from idle
    ck(cudaStreamSynchronize(st), "warm-up");
    for (int n = 1; n <= T; ++n) {
        if (only && n != only) continue;
        // timed as the engine runs it: captured in a CUDA graph (the host-side launch cost is not the GPU's)
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int r = 0; r < 20; ++r) k::fused_gr_read_multi(a.data(), n, xn, st);
        ck(cudaStreamEndCapture(st, &gr), "capture end");
        ck(cudaGraphInstantiate(&ge, gr, 0), "instantiate");
        ck(cudaGraphLaunch(ge, st), "launch");
        ck(cudaEventRecord(e0, st), "record");
        for (int r = 0; r < reps / 20; ++r) ck(cudaGraphLaunch(ge, st), "launch");
        ck(cudaEventRecord(e1, st), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        uint64_t s = 1469598103934665603ull;
        for (int t = 0; t < n; ++t) {
            mix(s, a[t].lo, LR);
            mix(s, a[t].rs, HC);
            mix(s, a[t].inject_out, HC);
            mix(s, a[t].mixed, N);
            mix(s, a[t].R_out, D);
        }
        std::printf("tokens %d: %7.2f us per call, checksum %016llx\n", n, 1000.0 * ms / (reps / 20 * 20),
                    (unsigned long long) s);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
    }
    return 0;
}
