// The prompt path's MMQ expert products at the IQ3_XXS pack's real shapes, timed (GPU, synthetic, no model):
// one group of 16 experts, gate/up [1280 x 2560] in `gu_type` and down [2560 x 640] in `d_type`, `rows` activation
// rows per expert (a +4.4K-token turn has ~87).  Prints the time per product and a checksum of the outputs, so two
// builds or settings can be compared for speed and for identical results; run it under ncu for the kernel's limits.
//   prefill_mmq_bench [gu_type=18 (IQ3_XXS)] [d_type=42 (Q2_0)] [rows=87] [reps=20] [experts=16]
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// random quantized blocks with a finite, small fp16 scale at the start of every `block` bytes
std::vector<uint8_t> weights(size_t bytes, size_t block, std::mt19937& rng) {
    std::vector<uint8_t> w(bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o + 2 <= bytes; o += block) {
        const uint16_t h = (uint16_t) (0x1c00 + byte(rng));   // ~2^-8 .. 2^-7
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

uint64_t checksum(const float* d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "copy back");
    uint64_t s = 1469598103934665603ull;
    for (float f : h) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        s = (s ^ b) * 1099511628211ull;
    }
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    const int gu_t = argc > 1 ? std::atoi(argv[1]) : 18, d_t = argc > 2 ? std::atoi(argv[2]) : 42;
    const int rows = argc > 3 ? std::atoi(argv[3]) : 87, reps = argc > 4 ? std::atoi(argv[4]) : 20;
    const int G = argc > 5 ? std::atoi(argv[5]) : 16;
    constexpr int N = 2560, FF2 = 1280, FF = 640;
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    if (!mmq::supported(gu_t) || !mmq::supported(d_t)) { std::puts("SKIP: type not covered"); return 77; }
    std::mt19937 rng(424242);
    const size_t gub = mmq::matrix_bytes(gu_t, FF2, N), db = mmq::matrix_bytes(d_t, N, FF);
    auto qk = [](int t) { return t == 42 ? 64 : t == 20 || t == 8 ? 32 : 256; };   // values per block
    const size_t gu_blk = gub / ((size_t) FF2 * N / qk(gu_t)), d_blk = db / ((size_t) N * FF / qk(d_t));
    const auto wgu = weights(G * gub + 4096, gu_blk, rng), wd = weights(G * db + 4096, d_blk, rng);
    const int64_t T = (int64_t) G * rows;
    std::vector<float> x((size_t) T * N), h((size_t) T * FF);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    for (auto& v : h) v = nd(rng);
    std::vector<int32_t> bounds(G + 1), ids((size_t) T);
    for (int i = 0; i <= G; ++i) bounds[i] = i * rows;
    for (int64_t i = 0; i < T; ++i) ids[(size_t) i] = (int32_t) i;
    uint8_t *dgu, *dd;
    float *dx, *dh, *dGU, *dD;
    void *xq, *hq;
    int32_t *dbounds, *dids;
    ck(cudaMalloc(&dgu, wgu.size()), "malloc"); ck(cudaMalloc(&dd, wd.size()), "malloc");
    ck(cudaMalloc(&dx, x.size() * 4), "malloc"); ck(cudaMalloc(&dh, h.size() * 4), "malloc");
    ck(cudaMalloc(&dGU, (size_t) T * FF2 * 4), "malloc"); ck(cudaMalloc(&dD, (size_t) T * N * 4), "malloc");
    ck(cudaMalloc(&xq, mmq::q8_bytes(T, N)), "malloc"); ck(cudaMalloc(&hq, mmq::q8_bytes(T, FF)), "malloc");
    ck(cudaMalloc(&dbounds, bounds.size() * 4), "malloc"); ck(cudaMalloc(&dids, ids.size() * 4), "malloc");
    ck(cudaMemcpy(dgu, wgu.data(), wgu.size(), cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dd, wd.data(), wd.size(), cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dbounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    mmq::quantize(dx, nullptr, xq, gu_t, N, N, T, s);
    mmq::quantize(dh, nullptr, hq, d_t, FF, FF, T, s);
    mmq::Context ctx;
    mmq::Product gu;
    gu.w = dgu; gu.type = gu_t; gu.w_rows = FF2; gu.w_cols = N; gu.expert_bytes = gub; gu.n = G; gu.xq = xq;
    gu.bounds = dbounds; gu.ids = dids; gu.total_rows = T; gu.max_rows = rows; gu.dst = dGU; gu.ld_dst = FF2;
    mmq::Product dn;
    dn.w = dd; dn.type = d_t; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db; dn.n = G; dn.xq = hq;
    dn.bounds = dbounds; dn.ids = dids; dn.total_rows = T; dn.max_rows = rows; dn.dst = dD; dn.ld_dst = N;
    // warm-up: the card leaves its idle clocks only after a few hundred ms of work
    for (int r = 0; r < 4000 / G + 20; ++r) { ctx.run(gu, s); ctx.run(dn, s); }
    ck(cudaStreamSynchronize(s), "warm-up");
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e0, s);
    for (int r = 0; r < reps; ++r) ctx.run(gu, s);
    cudaEventRecord(e1, s);
    for (int r = 0; r < reps; ++r) ctx.run(dn, s);
    cudaEventRecord(e2, s);
    ck(cudaEventSynchronize(e2), "run");
    float t_gu = 0, t_d = 0;
    cudaEventElapsedTime(&t_gu, e0, e1);
    cudaEventElapsedTime(&t_d, e1, e2);
    const double fl_gu = 2.0 * T * FF2 * N, fl_d = 2.0 * T * N * FF;
    std::printf("types %d / %d, %d experts x %d rows: gate/up %.3f ms (%.1f TOPS), down %.3f ms (%.1f TOPS); "
                "checksums %016llx %016llx\n", gu_t, d_t, G, rows, t_gu / reps, fl_gu / (t_gu / reps * 1e9), t_d / reps,
                fl_d / (t_d / reps * 1e9), (unsigned long long) checksum(dGU, (size_t) T * FF2),
                (unsigned long long) checksum(dD, (size_t) T * N));
    return 0;
}
