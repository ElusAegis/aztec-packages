// Templated microbenchmark for four BN254 scalar-field Montgomery kernels:
//
//   Mul       : x = x * y                        (single montgomery_mul)
//   Sqr       : x = x^2                          (single montgomery_square)
//   MulPaired : (x1, x2) = (x1*y1, x2*y2)        (montgomery_mul_paired)
//   SqrPaired : (x1, x2) = (x1^2,  x2^2)         (montgomery_sqr_paired)
//
// Each kernel runs in two fixtures:
//
//   Latency    : one dependent kernel call per loop iteration. Each iter
//                reads the output of the previous iter, serialising the
//                kernel so we measure per-op latency.
//
//   Throughput : BATCH independent kernel calls per loop iteration. Each
//                kernel touches its own chunk of storage, so the CPU can
//                pipeline them — measures steady-state items/s and exposes
//                ILP that the latency chain hides.
//
// Only Fr (BN254 scalar) is benchmarked. Fq and Fr share the same montmul
// algorithm — only the modulus constants differ — so running both produces
// duplicate numbers and doubles bench wall-time for no new signal.
//
// `kWidth` on each op tag records how many field multiplications a single
// kernel call performs (1 for single, 2 for paired). Latency and throughput
// reports divide wall time by width to normalise everything to ns/mul, so
// the single-vs-paired speedup reads straight off the per-mul column.

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <string>

using Fr = bb::curve::BN254::ScalarField;

namespace {

// --- Operation tags -----------------------------------------------------------

struct MulOp {
    static constexpr size_t kWidth = 1;
    static constexpr const char* kName = "Mul";
    template <typename F> BB_INLINE static void run(F* x, const F* y) { x[0] *= y[0]; }
};

struct SqrOp {
    static constexpr size_t kWidth = 1;
    static constexpr const char* kName = "Sqr";
    template <typename F> BB_INLINE static void run(F* x, const F* /*y*/) { x[0].self_sqr(); }
};

struct MulPairedOp {
    static constexpr size_t kWidth = 2;
    static constexpr const char* kName = "MulPaired";
    template <typename F> BB_INLINE static void run(F* x, const F* y)
    {
        F::montgomery_mul_paired(x[0], y[0], x[1], y[1], x[0], x[1]);
    }
};

struct SqrPairedOp {
    static constexpr size_t kWidth = 2;
    static constexpr const char* kName = "SqrPaired";
    template <typename F> BB_INLINE static void run(F* x, const F* /*y*/)
    {
        F::montgomery_sqr_paired(x[0], x[1], x[0], x[1]);
    }
};

// --- Fixtures ---------------------------------------------------------------

template <typename F, typename Op> void Latency(benchmark::State& state)
{
    std::array<F, Op::kWidth> x;
    std::array<F, Op::kWidth> y;
    for (size_t i = 0; i < Op::kWidth; ++i) {
        x[i] = F::random_element();
        y[i] = F::random_element();
    }
    for (auto _ : state) {
        Op::template run<F>(x.data(), y.data());
        benchmark::DoNotOptimize(x);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * Op::kWidth);
}

template <typename F, typename Op> void Throughput(benchmark::State& state)
{
    constexpr size_t BATCH = 64;
    constexpr size_t N = BATCH * Op::kWidth;
    std::array<F, N> xs;
    std::array<F, N> ys;
    for (size_t i = 0; i < N; ++i) {
        xs[i] = F::random_element();
        ys[i] = F::random_element();
    }
    for (auto _ : state) {
        for (size_t i = 0; i < BATCH; ++i) {
            Op::template run<F>(&xs[i * Op::kWidth], &ys[i * Op::kWidth]);
        }
        benchmark::DoNotOptimize(xs);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * BATCH * Op::kWidth);
}

} // namespace

// --- Registration -------------------------------------------------------------
//
// Iterations are fixed at 1<<20 (latency) / 1<<16 (throughput) to match the
// previous baseline so historical tier-1 runs remain directly comparable.

#define REGISTER_OP(Op)                                                                                                \
    BENCHMARK_TEMPLATE(Latency, Fr, Op)->Name(std::string("Fr_") + Op::kName + "_Latency")->Iterations(1 << 20);       \
    BENCHMARK_TEMPLATE(Throughput, Fr, Op)->Name(std::string("Fr_") + Op::kName + "_Throughput")->Iterations(1 << 16)

REGISTER_OP(MulOp);
REGISTER_OP(SqrOp);
REGISTER_OP(MulPairedOp);
REGISTER_OP(SqrPairedOp);

#undef REGISTER_OP

BENCHMARK_MAIN();
