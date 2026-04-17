// Templated microbenchmark for four BN254 scalar-field Montgomery kernels:
//
// Each kernel runs in two fixtures:
//
//   Latency    : one dependent kernel call per benchmark iteration. The
//                output feeds the next iteration, so this measures
//                per-call latency.
//
//   Throughput : BATCH independent kernel calls per benchmark iteration.
//                Each call has its own storage, exposing ILP that the
//                latency chain hides.
//
// Only Fr is benchmarked. Fq uses the same Montgomery algorithms with
// different constants, so it would duplicate the signal without adding a
// new code path.
//
// `kWidth` controls the batch size, so one kernel call performs `kWidth`
// independent muls or sqrs.

#include "barretenberg/ecc/curves/bn254/bn254.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <string>

using Fr = bb::curve::BN254::ScalarField;

namespace {

// --- Operation tags -----------------------------------------------------------

template <typename F, size_t Width> BB_INLINE auto make_ptrs(std::array<F, Width>& values)
{
    std::array<F*, Width> ptrs{};
    for (size_t i = 0; i < Width; ++i) {
        ptrs[i] = &values[i];
    }
    return ptrs;
}

template <typename F, size_t Width> BB_INLINE auto make_const_ptrs(const std::array<F, Width>& values)
{
    std::array<const F*, Width> ptrs{};
    for (size_t i = 0; i < Width; ++i) {
        ptrs[i] = &values[i];
    }
    return ptrs;
}

template <typename F, size_t Width> struct BatchViews {
    std::array<const F*, Width> x_inputs;
    std::array<const F*, Width> y_inputs;
    std::array<F*, Width> x_outputs;
};

template <typename F, size_t Width>
BB_INLINE auto randomize_batch(std::array<F, Width>& x, std::array<F, Width>& y) -> BatchViews<F, Width>
{
    for (size_t i = 0; i < Width; ++i) {
        x[i] = F::random_element();
        y[i] = F::random_element();
    }
    return { make_const_ptrs(x), make_const_ptrs(y), make_ptrs(x) };
}

template <typename Op> std::string bench_name(const char* mode)
{
    std::string name = "Fr_";
    name += Op::kIdentifier;
    if constexpr (Op::kWidth != 1) {
        name += "Batch";
        name += std::to_string(Op::kWidth);
    }
    name += "_";
    name += mode;
    return name;
}

template <size_t Width> struct MulOp {
    static constexpr size_t kWidth = Width;
    static constexpr const char* kIdentifier = "Mul";

    template <typename F>
    BB_INLINE static void run(const std::array<const F*, Width>& x_inputs,
                              const std::array<const F*, Width>& y_inputs,
                              const std::array<F*, Width>& x_outputs)
    {
        F::template montgomery_mul_batched<Width>(x_inputs, y_inputs, x_outputs);
    }
};

template <size_t Width> struct SqrOp {
    static constexpr size_t kWidth = Width;
    static constexpr const char* kIdentifier = "Sqr";

    template <typename F>
    BB_INLINE static void run(const std::array<const F*, Width>& x_inputs,
                              const std::array<const F*, Width>& /*y_inputs*/,
                              const std::array<F*, Width>& x_outputs)
    {
        F::template montgomery_sqr_batched<Width>(x_inputs, x_outputs);
    }
};

// --- Fixtures ---------------------------------------------------------------

template <typename F, typename Op> void Latency(benchmark::State& state)
{
    std::array<F, Op::kWidth> x;
    std::array<F, Op::kWidth> y;
    const auto batch = randomize_batch(x, y);
    for (auto _ : state) {
        Op::template run<F>(batch.x_inputs, batch.y_inputs, batch.x_outputs);
        benchmark::DoNotOptimize(x);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * Op::kWidth);
}

template <typename F, typename Op> void Throughput(benchmark::State& state)
{
    constexpr size_t BATCH = 64;
    std::array<std::array<F, Op::kWidth>, BATCH> xs;
    std::array<std::array<F, Op::kWidth>, BATCH> ys;
    std::array<BatchViews<F, Op::kWidth>, BATCH> batches;
    for (size_t i = 0; i < BATCH; ++i) {
        batches[i] = randomize_batch(xs[i], ys[i]);
    }
    for (auto _ : state) {
        for (size_t i = 0; i < BATCH; ++i) {
            Op::template run<F>(batches[i].x_inputs, batches[i].y_inputs, batches[i].x_outputs);
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
    BENCHMARK_TEMPLATE(Latency, Fr, Op)                                                                                \
        ->Name(bench_name<Op>("Latency"))                                                                              \
        ->Iterations(1 << 20);                                                                                         \
    BENCHMARK_TEMPLATE(Throughput, Fr, Op)                                                                             \
        ->Name(bench_name<Op>("Throughput"))                                                                           \
        ->Iterations(1 << 16)

#define REGISTER_WIDTH(Width)                                                                                          \
    REGISTER_OP(MulOp<Width>);                                                                                         \
    REGISTER_OP(SqrOp<Width>)

REGISTER_WIDTH(1);
REGISTER_WIDTH(2);
REGISTER_WIDTH(3);
REGISTER_WIDTH(5);

#undef REGISTER_WIDTH
#undef REGISTER_OP

BENCHMARK_MAIN();
