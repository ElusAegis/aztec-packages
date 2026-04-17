#pragma once

// WASM 24-bit FMA + SIMD Montgomery backend.
//
// Uses 11 × 24-bit limbs with exact f64 arithmetic (24×24 = 48 < 53-bit
// mantissa). R = 2^264 (11 × 24-bit reduction steps). Requires
// -msimd128 -mrelaxed-simd.
//
// Entry points:
//   mul          — single mul, routed through mul_batched<1>
//   sqr          — single sqr, routed through sqr_batched<1>
//   mul_batched  — N independent muls at once
//   sqr_batched  — N independent sqrs at once
//   mul_big      — single big-modulus mul, routed to int29 R=264 companion
//   wide_mul     — raw 256×256 → 512 integer multiply (R-independent)

#if BB_R_LIMB_BITS == 24 && defined(__wasm_simd128__)

#include <array>
#include <cmath>
#include <type_traits>
#include <wasm_simd128.h>

#include "../field_constexpr_helpers.hpp"
#include "../field_declarations.hpp"
#include "../field_montgomery_config.hpp"
#include "wasm_int29.hpp"

namespace bb::detail {

template <class Params> struct WasmFmaBackend {
    // Pure-integer Montgomery kernel at R = 2^264, used as the constexpr
    // fallback, for the N=1 mul/sqr path, and for the odd-tail sqr case.
    // Reuses the 9×29-bit int29 schoolbook kernel but retargets its final
    // reduction step to R = 2^264 so outputs are interchangeable with the
    // paired FMA kernel's outputs.
    using IntCompanion = WasmInt29Backend<Params, 264>;

    // ── Public MontBackend contract ──────────────────────────────────────

    BB_INLINE static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        field<Params> out;
        mul_batched<1>({ &lhs }, { &rhs }, { &out });
        return out;
    }

    BB_INLINE static constexpr field<Params> sqr(const field<Params>& x) noexcept
    {
        field<Params> out;
        sqr_batched<1>({ &x }, { &out });
        return out;
    }


    // Big-modulus delegation to int29's REXP=264 companion. Same R as the
    // paired FMA kernel, so outputs are interchangeable.
    BB_INLINE static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        return IntCompanion::mul_big(lhs, rhs);
    }

    BB_INLINE static constexpr typename field<Params>::wide_array wide_mul(const field<Params>& lhs,
                                                                           const field<Params>& rhs) noexcept
    {
        // R-independent: raw integer 256×256 → 512 multiply. WasmInt29's
        // implementation uses 29-bit splitting internally, but that's an
        // implementation detail — the output bits are the same for any R.
        return WasmInt29Backend<Params>::wide_mul(lhs, rhs);
    }

    // Batched Montgomery mul: outs[i] = as[i] * bs[i] for i in [0, N).
    //
    // Routing:
    //   constexpr → int29-R264 companion (constexpr-compatible, same R as FMA).
    //   N = 1     → int29-R264 companion. A single mul can't amortize the
    //               paired-FMA scheduling, so the integer kernel wins.
    //   N ≥ 2     → paired FMA across every full pair. If N is odd, the tail
    //               also runs through the paired FMA kernel with a duplicated
    //               operand (lane-1 result discarded). Benchmarks show the
    //               paired kernel has higher throughput than the int29 kernel
    //               even with one lane wasted, so the duplicate beats routing
    //               the tail through IntCompanion.
    template <size_t N>
    BB_INLINE static constexpr void mul_batched(std::array<const field<Params>*, N> as,
                                                std::array<const field<Params>*, N> bs,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        if (std::is_constant_evaluated()) {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = IntCompanion::mul(*as[i], *bs[i]);
            }
            return;
        }
        if constexpr (N == 0) {
            return;
        } else if constexpr (N == 1) {
            *outs[0] = IntCompanion::mul(*as[0], *bs[0]);
        } else {
            for (size_t i = 0; i + 1 < N; i += 2) {
                mul_paired_fma_simd(*as[i], *bs[i], *as[i + 1], *bs[i + 1], *outs[i], *outs[i + 1]);
            }
            if constexpr ((N % 2) == 1) {
                field<Params> scratch;
                mul_paired_fma_simd(
                    *as[N - 1], *bs[N - 1], *as[N - 1], *bs[N - 1], *outs[N - 1], scratch);
            }
        }
    }

    // Batched Montgomery sqr: outs[i] = as[i]^2 for i in [0, N).
    //
    // Routing:
    //   constexpr → int29-R264 companion (constexpr-compatible, same R as FMA).
    //   N = 1     → int29-R264 companion.
    //   N ≥ 2     → paired FMA across every full pair. If N is odd, the tail
    //               runs through the int29 companion — for a lone square the
    //               integer kernel matches or beats the "paired with duplicate"
    //               trick (empirical).
    template <size_t N>
    BB_INLINE static constexpr void sqr_batched(std::array<const field<Params>*, N> as,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        if (std::is_constant_evaluated()) {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = IntCompanion::sqr(*as[i]);
            }
            return;
        }
        if constexpr (N == 0) {
            return;
        } else if constexpr (N == 1) {
            *outs[0] = IntCompanion::sqr(*as[0]);
        } else {
            for (size_t i = 0; i + 1 < N; i += 2) {
                sqr_paired_fma_simd(*as[i], *as[i + 1], *outs[i], *outs[i + 1]);
            }
            if constexpr ((N % 2) == 1) {
                *outs[N - 1] = IntCompanion::sqr(*as[N - 1]);
            }
        }
    }

  private:
    // Paired f64x2 FMA kernels — one Montgomery reduction covers two
    // independent inputs (lane 0 and lane 1). Inlining is left to the
    // compiler's heuristics.
    static void mul_paired_fma_simd(const field<Params>& a1,
                                    const field<Params>& b1,
                                    const field<Params>& a2,
                                    const field<Params>& b2,
                                    field<Params>& out1,
                                    field<Params>& out2) noexcept;
    static void sqr_paired_fma_simd(const field<Params>& a1,
                                    const field<Params>& a2,
                                    field<Params>& out1,
                                    field<Params>& out2) noexcept;

    // Shared Phase 3 (reduction) + Phase 4 (extract & pack) for both paired
    // kernels. The kernel-specific product phase populates t[0..20] (two
    // 11-limb lanes worth of pre-reduced f64 values); this helper performs
    // 10 Yuval steps + 1 standard step of Montgomery reduction, then integer
    // carry-propagates each lane back into a 4×64-bit field<Params>.
    // BB_INLINE so it collapses into the two paired kernels.
    BB_INLINE static void reduce_and_finalize_paired(v128_t* t, field<Params>& out1, field<Params>& out2) noexcept;
};

// ═════════════════════════════════════════════════════════════════════════
// Dual Montgomery multiplication: out1 = a1*b1, out2 = a2*b2.
//
// Both f64x2 lanes carry independent multiplications from start to finish
// (lane 0 = first mul, lane 1 = second). This is the only mul kernel —
// odd-tail batches call it with a duplicated operand on lane 1.
//
// Point arithmetic (e.g., mixed addition) naturally has pairs of
// independent field muls, making this the preferred entry point.
// ═════════════════════════════════════════════════════════════════════════

template <class Params>
void WasmFmaBackend<Params>::mul_paired_fma_simd(const field<Params>& a1,
                                                 const field<Params>& b1,
                                                 const field<Params>& a2,
                                                 const field<Params>& b2,
                                                 field<Params>& out1,
                                                 field<Params>& out2) noexcept
{
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    // Phase 1: Convert — lane 0 = (a1,b1), lane 1 = (a2,b2)
#define LIMB24(D, IDX, SHIFT) static_cast<double>((D[IDX] >> SHIFT) & M24)
#define LIMB24_CROSS(D, I0, S0, I1, S1) static_cast<double>(((D[I0] >> S0) | (D[I1] << S1)) & M24)
#define LIMB24_TOP(D, IDX, SHIFT) static_cast<double>(D[IDX] >> SHIFT)

    v128_t af0 = wasm_f64x2_make(LIMB24(a1.data, 0, 0), LIMB24(a2.data, 0, 0));
    v128_t af1 = wasm_f64x2_make(LIMB24(a1.data, 0, 24), LIMB24(a2.data, 0, 24));
    v128_t af2 = wasm_f64x2_make(LIMB24_CROSS(a1.data, 0, 48, 1, 16), LIMB24_CROSS(a2.data, 0, 48, 1, 16));
    v128_t af3 = wasm_f64x2_make(LIMB24(a1.data, 1, 8), LIMB24(a2.data, 1, 8));
    v128_t af4 = wasm_f64x2_make(LIMB24(a1.data, 1, 32), LIMB24(a2.data, 1, 32));
    v128_t af5 = wasm_f64x2_make(LIMB24_CROSS(a1.data, 1, 56, 2, 8), LIMB24_CROSS(a2.data, 1, 56, 2, 8));
    v128_t af6 = wasm_f64x2_make(LIMB24(a1.data, 2, 16), LIMB24(a2.data, 2, 16));
    v128_t af7 = wasm_f64x2_make(LIMB24(a1.data, 2, 40), LIMB24(a2.data, 2, 40));
    v128_t af8 = wasm_f64x2_make(LIMB24(a1.data, 3, 0), LIMB24(a2.data, 3, 0));
    v128_t af9 = wasm_f64x2_make(LIMB24(a1.data, 3, 24), LIMB24(a2.data, 3, 24));
    v128_t af10 = wasm_f64x2_make(LIMB24_TOP(a1.data, 3, 48), LIMB24_TOP(a2.data, 3, 48));

    v128_t bf0 = wasm_f64x2_make(LIMB24(b1.data, 0, 0), LIMB24(b2.data, 0, 0));
    v128_t bf1 = wasm_f64x2_make(LIMB24(b1.data, 0, 24), LIMB24(b2.data, 0, 24));
    v128_t bf2 = wasm_f64x2_make(LIMB24_CROSS(b1.data, 0, 48, 1, 16), LIMB24_CROSS(b2.data, 0, 48, 1, 16));
    v128_t bf3 = wasm_f64x2_make(LIMB24(b1.data, 1, 8), LIMB24(b2.data, 1, 8));
    v128_t bf4 = wasm_f64x2_make(LIMB24(b1.data, 1, 32), LIMB24(b2.data, 1, 32));
    v128_t bf5 = wasm_f64x2_make(LIMB24_CROSS(b1.data, 1, 56, 2, 8), LIMB24_CROSS(b2.data, 1, 56, 2, 8));
    v128_t bf6 = wasm_f64x2_make(LIMB24(b1.data, 2, 16), LIMB24(b2.data, 2, 16));
    v128_t bf7 = wasm_f64x2_make(LIMB24(b1.data, 2, 40), LIMB24(b2.data, 2, 40));
    v128_t bf8 = wasm_f64x2_make(LIMB24(b1.data, 3, 0), LIMB24(b2.data, 3, 0));
    v128_t bf9 = wasm_f64x2_make(LIMB24(b1.data, 3, 24), LIMB24(b2.data, 3, 24));
    v128_t bf10 = wasm_f64x2_make(LIMB24_TOP(b1.data, 3, 48), LIMB24_TOP(b2.data, 3, 48));

#undef LIMB24
#undef LIMB24_CROSS
#undef LIMB24_TOP

    // Phase 2: Karatsuba product (all v128_t)
    v128_t sl0 = wasm_f64x2_add(af0, af6);
    v128_t sr0 = wasm_f64x2_add(bf0, bf6);
    v128_t sl1 = wasm_f64x2_add(af1, af7);
    v128_t sr1 = wasm_f64x2_add(bf1, bf7);
    v128_t sl2 = wasm_f64x2_add(af2, af8);
    v128_t sr2 = wasm_f64x2_add(bf2, bf8);
    v128_t sl3 = wasm_f64x2_add(af3, af9);
    v128_t sr3 = wasm_f64x2_add(bf3, bf9);
    v128_t sl4 = wasm_f64x2_add(af4, af10);
    v128_t sr4 = wasm_f64x2_add(bf4, bf10);
    v128_t sl5 = af5;
    v128_t sr5 = bf5;

    v128_t pl0 = wasm_f64x2_mul(af0, bf0);
    v128_t pl1 = fma_v(af1, bf0, wasm_f64x2_mul(af0, bf1));
    v128_t pl2 = fma_v(af2, bf0, fma_v(af1, bf1, wasm_f64x2_mul(af0, bf2)));
    v128_t pl3 = fma_v(af3, bf0, fma_v(af2, bf1, fma_v(af1, bf2, wasm_f64x2_mul(af0, bf3))));
    v128_t pl4 = fma_v(af4, bf0, fma_v(af3, bf1, fma_v(af2, bf2, fma_v(af1, bf3, wasm_f64x2_mul(af0, bf4)))));
    v128_t pl5 =
        fma_v(af5, bf0, fma_v(af4, bf1, fma_v(af3, bf2, fma_v(af2, bf3, fma_v(af1, bf4, wasm_f64x2_mul(af0, bf5))))));
    v128_t pl6 = fma_v(af5, bf1, fma_v(af4, bf2, fma_v(af3, bf3, fma_v(af2, bf4, wasm_f64x2_mul(af1, bf5)))));
    v128_t pl7 = fma_v(af5, bf2, fma_v(af4, bf3, fma_v(af3, bf4, wasm_f64x2_mul(af2, bf5))));
    v128_t pl8 = fma_v(af5, bf3, fma_v(af4, bf4, wasm_f64x2_mul(af3, bf5)));
    v128_t pl9 = fma_v(af5, bf4, wasm_f64x2_mul(af4, bf5));
    v128_t pl10 = wasm_f64x2_mul(af5, bf5);

    v128_t ph0 = wasm_f64x2_mul(af6, bf6);
    v128_t ph1 = fma_v(af7, bf6, wasm_f64x2_mul(af6, bf7));
    v128_t ph2 = fma_v(af8, bf6, fma_v(af7, bf7, wasm_f64x2_mul(af6, bf8)));
    v128_t ph3 = fma_v(af9, bf6, fma_v(af8, bf7, fma_v(af7, bf8, wasm_f64x2_mul(af6, bf9))));
    v128_t ph4 = fma_v(af10, bf6, fma_v(af9, bf7, fma_v(af8, bf8, fma_v(af7, bf9, wasm_f64x2_mul(af6, bf10)))));
    v128_t ph5 = fma_v(af10, bf7, fma_v(af9, bf8, fma_v(af8, bf9, wasm_f64x2_mul(af7, bf10))));
    v128_t ph6 = fma_v(af10, bf8, fma_v(af9, bf9, wasm_f64x2_mul(af8, bf10)));
    v128_t ph7 = fma_v(af10, bf9, wasm_f64x2_mul(af9, bf10));
    v128_t ph8 = wasm_f64x2_mul(af10, bf10);

    v128_t pc0 = wasm_f64x2_mul(sl0, sr0);
    v128_t pc1 = fma_v(sl1, sr0, wasm_f64x2_mul(sl0, sr1));
    v128_t pc2 = fma_v(sl2, sr0, fma_v(sl1, sr1, wasm_f64x2_mul(sl0, sr2)));
    v128_t pc3 = fma_v(sl3, sr0, fma_v(sl2, sr1, fma_v(sl1, sr2, wasm_f64x2_mul(sl0, sr3))));
    v128_t pc4 = fma_v(sl4, sr0, fma_v(sl3, sr1, fma_v(sl2, sr2, fma_v(sl1, sr3, wasm_f64x2_mul(sl0, sr4)))));
    v128_t pc5 =
        fma_v(sl5, sr0, fma_v(sl4, sr1, fma_v(sl3, sr2, fma_v(sl2, sr3, fma_v(sl1, sr4, wasm_f64x2_mul(sl0, sr5))))));
    v128_t pc6 = fma_v(sl5, sr1, fma_v(sl4, sr2, fma_v(sl3, sr3, fma_v(sl2, sr4, wasm_f64x2_mul(sl1, sr5)))));
    v128_t pc7 = fma_v(sl5, sr2, fma_v(sl4, sr3, fma_v(sl3, sr4, wasm_f64x2_mul(sl2, sr5))));
    v128_t pc8 = fma_v(sl5, sr3, fma_v(sl4, sr4, wasm_f64x2_mul(sl3, sr5)));
    v128_t pc9 = fma_v(sl5, sr4, wasm_f64x2_mul(sl4, sr5));
    v128_t pc10 = wasm_f64x2_mul(sl5, sr5);

    // Combine Karatsuba sub-products into a 21-limb t-array (both lanes
    // populated with pre-reduction limbs) and inject the Phase 3 bias as a
    // post-combine add per t[j]. Baking bias into leaf products is unsafe
    // here: the combine subtracts pl_j/ph_j with opposite signs across
    // t[j]/t[j+6] so constants leak to the wrong destinations, and the
    // pc[5] chain (6×≤2^50 terms) pushes intermediates past 2^53.
    const v128_t bias_v = wasm_f64x2_splat(0x1p52);
    const v128_t bias_minus_t1_comp_v = wasm_f64x2_splat(0x1p52 - 0x1p28);

    v128_t t[21];
    t[0] = wasm_f64x2_add(pl0, bias_v);
    t[1] = wasm_f64x2_add(pl1, bias_minus_t1_comp_v);
    t[2] = wasm_f64x2_add(pl2, bias_minus_t1_comp_v);
    t[3] = wasm_f64x2_add(pl3, bias_minus_t1_comp_v);
    t[4] = wasm_f64x2_add(pl4, bias_minus_t1_comp_v);
    t[5] = wasm_f64x2_add(pl5, bias_minus_t1_comp_v);
    t[6] = wasm_f64x2_add(bias_minus_t1_comp_v,
                          wasm_f64x2_add(pl6, wasm_f64x2_sub(pc0, wasm_f64x2_add(pl0, ph0))));
    t[7] = wasm_f64x2_add(bias_minus_t1_comp_v,
                          wasm_f64x2_add(pl7, wasm_f64x2_sub(pc1, wasm_f64x2_add(pl1, ph1))));
    t[8] = wasm_f64x2_add(bias_minus_t1_comp_v,
                          wasm_f64x2_add(pl8, wasm_f64x2_sub(pc2, wasm_f64x2_add(pl2, ph2))));
    t[9] = wasm_f64x2_add(bias_minus_t1_comp_v,
                          wasm_f64x2_add(pl9, wasm_f64x2_sub(pc3, wasm_f64x2_add(pl3, ph3))));
    t[10] = wasm_f64x2_add(bias_minus_t1_comp_v,
                           wasm_f64x2_add(pl10, wasm_f64x2_sub(pc4, wasm_f64x2_add(pl4, ph4))));
    t[11] = wasm_f64x2_add(bias_v, wasm_f64x2_sub(pc5, wasm_f64x2_add(pl5, ph5)));
    t[12] = wasm_f64x2_add(bias_v,
                           wasm_f64x2_add(wasm_f64x2_sub(pc6, wasm_f64x2_add(pl6, ph6)), ph0));
    t[13] = wasm_f64x2_add(bias_v,
                           wasm_f64x2_add(wasm_f64x2_sub(pc7, wasm_f64x2_add(pl7, ph7)), ph1));
    t[14] = wasm_f64x2_add(bias_v,
                           wasm_f64x2_add(wasm_f64x2_sub(pc8, wasm_f64x2_add(pl8, ph8)), ph2));
    t[15] = wasm_f64x2_add(bias_v, wasm_f64x2_add(wasm_f64x2_sub(pc9, pl9), ph3));
    t[16] = wasm_f64x2_add(bias_v, wasm_f64x2_add(wasm_f64x2_sub(pc10, pl10), ph4));
    t[17] = wasm_f64x2_add(ph5, bias_v);
    t[18] = wasm_f64x2_add(ph6, bias_v);
    t[19] = wasm_f64x2_add(ph7, bias_v);
    t[20] = wasm_f64x2_add(ph8, bias_v);

    // Phases 3 + 4: shared Montgomery reduction and output extraction.
    reduce_and_finalize_paired(t, out1, out2);
}

// ═════════════════════════════════════════════════════════════════════════
// Dual Montgomery squaring: out1 = a1², out2 = a2².
//
// Flat 11-limb schoolbook triangular square. Karatsuba's combine overhead
// (~30 adds + sum limbs) isn't worth it at n=11 for a square — the extra
// 9 mul/FMAs schoolbook pays are cheaper than the combine it avoids.
// Per lane: 66 mul/FMA + 10 pre-doubles vs. 57 + 49 for Karatsuba 6+5.
//
// Phases 3 + 4 delegate to reduce_and_finalize_paired.
// ═════════════════════════════════════════════════════════════════════════

template <class Params>
void WasmFmaBackend<Params>::sqr_paired_fma_simd(const field<Params>& a1,
                                                 const field<Params>& a2,
                                                 field<Params>& out1,
                                                 field<Params>& out2) noexcept
{
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    // Phase 1: Unpack a1/a2 into 11×24-bit limbs (lane 0 = a1, lane 1 = a2).
#define LIMB24(D, IDX, SHIFT) static_cast<double>((D[IDX] >> SHIFT) & M24)
#define LIMB24_CROSS(D, I0, S0, I1, S1) static_cast<double>(((D[I0] >> S0) | (D[I1] << S1)) & M24)
#define LIMB24_TOP(D, IDX, SHIFT) static_cast<double>(D[IDX] >> SHIFT)

    v128_t af0 = wasm_f64x2_make(LIMB24(a1.data, 0, 0), LIMB24(a2.data, 0, 0));
    v128_t af1 = wasm_f64x2_make(LIMB24(a1.data, 0, 24), LIMB24(a2.data, 0, 24));
    v128_t af2 = wasm_f64x2_make(LIMB24_CROSS(a1.data, 0, 48, 1, 16), LIMB24_CROSS(a2.data, 0, 48, 1, 16));
    v128_t af3 = wasm_f64x2_make(LIMB24(a1.data, 1, 8), LIMB24(a2.data, 1, 8));
    v128_t af4 = wasm_f64x2_make(LIMB24(a1.data, 1, 32), LIMB24(a2.data, 1, 32));
    v128_t af5 = wasm_f64x2_make(LIMB24_CROSS(a1.data, 1, 56, 2, 8), LIMB24_CROSS(a2.data, 1, 56, 2, 8));
    v128_t af6 = wasm_f64x2_make(LIMB24(a1.data, 2, 16), LIMB24(a2.data, 2, 16));
    v128_t af7 = wasm_f64x2_make(LIMB24(a1.data, 2, 40), LIMB24(a2.data, 2, 40));
    v128_t af8 = wasm_f64x2_make(LIMB24(a1.data, 3, 0), LIMB24(a2.data, 3, 0));
    v128_t af9 = wasm_f64x2_make(LIMB24(a1.data, 3, 24), LIMB24(a2.data, 3, 24));
    v128_t af10 = wasm_f64x2_make(LIMB24_TOP(a1.data, 3, 48), LIMB24_TOP(a2.data, 3, 48));

#undef LIMB24
#undef LIMB24_CROSS
#undef LIMB24_TOP

    // Phase 2: Schoolbook triangular square. Pre-double a[0..9] so each
    // output limb is a single FMA chain (no post-multiply x+x step).
    // a[10] is always the larger index — no doubled form needed.
    // Worst-case limb (t[10]): 5 cross + 1 diag ≈ 2^51.6 (f64-safe).
    //
    // Init bias (see reduce_and_finalize_paired) is folded into the innermost
    // mul of every chain, mul → fma(..., bias_addend), at zero extra op cost.
    // Longest chain (t[10]): 6 products of ≤ 2^49 → 2^52 + 6·2^49 < 2^53,
    // so every intermediate stays at exp 1075 (ULP=1, integer-exact).
    const v128_t bias_v = wasm_f64x2_splat(0x1p52);
    const v128_t bias_minus_t1_comp_v = wasm_f64x2_splat(0x1p52 - 0x1p28);

    v128_t a0x2 = wasm_f64x2_add(af0, af0);
    v128_t a1x2 = wasm_f64x2_add(af1, af1);
    v128_t a2x2 = wasm_f64x2_add(af2, af2);
    v128_t a3x2 = wasm_f64x2_add(af3, af3);
    v128_t a4x2 = wasm_f64x2_add(af4, af4);
    v128_t a5x2 = wasm_f64x2_add(af5, af5);
    v128_t a6x2 = wasm_f64x2_add(af6, af6);
    v128_t a7x2 = wasm_f64x2_add(af7, af7);
    v128_t a8x2 = wasm_f64x2_add(af8, af8);
    v128_t a9x2 = wasm_f64x2_add(af9, af9);

    v128_t t[21];
    // t[0]: init bias +2^52 (no T_1 role ever).
    t[0] = fma_v(af0, af0, bias_v);
    // t[1..10]: init bias +2^52 − 2^28 (compensation for T_1-receiver role).
    t[1] = fma_v(a0x2, af1, bias_minus_t1_comp_v);
    t[2] = fma_v(a0x2, af2, fma_v(af1, af1, bias_minus_t1_comp_v));
    t[3] = fma_v(a0x2, af3, fma_v(a1x2, af2, bias_minus_t1_comp_v));
    t[4] = fma_v(a0x2, af4, fma_v(a1x2, af3, fma_v(af2, af2, bias_minus_t1_comp_v)));
    t[5] = fma_v(a0x2, af5, fma_v(a1x2, af4, fma_v(a2x2, af3, bias_minus_t1_comp_v)));
    t[6] = fma_v(a0x2, af6, fma_v(a1x2, af5, fma_v(a2x2, af4, fma_v(af3, af3, bias_minus_t1_comp_v))));
    t[7] = fma_v(a0x2, af7, fma_v(a1x2, af6, fma_v(a2x2, af5, fma_v(a3x2, af4, bias_minus_t1_comp_v))));
    t[8] = fma_v(a0x2, af8, fma_v(a1x2, af7, fma_v(a2x2, af6, fma_v(a3x2, af5, fma_v(af4, af4, bias_minus_t1_comp_v)))));
    t[9] = fma_v(a0x2, af9, fma_v(a1x2, af8, fma_v(a2x2, af7, fma_v(a3x2, af6, fma_v(a4x2, af5, bias_minus_t1_comp_v)))));
    t[10] = fma_v(a0x2,
                  af10,
                  fma_v(a1x2,
                        af9,
                        fma_v(a2x2, af8, fma_v(a3x2, af7, fma_v(a4x2, af6, fma_v(af5, af5, bias_minus_t1_comp_v))))));
    // t[11..20]: init bias +2^52 (pure sink slots in Phase 3).
    t[11] = fma_v(a1x2, af10, fma_v(a2x2, af9, fma_v(a3x2, af8, fma_v(a4x2, af7, fma_v(a5x2, af6, bias_v)))));
    t[12] = fma_v(a2x2, af10, fma_v(a3x2, af9, fma_v(a4x2, af8, fma_v(a5x2, af7, fma_v(af6, af6, bias_v)))));
    t[13] = fma_v(a3x2, af10, fma_v(a4x2, af9, fma_v(a5x2, af8, fma_v(a6x2, af7, bias_v))));
    t[14] = fma_v(a4x2, af10, fma_v(a5x2, af9, fma_v(a6x2, af8, fma_v(af7, af7, bias_v))));
    t[15] = fma_v(a5x2, af10, fma_v(a6x2, af9, fma_v(a7x2, af8, bias_v)));
    t[16] = fma_v(a6x2, af10, fma_v(a7x2, af9, fma_v(af8, af8, bias_v)));
    t[17] = fma_v(a7x2, af10, fma_v(a8x2, af9, bias_v));
    t[18] = fma_v(a8x2, af10, fma_v(af9, af9, bias_v));
    t[19] = fma_v(a9x2, af10, bias_v);
    t[20] = fma_v(af10, af10, bias_v);

    // Phases 3 + 4: shared Montgomery reduction and output extraction.
    reduce_and_finalize_paired(t, out1, out2);
}

// ═════════════════════════════════════════════════════════════════════════
// Shared Phase 3 + Phase 4 for both paired kernels.
//
// Input: t[0..20] holding two independent 11-limb pre-reduction values
// (lane 0 and lane 1). Each limb is an f64 in [~-2^53, ~+2^53).
//
// Performs 10 Yuval reduction steps + 1 standard step (dividing by
// R = 2^264), then extracts each lane, propagates integer carries across
// 24-bit limbs, and packs back into a 4×64-bit field<Params>.
// ═════════════════════════════════════════════════════════════════════════

template <class Params>
void WasmFmaBackend<Params>::reduce_and_finalize_paired(v128_t* t, field<Params>& out1, field<Params>& out2) noexcept
{
    constexpr auto modulus = field<Params>::modulus;
    constexpr auto r_limbs = field<Params>::r_limbs;
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;
    constexpr double NP0_F = static_cast<double>(compute_r_inv(modulus.data[0]) & M24);

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    // AND-mask bias trick: each t[j] enters Phase 3 biased by +2^52, so its
    // f64 encoding pins exp=1075 and the mantissa's low 24 bits encode ki,
    // bits 24–51 encode qi·2^24. Two parallel ANDs extract both halves off
    // TI in the time of one, replacing mul→floor→fma with AND→sub→fma.
    //
    // The bias halves Phase 3's f64 headroom: exactness requires every FMA
    // intermediate (product + addend) to stay < 2^53. Only BN254 Fq and
    // BN254 Fr (= Grumpkin Fq) have a completed bounds proof (step-8
    // scatter peaks at ≈ 2^52.93–2^52.94, ≈ 0.06 bits of slack). Other
    // curves are gated out below: secp256k1 Fq's upper div_r_inv_f limbs
    // are near-saturated and drive the step-8 scatter past 2^53, and the
    // secp256k1 Fr / secp256r1 Fq/Fr margins have not been re-verified
    // against this kernel's actual FMA schedule.
    //
    // TODO(bounds): widen the whitelist once each additional curve has an
    // end-to-end per-step bounds proof. Currently proven: Bn254FqParams, Bn254FrParams
    const v128_t sd = wasm_f64x2_splat(0x1p-24);                 // 2^{-R_LIMB_BITS}
    const v128_t bias_v = wasm_f64x2_splat(0x1p52);              // 2^52
    const v128_t neg_t1_comp_v = wasm_f64x2_splat(-0x1p28);      // −2^28
    const v128_t mask_ki = wasm_i64x2_splat(static_cast<int64_t>(0x4330000000FFFFFFULL));        // exp(1075) | low-24 mantissa
    const v128_t mask_qi_scaled = wasm_i64x2_splat(static_cast<int64_t>(0x433FFFFFFF000000ULL)); // exp(1075) | mantissa bits 24–51

    const v128_t rinv0 = wasm_f64x2_splat(r_limbs.div_r_inv_f[0]);
    const v128_t rinv1 = wasm_f64x2_splat(r_limbs.div_r_inv_f[1]);
    const v128_t rinv2 = wasm_f64x2_splat(r_limbs.div_r_inv_f[2]);
    const v128_t rinv3 = wasm_f64x2_splat(r_limbs.div_r_inv_f[3]);
    const v128_t rinv4 = wasm_f64x2_splat(r_limbs.div_r_inv_f[4]);
    const v128_t rinv5 = wasm_f64x2_splat(r_limbs.div_r_inv_f[5]);
    const v128_t rinv6 = wasm_f64x2_splat(r_limbs.div_r_inv_f[6]);
    const v128_t rinv7 = wasm_f64x2_splat(r_limbs.div_r_inv_f[7]);
    const v128_t rinv8 = wasm_f64x2_splat(r_limbs.div_r_inv_f[8]);
    const v128_t rinv9 = wasm_f64x2_splat(r_limbs.div_r_inv_f[9]);
    const v128_t rinv10 = wasm_f64x2_splat(r_limbs.div_r_inv_f[10]);

    // Phase 3: Reduction — 10 Yuval + 1 standard (all v128_t).
    // The qi-FMA adds qi + 2^28 to T_1 (not just +qi). Each t[j] for j ∈
    // [1,10] plays the T_1 role in exactly one step, so its init bias
    // pre-subtracts 2^28; the running bias settles back to +2^52 before
    // t[j] enters its own Yuval step.
    v128_t ki, ki_biased, qi_scaled_biased;

#define YUVAL_STEP_V(TI, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)                                                 \
    ki_biased = wasm_v128_and(TI, mask_ki);                /* 2^52 + ki             */                                 \
    qi_scaled_biased = wasm_v128_and(TI, mask_qi_scaled);  /* 2^52 + qi·2^24        */                                 \
    ki = wasm_f64x2_sub(ki_biased, bias_v);                                                                            \
    T1 = fma_v(qi_scaled_biased, sd, T1);                  /* adds qi + 2^28        */                                 \
    T1 = fma_v(ki, rinv0, T1);                                                                                         \
    T2 = fma_v(ki, rinv1, T2);                                                                                         \
    T3 = fma_v(ki, rinv2, T3);                                                                                         \
    T4 = fma_v(ki, rinv3, T4);                                                                                         \
    T5 = fma_v(ki, rinv4, T5);                                                                                         \
    T6 = fma_v(ki, rinv5, T6);                                                                                         \
    T7 = fma_v(ki, rinv6, T7);                                                                                         \
    T8 = fma_v(ki, rinv7, T8);                                                                                         \
    T9 = fma_v(ki, rinv8, T9);                                                                                         \
    T10 = fma_v(ki, rinv9, T10);                                                                                       \
    T11 = fma_v(ki, rinv10, T11);

    YUVAL_STEP_V(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11])
    YUVAL_STEP_V(t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12])
    YUVAL_STEP_V(t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13])
    YUVAL_STEP_V(t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14])
    YUVAL_STEP_V(t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15])
    YUVAL_STEP_V(t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16])
    YUVAL_STEP_V(t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17])
    YUVAL_STEP_V(t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18])
    YUVAL_STEP_V(t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18], t[19])
    YUVAL_STEP_V(t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18], t[19], t[20])

#undef YUVAL_STEP_V

    // Standard step 10
    {
        // mod0 is only needed in scaled form (mod0_sd) for the carry computation.
        const v128_t mod0_sd = wasm_f64x2_splat(r_limbs.modulus_f[0] * 0x1p-24);
        const v128_t mod1 = wasm_f64x2_splat(r_limbs.modulus_f[1]);
        const v128_t mod2 = wasm_f64x2_splat(r_limbs.modulus_f[2]);
        const v128_t mod3 = wasm_f64x2_splat(r_limbs.modulus_f[3]);
        const v128_t mod4 = wasm_f64x2_splat(r_limbs.modulus_f[4]);
        const v128_t mod5 = wasm_f64x2_splat(r_limbs.modulus_f[5]);
        const v128_t mod6 = wasm_f64x2_splat(r_limbs.modulus_f[6]);
        const v128_t mod7 = wasm_f64x2_splat(r_limbs.modulus_f[7]);
        const v128_t mod8 = wasm_f64x2_splat(r_limbs.modulus_f[8]);
        const v128_t mod9 = wasm_f64x2_splat(r_limbs.modulus_f[9]);
        const v128_t mod10 = wasm_f64x2_splat(r_limbs.modulus_f[10]);
        const v128_t np0 = wasm_f64x2_splat(NP0_F);

        // AND-mask extraction mirrors the Yuval steps. t[10] carries +2^52
        // here (step 9's T_1 FMA restored the +2^28). fma(t[10], sd, −2^28)
        // yields clean t10_sd; k_product gets re-biased for a second AND.
        v128_t t10_sd = fma_v(t[10], sd, neg_t1_comp_v);
        v128_t k_base_biased = wasm_v128_and(t[10], mask_ki);
        v128_t k_base = wasm_f64x2_sub(k_base_biased, bias_v);
        v128_t k_product_biased = fma_v(k_base, np0, bias_v);
        v128_t ks_biased = wasm_v128_and(k_product_biased, mask_ki);
        v128_t ks = wasm_f64x2_sub(ks_biased, bias_v);

        // The standard step chooses ks so that t[10] + ks·p[0] ≡ 0 mod 2^24.
        // Therefore (t[10] + ks·p[0]) / 2^24 is already an integer — no floor
        // needed. And t[10] itself is dead after this (Phase 4 starts at t[11]),
        // so we compute the carry directly without updating t[10]:
        //   carry = t[10]/2^24 + ks · (p[0]/2^24) = fma(ks, mod0_sd, t10_sd)
        v128_t carry_10 = fma_v(ks, mod0_sd, t10_sd);
        t[11] = fma_v(ks, mod1, wasm_f64x2_add(t[11], carry_10));
        t[12] = fma_v(ks, mod2, t[12]);
        t[13] = fma_v(ks, mod3, t[13]);
        t[14] = fma_v(ks, mod4, t[14]);
        t[15] = fma_v(ks, mod5, t[15]);
        t[16] = fma_v(ks, mod6, t[16]);
        t[17] = fma_v(ks, mod7, t[17]);
        t[18] = fma_v(ks, mod8, t[18]);
        t[19] = fma_v(ks, mod9, t[19]);
        t[20] = fma_v(ks, mod10, t[20]);
    }

    // Phase 4: Extract lanes, integer carry propagation, output. Each t[j]
    // for j ∈ [11,20] still carries +2^52; strip it after the f64→int64.
    constexpr int64_t BIAS_I64 = 1LL << 52;
#define EXTRACT_AND_FINALIZE(LANE, OUT)                                                                                \
    {                                                                                                                  \
        uint64_t r11 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[11], LANE)) - BIAS_I64);   \
        uint64_t r12 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[12], LANE)) - BIAS_I64);   \
        uint64_t r13 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[13], LANE)) - BIAS_I64);   \
        uint64_t r14 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[14], LANE)) - BIAS_I64);   \
        uint64_t r15 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[15], LANE)) - BIAS_I64);   \
        uint64_t r16 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[16], LANE)) - BIAS_I64);   \
        uint64_t r17 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[17], LANE)) - BIAS_I64);   \
        uint64_t r18 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[18], LANE)) - BIAS_I64);   \
        uint64_t r19 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[19], LANE)) - BIAS_I64);   \
        uint64_t r20 = static_cast<uint64_t>(static_cast<int64_t>(wasm_f64x2_extract_lane(t[20], LANE)) - BIAS_I64);   \
        r12 += r11 >> 24;                                                                                              \
        r11 &= M24;                                                                                                    \
        r13 += r12 >> 24;                                                                                              \
        r12 &= M24;                                                                                                    \
        r14 += r13 >> 24;                                                                                              \
        r13 &= M24;                                                                                                    \
        r15 += r14 >> 24;                                                                                              \
        r14 &= M24;                                                                                                    \
        r16 += r15 >> 24;                                                                                              \
        r15 &= M24;                                                                                                    \
        r17 += r16 >> 24;                                                                                              \
        r16 &= M24;                                                                                                    \
        r18 += r17 >> 24;                                                                                              \
        r17 &= M24;                                                                                                    \
        r19 += r18 >> 24;                                                                                              \
        r18 &= M24;                                                                                                    \
        r20 += r19 >> 24;                                                                                              \
        r19 &= M24;                                                                                                    \
        OUT = field<Params>{ r11 | (r12 << 24) | (r13 << 48),                                                          \
                             (r13 >> 16) | (r14 << 8) | (r15 << 32) | (r16 << 56),                                     \
                             (r16 >> 8) | (r17 << 16) | (r18 << 40),                                                   \
                             r19 | (r20 << 24) };                                                                      \
    }

    EXTRACT_AND_FINALIZE(0, out1)
    EXTRACT_AND_FINALIZE(1, out2)

#undef EXTRACT_AND_FINALIZE
}

} // namespace bb::detail

#endif // BB_R_LIMB_BITS == 24 && __wasm_simd128__
