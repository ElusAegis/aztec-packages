#pragma once

// WASM 24-bit FMA + SIMD Montgomery backend.
//
// Uses 11 × 24-bit limbs with exact f64 arithmetic (24×24 = 48 < 53-bit
// mantissa). R = 2^264 (11 × 24-bit reduction steps). Requires
// -msimd128 -mrelaxed-simd.
//
// Entry points:
//   mul          — single mul, SIMD within (P_lo/P_cross parallel)
//   sqr          — dedicated triangular-product square, SIMD via paired kernel
//   mul_batched  — N independent muls at once (N=2 routes through the paired
//                  SIMD kernel; other N fall back to paired + singles)
//   sqr_batched  — N independent sqrs at once, symmetric to mul_batched
//
// ╔══════════════════════════════════════════════════════════════════════╗
// ║  ⚠  CORRECTNESS WARNING — FMA mul_big delegation                      ║
// ╠══════════════════════════════════════════════════════════════════════╣
// ║  FMA uses R = 2^264 (11 × 24-bit limbs).                              ║
// ║  mul_big delegates to ConstexprFallback::mul, which derives           ║
// ║  R^{-1} from the platform's R_EXPONENT (= 264 for FMA). Correct but   ║
// ║  O(256²) per multiply. Acceptable as a stopgap; a dedicated FMA       ║
// ║  big-modulus kernel is future work.                                   ║
// ║                                                                       ║
// ║  wide_mul IS safe to delegate to WasmInt29: it's a raw integer        ║
// ║  256×256 → 512 multiply — no Montgomery reduction, so R doesn't       ║
// ║  enter the computation.                                               ║
// ╚══════════════════════════════════════════════════════════════════════╝

#if BB_R_LIMB_BITS == 24 && defined(__wasm_simd128__)

#include <array>
#include <cmath>
#include <type_traits>
#include <wasm_simd128.h>

#include "../field_constexpr_helpers.hpp"
#include "../field_declarations.hpp"
#include "../field_montgomery_config.hpp"
#include "constexpr_fallback.hpp"
#include "wasm_int29.hpp"

namespace bb::detail {

template <class Params> struct WasmFmaBackend {
    // Pure-integer Montgomery kernel at R = 2^264, co-scheduled alongside the
    // paired f64x2 FMA kernel inside mul_batched<3>/<5> and sqr_batched<3>/<5>.
    // Reuses the 9×29-bit int29 schoolbook path (81 muls per mul vs. 121 for
    // the previous 11×24 int kernel) but keeps the FMA-compatible R so the
    // integer output can share a field<Params> value frame with the FMA lanes.
    using IntCompanion = WasmInt29Backend<Params, 264>;

    // ── Public MontBackend contract ──────────────────────────────────────

    BB_INLINE static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        if (std::is_constant_evaluated()) {
            return ConstexprFallback::mul(lhs, rhs);
        }
        return mul_via_paired_fma_simd(lhs, rhs);
    }

    BB_INLINE static constexpr field<Params> sqr(const field<Params>& x) noexcept
    {
        if (std::is_constant_evaluated()) {
            return ConstexprFallback::mul(x, x);
        }
        // Dedicated triangular-product FMA squaring kernel. Standalone sqr
        // still pays for the paired kernel and discards one lane — same shape
        // as the standalone mul entry point — but the kernel itself is ~41%
        // cheaper than mul thanks to the a·b = b·a symmetry, so every
        // .sqr() call benefits (especially the 255-sqr chain in invert()).
        field<Params> out1;
        field<Params> out2;
        sqr_paired_fma_simd(x, x, out1, out2);
        return out1;
    }

    // ⚠ WARNING: FMA R=2^264 vs WasmInt29 R=2^261 — cannot delegate here.
    //            See big banner comment at top of file.
    BB_INLINE static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        // ConstexprFallback uses R_EXPONENT (= 264 for FMA builds) — R-aware,
        // bit-correct. Slow (O(256²)), but FMA has no dedicated big-mul kernel.
        return ConstexprFallback::mul(lhs, rhs);
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
    //   N=1  → single-lane mul (SIMD within the kernel).
    //   N=2  → mul_paired_fma_simd — 2 SIMD lanes, 1 kernel call.
    //   N=3  → noinline helper that force-inlines both mul_paired_fma_simd
    //          and the int29-R264 companion mul into one function body. LLVM
    //          can then interleave the i64 ops with f64 FMA ops. The noinline
    //          boundary keeps the fat body out of every caller.
    //   N=5  → noinline helper with 2 paired FMAs + 1 int29-R264 mul, all
    //          inlined. Integer sandwiched between the two FMA calls so it
    //          can hide behind the combined FMA latency.
    //   N≥4 (except 5) → paired FMA for [0,1] + single mul() for [2..N).
    template <size_t N>
    BB_INLINE static constexpr void mul_batched(std::array<const field<Params>*, N> as,
                                                std::array<const field<Params>*, N> bs,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        if (std::is_constant_evaluated()) {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = ConstexprFallback::mul(*as[i], *bs[i]);
            }
            return;
        }
        if constexpr (N == 0) {
            return;
        } else if constexpr (N == 1) {
            *outs[0] = mul_via_paired_fma_simd(*as[0], *bs[0]);
        } else if constexpr (N == 3) {
            mul_triple_noinline(*as[0], *bs[0], *as[1], *bs[1], *as[2], *bs[2], *outs[0], *outs[1], *outs[2]);
        } else if constexpr (N == 5) {
            mul_quint_noinline(*as[0],
                               *bs[0],
                               *as[1],
                               *bs[1],
                               *as[2],
                               *bs[2],
                               *as[3],
                               *bs[3],
                               *as[4],
                               *bs[4],
                               *outs[0],
                               *outs[1],
                               *outs[2],
                               *outs[3],
                               *outs[4]);
        } else {
            mul_paired_fma_simd(*as[0], *bs[0], *as[1], *bs[1], *outs[0], *outs[1]);
            for (size_t i = 2; i < N; ++i) {
                *outs[i] = mul_via_paired_fma_simd(*as[i], *bs[i]);
            }
        }
    }

    // Batched Montgomery sqr: outs[i] = as[i]^2 for i in [0, N).
    //
    // N=2 uses the triangular-product paired kernel.
    // N=3 delegates to a noinline helper with both kernels force-inlined.
    // N=5 delegates to a noinline helper with 2 paired sqrs + 1 int29-R264 sqr.
    template <size_t N>
    BB_INLINE static constexpr void sqr_batched(std::array<const field<Params>*, N> as,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        if (std::is_constant_evaluated()) {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = ConstexprFallback::mul(*as[i], *as[i]);
            }
            return;
        }
        if constexpr (N == 0) {
            return;
        } else if constexpr (N == 1) {
            field<Params> scratch;
            sqr_paired_fma_simd(*as[0], *as[0], *outs[0], scratch);
        } else if constexpr (N == 3) {
            sqr_triple_noinline(*as[0], *as[1], *as[2], *outs[0], *outs[1], *outs[2]);
        } else if constexpr (N == 5) {
            sqr_quint_noinline(
                *as[0], *as[1], *as[2], *as[3], *as[4], *outs[0], *outs[1], *outs[2], *outs[3], *outs[4]);
        } else {
            sqr_paired_fma_simd(*as[0], *as[1], *outs[0], *outs[1]);
            for (size_t i = 2; i < N; ++i) {
                field<Params> scratch;
                sqr_paired_fma_simd(*as[i], *as[i], *outs[i], scratch);
            }
        }
    }

  private:
    static field<Params> mul_fma_simd(const field<Params>& lhs, const field<Params>& rhs) noexcept;

    // BB_INLINE _impl kernels: the raw kernel bodies. These inline into
    // whichever noinline wrapper calls them.
    BB_INLINE static void mul_paired_fma_simd_impl(const field<Params>& a1,
                                                   const field<Params>& b1,
                                                   const field<Params>& a2,
                                                   const field<Params>& b2,
                                                   field<Params>& out1,
                                                   field<Params>& out2) noexcept;
    BB_INLINE static void sqr_paired_fma_simd_impl(const field<Params>& a1,
                                                   const field<Params>& a2,
                                                   field<Params>& out1,
                                                   field<Params>& out2) noexcept;

    // Noinline wrappers around individual kernels. Called by N=1 and N=2
    // paths. Compact call site — the kernel body lives in exactly one
    // place in the WASM binary.
    __attribute__((noinline)) static void mul_paired_fma_simd(const field<Params>& a1,
                                                              const field<Params>& b1,
                                                              const field<Params>& a2,
                                                              const field<Params>& b2,
                                                              field<Params>& out1,
                                                              field<Params>& out2) noexcept
    {
        mul_paired_fma_simd_impl(a1, b1, a2, b2, out1, out2);
    }

    __attribute__((noinline)) static void sqr_paired_fma_simd(const field<Params>& a1,
                                                              const field<Params>& a2,
                                                              field<Params>& out1,
                                                              field<Params>& out2) noexcept
    {
        sqr_paired_fma_simd_impl(a1, a2, out1, out2);
    }

    // Noinline wrappers for N=3 co-scheduling. The paired _impl kernel is
    // BB_INLINE and the int29-R264 companion is BB_INLINE constexpr, so LLVM
    // merges their bodies into this one function. The scheduler can then
    // interleave i64 ops (IntCompanion::mul) with f64 FMA ops
    // (mul_paired_fma_simd) in the same basic block.
    __attribute__((noinline)) static void mul_triple_noinline(const field<Params>& a0,
                                                              const field<Params>& b0,
                                                              const field<Params>& a1,
                                                              const field<Params>& b1,
                                                              const field<Params>& a2,
                                                              const field<Params>& b2,
                                                              field<Params>& o0,
                                                              field<Params>& o1,
                                                              field<Params>& o2) noexcept
    {
        o2 = IntCompanion::mul(a2, b2);
        mul_paired_fma_simd_impl(a0, b0, a1, b1, o0, o1);
    }

    __attribute__((noinline)) static void sqr_triple_noinline(const field<Params>& a0,
                                                              const field<Params>& a1,
                                                              const field<Params>& a2,
                                                              field<Params>& o0,
                                                              field<Params>& o1,
                                                              field<Params>& o2) noexcept
    {
        o2 = IntCompanion::sqr(a2);
        sqr_paired_fma_simd_impl(a0, a1, o0, o1);
    }

    // N=5: two paired FMAs + one integer, all inlined. Integer sandwiched
    // between the two FMA calls so its i64 ops can hide in the gap.
    __attribute__((noinline)) static void mul_quint_noinline(const field<Params>& a0,
                                                             const field<Params>& b0,
                                                             const field<Params>& a1,
                                                             const field<Params>& b1,
                                                             const field<Params>& a2,
                                                             const field<Params>& b2,
                                                             const field<Params>& a3,
                                                             const field<Params>& b3,
                                                             const field<Params>& a4,
                                                             const field<Params>& b4,
                                                             field<Params>& o0,
                                                             field<Params>& o1,
                                                             field<Params>& o2,
                                                             field<Params>& o3,
                                                             field<Params>& o4) noexcept
    {
        mul_paired_fma_simd_impl(a0, b0, a1, b1, o0, o1);
        o2 = IntCompanion::mul(a2, b2);
        mul_paired_fma_simd_impl(a3, b3, a4, b4, o3, o4);
    }

    __attribute__((noinline)) static void sqr_quint_noinline(const field<Params>& a0,
                                                             const field<Params>& a1,
                                                             const field<Params>& a2,
                                                             const field<Params>& a3,
                                                             const field<Params>& a4,
                                                             field<Params>& o0,
                                                             field<Params>& o1,
                                                             field<Params>& o2,
                                                             field<Params>& o3,
                                                             field<Params>& o4) noexcept
    {
        sqr_paired_fma_simd_impl(a0, a1, o0, o1);
        o2 = IntCompanion::sqr(a2);
        sqr_paired_fma_simd_impl(a3, a4, o3, o4);
    }

    // Shared Phase 3 (reduction) + Phase 4 (extract & pack) for both
    // mul_paired_fma_simd_impl and sqr_paired_fma_simd_impl. The kernel-specific product
    // phase populates t[0..20] (two 11-limb lanes worth of pre-reduced f64
    // values); this helper performs 10 Yuval steps + 1 standard step of
    // Montgomery reduction, then integer carry-propagates each lane back into
    // a 4×64-bit field<Params>. BB_INLINE so it collapses into the caller.
    BB_INLINE static void reduce_and_finalize_paired(v128_t* t, field<Params>& out1, field<Params>& out2) noexcept;

    BB_INLINE static field<Params> mul_via_paired_fma_simd(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        field<Params> out1;
        field<Params> out2;
        mul_paired_fma_simd(lhs, rhs, lhs, rhs, out1, out2);
        return out1;
    }
};

// ═════════════════════════════════════════════════════════════════════════
// Single Montgomery multiplication: lhs * rhs mod p.
//
// Uses 11 × 24-bit limbs in f64 with Karatsuba 6+5 split.
// SIMD is used _within_ the single multiplication to extract parallelism:
//   - Product phase: P_lo and P_cross (both 6×6) computed in parallel via
//     f64x2 lanes; P_hi (5×5) is scalar (only 5 limbs, no SIMD partner).
//   - Reduction phase: consecutive Yuval scatter FMAs paired into f64x2.
// 10 Yuval steps + 1 standard step = division by 2^264 = R. Output in [0, 2p).
// ═════════════════════════════════════════════════════════════════════════

template <class Params>
field<Params> WasmFmaBackend<Params>::mul_fma_simd(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    constexpr auto modulus = field<Params>::modulus;
    constexpr auto r_limbs = field<Params>::r_limbs;
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;
    constexpr double SD = 0x1p-24; // 2^{-R_LIMB_BITS}
    constexpr double SU = 0x1p24;  // 2^{R_LIMB_BITS}
    // -(p^{-1}) mod 2^{R_LIMB_BITS}, for the standard reduction step.
    constexpr double NP0_F = static_cast<double>(compute_r_inv(modulus.data[0]) & M24);

    auto fma_s = [](double a, double b, double c) -> double {
        v128_t r = __builtin_wasm_relaxed_madd_f64x2(
            wasm_f64x2_make(a, 0.0), wasm_f64x2_make(b, 0.0), wasm_f64x2_make(c, 0.0));
        return wasm_f64x2_extract_lane(r, 0);
    };

    // ================================================================
    // Phase 1: Convert 4x64 -> 11x24 -> double
    // ================================================================
    const uint64_t* a = lhs.data;
    const uint64_t* b = rhs.data;

    double af0 = static_cast<double>(a[0] & M24);
    double af1 = static_cast<double>((a[0] >> 24) & M24);
    double af2 = static_cast<double>(((a[0] >> 48) | (a[1] << 16)) & M24);
    double af3 = static_cast<double>((a[1] >> 8) & M24);
    double af4 = static_cast<double>((a[1] >> 32) & M24);
    double af5 = static_cast<double>(((a[1] >> 56) | (a[2] << 8)) & M24);
    double af6 = static_cast<double>((a[2] >> 16) & M24);
    double af7 = static_cast<double>((a[2] >> 40) & M24);
    double af8 = static_cast<double>(a[3] & M24);
    double af9 = static_cast<double>((a[3] >> 24) & M24);
    double af10 = static_cast<double>(a[3] >> 48);

    double bf0 = static_cast<double>(b[0] & M24);
    double bf1 = static_cast<double>((b[0] >> 24) & M24);
    double bf2 = static_cast<double>(((b[0] >> 48) | (b[1] << 16)) & M24);
    double bf3 = static_cast<double>((b[1] >> 8) & M24);
    double bf4 = static_cast<double>((b[1] >> 32) & M24);
    double bf5 = static_cast<double>(((b[1] >> 56) | (b[2] << 8)) & M24);
    double bf6 = static_cast<double>((b[2] >> 16) & M24);
    double bf7 = static_cast<double>((b[2] >> 40) & M24);
    double bf8 = static_cast<double>(b[3] & M24);
    double bf9 = static_cast<double>((b[3] >> 24) & M24);
    double bf10 = static_cast<double>(b[3] >> 48);

    // ================================================================
    // Phase 2: Karatsuba product with SIMD
    // P_lo and P_cross share the same 6x6 structure.
    // lane 0 = P_lo, lane 1 = P_cross.
    // ================================================================
    double sl0 = af0 + af6;
    double sr0 = bf0 + bf6;
    double sl1 = af1 + af7;
    double sr1 = bf1 + bf7;
    double sl2 = af2 + af8;
    double sr2 = bf2 + bf8;
    double sl3 = af3 + af9;
    double sr3 = bf3 + bf9;
    double sl4 = af4 + af10;
    double sr4 = bf4 + bf10;
    double sl5 = af5;
    double sr5 = bf5;

    v128_t als0 = wasm_f64x2_make(af0, sl0);
    v128_t als1 = wasm_f64x2_make(af1, sl1);
    v128_t als2 = wasm_f64x2_make(af2, sl2);
    v128_t als3 = wasm_f64x2_make(af3, sl3);
    v128_t als4 = wasm_f64x2_make(af4, sl4);
    v128_t als5 = wasm_f64x2_make(af5, sl5);

    v128_t bsr0 = wasm_f64x2_make(bf0, sr0);
    v128_t bsr1 = wasm_f64x2_make(bf1, sr1);
    v128_t bsr2 = wasm_f64x2_make(bf2, sr2);
    v128_t bsr3 = wasm_f64x2_make(bf3, sr3);
    v128_t bsr4 = wasm_f64x2_make(bf4, sr4);
    v128_t bsr5 = wasm_f64x2_make(bf5, sr5);

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    v128_t plpc0 = wasm_f64x2_mul(als0, bsr0);
    v128_t plpc1 = fma_v(als1, bsr0, wasm_f64x2_mul(als0, bsr1));
    v128_t plpc2 = fma_v(als2, bsr0, fma_v(als1, bsr1, wasm_f64x2_mul(als0, bsr2)));
    v128_t plpc3 = fma_v(als3, bsr0, fma_v(als2, bsr1, fma_v(als1, bsr2, wasm_f64x2_mul(als0, bsr3))));
    v128_t plpc4 =
        fma_v(als4, bsr0, fma_v(als3, bsr1, fma_v(als2, bsr2, fma_v(als1, bsr3, wasm_f64x2_mul(als0, bsr4)))));
    v128_t plpc5 =
        fma_v(als5,
              bsr0,
              fma_v(als4, bsr1, fma_v(als3, bsr2, fma_v(als2, bsr3, fma_v(als1, bsr4, wasm_f64x2_mul(als0, bsr5))))));
    v128_t plpc6 =
        fma_v(als5, bsr1, fma_v(als4, bsr2, fma_v(als3, bsr3, fma_v(als2, bsr4, wasm_f64x2_mul(als1, bsr5)))));
    v128_t plpc7 = fma_v(als5, bsr2, fma_v(als4, bsr3, fma_v(als3, bsr4, wasm_f64x2_mul(als2, bsr5))));
    v128_t plpc8 = fma_v(als5, bsr3, fma_v(als4, bsr4, wasm_f64x2_mul(als3, bsr5)));
    v128_t plpc9 = fma_v(als5, bsr4, wasm_f64x2_mul(als4, bsr5));
    v128_t plpc10 = wasm_f64x2_mul(als5, bsr5);

    double pl0 = wasm_f64x2_extract_lane(plpc0, 0);
    double pl1 = wasm_f64x2_extract_lane(plpc1, 0);
    double pl2 = wasm_f64x2_extract_lane(plpc2, 0);
    double pl3 = wasm_f64x2_extract_lane(plpc3, 0);
    double pl4 = wasm_f64x2_extract_lane(plpc4, 0);
    double pl5 = wasm_f64x2_extract_lane(plpc5, 0);
    double pl6 = wasm_f64x2_extract_lane(plpc6, 0);
    double pl7 = wasm_f64x2_extract_lane(plpc7, 0);
    double pl8 = wasm_f64x2_extract_lane(plpc8, 0);
    double pl9 = wasm_f64x2_extract_lane(plpc9, 0);
    double pl10 = wasm_f64x2_extract_lane(plpc10, 0);

    double pc0 = wasm_f64x2_extract_lane(plpc0, 1);
    double pc1 = wasm_f64x2_extract_lane(plpc1, 1);
    double pc2 = wasm_f64x2_extract_lane(plpc2, 1);
    double pc3 = wasm_f64x2_extract_lane(plpc3, 1);
    double pc4 = wasm_f64x2_extract_lane(plpc4, 1);
    double pc5 = wasm_f64x2_extract_lane(plpc5, 1);
    double pc6 = wasm_f64x2_extract_lane(plpc6, 1);
    double pc7 = wasm_f64x2_extract_lane(plpc7, 1);
    double pc8 = wasm_f64x2_extract_lane(plpc8, 1);
    double pc9 = wasm_f64x2_extract_lane(plpc9, 1);
    double pc10 = wasm_f64x2_extract_lane(plpc10, 1);

    // P_hi = left[6..10] x right[6..10] -- 5x5 scalar
    double ph0 = af6 * bf6;
    double ph1 = fma_s(af7, bf6, af6 * bf7);
    double ph2 = fma_s(af8, bf6, fma_s(af7, bf7, af6 * bf8));
    double ph3 = fma_s(af9, bf6, fma_s(af8, bf7, fma_s(af7, bf8, af6 * bf9)));
    double ph4 = fma_s(af10, bf6, fma_s(af9, bf7, fma_s(af8, bf8, fma_s(af7, bf9, af6 * bf10))));
    double ph5 = fma_s(af10, bf7, fma_s(af9, bf8, fma_s(af8, bf9, af7 * bf10)));
    double ph6 = fma_s(af10, bf8, fma_s(af9, bf9, af8 * bf10));
    double ph7 = fma_s(af10, bf9, af9 * bf10);
    double ph8 = af10 * bf10;

    // Combine
    double t0 = pl0;
    double t1 = pl1;
    double t2 = pl2;
    double t3 = pl3;
    double t4 = pl4;
    double t5 = pl5;
    double t6 = pl6 + (pc0 - pl0 - ph0);
    double t7 = pl7 + (pc1 - pl1 - ph1);
    double t8 = pl8 + (pc2 - pl2 - ph2);
    double t9 = pl9 + (pc3 - pl3 - ph3);
    double t10 = pl10 + (pc4 - pl4 - ph4);
    double t11 = (pc5 - pl5 - ph5);
    double t12 = (pc6 - pl6 - ph6) + ph0;
    double t13 = (pc7 - pl7 - ph7) + ph1;
    double t14 = (pc8 - pl8 - ph8) + ph2;
    double t15 = (pc9 - pl9) + ph3;
    double t16 = (pc10 - pl10) + ph4;
    double t17 = ph5;
    double t18 = ph6;
    double t19 = ph7;
    double t20 = ph8;

    // ================================================================
    // Phase 3: Reduction -- 10 Yuval + 1 standard, with SIMD scatter
    // ================================================================
    const v128_t rinv_01 = wasm_f64x2_make(r_limbs.div_r_inv_f[0], r_limbs.div_r_inv_f[1]);
    const v128_t rinv_23 = wasm_f64x2_make(r_limbs.div_r_inv_f[2], r_limbs.div_r_inv_f[3]);
    const v128_t rinv_45 = wasm_f64x2_make(r_limbs.div_r_inv_f[4], r_limbs.div_r_inv_f[5]);
    const v128_t rinv_67 = wasm_f64x2_make(r_limbs.div_r_inv_f[6], r_limbs.div_r_inv_f[7]);
    const v128_t rinv_89 = wasm_f64x2_make(r_limbs.div_r_inv_f[8], r_limbs.div_r_inv_f[9]);

    double qi, ki;
    v128_t ks, p01, p23, p45, p67, p89;

#define YUVAL_STEP_SIMD(TI, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)                                              \
    qi = std::floor(TI * SD);                                                                                          \
    ki = TI - qi * SU;                                                                                                 \
    ks = wasm_f64x2_splat(ki);                                                                                         \
    p01 = fma_v(ks, rinv_01, wasm_f64x2_make(T1 + qi, T2));                                                            \
    T1 = wasm_f64x2_extract_lane(p01, 0);                                                                              \
    T2 = wasm_f64x2_extract_lane(p01, 1);                                                                              \
    p23 = fma_v(ks, rinv_23, wasm_f64x2_make(T3, T4));                                                                 \
    T3 = wasm_f64x2_extract_lane(p23, 0);                                                                              \
    T4 = wasm_f64x2_extract_lane(p23, 1);                                                                              \
    p45 = fma_v(ks, rinv_45, wasm_f64x2_make(T5, T6));                                                                 \
    T5 = wasm_f64x2_extract_lane(p45, 0);                                                                              \
    T6 = wasm_f64x2_extract_lane(p45, 1);                                                                              \
    p67 = fma_v(ks, rinv_67, wasm_f64x2_make(T7, T8));                                                                 \
    T7 = wasm_f64x2_extract_lane(p67, 0);                                                                              \
    T8 = wasm_f64x2_extract_lane(p67, 1);                                                                              \
    p89 = fma_v(ks, rinv_89, wasm_f64x2_make(T9, T10));                                                                \
    T9 = wasm_f64x2_extract_lane(p89, 0);                                                                              \
    T10 = wasm_f64x2_extract_lane(p89, 1);                                                                             \
    T11 = fma_s(ki, r_limbs.div_r_inv_f[10], T11);

    YUVAL_STEP_SIMD(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)
    YUVAL_STEP_SIMD(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12)
    YUVAL_STEP_SIMD(t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13)
    YUVAL_STEP_SIMD(t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14)
    YUVAL_STEP_SIMD(t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15)
    YUVAL_STEP_SIMD(t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16)
    YUVAL_STEP_SIMD(t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17)
    YUVAL_STEP_SIMD(t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18)
    YUVAL_STEP_SIMD(t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19)
    YUVAL_STEP_SIMD(t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20)

#undef YUVAL_STEP_SIMD

    // Standard step 10
    {
        // Limbs 0–1 use scalar fma below (t10, t11), so no vector pair for mod_01.
        const v128_t mod_23 = wasm_f64x2_make(r_limbs.modulus_f[2], r_limbs.modulus_f[3]);
        const v128_t mod_45 = wasm_f64x2_make(r_limbs.modulus_f[4], r_limbs.modulus_f[5]);
        const v128_t mod_67 = wasm_f64x2_make(r_limbs.modulus_f[6], r_limbs.modulus_f[7]);
        const v128_t mod_89 = wasm_f64x2_make(r_limbs.modulus_f[8], r_limbs.modulus_f[9]);

        double q10 = std::floor(t10 * SD);
        double k_base = t10 - q10 * SU;
        double k_product = k_base * NP0_F;
        double k_std = k_product - std::floor(k_product * SD) * SU;

        double carry_10 = fma_s(k_std, r_limbs.modulus_f[0] * SD, t10 * SD);

        ks = wasm_f64x2_splat(k_std);
        t11 = fma_s(k_std, r_limbs.modulus_f[1], t11 + carry_10);

        v128_t pm23 = fma_v(ks, mod_23, wasm_f64x2_make(t12, t13));
        t12 = wasm_f64x2_extract_lane(pm23, 0);
        t13 = wasm_f64x2_extract_lane(pm23, 1);

        v128_t pm45 = fma_v(ks, mod_45, wasm_f64x2_make(t14, t15));
        t14 = wasm_f64x2_extract_lane(pm45, 0);
        t15 = wasm_f64x2_extract_lane(pm45, 1);

        v128_t pm67 = fma_v(ks, mod_67, wasm_f64x2_make(t16, t17));
        t16 = wasm_f64x2_extract_lane(pm67, 0);
        t17 = wasm_f64x2_extract_lane(pm67, 1);

        v128_t pm89 = fma_v(ks, mod_89, wasm_f64x2_make(t18, t19));
        t18 = wasm_f64x2_extract_lane(pm89, 0);
        t19 = wasm_f64x2_extract_lane(pm89, 1);

        t20 = fma_s(k_std, r_limbs.modulus_f[10], t20);
    }

    // ================================================================
    // Phase 4: Integer carry propagation + output
    // ================================================================
    uint64_t r11 = static_cast<uint64_t>(static_cast<int64_t>(t11));
    uint64_t r12 = static_cast<uint64_t>(static_cast<int64_t>(t12));
    uint64_t r13 = static_cast<uint64_t>(static_cast<int64_t>(t13));
    uint64_t r14 = static_cast<uint64_t>(static_cast<int64_t>(t14));
    uint64_t r15 = static_cast<uint64_t>(static_cast<int64_t>(t15));
    uint64_t r16 = static_cast<uint64_t>(static_cast<int64_t>(t16));
    uint64_t r17 = static_cast<uint64_t>(static_cast<int64_t>(t17));
    uint64_t r18 = static_cast<uint64_t>(static_cast<int64_t>(t18));
    uint64_t r19 = static_cast<uint64_t>(static_cast<int64_t>(t19));
    uint64_t r20 = static_cast<uint64_t>(static_cast<int64_t>(t20));

    r12 += r11 >> 24;
    r11 &= M24;
    r13 += r12 >> 24;
    r12 &= M24;
    r14 += r13 >> 24;
    r13 &= M24;
    r15 += r14 >> 24;
    r14 &= M24;
    r16 += r15 >> 24;
    r15 &= M24;
    r17 += r16 >> 24;
    r16 &= M24;
    r18 += r17 >> 24;
    r17 &= M24;
    r19 += r18 >> 24;
    r18 &= M24;
    r20 += r19 >> 24;
    r19 &= M24;

    return { r11 | (r12 << 24) | (r13 << 48),
             (r13 >> 16) | (r14 << 8) | (r15 << 32) | (r16 << 56),
             (r16 >> 8) | (r17 << 16) | (r18 << 40),
             r19 | (r20 << 24) };
}

// ═════════════════════════════════════════════════════════════════════════
// Dual Montgomery multiplication: out1 = a1*b1, out2 = a2*b2.
//
// Same algorithm as mul_fma_simd, but both f64x2 lanes carry entirely
// separate multiplications from start to finish (lane 0 = first mul,
// lane 1 = second). Better SIMD utilization — no lane ever idles.
//
// Point arithmetic (e.g., mixed addition) naturally has pairs of
// independent field muls, making this the preferred entry point.
// ═════════════════════════════════════════════════════════════════════════

template <class Params>
void WasmFmaBackend<Params>::mul_paired_fma_simd_impl(const field<Params>& a1,
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
void WasmFmaBackend<Params>::sqr_paired_fma_simd_impl(const field<Params>& a1,
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
