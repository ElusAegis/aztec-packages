#pragma once

// WASM RNE (Round-to-Nearest-Even) FMA SIMD Montgomery backend.
//
// Keeps external 4x64-bit storage with R = 2^256 (byte-identical to the native
// int128 layout) but performs the multiplication internally on 5x51-bit limbs
// via f64x2 relaxed-FMA.
//
// C++ port of worldfnd/provekit skyscraper/bn254-multiplier/src/rne/mono.rs::mul
// (MIT, Copyright 2025 World Foundation), itself adapted from Emmart-Zheng-
// Weems, "Faster Modular Exponentiation using Double Precision Floating Point
// Arithmetic on the GPU" (ARITH 2018).

#include "../field_montgomery_config.hpp"

#if BB_R_LIMB_BITS == 64 && defined(MONTMUL_VARIANT_RNE) && defined(__wasm_simd128__)

#include <array>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <wasm_simd128.h>

#include "../field_constexpr_helpers.hpp"
#include "../field_declarations.hpp"
#include "wasm_int29.hpp"

namespace bb {
class Bn254FrParams;
class Bn254FqParams;
} // namespace bb

namespace bb::detail {

namespace rne_detail {

// ---- Helper macros: force every kernel helper to fully inline --------------
// BB_INLINE is the project's usual "please inline" marker; BB_ALWAYS adds the
// compiler-level always_inline to defeat WASM's conservative inliner on
// SIMD-heavy leaf functions. Every function in this kernel needs to inline
// or the compiler spills v128_t locals to linear memory.
#define BB_ALWAYS BB_INLINE __attribute__((always_inline))

inline constexpr uint64_t MASK51 = (1ULL << 51) - 1;

// ---- Anchor constants ------------------------------------------------------
inline constexpr double C1_D = 0x1p103;
inline constexpr double C2_D = 0x1p103 + 0x1p52 + 0x1p51;

// make_initial(low_count, high_count): per-limb bias that cancels the anchor
// residue accumulated by `low_count` p_lo terms plus `high_count` p_hi terms.
BB_ALWAYS constexpr int64_t make_initial(uint64_t low_count, uint64_t high_count) noexcept
{
    constexpr uint64_t C1_BITS = 0x4660000000000000ULL; // bits of 2^103
    constexpr uint64_t C3_BITS = 0x4338000000000000ULL; // bits of 1.5 * 2^52
    const uint64_t val = high_count * C1_BITS + low_count * C3_BITS;
    return -static_cast<int64_t>(val);
}

// Precompute each init bias pair as a constexpr int64_t, then materialize as a
// `wasm_i64x2_const` immediate -- this guarantees the value lands as a v128
// immediate in the bytecode rather than a pair of replace_lane ops.
inline constexpr int64_t INIT0_LO = make_initial(1, 0);
inline constexpr int64_t INIT0_HI = make_initial(2, 1);
inline constexpr int64_t INIT2_LO = make_initial(3, 2);
inline constexpr int64_t INIT2_HI = make_initial(4, 3);
inline constexpr int64_t INIT4_LO = make_initial(10, 4);
inline constexpr int64_t INIT4_HI = make_initial(10, 10);
inline constexpr int64_t INIT6_LO = make_initial(9, 10);
inline constexpr int64_t INIT6_HI = make_initial(8, 9);
inline constexpr int64_t INIT8_LO = make_initial(7, 8);
inline constexpr int64_t INIT8_HI = make_initial(6, 7);

BB_ALWAYS v128_t fma_v(v128_t a, v128_t b, v128_t c) noexcept
{
    return __builtin_wasm_relaxed_madd_f64x2(a, b, c);
}

// i2f bit-trick: OR the low 52 bits of the integer into the mantissa of 2^52,
// then subtract 2^52 to strip the exponent back out.
BB_ALWAYS v128_t i2f_v128(v128_t u) noexcept
{
    const v128_t exponent = wasm_i64x2_const_splat(0x4330000000000000LL);
    return wasm_f64x2_sub(wasm_v128_or(u, exponent), exponent);
}

BB_ALWAYS double i2f_scalar(uint64_t u) noexcept
{
    constexpr uint64_t EXP_BITS = 0x4330000000000000ULL;
    return std::bit_cast<double>(u | EXP_BITS) - std::bit_cast<double>(EXP_BITS);
}

// Compile-time shuffle over 64-bit lanes.
template <int a_lane, int b_lane> BB_ALWAYS v128_t shuffle_i64x2(v128_t a, v128_t b) noexcept
{
    static_assert(a_lane == 0 || a_lane == 1);
    static_assert(b_lane == 0 || b_lane == 1);
    if constexpr (a_lane == 0 && b_lane == 0) {
        return wasm_i8x16_shuffle(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
    } else if constexpr (a_lane == 0 && b_lane == 1) {
        return wasm_i8x16_shuffle(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31);
    } else if constexpr (a_lane == 1 && b_lane == 0) {
        return wasm_i8x16_shuffle(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
    } else {
        return wasm_i8x16_shuffle(a, b, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
    }
}

BB_ALWAYS constexpr std::array<uint64_t, 5> u256_to_u255(const uint64_t (&l)[4]) noexcept
{
    return {
        l[0] & MASK51,
        ((l[0] >> 51) | (l[1] << 13)) & MASK51,
        ((l[1] >> 38) | (l[2] << 26)) & MASK51,
        ((l[2] >> 25) | (l[3] << 39)) & MASK51,
        (l[3] >> 12) & MASK51,
    };
}

BB_ALWAYS constexpr std::array<uint64_t, 4> u255_to_u256_shr_1(const std::array<uint64_t, 5>& l) noexcept
{
    return {
        (l[0] >> 1) | (l[1] << 50),
        (l[1] >> 14) | (l[2] << 37),
        (l[2] >> 27) | (l[3] << 24),
        (l[3] >> 40) | (l[4] << 11),
    };
}

template <size_t N>
BB_ALWAYS constexpr std::array<uint64_t, N> redundant_carry(const std::array<int64_t, N>& t) noexcept
{
    static_assert(N > 0);
    std::array<uint64_t, N> res{};
    int64_t borrow = 0;
    if constexpr (N > 1) {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((res[Is] = [&] {
                 const int64_t tmp = t[Is] + borrow;
                 const int64_t out = static_cast<int64_t>(static_cast<uint64_t>(tmp) & MASK51);
                 borrow = tmp >> 51;
                 return static_cast<uint64_t>(out);
             }()),
             ...);
        }(std::make_index_sequence<N - 1>{});
    }
    res[N - 1] = static_cast<uint64_t>(t[N - 1] + borrow);
    return res;
}

BB_ALWAYS constexpr std::array<int64_t, 5> reduce_ct(std::array<int64_t, 5> a,
                                                     const std::array<uint64_t, 5>& u51_p) noexcept
{
    const int64_t mask = -(a[0] & 1);
    a[0] += static_cast<int64_t>(u51_p[0]) & mask;
    a[1] += static_cast<int64_t>(u51_p[1]) & mask;
    a[2] += static_cast<int64_t>(u51_p[2]) & mask;
    a[3] += static_cast<int64_t>(u51_p[3]) & mask;
    a[4] += static_cast<int64_t>(u51_p[4]) & mask;
    return a;
}

// smult_noinit: scalar s times 5-limb constant v, producing 6 accumulator
// vectors in redundant SIMD form. Returns by 6 separate refs (ugly but the
// only reliably-SROA-able signature on clang/WASM -- an aggregate return
// gets spilled through linear memory in at least one toolchain version).
BB_ALWAYS void smult_noinit_v128(uint64_t s_scalar,
                                 const std::array<uint64_t, 5>& v,
                                 v128_t& o0, v128_t& o1, v128_t& o2,
                                 v128_t& o3, v128_t& o4, v128_t& o5) noexcept
{
    const v128_t C1_V = wasm_f64x2_const_splat(C1_D);
    const v128_t C2_V = wasm_f64x2_const_splat(C2_D);
    const v128_t s_f = wasm_f64x2_splat(i2f_scalar(s_scalar));

    const v128_t v01 = wasm_f64x2_make(static_cast<double>(v[0]), static_cast<double>(v[1]));
    const v128_t p_hi01 = fma_v(s_f, v01, C1_V);
    const v128_t p_lo01 = fma_v(s_f, v01, wasm_f64x2_sub(C2_V, p_hi01));
    o1 = p_hi01;
    o0 = p_lo01;

    const v128_t v23 = wasm_f64x2_make(static_cast<double>(v[2]), static_cast<double>(v[3]));
    const v128_t p_hi23 = fma_v(s_f, v23, C1_V);
    const v128_t p_lo23 = fma_v(s_f, v23, wasm_f64x2_sub(C2_V, p_hi23));
    o3 = p_hi23;
    o2 = p_lo23;

    const v128_t v45 = wasm_f64x2_make(static_cast<double>(v[4]), 0.0);
    const v128_t p_hi45 = fma_v(s_f, v45, C1_V);
    const v128_t p_lo45 = fma_v(s_f, v45, wasm_f64x2_sub(C2_V, p_hi45));
    o5 = p_hi45;
    o4 = p_lo45;
}

} // namespace rne_detail


// =========================================================================
// RneBackend<Params>
// =========================================================================

template <class Params> struct RneBackend {
    static constexpr bool USE_RNE_KERNEL =
        std::is_same_v<Params, Bn254FrParams> || std::is_same_v<Params, Bn254FqParams>;

    using IntCompanion = WasmInt29Backend<Params>;

    // mul: no input/output reductions -- the kernel expects coarse form
    // in [0, 2p) on entry and returns coarse form on exit, matching the
    // Rust mono.rs contract exactly.
    //
    // TODO(perf): the original C++ wrapper ran lhs.reduce_once().reduce_once()
    // on each input plus one reduce_once on output, costing ~30-50 ns of the
    // observed 120 ns total. Those reductions existed to defend against
    // arbitrary 256-bit inputs (e.g. fr(uint256_t) constructors that store
    // raw bits). Removing them moves the responsibility to the caller:
    //   - Field constructors must reduce to [0, 2p) before storing.
    //   - Public API boundaries (serialization, comparison, ==) must
    //     reduce once on read.
    // Figure out the right place for each reduction in the surrounding
    // barretenberg field wrapper before shipping.
    BB_INLINE __attribute__((always_inline))
    static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        if (std::is_constant_evaluated()) {
            return constexpr_mont_mul(lhs, rhs);
        }
        if constexpr (USE_RNE_KERNEL) {
            field<Params> out;
            mul_rne_simd(lhs, rhs, out);
            return out;
        } else {
            return constexpr_mont_mul(lhs, rhs);
        }
    }

    BB_INLINE __attribute__((always_inline))
    static constexpr field<Params> sqr(const field<Params>& x) noexcept { return mul(x, x); }

    BB_INLINE __attribute__((always_inline))
    static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        return mul(lhs, rhs);
    }

    BB_INLINE static constexpr typename field<Params>::wide_array wide_mul(const field<Params>& lhs,
                                                                           const field<Params>& rhs) noexcept
    {
        return IntCompanion::wide_mul(lhs, rhs);
    }

    template <size_t N>
    BB_INLINE static constexpr void mul_batched(std::array<const field<Params>*, N> as,
                                                std::array<const field<Params>*, N> bs,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((*outs[Is] = mul(*as[Is], *bs[Is])), ...);
        }(std::make_index_sequence<N>{});
    }

    template <size_t N>
    BB_INLINE static constexpr void sqr_batched(std::array<const field<Params>*, N> as,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((*outs[Is] = sqr(*as[Is])), ...);
        }(std::make_index_sequence<N>{});
    }

  private:
    // Inlined definition so the class template always emits this as a fully-
    // inlinable body (templates-in-class are implicitly inline, but
    // always_inline forces the compiler's hand even when the body is large).
    BB_INLINE __attribute__((always_inline))
    static void mul_rne_simd(const field<Params>& lhs, const field<Params>& rhs, field<Params>& out) noexcept
    {
        using namespace rne_detail;

        // Direct access to the static-constexpr field constants. NO LOCAL COPY.
        // `field<Params>::r_limbs.rho[k]` lowers to an immediate-constant
        // `v128.const` in the emitted wasm once we inline smult_noinit_v128.
        constexpr const auto& rho0 = field<Params>::r_limbs.rho[0];
        constexpr const auto& rho1 = field<Params>::r_limbs.rho[1];
        constexpr const auto& rho2 = field<Params>::r_limbs.rho[2];
        constexpr const auto& rho3 = field<Params>::r_limbs.rho[3];
        constexpr const auto& u51_p = field<Params>::r_limbs.u51_p;
        constexpr uint64_t u51_np0 = field<Params>::r_limbs.u51_np0;

        // ---- Phase 1: 4x64 -> 5x51. -------------------------------------
        const auto a = u256_to_u255(lhs.data);
        const auto b = u256_to_u255(rhs.data);

        // ---- Phase 2: Schoolbook 5x5 FMA product. ----------------------
        //
        // All ten ts_N accumulators live in named locals so the WASM SROA
        // pass can keep them in SSA the entire function. No arrays, no
        // lambda captures, no reference escapes.
        const v128_t C1_V = wasm_f64x2_const_splat(C1_D);
        const v128_t C2_V = wasm_f64x2_const_splat(C2_D);

        v128_t ts0 = wasm_i64x2_const(INIT0_LO, INIT0_HI);
        v128_t ts1 = wasm_i64x2_const_splat(0);
        v128_t ts2 = wasm_i64x2_const(INIT2_LO, INIT2_HI);
        v128_t ts3 = wasm_i64x2_const_splat(0);
        v128_t ts4 = wasm_i64x2_const(INIT4_LO, INIT4_HI);
        v128_t ts5 = wasm_i64x2_const_splat(0);
        v128_t ts6 = wasm_i64x2_const(INIT6_LO, INIT6_HI);
        v128_t ts7 = wasm_i64x2_const_splat(0);
        v128_t ts8 = wasm_i64x2_const(INIT8_LO, INIT8_HI);
        v128_t ts9 = wasm_i64x2_const_splat(0);

        // Convert b-pair buckets once (outside the a-loop) since they are
        // reused for every a-limb.
        const v128_t b01 = i2f_v128(wasm_i64x2_make(static_cast<int64_t>(b[0]), static_cast<int64_t>(b[1])));
        const v128_t b23 = i2f_v128(wasm_i64x2_make(static_cast<int64_t>(b[2]), static_cast<int64_t>(b[3])));
        const v128_t b4x = wasm_f64x2_make(i2f_scalar(b[4]), 0.0);

        // The schoolbook is hand-unrolled into 5 explicit blocks. A templated
        // static_for would be equivalent in theory, but on clang/wasm a
        // reference-capturing lambda over ~10 named SSA locals can prevent
        // promotion. Flat is the safe choice here.

        #define RNE_SCHOOLBOOK_STEP(I, TS_LO_0, TS_HI_0, TS_LO_1, TS_HI_1, TS_LO_2, TS_HI_2)   \
            do {                                                                                \
                const v128_t ai = i2f_v128(wasm_i64x2_splat(static_cast<int64_t>(a[I])));       \
                const v128_t h01 = fma_v(ai, b01, C1_V);                                        \
                const v128_t l01 = fma_v(ai, b01, wasm_f64x2_sub(C2_V, h01));                   \
                TS_HI_0 = wasm_i64x2_add(TS_HI_0, h01);                                         \
                TS_LO_0 = wasm_i64x2_add(TS_LO_0, l01);                                         \
                const v128_t h23 = fma_v(ai, b23, C1_V);                                        \
                const v128_t l23 = fma_v(ai, b23, wasm_f64x2_sub(C2_V, h23));                   \
                TS_HI_1 = wasm_i64x2_add(TS_HI_1, h23);                                         \
                TS_LO_1 = wasm_i64x2_add(TS_LO_1, l23);                                         \
                const v128_t h4 = fma_v(ai, b4x, C1_V);                                         \
                const v128_t l4 = fma_v(ai, b4x, wasm_f64x2_sub(C2_V, h4));                     \
                TS_HI_2 = wasm_i64x2_add(TS_HI_2, h4);                                          \
                TS_LO_2 = wasm_i64x2_add(TS_LO_2, l4);                                          \
            } while (0)

        // i=0: ts[0,1]  ts[2,3]  ts[4,5]
        RNE_SCHOOLBOOK_STEP(0, ts0, ts1, ts2, ts3, ts4, ts5);
        // i=1: ts[1,2]  ts[3,4]  ts[5,6]
        RNE_SCHOOLBOOK_STEP(1, ts1, ts2, ts3, ts4, ts5, ts6);
        // i=2: ts[2,3]  ts[4,5]  ts[6,7]
        RNE_SCHOOLBOOK_STEP(2, ts2, ts3, ts4, ts5, ts6, ts7);
        // i=3: ts[3,4]  ts[5,6]  ts[7,8]
        RNE_SCHOOLBOOK_STEP(3, ts3, ts4, ts5, ts6, ts7, ts8);
        // i=4: ts[4,5]  ts[6,7]  ts[8,9]
        RNE_SCHOOLBOOK_STEP(4, ts4, ts5, ts6, ts7, ts8, ts9);

        #undef RNE_SCHOOLBOOK_STEP

        // ---- Phase 3: Fold odd-indexed hi-piles, extract low 4 scalars. -
        const v128_t zero_v = wasm_i64x2_const_splat(0);

        // k=0: ts0 += [0, ts1[0]],  ts2 += [ts1[1], 0]
        ts0 = wasm_i64x2_add(ts0, shuffle_i64x2<0, 0>(zero_v, ts1));
        ts2 = wasm_i64x2_add(ts2, shuffle_i64x2<1, 0>(ts1, zero_v));
        const int64_t t0 = wasm_i64x2_extract_lane(ts0, 0);
        const int64_t t1_raw = wasm_i64x2_extract_lane(ts0, 1);

        // k=2: ts2 += [0, ts3[0]],  ts4 += [ts3[1], 0]
        ts2 = wasm_i64x2_add(ts2, shuffle_i64x2<0, 0>(zero_v, ts3));
        ts4 = wasm_i64x2_add(ts4, shuffle_i64x2<1, 0>(ts3, zero_v));
        const int64_t t2_raw = wasm_i64x2_extract_lane(ts2, 0);
        const int64_t t3_raw = wasm_i64x2_extract_lane(ts2, 1);

        // Sequential signed carry propagation.
        const int64_t t1 = t1_raw + (t0 >> 51);
        const int64_t t2 = t2_raw + (t1 >> 51);
        const int64_t t3 = t3_raw + (t2 >> 51);

        // Lift the final carry into ts4 lane 0 (stay in SIMD).
        ts4 = wasm_i64x2_add(ts4, wasm_i64x2_make(t3 >> 51, 0));

        // ---- Phase 4: Parallel reduction. -------------------------------
        // t[i] * rho[3-i], summed into ss[0..5]. Outputs in named locals.
        v128_t r00, r01, r02, r03, r04, r05;
        v128_t r10, r11, r12, r13, r14, r15;
        v128_t r20, r21, r22, r23, r24, r25;
        v128_t r30, r31, r32, r33, r34, r35;
        smult_noinit_v128(static_cast<uint64_t>(t0) & MASK51, rho3, r00, r01, r02, r03, r04, r05);
        smult_noinit_v128(static_cast<uint64_t>(t1) & MASK51, rho2, r10, r11, r12, r13, r14, r15);
        smult_noinit_v128(static_cast<uint64_t>(t2) & MASK51, rho1, r20, r21, r22, r23, r24, r25);
        smult_noinit_v128(static_cast<uint64_t>(t3) & MASK51, rho0, r30, r31, r32, r33, r34, r35);

        // Tree-reduce the 4 rho contributions per limb to shorten the
        // critical-path dependency chain.
        v128_t ss0 = wasm_i64x2_add(ts4, wasm_i64x2_add(wasm_i64x2_add(r00, r10), wasm_i64x2_add(r20, r30)));
        v128_t ss1 = wasm_i64x2_add(ts5, wasm_i64x2_add(wasm_i64x2_add(r01, r11), wasm_i64x2_add(r21, r31)));
        v128_t ss2 = wasm_i64x2_add(ts6, wasm_i64x2_add(wasm_i64x2_add(r02, r12), wasm_i64x2_add(r22, r32)));
        v128_t ss3 = wasm_i64x2_add(ts7, wasm_i64x2_add(wasm_i64x2_add(r03, r13), wasm_i64x2_add(r23, r33)));
        v128_t ss4 = wasm_i64x2_add(ts8, wasm_i64x2_add(wasm_i64x2_add(r04, r14), wasm_i64x2_add(r24, r34)));
        v128_t ss5 = wasm_i64x2_add(ts9, wasm_i64x2_add(wasm_i64x2_add(r05, r15), wasm_i64x2_add(r25, r35)));

        // ---- Phase 5: Final Montgomery step. m = ss[0][0] * np0 mod 2^51.
        const uint64_t m = (static_cast<uint64_t>(wasm_i64x2_extract_lane(ss0, 0)) * u51_np0) & MASK51;
        v128_t mp0, mp1, mp2, mp3, mp4, mp5;
        smult_noinit_v128(m, u51_p, mp0, mp1, mp2, mp3, mp4, mp5);
        ss0 = wasm_i64x2_add(ss0, mp0);
        ss1 = wasm_i64x2_add(ss1, mp1);
        ss2 = wasm_i64x2_add(ss2, mp2);
        ss3 = wasm_i64x2_add(ss3, mp3);
        ss4 = wasm_i64x2_add(ss4, mp4);
        ss5 = wasm_i64x2_add(ss5, mp5);

        // Realign: same swizzle pattern as phase 3 + final ss5 fold.
        ss0 = wasm_i64x2_add(ss0, shuffle_i64x2<0, 0>(zero_v, ss1));
        ss2 = wasm_i64x2_add(ss2, shuffle_i64x2<1, 0>(ss1, zero_v));
        ss2 = wasm_i64x2_add(ss2, shuffle_i64x2<0, 0>(zero_v, ss3));
        ss4 = wasm_i64x2_add(ss4, shuffle_i64x2<1, 0>(ss3, zero_v));
        ss4 = wasm_i64x2_add(ss4, shuffle_i64x2<0, 0>(zero_v, ss5));

        // Extract ss[0,2,4] lanes into redundant scalar form s[0..5].
        const int64_t s0 = wasm_i64x2_extract_lane(ss0, 0);
        const int64_t s1_raw = wasm_i64x2_extract_lane(ss0, 1);
        const int64_t s2 = wasm_i64x2_extract_lane(ss2, 0);
        const int64_t s3 = wasm_i64x2_extract_lane(ss2, 1);
        const int64_t s4 = wasm_i64x2_extract_lane(ss4, 0);
        const int64_t s5 = wasm_i64x2_extract_lane(ss4, 1);

        // s0 is the forced-zero lowest limb; propagate carry into s1 and
        // drop it from the output.
        const int64_t s1 = s1_raw + (s0 >> 51);

        // ---- Phase 6: R=2^255 -> R=2^256 adjustment. --------------------
        std::array<int64_t, 5> s_arr = { s1, s2, s3, s4, s5 };
        s_arr = reduce_ct(s_arr, u51_p);
        const std::array<uint64_t, 5> reduced = redundant_carry<5>(s_arr);
        const auto result = u255_to_u256_shr_1(reduced);

        out.data[0] = result[0];
        out.data[1] = result[1];
        out.data[2] = result[2];
        out.data[3] = result[3];
    }
};

#undef BB_ALWAYS

} // namespace bb::detail

#endif // BB_R_LIMB_BITS == 64 && MONTMUL_VARIANT_RNE && __wasm_simd128__
