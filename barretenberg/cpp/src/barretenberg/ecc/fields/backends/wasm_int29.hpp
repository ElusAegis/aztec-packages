#pragma once

// WASM 29-bit integer Montgomery backend.
//
// Provides:
//   - 9×29-bit Karatsuba (5+4) montmul — 66 muls vs 81 for schoolbook
//   - 9×29-bit schoolbook montsqr
//   - big-modulus montmul with interleaved reduction
//   - raw 256×256 → 512 wide multiply (R-independent, also usable from FMA)
//
// Templated on R_EXPONENT so a single class body covers two R regimes:
//
//   REXP = 261 (default): native 9×29-bit reduction — today's behavior.
//                         Final step uses a 29-bit `k`, output packing is
//                         byte-identical to the legacy non-templated code.
//   REXP = 264: FMA-compatible R — 8 Yuval (29-bit each) + 1 final step
//                         widened to a 32-bit `k` (29·8 + 32 = 264).
//                         Output packing is shifted by 3 bits because the
//                         "virtual LSB" of the reduced value lives at bit 3
//                         of temp_9 in the 29-bit-limb reference frame.
//
// The R=264 instantiation is consumed by the FMA SIMD backend when it
// co-schedules a pure-integer kernel alongside the paired f64x2 FMA kernel
// (N=3/N=5 batched muls). Both must produce outputs at the same R for the
// FMA caller to mix their results. The Yuval chain is bit-identical across
// both regimes; only the final reduction step and the output pack differ.
//
// Active on WASM and any host without __int128. The 29-bit kernel bodies
// use *locally computed* 29-bit limb constants so they work correctly even
// when the platform R_LIMB_BITS ≠ 29 (specifically, under BB_R_LIMB_BITS==24
// the FMA backend instantiates WasmInt29Backend<Params, 264> for its integer
// companion kernel).
//
// Schoolbook paths are fully constexpr (pure integer arithmetic).

#if defined(__wasm__) || !defined(__SIZEOF_INT128__)

#include <array>

#include "../field_constexpr_helpers.hpp"
#include "../field_declarations.hpp"
#include "../field_montgomery_config.hpp"

namespace bb::detail {

template <class Params, unsigned REXP = 9 * 29> struct WasmInt29Backend {
    static_assert(REXP == 261 || REXP == 264,
                  "only R = 2^261 (native 29-bit) and R = 2^264 (FMA-compatible) supported");

    // ── Public MontBackend contract ──────────────────────────────────────

    BB_INLINE static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept;
    BB_INLINE static constexpr field<Params> sqr(const field<Params>& x) noexcept;
    BB_INLINE static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept;
    BB_INLINE static constexpr typename field<Params>::wide_array wide_mul(const field<Params>& lhs,
                                                                           const field<Params>& rhs) noexcept;

    // Batched Montgomery mul: outs[i] = as[i] * bs[i] for i in [0, N).
    // 9×29-bit backend has no SIMD kernel — N sequential schoolbook multiplies.
    template <size_t N>
    BB_INLINE static constexpr void mul_batched(std::array<const field<Params>*, N> as,
                                                std::array<const field<Params>*, N> bs,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            *outs[i] = mul(*as[i], *bs[i]);
        }
    }

    // Batched Montgomery sqr: outs[i] = as[i]^2 for i in [0, N).
    template <size_t N>
    BB_INLINE static constexpr void sqr_batched(std::array<const field<Params>*, N> as,
                                                std::array<field<Params>*, N> outs) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            *outs[i] = sqr(*as[i]);
        }
    }

    // Exposed for FMA backend (which reuses wide_mul's 29-bit splitting).
    BB_INLINE static constexpr std::array<uint64_t, 9> wasm_convert(const uint64_t* data)
    {
        return { data[0] & 0x1fffffff,
                 (data[0] >> 29) & 0x1fffffff,
                 ((data[0] >> 58) & 0x3f) | ((data[1] & 0x7fffff) << 6),
                 (data[1] >> 23) & 0x1fffffff,
                 ((data[1] >> 52) & 0xfff) | ((data[2] & 0x1ffff) << 12),
                 (data[2] >> 17) & 0x1fffffff,
                 ((data[2] >> 46) & 0x3ffff) | ((data[3] & 0x7ff) << 18),
                 (data[3] >> 11) & 0x1fffffff,
                 (data[3] >> 40) & 0x1fffffff };
    }

  private:
    // ── Local 29-bit constants ────────────────────────────────────────────
    //
    // Deliberately NOT using field<Params>::r_limbs (which is R_LIMB_BITS-wide
    // and therefore 24-bit on FMA builds). These kernels intrinsically operate
    // at 29 bits regardless of the platform's R_LIMB_BITS.
    static constexpr unsigned LIMB_BITS_29 = 29;
    static constexpr unsigned NUM_LIMBS_29 = 9;
    static constexpr uint64_t MASK_29 = 0x1fffffff;
    static constexpr auto r_limbs_29 = compute_limb_constants<LIMB_BITS_29, NUM_LIMBS_29>(field<Params>::modulus);

    // ── 29-bit limb helpers ──────────────────────────────────────────────

    // Multiply-accumulate: r[i] += left_limb * right_limbs[i] for i in 0..8.
    BB_INLINE static constexpr void wasm_madd(uint64_t& left_limb,
                                              const std::array<uint64_t, 9>& right_limbs,
                                              uint64_t* r)
    {
        r[0] += left_limb * right_limbs[0];
        r[1] += left_limb * right_limbs[1];
        r[2] += left_limb * right_limbs[2];
        r[3] += left_limb * right_limbs[3];
        r[4] += left_limb * right_limbs[4];
        r[5] += left_limb * right_limbs[5];
        r[6] += left_limb * right_limbs[6];
        r[7] += left_limb * right_limbs[7];
        r[8] += left_limb * right_limbs[8];
    }

    // Standard Montgomery reduction step with a 29-bit `k`. Used for the
    // final step of the R = 2^261 kernel (9×29-bit total) and for the
    // interleaved reduction of mul_big. Operates on r[0..8].
    BB_INLINE static constexpr void wasm_reduce(uint64_t* r)
    {
        constexpr uint64_t r_inv_29 = Params::r_inv & MASK_29;
        uint64_t k = (r[0] * r_inv_29) & MASK_29;
        r[0] += k * r_limbs_29.modulus[0];
        r[1] += k * r_limbs_29.modulus[1] + (r[0] >> LIMB_BITS_29);
        r[2] += k * r_limbs_29.modulus[2];
        r[3] += k * r_limbs_29.modulus[3];
        r[4] += k * r_limbs_29.modulus[4];
        r[5] += k * r_limbs_29.modulus[5];
        r[6] += k * r_limbs_29.modulus[6];
        r[7] += k * r_limbs_29.modulus[7];
        r[8] += k * r_limbs_29.modulus[8];
    }

    // Final Montgomery reduction step with a 32-bit `k`. Used only in the
    // REXP = 264 path: after 8 Yuval (29-bit) steps the accumulator has been
    // divided by 2^232; dividing by another 2^32 brings the total to 2^264,
    // matching the FMA SIMD backend's R. Operates on r[0..8].
    //
    // `k` spans r[0]'s 29 bits plus r[1]'s low 3 bits (29 + 3 = 32).
    // U64 safety: k ≤ 2^32 − 1, modulus_29[i] ≤ 2^29 − 1, so k·modulus_29[i]
    // ≤ 2^61 − …; post-Yuval accumulator limbs are ≤ 2^62; sum well below 2^64.
    BB_INLINE static constexpr void wasm_reduce_32bit_k(uint64_t* r)
    {
        constexpr uint64_t MASK_32 = 0xffffffffULL;
        // -p^{-1} mod 2^32: Params::r_inv is -p^{-1} mod 2^64; the low 32
        // bits are the same value mod 2^32.
        constexpr uint64_t r_inv_32 = Params::r_inv & MASK_32;

        // V mod 2^32 spans r[0]'s 29 bits + r[1]'s low 3 bits.
        const uint64_t v_low = ((r[0] & MASK_32) + ((r[1] & 0x7) << LIMB_BITS_29)) & MASK_32;
        const uint64_t k = (v_low * r_inv_32) & MASK_32;

        r[0] += k * r_limbs_29.modulus[0];
        r[1] += k * r_limbs_29.modulus[1] + (r[0] >> LIMB_BITS_29);
        r[2] += k * r_limbs_29.modulus[2];
        r[3] += k * r_limbs_29.modulus[3];
        r[4] += k * r_limbs_29.modulus[4];
        r[5] += k * r_limbs_29.modulus[5];
        r[6] += k * r_limbs_29.modulus[6];
        r[7] += k * r_limbs_29.modulus[7];
        r[8] += k * r_limbs_29.modulus[8];
    }

    // Yuval reduction step — folds r[0] into r[1..9] using the precomputed
    // div_r_inv constants, dividing the accumulator by 2^29. Operates on r[0..9].
    BB_INLINE static constexpr void wasm_reduce_yuval(uint64_t* r)
    {
        const uint64_t r0_masked = r[0] & MASK_29;
        r[1] += r0_masked * r_limbs_29.div_r_inv[0] + (r[0] >> LIMB_BITS_29);
        r[2] += r0_masked * r_limbs_29.div_r_inv[1];
        r[3] += r0_masked * r_limbs_29.div_r_inv[2];
        r[4] += r0_masked * r_limbs_29.div_r_inv[3];
        r[5] += r0_masked * r_limbs_29.div_r_inv[4];
        r[6] += r0_masked * r_limbs_29.div_r_inv[5];
        r[7] += r0_masked * r_limbs_29.div_r_inv[6];
        r[8] += r0_masked * r_limbs_29.div_r_inv[7];
        r[9] += r0_masked * r_limbs_29.div_r_inv[8];
    }

    // Phase 3+4 shared by mul/sqr: 8 Yuval reductions, REXP-dependent
    // final reduce, carry propagation, REXP-dependent output pack. The
    // product phase populates t[0..16] before this helper runs.
    BB_INLINE static constexpr field<Params> reduce_and_finalize(uint64_t* t) noexcept;
};

// Karatsuba Montgomery multiplication (5+4 split, 66 muls). 8 Yuval
// reductions (29-bit each) plus one REXP-dependent final reduce —
// 29-bit k for REXP=261 (total 2^261), 32-bit k for REXP=264 (total
// 2^264, output realigned by 3 bits in the pack).
template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);

    // Karatsuba multiplication: split 9 limbs into 5 (lo) + 4 (hi).
    // P_lo = left[0..4] * right[0..4]  (25 muls)
    // P_hi = left[5..8] * right[5..8]  (16 muls)
    // P_cross = (left_lo + left_hi) * (right_lo + right_hi)  (25 muls)
    // P_mid = P_cross - P_lo - P_hi
    // Total: 66 muls vs 81 for schoolbook 9x9.

    // P_lo = left[0..4] * right[0..4] — 5x5 schoolbook
    uint64_t pl0 = left[0] * right[0];
    uint64_t pl1 = left[0] * right[1] + left[1] * right[0];
    uint64_t pl2 = left[0] * right[2] + left[1] * right[1] + left[2] * right[0];
    uint64_t pl3 = left[0] * right[3] + left[1] * right[2] + left[2] * right[1] + left[3] * right[0];
    uint64_t pl4 =
        left[0] * right[4] + left[1] * right[3] + left[2] * right[2] + left[3] * right[1] + left[4] * right[0];
    uint64_t pl5 = left[1] * right[4] + left[2] * right[3] + left[3] * right[2] + left[4] * right[1];
    uint64_t pl6 = left[2] * right[4] + left[3] * right[3] + left[4] * right[2];
    uint64_t pl7 = left[3] * right[4] + left[4] * right[3];
    uint64_t pl8 = left[4] * right[4];

    // P_hi = left[5..8] * right[5..8] — 4x4 schoolbook
    uint64_t ph0 = left[5] * right[5];
    uint64_t ph1 = left[5] * right[6] + left[6] * right[5];
    uint64_t ph2 = left[5] * right[7] + left[6] * right[6] + left[7] * right[5];
    uint64_t ph3 = left[5] * right[8] + left[6] * right[7] + left[7] * right[6] + left[8] * right[5];
    uint64_t ph4 = left[6] * right[8] + left[7] * right[7] + left[8] * right[6];
    uint64_t ph5 = left[7] * right[8] + left[8] * right[7];
    uint64_t ph6 = left[8] * right[8];

    // Sums for the cross product (left_lo + left_hi, right_lo + right_hi)
    uint64_t sl0 = left[0] + left[5];
    uint64_t sl1 = left[1] + left[6];
    uint64_t sl2 = left[2] + left[7];
    uint64_t sl3 = left[3] + left[8];
    uint64_t sl4 = left[4];
    uint64_t sr0 = right[0] + right[5];
    uint64_t sr1 = right[1] + right[6];
    uint64_t sr2 = right[2] + right[7];
    uint64_t sr3 = right[3] + right[8];
    uint64_t sr4 = right[4];

    // P_cross = sum_left * sum_right — 5x5 schoolbook
    uint64_t pc0 = sl0 * sr0;
    uint64_t pc1 = sl0 * sr1 + sl1 * sr0;
    uint64_t pc2 = sl0 * sr2 + sl1 * sr1 + sl2 * sr0;
    uint64_t pc3 = sl0 * sr3 + sl1 * sr2 + sl2 * sr1 + sl3 * sr0;
    uint64_t pc4 = sl0 * sr4 + sl1 * sr3 + sl2 * sr2 + sl3 * sr1 + sl4 * sr0;
    uint64_t pc5 = sl1 * sr4 + sl2 * sr3 + sl3 * sr2 + sl4 * sr1;
    uint64_t pc6 = sl2 * sr4 + sl3 * sr3 + sl4 * sr2;
    uint64_t pc7 = sl3 * sr4 + sl4 * sr3;
    uint64_t pc8 = sl4 * sr4;

    // Combine: t[k] = P_lo[k] + P_mid[k-5] + P_hi[k-10]
    // where P_mid = P_cross - P_lo - P_hi
    uint64_t t[17] = { pl0,
                       pl1,
                       pl2,
                       pl3,
                       pl4,
                       pl5 + (pc0 - pl0 - ph0),
                       pl6 + (pc1 - pl1 - ph1),
                       pl7 + (pc2 - pl2 - ph2),
                       pl8 + (pc3 - pl3 - ph3),
                       pc4 - pl4 - ph4,
                       (pc5 - pl5 - ph5) + ph0,
                       (pc6 - pl6 - ph6) + ph1,
                       (pc7 - pl7) + ph2,
                       (pc8 - pl8) + ph3,
                       ph4,
                       ph5,
                       ph6 };
    return reduce_and_finalize(t);
}

// Schoolbook Montgomery squaring (triangular). 8 Yuval reductions
// (29-bit each) plus one REXP-dependent final reduce — 29-bit k for
// REXP=261 (total 2^261), 32-bit k for REXP=264 (total 2^264, output
// realigned by 3 bits in the pack).
template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::sqr(const field<Params>& x) noexcept
{
    auto left = wasm_convert(x.data);
    uint64_t t[17] = {};
    uint64_t acc;

    t[0] += left[0] * left[0];
    acc = 0;
    acc += left[0] * left[1];
    t[1] += (acc << 1);
    acc = 0;
    acc += left[0] * left[2];
    t[2] += left[1] * left[1];
    t[2] += (acc << 1);
    acc = 0;
    acc += left[0] * left[3];
    acc += left[1] * left[2];
    t[3] += (acc << 1);
    acc = 0;
    acc += left[0] * left[4];
    acc += left[1] * left[3];
    t[4] += left[2] * left[2];
    t[4] += (acc << 1);
    acc = 0;
    acc += left[0] * left[5];
    acc += left[1] * left[4];
    acc += left[2] * left[3];
    t[5] += (acc << 1);
    acc = 0;
    acc += left[0] * left[6];
    acc += left[1] * left[5];
    acc += left[2] * left[4];
    t[6] += left[3] * left[3];
    t[6] += (acc << 1);
    acc = 0;
    acc += left[0] * left[7];
    acc += left[1] * left[6];
    acc += left[2] * left[5];
    acc += left[3] * left[4];
    t[7] += (acc << 1);
    acc = 0;
    acc += left[0] * left[8];
    acc += left[1] * left[7];
    acc += left[2] * left[6];
    acc += left[3] * left[5];
    t[8] += left[4] * left[4];
    t[8] += (acc << 1);
    acc = 0;
    acc += left[1] * left[8];
    acc += left[2] * left[7];
    acc += left[3] * left[6];
    acc += left[4] * left[5];
    t[9] += (acc << 1);
    acc = 0;
    acc += left[2] * left[8];
    acc += left[3] * left[7];
    acc += left[4] * left[6];
    t[10] += left[5] * left[5];
    t[10] += (acc << 1);
    acc = 0;
    acc += left[3] * left[8];
    acc += left[4] * left[7];
    acc += left[5] * left[6];
    t[11] += (acc << 1);
    acc = 0;
    acc += left[4] * left[8];
    acc += left[5] * left[7];
    t[12] += left[6] * left[6];
    t[12] += (acc << 1);
    acc = 0;
    acc += left[5] * left[8];
    acc += left[6] * left[7];
    t[13] += (acc << 1);
    acc = 0;
    acc += left[6] * left[8];
    t[14] += left[7] * left[7];
    t[14] += (acc << 1);
    acc = 0;
    acc += left[7] * left[8];
    t[15] += (acc << 1);
    t[16] += left[8] * left[8];

    return reduce_and_finalize(t);
}

// Phase 3+4 shared by mul/sqr: 8 Yuval reductions, REXP-dependent final
// reduce, carry propagation, REXP-dependent output pack. The product
// phase populates t[0..16] before this helper runs.
template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::reduce_and_finalize(uint64_t* t) noexcept
{
    wasm_reduce_yuval(&t[0]);
    wasm_reduce_yuval(&t[1]);
    wasm_reduce_yuval(&t[2]);
    wasm_reduce_yuval(&t[3]);
    wasm_reduce_yuval(&t[4]);
    wasm_reduce_yuval(&t[5]);
    wasm_reduce_yuval(&t[6]);
    wasm_reduce_yuval(&t[7]);

    if constexpr (REXP == 261) {
        wasm_reduce(&t[8]);
    } else if constexpr (REXP == 264) {
        wasm_reduce_32bit_k(&t[8]);
    } else {
        static_assert(REXP == 261 || REXP == 264, "unsupported REXP for WasmInt29 final reduce");
    }

    t[10] += t[9] >> LIMB_BITS_29;
    t[9] &= MASK_29;
    t[11] += t[10] >> LIMB_BITS_29;
    t[10] &= MASK_29;
    t[12] += t[11] >> LIMB_BITS_29;
    t[11] &= MASK_29;
    t[13] += t[12] >> LIMB_BITS_29;
    t[12] &= MASK_29;
    t[14] += t[13] >> LIMB_BITS_29;
    t[13] &= MASK_29;
    t[15] += t[14] >> LIMB_BITS_29;
    t[14] &= MASK_29;
    t[16] += t[15] >> LIMB_BITS_29;
    t[15] &= MASK_29;

    if constexpr (REXP == 261) {
        return { (t[9] << 0) | (t[10] << 29) | (t[11] << 58),
                 (t[11] >> 6) | (t[12] << 23) | (t[13] << 52),
                 (t[13] >> 12) | (t[14] << 17) | (t[15] << 46),
                 (t[15] >> 18) | (t[16] << 11) };
    } else if constexpr (REXP == 264) {
        // Virtual LSB at bit 3 of t[9] → offsets {0, 26, 55, 84, 113, 142, 171, 200}.
        return { (t[9] >> 3) | (t[10] << 26) | (t[11] << 55),
                 (t[11] >> 9) | (t[12] << 20) | (t[13] << 49),
                 (t[13] >> 15) | (t[14] << 14) | (t[15] << 43),
                 (t[15] >> 21) | (t[16] << 8) };
    } else {
        static_assert(REXP == 261 || REXP == 264, "unsupported REXP for WasmInt29 output pack");
    }
}

// Big-modulus Montgomery multiplication. 9 interleaved madd+reduce
// pairs for moduli at MODULUS_TOP_LIMB_LARGE_THRESHOLD or above. The
// 9th reduce is REXP-dependent — 29-bit k for REXP=261 (total 2^261),
// 32-bit k for REXP=264 (total 2^264, output realigned by 3 bits before
// the conditional subtract).
template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::mul_big(const field<Params>& lhs,
                                                                const field<Params>& rhs) noexcept
{
    static_assert(field<Params>::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);

    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    uint64_t t[18] = {};

    wasm_madd(left[0], right, &t[0]);
    wasm_reduce(&t[0]);
    wasm_madd(left[1], right, &t[1]);
    wasm_reduce(&t[1]);
    wasm_madd(left[2], right, &t[2]);
    wasm_reduce(&t[2]);
    wasm_madd(left[3], right, &t[3]);
    wasm_reduce(&t[3]);
    wasm_madd(left[4], right, &t[4]);
    wasm_reduce(&t[4]);
    wasm_madd(left[5], right, &t[5]);
    wasm_reduce(&t[5]);
    wasm_madd(left[6], right, &t[6]);
    wasm_reduce(&t[6]);
    wasm_madd(left[7], right, &t[7]);
    wasm_reduce(&t[7]);
    wasm_madd(left[8], right, &t[8]);
    if constexpr (REXP == 261) {
        wasm_reduce(&t[8]);
    } else if constexpr (REXP == 264) {
        wasm_reduce_32bit_k(&t[8]);
    } else {
        static_assert(REXP == 261 || REXP == 264, "unsupported REXP for mul_big final reduce");
    }

    t[10] += t[9] >> LIMB_BITS_29;
    t[9] &= MASK_29;
    t[11] += t[10] >> LIMB_BITS_29;
    t[10] &= MASK_29;
    t[12] += t[11] >> LIMB_BITS_29;
    t[11] &= MASK_29;
    t[13] += t[12] >> LIMB_BITS_29;
    t[12] &= MASK_29;
    t[14] += t[13] >> LIMB_BITS_29;
    t[13] &= MASK_29;
    t[15] += t[14] >> LIMB_BITS_29;
    t[14] &= MASK_29;
    t[16] += t[15] >> LIMB_BITS_29;
    t[15] &= MASK_29;
    t[17] += t[16] >> LIMB_BITS_29;
    t[16] &= MASK_29;

    if constexpr (REXP == 264) {
        // Realign: the 32-bit final reduce leaves the output's virtual LSB at
        // bit 3 of t[9]. Shift the limb window down by 3 bits so the
        // conditional subtract (which expects clean 29-bit limbs aligned at
        // bit 0 of t[9]) can be reused verbatim.
        const uint64_t a_0 = ((t[9] >> 3) | (t[10] << 26)) & MASK_29;
        const uint64_t a_1 = ((t[10] >> 3) | (t[11] << 26)) & MASK_29;
        const uint64_t a_2 = ((t[11] >> 3) | (t[12] << 26)) & MASK_29;
        const uint64_t a_3 = ((t[12] >> 3) | (t[13] << 26)) & MASK_29;
        const uint64_t a_4 = ((t[13] >> 3) | (t[14] << 26)) & MASK_29;
        const uint64_t a_5 = ((t[14] >> 3) | (t[15] << 26)) & MASK_29;
        const uint64_t a_6 = ((t[15] >> 3) | (t[16] << 26)) & MASK_29;
        const uint64_t a_7 = ((t[16] >> 3) | (t[17] << 26)) & MASK_29;
        const uint64_t a_8 = (t[17] >> 3);
        t[9] = a_0;
        t[10] = a_1;
        t[11] = a_2;
        t[12] = a_3;
        t[13] = a_4;
        t[14] = a_5;
        t[15] = a_6;
        t[16] = a_7;
        t[17] = a_8;
    }

    uint64_t r[9];
    r[0] = t[9] - r_limbs_29.modulus[0];
    r[1] = t[10] - r_limbs_29.modulus[1] - (r[0] >> 63);
    r[2] = t[11] - r_limbs_29.modulus[2] - (r[1] >> 63);
    r[3] = t[12] - r_limbs_29.modulus[3] - (r[2] >> 63);
    r[4] = t[13] - r_limbs_29.modulus[4] - (r[3] >> 63);
    r[5] = t[14] - r_limbs_29.modulus[5] - (r[4] >> 63);
    r[6] = t[15] - r_limbs_29.modulus[6] - (r[5] >> 63);
    r[7] = t[16] - r_limbs_29.modulus[7] - (r[6] >> 63);
    r[8] = t[17] - r_limbs_29.modulus[8] - (r[7] >> 63);

    uint64_t new_mask = 0 - (r[8] >> 63);
    uint64_t inverse_mask = (~new_mask) & MASK_29;
    t[9] = (t[9] & new_mask) | (r[0] & inverse_mask);
    t[10] = (t[10] & new_mask) | (r[1] & inverse_mask);
    t[11] = (t[11] & new_mask) | (r[2] & inverse_mask);
    t[12] = (t[12] & new_mask) | (r[3] & inverse_mask);
    t[13] = (t[13] & new_mask) | (r[4] & inverse_mask);
    t[14] = (t[14] & new_mask) | (r[5] & inverse_mask);
    t[15] = (t[15] & new_mask) | (r[6] & inverse_mask);
    t[16] = (t[16] & new_mask) | (r[7] & inverse_mask);
    t[17] = (t[17] & new_mask) | (r[8] & inverse_mask);

    return { (t[9] << 0) | (t[10] << 29) | (t[11] << 58),
             (t[11] >> 6) | (t[12] << 23) | (t[13] << 52),
             (t[13] >> 12) | (t[14] << 17) | (t[15] << 46),
             (t[15] >> 18) | (t[16] << 11) | (t[17] << 40) };
}

// ── 256×256 → 512 wide multiply, WASM path (uses 29-bit internally) ──────
// R-independent: pure integer wide multiply. Safe to call from any WASM
// build, including FMA (where R_LIMB_BITS == 24).

template <class Params, unsigned REXP>
constexpr typename field<Params>::wide_array WasmInt29Backend<Params, REXP>::wide_mul(const field<Params>& lhs,
                                                                                      const field<Params>& rhs) noexcept
{
    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    uint64_t t[17] = {};

    wasm_madd(left[0], right, &t[0]);
    wasm_madd(left[1], right, &t[1]);
    wasm_madd(left[2], right, &t[2]);
    wasm_madd(left[3], right, &t[3]);
    wasm_madd(left[4], right, &t[4]);
    wasm_madd(left[5], right, &t[5]);
    wasm_madd(left[6], right, &t[6]);
    wasm_madd(left[7], right, &t[7]);
    wasm_madd(left[8], right, &t[8]);

    t[1] += t[0] >> 29;
    t[0] &= MASK_29;
    t[2] += t[1] >> 29;
    t[1] &= MASK_29;
    t[3] += t[2] >> 29;
    t[2] &= MASK_29;
    t[4] += t[3] >> 29;
    t[3] &= MASK_29;
    t[5] += t[4] >> 29;
    t[4] &= MASK_29;
    t[6] += t[5] >> 29;
    t[5] &= MASK_29;
    t[7] += t[6] >> 29;
    t[6] &= MASK_29;
    t[8] += t[7] >> 29;
    t[7] &= MASK_29;
    t[9] += t[8] >> 29;
    t[8] &= MASK_29;
    t[10] += t[9] >> 29;
    t[9] &= MASK_29;
    t[11] += t[10] >> 29;
    t[10] &= MASK_29;
    t[12] += t[11] >> 29;
    t[11] &= MASK_29;
    t[13] += t[12] >> 29;
    t[12] &= MASK_29;
    t[14] += t[13] >> 29;
    t[13] &= MASK_29;
    t[15] += t[14] >> 29;
    t[14] &= MASK_29;
    t[16] += t[15] >> 29;
    t[15] &= MASK_29;

    return { (t[0] << 0) | (t[1] << 29) | (t[2] << 58),
             (t[2] >> 6) | (t[3] << 23) | (t[4] << 52),
             (t[4] >> 12) | (t[5] << 17) | (t[6] << 46),
             (t[6] >> 18) | (t[7] << 11) | (t[8] << 40),
             (t[8] >> 24) | (t[9] << 5) | (t[10] << 34) | (t[11] << 63),
             (t[11] >> 1) | (t[12] << 28) | (t[13] << 57),
             (t[13] >> 7) | (t[14] << 22) | (t[15] << 51),
             (t[15] >> 13) | (t[16] << 16) };
}

} // namespace bb::detail

#endif // defined(__wasm__) || !defined(__SIZEOF_INT128__)
