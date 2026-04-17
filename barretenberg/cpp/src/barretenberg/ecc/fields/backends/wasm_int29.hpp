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

    BB_INLINE static constexpr void wasm_madd(uint64_t& left_limb,
                                              const std::array<uint64_t, 9>& right_limbs,
                                              uint64_t& result_0,
                                              uint64_t& result_1,
                                              uint64_t& result_2,
                                              uint64_t& result_3,
                                              uint64_t& result_4,
                                              uint64_t& result_5,
                                              uint64_t& result_6,
                                              uint64_t& result_7,
                                              uint64_t& result_8)
    {
        result_0 += left_limb * right_limbs[0];
        result_1 += left_limb * right_limbs[1];
        result_2 += left_limb * right_limbs[2];
        result_3 += left_limb * right_limbs[3];
        result_4 += left_limb * right_limbs[4];
        result_5 += left_limb * right_limbs[5];
        result_6 += left_limb * right_limbs[6];
        result_7 += left_limb * right_limbs[7];
        result_8 += left_limb * right_limbs[8];
    }

    // Standard Montgomery reduction step with a 29-bit `k`. Used for the
    // final step of the R = 2^261 kernel (9×29-bit total) and for the
    // interleaved reduction of mul_big.
    BB_INLINE static constexpr void wasm_reduce(uint64_t& result_0,
                                                uint64_t& result_1,
                                                uint64_t& result_2,
                                                uint64_t& result_3,
                                                uint64_t& result_4,
                                                uint64_t& result_5,
                                                uint64_t& result_6,
                                                uint64_t& result_7,
                                                uint64_t& result_8)
    {
        constexpr uint64_t r_inv_29 = Params::r_inv & MASK_29;
        uint64_t k = (result_0 * r_inv_29) & MASK_29;
        result_0 += k * r_limbs_29.modulus[0];
        result_1 += k * r_limbs_29.modulus[1] + (result_0 >> LIMB_BITS_29);
        result_2 += k * r_limbs_29.modulus[2];
        result_3 += k * r_limbs_29.modulus[3];
        result_4 += k * r_limbs_29.modulus[4];
        result_5 += k * r_limbs_29.modulus[5];
        result_6 += k * r_limbs_29.modulus[6];
        result_7 += k * r_limbs_29.modulus[7];
        result_8 += k * r_limbs_29.modulus[8];
    }

    // Final Montgomery reduction step with a 32-bit `k`. Used only in the
    // REXP = 264 path: after 8 Yuval (29-bit) steps the accumulator has been
    // divided by 2^232; dividing by another 2^32 brings the total to 2^264,
    // matching the FMA SIMD backend's R.
    //
    // `k` spans result_0's 29 bits plus result_1's low 3 bits (29 + 3 = 32).
    // U64 safety: k ≤ 2^32 − 1, modulus_29[i] ≤ 2^29 − 1, so k·modulus_29[i]
    // ≤ 2^61 − …; post-Yuval accumulator limbs are ≤ 2^62; sum well below 2^64.
    BB_INLINE static constexpr void wasm_reduce_32bit_k(uint64_t& result_0,
                                                        uint64_t& result_1,
                                                        uint64_t& result_2,
                                                        uint64_t& result_3,
                                                        uint64_t& result_4,
                                                        uint64_t& result_5,
                                                        uint64_t& result_6,
                                                        uint64_t& result_7,
                                                        uint64_t& result_8)
    {
        constexpr uint64_t MASK_32 = 0xffffffffULL;
        // -p^{-1} mod 2^32: Params::r_inv is -p^{-1} mod 2^64; the low 32
        // bits are the same value mod 2^32.
        constexpr uint64_t r_inv_32 = Params::r_inv & MASK_32;

        // V mod 2^32 spans result_0's 29 bits + result_1's low 3 bits.
        const uint64_t v_low = ((result_0 & MASK_32) + ((result_1 & 0x7) << LIMB_BITS_29)) & MASK_32;
        const uint64_t k = (v_low * r_inv_32) & MASK_32;

        result_0 += k * r_limbs_29.modulus[0];
        result_1 += k * r_limbs_29.modulus[1] + (result_0 >> LIMB_BITS_29);
        result_2 += k * r_limbs_29.modulus[2];
        result_3 += k * r_limbs_29.modulus[3];
        result_4 += k * r_limbs_29.modulus[4];
        result_5 += k * r_limbs_29.modulus[5];
        result_6 += k * r_limbs_29.modulus[6];
        result_7 += k * r_limbs_29.modulus[7];
        result_8 += k * r_limbs_29.modulus[8];
    }

    BB_INLINE static constexpr void wasm_reduce_yuval(uint64_t& result_0,
                                                      uint64_t& result_1,
                                                      uint64_t& result_2,
                                                      uint64_t& result_3,
                                                      uint64_t& result_4,
                                                      uint64_t& result_5,
                                                      uint64_t& result_6,
                                                      uint64_t& result_7,
                                                      uint64_t& result_8,
                                                      uint64_t& result_9)
    {
        const uint64_t result_0_masked = result_0 & MASK_29;
        result_1 += result_0_masked * r_limbs_29.div_r_inv[0] + (result_0 >> LIMB_BITS_29);
        result_2 += result_0_masked * r_limbs_29.div_r_inv[1];
        result_3 += result_0_masked * r_limbs_29.div_r_inv[2];
        result_4 += result_0_masked * r_limbs_29.div_r_inv[3];
        result_5 += result_0_masked * r_limbs_29.div_r_inv[4];
        result_6 += result_0_masked * r_limbs_29.div_r_inv[5];
        result_7 += result_0_masked * r_limbs_29.div_r_inv[6];
        result_8 += result_0_masked * r_limbs_29.div_r_inv[7];
        result_9 += result_0_masked * r_limbs_29.div_r_inv[8];
    }
};

// ── Karatsuba Montgomery multiplication (9×29-bit, 5+4 split, 66 muls) ───

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

    // Combine: temp[k] = P_lo[k] + P_mid[k-5] + P_hi[k-10]
    // where P_mid = P_cross - P_lo - P_hi
    uint64_t temp_0 = pl0;
    uint64_t temp_1 = pl1;
    uint64_t temp_2 = pl2;
    uint64_t temp_3 = pl3;
    uint64_t temp_4 = pl4;
    uint64_t temp_5 = pl5 + (pc0 - pl0 - ph0);
    uint64_t temp_6 = pl6 + (pc1 - pl1 - ph1);
    uint64_t temp_7 = pl7 + (pc2 - pl2 - ph2);
    uint64_t temp_8 = pl8 + (pc3 - pl3 - ph3);
    uint64_t temp_9 = pc4 - pl4 - ph4;
    uint64_t temp_10 = (pc5 - pl5 - ph5) + ph0;
    uint64_t temp_11 = (pc6 - pl6 - ph6) + ph1;
    uint64_t temp_12 = (pc7 - pl7) + ph2;
    uint64_t temp_13 = (pc8 - pl8) + ph3;
    uint64_t temp_14 = ph4;
    uint64_t temp_15 = ph5;
    uint64_t temp_16 = ph6;

    wasm_reduce_yuval(temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_reduce_yuval(temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_reduce_yuval(temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_reduce_yuval(temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_reduce_yuval(temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_reduce_yuval(temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_reduce_yuval(temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_reduce_yuval(temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    if constexpr (REXP == 261) {
        wasm_reduce(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

        temp_10 += temp_9 >> LIMB_BITS_29;
        temp_9 &= MASK_29;
        temp_11 += temp_10 >> LIMB_BITS_29;
        temp_10 &= MASK_29;
        temp_12 += temp_11 >> LIMB_BITS_29;
        temp_11 &= MASK_29;
        temp_13 += temp_12 >> LIMB_BITS_29;
        temp_12 &= MASK_29;
        temp_14 += temp_13 >> LIMB_BITS_29;
        temp_13 &= MASK_29;
        temp_15 += temp_14 >> LIMB_BITS_29;
        temp_14 &= MASK_29;
        temp_16 += temp_15 >> LIMB_BITS_29;
        temp_15 &= MASK_29;

        return { (temp_9 << 0) | (temp_10 << 29) | (temp_11 << 58),
                 (temp_11 >> 6) | (temp_12 << 23) | (temp_13 << 52),
                 (temp_13 >> 12) | (temp_14 << 17) | (temp_15 << 46),
                 (temp_15 >> 18) | (temp_16 << 11) };
    } else {
        // REXP = 264: widen the final step to 32 bits so total reduction is
        // 8·29 + 32 = 264. The "virtual LSB" of the output lives at bit 3
        // of temp_9 (equivalently bit 32 of temp_8 in the old frame) and
        // the pack offsets shift from {0,29,58,…} to {0,26,55,84,…}.
        wasm_reduce_32bit_k(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

        // Carry-propagate through temp_9..temp_16 (same chain as REXP=261).
        temp_10 += temp_9 >> LIMB_BITS_29;
        temp_9 &= MASK_29;
        temp_11 += temp_10 >> LIMB_BITS_29;
        temp_10 &= MASK_29;
        temp_12 += temp_11 >> LIMB_BITS_29;
        temp_11 &= MASK_29;
        temp_13 += temp_12 >> LIMB_BITS_29;
        temp_12 &= MASK_29;
        temp_14 += temp_13 >> LIMB_BITS_29;
        temp_13 &= MASK_29;
        temp_15 += temp_14 >> LIMB_BITS_29;
        temp_14 &= MASK_29;
        temp_16 += temp_15 >> LIMB_BITS_29;
        temp_15 &= MASK_29;

        // Virtual LSB at bit 3 of temp_9 → offsets {0, 26, 55, 84, 113, 142, 171, 200}.
        return { (temp_9 >> 3) | (temp_10 << 26) | (temp_11 << 55),
                 (temp_11 >> 9) | (temp_12 << 20) | (temp_13 << 49),
                 (temp_13 >> 15) | (temp_14 << 14) | (temp_15 << 43),
                 (temp_15 >> 21) | (temp_16 << 8) };
    }
}

// ── Schoolbook Montgomery squaring (9×29-bit) ────────────────────────────

template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::sqr(const field<Params>& x) noexcept
{
    auto left = wasm_convert(x.data);
    uint64_t temp_0 = 0;
    uint64_t temp_1 = 0;
    uint64_t temp_2 = 0;
    uint64_t temp_3 = 0;
    uint64_t temp_4 = 0;
    uint64_t temp_5 = 0;
    uint64_t temp_6 = 0;
    uint64_t temp_7 = 0;
    uint64_t temp_8 = 0;
    uint64_t temp_9 = 0;
    uint64_t temp_10 = 0;
    uint64_t temp_11 = 0;
    uint64_t temp_12 = 0;
    uint64_t temp_13 = 0;
    uint64_t temp_14 = 0;
    uint64_t temp_15 = 0;
    uint64_t temp_16 = 0;
    uint64_t acc;

    temp_0 += left[0] * left[0];
    acc = 0;
    acc += left[0] * left[1];
    temp_1 += (acc << 1);
    acc = 0;
    acc += left[0] * left[2];
    temp_2 += left[1] * left[1];
    temp_2 += (acc << 1);
    acc = 0;
    acc += left[0] * left[3];
    acc += left[1] * left[2];
    temp_3 += (acc << 1);
    acc = 0;
    acc += left[0] * left[4];
    acc += left[1] * left[3];
    temp_4 += left[2] * left[2];
    temp_4 += (acc << 1);
    acc = 0;
    acc += left[0] * left[5];
    acc += left[1] * left[4];
    acc += left[2] * left[3];
    temp_5 += (acc << 1);
    acc = 0;
    acc += left[0] * left[6];
    acc += left[1] * left[5];
    acc += left[2] * left[4];
    temp_6 += left[3] * left[3];
    temp_6 += (acc << 1);
    acc = 0;
    acc += left[0] * left[7];
    acc += left[1] * left[6];
    acc += left[2] * left[5];
    acc += left[3] * left[4];
    temp_7 += (acc << 1);
    acc = 0;
    acc += left[0] * left[8];
    acc += left[1] * left[7];
    acc += left[2] * left[6];
    acc += left[3] * left[5];
    temp_8 += left[4] * left[4];
    temp_8 += (acc << 1);
    acc = 0;
    acc += left[1] * left[8];
    acc += left[2] * left[7];
    acc += left[3] * left[6];
    acc += left[4] * left[5];
    temp_9 += (acc << 1);
    acc = 0;
    acc += left[2] * left[8];
    acc += left[3] * left[7];
    acc += left[4] * left[6];
    temp_10 += left[5] * left[5];
    temp_10 += (acc << 1);
    acc = 0;
    acc += left[3] * left[8];
    acc += left[4] * left[7];
    acc += left[5] * left[6];
    temp_11 += (acc << 1);
    acc = 0;
    acc += left[4] * left[8];
    acc += left[5] * left[7];
    temp_12 += left[6] * left[6];
    temp_12 += (acc << 1);
    acc = 0;
    acc += left[5] * left[8];
    acc += left[6] * left[7];
    temp_13 += (acc << 1);
    acc = 0;
    acc += left[6] * left[8];
    temp_14 += left[7] * left[7];
    temp_14 += (acc << 1);
    acc = 0;
    acc += left[7] * left[8];
    temp_15 += (acc << 1);
    temp_16 += left[8] * left[8];

    wasm_reduce_yuval(temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_reduce_yuval(temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_reduce_yuval(temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_reduce_yuval(temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_reduce_yuval(temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_reduce_yuval(temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_reduce_yuval(temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_reduce_yuval(temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    if constexpr (REXP == 261) {
        wasm_reduce(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

        temp_10 += temp_9 >> LIMB_BITS_29;
        temp_9 &= MASK_29;
        temp_11 += temp_10 >> LIMB_BITS_29;
        temp_10 &= MASK_29;
        temp_12 += temp_11 >> LIMB_BITS_29;
        temp_11 &= MASK_29;
        temp_13 += temp_12 >> LIMB_BITS_29;
        temp_12 &= MASK_29;
        temp_14 += temp_13 >> LIMB_BITS_29;
        temp_13 &= MASK_29;
        temp_15 += temp_14 >> LIMB_BITS_29;
        temp_14 &= MASK_29;
        temp_16 += temp_15 >> LIMB_BITS_29;
        temp_15 &= MASK_29;

        return { (temp_9 << 0) | (temp_10 << 29) | (temp_11 << 58),
                 (temp_11 >> 6) | (temp_12 << 23) | (temp_13 << 52),
                 (temp_13 >> 12) | (temp_14 << 17) | (temp_15 << 46),
                 (temp_15 >> 18) | (temp_16 << 11) };
    } else {
        wasm_reduce_32bit_k(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

        temp_10 += temp_9 >> LIMB_BITS_29;
        temp_9 &= MASK_29;
        temp_11 += temp_10 >> LIMB_BITS_29;
        temp_10 &= MASK_29;
        temp_12 += temp_11 >> LIMB_BITS_29;
        temp_11 &= MASK_29;
        temp_13 += temp_12 >> LIMB_BITS_29;
        temp_12 &= MASK_29;
        temp_14 += temp_13 >> LIMB_BITS_29;
        temp_13 &= MASK_29;
        temp_15 += temp_14 >> LIMB_BITS_29;
        temp_14 &= MASK_29;
        temp_16 += temp_15 >> LIMB_BITS_29;
        temp_15 &= MASK_29;

        return { (temp_9 >> 3) | (temp_10 << 26) | (temp_11 << 55),
                 (temp_11 >> 9) | (temp_12 << 20) | (temp_13 << 49),
                 (temp_13 >> 15) | (temp_14 << 14) | (temp_15 << 43),
                 (temp_15 >> 21) | (temp_16 << 8) };
    }
}

// ── Big-modulus Montgomery multiplication (>= 2^254), WASM 29-bit ────────
//
// mul_big interleaves reduction with the product phase (9 madd + 9 reduce
// in lockstep) rather than doing the full product then the full reduction.
// That chain *cannot* share the REXP = 264 shortcut: widening the final step
// to 32 bits only works when the preceding Yuval chain divides by 2^232 in
// one pass. The interleaved path divides by 2^29 after every madd, so all
// nine steps must be 29-bit. mul_big is therefore only valid for REXP = 261;
// the FMA backend uses its own constexpr_mont_mul-based big-mul path and
// never calls mul_big on a 264 instantiation.
template <class Params, unsigned REXP>
constexpr field<Params> WasmInt29Backend<Params, REXP>::mul_big(const field<Params>& lhs,
                                                                const field<Params>& rhs) noexcept
{
    static_assert(REXP == 261, "mul_big is only valid at REXP = 261 (native 29-bit reduction chain)");
    static_assert(field<Params>::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);

    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    uint64_t temp_0 = 0;
    uint64_t temp_1 = 0;
    uint64_t temp_2 = 0;
    uint64_t temp_3 = 0;
    uint64_t temp_4 = 0;
    uint64_t temp_5 = 0;
    uint64_t temp_6 = 0;
    uint64_t temp_7 = 0;
    uint64_t temp_8 = 0;
    uint64_t temp_9 = 0;
    uint64_t temp_10 = 0;
    uint64_t temp_11 = 0;
    uint64_t temp_12 = 0;
    uint64_t temp_13 = 0;
    uint64_t temp_14 = 0;
    uint64_t temp_15 = 0;
    uint64_t temp_16 = 0;
    uint64_t temp_17 = 0;

    wasm_madd(left[0], right, temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8);
    wasm_reduce(temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8);
    wasm_madd(left[1], right, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_reduce(temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_madd(left[2], right, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_reduce(temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_madd(left[3], right, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_reduce(temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_madd(left[4], right, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_reduce(temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_madd(left[5], right, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_reduce(temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_madd(left[6], right, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_reduce(temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_madd(left[7], right, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_reduce(temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_madd(left[8], right, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);
    wasm_reduce(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    temp_10 += temp_9 >> LIMB_BITS_29;
    temp_9 &= MASK_29;
    temp_11 += temp_10 >> LIMB_BITS_29;
    temp_10 &= MASK_29;
    temp_12 += temp_11 >> LIMB_BITS_29;
    temp_11 &= MASK_29;
    temp_13 += temp_12 >> LIMB_BITS_29;
    temp_12 &= MASK_29;
    temp_14 += temp_13 >> LIMB_BITS_29;
    temp_13 &= MASK_29;
    temp_15 += temp_14 >> LIMB_BITS_29;
    temp_14 &= MASK_29;
    temp_16 += temp_15 >> LIMB_BITS_29;
    temp_15 &= MASK_29;
    temp_17 += temp_16 >> LIMB_BITS_29;
    temp_16 &= MASK_29;

    uint64_t r_temp_0;
    uint64_t r_temp_1;
    uint64_t r_temp_2;
    uint64_t r_temp_3;
    uint64_t r_temp_4;
    uint64_t r_temp_5;
    uint64_t r_temp_6;
    uint64_t r_temp_7;
    uint64_t r_temp_8;

    r_temp_0 = temp_9 - r_limbs_29.modulus[0];
    r_temp_1 = temp_10 - r_limbs_29.modulus[1] - ((r_temp_0) >> 63);
    r_temp_2 = temp_11 - r_limbs_29.modulus[2] - ((r_temp_1) >> 63);
    r_temp_3 = temp_12 - r_limbs_29.modulus[3] - ((r_temp_2) >> 63);
    r_temp_4 = temp_13 - r_limbs_29.modulus[4] - ((r_temp_3) >> 63);
    r_temp_5 = temp_14 - r_limbs_29.modulus[5] - ((r_temp_4) >> 63);
    r_temp_6 = temp_15 - r_limbs_29.modulus[6] - ((r_temp_5) >> 63);
    r_temp_7 = temp_16 - r_limbs_29.modulus[7] - ((r_temp_6) >> 63);
    r_temp_8 = temp_17 - r_limbs_29.modulus[8] - ((r_temp_7) >> 63);

    uint64_t new_mask = 0 - (r_temp_8 >> 63);
    uint64_t inverse_mask = (~new_mask) & MASK_29;
    temp_9 = (temp_9 & new_mask) | (r_temp_0 & inverse_mask);
    temp_10 = (temp_10 & new_mask) | (r_temp_1 & inverse_mask);
    temp_11 = (temp_11 & new_mask) | (r_temp_2 & inverse_mask);
    temp_12 = (temp_12 & new_mask) | (r_temp_3 & inverse_mask);
    temp_13 = (temp_13 & new_mask) | (r_temp_4 & inverse_mask);
    temp_14 = (temp_14 & new_mask) | (r_temp_5 & inverse_mask);
    temp_15 = (temp_15 & new_mask) | (r_temp_6 & inverse_mask);
    temp_16 = (temp_16 & new_mask) | (r_temp_7 & inverse_mask);
    temp_17 = (temp_17 & new_mask) | (r_temp_8 & inverse_mask);

    return { (temp_9 << 0) | (temp_10 << 29) | (temp_11 << 58),
             (temp_11 >> 6) | (temp_12 << 23) | (temp_13 << 52),
             (temp_13 >> 12) | (temp_14 << 17) | (temp_15 << 46),
             (temp_15 >> 18) | (temp_16 << 11) | (temp_17 << 40) };
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
    uint64_t temp_0 = 0;
    uint64_t temp_1 = 0;
    uint64_t temp_2 = 0;
    uint64_t temp_3 = 0;
    uint64_t temp_4 = 0;
    uint64_t temp_5 = 0;
    uint64_t temp_6 = 0;
    uint64_t temp_7 = 0;
    uint64_t temp_8 = 0;
    uint64_t temp_9 = 0;
    uint64_t temp_10 = 0;
    uint64_t temp_11 = 0;
    uint64_t temp_12 = 0;
    uint64_t temp_13 = 0;
    uint64_t temp_14 = 0;
    uint64_t temp_15 = 0;
    uint64_t temp_16 = 0;

    wasm_madd(left[0], right, temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8);
    wasm_madd(left[1], right, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_madd(left[2], right, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_madd(left[3], right, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_madd(left[4], right, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_madd(left[5], right, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_madd(left[6], right, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_madd(left[7], right, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_madd(left[8], right, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    temp_1 += temp_0 >> 29;
    temp_0 &= MASK_29;
    temp_2 += temp_1 >> 29;
    temp_1 &= MASK_29;
    temp_3 += temp_2 >> 29;
    temp_2 &= MASK_29;
    temp_4 += temp_3 >> 29;
    temp_3 &= MASK_29;
    temp_5 += temp_4 >> 29;
    temp_4 &= MASK_29;
    temp_6 += temp_5 >> 29;
    temp_5 &= MASK_29;
    temp_7 += temp_6 >> 29;
    temp_6 &= MASK_29;
    temp_8 += temp_7 >> 29;
    temp_7 &= MASK_29;
    temp_9 += temp_8 >> 29;
    temp_8 &= MASK_29;
    temp_10 += temp_9 >> 29;
    temp_9 &= MASK_29;
    temp_11 += temp_10 >> 29;
    temp_10 &= MASK_29;
    temp_12 += temp_11 >> 29;
    temp_11 &= MASK_29;
    temp_13 += temp_12 >> 29;
    temp_12 &= MASK_29;
    temp_14 += temp_13 >> 29;
    temp_13 &= MASK_29;
    temp_15 += temp_14 >> 29;
    temp_14 &= MASK_29;
    temp_16 += temp_15 >> 29;
    temp_15 &= MASK_29;

    return { (temp_0 << 0) | (temp_1 << 29) | (temp_2 << 58),
             (temp_2 >> 6) | (temp_3 << 23) | (temp_4 << 52),
             (temp_4 >> 12) | (temp_5 << 17) | (temp_6 << 46),
             (temp_6 >> 18) | (temp_7 << 11) | (temp_8 << 40),
             (temp_8 >> 24) | (temp_9 << 5) | (temp_10 << 34) | (temp_11 << 63),
             (temp_11 >> 1) | (temp_12 << 28) | (temp_13 << 57),
             (temp_13 >> 7) | (temp_14 << 22) | (temp_15 << 51),
             (temp_15 >> 13) | (temp_16 << 16) };
}

} // namespace bb::detail

#endif // defined(__wasm__) || !defined(__SIZEOF_INT128__)
