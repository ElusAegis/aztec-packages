#pragma once

// WASM 29-bit integer Montgomery backend.
//
// Provides:
//   - 9×29-bit schoolbook montmul (small modulus, R = 2^261)
//   - 9×29-bit schoolbook montsqr
//   - big-modulus montmul with interleaved reduction
//   - raw 256×256 → 512 wide multiply (R-independent, also usable from FMA)
//
// Active on WASM and any host without __int128.
// Schoolbook paths are fully constexpr (pure integer arithmetic).

#if defined(__wasm__) || !defined(__SIZEOF_INT128__)

#include <array>

#include "../field_declarations.hpp"
#include "../field_montgomery_config.hpp"

namespace bb::detail {

template <class Params> struct WasmInt29Backend {
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

#if BB_R_LIMB_BITS == 29
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
        constexpr uint64_t mask = 0x1fffffff;
        constexpr uint64_t r_inv = Params::r_inv & mask;
        constexpr auto r_limbs = field<Params>::r_limbs;
        uint64_t k = (result_0 * r_inv) & mask;
        result_0 += k * r_limbs.modulus[0];
        result_1 += k * r_limbs.modulus[1] + (result_0 >> R_LIMB_BITS);
        result_2 += k * r_limbs.modulus[2];
        result_3 += k * r_limbs.modulus[3];
        result_4 += k * r_limbs.modulus[4];
        result_5 += k * r_limbs.modulus[5];
        result_6 += k * r_limbs.modulus[6];
        result_7 += k * r_limbs.modulus[7];
        result_8 += k * r_limbs.modulus[8];
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
        constexpr uint64_t mask = 0x1fffffff;
        constexpr auto r_limbs = field<Params>::r_limbs;
        const uint64_t result_0_masked = result_0 & mask;
        result_1 += result_0_masked * r_limbs.div_r_inv[0] + (result_0 >> R_LIMB_BITS);
        result_2 += result_0_masked * r_limbs.div_r_inv[1];
        result_3 += result_0_masked * r_limbs.div_r_inv[2];
        result_4 += result_0_masked * r_limbs.div_r_inv[3];
        result_5 += result_0_masked * r_limbs.div_r_inv[4];
        result_6 += result_0_masked * r_limbs.div_r_inv[5];
        result_7 += result_0_masked * r_limbs.div_r_inv[6];
        result_8 += result_0_masked * r_limbs.div_r_inv[7];
        result_9 += result_0_masked * r_limbs.div_r_inv[8];
    }
#endif // BB_R_LIMB_BITS == 29
};

// ── Schoolbook Montgomery multiplication (9×29-bit) ──────────────────────

#if BB_R_LIMB_BITS == 29

template <class Params>
constexpr field<Params> WasmInt29Backend<Params>::mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    constexpr uint64_t mask = 0x1fffffff;
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

    wasm_reduce_yuval(temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9);
    wasm_reduce_yuval(temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10);
    wasm_reduce_yuval(temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11);
    wasm_reduce_yuval(temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12);
    wasm_reduce_yuval(temp_4, temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13);
    wasm_reduce_yuval(temp_5, temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14);
    wasm_reduce_yuval(temp_6, temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15);
    wasm_reduce_yuval(temp_7, temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    wasm_reduce(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    temp_10 += temp_9 >> R_LIMB_BITS;
    temp_9 &= mask;
    temp_11 += temp_10 >> R_LIMB_BITS;
    temp_10 &= mask;
    temp_12 += temp_11 >> R_LIMB_BITS;
    temp_11 &= mask;
    temp_13 += temp_12 >> R_LIMB_BITS;
    temp_12 &= mask;
    temp_14 += temp_13 >> R_LIMB_BITS;
    temp_13 &= mask;
    temp_15 += temp_14 >> R_LIMB_BITS;
    temp_14 &= mask;
    temp_16 += temp_15 >> R_LIMB_BITS;
    temp_15 &= mask;

    return { (temp_9 << 0) | (temp_10 << 29) | (temp_11 << 58),
             (temp_11 >> 6) | (temp_12 << 23) | (temp_13 << 52),
             (temp_13 >> 12) | (temp_14 << 17) | (temp_15 << 46),
             (temp_15 >> 18) | (temp_16 << 11) };
}

// ── Schoolbook Montgomery squaring (9×29-bit) ────────────────────────────

template <class Params> constexpr field<Params> WasmInt29Backend<Params>::sqr(const field<Params>& x) noexcept
{
    auto left = wasm_convert(x.data);
    constexpr uint64_t mask = 0x1fffffff;
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

    wasm_reduce(temp_8, temp_9, temp_10, temp_11, temp_12, temp_13, temp_14, temp_15, temp_16);

    temp_10 += temp_9 >> R_LIMB_BITS;
    temp_9 &= mask;
    temp_11 += temp_10 >> R_LIMB_BITS;
    temp_10 &= mask;
    temp_12 += temp_11 >> R_LIMB_BITS;
    temp_11 &= mask;
    temp_13 += temp_12 >> R_LIMB_BITS;
    temp_12 &= mask;
    temp_14 += temp_13 >> R_LIMB_BITS;
    temp_13 &= mask;
    temp_15 += temp_14 >> R_LIMB_BITS;
    temp_14 &= mask;
    temp_16 += temp_15 >> R_LIMB_BITS;
    temp_15 &= mask;

    return { (temp_9 << 0) | (temp_10 << 29) | (temp_11 << 58),
             (temp_11 >> 6) | (temp_12 << 23) | (temp_13 << 52),
             (temp_13 >> 12) | (temp_14 << 17) | (temp_15 << 46),
             (temp_15 >> 18) | (temp_16 << 11) };
}

// ── Big-modulus Montgomery multiplication (>= 2^254), WASM 29-bit ────────

template <class Params>
constexpr field<Params> WasmInt29Backend<Params>::mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    static_assert(field<Params>::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);
    constexpr auto r_limbs = field<Params>::r_limbs;

    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    constexpr uint64_t mask = 0x1fffffff;
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

    temp_10 += temp_9 >> R_LIMB_BITS;
    temp_9 &= mask;
    temp_11 += temp_10 >> R_LIMB_BITS;
    temp_10 &= mask;
    temp_12 += temp_11 >> R_LIMB_BITS;
    temp_11 &= mask;
    temp_13 += temp_12 >> R_LIMB_BITS;
    temp_12 &= mask;
    temp_14 += temp_13 >> R_LIMB_BITS;
    temp_13 &= mask;
    temp_15 += temp_14 >> R_LIMB_BITS;
    temp_14 &= mask;
    temp_16 += temp_15 >> R_LIMB_BITS;
    temp_15 &= mask;
    temp_17 += temp_16 >> R_LIMB_BITS;
    temp_16 &= mask;

    uint64_t r_temp_0;
    uint64_t r_temp_1;
    uint64_t r_temp_2;
    uint64_t r_temp_3;
    uint64_t r_temp_4;
    uint64_t r_temp_5;
    uint64_t r_temp_6;
    uint64_t r_temp_7;
    uint64_t r_temp_8;

    r_temp_0 = temp_9 - r_limbs.modulus[0];
    r_temp_1 = temp_10 - r_limbs.modulus[1] - ((r_temp_0) >> 63);
    r_temp_2 = temp_11 - r_limbs.modulus[2] - ((r_temp_1) >> 63);
    r_temp_3 = temp_12 - r_limbs.modulus[3] - ((r_temp_2) >> 63);
    r_temp_4 = temp_13 - r_limbs.modulus[4] - ((r_temp_3) >> 63);
    r_temp_5 = temp_14 - r_limbs.modulus[5] - ((r_temp_4) >> 63);
    r_temp_6 = temp_15 - r_limbs.modulus[6] - ((r_temp_5) >> 63);
    r_temp_7 = temp_16 - r_limbs.modulus[7] - ((r_temp_6) >> 63);
    r_temp_8 = temp_17 - r_limbs.modulus[8] - ((r_temp_7) >> 63);

    uint64_t new_mask = 0 - (r_temp_8 >> 63);
    uint64_t inverse_mask = (~new_mask) & mask;
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

#else // BB_R_LIMB_BITS != 29 — stub mul/sqr/mul_big so the non-29 build doesn't need them

template <class Params>
constexpr field<Params> WasmInt29Backend<Params>::mul(const field<Params>&, const field<Params>&) noexcept
{
    // Not reachable — WasmInt29Backend is only selected when BB_R_LIMB_BITS == 29.
    if (!std::is_constant_evaluated()) {
        __builtin_trap();
    }
    return field<Params>{};
}

template <class Params> constexpr field<Params> WasmInt29Backend<Params>::sqr(const field<Params>&) noexcept
{
    if (!std::is_constant_evaluated()) {
        __builtin_trap();
    }
    return field<Params>{};
}

template <class Params>
constexpr field<Params> WasmInt29Backend<Params>::mul_big(const field<Params>&, const field<Params>&) noexcept
{
    if (!std::is_constant_evaluated()) {
        __builtin_trap();
    }
    return field<Params>{};
}

#endif // BB_R_LIMB_BITS == 29

// ── 256×256 → 512 wide multiply, WASM path (uses 29-bit internally) ──────
// R-independent: pure integer wide multiply. Safe to call from any WASM
// build, including FMA (where R_LIMB_BITS == 24).

template <class Params>
constexpr typename field<Params>::wide_array WasmInt29Backend<Params>::wide_mul(const field<Params>& lhs,
                                                                                const field<Params>& rhs) noexcept
{
    auto left = wasm_convert(lhs.data);
    auto right = wasm_convert(rhs.data);
    constexpr uint64_t mask = 0x1fffffff;
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
    temp_0 &= mask;
    temp_2 += temp_1 >> 29;
    temp_1 &= mask;
    temp_3 += temp_2 >> 29;
    temp_2 &= mask;
    temp_4 += temp_3 >> 29;
    temp_3 &= mask;
    temp_5 += temp_4 >> 29;
    temp_4 &= mask;
    temp_6 += temp_5 >> 29;
    temp_5 &= mask;
    temp_7 += temp_6 >> 29;
    temp_6 &= mask;
    temp_8 += temp_7 >> 29;
    temp_7 &= mask;
    temp_9 += temp_8 >> 29;
    temp_8 &= mask;
    temp_10 += temp_9 >> 29;
    temp_9 &= mask;
    temp_11 += temp_10 >> 29;
    temp_10 &= mask;
    temp_12 += temp_11 >> 29;
    temp_11 &= mask;
    temp_13 += temp_12 >> 29;
    temp_12 &= mask;
    temp_14 += temp_13 >> 29;
    temp_13 &= mask;
    temp_15 += temp_14 >> 29;
    temp_14 &= mask;
    temp_16 += temp_15 >> 29;
    temp_15 &= mask;

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
