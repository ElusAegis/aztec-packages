// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Raju], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include <array>
#include <cstdint>

#include "./field_impl.hpp"
#include "barretenberg/common/bb_bench.hpp"

namespace bb {

// NOLINTBEGIN(readability-implicit-bool-conversion)

// ══════════════════════════════════════════════════════════════════════
// Universal helpers — used by field arithmetic (add/sub/reduce) and by
// native montgomery_mul_big. Both have internal native/WASM branches.
// ══════════════════════════════════════════════════════════════════════

template <class T>
constexpr uint64_t field<T>::addc(const uint64_t a,
                                  const uint64_t b,
                                  const uint64_t carry_in,
                                  uint64_t& carry_out) noexcept
{
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    uint128_t res = static_cast<uint128_t>(a) + static_cast<uint128_t>(b) + static_cast<uint128_t>(carry_in);
    carry_out = static_cast<uint64_t>(res >> 64);
    return static_cast<uint64_t>(res);
#else
    uint64_t r = a + b;
    const uint64_t carry_temp = r < a;
    r += carry_in;
    carry_out = carry_temp + (r < carry_in);
    return r;
#endif
}

template <class T>
constexpr uint64_t field<T>::sbb(const uint64_t a,
                                 const uint64_t b,
                                 const uint64_t borrow_in,
                                 uint64_t& borrow_out) noexcept
{
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    uint128_t res = static_cast<uint128_t>(a) - (static_cast<uint128_t>(b) + static_cast<uint128_t>(borrow_in >> 63));
    borrow_out = static_cast<uint64_t>(res >> 64);
    return static_cast<uint64_t>(res);
#else
    uint64_t t_1 = a - (borrow_in >> 63ULL);
    uint64_t borrow_temp_1 = t_1 > a;
    uint64_t t_2 = t_1 - b;
    uint64_t borrow_temp_2 = t_2 > t_1;
    borrow_out = 0ULL - (borrow_temp_1 | borrow_temp_2);
    return t_2;
#endif
}

// ══════════════════════════════════════════════════════════════════════
// Field arithmetic — reduce, add, subtract
// ══════════════════════════════════════════════════════════════════════

template <class T> constexpr field<T> field<T>::reduce() const noexcept
{
    if constexpr (modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        uint256_t val{ data[0], data[1], data[2], data[3] };
        if (val >= modulus) {
            val -= modulus;
        }
        return { val.data[0], val.data[1], val.data[2], val.data[3] };
    }
    // not_modulus == 2^256 - modulus
    // do limb-based add-and-carry with `not_modulus`. this yields a _constant-time_ algorithm.
    uint64_t t0 = data[0] + not_modulus.data[0];
    uint64_t c = t0 < data[0];
    auto t1 = addc(data[1], not_modulus.data[1], c, c);
    auto t2 = addc(data[2], not_modulus.data[2], c, c);
    auto t3 = addc(data[3], not_modulus.data[3], c, c);
    // c != 0 iff val >= modulus.
    const uint64_t selection_mask = 0ULL - c; // 0xffffffff if we have overflowed.
    const uint64_t selection_mask_inverse = ~selection_mask;
    // if c == 0, then the original element is already reduced. if we overflow, we want to return the element whose
    // limbs are {t0, t1, t2, t3}.
    return {
        (data[0] & selection_mask_inverse) | (t0 & selection_mask),
        (data[1] & selection_mask_inverse) | (t1 & selection_mask),
        (data[2] & selection_mask_inverse) | (t2 & selection_mask),
        (data[3] & selection_mask_inverse) | (t3 & selection_mask),
    };
}

///////////////////////
///// ADD and SUB /////
///////////////////////

// Both `add` and `sub` use constexpr branching to distinguish the cases: modulus has <= 254 bits (fields associated to
// BN-254) and modulus has 256 bits. The former has the so-called "coarse" optimization: we allow the inputs to be in
// the range [0, 2p) and the outputs will similarly only be constrained to [0, 2p)

template <class T> constexpr field<T> field<T>::add(const field& other) const noexcept
{
    if constexpr (modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        uint64_t r0 = data[0] + other.data[0];
        uint64_t c = r0 < data[0];
        auto r1 = addc(data[1], other.data[1], c, c);
        auto r2 = addc(data[2], other.data[2], c, c);
        auto r3 = addc(data[3], other.data[3], c, c);
        if (c) {
            uint64_t b = 0;
            r0 = sbb(r0, modulus.data[0], b, b);
            r1 = sbb(r1, modulus.data[1], b, b);
            r2 = sbb(r2, modulus.data[2], b, b);
            r3 = sbb(r3, modulus.data[3], b, b);
            // Since both values are in [0, 2^256), the result is in [0, 2^257-2). Subtracting one p might not
            // be enough. We need to ensure that we've underflown the 0 and that might require subtracting an additional
            // p. This can only happen if at least one of the two arguments has uint256_t-element (derived from limbs)
            // LARGER than p (i.e., non-reduced).
            if (!b) {
                b = 0;
                r0 = sbb(r0, modulus.data[0], b, b);
                r1 = sbb(r1, modulus.data[1], b, b);
                r2 = sbb(r2, modulus.data[2], b, b);
                r3 = sbb(r3, modulus.data[3], b, b);
            }
        }
        // if c != 0, i.e., if there was no carry, we do no additional processing. Note that this means that the output
        // might be larger than p, even if the original self and other were in the range [0, p). This is witnessed in
        // the test AddYieldsLimbsBiggerThanModulus.
        return { r0, r1, r2, r3 };
    } else {
        uint64_t r0 = data[0] + other.data[0];
        uint64_t c = r0 < data[0];
        auto r1 = addc(data[1], other.data[1], c, c);
        auto r2 = addc(data[2], other.data[2], c, c);
        uint64_t r3 = data[3] + other.data[3] +
                      c; // in the small modulus branch so this will satisfy the right size bounds: both self
                         // and other are in the range [0, 2p), which means their sum is in [0, 4p-1).

        uint64_t t0 = r0 + twice_not_modulus.data[0];
        c = t0 < twice_not_modulus.data[0];
        uint64_t t1 = addc(r1, twice_not_modulus.data[1], c, c);
        uint64_t t2 = addc(r2, twice_not_modulus.data[2], c, c);
        uint64_t t3 = addc(r3, twice_not_modulus.data[3], c, c);
        // c == 1 iff self + other >= 2 * p.
        // if c == 0, then return the r_i (naive sum still in coarse form), if c == 1, return the t_i.
        const uint64_t selection_mask = 0ULL - c;
        const uint64_t selection_mask_inverse = ~selection_mask;

        field result{
            (r0 & selection_mask_inverse) | (t0 & selection_mask),
            (r1 & selection_mask_inverse) | (t1 & selection_mask),
            (r2 & selection_mask_inverse) | (t2 & selection_mask),
            (r3 & selection_mask_inverse) | (t3 & selection_mask),
        };
        if (!std::is_constant_evaluated()) {
            result.assert_coarse_form();
        }
        return result;
    }
}

template <class T> constexpr field<T> field<T>::subtract(const field& other) const noexcept
{
    uint64_t borrow = 0;
    uint64_t r0 = sbb(data[0], other.data[0], borrow, borrow);
    uint64_t r1 = sbb(data[1], other.data[1], borrow, borrow);
    uint64_t r2 = sbb(data[2], other.data[2], borrow, borrow);
    uint64_t r3 = sbb(data[3], other.data[3], borrow, borrow);

    // recall that borrow is in the size-2 set {0, 2^64 - 1}.
    if constexpr (modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        // add the modulus if borrow != 0, i.e., if other > self as uint256_t.
        r0 += (modulus.data[0] & borrow);
        uint64_t carry = r0 < (modulus.data[0] & borrow);
        r1 = addc(r1, modulus.data[1] & borrow, carry, carry);
        r2 = addc(r2, modulus.data[2] & borrow, carry, carry);
        r3 = addc(r3, modulus.data[3] & borrow, carry, carry);
        // The value being subtracted is in [0, 2^256); it is possible that adding one copy of
        // p still leaves us with a negative number. To check if we might need to add another copy of p, we check if
        // `carry == 0`; this means that (if we are "in the borrow branch"), the addition did not 2^256-overflow, which
        // means we are still negative. If we not in the borrow branch (i.e., if `borrow == 0`), `carry == 0` and we add
        // nothing using the
        // `& borrow` trick for the `addc` argument.
        if (!carry) {
            r0 += (modulus.data[0] & borrow);
            uint64_t carry = r0 < (modulus.data[0] & borrow);
            r1 = addc(r1, modulus.data[1] & borrow, carry, carry);
            r2 = addc(r2, modulus.data[2] & borrow, carry, carry);
            r3 = addc(r3, (modulus.data[3] & borrow), carry, carry);
        }
        return { r0, r1, r2, r3 };
    }
    // Recall that in this constexpr branch, we use _coarse representation_, meaning the underlying limbs of both self
    // and other yield uint256_t are in [0, 2p) . If there is a borrow, then it is possible that adding one copy of p
    // is insufficient to make the result positive (and adding two copies both preserves the residue mod p and keeps us
    // in the coarse-range).
    r0 += (twice_modulus.data[0] & borrow);
    uint64_t carry = r0 < (twice_modulus.data[0] & borrow);
    r1 = addc(r1, twice_modulus.data[1] & borrow, carry, carry);
    r2 = addc(r2, twice_modulus.data[2] & borrow, carry, carry);
    r3 += (twice_modulus.data[3] & borrow) + carry;

    field result{ r0, r1, r2, r3 };
    if (!std::is_constant_evaluated()) {
        result.assert_coarse_form();
    }
    return result;
}

// ══════════════════════════════════════════════════════════════════════
// Include compute paradigm files (each self-guards via preprocessor).
// Order matters: paradigm files may use addc/sbb defined above.
// ══════════════════════════════════════════════════════════════════════

#include "platform_variants/native_int128.hpp"
#include "platform_variants/wasm_int29.hpp"
#include "platform_variants/wasm_fma_simd.hpp"
#include "platform_variants/constexpr_fallback.hpp"

// ══════════════════════════════════════════════════════════════════════
// Dispatch — select paradigm based on platform/flags
// ══════════════════════════════════════════════════════════════════════

/**
 * @brief Unified Montgomery multiplication dispatch.
 *
 * Single entry point for all platforms: large-modulus, x86 asm, native CIOS, WASM variants.
 * operator* and operator*= are trivial one-liners that delegate here.
 */
template <class T> constexpr field<T> field<T>::montgomery_mul(const field& other) const noexcept
{
    if constexpr (modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        return montgomery_mul_big(other);
    }
#if BBERG_NO_ASM == 0
    if constexpr (!use_generic_arithmetic) {
        if (!std::is_constant_evaluated()) {
            field result = asm_mul_with_coarse_reduction(*this, other);
            result.assert_coarse_form();
            return result;
        }
    }
#endif
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    return montgomery_mul_native_cios(other);
#elif defined(MONTMUL_VARIANT_FMA)
    if (std::is_constant_evaluated()) {
        return montgomery_mul_constexpr_fallback(other);
    }
    return montgomery_mul_wasm_fma_simd(other);
#else
    return montgomery_mul_wasm_standard(other);
#endif
}

/**
 * @brief Unified Montgomery squaring dispatch.
 */
template <class T> constexpr field<T> field<T>::montgomery_square() const noexcept
{
    if constexpr (modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        return montgomery_mul_big(*this);
    }
#if BBERG_NO_ASM == 0
    if constexpr (!use_generic_arithmetic) {
        if (!std::is_constant_evaluated()) {
            field result = asm_sqr_with_coarse_reduction(*this);
            result.assert_coarse_form();
            return result;
        }
    }
#endif
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    return montgomery_square_native_cios();
#elif defined(MONTMUL_VARIANT_FMA)
    // TODO: implement dedicated FMA squaring (Karatsuba square saves ~30% FMAs)
    return montgomery_mul(*this);
#else
    return montgomery_square_wasm_standard();
#endif
}

/**
 * @brief Paired Montgomery multiplication: out1 = a1*b1, out2 = a2*b2.
 *
 * On FMA SIMD platforms, uses the native SIMD2 paired multiply for better throughput.
 * On all other platforms, falls back to two sequential multiplications.
 */
template <class T>
constexpr void field<T>::montgomery_mul_paired(
    const field& a1, const field& b1,
    const field& a2, const field& b2,
    field& out1, field& out2) noexcept
{
#if defined(MONTMUL_VARIANT_FMA) && defined(__wasm_simd128__)
    if (!std::is_constant_evaluated()) {
        montgomery_mul_wasm_fma_simd2(a1, b1, a2, b2, out1, out2);
        return;
    }
#endif
    out1 = a1.montgomery_mul(b1);
    out2 = a2.montgomery_mul(b2);
}

/**
 * @brief Montgomery multiplication for moduli >= 2^254.
 * Dispatches to paradigm-specific implementation.
 */
template <class T>
constexpr field<T> field<T>::montgomery_mul_big(const field& other) const noexcept
{
    static_assert(modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    return montgomery_mul_big_native(other);
#elif BB_R_LIMB_BITS == 29
    return montgomery_mul_big_wasm29(other);
#else
    // No dedicated 24-bit FMA kernel exists for large moduli yet.
    // Reuse the generic bigint fallback so FMA builds remain correct.
    return montgomery_mul_constexpr_fallback(other);
#endif
}

/**
 * @brief 256×256 → 512-bit wide multiply.
 * Dispatches to paradigm-specific implementation.
 */
template <class T> constexpr struct field<T>::wide_array field<T>::mul_512(const field& other) const noexcept
{
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    return mul_512_native(other);
#else
    // mul_512 is a raw 256×256→512 integer multiply (no Montgomery reduction).
    // The WASM implementation uses 29-bit limb splitting internally as an
    // implementation detail, independent of the Montgomery R representation
    // (BB_R_LIMB_BITS). Safe to call from any WASM build, including FMA (R=2^264).
    return mul_512_wasm(other);
#endif
}

// NOLINTEND(readability-implicit-bool-conversion)
} // namespace bb
