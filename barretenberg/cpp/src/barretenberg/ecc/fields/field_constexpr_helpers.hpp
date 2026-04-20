#pragma once
#include "field_montgomery_config.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include <array>
#include <cstdint>

namespace bb {

// Forward declaration: needed by constexpr_mont_mul below.
// Full definition lives in field_declarations.hpp, which includes this header.
template <class Params_> struct field;

// -p^{-1} mod 2^64 via Newton's method (6 iterations: 1->2->4->8->16->32->64 correct bits)
static constexpr uint64_t compute_r_inv(uint64_t p0)
{
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) {
        x = x * (2 - p0 * x);
    }
    return -x;
}

// Constexpr modular addition: returns (a + b) mod p, handling 256-bit overflow.
// Precondition: a < p, b < p.
static constexpr uint256_t mod_add(const uint256_t& a, const uint256_t& b, const uint256_t& p)
{
    uint256_t sum = a + b;
    bool carry = sum < a; // overflow detection
    if (carry || sum >= p) {
        sum = sum - p;
    }
    return sum;
}

// Constexpr 512-bit modular reduction: (hi * 2^256 + lo) mod p.
// Uses only uint256_t arithmetic, avoiding uint512_t.
// Strategy: reduce hi mod p, then compute (hi_reduced * 2^256 + lo) mod p
// by doubling hi_reduced 256 times, then adding lo.
static constexpr uint256_t wide_mod(uint256_t hi, uint256_t lo, const uint256_t& p)
{
    while (hi >= p) {
        hi = hi - p;
    }
    while (lo >= p) {
        lo = lo - p;
    }

    // Compute hi * 2^256 mod p by 256 modular doublings
    uint256_t acc = hi;
    for (unsigned i = 0; i < 256; ++i) {
        acc = mod_add(acc, acc, p);
    }

    return mod_add(acc, lo, p);
}

// Constexpr modular multiplication: (a * b) mod p using 512-bit intermediate.
static constexpr uint256_t mod_mul(const uint256_t& a, const uint256_t& b, const uint256_t& p)
{
    auto [lo, hi] = a.mul_extended(b);
    return wide_mod(hi, lo, p);
}

// 2^r_exponent mod p via repeated modular doubling.
static constexpr uint256_t pow2_mod(const uint256_t& p, unsigned r_exponent)
{
    uint256_t acc(1);
    for (unsigned i = 0; i < r_exponent; ++i) {
        acc = mod_add(acc, acc, p);
    }
    return acc;
}

// R^2 mod p for arbitrary R = 2^r_exponent, using only uint256_t constexpr arithmetic.
static constexpr uint256_t compute_r_squared(const uint256_t& modulus, unsigned r_exponent)
{
    uint256_t R_mod_p = pow2_mod(modulus, r_exponent);
    return mod_mul(R_mod_p, R_mod_p, modulus);
}

// Split uint256_t into NUM_LIMBS limbs of LIMB_BITS bits each (little-endian).
// When LIMB_BITS == 64, each limb is exactly one word (no masking/spanning needed).
//
// The ≥ 255 bound (not 256) is intentional — the RNE backend uses a 5×51-bit
// internal layout (255 bits) for values that are always < 2^255 by
// construction (BN254 Fr modulus is < 2^254). Callers with inputs that may
// actually occupy all 256 bits must pick a limbing that covers ≥ 256 bits.
template <unsigned LIMB_BITS, unsigned NUM_LIMBS>
static constexpr std::array<uint64_t, NUM_LIMBS> split_limbs(const uint256_t& v)
{
    static_assert(LIMB_BITS > 0 && LIMB_BITS <= 64, "LIMB_BITS must be in (0, 64]");
    static_assert(NUM_LIMBS * LIMB_BITS >= 255, "NUM_LIMBS * LIMB_BITS must cover ≥ 255 bits");
    std::array<uint64_t, NUM_LIMBS> limbs{};
    if constexpr (LIMB_BITS == 64) {
        for (unsigned i = 0; i < NUM_LIMBS; ++i) {
            limbs[i] = (i < 4) ? v.data[i] : 0;
        }
    } else {
        constexpr uint64_t mask = (1ULL << LIMB_BITS) - 1;
        for (unsigned i = 0; i < NUM_LIMBS; ++i) {
            unsigned bit_pos = i * LIMB_BITS;
            unsigned word_lo = bit_pos / 64;
            unsigned shift_lo = bit_pos % 64;
            uint64_t val = v.data[word_lo] >> shift_lo;
            if (shift_lo + LIMB_BITS > 64 && word_lo + 1 < 4) {
                val |= v.data[word_lo + 1] << (64 - shift_lo);
            }
            limbs[i] = val & mask;
        }
    }
    return limbs;
}

// Compute 2^{-LIMB_BITS} mod p by repeated halving.
// "Halve mod p" means: if odd, add p first (making it even), then right-shift by 1.
// The addition can overflow 256 bits, so we track the 257th bit as `carry`.
static constexpr uint256_t compute_div_r_inv(const uint256_t& p, unsigned limb_bits)
{
    uint256_t result(1);
    for (unsigned i = 0; i < limb_bits; ++i) {
        uint64_t carry = 0;
        if ((result.data[0] & 1) != 0) {
            uint256_t sum = result + p;
            carry = (sum < result) ? 1ULL : 0ULL;
            result = sum;
        }
        // Right-shift by 1, feeding carry into bit 255
        result = uint256_t((result.data[0] >> 1) | (result.data[1] << 63),
                           (result.data[1] >> 1) | (result.data[2] << 63),
                           (result.data[2] >> 1) | (result.data[3] << 63),
                           (result.data[3] >> 1) | (carry << 63));
    }
    return result;
}

// Convert a constexpr uint64_t array to a double array (for FMA paths).
template <size_t N> static constexpr std::array<double, N> to_double_array(const std::array<uint64_t, N>& arr)
{
    std::array<double, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<double>(arr[i]);
    }
    return result;
}

// Precomputed limb constants for a given limb representation.
template <unsigned LIMB_BITS, unsigned NUM_LIMBS> struct LimbConstants {
    std::array<uint64_t, NUM_LIMBS> modulus;
    std::array<uint64_t, NUM_LIMBS> div_r_inv;

#ifdef MONTMUL_VARIANT_FMA
    // The FMA SIMD montmul operates entirely in f64 arithmetic — it needs the
    // modulus and Yuval div_r_inv (2^{-LIMB_BITS} mod p) as double arrays for
    // direct use in relaxed-SIMD FMA instructions (f64x2.relaxed_madd).
    // Only materialized when the FMA variant is active; other paths use the
    // integer arrays above.
    std::array<double, NUM_LIMBS> modulus_f;
    std::array<double, NUM_LIMBS> div_r_inv_f;
#endif

#ifdef MONTMUL_VARIANT_RNE
    // The RNE SIMD montmul keeps external 4×64-bit storage but operates
    // internally on 5×51-bit limbs via f64x2 FMA. Its 51-bit constants are
    // invariant to the platform LIMB_BITS/NUM_LIMBS — the arrays are always
    // sized 5 (51*5 = 255 bits cover the field).
    std::array<uint64_t, 5> u51_p;              // modulus repacked as 5×51-bit limbs
    uint64_t u51_np0;                           // -p^{-1} mod 2^51
    std::array<std::array<uint64_t, 5>, 4> rho; // rho[i-1] = (2^(51*i + 255)) mod p, as 5×51-bit limbs
#endif
};

// Factory: compute LimbConstants from a uint256_t modulus.
template <unsigned LIMB_BITS, unsigned NUM_LIMBS>
static constexpr LimbConstants<LIMB_BITS, NUM_LIMBS> compute_limb_constants(const uint256_t& modulus)
{
    LimbConstants<LIMB_BITS, NUM_LIMBS> result{};
    result.modulus = split_limbs<LIMB_BITS, NUM_LIMBS>(modulus);
    uint256_t div_r = compute_div_r_inv(modulus, LIMB_BITS);
    result.div_r_inv = split_limbs<LIMB_BITS, NUM_LIMBS>(div_r);
#ifdef MONTMUL_VARIANT_FMA
    result.modulus_f = to_double_array(result.modulus);
    result.div_r_inv_f = to_double_array(result.div_r_inv);
#endif
#ifdef MONTMUL_VARIANT_RNE
    result.u51_p = split_limbs<51, 5>(modulus);
    // -p^{-1} mod 2^51 — mask the existing -p^{-1} mod 2^64 down to 51 bits.
    // Valid because 2^51 | 2^64 and p is odd, so the inverse congruence
    // p · x ≡ -1 mod 2^64 implies p · (x mod 2^51) ≡ -1 mod 2^51.
    result.u51_np0 = compute_r_inv(modulus.data[0]) & ((1ULL << 51) - 1);
    // rho[k-1] = 2^{-51·k} mod p.
    //
    // Derivation: smult_noinit(t[i], RHO_{4-i+1}) adds t[i]·RHO at ss[0..5]
    // which starts at intermediate bit position 204. We want the added term
    // to represent t[i] at its original bit position 51·i (so that after
    // Montgomery division by 2^255, the sum gives product · 2^{-255} mod p):
    //     t[i] · RHO · 2^204 ≡ t[i] · 2^{51·i} mod p
    //     RHO                ≡ 2^{51·i - 204} mod p
    //     RHO                ≡ 2^{-51·(4-i)} mod p
    // Pairing: RHO_k corresponds to t[4-k], so RHO_k = 2^{-51·k} mod p.
    //
    // `compute_div_r_inv(p, n)` computes 2^{-n} mod p by repeated halving.
    // Despite provekit's comment claiming "2^(51·i)·2^255 mod p", the
    // hardcoded constants match this 2^{-51·k} formula (confirmed by runtime
    // differential test).
    for (unsigned k = 1; k <= 4; ++k) {
        result.rho[k - 1] = split_limbs<51, 5>(compute_div_r_inv(modulus, 51U * k));
    }
#endif
    return result;
}

// Compute (canonical * 2^r_exponent) mod p — converts from canonical to Montgomery form.
static constexpr uint256_t to_montgomery_uint256(const uint256_t& canonical,
                                                 const uint256_t& modulus,
                                                 unsigned r_exponent)
{
    return mod_mul(canonical, pow2_mod(modulus, r_exponent), modulus);
}

// Convenience: compute limb constants using the platform R configuration.
// On native (R_LIMB_BITS=64, R_NUM_LIMBS=4), the limbs are just the 4 uint64_t words.
// On WASM (R_LIMB_BITS=29, R_NUM_LIMBS=9), the limbs are the 29-bit sub-word representation.
static constexpr auto compute_r_limb_constants(const uint256_t& modulus)
{
    return compute_limb_constants<R_LIMB_BITS, R_NUM_LIMBS>(modulus);
}

// Constexpr-safe Montgomery multiplication via uint256_t modular arithmetic.
//
// Works for any R — derives R^{-1} from R_EXPONENT, so the Montgomery form
// matches whatever R the platform backend uses (2^256, 2^261, 2^264, …).
// O(256²) doublings per multiply; use only at compile time or as a slow
// last-resort runtime path (e.g. FMA mul_big while no dedicated kernel exists).
//
// Requires field<Params> (full declaration via field_declarations.hpp) — callers
// must include that header before this helper.
template <class Params>
static constexpr field<Params> constexpr_mont_mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    const uint256_t a_ui{ lhs.data[0], lhs.data[1], lhs.data[2], lhs.data[3] };
    const uint256_t b_ui{ rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3] };

    // R^{-1} mod p, computed once per instantiation. R_EXPONENT is the same
    // constant that drives r_squared, so the Montgomery representation here
    // matches the runtime path bit-for-bit.
    constexpr uint256_t r_inv_mod_p = compute_div_r_inv(field<Params>::modulus, R_EXPONENT);

    // Montgomery mul: (a · b) / R mod p.
    const uint256_t ab_mod_p = mod_mul(a_ui, b_ui, field<Params>::modulus);
    const uint256_t result = mod_mul(ab_mod_p, r_inv_mod_p, field<Params>::modulus);

    return field<Params>{ result.data[0], result.data[1], result.data[2], result.data[3] };
}

} // namespace bb
