#pragma once

// Constexpr-safe Montgomery multiplication fallback — works for any R.
//
// Used when the optimized runtime path is not constexpr-evaluable
// (e.g., FMA uses std::floor, x86 asm uses inline assembly).
// Delegates to uint256_t mod_mul / compute_div_r_inv for correctness.
//
// Performance: O(256^2) doublings per multiply — suitable only for
// compile-time evaluation, or as a last-resort runtime path.
// Correctness: derives R^{-1} from the same R_EXPONENT that drives r_squared,
// so the Montgomery form matches the runtime path exactly (whatever R that
// path uses).

#include "../field_constexpr_helpers.hpp"
#include "../field_declarations.hpp"
#include "../field_montgomery_config.hpp"

namespace bb::detail {

/**
 * @brief Montgomery multiplication via uint256_t modular arithmetic.
 *
 * Not a full MontBackend — only provides `mul()`. Used by backends that
 * need a constexpr-safe fallback for their runtime-only implementations
 * (X86Asm → Native is preferred; FMA → this as a correctness-first, slow
 * fallback).
 */
struct ConstexprFallback {
    template <class Params>
    static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        const uint256_t a_ui{ lhs.data[0], lhs.data[1], lhs.data[2], lhs.data[3] };
        const uint256_t b_ui{ rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3] };

        // R^{-1} mod p, computed once per instantiation.
        // R_EXPONENT is the same constant that drives r_squared, so the
        // Montgomery representation produced here matches the runtime path
        // bit-for-bit (whether R = 2^256, 2^261, or 2^264).
        constexpr uint256_t r_inv_mod_p = compute_div_r_inv(field<Params>::modulus, R_EXPONENT);

        // Montgomery mul: (a · b) / R mod p.
        const uint256_t ab_mod_p = mod_mul(a_ui, b_ui, field<Params>::modulus);
        const uint256_t result = mod_mul(ab_mod_p, r_inv_mod_p, field<Params>::modulus);

        return field<Params>{ result.data[0], result.data[1], result.data[2], result.data[3] };
    }
};

} // namespace bb::detail
