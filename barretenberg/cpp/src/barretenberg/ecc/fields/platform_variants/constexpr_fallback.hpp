#pragma once

// Constexpr-safe Montgomery multiplication fallback — works for any R.
//
// Used when the optimized runtime path is not constexpr-evaluable
// (e.g., FMA uses std::floor, native uses __int128 widening).
// Delegates to uint256_t mod_mul / compute_div_r_inv for correctness.
//
// Performance: O(256^2) doublings per multiply — suitable only for compile-time.
// Correctness: derives R^{-1} from the same R_EXPONENT that drives r_squared,
// so the Montgomery form matches the runtime path exactly.

#if defined(__wasm__) || !defined(__SIZEOF_INT128__)

template <class T>
constexpr field<T> field<T>::montgomery_mul_constexpr_fallback(const field& other) const noexcept
{
    const uint256_t a_ui{ data[0], data[1], data[2], data[3] };
    const uint256_t b_ui{ other.data[0], other.data[1], other.data[2], other.data[3] };

    // R^{-1} mod p, computed once per instantiation.
    // R_EXPONENT is the same constant that drives r_squared, so the Montgomery
    // representation produced here matches the runtime path bit-for-bit.
    constexpr uint256_t r_inv_mod_p = compute_div_r_inv(modulus, R_EXPONENT);

    // Montgomery mul: (a · b) / R mod p.
    const uint256_t ab_mod_p = mod_mul(a_ui, b_ui, modulus);
    const uint256_t result   = mod_mul(ab_mod_p, r_inv_mod_p, modulus);

    return field{ result.data[0], result.data[1], result.data[2], result.data[3] };
}

#endif // defined(__wasm__) || !defined(__SIZEOF_INT128__)
