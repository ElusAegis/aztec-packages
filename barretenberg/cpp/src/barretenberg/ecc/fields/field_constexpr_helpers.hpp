#pragma once
#include "barretenberg/numeric/uint256/uint256.hpp"
#include <array>
#include <cstdint>

namespace bb {

// -p^{-1} mod 2^64 via Newton's method (6 iterations: 1->2->4->8->16->32->64 correct bits)
static constexpr uint64_t compute_r_inv(uint64_t p0)
{
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) {
        x = x * (2 - p0 * x);
    }
    return -x;
}

// Constexpr modular doubling: returns (2 * a) mod p, handling overflow.
// Precondition: a < p.
static constexpr uint256_t mod_dbl(const uint256_t& a, const uint256_t& p)
{
    bool msb = (a.data[3] >> 63) != 0;
    uint256_t doubled = a + a; // wraps mod 2^256
    // If the MSB was set before doubling, the true result is doubled + 2^256, which is certainly >= p.
    // Also check if the (wrapped) doubled value >= p.
    if (msb || doubled >= p) {
        doubled = doubled - p;
    }
    return doubled;
}

// Constexpr modular addition: returns (a + b) mod p, handling overflow.
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
    // Reduce hi mod p (hi may be >= p for large moduli)
    while (hi >= p) {
        hi = hi - p;
    }

    // Reduce lo mod p
    while (lo >= p) {
        lo = lo - p;
    }

    // Compute hi * 2^256 mod p by 256 modular doublings
    uint256_t acc = hi;
    for (unsigned i = 0; i < 256; ++i) {
        acc = mod_dbl(acc, p);
    }

    // Add lo
    return mod_add(acc, lo, p);
}

// 2^r_exponent mod p via repeated modular doubling.
static constexpr uint256_t pow2_mod(const uint256_t& p, unsigned r_exponent)
{
    uint256_t acc(1);
    for (unsigned i = 0; i < r_exponent; ++i) {
        acc = mod_dbl(acc, p);
    }
    return acc;
}

// R^2 mod p for arbitrary R = 2^r_exponent, using only uint256_t constexpr arithmetic.
static constexpr uint256_t compute_r_squared(const uint256_t& modulus, unsigned r_exponent)
{
    uint256_t R_mod_p = pow2_mod(modulus, r_exponent);
    auto [lo, hi] = R_mod_p.mul_extended(R_mod_p);
    return wide_mod(hi, lo, modulus);
}

// Split uint256_t into 9x29-bit limbs (little-endian)
static constexpr std::array<uint64_t, 9> split_29bit(const uint256_t& v)
{
    std::array<uint64_t, 9> limbs{};
    constexpr uint64_t mask = (1ULL << 29) - 1;
    uint64_t words[4] = { v.data[0], v.data[1], v.data[2], v.data[3] };

    uint64_t acc = words[0];
    unsigned bits_in_acc = 64;
    unsigned word_idx = 1;

    for (unsigned i = 0; i < 9; ++i) {
        if (bits_in_acc < 29 && word_idx < 4) {
            acc |= (words[word_idx] << bits_in_acc);
            bits_in_acc += 64;
            ++word_idx;
        }
        limbs[i] = acc & mask;
        acc >>= 29;
        bits_in_acc -= 29;
    }
    return limbs;
}

// Compute (canonical * 2^r_exponent) mod p — converts from canonical to Montgomery form.
static constexpr uint256_t to_montgomery_uint256(const uint256_t& canonical,
                                                 const uint256_t& modulus,
                                                 unsigned r_exponent)
{
    uint256_t R_mod_p = pow2_mod(modulus, r_exponent);
    auto [lo, hi] = canonical.mul_extended(R_mod_p);
    return wide_mod(hi, lo, modulus);
}

} // namespace bb
