// Integer Logjumps Montgomery multiplication for BN254 fields.
//
// Uses 4x64-bit limbs with R=2^256, matching native Montgomery form.
// Instead of sequential CIOS (9 x 29-bit limbs, R=2^261), this uses
// Logjumps-style parallel reduction: fold lower limbs via precomputed
// 2^(-64k) mod p constants, then one final Montgomery step.
//
// Multiplication count: 32 widening multiplications vs 162 i64.mul for 29-bit.
// On WASM, each widening mul compiles to ~4 i64.mul + recombination overhead,
// so the real comparison is ~128 i64.mul + overhead vs 162 i64.mul.
//
// The three Logjumps fold multiplications are independent and can be
// parallelized in future SIMD variants.
//
// References:
// - ProveKit (worldfnd/ProveKit) scalar.rs
// - Koh Wei Jie, "A Deep Dive into Logjumps", Bain Capital Crypto, 2025
#pragma once

#include <cstdint>
#include <type_traits>

namespace bb::float_montmul_alt {

// ============================================================================
// Logjumps constants for BN254 fields
// I_k = 2^(-64*k) mod p     (for folding lower limbs)
// MU0 = -p^{-1} mod 2^64    (for final Montgomery step)
// ============================================================================

struct LogjumpsConstantsFq {
    static constexpr uint64_t P[4] = {
        0x3C208C16D87CFD47ULL, 0x97816A916871CA8DULL,
        0xB85045B68181585DULL, 0x30644E72E131A029ULL,
    };
    static constexpr uint64_t MU0 = 0x87D20782E4866389ULL;
    static constexpr uint64_t I1[4] = {
        0x327d7c1b18f7bd41ULL, 0xdb8ed52f824ed32fULL,
        0x29b67b05eb29a6a1ULL, 0x19ac99126b459ddaULL,
    };
    static constexpr uint64_t I2[4] = {
        0x1da790e434ade680ULL, 0x27a2f342f9905883ULL,
        0xb5ab34890dfa3d61ULL, 0x1e07f71b064ef9b1ULL,
    };
    static constexpr uint64_t I3[4] = {
        0xb334aa7264874f53ULL, 0x62a52db096edbc9eULL,
        0x235878f5c0a1dafeULL, 0x28f5dd496ed1da9dULL,
    };
};

struct LogjumpsConstantsFr {
    static constexpr uint64_t P[4] = {
        0x43E1F593F0000001ULL, 0x2833E84879B97091ULL,
        0xB85045B68181585DULL, 0x30644E72E131A029ULL,
    };
    static constexpr uint64_t MU0 = 0xC2E1F593EFFFFFFFULL;
    static constexpr uint64_t I1[4] = {
        0x2d3e8053e396ee4dULL, 0xca478dbeab3c92cdULL,
        0xb2d8f06f77f52a93ULL, 0x24d6ba07f7aa8f04ULL,
    };
    static constexpr uint64_t I2[4] = {
        0x18ee753c76f9dc6fULL, 0x54ad7e14a329e70fULL,
        0x2b16366f4f7684dfULL, 0x133100d71fdf3579ULL,
    };
    static constexpr uint64_t I3[4] = {
        0x9bacb016127cbe4eULL, 0x0b2051fa31944124ULL,
        0xb064eea46091c76cULL, 0x2b062aaa49f80c7dULL,
    };
};

template <class Params> constexpr bool is_logjumps_supported_field()
{
    return Params::modulus_0 == 0x3C208C16D87CFD47ULL ||
           Params::modulus_0 == 0x43E1F593F0000001ULL;
}

template <class Params> struct LogjumpsConstants {
    using type = std::conditional_t<
        Params::modulus_0 == 0x3C208C16D87CFD47ULL,
        LogjumpsConstantsFq,
        LogjumpsConstantsFr>;
};

// ============================================================================
// Core arithmetic helpers
// ============================================================================

// Fused multiply-add with carry: (result, carry_out) = a*b + addend + carry_in
// Uses __uint128_t which emscripten supports (emits 32-bit split on WASM).
constexpr void carrying_mul_add(
    uint64_t a, uint64_t b, uint64_t addend, uint64_t carry_in,
    uint64_t& result, uint64_t& carry_out)
{
    __uint128_t prod = static_cast<__uint128_t>(a) * b + addend + carry_in;
    result = static_cast<uint64_t>(prod);
    carry_out = static_cast<uint64_t>(prod >> 64);
}

// 5-element carry-propagating addition: out = a + b
constexpr void addv5(const uint64_t a[5], const uint64_t b[5], uint64_t out[5])
{
    uint64_t carry = 0;
    for (int i = 0; i < 5; i++) {
        __uint128_t sum = static_cast<__uint128_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
}

// Constant-time conditional subtraction: if val >= sub, return val - sub.
// Returns true if subtraction was performed.
constexpr bool cond_subtract(uint64_t val[4], const uint64_t sub[4])
{
    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) {
        __uint128_t diff = static_cast<__uint128_t>(val[i]) - sub[i] - borrow;
        tmp[i] = static_cast<uint64_t>(diff);
        borrow = (diff >> 127) & 1; // borrow if negative
    }
    // Constant-time select: use tmp if no borrow, keep val otherwise
    uint64_t mask = (borrow == 0) ? UINT64_MAX : 0;
    for (int i = 0; i < 4; i++) {
        val[i] = (tmp[i] & mask) | (val[i] & ~mask);
    }
    return borrow == 0;
}

// ============================================================================
// Integer Logjumps Montgomery multiplication
//
// Computes a * b * R^{-1} mod p where R = 2^256.
// Inputs must be < p (fully reduced). Output is < p (fully reduced).
//
// Algorithm:
// 1. Schoolbook 4x4 multiplication -> 8-limb product t[0..7]
// 2. Logjumps fold: s = t[3..7] + t[0]*I3 + t[1]*I2 + t[2]*I1
//    (3 independent scalar-by-vector multiplications)
// 3. Final Montgomery step: m = s[0]*MU0, s += m*P, discard s[0]
// 4. Conditional subtraction of P (up to 3x: result can be < ~4P)
// ============================================================================

template <class Params>
constexpr void montgomery_mul_logjumps_64(
    const uint64_t a_data[4],
    const uint64_t b_data[4],
    uint64_t result[4])
{
    using C = typename LogjumpsConstants<Params>::type;

    // Phase 1: Schoolbook 4x4
    // t[0..7] = a * b (512-bit product in 8 x 64-bit limbs)
    uint64_t t[8] = {};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            carrying_mul_add(a_data[i], b_data[j], t[i + j], carry,
                             t[i + j], carry);
        }
        t[i + 4] = carry;
    }

    // Phase 2: Logjumps fold
    // sr_k = t[k] * I_{3-k} for k = 0, 1, 2
    // Each produces a 5-limb result (scalar * 4-limb constant -> 5 limbs)
    uint64_t sr[3][5];
    const uint64_t* I_consts[3] = { C::I3, C::I2, C::I1 };
    for (int k = 0; k < 3; k++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            carrying_mul_add(t[k], I_consts[k][j], 0, carry,
                             sr[k][j], carry);
        }
        sr[k][4] = carry;
    }

    // s = t[3..7] + sr[0] + sr[1] + sr[2]
    // Use carry-propagating addition to avoid 64-bit overflow.
    uint64_t t_upper[5] = { t[3], t[4], t[5], t[6], t[7] };
    uint64_t tmp1[5], tmp2[5], s[5];
    addv5(t_upper, sr[0], tmp1);
    addv5(sr[1], sr[2], tmp2);
    addv5(tmp1, tmp2, s);

    // Phase 3: Final Montgomery step
    // m = s[0] * MU0 (truncated to 64 bits)
    uint64_t m = s[0] * C::MU0;

    // mp = m * P (5-limb result)
    uint64_t mp[5] = {};
    {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            carrying_mul_add(m, C::P[j], 0, carry, mp[j], carry);
        }
        mp[4] = carry;
    }

    // s += mp (carry-propagating)
    uint64_t s_mp[5];
    addv5(s, mp, s_mp);

    // s_mp[0] is now zero mod 2^64 (by construction of m).
    // The carry from position 0 was propagated by addv5.
    // Result is s_mp[1..4].
    result[0] = s_mp[1];
    result[1] = s_mp[2];
    result[2] = s_mp[3];
    result[3] = s_mp[4];

    // Phase 4: Conditional subtraction (up to 3 times)
    // Output can be up to ~4p (since s < 2^320 and m*P < 2^318,
    // (s + m*P)/2^64 < 2^256 ~ 4p for BN254). Three subtractions
    // of P bring the result into [0, P).
    cond_subtract(result, C::P);
    cond_subtract(result, C::P);
    cond_subtract(result, C::P);
}

} // namespace bb::float_montmul_alt
