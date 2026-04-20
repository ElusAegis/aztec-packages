#pragma once

// Native __int128 Montgomery backend.
//
// Provides CIOS montmul/square (small modulus), big-modulus CIOS, and a
// raw 256×256 → 512-bit wide multiply. All paths use __int128 widening
// arithmetic and are inherently constexpr.
//
// Active on any host with __int128 support (excluding WASM).

#if defined(__SIZEOF_INT128__) && !defined(__wasm__)

#include <array>

#include "../field_declarations.hpp"

namespace bb::detail {

template <class Params> struct NativeBackend {
    // ── Public MontBackend contract ──────────────────────────────────────

    BB_INLINE static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept;
    BB_INLINE static constexpr field<Params> sqr(const field<Params>& x) noexcept;
    BB_INLINE static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept;
    BB_INLINE static constexpr typename field<Params>::wide_array wide_mul(const field<Params>& lhs,
                                                                           const field<Params>& rhs) noexcept;

    // Batched Montgomery mul: outs[i] = as[i] * bs[i] for i in [0, N).
    // Native backend has no SIMD kernel — N sequential CIOS multiplies.
    // Only WasmFmaBackend specializes this for N=2 (real SIMD pairing) and
    // for N=3 (paired FMA + integer co-scheduling).
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

  private:
    // ── Widening arithmetic helpers (private to this backend) ────────────

    BB_INLINE static constexpr std::pair<uint64_t, uint64_t> mul_wide(uint64_t a, uint64_t b) noexcept
    {
        const uint128_t res = (static_cast<uint128_t>(a) * static_cast<uint128_t>(b));
        return { static_cast<uint64_t>(res), static_cast<uint64_t>(res >> 64) };
    }

    BB_INLINE static constexpr uint64_t mac(
        uint64_t a, uint64_t b, uint64_t c, uint64_t carry_in, uint64_t& carry_out) noexcept
    {
        const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c)) +
                              static_cast<uint128_t>(carry_in);
        carry_out = static_cast<uint64_t>(res >> 64);
        return static_cast<uint64_t>(res);
    }

    BB_INLINE static constexpr void mac(
        uint64_t a, uint64_t b, uint64_t c, uint64_t carry_in, uint64_t& out, uint64_t& carry_out) noexcept
    {
        const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c)) +
                              static_cast<uint128_t>(carry_in);
        out = static_cast<uint64_t>(res);
        carry_out = static_cast<uint64_t>(res >> 64);
    }

    BB_INLINE static constexpr uint64_t mac_mini(uint64_t a, uint64_t b, uint64_t c, uint64_t& carry_out) noexcept
    {
        const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
        carry_out = static_cast<uint64_t>(res >> 64);
        return static_cast<uint64_t>(res);
    }

    BB_INLINE static constexpr void mac_mini(
        uint64_t a, uint64_t b, uint64_t c, uint64_t& out, uint64_t& carry_out) noexcept
    {
        const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
        out = static_cast<uint64_t>(res);
        carry_out = static_cast<uint64_t>(res >> 64);
    }

    BB_INLINE static constexpr uint64_t mac_discard_lo(uint64_t a, uint64_t b, uint64_t c) noexcept
    {
        const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
        return static_cast<uint64_t>(res >> 64);
    }

    BB_INLINE static constexpr uint64_t square_accumulate(uint64_t a,
                                                          uint64_t b,
                                                          uint64_t c,
                                                          uint64_t carry_in_lo,
                                                          uint64_t carry_in_hi,
                                                          uint64_t& carry_lo,
                                                          uint64_t& carry_hi) noexcept
    {
        const uint128_t product = static_cast<uint128_t>(b) * static_cast<uint128_t>(c);
        const auto r0 = static_cast<uint64_t>(product);
        const auto r1 = static_cast<uint64_t>(product >> 64);
        uint64_t out = r0 + r0;
        carry_lo = (out < r0);
        out += a;
        carry_lo += (out < a);
        out += carry_in_lo;
        carry_lo += (out < carry_in_lo);
        carry_lo += r1;
        carry_hi = (carry_lo < r1);
        carry_lo += r1;
        carry_hi += (carry_lo < r1);
        carry_lo += carry_in_hi;
        carry_hi += (carry_lo < carry_in_hi);
        return out;
    }
};

// ── CIOS Montgomery multiplication (small modulus, < 2^254) ──────────────

template <class Params>
constexpr field<Params> NativeBackend<Params>::mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    const uint64_t* a = lhs.data;
    const uint64_t* b = rhs.data;
    constexpr auto modulus = field<Params>::modulus;

    auto [t0, c] = mul_wide(a[0], b[0]);
    uint64_t k = t0 * Params::r_inv;
    uint64_t carry = mac_discard_lo(t0, k, modulus.data[0]);

    uint64_t t1 = mac_mini(carry, a[0], b[1], carry);
    mac(t1, k, modulus.data[1], c, t0, c);
    uint64_t t2 = mac_mini(carry, a[0], b[2], carry);
    mac(t2, k, modulus.data[2], c, t1, c);
    uint64_t t3 = mac_mini(carry, a[0], b[3], carry);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + carry;

    mac_mini(t0, a[1], b[0], t0, carry);
    k = t0 * Params::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, a[1], b[1], carry, t1, carry);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, a[1], b[2], carry, t2, carry);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, a[1], b[3], carry, t3, carry);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + carry;

    mac_mini(t0, a[2], b[0], t0, carry);
    k = t0 * Params::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, a[2], b[1], carry, t1, carry);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, a[2], b[2], carry, t2, carry);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, a[2], b[3], carry, t3, carry);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + carry;

    mac_mini(t0, a[3], b[0], t0, carry);
    k = t0 * Params::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, a[3], b[1], carry, t1, carry);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, a[3], b[2], carry, t2, carry);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, a[3], b[3], carry, t3, carry);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + carry;

    field<Params> result{ t0, t1, t2, t3 };
    if (!std::is_constant_evaluated()) {
        result.assert_coarse_form();
    }
    return result;
}

// ── CIOS Montgomery squaring (small modulus) ─────────────────────────────

template <class Params> constexpr field<Params> NativeBackend<Params>::sqr(const field<Params>& x) noexcept
{
    const uint64_t* a = x.data;
    constexpr auto modulus = field<Params>::modulus;

    uint64_t carry_hi = 0;

    auto [t0, carry_lo] = mul_wide(a[0], a[0]);
    uint64_t t1 = square_accumulate(0, a[1], a[0], carry_lo, carry_hi, carry_lo, carry_hi);
    uint64_t t2 = square_accumulate(0, a[2], a[0], carry_lo, carry_hi, carry_lo, carry_hi);
    uint64_t t3 = square_accumulate(0, a[3], a[0], carry_lo, carry_hi, carry_lo, carry_hi);

    uint64_t round_carry = carry_lo;
    uint64_t k = t0 * Params::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t1 = mac_mini(t1, a[1], a[1], carry_lo);
    carry_hi = 0;
    t2 = square_accumulate(t2, a[2], a[1], carry_lo, carry_hi, carry_lo, carry_hi);
    t3 = square_accumulate(t3, a[3], a[1], carry_lo, carry_hi, carry_lo, carry_hi);
    round_carry = carry_lo;
    k = t0 * Params::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t2 = mac_mini(t2, a[2], a[2], carry_lo);
    carry_hi = 0;
    t3 = square_accumulate(t3, a[3], a[2], carry_lo, carry_hi, carry_lo, carry_hi);
    round_carry = carry_lo;
    k = t0 * Params::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t3 = mac_mini(t3, a[3], a[3], carry_lo);
    k = t0 * Params::r_inv;
    round_carry = carry_lo;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    field<Params> result{ t0, t1, t2, t3 };
    if (!std::is_constant_evaluated()) {
        result.assert_coarse_form();
    }
    return result;
}

// ── Big-modulus Montgomery multiplication (>= 2^254) ─────────────────────

template <class Params>
constexpr field<Params> NativeBackend<Params>::mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
{
    static_assert(field<Params>::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);
    constexpr auto modulus = field<Params>::modulus;

    uint64_t c = 0;
    uint64_t t0 = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    uint64_t t3 = 0;
    uint64_t t4 = 0;
    uint64_t t5 = 0;
    uint64_t k = 0;

    for (const auto& element : lhs.data) {
        c = 0;
        mac(t0, element, rhs.data[0], c, t0, c);
        mac(t1, element, rhs.data[1], c, t1, c);
        mac(t2, element, rhs.data[2], c, t2, c);
        mac(t3, element, rhs.data[3], c, t3, c);
        t4 = field<Params>::addc(t4, c, 0, t5);

        k = t0 * Params::r_inv;
        c = mac_discard_lo(t0, k, modulus.data[0]);
        mac(t1, k, modulus.data[1], c, t0, c);
        mac(t2, k, modulus.data[2], c, t1, c);
        mac(t3, k, modulus.data[3], c, t2, c);
        t3 = field<Params>::addc(c, t4, 0, c);
        t4 = t5 + c;
    }

    uint64_t borrow = 0;
    uint64_t r0 = field<Params>::sbb(t0, modulus.data[0], borrow, borrow);
    uint64_t r1 = field<Params>::sbb(t1, modulus.data[1], borrow, borrow);
    uint64_t r2 = field<Params>::sbb(t2, modulus.data[2], borrow, borrow);
    uint64_t r3 = field<Params>::sbb(t3, modulus.data[3], borrow, borrow);
    borrow = borrow ^ (0ULL - t4);
    r0 += (modulus.data[0] & borrow);
    uint64_t carry = r0 < (modulus.data[0] & borrow);
    r1 = field<Params>::addc(r1, modulus.data[1] & borrow, carry, carry);
    r2 = field<Params>::addc(r2, modulus.data[2] & borrow, carry, carry);
    r3 += (modulus.data[3] & borrow) + carry;
    return { r0, r1, r2, r3 };
}

// ── 256×256 → 512 wide multiply ──────────────────────────────────────────

template <class Params>
constexpr typename field<Params>::wide_array NativeBackend<Params>::wide_mul(const field<Params>& lhs,
                                                                             const field<Params>& rhs) noexcept
{
    const uint64_t* a = lhs.data;
    const uint64_t* b = rhs.data;

    uint64_t carry_2 = 0;
    auto [r0, carry] = mul_wide(a[0], b[0]);
    uint64_t r1 = mac_mini(carry, a[0], b[1], carry);
    uint64_t r2 = mac_mini(carry, a[0], b[2], carry);
    uint64_t r3 = mac_mini(carry, a[0], b[3], carry_2);

    r1 = mac_mini(r1, a[1], b[0], carry);
    r2 = mac(r2, a[1], b[1], carry, carry);
    r3 = mac(r3, a[1], b[2], carry, carry);
    uint64_t r4 = mac(carry_2, a[1], b[3], carry, carry_2);

    r2 = mac_mini(r2, a[2], b[0], carry);
    r3 = mac(r3, a[2], b[1], carry, carry);
    r4 = mac(r4, a[2], b[2], carry, carry);
    uint64_t r5 = mac(carry_2, a[2], b[3], carry, carry_2);

    r3 = mac_mini(r3, a[3], b[0], carry);
    r4 = mac(r4, a[3], b[1], carry, carry);
    r5 = mac(r5, a[3], b[2], carry, carry);
    uint64_t r6 = mac(carry_2, a[3], b[3], carry, carry_2);

    return { r0, r1, r2, r3, r4, r5, r6, carry_2 };
}

} // namespace bb::detail

#endif // defined(__SIZEOF_INT128__) && !defined(__wasm__)
