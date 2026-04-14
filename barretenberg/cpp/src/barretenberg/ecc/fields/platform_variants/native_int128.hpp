#pragma once

// Native __int128 compute paradigm: helpers, CIOS montmul/square, big-modulus montmul, wide multiply.
// Guard: only compiled when __int128 is available and not on WASM.
// Included from field_impl_generic.hpp (already inside namespace bb, after addc/sbb are defined).

#if defined(__SIZEOF_INT128__) && !defined(__wasm__)

// ── Widening arithmetic helpers ──────────────────────────────────────

template <class T>
constexpr std::pair<uint64_t, uint64_t> field<T>::mul_wide(uint64_t a, uint64_t b) noexcept
{
    const uint128_t res = (static_cast<uint128_t>(a) * static_cast<uint128_t>(b));
    return { static_cast<uint64_t>(res), static_cast<uint64_t>(res >> 64) };
}

template <class T>
constexpr uint64_t field<T>::mac(const uint64_t a,
                                 const uint64_t b,
                                 const uint64_t c,
                                 const uint64_t carry_in,
                                 uint64_t& carry_out) noexcept
{
    const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c)) +
                          static_cast<uint128_t>(carry_in);
    carry_out = static_cast<uint64_t>(res >> 64);
    return static_cast<uint64_t>(res);
}

template <class T>
constexpr void field<T>::mac(const uint64_t a,
                             const uint64_t b,
                             const uint64_t c,
                             const uint64_t carry_in,
                             uint64_t& out,
                             uint64_t& carry_out) noexcept
{
    const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c)) +
                          static_cast<uint128_t>(carry_in);
    out = static_cast<uint64_t>(res);
    carry_out = static_cast<uint64_t>(res >> 64);
}

template <class T>
constexpr uint64_t field<T>::mac_mini(const uint64_t a,
                                      const uint64_t b,
                                      const uint64_t c,
                                      uint64_t& carry_out) noexcept
{
    const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
    carry_out = static_cast<uint64_t>(res >> 64);
    return static_cast<uint64_t>(res);
}

template <class T>
constexpr void field<T>::mac_mini(const uint64_t a,
                                  const uint64_t b,
                                  const uint64_t c,
                                  uint64_t& out,
                                  uint64_t& carry_out) noexcept
{
    const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
    out = static_cast<uint64_t>(res);
    carry_out = static_cast<uint64_t>(res >> 64);
}

template <class T>
constexpr uint64_t field<T>::mac_discard_lo(const uint64_t a, const uint64_t b, const uint64_t c) noexcept
{
    const uint128_t res = static_cast<uint128_t>(a) + (static_cast<uint128_t>(b) * static_cast<uint128_t>(c));
    return static_cast<uint64_t>(res >> 64);
}

template <class T>
constexpr uint64_t field<T>::square_accumulate(const uint64_t a,
                                               const uint64_t b,
                                               const uint64_t c,
                                               const uint64_t carry_in_lo,
                                               const uint64_t carry_in_hi,
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

// ── CIOS Montgomery multiplication (small modulus, < 2^254) ──────────

template <class T>
constexpr field<T> field<T>::montgomery_mul_native_cios(const field& other) const noexcept
{
    auto [t0, c] = mul_wide(data[0], other.data[0]);
    uint64_t k = t0 * T::r_inv;
    uint64_t a = mac_discard_lo(t0, k, modulus.data[0]);

    uint64_t t1 = mac_mini(a, data[0], other.data[1], a);
    mac(t1, k, modulus.data[1], c, t0, c);
    uint64_t t2 = mac_mini(a, data[0], other.data[2], a);
    mac(t2, k, modulus.data[2], c, t1, c);
    uint64_t t3 = mac_mini(a, data[0], other.data[3], a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[1], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[1], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[1], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[1], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[2], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[2], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[2], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[2], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[3], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[3], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[3], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[3], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    field result{ t0, t1, t2, t3 };
    if (!std::is_constant_evaluated()) {
        result.assert_coarse_form();
    }
    return result;
}

// ── CIOS Montgomery squaring (small modulus) ─────────────────────────

template <class T>
constexpr field<T> field<T>::montgomery_square_native_cios() const noexcept
{
    uint64_t carry_hi = 0;

    auto [t0, carry_lo] = mul_wide(data[0], data[0]);
    uint64_t t1 = square_accumulate(0, data[1], data[0], carry_lo, carry_hi, carry_lo, carry_hi);
    uint64_t t2 = square_accumulate(0, data[2], data[0], carry_lo, carry_hi, carry_lo, carry_hi);
    uint64_t t3 = square_accumulate(0, data[3], data[0], carry_lo, carry_hi, carry_lo, carry_hi);

    uint64_t round_carry = carry_lo;
    uint64_t k = t0 * T::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t1 = mac_mini(t1, data[1], data[1], carry_lo);
    carry_hi = 0;
    t2 = square_accumulate(t2, data[2], data[1], carry_lo, carry_hi, carry_lo, carry_hi);
    t3 = square_accumulate(t3, data[3], data[1], carry_lo, carry_hi, carry_lo, carry_hi);
    round_carry = carry_lo;
    k = t0 * T::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t2 = mac_mini(t2, data[2], data[2], carry_lo);
    carry_hi = 0;
    t3 = square_accumulate(t3, data[3], data[2], carry_lo, carry_hi, carry_lo, carry_hi);
    round_carry = carry_lo;
    k = t0 * T::r_inv;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    t3 = mac_mini(t3, data[3], data[3], carry_lo);
    k = t0 * T::r_inv;
    round_carry = carry_lo;
    carry_lo = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, k, modulus.data[1], carry_lo, t0, carry_lo);
    mac(t2, k, modulus.data[2], carry_lo, t1, carry_lo);
    mac(t3, k, modulus.data[3], carry_lo, t2, carry_lo);
    t3 = carry_lo + round_carry;

    field result{ t0, t1, t2, t3 };
    if (!std::is_constant_evaluated()) {
        result.assert_coarse_form();
    }
    return result;
}

// ── Big-modulus Montgomery multiplication (>= 2^254), native path ────

template <class T>
constexpr field<T> field<T>::montgomery_mul_big_native(const field& other) const noexcept
{
    static_assert(modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD);

    uint64_t c = 0;
    uint64_t t0 = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    uint64_t t3 = 0;
    uint64_t t4 = 0;
    uint64_t t5 = 0;
    uint64_t k = 0;

    for (const auto& element : data) {
        c = 0;
        mac(t0, element, other.data[0], c, t0, c);
        mac(t1, element, other.data[1], c, t1, c);
        mac(t2, element, other.data[2], c, t2, c);
        mac(t3, element, other.data[3], c, t3, c);
        t4 = addc(t4, c, 0, t5);

        k = t0 * T::r_inv;
        c = mac_discard_lo(t0, k, modulus.data[0]);
        mac(t1, k, modulus.data[1], c, t0, c);
        mac(t2, k, modulus.data[2], c, t1, c);
        mac(t3, k, modulus.data[3], c, t2, c);
        t3 = addc(c, t4, 0, c);
        t4 = t5 + c;
    }

    uint64_t borrow = 0;
    uint64_t r0 = sbb(t0, modulus.data[0], borrow, borrow);
    uint64_t r1 = sbb(t1, modulus.data[1], borrow, borrow);
    uint64_t r2 = sbb(t2, modulus.data[2], borrow, borrow);
    uint64_t r3 = sbb(t3, modulus.data[3], borrow, borrow);
    borrow = borrow ^ (0ULL - t4);
    r0 += (modulus.data[0] & borrow);
    uint64_t carry = r0 < (modulus.data[0] & borrow);
    r1 = addc(r1, modulus.data[1] & borrow, carry, carry);
    r2 = addc(r2, modulus.data[2] & borrow, carry, carry);
    r3 += (modulus.data[3] & borrow) + carry;
    return { r0, r1, r2, r3 };
}

// ── 256×256→512 wide multiply, native path ───────────────────────────

template <class T>
constexpr struct field<T>::wide_array field<T>::mul_512_native(const field& other) const noexcept
{
    uint64_t carry_2 = 0;
    auto [r0, carry] = mul_wide(data[0], other.data[0]);
    uint64_t r1 = mac_mini(carry, data[0], other.data[1], carry);
    uint64_t r2 = mac_mini(carry, data[0], other.data[2], carry);
    uint64_t r3 = mac_mini(carry, data[0], other.data[3], carry_2);

    r1 = mac_mini(r1, data[1], other.data[0], carry);
    r2 = mac(r2, data[1], other.data[1], carry, carry);
    r3 = mac(r3, data[1], other.data[2], carry, carry);
    uint64_t r4 = mac(carry_2, data[1], other.data[3], carry, carry_2);

    r2 = mac_mini(r2, data[2], other.data[0], carry);
    r3 = mac(r3, data[2], other.data[1], carry, carry);
    r4 = mac(r4, data[2], other.data[2], carry, carry);
    uint64_t r5 = mac(carry_2, data[2], other.data[3], carry, carry_2);

    r3 = mac_mini(r3, data[3], other.data[0], carry);
    r4 = mac(r4, data[3], other.data[1], carry, carry);
    r5 = mac(r5, data[3], other.data[2], carry, carry);
    uint64_t r6 = mac(carry_2, data[3], other.data[3], carry, carry_2);

    return { r0, r1, r2, r3, r4, r5, r6, carry_2 };
}

#endif // defined(__SIZEOF_INT128__) && !defined(__wasm__)
