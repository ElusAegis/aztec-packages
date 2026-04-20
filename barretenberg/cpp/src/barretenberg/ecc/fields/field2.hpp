// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Raju], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "./field2_declarations.hpp"

/**
 * @brief Note, this file contains the definitions. of `field2` class.
 *        Declarations are in `field2_declarations.hpp`.
 *        Include ordering ensures linter/language server has knowledge of declarations when parsing definitions
 *
 */
namespace bb {
template <class base, class T> constexpr field2<base, T> field2<base, T>::operator*(const field2& other) const noexcept
{
    // no funny primes please! we assume -1 is not a quadratic residue
    static_assert((base::modulus.data[0] & 0x3UL) == 0x3UL);
    // Karatsuba base products: t1 = c0*other.c0, t2 = c1*other.c1 are fully
    // independent. Pair them into a single W=2 batched kernel so the WASM-FMA
    // backend dispatches mul_paired_fma_simd instead of two W=1 int29 kernels.
    base t1;
    base t2;
    base::template montgomery_mul_batched<2>(
        { &c0, &c1 }, { &other.c0, &other.c1 }, { &t1, &t2 });
    base t3 = c0 + c1;
    base t4 = other.c0 + other.c1;

    return { t1 - t2, t3 * t4 - (t1 + t2) };
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator+(const field2& other) const noexcept
{
    return { c0 + other.c0, c1 + other.c1 };
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator-(const field2& other) const noexcept
{
    return { c0 - other.c0, c1 - other.c1 };
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator-() const noexcept
{
    return { -c0, -c1 };
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator/(const field2& other) const noexcept
{
    return operator*(other.invert());
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator*=(const field2& other) noexcept
{
    *this = operator*(other);
    return *this;
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator+=(const field2& other) noexcept
{
    *this = operator+(other);
    return *this;
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator-=(const field2& other) noexcept
{
    *this = operator-(other);
    return *this;
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::operator/=(const field2& other) noexcept
{
    *this = operator/(other);
    return *this;
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::sqr() const noexcept
{
    // Complex-squaring identity: (c0 + c1*u)^2 = (c0+c1)(c0-c1) + 2*c0*c1*u.
    // The two Fp muls -- (c0+c1)*(c0-c1) and c0*c1 -- are independent, so
    // pair them into a single W=2 batched kernel for the same reason as
    // operator*: the WASM-FMA backend promotes to mul_paired_fma_simd.
    base sum = c0 + c1;
    base diff = c0 - c1;
    base real_part;
    base t1;
    base::template montgomery_mul_batched<2>({ &sum, &c0 }, { &diff, &c1 }, { &real_part, &t1 });
    return { real_part, t1 + t1 };
}

template <class base, class T> constexpr void field2<base, T>::self_sqr() noexcept
{
    *this = sqr();
}

// Montgomery form conversions use the reduced variants to ensure each component
// is in canonical form [0, p) rather than the coarse internal representation [0, 2p).
template <class base, class T> constexpr field2<base, T> field2<base, T>::to_montgomery_form() const noexcept
{
    return { c0.to_montgomery_form_reduced(), c1.to_montgomery_form_reduced() };
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::from_montgomery_form() const noexcept
{
    return { c0.from_montgomery_form_reduced(), c1.from_montgomery_form_reduced() };
}

template <class base, class T> constexpr void field2<base, T>::self_to_montgomery_form() noexcept
{
    c0.self_to_montgomery_form_reduced();
    c1.self_to_montgomery_form_reduced();
}

template <class base, class T> constexpr void field2<base, T>::self_from_montgomery_form() noexcept
{
    c0.self_from_montgomery_form_reduced();
    c1.self_from_montgomery_form_reduced();
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::reduce_once() const noexcept
{
    return { c0.reduce_once(), c1.reduce_once() };
}

template <class base, class T> constexpr void field2<base, T>::self_reduce_once() noexcept
{
    c0.self_reduce_once();
    c1.self_reduce_once();
}

template <class base, class T> constexpr void field2<base, T>::self_neg() noexcept
{
    c0.self_neg();
    c1.self_neg();
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::pow(const uint256_t& exponent) const noexcept
{

    field2 accumulator = *this;
    field2 to_mul = *this;
    const uint64_t maximum_set_bit = exponent.get_msb();

    for (int i = static_cast<int>(maximum_set_bit) - 1; i >= 0; --i) {
        accumulator.self_sqr();
        if (exponent.get_bit(static_cast<uint64_t>(i))) {
            accumulator *= to_mul;
        }
    }

    if (*this == zero()) {
        accumulator = zero();
    } else if (exponent == uint256_t(0)) {
        accumulator = one();
    }
    return accumulator;
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::pow(const uint64_t exponent) const noexcept
{
    return pow({ exponent, 0, 0, 0 });
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::invert() const noexcept
{
    // Norm |c|^2 = c0^2 + c1^2: two independent Fp squares, then invert the
    // sum. Pair the squares into one W=2 batched kernel. The two follow-on
    // muls (c0 * t3 and c1 * t3) are also independent; pair them too.
    base c0_sqr;
    base c1_sqr;
    base::template montgomery_sqr_batched<2>({ &c0, &c1 }, { &c0_sqr, &c1_sqr });
    base t3 = (c0_sqr + c1_sqr).invert();
    base out_c0;
    base out_c1;
    base::template montgomery_mul_batched<2>({ &c0, &c1 }, { &t3, &t3 }, { &out_c0, &out_c1 });
    return { out_c0, -out_c1 };
}

template <class base, class T>
constexpr void field2<base, T>::self_conditional_negate(const uint64_t predicate) noexcept
{
    *this = predicate != 0U ? -(*this) : *this;
}

template <class base, class T> constexpr void field2<base, T>::self_set_msb() noexcept
{
    c0.data[3] = 0ULL | (1ULL << 63ULL);
}

template <class base, class T> constexpr bool field2<base, T>::is_msb_set() const noexcept
{
    return (c0.data[3] >> 63ULL) == 1ULL;
}

template <class base, class T> constexpr uint64_t field2<base, T>::is_msb_set_word() const noexcept
{
    return (c0.data[3] >> 63ULL);
}

template <class base, class T> constexpr bool field2<base, T>::is_zero() const noexcept
{
    return (c0.is_zero() && c1.is_zero());
}

template <class base, class T> constexpr bool field2<base, T>::operator==(const field2& other) const noexcept
{
    return (c0 == other.c0) && (c1 == other.c1);
}

template <class base, class T> constexpr field2<base, T> field2<base, T>::frobenius_map() const noexcept
{
    return { c0, -c1 };
}

template <class base, class T> constexpr void field2<base, T>::self_frobenius_map() noexcept
{
    c1.self_neg();
}

template <class base, class T> field2<base, T> field2<base, T>::random_element(numeric::RNG* engine)
{
    return { base::random_element(engine), base::random_element(engine) };
}
} // namespace bb