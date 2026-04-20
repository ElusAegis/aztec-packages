// === AUDIT STATUS ===
// internal:    { status: Planned, auditors: [Raju], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "barretenberg/numeric/uint256/uint256.hpp"
#include <array>

// forward declare RNG
namespace bb::numeric {
class RNG;
}

namespace bb {
template <class base_field, class Params> struct alignas(32) field2 {
  public:
    static constexpr size_t PUBLIC_INPUTS_SIZE = base_field::PUBLIC_INPUTS_SIZE + base_field::PUBLIC_INPUTS_SIZE;

    constexpr field2(const base_field& a = base_field::zero(), const base_field& b = base_field::zero())
        : c0(a)
        , c1(b)
    {}

    constexpr field2(const field2& other) noexcept
        : c0(other.c0)
        , c1(other.c1)
    {}
    constexpr field2(field2&& other) noexcept
        : c0(other.c0)
        , c1(other.c1)
    {}

    constexpr field2& operator=(const field2& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        c0 = other.c0;
        c1 = other.c1;
        return *this;
    }

    constexpr field2& operator=(field2&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        c0 = other.c0;
        c1 = other.c1;
        return *this;
    }

    constexpr ~field2() noexcept = default;

    base_field c0;
    base_field c1;

    static constexpr uint256_t modulus = base_field::modulus;

    static constexpr field2 zero() { return field2{ base_field::zero(), base_field::zero() }; }
    static constexpr field2 one() { return field2{ base_field::one(), base_field::zero() }; }
    static constexpr field2 twist_coeff_b() { return field2{ Params::twist_coeff_b_0, Params::twist_coeff_b_1 }; }
    static constexpr field2 frobenius_on_twisted_curve_x()
    {
        return field2{ Params::frobenius_on_twisted_curve_x_0, Params::frobenius_on_twisted_curve_x_1 };
    }
    static constexpr field2 frobenius_on_twisted_curve_y()
    {
        return field2{ Params::frobenius_on_twisted_curve_y_0, Params::frobenius_on_twisted_curve_y_1 };
    }

    constexpr field2 operator*(const field2& other) const noexcept;
    constexpr field2 operator+(const field2& other) const noexcept;
    constexpr field2 operator-(const field2& other) const noexcept;
    constexpr field2 operator-() const noexcept;
    constexpr field2 operator/(const field2& other) const noexcept;

    constexpr field2 operator*=(const field2& other) noexcept;
    constexpr field2 operator+=(const field2& other) noexcept;
    constexpr field2 operator-=(const field2& other) noexcept;
    constexpr field2 operator/=(const field2& other) noexcept;

    constexpr field2 mul_by_fq(const base_field& a) const noexcept
    {
        field2 r{ a * c0, a * c1 };
        return r;
    }

    constexpr bool operator==(const field2& other) const noexcept;
    constexpr bool operator!=(const field2& other) const noexcept { return !(*this == other); }
    constexpr field2 sqr() const noexcept;
    constexpr void self_sqr() noexcept;

    // Batched multiplication: *outs[i] = *as[i] * *bs[i] for i in [0, N).
    //
    // For N=2, the 6 independent Fp muls inside two Fq2 Karatsuba mults are
    // co-scheduled as 3 paired Fp-level montgomery_mul_batched<2> dispatches
    // (two base products per slot + one cross-term per slot). On the WASM
    // FMA-SIMD backend this issues 3 f64x2 paired kernels instead of 6 W=1
    // singles, cutting the per-Fq2-mul cost materially. Fq6 Karatsuba mul
    // issues 6 independent Fq2 muls in 3 pairs, the intended caller.
    //
    // For N=1 or N>=3, falls back to sequential Fq2 muls (each of which is
    // 3 sequential Fp muls via operator*).
    //
    // Exists so that element_impl.hpp can call Fq::montgomery_mul_batched
    // uniformly across Fq and Fq2 base fields.
    template <size_t N>
    BB_INLINE static constexpr void montgomery_mul_batched(std::array<const field2*, N> as,
                                                           std::array<const field2*, N> bs,
                                                           std::array<field2*, N> outs) noexcept
    {
        if constexpr (N == 2) {
            // Assumes -1 is not a quadratic residue (same invariant as operator*).
            static_assert((base_field::modulus.data[0] & 0x3UL) == 0x3UL);

            // Snapshot inputs into locals first. Aliasing between any input pointer
            // and any output pointer is legal at the caller; the reads must happen
            // before we start writing into *outs.
            const base_field a0c0 = as[0]->c0;
            const base_field a0c1 = as[0]->c1;
            const base_field a1c0 = as[1]->c0;
            const base_field a1c1 = as[1]->c1;
            const base_field b0c0 = bs[0]->c0;
            const base_field b0c1 = bs[0]->c1;
            const base_field b1c0 = bs[1]->c0;
            const base_field b1c1 = bs[1]->c1;

            // Precompute cross-term sums (Fq2 add = two Fp adds, no mul).
            const base_field a0_sum = a0c0 + a0c1;
            const base_field a1_sum = a1c0 + a1c1;
            const base_field b0_sum = b0c0 + b0c1;
            const base_field b1_sum = b1c0 + b1c1;

            // Pair 1: c0 * other.c0 for both slots.
            base_field t1_0;
            base_field t1_1;
            base_field::template montgomery_mul_batched<2>(
                { &a0c0, &a1c0 }, { &b0c0, &b1c0 }, { &t1_0, &t1_1 });

            // Pair 2: c1 * other.c1 for both slots.
            base_field t2_0;
            base_field t2_1;
            base_field::template montgomery_mul_batched<2>(
                { &a0c1, &a1c1 }, { &b0c1, &b1c1 }, { &t2_0, &t2_1 });

            // Pair 3: (c0+c1) * (other.c0+other.c1) for both slots.
            base_field t3_0;
            base_field t3_1;
            base_field::template montgomery_mul_batched<2>(
                { &a0_sum, &a1_sum }, { &b0_sum, &b1_sum }, { &t3_0, &t3_1 });

            // Reconstruct. Matches operator*'s formula exactly:
            //   c0 = t1 - t2,  c1 = t3 - (t1 + t2)
            *outs[0] = field2{ t1_0 - t2_0, t3_0 - (t1_0 + t2_0) };
            *outs[1] = field2{ t1_1 - t2_1, t3_1 - (t1_1 + t2_1) };
        } else {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = *as[i] * *bs[i];
            }
        }
    }

    // Batched squaring: *outs[i] = *as[i]^2 for i in [0, N).
    //
    // For N=2, the 4 independent Fp muls inside two Fq2 complex-squarings are
    // co-scheduled as 2 paired Fp-level montgomery_mul_batched<2> dispatches
    // (one "norm" pair (c0+c1)(c0-c1) and one "cross" pair c0*c1). On the
    // WASM FMA-SIMD backend this issues 2 f64x2 paired kernels instead of
    // 4 W=1 singles.
    //
    // For N=1 or N>=3, falls back to sequential Fq2 squares.
    //
    // Exists so element_impl.hpp can call Fq::montgomery_sqr_batched
    // uniformly (analogous to montgomery_mul_batched above).
    template <size_t N>
    BB_INLINE static constexpr void montgomery_sqr_batched(std::array<const field2*, N> as,
                                                           std::array<field2*, N> outs) noexcept
    {
        if constexpr (N == 2) {
            // Snapshot inputs before writing outputs (aliasing-safe).
            const base_field a0c0 = as[0]->c0;
            const base_field a0c1 = as[0]->c1;
            const base_field a1c0 = as[1]->c0;
            const base_field a1c1 = as[1]->c1;

            // Norm factors: (c0+c1) and (c0-c1).
            const base_field a0_sum = a0c0 + a0c1;
            const base_field a0_diff = a0c0 - a0c1;
            const base_field a1_sum = a1c0 + a1c1;
            const base_field a1_diff = a1c0 - a1c1;

            // Pair 1: (c0+c1)*(c0-c1) for both slots.
            base_field norm_0;
            base_field norm_1;
            base_field::template montgomery_mul_batched<2>(
                { &a0_sum, &a1_sum }, { &a0_diff, &a1_diff }, { &norm_0, &norm_1 });

            // Pair 2: c0 * c1 for both slots (will be doubled below).
            base_field cross_0;
            base_field cross_1;
            base_field::template montgomery_mul_batched<2>(
                { &a0c0, &a1c0 }, { &a0c1, &a1c1 }, { &cross_0, &cross_1 });

            *outs[0] = field2{ norm_0, cross_0 + cross_0 };
            *outs[1] = field2{ norm_1, cross_1 + cross_1 };
        } else {
            for (size_t i = 0; i < N; ++i) {
                *outs[i] = as[i]->sqr();
            }
        }
    }

    constexpr field2 pow(const uint256_t& exponent) const noexcept;
    constexpr field2 pow(uint64_t exponent) const noexcept;

    constexpr field2 invert() const noexcept;

    constexpr void self_neg() noexcept;
    constexpr field2 to_montgomery_form() const noexcept;
    constexpr field2 from_montgomery_form() const noexcept;

    constexpr void self_to_montgomery_form() noexcept;
    constexpr void self_from_montgomery_form() noexcept;

    constexpr void self_conditional_negate(uint64_t predicate) noexcept;

    constexpr field2 reduce_once() const noexcept;
    constexpr void self_reduce_once() noexcept;

    constexpr void self_set_msb() noexcept;
    [[nodiscard]] constexpr bool is_msb_set() const noexcept;
    [[nodiscard]] constexpr uint64_t is_msb_set_word() const noexcept;

    [[nodiscard]] constexpr bool is_zero() const noexcept;

    constexpr field2 frobenius_map() const noexcept;
    constexpr void self_frobenius_map() noexcept;

    static field2 random_element(numeric::RNG* engine = nullptr);
    static void serialize_to_buffer(const field2& value, uint8_t* buffer)
    {
        base_field::serialize_to_buffer(value.c0, buffer);
        base_field::serialize_to_buffer(value.c1, buffer + sizeof(base_field));
    }

    static field2 serialize_from_buffer(uint8_t* buffer)
    {
        field2 result{ base_field::zero(), base_field::zero() };
        result.c0 = base_field::serialize_from_buffer(buffer);
        result.c1 = base_field::serialize_from_buffer(buffer + sizeof(base_field));

        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const field2& a)
    {
        os << a.c0 << " , " << a.c1;
        return os;
    }
};

template <typename B, typename base_field, typename Params> void read(B& it, field2<base_field, Params>& value)
{
    using serialize::read;
    read(it, value.c0);
    read(it, value.c1);
}
template <typename B, typename base_field, typename Params> void write(B& buf, field2<base_field, Params> const& value)
{
    using serialize::write;
    write(buf, value.c0);
    write(buf, value.c1);
}
} // namespace bb
