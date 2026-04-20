/**
 * @brief Correctness tests for the BN254 Fermat addition chain inversions.
 *
 * The addition chains in `bn254_fermat_addchain.hpp` encode `x^(modulus-2)` for
 * the BN254 Fr and Fq primes. These tests compare the chain output against a
 * reference binary exponent-scan that raises `x` to `modulus_minus_two`, and
 * also check the inverse identity `x * x^{-1} == 1`.
 */

#include "barretenberg/ecc/curves/bn254/fq.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/ecc/fields/bn254_fermat_addchain.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include <gtest/gtest.h>

using namespace bb;

namespace {

auto& engine = numeric::get_debug_randomness();

// Reference implementation: unambiguous binary square-and-multiply over
// `modulus_minus_two`. Used instead of `field::pow()` because that function
// is the exact thing we are trying to optimize; keeping the reference fully
// independent makes the correctness proof more convincing.
template <class F> F reference_invert(const F& x)
{
    if (x == F::zero()) {
        return F::zero();
    }
    const numeric::uint256_t e = F::modulus_minus_two;
    const uint64_t msb = e.get_msb();

    F accumulator = x;
    for (int i = static_cast<int>(msb) - 1; i >= 0; --i) {
        accumulator.self_sqr();
        if (e.get_bit(static_cast<uint64_t>(i))) {
            accumulator *= x;
        }
    }
    return accumulator;
}

template <class F> F addchain_invert(const F& x)
{
    using namespace bb::detail::bn254_fermat_addchain;
    if constexpr (params_modulus_equals<typename F::Params>(bn254_fr_modulus)) {
        return invert_bn254_fr(x);
    } else {
        static_assert(params_modulus_equals<typename F::Params>(bn254_fq_modulus));
        return invert_bn254_fq(x);
    }
}

} // namespace

TEST(BN254FermatAddchain, FrMatchesReferenceOnRandom)
{
    constexpr size_t N = 1000;
    for (size_t i = 0; i < N; ++i) {
        fr x = fr::random_element(&engine);
        if (x.is_zero()) {
            continue;
        }
        const fr ref = reference_invert(x);
        const fr got = addchain_invert(x);
        ASSERT_EQ(ref, got) << "Fr addchain mismatch at iteration " << i;
    }
}

TEST(BN254FermatAddchain, FqMatchesReferenceOnRandom)
{
    constexpr size_t N = 1000;
    for (size_t i = 0; i < N; ++i) {
        fq x = fq::random_element(&engine);
        if (x.is_zero()) {
            continue;
        }
        const fq ref = reference_invert(x);
        const fq got = addchain_invert(x);
        ASSERT_EQ(ref, got) << "Fq addchain mismatch at iteration " << i;
    }
}

TEST(BN254FermatAddchain, FieldInvertProducesMultiplicativeInverse)
{
    constexpr size_t N = 1000;
    for (size_t i = 0; i < N; ++i) {
        fr xr = fr::random_element(&engine);
        if (!xr.is_zero()) {
            ASSERT_EQ(xr * xr.invert(), fr::one()) << "Fr invert round-trip failed at " << i;
        }
        fq xq = fq::random_element(&engine);
        if (!xq.is_zero()) {
            ASSERT_EQ(xq * xq.invert(), fq::one()) << "Fq invert round-trip failed at " << i;
        }
    }
}

TEST(BN254FermatAddchain, HandlesOneAndNegOne)
{
    // x = 1  -> invert = 1
    EXPECT_EQ(fr::one().invert(), fr::one());
    EXPECT_EQ(fq::one().invert(), fq::one());
    // x = -1 -> invert = -1 (self-inverse)
    EXPECT_EQ((-fr::one()).invert(), -fr::one());
    EXPECT_EQ((-fq::one()).invert(), -fq::one());
}

#ifdef MONTMUL_VARIANT_FMA
static_assert(bb::R_EXPONENT == 264, "FMA build must compile chain against R=2^264 backend");
#endif
