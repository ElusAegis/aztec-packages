// Differential test: runtime RNE backend vs uint512 modular reference.
//
// When MONTMUL_VARIANT_RNE is active the RNE kernel is the live backend for
// Bn254Fr and (empirically) Bn254Fq. Any disagreement with the
// representation-independent reference is a kernel bug. On mismatch the
// inputs and Montgomery limb outputs are printed so the offending case can
// be minimized outside the test harness.
//
// Also includes a BN254 R_EXPONENT check (expected == 256 under RNE) so the
// build can't silently fall through to a different backend.

#include "barretenberg/ecc/curves/bn254/fq.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/ecc/fields/field_montgomery_config.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include <array>
#include <cstdio>
#include <gtest/gtest.h>

#if BB_R_LIMB_BITS == 64 && defined(MONTMUL_VARIANT_RNE) && defined(__wasm_simd128__)

using namespace bb;

namespace {

auto& engine = numeric::get_debug_randomness();

template <class Field> void dump(const char* label, const Field& f)
{
    std::fprintf(stderr,
                 "  %s = {0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx}\n",
                 label,
                 static_cast<unsigned long long>(f.data[0]),
                 static_cast<unsigned long long>(f.data[1]),
                 static_cast<unsigned long long>(f.data[2]),
                 static_cast<unsigned long long>(f.data[3]));
}

void dump_canonical(const char* label, const uint256_t& x)
{
    std::fprintf(stderr,
                 "  %s (canonical) = {0x%016llx, 0x%016llx, 0x%016llx, 0x%016llx}\n",
                 label,
                 static_cast<unsigned long long>(x.data[0]),
                 static_cast<unsigned long long>(x.data[1]),
                 static_cast<unsigned long long>(x.data[2]),
                 static_cast<unsigned long long>(x.data[3]));
}

// Force a runtime Montgomery multiply — inlining would fold back into the
// constexpr reference path, never exercising the RNE SIMD kernel.
template <class Field> [[gnu::noinline]] Field runtime_mul(Field a, Field b)
{
    return a * b;
}
template <class Field> [[gnu::noinline]] Field runtime_sqr(Field a)
{
    return a.sqr();
}

} // namespace

// Compile-time sanity check — if MONTMUL_VARIANT_RNE is active the config
// file must pick R_EXPONENT == 256 (4×64-bit, matching native layout).
static_assert(bb::R_EXPONENT == 256, "RNE build must have R_EXPONENT == 256");

TEST(RneDifferential, ConfigSanity)
{
    // Runtime print so bench logs show which backend is actually live.
    std::fprintf(stderr, "[rne] bb::R_EXPONENT = %u  (expected 256)\n", bb::R_EXPONENT);
    EXPECT_EQ(bb::R_EXPONENT, 256U);
}

TEST(RneDifferential, Bn254FrMul)
{
    size_t failures = 0;
    constexpr size_t iterations = 1024;
    for (size_t i = 0; i < iterations; ++i) {
        uint256_t a_can = engine.get_random_uint256() % bb::fr::modulus;
        uint256_t b_can = engine.get_random_uint256() % bb::fr::modulus;

        // Reference: canonical × canonical mod p via uint512 (R-independent).
        uint512_t prod = uint512_t(a_can) * uint512_t(b_can);
        uint256_t expected_can = (prod % uint512_t(bb::fr::modulus)).lo;

        // Runtime path: canonical → Montgomery (constexpr conversion), runtime
        // mul (RNE kernel), Montgomery → canonical.
        bb::fr a(a_can);
        bb::fr b(b_can);
        bb::fr product = runtime_mul(a, b);
        uint256_t actual_can = static_cast<uint256_t>(product);

        if (actual_can != expected_can) {
            if (failures < 3) {
                std::fprintf(stderr, "[rne] Bn254FrMul MISMATCH (iter %zu):\n", i);
                dump_canonical("a", a_can);
                dump_canonical("b", b_can);
                dump<bb::fr>("a_mont", a);
                dump<bb::fr>("b_mont", b);
                dump<bb::fr>("product_mont", product);
                dump_canonical("expected", expected_can);
                dump_canonical("actual", actual_can);
            }
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0U) << "RNE Montgomery mul disagrees with uint512 reference in " << failures << "/"
                            << iterations << " iterations";
}

TEST(RneDifferential, Bn254FrSqr)
{
    size_t failures = 0;
    constexpr size_t iterations = 1024;
    for (size_t i = 0; i < iterations; ++i) {
        uint256_t a_can = engine.get_random_uint256() % bb::fr::modulus;

        uint512_t prod = uint512_t(a_can) * uint512_t(a_can);
        uint256_t expected_can = (prod % uint512_t(bb::fr::modulus)).lo;

        bb::fr a(a_can);
        bb::fr squared = runtime_sqr(a);
        uint256_t actual_can = static_cast<uint256_t>(squared);

        if (actual_can != expected_can) {
            if (failures < 3) {
                std::fprintf(stderr, "[rne] Bn254FrSqr MISMATCH (iter %zu):\n", i);
                dump_canonical("a", a_can);
                dump<bb::fr>("a_mont", a);
                dump<bb::fr>("squared_mont", squared);
                dump_canonical("expected", expected_can);
                dump_canonical("actual", actual_can);
            }
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0U) << "RNE Montgomery sqr disagrees with uint512 reference in " << failures << "/"
                            << iterations << " iterations";
}

// Edge cases: operands at the boundary of the 51-bit limb structure. Checks
// that carry propagation into the fifth 51-bit limb (which only holds the
// top 51 bits of the 255-bit value) is handled correctly.
TEST(RneDifferential, Bn254FrEdgeCases)
{
    struct Case {
        const char* name;
        uint256_t a;
        uint256_t b;
    };

    const std::array<Case, 6> cases = { {
        { "zero * random",
          uint256_t{ 0, 0, 0, 0 },
          uint256_t{ 0xdeadbeefcafebabeULL, 0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x1111222233334444ULL } },
        { "one * one", uint256_t{ 1, 0, 0, 0 }, uint256_t{ 1, 0, 0, 0 } },
        { "modulus-1 squared",
          bb::fr::modulus - uint256_t{ 1, 0, 0, 0 },
          bb::fr::modulus - uint256_t{ 1, 0, 0, 0 } },
        { "all 1s (saturated low bits)",
          uint256_t{ 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0FFFFFFFFFFFFFFFULL } %
              bb::fr::modulus,
          uint256_t{ 1, 0, 0, 0 } },
        { "2^204 * 2^51 (touches top limb)",
          uint256_t{ 0, 0, 0, 1ULL << 12 } % bb::fr::modulus,
          uint256_t{ 0, 1ULL << (51 - 13), 0, 0 } % bb::fr::modulus },
        { "near-top-limb cross",
          uint256_t{ 0x0123456789abcdefULL, 0, 0x0FEDCBA987654321ULL, 0 } % bb::fr::modulus,
          uint256_t{ 0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL } %
              bb::fr::modulus } } };

    for (const auto& c : cases) {
        uint512_t prod = uint512_t(c.a) * uint512_t(c.b);
        uint256_t expected_can = (prod % uint512_t(bb::fr::modulus)).lo;

        bb::fr a(c.a);
        bb::fr b(c.b);
        bb::fr product = runtime_mul(a, b);
        uint256_t actual_can = static_cast<uint256_t>(product);

        if (actual_can != expected_can) {
            std::fprintf(stderr, "[rne] edge case '%s' MISMATCH:\n", c.name);
            dump_canonical("a", c.a);
            dump_canonical("b", c.b);
            dump<bb::fr>("product_mont", product);
            dump_canonical("expected", expected_can);
            dump_canonical("actual", actual_can);
        }
        EXPECT_EQ(actual_can, expected_can) << "edge case: " << c.name;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bn254Fq variants. Fq is on the RNE whitelist by empirical extension —
// provekit's bounds proof is Fr-only. These tests backstop the
// "Fr bounds probably cover Fq" assumption.
// ─────────────────────────────────────────────────────────────────────

TEST(RneDifferential, Bn254FqMul)
{
    size_t failures = 0;
    constexpr size_t iterations = 1024;
    for (size_t i = 0; i < iterations; ++i) {
        uint256_t a_can = engine.get_random_uint256() % bb::fq::modulus;
        uint256_t b_can = engine.get_random_uint256() % bb::fq::modulus;

        uint512_t prod = uint512_t(a_can) * uint512_t(b_can);
        uint256_t expected_can = (prod % uint512_t(bb::fq::modulus)).lo;

        bb::fq a(a_can);
        bb::fq b(b_can);
        bb::fq product = runtime_mul(a, b);
        uint256_t actual_can = static_cast<uint256_t>(product);

        if (actual_can != expected_can) {
            if (failures < 3) {
                std::fprintf(stderr, "[rne] Bn254FqMul MISMATCH (iter %zu):\n", i);
                dump_canonical("a", a_can);
                dump_canonical("b", b_can);
                dump<bb::fq>("a_mont", a);
                dump<bb::fq>("b_mont", b);
                dump<bb::fq>("product_mont", product);
                dump_canonical("expected", expected_can);
                dump_canonical("actual", actual_can);
            }
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0U) << "RNE Montgomery mul (Fq) disagrees with uint512 reference in " << failures << "/"
                            << iterations << " iterations";
}

TEST(RneDifferential, Bn254FqSqr)
{
    size_t failures = 0;
    constexpr size_t iterations = 1024;
    for (size_t i = 0; i < iterations; ++i) {
        uint256_t a_can = engine.get_random_uint256() % bb::fq::modulus;

        uint512_t prod = uint512_t(a_can) * uint512_t(a_can);
        uint256_t expected_can = (prod % uint512_t(bb::fq::modulus)).lo;

        bb::fq a(a_can);
        bb::fq squared = runtime_sqr(a);
        uint256_t actual_can = static_cast<uint256_t>(squared);

        if (actual_can != expected_can) {
            if (failures < 3) {
                std::fprintf(stderr, "[rne] Bn254FqSqr MISMATCH (iter %zu):\n", i);
                dump_canonical("a", a_can);
                dump<bb::fq>("a_mont", a);
                dump<bb::fq>("squared_mont", squared);
                dump_canonical("expected", expected_can);
                dump_canonical("actual", actual_can);
            }
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0U) << "RNE Montgomery sqr (Fq) disagrees with uint512 reference in " << failures << "/"
                            << iterations << " iterations";
}

TEST(RneDifferential, Bn254FqEdgeCases)
{
    struct Case {
        const char* name;
        uint256_t a;
        uint256_t b;
    };

    const std::array<Case, 6> cases = { {
        { "zero * random",
          uint256_t{ 0, 0, 0, 0 },
          uint256_t{ 0xdeadbeefcafebabeULL, 0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x1111222233334444ULL } },
        { "one * one", uint256_t{ 1, 0, 0, 0 }, uint256_t{ 1, 0, 0, 0 } },
        { "modulus-1 squared",
          bb::fq::modulus - uint256_t{ 1, 0, 0, 0 },
          bb::fq::modulus - uint256_t{ 1, 0, 0, 0 } },
        { "all 1s (saturated low bits)",
          uint256_t{ 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0FFFFFFFFFFFFFFFULL } %
              bb::fq::modulus,
          uint256_t{ 1, 0, 0, 0 } },
        { "2^204 * 2^51 (touches top limb)",
          uint256_t{ 0, 0, 0, 1ULL << 12 } % bb::fq::modulus,
          uint256_t{ 0, 1ULL << (51 - 13), 0, 0 } % bb::fq::modulus },
        { "near-top-limb cross",
          uint256_t{ 0x0123456789abcdefULL, 0, 0x0FEDCBA987654321ULL, 0 } % bb::fq::modulus,
          uint256_t{ 0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL } %
              bb::fq::modulus } } };

    for (const auto& c : cases) {
        uint512_t prod = uint512_t(c.a) * uint512_t(c.b);
        uint256_t expected_can = (prod % uint512_t(bb::fq::modulus)).lo;

        bb::fq a(c.a);
        bb::fq b(c.b);
        bb::fq product = runtime_mul(a, b);
        uint256_t actual_can = static_cast<uint256_t>(product);

        if (actual_can != expected_can) {
            std::fprintf(stderr, "[rne] Fq edge case '%s' MISMATCH:\n", c.name);
            dump_canonical("a", c.a);
            dump_canonical("b", c.b);
            dump<bb::fq>("product_mont", product);
            dump_canonical("expected", expected_can);
            dump_canonical("actual", actual_can);
        }
        EXPECT_EQ(actual_can, expected_can) << "Fq edge case: " << c.name;
    }
}

#endif // BB_R_LIMB_BITS == 64 && MONTMUL_VARIANT_RNE && __wasm_simd128__
