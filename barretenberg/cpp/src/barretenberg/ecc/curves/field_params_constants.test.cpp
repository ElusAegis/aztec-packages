/**
 * @brief Typed test fixture verifying field parameter constants for all supported curves.
 *
 * Each field is described by a Config struct that bundles:
 *   - Params:                    the field params struct (e.g. Bn254FqParams)
 *   - Field:                     the instantiated field type (e.g. bb::fq = field<Bn254FqParams>)
 *   - expected_modulus_decimal:  the expected modulus as a decimal string (ground truth reference)
 *   - has_cube_root:             whether a meaningful cube root of unity exists in this field
 *   - has_primitive_root:        whether a high-2-adicity primitive root of unity is used
 *
 * Tests cover native (64-bit limb) and WASM (29-bit limb) representations of all constants.
 *
 * Fields tested:
 *   - BN254:     Fq (base field), Fr (scalar field)
 *   - secp256k1: Fq (base field), Fr (scalar field)
 *   - secp256r1: Fq (base field), Fr (scalar field)
 *
 * Note: Grumpkin reuses BN254's fq/fr (swapped), so no separate config is needed.
 */

#include "barretenberg/ecc/curves/bn254/fq.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/ecc/curves/secp256k1/secp256k1.hpp"
#include "barretenberg/ecc/curves/secp256r1/secp256r1.hpp"
#include "barretenberg/ecc/fields/field_constexpr_helpers.hpp"
#include "barretenberg/numeric/random/engine.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include <array>
#include <gtest/gtest.h>
#include <string>

using namespace bb;

// ---- Static assertions: derivation chain ----

static_assert(bb::FIELD_BITS == 256);
static_assert(bb::R_NUM_LIMBS == (bb::FIELD_BITS + bb::R_LIMB_BITS - 1) / bb::R_LIMB_BITS);
static_assert(bb::R_EXPONENT == bb::R_LIMB_BITS * bb::R_NUM_LIMBS);

// ---- Static assertions: computed 29-bit limb constants match old hardcoded values ----

// BN254 Fq: all 9 modulus limbs
static constexpr auto bn254_fq_r_limbs = compute_limb_constants<29, 9>(Bn254FqParams::modulus_uint256);
static_assert(bn254_fq_r_limbs.modulus[0] == 0x187cfd47);
static_assert(bn254_fq_r_limbs.modulus[1] == 0x10460b6);
static_assert(bn254_fq_r_limbs.modulus[2] == 0x1c72a34f);
static_assert(bn254_fq_r_limbs.modulus[3] == 0x2d522d0);
static_assert(bn254_fq_r_limbs.modulus[4] == 0x1585d978);
static_assert(bn254_fq_r_limbs.modulus[5] == 0x2db40c0);
static_assert(bn254_fq_r_limbs.modulus[6] == 0xa6e141);
static_assert(bn254_fq_r_limbs.modulus[7] == 0xe5c2634);
static_assert(bn254_fq_r_limbs.modulus[8] == 0x30644e);
// BN254 Fq: first and last div_r_inv limbs
static_assert(bn254_fq_r_limbs.div_r_inv[0] == 0x17789a9f);
static_assert(bn254_fq_r_limbs.div_r_inv[8] == 0x6d7c4);

// BN254 Fr: all 9 modulus limbs
static constexpr auto bn254_fr_r_limbs = compute_limb_constants<29, 9>(Bn254FrParams::modulus_uint256);
static_assert(bn254_fr_r_limbs.modulus[0] == 0x10000001);
static_assert(bn254_fr_r_limbs.modulus[1] == 0x1f0fac9f);
static_assert(bn254_fr_r_limbs.modulus[2] == 0xe5c2450);
static_assert(bn254_fr_r_limbs.modulus[3] == 0x7d090f3);
static_assert(bn254_fr_r_limbs.modulus[4] == 0x1585d283);
static_assert(bn254_fr_r_limbs.modulus[5] == 0x2db40c0);
static_assert(bn254_fr_r_limbs.modulus[6] == 0xa6e141);
static_assert(bn254_fr_r_limbs.modulus[7] == 0xe5c2634);
static_assert(bn254_fr_r_limbs.modulus[8] == 0x30644e);
// BN254 Fr: first and last div_r_inv limbs
static_assert(bn254_fr_r_limbs.div_r_inv[0] == 0x18f05361);
static_assert(bn254_fr_r_limbs.div_r_inv[8] == 0x183227);

// secp256k1 Fq: first and last limbs
static constexpr auto k1_fq_r_limbs = compute_limb_constants<29, 9>(secp256k1::FqParams::modulus_uint256);
static_assert(k1_fq_r_limbs.modulus[0] == 0x1ffffc2f);
static_assert(k1_fq_r_limbs.modulus[8] == 0xffffff);
static_assert(k1_fq_r_limbs.div_r_inv[0] == 0xed6544e);
static_assert(k1_fq_r_limbs.div_r_inv[8] == 0x9129a9);

// secp256k1 Fr: first and last limbs
static constexpr auto k1_fr_r_limbs = compute_limb_constants<29, 9>(secp256k1::FrParams::modulus_uint256);
static_assert(k1_fr_r_limbs.modulus[0] == 0x10364141);
static_assert(k1_fr_r_limbs.modulus[8] == 0xffffff);
static_assert(k1_fr_r_limbs.div_r_inv[0] == 0x3d864e);
static_assert(k1_fr_r_limbs.div_r_inv[8] == 0xac4589);

// secp256r1 Fq: first and last limbs
static constexpr auto r1_fq_r_limbs = compute_limb_constants<29, 9>(secp256r1::FqParams::modulus_uint256);
static_assert(r1_fq_r_limbs.modulus[0] == 0x1fffffff);
static_assert(r1_fq_r_limbs.modulus[8] == 0xffffff);
static_assert(r1_fq_r_limbs.div_r_inv[0] == 0x0);
static_assert(r1_fq_r_limbs.div_r_inv[8] == 0x0);

// secp256r1 Fr: first and last limbs
static constexpr auto r1_fr_r_limbs = compute_limb_constants<29, 9>(secp256r1::FrParams::modulus_uint256);
static_assert(r1_fr_r_limbs.modulus[0] == 0x1c632551);
static_assert(r1_fr_r_limbs.modulus[8] == 0xffffff);
static_assert(r1_fr_r_limbs.div_r_inv[0] == 0x8517c79);
static_assert(r1_fr_r_limbs.div_r_inv[8] == 0x7005e2);

// ---- end static assertions ----

namespace {

uint256_t from_decimal(const std::string& dec_str)
{
    uint256_t result = 0;
    for (char c : dec_str) {
        result = result * 10 + static_cast<uint64_t>(c - '0');
    }
    return result;
}

struct Bn254FqTestConfig {
    using Params = Bn254FqParams;
    using Field = bb::fq;
    // BN254 base field prime q
    // References: https://eips.ethereum.org/EIPS/eip-196, https://hackmd.io/@jpw/bn254
    static constexpr const char* expected_modulus_decimal =
        "21888242871839275222246405745257275088696311157297823662689037894645226208583";
    static constexpr bool has_cube_root = true;
    static constexpr bool has_primitive_root = false;
};

struct Bn254FrTestConfig {
    using Params = Bn254FrParams;
    using Field = bb::fr;
    // BN254 scalar field prime r (also Baby Jubjub base field)
    // References: https://eips.ethereum.org/EIPS/eip-196, https://hackmd.io/@jpw/bn254
    static constexpr const char* expected_modulus_decimal =
        "21888242871839275222246405745257275088548364400416034343698204186575808495617";
    static constexpr bool has_cube_root = true;
    static constexpr bool has_primitive_root = true;
};

struct Secp256k1FqTestConfig {
    using Params = secp256k1::FqParams;
    using Field = secp256k1::fq;
    // secp256k1 base field prime p = 2^256 - 2^32 - 977
    // Reference: https://www.secg.org/sec2-v2.pdf
    static constexpr const char* expected_modulus_decimal =
        "115792089237316195423570985008687907853269984665640564039457584007908834671663";
    static constexpr bool has_cube_root = true;
    static constexpr bool has_primitive_root = false;
};

struct Secp256k1FrTestConfig {
    using Params = secp256k1::FrParams;
    using Field = secp256k1::fr;
    // secp256k1 scalar field order
    // Reference: https://www.secg.org/sec2-v2.pdf
    static constexpr const char* expected_modulus_decimal =
        "115792089237316195423570985008687907852837564279074904382605163141518161494337";
    static constexpr bool has_cube_root = true;
    static constexpr bool has_primitive_root = false;
};

struct Secp256r1FqTestConfig {
    using Params = secp256r1::FqParams;
    using Field = secp256r1::fq;
    // secp256r1 (P-256) base field prime p = 2^256 - 2^224 + 2^192 + 2^96 - 1
    // Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    static constexpr const char* expected_modulus_decimal =
        "115792089210356248762697446949407573530086143415290314195533631308867097853951";
    static constexpr bool has_cube_root = false;
    static constexpr bool has_primitive_root = false;
};

struct Secp256r1FrTestConfig {
    using Params = secp256r1::FrParams;
    using Field = secp256r1::fr;
    // secp256r1 (P-256) scalar field order
    // Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    static constexpr const char* expected_modulus_decimal =
        "115792089210356248762697446949407573529996955224135760342422259061068512044369";
    static constexpr bool has_cube_root = false;
    static constexpr bool has_primitive_root = false;
};

} // namespace

template <typename Config> class FieldConstantsTest : public testing::Test {};

TYPED_TEST_SUITE_P(FieldConstantsTest);

TYPED_TEST_P(FieldConstantsTest, Modulus)
{
    using Params = typename TypeParam::Params;
    uint256_t expected = from_decimal(TypeParam::expected_modulus_decimal);
    uint256_t actual{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    EXPECT_EQ(expected, actual);
}

// Verify R^2 mod p is correctly derived from bb::R_EXPONENT
TYPED_TEST_P(FieldConstantsTest, RSquared)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    uint256_t expected = compute_r_squared(mod, bb::R_EXPONENT);
    uint256_t actual{ Params::r_squared_0, Params::r_squared_1, Params::r_squared_2, Params::r_squared_3 };
    EXPECT_EQ(expected, actual);
}

// Also verify R^2 for both R=256 and R=261 are correct (cross-platform validation)
TYPED_TEST_P(FieldConstantsTest, RSquaredBothPlatforms)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };

    // Verify R=2^256 path
    uint512_t R256 = (uint512_t(1) << 256) % mod;
    uint256_t r_sq_256 = ((R256 * R256) % mod).lo;
    EXPECT_EQ(r_sq_256, compute_r_squared(mod, 256));

    // Verify R=2^261 path
    uint512_t R261 = (uint512_t(1) << 261) % mod;
    uint256_t r_sq_261 = ((R261 * R261) % mod).lo;
    EXPECT_EQ(r_sq_261, compute_r_squared(mod, 261));
}

TYPED_TEST_P(FieldConstantsTest, RInv)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    uint512_t two_64 = uint512_t(1) << 64;
    uint512_t neg_mod{ -mod, 0 };
    uint64_t expected = neg_mod.invmod(two_64).lo.data[0];
    EXPECT_EQ(Params::r_inv, expected);
}

TYPED_TEST_P(FieldConstantsTest, PowMinus64)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    uint512_t two_64 = uint512_t(1) << 64;
    uint256_t expected = two_64.invmod(mod).lo;
    EXPECT_EQ(expected.data[0], Params::r_inv_0);
    EXPECT_EQ(expected.data[1], Params::r_inv_1);
    EXPECT_EQ(expected.data[2], Params::r_inv_2);
    EXPECT_EQ(expected.data[3], Params::r_inv_3);
}

TYPED_TEST_P(FieldConstantsTest, CubeRootOfUnity)
{
    if constexpr (!TypeParam::has_cube_root) {
        GTEST_SKIP() << "Cube root of unity is not defined for this field";
    } else {
        using Field = typename TypeParam::Field;
        Field beta = Field::cube_root_of_unity();
        EXPECT_EQ(beta * beta * beta, Field::one());
        EXPECT_NE(beta, Field::one());
    }
}

TYPED_TEST_P(FieldConstantsTest, PrimitiveRootOfUnity)
{
    if constexpr (!TypeParam::has_primitive_root) {
        GTEST_SKIP() << "Primitive root of unity is not used for this field";
    } else {
        using Field = typename TypeParam::Field;
        size_t order = Field::primitive_root_log_size();
        Field root = Field::get_root_of_unity(order);
        for (size_t i = 0; i < order; i++) {
            EXPECT_NE(root, Field::one());
            root = root.sqr();
        }
        EXPECT_EQ(root, Field::one());
    }
}

TYPED_TEST_P(FieldConstantsTest, CosetGenerator)
{
    using Params = typename TypeParam::Params;
    using Field = typename TypeParam::Field;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    Field coset_gen = Field::coset_generator();
    EXPECT_NE(coset_gen.pow((mod - 1) / 2), Field::one());
}

// Verify Montgomery-form constants are correctly derived from canonical values
TYPED_TEST_P(FieldConstantsTest, MontgomeryFormDerivation)
{
    using Params = typename TypeParam::Params;
    using Field = typename TypeParam::Field;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };

    // Verify coset_generator: field(canonical) should equal the Montgomery-form constant
    Field coset_from_canonical(Params::canonical_coset_generator);
    Field coset_from_limbs{
        Params::coset_generator_0, Params::coset_generator_1, Params::coset_generator_2, Params::coset_generator_3
    };
    EXPECT_EQ(coset_from_canonical, coset_from_limbs);
}

// Verify computed 29-bit modulus limbs reconstruct to the original modulus
TYPED_TEST_P(FieldConstantsTest, WasmModulusConsistency)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    constexpr auto lc = compute_limb_constants<29, 9>(Params::modulus_uint256);
    uint512_t reconstructed = 0;
    for (size_t i = 0; i < 9; i++) {
        reconstructed += uint512_t(lc.modulus[i]) << (29UL * i);
        EXPECT_LT(lc.modulus[i], uint64_t(1) << 29);
    }
    EXPECT_EQ(reconstructed.lo, mod);
    EXPECT_EQ(reconstructed.hi, uint256_t(0));
}

// Verify computed 2^{-29} mod p matches independent calculation
TYPED_TEST_P(FieldConstantsTest, WasmPowMinus29)
{
    using Params = typename TypeParam::Params;
    uint256_t mod{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };
    constexpr auto lc = compute_limb_constants<29, 9>(Params::modulus_uint256);
    uint512_t r_inv_wasm = 0;
    for (size_t i = 0; i < 9; i++) {
        r_inv_wasm += uint512_t(lc.div_r_inv[i]) << (29UL * i);
        EXPECT_LT(lc.div_r_inv[i], uint64_t(1) << 29);
    }
    uint512_t two_29 = uint512_t(1) << 29;
    uint512_t expected = two_29.invmod(mod);
    EXPECT_EQ(r_inv_wasm, expected);
    EXPECT_LT(r_inv_wasm, uint512_t(mod));
}

REGISTER_TYPED_TEST_SUITE_P(FieldConstantsTest,
                            Modulus,
                            RSquared,
                            RSquaredBothPlatforms,
                            RInv,
                            PowMinus64,
                            CubeRootOfUnity,
                            PrimitiveRootOfUnity,
                            CosetGenerator,
                            MontgomeryFormDerivation,
                            WasmModulusConsistency,
                            WasmPowMinus29);

using FieldTestTypes = ::testing::Types<Bn254FqTestConfig,
                                        Bn254FrTestConfig,
                                        Secp256k1FqTestConfig,
                                        Secp256k1FrTestConfig,
                                        Secp256r1FqTestConfig,
                                        Secp256r1FrTestConfig>;

INSTANTIATE_TYPED_TEST_SUITE_P(AllFields, FieldConstantsTest, FieldTestTypes);
