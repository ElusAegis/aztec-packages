// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../fields/field.hpp"
#include "../../fields/field_constexpr_helpers.hpp"
#include "../../groups/group.hpp"
#include "../types.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)

namespace bb::secp256k1 {

/**
 * @brief Parameters defining the base field of the secp256k1 curve.
 */
struct FqParams {
    static constexpr uint64_t modulus_0 = 0xFFFFFFFEFFFFFC2FULL;
    static constexpr uint64_t modulus_1 = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint64_t modulus_2 = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint64_t modulus_3 = 0xFFFFFFFFFFFFFFFFULL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    static constexpr unsigned R_EXPONENT = 256;
#else
    static constexpr unsigned R_EXPONENT = 261;
#endif

    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, R_EXPONENT);
    static constexpr uint64_t r_squared_0 = r_squared_uint256.data[0];
    static constexpr uint64_t r_squared_1 = r_squared_uint256.data[1];
    static constexpr uint64_t r_squared_2 = r_squared_uint256.data[2];
    static constexpr uint64_t r_squared_3 = r_squared_uint256.data[3];

    static constexpr uint64_t r_inv = 15580212934572586289ULL;

    static constexpr uint64_t r_inv_0 = 0xffffffff27c7f3a9UL;
    static constexpr uint64_t r_inv_1 = 0xffffffffffffffffUL;
    static constexpr uint64_t r_inv_2 = 0xffffffffffffffffUL;
    static constexpr uint64_t r_inv_3 = 0xd838091dd2253530UL;

    static constexpr uint256_t canonical_cube_root{
        0xC1396C28719501EEUL, 0x9CF0497512F58995UL, 0x6E64479EAC3434E9UL, 0x7AE96A2B657C0710UL
    };
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000003UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    static constexpr uint256_t cube_root_mont = to_montgomery_uint256(canonical_cube_root, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t cube_root_0 = cube_root_mont.data[0];
    static constexpr uint64_t cube_root_1 = cube_root_mont.data[1];
    static constexpr uint64_t cube_root_2 = cube_root_mont.data[2];
    static constexpr uint64_t cube_root_3 = cube_root_mont.data[3];

    static constexpr uint64_t primitive_root_0 = 0UL;
    static constexpr uint64_t primitive_root_1 = 0UL;
    static constexpr uint64_t primitive_root_2 = 0UL;
    static constexpr uint64_t primitive_root_3 = 0UL;

    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t coset_generator_0 = coset_generator_mont.data[0];
    static constexpr uint64_t coset_generator_1 = coset_generator_mont.data[1];
    static constexpr uint64_t coset_generator_2 = coset_generator_mont.data[2];
    static constexpr uint64_t coset_generator_3 = coset_generator_mont.data[3];

    static constexpr uint64_t modulus_wasm_0 = 0x1ffffc2f;
    static constexpr uint64_t modulus_wasm_1 = 0x1ffffff7;
    static constexpr uint64_t modulus_wasm_2 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_3 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_4 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_5 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_6 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_7 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_8 = 0xffffff;

    static constexpr uint64_t r_inv_wasm_0 = 0xed6544e;
    static constexpr uint64_t r_inv_wasm_1 = 0x1ffffffb;
    static constexpr uint64_t r_inv_wasm_2 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_3 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_4 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_5 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_6 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_7 = 0x10ffffff;
    static constexpr uint64_t r_inv_wasm_8 = 0x9129a9;

    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
    static constexpr char schema_name[] = "secp256k1_fq";
};
using fq = field<FqParams>;

/**
 * @brief Parameters defining the scalar field of the secp256k1 curve.
 */
struct FrParams {
    static constexpr uint64_t modulus_0 = 0xBFD25E8CD0364141ULL;
    static constexpr uint64_t modulus_1 = 0xBAAEDCE6AF48A03BULL;
    static constexpr uint64_t modulus_2 = 0xFFFFFFFFFFFFFFFEULL;
    static constexpr uint64_t modulus_3 = 0xFFFFFFFFFFFFFFFFULL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    static constexpr unsigned R_EXPONENT = 256;
#else
    static constexpr unsigned R_EXPONENT = 261;
#endif

    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, R_EXPONENT);
    static constexpr uint64_t r_squared_0 = r_squared_uint256.data[0];
    static constexpr uint64_t r_squared_1 = r_squared_uint256.data[1];
    static constexpr uint64_t r_squared_2 = r_squared_uint256.data[2];
    static constexpr uint64_t r_squared_3 = r_squared_uint256.data[3];

    static constexpr uint64_t r_inv = 5408259542528602431ULL;

    static constexpr uint64_t r_inv_0 = 0x9d4ad302583de6dcUL;
    static constexpr uint64_t r_inv_1 = 0xa09f710af0155525UL;
    static constexpr uint64_t r_inv_2 = 0xffffffffffffffffUL;
    static constexpr uint64_t r_inv_3 = 0x4b0dff665588b13eUL;

    static constexpr uint256_t canonical_cube_root{
        0xDF02967C1B23BD72UL, 0x122E22EA20816678UL, 0xA5261C028812645AUL, 0x5363AD4CC05C30E0UL
    };
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000005UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    static constexpr uint256_t cube_root_mont = to_montgomery_uint256(canonical_cube_root, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t cube_root_0 = cube_root_mont.data[0];
    static constexpr uint64_t cube_root_1 = cube_root_mont.data[1];
    static constexpr uint64_t cube_root_2 = cube_root_mont.data[2];
    static constexpr uint64_t cube_root_3 = cube_root_mont.data[3];

    static constexpr uint64_t primitive_root_0 = 0UL;
    static constexpr uint64_t primitive_root_1 = 0UL;
    static constexpr uint64_t primitive_root_2 = 0UL;
    static constexpr uint64_t primitive_root_3 = 0UL;

    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t coset_generator_0 = coset_generator_mont.data[0];
    static constexpr uint64_t coset_generator_1 = coset_generator_mont.data[1];
    static constexpr uint64_t coset_generator_2 = coset_generator_mont.data[2];
    static constexpr uint64_t coset_generator_3 = coset_generator_mont.data[3];

    static constexpr uint64_t endo_minus_b1_lo = 0x6F547FA90ABFE4C3ULL;
    static constexpr uint64_t endo_minus_b1_mid = 0xE4437ED6010E8828ULL;

    static constexpr uint64_t endo_b2_lo = 0xe86c90e49284eb15ULL;
    static constexpr uint64_t endo_b2_mid = 0x3086d221a7d46bcdULL;

    // 256-bit-shift constants: g1 = floor((-b1) * 2^256 / r), g2 = floor(b2 * 2^256 / r)
    // See endomorphism_scalars.py compute_splitting_constants() for derivation.
    static constexpr uint64_t endo_g1_lo = 0x6F547FA90ABFE4C4ULL;
    static constexpr uint64_t endo_g1_mid = 0xE4437ED6010E8828ULL;
    static constexpr uint64_t endo_g1_hi = 0x0ULL;

    static constexpr uint64_t endo_g2_lo = 0xE86C90E49284EB15ULL;
    static constexpr uint64_t endo_g2_mid = 0x3086D221A7D46BCDULL;

    static constexpr uint64_t modulus_wasm_0 = 0x10364141;
    static constexpr uint64_t modulus_wasm_1 = 0x1e92f466;
    static constexpr uint64_t modulus_wasm_2 = 0x12280eef;
    static constexpr uint64_t modulus_wasm_3 = 0x1db9cd5e;
    static constexpr uint64_t modulus_wasm_4 = 0x1fffebaa;
    static constexpr uint64_t modulus_wasm_5 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_6 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_7 = 0x1fffffff;
    static constexpr uint64_t modulus_wasm_8 = 0xffffff;

    static constexpr uint64_t r_inv_wasm_0 = 0x3d864e;
    static constexpr uint64_t r_inv_wasm_1 = 0x8b9f61c;
    static constexpr uint64_t r_inv_wasm_2 = 0x3df60c0;
    static constexpr uint64_t r_inv_wasm_3 = 0xa3c71eb;
    static constexpr uint64_t r_inv_wasm_4 = 0x1ffff251;
    static constexpr uint64_t r_inv_wasm_5 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_6 = 0x1fffffff;
    static constexpr uint64_t r_inv_wasm_7 = 0x1effffff;
    static constexpr uint64_t r_inv_wasm_8 = 0xac4589;

    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
    static constexpr char schema_name[] = "secp256k1_fr";
};
using fr = field<FrParams>;

struct G1Params {
    static constexpr bool USE_ENDOMORPHISM = false;
    static constexpr bool can_hash_to_curve = true;
    static constexpr bool has_a = false;

    static constexpr fq b = fq(7);
    static constexpr fq a = fq(0);

    static constexpr fq one_x =
        fq(0x59F2815B16F81798UL, 0x029BFCDB2DCE28D9UL, 0x55A06295CE870B07UL, 0x79BE667EF9DCBBACUL).to_montgomery_form();
    static constexpr fq one_y =
        fq(0x9C47D08FFB10D4B8UL, 0xFD17B448A6855419UL, 0x5DA4FBFC0E1108A8UL, 0x483ADA7726A3C465UL).to_montgomery_form();
};
using g1 = group<fq, fr, G1Params>;

// specialize the name in msgpack schema generation
// consumed by the typescript schema compiler, helps disambiguate templates
inline std::string msgpack_schema_name(g1::affine_element const& /*unused*/)
{
    return "Secp256k1Point";
}

} // namespace bb::secp256k1

namespace bb::curve {
class SECP256K1 {
  public:
    using ScalarField = secp256k1::fr;
    using BaseField = secp256k1::fq;
    using Group = secp256k1::g1;
    using Element = typename Group::element;
    using AffineElement = typename Group::affine_element;
};
} // namespace bb::curve

// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
