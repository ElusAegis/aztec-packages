// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../fields/field.hpp"
#include "../../fields/field_constexpr_helpers.hpp"
#include "../../groups/group.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
namespace bb::secp256r1 {

/**
 * @brief Parameters defining the base field of the secp256r1 curve.
 */
struct FqParams {
    static constexpr const char* schema_name = "secp256r1_fq";

    // p = 2^256 - 2^224 + 2^192 + 2^96 - 1
    static constexpr uint64_t modulus_0 = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint64_t modulus_1 = 0x00000000FFFFFFFFULL;
    static constexpr uint64_t modulus_2 = 0x0000000000000000ULL;
    static constexpr uint64_t modulus_3 = 0xFFFFFFFF00000001ULL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, bb::R_EXPONENT);

    static constexpr uint64_t r_inv = compute_r_inv(modulus_0);

    static constexpr uint256_t r_inv_uint256 = compute_div_r_inv(modulus_uint256, 64);
    static constexpr uint64_t r_inv_0 = r_inv_uint256.data[0];
    static constexpr uint64_t r_inv_1 = r_inv_uint256.data[1];
    static constexpr uint64_t r_inv_2 = r_inv_uint256.data[2];
    static constexpr uint64_t r_inv_3 = r_inv_uint256.data[3];

    // No cube root of unity exists for secp256r1 Fq (p mod 3 != 1)
    static constexpr uint256_t canonical_cube_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000003UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    static constexpr uint256_t cube_root_mont =
        to_montgomery_uint256(canonical_cube_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t primitive_root_mont =
        to_montgomery_uint256(canonical_primitive_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, bb::R_EXPONENT);

    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
};
using fq = field<FqParams>;

/**
 * @brief Parameters defining the scalar field of the secp256r1 curve.
 */
struct FrParams {
    static constexpr const char* schema_name = "secp256r1_fr";

    static constexpr uint64_t modulus_0 = 0xF3B9CAC2FC632551ULL;
    static constexpr uint64_t modulus_1 = 0xBCE6FAADA7179E84ULL;
    static constexpr uint64_t modulus_2 = 0xFFFFFFFFFFFFFFFFULL;
    static constexpr uint64_t modulus_3 = 0xFFFFFFFF00000000ULL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, bb::R_EXPONENT);

    static constexpr uint64_t r_inv = compute_r_inv(modulus_0);

    static constexpr uint256_t r_inv_uint256 = compute_div_r_inv(modulus_uint256, 64);
    static constexpr uint64_t r_inv_0 = r_inv_uint256.data[0];
    static constexpr uint64_t r_inv_1 = r_inv_uint256.data[1];
    static constexpr uint64_t r_inv_2 = r_inv_uint256.data[2];
    static constexpr uint64_t r_inv_3 = r_inv_uint256.data[3];

    static constexpr uint256_t canonical_cube_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000007UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    static constexpr uint256_t cube_root_mont =
        to_montgomery_uint256(canonical_cube_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t primitive_root_mont =
        to_montgomery_uint256(canonical_primitive_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, bb::R_EXPONENT);

    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
};
using fr = field<FrParams>;

struct G1Params {
    static constexpr bool USE_ENDOMORPHISM = false;
    static constexpr bool can_hash_to_curve = true;
    static constexpr bool has_a = true;

    static constexpr fq b =
        fq(0x3BCE3C3E27D2604B, 0x651D06B0CC53B0F6, 0xB3EBBD55769886BC, 0x5AC635D8AA3A93E7).to_montgomery_form();
    static constexpr fq a =
        fq(0xFFFFFFFFFFFFFFFC, 0x00000000FFFFFFFF, 0x0000000000000000, 0xFFFFFFFF00000001).to_montgomery_form();

    static constexpr fq one_x =
        fq(0xF4A13945D898C296, 0x77037D812DEB33A0, 0xF8BCE6E563A440F2, 0x6B17D1F2E12C4247).to_montgomery_form();
    static constexpr fq one_y =
        fq(0xCBB6406837BF51F5, 0x2BCE33576B315ECE, 0x8EE7EB4A7C0F9E16, 0x4FE342E2FE1A7F9B).to_montgomery_form();
};
using g1 = group<fq, fr, G1Params>;

// specialize the name in msgpack schema generation
// consumed by the typescript schema compiler, helps disambiguate templates
inline std::string msgpack_schema_name(g1::affine_element const& /*unused*/)
{
    return "Secp256r1Point";
}

} // namespace bb::secp256r1

namespace bb::curve {
class SECP256R1 {
  public:
    using ScalarField = secp256r1::fr;
    using BaseField = secp256r1::fq;
    using Group = secp256r1::g1;
    using Element = typename Group::element;
    using AffineElement = typename Group::affine_element;
};
} // namespace bb::curve

// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
