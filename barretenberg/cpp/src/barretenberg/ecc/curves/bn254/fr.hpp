// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include <cstdint>
#include <iomanip>
#include <ostream>

#include "../../fields/field.hpp"
#include "../../fields/field_constexpr_helpers.hpp"
#include "barretenberg/honk/types/public_inputs_type.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)

namespace bb {

/**
 * @brief Parameters defining the scalar field of the BN254 curve.
 *
 * @details When split into 4 64-bit words, the parameters are represented in little-endian, i.e. the least significant
 * bit comes first. For example, to recover the modulus from the 64-bit words we concatenate its limbs to obtain:
 *           0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
 *
 * @note These parameters can be extracted by running the script parameter_helper.py in ecc/fields
 */
class Bn254FrParams {
  public:
    // A little-endian representation of the modulus split into 4 64-bit words
    static constexpr uint64_t modulus_0 = 0x43E1F593F0000001UL;
    static constexpr uint64_t modulus_1 = 0x2833E84879B97091UL;
    static constexpr uint64_t modulus_2 = 0xB85045B68181585DUL;
    static constexpr uint64_t modulus_3 = 0x30644E72E131A029UL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

    // R^2 mod p, where R = 2^R_EXPONENT. Used to convert elements into Montgomery form.
    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, bb::R_EXPONENT);

    // -(p^{-1}) mod 2^64. See field_docs.hpp for Montgomery reduction details.
    static constexpr uint64_t r_inv = compute_r_inv(modulus_0);

    // 2^{-64} mod p. Used in the Yuval/Barrett-Montgomery reduction variant.
    static constexpr uint256_t r_inv_uint256 = compute_div_r_inv(modulus_uint256, 64);
    static constexpr uint64_t r_inv_0 = r_inv_uint256.data[0];
    static constexpr uint64_t r_inv_1 = r_inv_uint256.data[1];
    static constexpr uint64_t r_inv_2 = r_inv_uint256.data[2];
    static constexpr uint64_t r_inv_3 = r_inv_uint256.data[3];

    // Canonical (non-Montgomery) cube root of unity in Fr.
    // Used for the GLV endomorphism: k * P decomposed via lambda (cube root in the scalar field).
    static constexpr uint256_t canonical_cube_root{
        0x8B17EA66B99C90DDUL, 0x5BFC41088D8DAAA7UL, 0xB3C4D79D41A91758UL, 0x0000000000000000UL
    };

    // Canonical (non-Montgomery) primitive root of unity (order 2^28 subgroup) in Fr.
    // Fr has high 2-adicity (2^28 | p-1), enabling FFT-based polynomial arithmetic.
    static constexpr uint256_t canonical_primitive_root{
        0x9BD61B6E725B19F0UL, 0x402D111E41112ED4UL, 0x00E0A7EB8EF62ABCUL, 0x2A3C09F0A58A7E85UL
    };

    // Canonical (non-Montgomery) coset generator (= 5). Must be a quadratic non-residue mod p.
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000005UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    // Montgomery-form representations, derived from canonical values
    static constexpr uint256_t cube_root_mont =
        to_montgomery_uint256(canonical_cube_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t primitive_root_mont =
        to_montgomery_uint256(canonical_primitive_root, modulus_uint256, bb::R_EXPONENT);
    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, bb::R_EXPONENT);

    // Parameters used for quickly splitting a scalar into two endomorphism scalars for faster scalar multiplication
    // For specifics on how these have been derived, see ecc/fields/endomorphim_scalars.py
    static constexpr uint64_t endo_g1_lo = 0x7a7bd9d4391eb18dUL;
    static constexpr uint64_t endo_g1_mid = 0x4ccef014a773d2cfUL;
    static constexpr uint64_t endo_g1_hi = 0x0000000000000002UL;
    static constexpr uint64_t endo_g2_lo = 0xd91d232ec7e0b3d7UL;
    static constexpr uint64_t endo_g2_mid = 0x0000000000000002UL;
    static constexpr uint64_t endo_minus_b1_lo = 0x8211bbeb7d4f1128UL;
    static constexpr uint64_t endo_minus_b1_mid = 0x6f4d8248eeb859fcUL;
    static constexpr uint64_t endo_b2_lo = 0x89d3256894d213e3UL;
    static constexpr uint64_t endo_b2_mid = 0UL;

    // used in msgpack schema serialization
    static constexpr char schema_name[] = "fr";
    static constexpr bool has_high_2adicity = true;

    // This is a BN254 scalar, so it represents one BN254 scalar
    static constexpr size_t NUM_BN254_SCALARS = 1;
    static constexpr size_t MAX_BITS_PER_ENDOMORPHISM_SCALAR = 128;

    // A point in Fr is represented with 1 public input
    static constexpr size_t PUBLIC_INPUTS_SIZE = FR_PUBLIC_INPUTS_SIZE;
};

using fr = field<Bn254FrParams>;

template <> template <> inline fr fr::reconstruct_from_public(const std::span<const fr, PUBLIC_INPUTS_SIZE>& limbs)
{
    return fr(limbs[0]);
}

} // namespace bb

// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
