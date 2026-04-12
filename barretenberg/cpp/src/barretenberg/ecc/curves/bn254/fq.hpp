// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include <cstdint>
#include <iomanip>

#include "../../fields/field.hpp"
#include "../../fields/field_constexpr_helpers.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/stdlib/primitives/bigfield/constants.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
namespace bb {

/**
 * @brief Parameters defining the base field of the BN254 curve.
 *
 * @details When split into 4 64-bit words, the parameters are represented in little-endian, i.e. the least significant
 * bit comes first. For example, to recover the modulus from the 64-bit words we concatenate its limbs to obtain:
 *           0x30644e72e131a029b85045b68181585d97816a916871ca8d3C208C16D87CFD47
 *
 * @note These parameters can be extracted by running the script parameter_helper.py in ecc/fields
 */
class Bn254FqParams {
  public:
    // Little-endian 256-bit modulus: 0x30644e72e131a029b85045b68181585d97816a916871ca8d3C208C16D87CFD47
    static constexpr uint256_t modulus_uint256{
        0x3C208C16D87CFD47UL, 0x97816a916871ca8dUL, 0xb85045b68181585dUL, 0x30644e72e131a029UL
    };

    // R^2 mod p, where R = 2^R_EXPONENT. Used to convert elements into Montgomery form.
    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, bb::R_EXPONENT);

    // -(p^{-1}) mod 2^64. Used in Montgomery reduction: for each of the lowest four limbs of an
    // 8-limb product, we compute k_i = r_inv * limb_i and add k_i * p to zero out that limb,
    // then divide by 2^256 by taking the upper four limbs. See field_docs.hpp for details.
    static constexpr uint64_t r_inv = compute_r_inv(modulus_uint256.data[0]);

    // 2^{-64} mod p. Used in the Yuval/Barrett-Montgomery reduction variant: instead of computing k,
    // we multiply the lowest limb by this value and add to the following limbs.
    static constexpr uint256_t r_inv_uint256 = compute_div_r_inv(modulus_uint256, 64);

    // Canonical (non-Montgomery) cube root of unity in Fq.
    // Used for the GLV endomorphism: lambda * [P] = (beta * x, y) where beta = cube_root.
    static constexpr uint256_t canonical_cube_root{
        0x5763473177FFFFFEUL, 0xD4F263F1ACDB5C4FUL, 0x59E26BCEA0D48BACUL, 0x0000000000000000UL
    };

    // Not used for Fq (Fq is not used for FFT), but required by the field template interface
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };

    // Canonical (non-Montgomery) coset generator (= 3). Must be a quadratic non-residue mod p.
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000003UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
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
    static constexpr uint64_t endo_g1_lo = 0x7a7bd9d4391eb18d;
    static constexpr uint64_t endo_g1_mid = 0x4ccef014a773d2cfUL;
    static constexpr uint64_t endo_g1_hi = 0x0000000000000002UL;
    static constexpr uint64_t endo_g2_lo = 0xd91d232ec7e0b3d2UL;
    static constexpr uint64_t endo_g2_mid = 0x0000000000000002UL;
    static constexpr uint64_t endo_minus_b1_lo = 0x8211bbeb7d4f1129UL;
    static constexpr uint64_t endo_minus_b1_mid = 0x6f4d8248eeb859fcUL;
    static constexpr uint64_t endo_b2_lo = 0x89d3256894d213e2UL;
    static constexpr uint64_t endo_b2_mid = 0UL;

    // used in msgpack schema serialization
    static constexpr char schema_name[] = "fq";
    static constexpr bool has_high_2adicity = false;

    // The modulus is larger than BN254 scalar field modulus, so it maps to two BN254 scalars
    static constexpr size_t NUM_BN254_SCALARS = 2;
    static constexpr size_t MAX_BITS_PER_ENDOMORPHISM_SCALAR = 128;

    // A point in Fq is represented using 2 field elements in the public inputs (matching Codec)
    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
};

using fq = field<Bn254FqParams>;

} // namespace bb

// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
