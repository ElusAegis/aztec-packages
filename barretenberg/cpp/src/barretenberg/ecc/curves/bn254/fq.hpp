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
    // A little-endian representation of the modulus split into 4 64-bit words
    static constexpr uint64_t modulus_0 = 0x3C208C16D87CFD47UL;
    static constexpr uint64_t modulus_1 = 0x97816a916871ca8dUL;
    static constexpr uint64_t modulus_2 = 0xb85045b68181585dUL;
    static constexpr uint64_t modulus_3 = 0x30644e72e131a029UL;

    static constexpr uint256_t modulus_uint256{ modulus_0, modulus_1, modulus_2, modulus_3 };

    // R exponent: defines the Montgomery domain
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    static constexpr unsigned R_EXPONENT = 256;
#else
    static constexpr unsigned R_EXPONENT = 261;
#endif

    // R^2 mod p, auto-derived
    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, R_EXPONENT);
    static constexpr uint64_t r_squared_0 = r_squared_uint256.data[0];
    static constexpr uint64_t r_squared_1 = r_squared_uint256.data[1];
    static constexpr uint64_t r_squared_2 = r_squared_uint256.data[2];
    static constexpr uint64_t r_squared_3 = r_squared_uint256.data[3];

    // -(Modulus^-1) mod 2^64
    static constexpr uint64_t r_inv = 0x87d20782e4866389UL;

    // 2^(-64) mod Modulus
    static constexpr uint64_t r_inv_0 = 0x327d7c1b18f7bd41UL;
    static constexpr uint64_t r_inv_1 = 0xdb8ed52f824ed32fUL;
    static constexpr uint64_t r_inv_2 = 0x29b67b05eb29a6a1UL;
    static constexpr uint64_t r_inv_3 = 0x19ac99126b459ddaUL;

    // Canonical (non-Montgomery) cube root of unity in Fq
    static constexpr uint256_t canonical_cube_root{
        0x5763473177FFFFFEUL, 0xD4F263F1ACDB5C4FUL, 0x59E26BCEA0D48BACUL, 0x0000000000000000UL
    };

    // Not used for Fq (Fq is not used for FFT), but required for all field types
    static constexpr uint256_t canonical_primitive_root{ 0UL, 0UL, 0UL, 0UL };

    // Canonical (non-Montgomery) coset generator (= 3)
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000003UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    // Montgomery-form constants auto-derived
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

    // Modulus split into 9 29-bit limbs (for WASM arithmetic)
    static constexpr uint64_t modulus_wasm_0 = 0x187cfd47;
    static constexpr uint64_t modulus_wasm_1 = 0x10460b6;
    static constexpr uint64_t modulus_wasm_2 = 0x1c72a34f;
    static constexpr uint64_t modulus_wasm_3 = 0x2d522d0;
    static constexpr uint64_t modulus_wasm_4 = 0x1585d978;
    static constexpr uint64_t modulus_wasm_5 = 0x2db40c0;
    static constexpr uint64_t modulus_wasm_6 = 0xa6e141;
    static constexpr uint64_t modulus_wasm_7 = 0xe5c2634;
    static constexpr uint64_t modulus_wasm_8 = 0x30644e;

    // 2^(-29) mod Modulus as 9 29-bit limbs (for WASM reduction)
    static constexpr uint64_t r_inv_wasm_0 = 0x17789a9f;
    static constexpr uint64_t r_inv_wasm_1 = 0x5ffc3dc;
    static constexpr uint64_t r_inv_wasm_2 = 0xd6bde42;
    static constexpr uint64_t r_inv_wasm_3 = 0x1cf152e3;
    static constexpr uint64_t r_inv_wasm_4 = 0x18eb055f;
    static constexpr uint64_t r_inv_wasm_5 = 0xed815e2;
    static constexpr uint64_t r_inv_wasm_6 = 0x16626d2;
    static constexpr uint64_t r_inv_wasm_7 = 0xb8bab0f;
    static constexpr uint64_t r_inv_wasm_8 = 0x6d7c4;

    // Parameters used for quickly splitting a scalar into two endomorphism scalars
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
