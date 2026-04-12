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

    // R exponent: defines the Montgomery domain. Native uses R=2^256, WASM uses R=2^261 (=29*9 bits).
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
    static constexpr unsigned R_EXPONENT = 256;
#else
    static constexpr unsigned R_EXPONENT = 261;
#endif

    // R^2 mod p, auto-derived from modulus and R_EXPONENT
    static constexpr uint256_t r_squared_uint256 = compute_r_squared(modulus_uint256, R_EXPONENT);
    static constexpr uint64_t r_squared_0 = r_squared_uint256.data[0];
    static constexpr uint64_t r_squared_1 = r_squared_uint256.data[1];
    static constexpr uint64_t r_squared_2 = r_squared_uint256.data[2];
    static constexpr uint64_t r_squared_3 = r_squared_uint256.data[3];

    // -(Modulus^-1) mod 2^64
    static constexpr uint64_t r_inv = 0xc2e1f593efffffffUL;

    // 2^(-64) mod Modulus — used in native WASM-like reduction path
    static constexpr uint64_t r_inv_0 = 0x2d3e8053e396ee4dUL;
    static constexpr uint64_t r_inv_1 = 0xca478dbeab3c92cdUL;
    static constexpr uint64_t r_inv_2 = 0xb2d8f06f77f52a93UL;
    static constexpr uint64_t r_inv_3 = 0x24d6ba07f7aa8f04UL;

    // Canonical (non-Montgomery) cube root of unity in Fr
    static constexpr uint256_t canonical_cube_root{
        0x8B17EA66B99C90DDUL, 0x5BFC41088D8DAAA7UL, 0xB3C4D79D41A91758UL, 0x0000000000000000UL
    };

    // Canonical (non-Montgomery) primitive root of unity (order 2^28 subgroup) in Fr
    static constexpr uint256_t canonical_primitive_root{
        0x9BD61B6E725B19F0UL, 0x402D111E41112ED4UL, 0x00E0A7EB8EF62ABCUL, 0x2A3C09F0A58A7E85UL
    };

    // Canonical (non-Montgomery) coset generator (= 5)
    static constexpr uint256_t canonical_coset_generator{
        0x0000000000000005UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
    };

    // Montgomery-form constants auto-derived from canonicals + R_EXPONENT
    static constexpr uint256_t cube_root_mont = to_montgomery_uint256(canonical_cube_root, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t cube_root_0 = cube_root_mont.data[0];
    static constexpr uint64_t cube_root_1 = cube_root_mont.data[1];
    static constexpr uint64_t cube_root_2 = cube_root_mont.data[2];
    static constexpr uint64_t cube_root_3 = cube_root_mont.data[3];

    static constexpr uint256_t primitive_root_mont =
        to_montgomery_uint256(canonical_primitive_root, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t primitive_root_0 = primitive_root_mont.data[0];
    static constexpr uint64_t primitive_root_1 = primitive_root_mont.data[1];
    static constexpr uint64_t primitive_root_2 = primitive_root_mont.data[2];
    static constexpr uint64_t primitive_root_3 = primitive_root_mont.data[3];

    static constexpr uint256_t coset_generator_mont =
        to_montgomery_uint256(canonical_coset_generator, modulus_uint256, R_EXPONENT);
    static constexpr uint64_t coset_generator_0 = coset_generator_mont.data[0];
    static constexpr uint64_t coset_generator_1 = coset_generator_mont.data[1];
    static constexpr uint64_t coset_generator_2 = coset_generator_mont.data[2];
    static constexpr uint64_t coset_generator_3 = coset_generator_mont.data[3];

    // A little-endian representation of the modulus split into 9 29-bit limbs (for WASM arithmetic)
    static constexpr uint64_t modulus_wasm_0 = 0x10000001;
    static constexpr uint64_t modulus_wasm_1 = 0x1f0fac9f;
    static constexpr uint64_t modulus_wasm_2 = 0xe5c2450;
    static constexpr uint64_t modulus_wasm_3 = 0x7d090f3;
    static constexpr uint64_t modulus_wasm_4 = 0x1585d283;
    static constexpr uint64_t modulus_wasm_5 = 0x2db40c0;
    static constexpr uint64_t modulus_wasm_6 = 0xa6e141;
    static constexpr uint64_t modulus_wasm_7 = 0xe5c2634;
    static constexpr uint64_t modulus_wasm_8 = 0x30644e;

    // 2^(-29) mod Modulus as 9 29-bit limbs (for WASM reduction)
    static constexpr uint64_t r_inv_wasm_0 = 0x18f05361;
    static constexpr uint64_t r_inv_wasm_1 = 0x12bb1fe;
    static constexpr uint64_t r_inv_wasm_2 = 0xf5d8135;
    static constexpr uint64_t r_inv_wasm_3 = 0x1e6275f6;
    static constexpr uint64_t r_inv_wasm_4 = 0x7e7a880;
    static constexpr uint64_t r_inv_wasm_5 = 0x10c6bf1f;
    static constexpr uint64_t r_inv_wasm_6 = 0x11f74a6c;
    static constexpr uint64_t r_inv_wasm_7 = 0x6fdaecb;
    static constexpr uint64_t r_inv_wasm_8 = 0x183227;

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
