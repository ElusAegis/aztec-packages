#ifdef STARKNET_GARAGA_FLAVORS
#pragma once

#include "barretenberg/ecc/fields/field.hpp"
#include "barretenberg/ecc/fields/field_constexpr_helpers.hpp"
#include "barretenberg/honk/types/public_inputs_type.hpp"

namespace bb::starknet::stark252 {

struct FqParams {
    static constexpr uint64_t modulus_0 = 0x0000000000000001ULL;
    static constexpr uint64_t modulus_1 = 0x0000000000000000ULL;
    static constexpr uint64_t modulus_2 = 0x0000000000000000ULL;
    static constexpr uint64_t modulus_3 = 0x0800000000000011ULL;

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

    static constexpr uint64_t r_inv = 0xffffffffffffffffULL;

    static constexpr uint64_t r_inv_0 = 0x0000000000000001ULL;
    static constexpr uint64_t r_inv_1 = 0x0000000000000000ULL;
    static constexpr uint64_t r_inv_2 = 0xf7ffffffffffffefULL;
    static constexpr uint64_t r_inv_3 = 0x0800000000000010ULL;

    static constexpr uint64_t modulus_wasm_0 = 0x00000001;
    static constexpr uint64_t modulus_wasm_1 = 0x00000000;
    static constexpr uint64_t modulus_wasm_2 = 0x00000000;
    static constexpr uint64_t modulus_wasm_3 = 0x00000000;
    static constexpr uint64_t modulus_wasm_4 = 0x00000000;
    static constexpr uint64_t modulus_wasm_5 = 0x00000000;
    static constexpr uint64_t modulus_wasm_6 = 0x00440000;
    static constexpr uint64_t modulus_wasm_7 = 0x00000000;
    static constexpr uint64_t modulus_wasm_8 = 0x00080000;

    static constexpr uint64_t r_inv_wasm_0 = 0x00000001;
    static constexpr uint64_t r_inv_wasm_1 = 0x00000000;
    static constexpr uint64_t r_inv_wasm_2 = 0x00000000;
    static constexpr uint64_t r_inv_wasm_3 = 0x00000000;
    static constexpr uint64_t r_inv_wasm_4 = 0x00000000;
    static constexpr uint64_t r_inv_wasm_5 = 0x1fbc0000;
    static constexpr uint64_t r_inv_wasm_6 = 0x0043ffff;
    static constexpr uint64_t r_inv_wasm_7 = 0x1ff80000;
    static constexpr uint64_t r_inv_wasm_8 = 0x0007ffff;

    // For consistency with bb::fq, if we ever represent an element of bb::stark252::fq in the public inputs, we do so
    // as a bigfield element, so with 4 public inputs
    static constexpr size_t PUBLIC_INPUTS_SIZE = BIGFIELD_PUBLIC_INPUTS_SIZE;
};

using fq = field<FqParams>;

} // namespace bb::starknet::stark252
#endif
