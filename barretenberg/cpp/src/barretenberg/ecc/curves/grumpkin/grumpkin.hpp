// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../groups/group.hpp"
#include "../bn254/fq.hpp"
#include "../bn254/fr.hpp"

namespace bb::grumpkin {

// Max num bits such that all numbers represented by that many bits are smaller than fr::modulus
constexpr size_t MAX_NO_WRAP_INTEGER_BIT_LENGTH = 252;
static_assert((uint256_t(1) << (MAX_NO_WRAP_INTEGER_BIT_LENGTH + 1)) - 1 < fr::modulus,
              "MAX_NO_WRAP_INTEGER_BIT_LENGTH is too large");

using fq = bb::fr;
using fr = bb::fq;

struct G1Params {
    static constexpr bool USE_ENDOMORPHISM = true;
    static constexpr bool can_hash_to_curve = true;
    static constexpr bool has_a = false;

    // Grumpkin: y^2 = x^3 - 17. Canonical b = p - 17 (for BN254 Fr modulus p).
    // Canonical one_y: sqrt(-16) on Grumpkin. Auto-converted to Montgomery form via fr constructor.
    static constexpr bb::fr b =
        bb::fr(uint256_t{ 0x43E1F593EFFFFFF0UL, 0x2833E84879B97091UL, 0xB85045B68181585DUL, 0x30644E72E131A029UL });
    static constexpr bb::fr a{ 0UL, 0UL, 0UL, 0UL };

    // generator point = (x, y) = (1, sqrt(-16))
    static constexpr bb::fr one_x = bb::fr::one();
    static constexpr bb::fr one_y =
        bb::fr(uint256_t{ 0x833FC48D823F272CUL, 0x2D270D45F1181294UL, 0xCF135E7506A45D63UL, 0x0000000000000002UL });
};
using g1 = bb::group<bb::fr, bb::fq, G1Params>;

// specialize the name in msgpack schema generation
// consumed by the typescript schema compiler, helps disambiguate templates
inline std::string msgpack_schema_name(g1::affine_element const& /*unused*/)
{
    return "GrumpkinPoint";
}

}; // namespace bb::grumpkin

namespace bb::curve {
class Grumpkin {
  public:
    using ScalarField = bb::fq;
    using BaseField = bb::fr;
    using Group = typename grumpkin::g1;
    using Element = typename Group::element;
    using AffineElement = typename Group::affine_element;

    static constexpr const char* name = "Grumpkin";
    static constexpr bool is_stdlib_type = false;

    static constexpr size_t SUBGROUP_SIZE = 87;
    static constexpr ScalarField subgroup_generator =
        ScalarField(uint256_t("0x147c647c09fb639514909e9f0513f31ec1a523bf8a0880bc7c24fbc962a9586b"));
    static constexpr ScalarField subgroup_generator_inverse =
        ScalarField("0x0c68e27477b5e78cfab790bd3b59806fa871771f71ec7452cde5384f6e3a1988");
    static constexpr uint32_t LIBRA_UNIVARIATES_LENGTH = 3;
};
} // namespace bb::curve
