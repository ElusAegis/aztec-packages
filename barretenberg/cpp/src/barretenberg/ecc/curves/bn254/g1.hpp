// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../groups/group.hpp"
#include "./fq.hpp"
#include "./fr.hpp"

namespace bb {
struct Bn254G1Params {
    static constexpr bool USE_ENDOMORPHISM = true;
    static constexpr bool can_hash_to_curve = true;
    static constexpr bool has_a = false;

    // Generator = (1, 2), b = 3 — canonical values, auto-converted to Montgomery form
    static constexpr fq one_x = fq::one();
    static constexpr fq one_y = fq(2);
    static constexpr fq a{ 0UL, 0UL, 0UL, 0UL };
    static constexpr fq b = fq(3);
};

using g1 = group<fq, fr, Bn254G1Params>;

// specialize the name in msgpack schema generation
// consumed by the typescript schema compiler, helps disambiguate templates
inline std::string msgpack_schema_name(g1::affine_element const& /*unused*/)
{
    return "Bn254G1Point";
}

} // namespace bb
