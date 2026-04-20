// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../groups/group.hpp"
#include "./fq2.hpp"
#include "./fr.hpp"

namespace bb {
struct Bn254G2Params {
    static constexpr bool USE_ENDOMORPHISM = false;
    static constexpr bool can_hash_to_curve = false;
    static constexpr bool has_a = false;

    static constexpr fq2 one_x{
        fq(uint256_t{ 0x46DEBD5CD992F6EDUL, 0x674322D4F75EDADDUL, 0x426A00665E5C4479UL, 0x1800DEEF121F1E76UL }),
        fq(uint256_t{ 0x97E485B7AEF312C2UL, 0xF1AA493335A9E712UL, 0x7260BFB731FB5D25UL, 0x198E9393920D483AUL })
    };
    static constexpr fq2 one_y{
        fq(uint256_t{ 0x4CE6CC0166FA7DAAUL, 0xE3D1E7690C43D37BUL, 0x4AAB71808DCB408FUL, 0x12C85EA5DB8C6DEBUL }),
        fq(uint256_t{ 0x55ACDADCD122975BUL, 0xBC4B313370B38EF3UL, 0xEC9E99AD690C3395UL, 0x090689D0585FF075UL })
    };
    static constexpr fq2 a = fq2::zero();
    static constexpr fq2 b = fq2::twist_coeff_b();
};

using g2 = group<fq2, fr, Bn254G2Params>;

// specialize the name in msgpack schema generation
// consumed by the typescript schema compiler, helps disambiguate templates
inline std::string msgpack_schema_name(g2::affine_element const& /*unused*/)
{
    return "Bn254G2Point";
}

} // namespace bb
