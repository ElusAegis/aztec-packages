// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../fields/field2.hpp"
#include "./fq.hpp"

namespace bb {

/**
 * @brief Quadratic extension of the base field of BN254
 *
 * @details The quadratic extension Fq2 is defined as Fq[u] / (u^2 + 1). Fq2 is the base field of
 * the twist of BN254, thus points in G2 have coordinates in Fq2.
 *
 * Constants are stored in canonical (non-Montgomery) form and auto-converted to Montgomery form
 * via the fq(uint256_t) constructor, which calls to_montgomery_form() using the platform-specific R.
 */
struct Bn254Fq2Params {
    static constexpr fq twist_coeff_b_0 =
        fq(uint256_t{ 0x3267E6DC24A138E5UL, 0xB5B4C5E559DBEFA3UL, 0x81BE18991BE06AC3UL, 0x2B149D40CEB8AAAEUL });
    static constexpr fq twist_coeff_b_1 =
        fq(uint256_t{ 0xE4A2BD0685C315D2UL, 0xA74FA084E52D1852UL, 0xCD2CAFADEED8FDF4UL, 0x009713B03AF0FED4UL });
    static constexpr fq frobenius_on_twisted_curve_x_0 =
        fq(uint256_t{ 0x99E39557176F553DUL, 0xB78CC310C2C3330CUL, 0x4C0BEC3CF559B143UL, 0x2FB347984F7911F7UL });
    static constexpr fq frobenius_on_twisted_curve_x_1 =
        fq(uint256_t{ 0x1665D51C640FCBA2UL, 0x32AE2A1D0B7C9DCEUL, 0x4BA4CC8BD75A0794UL, 0x16C9E55061EBAE20UL });
    static constexpr fq frobenius_on_twisted_curve_y_0 =
        fq(uint256_t{ 0xDC54014671A0135AUL, 0xDBAAE0EDA9C95998UL, 0xDC5EC698B6E2F9B9UL, 0x063CF305489AF5DCUL });
    static constexpr fq frobenius_on_twisted_curve_y_1 =
        fq(uint256_t{ 0x82D37F632623B0E3UL, 0x21807DC98FA25BD2UL, 0x0704B5A7EC796F2BUL, 0x07C03CBCAC41049AUL });
};

using fq2 = field2<fq, Bn254Fq2Params>;
} // namespace bb
