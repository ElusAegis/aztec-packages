// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../fields/field12.hpp"
#include "./fq2.hpp"
#include "./fq6.hpp"

namespace bb {

/**
 * @brief The twelfth degree extension of the base field of BN254
 *
 * @details Fq12 is defined as Fq6[w] / (w^2 - v). Frobenius coefficients stored in canonical form,
 * auto-converted to Montgomery form via the fq(uint256_t) constructor.
 */
struct Bn254Fq12Params {

    static constexpr fq2 frobenius_coefficients_1{
        fq(uint256_t{ 0xD60B35DADCC9E470UL, 0x5C521E08292F2176UL, 0xE8B99FDD76E68B60UL, 0x1284B71C2865A7DFUL }),
        fq(uint256_t{ 0xCA5CF05F80F362ACUL, 0x747992778EEEC7E5UL, 0xA6327CFE12150B8EUL, 0x246996F3B4FAE7E6UL })
    };

    static constexpr fq2 frobenius_coefficients_2{
        fq(uint256_t{ 0xE4BD44E5607CFD49UL, 0xC28F069FBB966E3DUL, 0x5E6DD9E7E0ACCCB0UL, 0x30644E72E131A029UL }), fq(0)
    };

    static constexpr fq2 frobenius_coefficients_3{
        fq(uint256_t{ 0xE86F7D391ED4A67FUL, 0x894CB38DBE55D24AUL, 0xEFE9608CD0ACAA90UL, 0x19DC81CFCC82E4BBUL }),
        fq(uint256_t{ 0x7694AA2BF4C0C101UL, 0x7F03A5E397D439ECUL, 0x06CBEEE33576139DUL, 0x00ABF8B60BE77D73UL })
    };
};

using fq12 = field12<fq2, fq6, Bn254Fq12Params>;
} // namespace bb
