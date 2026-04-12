// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Federico], commit: 158dd845c99f8f702979c20f1625730d126c4b20}
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

#include "../../fields/field6.hpp"
#include "./fq.hpp"
#include "./fq2.hpp"

namespace bb {

/**
 * @brief Sextic extension of the base field of BN254
 *
 * @details Fq6 is defined as Fq2[v] / (v^3 - \xi), where \xi = 9 + u is not a cubic residue in Fq2. We store in the
 * struct the coefficients to compute the frobenius morphism (we need powers up to q^3 to compute the final
 * exponentiation in the pairing calculation)
 * 1. Power q
 * \f[
 *  (a + bv + cv^2)^q = a^q + b^q * v^q + c^q * v^{2q} = a^q + b^q * \xi^{(q-1)/3} * v + c^q * \xi^{2(q-1)/3} * v^2
 * \f]
 * 2. Power q^2
 * \f[
 *  (a + bv + cv^2)^{q^2} = a^{q^2} + b^{q^2} * v^{q^2} + c^{q^2} * v^{2q^2} =
 *                                  a + b * \xi^{(q^2-1)/3} * v + c * \xi^{2(q^2-1)/3} * v^2
 * \f]
 * 3. Power q^3
 * \f[
 *  (a + bv + cv^2)^{q^3} = a^{q^3} + b^{q^3} * v^{q^3} + c^{q^3} * v^{2q^3} =
 *                                  a^q + b^q * \xi^{(q^3-1)/3} * v + c^q * \xi^{2(q^3-1)/3} * v^2
 * \f]
 *
 * Constants are stored in canonical (non-Montgomery) form and converted to Montgomery form via the fq(uint256_t)
 * constructor.
 */
struct Bn254Fq6Params {

    static constexpr fq2 frobenius_coeffs_c1_1{
        fq(uint256_t{ 0x99E39557176F553DUL, 0xB78CC310C2C3330CUL, 0x4C0BEC3CF559B143UL, 0x2FB347984F7911F7UL }),
        fq(uint256_t{ 0x1665D51C640FCBA2UL, 0x32AE2A1D0B7C9DCEUL, 0x4BA4CC8BD75A0794UL, 0x16C9E55061EBAE20UL })
    };

    static constexpr fq2 frobenius_coeffs_c1_2{
        fq(uint256_t{ 0xE4BD44E5607CFD48UL, 0xC28F069FBB966E3DUL, 0x5E6DD9E7E0ACCCB0UL, 0x30644E72E131A029UL }), fq(0)
    };

    static constexpr fq2 frobenius_coeffs_c1_3{
        fq(uint256_t{ 0x7B746EE87BDCFB6DUL, 0x805FFD3D5D6942D3UL, 0xBAFF1C77959F25ACUL, 0x0856E078B755EF0AUL }),
        fq(uint256_t{ 0x380CAB2BAAA586DEUL, 0x0FDF31BF98FF2631UL, 0xA9F30E6DEC26094FUL, 0x04F1DE41B3D1766FUL })
    };

    static constexpr fq2 frobenius_coeffs_c2_1{
        fq(uint256_t{ 0x848A1F55921EA762UL, 0xD33365F7BE94EC72UL, 0x80F3C0B75A181E84UL, 0x05B54F5E64EEA801UL }),
        fq(uint256_t{ 0xC13B4711CD2B8126UL, 0x3685D2EA1BDEC763UL, 0x9F3A80B03B0B1C92UL, 0x2C145EDBE7FD8AEEUL })
    };

    static constexpr fq2 frobenius_coeffs_c2_2{
        fq(uint256_t{ 0x5763473177FFFFFEUL, 0xD4F263F1ACDB5C4FUL, 0x59E26BCEA0D48BACUL, 0x0000000000000000UL }), fq(0)
    };

    static constexpr fq2 frobenius_coeffs_c2_3{
        fq(uint256_t{ 0x0E1A92BC3CCBF066UL, 0xE633094575B06BCBUL, 0x19BEE0F7B5B2444EUL, 0x0BC58C6611C08DABUL }),
        fq(uint256_t{ 0x5FE3ED9D730C239FUL, 0xA44A9E08737F96E5UL, 0xFEB0F6EF0CD21D04UL, 0x23D5E999E1910A12UL })
    };

    static inline constexpr fq2 mul_by_non_residue(const fq2& a)
    {
        // non_residue = 9 + u
        // (a + bu) * (9 + u) = (9a - b) + (9b + a)u

        // 9a
        fq T0 = a.c0 + a.c0;
        T0 += T0;
        T0 += T0;
        T0 += a.c0;

        // 9b
        fq T1 = a.c1 + a.c1;
        T1 += T1;
        T1 += T1;
        T1 += a.c1;

        return { T0 - a.c1, T1 + a.c0 };
    }
};

using fq6 = field6<fq2, Bn254Fq6Params>;
} // namespace bb
