#pragma once

// 24-bit FMA SIMD Montgomery multiplication for WASM.
// Uses 11 × 24-bit limbs with exact f64 arithmetic (24×24 = 48 < 53-bit mantissa).
// R = 2^264 (11 × 24-bit reduction steps). No correction needed.
// Only valid when R_LIMB_BITS == 24. Requires -msimd128 -mrelaxed-simd.
//
// Two entry points:
//   fma_simd  — single mul, SIMD used within (P_lo/P_cross parallel, scatter pairing)
//   fma_simd2 — dual mul, each SIMD lane carries a full independent multiplication

#if BB_R_LIMB_BITS == 24 && defined(__wasm_simd128__)
#include <wasm_simd128.h>
/**
 * @brief Single Montgomery multiplication: this * other mod p.
 *
 * Uses 11 × 24-bit limbs in f64 with Karatsuba 6+5 split.
 * SIMD is used _within_ the single multiplication to extract parallelism:
 *   - Product phase: P_lo and P_cross (both 6×6) computed in parallel via
 *     f64x2 lanes; P_hi (5×5) is scalar (only 5 limbs, no SIMD partner).
 *   - Reduction phase: consecutive Yuval scatter FMAs paired into f64x2.
 * 10 Yuval steps + 1 standard step = division by 2^264 = R. Output in [0, 2p).
 *
 * See montgomery_mul_wasm_fma_simd2 for the dual-multiplication variant that
 * uses both SIMD lanes for two independent multiplications end-to-end.
 */
template <class T>
field<T> field<T>::montgomery_mul_wasm_fma_simd(const field& other) const noexcept
{
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;
    constexpr double SD = 0x1p-24; // 2^{-R_LIMB_BITS}
    constexpr double SU = 0x1p24;  // 2^{R_LIMB_BITS}
    // -(p^{-1}) mod 2^{R_LIMB_BITS}, for the standard reduction step.
    constexpr double NP0_F = static_cast<double>(compute_r_inv(modulus.data[0]) & M24);

    auto fma_s = [](double a, double b, double c) -> double {
        v128_t r = __builtin_wasm_relaxed_madd_f64x2(
            wasm_f64x2_make(a, 0.0),
            wasm_f64x2_make(b, 0.0),
            wasm_f64x2_make(c, 0.0));
        return wasm_f64x2_extract_lane(r, 0);
    };

    // ================================================================
    // Phase 1: Convert 4x64 -> 11x24 -> double
    // ================================================================
    const uint64_t* a = data;
    const uint64_t* b = other.data;

    double af0  = static_cast<double>(a[0] & M24);
    double af1  = static_cast<double>((a[0] >> 24) & M24);
    double af2  = static_cast<double>(((a[0] >> 48) | (a[1] << 16)) & M24);
    double af3  = static_cast<double>((a[1] >> 8) & M24);
    double af4  = static_cast<double>((a[1] >> 32) & M24);
    double af5  = static_cast<double>(((a[1] >> 56) | (a[2] << 8)) & M24);
    double af6  = static_cast<double>((a[2] >> 16) & M24);
    double af7  = static_cast<double>((a[2] >> 40) & M24);
    double af8  = static_cast<double>(a[3] & M24);
    double af9  = static_cast<double>((a[3] >> 24) & M24);
    double af10 = static_cast<double>(a[3] >> 48);

    double bf0  = static_cast<double>(b[0] & M24);
    double bf1  = static_cast<double>((b[0] >> 24) & M24);
    double bf2  = static_cast<double>(((b[0] >> 48) | (b[1] << 16)) & M24);
    double bf3  = static_cast<double>((b[1] >> 8) & M24);
    double bf4  = static_cast<double>((b[1] >> 32) & M24);
    double bf5  = static_cast<double>(((b[1] >> 56) | (b[2] << 8)) & M24);
    double bf6  = static_cast<double>((b[2] >> 16) & M24);
    double bf7  = static_cast<double>((b[2] >> 40) & M24);
    double bf8  = static_cast<double>(b[3] & M24);
    double bf9  = static_cast<double>((b[3] >> 24) & M24);
    double bf10 = static_cast<double>(b[3] >> 48);

    // ================================================================
    // Phase 2: Karatsuba product with SIMD
    // P_lo and P_cross share the same 6x6 structure.
    // lane 0 = P_lo, lane 1 = P_cross.
    // ================================================================
    double sl0 = af0 + af6;   double sr0 = bf0 + bf6;
    double sl1 = af1 + af7;   double sr1 = bf1 + bf7;
    double sl2 = af2 + af8;   double sr2 = bf2 + bf8;
    double sl3 = af3 + af9;   double sr3 = bf3 + bf9;
    double sl4 = af4 + af10;  double sr4 = bf4 + bf10;
    double sl5 = af5;         double sr5 = bf5;

    v128_t als0 = wasm_f64x2_make(af0, sl0);
    v128_t als1 = wasm_f64x2_make(af1, sl1);
    v128_t als2 = wasm_f64x2_make(af2, sl2);
    v128_t als3 = wasm_f64x2_make(af3, sl3);
    v128_t als4 = wasm_f64x2_make(af4, sl4);
    v128_t als5 = wasm_f64x2_make(af5, sl5);

    v128_t bsr0 = wasm_f64x2_make(bf0, sr0);
    v128_t bsr1 = wasm_f64x2_make(bf1, sr1);
    v128_t bsr2 = wasm_f64x2_make(bf2, sr2);
    v128_t bsr3 = wasm_f64x2_make(bf3, sr3);
    v128_t bsr4 = wasm_f64x2_make(bf4, sr4);
    v128_t bsr5 = wasm_f64x2_make(bf5, sr5);

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    v128_t plpc0 = wasm_f64x2_mul(als0, bsr0);
    v128_t plpc1 = fma_v(als1, bsr0, wasm_f64x2_mul(als0, bsr1));
    v128_t plpc2 = fma_v(als2, bsr0, fma_v(als1, bsr1, wasm_f64x2_mul(als0, bsr2)));
    v128_t plpc3 = fma_v(als3, bsr0, fma_v(als2, bsr1, fma_v(als1, bsr2,
                   wasm_f64x2_mul(als0, bsr3))));
    v128_t plpc4 = fma_v(als4, bsr0, fma_v(als3, bsr1, fma_v(als2, bsr2,
                   fma_v(als1, bsr3, wasm_f64x2_mul(als0, bsr4)))));
    v128_t plpc5 = fma_v(als5, bsr0, fma_v(als4, bsr1, fma_v(als3, bsr2,
                   fma_v(als2, bsr3, fma_v(als1, bsr4, wasm_f64x2_mul(als0, bsr5))))));
    v128_t plpc6 = fma_v(als5, bsr1, fma_v(als4, bsr2, fma_v(als3, bsr3,
                   fma_v(als2, bsr4, wasm_f64x2_mul(als1, bsr5)))));
    v128_t plpc7 = fma_v(als5, bsr2, fma_v(als4, bsr3, fma_v(als3, bsr4,
                   wasm_f64x2_mul(als2, bsr5))));
    v128_t plpc8 = fma_v(als5, bsr3, fma_v(als4, bsr4, wasm_f64x2_mul(als3, bsr5)));
    v128_t plpc9 = fma_v(als5, bsr4, wasm_f64x2_mul(als4, bsr5));
    v128_t plpc10 = wasm_f64x2_mul(als5, bsr5);

    double pl0  = wasm_f64x2_extract_lane(plpc0, 0);
    double pl1  = wasm_f64x2_extract_lane(plpc1, 0);
    double pl2  = wasm_f64x2_extract_lane(plpc2, 0);
    double pl3  = wasm_f64x2_extract_lane(plpc3, 0);
    double pl4  = wasm_f64x2_extract_lane(plpc4, 0);
    double pl5  = wasm_f64x2_extract_lane(plpc5, 0);
    double pl6  = wasm_f64x2_extract_lane(plpc6, 0);
    double pl7  = wasm_f64x2_extract_lane(plpc7, 0);
    double pl8  = wasm_f64x2_extract_lane(plpc8, 0);
    double pl9  = wasm_f64x2_extract_lane(plpc9, 0);
    double pl10 = wasm_f64x2_extract_lane(plpc10, 0);

    double pc0  = wasm_f64x2_extract_lane(plpc0, 1);
    double pc1  = wasm_f64x2_extract_lane(plpc1, 1);
    double pc2  = wasm_f64x2_extract_lane(plpc2, 1);
    double pc3  = wasm_f64x2_extract_lane(plpc3, 1);
    double pc4  = wasm_f64x2_extract_lane(plpc4, 1);
    double pc5  = wasm_f64x2_extract_lane(plpc5, 1);
    double pc6  = wasm_f64x2_extract_lane(plpc6, 1);
    double pc7  = wasm_f64x2_extract_lane(plpc7, 1);
    double pc8  = wasm_f64x2_extract_lane(plpc8, 1);
    double pc9  = wasm_f64x2_extract_lane(plpc9, 1);
    double pc10 = wasm_f64x2_extract_lane(plpc10, 1);

    // P_hi = left[6..10] x right[6..10] -- 5x5 scalar
    double ph0 = af6 * bf6;
    double ph1 = fma_s(af7, bf6, af6 * bf7);
    double ph2 = fma_s(af8, bf6, fma_s(af7, bf7, af6 * bf8));
    double ph3 = fma_s(af9, bf6, fma_s(af8, bf7, fma_s(af7, bf8, af6 * bf9)));
    double ph4 = fma_s(af10, bf6, fma_s(af9, bf7, fma_s(af8, bf8,
                 fma_s(af7, bf9, af6 * bf10))));
    double ph5 = fma_s(af10, bf7, fma_s(af9, bf8, fma_s(af8, bf9, af7 * bf10)));
    double ph6 = fma_s(af10, bf8, fma_s(af9, bf9, af8 * bf10));
    double ph7 = fma_s(af10, bf9, af9 * bf10);
    double ph8 = af10 * bf10;

    // Combine
    double t0  = pl0;
    double t1  = pl1;
    double t2  = pl2;
    double t3  = pl3;
    double t4  = pl4;
    double t5  = pl5;
    double t6  = pl6  + (pc0 - pl0 - ph0);
    double t7  = pl7  + (pc1 - pl1 - ph1);
    double t8  = pl8  + (pc2 - pl2 - ph2);
    double t9  = pl9  + (pc3 - pl3 - ph3);
    double t10 = pl10 + (pc4 - pl4 - ph4);
    double t11 = (pc5 - pl5 - ph5);
    double t12 = (pc6 - pl6 - ph6) + ph0;
    double t13 = (pc7 - pl7 - ph7) + ph1;
    double t14 = (pc8 - pl8 - ph8) + ph2;
    double t15 = (pc9 - pl9) + ph3;
    double t16 = (pc10 - pl10) + ph4;
    double t17 = ph5;
    double t18 = ph6;
    double t19 = ph7;
    double t20 = ph8;

    // ================================================================
    // Phase 3: Reduction -- 10 Yuval + 1 standard, with SIMD scatter
    // ================================================================
    const v128_t rinv_01 = wasm_f64x2_make(r_limbs.div_r_inv_f[0], r_limbs.div_r_inv_f[1]);
    const v128_t rinv_23 = wasm_f64x2_make(r_limbs.div_r_inv_f[2], r_limbs.div_r_inv_f[3]);
    const v128_t rinv_45 = wasm_f64x2_make(r_limbs.div_r_inv_f[4], r_limbs.div_r_inv_f[5]);
    const v128_t rinv_67 = wasm_f64x2_make(r_limbs.div_r_inv_f[6], r_limbs.div_r_inv_f[7]);
    const v128_t rinv_89 = wasm_f64x2_make(r_limbs.div_r_inv_f[8], r_limbs.div_r_inv_f[9]);

    double qi, ki;
    v128_t ks, p01, p23, p45, p67, p89;

#define YUVAL_STEP_SIMD(TI, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    qi = std::floor(TI * SD);                                               \
    ki = TI - qi * SU;                                                       \
    ks = wasm_f64x2_splat(ki);                                               \
    p01 = fma_v(ks, rinv_01, wasm_f64x2_make(T1 + qi, T2));                 \
    T1 = wasm_f64x2_extract_lane(p01, 0);                                   \
    T2 = wasm_f64x2_extract_lane(p01, 1);                                   \
    p23 = fma_v(ks, rinv_23, wasm_f64x2_make(T3, T4));                      \
    T3 = wasm_f64x2_extract_lane(p23, 0);                                   \
    T4 = wasm_f64x2_extract_lane(p23, 1);                                   \
    p45 = fma_v(ks, rinv_45, wasm_f64x2_make(T5, T6));                      \
    T5 = wasm_f64x2_extract_lane(p45, 0);                                   \
    T6 = wasm_f64x2_extract_lane(p45, 1);                                   \
    p67 = fma_v(ks, rinv_67, wasm_f64x2_make(T7, T8));                      \
    T7 = wasm_f64x2_extract_lane(p67, 0);                                   \
    T8 = wasm_f64x2_extract_lane(p67, 1);                                   \
    p89 = fma_v(ks, rinv_89, wasm_f64x2_make(T9, T10));                     \
    T9 = wasm_f64x2_extract_lane(p89, 0);                                   \
    T10 = wasm_f64x2_extract_lane(p89, 1);                                  \
    T11 = fma_s(ki, r_limbs.div_r_inv_f[10], T11);

    YUVAL_STEP_SIMD(t0,  t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11)
    YUVAL_STEP_SIMD(t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12)
    YUVAL_STEP_SIMD(t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13)
    YUVAL_STEP_SIMD(t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14)
    YUVAL_STEP_SIMD(t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15)
    YUVAL_STEP_SIMD(t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16)
    YUVAL_STEP_SIMD(t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17)
    YUVAL_STEP_SIMD(t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18)
    YUVAL_STEP_SIMD(t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18, t19)
    YUVAL_STEP_SIMD(t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20)

#undef YUVAL_STEP_SIMD

    // Standard step 10
    {
        // Limbs 0–1 use scalar fma below (t10, t11), so no vector pair for mod_01.
        const v128_t mod_23 = wasm_f64x2_make(r_limbs.modulus_f[2], r_limbs.modulus_f[3]);
        const v128_t mod_45 = wasm_f64x2_make(r_limbs.modulus_f[4], r_limbs.modulus_f[5]);
        const v128_t mod_67 = wasm_f64x2_make(r_limbs.modulus_f[6], r_limbs.modulus_f[7]);
        const v128_t mod_89 = wasm_f64x2_make(r_limbs.modulus_f[8], r_limbs.modulus_f[9]);

        double q10 = std::floor(t10 * SD);
        double k_base = t10 - q10 * SU;
        double k_product = k_base * NP0_F;
        double k_std = k_product - std::floor(k_product * SD) * SU;

        t10 = fma_s(k_std, r_limbs.modulus_f[0], t10);
        double carry_10 = std::floor(t10 * SD);

        ks = wasm_f64x2_splat(k_std);
        t11 = fma_s(k_std, r_limbs.modulus_f[1], t11 + carry_10);

        v128_t pm23 = fma_v(ks, mod_23, wasm_f64x2_make(t12, t13));
        t12 = wasm_f64x2_extract_lane(pm23, 0);
        t13 = wasm_f64x2_extract_lane(pm23, 1);

        v128_t pm45 = fma_v(ks, mod_45, wasm_f64x2_make(t14, t15));
        t14 = wasm_f64x2_extract_lane(pm45, 0);
        t15 = wasm_f64x2_extract_lane(pm45, 1);

        v128_t pm67 = fma_v(ks, mod_67, wasm_f64x2_make(t16, t17));
        t16 = wasm_f64x2_extract_lane(pm67, 0);
        t17 = wasm_f64x2_extract_lane(pm67, 1);

        v128_t pm89 = fma_v(ks, mod_89, wasm_f64x2_make(t18, t19));
        t18 = wasm_f64x2_extract_lane(pm89, 0);
        t19 = wasm_f64x2_extract_lane(pm89, 1);

        t20 = fma_s(k_std, r_limbs.modulus_f[10], t20);
    }

    // ================================================================
    // Phase 4: Integer carry propagation + output
    // ================================================================
    uint64_t r11 = static_cast<uint64_t>(static_cast<int64_t>(t11));
    uint64_t r12 = static_cast<uint64_t>(static_cast<int64_t>(t12));
    uint64_t r13 = static_cast<uint64_t>(static_cast<int64_t>(t13));
    uint64_t r14 = static_cast<uint64_t>(static_cast<int64_t>(t14));
    uint64_t r15 = static_cast<uint64_t>(static_cast<int64_t>(t15));
    uint64_t r16 = static_cast<uint64_t>(static_cast<int64_t>(t16));
    uint64_t r17 = static_cast<uint64_t>(static_cast<int64_t>(t17));
    uint64_t r18 = static_cast<uint64_t>(static_cast<int64_t>(t18));
    uint64_t r19 = static_cast<uint64_t>(static_cast<int64_t>(t19));
    uint64_t r20 = static_cast<uint64_t>(static_cast<int64_t>(t20));

    r12 += r11 >> 24; r11 &= M24;
    r13 += r12 >> 24; r12 &= M24;
    r14 += r13 >> 24; r13 &= M24;
    r15 += r14 >> 24; r14 &= M24;
    r16 += r15 >> 24; r15 &= M24;
    r17 += r16 >> 24; r16 &= M24;
    r18 += r17 >> 24; r17 &= M24;
    r19 += r18 >> 24; r18 &= M24;
    r20 += r19 >> 24; r19 &= M24;

    return {
        r11 | (r12 << 24) | (r13 << 48),
        (r13 >> 16) | (r14 << 8) | (r15 << 32) | (r16 << 56),
        (r16 >> 8) | (r17 << 16) | (r18 << 40),
        r19 | (r20 << 24)
    };
}

/**
 * @brief Dual Montgomery multiplication: two independent muls in one call.
 *
 * out1 = a1 * b1 mod p, out2 = a2 * b2 mod p.
 *
 * Same algorithm as fma_simd, but both f64x2 lanes carry entirely separate
 * multiplications from start to finish (lane 0 = first mul, lane 1 = second).
 * This gives better SIMD utilization than fma_simd — no lane ever idles
 * (fma_simd leaves lane 1 idle during the 5×5 P_hi phase).
 *
 * Point arithmetic (e.g., mixed addition) naturally has pairs of independent
 * field muls, making this the preferred entry point when two muls are available.
 */
template <class T>
void field<T>::montgomery_mul_wasm_fma_simd2(
    const field& a1, const field& b1,
    const field& a2, const field& b2,
    field& out1, field& out2) noexcept
{
    constexpr uint64_t M24 = (1ULL << R_LIMB_BITS) - 1;
    constexpr double NP0_F = static_cast<double>(compute_r_inv(modulus.data[0]) & M24);

    auto fma_v = [](v128_t va, v128_t vb, v128_t vc) -> v128_t {
        return __builtin_wasm_relaxed_madd_f64x2(va, vb, vc);
    };

    const v128_t sd = wasm_f64x2_splat(0x1p-24); // 2^{-R_LIMB_BITS}
    const v128_t su = wasm_f64x2_splat(0x1p24);  // 2^{R_LIMB_BITS}

    const v128_t rinv0  = wasm_f64x2_splat(r_limbs.div_r_inv_f[0]);
    const v128_t rinv1  = wasm_f64x2_splat(r_limbs.div_r_inv_f[1]);
    const v128_t rinv2  = wasm_f64x2_splat(r_limbs.div_r_inv_f[2]);
    const v128_t rinv3  = wasm_f64x2_splat(r_limbs.div_r_inv_f[3]);
    const v128_t rinv4  = wasm_f64x2_splat(r_limbs.div_r_inv_f[4]);
    const v128_t rinv5  = wasm_f64x2_splat(r_limbs.div_r_inv_f[5]);
    const v128_t rinv6  = wasm_f64x2_splat(r_limbs.div_r_inv_f[6]);
    const v128_t rinv7  = wasm_f64x2_splat(r_limbs.div_r_inv_f[7]);
    const v128_t rinv8  = wasm_f64x2_splat(r_limbs.div_r_inv_f[8]);
    const v128_t rinv9  = wasm_f64x2_splat(r_limbs.div_r_inv_f[9]);
    const v128_t rinv10 = wasm_f64x2_splat(r_limbs.div_r_inv_f[10]);

    // Phase 1: Convert — lane 0 = (a1,b1), lane 1 = (a2,b2)
#define LIMB24(D, IDX, SHIFT) static_cast<double>((D[IDX] >> SHIFT) & M24)
#define LIMB24_CROSS(D, I0, S0, I1, S1) \
    static_cast<double>(((D[I0] >> S0) | (D[I1] << S1)) & M24)
#define LIMB24_TOP(D, IDX, SHIFT) static_cast<double>(D[IDX] >> SHIFT)

    v128_t af0  = wasm_f64x2_make(LIMB24(a1.data,0,0),  LIMB24(a2.data,0,0));
    v128_t af1  = wasm_f64x2_make(LIMB24(a1.data,0,24), LIMB24(a2.data,0,24));
    v128_t af2  = wasm_f64x2_make(LIMB24_CROSS(a1.data,0,48,1,16), LIMB24_CROSS(a2.data,0,48,1,16));
    v128_t af3  = wasm_f64x2_make(LIMB24(a1.data,1,8),  LIMB24(a2.data,1,8));
    v128_t af4  = wasm_f64x2_make(LIMB24(a1.data,1,32), LIMB24(a2.data,1,32));
    v128_t af5  = wasm_f64x2_make(LIMB24_CROSS(a1.data,1,56,2,8), LIMB24_CROSS(a2.data,1,56,2,8));
    v128_t af6  = wasm_f64x2_make(LIMB24(a1.data,2,16), LIMB24(a2.data,2,16));
    v128_t af7  = wasm_f64x2_make(LIMB24(a1.data,2,40), LIMB24(a2.data,2,40));
    v128_t af8  = wasm_f64x2_make(LIMB24(a1.data,3,0),  LIMB24(a2.data,3,0));
    v128_t af9  = wasm_f64x2_make(LIMB24(a1.data,3,24), LIMB24(a2.data,3,24));
    v128_t af10 = wasm_f64x2_make(LIMB24_TOP(a1.data,3,48), LIMB24_TOP(a2.data,3,48));

    v128_t bf0  = wasm_f64x2_make(LIMB24(b1.data,0,0),  LIMB24(b2.data,0,0));
    v128_t bf1  = wasm_f64x2_make(LIMB24(b1.data,0,24), LIMB24(b2.data,0,24));
    v128_t bf2  = wasm_f64x2_make(LIMB24_CROSS(b1.data,0,48,1,16), LIMB24_CROSS(b2.data,0,48,1,16));
    v128_t bf3  = wasm_f64x2_make(LIMB24(b1.data,1,8),  LIMB24(b2.data,1,8));
    v128_t bf4  = wasm_f64x2_make(LIMB24(b1.data,1,32), LIMB24(b2.data,1,32));
    v128_t bf5  = wasm_f64x2_make(LIMB24_CROSS(b1.data,1,56,2,8), LIMB24_CROSS(b2.data,1,56,2,8));
    v128_t bf6  = wasm_f64x2_make(LIMB24(b1.data,2,16), LIMB24(b2.data,2,16));
    v128_t bf7  = wasm_f64x2_make(LIMB24(b1.data,2,40), LIMB24(b2.data,2,40));
    v128_t bf8  = wasm_f64x2_make(LIMB24(b1.data,3,0),  LIMB24(b2.data,3,0));
    v128_t bf9  = wasm_f64x2_make(LIMB24(b1.data,3,24), LIMB24(b2.data,3,24));
    v128_t bf10 = wasm_f64x2_make(LIMB24_TOP(b1.data,3,48), LIMB24_TOP(b2.data,3,48));

#undef LIMB24
#undef LIMB24_CROSS
#undef LIMB24_TOP

    // Phase 2: Karatsuba product (all v128_t)
    v128_t sl0 = wasm_f64x2_add(af0, af6);   v128_t sr0 = wasm_f64x2_add(bf0, bf6);
    v128_t sl1 = wasm_f64x2_add(af1, af7);   v128_t sr1 = wasm_f64x2_add(bf1, bf7);
    v128_t sl2 = wasm_f64x2_add(af2, af8);   v128_t sr2 = wasm_f64x2_add(bf2, bf8);
    v128_t sl3 = wasm_f64x2_add(af3, af9);   v128_t sr3 = wasm_f64x2_add(bf3, bf9);
    v128_t sl4 = wasm_f64x2_add(af4, af10);  v128_t sr4 = wasm_f64x2_add(bf4, bf10);
    v128_t sl5 = af5;                         v128_t sr5 = bf5;

    v128_t pl0  = wasm_f64x2_mul(af0, bf0);
    v128_t pl1  = fma_v(af1, bf0, wasm_f64x2_mul(af0, bf1));
    v128_t pl2  = fma_v(af2, bf0, fma_v(af1, bf1, wasm_f64x2_mul(af0, bf2)));
    v128_t pl3  = fma_v(af3, bf0, fma_v(af2, bf1, fma_v(af1, bf2, wasm_f64x2_mul(af0, bf3))));
    v128_t pl4  = fma_v(af4, bf0, fma_v(af3, bf1, fma_v(af2, bf2,
                  fma_v(af1, bf3, wasm_f64x2_mul(af0, bf4)))));
    v128_t pl5  = fma_v(af5, bf0, fma_v(af4, bf1, fma_v(af3, bf2,
                  fma_v(af2, bf3, fma_v(af1, bf4, wasm_f64x2_mul(af0, bf5))))));
    v128_t pl6  = fma_v(af5, bf1, fma_v(af4, bf2, fma_v(af3, bf3,
                  fma_v(af2, bf4, wasm_f64x2_mul(af1, bf5)))));
    v128_t pl7  = fma_v(af5, bf2, fma_v(af4, bf3, fma_v(af3, bf4, wasm_f64x2_mul(af2, bf5))));
    v128_t pl8  = fma_v(af5, bf3, fma_v(af4, bf4, wasm_f64x2_mul(af3, bf5)));
    v128_t pl9  = fma_v(af5, bf4, wasm_f64x2_mul(af4, bf5));
    v128_t pl10 = wasm_f64x2_mul(af5, bf5);

    v128_t ph0 = wasm_f64x2_mul(af6, bf6);
    v128_t ph1 = fma_v(af7, bf6, wasm_f64x2_mul(af6, bf7));
    v128_t ph2 = fma_v(af8, bf6, fma_v(af7, bf7, wasm_f64x2_mul(af6, bf8)));
    v128_t ph3 = fma_v(af9, bf6, fma_v(af8, bf7, fma_v(af7, bf8, wasm_f64x2_mul(af6, bf9))));
    v128_t ph4 = fma_v(af10, bf6, fma_v(af9, bf7, fma_v(af8, bf8,
                 fma_v(af7, bf9, wasm_f64x2_mul(af6, bf10)))));
    v128_t ph5 = fma_v(af10, bf7, fma_v(af9, bf8, fma_v(af8, bf9, wasm_f64x2_mul(af7, bf10))));
    v128_t ph6 = fma_v(af10, bf8, fma_v(af9, bf9, wasm_f64x2_mul(af8, bf10)));
    v128_t ph7 = fma_v(af10, bf9, wasm_f64x2_mul(af9, bf10));
    v128_t ph8 = wasm_f64x2_mul(af10, bf10);

    v128_t pc0  = wasm_f64x2_mul(sl0, sr0);
    v128_t pc1  = fma_v(sl1, sr0, wasm_f64x2_mul(sl0, sr1));
    v128_t pc2  = fma_v(sl2, sr0, fma_v(sl1, sr1, wasm_f64x2_mul(sl0, sr2)));
    v128_t pc3  = fma_v(sl3, sr0, fma_v(sl2, sr1, fma_v(sl1, sr2, wasm_f64x2_mul(sl0, sr3))));
    v128_t pc4  = fma_v(sl4, sr0, fma_v(sl3, sr1, fma_v(sl2, sr2,
                  fma_v(sl1, sr3, wasm_f64x2_mul(sl0, sr4)))));
    v128_t pc5  = fma_v(sl5, sr0, fma_v(sl4, sr1, fma_v(sl3, sr2,
                  fma_v(sl2, sr3, fma_v(sl1, sr4, wasm_f64x2_mul(sl0, sr5))))));
    v128_t pc6  = fma_v(sl5, sr1, fma_v(sl4, sr2, fma_v(sl3, sr3,
                  fma_v(sl2, sr4, wasm_f64x2_mul(sl1, sr5)))));
    v128_t pc7  = fma_v(sl5, sr2, fma_v(sl4, sr3, fma_v(sl3, sr4, wasm_f64x2_mul(sl2, sr5))));
    v128_t pc8  = fma_v(sl5, sr3, fma_v(sl4, sr4, wasm_f64x2_mul(sl3, sr5)));
    v128_t pc9  = fma_v(sl5, sr4, wasm_f64x2_mul(sl4, sr5));
    v128_t pc10 = wasm_f64x2_mul(sl5, sr5);

    // Combine
    v128_t t0  = pl0;
    v128_t t1  = pl1;
    v128_t t2  = pl2;
    v128_t t3  = pl3;
    v128_t t4  = pl4;
    v128_t t5  = pl5;
    v128_t t6  = wasm_f64x2_add(pl6,  wasm_f64x2_sub(pc0, wasm_f64x2_add(pl0, ph0)));
    v128_t t7  = wasm_f64x2_add(pl7,  wasm_f64x2_sub(pc1, wasm_f64x2_add(pl1, ph1)));
    v128_t t8  = wasm_f64x2_add(pl8,  wasm_f64x2_sub(pc2, wasm_f64x2_add(pl2, ph2)));
    v128_t t9  = wasm_f64x2_add(pl9,  wasm_f64x2_sub(pc3, wasm_f64x2_add(pl3, ph3)));
    v128_t t10 = wasm_f64x2_add(pl10, wasm_f64x2_sub(pc4, wasm_f64x2_add(pl4, ph4)));
    v128_t t11 = wasm_f64x2_sub(pc5, wasm_f64x2_add(pl5, ph5));
    v128_t t12 = wasm_f64x2_add(wasm_f64x2_sub(pc6, wasm_f64x2_add(pl6, ph6)), ph0);
    v128_t t13 = wasm_f64x2_add(wasm_f64x2_sub(pc7, wasm_f64x2_add(pl7, ph7)), ph1);
    v128_t t14 = wasm_f64x2_add(wasm_f64x2_sub(pc8, wasm_f64x2_add(pl8, ph8)), ph2);
    v128_t t15 = wasm_f64x2_add(wasm_f64x2_sub(pc9, pl9), ph3);
    v128_t t16 = wasm_f64x2_add(wasm_f64x2_sub(pc10, pl10), ph4);
    v128_t t17 = ph5;
    v128_t t18 = ph6;
    v128_t t19 = ph7;
    v128_t t20 = ph8;

    // Phase 3: Reduction — 10 Yuval + 1 standard (all v128_t)
    v128_t qi, ki;

#define YUVAL_STEP_V(TI, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    qi = wasm_f64x2_floor(wasm_f64x2_mul(TI, sd));                       \
    ki = wasm_f64x2_sub(TI, wasm_f64x2_mul(qi, su));                     \
    T1  = fma_v(ki, rinv0,  wasm_f64x2_add(T1, qi));                     \
    T2  = fma_v(ki, rinv1,  T2);                                          \
    T3  = fma_v(ki, rinv2,  T3);                                          \
    T4  = fma_v(ki, rinv3,  T4);                                          \
    T5  = fma_v(ki, rinv4,  T5);                                          \
    T6  = fma_v(ki, rinv5,  T6);                                          \
    T7  = fma_v(ki, rinv6,  T7);                                          \
    T8  = fma_v(ki, rinv7,  T8);                                          \
    T9  = fma_v(ki, rinv8,  T9);                                          \
    T10 = fma_v(ki, rinv9,  T10);                                         \
    T11 = fma_v(ki, rinv10, T11);

    YUVAL_STEP_V(t0,  t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11)
    YUVAL_STEP_V(t1,  t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12)
    YUVAL_STEP_V(t2,  t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13)
    YUVAL_STEP_V(t3,  t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14)
    YUVAL_STEP_V(t4,  t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15)
    YUVAL_STEP_V(t5,  t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16)
    YUVAL_STEP_V(t6,  t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17)
    YUVAL_STEP_V(t7,  t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18)
    YUVAL_STEP_V(t8,  t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18, t19)
    YUVAL_STEP_V(t9,  t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20)

#undef YUVAL_STEP_V

    // Standard step 10
    {
        const v128_t mod0  = wasm_f64x2_splat(r_limbs.modulus_f[0]);
        const v128_t mod1  = wasm_f64x2_splat(r_limbs.modulus_f[1]);
        const v128_t mod2  = wasm_f64x2_splat(r_limbs.modulus_f[2]);
        const v128_t mod3  = wasm_f64x2_splat(r_limbs.modulus_f[3]);
        const v128_t mod4  = wasm_f64x2_splat(r_limbs.modulus_f[4]);
        const v128_t mod5  = wasm_f64x2_splat(r_limbs.modulus_f[5]);
        const v128_t mod6  = wasm_f64x2_splat(r_limbs.modulus_f[6]);
        const v128_t mod7  = wasm_f64x2_splat(r_limbs.modulus_f[7]);
        const v128_t mod8  = wasm_f64x2_splat(r_limbs.modulus_f[8]);
        const v128_t mod9  = wasm_f64x2_splat(r_limbs.modulus_f[9]);
        const v128_t mod10 = wasm_f64x2_splat(r_limbs.modulus_f[10]);
        const v128_t np0   = wasm_f64x2_splat(NP0_F);

        v128_t q10 = wasm_f64x2_floor(wasm_f64x2_mul(t10, sd));
        v128_t k_base = wasm_f64x2_sub(t10, wasm_f64x2_mul(q10, su));
        v128_t k_product = wasm_f64x2_mul(k_base, np0);
        v128_t ks = wasm_f64x2_sub(k_product,
                    wasm_f64x2_mul(wasm_f64x2_floor(wasm_f64x2_mul(k_product, sd)), su));

        t10 = fma_v(ks, mod0, t10);
        v128_t carry_10 = wasm_f64x2_floor(wasm_f64x2_mul(t10, sd));
        t11 = fma_v(ks, mod1,  wasm_f64x2_add(t11, carry_10));
        t12 = fma_v(ks, mod2,  t12);
        t13 = fma_v(ks, mod3,  t13);
        t14 = fma_v(ks, mod4,  t14);
        t15 = fma_v(ks, mod5,  t15);
        t16 = fma_v(ks, mod6,  t16);
        t17 = fma_v(ks, mod7,  t17);
        t18 = fma_v(ks, mod8,  t18);
        t19 = fma_v(ks, mod9,  t19);
        t20 = fma_v(ks, mod10, t20);
    }

    // Phase 4: Extract lanes, integer carry propagation, output
#define EXTRACT_AND_FINALIZE(LANE, OUT)                                    \
    {                                                                       \
        uint64_t r11 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t11, LANE)));                          \
        uint64_t r12 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t12, LANE)));                          \
        uint64_t r13 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t13, LANE)));                          \
        uint64_t r14 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t14, LANE)));                          \
        uint64_t r15 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t15, LANE)));                          \
        uint64_t r16 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t16, LANE)));                          \
        uint64_t r17 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t17, LANE)));                          \
        uint64_t r18 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t18, LANE)));                          \
        uint64_t r19 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t19, LANE)));                          \
        uint64_t r20 = static_cast<uint64_t>(static_cast<int64_t>(         \
            wasm_f64x2_extract_lane(t20, LANE)));                          \
        r12 += r11 >> 24; r11 &= M24;                                     \
        r13 += r12 >> 24; r12 &= M24;                                     \
        r14 += r13 >> 24; r13 &= M24;                                     \
        r15 += r14 >> 24; r14 &= M24;                                     \
        r16 += r15 >> 24; r15 &= M24;                                     \
        r17 += r16 >> 24; r16 &= M24;                                     \
        r18 += r17 >> 24; r17 &= M24;                                     \
        r19 += r18 >> 24; r18 &= M24;                                     \
        r20 += r19 >> 24; r19 &= M24;                                     \
        OUT = field{                                                        \
            r11 | (r12 << 24) | (r13 << 48),                               \
            (r13 >> 16) | (r14 << 8) | (r15 << 32) | (r16 << 56),         \
            (r16 >> 8) | (r17 << 16) | (r18 << 40),                       \
            r19 | (r20 << 24)                                              \
        };                                                                  \
    }

    EXTRACT_AND_FINALIZE(0, out1)
    EXTRACT_AND_FINALIZE(1, out2)

#undef EXTRACT_AND_FINALIZE
}
#endif // BB_R_LIMB_BITS == 24 && __wasm_simd128__
