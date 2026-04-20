// === AUDIT STATUS ===
// internal:    { status: Planned, auditors: [], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/ecc/groups/element.hpp"
#include "element.hpp"
#include <cstdint>

// NOLINTBEGIN(readability-implicit-bool-conversion, cppcoreguidelines-avoid-c-arrays)
namespace bb::group_elements {
template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>::element(const Fq& a, const Fq& b, const Fq& c) noexcept
    : x(a)
    , y(b)
    , z(c)
{}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>::element(const element& other) noexcept
    : x(other.x)
    , y(other.y)
    , z(other.z)
{}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>::element(element&& other) noexcept
    : x(other.x)
    , y(other.y)
    , z(other.z)
{}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>::element(const affine_element<Fq, Fr, T>& other) noexcept
    : x(other.x)
    , y(other.y)
    , z(Fq::one())
{}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>& element<Fq, Fr, T>::operator=(const element& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T>& element<Fq, Fr, T>::operator=(element&& other) noexcept
{
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template <class Fq, class Fr, class T> constexpr element<Fq, Fr, T>::operator affine_element<Fq, Fr, T>() const noexcept
{
    if (is_point_at_infinity()) {
        affine_element<Fq, Fr, T> result;
        result.x = Fq(0);
        result.y = Fq(0);
        result.self_set_infinity();
        return result;
    }
    Fq z_inv = z.invert();
    Fq zz_inv = z_inv.sqr();
    Fq zzz_inv = zz_inv * z_inv;
    affine_element<Fq, Fr, T> result(x * zz_inv, y * zzz_inv);
    return result;
}

template <class Fq, class Fr, class T> constexpr void element<Fq, Fr, T>::self_dbl() noexcept
{
    if constexpr (Fq::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        if (is_point_at_infinity()) {
            return;
        }
    } else {
        if (x.is_msb_set_word()) {
            return;
        }
    }

    // Scheduled as 2 paired squarings + 1 paired mul + 1 single (vs. 7 singles).
    // Pairs 1 and 2 are pure squaring pairs (both lanes compute a²), so they
    // route to the triangular-product sqr_paired kernel (~41% fewer FMAs per
    // lane than mul_paired). Pair 3 is mixed (mul + sqr) and stays on
    // mul_paired — no independent sqr partner available at that point.

    // Pair 1: (x², y²) — the two independent input-coordinate squarings.
    Fq T0;
    Fq T1;
    Fq::template montgomery_sqr_batched<2>({ &x, &y }, { &T0, &T1 });

    // Pair 2: (T1², (T1 + x)²) — the two downstream squarings both depend on
    // T1 but not on each other.
    Fq T1_plus_x = T1 + x;
    Fq T2;
    Fq T1_sq2;
    Fq::template montgomery_sqr_batched<2>({ &T1, &T1_plus_x }, { &T2, &T1_sq2 });

    // T3 = T0 + T2;  T1 = T1_sq2 - T3;  T1 += T1;  // T1 = 4*S
    Fq T3 = T0 + T2;
    T1 = T1_sq2 - T3;
    T1 += T1;

    // T3 = 3*T0
    T3 = T0 + T0;
    T3 += T0;
    if constexpr (T::has_a) {
        // Not on the BN254/Grumpkin MSM hot path (has_a is false there).
        // Kept sequential; dedicated pairing is future work if needed.
        T3 += (T::a * z.sqr().sqr());
    }

    // Pair 3: (z_doubled · y, T3²) — the final mul and sqr are independent
    // once T3 (= 3*T0 [+ a*z⁴]) is known.
    Fq z_doubled = z + z;
    Fq new_z;
    Fq x_new;
    Fq::template montgomery_mul_batched<2>({ &z_doubled, &T3 }, { &y, &T3 }, { &new_z, &x_new });

    // x = x_new - 2*T1
    Fq twoT1 = T1 + T1;
    x = x_new - twoT1;

    // T2 = 8*T2
    T2 += T2;
    T2 += T2;
    T2 += T2;

    // y = (T1 - x) * T3 - T2 — last mul stays single (nothing left to pair).
    Fq y_pre = T1 - x;
    y = y_pre * T3;
    y -= T2;

    z = new_z;
}

template <class Fq, class Fr, class T> constexpr element<Fq, Fr, T> element<Fq, Fr, T>::dbl() const noexcept
{
    element result(*this);
    result.self_dbl();
    return result;
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator+=(const affine_element<Fq, Fr, T>& other) noexcept
{
    if constexpr (Fq::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        if (is_point_at_infinity()) {
            *this = { other.x, other.y, Fq::one() };
            return *this;
        }
    } else {
        const bool edge_case_trigger = x.is_msb_set() || other.x.is_msb_set();
        if (edge_case_trigger) {
            if (x.is_msb_set()) {
                *this = { other.x, other.y, Fq::one() };
            }
            return *this;
        }
    }

    // T0 = z1^2
    Fq T0 = z.sqr();

    // Pair 1: T1 = other.x * T0, T2 = z * T0 (both use T0 = z1^2)
    Fq T1;
    Fq T2;
    Fq::template montgomery_mul_batched<2>({ &other.x, &z }, { &T0, &T0 }, { &T1, &T2 });
    T1 -= x; // H = x2*z1^2 - x1

    // Pair 2: T2 = y2*z1^3, T3 = HH = H^2.
    // T3 is speculatively hoisted above the edge-case branch — the branch discards
    // it on either exit, and the branch is __builtin_expect(..., 0) so the wasted
    // paired lane is essentially never hit on random MSM input.
    Fq T3;
    Fq::template montgomery_mul_batched<2>({ &T2, &T1 }, { &other.y, &T1 }, { &T2, &T3 });
    T2 -= y; // y2*z1^3 - y1

    if (__builtin_expect(T1.is_zero(), 0)) {
        if (T2.is_zero()) {
            self_dbl();
            return *this;
        }
        self_set_infinity();
        return *this;
    }

    // T2 = 2R = 2(y2*z1^3 - y1); z' = z1 + H
    T2 += T2;
    z += T1;

    // T0 = z1^2 + HH (T3 already has HH from Pair 2)
    T0 += T3;

    // Pair 3: z_sq = (z1 + H)^2, R_sq = R^2. Both are true squarings —
    // dispatched as the dedicated sqr_paired kernel. R_sq is parked in a temp
    // so Pair 4 below can still consume the pre-update x.
    Fq z_sq;
    Fq R_sq;
    Fq::template montgomery_sqr_batched<2>({ &z, &T2 }, { &z_sq, &R_sq });
    z = z_sq - T0; // z3 = (z1 + H)^2 - z1^2 - HH

    // T3 = 4HH
    T3 += T3;
    T3 += T3;

    // Pair 4: T1 = T1*T3 (4HHH), T3 = T3*x (4HH*x1)
    Fq::template montgomery_mul_batched<2>({ &T1, &T3 }, { &T3, &x }, { &T1, &T3 });

    T0 = T3 + T3;  // 8HH*x1
    T0 += T1;      // 8HH*x1 + 4HHH
    x = R_sq - T0; // x3 = R^2 - 8HH*x1 - 4HHH
    T3 -= x;       // 4HH*x1 - x3

    // Pair 5: T1 = T1*y (4HHH*y1), T3 = T3*T2 (R*(4HH*x1-x3))
    Fq::template montgomery_mul_batched<2>({ &T1, &T3 }, { &y, &T2 }, { &T1, &T3 });

    T1 += T1;    // 8HHH*y1
    y = T3 - T1; // y3 = R*(4HH*x1-x3) - 8HHH*y1
    return *this;
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator+(const affine_element<Fq, Fr, T>& other) const noexcept
{
    element result(*this);
    return (result += other);
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator-=(const affine_element<Fq, Fr, T>& other) noexcept
{
    const affine_element<Fq, Fr, T> to_add{ other.x, -other.y };
    return operator+=(to_add);
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator-(const affine_element<Fq, Fr, T>& other) const noexcept
{
    element result(*this);
    return (result -= other);
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator+=(const element& other) noexcept
{
    if constexpr (Fq::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        bool p1_zero = is_point_at_infinity();
        bool p2_zero = other.is_point_at_infinity();
        if (__builtin_expect((p1_zero || p2_zero), 0)) {
            if (p1_zero && !p2_zero) {
                *this = other;
                return *this;
            }
            if (p2_zero && !p1_zero) {
                return *this;
            }
            self_set_infinity();
            return *this;
        }
    } else {
        bool p1_zero = x.is_msb_set();
        bool p2_zero = other.x.is_msb_set();
        if (__builtin_expect((p1_zero || p2_zero), 0)) {
            if (p1_zero && !p2_zero) {
                *this = other;
                return *this;
            }
            if (p2_zero && !p1_zero) {
                return *this;
            }
            self_set_infinity();
            return *this;
        }
    }
    // Jacobian + Jacobian addition, scheduled as eight paired Montgomery
    // kernel calls (16 logical muls/sqrs, zero single-kernel calls).
    //
    // Baseline: sqr (Z1Z1, Z2Z2); mul (S2=Z1Z1·z, U2=Z1Z1·ox, S2·=oy, U1=Z2Z2·x,
    //   S1=Z2Z2·oz, S1·=y); sqr (I=(2H)²); mul (J=H·I, U1·=I); sqr (x=F²);
    //   mul (J·=S1); mul (y·=F); sqr ((z+oz)²); mul (z·=H).
    //
    // The schedule below pairs all 16 ops across 8 paired kernel invocations.
    // Pair 1 is a pure-square pair and routes to sqr_paired (triangular kernel,
    // ~41% cheaper than mul). Pairs 4, 5, 7 mix one mul with one sqr; since no
    // two independent sqrs are simultaneously available without breaking the
    // mul dependency chain, those stay on mul_paired (the sqr lane benefits
    // transparently via WasmFmaBackend's internal sqr(x)=mul(x,x) routing).
    //
    // Data-flow exploits: z and Z1Z1 are dead after Pair 2 (allowing us to fold
    // `z += other.z` and `Z1Z1 += Z2Z2` early), and op15 (z² = (z+oz)²) and
    // op9 (I²) can run speculatively because they never overflow even on the
    // H==0 edge-case path (which is ~never hit for random MSM inputs).

    // Pair 1: (z², other.z²) — pure squaring pair, routes to the sqr-batched
    // triangular-product kernel (~41% fewer FMAs per lane than mul-batched).
    Fq Z1Z1;
    Fq Z2Z2;
    Fq::template montgomery_sqr_batched<2>({ &z, &other.z }, { &Z1Z1, &Z2Z2 });

    // Pair 2: (Z1Z1·z, Z1Z1·other.x)
    Fq S2;
    Fq U2;
    Fq::template montgomery_mul_batched<2>({ &Z1Z1, &Z1Z1 }, { &z, &other.x }, { &S2, &U2 });

    // Precompute (z + other.z) and (Z1Z1 + Z2Z2) into locals so *this stays
    // intact if we bail out to self_dbl() / self_set_infinity() on the edge
    // case below. Both feed the final z-update: z = ((z+oz)² - (Z1Z1+Z2Z2)) · H.
    Fq z_sum = z + other.z;
    Fq Z_sum = Z1Z1 + Z2Z2;

    // Pair 3: (S2·other.y, Z2Z2·x)  — S2 output aliases S2 input (safe: the
    // batched kernel reads all inputs to locals before writing any output)
    Fq U1;
    Fq::template montgomery_mul_batched<2>({ &S2, &Z2Z2 }, { &other.y, &x }, { &S2, &U1 });

    // H and 2H are ready as soon as U1 is.
    Fq H(U2 - U1);
    Fq twoH = H + H;

    // Pair 4: (Z2Z2·other.z, (z+oz)²) — z_sum² is speculative w.r.t. the H==0
    // edge case but is always numerically safe (never overflows).
    Fq S1_pre;
    Fq z_sq;
    Fq::template montgomery_mul_batched<2>({ &Z2Z2, &z_sum }, { &other.z, &z_sum }, { &S1_pre, &z_sq });

    // z's remaining subtraction before the final mul.
    Fq z_after_sub = z_sq - Z_sum;

    // Pair 5: (S1·y, (2H)²) — (2H)² is also speculative but safe.
    Fq S1;
    Fq I_var;
    Fq::template montgomery_mul_batched<2>({ &S1_pre, &twoH }, { &y, &twoH }, { &S1, &I_var });

    // F and 2F are ready as soon as S1 is (original code's `F += F`).
    Fq F(S2 - S1);
    Fq twoF = F + F;

    // Edge case: H == 0 means the two points share an x-coordinate.
    // F == 0 ⇒ P == Q (doubling); F != 0 ⇒ P == -Q (infinity).
    // Pairs 4 and 5 computed (z+oz)² and (2H)² speculatively; that work is
    // wasted here but H==0 is astronomically rare for random MSM inputs.
    if (__builtin_expect(H.is_zero(), 0)) {
        if (F.is_zero()) {
            self_dbl();
            return *this;
        }
        self_set_infinity();
        return *this;
    }

    // Pair 6: (H·I, U1·I)
    Fq J;
    Fq U1_new;
    Fq::template montgomery_mul_batched<2>({ &H, &U1 }, { &I_var, &I_var }, { &J, &U1_new });

    // U2 update = 2·U1_new + J (all adds).
    Fq U2_sum = U1_new + U1_new;
    U2_sum += J;

    // Pair 7: ((2F)², z_after_sub·H) — final x squaring fused with final z mul.
    Fq x_F_sq;
    Fq z_new;
    Fq::template montgomery_mul_batched<2>({ &twoF, &z_after_sub }, { &twoF, &H }, { &x_F_sq, &z_new });

    x = x_F_sq - U2_sum;
    Fq y_pre = U1_new - x;

    // Pair 8: (J·S1, y_pre·(2F))
    Fq J_S1;
    Fq y_mul;
    Fq::template montgomery_mul_batched<2>({ &J, &y_pre }, { &S1, &twoF }, { &J_S1, &y_mul });

    Fq J_doubled = J_S1 + J_S1;
    y = y_mul - J_doubled;
    z = z_new;
    return *this;
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator+(const element& other) const noexcept
{
    element result(*this);
    return (result += other);
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator-=(const element& other) noexcept
{
    const element to_add{ other.x, -other.y, other.z };
    return operator+=(to_add);
}

template <class Fq, class Fr, class T>
constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator-(const element& other) const noexcept
{
    element result(*this);
    return (result -= other);
}

template <class Fq, class Fr, class T> constexpr element<Fq, Fr, T> element<Fq, Fr, T>::operator-() const noexcept
{
    return { x, -y, z };
}

template <class Fq, class Fr, class T>
element<Fq, Fr, T> element<Fq, Fr, T>::operator*(const Fr& exponent) const noexcept
{
    if constexpr (T::USE_ENDOMORPHISM) {
        return mul_with_endomorphism(exponent);
    }
    return mul_without_endomorphism(exponent);
}

template <class Fq, class Fr, class T> element<Fq, Fr, T> element<Fq, Fr, T>::operator*=(const Fr& exponent) noexcept
{
    *this = operator*(exponent);
    return *this;
}

template <class Fq, class Fr, class T> constexpr element<Fq, Fr, T> element<Fq, Fr, T>::normalize() const noexcept
{
    const affine_element<Fq, Fr, T> converted = *this;
    return element(converted);
}

template <class Fq, class Fr, class T> element<Fq, Fr, T> element<Fq, Fr, T>::infinity()
{
    element<Fq, Fr, T> e{};
    e.self_set_infinity();
    return e;
}

template <class Fq, class Fr, class T> constexpr element<Fq, Fr, T> element<Fq, Fr, T>::set_infinity() const noexcept
{
    element result(*this);
    result.self_set_infinity();
    return result;
}

template <class Fq, class Fr, class T> constexpr void element<Fq, Fr, T>::self_set_infinity() noexcept
{
    if constexpr (Fq::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        // We set the value of x equal to modulus to represent inifinty
        x.data[0] = Fq::modulus.data[0];
        x.data[1] = Fq::modulus.data[1];
        x.data[2] = Fq::modulus.data[2];
        x.data[3] = Fq::modulus.data[3];
    } else {
        (*this).x = Fq::zero();
        (*this).y = Fq::zero();
        (*this).z = Fq::zero();
        x.self_set_msb();
    }
}

template <class Fq, class Fr, class T> constexpr bool element<Fq, Fr, T>::is_point_at_infinity() const noexcept
{
    if constexpr (Fq::modulus.data[3] >= MODULUS_TOP_LIMB_LARGE_THRESHOLD) {
        // We check if the value of x is equal to modulus to represent inifinty
        return ((x.data[0] ^ Fq::modulus.data[0]) | (x.data[1] ^ Fq::modulus.data[1]) |
                (x.data[2] ^ Fq::modulus.data[2]) | (x.data[3] ^ Fq::modulus.data[3])) == 0;
    } else {
        return (x.is_msb_set());
    }
}

template <class Fq, class Fr, class T> constexpr bool element<Fq, Fr, T>::on_curve() const noexcept
{
    if (is_point_at_infinity()) {
        return true;
    }
    // We specify the point at inifinity not by (0 \lambda 0), so z should not be 0
    if (z.is_zero()) {
        return false;
    }
    Fq zz = z.sqr();
    Fq zzzz = zz.sqr();
    Fq bz_6 = zzzz * zz * T::b;
    if constexpr (T::has_a) {
        bz_6 += (x * T::a) * zzzz;
    }
    Fq xxx = x.sqr() * x + bz_6;
    Fq yy = y.sqr();
    return (xxx == yy);
}

template <class Fq, class Fr, class T>
constexpr bool element<Fq, Fr, T>::operator==(const element& other) const noexcept
{
    // If one of points is not on curve, we have no business comparing them.
    if ((!on_curve()) || (!other.on_curve())) {
        return false;
    }
    bool am_infinity = is_point_at_infinity();
    bool is_infinity = other.is_point_at_infinity();
    bool both_infinity = am_infinity && is_infinity;
    // If just one is infinity, then they are obviously not equal.
    if ((!both_infinity) && (am_infinity || is_infinity)) {
        return false;
    }
    const Fq lhs_zz = z.sqr();
    const Fq lhs_zzz = lhs_zz * z;
    const Fq rhs_zz = other.z.sqr();
    const Fq rhs_zzz = rhs_zz * other.z;

    const Fq lhs_x = x * rhs_zz;
    const Fq lhs_y = y * rhs_zzz;

    const Fq rhs_x = other.x * lhs_zz;
    const Fq rhs_y = other.y * lhs_zzz;
    return both_infinity || ((lhs_x == rhs_x) && (lhs_y == rhs_y));
}

template <class Fq, class Fr, class T>
element<Fq, Fr, T> element<Fq, Fr, T>::random_element(numeric::RNG* engine) noexcept
{
    if constexpr (T::can_hash_to_curve) {
        element result = random_coordinates_on_curve(engine);
        result.z = Fq::random_element(engine);
        Fq zz = result.z.sqr();
        Fq zzz = zz * result.z;
        result.x *= zz;
        result.y *= zzz;
        return result;
    } else {
        Fr scalar = Fr::random_element(engine);
        return (element{ T::one_x, T::one_y, Fq::one() } * scalar);
    }
}

template <class Fq, class Fr, class T>
element<Fq, Fr, T> element<Fq, Fr, T>::mul_without_endomorphism(const Fr& scalar) const noexcept
{
    const uint256_t converted_scalar(scalar);

    if (converted_scalar == 0) {
        return element::infinity();
    }

    element accumulator(*this);
    const uint64_t maximum_set_bit = converted_scalar.get_msb();
    // This is simpler and doublings of infinity should be fast. We should think if we want to defend against the
    // timing leak here (if used with ECDSA it can sometimes lead to private key compromise)
    for (uint64_t i = maximum_set_bit - 1; i < maximum_set_bit; --i) {
        accumulator.self_dbl();
        if (converted_scalar.get_bit(i)) {
            accumulator += *this;
        }
    }
    return accumulator;
}

namespace detail {
// Represents the result of
using EndoScalars = std::pair<std::array<uint64_t, 2>, std::array<uint64_t, 2>>;

/**
 * @brief Handles the WNAF computation for scalars that are split using an endomorphism,
 * achieved through `split_into_endomorphism_scalars`. It facilitates efficient computation of elliptic curve
 * point multiplication by optimizing the representation of these scalars.
 *
 * @tparam Element The data type of elements in the elliptic curve.
 * @tparam NUM_ROUNDS The number of computation rounds for WNAF.
 */
template <typename Element, std::size_t NUM_ROUNDS> struct EndomorphismWnaf {
    // NUM_WNAF_BITS: Number of bits per window in the WNAF representation.
    static constexpr size_t NUM_WNAF_BITS = 4;
    // table: Stores the WNAF representation of the scalars.
    std::array<uint64_t, NUM_ROUNDS * 2> table;
    // skew and endo_skew: Indicate if our original scalar is even or odd.
    bool skew = false;
    bool endo_skew = false;

    /**
     * @param scalars A pair of 128-bit scalars (as two uint64_t arrays), split using an endomorphism.
     */
    EndomorphismWnaf(const EndoScalars& scalars)
    {
        wnaf::fixed_wnaf(&scalars.first[0], &table[0], skew, 0, 2, NUM_WNAF_BITS);
        wnaf::fixed_wnaf(&scalars.second[0], &table[1], endo_skew, 0, 2, NUM_WNAF_BITS);
    }
};

} // namespace detail

template <class Fq, class Fr, class T>
element<Fq, Fr, T> element<Fq, Fr, T>::mul_with_endomorphism(const Fr& scalar) const noexcept
{
    // Consider the infinity flag, return infinity if set
    if (is_point_at_infinity()) {
        return element::infinity();
    }
    constexpr size_t NUM_ROUNDS = 32;
    const Fr converted_scalar = scalar.from_montgomery_form();

    if (converted_scalar.is_zero()) {
        return element::infinity();
    }
    static constexpr size_t LOOKUP_SIZE = 8;
    std::array<element, LOOKUP_SIZE> lookup_table;

    element d2 = dbl();
    lookup_table[0] = element(*this);
    for (size_t i = 1; i < LOOKUP_SIZE; ++i) {
        lookup_table[i] = lookup_table[i - 1] + d2;
    }

    detail::EndoScalars endo_scalars = Fr::split_into_endomorphism_scalars(converted_scalar);
    detail::EndomorphismWnaf<element, NUM_ROUNDS> wnaf{ endo_scalars };
    element accumulator{ T::one_x, T::one_y, Fq::one() };
    accumulator.self_set_infinity();
    Fq beta = Fq::cube_root_of_unity();

    for (size_t i = 0; i < NUM_ROUNDS * 2; ++i) {
        uint64_t wnaf_entry = wnaf.table[i];
        uint64_t index = wnaf_entry & 0x0fffffffU;
        bool sign = static_cast<bool>((wnaf_entry >> 31) & 1);
        const bool is_odd = ((i & 1) == 1);
        auto to_add = lookup_table[static_cast<size_t>(index)];
        to_add.y.self_conditional_negate(sign ^ is_odd);
        if (is_odd) {
            to_add.x *= beta;
        }
        accumulator += to_add;

        if (i != ((2 * NUM_ROUNDS) - 1) && is_odd) {
            for (size_t j = 0; j < 4; ++j) {
                accumulator.self_dbl();
            }
        }
    }

    if (wnaf.skew) {
        accumulator += -lookup_table[0];
    }
    if (wnaf.endo_skew) {
        accumulator += element{ lookup_table[0].x * beta, lookup_table[0].y, lookup_table[0].z };
    }

    return accumulator;
}

/**
 * @brief Batch affine addition for parallel arrays: (lhs[i], rhs[i]) → rhs[i]
 * @details Uses Montgomery's batch inversion trick. lhs and rhs are separate arrays so no aliasing issues.
 *
 * @param lhs        Input array of first summands (read-only)
 * @param rhs        Input array of second summands; results are written here (rhs[i] = lhs[i] + rhs[i])
 * @param num_pairs  Number of point pairs to add
 * @param scratch_space Temporary storage for batch inversion, size >= num_pairs
 *
 * @warning ASSUMES NO EDGE CASES:
 *   - All points must be valid (not point at infinity)
 *   - lhs[i] != rhs[i] for all i (no point doubling cases)
 *   - lhs[i] != -rhs[i] for all i (no point at infinity results)
 */
template <typename AffineElement, typename Fq>
__attribute__((always_inline)) inline void batch_affine_add_impl(const AffineElement* lhs,
                                                                 AffineElement* rhs,
                                                                 const size_t num_pairs,
                                                                 Fq* scratch_space) noexcept
{
    Fq batch_inversion_accumulator = Fq::one();

    // Forward pass: prepare batch inversion
    for (size_t i = 0; i < num_pairs; ++i) {
        scratch_space[i] = lhs[i].x + rhs[i].x;
        rhs[i].x -= lhs[i].x;
        rhs[i].y -= lhs[i].y;
        rhs[i].y *= batch_inversion_accumulator;
        batch_inversion_accumulator *= rhs[i].x;
    }

    if (batch_inversion_accumulator == Fq::zero()) {
        throw_or_abort("attempted to invert zero in batch_affine_add_impl");
    }
    batch_inversion_accumulator = batch_inversion_accumulator.invert();

    // Backward pass: compute additions
    for (size_t i = num_pairs - 1; i < num_pairs; --i) {
        // lambda = (y2 - y1) / (x2 - x1)
        rhs[i].y *= batch_inversion_accumulator;
        batch_inversion_accumulator *= rhs[i].x;
        rhs[i].x = rhs[i].y.sqr();
        rhs[i].x -= scratch_space[i]; // x3 = lambda^2 - (x1 + x2)

        // y3 = lambda * (x1 - x3) - y1
        Fq temp = lhs[i].x - rhs[i].x;
        temp *= rhs[i].y;
        rhs[i].y = temp - lhs[i].y;
    }
}

/**
 * @brief Batch affine addition for interleaved arrays: pairs (points[2i], points[2i+1]) → points[num_points/2 + i]
 * @details Optimized for the pippenger interleaved memory layout where lhs and rhs live in the same contiguous array.
 *          Uses direct address arithmetic and hardcoded prefetch to avoid aliasing penalties that arise when the
 *          generic batch_affine_add_impl is called with lhs_base == rhs_base (the compiler cannot prove that writes
 *          to `output` don't alias reads from `lhs`, forcing unnecessary reloads).
 *
 * @param points     Interleaved array: [lhs0, rhs0, lhs1, rhs1, ...]. Results written to top half.
 * @param num_points Total number of points (must be even). Number of pairs = num_points / 2.
 * @param scratch_space Temporary storage for batch inversion, size >= num_points / 2.
 */
template <typename AffineElement, typename Fq>
__attribute__((always_inline)) inline void batch_affine_add_interleaved(AffineElement* points,
                                                                        const size_t num_points,
                                                                        Fq* scratch_space) noexcept
{
    Fq batch_inversion_accumulator = Fq::one();

    // Forward pass: accumulate (x2 - x1) products for batch inversion.
    // The two multiplications below both read the *current* value of
    // batch_inversion_accumulator, so they are independent and can be fused
    // into a single paired Montgomery multiply (2 SIMD lanes per kernel call).
    for (size_t i = 0; i < num_points; i += 2) {
        scratch_space[i >> 1] = points[i].x + points[i + 1].x; // x1 + x2 (saved for later)
        points[i + 1].x -= points[i].x;                        // x2 - x1
        points[i + 1].y -= points[i].y;                        // y2 - y1
        // Pair: points[i+1].y = points[i+1].y * acc;  acc = acc * points[i+1].x
        // (Both use the old acc; output aliasing is safe — the batched kernel
        //  reads all inputs into locals before writing any output.)
        Fq::template montgomery_mul_batched<2>({ &points[i + 1].y, &batch_inversion_accumulator },
                                               { &batch_inversion_accumulator, &points[i + 1].x },
                                               { &points[i + 1].y, &batch_inversion_accumulator });
    }

    if (batch_inversion_accumulator == Fq::zero()) {
        throw_or_abort("attempted to invert zero in batch_affine_add_interleaved");
    }
    batch_inversion_accumulator = batch_inversion_accumulator.invert();

    // Backward pass: complete inversions and compute additions — unrolled by 2.
    //
    // Each iteration of the single-pair form emits three kernels after the
    // inversion accumulator update:
    //   A: paired mul  (λ = (y2-y1)·acc;  acc *= (x2-x1))   — SERIAL on acc
    //   B: single sqr  (λ²)                                 — local to the iteration
    //   C: single mul  ((x1-x3)·λ)                          — local to the iteration
    //
    // A is a strict recurrence on batch_inversion_accumulator and stays the
    // critical path. B and C depend only on the iteration's local λ, so
    // across two adjacent iterations (hi and lo = hi − 2) they are
    // independent and pair:
    //   B_hi, B_lo → one sqr_paired<2>
    //   C_hi, C_lo → one mul_paired<2>
    // Per 2 iterations: 4 kernels (all paired) vs 6 before (2 paired + 4 singles).
    //
    // Tail: if num_points / 2 is odd, the remaining single pair falls back
    // to the original 3-kernel form.
    size_t i = num_points;
    while (i >= 4) {
        // Two adjacent pairs per step, high-index first (matches the
        // original loop's descending order of consumption).
        const size_t hi = i - 2; // lhs index of the higher pair
        const size_t lo = i - 4; // lhs index of the lower pair
        const size_t hi_out = (hi + num_points) >> 1;
        const size_t lo_out = (lo + num_points) >> 1;
        const size_t hi_scratch = hi >> 1;
        const size_t lo_scratch = lo >> 1;

        // A_hi — critical-path paired mul: λ_hi = (y2-y1)_hi · acc_in,  acc *= (x2-x1)_hi.
        Fq::template montgomery_mul_batched<2>({ &points[hi + 1].y, &batch_inversion_accumulator },
                                               { &batch_inversion_accumulator, &points[hi + 1].x },
                                               { &points[hi + 1].y, &batch_inversion_accumulator });

        // A_lo — critical-path paired mul: λ_lo = (y2-y1)_lo · acc_after_hi,  acc *= (x2-x1)_lo.
        // Must follow A_hi (serial on batch_inversion_accumulator).
        Fq::template montgomery_mul_batched<2>({ &points[lo + 1].y, &batch_inversion_accumulator },
                                               { &batch_inversion_accumulator, &points[lo + 1].x },
                                               { &points[lo + 1].y, &batch_inversion_accumulator });
        // Now: points[hi+1].y == λ_hi  and  points[lo+1].y == λ_lo.

        // B_pair — paired squaring of the two lambdas. Output to locals, NOT
        // to points[hi+1].x / points[lo+1].x directly, because the output
        // slot for pair `lo` (lo_out = (lo+N)/2) equals `hi` for adjacent
        // pairs — writing x3_lo to points[lo_out].x would clobber x1_hi
        // before the C_pair mul below needs it. Staging in locals makes the
        // two pairs' writes independent of each other's reads.
        Fq lambda_sq_hi;
        Fq lambda_sq_lo;
        Fq::template montgomery_sqr_batched<2>({ &points[hi + 1].y, &points[lo + 1].y },
                                               { &lambda_sq_hi, &lambda_sq_lo });

        // x3 = λ² − (x1 + x2). Subtractions into locals.
        const Fq x3_hi = lambda_sq_hi - scratch_space[hi_scratch];
        const Fq x3_lo = lambda_sq_lo - scratch_space[lo_scratch];

        // (x1 − x3) into locals — captured BEFORE any output write touches
        // points[hi].x or points[lo].x.
        const Fq x_diff_hi = points[hi].x - x3_hi;
        const Fq x_diff_lo = points[lo].x - x3_lo;

        // Prefetch the next unrolled step's working set (pairs at i−6 and i−8).
        // Matches the single-iter loop's 4-prefetches-per-iteration density.
        if (i >= 8) {
            __builtin_prefetch(points + (i - 6));
            __builtin_prefetch(points + (i - 5));
            __builtin_prefetch(points + (i - 8));
            __builtin_prefetch(points + (i - 7));
            __builtin_prefetch(points + ((i + num_points - 6) >> 1));
            __builtin_prefetch(points + ((i + num_points - 8) >> 1));
            __builtin_prefetch(scratch_space + ((i - 6) >> 1));
            __builtin_prefetch(scratch_space + ((i - 8) >> 1));
        }

        // C_pair — paired mul: {(x1−x3)_hi · λ_hi,  (x1−x3)_lo · λ_lo} into locals.
        Fq prod_hi;
        Fq prod_lo;
        Fq::template montgomery_mul_batched<2>({ &x_diff_hi, &x_diff_lo },
                                               { &points[hi + 1].y, &points[lo + 1].y },
                                               { &prod_hi, &prod_lo });

        // y3 = (x1 − x3) · λ − y1. Read lhs.y BEFORE the output .y writes
        // (for adjacent pairs, points[lo_out].y aliases points[hi].y).
        const Fq y3_hi = prod_hi - points[hi].y;
        const Fq y3_lo = prod_lo - points[lo].y;

        // All inputs captured; commit outputs. Order within each .x / .y
        // pair is irrelevant now — locals break the aliasing.
        points[hi_out].x = x3_hi;
        points[lo_out].x = x3_lo;
        points[hi_out].y = y3_hi;
        points[lo_out].y = y3_lo;

        i -= 4;
    }

    // Tail: at most one pair remains (lhs index 0). This happens iff
    // num_points / 2 was odd. No cross-iteration partner is available,
    // so fall back to the original 3-kernel single-pair form.
    if (i == 2) {
        Fq::template montgomery_mul_batched<2>({ &points[1].y, &batch_inversion_accumulator },
                                               { &batch_inversion_accumulator, &points[1].x },
                                               { &points[1].y, &batch_inversion_accumulator });
        points[1].x = points[1].y.sqr();
        points[num_points >> 1].x = points[1].x - scratch_space[0];
        points[0].x -= points[num_points >> 1].x;
        points[0].x *= points[1].y;
        points[num_points >> 1].y = points[0].x - points[0].y;
    }
}

/**
 * @brief Batch affine point doubling using Montgomery's trick
 * @tparam AffineElement Affine point type
 * @tparam Fq Base field type
 *
 * @warning ASSUMES NO EDGE CASES:
 *   - All points must be valid (not point at infinity)
 *   - points[i].y != 0 for all i (no vertical tangents)
 *   - No points with order 2 (where 2P = point at infinity)
 *
 * @note This is the "unsafe" fast path. For general point doubling with edge case handling,
 *       use Jacobian arithmetic or check for edge cases before calling this function.
 */
template <typename AffineElement, typename Fq>
__attribute__((always_inline)) inline void batch_affine_double_impl(AffineElement* points,
                                                                    const size_t num_points,
                                                                    Fq* scratch_space) noexcept
{
    Fq batch_inversion_accumulator = Fq::one();

    // Forward pass: prepare batch inversion
    for (size_t i = 0; i < num_points; ++i) {
        scratch_space[i] = points[i].x.sqr();
        scratch_space[i] = scratch_space[i] + scratch_space[i] + scratch_space[i];
        scratch_space[i] *= batch_inversion_accumulator;
        batch_inversion_accumulator *= (points[i].y + points[i].y);
    }

    if (batch_inversion_accumulator == Fq::zero()) {
        throw_or_abort("attempted to invert zero in batch_affine_double_impl");
    }
    batch_inversion_accumulator = batch_inversion_accumulator.invert();

    // Backward pass: compute doublings
    Fq temp_x;
    for (size_t i_plus_1 = num_points; i_plus_1 > 0; --i_plus_1) {
        size_t i = i_plus_1 - 1;

        scratch_space[i] *= batch_inversion_accumulator;
        batch_inversion_accumulator *= (points[i].y + points[i].y);

        temp_x = points[i].x;
        points[i].x = scratch_space[i].sqr() - (points[i].x + points[i].x);
        points[i].y = scratch_space[i] * (temp_x - points[i].x) - points[i].y;
    }
}

/**
 * @brief Pairwise affine add points in first and second group
 *
 * @param first_group Left-hand points
 * @param second_group Right-hand points
 * @param results Output array for results[i] = first_group[i] + second_group[i]
 *
 * @warning This function does NOT handle edge cases (point at infinity, point doubling, etc.).
 *          For generic point addition with edge case handling, use Jacobian coordinates instead.
 *          Only use this when you know points are in generic position (e.g., in Pippenger/MSM).
 */
template <class Fq, class Fr, class T>
void element<Fq, Fr, T>::batch_affine_add(const std::span<affine_element<Fq, Fr, T>>& first_group,
                                          const std::span<affine_element<Fq, Fr, T>>& second_group,
                                          const std::span<affine_element<Fq, Fr, T>>& results) noexcept
{
    using affine_element = affine_element<Fq, Fr, T>;
    const size_t num_points = first_group.size();
    BB_ASSERT_EQ(second_group.size(), first_group.size());

    // Space for temporary values
    std::vector<Fq> scratch_space(num_points);

    parallel_for_heuristic(
        num_points, [&](size_t i) { results[i] = first_group[i]; }, thread_heuristics::FF_COPY_COST * 2);

    // Perform batch affine addition: (lhs[i], rhs[i]) -> rhs[i]
    parallel_for_heuristic(
        num_points,
        [&](size_t start, size_t end, BB_UNUSED size_t chunk_index) {
            batch_affine_add_impl<affine_element, Fq>(
                &second_group[start], &results[start], end - start, &scratch_space[start]);
        },
        thread_heuristics::FF_ADDITION_COST * 6 + thread_heuristics::FF_MULTIPLICATION_COST * 6);
}

/**
 * @brief Multiply each point by the same scalar
 *
 * @details We use the fact that all points are being multiplied by the same scalar to batch the operations (perform
 * batch affine additions and doublings with batch inversion trick)
 *
 * @param points The span of individual points that need to be scaled
 * @param scalar The scalar we multiply all the points by
 * @return std::vector<affine_element<Fq, Fr, T>> Vector of new points where each point is exponent⋅points[i]
 */
template <class Fq, class Fr, class T>
std::vector<affine_element<Fq, Fr, T>> element<Fq, Fr, T>::batch_mul_with_endomorphism(
    const std::span<const affine_element<Fq, Fr, T>>& points, const Fr& scalar) noexcept
{
    BB_BENCH();
    using affine_element = affine_element<Fq, Fr, T>;
    const size_t num_points = points.size();

    // Space for temporary values
    std::vector<Fq> scratch_space(num_points);

    // We compute the resulting point through WNAF by evaluating (the (\sum_i (16ⁱ⋅
    // (a_i ∈ {-15,-13,-11,-9,-7,-5,-3,-1,1,3,5,7,9,11,13,15}))) - skew), where skew is 0 or 1. The result of the sum is
    // always odd and skew is used to reconstruct an even scalar. This means that to construct scalar p-1, where p is
    // the order of the scalar field, we first compute p through the sums and then subtract -1. Howver, since we are
    // computing p⋅Point, we get a point at infinity, which is an edgecase, and we don't want to handle edgecases in the
    // hot loop since the slow the computation down. So it's better to just handle it here.
    if (scalar == -Fr::one()) {
        std::vector<affine_element> results(num_points);
        parallel_for_heuristic(num_points, [&](size_t i) { results[i] = -points[i]; }, thread_heuristics::FF_COPY_COST);
        return results;
    }
    // Compute wnaf for scalar
    const Fr converted_scalar = scalar.from_montgomery_form();

    // If the scalar is zero, just set results to the point at infinity
    if (converted_scalar.is_zero()) {
        affine_element result{ Fq::zero(), Fq::zero() };
        result.self_set_infinity();
        std::vector<affine_element> results(num_points);
        parallel_for_heuristic(num_points, [&](size_t i) { results[i] = result; }, thread_heuristics::FF_COPY_COST);
        return results;
    }

    constexpr size_t LOOKUP_SIZE = 8;
    constexpr size_t NUM_ROUNDS = 32;

    detail::EndoScalars endo_scalars = Fr::split_into_endomorphism_scalars(converted_scalar);
    detail::EndomorphismWnaf<element, NUM_ROUNDS> wnaf{ endo_scalars };

    std::vector<affine_element> work_elements(num_points);
    std::array<std::vector<affine_element>, LOOKUP_SIZE> lookup_table;
    for (auto& table : lookup_table) {
        table.resize(num_points);
    }
    std::vector<affine_element> temp_point_vector(num_points);

    auto execute_range = [&](size_t start, size_t end) {
        // Perform batch affine addition in parallel
        const auto add_chunked = [&](const affine_element* lhs, affine_element* rhs) {
            batch_affine_add_impl<affine_element, Fq>(&lhs[start], &rhs[start], end - start, &scratch_space[start]);
        };

        // Perform point doubling in parallel
        const auto double_chunked = [&](affine_element* lhs) {
            batch_affine_double_impl<affine_element, Fq>(&lhs[start], end - start, &scratch_space[start]);
        };

        // Initialize first entries in lookup table
        for (size_t i = start; i < end; ++i) {
            if (points[i].is_point_at_infinity()) {
                temp_point_vector[i] = affine_element::one();
                lookup_table[0][i] = affine_element::one();
            } else {
                temp_point_vector[i] = points[i];
                lookup_table[0][i] = points[i];
            }
        }
        // Costruct lookup table
        double_chunked(&temp_point_vector[0]);
        for (size_t j = 1; j < LOOKUP_SIZE; ++j) {
            for (size_t i = start; i < end; ++i) {
                lookup_table[j][i] = lookup_table[j - 1][i];
            }
            add_chunked(&temp_point_vector[0], &lookup_table[j][0]);
        }

        constexpr Fq beta = Fq::cube_root_of_unity();
        uint64_t wnaf_entry = 0;
        uint64_t index = 0;
        bool sign = 0;
        // Prepare elements for the first batch addition
        for (size_t j = 0; j < 2; ++j) {
            wnaf_entry = wnaf.table[j];
            index = wnaf_entry & 0x0fffffffU;
            sign = static_cast<bool>((wnaf_entry >> 31) & 1);
            const bool is_odd = ((j & 1) == 1);
            for (size_t i = start; i < end; ++i) {
                auto to_add = lookup_table[static_cast<size_t>(index)][i];
                to_add.y.self_conditional_negate(sign ^ is_odd);
                if (is_odd) {
                    to_add.x *= beta;
                }
                if (j == 0) {
                    work_elements[i] = to_add;
                } else {
                    temp_point_vector[i] = to_add;
                }
            }
        }
        add_chunked(&temp_point_vector[0], &work_elements[0]);
        // Run through SM logic in wnaf form (excluding the skew)
        for (size_t j = 2; j < NUM_ROUNDS * 2; ++j) {
            wnaf_entry = wnaf.table[j];
            index = wnaf_entry & 0x0fffffffU;
            sign = static_cast<bool>((wnaf_entry >> 31) & 1);
            const bool is_odd = ((j & 1) == 1);
            if (!is_odd) {
                for (size_t k = 0; k < 4; ++k) {
                    double_chunked(&work_elements[0]);
                }
            }
            for (size_t i = start; i < end; ++i) {
                auto to_add = lookup_table[static_cast<size_t>(index)][i];
                to_add.y.self_conditional_negate(sign ^ is_odd);
                if (is_odd) {
                    to_add.x *= beta;
                }
                temp_point_vector[i] = to_add;
            }
            add_chunked(&temp_point_vector[0], &work_elements[0]);
        }
        // Apply skew for the first endo scalar
        // Use affine_element::operator+ (via Jacobian) to handle edge cases related to the point at infinity.
        if (wnaf.skew) {
            for (size_t i = start; i < end; ++i) {
                work_elements[i] = work_elements[i] + (-lookup_table[0][i]);
            }
        }
        // Apply skew for the second endo scalar
        if (wnaf.endo_skew) {
            for (size_t i = start; i < end; ++i) {
                affine_element endo_point = lookup_table[0][i];
                endo_point.x *= beta;
                work_elements[i] = work_elements[i] + endo_point;
            }
        }
        // Handle points at infinity explicitly
        for (size_t i = start; i < end; ++i) {
            work_elements[i] = points[i].is_point_at_infinity() ? work_elements[i].set_infinity() : work_elements[i];
        }
    };
    parallel_for_range(num_points, execute_range);

    return work_elements;
}

template <typename Fq, typename Fr, typename T>
void element<Fq, Fr, T>::batch_normalize(element* elements, const size_t num_elements) noexcept
{
    std::vector<Fq> temporaries;
    temporaries.reserve(num_elements * 2);
    Fq accumulator = Fq::one();

    // Iterate over the points, computing the product of their z-coordinates.
    // At each iteration, store the currently-accumulated z-coordinate in `temporaries`
    for (size_t i = 0; i < num_elements; ++i) {
        temporaries.emplace_back(accumulator);
        if (!elements[i].is_point_at_infinity()) {
            accumulator *= elements[i].z;
        }
    }
    // For the rest of this method we refer to the product of all z-coordinates as the 'global' z-coordinate
    // Invert the global z-coordinate and store in `accumulator`
    accumulator = accumulator.invert();

    /**
     * We now proceed to iterate back down the array of points.
     * At each iteration we update the accumulator to contain the z-coordinate of the currently worked-upon
     *z-coordinate. We can then multiply this accumulator with `temporaries`, to get a scalar that is equal to the
     *inverse of the z-coordinate of the point at the next iteration cycle e.g. Imagine we have 4 points, such that:
     *
     * accumulator = 1 / z.data[0]*z.data[1]*z.data[2]*z.data[3]
     * temporaries[3] = z.data[0]*z.data[1]*z.data[2]
     * temporaries[2] = z.data[0]*z.data[1]
     * temporaries[1] = z.data[0]
     * temporaries[0] = 1
     *
     * At the first iteration, accumulator * temporaries[3] = z.data[0]*z.data[1]*z.data[2] /
     *z.data[0]*z.data[1]*z.data[2]*z.data[3]  = (1 / z.data[3]) We then update accumulator, such that:
     *
     * accumulator = accumulator * z.data[3] = 1 / z.data[0]*z.data[1]*z.data[2]
     *
     * At the second iteration, accumulator * temporaries[2] = z.data[0]*z.data[1] / z.data[0]*z.data[1]*z.data[2] =
     *(1 z.data[2]) And so on, until we have computed every z-inverse!
     *
     * We can then convert out of Jacobian form (x = X / Z^2, y = Y / Z^3) with 4 muls and 1 square.
     **/
    for (size_t i = num_elements - 1; i < num_elements; --i) {
        if (!elements[i].is_point_at_infinity()) {
            Fq z_inv = accumulator * temporaries[i];
            Fq zz_inv = z_inv.sqr();
            elements[i].x *= zz_inv;
            elements[i].y *= (zz_inv * z_inv);
            accumulator *= elements[i].z;
        }
        elements[i].z = Fq::one();
    }
}

template <typename Fq, typename Fr, typename T>
template <typename>
element<Fq, Fr, T> element<Fq, Fr, T>::random_coordinates_on_curve(numeric::RNG* engine) noexcept
{
    bool found_one = false;
    Fq yy;
    Fq x;
    Fq y;
    while (!found_one) {
        x = Fq::random_element(engine);
        yy = x.sqr() * x + T::b;
        if constexpr (T::has_a) {
            yy += (x * T::a);
        }
        auto [found_root, y1] = yy.sqrt();
        y = y1;
        found_one = found_root;
    }
    return { x, y, Fq::one() };
}

} // namespace bb::group_elements
// NOLINTEND(readability-implicit-bool-conversion, cppcoreguidelines-avoid-c-arrays)
