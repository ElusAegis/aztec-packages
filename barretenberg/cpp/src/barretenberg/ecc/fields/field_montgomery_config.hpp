#pragma once
#include <cstdint>

namespace bb {

inline constexpr unsigned FIELD_BITS = 256;

// R_LIMB_BITS: the single hardware-motivated input that determines the Montgomery representation.
// 64 — Native: 64x64->128 widening mul (MUL+UMULH); also WASM RNE (external layout)
// 29 — WASM int: 29x29=58 bits fits i64 with accumulation headroom
// 24 — WASM FMA: 24x24=48 bits fits f64 mantissa (53 bits)
//
// Exposed as a macro (BB_R_LIMB_BITS) so preprocessor `#if` directives in
// platform-dispatch code can switch on it. The constexpr below mirrors the
// macro for use in C++ expressions.
//
// Note: the RNE WASM backend keeps the 4×64-bit storage layout (R=2^256,
// matching native) but multiplies via 5×51-bit f64 FMA internally. Its
// gate is placed *before* the native int128 branch so WASM builds —
// where clang still defines __SIZEOF_INT128__ — pick RNE over int128.
#if defined(MONTMUL_VARIANT_RNE) && defined(__wasm__)
#define BB_R_LIMB_BITS 64
#elif defined(__SIZEOF_INT128__) && !defined(__wasm__)
#define BB_R_LIMB_BITS 64
#elif defined(MONTMUL_VARIANT_FMA)
#define BB_R_LIMB_BITS 24
#else
#define BB_R_LIMB_BITS 29
#endif

inline constexpr unsigned R_LIMB_BITS = BB_R_LIMB_BITS;

inline constexpr unsigned R_NUM_LIMBS = (FIELD_BITS + R_LIMB_BITS - 1) / R_LIMB_BITS;
inline constexpr unsigned R_EXPONENT = R_LIMB_BITS * R_NUM_LIMBS;

static_assert(R_NUM_LIMBS * R_LIMB_BITS >= FIELD_BITS);
// Platform verification:
// Native:  R_LIMB_BITS=64 -> R_NUM_LIMBS=4  -> R_EXPONENT=256
// WASM RNE:R_LIMB_BITS=64 -> R_NUM_LIMBS=4  -> R_EXPONENT=256 (external layout; 5×51-bit internal)
// WASM 29: R_LIMB_BITS=29 -> R_NUM_LIMBS=9  -> R_EXPONENT=261
// FMA 24:  R_LIMB_BITS=24 -> R_NUM_LIMBS=11 -> R_EXPONENT=264

} // namespace bb
