#pragma once
#include <cstdint>

namespace bb {

inline constexpr unsigned FIELD_BITS = 256;

// R_LIMB_BITS: the single hardware-motivated input that determines the Montgomery representation.
// 64 — Native: 64x64->128 widening mul (MUL+UMULH)
// 29 — WASM int: 29x29=58 bits fits i64 with accumulation headroom
// 24 — WASM FMA: 24x24=48 bits fits f64 mantissa (53 bits)
//
// Exposed as a macro (BB_R_LIMB_BITS) so preprocessor `#if` directives in
// platform-dispatch code can switch on it. The constexpr below mirrors the
// macro for use in C++ expressions.
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
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
// WASM 29: R_LIMB_BITS=29 -> R_NUM_LIMBS=9  -> R_EXPONENT=261
// FMA 24:  R_LIMB_BITS=24 -> R_NUM_LIMBS=11 -> R_EXPONENT=264

} // namespace bb
