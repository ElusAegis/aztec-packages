#pragma once

// Compile-time Montgomery backend selector.
//
// Picks one of the 4 concrete backends based on platform and build flags.
// field<T> uses detail::MontBackend<Params> as a blind delegate — it doesn't
// know or care which platform it's on. Add a new backend? Add one line here.
//
// Selection precedence:
//   1. x86 BMI2 asm   — host + BMI2 available + small modulus
//   2. native __int128 — host + any other case that wants the CIOS path
//   3. WASM FMA SIMD   — WASM with MONTMUL_VARIANT_FMA defined
//   4. WASM int29     — WASM fallback (9 × 29-bit schoolbook)

#include "../field_declarations.hpp" // provides BBERG_NO_ASM

// Include all backend headers — each self-guards via preprocessor so only
// the applicable ones define their struct.
#include "native_int128.hpp"
#include "wasm_fma_simd.hpp"
#include "wasm_int29.hpp"
#include "x86_asm.hpp"

namespace bb::detail {

// Note: the field<T> dispatch methods already guard the asm path with
// `use_generic_arithmetic` — so on hosts where BBERG_NO_ASM == 0 but the
// modulus is large (>= 2^254) or tiny (<= 64 bits), the `if constexpr`
// check there bypasses X86AsmBackend and calls into NativeBackend directly
// via the constexpr fallback in X86AsmBackend::mul (which delegates).
//
// For the simple selector here we just pick the "preferred" runtime
// backend; per-modulus opt-out is the caller's job.

template <class Params>
using MontBackend =
#if BBERG_NO_ASM == 0
    X86AsmBackend<Params>;
#elif defined(__SIZEOF_INT128__) && !defined(__wasm__)
    NativeBackend<Params>;
#elif defined(MONTMUL_VARIANT_FMA)
    WasmFmaBackend<Params>;
#else
    WasmInt29Backend<Params>;
#endif

} // namespace bb::detail
