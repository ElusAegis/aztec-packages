#pragma once

// x86 BMI2 assembly Montgomery backend.
//
// Provides inline-asm montmul/square (small modulus). Delegates everything
// else to NativeBackend:
//   - mul_big   → NativeBackend::mul_big (no dedicated asm big-mul kernel)
//   - wide_mul  → NativeBackend::wide_mul
//   - constexpr → NativeBackend::{mul,sqr} (asm isn't constexpr-evaluable)
//
// Active only on hosts with BMI2 (BBERG_NO_ASM == 0). If asm is disabled
// (DISABLE_ASM, no __BMI2__, large modulus, or tiny modulus), the selector
// falls through to NativeBackend directly.

#if (BBERG_NO_ASM == 0)

#include "../asm_macros.hpp"
#include "../field_declarations.hpp"
#include "native_int128.hpp"

namespace bb::detail {

template <class Params> struct X86AsmBackend {
    // ── Public MontBackend contract ──────────────────────────────────────

    BB_INLINE static constexpr field<Params> mul(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        if (std::is_constant_evaluated()) {
            return NativeBackend<Params>::mul(lhs, rhs);
        }
        field<Params> result = asm_mul(lhs, rhs);
        result.assert_coarse_form();
        return result;
    }

    BB_INLINE static constexpr field<Params> sqr(const field<Params>& x) noexcept
    {
        if (std::is_constant_evaluated()) {
            return NativeBackend<Params>::sqr(x);
        }
        field<Params> result = asm_sqr(x);
        result.assert_coarse_form();
        return result;
    }

    BB_INLINE static constexpr field<Params> mul_big(const field<Params>& lhs, const field<Params>& rhs) noexcept
    {
        // No dedicated asm big-mul kernel; delegate to native CIOS (fully
        // constexpr and performant for both runtime and constexpr paths).
        return NativeBackend<Params>::mul_big(lhs, rhs);
    }

    BB_INLINE static constexpr typename field<Params>::wide_array wide_mul(const field<Params>& lhs,
                                                                           const field<Params>& rhs) noexcept
    {
        return NativeBackend<Params>::wide_mul(lhs, rhs);
    }

    BB_INLINE static constexpr void mul_paired(const field<Params>& a1,
                                               const field<Params>& b1,
                                               const field<Params>& a2,
                                               const field<Params>& b2,
                                               field<Params>& out1,
                                               field<Params>& out2) noexcept
    {
        out1 = mul(a1, b1);
        out2 = mul(a2, b2);
    }

    BB_INLINE static constexpr void sqr_paired(const field<Params>& a1,
                                               const field<Params>& a2,
                                               field<Params>& out1,
                                               field<Params>& out2) noexcept
    {
        out1 = sqr(a1);
        out2 = sqr(a2);
    }

  private:
    // ── Inline asm montmul (BMI2 ADX when __ADX__, else plain MUL) ───────

    BB_INLINE static field<Params> asm_mul(const field<Params>& a, const field<Params>& b) noexcept
    {
        field<Params> r;
        constexpr uint64_t r_inv = Params::r_inv;
        constexpr uint64_t modulus_0 = field<Params>::modulus.data[0];
        constexpr uint64_t modulus_1 = field<Params>::modulus.data[1];
        constexpr uint64_t modulus_2 = field<Params>::modulus.data[2];
        constexpr uint64_t modulus_3 = field<Params>::modulus.data[3];
        constexpr uint64_t zero_ref = 0;

        __asm__(MUL("0(%0)", "8(%0)", "16(%0)", "24(%0)", "%1")
                    STORE_FIELD_ELEMENT("%2", "%%r12", "%%r13", "%%r14", "%%r15")
                :
                : "%r"(&a),
                  "%r"(&b),
                  "r"(&r),
                  [modulus_0] "m"(modulus_0),
                  [modulus_1] "m"(modulus_1),
                  [modulus_2] "m"(modulus_2),
                  [modulus_3] "m"(modulus_3),
                  [r_inv] "m"(r_inv),
                  [zero_reference] "m"(zero_ref)
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "cc", "memory");
        return r;
    }

    BB_INLINE static field<Params> asm_sqr(const field<Params>& a) noexcept
    {
        field<Params> r;
        constexpr uint64_t r_inv = Params::r_inv;
        constexpr uint64_t modulus_0 = field<Params>::modulus.data[0];
        constexpr uint64_t modulus_1 = field<Params>::modulus.data[1];
        constexpr uint64_t modulus_2 = field<Params>::modulus.data[2];
        constexpr uint64_t modulus_3 = field<Params>::modulus.data[3];
        constexpr uint64_t zero_ref = 0;

        // Our SQR implementation with BMI2 but without ADX has a bug. The
        // case is extremely rare so we use MUL instead (same algorithm,
        // slightly slower).
#if !defined(__ADX__) || defined(DISABLE_ADX)
        __asm__(MUL("0(%0)", "8(%0)", "16(%0)", "24(%0)", "%1")
                    STORE_FIELD_ELEMENT("%2", "%%r12", "%%r13", "%%r14", "%%r15")
                :
                : "%r"(&a),
                  "%r"(&a),
                  "r"(&r),
                  [modulus_0] "m"(modulus_0),
                  [modulus_1] "m"(modulus_1),
                  [modulus_2] "m"(modulus_2),
                  [modulus_3] "m"(modulus_3),
                  [r_inv] "m"(r_inv),
                  [zero_reference] "m"(zero_ref)
                : "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "cc", "memory");
#else
        __asm__(SQR("%0") STORE_FIELD_ELEMENT("%1", "%%r12", "%%r13", "%%r14", "%%r15")
                :
                : "r"(&a),
                  "r"(&r),
                  [zero_reference] "m"(zero_ref),
                  [modulus_0] "m"(modulus_0),
                  [modulus_1] "m"(modulus_1),
                  [modulus_2] "m"(modulus_2),
                  [modulus_3] "m"(modulus_3),
                  [r_inv] "m"(r_inv)
                : "%rcx", "%rdx", "%rdi", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "cc", "memory");
#endif
        return r;
    }
};

} // namespace bb::detail

#endif // BBERG_NO_ASM == 0
