// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Raju], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

/**
 * @brief Include order of header-only field class is structured to ensure linter/language server can resolve paths.
 *        Declarations are defined in "field_declarations.hpp"; operators and shared
 *        helpers in "field_impl.hpp"; constexpr addc/sbb/add/sub/reduce and Montgomery
 *        dispatch in "field_impl_generic.hpp" (which includes backends/mont_backend.hpp
 *        for platform-specific Montgomery multiplication); x86 inline-asm add/sub/reduce/
 *        negate in "x86_asm_arith.hpp".
 */
#include "./field_impl_generic.hpp"
#include "./x86_asm_arith.hpp"
