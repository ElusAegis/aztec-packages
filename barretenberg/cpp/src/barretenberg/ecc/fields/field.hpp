// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Raju], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

/**
 * @brief Include order of header-only field class is structured to ensure linter/language server can resolve paths.
 *        Declarations are defined in "field_declarations.hpp", definitions in "field_impl.hpp" (which includes
 *        declarations header) Specialized definitions are in "field_impl_generic.hpp" and
 *        "platform_variants/x86_asm.hpp" (which include "field_impl.hpp")
 */
#include "./field_impl_generic.hpp"
#include "./platform_variants/x86_asm.hpp"
