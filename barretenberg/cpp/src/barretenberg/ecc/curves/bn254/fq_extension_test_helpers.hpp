#pragma once

#include "barretenberg/ecc/fields/field_montgomery_config.hpp"
#include <gtest/gtest.h>

// Used in fq2/fq6/fq12 *CheckAgainstConstants tests.
// TODO(fma-24bit-constants): the #else branch of those tests pins Montgomery
// limbs to R=2^261 (WASM int29). FMA SIMD 24-bit uses R=2^264 — limbs differ
// even for a correct kernel. Remove these skips once the #else gets a third
// branch (or the tests are rewritten representation-agnostically).
#define SKIP_IF_FMA_24BIT()                                                                                            \
    do {                                                                                                               \
        if (bb::R_LIMB_BITS == 24) {                                                                                   \
            GTEST_SKIP() << "TODO: hardcoded Montgomery limbs are R=2^261; FMA 24-bit uses R=2^264";                   \
        }                                                                                                              \
    } while (0)
