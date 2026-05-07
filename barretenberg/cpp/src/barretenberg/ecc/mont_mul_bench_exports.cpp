#include "barretenberg/common/wasm_export.hpp"
#include "barretenberg/ecc/curves/bn254/fq.hpp"
#include "barretenberg/ecc/curves/bn254/fr.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"

#include <cstdint>

namespace {

template <typename Field> void init_mont_mul_inputs(Field* lhs0, Field* rhs0, Field* lhs1, Field* rhs1) noexcept
{
    *lhs0 = Field(bb::numeric::uint256_t{ 0x123456789abcdef0ULL,
                                          0x0fedcba987654321ULL,
                                          0x1111222233334444ULL,
                                          0x0000000000000001ULL });
    *rhs0 = Field(bb::numeric::uint256_t{ 0x8877665544332211ULL,
                                          0xf0e1d2c3b4a59687ULL,
                                          0x5555666677778888ULL,
                                          0x0000000000000002ULL });
    *lhs1 = Field(bb::numeric::uint256_t{ 0x13579bdf2468ace0ULL,
                                          0x02468ace13579bdfULL,
                                          0x9999aaaabbbbccccULL,
                                          0x0000000000000003ULL });
    *rhs1 = Field(bb::numeric::uint256_t{ 0xdecafbadfeedbeefULL,
                                          0x1122334455667788ULL,
                                          0xddddeeeeffff0001ULL,
                                          0x0000000000000004ULL });
}

template <typename Field> void single_mont_mul(const Field* lhs, const Field* rhs, Field* out) noexcept
{
    *out = *lhs * *rhs;
}

template <typename Field>
void paired_mont_mul(const Field* lhs0,
                     const Field* rhs0,
                     const Field* lhs1,
                     const Field* rhs1,
                     Field* out0,
                     Field* out1) noexcept
{
    const auto [product0, product1] = Field::paired_mul(*lhs0, *rhs0, *lhs1, *rhs1);
    *out0 = product0;
    *out1 = product1;
}

} // namespace

static_assert(sizeof(bb::fr) == 32);
static_assert(sizeof(bb::fq) == 32);

WASM_EXPORT uint32_t wasm_bench_field_element_size_bytes()
{
    return sizeof(bb::fr);
}

WASM_EXPORT void wasm_bench_bn254_fr_init_mont_mul_inputs(bb::fr* lhs0, bb::fr* rhs0, bb::fr* lhs1, bb::fr* rhs1)
{
    init_mont_mul_inputs(lhs0, rhs0, lhs1, rhs1);
}

WASM_EXPORT void wasm_bench_bn254_fr_mont_mul(const bb::fr* lhs, const bb::fr* rhs, bb::fr* out)
{
    single_mont_mul(lhs, rhs, out);
}

WASM_EXPORT void wasm_bench_bn254_fq_init_mont_mul_inputs(bb::fq* lhs0, bb::fq* rhs0, bb::fq* lhs1, bb::fq* rhs1)
{
    init_mont_mul_inputs(lhs0, rhs0, lhs1, rhs1);
}

WASM_EXPORT void wasm_bench_bn254_fq_mont_mul(const bb::fq* lhs, const bb::fq* rhs, bb::fq* out)
{
    single_mont_mul(lhs, rhs, out);
}

#if defined(__wasm__) && defined(__wasm_relaxed_simd__)
WASM_EXPORT void wasm_bench_bn254_fr_paired_mont_mul(const bb::fr* lhs0,
                                                     const bb::fr* rhs0,
                                                     const bb::fr* lhs1,
                                                     const bb::fr* rhs1,
                                                     bb::fr* out0,
                                                     bb::fr* out1)
{
    paired_mont_mul(lhs0, rhs0, lhs1, rhs1, out0, out1);
}

WASM_EXPORT void wasm_bench_bn254_fq_paired_mont_mul(const bb::fq* lhs0,
                                                     const bb::fq* rhs0,
                                                     const bb::fq* lhs1,
                                                     const bb::fq* rhs1,
                                                     bb::fq* out0,
                                                     bb::fq* out1)
{
    paired_mont_mul(lhs0, rhs0, lhs1, rhs1, out0, out1);
}
#endif
