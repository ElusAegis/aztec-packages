Prime field documentation    {#field_docs}
===
Barretenberg has its own implementation of finite field arithmetic. The implementation targets 254-bit (bn254, grumpkin) and 256-bit (secp256k1, secp256r1) fields. Internally the field is represented as a little-endian C-array of 4 uint64_t limbs. For 254-bit fields, the internal representation must be in the range $[0, 2p)$ (which we refer to as the _coarse representation_), while for 256-bit fields the internal representation is an arbitrary `uint256_t`.

## Field arithmetic
### Introduction to Montgomery form {#field_docs_montgomery_explainer}
We use Montgomery multiplication to speed up field multiplication. For an original element  $ a \in \mathbb F_p$ the element is represented internally as $$ a⋅R\ mod\ p$$ where $R = 2^d\ mod\ p$ and $d=64⋅4=256$ on every backend (x86_64, generic 64-bit, and WASM). On WASM we still expand to 9 × 29-bit limbs internally during a multiplication, but the canonical 4 × 64-bit Montgomery form on input/output is the same as on x86_64. Consequently, Montgomery-form constants (`r_squared`, `cube_root`, `coset_generator`) are single shared values across all builds.

The goal of using Montgomery form is to avoid heavy division modulo $p$. To compute a representative of element $$c = a⋅b\ mod\ p$$ we compute $$c⋅R = (a⋅R)⋅(b⋅R) / R\ mod\ p,$$ but we use an efficient division trick to avoid the naive modular division. Let's look into the standard 4⋅64 case:
1. First, we compute the value $$c_r=c⋅R⋅R = aR⋅bR$$ in integers and get a value with 8 64-bit limbs
2. Then we take the lowest limb of $c_r$ (i.e., $c_r[0]$) and multiply it by a special _precomputed_ value $$r_{inv} = -1 ⋅ p^{-1}\ mod\  2^{64}$$ As a result we get $$k = r_{inv}⋅ c_r[0]\ mod\ 2^{64}$$
3. Next we update $c_r$ in integers by adding $k⋅p$: $$c_r += k⋅p$$ You might notice that the value of $c_r\ mod\ p$ hasn't changed, since we've added a multiple of the modulus. At the same time, if we look at the expression modulo $2^{64}$: $$c_r + k⋅p = c_r + c_r⋅r_{inv}⋅p = c_r + c_r⋅ (-1)⋅p^{-1}⋅p = c_r - c_r = 0\ mod\ 2^{64}.$$ The result is equivalent modulo $p$, but we zeroed out the lowest limb
4. We perform the same operation for $c_r[1]$, but instead of adding $k⋅p$, we add $2^{64}⋅k⋅p$. In the implementation, instead of adding $k⋅ p$ to limbs of $c_r$ starting with zero, we just start with limb 1. This ensures that $c_r[1]=0$. We then perform the same operation for 2 more limbs.
5. At this stage the array $c_r$ has the property that the first 4 limbs of the total 8 limbs are zero. So if we treat the 4 high limbs as a separate integer $c_{r.high}$, $$c_r = c_{r.high}⋅2^{256}=c_{r.high}⋅R\ mod\ p \Rightarrow c_{r.high} = c\cdot R\ mod\ p$$ and we can get the evaluation simply by taking the 4 high limbs of $c_r$.
6. The previous step has reduced the intermediate value of $cR$ to range $[0,2p)$, so we must check if it is more than $p$ and subtract the modulus once if it overflows.

On a high level, what we are doing is iteratively adding a multiple of $p$ until the current bottom limb is zero, then shifting by a limb (amounting to dividing by $2^{64}$).
#### Bounds analysis
Why does this work? We present several versions of the analysis, for completeness.

* Suppose both $aR$ and $bR$ are less than the modulus $p$ in integers, so $$aR\cdot bR <= (p-1)^2.$$ During each of the $k\cdot p$ addition rounds we can add at most $(2^{64}-1)p$ to the corresponding digits, so at most we add $(2^{256}-1)p$ and the total is $$aR\cdot bR + k_{0,1,2,3}p \le (p-1)^2+(2^{256}-1)p < 2\cdot 2^{256}p \Rightarrow c_{r.high} = \frac{aR\cdot bR + k_{0,1,2,3}p}{2^{256}} < 2p.$$

* For our 256-bit fields, we _cannot_ assume that $aR$ and $bR$ are less than the modulus $p$; we simply know that they are 256-bit numbers. Nonetheless, the same analysis shows that the output is less than $2^{256} + p -1$. This means that (conditionally) subtracting one copy of $p$ is enough to get us to the valid range of $[0, 2^{256})$.

* For 254-bit fields (e.g. the BN-254 base and scalar fields) we can do even better by employing a simple trick. Note that 4 64-bit limbs allow 256 bits of storage. We relax the internal representation to use values in range $[0,2p)$. The addition, negation and subtraction operation logic doesn't change, we simply replace the modulus $p$ with $2p$, but the multiplication becomes more efficient. The multiplicands are in range $[0,2p)$, but we add multiples of modulus $p$ to reduce limbs, not $2p$. If we revisit the $c_r$ formula:
$$aR\cdot bR + k_{0,1,2,3}p \le (2p-1)^2+(2^{256}-1)p = 2^{256}p+4p^2-5p+1 \Rightarrow$$ $$\Rightarrow c_{r.high} = \frac{aR\cdot bR + k_{0,1,2,3}p}{2^{256}} \le \frac{2^{256}p+4p^2-5p+1}{2^{256}}=p +\frac{4p^2 - 5p +1}{2^{256}}, 4p < 2^{256} \Rightarrow$$ $$\Rightarrow p +\frac{4p^2 - 5p +1}{2^{256}} < 2p$$ So we ended in the same range and we don't have to perform additional reductions.

**N.B.** In the code we refer to this form, when the limbs are only constrained to be in the range $[0,2p)$, as the coarse-representation.

### Yuval reduction
For our 254-bit multiplication in WASM, we use a reduction technique found by Yuval. For a reference, please see this [hackmd](https://hackmd.io/@Ingonyama/Barret-Montgomery).

Recall that in standard Montgomery reduction, we zero out the lowest limb by adding a carefully chosen multiple of the modulus $p$. In particular, if we were to use standard Montgomery reduction given our limb-decomposition for WASM: given an accumulator $x = \sum_{i=0}^{n} \text{result}_i \cdot 2^{29i}$, we compute $k = \text{result}_0 \cdot (-p^{-1}) \mod 2^{29}$ and add $k \cdot p$ to $x$. This makes the lowest 29 bits zero (since $\text{result}_0 + k \cdot p_0 \equiv 0 \mod 2^{29}$), allowing us to "shift right" by discarding the zeroed limb.

Yuval's method takes a different approach. Instead of adding a multiple of $p$ to zero out the low bits, we directly compute the equivalent value after the divide by $2^{29}$ step. Given the same accumulator $x$, we want to find $x / 2^{29} \mod p$. We can rewrite this as:
$$x / 2^{29} = (x - \text{result}_0) / 2^{29} + \text{result}_0 / 2^{29} \mod p.$$

The first term $(x - \text{result}_0) / 2^{29}$ is simply the higher limbs shifted down. The second term requires computing $\text{result}_0 \cdot 2^{-29} \mod p$, which we precompute as `r_inv_wasm` (stored in 9 limbs).

So instead of computing $k = \text{result}_0 \cdot (-p^{-1})$ and adding $k \cdot p$ (9 multiply-accumulates), we compute $\text{result}_0 \cdot r\_inv\_wasm$ and add it to the higher limbs (also 9 multiply-accumulates). The key insight is that both approaches require the same number of operations, but Yuval's method avoids the need for a separate "zero out and shift" step—the shift is implicit in how we interpret the result.

In code, `wasm_reduce_yuval` implements this as:
```cpp
result_1 += result_0_masked * wasm_r_inv[0] + (result_0 >> 29);
result_2 += result_0_masked * wasm_r_inv[1];
// ... and so on for result_3 through result_9
```

The term `(result_0 >> 29)` handles any overflow bits in `result_0` beyond the lowest 29 bits, propagating them to `result_1`. After this operation, `result_0` is effectively discarded, and `result_1` through `result_9` hold the Montgomery-reduced value.

#### Structure of WASM Montgomery multiplication

In 254-bit WASM multiplication the cumulative shift across all reductions must equal $R = 2^{256}$ (not $2^{261}$), since the canonical output form is the same 4 × 64-bit Montgomery layout used by x86_64. We achieve this with 9 limb-reductions whose widths sum to 256: $256 = 7 \cdot 29 + 29 + 24$. **We apply Yuval's method for the first 7 reductions, a standard Montgomery 29-bit reduction for the 8th, and a special 24-bit Montgomery reduction (`wasm_reduce_24`) for the 9th (final) step.** The schoolbook multiplication that produces the 17-limb intermediate is a Karatsuba 5+4 split (`wasm_karatsuba_mul`) that costs 66 multiplications instead of the naïve 81.

Why not use Yuval for all 8 of the 29-bit reductions? Yuval's per-step slack on the running bound is $2^{29} \cdot p$ (vs. Montgomery's $p$). Eight Yuvals would push the high limb to $\approx 2^{283}$, and the final $/2^{24}$ step would land at $\approx 64p$ — outside the coarse range $[0, 2p)$. Replacing the eighth Yuval with a Montgomery 29-bit step tightens that bound back to $\approx p$, so after the final 24-bit reduction the result is in $[0, 2p)$ with no conditional subtraction.
#### Bounds analysis

We must verify that the output is in $[0, 2p)$ (the coarse representation) without requiring an additional subtraction of $p$.

After the Karatsuba multiplication, we have $aR \cdot bR$ stored across 17 limbs. Since both $aR$ and $bR$ are in $[0, 2p)$, this product is at most $4p^2$.

After 7 Yuval reductions, 1 standard 29-bit Montgomery reduction, and 1 standard 24-bit Montgomery reduction, we have computed:
$$\frac{aR \cdot bR + k_0 \cdot r_{inv} + k_1 \cdot r_{inv} + \cdots + k_6 \cdot r_{inv} + k_7 \cdot p + k_8 \cdot p}{2^{256}}$$

where each $k_i$ is the masked low limb at reduction step $i$. By construction:
- $k_0 < 2^{29}$
- $k_1 < 2^{58}$ (since it includes carries from the previous step)
- For $i \le 7$ we have $k_i < 2^{29(i+1)}$
- The Yuval sum $\sum_{i=0}^{6} k_i < 2^{203}$ (geometric series)
- $k_7 < 2^{232}$ (Montgomery 29-bit)
- $k_8 < 2^{256} - 2^{232}$ (Montgomery 24-bit, at limb position $7 \cdot 29 = 203$, masked to 24 bits)

Since $r_{inv} = 2^{-29} \mod p < p$, the total added via Yuval reductions is bounded by $(2^{203} - 1) \cdot p$. The two standard reductions together add at most $(2^{256} - 2^{203}) \cdot p$.

Therefore, the numerator is bounded by:
$$4p^2 + (2^{203} - 1) \cdot p + (2^{256} - 2^{203}) \cdot p < 4p^2 + 2^{256} \cdot p$$

Dividing by $2^{256}$ (the cumulative reduction width $7 \cdot 29 + 29 + 24$):
$$\frac{4p^2 + 2^{256} \cdot p}{2^{256}} = p + \frac{4p^2}{2^{256}}$$

For 254-bit primes, $p < 2^{254}$, so $4p < 2^{256}$ and hence:
$$\frac{4p^2}{2^{256}} = \frac{(4p) \cdot p}{2^{256}} < p$$

Thus the result is less than $2p$, which is in the coarse representation range $[0, 2p)$. No additional reduction is required.

### Paired Montgomery multiplication {#field_docs_paired_explainer}

On WASM targets that enable relaxed-SIMD (`__wasm_relaxed_simd__`), we additionally expose a *paired* kernel that computes two independent Montgomery products $a \cdot b$ and $c \cdot d$ in a single pass by riding the two `f64x2` SIMD lanes:
```cpp
auto [o1, o2] = field::paired_mul(a, b, c, d);                         // o1 = a*b, o2 = c*d (Montgomery)
auto [o1, o2] = field::paired_sqr(a, b);                               // o1 = a^2, o2 = b^2 (Montgomery)
auto [m1, m2] = field::paired_to_montgomery_form(a, b);                // m1, m2 in Montgomery form
auto [r1, r2] = field::paired_from_montgomery_form_reduced(a, b);      // r1, r2 in canonical [0, p)
```
On non-relaxed-SIMD builds, or when the modulus is 256-bit (secp curves), these dispatch to two ordinary single-lane multiplications. The paired kernel is restricted to small (<254-bit) moduli because its internal limb shape does not have headroom for the looser 256-bit-modulus arithmetic.

Internally the paired kernel uses a different limb decomposition from the standard WASM kernel: **5 × 51-bit limbs** (a 255-bit form, denoted `u51` in code) rather than 9 × 29-bit. This is forced by `f64x2` FMA: the relaxed-FMA result has a 53-bit mantissa, so 51-bit operands give a 102-bit product with one bit of headroom, which is the sweet spot. The high/low halves of each 51 × 51 product are extracted using two carefully chosen IEEE-754 bias constants (`C1 = 2^103`, `C2 = C1 + 2^52 + 2^51`) — a round-to-nearest-even FMA trick that splits the mantissa with no integer multiplication. The Montgomery target is still $R = 2^{256}$: the 5 × 51-bit reduced result is converted back to canonical 4 × 64-bit limbs via `u255_to_u256_shr_1` (the reduced value is in $[0, 2p) \subset [0, 2^{255})$ so the shift is exact).

#### Threshold and setup

The kernel's 5 × 51-bit layout holds values up to $2^{255}$. Coarse-form inputs are bounded by $2p$, so the kernel applies precisely when $2p < 2^{255}$, i.e. $p < 2^{254}$. Larger moduli (the secp curves) fall back to single-lane `montgomery_mul` and `montgomery_sqr`.

Before reduction, `paired_mul` (resp. `paired_sqr`) packs each input into 5 × 51-bit lane-paired form (lane 0 carries the first product, lane 1 the second), converts each limb to f64 via `i2f_v128`, runs a 5 × 5 schoolbook (`paired_mul`) or its triangular off-diagonal-doubled equivalent (`paired_sqr`), and writes the 10-limb column accumulator. Both shapes produce the same per-column $p_{\text{lo}}$ / $p_{\text{hi}}$ histogram, encoded once in `LO_BIAS_COUNTS` and `HI_BIAS_COUNTS`; `make_initial` seeds each column with the negation of that bias so the IEEE-754 anchor cancels exactly when the column finishes summing. After the schoolbook, the per-column integer sum $t_{\text{in}} = \sum_{k=0}^{9} t_k \cdot \beta^k$ satisfies, in each lane,
$$t_{\text{in}} < (2p)^2 = 4 p^2, \quad \beta = 2^{51},$$
and the kernel target is $t_{\text{in}} \cdot \beta^{-5} \bmod p$, i.e. the kernel's internal Montgomery factor is $R_{\text{kernel}} = \beta^5 = 2^{255}$.

#### Reduction pipeline

`reduce_and_finalize_paired_rne` reduces $t_{\text{in}}$ to outer Montgomery form in five phases:

**Phase 1 — signed carry propagation through $t_0, t_1, t_2, t_3$.** Phase 2 will mask each of $t_0$, $t_1$, $t_2$ to 51 bits when feeding the rho folds, so any high bits live in those slots must be pushed up first or they would be silently dropped. After phase 1, $t_0$, $t_1$, $t_2$ are at most 51 bits live, and the carry-out of $t_2$ has joined $t_3$ in the high window.

**Phase 2 — three parallel rho folds.** For each $k \in \{0, 1, 2\}$,
$$t_k \cdot \beta^k \equiv (t_k \cdot \beta^{k-3}) \cdot \beta^3 \pmod{p}.$$
Applied to the schoolbook expansion, this gives
$$t_{\text{in}} = \sum_{k=0}^{9} t_k \cdot \beta^k \equiv \beta^3 \cdot \sum_{k=0}^{9} t_k \cdot \beta^{k-3} \pmod{p},$$
so the bottom three limbs (153 bits) can be dropped: the residue is now represented in a 7-limb high window at positions $\beta^3, \beta^4, \ldots, \beta^9$. The constants $\rho_{2-k} = \beta^{k-3} \bmod p$ are precomputed at compile time in 5 × 51-bit form by `compute_div_r_inv_local` (see `paired_rne_constants::rho`). The three folds share no inputs, so they execute on independent SIMD lane-pairs and combine via a balanced add tree. Phase 2 preserves the residue mod $p$ but **does not shrink the integer magnitude**: after it, $\mathrm{ss}$ is on the order of $\beta^2 \cdot p$.

**Phase 3 — two CIOS reductions.** Each step computes the scalar
$$m \;=\; \mathrm{ss}[i] \cdot n_p \bmod \beta, \qquad n_p \;=\; -p^{-1} \bmod \beta$$
(the `u51_np0` constant), adds $m \cdot p$ to the live window so the bottom limb is divisible by $\beta$, and propagates the carry forward — equivalently,
$$\mathrm{ss} \leftarrow (\mathrm{ss} + m \cdot p) / \beta.$$
Each step both divides by $\beta$ and tightens the bound: the per-step transform is
$$\mathrm{ss} < B \;\implies\; \mathrm{ss} < B/\beta + p$$
because $m \cdot p < \beta \cdot p$ before the divide. Two steps starting from $\sim \beta^2 \cdot p$ chain as $\beta^2 p \to \beta p + p \to 2p + p/\beta$. After phase 3, $\mathrm{ss} < 2p + p/\beta$ in kernel form $R_{\text{kernel}} = 2^{255}$ — the residual $p/\beta$ slack is absorbed by phase 5's halving.

**Phase 4 — parity fix.** To convert $R_{\text{kernel}} = 2^{255}$ to outer $R = 2^{256}$, one more halving mod $p$ is needed. If $\mathrm{ss}$ is odd, we add $p$ — since $p$ is odd, $\mathrm{ss} + p$ is even and still $\equiv \mathrm{ss} \pmod p$ — then run one carry-propagation pass to renormalize the 5 × 51-bit limbs. After phase 4, $\mathrm{ss} < 3p + p/\beta$.

**Phase 5 — lane split + fused $\gg 1$ repack.** `pack_to_4x64_shr_1` extracts each lane's scalar 5 × 51-bit array and repacks to 4 × 64-bit with a fused $\gg 1$, performing the halving prepared in phase 4. This is the final $\beta^{-1}$-equivalent step: combined with phases 2 and 3 the cumulative reduction is $3 \cdot 51 + 2 \cdot 51 + 1 = 256$ bits, which exactly converts $R_{\text{kernel}} = 2^{255}$ to outer $R = 2^{256}$. The output remains in coarse Montgomery form, matching the rest of the field API.

#### Bounds analysis

We must verify that the output is in $[0, 2p)$ without requiring an additional subtraction of $p$.

After the schoolbook, both factors are coarse ($< 2p$), so each column-sum-as-integer satisfies
$$\mathrm{ss} \;<\; (2p)^2 = 4p^2.$$

Phase 1 is a pure signed carry shuffle and does not change the integer value of $\mathrm{ss}$. Phase 2 replaces $t_{\text{in}}$ with a residue-equivalent value in a 7-limb $\beta^3 \ldots \beta^9$ window; bookkeeping (each $t_k < \beta$ contributes $t_k \cdot \rho_{2-k} < \beta \cdot p$ to the high window) puts the post-phase-2 bound at the order of $\beta^2 \cdot p$, which we take as the starting bound $B_3 = \beta^2 p$ for the CIOS chain.

Phase 3 applies the bound transform $B \mapsto B/\beta + p$ twice:
$$\beta^2 p \;\longrightarrow\; \beta p + p \;\longrightarrow\; \frac{\beta p + p}{\beta} + p \;=\; 2p + \frac{p}{\beta},$$
so after phase 3, $\mathrm{ss} < 2p + p/\beta$ in kernel form. Note this is *strictly larger* than $2p$ — the residual $p/\beta$ slack is by design, and phase 5 absorbs it.

Phase 4 conditionally adds $p$, giving $\mathrm{ss} < 3p + p/\beta$. Phase 5's fused halving then yields
$$\frac{\mathrm{ss} + 0 \text{ or } p}{2} \;<\; \frac{3p + p/\beta}{2} \;=\; \frac{3p}{2} + \frac{p}{2\beta} \;<\; 2p$$
(the final inequality follows from $p/(2\beta) < p/2$ for $\beta > 1$, and $\beta = 2^{51}$ comfortably satisfies this with $p/(2\beta) \approx p \cdot 2^{-52}$). The final 4 × 64-bit output is in coarse Montgomery form $[0, 2p)$, as required. No conditional subtraction is needed.

The two CIOS $m$-factors in phase 3 are computed with scalar 64-bit multiplications because `wasm_i64x2_mul` lowers to ~6 micro-ops on x86 V8, which is more expensive than two GPR `imul`s + extract/make.

### Converting to and from Montgomery form
Obviously we want to avoid using standard form division when converting between forms, so we use Montgomery form to convert to Montgomery form. If we look at a value $a\ mod\ p$ we can notice that this is the Montgomery form of $a\cdot R^{-1}\ mod\ p$, so if we want to get $aR$ from it, we need to multiply it by the Montgomery form of $R\ mod\ p$, which is $R\cdot R\ mod\ p$. So using Montgomery multiplication we compute

$$a \cdot R^2 / R  = a\cdot R\ mod\ p$$

To convert from Montgomery form into standard form we multiply the element in Montgomery form by 1:

$$ aR \cdot 1 / R = a\ mod\ p$$

## Architecture details {#field_docs_architecture_details}
You could say that for each multiplication or squaring primitive there are 3 implementations:
1. Generic 64-bit implementation when uint128_t type is available (there is efficient multiplication of 64-bit values)
2. Assembly 64-bit implementation (Intel ADX and no Intel ADX versions)
3. Implementation targeting WASM

The generic implementation has 2 purposes:
1. Building barretenberg on platforms we haven't targeted in the past (new ARM-based Macs, for example)
2. Compile-time computation of constant expressions, since we can't use the assembly implementation for those.

The assembly implementation for x86_64 is optimized. There are 2 versions:
1. General x86_64 implementation that uses 64-bit registers. The squaring operation is equivalent to multiplication for simplicity and because the original squaring implementation was quite buggy.
2. Implementation using Intel ADX. It allows simultaneous use of two addition-with carry operations (adox and adcx) on two separate CPU gates (units of execution that can work simultaneously on the same core), which almost halves the time spent adding up the results of uint64_t multiplication.

Implementation for WASM:

We use 9 29-bit limbs for computation while keeping the canonical 4 × 64-bit storage and the same $R = 2^{256}$ Montgomery form as native. The reason for the different internal limb width is that WASM doesn't have:
1. 128-bit result 64*64 bit multiplication
2. 64-bit addition with carry

On WASM targets that also expose relaxed SIMD, an additional *paired* implementation (see [Paired Montgomery multiplication](#field_docs_paired_explainer)) computes two independent Montgomery products at once using a 5 × 51-bit `f64x2` SIMD pipeline. It is an opt-in API surface (`paired_mul`, `paired_sqr`, …); the standard `montgomery_mul` still uses the 9 × 29-bit pipeline.

In the past we implemented a version with 32-bit limbs, but as a result, when we accumulated limb products we always had to split 64-bit results of 32-bit multiplication back into 32-bit chunks. Had we not, the addition of 2 64-bit products would have lost the carry flag and the result would be incorrect. There were 2 issues with this:
1. This spawned in a lot of masking operations
2. We didn't use more efficient algorithms for squaring, because multiplication by 2 of intermediate products would once again overflow.

Switching to 9 29-bit limbs increased the number of multiplications from 136 to 171. However, since the product of 2 limbs is 58 bits, we can safely accumulate 64 of those before we have to reduce. This allowed us to get rid of a lot of intermediate masking operations, shifts and additions, so the resulting computation turned out to be more efficient.

## Interaction of field object with other objects
Most of the time field is used with uint64_t or uint256_t in our codebase, but there is general logic of how we generate field elements from integers:
1. Converting from signed int takes the sign into account. It takes the absolute value, converts it to montgomery and then negates the result if the original value was negative
2. Unsigned integers ( <= 64 bits) are just converted to montgomery
3. uint256_t and uint512_t:
    1. Truncate to 256 bits
    2. Subtract the modulus until the value is within field
    3. Convert to montgomery

Conversion from field elements exists only to unsigned integers and bools. The value is converted from montgomery and appropriate number of lowest bits is used to initialize the value.

**N.B.** Functions for converting from uint256_t and back are not bijective, since values $ \ge p$ will be reduced.

## Field parameters

The field template is instantiated with field parameter classes, for example, class bb::Bn254FqParams. Each such class contains at least the modulus (in 64-bit and 29-bit form), r_inv (used for efficient reductions; the WASM Yuval-style reduction uses an additional `r_inv_wasm = 2^{-29} mod p` precomputation in 9 × 29-bit form), and r_squared used for converting to Montgomery form. Since $R = 2^{256}$ is shared across native and WASM, r_squared is a single value (no separate WASM version), and likewise cube_root, primitive_root and coset_generator — values already in Montgomery form — are defined once and used by every backend.

## Helpful python snippets

Parse field parameters out of a parameter class (doesn't check and reconstitute endomorphism parameters, but checks correctness of everything else)
```python
import re
def parse_field_params(s):
    def parse_number(line):
        """Expects a string without whitespaces"""
        line=line.replace('U','').replace('L','') # Clear away all postfixes
        if line.find('0x')!=-1: # We have to parse hex
            value= int(line,16)
        else:
            value = int(line)
        return value

    def recover_single_value(name):
        nonlocal s
        index=s.find(name)
        if index==-1:
            raise ValueError("Couldn't find value with name "+name)
        eq_position=s[index:].find('=')
        line_end=s[index:].find(';')
        return parse_number(s[index+eq_position+1:index+line_end])

    def recover_single_value_if_present(name):
        nonlocal s
        index=s.find(name)
        if index==-1:
            return None
        eq_position=s[index:].find('=')
        line_end=s[index:].find(';')
        return parse_number(s[index+eq_position+1:index+line_end])

    def recover_array(name):
        nonlocal s
        index = s.find(name)
        number_of_elements=int(re.findall(r'(?<='+name+r'\[)\d+',s)[0])
        start_index=s[index:].find('{')
        end_index=s[index:].find('}')
        all_values=s[index+start_index+1:index+end_index]
        result=[parse_number(x) for (i,x) in enumerate(all_values.split(',')) if i<number_of_elements]
        return result

    def recover_multiple_arrays(prefix):
        chunk_names=re.findall(prefix+r'_\d+',s)
        recovered=dict()
        for name in chunk_names:
            recovered[name]=recover_array(name)
        return recovered

    def recover_element_from_parts(prefix,shift):
        """Recover a field element from its parts"""
        chunk_names=re.findall(prefix+r'_\d+',s)
        val_dict=dict()
        for name in chunk_names:
            val_dict[int(name[len(prefix)+1:])]=recover_single_value(name)
        result=0
        for i in range(len(val_dict)):
            result|=val_dict[i]<<(i*shift)
        return result

    def reconstruct_field_from_4_parts(arr):
        result=0
        for i, v in enumerate(arr):
            result|=v<<(i*64)
        return result
    parameter_dictionary=dict()
    parameter_dictionary['modulus']=recover_element_from_parts('modulus',64)
    parameter_dictionary['r_squared']=recover_element_from_parts('r_squared',64)
    parameter_dictionary['cube_root']=recover_element_from_parts('cube_root',64)
    parameter_dictionary['primitive_root']=recover_element_from_parts('primitive_root',64)

    parameter_dictionary['modulus_wasm']=recover_element_from_parts('modulus_wasm',29)
    parameter_dictionary['r_inv_wasm']=recover_element_from_parts('r_inv_wasm',29)
    parameter_dictionary={**parameter_dictionary,**recover_multiple_arrays('coset_generators')}
    parameter_dictionary['endo_g1_lo']=recover_single_value_if_present('endo_g1_lo')
    parameter_dictionary['endo_g1_mid']=recover_single_value_if_present('endo_g1_mid')
    parameter_dictionary['endo_g1_hi']=recover_single_value_if_present('endo_g1_hi')
    parameter_dictionary['endo_g2_lo']=recover_single_value_if_present('endo_g2_lo')
    parameter_dictionary['endo_g2_mid']=recover_single_value_if_present('endo_g2_mid')
    parameter_dictionary['endo_minus_b1_lo']=recover_single_value_if_present('endo_minus_b1_lo')
    parameter_dictionary['endo_minus_b1_mid']=recover_single_value_if_present('endo_minus_b1_mid')
    parameter_dictionary['endo_b2_lo']=recover_single_value_if_present('endo_b2_lo')
    parameter_dictionary['endo_b2_mid']=recover_single_value_if_present('endo_b2_mid')

    assert(parameter_dictionary['modulus']==parameter_dictionary['modulus_wasm']) # Check modulus representations are equivalent
    modulus=parameter_dictionary['modulus']
    assert(parameter_dictionary['r_squared']==pow(2,512,modulus)) # Check r_squared (R = 2^256)
    assert(parameter_dictionary['r_inv_wasm']*(1<<29)%modulus==1) # Check r_inv_wasm = 2^{-29} mod p
    assert(pow(parameter_dictionary['cube_root']*pow(2,-256,modulus),3,modulus)==1) # Check cubic root

    return parameter_dictionary
```

Convert value from python to string for easy addition to bb's tests:
```python
def to_ff(value):
	print ("FF(uint256_t{"+','.join(["0x%xUL"%((value>>(i*64))&((1<<64)-1))for i in range(4)])+"})")
```
