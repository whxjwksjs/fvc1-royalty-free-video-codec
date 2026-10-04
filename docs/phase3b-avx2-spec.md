# Phase 3b — AVX2 SIMD + rANS/Prediction Optimization

**Goal:** 1280×768 intra decode <15ms median on x86_64 (from 71.2ms v0.2.2).

**Status:** DRAFT 2026-10-04. Branch from v0.2.2 tag.

## Background

v0.2.2 (scalar integer IDCT) achieves 71.2ms, 2.3x over f64. Breakdown:
- IDCT (scalar): ~30ms (42%)
- rANS + dequant: ~25ms (35%)
- Intra prediction + reconstruction: ~16ms (23%)

To reach 15ms, need 4.7x overall. This requires:
1. AVX2 for IDCT (target 4x on IDCT → 30ms → 7.5ms)
2. rANS table precomputation (target 2x → 25ms → 12.5ms)
3. Prediction SIMD (target 2x → 16ms → 8ms)
Total: ~28ms → need further 2x from combined effects + reduced overhead.

## 1. AVX2 IDCT (Bit-Identical)

### Constraints (NON-NEGOTIABLE)

- **Bit-identical to Appendix A:** Same integer operations, same rounding,
  same shifts, same saturation. No floating-point. No reordering that changes
  rounding.
- **Bit-identical to Python:** Rust AVX2 output must match `src/common/idct.py`
  bit-for-bit on 10M random blocks.
- **Unsafe confined:** Only in `src/decoder/src/idct.rs::simd` module.
  All other modules keep `#[forbid(unsafe)]`.

### Implementation Strategy

The even-odd decomposition is recursive and branchy — not ideal for SIMD.
Instead, implement **direct 8×8 IDCT via LLM rotations** with AVX2:

**8-point LLM IDCT (inverse):**

The forward DCT uses butterflies + rotations. The inverse reverses them.

Stage 1 (input): De-interleave (reverse of forward output permutation)
Stage 2: 4 plane rotations (angles: π/16, 3π/16, 5π/16, 7π/16, negated for inverse)
Stage 3: 2 rotations (π/8, 3π/8) + 2 butterflies
Stage 4: 1 rotation (π/4) + 3 butterflies
Stage 5: Output scaling by 1/√2 (using 11585/16384 as in Appendix A)

**Rotation via AVX2:**
A plane rotation `[c -s; s c] * [x0; x1]` can be done as:
```
tmp0 = x0 * c - x1 * s
tmp1 = x0 * s + x1 * c
```
With AVX2, process 4 rotations in parallel (8 int32s):
- Load 8 x-values into `__m256i`
- Load 8 coefficients (c,s pairs) 
- Use `_mm256_mullo_epi32` + `_mm256_add_epi32`/`_mm256_sub_epi32`
- Shift right by 14 with rounding: `(x + 8192) >> 14`

**Fixed-point coefficients:**
Precompute `C[k] = round(cos(k*π/16) * 2^14)` for k=1,3,5,7, etc.
All rotations use Q14 coefficients, Q14 shift with round-half-up.

**Verification:**
- Unit test: AVX2 vs scalar on 1M random 8×8 blocks, must match bit-for-bit.
- If ANY mismatch, AVX2 is disabled (fallback to scalar).

### 4×4, 16×16, 32×32

- **4×4:** Direct matrix via AVX2 (16 coeffs, 4 outputs — fits in 2 AVX2 registers).
- **16×16:** Two 8×8 IDCTs + combine (even-odd, but with AVX2 8×8 as base).
- **32×32:** Two 16×16 + combine.

This avoids the DCT-IV matrices entirely for the AVX2 path.

## 2. rANS Table Precomputation

**Current:** Symbol-by-symbol decode with per-symbol table lookups.

**Optimization:**
- Precompute `decode_table[symbol] = (freq, cum_freq)` for each context at stream init.
- Use direct array indexing instead of HashMap.
- Batch: decode 4 symbols at once where dependencies allow (rANS is serial, but table lookups can be pipelined).

**Constraint:** Must remain single-state vanilla rANS (no interleaving — patent avoidance).
Precomputation is just caching, not a coding tool change.

**Target:** 25ms → 12ms (2x).

## 3. Intra Prediction SIMD

**Current:** Per-pixel prediction with bounds checks.

**Optimization:**
- **H/V modes:** Use `_mm256_loadu_si256` / `_mm256_storeu_si256` for 16-pixel copies.
  These are just memcpys — very fast with AVX2.
- **Diagonal modes:** Use `_mm256_avg_epu16` for (a+b+1)/2 averaging.
  Process 16 pixels at once.
- **DC mode:** Broadcast via `_mm256_set1_epi16`.

**Constraint:** Must handle edge cases (block at frame boundary) with scalar fallback.
Only the interior fast path uses SIMD.

**Target:** 16ms → 8ms (2x).

## 4. Verification Requirements

- [ ] Rust AVX2 vs Python `idct.py`: 10M random blocks, bit-identical.
- [ ] All 15 vectors: Rust vs Python `cmp` clean.
- [ ] 5M libFuzzer + ASan: 0 panics.
- [ ] Bench: 1280×768 <15ms median, breakdown IDCT<20% / rANS<30% / pred<30%.
- [ ] `cargo test`: all pass.
- [ ] Zero `unsafe` outside `idct.rs::simd`.

## 5. What NOT to Do

- Do NOT change Appendix A (the integer math is frozen).
- Do NOT change the bitstream (stays v0.2.2).
- Do NOT add new coding tools.
- Do NOT use floating-point in IDCT.
- Do NOT interleave rANS states (patent).

## 6. Deliverables

- [ ] `src/decoder/src/idct.rs`: AVX2 8×8 LLM rotations, bit-identical.
- [ ] `src/decoder/src/rans.rs`: Precomputed decode tables.
- [ ] `src/decoder/src/intra.rs`: AVX2 for H/V/DC modes.
- [ ] `bench/BENCH.md`: <15ms with breakdown.
- [ ] 5M fuzz clean.
- [ ] Merge to main, tag v0.2.3.

## 7. Fallback Plan

If AVX2 cannot achieve bit-identicality (rounding differences):
- Keep scalar integer IDCT (v0.2.2 is already 2.3x faster than f64).
- Focus on rANS + prediction optimization (target 71ms → 30ms).
- Document: "AVX2 attempted, rounding mismatch, scalar retained for correctness."

Correctness > speed. A fast wrong decoder is useless.
