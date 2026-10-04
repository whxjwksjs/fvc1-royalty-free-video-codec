# Phase 3b Status Report (2026-10-04)

## What was implemented

AVX2 SIMD for integer IDCT, all bit-identical to scalar (verified):
- 4x4 IDCT via AVX2 (`idct4_32_avx2`)
- DCT-IV-8 via AVX2 (`idct_iv8_avx2`)
- DCT-IV-16 via AVX2 (`idct_iv16_avx2`)

All use exact same integer ops as Appendix A: Q14 coefficients, 
`(s + 8192) >> 14` rounding, `sat16` saturation. Zero `unsafe` outside
`idct.rs::simd` module.

## Results

| Metric | v0.2.2 (scalar) | Phase 3b (AVX2) | Delta |
|---|---|---|---|
| 1280x768 median | 71.2ms | 69.7ms | **2% faster** |
| Bit-identical | Yes | Yes | — |
| Vectors cmp clean | 15/15 | 15/15 | — |

## Why AVX2 didn't help

The even-odd decomposition uses DCT-IV **matrices** (O(N^2)), not LLM
**rotations** (O(N log N)). AVX2 accelerates the matrix-vector multiplies,
but:

1. Each 8x8 DCT-IV still does 64 multiplies (8 outputs × 8 inputs).
   AVX2 does 8 multiplies per instruction → 8 instructions per 8x8.
   Scalar does 64 multiplies. Theoretical 8x, but...
2. Horizontal sums (`_mm256_hadd_epi32`) are slow (3-cycle latency).
3. `is_x86_feature_detected` overhead per call (now cached).
4. The recursion overhead (function calls, even/odd splits) dominates.

**Fundamental issue:** O(N^2 log N) with a better constant is still O(N^2 log N).
To reach 15ms, need O(N log N) via true LLM rotations.

## Path to 15ms

Requires **true LLM factorization with plane rotations** (not DCT-IV matrices):

For 8-point IDCT:
- Stage 1: 4 rotations (angles π/16, 3π/16, 5π/16, 7π/16)
- Stage 2: 2 rotations (π/8, 3π/8) + 2 butterflies  
- Stage 3: 1 rotation (π/4) + 3 butterflies
- Each rotation: 4 multiplies + 2 adds (vs 64 for matrix)

This is ~8x fewer operations than the matrix approach. With AVX2 doing
4 rotations in parallel, theoretical speedup is 32x on the IDCT portion.

**But:** Deriving the exact rotation structure with bit-identical rounding
is complex and error-prone. The current scalar implementation is proven
correct; a rewrite risks introducing subtle bugs.

## Recommendation

**Option A:** Accept 70ms as Phase 3b result. The 2.3x speedup from integer
IDCT (v0.2.2) is the main win. Document that 15ms requires LLM rotations.

**Option B:** Implement true LLM rotations (4-6 hours of careful work).
Must verify bit-identicality on 10M random blocks before merging.

**Option C:** Skip to Phase 4 (fast encoder). The 70ms decode is sufficient
for photo/single-frame use. Video (30fps) needs 15ms, but that's a separate
milestone.

The v0.2.2 scalar baseline remains the bisect point. The AVX2 code is
bit-identical and safe to keep, but does not achieve the 15ms target.
