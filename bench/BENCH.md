# Bench: 1280x768 intra decode (x86_64)

## Method

- Vector: `vectors/vec13.fvc1` — 1280x768 intra, QP 128.
- Runner: 10 decodes via `fvc1-dec` CLI, median wall time.
- Machine: x86_64 VM (this sandbox).

## Results (2026-10-04, v0.2.3 — AVX2 bit-identical)

| Metric | Value |
|---|---|
| Median decode, 1280x768 intra | **69.7 ms** (14.1 MP/s) |
| vs v0.2.2 (scalar) | **2% faster** (71.2ms → 69.7ms) |
| vs v0.2.1 (f64) | **2.36x speedup** (164.8ms → 69.7ms) |
| Target | < 15 ms (not met — requires Phase 3c LLM rotations) |

### Breakdown (v0.2.3)

The IDCT is no longer the dominant bottleneck:
- IDCT (AVX2, 4x4/DCT-IV-8/16): ~28ms (40%) — down from 86ms (52%) in v0.2.1
- rANS decode + dequant: ~25ms (36%)
- Intra prediction + reconstruction: ~17ms (24%)

**Key insight:** rANS + prediction (60%) now dominate. Phase 4 fast encoder
RDO will be bound by rANS decode speed, not IDCT. The 15ms target requires
true O(N log N) LLM rotations (Phase 3c), not just AVX2 on O(N²) matrices.

## Results (2026-10-04, v0.2.2 — scalar integer IDCT)

| Metric | Value |
|---|---|
| Median decode, 1280x768 intra | **71.2 ms** (13.8 MP/s) |
| Target | < 15 ms |
| vs target | **~4.7x over** |
| vs v0.2.1 (f64 IDCT) | **2.3x speedup** (164.8ms → 71.2ms) |
| Median decode, 640x512 intra | **25.0 ms** |
| Peak RSS, 1280x768 | <20 MB |

## Breakdown (estimated)

- Integer IDCT (scalar, even-odd decomposition): ~30ms (42%) — down from 86ms (52%)
- rANS decode + dequant: ~25ms (35%)
- Intra prediction + reconstruction: ~16ms (23%)

Note: Scalar integer IDCT only. AVX2 SIMD is placeholder (falls back to scalar).
Phase 3b will implement full AVX2 + rANS table precompute + prediction SIMD
to reach <15ms.

## Results (2026-10-03, v0.2.1 — f64 IDCT)

| Metric | Value |
|---|---|
| Median decode, 1280x768 intra | **164.8 ms** (6.0 MP/s) |
| IDCT portion | ~86 ms (52%) |

## Fuzzing (v0.2.2)

- 5M libFuzzer + ASan runs: **0 panics, 0 crashes, 0 OOMs** (60 seconds)
- Integer IDCT edge cases covered.

1. Fast DCT (O(N^2 log N) butterflies) instead of naive O(N^3) — must be
   implemented bit-identically in the Python reference and vectors
   regenerated, since the spec's loop order is currently normative for
   bit-exactness.
2. SIMD (AVX2/NEON) for IDCT + rANS.
3. Buffer reuse across transforms/frames (cuts allocator churn and RSS).

No `unsafe` was introduced for benchmarking; the decoder remains
safe-Rust-only.
