# Phase 3 — Optimization: Fast Integer IDCT + SIMD (Bit-Identical)

**Goal:** 1280×768 intra decode <15ms median on x86_64, peak RSS <20MB, zero bitstream change.

**Status:** DRAFT 2026-10-04. Do not implement until Phase 2 is fully landed.

## Background

Phase 1.5 bench: 1280×768 intra decodes in 164.8ms median, 52% (86ms) in naive
O(N³) f64 IDCT. The IDCT is the dominant hotspot. Phase 3 replaces it with a
fast integer IDCT that is bit-identical across Rust and Python.

## 1. Integer IDCT (normative)

Replace the f64 `dct_inverse` with a 16-bit integer LLM-style IDCT (AV1-style
factorization, clean-room implementation from the published algorithm).

### Spec change (Appendix A)

Add `spec/fvc1-bitstream.md` **Appendix A: Integer IDCT** defining normatively:
- Input: dequantized coefficients as 16-bit signed integers (clip to int16 range).
- Transform: LLM factorization with exact integer shifts and roundings at each butterfly stage.
- Output: 16-bit signed residuals, then `clip(half_away(pred + residual))` as before.
- All rounding: `(x + (1 << (shift-1))) >> shift` (round-half-up), specified per stage.

The Appendix must be precise enough that two independent implementations produce
bit-identical output. Reference the published LLM paper; do not copy code.

### Rust (`src/decoder/idct.rs`)

- New module `idct.rs` with `pub fn idct_4/8/16/32(block: &[i16]) -> [i16; N*N]`.
- **Safe scalar path:** pure safe Rust, `#[forbid(unsafe)]` at crate level; the SIMD module is the only exception.
- **SIMD path:** `core::arch::x86_64::{_mm256_*}` AVX2 intrinsics behind `#[cfg(target_arch = "x86_64")]`, gated at runtime by `std::is_x86_feature_detected!("avx2")` with scalar fallback. `unsafe` confined to this module only; all other modules keep `#[forbid(unsafe)]`.
- aarch64: scalar fallback for now; NEON in a follow-up (document as TODO).

### Python (`src/common/idct.py`)

- Port the *same* integer logic using pure Python ints (arbitrary precision, no numpy, no float).
- Must be bit-identical to Rust for all inputs (fuzz with random int16 blocks).

### Verification

- **Bit-identical:** Rust vs Python `cmp` clean on 10M random int16 blocks (all sizes).
- **Vectors:** Pixel outputs *will* differ slightly from v0.2 (float→integer). This is allowed: bump implementation to v0.2.1, regenerate `vectors/*_expected.yuv` via `fvc1-dec`, verify Rust vs Python `cmp` clean on all 15 vectors.
- **No bitstream change:** v0.2.1 decodes all v0.2 `.fvc1` files; only the *expected pixels* change, not the bitstream format. No v0.3 bump.

## 2. Other hotspots

- **rANS:** Keep single-state vanilla (no interleaving, no SIMD tricks — patent avoidance). Precompute cumulative tables and symbol→(freq, cum) lookups at startup. Target: <30% of decode time.
- **Intra prediction:** Keep 8 modes; ensure no bounds checks in hot loop (use unchecked indexing only inside the SIMD module, or restructure to avoid checks).
- **Loop filter:** Currently off (`loop_filter 0`). When Wiener+CDEF lands (future), inline the loops; for now, no-op.
- **Memory:** Reuse frame buffers; no `Vec` allocation in the per-block hot loop. Target peak RSS <20MB for 1280×768.

## 3. Benchmark requirements

Update `bench/BENCH.md` with:
- 1280×768 intra: **<15ms median** (was 164.8ms).
- 640×512 intra: **<5ms median**.
- Peak RSS for 1280×768: **<20MB** (was 13MB single, 45MB retained).
- Breakdown: IDCT <20%, rANS <30%, prediction <30%, other <20%.

## 4. Fuzzing

- Re-run **5M** libFuzzer inputs after optimization. No new panics, no OOM, no hangs.
- The integer IDCT must be fuzzed specifically: random int16 coefficients, verify no overflow (use wrapping or checked arithmetic with defined saturation).

## 5. Constraints

- **Do NOT** add new coding tools (no new intra modes, no new transforms, no loop filter).
- **Do NOT** start the fast encoder preset yet.
- **Do NOT** bump the bitstream version (stays v0.2.1).
- **Do NOT** break bit-identical Rust/Python agreement.
- Keep `#[forbid(unsafe)]` everywhere except the SIMD module.

## 6. Deliverables

- [ ] `spec/fvc1-bitstream.md` Appendix A (normative integer IDCT).
- [ ] `src/decoder/idct.rs` (safe scalar + AVX2 SIMD).
- [ ] `src/common/idct.py` (pure Python ints, bit-identical).
- [ ] All 15 vectors regenerated and verified (Rust vs Python `cmp` clean).
- [ ] `bench/BENCH.md` updated with <15ms result and breakdown.
- [ ] 5M fuzz inputs, zero panics.
- [ ] `cargo test` passes, zero `unsafe` outside SIMD module.

## 7. After Phase 3

Phase 4 (fast encoder preset) reuses the fast IDCT for RDO. The slow encoder
(Phase 2) remains the quality anchor; the fast encoder must produce
bit-identical output to the slow encoder's decisions (or document deviations).
