# Phase 4 — Fast Encoder Preset (Realtime)

**Goal:** 1080p30 on i7-12700 single-thread; <5× x264 medium speed; quality within 5% BDBR of slow encoder.

**Status:** DRAFT 2026-10-04. Do not implement until Phase 3 (fast IDCT) is landed.

## Background

Phase 2 (slow encoder) does full RDO: partition depth 0–5, 8 intra modes,
Q 0–255, diamond + 1/8-pel MV search. It takes ~60s for 640×512 intra.
Phase 4 builds a fast preset that reuses Phase 3's fast IDCT for RDO but
adds early-skip heuristics to reach realtime.

## 1. Preset definition

`fvc1-enc --preset fast` (alongside `--preset slow`).

### Intra fast path

- **Partition:** Limit depth to 0–3 (blocks ≥16×16) by default; depth 4–5 only if variance exceeds threshold. Early skip: if block variance < T_qp, code as skip without RDO.
- **Modes:** SATD-based mode pre-selection: compute SATD for 8 modes, full RDO only on top 3. (SATD uses Hadamard transform, not DCT — faster.)
- **Q:** Do not brute-force 0–255. Use QP from `--qp` with ±8 refinement search.
- **RDO metric:** D + λR with D = SATD (not SSE) for mode/partition decisions; SSE only for final Q refinement.

### Inter fast path

- **MV search:** Hexagonal search (not diamond) at integer pel, range ±32 (not ±64). Skip 1/8-pel refinement if best MV is (0,0) and SAD < threshold (early skip).
- **Mode decision:** Compare inter (best MV) vs intra (top-3 modes) with SATD-based cost; full RDO only for the winner.
- **Q:** Same ±8 refinement as intra.

### What stays exact

- Bitstream format: v0.2.1 (no changes).
- Reconstruction: bit-identical to slow encoder *given the same decisions*. (Decisions may differ due to heuristics; that's allowed.)
- rANS, CRC32, partition signaling: unchanged.

## 2. Quality requirement

- BDBR vs slow encoder: **<5%** worse (on the 5 RD clips from Phase 2).
- If a heuristic causes >5% loss, it must be gated behind a stricter threshold or removed.
- The slow encoder remains the reference; fast must not produce a stream the slow decoder rejects.

## 3. Speed requirement

- **1080p30 on i7-12700 single-thread:** ~33ms/frame budget. Target <30ms median.
- **<5× x264 medium:** x264 medium does 1080p30 in ~X ms; fvc1-enc fast must be within 5× of that.
- No specific target for 640×512, but should be <10ms.

## 4. Implementation

- New module `src/encoder/fast.rs` (or `preset_fast.rs`).
- Reuse: `CostTables`, `collect_txs`, `best_q` (with limited range), `motion_compensate`, `write_frame`.
- New: `satd()` function (Hadamard), hexagonal search, early-skip thresholds.
- CLI: `--preset fast` selects the fast path; `--preset slow` (default) keeps Phase 2 behavior.

## 5. Verification

- **Round-trip:** `fvc1-dec` must decode fast-encoder output bit-exactly (same as slow).
- **Vectors:** Generate `vec16-vec20` with fast preset; verify Rust vs Python.
- **BDBR:** vs slow encoder on 5 clips, must be <5%.
- **Speed:** Bench on i7-12700 (or document the test machine).

## 6. Constraints

- **Do NOT** change the bitstream (stays v0.2.1).
- **Do NOT** add new coding tools.
- **Do NOT** break the slow encoder (it stays as `--preset slow`).
- Keep safe Rust, no unsafe, no external crates.

## 7. Deliverables

- [ ] `fvc1-enc --preset fast` implemented.
- [ ] BDBR <5% vs slow on 5 clips (document in `bench/rd-fast.md`).
- [ ] Speed: 1080p30 single-thread on i7-12700 (document in `bench/BENCH.md`).
- [ ] vec16–20 (fast preset vectors), verified Rust vs Python.
- [ ] Fuzz the fast encoder output: 1M inputs through `fvc1-dec`, zero panics.

## 8. After Phase 4

- WASM demo, FFmpeg integration, GStreamer, MP4 fourcc, Verilog (per original roadmap).
- The fast preset becomes the default for realtime use; slow remains for quality anchor.
