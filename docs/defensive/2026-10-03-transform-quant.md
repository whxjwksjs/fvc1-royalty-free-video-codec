# Defensive Publication — FVC1 transform, quantization, and reconstruction

**Date of first public disclosure:** 2026-10-03 (commit 79bdae9, https://github.com/whxjwksjs/fvc1-royalty-free-video-codec)
**Project:** FVC1 (Free Video Codec 1)

## Purpose

Defensive publication: this processing chain is disclosed as prior art to
block future patent claims on it. Not legal advice.

## What is disclosed

Per coding block, per plane (Y, then U, then V), in raster block order:

1. Predict with the block's intra mode from reconstructed references.
2. Unless the plane's skip flag is set: inverse-quantize
   (`dqcoeff = qcoeff * qstep`, `qstep(Q) = max(1, round(2^(Q/16)))`),
   apply the separable inverse orthonormal DCT-II
   (`x = M^T * F * M`, f64 reference, sizes 4x4–32x32 with the rule
   `tx = min(block_dim, 32)`), and reconstruct
   `rec = clip(round_half_away(pred + residual), 0, 1023)`.
   If skipped, `rec = pred`.
3. Encoder-side (for conformance-vector generation): forward orthonormal
   DCT-II (`F = M * x * M^T`), `qcoeff = round_half_away(coeff / qstep)`,
   with prediction computed from the encoder's own reconstruction loop.

The v0.2 reference uses f64 with the exact loop order published in the
bitstream spec (section 7.1); migration to an integer exact-match transform
is declared future work.

## Prior art cited

Ahmed–Natarajan–Rao DCT-II (1974); orthonormal separable transform theory
(1970s); uniform scalar quantization (foundational, pre-2004).

---
*Not legal advice. Defensive publication for prior-art purposes only.*
