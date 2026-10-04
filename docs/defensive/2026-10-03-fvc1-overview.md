# Defensive Publication — FVC1 overall codec design

**Date of first public disclosure:** 2026-10-03 (commit 79bdae9, https://github.com/whxjwksjs/fvc1-royalty-free-video-codec)
**Project:** FVC1 (Free Video Codec 1), clean-room royalty-free video codec
**Disclosed by:** Gc / R2D2 (design agent)

## Purpose

This document is a defensive publication. It describes the FVC1 video codec
design so that the design constitutes prior art against any future patent
application that would claim it. It is published to prevent others from
patenting these techniques, not to assert any patent rights.

## What is disclosed

A video codec ("FVC1") with the following combination, as specified in
`spec/fvc1-bitstream.md` v0.2:

- 128x128 superblocks with quad-tree partitioning (2 bits per node,
  preorder, max depth 5 in v0.2), KEY/INTER frame types.
- YUV420 10-bit default color format.
- Per-block intra/inter flag; intra modes drawn from an 8-mode directional
  set with extended references (see companion publication on intra modes).
- Separable orthonormal DCT-II transforms, 4x4 to 32x32, with transform size
  `min(block_dim, 32)`; DST-VII allowed 4x4–16x16.
- Coefficient coding with rANS using exactly four fixed contexts
  (DC / AC-low / AC-high / sign), a 17-symbol magnitude alphabet with an
  escape symbol plus raw u16 escapes, and fixed probability tables published
  in the bitstream spec.
- Uniform scalar quantization with `qstep(Q) = max(1, round(2^(Q/16)))`,
  Q in 0..255, signaled per block.
- Skip flags per block for luma and chroma.
- Planned (not in v0.2 bitstream): Wiener + CDEF-variant loop filtering,
  Paeth/Smooth intra predictors, Daala-style lapped transform option.

## Prior art cited

Duda "Asymmetric numeral systems" (arXiv:1311.2540, 2013); Ahmed–Natarajan–Rao
DCT (1974); JPEG zigzag scan (1992); PNG Paeth predictor (1995); H.264
directional intra prediction (2003); VP9/AV1 (AOMedia, royalty-free); Daala
(Xiph, public domain); Wiener (1940s).

---
*Not legal advice. Defensive publication for prior-art purposes only.*
