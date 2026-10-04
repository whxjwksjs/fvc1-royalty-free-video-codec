# Defensive Publication — FVC1 8-mode directional intra prediction set

**Date of first public disclosure:** 2026-10-03
**Project:** FVC1 (Free Video Codec 1)

## Purpose

Defensive publication: this predictor set is disclosed as prior art to block
future patent claims on it. Not legal advice.

## What is disclosed

Intra prediction for a WxH block from reconstructed neighbors, where
unavailable reference samples read as 512 (10-bit mid), with extended
reference arrays:

- `A_ext[i]`, i in [-1, 2W): above_left at -1, `above` at 0..W-1,
  `above_right` (or edge-replicated `above[W-1]`) at W..2W-1, clamped.
- `L_ext[i]`, i in [-1, 2H): above_left at -1, `left` at 0..H-1,
  `below_left` (or edge-replicated `left[H-1]`) at H..2H-1, clamped.

Eight modes:

```
0 V:      pred(x,y) = above[x]
1 H:      pred(x,y) = left[y]
2 D_DR:   pred(x,y) = A_ext[x + y]
3 D_DL:   pred(x,y) = A_ext[x - y]
4 D_HD:   pred(x,y) = L_ext[y + x]
5 D_HU:   pred(x,y) = L_ext[y - x]
6 V_OBL:  pred(x,y) = A_ext[x + (y >> 1)]
7 H_OBL:  pred(x,y) = L_ext[y + (x >> 1)]
```

Chroma uses the luma block's mode on the subsampled planes. This set is an
original simple formulation for FVC1; it deliberately avoids matrix-based
intra prediction (MIP) and the VP9/AV1 angle tables.

## Prior art cited

H.264 directional intra prediction (2003, expired); VP9 intra modes
(AOMedia royalty-free); Daala intra predictors (Xiph, public domain).

---
*Not legal advice. Defensive publication for prior-art purposes only.*
