# Defensive Publication — rANS with four fixed contexts for video coefficients

**Date of first public disclosure:** 2026-10-03 (commit 79bdae9, https://github.com/whxjwksjs/fvc1-royalty-free-video-codec)
**Project:** FVC1 (Free Video Codec 1)

## Purpose

Defensive publication: this design is disclosed as prior art to block future
patent claims on it. Not legal advice.

## What is disclosed

Entropy coding of quantized transform coefficients in a video codec using:

1. Exactly **four fixed rANS contexts**: DC (coefficient position 0),
   AC_LOW (zigzag positions 1–16), AC_HIGH (positions 17+), SIGN.
2. Fixed (non-adaptive) probability tables summing to 65536, published in
   the bitstream specification, e.g. v0.2 tables:
   - DC: `[16384,12288,8192,6144,4096,3072,2048,1536,1024,768,512,384,256,192,128,96,8416]`
   - AC_LOW: `[60000,2048,1024,512,256,128,64,32×9,1216]`
   - AC_HIGH: `[63000,1024,512,256,128,64,32,16,8×8,440]`
   - SIGN: `[32768,32768]`
3. A 17-symbol alphabet per coefficient context where symbols 0–15 encode
   magnitudes 0–15 directly and symbol 16 is an escape, with the true
   magnitude appended as a raw u16 in a trailing escape section ordered:
   all DC escapes, then AC_LOW escapes, then AC_HIGH escapes (transforms in
   enumeration order).
4. Signs coded once per nonzero coefficient (0 = positive, 1 = negative) in
   decode-encounter order: DC signs, then AC signs.
5. Standard byte-aligned rANS with L = 2^23, 16-bit probability precision,
   symbols encoded last-to-first, renorm bytes consumed last-emitted-first
   (LIFO), each context stored as (u32 final_state, u32 num_bytes, bytes).

## Prior art cited

J. Duda, "Asymmetric numeral systems" (arXiv:1311.2540, 2013), public domain
by author's intent; public-domain rANS implementations (ryg_rans, CC0).

---
*Not legal advice. Defensive publication for prior-art purposes only.*
