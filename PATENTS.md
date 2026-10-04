# PATENTS.md — FVC1 patent position

> **THIS IS NOT LEGAL ADVICE.** This file is an engineering prior-art survey,
> not a legal opinion. It does not grant any license and does not conclude
> that FVC1 is free of third-party patent claims. **Attorney review is
> required before any release, and before any public "royalty-free" claim.**
> Patent law varies by jurisdiction; expiry dates below are US-centric
> approximations and must be verified by counsel.

## Design philosophy

FVC1 is a clean-room design. No code or text was copied from x264/x265,
libaom/SVT-AV1, VVenC, dav1d, or any other codec implementation. Techniques
were re-derived from papers and public specifications only.

The strategy has three legs:

1. **Use old or openly-dedicated techniques** whose patents (if any) expired
   long ago, or whose authors dedicated them to the public.
2. **Use AOMedia-licensed techniques** (CDEF, Wiener restoration) only in the
   form released royalty-free inside the AV1 royalty-free codec, under the
   AOMedia patent license — while noting the license covers member patents,
   and non-members have asserted AV1-related patents (e.g. Dolby litigation;
   Access Advance's 2025–2026 call for patents for a potential AV1 pool).
3. **Defensively publish everything** in `docs/defensive/` with dates, so the
   design itself becomes prior art against future filings.

**Banned list (never implement):** CABAC / context-adaptive binary arithmetic
coding, SAO, large asymmetric partitions (VVC), matrix-based intra prediction
(MIP), 6-parameter affine motion (VVC-style), trellis-coded quantization
(AV2-era; too new to trust).

**Marketing rule:** never claim "patent free". The allowed claim is:
*"designed to be royalty-free under the AOMedia patent license, with all
known techniques defensively published — pending legal review."*

## Per-tool survey (v0.1)

| # | Coding tool | Prior art / basis | Filing / publication | Why believed safe | Residual risk |
|---|-------------|-------------------|----------------------|-------------------|---------------|
| 1 | rANS entropy coding, fixed contexts | J. Duda, "Asymmetric numeral systems", arXiv:1311.2540 (2013). Public-domain implementations (ryg_rans, CC0). Author explicitly intended public domain. | Paper 2013; Duda never filed. Google's 2015 "Mixed boolean-token ans coefficient coding" patent was **rejected by USPTO (2018) and abandoned** after Duda's third-party challenge. | Author dedication + failed Google patent + widespread use (Linux kernel, Android, JPEG XL). | **Microsoft was granted a US patent in Jan 2022** on "Features of range asymmetric number system encoding and decoding" despite prior art. Counsel must evaluate whether FVC1's fixed-context rANS reads on it. |
| 2 | DCT-II 4x4–32x32, separable | Ahmed, Natarajan & Rao, 1974. | 1974 — any patents expired decades ago (US term: 20 years). | Foundational 1970s mathematics; basis of JPEG/MPEG-1/2/H.264. | None known. |
| 3 | DST-VII 4x4–16x16 (allowed, not yet in bitstream) | Sinusoidal transforms, 1970s–80s literature; used in AV1/HEVC. | Pre-2004. | Old mathematics. | Low; verify specific integer approximations before use. |
| 4 | 8-mode directional intra prediction (v0.1 set) | Original simple formulation for FVC1 (axis/diagonal/oblique projection onto extended references). Directional prediction concept dates to H.264 (2003, expired). | FVC1 original, defensively published 2026-10-03. | Simple, differs from VP9/AV1 angle tables; H.264-era directional prediction patents expired. | Low; counsel to confirm no overlap with active angular-prediction filings. |
| 5 | Paeth + Smooth predictors (planned, not in v0.1) | Paeth: PNG spec (1995, never patented). Smooth: VP9/AV1, released under AOMedia royalty-free license. | PNG 1995; VP9 2013 / AV1 2018 (AOMedia). | PNG-era: ancient. VP9/AV1 versions: AOMedia member grant. | AOMedia grant covers members only; non-member assertions (cf. Dolby/AV1 suits) are the known gap. |
| 6 | Zigzag coefficient scan | JPEG (1992, patents long expired). | Pre-2004. | Ubiquitous expired prior art. | None known. |
| 7 | Uniform scalar quantization, qstep table | Foundational; every block codec since the 1980s. | Pre-2004. | Expired/never-patented basics. | None known. |
| 8 | Quad-tree partitioning (128→64→32) | Quadtrees predate video coding; used in HEVC (2013). | Concept pre-2004. | Generic data structure, not the VVC asymmetric/broad partition set. | Low. |
| 9 | Skip flags | Foundational (all modern codecs). | Pre-2004. | — | None known. |
| 10 | Wiener restoration filter (planned loop filter) | N. Wiener, 1940s. Released royalty-free inside AV1 under the AOMedia patent license. | 1940s; AV1 2018 (AOMedia). | 1940s mathematics + AOMedia grant. | Same AOMedia member-only caveat as #5. |
| 11 | CDEF variant (planned loop filter) | Midtskogen & Valin (Cisco/Mozilla), "The AV1 Constrained Directional Enhancement Filter", designed for the royalty-free AV1 codec; AV1 2018 under AOMedia patent license. | 2018 (AOMedia). | Released royalty-free by AOMedia members. | Same AOMedia member-only caveat as #5; FVC1 will use a *variant*, which counsel must clear independently. |
| 12 | Daala lapped biorthogonal transform (planned option) | Xiph.Org Daala project; authors placed the work in the public domain. | 2010s, public-domain dedication. | Explicit public-domain dedication by Xiph. | Low; verify no overlapping third-party filings. |

## What was deliberately NOT searched

A full USPTO/Espacenet freedom-to-operate search per tool has **not** been
performed. The per-tool notes above are engineering judgment from public
sources, which is exactly what the "pending legal review" qualifier is for.

## Process requirements (ongoing)

- Before adding any coding tool: check filing dates; if any relevant filing
  is newer than 2004, assume it is active and do not implement.
- Every tool gets a defensive publication in `docs/defensive/` with date +
  diagram, pushed to public git promptly.
- Every source file touching a coding tool carries:
  `/* Patent Status: ... */` with its basis.
