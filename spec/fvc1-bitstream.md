# FVC1 Bitstream Specification — v0.2.1

**Status: FROZEN 2026-10-04 (errata).** No bitstream changes without a spec version
bump and full regeneration of `vectors/`. v0.2.1 is bitstream-identical to
v0.2; it clarifies reconstruction ordering (§11) only.

**Version history:**
- v0.1 (2026-10-03): decoder-first subset — KEY frames only, single frame per
  file, quad partitions max depth 2, no checksums.
- v0.2 (2026-10-03): multi-frame container, INTER (P) frames with 1/8-pel MVs
  and bilinear subpel filter, per-stream CRC32, partition depth 5 (4x4
  blocks), decode-bomb guards. Supersedes v0.1; v0.1 files are not decodable
  by v0.2 decoders (container changed).
- v0.2.1 (2026-10-04): errata only — clarifies §11: reconstruction follows
  partition preorder (causal), transform coefficient enumeration follows
  frame-raster. No bitstream changes; v0.2.1 decodes all v0.2 streams
  bit-identically.

Clean-room design. No code or text was copied from AV1/HEVC/VVC encoders or
decoders. Techniques used here are documented in `PATENTS.md` with prior art.

> **Not legal advice.** This spec does not grant any patent license and is not
> a legal opinion. Attorney review is required before any release.

## 1. Conventions

- All multi-byte integers are **little-endian**.
- Bit fields are packed **MSB-first** within each byte.
- 10-bit samples are stored in 16-bit words; valid range is 0..1023.
  Out-of-range values after reconstruction are clipped.
- `half_away(x)`: round half away from zero —
  `x >= 0 ? floor(x + 0.5) : -floor(-x + 0.5)`.
- `clip(v, lo, hi)`.
- `crc32(data)`: IEEE 802.3 CRC-32 (polynomial 0xEDB88320, init 0xFFFFFFFF,
  refin/refout true, xorout 0xFFFFFFFF). Matches `zlib.crc32`.
- Arithmetic right shift `>>` on signed values rounds toward negative
  infinity (floor). `x & 7` after such a shift yields the positive remainder:
  for any integer `f`, `ix = f >> 3`, `fr = f & 7` satisfies `f = ix*8 + fr`
  with `0 <= fr <= 7`.

## 2. Container

| Field       | Type | Value                          |
|-------------|------|--------------------------------|
| magic       | 4 bytes | `"FVC1"`                    |
| version     | u8   | 2                              |
| num_frames  | u32  | >= 1                           |

Then per frame, in decode order:

| Field       | Type | Value                          |
|-------------|------|--------------------------------|
| frame_len   | u32  | bytes in this frame's data     |
| frame_data  | frame_len bytes | one frame (§3–§10)   |

A decoder must consume exactly `frame_len` bytes per frame; trailing bytes
inside a frame are an error.

## 3. Frame header

| Field        | Type | Value (v0.2)                        |
|--------------|------|-------------------------------------|
| frame_type   | u8   | 0 = KEY, 1 = INTER (P)              |
| width        | u16  | multiple of 128                     |
| height       | u16  | multiple of 128                     |
| color        | u8   | 0 = YUV420 10-bit (only value)      |
| loop_filter  | u8   | 0 = off (only value)                |
| reserved     | u8   | 0                                   |

INTER frames must repeat the KEY frame's width/height/color exactly.
The decoder retains the most recent decoded frame as the reference picture
(`ref_idx 0`). An INTER frame with no decoded reference is an error.

## 4. Superblocks and partitions

The frame is divided into 128x128 superblocks in raster order. Chroma planes
are half resolution in each dimension (YUV420).

Per superblock, a partition tree is coded **preorder**, 2 bits per node,
MSB-first, padded with zero bits to the next byte boundary:

- `0` = quad split (four children follow, each a full subtree)
- `1` = leaf (coding block)
- `2`, `3` = reserved (error)

Maximum depth is 5, so leaf blocks range from 128x128 down to 4x4.
(Binary splits remain reserved.)

## 5. Coding block syntax (per leaf, raster order within the superblock)

| Field        | Type | Condition / value (v0.2)          |
|--------------|------|-----------------------------------|
| intra_flag   | u8   | 0 = inter, 1 = intra              |
| intra_mode   | u8   | if intra: 0..7 (see §6)           |
| ref_idx      | u8   | if inter: must be 0 (v0.2)        |
| mv_x         | i16  | if inter: 1/8-pel units (§7)      |
| mv_y         | i16  | if inter: 1/8-pel units (§7)      |
| warp         | u8   | if inter: must be 0 (v0.2)        |
| q_index      | u8   | 0..255 (see §8)                   |
| skip_luma    | u8   | 0/1 — 1 means all luma coeffs zero |
| skip_chroma  | u8   | 0/1 — 1 means all chroma coeffs zero |

v0.2 codes headers as raw bytes. Header entropy coding is deferred.

## 6. Intra prediction

Unchanged from v0.1. Reference samples for a WxH block at (bx, by):

- `above[x]`, `left[y]`, `above_left`, `above_right[x]`, `below_left[y]`
  with availability rules per v0.1; unavailable samples read as 512.

Extended arrays `A_ext[i]`, `L_ext[i]` (i in [-1, 2W), clamped) as in v0.1.

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

Chroma uses the luma block's mode on the subsampled planes.

## 7. Motion compensation (INTER blocks)

For each pixel (x, y) in the block, with MV (mvx, mvy) in 1/8-pel units:

```
fx_full = x*8 + mvx;  fy_full = y*8 + mvy
ix = fx_full >> 3;    fx = fx_full & 7
iy = fy_full >> 3;    fy = fy_full & 7
```

Bilinear interpolation on the reference frame (edge-clamped samples):

```
p00 = ref[ix][iy];  p10 = ref[ix+1][iy]
p01 = ref[ix][iy+1]; p11 = ref[ix+1][iy+1]
pred(x,y) = ((8-fx)*(8-fy)*p00 + fx*(8-fy)*p10
             + (8-fx)*fy*p01 + fx*fy*p11 + 32) / 64     // integer division
```

All terms are non-negative, so `(v + 32) / 64` is exact round-half-up.
v0.2 uses this bilinear filter deliberately (simple, fully specified); a
higher-quality separable filter is a planned upgrade and will be versioned.

Chroma (4:2:0): the chroma MV is `(mvx >> 1, mvy >> 1)` in 1/8-pel chroma
units, applied to chroma coordinates. (For odd luma MVs this truncates the
chroma subpel position toward negative infinity — documented v0.2 behavior.)

## 8. Quantization

```
qstep(Q) = max(1, half_away(2^(Q/16)))
```

Forward (encoder): `qcoeff = half_away(coeff / qstep)`.
Inverse (decoder): `dqcoeff = qcoeff * qstep`.

## 9. Transforms

Transform size rule: `tx = min(block_width, 32)`, square only.
Enumeration per coding block: all luma transforms raster, then chroma U
transforms raster, then chroma V transforms raster.

Orthonormal DCT-II only in v0.2 (DST-VII allowed by the project, not yet in
the bitstream), separable, f64 reference with the normative loop order:

Forward: `F = M * x * M^T`. Inverse: `x = M^T * F * M`.
(`M[k][n] = sqrt(2/N) * c[k] * cos(pi*(2n+1)*k/(2N))`, `c[0] = 1/sqrt(2)`.)

Loop order (normative for bit-exactness) — forward:
```
# T = M @ x
for i, j:  T[i][j] = sum_n M[i][n] * x[n][j]
# F = T @ M^T
for i, j:  F[i][j] = sum_n T[i][n] * M[j][n]
```
Inverse:
```
# T = M^T @ F
for i, j:  T[i][j] = sum_n M[n][i] * F[n][j]
# x = T @ M
for i, j:  x[i][j] = sum_n T[i][n] * M[n][j]
```

An integer exact-match transform is a defined future step (required before
hardware work); it is not part of v0.2 conformance.

## 10. Coefficient coding (rANS)

Zigzag scan (JPEG-style diagonal, extended to 32x32) as in v0.1.
Position 0 → context **DC**; positions 1..16 → **AC_LOW**; 17+ → **AC_HIGH**;
signs → **SIGN**.

Symbol alphabet per coefficient context: 0..15 = `|qcoeff|` 0..15;
16 = escape, true `|qcoeff|` appended as u16 to the escape section.
Signs: 0 = positive, 1 = negative, in decode-encounter order (all DC signs
in transform enumeration order, then all AC signs in transform enumeration
order, zigzag scan order within).

rANS: byte-aligned, `L = 2^23`, 16-bit precision. Symbols encoded in reverse,
decoded forward; renorm bytes are LIFO (last emitted = first consumed).
Fixed probability tables (weights sum to 65536):

```
DC:      [16384,12288,8192,6144,4096,3072,2048,1536,1024,768,512,384,256,192,128,96,8416]
AC_LOW:  [60000,2048,1024,512,256,128,64,32,32,32,32,32,32,32,32,32,1216]
AC_HIGH: [63000,1024,512,256,128,64,32,16,8,8,8,8,8,8,8,8,440]
SIGN:    [32768,32768]
```

Per context, in fixed order DC, AC_LOW, AC_HIGH, SIGN:

| Field       | Type |
|-------------|------|
| final_state | u32  |
| num_bytes   | u32  |
| crc32       | u32 — IEEE CRC-32 of `bytes` (§1) |
| bytes       | num_bytes bytes |

The decoder **must verify `crc32` before decoding** the stream; a mismatch
is a hard error (`ChecksumMismatch`), never silent garbage. A context with
zero symbols stores `final_state = L`, `num_bytes = 0`, `crc32 = crc32("")`.

Escape section: `num_escapes u32`, then `num_escapes × u16` values in
decode-encounter order (DC escapes, then AC_LOW, then AC_HIGH; transforms
in enumeration order).

## 11. Reconstruction

**Ordering (v0.2.1 errata):** Transform *coefficients* are enumerated in
**frame-raster order** (all superblocks' blocks sorted by (y, x); §9), but
pixel *reconstruction* must follow **partition preorder** (the order blocks
appear in the §4 partition trees, SB-by-SB in raster). Preorder is causal:
a block's top and left neighbors are always reconstructed before it, which
intra prediction (§6) requires. Frame-raster order is *not* causal in
general. (The block *headers* in §5 appear SB-by-SB in bitstream order.)

For each coding block in partition preorder, each plane (Y, then U, then V):

```
pred  = intra_predict(mode, block)   // §6, if intra
        motion_compensate(mv, block) // §7, if inter
if not skip:
  qcoeffs = rans_decode(...)         // §10 (CRC verified first)
  dq      = qcoeffs * qstep
  resid   = idct(dq)                 // §9
  // fast path: all-zero qcoeffs  =>  resid is all zeros, skip the IDCT
  rec     = clip(half_away(pred + resid), 0, 1023)
else:
  rec     = pred
```

## 12. Decode bombs and decoder obligations

A decoder MUST NOT panic, hang, or over-allocate on malicious input. It
returns a typed error instead. Minimum obligations:

- Every read is bounds-checked against the input (`Truncated` on overrun).
- `width * height <= 3_000_000` pixels (total allocation stays under 10 MB);
  dimensions are multiples of 128.
- Partition depth <= 5; leaves per superblock <= 1024.
- `num_bytes` of any rANS stream <= remaining input bytes.
- `num_escapes <=` total coded symbols.
- CRC32 verified per stream before decode; escape count exact.
- Frame bytes consumed must equal `frame_len` exactly.
- INTER with no reference frame, or dimension mismatch, is an error.
- No recursion deeper than the partition depth; no allocation proportional
  to anything but frame geometry and declared stream lengths.

## 13. Patent status

/* Patent Status: clean-room design; see PATENTS.md and docs/defensive/.
   This is not legal advice. Requires attorney review before release. */

## Appendix A: Integer IDCT (Normative, v0.2.1)

This Appendix defines the exact integer inverse DCT used by FVC1 decoders
from v0.2.1 onward. It replaces the f64 reference in §7. All operations are
on signed integers; bit-identical output is required across implementations.

### A.1 Input and Output

- **Input:** Dequantized coefficients `dq[k]` as signed 32-bit integers,
  obtained by `half_away(qcoeff * qstep)` (round half away from zero).
  Range: `[-2^20, 2^20]` (enforced by coefficient clipping).
- **Output:** Spatial residuals `r[n]` as signed 16-bit integers.
  Range: `[-2^15, 2^15 - 1]` (saturated).

### A.2 1D Integer IDCT

The 1D N-point integer IDCT (N = 4, 8, 16, 32) is defined recursively
via even-odd decomposition (Loeffler-style factorization).

**Base case (N=4):** Direct matrix multiplication with fixed-point coefficients.

Define `W4[k][n] = round(M4[k][n] * 2^14)` where M4 is the 4-point orthonormal
DCT-II matrix (§7). The 1D IDCT-4 is:

```
for n in 0..4:
    s = 0
    for k in 0..4:
        s += W4[k][n] * dq[k]   // 32-bit accumulator
    r[n] = sat16((s + (1 << 13)) >> 14)   // round, shift, saturate to int16
```

The `W4` coefficients (row k, column n) are:
```
W4 = [
  [ 8192,  8192,  8192,  8192],  // k=0: sqrt(1/4)*2^14
  [10703,  4433, -4433,-10703],  // k=1
  [ 8192, -8192, -8192,  8192],  // k=2
  [ 4433,-10703, 10703, -4433],  // k=3
]
```
(Values are `round(sqrt(2/4) * c_k * cos(pi*(2n+1)*k/8) * 2^14)`.)

**Recursive case (N=8, 16, 32):** Even-odd decomposition.

Given input `dq[0..N-1]`:
1. Split: `E[j] = dq[2*j]`, `O[j] = dq[2*j+1]` for j=0..N/2-1.
2. Compute `a = IDCT_{N/2}(E)` (N/2-point integer IDCT, recursively).
3. Compute `b = IDCT-IV_{N/2}(O)` (see A.3).
4. Combine for n=0..N/2-1:
   ```
   even = (a[n] + 1) >> 1   // divide by sqrt(2), rounded; see note
   odd  = (b[n] + 1) >> 1
   r[n]     = sat16(even + odd)
   r[N-1-n] = sat16(even - odd)
   ```

**Note on sqrt(2) scaling:** The even-odd decomposition requires division by
sqrt(2). We approximate `1/sqrt(2) ≈ 11585/16384` (i.e., `round(2^14 / sqrt(2))`).
So: `even = (a[n] * 11585 + (1 << 13)) >> 14`.

### A.3 Integer DCT-IV (for odd part)

The (N/2)-point integer DCT-IV is defined via direct matrix multiplication.

Define `V_M[k][n] = round(sqrt(2/M) * cos(pi*(2n+1)*(2k+1)/(4M)) * 2^14)`
for M = N/2. Then:

```
for n in 0..M-1:
    s = 0
    for k in 0..M-1:
        s += V_M[k][n] * O[k]
    b[n] = sat16((s + (1 << 13)) >> 14)
```

The `V` matrices are precomputed constants (given in the reference implementation).

### A.4 2D Integer IDCT

The 2D N×N integer IDCT applies the 1D IDCT to each row, then to each column.
Intermediate row results are 16-bit; column IDCT takes 16-bit input.

### A.5 Bit-Exactness Requirements

1. All multiplications use 32-bit signed accumulators.
2. Rounding is round-half-up: `(s + (1 << (SHIFT-1))) >> SHIFT`.
3. Right shifts are arithmetic (sign-extending).
4. Saturation to int16: `sat16(x) = max(-32768, min(32767, x))`.
5. The order of operations (loop nesting) MUST match the reference implementation.
6. No floating-point operations are permitted in the IDCT.

### A.6 Verification

A decoder is compliant iff, for 10,000,000 random int32 coefficient blocks
(all sizes 4, 8, 16, 32), its output matches the Python reference
(`src/common/idct.py`) bit-for-bit.
