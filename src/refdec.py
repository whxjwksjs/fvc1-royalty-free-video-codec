"""FVC1 Python reference decoder (spec v0.2).

Implements spec/fvc1-bitstream.md v0.2 exactly: multi-frame container,
INTER frames with 1/8-pel MVs + bilinear filter, per-stream CRC32,
partition depth 5, typed errors (no exceptions on malformed input beyond
FvcError). Used to cross-check the Rust fvc1-dec.

/* Patent Status: clean-room implementation; see PATENTS.md.
   This is not legal advice. */
"""

import hashlib
import math
import struct
import sys
import zlib

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "common"))
from rans import rans_decode, CTX_ORDER  # noqa: E402
from idct import idct_2d  # noqa: E402 — integer IDCT (Appendix A, v0.2.1)


class FvcError(Exception):
    pass


MAX_PIXELS = 3_000_000
MAX_DEPTH = 5
MAX_LEAVES_PER_SB = 1024


def half_away(x):
    if x >= 0:
        return math.floor(x + 0.5)
    return -math.floor(-x + 0.5)


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def qstep_of(q_index):
    return max(1, half_away(2.0 ** (q_index / 16.0)))


# ---------------------------------------------------------------- transforms

_DCT_MATRICES = {}


def dct_matrix(n):
    if n not in _DCT_MATRICES:
        m = [[0.0] * n for _ in range(n)]
        s = math.sqrt(2.0 / n)
        for k in range(n):
            ck = (1.0 / math.sqrt(2.0)) if k == 0 else 1.0
            for nn in range(n):
                m[k][nn] = s * ck * math.cos(math.pi * (2 * nn + 1) * k / (2 * n))
        _DCT_MATRICES[n] = m
    return _DCT_MATRICES[n]


def dct_forward(block):
    n = len(block)
    m = dct_matrix(n)
    t = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for nn in range(n):
                s += m[i][nn] * block[nn][j]
            t[i][j] = s
    f = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for nn in range(n):
                s += t[i][nn] * m[j][nn]
            f[i][j] = s
    return f


def dct_inverse(coeffs):
    """Integer IDCT (Appendix A, v0.2.1). Input: 2D list. Output: 2D list."""
    n = len(coeffs)
    # Flatten, round to int, apply integer IDCT, unflatten
    flat = [half_away(coeffs[y][x]) for y in range(n) for x in range(n)]
    out = idct_2d(flat, n)
    return [[out[y * n + x] for x in range(n)] for y in range(n)]


_ZIGZAG = {}


def zigzag(n):
    if n not in _ZIGZAG:
        order = []
        for s in range(2 * n - 1):
            if s % 2 == 0:
                r, c = min(s, n - 1), s - min(s, n - 1)
                while r >= 0 and c < n:
                    order.append((r, c))
                    r -= 1
                    c += 1
            else:
                c, r = min(s, n - 1), s - min(s, n - 1)
                while c >= 0 and r < n:
                    order.append((r, c))
                    r += 1
                    c -= 1
        assert len(order) == n * n
        _ZIGZAG[n] = order
    return _ZIGZAG[n]


# ------------------------------------------------------------- intra predict

def _ref_sample(plane, x, y, w, h):
    if 0 <= x < w and 0 <= y < h:
        return plane[y][x]
    return 512


def intra_predict(mode, plane, bx, by, bw, bh, fw, fh):
    above = [_ref_sample(plane, bx + x, by - 1, fw, fh) for x in range(bw)]
    left = [_ref_sample(plane, bx - 1, by + y, fw, fh) for y in range(bh)]
    above_left = _ref_sample(plane, bx - 1, by - 1, fw, fh)
    a_ext = {}
    a_ext[-1] = above_left
    for x in range(bw):
        a_ext[x] = above[x]
    ar_av = by > 0 and bx + 2 * bw <= fw
    fill = above[bw - 1]
    for x in range(bw, 2 * bw):
        a_ext[x] = _ref_sample(plane, bx + x, by - 1, fw, fh) if ar_av else fill
    l_ext = {}
    l_ext[-1] = above_left
    for y in range(bh):
        l_ext[y] = left[y]
    bl_av = bx > 0 and by + 2 * bh <= fh
    fill = left[bh - 1]
    for y in range(bh, 2 * bh):
        l_ext[y] = _ref_sample(plane, bx - 1, by + y, fw, fh) if bl_av else fill

    def A(i):
        return a_ext[max(-1, min(2 * bw - 1, i))]

    def Ld(i):
        return l_ext[max(-1, min(2 * bh - 1, i))]

    pred = [[0] * bw for _ in range(bh)]
    for y in range(bh):
        for x in range(bw):
            if mode == 0:
                pred[y][x] = above[x]
            elif mode == 1:
                pred[y][x] = left[y]
            elif mode == 2:
                pred[y][x] = A(x + y)
            elif mode == 3:
                pred[y][x] = A(x - y)
            elif mode == 4:
                pred[y][x] = Ld(y + x)
            elif mode == 5:
                pred[y][x] = Ld(y - x)
            elif mode == 6:
                pred[y][x] = A(x + (y >> 1))
            elif mode == 7:
                pred[y][x] = Ld(y + (x >> 1))
            else:
                raise FvcError(f"bad intra mode {mode}")
    return pred


# ------------------------------------------------------- motion compensation

def motion_compensate(ref_plane, mvx, mvy, bx, by, bw, bh, fw, fh):
    """1/8-pel MVs, bilinear subpel filter (spec section 7).

    Reference samples are edge-clamped (NOT 512-filled like intra).
    """
    def clamped(x, y):
        x = min(max(x, 0), fw - 1)
        y = min(max(y, 0), fh - 1)
        return ref_plane[y][x]

    pred = [[0] * bw for _ in range(bh)]
    for y in range(bh):
        for x in range(bw):
            fxf = (bx + x) * 8 + mvx
            fyf = (by + y) * 8 + mvy
            # floor divmod: Python // and % are floor-based, matching spec
            ix, fx = divmod(fxf, 8)
            iy, fy = divmod(fyf, 8)
            p00 = clamped(ix, iy)
            p10 = clamped(ix + 1, iy)
            p01 = clamped(ix, iy + 1)
            p11 = clamped(ix + 1, iy + 1)
            v = (8 - fx) * (8 - fy) * p00 + fx * (8 - fy) * p10 + \
                (8 - fx) * fy * p01 + fx * fy * p11
            pred[y][x] = (v + 32) // 64
    return pred


# ------------------------------------------------------------- bitstream I/O

class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def need(self, n):
        if self.p + n > len(self.d):
            raise FvcError("truncated bitstream")

    def u8(self):
        self.need(1)
        v = self.d[self.p]
        self.p += 1
        return v

    def u16(self):
        self.need(2)
        v = struct.unpack_from("<H", self.d, self.p)[0]
        self.p += 2
        return v

    def i16(self):
        self.need(2)
        v = struct.unpack_from("<h", self.d, self.p)[0]
        self.p += 2
        return v

    def u32(self):
        self.need(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def bytes(self, n):
        self.need(n)
        s = self.d[self.p:self.p + n]
        self.p += n
        return s

    def remaining(self):
        return len(self.d) - self.p


class BitReader:
    def __init__(self, data, pos):
        self.d = data
        self.p = pos
        self.bits_left = 0
        self.cur = 0

    def get_bits(self, n):
        v = 0
        for _ in range(n):
            if self.bits_left == 0:
                if self.p >= len(self.d):
                    raise FvcError("truncated bitstream")
                self.cur = self.d[self.p]
                self.p += 1
                self.bits_left = 8
            self.bits_left -= 1
            v = (v << 1) | ((self.cur >> self.bits_left) & 1)
        return v

    def align(self):
        self.bits_left = 0
        return self.p


def parse_partition(br, x, y, size, depth, leaves):
    t = br.get_bits(2)
    if t == 0:
        if depth >= MAX_DEPTH:
            raise FvcError("partition too deep")
        h = size // 2
        parse_partition(br, x, y, h, depth + 1, leaves)
        parse_partition(br, x + h, y, h, depth + 1, leaves)
        parse_partition(br, x, y + h, h, depth + 1, leaves)
        parse_partition(br, x + h, y + h, h, depth + 1, leaves)
    elif t == 1:
        leaves.append((x, y, size))
    else:
        raise FvcError(f"reserved partition node {t}")


def decode_frame(data, reference):
    r = Reader(data)
    frame_type = r.u8()
    if frame_type > 1:
        raise FvcError(f"unsupported frame type {frame_type}")
    w = r.u16()
    h = r.u16()
    if w < 128 or h < 128 or w % 128 != 0 or h % 128 != 0:
        raise FvcError("bad dimensions")
    if w * h > MAX_PIXELS:
        raise FvcError("frame too large")
    color = r.u8()
    if color != 0:
        raise FvcError("unsupported color")
    if r.u8() != 0:
        raise FvcError("loop filter on")
    r.u8()  # reserved

    if frame_type == 1 and reference is None:
        raise FvcError("INTER with no reference")

    num_sb = (w // 128) * (h // 128)
    blocks = []
    # Parse all SB partitions first (spec section 4), then headers (section 5).
    sb_leaves = []
    for sby in range(0, h, 128):
        for sbx in range(0, w, 128):
            br = BitReader(data, r.p)
            leaves = []
            parse_partition(br, sbx, sby, 128, 0, leaves)
            r.p = br.align()
            if len(leaves) > MAX_LEAVES_PER_SB:
                raise FvcError("too many blocks")
            sb_leaves.append(leaves)
    if sum(len(v) for v in sb_leaves) > num_sb * MAX_LEAVES_PER_SB:
        raise FvcError("too many blocks")
    blocks = []
    for leaves in sb_leaves:
        raster_leaves = sorted(leaves, key=lambda b: (b[1], b[0]))
        headers = {}
        for (bx, by, bs) in raster_leaves:
            intra_flag = r.u8()
            if intra_flag == 1:
                mode = r.u8()
                if mode > 7:
                    raise FvcError(f"bad intra mode {mode}")
                pred = ("intra", mode)
            elif intra_flag == 0:
                ref_idx = r.u8()
                if ref_idx != 0:
                    raise FvcError(f"bad ref_idx {ref_idx}")
                mvx = r.i16()
                mvy = r.i16()
                if r.u8() != 0:
                    raise FvcError("warp on")
                pred = ("inter", mvx, mvy)
            else:
                raise FvcError(f"bad intra_flag {intra_flag}")
            q_index = r.u8()
            skip_luma = r.u8() != 0
            skip_chroma = r.u8() != 0
            headers[(bx, by)] = (pred, q_index, skip_luma, skip_chroma)
        for (bx, by, bs) in leaves:
            (pred, q_index, skip_luma, skip_chroma) = headers[(bx, by)]
            blocks.append((bx, by, bs, pred, q_index, skip_luma, skip_chroma))

    # Blocks arrive in partition-preorder (bitstream order), which is causal.
    # Coefficients are enumerated in frame-raster order (spec section 9),
    # so build tx_list via a raster-sorted index; reconstruct in preorder.
    raster_idx = sorted(range(len(blocks)), key=lambda i: (blocks[i][1], blocks[i][0]))

    tx_list = []
    block_tx_range = [(0, 0)] * len(blocks)
    for bi in raster_idx:
        (bx, by, bs, pred, q_index, skip_luma, skip_chroma) = blocks[bi]
        start = len(tx_list)
        tn = min(bs, 32)
        if not skip_luma:
            for ty in range(0, bs, tn):
                for tx in range(0, bs, tn):
                    tx_list.append((0, bx + tx, by + ty, tn))
        if not skip_chroma:
            cbs = bs // 2
            ctn = min(cbs, 32)
            # Spec section 9: all U transforms raster, then all V transforms raster.
            for p in (1, 2):
                for ty in range(0, cbs, ctn):
                    for tx in range(0, cbs, ctn):
                        tx_list.append((p, bx // 2 + tx, by // 2 + ty, ctn))
        if len(tx_list) > len(blocks) * 24:
            raise FvcError("too many symbols")
        block_tx_range[bi] = (start, len(tx_list))

    streams = {}
    for ctx in CTX_ORDER:
        final_state = r.u32()
        nbytes = r.u32()
        if nbytes > r.remaining():
            raise FvcError("stream longer than input")
        want_crc = r.u32()
        blob = r.bytes(nbytes)
        if zlib.crc32(blob) != want_crc:
            raise FvcError(f"checksum mismatch in {ctx} stream")
        streams[ctx] = (final_state, blob)
    n_esc = r.u32()
    n_dc = len(tx_list)
    n_low = sum(min(16, tn * tn - 1) for (_, _, _, tn) in tx_list)
    n_high = sum(max(0, tn * tn - 1 - 16) for (_, _, _, tn) in tx_list)
    if n_esc > n_dc + n_low + n_high:
        raise FvcError("too many escapes")
    escapes = [r.u16() for _ in range(n_esc)]

    try:
        dc_syms = rans_decode(*streams["DC"], n_dc, "DC")
        low_syms = rans_decode(*streams["AC_LOW"], n_low, "AC_LOW")
        high_syms = rans_decode(*streams["AC_HIGH"], n_high, "AC_HIGH")
    except AssertionError as e:
        raise FvcError(f"rANS failure: {e}")

    esc_pos = 0

    def take_escape():
        nonlocal esc_pos
        if esc_pos >= len(escapes):
            raise FvcError("escape overrun")
        v = escapes[esc_pos]
        esc_pos += 1
        return v

    qcoeff_blocks = [[0] * (tn * tn) for (_, _, _, tn) in tx_list]
    pending_levels = []  # (tx_idx, scan_pos, level), sign order per spec:
                         # DC signs (tx order), then AC signs (tx order,
                         # zigzag scan order within = low/high interleaved)
    di = li = hi = 0
    for ti, (_, _, _, tn) in enumerate(tx_list):
        s = dc_syms[di]; di += 1
        lv = s if s < 16 else take_escape()
        qcoeff_blocks[ti][0] = lv
        if lv:
            pending_levels.append((ti, 0, lv))
    for ti, (_, _, _, tn) in enumerate(tx_list):
        for pos in range(1, min(16, tn * tn - 1) + 1):
            s = low_syms[li]; li += 1
            lv = s if s < 16 else take_escape()
            qcoeff_blocks[ti][pos] = lv
    for ti, (_, _, _, tn) in enumerate(tx_list):
        for pos in range(17, tn * tn):
            s = high_syms[hi]; hi += 1
            lv = s if s < 16 else take_escape()
            qcoeff_blocks[ti][pos] = lv
    for ti, (_, _, _, tn) in enumerate(tx_list):
        coeffs = qcoeff_blocks[ti]
        for pos in range(1, tn * tn):
            if coeffs[pos]:
                pending_levels.append((ti, pos, coeffs[pos]))
    if esc_pos != n_esc:
        raise FvcError("escape count mismatch")

    try:
        sign_syms = rans_decode(*streams["SIGN"], len(pending_levels), "SIGN")
    except AssertionError as e:
        raise FvcError(f"rANS failure: {e}")
    for (ti, pos, lv), ss in zip(pending_levels, sign_syms):
        qcoeff_blocks[ti][pos] = lv if ss == 0 else -lv

    planes = [
        [[512] * w for _ in range(h)],
        [[512] * (w // 2) for _ in range(h // 2)],
        [[512] * (w // 2) for _ in range(h // 2)],
    ]
    fws = [w, w // 2, w // 2]
    fhs = [h, h // 2, h // 2]
    # Reconstruct in partition-preorder (causal); ti via raster-order tx range.
    for bi, (bx, by, bs, pred, q_index, skip_luma, skip_chroma) in enumerate(blocks):
        ti = block_tx_range[bi][0]
        qs = qstep_of(q_index)
        tn = min(bs, 32)
        # luma prediction
        if pred[0] == "intra":
            pred_y = intra_predict(pred[1], planes[0], bx, by, bs, bs, w, h)
        else:
            _, mvx, mvy = pred
            pred_y = motion_compensate(reference["planes"][0], mvx, mvy,
                                      bx, by, bs, bs, w, h)
        for ty in range(0, bs, tn):
            for tx in range(0, bs, tn):
                if not skip_luma:
                    coeffs = qcoeff_blocks[ti]
                    ti += 1
                    if any(coeffs):
                        zz = zigzag(tn)
                        mat = [[0] * tn for _ in range(tn)]
                        for pos, (qy, qx) in enumerate(zz):
                            mat[qy][qx] = coeffs[pos] * qs
                        resid = dct_inverse(mat)
                    else:
                        resid = [[0.0] * tn for _ in range(tn)]
                    for y in range(tn):
                        row = planes[0][by + ty + y]
                        pr = pred_y[ty + y]
                        rs = resid[y]
                        for x in range(tn):
                            row[bx + tx + x] = clip(
                                half_away(pr[tx + x] + rs[x]), 0, 1023)
                else:
                    for y in range(tn):
                        row = planes[0][by + ty + y]
                        pr = pred_y[ty + y]
                        for x in range(tn):
                            row[bx + tx + x] = clip(pr[tx + x], 0, 1023)
        # chroma
        cbs = bs // 2
        cbx, cby = bx // 2, by // 2
        for p in (1, 2):
            if pred[0] == "intra":
                pred_c = intra_predict(pred[1], planes[p], cbx, cby, cbs, cbs,
                                       fws[p], fhs[p])
            else:
                _, mvx, mvy = pred
                pred_c = motion_compensate(reference["planes"][p], mvx // 2,
                                           mvy // 2, cbx, cby, cbs, cbs,
                                           fws[p], fhs[p])
            ctn = min(cbs, 32)
            for ty in range(0, cbs, ctn):
                for tx in range(0, cbs, ctn):
                    if not skip_chroma:
                        coeffs = qcoeff_blocks[ti]
                        ti += 1
                        if any(coeffs):
                            zz = zigzag(ctn)
                            mat = [[0] * ctn for _ in range(ctn)]
                            for pos, (qy, qx) in enumerate(zz):
                                mat[qy][qx] = coeffs[pos] * qs
                            resid = dct_inverse(mat)
                        else:
                            resid = [[0.0] * ctn for _ in range(ctn)]
                        for y in range(ctn):
                            row = planes[p][cby + ty + y]
                            pr = pred_c[ty + y]
                            rs = resid[y]
                            for x in range(ctn):
                                row[cbx + tx + x] = clip(
                                    half_away(pr[tx + x] + rs[x]), 0, 1023)
                    else:
                        for y in range(ctn):
                            row = planes[p][cby + ty + y]
                            pr = pred_c[ty + y]
                            for x in range(ctn):
                                row[cbx + tx + x] = clip(pr[tx + x], 0, 1023)
    # (ti is per-block now; ranges partition tx_list exactly by construction)
    if r.p != len(data):
        raise FvcError("trailing data after frame")
    return {"width": w, "height": h, "frame_type": frame_type, "planes": planes}


def decode_file(data):
    r = Reader(data)
    if len(data) < 4 or data[0:4] != b"FVC1":
        raise FvcError("bad magic")
    r.p = 4
    version = r.u8()
    if version != 2:
        raise FvcError(f"unsupported version {version}")
    num_frames = r.u32()
    if num_frames == 0:
        raise FvcError("no frames")
    frames = []
    reference = None
    for _ in range(num_frames):
        frame_len = r.u32()
        fdata = r.bytes(frame_len)
        frame = decode_frame(fdata, reference)
        if frame["frame_type"] == 1:
            rf = reference
            if frame["width"] != rf["width"] or frame["height"] != rf["height"]:
                raise FvcError("dimension mismatch")
        reference = frame
        frames.append(frame)
    return frames


def write_yuv(path, frame):
    with open(path, "wb") as f:
        for plane in frame["planes"]:
            for row in plane:
                f.write(struct.pack("<%dH" % len(row), *row))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


if __name__ == "__main__":
    src, prefix = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        data = f.read()
    frames = decode_file(data)
    for i, frame in enumerate(frames):
        dst = f"{prefix}_f{i}.yuv"
        write_yuv(dst, frame)
        ftype = "KEY" if frame["frame_type"] == 0 else "INTER"
        print(f"frame {i} ({ftype}) {frame['width']}x{frame['height']} -> {dst}")
        print("  sha256:", sha256_file(dst))
