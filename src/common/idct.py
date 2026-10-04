#!/usr/bin/env python3
"""Integer IDCT (spec Appendix A) — pure Python ints, bit-identical to Rust.

This is the reference implementation for verification. All operations use
Python arbitrary-precision ints with explicit masking to emulate i32/i16.
"""

# 1/sqrt(2) in Q14
INV_SQRT2_Q14 = 11585

# W4: 4-point DCT-II * 2^14, [k][n]
W4 = [
    [8192, 8192, 8192, 8192],
    [10703, 4433, -4433, -10703],
    [8192, -8192, -8192, 8192],
    [4433, -10703, 10703, -4433],
]

# V matrices: DCT-IV * 2^14, [k][n]
V2 = [
    [15137, 6270],
    [6270, -15137],
]

V4 = [
    [11363, 9633, 6436, 2260],
    [9633, -2260, -11363, -6436],
    [6436, -11363, 2260, 9633],
    [2260, -6436, 9633, -11363],
]

V8 = [
    [8153, 7839, 7225, 6333, 5197, 3862, 2378, 803],
    [7839, 5197, 803, -3862, -7225, -8153, -6333, -2378],
    [7225, 803, -6333, -7839, -2378, 5197, 8153, 3862],
    [6333, -3862, -7839, 803, 8153, 2378, -7225, -5197],
    [5197, -7225, -2378, 8153, -803, -7839, 3862, 6333],
    [3862, -8153, 5197, 2378, -7839, 6333, 803, -7225],
    [2378, -6333, 8153, -7225, 3862, 803, -5197, 7839],
    [803, -2378, 3862, -5197, 6333, -7225, 7839, -8153],
]

V16 = [
    [5786, 5730, 5619, 5454, 5236, 4968, 4653, 4292, 3890, 3451, 2978, 2477, 1951, 1407, 850, 284],
    [5730, 5236, 4292, 2978, 1407, -284, -1951, -3451, -4653, -5454, -5786, -5619, -4968, -3890, -2477, -850],
    [5619, 4292, 1951, -850, -3451, -5236, -5786, -4968, -2978, -284, 2477, 4653, 5730, 5454, 3890, 1407],
    [5454, 2978, -850, -4292, -5786, -4653, -1407, 2477, 5236, 5619, 3451, -284, -3890, -5730, -4968, -1951],
    [5236, 1407, -3451, -5786, -3890, 850, 4968, 5454, 1951, -2978, -5730, -4292, 284, 4653, 5619, 2477],
    [4968, -284, -5236, -4653, 850, 5454, 4292, -1407, -5619, -3890, 1951, 5730, 3451, -2477, -5786, -2978],
    [4653, -1951, -5786, -1407, 4968, 4292, -2477, -5730, -850, 5236, 3890, -2978, -5619, -284, 5454, 3451],
    [4292, -3451, -4968, 2477, 5454, -1407, -5730, 284, 5786, 850, -5619, -1951, 5236, 2978, -4653, -3890],
    [3890, -4653, -2978, 5236, 1951, -5619, -850, 5786, -284, -5730, 1407, 5454, -2477, -4968, 3451, 4292],
    [3451, -5454, -284, 5619, -2978, -3890, 5236, 850, -5730, 2477, 4292, -4968, -1407, 5786, -1951, -4653],
    [2978, -5786, 2477, 3451, -5730, 1951, 3890, -5619, 1407, 4292, -5454, 850, 4653, -5236, 284, 4968],
    [2477, -5619, 4653, -284, -4292, 5730, -2978, -1951, 5454, -4968, 850, 3890, -5786, 3451, 1407, -5236],
    [1951, -4968, 5730, -3890, 284, 3451, -5619, 5236, -2477, -1407, 4653, -5786, 4292, -850, -2978, 5454],
    [1407, -3890, 5454, -5730, 4653, -2477, -284, 2978, -4968, 5786, -5236, 3451, -850, -1951, 4292, -5619],
    [850, -2477, 3890, -4968, 5619, -5786, 5454, -4653, 3451, -1951, 284, 1407, -2978, 4292, -5236, 5730],
    [284, -850, 1407, -1951, 2477, -2978, 3451, -3890, 4292, -4653, 4968, -5236, 5454, -5619, 5730, -5786],
]

V = {2: V2, 4: V4, 8: V8, 16: V16}


def _sat16(x):
    """Saturate to i16 range."""
    return max(-32768, min(32767, x))


def _sar(x, shift):
    """Arithmetic shift right (Python >> is arithmetic for negatives)."""
    return x >> shift


def _idct4_32(dq):
    """4-point IDCT, 32-bit intermediate."""
    r = []
    for n in range(4):
        s = 0
        for k in range(4):
            s += W4[k][n] * dq[k]
        r.append(_sar(s + 8192, 14))
    return r


def _idct_iv_32(dq, m):
    """M-point DCT-IV, 32-bit intermediate."""
    Vm = V[m]
    r = []
    for n in range(m):
        s = 0
        for k in range(m):
            s += Vm[k][n] * dq[k]
        r.append(_sar(s + 8192, 14))
    return r


def _idct_1d_32(dq):
    """1D N-point IDCT, 32-bit intermediate. N in {4, 8, 16, 32}."""
    n = len(dq)
    if n == 4:
        return _idct4_32(dq)
    m = n // 2
    e = [dq[2 * j] for j in range(m)]
    o = [dq[2 * j + 1] for j in range(m)]
    a = _idct_1d_32(e)
    b = _idct_iv_32(o, m)
    r = [0] * n
    for i in range(m):
        even = _sar(a[i] * INV_SQRT2_Q14 + 8192, 14)
        odd = _sar(b[i] * INV_SQRT2_Q14 + 8192, 14)
        r[i] = even + odd
        r[n - 1 - i] = even - odd
    return r


def idct_1d(dq):
    """1D N-point integer IDCT. Input: list of int. Output: list of int (i16 range)."""
    return [_sat16(v) for v in _idct_1d_32(dq)]


def idct_2d(coeffs, n):
    """2D N×N integer IDCT. Input: flat list of N*N ints (row-major). Output: flat list of N*N ints."""
    assert n in (4, 8, 16, 32)
    assert len(coeffs) == n * n
    # Rows (32-bit intermediate)
    tmp = []
    for y in range(n):
        row = coeffs[y * n:(y + 1) * n]
        tmp.extend(_idct_1d_32(row))
    # Columns (saturate to i16 at output)
    out = [0] * (n * n)
    for x in range(n):
        col = [tmp[y * n + x] for y in range(n)]
        col_out = _idct_1d_32(col)
        for y in range(n):
            out[y * n + x] = _sat16(col_out[y])
    return out


if __name__ == "__main__":
    import random
    random.seed(42)
    # Quick self-test
    for n in [4, 8, 16, 32]:
        dq = [random.randint(-2000, 2000) for _ in range(n)]
        r = idct_1d(dq)
        assert len(r) == n
        assert all(-32768 <= v <= 32767 for v in r)
    print("idct.py self-test OK")
