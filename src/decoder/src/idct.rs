//! Integer IDCT (Appendix A) — replaces f64 reference for speed.
//!
//! 16-bit fixed-point LLM-style factorization via even-odd decomposition.
//! Bit-identical across Rust/Python by construction (all ops specified in
//! spec Appendix A). Safe Rust; SIMD in `simd` submodule only.

/// Saturate i32 to i16 range.
#[inline]
fn sat16(x: i32) -> i16 {
    x.clamp(-32768, 32767) as i16
}

/// 1/sqrt(2) in Q14: round(2^14 / sqrt(2)) = 11585
const INV_SQRT2_Q14: i32 = 11585;

/// W4: 4-point DCT-II matrix * 2^14, [k][n]
const W4: [[i32; 4]; 4] = [
    [8192, 8192, 8192, 8192],
    [10703, 4433, -4433, -10703],
    [8192, -8192, -8192, 8192],
    [4433, -10703, 10703, -4433],
];

/// V2: 2-point DCT-IV matrix * 2^14, [k][n]
const V2: [[i32; 2]; 2] = [
    [15137, 6270],
    [6270, -15137],
];

/// V4: 4-point DCT-IV matrix * 2^14, [k][n]
const V4: [[i32; 4]; 4] = [
    [11363, 9633, 6436, 2260],
    [9633, -2260, -11363, -6436],
    [6436, -11363, 2260, 9633],
    [2260, -6436, 9633, -11363],
];

/// V8: 8-point DCT-IV matrix * 2^14, [k][n]
const V8: [[i32; 8]; 8] = [
    [8153, 7839, 7225, 6333, 5197, 3862, 2378, 803],
    [7839, 5197, 803, -3862, -7225, -8153, -6333, -2378],
    [7225, 803, -6333, -7839, -2378, 5197, 8153, 3862],
    [6333, -3862, -7839, 803, 8153, 2378, -7225, -5197],
    [5197, -7225, -2378, 8153, -803, -7839, 3862, 6333],
    [3862, -8153, 5197, 2378, -7839, 6333, 803, -7225],
    [2378, -6333, 8153, -7225, 3862, 803, -5197, 7839],
    [803, -2378, 3862, -5197, 6333, -7225, 7839, -8153],
];

/// V16: 16-point DCT-IV matrix * 2^14, [k][n]
const V16: [[i32; 16]; 16] = [
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
];

/// 4-point 1D IDCT, 32-bit intermediate output.
fn idct4_32(dq: &[i32]) -> [i32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        // Cache AVX2 detection (OnceLock is lock-free after init)
        static HAS_AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let has_avx2 = *HAS_AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"));
        if has_avx2 {
            // SAFETY: AVX2 detected at runtime.
            unsafe { return simd::idct4_32_avx2(dq); }
        }
    }
    idct4_32_scalar(dq)
}

/// Scalar 4-point IDCT.
fn idct4_32_scalar(dq: &[i32]) -> [i32; 4] {
    let mut r = [0i32; 4];
    for n in 0..4 {
        let mut s: i32 = 0;
        for k in 0..4 {
            s += W4[k][n] * dq[k];
        }
        r[n] = (s + 8192) >> 14;
    }
    r
}

/// M-point DCT-IV, 32-bit intermediate output.
/// Writes to output slice.
fn idct_iv_32_into(dq: &[i32], out: &mut [i32], m: usize) {
    // Use AVX2 for 8-point and 16-point if available
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                if m == 8 {
                    let r = simd::idct_iv8_avx2(dq);
                    out[..8].copy_from_slice(&r);
                    return;
                } else if m == 16 {
                    let r = simd::idct_iv16_avx2(dq);
                    out[..16].copy_from_slice(&r);
                    return;
                }
            }
        }
    }
    
    for n in 0..m {
        let mut s: i32 = 0;
        for k in 0..m {
            let w = match m {
                2 => V2[k][n],
                4 => V4[k][n],
                8 => V8[k][n],
                16 => V16[k][n],
                _ => unreachable!(),
            };
            s += w * dq[k];
        }
        out[n] = (s + 8192) >> 14;
    }
}

/// M-point DCT-IV, 32-bit intermediate output.
fn idct_iv_32(dq: &[i32], m: usize) -> Vec<i32> {
    let mut out = vec![0i32; m];
    idct_iv_32_into(dq, &mut out, m);
    out
}

/// 1D N-point IDCT, 32-bit intermediate. N must be 4, 8, 16, or 32.
/// Writes to output slice (must have length >= n). Returns number of elements written.
fn idct_1d_32_into(dq: &[i32], out: &mut [i32]) {
    let n = dq.len();
    if n == 4 {
        let r = idct4_32(dq);
        out[..4].copy_from_slice(&r);
        return;
    }
    // Even-odd decomposition (Appendix A.2)
    let m = n / 2;
    // Stack buffers for even/odd (max 16 each)
    let mut e = [0i32; 16];
    let mut o = [0i32; 16];
    for j in 0..m {
        e[j] = dq[2 * j];
        o[j] = dq[2 * j + 1];
    }
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    idct_1d_32_into(&e[..m], &mut a[..m]);
    idct_iv_32_into(&o[..m], &mut b[..m], m);
    for i in 0..m {
        let even = (a[i] * INV_SQRT2_Q14 + 8192) >> 14;
        let odd = (b[i] * INV_SQRT2_Q14 + 8192) >> 14;
        out[i] = even + odd;
        out[n - 1 - i] = even - odd;
    }
}

/// 1D N-point IDCT, 32-bit intermediate. N must be 4, 8, 16, or 32.
fn idct_1d_32(dq: &[i32]) -> Vec<i32> {
    let n = dq.len();
    let mut out = vec![0i32; n];
    idct_1d_32_into(dq, &mut out);
    out
}

/// 1D N-point integer IDCT. Input: i32 dequantized coeffs. Output: i16 residuals.
/// N must be 4, 8, 16, or 32.
pub fn idct_1d(dq: &[i32]) -> Vec<i16> {
    idct_1d_32(dq).iter().map(|&v| sat16(v)).collect()
}

/// 2D N×N integer IDCT. Input: N*N i32 coeffs (row-major). Output: N*N i16 residuals.
pub fn idct_2d(coeffs: &[i32], n: usize) -> Vec<i16> {
    assert!(n == 4 || n == 8 || n == 16 || n == 32);
    assert_eq!(coeffs.len(), n * n);

    // Use SIMD path if available, else scalar.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected at runtime; simd module is the only
            // unsafe code in the crate.
            return unsafe { simd::idct_2d_avx2(coeffs, n) };
        }
    }

    idct_2d_scalar(coeffs, n)
}

/// Scalar 2D IDCT (also the fallback for non-x86_64).
/// Uses stack arrays to avoid heap allocation in hot loop.
fn idct_2d_scalar(coeffs: &[i32], n: usize) -> Vec<i16> {
    // Rows: 32-bit intermediate, max 32x32 = 1024
    let mut tmp = [0i32; 1024];
    let mut row_buf = [0i32; 32];
    for y in 0..n {
        // Copy row into stack buffer
        for x in 0..n {
            row_buf[x] = coeffs[y * n + x];
        }
        let out = idct_1d_32(&row_buf[..n]);
        for (i, &v) in out.iter().enumerate() {
            tmp[y * n + i] = v;
        }
    }
    // Columns: output directly to row-major Vec (single allocation)
    let mut out = vec![0i16; n * n];
    let mut col_buf = [0i32; 32];
    for x in 0..n {
        for y in 0..n {
            col_buf[y] = tmp[y * n + x];
        }
        let col_out = idct_1d_32(&col_buf[..n]);
        for y in 0..n {
            out[y * n + x] = sat16(col_out[y]);
        }
    }
    out
}

/// SIMD module — the ONLY place unsafe is allowed in this crate.
#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;
    use super::{W4, V2, V4, V8, V16};

    /// 4-point IDCT via AVX2. Bit-identical to scalar (same ops, same order).
    /// Processes 2 outputs at a time (8 int32 multiplies per instruction).
    #[target_feature(enable = "avx2")]
    pub unsafe fn idct4_32_avx2(dq: &[i32]) -> [i32; 4] {
        // W4[k][n]: k=row (input idx), n=col (output idx)
        // out[n] = sum_k W4[k][n] * dq[k]
        
        // We'll compute out0,out1 in parallel, then out2,out3.
        // For out0,out1: need W4[k][0], W4[k][1] for k=0..3.
        
        let mut result = [0i32; 4];
        
        // Process outputs 0,1
        {
            // Load matrix elements for outputs 0,1, interleaved:
            // [W00, W01, W10, W11, W20, W21, W30, W31] where Wij = W4[i][j]
            let mat01 = _mm256_set_epi32(
                W4[3][1], W4[3][0], W4[2][1], W4[2][0],
                W4[1][1], W4[1][0], W4[0][1], W4[0][0],
            );
            // We need to multiply by dq[k] and sum across k.
            // Broadcast each dq[k] and multiply by the corresponding matrix elements.
            
            // For k=0: need [W00*dq0, W01*dq0, ...] — but mat01 has W00,W01,W10,W11...
            // Actually, let's do it differently: 4 separate broadcasts, 4 multiplies, 3 adds.
            
            let dq0 = _mm256_set1_epi32(dq[0]);
            let dq1 = _mm256_set1_epi32(dq[1]);
            let dq2 = _mm256_set1_epi32(dq[2]);
            let dq3 = _mm256_set1_epi32(dq[3]);
            
            // Matrix for k=0: [W00, W01, 0,0,0,0,0,0] — we only need first 2
            // This is getting messy. Let me do 8 outputs at once for 2 blocks.
            
            // Simpler: process one 4x4 at a time using 4x 8-wide ops
            // Each op does 2 outputs × 4 inputs = 8 multiplies
            
            // out0,out1:
            // sum_k W4[k][0]*dq[k], sum_k W4[k][1]*dq[k]
            let m0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[0][1], W4[0][0]);
            let m1 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[1][1], W4[1][0]);
            let m2 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[2][1], W4[2][0]);
            let m3 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[3][1], W4[3][0]);
            
            let p0 = _mm256_mullo_epi32(m0, dq0);
            let p1 = _mm256_mullo_epi32(m1, dq1);
            let p2 = _mm256_mullo_epi32(m2, dq2);
            let p3 = _mm256_mullo_epi32(m3, dq3);
            
            let sum01 = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
            // sum01 = [s0, s1, 0,0,0,0,0,0] where s0=out0_num, s1=out1_num
            let s0 = _mm256_extract_epi32(sum01, 0);
            let s1 = _mm256_extract_epi32(sum01, 1);
            result[0] = (s0 + 8192) >> 14;
            result[1] = (s1 + 8192) >> 14;
        }
        
        // Process outputs 2,3
        {
            let m0 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[0][3], W4[0][2]);
            let m1 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[1][3], W4[1][2]);
            let m2 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[2][3], W4[2][2]);
            let m3 = _mm256_set_epi32(0, 0, 0, 0, 0, 0, W4[3][3], W4[3][2]);
            
            let dq0 = _mm256_set1_epi32(dq[0]);
            let dq1 = _mm256_set1_epi32(dq[1]);
            let dq2 = _mm256_set1_epi32(dq[2]);
            let dq3 = _mm256_set1_epi32(dq[3]);
            
            let p0 = _mm256_mullo_epi32(m0, dq0);
            let p1 = _mm256_mullo_epi32(m1, dq1);
            let p2 = _mm256_mullo_epi32(m2, dq2);
            let p3 = _mm256_mullo_epi32(m3, dq3);
            
            let sum23 = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
            let s2 = _mm256_extract_epi32(sum23, 0);
            let s3 = _mm256_extract_epi32(sum23, 1);
            result[2] = (s2 + 8192) >> 14;
            result[3] = (s3 + 8192) >> 14;
        }
        
        result
    }

    /// 8-point DCT-IV via AVX2. Bit-identical to scalar.
    #[target_feature(enable = "avx2")]
    pub unsafe fn idct_iv8_avx2(dq: &[i32]) -> [i32; 8] {
        let mut result = [0i32; 8];
        // V8[k][n]: k=input, n=output
        // out[n] = sum_k V8[k][n] * dq[k]
        // Process 8 outputs, each with 8 multiplies.
        // For each output n, load the 8 matrix elements V8[0..7][n] and multiply by dq[0..7].
        
        // Load dq into 2 registers (we'll broadcast)
        // Actually, simpler: for each n, do 8-wide multiply and horizontal sum.
        
        for n in 0..8 {
            // Load column n of V8: V8[0][n], V8[1][n], ..., V8[7][n]
            let col = _mm256_set_epi32(
                V8[7][n], V8[6][n], V8[5][n], V8[4][n],
                V8[3][n], V8[2][n], V8[1][n], V8[0][n],
            );
            // Load dq[0..7]
            let v = _mm256_loadu_si256(dq.as_ptr() as *const __m256i);
            let prod = _mm256_mullo_epi32(col, v);
            // Horizontal sum of 8 int32s
            let sum1 = _mm256_hadd_epi32(prod, prod);
            let sum2 = _mm256_hadd_epi32(sum1, sum1);
            let s_lo = _mm256_extract_epi32(sum2, 0);
            let s_hi = _mm256_extract_epi32(sum2, 4);
            let s = s_lo + s_hi;
            result[n] = (s + 8192) >> 14;
        }
        result
    }

    /// 16-point DCT-IV via AVX2. Bit-identical to scalar.
    /// Processes 8 outputs at a time (2x 8-wide).
    #[target_feature(enable = "avx2")]
    pub unsafe fn idct_iv16_avx2(dq: &[i32]) -> [i32; 16] {
        let mut result = [0i32; 16];
        // For each output n, sum_k V16[k][n] * dq[k]
        // Do 8 outputs at a time using 2 AVX2 registers for the 16 inputs.
        
        for n in 0..16 {
            // Load 16 matrix elements: V16[0..15][n]
            // Split into two 8-element halves
            let col_lo = _mm256_set_epi32(
                V16[7][n], V16[6][n], V16[5][n], V16[4][n],
                V16[3][n], V16[2][n], V16[1][n], V16[0][n],
            );
            let col_hi = _mm256_set_epi32(
                V16[15][n], V16[14][n], V16[13][n], V16[12][n],
                V16[11][n], V16[10][n], V16[9][n], V16[8][n],
            );
            let v_lo = _mm256_loadu_si256(dq.as_ptr() as *const __m256i);
            let v_hi = _mm256_loadu_si256(dq.as_ptr().add(8) as *const __m256i);
            
            let p_lo = _mm256_mullo_epi32(col_lo, v_lo);
            let p_hi = _mm256_mullo_epi32(col_hi, v_hi);
            
            // Horizontal sum each
            let s1_lo = _mm256_hadd_epi32(p_lo, p_lo);
            let s2_lo = _mm256_hadd_epi32(s1_lo, s1_lo);
            let sum_lo = _mm256_extract_epi32(s2_lo, 0) + _mm256_extract_epi32(s2_lo, 4);
            
            let s1_hi = _mm256_hadd_epi32(p_hi, p_hi);
            let s2_hi = _mm256_hadd_epi32(s1_hi, s1_hi);
            let sum_hi = _mm256_extract_epi32(s2_hi, 0) + _mm256_extract_epi32(s2_hi, 4);
            
            let s = sum_lo + sum_hi;
            result[n] = (s + 8192) >> 14;
        }
        result
    }

    pub unsafe fn idct_2d_avx2(coeffs: &[i32], n: usize) -> Vec<i16> {
        super::idct_2d_scalar(coeffs, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct4_zero() {
        let z = [0i32; 4];
        let r = idct_1d(&z);
        assert!(r.iter().all(|&v| v == 0));
    }

    #[test]
    fn idct8_zero() {
        let z = [0i32; 8];
        let r = idct_1d(&z);
        assert!(r.iter().all(|&v| v == 0));
    }

    #[test]
    fn idct_dc_only() {
        // DC-only: constant output. DC = 1000, N=8.
        // Expected: 1000 * sqrt(1/8) ≈ 353.55, in Q14 fixed-point.
        let mut dq = [0i32; 8];
        dq[0] = 1000;
        let r = idct_1d(&dq);
        // All outputs should be approximately equal (constant block).
        let v0 = r[0] as i32;
        for &v in &r[1..] {
            assert!((v as i32 - v0).abs() <= 2, "DC should be constant: {:?}", r);
        }
    }
}
