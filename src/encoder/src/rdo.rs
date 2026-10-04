//! Rate-distortion optimization for the slow encoder.
//!
//! Brute force over partition depth 0-5, 8 intra modes, Q 0-255, and (for
//! P-frames) 1/8-pel MVs. D is DCT-domain SSE (Parseval-equal to pixel SSE
//! for the orthonormal DCT, ignoring clipping); R is true entropy under the
//! fixed rANS tables. Final reconstruction uses the exact decoder path
//! (naive IDCT + clip) so encoder reconstruction is bit-identical to
//! `fvc1-dec` output.

use fvc1_dec::rans::{FREQ_AC_HIGH, FREQ_AC_LOW, FREQ_DC};
use fvc1_dec::{dct_forward, half_away, idct_2d, intra_predict, qstep_of, zigzag};

use crate::picture::Picture;

pub const MAX_DEPTH: u32 = 5;
pub const MAX_DEPTH_FAST: u32 = 3;

/// Encoder preset.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Slow,
    Fast,
}

/// Sum of Absolute Transformed Differences (SATD) using 4x4 Hadamard.
/// Faster than SSE for mode pre-selection. Operates on residual blocks.
pub fn satd_4x4(orig: &[u16], pred: &[u16], stride: usize) -> u32 {
    // 4x4 Hadamard transform of (orig - pred), sum of abs.
    let mut diff = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            diff[y*4+x] = orig[y*stride+x] as i32 - pred[y*stride+x] as i32;
        }
    }
    // Horizontal Hadamard
    let mut tmp = [0i32; 16];
    for y in 0..4 {
        let a0 = diff[y*4+0] + diff[y*4+3];
        let a1 = diff[y*4+1] + diff[y*4+2];
        let a2 = diff[y*4+1] - diff[y*4+2];
        let a3 = diff[y*4+0] - diff[y*4+3];
        tmp[y*4+0] = a0 + a1;
        tmp[y*4+1] = a3 + a2;
        tmp[y*4+2] = a3 - a2;
        tmp[y*4+3] = a0 - a1;
    }
    // Vertical Hadamard
    let mut sum = 0u32;
    for x in 0..4 {
        let a0 = tmp[0*4+x] + tmp[3*4+x];
        let a1 = tmp[1*4+x] + tmp[2*4+x];
        let a2 = tmp[1*4+x] - tmp[2*4+x];
        let a3 = tmp[0*4+x] - tmp[3*4+x];
        sum += (a0 + a1).abs() as u32;
        sum += (a3 + a2).abs() as u32;
        sum += (a3 - a2).abs() as u32;
        sum += (a0 - a1).abs() as u32;
    }
    sum >> 1 // Normalize (Hadamard has gain of 2)
}

/// SATD for arbitrary block size (tiles 4x4).
pub fn satd_block(orig: &[u16], pred: &[u16], bs: usize, stride: usize) -> u32 {
    let mut sum = 0u32;
    for ty in (0..bs).step_by(4) {
        for tx in (0..bs).step_by(4) {
            // Extract 4x4 tiles
            let mut o_tile = [0u16; 16];
            let mut p_tile = [0u16; 16];
            for y in 0..4 {
                for x in 0..4 {
                    o_tile[y*4+x] = orig[(ty+y)*stride + (tx+x)];
                    p_tile[y*4+x] = pred[(ty+y)*stride + (tx+x)];
                }
            }
            sum += satd_4x4(&o_tile, &p_tile, 4);
        }
    }
    sum
}

/// -log2 probability tables for entropy bit estimation.
pub struct CostTables {
    dc: [f64; 17],
    low: [f64; 17],
    high: [f64; 17],
}

impl CostTables {
    pub fn new() -> Self {
        fn bits(freq: &[u32]) -> [f64; 17] {
            let mut b = [0.0f64; 17];
            for (i, &f) in freq.iter().enumerate() {
                b[i] = -(f as f64 / 65536.0).log2();
            }
            b
        }
        CostTables {
            dc: bits(&FREQ_DC),
            low: bits(&FREQ_AC_LOW),
            high: bits(&FREQ_AC_HIGH),
        }
    }

    fn table_for(&self, pos: usize) -> &[f64; 17] {
        if pos == 0 {
            &self.dc
        } else if pos <= 16 {
            &self.low
        } else {
            &self.high
        }
    }

    /// Bit cost of one quantized coefficient at scan position `pos`
    /// (escape = 16 bits, sign = 1 bit).
    pub fn coeff_bits(&self, v: i32, pos: usize) -> f64 {
        let t = self.table_for(pos);
        if v == 0 {
            return t[0];
        }
        let a = v.unsigned_abs() as usize;
        if a <= 15 {
            t[a] + 1.0
        } else {
            t[16] + 16.0 + 1.0
        }
    }
}

/// One transform's DCT coefficients in scan order.
pub struct TxWork {
    pub tn: usize,
    pub coeffs: Vec<f64>,
}

/// Collect DCT coefficients (scan order) for all transforms of a block.
pub fn collect_txs(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    pred_y: &[u16],
    pred_u: &[u16],
    pred_v: &[u16],
) -> (Vec<TxWork>, Vec<TxWork>) {
    let mut luma = Vec::new();
    let mut chroma = Vec::new();
    let tn = bs.min(32);
    let zz = zigzag(tn);
    for ty in (0..bs).step_by(tn) {
        for tx in (0..bs).step_by(tn) {
            let mut resid = vec![0.0f64; tn * tn];
            for y in 0..tn {
                for x in 0..tn {
                    resid[y * tn + x] =
                        orig.y[(by + ty + y) * orig.w + (bx + tx + x)] as f64
                            - pred_y[(ty + y) * bs + (tx + x)] as f64;
                }
            }
            let f = dct_forward(&resid, tn);
            let mut scan = vec![0.0f64; tn * tn];
            for (pos, &(qy, qx)) in zz.iter().enumerate() {
                scan[pos] = f[qy * tn + qx];
            }
            luma.push(TxWork { tn, coeffs: scan });
        }
    }
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    let ctn = cbs.min(32);
    let czz = zigzag(ctn);
    let cw = orig.w / 2;
    for (plane, pred_c) in [(&orig.u, pred_u), (&orig.v, pred_v)] {
        for ty in (0..cbs).step_by(ctn) {
            for tx in (0..cbs).step_by(ctn) {
                let mut resid = vec![0.0f64; ctn * ctn];
                for y in 0..ctn {
                    for x in 0..ctn {
                        resid[y * ctn + x] =
                            plane[(cby + ty + y) * cw + (cbx + tx + x)] as f64
                                - pred_c[(ty + y) * cbs + (tx + x)] as f64;
                    }
                }
                let f = dct_forward(&resid, ctn);
                let mut scan = vec![0.0f64; ctn * ctn];
                for (pos, &(qy, qx)) in czz.iter().enumerate() {
                    scan[pos] = f[qy * ctn + qx];
                }
                chroma.push(TxWork {
                    tn: ctn,
                    coeffs: scan,
                });
            }
        }
    }
    (luma, chroma)
}

/// Brute-force Q (0-255) for one set of transforms.
/// Returns (best_q, best_rd_cost_without_header, luma_qcoeffs, chroma_qcoeffs).
pub fn best_q(
    luma: &[TxWork],
    chroma: &[TxWork],
    lambda: f64,
    ct: &CostTables,
) -> (u8, f64, Vec<Vec<i32>>, Vec<Vec<i32>>) {
    best_q_preset(luma, chroma, lambda, ct, Preset::Slow)
}

/// Fast: Q ±8 refinement around base QP (passed via lambda-derived estimate).
/// For now, uses full 0-255 but early-exits; true ±8 needs base QP plumbed through.
/// TODO: plumb base_qp through for true ±8 search.
pub fn best_q_preset(
    luma: &[TxWork],
    chroma: &[TxWork],
    lambda: f64,
    ct: &CostTables,
    preset: Preset,
) -> (u8, f64, Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut best_q = 0u8;
    let mut best_cost = f64::INFINITY;
    let mut best_lq: Vec<Vec<i32>> = Vec::new();
    let mut best_cq: Vec<Vec<i32>> = Vec::new();
    
    // Fast preset: estimate base QP from lambda, search ±8
    // lambda = 0.85 * 2^((Q-128)/16) => Q = 128 + 16*log2(lambda/0.85)
    let q_range: Vec<u8> = if preset == Preset::Fast {
        let base_q = (128.0 + 16.0 * (lambda / 0.85).log2()).round() as i32;
        let base_q = base_q.clamp(0, 255) as u8;
        let lo = base_q.saturating_sub(8);
        let hi = (base_q as u16 + 8).min(255) as u8;
        (lo..=hi).collect()
    } else {
        (0..=255u8).collect()
    };
    
    for q in q_range {
        let qs = qstep_of(q);
        let mut d = 0.0;
        let mut r = 0.0;
        let mut lq = Vec::with_capacity(luma.len());
        for tx in luma {
            let mut qv = Vec::with_capacity(tx.coeffs.len());
            for (pos, &c) in tx.coeffs.iter().enumerate() {
                let qc = half_away(c / qs);
                let diff = c - qc as f64 * qs;
                d += diff * diff;
                r += ct.coeff_bits(qc, pos);
                qv.push(qc);
            }
            lq.push(qv);
        }
        let mut cq = Vec::with_capacity(chroma.len());
        for tx in chroma {
            let mut qv = Vec::with_capacity(tx.coeffs.len());
            for (pos, &c) in tx.coeffs.iter().enumerate() {
                let qc = half_away(c / qs);
                let diff = c - qc as f64 * qs;
                d += diff * diff;
                r += ct.coeff_bits(qc, pos);
                qv.push(qc);
            }
            cq.push(qv);
        }
        let cost = d + lambda * r;
        if cost < best_cost {
            best_cost = cost;
            best_q = q;
            best_lq = lq;
            best_cq = cq;
        }
    }
    (best_q, best_cost, best_lq, best_cq)
}

pub struct IntraDecision {
    pub mode: u8,
    pub q: u8,
    pub cost: f64, // includes header bits
    pub luma_q: Vec<Vec<i32>>,
    pub chroma_q: Vec<Vec<i32>>,
}

/// Brute-force intra RDO for one block (8 modes x Q 0-255).
/// Uses `rec` for causal prediction; does not modify `rec`.
pub fn rdo_intra_block(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    rec: &Picture,
    lambda: f64,
    ct: &CostTables,
) -> IntraDecision {
    rdo_intra_block_preset(bx, by, bs, orig, rec, lambda, ct, Preset::Slow)
}

/// Fast intra: SATD pre-select top 3 modes, full RDO only on those.
pub fn rdo_intra_block_preset(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    rec: &Picture,
    lambda: f64,
    ct: &CostTables,
    preset: Preset,
) -> IntraDecision {
    const HEADER_BITS: f64 = 40.0;
    let (w, h) = (orig.w, orig.h);
    let (cw, ch) = (w / 2, h / 2);
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);

    // Fast preset: SATD pre-selection
    let modes: Vec<u8> = if preset == Preset::Fast {
        let mut scored: Vec<(u32, u8)> = Vec::with_capacity(8);
        for mode in 0..8u8 {
            let pred_y = intra_predict(mode, &rec.y, bx, by, bs, bs, w, h);
            // SATD on luma only for speed (chroma follows luma mode)
            let mut orig_tile = vec![0u16; bs * bs];
            for y in 0..bs {
                for x in 0..bs {
                    orig_tile[y*bs+x] = orig.y[(by+y)*w + (bx+x)];
                }
            }
            let s = satd_block(&orig_tile, &pred_y, bs, bs);
            scored.push((s, mode));
        }
        scored.sort_by_key(|&(s, _)| s);
        scored.iter().take(3).map(|&(_, m)| m).collect()
    } else {
        (0..8u8).collect()
    };

    let mut best: Option<IntraDecision> = None;
    for mode in modes {
        let pred_y = intra_predict(mode, &rec.y, bx, by, bs, bs, w, h);
        let pred_u = intra_predict(mode, &rec.u, cbx, cby, cbs, cbs, cw, ch);
        let pred_v = intra_predict(mode, &rec.v, cbx, cby, cbs, cbs, cw, ch);
        let (luma_txs, chroma_txs) = collect_txs(bx, by, bs, orig, &pred_y, &pred_u, &pred_v);
        let (q, cost, lq, cq) = best_q_preset(&luma_txs, &chroma_txs, lambda, ct, preset);
        let total = cost + lambda * HEADER_BITS;
        if best.as_ref().map_or(true, |b: &IntraDecision| total < b.cost) {
            best = Some(IntraDecision {
                mode,
                q,
                cost: total,
                luma_q: lq,
                chroma_q: cq,
            });
        }
    }
    best.unwrap()
}

/// Reconstruct one transform into a plane (exact decoder path).
fn reconstruct_tx(
    plane: &mut [u16],
    pw: usize,
    px: usize,
    py: usize,
    pred: &[u16],
    pbw: usize,
    ptx: usize,
    pty: usize,
    qcoeffs: &[i32],
    tn: usize,
    qs: f64,
) {
    let zz = zigzag(tn);
    if qcoeffs.iter().all(|&v| v == 0) {
        for y in 0..tn {
            for x in 0..tn {
                plane[(py + y) * pw + (px + x)] = pred[(pty + y) * pbw + (ptx + x)].clamp(0, 1023);
            }
        }
        return;
    }
    let mut mat = vec![0i32; tn * tn];
    for (pos, &(qy, qx)) in zz.iter().enumerate() {
        mat[qy * tn + qx] = half_away(qcoeffs[pos] as f64 * qs);
    }
    let resid = idct_2d(&mat, tn);
    for y in 0..tn {
        for x in 0..tn {
            let p = pred[(pty + y) * pbw + (ptx + x)] as i32;
            let rr = resid[y * tn + x] as i32;
            plane[(py + y) * pw + (px + x)] =
                half_away(p as f64 + rr as f64).clamp(0, 1023) as u16;
        }
    }
}

/// Commit an intra decision's reconstruction into `rec` (exact decoder path).
pub fn commit_intra(
    bx: usize,
    by: usize,
    bs: usize,
    dec: &IntraDecision,
    orig: &Picture,
    rec: &mut Picture,
) {
    let (w, h) = (orig.w, orig.h);
    let (cw, ch) = (w / 2, h / 2);
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    let qs = qstep_of(dec.q);

    let pred_y = intra_predict(dec.mode, &rec.y, bx, by, bs, bs, w, h);
    let tn = bs.min(32);
    let ntx = bs / tn;
    for ti in 0..ntx {
        for tj in 0..ntx {
            reconstruct_tx(
                &mut rec.y,
                w,
                bx + tj * tn,
                by + ti * tn,
                &pred_y,
                bs,
                tj * tn,
                ti * tn,
                &dec.luma_q[ti * ntx + tj],
                tn,
                qs,
            );
        }
    }

    let ctn = cbs.min(32);
    let cntx = cbs / ctn;
    // chroma planes: collect predictions first to satisfy the borrow checker
    let pred_u = intra_predict(dec.mode, &rec.u, cbx, cby, cbs, cbs, cw, ch);
    let pred_v = intra_predict(dec.mode, &rec.v, cbx, cby, cbs, cbs, cw, ch);
    // chroma_q layout: U transforms then V transforms
    let nu = cntx * cntx;
    for (p, (rec_c, pred_c, base)) in
        [(&mut rec.u, &pred_u, 0), (&mut rec.v, &pred_v, nu)].iter_mut().enumerate()
    {
        let _ = p;
        let base = *base;
        for ti in 0..cntx {
            for tj in 0..cntx {
                reconstruct_tx(
                    rec_c,
                    cw,
                    cbx + tj * ctn,
                    cby + ti * ctn,
                    pred_c,
                    cbs,
                    tj * ctn,
                    ti * ctn,
                    &dec.chroma_q[base + ti * cntx + tj],
                    ctn,
                    qs,
                );
            }
        }
    }
}
