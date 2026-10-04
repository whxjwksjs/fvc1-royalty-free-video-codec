//! Inter prediction for the slow encoder: diamond MV search at integer pel
//! (SSE-based, no transform) followed by exhaustive 1/8-pel refinement with
//! full RDO. Motion compensation matches the spec exactly (bilinear,
//! edge-clamped).

use fvc1_dec::motion_compensate;

use crate::picture::Picture;
use crate::rdo::Preset;
use crate::rdo::{best_q, collect_txs, CostTables};

pub const MV_RANGE: i32 = 64; // integer-pel diamond search range

pub struct InterDecision {
    pub mvx: i32, // 1/8-pel units
    pub mvy: i32,
    pub q: u8,
    pub cost: f64, // includes header bits
    pub luma_q: Vec<Vec<i32>>,
    pub chroma_q: Vec<Vec<i32>>,
}

/// SSE of motion-compensated residual for luma only (fast integer search).
fn mc_sse(
    ref_y: &[u16],
    orig_y: &[u16],
    ow: usize,
    bx: usize,
    by: usize,
    bs: usize,
    mvx: i32,
    mvy: i32,
    fw: usize,
    fh: usize,
) -> f64 {
    let pred = motion_compensate(ref_y, mvx, mvy, bx, by, bs, bs, fw, fh);
    let mut sse = 0.0;
    for y in 0..bs {
        for x in 0..bs {
            let d = orig_y[(by + y) * ow + (bx + x)] as f64 - pred[y * bs + x] as f64;
            sse += d * d;
        }
    }
    sse
}

/// Diamond search at integer pel. Returns best (mvx, mvy) in 1/8-pel units.
fn diamond_search(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    ref_pic: &Picture,
) -> (i32, i32) {
    let (fw, fh) = (ref_pic.w, ref_pic.h);
    let mut best = (0i32, 0i32);
    let mut best_cost = mc_sse(&ref_pic.y, &orig.y, orig.w, bx, by, bs, 0, 0, fw, fh);
    // clamp helper for candidate MVs
    for step in [32, 16, 8, 4, 2, 1] {
        loop {
            let mut improved = false;
            // 8-point diamond
            for (dx, dy) in [
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ] {
                let cx = (best.0 + dx * 8).clamp(-MV_RANGE * 8, MV_RANGE * 8);
                let cy = (best.1 + dy * 8).clamp(-MV_RANGE * 8, MV_RANGE * 8);
                if cx == best.0 && cy == best.1 {
                    continue;
                }
                let c = mc_sse(&ref_pic.y, &orig.y, orig.w, bx, by, bs, cx, cy, fw, fh);
                if c < best_cost {
                    best_cost = c;
                    best = (cx, cy);
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    best
}

/// Hexagonal search (fast preset). 6-point hex pattern, fewer candidates.
fn hex_search(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    ref_pic: &Picture,
) -> (i32, i32) {
    let (fw, fh) = (ref_pic.w, ref_pic.h);
    let mut best = (0i32, 0i32);
    let mut best_cost = mc_sse(&ref_pic.y, &orig.y, orig.w, bx, by, bs, 0, 0, fw, fh);
    if best_cost < (bs * bs * 2) as f64 {
        return best;
    }
    for step in [16, 8, 4, 2, 1] {
        loop {
            let mut improved = false;
            for (dx, dy) in [
                (step, 0), (-step, 0),
                (step / 2, step), (-step / 2, step),
                (step / 2, -step), (-step / 2, -step),
            ] {
                let cx = (best.0 + dx * 8).clamp(-MV_RANGE * 8, MV_RANGE * 8);
                let cy = (best.1 + dy * 8).clamp(-MV_RANGE * 8, MV_RANGE * 8);
                if cx == best.0 && cy == best.1 { continue; }
                let c = mc_sse(&ref_pic.y, &orig.y, orig.w, bx, by, bs, cx, cy, fw, fh);
                if c < best_cost { best_cost = c; best = (cx, cy); improved = true; }
            }
            if !improved { break; }
        }
    }
    best
}

/// Full RDO for one inter block at a given MV (tries Q 0-255).
/// Returns (cost_without_mv_header, q, luma_q, chroma_q).
fn rdo_at_mv(
    bx: usize,
    by: usize,
    bs: usize,
    mvx: i32,
    mvy: i32,
    orig: &Picture,
    ref_pic: &Picture,
    lambda: f64,
    ct: &CostTables,
) -> (f64, u8, Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let (w, h) = (orig.w, orig.h);
    let (cw, ch) = (w / 2, h / 2);
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    let pred_y = motion_compensate(&ref_pic.y, mvx, mvy, bx, by, bs, bs, w, h);
    let pred_u = motion_compensate(&ref_pic.u, mvx >> 1, mvy >> 1, cbx, cby, cbs, cbs, cw, ch);
    let pred_v = motion_compensate(&ref_pic.v, mvx >> 1, mvy >> 1, cbx, cby, cbs, cbs, cw, ch);
    let (luma_txs, chroma_txs) = collect_txs(bx, by, bs, orig, &pred_y, &pred_u, &pred_v);
    let (q, cost, lq, cq) = best_q(&luma_txs, &chroma_txs, lambda, ct);
    (cost, q, lq, cq)
}

/// RD cost at a fixed Q (for fast MV search). Returns cost without header.
fn cost_at_mv_fixed_q(
    bx: usize,
    by: usize,
    bs: usize,
    mvx: i32,
    mvy: i32,
    q: u8,
    orig: &Picture,
    ref_pic: &Picture,
    lambda: f64,
    ct: &CostTables,
) -> f64 {
    let (w, h) = (orig.w, orig.h);
    let (cw, ch) = (w / 2, h / 2);
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    let pred_y = motion_compensate(&ref_pic.y, mvx, mvy, bx, by, bs, bs, w, h);
    let pred_u = motion_compensate(&ref_pic.u, mvx >> 1, mvy >> 1, cbx, cby, cbs, cbs, cw, ch);
    let pred_v = motion_compensate(&ref_pic.v, mvx >> 1, mvy >> 1, cbx, cby, cbs, cbs, cw, ch);
    let (luma_txs, chroma_txs) = collect_txs(bx, by, bs, orig, &pred_y, &pred_u, &pred_v);
    let qs = fvc1_dec::qstep_of(q);
    let mut d = 0.0;
    let mut r = 0.0;
    for tx in luma_txs.iter().chain(chroma_txs.iter()) {
        for (pos, &c) in tx.coeffs.iter().enumerate() {
            let qc = fvc1_dec::half_away(c / qs);
            let diff = c - qc as f64 * qs;
            d += diff * diff;
            r += ct.coeff_bits(qc, pos);
        }
    }
    d + lambda * r
}

/// Brute-force inter RDO for one block.
/// Header bits: flag(8) + ref_idx(8) + mvx(16) + mvy(16) + warp(8) + q(8) + skips(16) = 80.
pub fn rdo_inter_block(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    ref_pic: &Picture,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
) -> InterDecision {
    rdo_inter_block_preset(bx, by, bs, orig, ref_pic, lambda, qp, ct, Preset::Slow)
}

pub fn rdo_inter_block_preset(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    ref_pic: &Picture,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
    preset: Preset,
) -> InterDecision {
    const HEADER_BITS: f64 = 80.0;
    // 1. integer MV search (diamond for slow, hex for fast)
    let (imvx, imvy) = if preset == Preset::Fast {
        hex_search(bx, by, bs, orig, ref_pic)
    } else {
        diamond_search(bx, by, bs, orig, ref_pic)
    };
    // 2. 1/8-pel refinement: exhaustive 17x17 for slow, best-2 for fast
    let mut best_mv = (imvx, imvy);
    let mut best_mv_cost = f64::INFINITY;
    if preset == Preset::Fast {
        let mut candidates = vec![(imvx, imvy)];
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 { continue; }
                candidates.push((imvx + dx*8, imvy + dy*8));
            }
        }
        let mut scored: Vec<(f64, (i32, i32))> = candidates.into_iter().map(|(mvx, mvy)| {
            let c = cost_at_mv_fixed_q(bx, by, bs, mvx, mvy, qp, orig, ref_pic, lambda, ct);
            (c, (mvx, mvy))
        }).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(_, (cmvx, cmvy)) in scored.iter().take(2) {
            for dy in -4..=4 {
                for dx in -4..=4 {
                    let mvx = cmvx + dx;
                    let mvy = cmvy + dy;
                    let c = cost_at_mv_fixed_q(bx, by, bs, mvx, mvy, qp, orig, ref_pic, lambda, ct);
                    if c < best_mv_cost { best_mv_cost = c; best_mv = (mvx, mvy); }
                }
            }
        }
    } else {
        for dy in -8..=8 {
            for dx in -8..=8 {
                let mvx = imvx + dx;
                let mvy = imvy + dy;
                let c = cost_at_mv_fixed_q(bx, by, bs, mvx, mvy, qp, orig, ref_pic, lambda, ct);
                if c < best_mv_cost { best_mv_cost = c; best_mv = (mvx, mvy); }
            }
        }
    }
    // 3. full Q search for the best MV
    let (cost, q, lq, cq) = rdo_at_mv(
        bx, by, bs, best_mv.0, best_mv.1, orig, ref_pic, lambda, ct,
    );
    InterDecision {
        mvx: best_mv.0,
        mvy: best_mv.1,
        q,
        cost: cost + lambda * HEADER_BITS,
        luma_q: lq,
        chroma_q: cq,
    }
}

/// Commit an inter decision's reconstruction into `rec` (exact decoder path).
pub fn commit_inter(
    bx: usize,
    by: usize,
    bs: usize,
    dec: &InterDecision,
    orig: &Picture,
    ref_pic: &Picture,
    rec: &mut Picture,
) {
    use fvc1_dec::{half_away, idct_2d, qstep_of, zigzag};
    let (w, h) = (orig.w, orig.h);
    let (cw, ch) = (w / 2, h / 2);
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    let qs = qstep_of(dec.q);

    // predictions (computed before mutable borrows)
    let pred_y = motion_compensate(&ref_pic.y, dec.mvx, dec.mvy, bx, by, bs, bs, w, h);
    let pred_u = motion_compensate(
        &ref_pic.u, dec.mvx >> 1, dec.mvy >> 1, cbx, cby, cbs, cbs, cw, ch,
    );
    let pred_v = motion_compensate(
        &ref_pic.v, dec.mvx >> 1, dec.mvy >> 1, cbx, cby, cbs, cbs, cw, ch,
    );

    // luma
    let tn = bs.min(32);
    let ntx = bs / tn;
    let zz = zigzag(tn);
    for ti in 0..ntx {
        for tj in 0..ntx {
            let qcoeffs = &dec.luma_q[ti * ntx + tj];
            if qcoeffs.iter().all(|&v| v == 0) {
                for y in 0..tn {
                    for x in 0..tn {
                        rec.y[(by + ti * tn + y) * w + (bx + tj * tn + x)] =
                            pred_y[(ti * tn + y) * bs + (tj * tn + x)].clamp(0, 1023);
                    }
                }
            } else {
                let mut mat = vec![0i32; tn * tn];
                for (pos, &(qy, qx)) in zz.iter().enumerate() {
                    mat[qy * tn + qx] = half_away(qcoeffs[pos] as f64 * qs);
                }
                let resid = idct_2d(&mat, tn);
                for y in 0..tn {
                    for x in 0..tn {
                        let p = pred_y[(ti * tn + y) * bs + (tj * tn + x)] as i32;
                        let rr = resid[y * tn + x] as i32;
                        rec.y[(by + ti * tn + y) * w + (bx + tj * tn + x)] =
                            half_away(p as f64 + rr as f64).clamp(0, 1023) as u16;
                    }
                }
            }
        }
    }
    // chroma
    let ctn = cbs.min(32);
    let cntx = cbs / ctn;
    let czz = zigzag(ctn);
    let nu = cntx * cntx;
    for (rec_c, pred_c, base) in
        [(&mut rec.u, &pred_u, 0), (&mut rec.v, &pred_v, nu)].iter_mut()
    {
        for ti in 0..cntx {
            for tj in 0..cntx {
                let qcoeffs = &dec.chroma_q[*base + ti * cntx + tj];
                if qcoeffs.iter().all(|&v| v == 0) {
                    for y in 0..ctn {
                        for x in 0..ctn {
                            rec_c[(cby + ti * ctn + y) * cw + (cbx + tj * ctn + x)] =
                                pred_c[(ti * ctn + y) * cbs + (tj * ctn + x)].clamp(0, 1023);
                        }
                    }
                } else {
                    let mut mat = vec![0i32; ctn * ctn];
                    for (pos, &(qy, qx)) in czz.iter().enumerate() {
                        mat[qy * ctn + qx] = half_away(qcoeffs[pos] as f64 * qs);
                    }
                    let resid = idct_2d(&mat, ctn);
                    for y in 0..ctn {
                        for x in 0..ctn {
                            let p = pred_c[(ti * ctn + y) * cbs + (tj * ctn + x)] as i32;
                            let rr = resid[y * ctn + x] as i32;
                            rec_c[(cby + ti * ctn + y) * cw + (cbx + tj * ctn + x)] =
                                half_away(p as f64 + rr as f64).clamp(0, 1023) as u16;
                        }
                    }
                }
            }
        }
    }
}
