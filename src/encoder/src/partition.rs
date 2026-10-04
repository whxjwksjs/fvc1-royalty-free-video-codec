//! Partition-tree RDO with causal reconstruction.
//!
//! Recursive DP: at each node, compare "code as leaf" vs "split into four".
//! Split is tried with save/restore of the reconstruction buffer; the
//! winner's reconstruction stays. SBs are processed in raster order and
//! children in raster order, so intra/inter causal neighbors are always
//! committed before use.

use crate::inter::{commit_inter, rdo_inter_block, rdo_inter_block_preset, InterDecision};
use crate::picture::Picture;
use crate::rdo::{commit_intra, rdo_intra_block, rdo_intra_block_preset, CostTables, IntraDecision, Preset, MAX_DEPTH, MAX_DEPTH_FAST};

pub enum BlockMode {
    Intra(IntraDecision),
    Inter(InterDecision),
}

pub struct BlockDecision {
    pub bx: usize,
    pub by: usize,
    pub bs: usize,
    pub mode: BlockMode,
}

impl BlockDecision {
    pub fn q(&self) -> u8 {
        match &self.mode {
            BlockMode::Intra(d) => d.q,
            BlockMode::Inter(d) => d.q,
        }
    }
    pub fn skip_luma(&self) -> bool {
        match &self.mode {
            BlockMode::Intra(d) => d.luma_q.iter().all(|t| t.iter().all(|&v| v == 0)),
            BlockMode::Inter(d) => d.luma_q.iter().all(|t| t.iter().all(|&v| v == 0)),
        }
    }
    pub fn skip_chroma(&self) -> bool {
        match &self.mode {
            BlockMode::Intra(d) => d.chroma_q.iter().all(|t| t.iter().all(|&v| v == 0)),
            BlockMode::Inter(d) => d.chroma_q.iter().all(|t| t.iter().all(|&v| v == 0)),
        }
    }
}

/// Partition node in preorder (bitstream order).
pub struct PartNode {
    pub x: usize,
    pub y: usize,
    pub size: usize,
    pub split: bool,
}

struct SavedRegion {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

fn save_region(rec: &Picture, bx: usize, by: usize, bs: usize) -> SavedRegion {
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    SavedRegion {
        y: rec.save_y_region(bx, by, bs, bs),
        u: {
            let mut v = Vec::with_capacity(cbs * cbs);
            for yy in cby..cby + cbs {
                v.extend_from_slice(&rec.u[yy * rec.w / 2 + cbx..yy * rec.w / 2 + cbx + cbs]);
            }
            v
        },
        v: {
            let mut v = Vec::with_capacity(cbs * cbs);
            for yy in cby..cby + cbs {
                v.extend_from_slice(&rec.v[yy * rec.w / 2 + cbx..yy * rec.w / 2 + cbx + cbs]);
            }
            v
        },
    }
}

fn restore_region(rec: &mut Picture, bx: usize, by: usize, bs: usize, s: &SavedRegion) {
    let (cbx, cby, cbs) = (bx / 2, by / 2, bs / 2);
    rec.restore_y_region(bx, by, bs, bs, &s.y);
    for (yy, row) in (cby..cby + cbs).enumerate() {
        rec.u[row * rec.w / 2 + cbx..row * rec.w / 2 + cbx + cbs]
            .copy_from_slice(&s.u[yy * cbs..(yy + 1) * cbs]);
        rec.v[row * rec.w / 2 + cbx..row * rec.w / 2 + cbx + cbs]
            .copy_from_slice(&s.v[yy * cbs..(yy + 1) * cbs]);
    }
}

fn commit_decision(bx: usize, by: usize, bs: usize, dec: &BlockDecision, orig: &Picture, rec: &mut Picture, ref_pic: Option<&Picture>) {
    match &dec.mode {
        BlockMode::Intra(d) => commit_intra(bx, by, bs, d, orig, rec),
        BlockMode::Inter(d) => commit_inter(bx, by, bs, d, orig, ref_pic.unwrap(), rec),
    }
}

/// Evaluate one block as a leaf (intra, plus inter if P-frame).
/// Returns (rd_cost_without_partition_flag, BlockDecision). Does not commit.
fn eval_leaf(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    rec: &Picture,
    ref_pic: Option<&Picture>,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
) -> (f64, BlockDecision) {
    eval_leaf_preset(bx, by, bs, orig, rec, ref_pic, lambda, qp, ct, Preset::Slow)
}

fn eval_leaf_preset(
    bx: usize,
    by: usize,
    bs: usize,
    orig: &Picture,
    rec: &Picture,
    ref_pic: Option<&Picture>,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
    preset: Preset,
) -> (f64, BlockDecision) {
    let intra = rdo_intra_block_preset(bx, by, bs, orig, rec, lambda, ct, preset);
    let mut best_cost = intra.cost;
    let mut best_mode = BlockMode::Intra(intra);
    if let Some(rp) = ref_pic {
        let inter = rdo_inter_block_preset(bx, by, bs, orig, rp, lambda, qp, ct, preset);
        if inter.cost < best_cost {
            best_cost = inter.cost;
            best_mode = BlockMode::Inter(inter);
        }
    }
    (
        best_cost,
        BlockDecision {
            bx,
            by,
            bs,
            mode: best_mode,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn rdo_node(
    bx: usize,
    by: usize,
    bs: usize,
    depth: u32,
    orig: &Picture,
    rec: &mut Picture,
    ref_pic: Option<&Picture>,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
    decisions: &mut Vec<BlockDecision>,
    parts: &mut Vec<PartNode>,
    preset: Preset,
) -> f64 {
    const FLAG_BITS: f64 = 2.0; // partition flag
    let max_depth = if preset == Preset::Fast { MAX_DEPTH_FAST } else { MAX_DEPTH };
    let (leaf_cost, leaf_dec) = eval_leaf_preset(bx, by, bs, orig, rec, ref_pic, lambda, qp, ct, preset);
    let leaf_total = leaf_cost + lambda * FLAG_BITS;

    // Minimum block size is 8 (chroma needs at least 4x4)
    if depth >= max_depth || bs <= 8 {
        let dec = leaf_dec;
        commit_decision(bx, by, bs, &dec, orig, rec, ref_pic);
        parts.push(PartNode {
            x: bx,
            y: by,
            size: bs,
            split: false,
        });
        decisions.push(dec);
        return leaf_total;
    }

    // Try split: buffer children's output, save rec.
    let saved = save_region(rec, bx, by, bs);
    let mut child_parts = Vec::new();
    let mut child_decs = Vec::new();
    let h = bs / 2;
    let mut split_cost = lambda * FLAG_BITS;
    for (cx, cy) in [(bx, by), (bx + h, by), (bx, by + h), (bx + h, by + h)] {
        split_cost += rdo_node(
            cx, cy, h, depth + 1, orig, rec, ref_pic, lambda, qp, ct, &mut child_decs,
            &mut child_parts, preset,
        );
    }

    if leaf_total <= split_cost {
        // Leaf wins: restore rec, commit leaf.
        restore_region(rec, bx, by, bs, &saved);
        let dec = leaf_dec;
        commit_decision(bx, by, bs, &dec, orig, rec, ref_pic);
        parts.push(PartNode {
            x: bx,
            y: by,
            size: bs,
            split: false,
        });
        decisions.push(dec);
        leaf_total
    } else {
        // Split wins: keep children's reconstructions and output.
        parts.push(PartNode {
            x: bx,
            y: by,
            size: bs,
            split: true,
        });
        parts.extend(child_parts);
        decisions.extend(child_decs);
        split_cost
    }
}

/// Encode one frame with full RDO.
/// Returns (partition nodes in bitstream order, block decisions, reconstruction).
pub fn encode_frame_rdo(
    orig: &Picture,
    ref_pic: Option<&Picture>,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
) -> (Vec<PartNode>, Vec<BlockDecision>, Picture) {
    encode_frame_rdo_preset(orig, ref_pic, lambda, qp, ct, Preset::Slow)
}

pub fn encode_frame_rdo_preset(
    orig: &Picture,
    ref_pic: Option<&Picture>,
    lambda: f64,
    qp: u8,
    ct: &CostTables,
    preset: Preset,
) -> (Vec<PartNode>, Vec<BlockDecision>, Picture) {
    let mut rec = Picture::new(orig.w, orig.h);
    let mut decisions = Vec::new();
    let mut parts = Vec::new();
    for sby in (0..orig.h).step_by(128) {
        for sbx in (0..orig.w).step_by(128) {
            rdo_node(
                sbx, sby, 128, 0, orig, &mut rec, ref_pic, lambda, qp, ct, &mut decisions,
                &mut parts, preset,
            );
        }
    }
    // Decisions are in DP (preorder) order; the writer sorts for bitstream.
    (parts, decisions, rec)
}
