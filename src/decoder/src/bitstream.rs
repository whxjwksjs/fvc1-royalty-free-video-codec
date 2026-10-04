//! Bitstream parsing and frame reconstruction — spec v0.2.
//!
//! All malformed input produces `DecodeError`; this module never panics on
//! attacker-controlled bytes. Mirrors src/refdec.py.

use crate::crc32::crc32;
use crate::error::{DecodeError, Result};
use crate::intra::intra_predict;
use crate::mc::motion_compensate;
use crate::rans;
use crate::idct::idct_2d;
use crate::transform::zigzag;

/// Max frame pixels: total allocation stays under 10 MB
/// (Y 6 MB + U/V 3 MB for 3M pixels at 2 bytes/sample).
const MAX_PIXELS: u64 = 3_000_000;
const MAX_DEPTH: u32 = 5;
const MAX_LEAVES_PER_SB: usize = 1024;
/// Max transforms per coding block: 16 luma (128x128 -> 32x32) + 8 chroma.
const MAX_TX_PER_BLOCK: usize = 24;

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub frame_type: u8,
    /// Y, U, V planes, row-major u16 (10-bit values).
    pub planes: [Vec<u16>; 3],
}

/// Round half away from zero (spec section 1).
pub fn half_away(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5).floor() as i32
    } else {
        -((-x + 0.5).floor() as i32)
    }
}

fn clip(v: i32) -> u16 {
    v.clamp(0, 1023) as u16
}

/// Quantizer step for a q_index (spec section 8).
pub fn qstep_of(q_index: u8) -> f64 {
    (2.0f64.powf(q_index as f64 / 16.0) + 0.5).floor().max(1.0)
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn need(&self, n: usize) -> Result<()> {
        if self.p.checked_add(n).map_or(true, |e| e > self.d.len()) {
            Err(DecodeError::Truncated)
        } else {
            Ok(())
        }
    }
    fn remaining(&self) -> usize {
        self.d.len() - self.p
    }
    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.d[self.p];
        self.p += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        Ok(v)
    }
    fn i16(&mut self) -> Result<i16> {
        self.need(2)?;
        let v = i16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes([
            self.d[self.p],
            self.d[self.p + 1],
            self.d[self.p + 2],
            self.d[self.p + 3],
        ]);
        self.p += 4;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
}

struct BitReader<'a> {
    d: &'a [u8],
    p: usize,
    bits_left: u8,
    cur: u8,
}

impl<'a> BitReader<'a> {
    fn get_bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            if self.bits_left == 0 {
                if self.p >= self.d.len() {
                    return Err(DecodeError::Truncated);
                }
                self.cur = self.d[self.p];
                self.p += 1;
                self.bits_left = 8;
            }
            self.bits_left -= 1;
            v = (v << 1) | ((self.cur >> self.bits_left) & 1) as u32;
        }
        Ok(v)
    }
    fn align(&mut self) -> usize {
        self.bits_left = 0;
        self.p
    }
}

enum PredMode {
    Intra(u8),
    Inter { mvx: i32, mvy: i32 },
}

struct Block {
    bx: usize,
    by: usize,
    bs: usize,
    pred: PredMode,
    q_index: u8,
    skip_luma: bool,
    skip_chroma: bool,
}

fn parse_partition(
    br: &mut BitReader,
    x: usize,
    y: usize,
    size: usize,
    depth: u32,
    leaves: &mut Vec<(usize, usize, usize)>,
) -> Result<()> {
    let t = br.get_bits(2)?;
    match t {
        0 => {
            if depth >= MAX_DEPTH {
                return Err(DecodeError::BadPartition);
            }
            let h = size / 2;
            parse_partition(br, x, y, h, depth + 1, leaves)?;
            parse_partition(br, x + h, y, h, depth + 1, leaves)?;
            parse_partition(br, x, y + h, h, depth + 1, leaves)?;
            parse_partition(br, x + h, y + h, h, depth + 1, leaves)?;
        }
        1 => leaves.push((x, y, size)),
        _ => return Err(DecodeError::BadPartition),
    }
    Ok(())
}

struct Tx {
    tn: usize,
}

/// Decode a whole container into frames. The last decoded frame is kept as
/// the reference for INTER frames.
pub fn decode_file(data: &[u8]) -> Result<Vec<Frame>> {
    let mut r = Reader { d: data, p: 0 };
    if r.remaining() < 4 || &data[0..4] != b"FVC1" {
        return Err(DecodeError::BadMagic);
    }
    r.p = 4;
    let version = r.u8()?;
    if version != 2 {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let num_frames = r.u32()?;
    if num_frames == 0 {
        return Err(DecodeError::EmptyFrames);
    }
    let mut frames: Vec<Frame> = Vec::new();
    let mut reference: Option<Frame> = None;
    let keep_ref = num_frames > 1;
    for _ in 0..num_frames {
        let frame_len = r.u32()? as usize;
        let fdata = r.bytes(frame_len)?;
        let frame = decode_frame(fdata, reference.as_ref())?;
        if frame.frame_type == 1 {
            let rf = reference.as_ref().ok_or(DecodeError::NoReferenceFrame)?;
            if frame.width != rf.width || frame.height != rf.height {
                return Err(DecodeError::DimensionMismatch);
            }
        }
        if keep_ref {
            reference = Some(Frame {
                width: frame.width,
                height: frame.height,
                frame_type: frame.frame_type,
                planes: [
                    frame.planes[0].clone(),
                    frame.planes[1].clone(),
                    frame.planes[2].clone(),
                ],
            });
        }
        frames.push(frame);
    }
    Ok(frames)
}

fn decode_frame(data: &[u8], reference: Option<&Frame>) -> Result<Frame> {
    let mut r = Reader { d: data, p: 0 };
    let frame_type = r.u8()?;
    if frame_type > 1 {
        return Err(DecodeError::UnsupportedFrameType(frame_type));
    }
    let w = r.u16()? as usize;
    let h = r.u16()? as usize;
    if w < 128 || h < 128 || w % 128 != 0 || h % 128 != 0 {
        return Err(DecodeError::BadDimensions);
    }
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(DecodeError::FrameTooLarge);
    }
    let color = r.u8()?;
    if color != 0 {
        return Err(DecodeError::UnsupportedColor(color));
    }
    if r.u8()? != 0 {
        return Err(DecodeError::LoopFilterOn);
    }
    r.u8()?; // reserved

    let num_sb = (w / 128) * (h / 128);
    // Parse all SB partitions first (spec section 4), then read all
    // block headers (spec section 5, SB-by-SB raster).
    let mut sb_leaves: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for sby in (0..h).step_by(128) {
        for sbx in (0..w).step_by(128) {
            let mut br = BitReader { d: data, p: r.p, bits_left: 0, cur: 0 };
            let mut leaves = Vec::new();
            parse_partition(&mut br, sbx, sby, 128, 0, &mut leaves)?;
            r.p = br.align();
            if leaves.len() > MAX_LEAVES_PER_SB {
                return Err(DecodeError::TooManyBlocks);
            }
            sb_leaves.push(leaves);
        }
    }
    let total_leaves: usize = sb_leaves.iter().map(|v| v.len()).sum();
    if total_leaves > num_sb * MAX_LEAVES_PER_SB {
        return Err(DecodeError::TooManyBlocks);
    }
    // Read headers SB-by-SB in raster order within SB.
    // Store blocks in partition-preorder (causal for reconstruction).
    let mut blocks: Vec<Block> = Vec::new();
    for leaves in &sb_leaves {
        let mut raster_leaves = leaves.clone();
        raster_leaves.sort_by_key(|&(x, y, _)| (y, x));
        let mut headers: Vec<(usize, usize, usize, PredMode, u8, bool, bool)> = Vec::new();
        for (bx, by, bs) in raster_leaves {
            let intra_flag = r.u8()?;
            let pred = match intra_flag {
                1 => {
                    let mode = r.u8()?;
                    if mode > 7 {
                        return Err(DecodeError::BadIntraMode(mode));
                    }
                    PredMode::Intra(mode)
                }
                0 => {
                    let ref_idx = r.u8()?;
                    if ref_idx != 0 {
                        return Err(DecodeError::BadRefIdx(ref_idx));
                    }
                    let mvx = r.i16()? as i32;
                    let mvy = r.i16()? as i32;
                    if r.u8()? != 0 {
                        return Err(DecodeError::WarpOn);
                    }
                    PredMode::Inter { mvx, mvy }
                }
                v => return Err(DecodeError::BadIntraFlag(v)),
            };
            let q_index = r.u8()?;
            let skip_luma = r.u8()? != 0;
            let skip_chroma = r.u8()? != 0;
            headers.push((bx, by, bs, pred, q_index, skip_luma, skip_chroma));
        }
        for (bx, by, bs) in leaves {
            let (_, _, _, pred, q_index, skip_luma, skip_chroma) = headers
                .iter()
                .find(|(hx, hy, _, _, _, _, _)| *hx == *bx && *hy == *by)
                .expect("header missing");
            blocks.push(Block {
                bx: *bx,
                by: *by,
                bs: *bs,
                pred: match pred {
                    PredMode::Intra(m) => PredMode::Intra(*m),
                    PredMode::Inter { mvx, mvy } => PredMode::Inter {
                        mvx: *mvx,
                        mvy: *mvy,
                    },
                },
                q_index: *q_index,
                skip_luma: *skip_luma,
                skip_chroma: *skip_chroma,
            });
        }
    }
    if frame_type == 1 && reference.is_none() {
        return Err(DecodeError::NoReferenceFrame);
    }

    // Blocks arrive in partition-preorder (bitstream order), which is causal:
    // a block's top/left neighbors are always reconstructed before it.
    // Transform coefficients are enumerated in frame-raster block order
    // (spec section 9), so we build tx_list via a raster-sorted index while
    // keeping `blocks` in preorder for reconstruction.
    let mut raster_idx: Vec<usize> = (0..blocks.len()).collect();
    raster_idx.sort_by_key(|&i| (blocks[i].by, blocks[i].bx));

    // transform enumeration: per block luma txs, then U, then V
    if blocks.len() > num_sb * MAX_LEAVES_PER_SB {
        return Err(DecodeError::TooManyBlocks);
    }
    let mut tx_list: Vec<Tx> = Vec::new();
    // per-block (preorder index) -> tx_list range
    let mut block_tx_range = vec![(0usize, 0usize); blocks.len()];
    for &bi in &raster_idx {
        let b = &blocks[bi];
        let start = tx_list.len();
        let tn = b.bs.min(32);
        if !b.skip_luma {
            for _ty in (0..b.bs).step_by(tn) {
                for _tx in (0..b.bs).step_by(tn) {
                    tx_list.push(Tx { tn });
                }
            }
        }
        if !b.skip_chroma {
            let cbs = b.bs / 2;
            let ctn = cbs.min(32);
            // Spec section 9: all U transforms raster, then all V transforms raster.
            for _ in 0..2 {
                for _ty in (0..cbs).step_by(ctn) {
                    for _tx in (0..cbs).step_by(ctn) {
                        tx_list.push(Tx { tn: ctn });
                    }
                }
            }
        }
        if tx_list.len() > blocks.len() * MAX_TX_PER_BLOCK {
            return Err(DecodeError::TooManySymbols);
        }
        block_tx_range[bi] = (start, tx_list.len());
    }

    // rANS streams (CRC verified before decode)
    let cum_dc = rans::cum_table(&rans::FREQ_DC);
    let cum_low = rans::cum_table(&rans::FREQ_AC_LOW);
    let cum_high = rans::cum_table(&rans::FREQ_AC_HIGH);
    let cum_sign = rans::cum_table(&rans::FREQ_SIGN);
    let ctx_names = ["DC", "AC_LOW", "AC_HIGH", "SIGN"];
    let mut streams: Vec<(u32, &[u8])> = Vec::new();
    for name in ctx_names {
        let final_state = r.u32()?;
        let nbytes = r.u32()? as usize;
        if nbytes > r.remaining() {
            return Err(DecodeError::StreamTooLong);
        }
        let want_crc = r.u32()?;
        let blob = r.bytes(nbytes)?;
        if crc32(blob) != want_crc {
            return Err(DecodeError::CrcMismatch(name));
        }
        streams.push((final_state, blob));
    }
    let n_esc = r.u32()? as usize;
    let n_dc = tx_list.len();
    let n_low: usize = tx_list.iter().map(|t| (t.tn * t.tn - 1).min(16)).sum();
    let n_high: usize = tx_list
        .iter()
        .map(|t| (t.tn * t.tn - 1).saturating_sub(16))
        .sum();
    if n_esc > n_dc + n_low + n_high {
        return Err(DecodeError::TooManyEscapes);
    }
    if n_esc.checked_mul(2).map_or(true, |b| b > r.remaining()) {
        return Err(DecodeError::Truncated);
    }
    let mut escapes = Vec::with_capacity(n_esc.min(1 << 20));
    for _ in 0..n_esc {
        escapes.push(r.u16()?);
    }

    let dc_syms = rans::decode_symbols(streams[0].0, &streams[0].1, n_dc, &rans::FREQ_DC, &cum_dc)?;
    let low_syms = rans::decode_symbols(
        streams[1].0,
        &streams[1].1,
        n_low,
        &rans::FREQ_AC_LOW,
        &cum_low,
    )?;
    let high_syms = rans::decode_symbols(
        streams[2].0,
        &streams[2].1,
        n_high,
        &rans::FREQ_AC_HIGH,
        &cum_high,
    )?;

    // rebuild coefficients.
    // Escape order: DC, then AC_LOW, then AC_HIGH (tx enum order).
    // Sign order (spec): DC signs in tx order, then AC signs in tx order
    // with zigzag scan order within each transform.
    let mut esc_pos = 0usize;
    let mut qcoeff_blocks: Vec<Vec<i32>> =
        tx_list.iter().map(|t| vec![0i32; t.tn * t.tn]).collect();
    let mut allzero: Vec<bool> = vec![true; tx_list.len()];
    let mut pending: Vec<(usize, usize, i32)> = Vec::new();
    let (mut di, mut li, mut hi) = (0usize, 0usize, 0usize);
    for (ti, t) in tx_list.iter().enumerate() {
        let _ = t;
        let s = dc_syms[di];
        di += 1;
        let lv = if s < 16 {
            s as i32
        } else {
            if esc_pos >= escapes.len() {
                return Err(DecodeError::TooManyEscapes);
            }
            let v = escapes[esc_pos] as i32;
            esc_pos += 1;
            v
        };
        qcoeff_blocks[ti][0] = lv;
        if lv != 0 {
            pending.push((ti, 0, lv));
            allzero[ti] = false;
        }
    }
    for (ti, t) in tx_list.iter().enumerate() {
        let n = t.tn * t.tn;
        for pos in 1..(16.min(n - 1) + 1) {
            let s = low_syms[li];
            li += 1;
            let lv = if s < 16 {
                s as i32
            } else {
                if esc_pos >= escapes.len() {
                    return Err(DecodeError::TooManyEscapes);
                }
                let v = escapes[esc_pos] as i32;
                esc_pos += 1;
                v
            };
            qcoeff_blocks[ti][pos] = lv;
            if lv != 0 {
                allzero[ti] = false;
            }
        }
    }
    for (ti, t) in tx_list.iter().enumerate() {
        let n = t.tn * t.tn;
        for pos in 17..n {
            let s = high_syms[hi];
            hi += 1;
            let lv = if s < 16 {
                s as i32
            } else {
                if esc_pos >= escapes.len() {
                    return Err(DecodeError::TooManyEscapes);
                }
                let v = escapes[esc_pos] as i32;
                esc_pos += 1;
                v
            };
            qcoeff_blocks[ti][pos] = lv;
            if lv != 0 {
                allzero[ti] = false;
            }
        }
    }
    // AC signs in zigzag scan order per transform (low/high interleaved),
    // matching the encoder's emission order (spec section 10).
    for (ti, coeffs) in qcoeff_blocks.iter().enumerate() {
        for (pos, &lv) in coeffs.iter().enumerate().skip(1) {
            if lv != 0 {
                pending.push((ti, pos, lv));
            }
        }
    }
    if esc_pos != n_esc {
        return Err(DecodeError::TooManyEscapes);
    }

    let sign_syms = rans::decode_symbols(
        streams[3].0,
        &streams[3].1,
        pending.len(),
        &rans::FREQ_SIGN,
        &cum_sign,
    )?;
    for (i, (ti, pos, lv)) in pending.iter().enumerate() {
        let v = if sign_syms[i] == 0 { *lv } else { -*lv };
        qcoeff_blocks[*ti][*pos] = v;
    }

    // reconstruct
    let mut planes = [
        vec![512u16; w * h],
        vec![512u16; (w / 2) * (h / 2)],
        vec![512u16; (w / 2) * (h / 2)],
    ];
    let fws = [w, w / 2, w / 2];
    let fhs = [h, h / 2, h / 2];
    // Reconstruct in partition-preorder (causal). ti indexes qcoeff_blocks
    // via the block's raster-order tx range.
    for (bi, b) in blocks.iter().enumerate() {
        let mut ti = block_tx_range[bi].0;
        let qs = qstep_of(b.q_index);
        let tn = b.bs.min(32);
        // prediction for luma
        let pred_y: Vec<u16> = match &b.pred {
            PredMode::Intra(mode) => intra_predict(*mode, &planes[0], b.bx, b.by, b.bs, b.bs, w, h),
            PredMode::Inter { mvx, mvy } => {
                let rf = reference.ok_or(DecodeError::NoReferenceFrame)?;
                motion_compensate(&rf.planes[0], *mvx, *mvy, b.bx, b.by, b.bs, b.bs, w, h)
            }
        };
        for ty in (0..b.bs).step_by(tn) {
            for tx in (0..b.bs).step_by(tn) {
                if !b.skip_luma && !allzero[ti] {
                    let coeffs = &qcoeff_blocks[ti];
                    ti += 1;
                    let zz = zigzag(tn);
                    // Stack-allocated (no heap allocation in hot loop)
                    let mut mat = [0i32; 1024];
                    for (pos, &(qy, qx)) in zz.iter().enumerate() {
                        mat[qy * tn + qx] = half_away(coeffs[pos] as f64 * qs);
                    }
                    let resid = idct_2d(&mat[..tn*tn], tn);
                    for y in 0..tn {
                        for x in 0..tn {
                            let p = pred_y[(ty + y) * b.bs + (tx + x)] as i32;
                            let rr = resid[y * tn + x] as i32;
                            planes[0][(b.by + ty + y) * w + (b.bx + tx + x)] =
                                clip(half_away(p as f64 + rr as f64));
                        }
                    }
                } else {
                    if !b.skip_luma {
                        ti += 1; // all-zero coeffs: no IDCT needed
                    }
                    for y in 0..tn {
                        for x in 0..tn {
                            planes[0][(b.by + ty + y) * w + (b.bx + tx + x)] =
                                clip(pred_y[(ty + y) * b.bs + (tx + x)] as i32);
                        }
                    }
                }
            }
        }
        // chroma
        let cbs = b.bs / 2;
        let (cbx, cby) = (b.bx / 2, b.by / 2);
        for p in 1..=2usize {
            let fw = fws[p];
            let fh = fhs[p];
            let pred_c: Vec<u16> = match &b.pred {
                PredMode::Intra(mode) => {
                    intra_predict(*mode, &planes[p], cbx, cby, cbs, cbs, fw, fh)
                }
                PredMode::Inter { mvx, mvy } => {
                    let rf = reference.ok_or(DecodeError::NoReferenceFrame)?;
                    motion_compensate(
                        &rf.planes[p],
                        mvx >> 1,
                        mvy >> 1,
                        cbx,
                        cby,
                        cbs,
                        cbs,
                        fw,
                        fh,
                    )
                }
            };
            let ctn = cbs.min(32);
            for ty in (0..cbs).step_by(ctn) {
                for tx in (0..cbs).step_by(ctn) {
                    if !b.skip_chroma && !allzero[ti] {
                        let coeffs = &qcoeff_blocks[ti];
                        ti += 1;
                        let zz = zigzag(ctn);
                        let mut mat = [0i32; 1024];
                        for (pos, &(qy, qx)) in zz.iter().enumerate() {
                            mat[qy * ctn + qx] = half_away(coeffs[pos] as f64 * qs);
                        }
                        let resid = idct_2d(&mat[..ctn*ctn], ctn);
                        for y in 0..ctn {
                            for x in 0..ctn {
                                let pr = pred_c[(ty + y) * cbs + (tx + x)] as i32;
                                let rr = resid[y * ctn + x] as i32;
                                planes[p][(cby + ty + y) * fw + (cbx + tx + x)] =
                                    clip(half_away(pr as f64 + rr as f64));
                            }
                        }
                    } else {
                        if !b.skip_chroma {
                            ti += 1;
                        }
                        for y in 0..ctn {
                            for x in 0..ctn {
                                planes[p][(cby + ty + y) * fw + (cbx + tx + x)] =
                                    clip(pred_c[(ty + y) * cbs + (tx + x)] as i32);
                            }
                        }
                    }
                }
            }
        }
    }
    // (ti is per-block now; ranges partition tx_list exactly by construction)
    if r.p != data.len() {
        return Err(DecodeError::TrailingData);
    }

    Ok(Frame { width: w, height: h, frame_type, planes })
}
