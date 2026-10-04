//! fvc1-enc: FVC1 slow encoder CLI (spec v0.2).
//!
//! Usage: fvc1-enc --preset slow --input in.yuv --width W --height H
//!                  --output out.fvc1 --qp Q [--frames N]
//!
//! Input: planar YUV420 10-bit, u16 LE, N frames concatenated.
//! Frame 0 is KEY; frames 1..N-1 are P (INTER).
//! Writes out.fvc1 and out_rec.yuv (encoder reconstruction, for round-trip check).

use fvc1_dec::decode_file;
use fvc1_enc::partition::{encode_frame_rdo, encode_frame_rdo_preset};
use fvc1_enc::rdo::Preset;
use fvc1_enc::rdo::CostTables;
use fvc1_enc::writer::{write_container, write_frame, FrameBits};
use fvc1_enc::Picture;
use std::fs;

fn usage() -> ! {
    eprintln!(
        "usage: fvc1-enc --preset slow --input in.yuv --width W --height H \
         --output out.fvc1 --qp Q [--frames N]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut preset = String::new();
    let mut input = String::new();
    let mut width = 0usize;
    let mut height = 0usize;
    let mut output = String::new();
    let mut qp = 128u8;
    let mut frames = 1usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" => {
                i += 1;
                preset = args.get(i).cloned().unwrap_or_default();
            }
            "--input" => {
                i += 1;
                input = args.get(i).cloned().unwrap_or_default();
            }
            "--width" => {
                i += 1;
                width = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--height" => {
                i += 1;
                height = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--output" => {
                i += 1;
                output = args.get(i).cloned().unwrap_or_default();
            }
            "--qp" => {
                i += 1;
                qp = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            "--frames" => {
                i += 1;
                frames = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            _ => usage(),
        }
        i += 1;
    }
    if (preset != "slow" && preset != "fast") || input.is_empty() || width == 0 || height == 0 || output.is_empty() {
        usage();
    }
    if width % 128 != 0 || height % 128 != 0 {
        eprintln!("error: dimensions must be multiples of 128 (spec v0.2)");
        std::process::exit(2);
    }

    let data = fs::read(&input).unwrap_or_else(|e| {
        eprintln!("read error: {e}");
        std::process::exit(2);
    });
    let frame_bytes = width * height * 3; // u16 * 1.5
    if data.len() < frame_bytes * frames {
        eprintln!("error: input too short for {frames} frame(s)");
        std::process::exit(2);
    }
    let mut orig_frames = Vec::new();
    for f in 0..frames {
        let start = f * frame_bytes;
        let pic = Picture::from_yuv420_10bit(&data[start..start + frame_bytes], width, height)
            .unwrap_or_else(|| {
                eprintln!("error: bad frame data");
                std::process::exit(2);
            });
        orig_frames.push(pic);
    }

    // lambda = 0.85 * 2^((Q-128)/16)
    let lambda = 0.85 * 2.0f64.powf((qp as f64 - 128.0) / 16.0);
    println!("fvc1-enc (slow): {width}x{height}, {frames} frame(s), qp={qp}, lambda={lambda:.4}");
    let ct = CostTables::new();

    let mut frame_bits = Vec::new();
    let mut rec_frames: Vec<Picture> = Vec::new();
    let mut ref_pic: Option<Picture> = None;
    for (fi, orig) in orig_frames.iter().enumerate() {
        let is_inter = fi > 0;
        println!("  frame {fi} ({}): RDO...", if is_inter { "P" } else { "KEY" });
        let t0 = std::time::Instant::now();
        let preset_enum = if preset == "fast" { Preset::Fast } else { Preset::Slow };
        let (parts, decisions, rec) =
            encode_frame_rdo_preset(orig, ref_pic.as_ref(), lambda, qp, &ct, preset_enum);
        let fb = write_frame(width, height, is_inter, &parts, &decisions, &rec);
        println!(
            "    {} blocks, {} bytes, {:.1}s",
            decisions.len(),
            fb.data.len(),
            t0.elapsed().as_secs_f64()
        );
        frame_bits.push(fb);
        ref_pic = Some(rec.clone());
        rec_frames.push(rec);
    }

    let container = write_container(
        &frame_bits.iter().map(|f| FrameBits { data: f.data.clone() }).collect::<Vec<_>>(),
    );
    fs::write(&output, &container).unwrap_or_else(|e| {
        eprintln!("write error: {e}");
        std::process::exit(2);
    });
    println!("wrote {output} ({} bytes)", container.len());

    // write encoder reconstruction for round-trip verification
    let rec_path = output.replace(".fvc1", "_rec.yuv");
    let mut rec_bytes = Vec::new();
    for rec in &rec_frames {
        for s in rec.y.iter().chain(rec.u.iter()).chain(rec.v.iter()) {
            rec_bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    fs::write(&rec_path, &rec_bytes).unwrap_or_else(|e| {
        eprintln!("write error: {e}");
        std::process::exit(2);
    });

    // round-trip check: fvc1-dec must reproduce our reconstruction bit-exactly
    let decoded = decode_file(&container).unwrap_or_else(|e| {
        eprintln!("ROUND-TRIP FAIL: decoder rejected our stream: {e}");
        std::process::exit(1);
    });
    if decoded.len() != rec_frames.len() {
        eprintln!("ROUND-TRIP FAIL: frame count mismatch");
        std::process::exit(1);
    }
    for (fi, (dec, rec)) in decoded.iter().zip(rec_frames.iter()).enumerate() {
        if dec.planes[0] != rec.y.as_slice()
            || dec.planes[1] != rec.u.as_slice()
            || dec.planes[2] != rec.v.as_slice()
        {
            eprintln!("ROUND-TRIP FAIL: frame {fi} pixels differ");
            std::process::exit(1);
        }
    }
    println!("round-trip OK: decoder reproduces encoder reconstruction bit-exactly");
}
