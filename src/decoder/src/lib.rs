//! fvc1-dec: FVC1 reference decoder library (spec v0.2).
//!
//! Safe Rust only. All malformed input yields `DecodeError`; the decoder
//! never panics on attacker-controlled bytes.

pub mod bitstream;
pub mod crc32;
pub mod error;
pub mod idct;
pub mod intra;
pub mod mc;
pub mod rans;
pub mod transform;

pub use bitstream::{decode_file, half_away, qstep_of, Frame};
pub use error::{DecodeError, Result};
pub use idct::{idct_1d, idct_2d};
pub use intra::intra_predict;
pub use mc::motion_compensate;
pub use transform::{dct_forward, dct_inverse, zigzag};
