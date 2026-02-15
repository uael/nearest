//! Pure-Rust CRC32 (ISO 3309 / ITU-T V.42, polynomial `0xEDB8_8320`).
//!
//! Uses a 256-entry lookup table computed at compile time. The algorithm
//! matches the widely deployed CRC-32/ISO-HDLC variant (sometimes called
//! CRC-32b), which is the same checksum used by gzip, PNG, and zlib.

/// Pre-computed CRC32 lookup table (reflected polynomial `0xEDB8_8320`).
const TABLE: [u32; 256] = {
  let mut table = [0u32; 256];
  let mut i: u32 = 0;
  while i < 256 {
    let mut crc = i;
    let mut j = 0;
    while j < 8 {
      if crc & 1 != 0 {
        crc = (crc >> 1) ^ 0xEDB8_8320;
      } else {
        crc >>= 1;
      }
      j += 1;
    }
    table[i as usize] = crc;
    i += 1;
  }
  table
};

/// Compute CRC32 of `data`.
///
/// Returns the standard CRC-32/ISO-HDLC checksum.
pub fn checksum(data: &[u8]) -> u32 {
  let mut crc: u32 = 0xFFFF_FFFF;
  let mut i = 0;
  while i < data.len() {
    let index = ((crc ^ u32::from(data[i])) & 0xFF) as usize;
    crc = (crc >> 8) ^ TABLE[index];
    i += 1;
  }
  crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn known_vector() {
    // "123456789" has well-known CRC32 = 0xCBF4_3926
    let data = b"123456789";
    assert_eq!(checksum(data), 0xCBF4_3926);
  }

  #[test]
  fn empty_input() {
    assert_eq!(checksum(b""), 0x0000_0000);
  }

  #[test]
  fn single_byte() {
    // CRC32 of a single 0x00 byte is 0xD202_EF8D
    assert_eq!(checksum(&[0x00]), 0xD202_EF8D);
  }

  #[test]
  fn deterministic() {
    let data = b"hello world";
    let c1 = checksum(data);
    let c2 = checksum(data);
    assert_eq!(c1, c2);
  }
}
