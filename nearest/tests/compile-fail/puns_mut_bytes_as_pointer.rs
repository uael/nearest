//! No `&mut` to a pointer from `puns` either (I2): bytes written through it would aim anywhere.
use nearest::Near;

fn main() {
  let mut bytes = [4u8, 0, 0, 0];
  let _ = nearest::puns::from_bytes_mut::<Near<u32>>(&mut bytes);
}
