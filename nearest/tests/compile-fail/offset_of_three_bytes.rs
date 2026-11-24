//! An offset is 2, 4 or 8 bytes, which is checked when a region uses it.
use std::num::NonZero;

use nearest::{Near, Offset, Region, flat, puns::layout::Endian};

#[flat]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
struct Three([u8; 3]);

#[derive(Clone, Copy, nearest::puns::Pun)]
#[pun(crate = nearest::puns)]
#[repr(C)]
struct NonZeroThree(NonZero<u8>, [u8; 2]);

impl From<NonZeroThree> for Three {
  fn from(n: NonZeroThree) -> Self { Three([n.0.get(), n.1[0], n.1[1]]) }
}

// SAFETY: not kept: three bytes.
unsafe impl Offset for Three {
  type NonZero = NonZeroThree;
  const ENDIAN: Endian = Endian::Little;
  fn from_isize(n: isize) -> Self { Three([n as u8, (n >> 8) as u8, (n >> 16) as u8]) }
  fn to_isize(self) -> isize { (self.0[0] as isize) | (self.0[1] as isize) << 8 }
}

#[flat]
struct Text { name: Near<str, Three> }

fn main() {
  let _ = Region::new(Text::new("x"));
}
