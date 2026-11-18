//! A `None` pointer is a zero offset in the place of a `Near`: the `NonZero` of an offset must
//! leave room for it, which is checked when a region uses the offset.
use nearest::{Near, Offset, Region, flat, puns::layout::Endian};

#[flat]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Wide(i32);

// SAFETY: not kept: `NonZero` is `Self`, zero included, so `Option` of it is larger.
unsafe impl Offset for Wide {
  type NonZero = Wide;
  const ENDIAN: Endian = Endian::NATIVE;
  fn from_isize(n: isize) -> Self { Wide(n as i32) }
  fn to_isize(self) -> isize { self.0 as isize }
}

#[flat]
struct List { next: Option<Near<List, Wide>> }

fn main() {
  let _ = Region::new(List::new(None));
}
