//! An offset promises what `Offset` documents: implementing it is `unsafe`.
use std::num::NonZero;

use nearest::{Offset, flat};

#[flat]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Le(i32);

impl From<NonZero<i32>> for Le {
  fn from(n: NonZero<i32>) -> Self { Le(n.get()) }
}

impl Offset for Le {
  type NonZero = NonZero<i32>;
  fn from_isize(n: isize) -> Self { Le((n as i32).to_le()) }
  fn to_isize(self) -> isize { i32::from_le(self.0) as isize }
}

fn main() {}
