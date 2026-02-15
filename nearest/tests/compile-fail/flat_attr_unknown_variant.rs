//! An unknown `#[flat(...)]` attribute on a variant must not compile.

use nearest::Flat;

#[derive(Flat)]
#[repr(u8)]
enum Bad {
  #[flat(skip)]
  A,
  B,
}

fn main() {}
