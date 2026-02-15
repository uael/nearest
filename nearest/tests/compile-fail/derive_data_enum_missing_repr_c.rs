//! A data enum with #[repr(u8)] but missing #[repr(C, u8)] must not compile.

use nearest::Flat;

#[derive(Flat)]
#[repr(u8)]
enum Bad {
  A,
  B { x: u32 },
}

fn main() {}
