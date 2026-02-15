//! An enum with explicit discriminants must not compile.

use nearest::Flat;

#[derive(Flat)]
#[repr(u8)]
enum Bad {
  A = 5,
  B = 10,
}

fn main() {}
