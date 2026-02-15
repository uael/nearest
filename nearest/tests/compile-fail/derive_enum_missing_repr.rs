//! An enum without #[repr(u8)] must not compile.

use nearest::Flat;

#[derive(Flat)]
enum Bad {
  A,
  B,
}

fn main() {}
