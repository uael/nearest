//! An unknown `#[flat(...)]` attribute on a field must not compile.

use nearest::Flat;

#[derive(Flat)]
struct Bad {
  #[flat(skip)]
  x: u32,
}

fn main() {}
