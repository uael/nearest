//! A type implementing Drop must not derive Flat.

use nearest::Flat;

#[derive(Flat)]
struct Bad {
  x: u32,
}

impl Drop for Bad {
  fn drop(&mut self) {}
}

fn main() {}
