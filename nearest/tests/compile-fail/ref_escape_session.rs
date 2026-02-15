//! A `Ref` must not escape the session closure.

use nearest::{Flat, NearList, Ref, Region, empty};

#[derive(Flat, Debug)]
struct Block {
  name: u32,
  items: NearList<u32>,
}

fn main() {
  let mut region: Region<Block> = Region::new(Block::make(42, empty()));
  let escaped: Ref<'_, Block> = region.session(|s| {
    s.root() // Ref escapes the closure — must not compile.
  });
  let _ = escaped;
}
