//! A `Ref` from one session must not work in another session.

use nearest::{Flat, NearList, Region, empty};

#[derive(Flat, Debug)]
struct Block {
  name: u32,
  items: NearList<u32>,
}

fn main() {
  let mut region: Region<Block> = Region::new(Block::make(42, empty()));
  let mut region2: Region<Block> = Region::new(Block::make(99, empty()));

  region.session(|s1| {
    let r = s1.root();
    region2.session(|s2| {
      // Use a Ref from s1 inside s2 — must not compile.
      let _ = s2.at(r);
    });
  });
}
