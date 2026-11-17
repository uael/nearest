//! A list cannot push a copy of its own element.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  let insts = r.edit().insts();
  insts.push(&insts[0]);
}
