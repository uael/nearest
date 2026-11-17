//! A list of values holding a `Near` is not a `&mut [T]`.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  r.edit().insts().reverse();
}
