//! An element handle cannot be used after its list was pushed to.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  let insts = r.edit().insts();
  let first = insts.at(0);
  insts.push(Inst::new(2, ()));
  *first.op() = 3;
}
