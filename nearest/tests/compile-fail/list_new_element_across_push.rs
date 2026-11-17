//! The handle returned by `push` cannot be used after the next `push`.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  let insts = r.edit().insts();
  let new = insts.push(Inst::new(2, ()));
  insts.push(Inst::new(3, ()));
  *new.op() = 4;
}
