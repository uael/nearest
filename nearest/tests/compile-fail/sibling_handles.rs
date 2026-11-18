//! Two fields cannot be edited at once.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  let b = r.edit();
  let mut id = b.id();
  b.insts().push(Inst::new(2, ()));
  *id = 1;
}
