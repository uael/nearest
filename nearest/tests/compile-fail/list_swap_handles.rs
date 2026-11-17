//! List handles are unsized, so they cannot be swapped.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let (mut a, mut b) = (mk(), mk());
  std::mem::swap(a.edit().insts(), b.edit().insts());
}
