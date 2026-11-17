//! A `&mut` to plain data cannot be held across a `push`.
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn mk() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [1, 2])], None)) }

fn main() {
  let mut r = mk();
  let args = r.edit().insts().at(0).args();
  let s: &mut [u32] = &mut *args;
  args.push(3);
  s[0] = 1;
}
