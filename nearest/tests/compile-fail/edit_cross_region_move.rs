//! A pin from one region's edit cannot be moved into another region's slot.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Near, Region, flat};

#[flat]
struct Func { name: Near<str>, entry: Near<Block> }
#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn func() -> Region<Func> {
  Region::new(Func::new("f", Near::new(Block::new(0, (Inst::new(1, [1]), Inst::new(2, [2])), None))))
}

fn main() {
  let (mut a, mut b) = (func(), func());
  a.edit(|ra, sa| {
    b.edit(|rb, _| {
      let Func { entry, .. } = rb;
      let Func { entry: mine, .. } = ra;
      sa.at(mine, |_, new| new.put(entry));
    });
  });
}
