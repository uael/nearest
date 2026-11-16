//! `at` needs the place for the whole level, not a reborrow.
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
  let mut f = func();
  f.edit(|root, slot| {
    let Func { entry, .. } = root;
    let Block { insts, .. } = entry.as_pin_mut();
    let mut insts = insts;
    slot.at(insts.as_mut(), |_, new| new.put(()));
    let _ = insts.len();
  });
}
