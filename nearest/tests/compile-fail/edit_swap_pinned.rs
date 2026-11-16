//! Values that contain a `Near` are pinned: no `&mut`, so no `swap`.
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
  f.edit(|root, _| {
    let Func { entry, .. } = root;
    let Block { insts, .. } = entry.as_pin_mut();
    let mut it = insts.into_iter();
    let (x, y) = (it.next().unwrap(), it.next().unwrap());
    core::mem::swap(x.get_mut(), y.get_mut());
  });
}
