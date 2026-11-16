//! The valid counterparts of the programs in `compile-fail/`.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Near, Region, flat};

#[flat]
struct Func { name: Near<str>, entry: Near<Block> }
#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }
#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
#[flat]
#[repr(C, align(8))]
struct A { x: u64 }

fn main() {
  let mut f = Region::new(Func::new("f", Near::new(Block::new(0, (Inst::new(1, [1]), Inst::new(2, [2])), None))));
  let other = f.clone();
  f.edit(|root, slot| {
    let Func { entry, name } = root;
    let Block { insts, next, .. } = entry.as_pin_mut();
    let n = insts.len();
    slot.at(insts, |old, new| {
      for i in old {
        new.push(i);
      }
    });
    slot.at(next, |_, new| new.put(Near::new(Block::new(1, &other.entry.insts[..], None))));
    slot.at(name, |old, new| new.put(&*old));
    assert_eq!(n, 2);
  });
  let mut e = Region::new(Expr::add(Near::new(Expr::Lit(1)), Near::new(Expr::Lit(2))));
  e.edit(|old, slot| {
    if let Expr::Add(a, b) = old {
      slot.at(a.as_pin_mut(), |old_a, new_a| new_a.put(Expr::neg(Near::new(old_a))));
      slot.put(Expr::add(b, Near::new(Expr::Lit(3))));
    }
  });
  if let Expr::Add(a, _) = &*e {
    let _copy = Region::new(&**a);
  }
  let _ = Region::new(A::new(1));
}
