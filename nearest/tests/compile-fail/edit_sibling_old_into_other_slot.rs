//! The old value of one place cannot be moved into a sibling's slot.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn expr() -> Region<Expr> {
  Region::new(Expr::add(Near::new(Expr::Lit(1)), Near::new(Expr::Lit(2))))
}

fn main() {
  let mut e = expr();
  e.edit(|old, slot| {
    if let Expr::Add(a, b) = old {
      slot.at(a.as_pin_mut(), |old_a, _| {
        slot.at(b.as_pin_mut(), |_, new_b| new_b.put(old_a));
      });
    }
  });
}
