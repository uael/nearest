//! A nested slot only accepts moves from its own old value.
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
      slot.at(a.as_pin_mut(), |_, new_a| new_a.put(Expr::neg(b)));
    }
  });
}
