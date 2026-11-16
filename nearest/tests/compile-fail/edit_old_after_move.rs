//! A moved value cannot be read afterwards.
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
    slot.put(Expr::neg(Near::new(old)));
    let _ = matches!(*old, Expr::Lit(_));
  });
}
