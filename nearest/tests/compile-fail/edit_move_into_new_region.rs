//! A pin from an edit cannot be moved into another region.
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
  e.edit(|old, _| {
    let _stolen: Region<Expr> = Region::new(old);
  });
}
