//! A field written `Near<X>` takes a builder of `X`, not of the pointer.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let _ = Region::new(Expr::neg(Near::new(Expr::Lit(1))));
}
