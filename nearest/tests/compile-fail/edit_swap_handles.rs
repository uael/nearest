//! Handles are unsized, so they cannot be swapped.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let (mut a, mut b) = (mk(), mk());
  std::mem::swap(a.edit(), b.edit());
}
