//! A reference read through a handle cannot be held across an edit.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  let e = r.edit();
  let v: &Expr = &**e;
  e.set(Expr::Lit(1));
  let _ = matches!(v, Expr::Lit(_));
}
