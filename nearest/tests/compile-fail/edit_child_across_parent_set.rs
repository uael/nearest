//! A child handle cannot be used after its parent was replaced.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  let e = r.edit();
  let ExprMut::Neg(x) = e.as_mut() else { return };
  let ExprMut::Lit(n) = x.as_mut() else { return };
  e.set(Expr::Lit(2));
  *n = 3;
}
