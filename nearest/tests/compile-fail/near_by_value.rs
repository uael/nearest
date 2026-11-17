//! A `Near` cannot be moved out of a region.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let r = mk();
  let Expr::Neg(x) = &*r else { return };
  let stolen: Near<Expr> = *x;
  drop(stolen);
}
