//! Two handles of one region cannot be live at once.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  let a = r.edit();
  let b = r.edit();
  a.set(Expr::Lit(1));
  b.set(Expr::Lit(2));
}
