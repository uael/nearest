//! A handle cannot outlive its region.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let e = {
    let mut r = mk();
    r.edit()
  };
  e.set(Expr::Lit(1));
}
