//! An `Old` is used at most once.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  r.edit().replace(|old| {
    let again = old;
    Expr::add(old, again)
  });
}
