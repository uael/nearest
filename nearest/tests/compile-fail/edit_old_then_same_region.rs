//! An `Old` kept out of its closure keeps the region borrowed.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  let mut stash = None;
  r.edit().replace(|old| {
    stash = Some(old);
    Expr::Lit(1)
  });
  r.edit().set(Expr::Lit(2));
  drop(stash);
}
