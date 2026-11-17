//! The reference given to `Old::map` cannot escape.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut r = mk();
  let mut out: Option<&Expr> = None;
  r.edit().replace(|old| {
    old.map(|e| {
      out = Some(e);
      e
    })
  });
  drop(out);
}
