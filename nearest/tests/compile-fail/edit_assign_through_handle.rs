//! A value holding a `Near` cannot be assigned through its handle.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn main() {
  let mut a = mk();
  **a.edit() = Expr::Lit(1);
}
