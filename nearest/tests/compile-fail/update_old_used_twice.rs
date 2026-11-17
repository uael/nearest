//! In `update`, the old value is used at most once.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::Lit(1));
  r.edit().update(|old| Expr::add(old, old));
}
