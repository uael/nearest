//! The handle is borrowed while the closure of `update` runs.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::Lit(1));
  let e = r.edit();
  e.update(|old| {
    e.set(Expr::Lit(2));
    Expr::neg(old)
  });
}
