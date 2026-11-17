//! The handle stays borrowed for as long as the old value of `update` is kept, so it cannot
//! move back in later.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::neg(Expr::Lit(1)));
  let e = r.edit();
  let mut stash = None;
  e.update(|old| {
    stash = Some(old);
    Expr::Lit(2)
  });
  e.set(Expr::Lit(3));
  drop(stash);
}
