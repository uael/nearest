//! A pinned value that contains a pointer gives no `&mut`.
use core::pin::Pin;

use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::neg(Expr::Lit(1)));
  let pinned: Pin<&mut Expr> = r.edit().pin();
  let _: &mut Expr = Pin::get_mut(pinned);
}
