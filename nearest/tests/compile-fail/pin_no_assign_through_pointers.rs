//! A pinned pointer is not assignable: there is no `Near` to assign.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::Lit(2)));
  if let ExprPin::Add(mut a, b) = r.edit().pin().project() {
    std::mem::swap(&mut *a, &mut *b.get_mut());
  }
}
