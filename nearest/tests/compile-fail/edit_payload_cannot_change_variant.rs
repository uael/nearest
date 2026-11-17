//! The handle of a variant's fields cannot replace the value, which would change the variant.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }

fn main() {
  let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::Lit(2)));
  if let ExprMut::Add(args) = r.edit().as_mut() {
    args.0.set(Expr::Lit(3));
    args._0().set(Expr::Lit(4));
  }
}
