//! `Edit<X>` is only ever behind a `&mut`.
use nearest::{Edit, Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn mk() -> Region<Expr> { Region::new(Expr::neg(Expr::Lit(1))) }

fn take(_: Edit<Expr>) {}

fn main() {}
