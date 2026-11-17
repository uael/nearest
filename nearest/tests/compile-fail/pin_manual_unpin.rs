//! `Unpin` cannot be implemented by hand for a stored type: a pinned value keeps its pointers.
use nearest::{Near, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

impl Unpin for Expr {}

fn main() {}
