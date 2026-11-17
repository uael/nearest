//! Generated public items are documented: a crate that denies `missing_docs` builds.
#![deny(missing_docs)]

use nearest::{Near, Region, flat};

/// A tree.
#[flat]
pub struct Tree {
  /// Its number.
  pub id: u32,
  /// Its children.
  pub kids: Near<[Tree]>,
}

/// An expression.
#[flat]
pub enum Expr {
  /// A number.
  Lit(i64),
  /// A sum.
  Add {
    /// The left side.
    left: Near<Expr>,
    /// The right side.
    right: Near<Expr>,
  },
}

fn main() {
  let mut t = Region::new(Tree::new(1, ()));
  *t.edit().pin().project().id += 1;
  let mut e = Region::new(Expr::add(Expr::Lit(1), Expr::Lit(2)));
  if let ExprPin::Add { left, right } = e.edit().pin().project() {
    let _ = (left.as_pin_mut(), right.as_pin_mut());
  }
  assert_eq!(t.id, 2);
}
