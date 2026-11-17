//! An expression tree simplified by rewriting in place.
//!
//! Each pass looks at a node; if a rule applies it replaces the node, moving subtrees instead
//! of copying them, and otherwise descends into the children through their handles. Passes
//! repeat until nothing changes, then the region is compacted.

use nearest::{Edit, Near, Region, flat};

#[flat]
enum Expr {
  Num(i64),
  Var(u8),
  Neg(Near<Expr>),
  Add(Near<Expr>, Near<Expr>),
  Mul(Near<Expr>, Near<Expr>),
}

const fn num(e: &Expr) -> Option<i64> {
  if let Expr::Num(n) = e { Some(*n) } else { None }
}

/// The handle of child `i` of a node that has children.
fn kid<'a, 'e>(e: &'a mut Edit<'e, Expr>, i: usize) -> &'a mut Edit<'e, Expr> {
  match e.as_mut() {
    ExprMut::Neg(a) => a,
    ExprMut::Add(args) => {
      if i == 0 {
        args._0()
      } else {
        args._1()
      }
    }
    ExprMut::Mul(args) => {
      if i == 0 {
        args._0()
      } else {
        args._1()
      }
    }
    ExprMut::Num(_) | ExprMut::Var(_) => unreachable!("a leaf has no children"),
  }
}

fn eval(e: &Expr, vars: &[i64]) -> i64 {
  match e {
    Expr::Num(n) => *n,
    Expr::Var(v) => vars[usize::from(*v)],
    Expr::Neg(a) => -eval(a, vars),
    Expr::Add(a, b) => eval(a, vars) + eval(b, vars),
    Expr::Mul(a, b) => eval(a, vars) * eval(b, vars),
  }
}

fn show(e: &Expr) -> String {
  match e {
    Expr::Num(n) => n.to_string(),
    Expr::Var(v) => char::from(b'x' + v).to_string(),
    Expr::Neg(a) => format!("-{}", show(a)),
    Expr::Add(a, b) => format!("({} + {})", show(a), show(b)),
    Expr::Mul(a, b) => format!("({} * {})", show(a), show(b)),
  }
}

/// What a rule does to a node.
enum Rule {
  /// Becomes this number.
  Num(i64),
  /// Child `i` takes the node's place.
  Hoist(usize),
  /// Grandchild `(i, j)` takes the node's place.
  Hoist2(usize, usize),
  /// `c * b` becomes `b * c`: constants go to the right.
  Swap(i64),
}

fn rule(e: &Expr) -> Option<Rule> {
  Some(match e {
    Expr::Neg(a) => match &**a {
      Expr::Num(n) => Rule::Num(-n),
      Expr::Neg(_) => Rule::Hoist2(0, 0), // --x -> x
      _ => return None,
    },
    Expr::Add(lhs, rhs) => match (num(lhs), num(rhs)) {
      (Some(x), Some(y)) => Rule::Num(x + y),
      (Some(0), _) => Rule::Hoist(1),
      (_, Some(0)) => Rule::Hoist(0),
      _ => return None,
    },
    Expr::Mul(lhs, rhs) => match (num(lhs), num(rhs)) {
      (Some(x), Some(y)) => Rule::Num(x * y),
      (Some(0), _) | (_, Some(0)) => Rule::Num(0),
      (Some(1), _) => Rule::Hoist(1),
      (_, Some(1)) => Rule::Hoist(0),
      (Some(c), None) => Rule::Swap(c),
      _ => return None,
    },
    Expr::Num(_) | Expr::Var(_) => return None,
  })
}

/// One rewriting pass over `e`; returns whether anything changed. Subtrees move: `replace`
/// takes a child out, `set` puts it where its parent was.
fn rewrite(e: &mut Edit<Expr>) -> bool {
  match rule(e) {
    Some(Rule::Num(n)) => e.set(Expr::Num(n)),
    Some(Rule::Hoist(i)) => {
      let child = kid(e, i).replace(Expr::Num(0));
      e.set(child);
    }
    Some(Rule::Hoist2(i, j)) => {
      let grandchild = kid(kid(e, i), j).replace(Expr::Num(0));
      e.set(grandchild);
    }
    Some(Rule::Swap(c)) => {
      let b = kid(e, 1).replace(Expr::Num(0));
      e.set(Expr::mul(b, Expr::Num(c)));
    }
    None => {
      return match e.as_mut() {
        ExprMut::Num(_) | ExprMut::Var(_) => false,
        ExprMut::Neg(a) => rewrite(a),
        ExprMut::Add(args) => rewrite(args._0()) | rewrite(args._1()),
        ExprMut::Mul(args) => rewrite(args._0()) | rewrite(args._1()),
      };
    }
  }
  true
}

fn main() {
  let (x, y, n) = (Expr::Var(0), Expr::Var(1), Expr::Num);
  // ((x + 0) * (2 + 3)) + --(1 * y)
  let mut e = Region::new(Expr::add(
    Expr::mul(Expr::add(x, n(0)), Expr::add(n(2), n(3))),
    Expr::neg(Expr::neg(Expr::mul(n(1), y))),
  ));
  let original = e.clone();
  println!("before: {}  ({} bytes)", show(&e), e.byte_len());

  let mut passes = 0;
  while rewrite(e.edit()) {
    passes += 1;
  }
  println!("after {passes} passes: {}  ({} bytes)", show(&e), e.byte_len());
  assert_eq!(show(&e), "((x * 5) + y)");
  assert_eq!(show(&original), "(((x + 0) * (2 + 3)) + --(1 * y))");
  for vars in [[0, 0], [3, 4], [-7, 11]] {
    assert_eq!(eval(&e, &vars), eval(&original, &vars));
  }

  // Edits append; a copy keeps only what is reachable.
  let compact = Region::new(&*e);
  println!("compacted: {} bytes", compact.byte_len());
  assert!(compact.byte_len() < original.byte_len());
  assert_eq!(show(&compact), show(&e));
}
