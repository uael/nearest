//! An expression tree simplified by rewriting in place.
//!
//! Each pass looks at a node; if a rule applies it replaces the node, moving subtrees instead
//! of copying them, and otherwise descends into the children with `Slot::at`. Passes repeat
//! until nothing changes, then the region is compacted.
#![feature(pin_ergonomics)]
#![expect(incomplete_features, reason = "pin_ergonomics")]

use std::pin::Pin;

use nearest::{Near, Region, Slot, flat};

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

/// One rewriting pass over `e`; returns whether anything changed.
fn rewrite<'o>(e: Pin<&'o mut Expr>, slot: &mut Slot<'o, Expr>) -> bool {
  match e {
    Expr::Num(_) | Expr::Var(_) => false,
    Expr::Neg(a) => match &**a {
      Expr::Num(n) => {
        slot.put(Expr::Num(-n));
        true
      }
      // --x -> x: the grandchild moves up.
      Expr::Neg(_) => {
        if let Expr::Neg(x) = a.as_pin_mut() {
          slot.put(x.as_pin_mut());
        }
        true
      }
      _ => slot.at(a.as_pin_mut(), rewrite),
    },
    Expr::Add(a, b) => match (num(&a), num(&b)) {
      (Some(x), Some(y)) => {
        slot.put(Expr::Num(x + y));
        true
      }
      (Some(0), _) => {
        slot.put(b.as_pin_mut());
        true
      }
      (_, Some(0)) => {
        slot.put(a.as_pin_mut());
        true
      }
      _ => slot.at(a.as_pin_mut(), rewrite) | slot.at(b.as_pin_mut(), rewrite),
    },
    Expr::Mul(a, b) => match (num(&a), num(&b)) {
      (Some(x), Some(y)) => {
        slot.put(Expr::Num(x * y));
        true
      }
      (Some(0), _) | (_, Some(0)) => {
        slot.put(Expr::Num(0));
        true
      }
      (Some(1), _) => {
        slot.put(b.as_pin_mut());
        true
      }
      (_, Some(1)) => {
        slot.put(a.as_pin_mut());
        true
      }
      // Constants go to the right: the children swap places.
      (Some(_), None) => {
        slot.put(Expr::mul(b, a));
        true
      }
      _ => slot.at(a.as_pin_mut(), rewrite) | slot.at(b.as_pin_mut(), rewrite),
    },
  }
}

fn main() {
  let (x, y) = (|| Near::new(Expr::Var(0)), || Near::new(Expr::Var(1)));
  let n = |n| Near::new(Expr::Num(n));
  // ((x + 0) * (2 + 3)) + --(1 * y)
  let mut e = Region::new(Expr::add(
    Near::new(Expr::mul(Near::new(Expr::add(x(), n(0))), Near::new(Expr::add(n(2), n(3))))),
    Near::new(Expr::neg(Near::new(Expr::neg(Near::new(Expr::mul(n(1), y())))))),
  ));
  let original = e.clone();
  println!("before: {}  ({} bytes)", show(&e), e.byte_len());

  let mut passes = 0;
  while e.edit(rewrite) {
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
