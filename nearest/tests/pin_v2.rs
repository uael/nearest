//! On nightly, a type also marked `#[pin_v2]` is projected by patterns: no `project` needed.
//! Built only with `--cfg nearest_nightly` (the Miri job of CI), on a nightly compiler.
#![cfg(nearest_nightly)]
#![feature(pin_ergonomics)]
#![expect(incomplete_features, reason = "pin_ergonomics")]

use std::pin::Pin;

use nearest::{Near, Region, flat};

#[flat]
#[pin_v2]
enum Expr {
  Lit(i64),
  Neg(Near<Expr>),
  Add(Near<Expr>, Near<Expr>),
}

#[flat]
#[pin_v2]
struct Block {
  id: u32,
  insts: Near<[Inst]>,
}

#[flat]
#[pin_v2]
struct Inst {
  op: u8,
  args: Near<[u32]>,
}

/// Adds `by` to every literal.
fn bump(e: Pin<&mut Expr>, by: i64) {
  match e {
    // A binding is the pinned field; `mut` would copy the number out instead.
    Expr::Lit(n) => *n.get_mut() += by,
    Expr::Neg(x) => bump(x.as_pin_mut(), by),
    Expr::Add(a, b) => {
      let (a, b) = (a.as_pin_mut(), b.as_pin_mut());
      bump(b, by);
      bump(a, by);
    }
  }
}

fn eval(e: &Expr) -> i64 {
  match e {
    Expr::Lit(n) => *n,
    Expr::Neg(x) => -eval(x),
    Expr::Add(a, b) => eval(a) + eval(b),
  }
}

#[test]
fn patterns_on_pinned_values() {
  let mut r = Region::new(Expr::add(Expr::neg(Expr::Lit(1)), Expr::Lit(2)));
  bump(r.edit().pin(), 5);
  assert_eq!(eval(&r), -6 + 7);
  // `project` is generated all the same.
  if let ExprPin::Add(a, _) = r.edit().pin().project() {
    bump(a.as_pin_mut(), 1);
  }
  assert_eq!(eval(&r), -7 + 7);

  let mut b = Region::new(Block::new(1, [Inst::new(1, [1])]));
  let Block { id, insts } = b.edit().pin();
  let id = id.get_mut();
  for inst in insts.iter_pin_mut() {
    let Inst { op, args } = inst;
    *op.get_mut() += 1;
    *id += args.len() as u32;
  }
  assert_eq!((b.id, b.insts[0].op), (2, 2));
  b.edit().insts().push(Inst::new(3, ()));
  assert_eq!(b.insts.len(), 2);
}
