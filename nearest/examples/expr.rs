#![feature(offset_of_enum)]

//! Expression tree evaluator using self-relative pointers.
//!
//! Demonstrates recursive `Near<Expr>` trees, nested construction,
//! `Region::clone`, session mutation with `splice`, and `trim`.

use nearest::{Flat, Near, Region};

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Op {
  Add,
  Sub,
  Mul,
}

#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum Expr {
  Lit(i64),
  Bin { op: Op, lhs: Near<Expr>, rhs: Near<Expr> },
}

fn eval(expr: &Expr) -> i64 {
  match expr {
    Expr::Lit(n) => *n,
    Expr::Bin { op, lhs, rhs } => {
      let l = eval(lhs);
      let r = eval(rhs);
      match op {
        Op::Add => l + r,
        Op::Sub => l - r,
        Op::Mul => l * r,
      }
    }
  }
}

fn pretty(expr: &Expr) -> String {
  match expr {
    Expr::Lit(n) => n.to_string(),
    Expr::Bin { op, lhs, rhs } => {
      let sym = match op {
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
      };
      format!("({} {} {})", pretty(lhs), sym, pretty(rhs))
    }
  }
}

fn main() {
  // Build: (2 + 3) * (10 - 4)
  let region = Region::new(Expr::make_bin(
    Op::Mul,
    Expr::make_bin(Op::Add, Expr::make_lit(2), Expr::make_lit(3)),
    Expr::make_bin(Op::Sub, Expr::make_lit(10), Expr::make_lit(4)),
  ));

  println!("expr:  {}", pretty(&region));
  println!("value: {}", eval(&region));
  assert_eq!(eval(&region), 30);

  // Clone is a plain memcpy — no pointer fixup.
  let mut cloned = region.clone();
  assert_eq!(eval(&cloned), eval(&region));
  println!("\nclone matches: {}", eval(&cloned) == eval(&region));

  // Mutate the clone: replace the `4` literal with `1`.
  // (2 + 3) * (10 - 1) = 45
  cloned.session(|s| {
    // Navigate to the rhs of the outer Mul, then the rhs of the inner Sub.
    let rhs_of_sub = s.nav(s.root(), |e| match e {
      Expr::Bin { rhs, .. } => match &**rhs {
        Expr::Bin { rhs: inner_rhs, .. } => inner_rhs,
        _ => panic!("expected Bin"),
      },
      _ => panic!("expected Bin"),
    });
    s.splice(rhs_of_sub, Expr::make_lit(1));
  });

  println!("\nmutated: {}", pretty(&cloned));
  println!("value:   {}", eval(&cloned));
  assert_eq!(eval(&cloned), 45);

  // Trim reclaims dead bytes from the old `Lit(4)`.
  let before = cloned.byte_len();
  cloned.trim();
  println!("\ntrim: {} -> {} bytes", before, cloned.byte_len());
  assert_eq!(eval(&cloned), 45);
}
