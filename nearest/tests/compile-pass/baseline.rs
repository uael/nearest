//! The valid counterparts of the programs in `compile-fail/`.
use nearest::{Edit, Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }
#[flat]
#[repr(C, align(8))]
struct A { x: u64 }
#[flat]
struct S { maybe: Option<Near<u32>> }

fn take(_: &mut Edit<Expr>) {}
fn first<'a, 'e>(e: &'a mut Edit<'e, Expr>) -> &'a mut Edit<'e, Expr> {
  e
}

fn main() {
  let mut r = Region::new(Expr::neg(Expr::Lit(1)));
  let e = r.edit();
  if let ExprMut::Neg(x) = e.as_mut() {
    if let ExprMut::Lit(n) = x.as_mut() {
      *n = 3;
    }
    x.set(Expr::Lit(2));
  }
  let old = e.replace(Expr::Lit(0));
  e.set(Expr::add(old, Expr::Lit(1)));
  take(first(e));
  let v: &Expr = &**e;
  let _ = matches!(v, Expr::Add(..));
  let old = r.edit().replace(Expr::Lit(1));
  let copy = Region::new(old);
  r.edit().set(&*copy);
  let (mut s, mut t) = (Region::new(Expr::Lit(1)), Region::new(Expr::Lit(2)));
  let old = s.edit().replace(Expr::Lit(0));
  t.edit().set(Expr::neg(old));

  let mut b = Region::new(Block::new(0, [Inst::new(1, [1, 2])], None));
  let insts = b.edit().insts();
  let first = Region::new(&insts[0]);
  insts.push(&*first);
  insts.push(Inst::new(2, ())).args().push(1);
  *insts.at(0).op() = 3;
  let args = insts.at(0).args();
  args[0] = 1;
  args.push(3);
  b.edit().next().insert(Block::new(1, (), None)).insts().push(Inst::new(4, ()));
  b.edit().next().set(None);
  *b.edit().id() = 1;
  b.edit().insts().push(Inst::new(5, ()));
  let _ = Region::new(A::new(1));
  let _ = Region::new(S::new(Near::new(1)));

  let mut inline = Region::new_in(nearest::Inline::<256>::default(), Block::new(0, (), None));
  inline.edit().insts().push(Inst::new(1, ()));
  let moved = inline;
  let _ = moved.insts.len();
}
