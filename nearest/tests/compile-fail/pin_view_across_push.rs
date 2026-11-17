//! A pinned view borrows its handle: nothing grows the buffer while it lives.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

#[flat]
struct List { items: Near<[u32]> }

fn main() {
  let mut r = Region::new(List::new([1]));
  let l = r.edit();
  let ListPin { items } = l.pin().project();
  let first = items.iter_pin_mut().next().unwrap();
  l.items().push(2);
  let _ = *first;
}
