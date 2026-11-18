//! A value is not read as a type that has fewer values: here a variant less, behind a pointer.
use nearest::{Near, Region, flat};

#[flat]
enum New {
  Leaf(u32),
  Node(Near<New>),
  More(Near<[New]>),
}

#[flat]
enum Old {
  Leaf(u32),
  Node(Near<Old>),
}

fn main() {
  let region = Region::new(New::Leaf(1));
  let _: &Old = nearest::puns::view_ref(&*region);
}
