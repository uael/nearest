//! Loading bytes checks the widths of the pointers too, here behind a list.
use nearest::{Near, Region, flat};

#[flat]
struct Inner { next: Option<Near<Inner, i16>> }

// Not a `Pun` type until `Option<Near<..>>` is one.
#[flat(no_pun)]
struct Outer { items: Near<[Inner]> }

fn main() {
  let _ = Region::<Outer>::from_bytes(&[]);
}
