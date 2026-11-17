//! A pointer cannot lead to a type holding a narrower pointer: the root's pointers bound the
//! region, so a narrower pointer below could not reach across it.
use nearest::{Near, Region, flat};

#[flat]
struct Inner { name: Near<str, i16> }

#[flat]
struct Outer { inner: Near<Inner, i64> }

fn main() {
  let _ = Region::new(Outer::new(Inner::new("x")));
}
