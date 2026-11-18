//! No bytes are a value that holds a pointer: where a pointer aims is not a property of its
//! bytes (I1: a `Near` only exists in a region). Bytes are loaded by `nearest`, which checks the
//! whole region.
use nearest::{Near, flat};

#[flat]
struct Node {
  id: u32,
  next: Near<Node>,
}

fn main() {
  let _ = nearest::puns::from_bytes::<Node>(&[0; 8]);
}
