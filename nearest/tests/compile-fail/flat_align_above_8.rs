//! A region is 8-aligned, so its types are aligned to at most 8.
use nearest::{Region, flat};

#[flat]
#[repr(C, align(16))]
struct A { x: u64 }

fn main() {
  let _ = Region::new(A::new(1));
}
