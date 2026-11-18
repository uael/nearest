//! A region is aligned as a `u64` (at most 8 bytes), so its types are aligned to no more.
use nearest::{Region, flat};

#[flat]
#[repr(C, align(16))]
struct A { x: u64 }

fn main() {
  let _ = Region::new(A::new(1));
}
