//! Loading and viewing bytes reject the same types as building: a region is 8-aligned.
use nearest::{Region, flat};

#[flat]
#[repr(C, align(16))]
struct A { x: u64 }

fn main() {
  let _ = Region::<A>::view(&[0; 16]);
}
