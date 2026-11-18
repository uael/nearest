//! Loading and viewing bytes reject the same types as building: a region is aligned as a `u64`.
use nearest::flat;

#[flat]
#[repr(C, align(16))]
struct A { x: u64 }

fn main() {
  let _ = nearest::view::<A>(&[0; 16]);
}
