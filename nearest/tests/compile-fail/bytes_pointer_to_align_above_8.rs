//! The types behind pointers are checked too when bytes are loaded.
use nearest::{Near, Region, flat};

#[flat]
#[repr(C, align(16))]
struct A { x: u64 }

#[flat]
struct B { list: Near<[A]> }

fn main() {
  let _ = Region::<B>::from_bytes(&[]);
}
