//! Nor one of another width, in a struct of the same size.
use nearest::{
  Near, Region, flat,
  puns::num::{i16le, i32le},
};

#[flat]
struct Wide {
  pointer: Near<u32, i32le>,
}

#[flat]
struct Narrow {
  pointer: Near<u32, i16le>,
  more: [u8; 2],
}

fn main() {
  let region = Region::new(Wide::new(1));
  let _: &Narrow = nearest::puns::view_ref(&*region);
}
