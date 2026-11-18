//! `puns` reads a pointer as a pointer whose offset is the same integer: not one of another byte
//! order, whose bytes are another distance.
use nearest::{
  Near, Region,
  puns::num::{i32be, i32le},
};

fn main() {
  let region = Region::<Near<u32, i32le>>::new(Near::new(1));
  let _: &Near<u32, i32be> = nearest::puns::view_ref(&*region);
}
