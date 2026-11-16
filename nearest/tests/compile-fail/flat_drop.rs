//! A type with drop glue cannot live in a region.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Region, flat};

#[flat]
struct D { x: u32 }

impl Drop for D {
  fn drop(&mut self) {}
}

fn main() {
  let _ = Region::new(D::new(1));
}
