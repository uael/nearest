//! The attribute takes no arguments.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::flat;

#[flat(copy)]
struct S { a: u8 }

fn main() {}
