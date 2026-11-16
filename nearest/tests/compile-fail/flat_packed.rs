//! Packed structs would have unaligned fields.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::flat;

#[flat]
#[repr(C, packed)]
struct P { a: u8, b: u32 }

fn main() {}
