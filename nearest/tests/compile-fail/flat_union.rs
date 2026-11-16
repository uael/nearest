//! Unions are rejected.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::flat;

#[flat]
union U { a: u32, b: f32 }

fn main() {}
