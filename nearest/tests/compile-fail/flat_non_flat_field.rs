//! Every field must be `Flat`.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::flat;

#[flat]
struct S { name: String }

fn main() {}
