//! Variants are numbered in order.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::flat;

#[flat]
enum E { A = 1, B }

fn main() {}
