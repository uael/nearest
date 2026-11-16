//! `Near` is not `Clone`, so neither is anything holding one.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::Near;

#[derive(Clone)]
struct Wrap(Near<u32>);

fn main() {}
