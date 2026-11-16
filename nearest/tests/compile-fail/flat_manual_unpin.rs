//! `#[flat]` types are `#[pin_v2]`, which forbids a manual `Unpin`.
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Near, flat};

#[flat]
struct S { p: Near<u32> }

impl Unpin for S {}

fn main() {}
