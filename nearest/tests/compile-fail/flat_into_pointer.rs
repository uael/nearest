//! `into` is for plain data: a pointer's parameter takes a builder of its pointee.
use nearest::flat;

#[flat]
struct S { #[flat(into)] p: nearest::Near<u32> }

fn main() {}
