//! `into` is for plain data: a pointer's parameter takes a builder of its target.
use nearest::flat;

#[flat]
struct S { #[flat(into)] p: nearest::Near<u32> }

fn main() {}
