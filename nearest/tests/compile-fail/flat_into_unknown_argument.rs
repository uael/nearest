//! A field takes `into` and nothing else.
use nearest::flat;

#[flat]
struct S { #[flat(from)] a: u8 }

fn main() {}
