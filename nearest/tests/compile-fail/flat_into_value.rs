//! `into` is a flag.
use nearest::flat;

#[flat]
struct S { #[flat(into = "u8")] a: u64 }

fn main() {}
