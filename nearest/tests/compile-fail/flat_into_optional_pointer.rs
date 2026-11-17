//! `into` is for plain data, not optional pointers.
use nearest::flat;

#[flat]
struct S { a: u8, #[flat(into)] next: Option<nearest::Near<S>> }

fn main() {}
