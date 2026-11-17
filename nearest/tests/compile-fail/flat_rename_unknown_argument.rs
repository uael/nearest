//! A variant takes `rename` and nothing else.
use nearest::flat;

#[flat]
enum E { #[flat(name = "make")] A(u8) }

fn main() {}
