//! The new name is an identifier.
use nearest::flat;

#[flat]
enum E { #[flat(rename = "make one")] A(u8) }

fn main() {}
