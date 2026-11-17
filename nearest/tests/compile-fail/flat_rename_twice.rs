//! One name per variant.
use nearest::flat;

#[flat]
enum E { #[flat(rename = "a", rename = "b")] A(u8) }

fn main() {}
