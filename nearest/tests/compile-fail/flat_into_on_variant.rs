//! `into` goes on a field, not on its variant.
use nearest::flat;

#[flat]
enum E { #[flat(into)] A(u64) }

fn main() {}
