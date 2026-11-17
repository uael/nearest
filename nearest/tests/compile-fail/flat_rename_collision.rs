//! A new name may not be another variant's constructor.
use nearest::flat;

#[flat]
enum E { #[flat(rename = "two")] One(u8), Two(u16) }

fn main() {}
