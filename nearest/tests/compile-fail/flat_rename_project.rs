//! `project` is the generated method for pinned values.
use nearest::flat;

#[flat]
enum E { #[flat(rename = "project")] A(u8) }

fn main() {}
