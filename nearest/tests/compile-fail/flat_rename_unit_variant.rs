//! A unit variant has no constructor to rename.
use nearest::flat;

#[flat]
enum E { A(u8), #[flat(rename = "nothing")] B }

fn main() {}
