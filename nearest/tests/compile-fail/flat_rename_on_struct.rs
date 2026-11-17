//! A struct's constructor is `new`: `rename` is for enum variants.
use nearest::flat;

#[flat]
#[flat(rename = "make")]
struct S { a: u8 }

fn main() {}
