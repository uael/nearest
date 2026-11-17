//! Packed structs would have unaligned fields.
use nearest::flat;

#[flat]
#[repr(C, packed)]
struct P { a: u8, b: u32 }

fn main() {}
