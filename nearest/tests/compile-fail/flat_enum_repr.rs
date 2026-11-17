//! Enums are `#[repr(u8)]`.
use nearest::flat;

#[flat]
#[repr(u16)]
enum Wide { A, B }

#[flat]
#[repr(C, u8)]
enum Tagged { A(u32), B }

fn main() {}
