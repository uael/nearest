//! Optional pointers are to sized values only.
use nearest::{Near, flat};

#[flat]
struct S { maybe: Option<Near<[u32]>> }

fn main() {}
