//! A pointer stores its distance in `i16`, `i32` or `i64`, nothing else.
use nearest::{Near, flat};

#[flat]
struct Byte { next: Option<Near<Byte, u8>> }

fn main() {}
