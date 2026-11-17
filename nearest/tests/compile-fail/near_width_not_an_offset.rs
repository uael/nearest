//! A pointer stores its distance in an `Offset`: `u8` is not one.
use nearest::{Near, flat};

#[flat]
struct Byte { next: Option<Near<Byte, u8>> }

fn main() {}
