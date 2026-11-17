//! Unions are rejected.
use nearest::flat;

#[flat]
union U { a: u32, b: f32 }

fn main() {}
