//! Every field must be `Flat`.
use nearest::flat;

#[flat]
struct S { name: String }

fn main() {}
