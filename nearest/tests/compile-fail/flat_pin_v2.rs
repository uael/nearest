//! A `#[flat]` type cannot be marked `#[pin_v2]`: it gets the bounded `Unpin` impl, which such a
//! type cannot have.
use nearest::flat;

#[flat]
#[pin_v2]
struct Node { value: u32 }

fn main() {}
