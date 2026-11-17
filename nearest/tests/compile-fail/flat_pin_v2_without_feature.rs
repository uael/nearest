//! Without the `nightly` feature, a `#[flat]` type cannot be marked `#[pin_v2]`: it gets the
//! bounded `Unpin` impl, which such a type cannot have. The feature adds the mark by itself.
use nearest::flat;

#[flat]
#[pin_v2]
struct Node { value: u32 }

fn main() {}
