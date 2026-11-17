//! A region holds only types of non-zero size.
use nearest::{Region, flat};

#[flat]
struct Z;

fn main() {
  let _ = Region::new(Z);
}
