//! `Near` is not `Copy`.
use nearest::Near;

fn copy(n: &Near<u32>) -> Near<u32> {
  *n
}

fn main() {}
