//! `into` once per field.
use nearest::flat;

#[flat]
struct S {
  #[flat(into)]
  #[flat(into)]
  a: u64,
}

fn main() {}
