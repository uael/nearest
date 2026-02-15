//! A struct with a non-Flat field (String) must not compile.

use nearest::Flat;

#[derive(Flat)]
struct Bad {
  name: String,
}

fn main() {}
