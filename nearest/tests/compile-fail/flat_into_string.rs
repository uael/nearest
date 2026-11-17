//! `into` is for plain data, not strings or lists.
use nearest::flat;

#[flat]
enum E { A(#[flat(into)] nearest::Near<str>) }

fn main() {}
