//! A region with inline storage cannot move while a handle of its edit lives: the bytes would
//! move under it.
use nearest::{Inline, Near, Region, flat};

#[flat]
struct Msg { id: u32, body: Near<str> }

fn main() {
  let mut r = Region::new_in(Inline::<64>::default(), Msg::new(1, "a"));
  let body = r.edit().body();
  let moved = r;
  body.set("b");
  drop(moved);
}
