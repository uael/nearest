//! A reference into a region with inline storage cannot outlive a move of the region.
use nearest::{Inline, Near, Region, flat};

#[flat]
struct Msg { id: u32, body: Near<str> }

fn main() {
  let r = Region::new_in(Inline::<64>::default(), Msg::new(1, "a"));
  let body: &str = &r.body;
  let moved = r;
  assert_eq!(body, "a");
  drop(moved);
}
