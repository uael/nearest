//! A moved value stays on the thread of its edit.
use nearest::{Near, Region, flat};

#[flat]
struct Row { id: u32, kids: Near<[Row]> }

fn main() {
  let mut a = Region::new(Row::new(0, [Row::new(1, ())]));
  let h = a.edit();
  let old = h.kids().remove(0);
  std::thread::scope(|s| {
    s.spawn(move || drop(Region::new(old)));
    h.kids().push(Row::new(2, ()));
  });
}
