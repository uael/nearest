//! An edit stays on one thread: a handle sent away could grow the buffer while a moved value
//! is copied out of it here.
use nearest::{Near, Region, flat};

#[flat]
struct Row { id: u32, kids: Near<[Row]> }

fn main() {
  let mut a = Region::new(Row::new(0, [Row::new(1, ())]));
  let mut b = Region::new(Row::new(0, ()));
  let h = a.edit();
  let old = h.kids().remove(0);
  std::thread::scope(|s| {
    s.spawn(|| {
      h.kids().push(Row::new(2, ()));
    });
    b.edit().kids().push(old);
  });
}
