//! A target is not read as a more aligned one: a region places it for its own alignment.
use nearest::{Near, Region};

fn main() {
  let region = Region::<Near<[u8; 4]>>::new(Near::new([1, 2, 3, 4]));
  let _: &Near<u32> = nearest::puns::view_ref(&*region);
}
