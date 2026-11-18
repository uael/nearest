//! A list is read as a list of elements of the same size: its length is a number of elements.
use nearest::{Near, Region};

fn main() {
  let region = Region::<Near<[u32]>>::new(Near::from_iter([1u32]));
  let _: &Near<[u16]> = nearest::puns::view_ref(&*region);
}
