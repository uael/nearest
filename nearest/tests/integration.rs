//! The integration tests, with every region on the heap; `inline.rs` runs them again with
//! inline storage.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

/// The storage of every region: the heap.
type Store = nearest::Heap;

include!("common/integration.rs");

// ---------------------------------------------------------------------------------------------
// The heap

/// The buffer does not move until the region holds more than its capacity.
#[test]
fn with_capacity_reserves_up_front() {
  let mut r = Region::with_capacity(4096, Lst::<u32>::new([0]));
  let root = std::ptr::from_ref(&*r);
  for i in 1..200 {
    r.edit().items().push(i);
  }
  assert!(r.byte_len() <= 4096 && std::ptr::eq(std::ptr::from_ref(&*r), root));
  assert!(r.items.iter().copied().eq(0..200));
  assert_eq!(Region::with_capacity(0, &*r), r);
}
