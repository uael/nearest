//! The integration tests, with every region in the global allocator; `user_alloc.rs` runs them
//! again with an allocator written there.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

/// The allocator of every region: the global one.
type Store = std::alloc::Global;

include!("common/integration.rs");

// ---------------------------------------------------------------------------------------------
// The global allocator

/// The block does not move until the region holds more than was reserved.
#[test]
fn reserve_makes_room_up_front() {
  let mut r = nearest::Region::new(Lst::<u32>::new([0]));
  r.reserve(4096);
  let root = std::ptr::from_ref(&*r);
  for i in 1..200 {
    r.edit().items().push(i);
  }
  assert!(r.byte_len() <= 4096 && std::ptr::eq(std::ptr::from_ref(&*r), root));
  assert!(r.items.iter().copied().eq(0..200));
}
