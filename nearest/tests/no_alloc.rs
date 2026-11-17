//! Inline storage needs no heap: these tests also run with the library built without `alloc`,
//! `cargo test -p nearest --no-default-features --test no_alloc`.

use core::pin::Pin;

use nearest::{Error, Inline, Near, Pending, Region, flat};

#[flat]
#[derive(Debug, PartialEq)]
struct Msg {
  id: u32,
  body: Near<str>,
  tags: Near<[u16]>,
}

#[flat]
#[derive(Debug, PartialEq)]
enum Expr {
  Lit(i64),
  Add(Near<Expr>, Near<Expr>),
}

#[test]
fn build_edit_move_and_load() {
  let mut r = Region::new_in(Inline::<128>::default(), Msg::new(1, "hi", [1, 2]));
  let m = r.edit();
  m.body().set("hello");
  m.tags().push(3);
  let mut pinned: Pin<&mut Msg> = m.pin();
  let MsgPin { id, .. } = pinned.as_mut().project();
  *id.get_mut() = 2;
  let moved = r;
  assert_eq!((moved.id, &*moved.body, &moved.tags[..]), (2, "hello", &[1, 2, 3][..]));
  assert!(moved.as_bytes().is_none());
  let compact = Region::new_in(Inline::<128>::default(), &*moved);
  let bytes = compact.as_bytes().unwrap();
  let mut scratch = [Pending::default(); 128 / 4];
  assert_eq!(nearest::view_with::<Msg>(bytes, &mut scratch), Ok(&*moved));
  let loaded = Region::<Msg, Inline<64>>::from_bytes_with(bytes, &mut scratch);
  assert_eq!(*loaded.unwrap(), *moved);
  // A valid image larger than the storage is an error, not a panic.
  let small = Region::<Msg, Inline<32>>::from_bytes_with(bytes, &mut scratch);
  assert_eq!(small.err(), Some(Error::StorageFull));
  #[cfg(feature = "alloc")]
  assert_eq!(Region::<Msg, Inline<32>>::from_bytes(bytes).err(), Some(Error::StorageFull));
  assert!(std::panic::catch_unwind(|| Region::new_in(Inline::<32>::default(), &*moved)).is_err());
}

/// The storage of the deep sums, and so their largest image.
const SIZE: usize = 1 << 14;
type Big = Region<Expr, Inline<SIZE>>;

/// A left-deep sum of `depth` terms, compact: checking it keeps about `depth` pointers waiting.
fn sum(depth: i64) -> Big {
  let mut r = Region::new_in(Inline::<{ 1 << 16 }>::default(), Expr::Lit(0));
  for i in 1..=depth {
    r.edit().update(|old| Expr::add(old, Expr::Lit(i)));
  }
  Region::new_in(Inline::default(), &*r)
}

/// Checking keeps the pointers still to visit in scratch memory from the caller: a quarter of
/// the size of the image, in entries, is always enough, and too little is not an invalid image.
#[test]
fn checking_takes_scratch_memory_from_the_caller() {
  #[expect(clippy::large_stack_arrays, reason = "memory without a heap, as a caller would have")]
  let mut scratch = [Pending::default(); SIZE / 4];
  let shallow = sum(50);
  let bytes = shallow.as_bytes().unwrap();
  assert_eq!(nearest::view_with::<Expr>(bytes, &mut scratch[..64]), Ok(&*shallow));
  let deep = sum(500);
  let bytes = deep.as_bytes().unwrap();
  assert!(bytes.len() > 16_000);
  assert_eq!(nearest::view_with::<Expr>(bytes, &mut scratch), Ok(&*deep));
  assert_eq!(*Big::from_bytes_with(bytes, &mut scratch).unwrap(), *deep);
  // Too little: the check could not finish.
  assert_eq!(nearest::view_with::<Expr>(bytes, &mut scratch[..400]), Err(Error::ScratchFull));
  assert_eq!(Big::from_bytes_with(bytes, &mut scratch[..400]).err(), Some(Error::ScratchFull));
  // Invalid bytes are invalid, given enough.
  let cut = &bytes[..bytes.len() - 16];
  assert!(matches!(nearest::view_with::<Expr>(cut, &mut scratch), Err(Error::Invalid(_))));
  assert!(matches!(Big::from_bytes_with(&bytes[1..], &mut scratch), Err(Error::Invalid(_))));
  // With `alloc`, the checker's own memory grows as needed.
  #[cfg(feature = "alloc")]
  {
    assert_eq!(nearest::view::<Expr>(bytes), Ok(&*deep));
    assert_eq!(*Big::from_bytes(bytes).unwrap(), *deep);
    assert!(matches!(nearest::view::<Expr>(cut), Err(Error::Invalid(_))));
  }
}
