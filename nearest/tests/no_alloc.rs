//! Inline storage needs no heap: these tests also run with the library built without `alloc`,
//! `cargo test -p nearest --no-default-features --test no_alloc`.

use core::pin::Pin;

use nearest::{Error, Inline, Near, Region, flat};

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
  assert_eq!(nearest::view::<Msg>(bytes), Ok(&*moved));
  let loaded = Region::<Msg, Inline<64>>::from_bytes(bytes);
  assert_eq!(*loaded.unwrap(), *moved);
  // A valid image larger than the storage is an error, not a panic.
  assert_eq!(Region::<Msg, Inline<32>>::from_bytes(bytes).err(), Some(Error::StorageFull));
  assert!(std::panic::catch_unwind(|| Region::new_in(Inline::<32>::default(), &*moved)).is_err());
}

/// Under Miri everything below is eight times smaller: the same paths, in seconds rather than
/// minutes (the time that Miri reports is not the one that passes).
const SCALE: usize = if cfg!(miri) { 8 } else { 1 };

/// The storage of the deep sums, and so their largest image.
const SIZE: usize = (1 << 14) / SCALE;
type Big = Region<Expr, Inline<SIZE>>;

/// The terms of a deep sum: its image nearly fills `SIZE`.
const DEPTH: usize = 500 / SCALE;

/// A left-deep sum of `depth` terms, compact: checking it keeps about `depth` pointers waiting.
fn sum(depth: usize) -> Big {
  // Each term leaves the sum it wraps behind: four times the image is room enough.
  let mut r = Region::new_in(Inline::<{ 4 * SIZE }>::default(), Expr::Lit(0));
  for i in 1..=depth {
    r.edit().update(|old| Expr::add(old, Expr::Lit(i64::try_from(i).unwrap())));
  }
  Region::new_in(Inline::default(), &*r)
}

/// Checking takes no memory from the caller, and does not recurse with the depth of the data: a
/// deep sum, which keeps about `DEPTH` pointers waiting, is checked within the frames of the check.
#[test]
fn checking_takes_no_memory_from_the_caller() {
  let deep = sum(DEPTH);
  let bytes = deep.as_bytes().unwrap();
  assert!(bytes.len() > SIZE / 2);
  assert_eq!(nearest::view::<Expr>(bytes), Ok(&*deep));
  assert_eq!(*Big::from_bytes(bytes).unwrap(), *deep);
  let cut = &bytes[..bytes.len() - 16];
  assert!(matches!(nearest::view::<Expr>(cut), Err(Error::Invalid(_))));
  assert!(matches!(Big::from_bytes(cut), Err(Error::Invalid(_))));
}
