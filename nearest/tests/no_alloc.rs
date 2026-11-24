//! Inline storage needs no heap: these tests also run with the library built without `alloc`,
//! `cargo test -p nearest --no-default-features --test no_alloc`.

use core::pin::Pin;

use nearest::{Error, Inline, Near, Region, flat, puns};

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
  // An image larger than the storage is an error, not a panic.
  let small = Region::<Msg, Inline<32>>::from_bytes(bytes);
  assert_eq!(small.err(), Some(Error::StorageFull));
  assert!(std::panic::catch_unwind(|| Region::new_in(Inline::<32>::default(), &*moved)).is_err());
}

/// The storage of the deep sums, and so their largest image.
type Big = Region<Expr, Inline<4096>>;

/// A left-deep sum of `depth` terms, compact: its first term is `depth` pointers deep.
fn sum(depth: usize) -> Big {
  // Each term leaves the sum it wraps behind: four times the image is room enough.
  let mut r = Region::new_in(Inline::<{ 4 * 4096 }>::default(), Expr::Lit(0));
  for i in 1..=depth {
    r.edit().update(|old| Expr::add(old, Expr::Lit(i64::try_from(i).unwrap())));
  }
  Region::new_in(Inline::default(), &*r)
}

/// Checking takes no memory but the stack: `view` and `from_bytes` check bytes where they are,
/// without `alloc` too, as deep as `puns` follows pointers (`AIM_DEPTH`); one level deeper, the
/// check stops, which says nothing of the bytes. Under Miri, a shallow sum: the same paths.
#[test]
fn checking_takes_no_memory() {
  let depth = if cfg!(miri) { 4 } else { puns::AIM_DEPTH };
  let deep = sum(depth);
  let bytes = deep.as_bytes().unwrap();
  assert_eq!(nearest::view::<Expr>(bytes), Ok(&*deep));
  assert_eq!(*Big::from_bytes(bytes).unwrap(), *deep);
  if !cfg!(miri) {
    let deeper = sum(depth + 1);
    assert_eq!(nearest::view::<Expr>(deeper.as_bytes().unwrap()), Err(Error::Limit));
    assert_eq!(Big::from_bytes(deeper.as_bytes().unwrap()).err(), Some(Error::Limit));
  }
  // Invalid bytes are invalid.
  assert_eq!(nearest::view::<Expr>(&bytes[..bytes.len() - 16]), Err(Error::Invalid));
  assert_eq!(Big::from_bytes(&bytes[1..]).err(), Some(Error::Invalid));
}
