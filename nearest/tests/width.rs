//! Pointers of every width, `Near<T, O>` with `O` = `i16`, `i32` (the default) or `i64`, and
//! offsets of the user's of each width in both byte orders: the same tests for each, on types
//! generic over the width; the limit a width sets; mixed widths; byte orders.

use std::{
  num::NonZero,
  panic::{AssertUnwindSafe, catch_unwind},
};

use nearest::{Edit, Error, Heap, Inline, Near, Offset, Pending, Region, flat};

#[flat]
#[derive(Debug, PartialEq)]
struct Tree<O: Offset> {
  /// Aligned as a `u64` (8 bytes on x86-64, 4 on `i686`), so that lists of trees are aligned so
  /// behind 2-byte headers too.
  id: u64,
  name: Near<str, O>,
  kids: Near<[Tree<O>], O>,
  next: Option<Near<Tree<O>, O>>,
}

#[flat]
#[derive(Debug, PartialEq)]
enum Expr<O: Offset> {
  Lit(i64),
  Neg(Near<Expr<O>, O>),
  Add(Near<Expr<O>, O>, Near<Expr<O>, O>),
}

fn leaf<O: Offset>(id: u64, name: &str) -> impl nearest::Emit<Tree<O>> {
  Tree::new(id, name, (), None)
}

fn tree<O: Offset>() -> Region<Tree<O>> {
  Region::new(Tree::new(
    1,
    "root",
    (leaf(2, "a"), Tree::new(3, "b", [leaf(4, "c")], None)),
    Near::new(leaf(5, "next")),
  ))
}

fn ids<O: Offset>(t: &Tree<O>) -> Vec<u64> {
  let mut out = vec![t.id];
  for k in &t.kids {
    out.extend(ids(k));
  }
  out.extend(t.next.as_deref().map(ids).unwrap_or_default());
  out
}

fn show<O: Offset>(e: &Expr<O>) -> String {
  match e {
    Expr::Lit(n) => n.to_string(),
    Expr::Neg(x) => format!("-{}", show(x)),
    Expr::Add(a, b) => format!("({} + {})", show(a), show(b)),
  }
}

/// A pointer, an optional pointer, a list and a string are each one `O`.
fn sizes<O: Offset>() {
  let w = size_of::<O>();
  assert_eq!(size_of::<Near<u8, O>>(), w);
  assert_eq!(size_of::<Option<Near<u8, O>>>(), w);
  assert_eq!(size_of::<Near<[u8], O>>(), w);
  assert_eq!(size_of::<Near<str, O>>(), w);
  // A `u64`, then three `O`s; the struct is aligned as the larger of the two.
  let align = align_of::<u64>().max(align_of::<O>());
  assert_eq!(size_of::<Tree<O>>(), (8 + 3 * w).next_multiple_of(align));
}

fn build_and_read<O: Offset>() {
  let r = tree::<O>();
  assert_eq!(ids(&r), [1, 2, 3, 4, 5]);
  assert_eq!((&*r.name, &*r.kids[1].kids[0].name), ("root", "c"));
  assert_eq!(r.next.as_ref().map(|n| &*n.name), Some("next"));
  assert!(r.kids.iter().all(|k| (&raw const *k).addr().is_multiple_of(align_of::<u64>())));
  assert_eq!(r.clone(), r);
  assert_eq!(Region::new(&*r), r);
  let e = Region::new(Expr::<O>::add(Expr::neg(Expr::Lit(1)), Expr::Lit(2)));
  assert_eq!(show(&e), "(-1 + 2)");
  assert_eq!(format!("{:?}", r.kids[0].name), "\"a\"");
}

/// Every handle, and moves inside the region and copies out of it.
fn edit<O: Offset>() {
  let mut r = tree::<O>();
  let t = r.edit();
  *t.id() = 10;
  t.name().set("renamed");
  let old = t.kids().at(1).name().replace("b2");
  t.kids().at(0).name().set(old); // a move
  t.kids().push(leaf(6, "d"));
  for i in 0..20 {
    t.kids().at(1).kids().push(leaf(100 + i, "x")); // grows, moves the block
  }
  t.kids().at(1).kids().retain(|k| k.id % 2 == 0);
  t.kids().at(1).kids().truncate(5);
  let c = t.kids().at(1).kids().remove(0);
  t.kids().insert(0, c);
  t.kids().swap(0, 3);
  let next = t.next().take().unwrap();
  t.kids().at(0).next().set(Near::new(next));
  t.kids().at(0).update(|old| Tree::new(7, "wrapped", [old], None));
  t.next().insert(leaf(8, "new")).name().update(|s| s);
  assert_eq!(t.kids().try_at(9).map(|k| k.id), None);
  assert_eq!(ids(&r), [10, 7, 6, 5, 2, 3, 100, 102, 104, 106, 4, 8]);
  let names = r.kids.iter().map(|k| &*k.name).collect::<Vec<_>>();
  assert_eq!((&*r.name, names), ("renamed", vec!["wrapped", "b", "b2", "c"]));
  let compact = Region::new(&*r);
  assert!(compact.byte_len() < r.byte_len());
  assert_eq!(compact, r);
  // Into another region, of another storage: a copy.
  let mut other = Region::new_in(Inline::<4096>::default(), leaf::<O>(0, ""));
  let e = r.edit();
  let moved = e.kids().remove(0);
  other.edit().kids().push(moved);
  assert_eq!(ids(&other), [0, 7, 6, 5]);
  assert_eq!(ids(&r), [10, 2, 3, 100, 102, 104, 106, 4, 8]);
  let mut e = Region::new(Expr::<O>::Lit(1));
  for i in 2..=5 {
    e.edit().update(|sum| Expr::add(sum, Expr::neg(Expr::Lit(i))));
  }
  assert_eq!(show(&e), "((((1 + -2) + -3) + -4) + -5)");
  // Several fields at once, pinned.
  let TreePin { mut id, kids, .. } = r.edit().pin().project();
  for kid in kids.iter_pin_mut() {
    *id += kid.id;
  }
  assert_eq!(r.id, 10 + 2 + 3 + 4);
}

/// A helper that takes the handle of a generic type, and returns one.
fn first<'a, 'e, O: Offset>(t: &'a mut Edit<'e, Tree<O>>) -> &'a mut Edit<'e, Tree<O>> {
  t.kids().at(0)
}

fn helpers<O: Offset>() {
  let mut r = tree::<O>();
  *first(r.edit()).id() = 20;
  assert_eq!(r.kids[0].id, 20);
}

/// The `O` at `at`, which is not negative: a distance forward or a length.
fn pos<O: Offset>(bytes: &[u8], at: usize) -> usize {
  usize::try_from(int::<O>(bytes, at)).unwrap()
}

/// The `O` at `at`.
fn int<O: Offset>(bytes: &[u8], at: usize) -> i64 {
  let b = &bytes[at..at + size_of::<O>()];
  // SAFETY: `size_of::<O>()` initialized bytes, which make an `O` whatever they are (`Offset`).
  let o = unsafe { b.as_ptr().cast::<O>().read_unaligned() };
  o.to_isize() as i64
}

/// Writes `v` as the `O` at `at`.
fn put<O: Offset>(bytes: &mut [u8], at: usize, v: i64) {
  let b = &mut bytes[at..at + size_of::<O>()];
  let o = O::from_isize(isize::try_from(v).unwrap());
  // SAFETY: `size_of::<O>()` bytes, written unaligned.
  unsafe { b.as_mut_ptr().cast::<O>().write_unaligned(o) };
}

/// The layout: an `O` pointer, list headers of two `O`s before elements aligned for their type,
/// string lengths of one `O`.
fn layout<O: Offset>() {
  let w = size_of::<O>();
  let list = Region::<Near<[u64], O>>::new(Near::from_iter([7u64]));
  let bytes = list.as_bytes().unwrap();
  // Elements at a position aligned to `max(w, align_of::<u64>())`, after the header.
  let elems = (3 * w).next_multiple_of(w.max(align_of::<u64>()));
  assert_eq!(bytes.len(), elems + 8);
  assert_eq!(pos::<O>(bytes, 0), elems - w); // to the length
  assert_eq!((pos::<O>(bytes, elems - 2 * w), pos::<O>(bytes, elems - w)), (1, 1));
  assert_eq!(bytes[elems..], 7u64.to_ne_bytes());
  let text = Region::<Near<str, O>>::new("hi");
  let bytes = text.as_bytes().unwrap();
  assert_eq!(bytes.len(), 2 * w + 2);
  assert_eq!((pos::<O>(bytes, 0), pos::<O>(bytes, w)), (w, 2));
  assert_eq!(&bytes[2 * w..], b"hi");
}

/// Bytes: saved, checked, loaded and viewed; a bad image is an error.
fn bytes<O: Offset>() {
  let mut r = tree::<O>();
  r.edit().kids().at(0).kids().push(leaf(9, "late"));
  let compact = Region::new(&*r);
  let image = compact.as_bytes().unwrap();
  assert_eq!(Region::<Tree<O>>::from_bytes(image), Ok(Region::new(&*r)));
  assert_eq!(nearest::view::<Tree<O>>(image), Ok(&*r));
  let mut scratch = vec![Pending::default(); image.len() / size_of::<O>().min(4)];
  assert_eq!(nearest::view_with::<Tree<O>>(image, &mut scratch), Ok(&*r));
  assert_eq!(Region::<Tree<O>>::from_bytes_with(image, &mut []).err(), Some(Error::ScratchFull));
  // Cut, padded, or a pointer one byte off.
  let cut = &image[..image.len() - 1];
  assert!(matches!(Region::<Tree<O>>::from_bytes(cut), Err(Error::Invalid(_))));
  let mut long = image.to_vec();
  long.push(0);
  assert_eq!(Region::<Tree<O>>::from_bytes(&long).err(), Some(Error::Invalid(image.len())));
  let name = std::mem::offset_of!(Tree<O>, name);
  for n in [1, -1] {
    let mut bad = image.to_vec();
    put::<O>(&mut bad, name, int::<O>(image, name) + n);
    assert_eq!(Region::<Tree<O>>::from_bytes(&bad).err(), Some(Error::Invalid(name)));
  }
  // A negative length is as bad as one too long.
  let text = Region::<Near<str, O>>::new("hi");
  let mut bad = text.as_bytes().unwrap().to_vec();
  let w = size_of::<O>();
  put::<O>(&mut bad, w, -1);
  assert_eq!(Region::<Near<str, O>>::from_bytes(&bad).err(), Some(Error::Invalid(w)));
}

macro_rules! widths {
  ($($test:ident),*) => {
    mod w16 { $(#[test] fn $test() { super::$test::<i16>() })* }
    mod w32 { $(#[test] fn $test() { super::$test::<i32>() })* }
    mod w64 { $(#[test] fn $test() { super::$test::<i64>() })* }
    mod le16 { $(#[test] fn $test() { super::$test::<super::Le16>() })* }
    mod le32 { $(#[test] fn $test() { super::$test::<super::Le32>() })* }
    mod le64 { $(#[test] fn $test() { super::$test::<super::Le64>() })* }
    mod be16 { $(#[test] fn $test() { super::$test::<super::Be16>() })* }
    mod be32 { $(#[test] fn $test() { super::$test::<super::Be32>() })* }
    mod be64 { $(#[test] fn $test() { super::$test::<super::Be64>() })* }
  };
}
widths!(sizes, build_and_read, edit, helpers, layout, bytes);

// ---------------------------------------------------------------------------------------------
// Offsets of the user's: the byte order of a region's own integers

/// An integer of a given byte order, whatever the target's, and the same without zero.
macro_rules! offsets {
  ($($name:ident $non_zero:ident: $int:ty, $to:ident, $from:ident;)*) => {$(
    #[flat]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[repr(transparent)]
    struct $name($int);

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct $non_zero(NonZero<$int>);

    impl From<$non_zero> for $name {
      fn from(n: $non_zero) -> Self {
        Self(n.0.get())
      }
    }

    // SAFETY: an integer whose bytes are those of the integer in the given byte order: any
    // bit pattern, a round trip within its range, zero is the zero bytes (swapping keeps it);
    // `$non_zero` is the same bytes without zero, a `#[repr(transparent)]` `NonZero`, so
    // `Option` of it is its size with `None` zero; `From` keeps the bytes.
    #[expect(clippy::cast_possible_truncation, reason = "values within a region fit both")]
    unsafe impl Offset for $name {
      type NonZero = $non_zero;
      fn from_isize(n: isize) -> Self {
        Self((n as $int).$to())
      }
      fn to_isize(self) -> isize {
        <$int>::$from(self.0) as isize
      }
    }
  )*};
}
offsets! {
  Le16 NonZeroLe16: i16, to_le, from_le;
  Le32 NonZeroLe32: i32, to_le, from_le;
  Le64 NonZeroLe64: i64, to_le, from_le;
  Be16 NonZeroBe16: i16, to_be, from_be;
  Be32 NonZeroBe32: i32, to_be, from_be;
  Be64 NonZeroBe64: i64, to_be, from_be;
}

/// The bytes of a region are the same on every target with offsets of a fixed byte order: a
/// pointer to a length, the length, the text.
#[test]
fn byte_orders() {
  let le = Region::<Near<str, Le32>>::new("hi");
  assert_eq!(le.as_bytes().unwrap(), [4, 0, 0, 0, 2, 0, 0, 0, b'h', b'i']);
  let be = Region::<Near<str, Be32>>::new("hi");
  assert_eq!(be.as_bytes().unwrap(), [0, 0, 0, 4, 0, 0, 0, 2, b'h', b'i']);
  let be = Region::<Near<[u8], Be16>>::new(Near::from_iter([7u8]));
  // A pointer to the length, the capacity and the length, the element.
  assert_eq!(be.as_bytes().unwrap(), [0, 4, 0, 1, 0, 1, 7]);
  // Loaded with the byte order it was written with, and only that one.
  let image = Region::<Near<str, Be32>>::new("hello").as_bytes().unwrap().to_vec();
  assert_eq!(&**Region::<Near<str, Be32>>::from_bytes(&image).unwrap(), "hello");
  assert!(Region::<Near<str, Le32>>::from_bytes(&image).is_err());
  // A region built with one byte order is read and edited the same as with any other.
  let (mut be, mut native) = (tree::<Be32>(), tree::<i32>());
  be.edit().kids().push(leaf(6, "pushed"));
  native.edit().kids().push(leaf(6, "pushed"));
  assert_eq!(ids(&be), ids(&native));
}

// ---------------------------------------------------------------------------------------------
// The limit a width sets

/// A list behind a 2-byte pointer: a region of at most 32 KiB, so the limit is cheap to reach.
type Small = Near<[u32], i16>;

/// What a 2-byte pointer reaches: 32 KiB, rounded down to 8 bytes.
const LIMIT: usize = i16::MAX as usize & !7;

#[test]
fn the_size_limit_panics_and_leaves_a_valid_region() {
  assert!(catch_unwind(|| Region::<Small>::new(Near::from_iter(0..10_000u32))).is_err());
  let mut r = Region::<Small>::new(Near::from_iter(0..10u32));
  let full = catch_unwind(AssertUnwindSafe(|| {
    for i in 0..u32::MAX {
      r.edit().push(i);
    }
  }));
  assert!(full.is_err());
  assert!(r.byte_len() > LIMIT / 2 && r.byte_len() <= LIMIT);
  assert_eq!(r[..10], (0..10).collect::<Vec<_>>());
  let before = r.to_vec();
  assert!(r[10..].iter().copied().eq(0..before.len() as u32 - 10));
  // What does not fit fails the same way, and the region is still intact.
  assert!(catch_unwind(AssertUnwindSafe(|| r.edit().set(Near::from_iter(0..8_192u32)))).is_err());
  assert_eq!(r[..], before);
  // The root, 2 bytes of padding, the header, then the elements, which are 4-aligned.
  let compact = Region::new(&*r);
  assert_eq!(compact.byte_len(), 2 + 2 + 4 + 4 * before.len());
  assert_eq!(compact[..], r[..]);
}

#[test]
fn capacity_past_the_size_limit_is_not_used() {
  let mut r = Region::<Small>::new_in(Heap::with_capacity(2 * LIMIT), Near::from_iter([0u32]));
  let full = catch_unwind(AssertUnwindSafe(|| {
    for i in 0..LIMIT as u32 {
      r.edit().push(i);
    }
  }));
  assert!(full.is_err() && r.byte_len() <= LIMIT);
}

#[test]
fn bytes_past_the_size_limit_are_invalid() {
  let r = Region::<Small>::new(Near::from_iter([1u32, 2]));
  let mut long = r.as_bytes().unwrap().to_vec();
  assert_eq!(Region::<Small>::from_bytes(&long).map(|r| r.to_vec()), Ok(vec![1, 2]));
  long.resize(LIMIT + 8, 0);
  assert_eq!(Region::<Small>::from_bytes(&long).err(), Some(Error::Invalid(0)));
}

// ---------------------------------------------------------------------------------------------
// Mixed widths

#[flat]
#[derive(Debug, PartialEq)]
struct Wide {
  items: Near<[u8], i64>,
}

/// Narrow pointers above wide ones: the root's pointers bound the region.
#[flat]
#[derive(Debug, PartialEq)]
struct Narrow {
  wide: Near<Wide, i16>,
  name: Near<str>,
}

#[test]
fn wider_pointers_below_narrower_ones() {
  let mut r = Region::new(Narrow::new(Wide::new([1, 2]), "n"));
  assert_eq!((&r.wide.items[..], &*r.name), (&[1, 2][..], "n"));
  r.edit().wide().items().push(3);
  assert_eq!(r.wide.items[..], [1, 2, 3]);
  let bytes = Region::new(&*r).as_bytes().unwrap().to_vec();
  assert_eq!(Region::<Narrow>::from_bytes(&bytes).unwrap(), r);
  // The region is bounded by the `i16` at its root.
  let bytes = vec![0u8; LIMIT];
  let big = catch_unwind(AssertUnwindSafe(|| r.edit().wide().items().set(&bytes[..])));
  assert!(big.is_err());
  assert_eq!(r.wide.items[..], [1, 2, 3]);
}

// ---------------------------------------------------------------------------------------------
// Past 2 GiB

#[flat]
struct Big {
  data: Near<[u8], i64>,
  tail: Near<str, i64>,
}

/// A region larger than 2 GiB, with a pointer that reaches past 2 GiB: built, edited, checked.
#[cfg(target_pointer_width = "64")]
#[test]
#[ignore = "builds a region of more than 2 GiB; CI runs it in release"]
fn a_region_past_2_gib() {
  let n = (1usize << 31) + 1000;
  let mut r = Region::new_in(
    Heap::with_capacity(n + 4096),
    Big::new(Near::from_iter(std::iter::repeat_n(7u8, n)), "end"),
  );
  assert!(r.byte_len() > n);
  assert_eq!((r.data.len(), r.data[n - 1], &*r.tail), (n, 7, "end"));
  let e = r.edit();
  e.data()[n - 1] = 8;
  e.tail().set("moved");
  assert_eq!((r.data[n - 1], &*r.tail), (8, "moved"));
  let copy = Region::new_in(Heap::with_capacity(r.byte_len()), &*r);
  let bytes = copy.as_bytes().unwrap();
  let big = nearest::view::<Big>(bytes).unwrap();
  assert_eq!((big.data.len(), big.data[n - 1], &*big.tail), (n, 8, "moved"));
}

// ---------------------------------------------------------------------------------------------
// Offsets, lengths and capacities that do not fit `isize`

/// `bytes`, with the `i64` at `at` set to `v`.
fn with_i64(bytes: &[u8], at: usize, v: i64) -> Vec<u8> {
  let mut out = bytes.to_vec();
  out[at..at + 8].copy_from_slice(&v.to_ne_bytes());
  out
}

/// An `i64` past what `isize` holds on a 32-bit target is not valid in a region: on such a
/// target `to_isize` would truncate it to the distance or the length it was made from, and on a
/// 64-bit one it is out of bounds. The loaders refuse it on every target.
#[test]
fn integers_wider_than_isize_are_invalid() {
  const HIGH: i64 = 1 << 32;
  let r = Region::<Near<u8, i64>>::new(Near::new(7));
  let bytes = r.as_bytes().unwrap();
  assert_eq!(bytes, [&8i64.to_ne_bytes()[..], &[7]].concat());
  let far = with_i64(bytes, 0, HIGH + 8);
  assert_eq!(Region::<Near<u8, i64>>::from_bytes(&far).err(), Some(Error::Invalid(0)));
  assert_eq!(view_in_words::<Near<u8, i64>>(&far), Err(Error::Invalid(0)));
  assert_eq!(view_in_words::<Near<u8, i64>>(bytes), Ok(()));
  // The length and the capacity of a list, and the length of a text.
  let list = Region::<Near<[u8], i64>>::new(Near::from_iter([1u8, 2]));
  let bytes = list.as_bytes().unwrap();
  let len = usize::try_from(i64::from_ne_bytes(bytes[..8].try_into().unwrap())).unwrap();
  assert_eq!(Region::<Near<[u8], i64>>::from_bytes(bytes).map(|r| r.len()), Ok(2));
  for at in [len, len - 8] {
    let bad = with_i64(bytes, at, HIGH + 2);
    assert!(matches!(Region::<Near<[u8], i64>>::from_bytes(&bad), Err(Error::Invalid(_))), "{at}");
  }
  let both = with_i64(&with_i64(bytes, len, HIGH + 2), len - 8, HIGH + 2);
  assert!(matches!(Region::<Near<[u8], i64>>::from_bytes(&both), Err(Error::Invalid(_))));
  let text = Region::<Near<str, i64>>::new("hi");
  let bytes = text.as_bytes().unwrap();
  let bad = with_i64(bytes, 8, HIGH + 2);
  assert_eq!(Region::<Near<str, i64>>::from_bytes(&bad).err(), Some(Error::Invalid(8)));
  // Nor any `i64` that is not what `from_isize` makes: the most negative one.
  let low = with_i64(r.as_bytes().unwrap(), 0, i64::MIN);
  assert_eq!(Region::<Near<u8, i64>>::from_bytes(&low).err(), Some(Error::Invalid(0)));
}

/// `nearest::view` of `bytes`, copied to words: an address aligned as a region is.
fn view_in_words<T: nearest::Flat>(bytes: &[u8]) -> Result<(), Error> {
  let mut words = vec![0u64; bytes.len().div_ceil(8)];
  // SAFETY: the words hold at least `bytes.len()` bytes, all initialized.
  let at = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
  at.copy_from_slice(bytes);
  nearest::view::<T>(at).map(|_| ())
}

/// An `i32` whose `to_isize` gives 4 for every integer past its `LIMIT`: within the contract of
/// `Offset`, which asks for a round trip within `LIMIT` only. Only the `Lenient`s that
/// `from_isize` makes are valid in a region, so the loaders refuse the others, for an `Offset` of
/// the user's as for `i64`.
#[flat]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Lenient(i32);

#[derive(Clone, Copy)]
#[repr(transparent)]
struct NonZeroLenient(NonZero<i32>);

impl From<NonZeroLenient> for Lenient {
  fn from(n: NonZeroLenient) -> Self {
    Self(n.0.get())
  }
}

// SAFETY: an `i32` of the target, every bit pattern; `as` round-trips within `LIMIT`; only 0 is
// 0 (past `LIMIT`, 4); `NonZeroLenient` is the same bytes without zero.
unsafe impl Offset for Lenient {
  type NonZero = NonZeroLenient;
  fn from_isize(n: isize) -> Self {
    Self(i32::try_from(n).unwrap())
  }
  fn to_isize(self) -> isize {
    if u64::from(self.0.unsigned_abs()) > Self::LIMIT { 4 } else { self.0 as isize }
  }
}

#[test]
fn offsets_that_from_isize_does_not_make_are_invalid() {
  let r = Region::<Near<u8, Lenient>>::new(Near::new(7));
  let bytes = r.as_bytes().unwrap();
  assert_eq!(bytes, [&4i32.to_ne_bytes()[..], &[7]].concat());
  assert_eq!(Region::<Near<u8, Lenient>>::from_bytes(bytes).map(|r| **r), Ok(7));
  let mut far = bytes.to_vec();
  far[..4].copy_from_slice(&i32::MAX.to_ne_bytes()); // read as 4
  assert_eq!(Region::<Near<u8, Lenient>>::from_bytes(&far).err(), Some(Error::Invalid(0)));
}
