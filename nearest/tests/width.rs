//! Pointers of every width, `Near<T, O>` with `O` = `i16`, `i32` (the default) or `i64`, and
//! offsets of the user's of each width in both byte orders: the same tests for each, on types
//! generic over the width; the limit a width sets; mixed widths; byte orders.

use std::{
  num::NonZero,
  panic::{AssertUnwindSafe, catch_unwind},
};

use nearest::{
  Edit, Error, Heap, Inline, Near, Offset, Pending, Region, flat, puns::layout::Endian,
};

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
  out.extend(t.next.as_deref().map_or_default(ids));
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

/// The integer that `bytes` are in two's complement, in the byte order `endian`.
fn decode(bytes: &[u8], endian: Endian) -> i64 {
  let last = if endian == Endian::Little { bytes.len() - 1 } else { 0 };
  let mut wide = [if bytes[last] & 0x80 == 0 { 0 } else { 0xff }; 8];
  for (i, byte) in bytes.iter().enumerate() {
    wide[if endian == Endian::Little { i } else { bytes.len() - 1 - i }] = *byte;
  }
  i64::from_le_bytes(wide)
}

/// The bytes of an `O`.
const fn bytes_of<O: Offset>(o: &O) -> &[u8] {
  // SAFETY: an `O` is `size_of::<O>()` initialized bytes, without padding (`Offset`).
  unsafe { std::slice::from_raw_parts(std::ptr::from_ref(o).cast::<u8>(), size_of::<O>()) }
}

/// The contract of `Offset`, on values: the bytes are the integer in two's complement in the byte
/// order `ENDIAN` says, which `to_isize` reads; a round trip within `LIMIT`; zero is the zero
/// bytes and nothing else is; the `NonZero` is the same bytes, and `None` of it is zero. This is
/// what `puns` relies on when it reads a pointer of one offset as a pointer of another.
fn contract<O: Offset>() {
  let w = size_of::<O>();
  let limit = i64::try_from(O::LIMIT).unwrap();
  let (least, most) = (i64::MIN >> (64 - 8 * w), i64::MAX >> (64 - 8 * w));
  let mut samples = vec![0, 1, -1, 2, 127, 128, 255, 256, -255, -256, 0x1234, -0x1234, limit];
  samples.extend([-limit, limit - 1, 8 - limit]);
  // Under Miri, one bit in 8: the same code.
  samples.extend((0..63).step_by(if cfg!(miri) { 8 } else { 1 }).map(|bit| (1i64 << bit) & limit));
  for n in samples {
    let Ok(wanted) = isize::try_from(n) else { continue };
    let o = O::from_isize(wanted);
    assert_eq!(o.to_isize(), wanted, "{n} does not round trip");
    let bytes = bytes_of(&o);
    assert_eq!(decode(bytes, O::ENDIAN), n, "{n} is {bytes:?}");
    assert_eq!(bytes.iter().all(|byte| *byte == 0), n == 0);
    assert_eq!(non_zero::<O>(bytes).is_some(), n != 0);
    if let Some(stored) = non_zero::<O>(bytes) {
      assert_eq!(bytes_of(&stored.into()), bytes);
    }
  }
  // Every integer of the width, as `to_isize` reads it: also those a region never holds.
  for n in [least, most, least + 1, most - 1] {
    let mut bytes = n.to_le_bytes()[..w].to_vec();
    if O::ENDIAN == Endian::Big {
      bytes.reverse();
    }
    // SAFETY: `size_of::<O>()` bytes, any of which are an `O`.
    let o = unsafe { bytes.as_ptr().cast::<O>().read_unaligned() };
    assert_eq!(o.to_isize(), n as isize, "{bytes:?}");
  }
}

/// The `Option<O::NonZero>` whose bytes are `bytes`.
fn non_zero<O: Offset>(bytes: &[u8]) -> Option<O::NonZero> {
  assert_eq!((bytes.len(), size_of::<Option<O::NonZero>>()), (size_of::<O>(), size_of::<O>()));
  // SAFETY: an `Option` of the `NonZero` of an offset has the size of the offset, and every bit
  // pattern of that size is one (`None` is zero): the contract of `Offset`.
  unsafe { bytes.as_ptr().cast::<Option<O::NonZero>>().read_unaligned() }
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

/// Each test with each offset. Under Miri, the tests that build and check regions run with seven
/// of the fifteen offsets (each width of the target, each width in big-endian, a user's and one
/// of `puns::num` of either byte order), the others with `sizes` and `contract` only: they run
/// the same code, and all of them would take the job past a minute.
macro_rules! widths {
  ($($test:ident),*) => {
    widths!(@ true ($($test),*) w16: i16, w32: i32, w64: i64, be32: super::Be32);
    widths!(@ false ($($test),*) le16: super::Le16, le32: super::Le32, le64: super::Le64,
      be16: super::Be16, be64: super::Be64);
    // The offsets of a fixed byte order of `puns::num`, which `nearest` implements.
    widths!(@ true ($($test),*) i32le: nearest::puns::num::i32le,
      i16be: nearest::puns::num::i16be, i64be: nearest::puns::num::i64be);
    widths!(@ false ($($test),*) i16le: nearest::puns::num::i16le,
      i64le: nearest::puns::num::i64le, i32be: nearest::puns::num::i32be);
  };
  (@ $miri:literal $tests:tt $($name:ident: $offset:ty),*) => {
    $(widths!(@@ $miri $tests $name: $offset);)*
  };
  (@@ $miri:literal ($($test:ident),*) $name:ident: $offset:ty) => {
    mod $name {
      $(#[test] fn $test() {
        if $miri || !cfg!(miri) || matches!(stringify!($test), "sizes" | "contract") {
          super::$test::<$offset>();
        }
      })*
    }
  };
}
widths!(sizes, contract, build_and_read, edit, helpers, layout, bytes);

// ---------------------------------------------------------------------------------------------
// Offsets of the user's: the byte order of a region's own integers

/// An integer of a given byte order, whatever the target's, and the same without zero.
macro_rules! offsets {
  ($($name:ident $non_zero:ident: $int:ty, $to:ident, $from:ident, $endian:expr;)*) => {$(
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
      const ENDIAN: Endian = $endian;
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
  Le16 NonZeroLe16: i16, to_le, from_le, Endian::Little;
  Le32 NonZeroLe32: i32, to_le, from_le, Endian::Little;
  Le64 NonZeroLe64: i64, to_le, from_le, Endian::Little;
  Be16 NonZeroBe16: i16, to_be, from_be, Endian::Big;
  Be32 NonZeroBe32: i32, to_be, from_be, Endian::Big;
  Be64 NonZeroBe64: i64, to_be, from_be, Endian::Big;
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

/// An `i32` of the target whose `LIMIT` is 1 MiB: an integer past it is no distance or length of
/// a region, although `to_isize` reads it as it is. Only the `Lenient`s that `from_isize` makes
/// within `LIMIT` are valid in a region, so the loaders refuse the others, for an `Offset` of the
/// user's as for `i64`.
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

// SAFETY: an `i32` of the target, in its byte order, every bit pattern, read with `as`; `as`
// round-trips within `LIMIT`; `NonZeroLenient` is the same bytes without zero.
unsafe impl Offset for Lenient {
  type NonZero = NonZeroLenient;
  const LIMIT: u64 = 1 << 20;
  const ENDIAN: Endian = Endian::NATIVE;
  fn from_isize(n: isize) -> Self {
    Self(i32::try_from(n).unwrap())
  }
  fn to_isize(self) -> isize {
    self.0 as isize
  }
}

#[test]
fn offsets_that_from_isize_does_not_make_are_invalid() {
  let r = Region::<Near<u8, Lenient>>::new(Near::new(7));
  let bytes = r.as_bytes().unwrap();
  assert_eq!(bytes, [&4i32.to_ne_bytes()[..], &[7]].concat());
  assert_eq!(Region::<Near<u8, Lenient>>::from_bytes(bytes).map(|r| **r), Ok(7));
  let mut far = bytes.to_vec();
  far[..4].copy_from_slice(&((1 << 20) + 4i32).to_ne_bytes()); // past `LIMIT`
  assert_eq!(Region::<Near<u8, Lenient>>::from_bytes(&far).err(), Some(Error::Invalid(0)));
}

// ---------------------------------------------------------------------------------------------
// Offsets of the same width and byte order read each other's regions

/// A region of `Tree<A>` read as `Tree<B>` with `puns::view_ref`: the same targets, since both
/// offsets are the same two's complement integer of the same bytes (the contract of `Offset`).
fn read_as<A: Offset, B: Offset>() {
  let r = tree::<A>();
  let view: &Tree<B> = nearest::puns::view_ref(&*r);
  assert_eq!(ids(view), ids(&r));
  assert_eq!((&*view.name, &*view.kids[1].kids[0].name), ("root", "c"));
  // A deep copy of the view is the same image.
  assert_eq!(Region::new(view).as_bytes(), Region::new(&*r).as_bytes());
}

#[test]
fn offsets_of_the_same_width_and_byte_order_read_each_other() {
  use nearest::puns::num::{i16be, i16le, i32be, i32le, i64be, i64le};
  read_as::<Le16, i16le>();
  read_as::<i16le, Le16>();
  read_as::<Le32, i32le>();
  read_as::<i32le, Le32>();
  read_as::<Le64, i64le>();
  read_as::<Be16, i16be>();
  read_as::<Be32, i32be>();
  read_as::<i32be, Be32>();
  read_as::<Be64, i64be>();
  read_as::<i64be, Be64>();
  // The integers of the target are of its byte order.
  #[cfg(target_endian = "little")]
  {
    read_as::<i16, i16le>();
    read_as::<i32, Le32>();
    read_as::<i64le, i64>();
  }
  #[cfg(target_endian = "big")]
  {
    read_as::<i16, i16be>();
    read_as::<i32, Be32>();
    read_as::<i64be, i64>();
  }
}
