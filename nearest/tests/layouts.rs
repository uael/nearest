//! What `nearest` takes from the layouts of `puns`: numbers of a fixed byte order in a region, a
//! value that holds pointers read as plain data, and from bytes once `puns` checked its targets,
//! optional pointers, generic types, the alignment of the elements of a list, and the rules of
//! `#[pun(..)]`, checked at load.

use nearest::{
  Flat, Near, Region, flat,
  puns::{
    self,
    num::{i16be, i32le, u16le, u32le},
  },
};

// -------------------------------------------------------------------------------------------------
// Generic types

/// Two generic types that hold each other: the bounds that the derive would find go round in a
/// circle (`Ping<O>` is `Pun` if the pointer to `Pong<O>` is, which is if `Pong<O>` is...), so
/// they are told: none but the ones that the types declare.
#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(bound())]
struct Ping<O: nearest::Offset> {
  id: u8,
  pongs: Near<[Pong<O>], O>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(bound())]
enum Pong<O: nearest::Offset> {
  End(bool),
  Ping(Near<Ping<O>, O>),
}

/// A generic type whose parameter is what it holds: the derive finds its bounds (and `#[flat]`
/// its own: the parameter is `Flat`). `Debug` and `PartialEq` read the targets, whose `Deref`
/// requires the parameter to be `Flat`, so it is written, and `#[flat]` does not add it again.
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Boxed<T: nearest::Flat> {
  id: u8,
  value: Near<T, i32le>,
  more: Near<[T], i32le>,
}

#[test]
fn generic_types() {
  let r = Region::new(Ping::<i16be>::new(1, [Pong::ping(Ping::new(2, [Pong::End(true)]))]));
  let ping: &Ping<i16be> = nearest::view(r.as_bytes().unwrap()).unwrap();
  assert_eq!((ping.id, ping.pongs.len()), (1, 1));
  let flags = Region::new(Boxed::<bool>::new(1, true, [false, true]));
  let bytes: &Boxed<u8> = nearest::view(flags.as_bytes().unwrap()).unwrap();
  assert_eq!((bytes.id, *bytes.value, &bytes.more[..]), (1, 1, &[0, 1][..]));
  // A `u8` of 2 is no `bool`: refused.
  let two = Region::new(Boxed::<u8>::new(1, 2, [0]));
  assert_eq!(
    nearest::view::<Boxed<bool>>(two.as_bytes().unwrap()).err(),
    Some(nearest::Error::Invalid)
  );
}

// -------------------------------------------------------------------------------------------------
// Numbers of a fixed byte order, and pointers read as bytes

/// Numbers of a given byte order are stored as their bytes are, and edited in place.
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Table {
  version: u8,
  count: u32le,
  rows: Near<[u16le], i32le>,
}

#[test]
fn numbers_of_a_byte_order_in_a_region() {
  let mut r = Region::new(Table::new(1, u32le::new(2), [u16le::new(0x0102), u16le::new(3)]));
  // The version, the count, the distance to the length of the rows; the capacity and the length,
  // so placed that the rows are aligned to the width of the offset; the rows.
  let image = [1, 2, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 3, 0];
  assert_eq!(r.as_bytes().unwrap(), image);
  assert_eq!(nearest::view::<Table>(r.as_bytes().unwrap()), Ok(&*r));
  // A byte array: a pointer may be anywhere.
  assert_eq!((align_of::<i32le>(), align_of::<Near<u64, i32le>>()), (1, 1));
  assert_eq!((size_of::<Table>(), align_of::<Table>()), (9, 1));
  // Plain data: a handle derefs to the number itself.
  let t = r.edit();
  *t.count() = u32le::new(3);
  t.rows().push(u16le::new(0xffff));
  let sum = r.rows.iter().map(|row| u32::from(row.get())).sum::<u32>();
  assert_eq!((r.count.get(), sum), (3, 0x0102 + 3 + 0xffff));
  const { assert!(<u32le as Flat>::PLAIN && <i16be as Flat>::PLAIN) };
}

/// A value that holds pointers is read as plain data: an offset is initialized bytes.
#[test]
fn pointers_are_read_as_bytes() {
  let r = Region::new(Table::new(1, u32le::new(2), [u16le::new(3)]));
  assert_eq!(puns::as_bytes(&*r), [1, 2, 0, 0, 0, 11, 0, 0, 0]);
  assert_eq!(puns::head_ref::<Table, u8>(&r), &1);
  assert_eq!(puns::tail_ref::<Table, i32le>(&r).get(), 11);
  assert_eq!(puns::try_pun_ref::<Table, Flagged>(&r).map(|f| f.set), Ok(true));
}

/// Two pointers, which `puns` reads from bytes once it checked their targets.
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Two {
  a: Near<u32, i32le>,
  b: Near<u32, i32le>,
}

/// `puns` reads a value that holds pointers from bytes as it reads any value, once it checked
/// every target (`Deref` then reads them). It does not require the image to be compact: two
/// pointers may share a target, which a region refuses, since an edit of one target through a
/// `&mut` would be seen through the other pointer.
#[test]
fn puns_reads_pointers_from_bytes_once_it_checked_their_targets() {
  let r = Region::new(Two::new(7, 9));
  let image = r.as_bytes().unwrap();
  let (two, _) = puns::from_head::<Two>(image).unwrap();
  assert_eq!((*two.a, *two.b), (7, 9));
  // In words: an address aligned as a region is, for the targets.
  let in_words = |bytes: &[u8], f: &dyn Fn(&[u8])| {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: the words hold at least `bytes.len()` bytes, all initialized.
    let at =
      unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    at.copy_from_slice(bytes);
    f(at);
  };
  // `b` aims at the target of `a`: valid for `puns`, not a region.
  let at = core::mem::offset_of!(Two, b);
  let a_target = i32::from_le_bytes(image[..4].try_into().unwrap());
  let b = a_target - i32::try_from(at).unwrap();
  let shared = [&image[..at], &b.to_le_bytes(), &image[at + 4..]].concat();
  in_words(&shared, &|bytes| {
    let (two, _) = puns::from_head::<Two>(bytes).unwrap();
    assert_eq!((*two.a, *two.b), (7, 7));
    assert_eq!(nearest::view::<Two>(bytes).err(), Some(nearest::Error::Invalid));
  });
  // `b` aims past the bytes: invalid for both.
  let far = [&image[..at], &1000i32.to_le_bytes(), &image[at + 4..]].concat();
  in_words(&far, &|bytes| {
    assert_eq!(puns::from_head::<Two>(bytes).err(), Some(puns::Error::Validity));
    assert_eq!(nearest::view::<Two>(bytes).err(), Some(nearest::Error::Invalid));
  });
}

/// A flag, then any bytes.
#[derive(puns::Pun)]
#[pun(crate = nearest::puns)]
#[repr(C)]
struct Flagged {
  set: bool,
  rest: [u8; 8],
}

// -------------------------------------------------------------------------------------------------
// Optional pointers

/// A pointer that may be null.
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Maybe {
  level: u8,
  owner: Option<Near<u32, i32le>>,
  more: Near<[Maybe], i32le>,
}

/// One that may not.
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Named {
  level: u8,
  owner: Near<u32, i32le>,
  more: Near<[Maybe], i32le>,
}

#[test]
fn optional_pointers() {
  // `Option<Near>` has the layout of `Near`: zero is `None`.
  assert_eq!(size_of::<Maybe>(), size_of::<Named>());
  let owned = Region::new(Maybe::new(1, Near::new(7), [Maybe::new(9, None, ())]));
  assert_eq!(Region::<Maybe>::from_bytes(owned.as_bytes().unwrap()).as_deref(), Ok(&*owned));
  // An image whose pointer is not null is one whose pointer may not be.
  let named = nearest::view::<Named>(owned.as_bytes().unwrap()).unwrap();
  assert_eq!((named.level, *named.owner, named.more[0].level), (1, 7, 9));
  assert_eq!(named.more[0].owner, None);
  // One that is null is not.
  let anonymous = Region::new(Maybe::new(2, None, ()));
  assert_eq!(
    nearest::view::<Named>(anonymous.as_bytes().unwrap()).err(),
    Some(nearest::Error::Invalid)
  );
}

// -------------------------------------------------------------------------------------------------
// The alignment of the elements of a list

/// Elements of alignment 1, of the size of an aligned integer.
#[flat]
struct Bytes<O: nearest::Offset> {
  quads: Near<[[u8; 4]], O>,
  numbers: Near<[i32le], O>,
  octets: Near<[[u8; 8]], O>,
  pairs: Near<[[u8; 2]], O>,
}

/// Asserts that the elements of each list are at a position, from the root at `base`, that is a
/// multiple of the larger of their alignment and the width of the offset: the layout of a list
/// block, which every way of making or moving a list keeps, and which a check at load requires.
fn aligned<O: nearest::Offset>(bytes: &Bytes<O>, base: usize) {
  fn at<T, O>(list: &[T], base: usize) {
    let align = align_of::<T>().max(size_of::<O>());
    assert_eq!((list.as_ptr().addr() - base) % align, 0, "{}", core::any::type_name::<T>());
  }
  at::<_, O>(&bytes.quads, base);
  at::<_, O>(&bytes.numbers, base);
  at::<_, O>(&bytes.octets, base);
  at::<_, O>(&bytes.pairs, base);
}

fn quad(i: u8) -> [u8; 4] {
  u32::from(i).to_ne_bytes()
}

/// Built, then edited every way that makes or moves a list block (pushes past the capacity,
/// insertions, removals, `retain`, `swap`, a list moved from one place to another), compacted,
/// cloned, loaded at an odd address, in a storage of its own and inline: aligned each time.
fn lists_stay_aligned<O: nearest::Offset, S: nearest::Storage + Default>() {
  let mut r = nearest::Region::<Bytes<O>, S>::new_in(
    S::default(),
    Bytes::new([quad(1)], [i32le::new(2)], [[1u8; 8]], [[3u8, 0]]),
  );
  let base = |r: &Bytes<O>| (&raw const *r).addr();
  aligned(&r, base(&r));
  let e = r.edit();
  for i in 2..12 {
    e.quads().push(quad(i)); // grows: new blocks
    e.numbers().push(i32le::new(i32::from(i)));
    e.octets().insert(0, [i; 8]);
    e.pairs().push([i, 0]);
  }
  e.quads().retain(|q| u32::from_ne_bytes(*q) % 3 != 0);
  e.quads().swap(0, 2);
  let _ = e.numbers().remove(1);
  // A list moved to another place.
  let old = e.quads().replace([quad(7), quad(8)]);
  e.quads().set(old);
  aligned(&r, base(&r));
  let compact = nearest::Region::<Bytes<O>, S>::new_in(S::default(), &*r);
  aligned(&compact, base(&compact));
  let clone = compact.clone();
  aligned(&clone, base(&clone));
  let image = compact.as_bytes().unwrap();
  let mut odd = vec![0u8; image.len() + 1];
  odd[1..].copy_from_slice(image);
  let loaded = nearest::Region::<Bytes<O>, S>::from_bytes(&odd[1..]).unwrap();
  aligned(&loaded, base(&loaded));
  let view = nearest::view::<Bytes<O>>(loaded.as_bytes().unwrap()).unwrap();
  aligned(view, base(view));
}

#[test]
fn the_elements_of_a_list_are_aligned_as_declared() {
  lists_stay_aligned::<i16, nearest::Heap>();
  lists_stay_aligned::<i32, nearest::Heap>();
  lists_stay_aligned::<i32le, nearest::Heap>();
  lists_stay_aligned::<i64, nearest::Heap>();
  lists_stay_aligned::<i64, nearest::Inline<4096>>();
  lists_stay_aligned::<nearest::puns::num::i64be, nearest::Inline<4096>>();
}

// -------------------------------------------------------------------------------------------------
// Rules

#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(assert(lo <= hi))]
struct Span {
  lo: u16le,
  #[pun(valid(..1000))]
  hi: u16le,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Spans {
  spans: Near<[Span], i32le>,
}

/// A rule on a type that holds pointers.
#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(assert(lo <= hi))]
struct Ruled {
  lo: u16le,
  hi: u16le,
  name: Near<str, i32le>,
  next: Option<Near<Ruled, i32le>>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Book {
  names: Near<[Ruled], i32le>,
}

/// A value that breaks a rule of its type, and holds pointers, is copied and moved as any
/// other (`Flat::store` and `shift` do not read its rules, which `puns` checks at load):
/// compacted, moved in its list (`insert`, `swap`, `retain`, `replace` and `set`), copied into
/// another region, and still the value it was (under Miri on every target of CI).
#[test]
fn a_value_that_breaks_a_rule_is_copied_and_moved() {
  let leaf = |lo, hi, name| Ruled::new(u16le::new(lo), u16le::new(hi), name, None);
  let show = |b: &Book| {
    (b.names.iter())
      .map(|n| {
        let next = n.next.as_ref().map(|m| (m.lo.get(), &*m.name));
        format!("{} {} {} {next:?}", n.lo.get(), n.hi.get(), &*n.name)
      })
      .collect::<Vec<_>>()
  };
  // `9 > 1` and `8 > 2`: both break `lo <= hi`, one inside the other.
  let bad = Ruled::new(u16le::new(9), u16le::new(1), "bad", Near::new(leaf(8, 2, "worse")));
  let mut r = Region::new(Book::new((bad, leaf(1, 2, "good"))));
  let names = r.edit().names();
  names.insert(0, leaf(7, 3, "front"));
  names.swap(0, 2);
  names.retain(|n| &*n.name != "front");
  // `bad` moves out of its place, and into the place of `good`.
  let old = names.at(1).replace(leaf(5, 4, "other"));
  names.at(0).set(old);
  let compact = Region::new(&*r);
  let copy = Region::new(Book::new(&r.names[..]));
  let expected = vec!["9 1 bad Some((8, \"worse\"))".to_owned(), "5 4 other None".to_owned()];
  assert_eq!(show(&r), expected);
  assert_eq!(show(&compact), expected);
  assert_eq!(show(&copy), expected);
  // And it is still refused at load.
  assert!(Region::<Book>::from_bytes(compact.as_bytes().unwrap()).is_err());
}

/// A value that breaks a rule of its type can be built and saved, not loaded.
#[test]
fn a_value_that_breaks_a_rule_is_saved_not_loaded() {
  let span = |lo, hi| Span::new(u16le::new(lo), u16le::new(hi));
  let good = Region::new(Spans::new([span(1, 2), span(3, 3)]));
  assert_eq!(Region::<Spans>::from_bytes(good.as_bytes().unwrap()).as_deref(), Ok(&*good));
  // Built and edited without a check: the rules are requirements, not guarantees.
  let mut bad = Region::new(Spans::new([span(1, 2), span(5, 3)]));
  let image = bad.as_bytes().unwrap().to_vec();
  assert_eq!(Region::<Spans>::from_bytes(&image).err(), Some(nearest::Error::Invalid));
  *bad.edit().spans().at(1).hi() = u16le::new(1000);
  *bad.edit().spans().at(1).lo() = u16le::new(0);
  let image = Region::new(&*bad).as_bytes().unwrap().to_vec();
  // `valid(..1000)` of `hi`.
  assert_eq!(Region::<Spans>::from_bytes(&image).err(), Some(nearest::Error::Invalid));
}
