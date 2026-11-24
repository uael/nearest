//! The loaders on a corpus of images and their corruptions, against expected results.
//!
//! Every image of a deterministic corpus (structs, enums, lists, text, optional pointers, numbers
//! of given values, each offset width and byte order), then every byte of it changed by a few
//! masks, every truncation, a few extensions and an address that is not aligned: what `view`,
//! `from_bytes` (at an odd address) and `view_with` (the least scratch that suffices) say of each
//! is folded into a digest per image. `corpus.expected` holds the digests of the check of `puns`
//! (`puns::check`), written with `NEAREST_CORPUS_WRITE=1`. The images are those of 0.8.1, byte for
//! byte, and so are the records, but in two ways (`NEAREST_CORPUS_DUMP` writes them, to compare
//! two versions):
//!
//! - The least scratch. `puns` checks in place a pointee that holds no pointer, when nothing waits
//!   before it, and keeps each other pointer that waits, where 0.8.1 kept the rest of a list in
//!   one entry: most images need fewer entries, and some invalid ones more, since `puns` finds
//!   where bytes fail with every pointee waiting. In both, an entry waits for a pointer of its own
//!   (see `Pending`).
//! - The length of a list that is negative, within the `LIMIT` of its offset, is refused where the
//!   length is, where 0.8.1 refused it where the list starts, at its capacity, as it refused a
//!   length that does not fit.
//!
//! 0.8.1 gave the results of 0.7 (the code that `#[flat]` generated, before the one validator of
//! `puns`), but for four errors at the second pointer of a `Call`, refused where it is.
//!
//! Under Miri, a fraction of the corpus runs, for the unsafe code that it reaches; the digests
//! are compared natively.

use std::{fmt::Write as _, hash::Hasher};

use nearest::{
  Error, Flat, Near, Offset, Pending, Region, flat,
  puns::num::{i16be, i32le, i64le, u8c, u8in, u16bec, u16le, u32le, u64lec},
};

// -------------------------------------------------------------------------------------------------
// A schema, generic over the offset

#[flat]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Kind {
  File,
  Dir,
  Link,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Item<O: Offset> {
  pub id: u32,
  pub hidden: bool,
  pub kind: Kind,
  pub name: Near<str, O>,
  pub tags: Near<[u16], O>,
  pub next: Option<Near<Item<O>, O>>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Dir<O: Offset> {
  pub size: u64,
  pub flags: [bool; 3],
  pub entries: Near<[Item<O>], O>,
  pub parent: Option<Near<Dir<O>, O>>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub enum Expr<O: Offset> {
  Lit(i64),
  Neg(Near<Expr<O>, O>),
  Add(Near<Expr<O>, O>, Near<Expr<O>, O>),
  Var(Near<str, O>),
  Call(Near<str, O>, Near<[Expr<O>], O>),
  Cond(Option<Near<Expr<O>, O>>, char),
}

/// Numbers of a given byte order.
#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Table {
  version: u8,
  count: u32le,
  rows: Near<[u16le], i32le>,
}

/// Numbers of a given value, and of given values.
#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Constants {
  magic: u16bec<0x0102>,
  version: u8c<3>,
  level: u8in<1, 5>,
  marks: Near<[u64lec<7>], i32le>,
}

// -------------------------------------------------------------------------------------------------
// The corpus

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
  const fn next(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
  }

  const fn below(&mut self, n: u64) -> usize {
    (self.next() % n) as usize
  }
}

const fn name(rng: &mut Rng) -> &'static str {
  ["a", "bé", "ç∂", "dir", "", "😀x", "long-name-of-a-file"][rng.below(7)]
}

fn item_chain<O: Offset>(rng: &mut Rng, len: usize) -> Region<Item<O>> {
  let kinds = [Kind::File, Kind::Dir, Kind::Link];
  let tags: Vec<u16> = (0..rng.below(4)).map(|_| rng.next() as u16).collect();
  let mut r =
    Region::new(Item::new(rng.next() as u32, false, Kind::Dir, name(rng), &tags[..], None));
  let mut cur = r.edit();
  for _ in 0..len {
    let tags: Vec<u16> = (0..rng.below(3)).map(|_| rng.next() as u16).collect();
    let (id, hidden, kind) = (rng.next() as u32, rng.below(2) == 1, kinds[rng.below(3)]);
    cur = cur.next().insert(Item::new(id, hidden, kind, name(rng), &tags[..], None));
  }
  Region::new(&*r)
}

fn expr<O: Offset>(rng: &mut Rng, depth: usize) -> Region<Expr<O>> {
  if depth == 0 {
    return match rng.below(3) {
      0 => Region::new(Expr::Lit(rng.next().cast_signed())),
      1 => Region::new(Expr::var(name(rng))),
      _ => Region::new(Expr::cond(None, char::from_u32(rng.next() as u32 >> 12).unwrap_or('?'))),
    };
  }
  match rng.below(5) {
    0 => Region::new(Expr::neg(&*expr(rng, depth - 1))),
    1 => {
      let (a, b) = (expr(rng, depth - 1), expr(rng, depth - 1));
      Region::new(Expr::add(&*a, &*b))
    }
    2 => {
      let args: Vec<_> = (0..rng.below(4)).map(|_| expr(rng, depth - 1)).collect();
      Region::new(Expr::call(name(rng), Near::from_iter(args.iter().map(|a| &**a))))
    }
    3 => Region::new(Expr::cond(Near::new(&*expr(rng, depth - 1)), 'x')),
    _ => expr(rng, 0),
  }
}

fn dir<O: Offset>(rng: &mut Rng, depth: usize) -> Region<Dir<O>> {
  let entries: Vec<_> = (0..rng.below(3))
    .map(|_| {
      let len = rng.below(2);
      item_chain::<O>(rng, len)
    })
    .collect();
  let parent = (depth > 0).then(|| dir::<O>(rng, depth - 1));
  let (size, flags) = (rng.next(), [rng.below(2) == 1, true, false]);
  let entries = || Near::from_iter(entries.iter().map(|e| &**e));
  parent.as_ref().map_or_else(
    || Region::new(Dir::new(size, flags, entries(), None)),
    |parent| Region::new(Dir::new(size, flags, entries(), Near::new(&**parent))),
  )
}

/// A copy of bytes at an address aligned as a region is: what `view` takes.
struct Aligned {
  words: Vec<u64>,
  len: usize,
}

impl Aligned {
  fn new(bytes: &[u8]) -> Self {
    let mut words = vec![0u64; bytes.len().div_ceil(8) + 1];
    // SAFETY: the words hold at least `bytes.len()` bytes, of any value.
    unsafe {
      core::ptr::copy_nonoverlapping(bytes.as_ptr(), words.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    Self { words, len: bytes.len() }
  }

  const fn bytes(&self) -> &[u8] {
    // SAFETY: `len` bytes of the words, initialized.
    unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
  }

  const fn bytes_mut(&mut self) -> &mut [u8] {
    // SAFETY: as in `bytes`.
    unsafe { core::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.len) }
  }
}

/// What the loaders say of bytes, written to the record.
fn verdict<T: Flat>(bytes: &[u8], out: &mut String) {
  let shown = |r: Result<(), Error>| match r {
    Ok(()) => "ok".to_owned(),
    Err(e) => format!("{e:?}"),
  };
  let viewed = nearest::view::<T>(bytes).map(|_| ());
  let _ = write!(out, "{};", shown(viewed));
}

/// What `from_bytes` says at an odd address, and the least scratch with which `view_with`
/// finishes (with what it says then).
fn loaded<T: Flat>(bytes: &[u8], out: &mut String) {
  let mut odd = vec![0u8; bytes.len() + 1];
  odd[1..].copy_from_slice(bytes);
  let from = Region::<T>::from_bytes(&odd[1..]).map(|_| ());
  let _ = write!(out, "from {from:?};");
  for size in 0.. {
    let mut scratch = vec![Pending::default(); size];
    let result = nearest::view_with::<T>(bytes, &mut scratch).map(|_| ());
    if result != Err(Error::ScratchFull) {
      let _ = write!(out, "scratch {size} {result:?};");
      break;
    }
  }
}

/// The record of an image and its corruptions.
fn record<T: Flat>(image: &[u8]) -> String {
  let mut out = String::new();
  let mut copy = Aligned::new(image);
  verdict::<T>(copy.bytes(), &mut out);
  loaded::<T>(copy.bytes(), &mut out);
  let masks: &[u8] = if cfg!(miri) { &[0xff] } else { &[0x01, 0x02, 0x04, 0x10, 0x80, 0xff] };
  let step = if cfg!(miri) { 11 } else { 1 };
  for at in (0..image.len()).step_by(step) {
    for &mask in masks {
      copy.bytes_mut()[at] ^= mask;
      verdict::<T>(copy.bytes(), &mut out);
      if mask == 0xff && !cfg!(miri) {
        loaded::<T>(copy.bytes(), &mut out);
      }
      copy.bytes_mut()[at] ^= mask;
    }
  }
  for cut in (0..image.len()).step_by(step) {
    verdict::<T>(&copy.bytes()[..cut], &mut out);
  }
  for extra in 1..=7 {
    verdict::<T>(Aligned::new(&[image, &vec![0; extra][..]].concat()).bytes(), &mut out);
  }
  // Not aligned as a region is.
  let shifted = Aligned::new(&[&[0u8][..], image].concat());
  verdict::<T>(&shifted.bytes()[1..], &mut out);
  out
}

/// A digest of a record: FNV-1a, 64 bits.
fn digest(record: &str) -> u64 {
  struct Fnv(u64);
  impl Hasher for Fnv {
    fn finish(&self) -> u64 {
      self.0
    }
    fn write(&mut self, bytes: &[u8]) {
      for byte in bytes {
        self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3);
      }
    }
  }
  let mut fnv = Fnv(0xcbf2_9ce4_8422_2325);
  fnv.write(record.as_bytes());
  fnv.finish()
}

/// Every image of the corpus: its name, and its record.
fn corpus() -> Vec<(String, String)> {
  fn of<O: Offset>(out: &mut Vec<(String, String)>, offset: &str, scale: usize) {
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    for i in 0..scale {
      let r = item_chain::<O>(&mut rng, i % 4);
      out.push((format!("item {offset} {i}"), record::<Item<O>>(r.as_bytes().unwrap())));
      let r = expr::<O>(&mut rng, 1 + i % 3);
      out.push((format!("expr {offset} {i}"), record::<Expr<O>>(r.as_bytes().unwrap())));
      if !cfg!(miri) {
        let r = dir::<O>(&mut rng, i % 2);
        out.push((format!("dir {offset} {i}"), record::<Dir<O>>(r.as_bytes().unwrap())));
      }
    }
  }
  let mut out = vec![];
  let scale = if cfg!(miri) { 1 } else { 6 };
  of::<i32le>(&mut out, "i32le", scale);
  if !cfg!(miri) {
    of::<i32>(&mut out, "i32", scale);
    of::<i16>(&mut out, "i16", scale);
    of::<i64>(&mut out, "i64", scale);
    of::<i16be>(&mut out, "i16be", scale);
    of::<i64le>(&mut out, "i64le", scale);
    let kind = Region::new(Kind::Link);
    out.push(("kind".into(), record::<Kind>(kind.as_bytes().unwrap())));
    let flags = Region::<[bool; 3]>::new([true, false, true]);
    out.push(("flags".into(), record::<[bool; 3]>(flags.as_bytes().unwrap())));
    let table = Region::new(Table::new(1, u32le::new(2), [u16le::new(3)]));
    out.push(("table".into(), record::<Table>(table.as_bytes().unwrap())));
    let constants = Region::new(Constants::new(
      u16bec::new(),
      u8c::new(),
      u8in::new(3).unwrap(),
      [u64lec::new(), u64lec::new()],
    ));
    out.push(("constants".into(), record::<Constants>(constants.as_bytes().unwrap())));
  }
  out
}

/// The expected digests, by name of image.
const EXPECTED: &str = include_str!("corpus.expected");

#[test]
fn the_loaders_give_the_recorded_results() {
  let corpus = corpus();
  // The records themselves, to compare two versions where the digests differ.
  if let Some(path) = std::env::var_os("NEAREST_CORPUS_DUMP") {
    let mut dump = String::new();
    for (name, record) in &corpus {
      let _ = writeln!(dump, "{name}\n{record}");
    }
    std::fs::write(path, dump.replace(';', "\n")).unwrap();
  }
  if std::env::var_os("NEAREST_CORPUS_WRITE").is_some() {
    let mut file = String::new();
    for (name, record) in &corpus {
      let _ = writeln!(file, "{name}\t{:016x}\t{}", digest(record), record.len());
    }
    std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/corpus.expected"), file).unwrap();
    return;
  }
  // Under Miri the corpus is smaller, and so is each record: the unsafe code is what is looked at.
  if cfg!(miri) {
    assert_ne!(corpus.len(), 0);
    return;
  }
  let expected: Vec<(&str, &str)> =
    EXPECTED.lines().map(|line| line.split_once('\t').expect("a name and a digest")).collect();
  assert_eq!(expected.len(), corpus.len(), "the corpus is not the one of the expected digests");
  for ((name, record), (expected_name, digest_and_len)) in corpus.iter().zip(&expected) {
    assert_eq!(name, expected_name);
    let ours = format!("{:016x}\t{}", digest(record), record.len());
    assert_eq!(&ours, digest_and_len, "{name}: not the recorded results");
  }
}

// -------------------------------------------------------------------------------------------------
// What `#[pun(..)]` says of a type, which 0.7 did not check

#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(assert(lo <= hi))]
struct Span {
  lo: u16le,
  hi: u16le,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(assert(count == 2 * pairs))]
struct Checked {
  #[pun(valid(1..=3))]
  level: u8,
  count: u8,
  pairs: u8,
  spans: Near<[Span], i32le>,
  next: Option<Near<Checked, i32le>>,
}

/// Each rule refused where its value is, behind pointers too: the results of `puns::nearest` (the
/// loaders of `puns` with the feature `nearest`, at `e5875f5`), which checked them.
#[test]
fn rules_are_checked_at_load() {
  let span = |lo, hi| Span::new(u16le::new(lo), u16le::new(hi));
  let r = Region::new(Checked::new(
    1,
    4,
    2,
    [span(1, 2), span(3, 3)],
    Near::new(Checked::new(3, 0, 0, (), None)),
  ));
  let image = r.as_bytes().unwrap().to_vec();
  assert_eq!(Region::<Checked>::from_bytes(&image).as_deref(), Ok(&*r));
  let base = r.as_bytes().unwrap().as_ptr().addr();
  let spans = (&raw const r.spans[0]).addr() - base;
  let next = (&raw const **r.next.as_ref().unwrap()).addr() - base;
  for (at, value, error) in [
    (0, 4, 0),             // `level`, out of its valid values
    (1, 5, 0),             // `count == 2 * pairs`, of the root
    (spans + 2, 0, spans), // `lo <= hi`, of the first span
    (spans + 4, 4, spans + 4),
    (next, 0, next),     // `level`, behind a pointer
    (next + 1, 1, next), // `count == 2 * pairs`, behind a pointer
  ] {
    let mut bad = image.clone();
    bad[at] = value;
    assert_eq!(Region::<Checked>::from_bytes(&bad).err(), Some(Error::Invalid(error)), "at {at}");
  }
}
