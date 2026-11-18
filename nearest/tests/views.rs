//! A value of a region read as another type, with everything it points to (`puns::view_ref`,
//! `puns::try_view_ref`): proven at compile time, or checked for what is not; numbers of a fixed
//! byte order in a region; a value that holds pointers read as plain data.
//!
//! The types read as others hold no `Option<Near<..>>`: `puns` has no layout of it yet.

use nearest::{
  Flat, Near, Region, flat,
  puns::{
    self,
    num::{i16be, i32le, u16le, u32le},
  },
};

// -------------------------------------------------------------------------------------------------
// A schema and its next version

/// An enum without fields: a scalar.
#[flat]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Kind {
  File,
  Dir,
  Link,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Item {
  pub id: u32,
  pub hidden: bool,
  pub kind: Kind,
  pub name: Near<str, i32le>,
  pub tags: Near<[u16], i32le>,
  pub more: Near<[Item], i32le>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Dir {
  pub size: u64,
  pub flags: [bool; 3],
  pub entries: Near<[Item], i32le>,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
  Lit(i64),
  Neg(Near<Expr, i32le>),
  Add(Near<Expr, i32le>, Near<Expr, i32le>),
  Var(Near<str, i32le>),
  Call(Near<str, i32le>, Near<[Expr], i32le>),
  Char(char),
}

/// The next version of the schema, which reads every value of the first as it is: a variant
/// added at the end of each enum, a `bool` read as a `u8`, a `char` as a `u32`, `[u16]` as
/// `[[u8; 2]]`.
mod v2 {
  use nearest::{Near, flat, puns::num::i32le};

  #[flat]
  #[derive(Debug, PartialEq, Eq, Clone, Copy)]
  pub enum Kind {
    File,
    Dir,
    Link,
    /// New.
    Socket,
  }

  #[flat]
  #[derive(Debug, PartialEq, Eq)]
  pub struct Item {
    pub id: u32,
    /// Was a `bool`: there are more flags now.
    pub hidden: u8,
    pub kind: Kind,
    pub name: Near<str, i32le>,
    /// Was `[u16]`: read as bytes.
    pub tags: Near<[[u8; 2]], i32le>,
    pub more: Near<[Item], i32le>,
  }

  #[flat]
  #[derive(Debug, PartialEq, Eq)]
  pub struct Dir {
    pub size: u64,
    pub flags: [u8; 3],
    pub entries: Near<[Item], i32le>,
  }

  #[flat]
  #[derive(Debug, PartialEq, Eq)]
  pub enum Expr {
    Lit(i64),
    Neg(Near<Expr, i32le>),
    Add(Near<Expr, i32le>, Near<Expr, i32le>),
    Var(Near<str, i32le>),
    Call(Near<str, i32le>, Near<[Expr], i32le>),
    /// Was a `char`.
    Char(u32),
    /// New.
    Mul(Near<Expr, i32le>, Near<Expr, i32le>),
  }

  /// Evaluates an expression; variables are 0, calls sum their arguments.
  pub fn eval(e: &Expr) -> i64 {
    match e {
      Expr::Lit(n) => *n,
      Expr::Neg(a) => eval(a).wrapping_neg(),
      Expr::Add(a, b) => eval(a).wrapping_add(eval(b)),
      Expr::Mul(a, b) => eval(a).wrapping_mul(eval(b)),
      Expr::Var(_) => 0,
      Expr::Call(_, args) => args.iter().map(eval).fold(0, i64::wrapping_add),
      Expr::Char(c) => i64::from(*c),
    }
  }
}

#[test]
fn an_expression_as_its_next_version() {
  let r = Region::new(Expr::call(
    "f",
    (Expr::add(Expr::Lit(2), Expr::neg(Expr::Lit(5))), Expr::var("x"), Expr::Char('A')),
  ));
  let new: &v2::Expr = puns::view_ref(&*r);
  // Nothing is copied: the same bytes.
  assert!(core::ptr::addr_eq(new, &raw const *r));
  assert_eq!(v2::eval(new), 2 - 5 + 65);
  let v2::Expr::Call(name, args) = new else { panic!("not what was written") };
  assert_eq!(&**name, "f");
  assert!(matches!(args[2], v2::Expr::Char(65)));
  // A deep copy of the view is a region of the new type, which loads as one.
  let copy = Region::<v2::Expr>::new(new);
  assert_eq!(nearest::view::<v2::Expr>(copy.as_bytes().unwrap()), Ok(new));
  // The image of the old one is the image of the new one here: the same sizes everywhere.
  assert_eq!(copy.as_bytes(), r.as_bytes());
}

#[test]
fn a_directory_as_its_next_version() {
  let entry = Region::new(Item::new(
    7,
    true,
    Kind::Link,
    "a",
    &[0x0102u16, 0x0304][..],
    [Item::new(8, false, Kind::File, "b", (), ())],
  ));
  let r = Region::new(Dir::new(1 << 40, [true, false, true], Near::from_iter([&*entry, &*entry])));
  let new: &v2::Dir = puns::view_ref(&*r);
  assert_eq!((new.size, new.flags, new.entries.len()), (1 << 40, [1, 0, 1], 2));
  let e = &new.entries[1];
  assert_eq!((e.id, e.hidden, e.kind, &*e.name), (7, 1, v2::Kind::Link, "a"));
  assert_eq!(&e.tags[..], &[0x0102u16.to_ne_bytes(), 0x0304u16.to_ne_bytes()][..]);
  let more = &e.more[0];
  assert_eq!((more.id, more.kind, &*more.name, more.tags.len()), (8, v2::Kind::File, "b", 0));
  // The elements of a list, as a slice of the new ones.
  let entries: &[v2::Item] = puns::view_ref(&r.entries[..]);
  assert_eq!(entries.len(), 2);
  assert!(core::ptr::addr_eq(entries, &raw const r.entries[..]));
  assert_eq!(&*entries[0].more[0].name, "b");
}

/// Bytes checked when loaded, then read as the next version: the address of `view`.
#[test]
fn loaded_then_read_as_another_type() {
  let r = Region::new(Expr::neg(Expr::add(Expr::Lit(1), Expr::var("y"))));
  let old: &Expr = nearest::view(r.as_bytes().unwrap()).unwrap();
  assert_eq!(v2::eval(puns::view_ref(old)), -1);
}

/// The first variants of `Expr`.
#[flat]
#[derive(Debug, PartialEq, Eq)]
enum Small {
  Lit(i64),
  Neg(Near<Expr, i32le>),
}

/// A level of at most 3: a requirement of the type, checked where it is not proven.
#[flat]
struct Level {
  #[pun(valid(..=3))]
  level: u8,
  owner: Near<u32, i32le>,
}

#[flat]
struct Plain {
  level: u8,
  owner: Near<u32, i32le>,
}

/// What only some values are: checked, for the value itself; what they point to is proven.
#[test]
fn checked_at_run_time() {
  let low = Region::new(Plain::new(1, 7));
  let view = puns::try_view_ref::<Plain, Level>(&low).unwrap();
  assert_eq!((view.level, *view.owner), (1, 7));
  let high = Region::new(Plain::new(4, 7));
  assert_eq!(puns::try_view_ref::<Plain, Level>(&high).err(), Some(puns::Error::Validity));
  // Every `Level` is a `Plain`: proven.
  assert_eq!(puns::view_ref::<Level, Plain>(view).level, 1);
  // A variant that the target does not have.
  let lit = Region::new(Expr::Lit(3));
  assert!(puns::try_view_ref::<Expr, Small>(&lit).is_ok_and(|small| *small == Small::Lit(3)));
  let neg = Region::new(Expr::neg(Expr::Lit(3)));
  assert!(
    matches!(puns::try_view_ref::<Expr, Small>(&neg), Ok(Small::Neg(e)) if **e == Expr::Lit(3))
  );
  let var = Region::new(Expr::var("x"));
  assert!(puns::try_view_ref::<Expr, Small>(&var).is_err());
}

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
/// its own: the parameter is `Flat`).
#[flat]
#[derive(Debug, PartialEq, Eq)]
struct Boxed<T> {
  id: u8,
  value: Near<T, i32le>,
  more: Near<[T], i32le>,
}

#[test]
fn generic_types() {
  let r = Region::new(Ping::<i16be>::new(1, [Pong::ping(Ping::new(2, [Pong::End(true)]))]));
  let ping: &Ping<i16be> = puns::view_ref(&*r);
  assert_eq!((ping.id, ping.pongs.len()), (1, 1));
  let flags = Region::new(Boxed::<bool>::new(1, true, [false, true]));
  let bytes: &Boxed<u8> = puns::view_ref(&*flags);
  assert_eq!((bytes.id, *bytes.value, &bytes.more[..]), (1, 1, &[0, 1][..]));
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
  const { assert!(<u32le as Flat>::PLAIN && <u32le as Flat>::ANY && <i16be as Flat>::ANY) };
}

/// A value that holds pointers is read as plain data: an offset is initialized bytes.
#[test]
fn pointers_are_read_as_bytes() {
  let r = Region::new(Table::new(1, u32le::new(2), [u16le::new(3)]));
  assert_eq!(puns::as_bytes(&*r), [1, 2, 0, 0, 0, 11, 0, 0, 0]);
  assert_eq!(puns::head_ref::<Table, u8>(&r), &1);
  assert_eq!(puns::tail_ref::<Table, i32le>(&r).get(), 11);
  assert_eq!(puns::try_pun_ref::<Table, (bool, [u8; 8])>(&r).map(|pair| pair.0), Ok(true));
}
