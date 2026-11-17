//! Self-relative pointers in one buffer.
//!
//! A [`Region<T>`] keeps a `T` and everything it points to in one buffer. A pointer is a
//! [`Near<T>`]: 4 bytes (or 2, or 8: see [Pointer widths](#pointer-widths)) holding the distance
//! from its own address to its target. So copying the buffer copies the whole value (`Clone` is
//! a `memcpy`), reading is plain `&T` with `Near` dereferencing like `Box`, and the bytes mean
//! the same at any address, so a region can be written out and read back.
//!
//! ```
//! use nearest::{Edit, Near, Region, flat};
//!
//! #[flat]
//! struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
//! #[flat]
//! struct Inst { op: u8, args: Near<[u32]> }
//!
//! // Build. Constructors take anything that can be written as the field's type.
//! let mut r = Region::new(Block::new(0, (Inst::new(1, [10, 20, 30]), Inst::new(2, ())), None));
//!
//! // Read. It is `&Block`.
//! assert_eq!(r.insts[0].args[1], 20);
//!
//! // Edit. `r.edit()` is `&mut Edit<Block>`; every field has a method of the same name.
//! let b = r.edit();
//! *b.id() = 7;
//! b.insts().push(Inst::new(9, [1, 2])).args().push(3);   // `push` returns the new element
//! b.insts().at(0).args()[1] = 21;                         // plain data is a real slice
//! b.insts().retain(|inst| inst.op != 2);
//!
//! // Move. A value that leaves its place can be written anywhere else in the region, in O(1).
//! let inst = b.insts().remove(0);                                 // like `Vec::remove`
//! b.next().insert(Block::new(1, (), None)).insts().push(inst);    // the instruction changes block
//! assert_eq!(b.next().as_ref().map(|n| &n.insts[0].args[..]), Some(&[10, 21, 30][..]));
//! let tail = b.next().take();                                     // like `Option::take`
//! drop(tail);
//!
//! // Compact. A copy into a new region keeps only what is reachable.
//! r = Region::new(&*r);
//! assert_eq!((r.id, r.insts.len(), &r.insts[0].args[..]), (7, 1, &[1, 2, 3][..]));
//! assert!(r.next.is_none());
//!
//! // Save and load. A compact region is its own image; loading checks the bytes.
//! let bytes = r.as_bytes().unwrap();
//! let copy = Region::<Block>::from_bytes(bytes)?;
//! assert_eq!(copy.insts[0].op, 9);
//! // A memory-mapped file, or any 8-aligned slice, is read in place.
//! let block: &Block = nearest::view(bytes)?;
//! assert_eq!(block.id, 7);
//! # Ok::<(), nearest::Error>(())
//! ```
//!
//! Enums get a view of their variant, and `replace` returns the old value, like
//! `mem::replace`; written somewhere else in the region, it moves there:
//!
//! ```
//! use nearest::{Edit, Near, Region, flat};
//!
//! #[flat]
//! enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
//!
//! /// `--x` becomes `x`, bottom up.
//! fn simplify(e: &mut Edit<Expr>) {
//!   match e.as_mut() {
//!     ExprMut::Lit(_) => {}
//!     ExprMut::Neg(x) => simplify(x),
//!     ExprMut::Add(args) => { simplify(args._0()); simplify(args._1()); }
//!   }
//!   if let ExprMut::Neg(x) = e.as_mut() && let ExprMut::Neg(y) = x.as_mut() {
//!     let inner = y.replace(Expr::Lit(0));     // like `mem::replace`: the old value comes back
//!     e.set(inner);                            // and moves up, O(1)
//!   }
//! }
//!
//! fn show(e: &Expr) -> String {
//!   match e {
//!     Expr::Lit(n) => n.to_string(),
//!     Expr::Neg(x) => format!("-{}", show(x)),
//!     Expr::Add(a, b) => format!("({} + {})", show(a), show(b)),
//!   }
//! }
//!
//! let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2))));
//! let e = r.edit();
//! simplify(e);
//! let old = e.replace(Expr::Lit(0));
//! e.set(Expr::neg(old));                       // wrap: the old value moves, O(1)
//! if let ExprMut::Neg(x) = e.as_mut() && let ExprMut::Add(args) = x.as_mut() {
//!   let a = args._0().replace(Expr::Lit(0));   // swap two children: three moves
//!   let b = args._1().replace(a);
//!   args._0().set(b);
//! }
//! assert_eq!(show(&r), "-(-2 + 1)");
//! let e = r.edit();
//! let old = e.replace(Expr::Lit(0));
//! e.set(Expr::neg(Expr::neg(old)));
//! simplify(e);
//! assert_eq!(show(&r), "-(-2 + 1)");
//! ```
//!
//! # Editing
//!
//! [`Region::edit`] returns the handle of the root, a `&mut` [`Edit<T>`]. Handles reach every
//! place in the region:
//!
//! | Place | Handle | Operations |
//! |---|---|---|
//! | plain data (`T: Copy`) | `&mut T` | ordinary Rust |
//! | `#[flat]` struct | generated | a method per field, `set`, `replace`, `update` |
//! | `#[flat]` enum `X` | generated | `as_mut()` (returns `XMut`), `set`, `replace`, `update` |
//! | `Near<T>` | the handle of the `T` | |
//! | `Near<[T]>` | [`ListEdit<T>`] | `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `try_at`, `set`, `replace`, `update`, `iter_pin_mut` |
//! | `Option<Near<T>>` | [`OptionEdit<T>`] | `as_mut`, `insert`, `take`, `set`, `update` |
//! | `Near<str>` | [`StrEdit`] | `set`, `replace`, `update` |
//!
//! Handles dereference to the value they designate, and mutably so for plain data. A handle
//! borrows its parent handle, so one handle is usable at a time, and no reference into the
//! buffer outlives a change that may grow it.
//!
//! A handle has two lifetimes, `&'a mut Edit<'e, T>`: `'a` borrows the parent handle, `'e` is the
//! edit, the borrow of the region taken by `Region::edit`. Both can be left out in the usual
//! case, `fn pass(e: &mut Edit<Expr>)`; a function that returns a handle or an [`Old`] names
//! them: `fn pop<'e>(list: &mut Edit<'e, Near<[Inst]>>) -> Old<'e, Inst>`.
//!
//! `set` assigns; `replace`, [`ListEdit::remove`] and [`OptionEdit::take`] return the value that
//! left its place as an [`Old`], which can be written into any other place of the region in
//! O(1). `update` builds the new value from the old one in a closure, so that wrapping needs no
//! placeholder: `e.update(Expr::neg)`, or `e.update(|old| Expr::add(old, Expr::Lit(1)))`. Edits
//! append: compact with `Region::new(&*region)`.
//!
//! # Several fields at once
//!
//! One handle is usable at a time, because any handle may grow the buffer. `pin()` on the
//! handle of a `#[flat]` type gives a `Pin<&mut T>` in which several places are borrowed
//! together, for reading, writing plain data and assigning values without pointers; nothing
//! is pushed, built or moved through it. `project()` returns the fields, pinned;
//! [`Near::as_pin_mut`] goes to the target of a pointer and `iter_pin_mut` to the elements of a
//! list.
//!
//! ```
//! use core::pin::Pin;
//!
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
//!
//! /// Adds `by` to every literal.
//! fn bump(e: Pin<&mut Expr>, by: i64) {
//!   match e.project() {
//!     ExprPin::Lit(mut n) => *n += by,
//!     ExprPin::Neg(x) => bump(x.as_pin_mut(), by),
//!     ExprPin::Add(a, b) => {
//!       let (a, b) = (a.as_pin_mut(), b.as_pin_mut()); // both at once
//!       bump(a, by);
//!       bump(b, by);
//!     }
//!   }
//! }
//!
//! let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2))));
//! bump(r.edit().pin(), 10);
//! assert!(matches!(&*r, Expr::Add(a, _) if matches!(**a, Expr::Lit(11))));
//! ```
//!
//! On nightly Rust, with the `nightly` feature, every `#[flat]` type is also marked `#[pin_v2]`
//! and is projected by patterns, as `project` does, in a crate that enables `pin_ergonomics`:
//! `match e { Expr::Add(a, b) => .. }`. Without the feature, `#[flat]` rejects a `#[pin_v2]`
//! written on the type.
//!
//! # Pointer widths
//!
//! A `Near<T>` keeps its distance in an `i32`, so a region holds at most 2 GiB. The second
//! parameter of `Near<T, O>` picks another [`Offset`]: `i16`, 2 bytes, for regions of at most
//! 32 KiB, or `i64`, 8 bytes, for regions as large as memory. List and string lengths are `O`s
//! too. Each field picks its width, and a type can take it as a parameter. The pointers of the
//! root bound the region, and a pointer may only lead to types whose pointers are at least as
//! wide (checked at compile time), so every distance fits. Building or editing past the bound
//! panics, as growing past a storage does, and bytes longer than it do not load.
//!
//! ```
//! use nearest::{Near, Offset, Region, flat};
//!
//! #[flat]
//! struct Table<O: Offset> { name: Near<str, O>, rows: Near<[u64], O> }
//!
//! let small = Region::new(Table::<i16>::new("small", [1, 2, 3]));
//! let large = Region::new(Table::<i64>::new("large", [1, 2, 3]));
//! assert_eq!((small.rows[2], large.rows[2]), (3, 3));
//! assert_eq!((size_of::<Table<i16>>(), size_of::<Table<i64>>()), (4, 16));
//! ```
//!
//! # The `nightly` feature
//!
//! The opt-in cargo feature `nightly` needs a nightly compiler for the whole build. With it
//! `#[flat]` marks every type `#[pin_v2]`, and gives it no `Unpin` impl of its own. It uses
//! `specialization` and `allow_internal_unstable` inside this crate and `nearest-derive`; crates
//! that use `#[flat]` need no `#![feature]` for it, only code that writes a pattern on a pinned
//! value needs `pin_ergonomics`. An `Unpin` impl written by hand for a type that holds a `Near`
//! would give a `&mut` to it through `Pin::get_mut`: in a crate with `pin_ergonomics` the
//! compiler rejects it, and in one without, the build fails where the type, or a type holding
//! it by value, is pinned (a build that generates code: `cargo check` does not report it).
//!
//! # Bytes
//!
//! [`Region::as_bytes`] returns the compact image, borrowed: the bytes a deep copy writes, which
//! a region is until it is edited. Save an edited region by copying it first:
//! `Region::new(&*r).as_bytes()`. [`Region::from_bytes`] checks and copies bytes; [`view`]
//! checks an 8-aligned slice and reads it in place. Checking accepts exactly the compact images
//! of valid values, in one pass, without recursion or panics; an [`Error`] says where the
//! bytes went wrong, or that a valid image does not fit the storage. Checking keeps the
//! pointers still to visit in memory that grows with the nesting of the data: [`view_with`] and
//! [`Region::from_bytes_with`] take it from the caller, as a slice of [`Pending`]
//! (`bytes.len() / 4` entries are always enough, `/ 2` with 2-byte pointers), and say if it
//! runs out. For bytes that are
//! trusted, [`Region::from_bytes_unchecked`] and [`view_unchecked`] skip the check and are
//! `unsafe`. The bytes hold native-endian integers and the layout of the target.
//!
//! # The `serde` feature
//!
//! With the opt-in cargo feature `serde` (which turns on `alloc`), a [`Region`] is `Serialize`
//! and `Deserialize`, as its bytes: it is written with `serialize_bytes` ([`Region::as_bytes`],
//! or the bytes of a compact copy if the region was edited), and read back through
//! [`Region::from_bytes`], so the bytes are checked and an [`Error`] becomes the format's error.
//! Formats that have no bytes, such as JSON, write a sequence of numbers, which is read back
//! too. Any storage with `Default` can be read into. The bytes are the same as `as_bytes` gives:
//! native-endian, with the layout of the target, so they suit caches and messages between
//! builds of the same program, not a format shared across architectures.
//!
//! ```toml
//! nearest = { version = "0.6", features = ["serde"] }
//! ```
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! #[derive(Debug, PartialEq)]
//! struct Msg { id: u32, body: Near<str> }
//!
//! let mut r = Region::new(Msg::new(1, "hi"));
//! r.edit().body().set("hello"); // edited: written as a compact copy
//! let json = serde_json::to_string(&r).unwrap();
//! let back: Region<Msg> = serde_json::from_str(&json).unwrap();
//! assert_eq!(back, r);
//! // Bytes that are not a `Msg` are an error of the format, not a panic.
//! assert!(serde_json::from_str::<Region<Msg>>("[1, 2, 3]").is_err());
//! # }
//! ```
//!
//! # Storage
//!
//! A region keeps its buffer in a [`Storage`]: `Region<T>` is `Region<T, Heap>`, on the [`Heap`],
//! which grows as needed ([`Heap::with_capacity`] makes room up front). [`Inline<N>`] keeps it in
//! `N` bytes inside the `Region` value, with no heap; growing past them panics, as growing past
//! what the pointers reach does. A storage of your own, such as an arena or a memory-mapped file,
//! implements the `unsafe` trait [`Storage`]: two accessors to its memory, the second of which
//! grows it if it can. A region in any storage is built with [`Region::new_in`], or with
//! [`Region::from_bytes`] when the storage has a `Default`; everything else is the same. A storage
//! may move with its region; a region cannot move while something borrows it, and every edit starts
//! from where it is.
//!
//! ```
//! use nearest::{Inline, Near, Region, flat};
//!
//! #[flat]
//! #[derive(Debug, PartialEq)]
//! struct Block { id: u32, insts: Near<[u8]> }
//!
//! let mut r = Region::new_in(Inline::<256>::default(), Block::new(0, [1, 2]));
//! r.edit().insts().push(3);
//! let moved = r;
//! assert_eq!(moved.insts[..], [1, 2, 3]);
//! assert_eq!(*Region::new(&*moved), *moved); // the same value, on the heap
//! ```
//!
//! The default feature `alloc` provides the [`Heap`]. Without it the crate needs no allocator:
//! a region names its storage (`Region<T, Inline<N>>`), and `from_bytes` and [`view`] are not
//! available, while [`Region::as_bytes`], [`Region::from_bytes_with`] and [`view_with`], which
//! take the memory for checking from the caller, are.
//!
//! # Safety model
//!
//! - A `Near` has no constructor and is not `Clone`: it only exists inside a region.
//! - No safe API gives a `&mut T` for a `T` that contains a `Near`; such places are reached
//!   through handles, which are unsized, so they cannot be swapped or assigned through.
//! - A region is a tree: builders make fresh targets, `&T` copies, and the only moves are
//!   [`Old`] (used at most once) and list operations. Bytes no longer reachable are never
//!   written again, so an `Old` stays intact for the whole edit, which it keeps borrowed.
//! - A handle holds a position, not an address, so it stays valid when the buffer grows; real
//!   references into the buffer borrow the handle.
//! - Handles and [`Old`] are neither `Send` nor `Sync`: an edit stays on one thread.
//! - A panic leaves every place either unchanged or holding its complete new value.
//! - No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
//!   provenance.
//!
//! Stored types must have alignment at most 8, non-zero size and no drop glue (checked at
//! compile time). A region holds at most what the pointers of its root reach (2 GiB with the
//! default `i32`; see [Pointer widths](#pointer-widths)), or what its storage gives; growing past
//! that panics.
#![no_std]
#![deny(missing_docs)]
// The `nightly` feature: `unpin` asks whether a type implements `Unpin` (see `mod unpin`).
#![cfg_attr(feature = "nightly", feature(specialization))]
#![cfg_attr(
  feature = "nightly",
  expect(incomplete_features, reason = "`specialization`, only to ask whether a type is `Unpin`")
)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod bytes;
mod edit;
mod flat;
mod near;
mod region;
mod storage;
#[cfg(feature = "nightly")]
mod unpin;

#[cfg(feature = "alloc")]
pub use bytes::view;
pub use bytes::{Error, Pending, view_unchecked, view_with};
pub use edit::{ListEdit, Old, OptionEdit, StrEdit};
pub use flat::{Emit, Flat};
pub use near::{Near, Offset};
/// Makes a struct or enum storable in a region.
///
/// It implements [`Flat`] and adds `#[repr(C)]` to structs (unless they are already `C` or
/// `transparent`) and `#[repr(u8)]` to enums. It generates:
///
/// - constructors whose parameters take anything that can be written as the field's type
///   ([`Emit`]): `new` for structs with fields, and one function per non-unit enum variant
///   named after it in snake case (`project_` for a variant called `Project`), or as
///   `#[flat(rename = "name")]` on the variant says. A parameter
///   for a field written `Near<X>` takes a builder of `X` and allocates the target (a list or
///   string builder if `X` is a slice or `str`); one for a field of plain data marked
///   `#[flat(into)]` takes `impl Into<FieldType>`;
/// - the handle, [`Edit<'e, X>`](Edit), with `set` (assign), `replace` (assign and return the
///   old value as an [`Old`]), `update` (build the new value from the old one) and `pin` (a
///   pinned view), and for structs one method per field
///   (`_0`, `_1`, .. for tuple fields; a trailing underscore for fields named `set`,
///   `replace`, `pin` or `update`) returning the field's handle, with the field's visibility;
/// - `project(self: Pin<&mut X>) -> XPin<'_>`: the fields of a pinned value, all pinned, as a
///   struct or an enum with the shape of `X`. `X` is `Unpin` exactly when its fields are, and
///   cannot implement it by hand. The types `XPin` and, for enums with fields, `XMut` are
///   defined next to `X`;
/// - for enums with fields, `as_mut` on the handle and the enum `XMut<'a, 'e>` it returns, all
///   of whose variants are tuple-like: unit variants stay units, a variant with one field holds
///   that field's handle, and a variant with several fields holds a handle with a method per
///   field (and no `set` or `replace`, so the variant cannot change under it).
///
/// Generic type parameters must be [`Flat`]. Lifetime parameters, unions, `repr(packed)`, enum
/// `repr`s other than `u8`, explicit discriminants, empty enums, enums of more than 256
/// variants, variants whose constructors would have the same name (`rename` one) and, without
/// the `nightly` feature, `#[pin_v2]` are rejected. `#[flat(..)]` on a variant or a field is
/// read and removed. A variant with fields takes `rename = "name"`, a new name for its
/// constructor, which no other constructor and not `project` may have; a field takes `into`,
/// unless it is written as a pointer (`Near<..>`, `Option<Near<..>>`).
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// enum Json { Null, Num(f64), Str(Near<str>), Arr(Near<[Json]>), Obj(Near<[Field]>) }
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// struct Field { key: Near<str>, value: Json }
///
/// // Tuples are lists whose elements may be different builders.
/// let mut doc = Region::new(Json::obj((
///   Field::new("a", Json::Num(1.0)),
///   Field::new("b", Json::arr((Json::Null, Json::str("x")))),
/// )));
/// if let JsonMut::Obj(fields) = doc.edit().as_mut() {
///   fields.at(0).value().set(Json::Null);
///   let key = fields.at(1).key().replace("c");
///   fields.at(0).key().set(key); // the old key moves
/// }
/// let Json::Obj(fields) = &*doc else { unreachable!() };
/// assert_eq!((&*fields[0].key, &fields[0].value), ("b", &Json::Null));
/// assert_eq!(fields[1].value, *Region::new(Json::arr((Json::Null, Json::str("x")))));
/// ```
///
/// A variant's constructor can be given another name, and a parameter of plain data can take
/// anything that converts to it:
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// enum Value {
///   Int(#[flat(into)] i64),
///   #[flat(rename = "text")]
///   Str(Near<str>),
/// }
///
/// let r = Region::new(Value::int(7u8));
/// assert!(matches!(&*r, Value::Int(7)));
/// let r = Region::new(Value::text("hi"));
/// assert!(matches!(&*r, Value::Str(s) if &**s == "hi"));
/// ```
pub use nearest_derive::flat;
pub use region::Region;
#[cfg(feature = "alloc")]
pub use storage::Heap;
pub use storage::{Inline, Storage};

/// The handle of a place holding a `T`, during the edit `'e`: `&mut Edit<'e, T>` is what
/// [`Region::edit`], field methods and [`ListEdit::at`] return.
///
/// Both lifetimes can be left out in a function that only takes a handle; one that returns a
/// handle or an [`Old`] names them, as in
/// `fn first<'a, 'e>(b: &'a mut Edit<'e, Block>) -> &'a mut Edit<'e, Inst>`.
///
/// For plain data it is `T` itself; for `Near<T>` it is the handle of the target; for lists,
/// optional pointers and strings it is [`ListEdit`], [`OptionEdit`] and [`StrEdit`]; for
/// `#[flat]` types it is generated.
///
/// ```
/// use nearest::{Edit, Near, Region, flat};
///
/// #[flat]
/// struct Counter { hits: u64, log: Near<[u64]> }
///
/// fn hit(c: &mut Edit<Counter>) {
///   *c.hits() += 1;
///   let n = c.hits;
///   c.log().push(n);
/// }
///
/// let mut r = Region::new(Counter::new(0, ()));
/// hit(r.edit());
/// hit(r.edit());
/// assert_eq!((r.hits, &r.log[..]), (2, &[1, 2][..]));
/// ```
pub type Edit<'e, T> = <T as Flat>::Edit<'e>;

/// Plumbing for the code [`#[flat]`](flat) generates. Not public API.
#[doc(hidden)]
pub mod __private {
  #[cfg(feature = "nightly")]
  pub use crate::unpin::unpin;
  use crate::{Emit, Flat, Near, Offset};
  pub use crate::{
    bytes::Check,
    edit::Raw,
    flat::{Buf, Sink, fits, make},
  };

  /// Names the handle of a variant with several fields, for `XMut`.
  pub trait Variant<const I: u8> {
    /// The handle, during the edit `'e`.
    type Edit<'e>: ?Sized;
  }

  /// What a constructor takes for a field written `Near<..>`: for `Near<T>` a builder of the
  /// target, which it allocates; for a list or a string, a builder of the pointer itself.
  pub trait Pointer: Flat {
    /// What the parameter builds.
    type Arg: Flat;
    /// The builder of the field.
    fn wrap(arg: impl Emit<Self::Arg>) -> impl Emit<Self>;
  }

  impl<T: Flat, O: Offset> Pointer for Near<T, O> {
    type Arg = T;
    fn wrap(arg: impl Emit<T>) -> impl Emit<Self> {
      Self::new(arg)
    }
  }

  impl<T: Flat, O: Offset> Pointer for Near<[T], O> {
    type Arg = Self;
    fn wrap(arg: impl Emit<Self>) -> impl Emit<Self> {
      arg
    }
  }

  impl<O: Offset> Pointer for Near<str, O> {
    type Arg = Self;
    fn wrap(arg: impl Emit<Self>) -> impl Emit<Self> {
      arg
    }
  }
}
