//! Self-relative pointers in one buffer.
//!
//! A [`Region<T>`] keeps a `T` and everything it points to in one buffer. A pointer is a
//! [`Near<T>`]: 4 bytes holding the distance from its own address to its target. So copying
//! the buffer copies the whole value (`Clone` is a `memcpy`), reading is plain `&T` with `Near`
//! dereferencing like `Box`, and the bytes mean the same at any address, so a region can be
//! written out and read back.
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
//! // Save and load. Loading checks the bytes.
//! let bytes = r.to_bytes();
//! let copy = Region::<Block>::from_bytes(&bytes)?;
//! assert_eq!(copy.insts[0].op, 9);
//! // A memory-mapped file, or any 8-aligned slice, is read in place.
//! let block: &Block = Region::view(&bytes)?;
//! assert_eq!(block.id, 7);
//! # Ok::<(), nearest::Invalid>(())
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
//! | `#[flat]` struct | generated | a method per field, `set`, `replace` |
//! | `#[flat]` enum `X` | generated | `as_mut()` (returns `XMut`), `set`, `replace` |
//! | `Near<T>` | the handle of the `T` | |
//! | `Near<[T]>` | [`ListEdit<T>`] | `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `get_mut`, `set`, `replace` |
//! | `Option<Near<T>>` | [`OptionEdit<T>`] | `as_mut`, `insert`, `take`, `set` |
//! | `Near<str>` | [`StrEdit`] | `set`, `replace` |
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
//! O(1). Edits append: compact with `Region::new(&*region)`.
//!
//! # Bytes
//!
//! [`Region::to_bytes`] returns the compact image: the bytes a deep copy writes, borrowed from the
//! region when it has not been edited since it was built. [`Region::from_bytes`] checks and
//! copies bytes; [`Region::view`] checks an 8-aligned slice and reads it in place. Checking
//! accepts exactly the compact images of valid values, in one pass, without recursion or
//! panics. For bytes that are trusted, [`Region::from_bytes_unchecked`] and
//! [`Region::view_unchecked`] skip the check and are `unsafe`. The bytes hold native-endian
//! integers and the layout of the target.
//!
//! # The `unstable` feature
//!
//! The cargo feature `unstable` adds `update` to every handle, the closure form of `replace`:
//! `e.update(Expr::neg)` or `e.update(|old| Expr::add(old, Expr::Lit(1)))` wraps a value
//! without a placeholder. It works on stable
//! Rust; what the feature gates may change in any release.
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
//! compile time). A region holds at most `i32::MAX` bytes; growing past that panics.
#![no_std]
#![deny(missing_docs)]

extern crate alloc;

mod bytes;
mod edit;
mod flat;
mod near;
mod region;

pub use bytes::Invalid;
pub use edit::{ListEdit, Old, OptionEdit, StrEdit};
pub use flat::{Emit, Flat};
pub use near::Near;
/// Makes a struct or enum storable in a region.
///
/// It implements [`Flat`] and adds `#[repr(C)]` to structs (unless they are already `C` or
/// `transparent`) and `#[repr(u8)]` to enums. It generates:
///
/// - constructors whose parameters take anything that can be written as the field's type
///   ([`Emit`]): `new` for structs with fields, and one function per non-unit enum variant
///   named after it in snake case. A parameter for a field written `Near<X>` takes a builder
///   of `X` and allocates the target (a list or string builder if `X` is a slice or `str`);
/// - the handle, [`Edit<'e, X>`](Edit), with `set` (assign) and `replace` (assign and return
///   the old value as an [`Old`]), and for structs one method per field (`_0`, `_1`, .. for
///   tuple fields; `set_` and `replace_` for fields named `set` and `replace`) returning the
///   field's handle, with the field's visibility;
/// - for enums with fields, `as_mut` on the handle and the enum `XMut<'a, 'e>` it returns, all
///   of whose variants are tuple-like: unit variants stay units, a variant with one field holds
///   that field's handle, and a variant with several fields holds a handle with a method per
///   field (and no `set` or `replace`, so the variant cannot change under it).
///
/// Generic type parameters must be [`Flat`]. Lifetime parameters, unions, `repr(packed)`, enum
/// `repr`s other than `u8`, explicit discriminants, empty enums, enums of more than 256
/// variants and variants whose constructors would have the same name are rejected.
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
pub use nearest_derive::flat;
pub use region::Region;

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
  use crate::{Emit, Flat, Near};
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

  impl<T: Flat> Pointer for Near<T> {
    type Arg = T;
    fn wrap(arg: impl Emit<T>) -> impl Emit<Self> {
      Self::new(arg)
    }
  }

  impl<T: Flat> Pointer for Near<[T]> {
    type Arg = Self;
    fn wrap(arg: impl Emit<Self>) -> impl Emit<Self> {
      arg
    }
  }

  impl Pointer for Near<str> {
    type Arg = Self;
    fn wrap(arg: impl Emit<Self>) -> impl Emit<Self> {
      arg
    }
  }
}

#[cfg(test)]
mod tests {
  extern crate std;
  use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec::Vec,
  };

  use crate::{Near, Region};

  /// Unit tests build with a 64 KiB limit.
  #[test]
  fn the_size_limit_panics_and_leaves_a_valid_region() {
    assert!(catch_unwind(|| Region::new(Near::from_iter(0..20_000u32))).is_err());
    let mut r = Region::new(Near::from_iter(0..10u32));
    let full = catch_unwind(AssertUnwindSafe(|| {
      for i in 0..u32::MAX {
        r.edit().push(i);
      }
    }));
    assert!(full.is_err());
    assert!(r.byte_len() > 1 << 15 && r.byte_len() <= 1 << 16);
    assert_eq!(r[..10], (0..10).collect::<Vec<_>>());
    let before = r.to_vec();
    assert!(r[10..].iter().copied().eq(0..before.len() as u32 - 10));
    // What does not fit fails the same way, and the region is still intact.
    assert!(
      catch_unwind(AssertUnwindSafe(|| r.edit().set(Near::from_iter(0..16_384u32)))).is_err()
    );
    assert_eq!(r[..], before);
    let compact = Region::new(&*r);
    assert_eq!(compact.byte_len(), 4 + 4 + 4 + 4 * before.len());
    assert_eq!(compact[..], r[..]);
  }
}
