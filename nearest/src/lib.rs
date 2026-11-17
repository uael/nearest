//! Self-relative pointers in one buffer.
//!
//! A [`Region<T>`] keeps a `T` and everything it points to in one buffer. A pointer is a
//! [`Near<T>`]: 4 bytes holding the distance from its own address to its target. So copying
//! the buffer copies the whole value (`Clone` is a `memcpy`), reading is plain `&T` with `Near`
//! dereferencing like `Box`, and the bytes mean the same at any address.
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
//! b.next().insert(Block::new(1, (), None)).insts().push(Inst::new(4, ()));
//! assert_eq!(b.next().as_ref().map(|n| n.insts.len()), Some(1));
//! b.next().set(None);
//!
//! // Compact. A copy into a new region keeps only what is reachable.
//! r = Region::new(&*r);
//! assert_eq!(r.id, 7);
//! assert_eq!(r.insts.iter().map(|i| (i.op, &i.args[..])).collect::<Vec<_>>(),
//!            [(1, &[10, 21, 30][..]), (9, &[1, 2, 3][..])]);
//! assert!(r.next.is_none());
//! ```
//!
//! Enums get a view of their variant, and `replace` hands over the old value, which moves
//! into the new one instead of being copied:
//!
//! ```
//! use nearest::{Edit, Near, Region, flat};
//!
//! #[flat]
//! enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
//!
//! fn inner(e: &Expr) -> &Expr { match e { Expr::Neg(x) => x, e => e } }
//!
//! /// `--x` becomes `x`, bottom up.
//! fn simplify(e: &mut Edit<Expr>) {
//!   match e.as_mut() {
//!     ExprMut::Lit(_) => {}
//!     ExprMut::Neg(x) => simplify(x),
//!     ExprMut::Add(args) => { simplify(args._0()); simplify(args._1()); }
//!   }
//!   if let Expr::Neg(x) = &**e && let Expr::Neg(_) = &**x {
//!     e.replace(|old| old.map(|e| inner(inner(e))));   // the inner value moves up, O(1)
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
//! simplify(r.edit());
//! r.edit().replace(|old| Expr::neg(old));              // wrap: the old value moves, O(1)
//! assert_eq!(show(&r), "-(1 + -2)");
//! r.edit().replace(|old| Expr::neg(old));
//! simplify(r.edit());
//! assert_eq!(show(&r), "(1 + -2)");
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
//! | `Near<[T]>` | [`ListEdit<T>`] | `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `set`, `replace` |
//! | `Option<Near<T>>` | [`OptionEdit<T>`] | `as_mut`, `insert`, `set`, `replace` |
//! | `Near<str>` | [`StrEdit`] | `set`, `replace` |
//!
//! Handles dereference to the value they designate, and mutably so for plain data. A handle
//! borrows its parent handle, so one handle is usable at a time, and no reference into the
//! buffer outlives a change that may grow it. Edits append: compact with
//! `Region::new(&*region)`.
//!
//! # Safety model
//!
//! - A `Near` has no constructor and is not `Clone`: it only exists inside a region.
//! - No safe API gives a `&mut T` for a `T` that contains a `Near`; such places are reached
//!   through handles, which are unsized, so they cannot be swapped or assigned through.
//! - A region is a tree: builders make fresh targets, `&T` copies, and the only moves are
//!   [`Old`] (used at most once) and list operations.
//! - A handle holds a position, not an address, so it stays valid when the buffer grows; real
//!   references into the buffer borrow the handle.
//! - A panic leaves every place either unchanged or holding its complete new value.
//! - No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
//!   provenance.
//!
//! Stored types must have alignment at most 8, non-zero size and no drop glue (checked at
//! compile time). A region holds at most `i32::MAX` bytes; growing past that panics.
#![no_std]
#![deny(missing_docs)]

extern crate alloc;

mod edit;
mod flat;
mod near;
mod region;

pub use edit::{ListEdit, Old, OptionEdit, StrEdit};
pub use flat::{Emit, Flat};
pub use near::Near;
/// Makes a struct or enum storable in a region.
///
/// It implements [`Flat`] and adds `#[repr(C)]` to structs without a `repr` and `#[repr(u8)]`
/// to enums. It generates:
///
/// - constructors whose parameters take anything that can be written as the field's type
///   ([`Emit`]): `new` for structs with fields, and one function per non-unit enum variant
///   named after it in snake case. A parameter for a field written `Near<X>` (with `X` not a
///   slice or `str`) takes a builder of `X` and allocates the target;
/// - the handle, [`Edit<X>`](Edit), with `set` and `replace`, and for structs one method per
///   field (`_0`, `_1`, .. for tuple fields; `set_` and `replace_` for fields named `set` and
///   `replace`) returning the field's handle, with the field's visibility;
/// - for enums with fields, `as_mut` on the handle and the enum `XMut<'a>` it returns: unit
///   variants stay units, a variant with one field holds that field's handle, and a variant
///   with several fields holds a handle with a method per field.
///
/// Generic type parameters must be [`Flat`]. Unions, `repr(packed)`, enum `repr`s other than
/// `u8`, explicit discriminants, empty enums and enums of more than 256 variants are rejected.
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
///   fields.at(1).key().set("c");
/// }
/// let Json::Obj(fields) = &*doc else { unreachable!() };
/// assert_eq!((&*fields[0].key, &fields[0].value), ("a", &Json::Null));
/// assert_eq!(fields[1].value, *Region::new(Json::arr((Json::Null, Json::str("x")))));
/// ```
pub use nearest_derive::flat;
pub use region::Region;

/// The handle of a place holding a `T`: `&mut Edit<T>` is what [`Region::edit`], field methods
/// and [`ListEdit::at`] return.
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
pub type Edit<T> = <T as Flat>::Edit;

/// Plumbing for the code [`#[flat]`](flat) generates. Not public API.
#[doc(hidden)]
pub mod __private {
  pub use crate::{
    edit::Raw,
    flat::{Buf, Sink, fits, make},
  };

  /// Names the handle of a variant with several fields, for `XMut`.
  pub trait Variant<const I: u8> {
    /// The handle.
    type Edit: ?Sized;
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
