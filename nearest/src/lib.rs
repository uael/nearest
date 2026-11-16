//! Self-relative pointers in one buffer.
//!
//! A [`Region<T>`] holds a `T` and everything it points to in one contiguous buffer. Every
//! pointer is a [`Near<T>`]: 4 bytes holding the distance from its own address to its target.
//! Copying the buffer copies the whole graph, so `Clone` is a `memcpy`, and reading is plain
//! `&T`. Edits are checked by the borrow checker, with no runtime checks.
//!
//! ```
//! #![feature(pin_ergonomics)]
//! #![allow(incomplete_features)]
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
//! #[flat]
//! struct Inst { op: u8, args: Near<[u32]> }
//!
//! // Build. Constructors take anything that can be written as the field's type.
//! let mut r = Region::new(Block::new(0, (Inst::new(1, [10, 20, 30]), Inst::new(2, ())), None));
//!
//! // Read. It is just `&Block`.
//! assert_eq!(r.insts[0].args[1], 20);
//!
//! // Clone. One memcpy.
//! let snapshot = r.clone();
//!
//! // Edit. `root` is `Pin<&mut Block>`; ordinary patterns project it.
//! r.edit(|root, slot| {
//!   let Block { id, insts, .. } = root;
//!   *id.get_mut() = 7;                                 // in place
//!   slot.at(insts, |old, new| {                        // a change that needs new memory
//!     for inst in old { if inst.op == 1 { new.push(inst) } }  // old elements move
//!     new.push(Inst::new(9, [1, 2]));
//!   });
//! });
//! assert_eq!(r.insts.iter().map(|i| i.op).collect::<Vec<_>>(), [1, 9]);
//! assert_eq!(snapshot.insts.len(), 2);
//!
//! // Compact. A copy into a new region keeps only what is reachable.
//! r = Region::new(&*r);
//! assert_eq!(r.id, 7);
//! ```
//!
//! # Editing
//!
//! [`Region::edit`] hands out the root as `Pin<&mut T>` and a [`Slot`] for it. Plain data is
//! `Unpin`, so `Pin::get_mut` gives a real `&mut`. A change that needs new memory goes through a
//! slot: [`Slot::put`] stages a replacement, [`Slot::push`] builds a replacement list, and
//! [`Slot::at`] narrows to a place inside the old value.
//!
//! | To do this | Write |
//! |---|---|
//! | assign a plain field | `*id.get_mut() = 7` |
//! | overwrite with a plain value | `let mut e = e; e.set(Expr::Lit(0))` |
//! | clear an optional pointer | `let mut next = next; next.set(None)` |
//! | follow an optional pointer | `next.as_pin_mut().map(Near::as_pin_mut)` |
//! | sort a list of plain data | `list.as_pin_mut().get_mut().sort()` |
//! | visit list elements | `for x in list { .. }` |
//! | compact | `region = Region::new(&*region)` |
//!
//! # Safety model
//!
//! - A `Near` has no constructor and is not `Clone`: it only exists inside a region.
//! - `Near` is `!Unpin`, so values containing one are only reachable as `&T` or `Pin<&mut T>`;
//!   [`#[flat]`](flat) makes types `#[pin_v2]` so that patterns project the pin.
//! - Each target has one owning pointer: builders make fresh targets, `&T` copies, and a
//!   `Pin<&mut T>` moves. A slot accepts moves only from the value it replaces, which the
//!   brand `'o` on [`Slot`] and [`Emit`] enforces.
//! - New data is built in a scratch buffer and appended when the edit ends, so no reference
//!   sees the buffer move. No pointer is made from an integer; Miri runs with Tree Borrows
//!   and strict provenance.
//!
//! Types must have alignment at most 8, non-zero size and no drop glue (checked at compile
//! time). A region holds at most `i32::MAX` bytes; growing past that panics.
//!
//! # Toolchain
//!
//! Requires nightly (`nightly-2026-02-10`). Crates that use [`#[flat]`](flat) enable
//! `#![feature(pin_ergonomics)]`.
#![no_std]
#![feature(ptr_metadata)]
#![deny(missing_docs)]

extern crate alloc;

mod flat;
mod near;
mod region;

pub use flat::{Emit, Flat};
pub use near::Near;
/// Makes a struct or enum storable in a region.
///
/// It implements [`Flat`], adds `#[pin_v2]`, `#[repr(C)]` to structs without a `repr` and
/// `#[repr(u8)]` to enums, and generates constructors whose parameters take anything that can
/// be written as the field's type ([`Emit`]): `new` for structs with fields, and one function
/// per non-unit enum variant named after it in snake case. Generic type parameters must be
/// [`Flat`]. Unions, `repr(packed)`, enum `repr`s other than `u8` and explicit discriminants are
/// rejected.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
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
/// let doc = Region::new(Json::obj((
///   Field::new("a", Json::Num(1.0)),
///   Field::new("b", Json::arr((Json::Null, Json::str("x")))),
/// )));
/// let Json::Obj(fields) = &*doc else { unreachable!() };
/// assert_eq!(&*fields[0].key, "a");
/// assert_eq!(fields[1].value, *Region::new(Json::arr((Json::Null, Json::str("x")))));
/// ```
pub use nearest_derive::flat;
pub use region::{Region, Slot};

/// Plumbing for the code [`#[flat]`](flat) generates. Not public API.
#[doc(hidden)]
pub mod __private {
  pub use crate::flat::{Sink, make};
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
      r.edit(|_, slot| {
        for i in 0.. {
          slot.push(i);
        }
      });
    }));
    assert!(full.is_err());
    assert!(r.byte_len() > 1 << 15 && r.byte_len() <= 1 << 16);
    assert_eq!(r[..], (0..10).collect::<Vec<_>>());
    // What does not fit fails the same way, and the region is still intact.
    assert!(
      catch_unwind(AssertUnwindSafe(|| r.edit(|_, slot| slot.put(Near::from_iter(0..16_384u32)))))
        .is_err()
    );
    assert_eq!(r[..], (0..10).collect::<Vec<_>>());
    let compact = Region::new(&*r);
    assert_eq!(compact.byte_len(), 4 + 4 + 40);
    assert_eq!(compact[..], r[..]);
  }
}
