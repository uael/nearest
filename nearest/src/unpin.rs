//! The `nightly` feature: `#[flat]` makes every type `#[pin_v2]` and gives it no `Unpin` impl of
//! its own. In a crate without `pin_ergonomics`, the compiler then accepts an `Unpin` impl
//! written by hand, which would give `&mut` to a value holding a `Near` through `Pin::get_mut`.
//! [`unpin`] rejects such an impl at compile time, when the type, or a type that holds it by
//! value, is pinned. In a crate with `pin_ergonomics`, the compiler rejects the impl itself.
//!
//! Each of these fails to build with the message of [`unpin`], and builds without the `impl`.
//!
//! Pinned through `pin()` on its handle:
//!
//! ```compile_fail
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Node { value: u32, next: Option<Near<Node>> }
//! impl Unpin for Node {}
//! let mut r = Region::new(Node::new(1, None));
//! let _ = r.edit().pin();
//! ```
//!
//! And without the impl:
//!
//! ```
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Node { value: u32, next: Option<Near<Node>> }
//! let mut r = Region::new(Node::new(1, None));
//! let _ = r.edit().pin();
//! ```
//!
//! Held by value in a type that is pinned:
//!
//! ```compile_fail
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Inner { next: Option<Near<Inner>> }
//! impl Unpin for Inner {}
//! #[flat(no_pun)] // holds an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Outer { id: u32, inner: Inner }
//! let mut r = Region::new(Outer::new(1, Inner::new(None)));
//! let _ = r.edit().pin();
//! ```
//!
//! And without the impl:
//!
//! ```
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Inner { next: Option<Near<Inner>> }
//! #[flat(no_pun)] // holds an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Outer { id: u32, inner: Inner }
//! let mut r = Region::new(Outer::new(1, Inner::new(None)));
//! let _ = r.edit().pin();
//! ```
//!
//! Reached through `Near::as_pin_mut`:
//!
//! ```compile_fail
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Leaf { next: Option<Near<Leaf>> }
//! impl Unpin for Leaf {}
//! #[flat(no_pun)] // points to an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Root { leaf: Near<Leaf> }
//! let mut r = Region::new(Root::new(Leaf::new(None)));
//! let RootPin { leaf } = r.edit().pin().project();
//! let _ = leaf.as_pin_mut();
//! ```
//!
//! And without the impl:
//!
//! ```
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Leaf { next: Option<Near<Leaf>> }
//! #[flat(no_pun)] // points to an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Root { leaf: Near<Leaf> }
//! let mut r = Region::new(Root::new(Leaf::new(None)));
//! let RootPin { leaf } = r.edit().pin().project();
//! let _ = leaf.as_pin_mut();
//! ```
//!
//! Reached through `iter_pin_mut`:
//!
//! ```compile_fail
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Leaf { next: Option<Near<Leaf>> }
//! impl Unpin for Leaf {}
//! #[flat(no_pun)] // points to an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Root { leaves: Near<[Leaf]> }
//! let mut r = Region::new(Root::new([Leaf::new(None)]));
//! let _ = r.edit().leaves().iter_pin_mut().count();
//! ```
//!
//! And without the impl:
//!
//! ```
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Leaf { next: Option<Near<Leaf>> }
//! #[flat(no_pun)] // points to an `Option<Near<..>>`, which `puns` does not describe yet
//! struct Root { leaves: Near<[Leaf]> }
//! let mut r = Region::new(Root::new([Leaf::new(None)]));
//! let _ = r.edit().leaves().iter_pin_mut().count();
//! ```
//!
//! In a crate with `pin_ergonomics`, the compiler rejects the impl itself, pinned or not:
//!
//! ```compile_fail
//! #![feature(pin_ergonomics)]
//! #![expect(incomplete_features, reason = "pin_ergonomics")]
//! use nearest::{Near, flat};
//!
//! #[flat]
//! struct Node { value: u32, next: Option<Near<Node>> }
//! impl Unpin for Node {}
//! ```
//!
//! And without the impl, a pattern projects the pinned value:
//!
//! ```
//! #![feature(pin_ergonomics)]
//! #![expect(incomplete_features, reason = "pin_ergonomics")]
//! use nearest::{Near, Region, flat};
//!
//! #[flat]
//! struct Node { value: u32, next: Option<Near<Node>> }
//! let mut r = Region::new(Node::new(1, None));
//! let Node { value, next } = r.edit().pin();
//! *value.get_mut() += 1;
//! assert!(next.is_none());
//! assert_eq!(r.value, 2);
//! ```

use crate::Flat;

/// Whether a type implements `Unpin`, asked of a type in general.
trait IsUnpin {
  const V: bool;
}

impl<T: ?Sized> IsUnpin for T {
  default const V: bool = false;
}

impl<T: ?Sized + Unpin> IsUnpin for T {
  const V: bool = true;
}

/// `Flat::UNPIN` of a `#[flat]` type, given whether all its fields are `Unpin`: the type may
/// be `Unpin` only if they all are.
///
/// # Panics
/// At compile time, where `UNPIN` is evaluated, if `T` is `Unpin` but `fields` is `false`.
#[must_use]
pub const fn unpin<T>(fields: bool) -> bool {
  assert!(
    fields || !<T as IsUnpin>::V,
    "`Unpin` implemented by hand for a type that holds a `Near`: it would give `&mut` to it"
  );
  fields
}

/// Checks, at compile time, a type about to be pinned: evaluates its `UNPIN`, which checks it
/// and every type it holds by value.
pub const fn pinned<T: Flat>() {
  const {
    let _: bool = T::UNPIN;
  };
}
