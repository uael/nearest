//! Self-relative pointer library for region-based allocation.
//!
//! `nearest` provides [`Near<T>`] (self-relative pointers) and [`NearList<T>`]
//! (segmented lists) stored in a contiguous [`Region<T>`] buffer. All pointers
//! are 4-byte `i32` offsets relative to their own address — cloning a region is
//! a plain `memcpy` with no fixup.
//!
//! Mutations use a branded [`Session`] API (ghost-cell pattern) where [`Ref<T>`]
//! tokens carry a compile-time brand preventing cross-session use or escape.
//!
//! # Safety model
//!
//! All data in a [`Region`] satisfies the [`Flat`] trait: no `Drop`, no heap
//! pointers, no interior mutability. Indirection is expressed exclusively
//! through [`Near<T>`] and [`NearList<T>`], which store `i32` offsets relative
//! to their own address.
//!
//! **Provenance**: Self-relative pointers cannot use strict Rust provenance
//! because `&self.offset` has provenance over only 4 bytes, while the target
//! `T` may be larger. Instead, every allocation exposes its provenance via
//! `expose_provenance`, and reads recover it via `with_exposed_provenance`.
//! This is the canonical pattern for self-relative pointers in Rust, accepted
//! by Miri (which emits int-to-ptr cast warnings but detects no UB).
//!
//! **Aliasing**: Mutation through [`Session`] uses pre-reservation to prevent
//! buffer reallocation, then recovers provenance for reads so the `&T` is not
//! derived from the `&mut Region` — avoiding Stacked Borrows violations.
//!
//! **Branding**: The [`Session`] API uses the ghost-cell pattern
//! (`for<'id> FnOnce(&mut Session<'id, …>)`) to make [`Ref<'id, T>`] tokens
//! invariant in `'id`. This prevents refs from escaping the session closure or
//! being used across sessions — verified by compile-fail tests.
//!
//! **Alignment**: The buffer base is aligned to `max(align_of::<Root>(), 8)`.
//! Every allocation within the buffer is aligned to `align_of::<T>()` via
//! padding. A compile-time assertion ensures `align_of::<T>() <= BUF_ALIGN`.
//!
//! **No-drop**: The derive macro emits `const { assert!(!needs_drop::<T>()) }`
//! to reject types with `Drop` impls at compile time.
//!
//! # Example
//!
//! ```
//! use nearest::{Flat, NearList, Region, empty};
//!
//! #[derive(Flat, Debug)]
//! struct Block {
//!   id: u32,
//!   items: NearList<u32>,
//! }
//!
//! let mut region = Region::new(Block::make(1, [10u32, 20, 30]));
//! assert_eq!(region.items.len(), 3);
//! assert_eq!(region.items[0], 10);
//!
//! // Mutate via a branded session.
//! region.session(|s| {
//!   let items = s.nav(s.root(), |b| &b.items);
//!   s.splice_list(items, [40u32, 50]);
//! });
//!
//! assert_eq!(region.items.len(), 2);
//! assert_eq!(region.items[0], 40);
//! assert_eq!(region.items[1], 50);
//! ```

#![feature(offset_of_enum)]
#![deny(missing_docs)]

mod buf;
mod emit;
mod emitter;
mod flat;
pub(crate) mod list;
mod near;
mod patch;
mod region;
/// Branded session API for safe region mutation.
///
/// The ghost-cell pattern (`for<'id>` universally quantified lifetime) ensures
/// that [`Ref<'id, T>`](session::Ref) tokens cannot escape the session closure
/// or be used across different sessions. This gives **compile-time** safety
/// with **zero runtime cost** — `Ref` is just a `u32` position + phantom brand.
pub mod session;

pub use emit::Emit;

/// Not part of the public API. Used by the derive macro.
#[doc(hidden)]
pub mod __private {
  pub use crate::emitter::Pos;

  /// Byte offset from a segment header to its first value.
  ///
  /// Equal to `size_of::<Segment<T>>()`. The derive macro uses this to
  /// compute value positions without exposing `Segment` publicly.
  #[must_use]
  pub const fn segment_values_offset<T>() -> usize {
    std::mem::size_of::<crate::list::Segment<T>>()
  }
}
pub use flat::Flat;
pub use list::NearList;
pub use near::Near;
pub use nearest_derive::{Emit, Flat};
pub use patch::Patch;
pub use region::Region;
pub use session::{ListTail, Ref, Session};

/// Returns an empty iterator suitable for any `NearList<T>` emitter parameter.
///
/// Since [`Infallible`](std::convert::Infallible) implements [`Emit<T>`] for all
/// `T: Flat`, an `Empty<Infallible>` satisfies any `IntoIterator<Item: Emit<T>>` bound.
///
/// # Examples
///
/// ```
/// use nearest::{Flat, NearList, Region, empty};
///
/// #[derive(Flat)]
/// struct Root { items: NearList<u32> }
///
/// // Use empty() when building a struct with an empty NearList.
/// let region = Region::new(Root::make(empty()));
/// assert!(region.items.is_empty());
/// ```
pub const fn empty() -> std::iter::Empty<std::convert::Infallible> {
  std::iter::empty()
}
