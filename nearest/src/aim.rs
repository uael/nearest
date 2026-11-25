//! What a pointer points at, as `puns` sees it: the pointee of its eager aim
//! ([`puns::layout::Aim`]).
//!
//! A `Near<T, O>` derives `Pun` with an eager aim at the pointee that [`Pointee`] maps `T` to, at
//! the address of the pointer plus its offset. A check of a value by `puns` follows every pointer
//! and checks the pointee there, in the bytes that it checks, so a reference that `puns` makes to
//! a value that holds a pointer is a reference to a valid pointee, which `Deref` reads without a
//! check.
//!
//! | Pointer | Pointee | Points at |
//! |---|---|---|
//! | `Near<T, O>` | `T` | the `T` |
//! | `Near<[T], O>` | [`Thin<[T], ListLen<O>>`](puns::Thin): the capacity, the length, the elements | the length |
//! | `Near<str, O>` | [`Thin<str, O>`](puns::Thin): the length, the UTF-8 bytes | the length |
//!
//! The format is that of [`puns::Thin`]: the length right before the elements, which are aligned
//! for them and to the size of an `O`, and the capacity right before the length. The offset, the
//! capacity and the length are read as their layouts say ([`Offset`]): integers of the size, the
//! byte order and the sign of the offset. A length is not negative.
//!
//! # Reading a pointee
//!
//! `puns` reads the pointee of a pointer, with the methods `Near::pointee` and `Near::pointee_mut`
//! that its derive writes. `pointee` is safe, and `Deref` calls it: the offset of a `Near` is an
//! [`InPlace`](puns::InPlace), which no code makes, so a `&Near` comes from `puns`, which checked
//! its pointee (`view`, `from_bytes`), or from unsafe code of `nearest` that takes the obligation
//! that `InPlace` states ("Values made by unsafe code"):
//!
//! - **Read.** A region makes its values with its writers, which put the pointee of each pointer
//!   at its offset, a valid value behind its length, and gives a `&T` to its root from its buffer
//!   (`Deref` of `Region`), which has the provenance of the bytes of every pointee under tree
//!   borrows, which Miri runs. The region is borrowed with the `&T`, so no pointee is written
//!   while it lives. `view_unchecked` gives a `&T` for bytes that its caller vouches `view` would
//!   accept. Under stacked borrows, a reference to a `Near` gives access to its own bytes only, and
//!   the read is undefined behavior: `nearest` relies on tree borrows here, as everywhere it
//!   resolves a pointer from `&self`.
//! - **Write.** `pointee_mut` is `unsafe`: a `&mut Near` does not cover its pointee, which a check
//!   may find shared. A `Pin<&mut Near>` comes from a pinned view, which borrows the handle of the
//!   region mutably, and a `Near` that a region wrote owns its pointee alone, so the pointee is
//!   reached only through the result while it lives. A `Near` is not `Unpin`, so safe code cannot
//!   move one out of its place, which would move where its offset leads.
//!
//! The rules of a pointee other than its length, such as an `assert` or `cap == len`, are not
//! needed by a read, and a region may hold a value that breaks one, as safe code can write it.

use puns::{Pun, Thin};

use crate::Offset;

/// What a `Near<Self, O>` points at, as `puns` checks and reads it: the pointee of its aim, whose
/// value is `Self`.
pub trait Pointee<O: Offset> {
  /// The pointee.
  type Pointee: ?Sized + puns::Pointee<Value = Self>;
}

impl<T: Pun, O: Offset> Pointee<O> for T {
  type Pointee = T;
}

impl<T: Pun, O: Offset> Pointee<O> for [T] {
  type Pointee = Thin<[T], ListLen<O>>;
}

impl<O: Offset> Pointee<O> for str {
  type Pointee = Thin<str, O>;
}

/// What is stored before the elements of a list: its capacity, then its length, `O`s.
///
/// A list in a region may have spare room, which the bytes of a region do not: in the image that
/// loading checks, the capacity is the length (`assert`). No value of it is made: it is the type
/// of what `puns` checks before the elements.
#[derive(Pun)]
#[pun(bound(O: Pun))]
#[pun(assert(cap as u128 == len as u128))]
#[repr(C)]
pub struct ListLen<O: Offset> {
  cap: O,
  /// Within `LIMIT` of 0, as its bits say in the byte order of `O`: a length past it is refused
  /// where it is, and a negative one within it with the capacity, which it is not.
  #[pun(valid(..=O::LIMIT, ((1u128 << (8 * size_of::<O>())) - O::LIMIT as u128)..))]
  len: O,
}
