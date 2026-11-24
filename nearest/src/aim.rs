//! What a pointer points at, as `puns` sees it: the target of its eager aim
//! ([`puns::layout::Aim`]).
//!
//! A `Near<T, O>` derives `Pun` with an eager aim at a [`Pointee::Target`], at the address of the
//! pointer plus its offset, minus the bytes of the target before the place where the pointer aims
//! ([`Pointee::LEAD`]). A check of a value by `puns` follows every pointer and checks the target
//! there, in the bytes that it checks, so a reference that `puns` makes to a value that holds a
//! pointer is a reference to a valid target, which `Deref` reads without a check.
//!
//! The target of an aim holds its own metadata (the aim is thin), so the target of a pointer to a
//! list or a text is a [`Block`], which holds its length:
//!
//! | Pointer | Target | Aims at |
//! |---|---|---|
//! | `Near<T, O>` | `T` | the `T` |
//! | `Near<[T], O>` | `ListBlock<T, O>`: the capacity, the length, the elements | the length |
//! | `Near<str, O>` | `TextBlock<O>`: the length, the UTF-8 bytes | the length |
//!
//! The integers of the offset and of the length are read in the byte order of the `Offset`
//! ([`Offset::ENDIAN`]), which is what `to_isize` reads, and not as the layout of `O` says: an
//! `Offset` of the user's may be a struct whose layout is that of a native integer.
//!
//! # Reading a target
//!
//! `puns` reads the target of a pointer, with the methods `Near::target` and `Near::target_mut`
//! that its derive writes, which are `unsafe`. `Deref` and the pinned view call them, and uphold
//! their requirements so:
//!
//! - **Read.** A `&Near` exists only inside a region, whose writers put the target of each pointer
//!   at its offset, a valid value whose block holds its length, or inside bytes that `puns`
//!   checked with its targets (`view`), or that the caller of `view_unchecked` vouches `view`
//!   would accept. The `&Near` is made from the buffer that holds both, so it
//!   has its provenance under tree borrows, which Miri runs. Under stacked borrows, a reference to
//!   a `Near` gives access to its own bytes only, and the read is undefined behavior: `nearest`
//!   relies on tree borrows here, as everywhere it resolves a pointer from `&self`. The region or
//!   the bytes are borrowed with the `&Near`, so the target is not written while it lives.
//! - **Write.** A `Pin<&mut Near>` comes from a pinned view, which borrows the handle of the region
//!   mutably, and a `Near` owns its target alone, so the target is reached only through the
//!   result while it lives. The other requirements are those of a read.
//!
//! The rules of a target other than its length, such as an `assert`, are not needed, and a region
//! may hold a value that breaks one, as safe code can write it.

use core::mem::offset_of;

use puns::Pun;

use crate::{Flat, Offset, flat::fits_behind, near::Packed};

/// What a `Near<Self, O>` points at, as `puns` checks it.
pub trait Pointee<O: Offset> {
  /// The type of the target.
  type Target: ?Sized + Pun;
  /// The bytes of the target before the place where the pointer aims.
  const LEAD: usize;
}

impl<T: Flat, O: Offset> Pointee<O> for T {
  type Target = T;
  const LEAD: usize = {
    fits_behind::<T, O>();
    0
  };
}

impl<T: Flat, O: Offset> Pointee<O> for [T] {
  type Target = ListBlock<T, O>;
  const LEAD: usize = {
    fits_behind::<T, O>();
    ListBlock::<T, O>::LEN
  };
}

impl<O: Offset> Pointee<O> for str {
  type Target = TextBlock<O>;
  const LEAD: usize = TextBlock::<O>::LEN;
}

/// What a pointer to a list or a text aims at, as `puns` checks it.
///
/// A `head` (the capacity of a list, and nothing for a text), the length of the `data`, then the
/// data, a slice or a `str`, aligned for it. The pointer aims at the length, which is not negative.
/// The capacity and the length are `O`s of alignment 1 ([`Packed`]), as the image places them:
/// the block is then aligned as its data, and ends where its data ends (`size_of_val`, which
/// `puns` requires of a target, has no padding after the data).
///
/// No value of it is made: it is the type of the target of the aim, whose layout `puns` checks,
/// and whose offsets are those of the image (`ListBlock`, `TextBlock`).
#[derive(Pun)]
#[pun(bound(H: Pun, D: Pun, O: Offset + Pun))]
#[repr(C)]
pub struct Block<H, D: ?Sized, O: Copy> {
  head: H,
  len: Packed<O>,
  /// The elements or the bytes, which `Deref` returns.
  #[pun(len = signed(offset_of!(Self, len), size_of::<O>(), O::ENDIAN))]
  pub(crate) data: D,
}

/// The block of a list of `T`: its capacity, its length, then the elements.
pub type ListBlock<T, O> = Block<Packed<O>, [T], O>;

/// The block of a text: its length, then the UTF-8 bytes.
pub type TextBlock<O> = Block<(), str, O>;

impl<H, D: ?Sized, O: Copy> Block<H, D, O> {
  /// The offset of the length, where the pointer aims.
  pub const LEN: usize = offset_of!(Self, len);
}

// The offset of the data is that of an empty array of its elements in the same block: `repr(C)`
// places each field at the first offset after the field before it that is aligned for it, sized or
// not, and `[E; 0]` and `[E]` are aligned as `E` ([`r-layout.repr.c.struct`]).
//
// [`r-layout.repr.c.struct`]: https://doc.rust-lang.org/1.99.0/reference/type-layout.html#r-layout.repr.c.struct

impl<T, O: Copy> ListBlock<T, O> {
  /// The offset of the elements.
  pub const DATA: usize = offset_of!(Block<Packed<O>, [T; 0], O>, data);
  /// The bytes from the length to the elements.
  pub const SKIP: usize = Self::DATA - Self::LEN;
}

impl<O: Copy> TextBlock<O> {
  /// The offset of the UTF-8 bytes.
  pub const DATA: usize = offset_of!(Block<(), [u8; 0], O>, data);
  /// The bytes from the length to the UTF-8 bytes.
  pub const SKIP: usize = Self::DATA - Self::LEN;
}
