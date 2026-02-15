use std::{mem, num::NonZero};

use crate::{Flat, Region, buf::AlignedBuf};

#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pos(pub(crate) u32);

impl Pos {
  #[doc(hidden)]
  pub const ZERO: Self = Self(0);

  #[doc(hidden)]
  #[must_use]
  pub const fn offset(self, bytes: usize) -> Self {
    Self(self.0 + bytes as u32)
  }
}

/// Builder for a [`Region<T>`].
///
/// Values are appended to an internal byte buffer. Self-relative pointers
/// ([`Near`](crate::Near)/[`NearList`](crate::NearList)) are written in a
/// two-phase pattern: first reserve or push the struct containing a dangling
/// pointer, then call [`patch_near`](Self::patch_near) /
/// [`patch_list_header`](Self::patch_list_header) once the target position is known.
///
/// Consumed by [`Emitter::finish`] to produce an immutable [`Region<T>`].
/// The emitter is also used internally by [`Region::trim`](crate::Region::trim)
/// for deep-copy compaction.
pub struct Emitter<T: Flat> {
  buf: AlignedBuf<T>,
}

impl<T: Flat> Emitter<T> {
  /// Create an empty emitter.
  #[doc(hidden)]
  #[must_use]
  pub const fn new() -> Self {
    Self { buf: AlignedBuf::new() }
  }

  /// Create an emitter pre-allocated to hold at least `capacity` bytes.
  #[doc(hidden)]
  #[must_use]
  pub fn with_capacity(capacity: u32) -> Self {
    Self { buf: AlignedBuf::with_capacity(capacity) }
  }

  /// Current write-cursor position.
  #[doc(hidden)]
  #[must_use]
  pub const fn pos(&self) -> Pos {
    Pos(self.buf.len())
  }

  /// Reserve aligned zeroed space for one `U`, returning its position.
  #[doc(hidden)]
  pub fn reserve<U: Flat>(&mut self) -> Pos {
    self.buf.alloc::<U>()
  }

  /// Write a value at a previously reserved position.
  ///
  /// # Panics
  ///
  /// Panics if the write would exceed the buffer bounds.
  pub(crate) fn write<U: Flat>(&mut self, pos: Pos, val: U) {
    let start = pos.0 as usize;
    let size = mem::size_of::<U>();
    assert!(
      start + size <= self.buf.len() as usize,
      "write out of bounds: {}..{} but buffer len is {}",
      start,
      start + size,
      self.buf.len()
    );
    // SAFETY: Bounds checked above. `pos` was previously allocated for `U` via
    // `reserve::<U>()` which ensures correct alignment. The source pointer is
    // valid for `size` bytes, and `mem::forget` prevents double-drop.
    unsafe {
      let src = std::ptr::from_ref(&val).cast::<u8>();
      std::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), size);
    }
    mem::forget(val);
  }

  /// Patch a [`Near<U>`](crate::Near) at position `at` to point to `target`.
  ///
  /// Computes the relative offset from `at` (the `off` field of `Near`) to `target`
  /// and writes it as a `NonZero<i32>`.
  ///
  /// # Panics
  ///
  /// Panics if the offset overflows `i32`, if `target == at`, or if out of bounds.
  pub(crate) fn patch_near(&mut self, at: Pos, target: Pos) {
    let rel = i64::from(target.0) - i64::from(at.0);
    let rel_i32: i32 = rel.try_into().expect("near offset overflow");
    let nz = NonZero::new(rel_i32).expect("near offset must be non-zero (target == at)");

    let start = at.0 as usize;
    let size = mem::size_of::<NonZero<i32>>();
    assert!(start + size <= self.buf.len() as usize, "patch_near out of bounds");
    // SAFETY: Bounds checked above. `at` points to an allocated `Near<U>` field
    // whose first 4 bytes are the `NonZero<i32>` offset.
    unsafe {
      let src = std::ptr::from_ref(&nz).cast::<u8>();
      std::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), size);
    }
  }

  /// Copy raw bytes to position `at`.
  ///
  /// # Panics
  ///
  /// Panics if the write would exceed the buffer bounds.
  pub(crate) fn write_bytes_internal(&mut self, at: Pos, src: *const u8, len: usize) {
    let start = at.0 as usize;
    assert!(
      start + len <= self.buf.len() as usize,
      "write_bytes out of bounds: {}..{} but buffer len is {}",
      start,
      start + len,
      self.buf.len()
    );
    // SAFETY: Bounds checked above. `src` is valid for `len` bytes (caller
    // contract). The destination range does not overlap with the source.
    unsafe {
      std::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), len);
    }
  }

  /// Patch a [`NearList<U>`](crate::NearList) header at position `at`.
  ///
  /// Writes the self-relative offset to the first node and the element count.
  ///
  /// # Panics
  ///
  /// Panics if the offset overflows `i32` or if out of bounds.
  pub(crate) fn patch_list_header(&mut self, at: Pos, target: Pos, len: u32) {
    let off_field_pos = at.0 as usize;
    let len_field_pos = off_field_pos + mem::size_of::<i32>();

    assert!(
      len_field_pos + mem::size_of::<u32>() <= self.buf.len() as usize,
      "patch_list_header out of bounds"
    );

    let rel: i32 = if len == 0 {
      0
    } else {
      let r = i64::from(target.0) - i64::from(at.0);
      r.try_into().expect("list header offset overflow")
    };

    // SAFETY: Bounds checked above. The list header at `at` has layout
    // `[i32 offset, u32 len]`, and both writes are within bounds.
    unsafe {
      let buf_ptr = self.buf.as_mut_ptr();
      std::ptr::copy_nonoverlapping(
        std::ptr::from_ref(&rel).cast::<u8>(),
        buf_ptr.add(off_field_pos),
        mem::size_of::<i32>(),
      );
      std::ptr::copy_nonoverlapping(
        std::ptr::from_ref(&len).cast::<u8>(),
        buf_ptr.add(len_field_pos),
        mem::size_of::<u32>(),
      );
    }
  }

  /// Consume the emitter and produce an aligned [`Region<T>`].
  ///
  /// Zero-copy: moves the `AlignedBuf` directly into the `Region`.
  #[doc(hidden)]
  pub fn finish(self) -> Region<T> {
    Region::from_buf(self.buf)
  }

  /// Mutable pointer to buffer start (for Patch impls).
  pub(crate) const fn buf_mut_ptr(&self) -> *mut u8 {
    self.buf.as_mut_ptr()
  }

  /// Const pointer to buffer start.
  pub(crate) const fn buf_ptr(&self) -> *const u8 {
    self.buf.as_ptr()
  }

  /// Ensure at least `additional` bytes of spare capacity.
  pub(crate) fn reserve_bytes(&mut self, additional: u32) {
    self.buf.reserve(additional);
  }

  /// Allocate a segment header plus `count` contiguous values of type `U`.
  ///
  /// Returns the position of the segment header. The segment's `len` field
  /// is initialized to `count`; `next` is 0 (end of chain, from zero-fill).
  /// Values start at offset `size_of::<Segment<U>>()` from the returned pos.
  pub(crate) fn alloc_segment_internal<U: Flat>(&mut self, count: u32) -> Pos {
    use crate::list::Segment;
    self.buf.align_to(align_of::<Segment<U>>());
    let pos = Pos(self.buf.len());
    let values_size = count.checked_mul(size_of::<U>() as u32).expect("segment values overflow");
    let total = (size_of::<Segment<U>>() as u32)
      .checked_add(values_size)
      .expect("segment total size overflow");
    self.buf.resize(self.buf.len() + total, 0);
    // Write segment len at offset 4 (next is already 0 from zero-fill).
    let len_offset = pos.0 as usize + size_of::<i32>();
    // SAFETY: `resize` just allocated `total` bytes starting at `pos`.
    // The `len` field is at `pos + 4`, within the freshly allocated region.
    unsafe {
      std::ptr::copy_nonoverlapping(
        std::ptr::from_ref(&count).cast::<u8>(),
        self.buf.as_mut_ptr().add(len_offset),
        size_of::<u32>(),
      );
    }
    pos
  }

  /// Extract the underlying buffer (used by [`Region::trim`]).
  pub(crate) fn into_buf(self) -> AlignedBuf<T> {
    self.buf
  }
}

impl<T: Flat> Default for Emitter<T> {
  fn default() -> Self {
    Self::new()
  }
}
