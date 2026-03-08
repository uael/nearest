use crate::{Flat, buf::Buf};

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

/// Builder for a [`Region<T>`](crate::Region).
///
/// Values are appended to an internal byte buffer. Self-relative pointers
/// ([`Near`](crate::Near)/[`NearList`](crate::NearList)) are written in a
/// two-phase pattern: first reserve or push the struct containing a dangling
/// pointer, then call [`patch_near`](Self::patch_near) /
/// [`patch_list_header`](Self::patch_list_header) once the target position is known.
///
/// Consumed by [`Emitter::finish`] to produce an immutable [`Region<T>`](crate::Region).
/// The emitter is also used internally by [`Region::trim`](crate::Region::trim)
/// for deep-copy compaction.
#[cfg(feature = "alloc")]
pub struct Emitter<T: Flat, B: Buf = crate::buf::AlignedBuf<T>> {
  buf: B,
  _type: core::marker::PhantomData<T>,
}

#[cfg(not(feature = "alloc"))]
pub struct Emitter<T: Flat, B: Buf> {
  buf: B,
  _type: core::marker::PhantomData<T>,
}

impl<T: Flat, B: Buf> Emitter<T, B> {
  /// Create an empty emitter.
  #[doc(hidden)]
  #[must_use]
  pub fn new() -> Self {
    Self { buf: B::empty(), _type: core::marker::PhantomData }
  }

  /// Create an emitter pre-allocated to hold at least `capacity` bytes.
  #[doc(hidden)]
  #[must_use]
  pub fn with_capacity(capacity: u32) -> Self {
    let mut em = Self::new();
    if capacity > 0 {
      em.buf.reserve(capacity);
    }
    em
  }

  /// Current write-cursor position.
  #[doc(hidden)]
  #[must_use]
  pub fn pos(&self) -> Pos {
    Pos(self.buf.len())
  }

  /// Reserve aligned zeroed space for one `U`, returning its position.
  #[doc(hidden)]
  pub fn reserve<U: Flat>(&mut self) -> Pos {
    self.buf.alloc::<U>()
  }

  /// Write a value at a previously reserved position.
  ///
  /// # Safety
  ///
  /// `pos` must have been allocated for `U` via `reserve::<U>()`, ensuring
  /// correct alignment. Writing at a misaligned or otherwise invalid position
  /// can produce an invalid region that causes UB when later read.
  ///
  /// # Panics
  ///
  /// Panics if the write would exceed the buffer bounds.
  pub(crate) unsafe fn write<U: Flat>(&mut self, pos: Pos, val: U) {
    // SAFETY: Caller guarantees `pos` was allocated for `U`.
    unsafe { crate::buf::write_flat(&mut self.buf, pos, val) };
  }

  /// Patch a [`Near<U>`](crate::Near) at position `at` to point to `target`.
  ///
  /// Computes the relative offset from `at` (the `off` field of `Near`) to `target`
  /// and writes it as a `NonZero<i32>`.
  ///
  /// # Safety
  ///
  /// `at` must point to a `Near<U>` field within a previously allocated value,
  /// and `target` must be a position allocated for `U`.
  ///
  /// # Panics
  ///
  /// Panics if the offset overflows `i32`, if `target == at`, or if out of bounds.
  pub(crate) unsafe fn patch_near(&mut self, at: Pos, target: Pos) {
    // SAFETY: Caller guarantees `at` points to a `Near<U>` and `target`
    // was allocated for `U`.
    unsafe { crate::buf::patch_near(&mut self.buf, at, target) };
  }

  /// Copy raw bytes to position `at`.
  ///
  /// # Safety
  ///
  /// `src` must be valid for reading `len` bytes. `at` must be a valid
  /// position with at least `len` bytes available.
  ///
  /// # Panics
  ///
  /// Panics if the write would exceed the buffer bounds.
  pub(crate) unsafe fn write_bytes_internal(&mut self, at: Pos, src: *const u8, len: usize) {
    // SAFETY: Caller guarantees `src` is valid for `len` bytes and `at`
    // has at least `len` bytes available.
    unsafe { crate::buf::write_bytes(&mut self.buf, at, src, len) };
  }

  /// Patch a [`NearList<U>`](crate::NearList) header at position `at`.
  ///
  /// Writes the self-relative offset to the first node and the element count.
  ///
  /// # Safety
  ///
  /// `at` must point to a `NearList<U>` field within a previously allocated
  /// value, and `target` must be a position of a `Segment<U>` (or `Pos::ZERO`
  /// when `len == 0`).
  ///
  /// # Panics
  ///
  /// Panics if the offset overflows `i32` or if out of bounds.
  pub(crate) unsafe fn patch_list_header(&mut self, at: Pos, target: Pos, len: u32) {
    // SAFETY: Caller guarantees `at` points to a `NearList<U>` and `target`
    // is a valid segment position (or `Pos::ZERO` when `len == 0`).
    unsafe { crate::buf::patch_list_header(&mut self.buf, at, target, len) };
  }

  /// Consume the emitter and produce an aligned [`Region<T>`](crate::Region).
  ///
  /// Zero-copy: moves the buffer directly into the `Region`.
  #[doc(hidden)]
  pub fn finish(self) -> crate::Region<T, B> {
    // SAFETY: The emitter constructed this buffer through the `Emit` trait,
    // which guarantees the root `T` and all reachable data are valid.
    unsafe { crate::Region::from_buf(self.buf) }
  }

  /// Const pointer to buffer start.
  pub(crate) fn buf_ptr(&self) -> *const u8 {
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
    crate::buf::alloc_segment::<U>(&mut self.buf, count)
  }

  /// Patch the `next` pointer of a segment at `seg_pos`.
  ///
  /// # Safety
  ///
  /// `seg_pos` must be a position of a previously allocated `Segment<T>`.
  pub(crate) unsafe fn patch_segment_next(&mut self, seg_pos: Pos, next_seg_pos: Pos) {
    // SAFETY: Caller guarantees `seg_pos` is a previously allocated segment.
    unsafe { crate::buf::patch_segment_next(&mut self.buf, seg_pos, next_seg_pos) };
  }

  /// Extract the underlying buffer (used by [`Region::trim`](crate::Region::trim)).
  pub(crate) fn into_buf(self) -> B {
    self.buf
  }
}

impl<T: Flat, B: Buf> Default for Emitter<T, B> {
  fn default() -> Self {
    Self::new()
  }
}
