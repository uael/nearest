use crate::{
  Flat, Region,
  buf::Buf,
  emitter::{Emitter, Pos},
};

/// Abstraction over [`Emitter`] and [`Region`] for position-based writes.
///
/// Both types support writing flat values and patching self-relative pointers
/// at known byte positions. This trait enables [`Emit::write_at`](crate::Emit::write_at)
/// to work with either during initial construction (via `Emitter`) or
/// post-construction mutation (via `Region`).
///
/// # Safety
///
/// All position arguments (`at`, `target`, `seg_pos`) must have been obtained
/// from prior `alloc` / `alloc_segment` calls on the same buffer. Out-of-bounds
/// writes are caught by assertions in every implementation.
#[doc(hidden)]
pub trait Patch {
  /// Allocate aligned space for one `T`, returning its position.
  fn alloc<T: Flat>(&mut self) -> Pos;

  /// Write a [`Flat`] value at position `at`.
  ///
  /// # Safety
  ///
  /// `at` must have been allocated for a type whose layout includes `T` at the
  /// corresponding offset.
  unsafe fn write_flat<T: Flat>(&mut self, at: Pos, val: T);

  /// Patch a [`Near<T>`](crate::Near) at position `at` to point to `target`.
  ///
  /// # Safety
  ///
  /// `at` must point to the `Near<T>` field within a previously allocated value,
  /// and `target` must be a position allocated for `T`.
  unsafe fn patch_near<T: Flat>(&mut self, at: Pos, target: Pos);

  /// Patch a [`NearList<T>`](crate::NearList) header at position `at` to point
  /// to `target` with `len` elements.
  ///
  /// Writes the self-relative offset and element count into the list header
  /// (same binary layout: `i32` offset + `u32` len).
  ///
  /// # Safety
  ///
  /// `at` must point to the `NearList<T>` field within a previously allocated
  /// value, and `target` must be a position allocated for a `Segment<T>` (or
  /// `Pos::ZERO` when `len == 0`).
  unsafe fn patch_list_header<T: Flat>(&mut self, at: Pos, target: Pos, len: u32);

  /// Copy raw bytes to position `at`.
  ///
  /// # Safety
  ///
  /// `at` must be a valid position with at least `len` bytes available.
  /// `src` must be valid for reading `len` bytes.
  unsafe fn write_bytes(&mut self, at: Pos, src: *const u8, len: usize);

  /// Allocate a [`Segment<T>`] with space for `count` contiguous values,
  /// returning the position of the segment header.
  ///
  /// The segment's `len` field is initialized to `count` and `next` is `0`
  /// (end of chain). Values start at offset `size_of::<Segment<T>>()`.
  fn alloc_segment<T: Flat>(&mut self, count: u32) -> Pos;

  /// Patch the `next` pointer of a segment at `seg_pos` to point to
  /// `next_seg_pos`.
  ///
  /// # Safety
  ///
  /// `seg_pos` must be a position of a previously allocated `Segment<T>`.
  unsafe fn patch_segment_next<T: Flat>(&mut self, seg_pos: Pos, next_seg_pos: Pos);

  /// Current buffer byte length.
  fn byte_len(&self) -> usize;

  /// Raw const pointer to buffer start.
  fn raw_ptr(&self) -> *const u8;

  /// Ensure at least `additional` bytes of spare capacity.
  fn reserve(&mut self, additional: u32);
}

impl<R: Flat, B: Buf> Patch for Emitter<R, B> {
  fn alloc<T: Flat>(&mut self) -> Pos {
    self.reserve::<T>()
  }

  unsafe fn write_flat<T: Flat>(&mut self, at: Pos, val: T) {
    self.write(at, val);
  }

  unsafe fn patch_near<T: Flat>(&mut self, at: Pos, target: Pos) {
    self.patch_near(at, target);
  }

  unsafe fn patch_list_header<T: Flat>(&mut self, at: Pos, target: Pos, len: u32) {
    self.patch_list_header(at, target, len);
  }

  unsafe fn write_bytes(&mut self, at: Pos, src: *const u8, len: usize) {
    self.write_bytes_internal(at, src, len);
  }

  fn alloc_segment<T: Flat>(&mut self, count: u32) -> Pos {
    self.alloc_segment_internal::<T>(count)
  }

  unsafe fn patch_segment_next<T: Flat>(&mut self, seg_pos: Pos, next_seg_pos: Pos) {
    // next field is at offset 0 of Segment<T>
    let rel = i64::from(next_seg_pos.0) - i64::from(seg_pos.0);
    let rel_i32: i32 = rel.try_into().expect("segment next offset overflow");
    let start = seg_pos.0 as usize;
    assert!(start + size_of::<i32>() <= self.pos().0 as usize, "patch_segment_next out of bounds");
    // SAFETY: Bounds checked above. The `next` field is at offset 0 of
    // `Segment<T>`, and we write exactly `size_of::<i32>()` bytes.
    unsafe {
      core::ptr::copy_nonoverlapping(
        core::ptr::from_ref(&rel_i32).cast::<u8>(),
        self.buf_mut_ptr().add(start),
        size_of::<i32>(),
      );
    }
  }

  fn byte_len(&self) -> usize {
    self.pos().0 as usize
  }

  fn raw_ptr(&self) -> *const u8 {
    self.buf_ptr()
  }

  fn reserve(&mut self, additional: u32) {
    self.reserve_bytes(additional);
  }
}

impl<R: Flat, B: Buf> Patch for Region<R, B> {
  fn alloc<T: Flat>(&mut self) -> Pos {
    self.alloc_internal::<T>()
  }

  unsafe fn write_flat<T: Flat>(&mut self, at: Pos, val: T) {
    self.write_flat_internal(at, val);
  }

  unsafe fn patch_near<T: Flat>(&mut self, at: Pos, target: Pos) {
    self.patch_near_internal(at, target);
  }

  unsafe fn patch_list_header<T: Flat>(&mut self, at: Pos, target: Pos, len: u32) {
    self.patch_list_header_internal(at, target, len);
  }

  unsafe fn write_bytes(&mut self, at: Pos, src: *const u8, len: usize) {
    self.write_bytes_internal(at, src, len);
  }

  fn alloc_segment<T: Flat>(&mut self, count: u32) -> Pos {
    self.alloc_segment_internal::<T>(count)
  }

  unsafe fn patch_segment_next<T: Flat>(&mut self, seg_pos: Pos, next_seg_pos: Pos) {
    self.patch_segment_next_internal(seg_pos, next_seg_pos);
  }

  fn byte_len(&self) -> usize {
    Self::byte_len(self)
  }

  fn raw_ptr(&self) -> *const u8 {
    self.deref_raw()
  }

  fn reserve(&mut self, additional: u32) {
    self.reserve_internal(additional);
  }
}
