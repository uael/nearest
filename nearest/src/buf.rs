use std::{alloc, marker::PhantomData, ptr::NonNull};

use crate::{Flat, emitter::Pos};

/// Growable byte buffer with base pointer aligned to `max(align_of::<T>(), 8)`.
///
/// Uses `u32` for len/cap (max ~4 GiB, matches `Pos(u32)`).
///
/// The buffer base is always at least 8-byte aligned, which guarantees correct
/// alignment for all standard primitive types (up to `i64`/`u64`). If the root
/// type `T` has stricter alignment, the buffer uses that instead.
///
/// # Soundness
///
/// **Allocation**: Uses `std::alloc::{alloc, realloc, dealloc}` with a layout
/// whose alignment is `BUF_ALIGN`. New regions are zero-filled. Allocation
/// failure calls [`handle_alloc_error`](std::alloc::handle_alloc_error) (never
/// returns null silently).
///
/// **Provenance exposure**: After every `grow` or `clone`, the new pointer's
/// provenance is exposed via [`expose_provenance`](pointer::expose_provenance).
/// This allows [`Near::get`](crate::Near::get) and session operations to
/// recover the full allocation's provenance using
/// [`with_exposed_provenance`](std::ptr::with_exposed_provenance) when
/// following self-relative offsets.
///
/// **`Send`/`Sync`**: The buffer is exclusively owned (no aliased mutable
/// pointers). `&AlignedBuf` only provides `&[u8]` access. Both impls are
/// sound.
pub struct AlignedBuf<T> {
  ptr: NonNull<u8>,
  len: u32,
  cap: u32,
  _type: PhantomData<T>,
}

impl<T> AlignedBuf<T> {
  /// Buffer alignment: covers the root type and all standard primitives (up to 8).
  pub(crate) const BUF_ALIGN: usize = if align_of::<T>() >= 8 { align_of::<T>() } else { 8 };

  /// Create an empty buffer with a dangling aligned pointer.
  pub const fn new() -> Self {
    // Non-null, properly aligned, never dereferenced when cap == 0.
    // `without_provenance_mut` avoids an integer-to-pointer cast that would
    // confuse Miri's provenance tracking.
    // SAFETY: `BUF_ALIGN` is a power-of-two > 0, so `without_provenance_mut`
    // returns a non-null, well-aligned dangling pointer.
    let ptr = unsafe { NonNull::new_unchecked(std::ptr::without_provenance_mut(Self::BUF_ALIGN)) };
    Self { ptr, len: 0, cap: 0, _type: PhantomData }
  }

  /// Create an empty buffer pre-allocated to hold at least `capacity` bytes.
  pub fn with_capacity(capacity: u32) -> Self {
    let mut buf = Self::new();
    if capacity > 0 {
      buf.reserve(capacity);
    }
    buf
  }

  /// Current byte length.
  pub const fn len(&self) -> u32 {
    self.len
  }

  /// View the buffer contents as a byte slice.
  pub const fn as_bytes(&self) -> &[u8] {
    if self.len == 0 {
      return &[];
    }
    // SAFETY: When `len > 0`, the buffer was allocated via `grow()` which
    // guarantees `ptr` is valid for `cap >= len` bytes, all initialized.
    unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
  }

  /// Raw pointer to the buffer start.
  pub const fn as_ptr(&self) -> *const u8 {
    self.ptr.as_ptr()
  }

  /// Mutable raw pointer to the buffer start.
  pub const fn as_mut_ptr(&self) -> *mut u8 {
    self.ptr.as_ptr()
  }

  /// Grow or shrink the buffer, filling new bytes with `fill`.
  pub fn resize(&mut self, new_len: u32, fill: u8) {
    if new_len > self.len {
      self.reserve(new_len - self.len);
      // SAFETY: `reserve` ensures `cap >= new_len`, so writing
      // `[len..new_len)` is within the allocation.
      unsafe {
        std::ptr::write_bytes(
          self.ptr.as_ptr().add(self.len as usize),
          fill,
          (new_len - self.len) as usize,
        );
      }
    }
    self.len = new_len;
  }

  /// Ensure at least `additional` bytes of spare capacity.
  pub fn reserve(&mut self, additional: u32) {
    let required = self.len.checked_add(additional).expect("capacity overflow");
    if required <= self.cap {
      return;
    }
    let new_cap = required.max(self.cap.saturating_mul(2)).max(64);
    self.grow(new_cap);
  }

  /// Append bytes from a slice.
  pub fn extend_from_slice(&mut self, data: &[u8]) {
    let n = data.len() as u32;
    self.reserve(n);
    // SAFETY: `reserve` ensures `cap >= len + n`. The source slice is valid
    // for `n` bytes, and the destination `[len..len+n)` does not overlap.
    unsafe {
      std::ptr::copy_nonoverlapping(
        data.as_ptr(),
        self.ptr.as_ptr().add(self.len as usize),
        data.len(),
      );
    }
    self.len += n;
  }

  /// Pad `len` up to the next multiple of `align`.
  pub fn align_to(&mut self, align: usize) {
    let rem = (self.len as usize) % align;
    if rem != 0 {
      let pad = (align - rem) as u32;
      self.resize(self.len + pad, 0);
    }
  }

  /// Align + allocate space for one `U`, return its position.
  ///
  /// # Compile-time invariant
  ///
  /// `align_of::<U>() <= BUF_ALIGN` — the buffer base is aligned to `BUF_ALIGN`
  /// (at least 8), so any position that is a multiple of `align_of::<U>()` yields
  /// a correctly aligned absolute address.
  pub fn alloc<U: Flat>(&mut self) -> Pos {
    const {
      assert!(
        align_of::<U>() <= Self::BUF_ALIGN,
        "allocated type alignment exceeds buffer alignment"
      );
    }
    self.align_to(align_of::<U>());
    let pos = Pos(self.len);
    let size = size_of::<U>() as u32;
    self.resize(self.len + size, 0);
    pos
  }

  /// Reallocate to `new_cap`, preserving existing data.
  fn grow(&mut self, new_cap: u32) {
    debug_assert!(new_cap > self.cap);
    let align = Self::BUF_ALIGN;
    let new_size = new_cap as usize;

    let ptr = if self.cap == 0 {
      // First allocation — cannot realloc a dangling pointer.
      let layout = alloc::Layout::from_size_align(new_size, align).expect("invalid layout");
      // SAFETY: `layout` has non-zero size (`new_cap > 0` by `debug_assert`).
      let p = unsafe { alloc::alloc(layout) };
      if p.is_null() {
        alloc::handle_alloc_error(layout);
      }
      p
    } else {
      let old_layout =
        alloc::Layout::from_size_align(self.cap as usize, align).expect("invalid layout");
      // SAFETY: `self.ptr` was allocated with `old_layout`. `new_size >= old_size`
      // (guaranteed by callers). The layout has non-zero size.
      let p = unsafe { alloc::realloc(self.ptr.as_ptr(), old_layout, new_size) };
      if p.is_null() {
        alloc::handle_alloc_error(
          alloc::Layout::from_size_align(new_size, align).expect("invalid layout"),
        );
      }
      p
    };

    // SAFETY: `ptr` is non-null (checked above), valid for `new_cap` bytes.
    // Zero-fill the new region `[cap..new_cap)`. Expose provenance so that
    // `Near::get` can recover it via `with_exposed_provenance`.
    unsafe {
      std::ptr::write_bytes(ptr.add(self.cap as usize), 0, (new_cap - self.cap) as usize);
      let _ = ptr.expose_provenance();
      self.ptr = NonNull::new_unchecked(ptr);
    }
    self.cap = new_cap;
  }
}

impl<T> Clone for AlignedBuf<T> {
  fn clone(&self) -> Self {
    if self.cap == 0 {
      return Self::new();
    }
    let align = Self::BUF_ALIGN;
    let layout = alloc::Layout::from_size_align(self.cap as usize, align).expect("invalid layout");
    // SAFETY: `layout` has non-zero size (`cap > 0`). After allocation, we
    // copy `len` bytes from the source buffer, expose provenance for
    // `Near::get`, and wrap in `NonNull` (checked non-null above).
    let ptr = unsafe {
      let p = alloc::alloc(layout);
      if p.is_null() {
        alloc::handle_alloc_error(layout);
      }
      std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), p, self.len as usize);
      let _ = p.expose_provenance();
      NonNull::new_unchecked(p)
    };
    Self { ptr, len: self.len, cap: self.cap, _type: PhantomData }
  }
}

impl<T> Drop for AlignedBuf<T> {
  fn drop(&mut self) {
    if self.cap == 0 {
      return; // dangling pointer — never allocated
    }
    let align = Self::BUF_ALIGN;
    // SAFETY: `self.ptr` was allocated with this layout (`cap > 0`).
    // `from_size_align_unchecked` is safe because `align` is a power of two
    // and `cap` was previously accepted by the allocator.
    unsafe {
      let layout = alloc::Layout::from_size_align_unchecked(self.cap as usize, align);
      alloc::dealloc(self.ptr.as_ptr(), layout);
    }
  }
}

// SAFETY: AlignedBuf owns its buffer exclusively; no shared mutable state.
unsafe impl<T> Send for AlignedBuf<T> {}

// SAFETY: &AlignedBuf only provides &[u8] access.
unsafe impl<T> Sync for AlignedBuf<T> {}
