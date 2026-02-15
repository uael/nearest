use core::mem::MaybeUninit;
#[cfg(feature = "alloc")]
use core::{marker::PhantomData, ptr::NonNull};

use crate::{Flat, emitter::Pos};

/// Backing storage for a [`Region`](crate::Region).
///
/// # Safety
///
/// Implementors must guarantee:
/// - `as_ptr()` / `as_mut_ptr()` return pointers valid for `len()` bytes
/// - Buffer base is aligned to at least `ALIGN` bytes
/// - `resize` zero-fills new bytes `[old_len..new_len)` and preserves alignment
pub unsafe trait Buf: Sized {
  /// Buffer alignment guarantee.
  const ALIGN: usize;

  /// Create an empty buffer (len = 0).
  fn empty() -> Self;

  /// Raw pointer to the buffer start.
  fn as_ptr(&self) -> *const u8;

  /// Mutable raw pointer to the buffer start.
  fn as_mut_ptr(&mut self) -> *mut u8;

  /// View the buffer contents as a byte slice.
  fn as_bytes(&self) -> &[u8];

  /// Current byte length.
  fn len(&self) -> u32;

  /// Returns `true` if the buffer contains no bytes.
  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Current capacity in bytes.
  fn capacity(&self) -> u32;

  /// Grow or shrink the buffer, filling new bytes with `fill`.
  fn resize(&mut self, new_len: u32, fill: u8);

  /// Ensure at least `additional` bytes of spare capacity.
  fn reserve(&mut self, additional: u32);

  /// Append bytes from a slice.
  fn extend_from_slice(&mut self, data: &[u8]);

  /// Pad `len` up to the next multiple of `align`.
  fn align_to(&mut self, align: usize) {
    let rem = (self.len() as usize) % align;
    if rem != 0 {
      let pad = (align - rem) as u32;
      self.resize(self.len() + pad, 0);
    }
  }

  /// Align + allocate space for one `U`, return its position.
  ///
  /// # Compile-time invariant
  ///
  /// `align_of::<U>() <= ALIGN` — the buffer base is aligned to `ALIGN`
  /// (at least 8), so any position that is a multiple of `align_of::<U>()` yields
  /// a correctly aligned absolute address.
  fn alloc<U: Flat>(&mut self) -> Pos {
    const {
      assert!(align_of::<U>() <= Self::ALIGN, "allocated type alignment exceeds buffer alignment");
    }
    self.align_to(align_of::<U>());
    let pos = Pos(self.len());
    let size = size_of::<U>() as u32;
    self.resize(self.len() + size, 0);
    pos
  }

  /// Expose provenance so [`Near::get`](crate::Near::get) can recover it
  /// via `with_exposed_provenance`.
  fn expose_provenance(&self) {
    let _ = self.as_ptr().expose_provenance();
  }
}

// ---------------------------------------------------------------------------
// FixedBuf
// ---------------------------------------------------------------------------

/// Stack-backed fixed-capacity buffer with 8-byte alignment.
///
/// `FixedBuf<N>` provides `N` bytes of inline storage with no heap allocation,
/// enabling `nearest` regions in `no_std` environments or on the stack.
///
/// # Panics
///
/// [`reserve`](Buf::reserve) and [`resize`](Buf::resize) panic if the
/// requested size exceeds `N`.
#[repr(C, align(8))]
pub struct FixedBuf<const N: usize> {
  data: [MaybeUninit<u8>; N],
  len: u32,
}

impl<const N: usize> FixedBuf<N> {
  /// Create an empty fixed buffer.
  ///
  /// This is `const`, enabling `static` regions.
  #[must_use]
  pub const fn new() -> Self {
    Self { data: [MaybeUninit::uninit(); N], len: 0 }
  }
}

impl<const N: usize> Default for FixedBuf<N> {
  fn default() -> Self {
    Self::new()
  }
}

// SAFETY: FixedBuf is repr(C, align(8)), so the data pointer is 8-byte aligned.
// resize zero-fills new bytes. as_ptr/as_mut_ptr return pointers to self.data.
unsafe impl<const N: usize> Buf for FixedBuf<N> {
  const ALIGN: usize = 8;

  fn empty() -> Self {
    Self::new()
  }

  fn as_ptr(&self) -> *const u8 {
    self.data.as_ptr().cast()
  }

  fn as_mut_ptr(&mut self) -> *mut u8 {
    self.data.as_mut_ptr().cast()
  }

  fn as_bytes(&self) -> &[u8] {
    if self.len == 0 {
      return &[];
    }
    // SAFETY: data[..len] has been initialized by resize/extend_from_slice.
    unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.len as usize) }
  }

  fn len(&self) -> u32 {
    self.len
  }

  fn capacity(&self) -> u32 {
    N as u32
  }

  fn resize(&mut self, new_len: u32, fill: u8) {
    assert!(new_len as usize <= N, "FixedBuf capacity exceeded: requested {new_len}, capacity {N}");
    if new_len > self.len {
      // SAFETY: new_len <= N, so data[len..new_len] is within bounds.
      unsafe {
        core::ptr::write_bytes(
          self.data.as_mut_ptr().add(self.len as usize).cast::<u8>(),
          fill,
          (new_len - self.len) as usize,
        );
      }
    }
    self.len = new_len;
  }

  fn reserve(&mut self, additional: u32) {
    let required = self.len.checked_add(additional).expect("capacity overflow");
    assert!(required as usize <= N, "FixedBuf capacity exceeded: need {required}, capacity {N}");
  }

  fn extend_from_slice(&mut self, data: &[u8]) {
    let n = data.len() as u32;
    self.reserve(n);
    // SAFETY: reserve checked capacity. data[len..len+n] is within bounds.
    unsafe {
      core::ptr::copy_nonoverlapping(
        data.as_ptr(),
        self.data.as_mut_ptr().add(self.len as usize).cast(),
        data.len(),
      );
    }
    self.len += n;
  }
}

impl<const N: usize> Clone for FixedBuf<N> {
  fn clone(&self) -> Self {
    let mut new = Self::new();
    if self.len > 0 {
      // SAFETY: data[..len] is initialized. The new buffer has capacity N >= len.
      unsafe {
        core::ptr::copy_nonoverlapping(
          self.data.as_ptr(),
          new.data.as_mut_ptr(),
          self.len as usize,
        );
      }
      new.len = self.len;
    }
    new
  }
}

// SAFETY: FixedBuf contains only [MaybeUninit<u8>; N] and u32 — no shared mutable state.
unsafe impl<const N: usize> Send for FixedBuf<N> {}

// SAFETY: &FixedBuf only provides &[u8] access.
unsafe impl<const N: usize> Sync for FixedBuf<N> {}

// ---------------------------------------------------------------------------
// AlignedBuf (alloc feature)
// ---------------------------------------------------------------------------

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
/// **Allocation**: Uses `alloc::{alloc, realloc, dealloc}` with a layout
/// whose alignment is `BUF_ALIGN`. New regions are zero-filled. Allocation
/// failure calls [`handle_alloc_error`](alloc::alloc::handle_alloc_error)
/// (never returns null silently).
///
/// **Provenance exposure**: After every `grow` or `clone`, the new pointer's
/// provenance is exposed via [`expose_provenance`](pointer::expose_provenance).
/// This allows [`Near::get`](crate::Near::get) and session operations to
/// recover the full allocation's provenance using
/// [`with_exposed_provenance`](core::ptr::with_exposed_provenance) when
/// following self-relative offsets.
///
/// **`Send`/`Sync`**: The buffer is exclusively owned (no aliased mutable
/// pointers). `&AlignedBuf` only provides `&[u8]` access. Both impls are
/// sound.
#[cfg(feature = "alloc")]
pub struct AlignedBuf<T> {
  ptr: NonNull<u8>,
  len: u32,
  cap: u32,
  _type: PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T> AlignedBuf<T> {
  const BUF_ALIGN: usize = if align_of::<T>() >= 8 { align_of::<T>() } else { 8 };

  /// Create an empty buffer with a dangling aligned pointer.
  #[must_use]
  pub const fn new() -> Self {
    // Non-null, properly aligned, never dereferenced when cap == 0.
    // `without_provenance_mut` avoids an integer-to-pointer cast that would
    // confuse Miri's provenance tracking.
    // SAFETY: `BUF_ALIGN` is a power-of-two > 0, so `without_provenance_mut`
    // returns a non-null, well-aligned dangling pointer.
    let ptr = unsafe { NonNull::new_unchecked(core::ptr::without_provenance_mut(Self::BUF_ALIGN)) };
    Self { ptr, len: 0, cap: 0, _type: PhantomData }
  }

  /// Create an empty buffer pre-allocated to hold at least `capacity` bytes.
  #[must_use]
  pub fn with_capacity(capacity: u32) -> Self {
    let mut buf = Self::new();
    if capacity > 0 {
      buf.reserve(capacity);
    }
    buf
  }

  /// Reallocate to `new_cap`, preserving existing data.
  fn grow(&mut self, new_cap: u32) {
    debug_assert!(new_cap > self.cap);
    let align = Self::BUF_ALIGN;
    let new_size = new_cap as usize;

    let ptr = if self.cap == 0 {
      // First allocation — cannot realloc a dangling pointer.
      let layout = alloc::alloc::Layout::from_size_align(new_size, align).expect("invalid layout");
      // SAFETY: `layout` has non-zero size (`new_cap > 0` by `debug_assert`).
      let p = unsafe { alloc::alloc::alloc(layout) };
      if p.is_null() {
        alloc::alloc::handle_alloc_error(layout);
      }
      p
    } else {
      let old_layout =
        alloc::alloc::Layout::from_size_align(self.cap as usize, align).expect("invalid layout");
      // SAFETY: `self.ptr` was allocated with `old_layout`. `new_size >= old_size`
      // (guaranteed by callers). The layout has non-zero size.
      let p = unsafe { alloc::alloc::realloc(self.ptr.as_ptr(), old_layout, new_size) };
      if p.is_null() {
        alloc::alloc::handle_alloc_error(
          alloc::alloc::Layout::from_size_align(new_size, align).expect("invalid layout"),
        );
      }
      p
    };

    // SAFETY: `ptr` is non-null (checked above), valid for `new_cap` bytes.
    // Zero-fill the new region `[cap..new_cap)`. Expose provenance so that
    // `Near::get` can recover it via `with_exposed_provenance`.
    unsafe {
      core::ptr::write_bytes(ptr.add(self.cap as usize), 0, (new_cap - self.cap) as usize);
      let _ = ptr.expose_provenance();
      self.ptr = NonNull::new_unchecked(ptr);
    }
    self.cap = new_cap;
  }
}

#[cfg(feature = "alloc")]
// SAFETY: AlignedBuf's BUF_ALIGN >= 8. Buffer base is properly aligned via Layout.
// resize zero-fills new bytes. as_ptr/as_mut_ptr return the allocation pointer.
unsafe impl<T> Buf for AlignedBuf<T> {
  const ALIGN: usize = Self::BUF_ALIGN;

  fn empty() -> Self {
    Self::new()
  }

  fn as_ptr(&self) -> *const u8 {
    self.ptr.as_ptr()
  }

  fn as_mut_ptr(&mut self) -> *mut u8 {
    self.ptr.as_ptr()
  }

  fn as_bytes(&self) -> &[u8] {
    if self.len == 0 {
      return &[];
    }
    // SAFETY: When `len > 0`, the buffer was allocated via `grow()` which
    // guarantees `ptr` is valid for `cap >= len` bytes, all initialized.
    unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
  }

  fn len(&self) -> u32 {
    self.len
  }

  fn capacity(&self) -> u32 {
    self.cap
  }

  fn resize(&mut self, new_len: u32, fill: u8) {
    if new_len > self.len {
      self.reserve(new_len - self.len);
      // SAFETY: `reserve` ensures `cap >= new_len`, so writing
      // `[len..new_len)` is within the allocation.
      unsafe {
        core::ptr::write_bytes(
          self.ptr.as_ptr().add(self.len as usize),
          fill,
          (new_len - self.len) as usize,
        );
      }
    }
    self.len = new_len;
  }

  fn reserve(&mut self, additional: u32) {
    let required = self.len.checked_add(additional).expect("capacity overflow");
    if required <= self.cap {
      return;
    }
    let new_cap = required.max(self.cap.saturating_mul(2)).max(64);
    self.grow(new_cap);
  }

  fn extend_from_slice(&mut self, data: &[u8]) {
    let n = data.len() as u32;
    self.reserve(n);
    // SAFETY: `reserve` ensures `cap >= len + n`. The source slice is valid
    // for `n` bytes, and the destination `[len..len+n)` does not overlap.
    unsafe {
      core::ptr::copy_nonoverlapping(
        data.as_ptr(),
        self.ptr.as_ptr().add(self.len as usize),
        data.len(),
      );
    }
    self.len += n;
  }
}

#[cfg(feature = "alloc")]
impl<T> Clone for AlignedBuf<T> {
  fn clone(&self) -> Self {
    if self.cap == 0 {
      return Self::new();
    }
    let align = Self::BUF_ALIGN;
    let layout =
      alloc::alloc::Layout::from_size_align(self.cap as usize, align).expect("invalid layout");
    // SAFETY: `layout` has non-zero size (`cap > 0`). After allocation, we
    // copy `len` bytes from the source buffer, expose provenance for
    // `Near::get`, and wrap in `NonNull` (checked non-null above).
    let ptr = unsafe {
      let p = alloc::alloc::alloc(layout);
      if p.is_null() {
        alloc::alloc::handle_alloc_error(layout);
      }
      core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), p, self.len as usize);
      let _ = p.expose_provenance();
      NonNull::new_unchecked(p)
    };
    Self { ptr, len: self.len, cap: self.cap, _type: PhantomData }
  }
}

#[cfg(feature = "alloc")]
impl<T> Default for AlignedBuf<T> {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(feature = "alloc")]
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
      let layout = alloc::alloc::Layout::from_size_align_unchecked(self.cap as usize, align);
      alloc::alloc::dealloc(self.ptr.as_ptr(), layout);
    }
  }
}

#[cfg(feature = "alloc")]
// SAFETY: AlignedBuf owns its buffer exclusively; no shared mutable state.
unsafe impl<T> Send for AlignedBuf<T> {}

#[cfg(feature = "alloc")]
// SAFETY: &AlignedBuf only provides &[u8] access.
unsafe impl<T> Sync for AlignedBuf<T> {}
