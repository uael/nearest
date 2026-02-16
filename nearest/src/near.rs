use core::{fmt, marker::PhantomData, num::NonZero, ops::Deref};

use crate::{Flat, Patch, emitter::Pos};

/// A self-relative pointer stored as a 4-byte `NonZero<i32>` offset.
///
/// # Layout
///
/// `#[repr(C)]` — 4 bytes, `align_of::<i32>()` alignment:
///
/// | Offset | Field | Type           |
/// |--------|-------|----------------|
/// | 0      | `off` | `NonZero<i32>` |
///
/// The offset is relative to the address of `off` itself (not the start of
/// the region). Non-zero guarantee ensures `Near<T>` never points to itself.
///
/// `Near<T>` is **not** `Copy` or `Clone` — moving it would invalidate the
/// self-relative offset. Use it only inside [`Region`](crate::Region) buffers
/// built by the emitter.
///
/// # Soundness
///
/// **Provenance recovery**: [`get`](Self::get) computes the target address by
/// adding the stored `i32` offset to the address of the offset field itself.
/// Because `&self.off` has provenance over only 4 bytes (insufficient for the
/// target `T`), the method uses `with_exposed_provenance` to recover the
/// full allocation's provenance — previously exposed by `AlignedBuf::grow`.
/// Miri validates this pattern with no UB detected.
///
/// **Non-zero invariant**: The offset is `NonZero<i32>`, so a `Near<T>` can
/// never point to itself (offset 0). This is enforced by the emitter's
/// `patch_near` method which panics on `target == at`.
pub struct Near<T> {
  off: NonZero<i32>,
  _type: PhantomData<T>,
}

// SAFETY: Near contains only a NonZero<i32> and PhantomData — no Drop, no heap.
// Unconditional impl: Near<T> is always Flat regardless of T, since it stores only
// an offset, not an actual T. This avoids circular trait bounds in recursive types.
//
// validate_option exploits the NonZero<i32> niche (0 = None) for Option<Near<T>>.
// Target validation is handled by the derive-generated code on the containing struct.
unsafe impl<T> Flat for Near<T> {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    // SAFETY: Caller guarantees `at` was allocated for `Near<T>`.
    // Byte-copy the 4-byte offset. Containing struct's deep_copy handles pointer following.
    unsafe {
      p.write_bytes(at, core::ptr::from_ref(self).cast(), size_of::<Self>());
    }
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Self>(addr, buf)?;
    let off = i32::from_ne_bytes(buf[addr..addr + 4].try_into().unwrap());
    if off == 0 {
      return Err(crate::ValidateError::NullNear { addr });
    }
    // Does NOT follow the offset — containing struct's derive code does that.
    Ok(())
  }

  fn validate_option(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    // Option<Near<T>> has niche layout: NonZero<i32> means 0 represents None.
    // Size is 4 bytes (same as Near<T>), no separate discriminant.
    crate::ValidateError::check::<i32>(addr, buf)?;
    let off = i32::from_ne_bytes(buf[addr..addr + 4].try_into().unwrap());
    if off == 0 {
      // None variant — valid.
      return Ok(());
    }
    // Some: offset is non-zero — Near header is valid.
    // Does NOT follow the offset to validate the target — the derive-generated
    // code on the containing struct handles that.
    Ok(())
  }
}

impl<T: Flat> Near<T> {
  /// Resolve the self-relative offset to a reference.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, Near, Region};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Wrapper { inner: Near<u32> }
  ///
  /// let region = Region::new(Wrapper::make(42u32));
  /// assert_eq!(*region.inner.get(), 42);
  ///
  /// // Deref also works (Near<T> implements Deref<Target = T>).
  /// assert_eq!(*region.inner, 42);
  /// ```
  #[must_use]
  pub fn get(&self) -> &T {
    // SAFETY: The offset was written by `Emitter::patch_near` which guarantees
    // the target lies within the same `Region` buffer. The region keeps the
    // buffer alive and properly aligned for `T`.
    //
    // Strict provenance is not possible here: `&self.off` has provenance over
    // only 4 bytes, but `T` may live anywhere in the buffer. A derived pointer
    // (e.g. `wrapping_byte_offset`) would fail Stacked Borrows retagging when
    // the target `&T` is larger than 4 bytes.
    //
    // Instead, `with_exposed_provenance` recovers the full allocation's
    // provenance (exposed by `AlignedBuf::grow`). This triggers Miri
    // int-to-ptr cast warnings but is not UB.
    unsafe {
      let base = core::ptr::from_ref(&self.off).cast::<u8>();
      let target = base.addr().wrapping_add_signed(self.off.get() as isize);
      &*core::ptr::with_exposed_provenance::<T>(target)
    }
  }
}

impl<T: Flat> Deref for Near<T> {
  type Target = T;

  fn deref(&self) -> &T {
    self.get()
  }
}

impl<T: Flat + PartialEq> PartialEq for Near<T> {
  fn eq(&self, other: &Self) -> bool {
    *self.get() == *other.get()
  }
}

impl<T: Flat + Eq> Eq for Near<T> {}

impl<T: Flat + fmt::Debug> fmt::Debug for Near<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(self.get(), f)
  }
}

impl<T: Flat + fmt::Display> fmt::Display for Near<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Display::fmt(self.get(), f)
  }
}
