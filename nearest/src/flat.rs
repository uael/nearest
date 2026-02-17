use core::convert::Infallible;

use crate::{Patch, emitter::Pos};

/// Marker trait for types that can be stored in a [`Region`](crate::Region).
///
/// # Safety
///
/// Implementors must satisfy **all four** invariants:
///
/// 1. **No `Drop`**: The type must not implement `Drop`. The derive macro
///    enforces this with `const { assert!(!needs_drop::<Self>()) }`.
/// 2. **All fields are `Flat`**: Every field type must itself implement `Flat`.
///    This is enforced by where-clause bounds in the generated impl.
/// 3. **No heap pointers**: Indirection is expressed exclusively through
///    [`Near<T>`](crate::Near) and [`NearList<T>`](crate::NearList), which
///    store self-relative `i32` offsets. Raw pointers, `Box`, `Vec`, `String`,
///    `Rc`, etc. are forbidden.
/// 4. **Correct `deep_copy`**: Must write all transitively reachable data into
///    the target buffer at the correct offsets, patching self-relative pointers.
///
/// # Derive macro
///
/// `#[derive(Flat)]` generates all of this automatically. For enums, a
/// `#[repr(u8)]` or `#[repr(C, u8)]` attribute is required (data variants
/// need `#[repr(C, u8)]`) so the discriminant byte is at a known position.
///
/// # Blanket impl
///
/// A blanket `Emit<T> for &T` impl exists, so `T: Flat` alone implies
/// `&T: Emit<T>` — enabling deep-copy via `Region::trim` and
/// `re_splice_list` without explicit builder types.
pub unsafe trait Flat: Sized {
  #[doc(hidden)]
  const _ASSERT_NO_DROP: () = ();

  /// Deep-copy `self` into the buffer at position `at`, patching all
  /// self-relative pointers so they remain valid in the new location.
  ///
  /// # Safety
  ///
  /// `at` must be a position previously allocated for `Self` in the same buffer.
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos);

  /// Validate that `buf[addr..]` contains a valid representation of `Self`.
  ///
  /// Checks bounds, alignment, and type-specific invariants (e.g. enum
  /// discriminants, `Near<T>` non-null offsets, `NearList<T>` consistency).
  ///
  /// For types with `Near<T>` or `NearList<T>` fields, the derive macro
  /// generates code that recursively validates targets.
  ///
  /// # Errors
  ///
  /// Returns [`ValidateError`](crate::ValidateError) if the bytes at `addr`
  /// do not form a valid `Self`.
  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError>;

  /// Validate an `Option<Self>` at `addr` in `buf`.
  ///
  /// The default implementation just checks bounds and alignment for
  /// `Option<Self>`. Types with niche layouts (e.g. [`Near<T>`](crate::Near)
  /// using `NonZero<i32>`) override this to inspect the discriminant and
  /// validate the inner value when present.
  ///
  /// # Errors
  ///
  /// Returns [`ValidateError`](crate::ValidateError) if the bytes at `addr`
  /// do not form a valid `Option<Self>`.
  fn validate_option(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Option<Self>>(addr, buf)
  }
}

macro_rules! impl_flat {
  ($($ty:ty),*) => {
    $(
      // SAFETY: Primitive types have no Drop, no heap pointers, and are Copy.
      unsafe impl Flat for $ty {
        unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
          // SAFETY: caller guarantees `at` was allocated for this type.
          unsafe { p.write_flat(at, *self) };
        }

        fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
          crate::ValidateError::check::<Self>(addr, buf)
        }
      }
    )*
  };
}

impl_flat!(u8, u16, u32, i32, u64, i64);

// SAFETY: bool is a primitive, no Drop, no heap pointers, and is Copy.
// Standalone impl to add value validation (must be 0 or 1).
unsafe impl Flat for bool {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    // SAFETY: caller guarantees `at` was allocated for this type.
    unsafe { p.write_flat(at, *self) };
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Self>(addr, buf)?;
    let value = buf[addr];
    if value > 1 {
      return Err(crate::ValidateError::InvalidBool { addr, value });
    }
    Ok(())
  }
}

// SAFETY: Infallible is uninhabited — deep_copy is unreachable.
unsafe impl Flat for Infallible {
  unsafe fn deep_copy(&self, _p: &mut impl Patch, _at: Pos) {
    match *self {}
  }

  fn validate(_addr: usize, _buf: &[u8]) -> Result<(), crate::ValidateError> {
    Err(crate::ValidateError::Uninhabited)
  }
}

// SAFETY: If both A and B are Flat (no Drop, no heap), then (A, B) is also Flat.
unsafe impl<A: Flat, B: Flat> Flat for (A, B) {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    // SAFETY: Caller guarantees `at` was allocated for `(A, B)`.
    // Sub-positions are computed by `offset_of!` so they are valid.
    unsafe {
      self.0.deep_copy(p, at.offset(core::mem::offset_of!((A, B), 0)));
      self.1.deep_copy(p, at.offset(core::mem::offset_of!((A, B), 1)));
    }
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Self>(addr, buf)?;
    A::validate(addr + core::mem::offset_of!((A, B), 0), buf)?;
    B::validate(addr + core::mem::offset_of!((A, B), 1), buf)?;
    Ok(())
  }
}

// SAFETY: If T is Flat, then [T; N] is also Flat (no Drop, no heap).
unsafe impl<T: Flat, const N: usize> Flat for [T; N] {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    for (i, elem) in self.iter().enumerate() {
      // SAFETY: Caller guarantees `at` was allocated for `[T; N]`.
      // Each element offset `i * size_of::<T>()` is within the allocation.
      unsafe { elem.deep_copy(p, at.offset(i * size_of::<T>())) };
    }
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Self>(addr, buf)?;
    for i in 0..N {
      T::validate(addr + i * size_of::<T>(), buf)?;
    }
    Ok(())
  }
}

// SAFETY: If T is Flat (no Drop, no heap), then Option<T> is also Flat.
// Niche optimization: the compiler may use unused bit patterns of T (e.g.
// NonZero's zero) as the None discriminant, so the inner T may not have a
// fixed offset within Option<T>. We compute the offset at runtime via
// pointer subtraction from `&self` to `&val`, which works for all layouts.
unsafe impl<T: Flat> Flat for Option<T> {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    // SAFETY: Caller guarantees `at` was allocated for `Option<T>`.
    // Byte-copy the full Option<T> (handles discriminant/niche layout).
    unsafe {
      p.write_bytes(at, core::ptr::from_ref(self).cast(), size_of::<Self>());
    }
    // Step 2: For Some, deep-copy the inner T to fix up self-relative pointers.
    // The inner offset is computed at runtime via pointer subtraction, which
    // correctly handles any niche-optimized layout.
    if let Some(val) = self {
      let inner_offset = (core::ptr::from_ref(val) as usize) - (core::ptr::from_ref(self) as usize);
      // SAFETY: `at` was allocated for `Option<T>` and `inner_offset`
      // is the runtime-computed offset of the `Some` payload.
      unsafe { val.deep_copy(p, at.offset(inner_offset)) };
    }
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    T::validate_option(addr, buf)
  }
}
