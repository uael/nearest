use std::{convert::Infallible, mem::size_of};

use crate::{Flat, Patch, emitter::Pos};

/// Builder trait for constructing values in a [`Region`](crate::Region).
///
/// Implementors describe how to serialize a value of type `T` into a region buffer.
/// The `#[derive(Flat)]` macro generates builder types (e.g. `Foo::make(...)`)
/// that implement this trait, enabling fully declarative, tree-shaped region
/// construction via [`Region::new`](crate::Region::new).
///
/// # Safety
///
/// Implementations must correctly write exactly `size_of::<T>()` bytes at the given
/// position, with correct field offsets and pointer patching. This invariant is upheld
/// by the derive macro; manual implementations must ensure the same.
///
/// # Key implementations
///
/// | Type | Behavior |
/// |------|----------|
/// | Primitives (`u32`, `bool`, …) | Self-emit via `write_flat` |
/// | `&T` where `T: Flat` | Deep-copy via `Flat::deep_copy` (blanket impl) |
/// | `Ref<'id, T>` | Deep-copy from existing buffer position |
/// | `Infallible` | Unreachable (used for empty `NearList` iterators) |
/// | Generated builders | Field-by-field construction with pointer patching |
pub unsafe trait Emit<T>: Sized {
  /// Reserve space for `T`, write this builder's data, and return the position.
  fn emit(self, p: &mut impl Patch) -> Pos
  where
    T: Flat,
  {
    let at = p.alloc::<T>();
    // SAFETY: `at` was just allocated for `T` by `alloc::<T>()`.
    unsafe { self.write_at(p, at) };
    at
  }

  /// Write this builder's data at position `at`.
  ///
  /// # Safety
  ///
  /// `at` must be a position previously allocated for `T` in the same buffer.
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos);
}

// --- Primitive impls: each primitive type emits itself ---

macro_rules! impl_emit_primitive {
  ($($ty:ty),* $(,)?) => {
    $(
      // SAFETY: Primitives are `Copy + Flat`; `write_flat` byte-copies the value.
      unsafe impl Emit<$ty> for $ty {
        unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
          // SAFETY: caller guarantees `at` was allocated for this type.
          unsafe { p.write_flat(at, self) };
        }
      }
    )*
  };
}

impl_emit_primitive!(u8, u16, u32, i32, u64, i64, bool);

// --- Tuple impl ---

// SAFETY: Delegates to `Emit<A>` and `Emit<B>` at their correct `offset_of!` positions.
unsafe impl<A: Flat, B: Flat, BA: Emit<A>, BB: Emit<B>> Emit<(A, B)> for (BA, BB) {
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    // SAFETY: caller guarantees `at` was allocated for `(A, B)`.
    // The offsets are computed by `offset_of!` so the sub-positions are valid.
    unsafe {
      self.0.write_at(p, at.offset(std::mem::offset_of!((A, B), 0)));
      self.1.write_at(p, at.offset(std::mem::offset_of!((A, B), 1)));
    }
  }
}

// --- Option impl ---

// SAFETY: `Option<T>` is `Flat` when `T: Flat`; `write_flat` byte-copies the entire value.
unsafe impl<T: Flat> Emit<Self> for Option<T> {
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    // SAFETY: caller guarantees `at` was allocated for `Option<T>`.
    unsafe { p.write_flat(at, self) };
  }
}

// SAFETY: `Infallible` is uninhabited — `write_at` is unreachable.
unsafe impl<T: Flat> Emit<T> for Infallible {
  unsafe fn write_at(self, _p: &mut impl Patch, _at: Pos) {
    unreachable!("Infallible should never be emitted")
  }
}

// --- Array impl: [B; N] emits as [T; N] element-by-element ---

// SAFETY: Emits each element at its stride-aligned offset within the array.
unsafe impl<T: Flat, B: Emit<T>, const N: usize> Emit<[T; N]> for [B; N] {
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    for (i, elem) in self.into_iter().enumerate() {
      // SAFETY: caller guarantees `at` was allocated for `[T; N]`.
      // Each element offset `i * size_of::<T>()` is within the allocation.
      unsafe { elem.write_at(p, at.offset(i * size_of::<T>())) };
    }
  }
}

// --- Blanket deep-copy impl: Emit<T> for &T via Flat::deep_copy ---

// SAFETY: Delegates to `Flat::deep_copy` which correctly copies all fields
// and patches self-relative pointers.
unsafe impl<T: Flat> Emit<T> for &T {
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    // SAFETY: caller guarantees `at` was allocated for `T`.
    unsafe { self.deep_copy(p, at) };
  }
}
