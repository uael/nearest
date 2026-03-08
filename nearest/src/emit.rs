use core::{convert::Infallible, marker::PhantomData, mem::size_of};

use crate::{Flat, Near, NearList, Patch, emitter::Pos, list::Segment};

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
      self.0.write_at(p, at.offset(core::mem::offset_of!((A, B), 0)));
      self.1.write_at(p, at.offset(core::mem::offset_of!((A, B), 1)));
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

// ---------------------------------------------------------------------------
// Wrapper functions for uniform Emit<Near<T>>, Emit<NearList<T>>, etc.
// ---------------------------------------------------------------------------

/// Construct a [`Near<T>`] from any `Emit<T>` builder.
///
/// Used as a wrapper when passing builders to `make()` for `Near<T>` fields:
/// ```ignore
/// Func::make(1, near(Block::make(0, list([10u32, 20, 30]))))
/// ```
pub fn near<T: Flat>(builder: impl Emit<T>) -> impl Emit<Near<T>> {
  struct W<B>(B);
  // SAFETY: Allocates space for T, emits the builder into it, then patches
  // the Near<T> offset at `at` to point to the allocated T.
  unsafe impl<T: Flat, B: Emit<T>> Emit<Near<T>> for W<B> {
    unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
      let target = Emit::<T>::emit(self.0, p);
      // SAFETY: `at` was allocated for `Near<T>` by the caller and `target`
      // was just allocated for `T` by `emit`.
      unsafe { p.patch_near::<T>(at, target) };
    }
  }
  W(builder)
}

/// Construct a [`NearList<T>`] from an iterator of `Emit<T>` builders.
///
/// Used as a wrapper when passing builders to `make()` for `NearList<T>` fields:
/// ```ignore
/// Block::make(0, list([10u32, 20, 30]))
/// ```
pub fn list<T: Flat>(
  iter: impl IntoIterator<IntoIter: ExactSizeIterator, Item: Emit<T>>,
) -> impl Emit<NearList<T>> {
  struct W<I>(I);
  // SAFETY: Allocates a segment with space for all elements, emits each element,
  // then patches the NearList header at `at`.
  unsafe impl<T: Flat, I> Emit<NearList<T>> for W<I>
  where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: Emit<T>,
  {
    unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
      let mut iter = self.0.into_iter();
      let len = iter.len() as u32;
      if len == 0 {
        // SAFETY: `at` was allocated for `NearList<T>` by the caller.
        unsafe { p.patch_list_header::<T>(at, Pos::ZERO, 0) };
      } else {
        let seg_pos = p.alloc_segment::<T>(len);
        let values_offset = size_of::<Segment<T>>();
        for i in 0..len as usize {
          let item = iter.next().expect("ExactSizeIterator lied");
          // SAFETY: `seg_pos` was allocated for `len` elements. Each
          // element position is within the segment allocation.
          unsafe {
            item.write_at(p, seg_pos.offset(values_offset + i * size_of::<T>()));
          }
        }
        // SAFETY: `at` was allocated for `NearList<T>` and `seg_pos`
        // for the segment.
        unsafe { p.patch_list_header::<T>(at, seg_pos, len) };
      }
    }
  }
  W(iter)
}

/// Construct an `Option<Near<T>>` from an optional `Emit<T>` builder.
///
/// Used as a wrapper when passing builders to `make()` for `Option<Near<T>>` fields:
/// ```ignore
/// Foo::make(maybe(Some(42u32)))
/// ```
pub fn maybe<T: Flat>(opt: Option<impl Emit<T>>) -> impl Emit<Option<Near<T>>> {
  struct W<B>(Option<B>);
  // SAFETY: If Some, allocates space for T, emits the builder, and patches the
  // Near<T> offset. If None, writes zero (the None niche for NonZero<i32>).
  unsafe impl<T: Flat, B: Emit<T>> Emit<Option<Near<T>>> for W<B> {
    unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
      match self.0 {
        Some(builder) => {
          let target = Emit::<T>::emit(builder, p);
          // SAFETY: `at` was allocated for `Option<Near<T>>` and `target`
          // for `T` by `emit`.
          unsafe { p.patch_near::<T>(at, target) };
        }
        // SAFETY: `at` was allocated for `Option<Near<T>>`. Writing zero
        // sets the None niche (NonZero<i32> uses zero as None).
        None => unsafe { p.write_flat::<i32>(at, 0) },
      }
    }
  }
  W(opt)
}

/// Construct a `None` value for `Option<Near<T>>`.
///
/// Avoids the turbofish needed with `maybe::<T>(None)`:
/// ```ignore
/// Foo::make(none::<Bar>())
/// ```
#[must_use]
pub fn none<T: Flat>() -> impl Emit<Option<Near<T>>> {
  struct W<T>(PhantomData<T>);
  // SAFETY: Writes zero at `at`, which is the None niche for Option<Near<T>>
  // (NonZero<i32> uses zero as the None discriminant).
  unsafe impl<T: Flat> Emit<Option<Near<T>>> for W<T> {
    unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
      // SAFETY: `at` was allocated for `Option<Near<T>>` by the caller.
      // Zero is the None niche for NonZero<i32>.
      unsafe { p.write_flat::<i32>(at, 0) };
    }
  }
  W(PhantomData)
}
