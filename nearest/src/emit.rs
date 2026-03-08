use core::marker::PhantomData;

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

// // SAFETY: `Infallible` is uninhabited — `write_at` is unreachable.
// unsafe impl<T: Flat> Emit<T> for Infallible {
//   unsafe fn write_at(self, _p: &mut impl Patch, _at: Pos) {
//     match self {}
//   }
// }

// --- Blanket deep-copy impl: Emit<T> for &T via Flat::deep_copy ---

// SAFETY: Delegates to `Flat::deep_copy` which correctly copies all fields
// and patches self-relative pointers.
unsafe impl<T: Flat> Emit<T> for T {
  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    // SAFETY: caller guarantees `at` was allocated for `T`.
    unsafe { self.deep_copy(p, at) };
  }
}

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

/// Returns an empty iterator suitable for any `NearList<T>` emitter parameter.
///
/// Since [`Infallible`](core::convert::Infallible) implements [`Emit<T>`] for all
/// `T: Flat`, an `Empty<Infallible>` satisfies any `IntoIterator<Item: Emit<T>>` bound.
///
/// # Examples
///
/// ```
/// use nearest::{Flat, NearList, Region, empty};
///
/// #[derive(Flat)]
/// struct Root { items: NearList<u32> }
///
/// // Use empty() when building a struct with an empty NearList.
/// let region = Region::new(Root::make(empty()));
/// assert!(region.items.is_empty());
/// ```
#[must_use]
pub fn empty<T: Flat>() -> impl Emit<NearList<T>> {
  list(core::iter::empty::<T>())
}
