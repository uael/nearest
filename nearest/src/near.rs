use core::{
  cmp::Ordering,
  fmt,
  hash::{Hash, Hasher},
  marker::{PhantomData, PhantomPinned},
  num::NonZero,
  ops::Deref,
  panic::{RefUnwindSafe, UnwindSafe},
  pin::Pin,
  ptr, slice, str,
};

use crate::{
  Emit, Flat,
  flat::{New, Seq},
};

/// A pointer to a `T` in the same [`Region`](crate::Region), stored as the distance from its
/// own address in an [`Offset`] `O`: 4 bytes (`i32`) unless `O` says otherwise. It
/// dereferences like a `Box`.
///
/// A `Near` has no constructor and is neither `Clone`, `Copy` nor `Unpin`: it exists only
/// inside a region, as a field of a value there, and a pinned value keeps its pointers.
/// `Near<[T]>` and `Near<str>` point at a length, an `O`, followed by the elements.
/// `Option<Near<T>>` is the size of a `Near<T>`.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Pair { name: Near<str>, values: Near<[u32]>, next: Option<Near<Pair>> }
///
/// let r = Region::new(Pair::new("a", [1, 2, 3], Near::new(Pair::new("b", (), None))));
/// assert_eq!(&*r.name, "a");
/// assert_eq!(r.values.iter().sum::<u32>(), 6);
/// assert_eq!(&*r.next.as_ref().unwrap().name, "b");
/// assert_eq!(size_of::<Option<Near<Pair>>>(), 4);
/// ```
#[repr(transparent)]
pub struct Near<T: ?Sized, O: Offset = i32> {
  pub(crate) off: O::NonZero,
  _t: PhantomData<T>,
  _pin: PhantomPinned,
}

/// The integer in which a [`Near`] keeps its distance: `i32`, the default, or `i16` or `i64`.
/// Implemented for these three only.
///
/// | `O` | `Near` | A region holds at most |
/// |---|---|---|
/// | `i16` | 2 bytes | 32 KiB |
/// | `i32` | 4 bytes | 2 GiB |
/// | `i64` | 8 bytes | what the address space holds (`isize::MAX` bytes) |
///
/// List and string lengths are `O`s too. The pointers of the root of a region bound it, so
/// pointers may only get wider going down: a `Near<T, O>` to a `T` that holds a narrower
/// pointer does not build. Types can take the width as a parameter.
///
/// ```
/// use nearest::{Near, Offset, Region, flat};
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// struct Tree<O: Offset> { id: u32, kids: Near<[Tree<O>], O> }
///
/// let small = Region::new(Tree::<i16>::new(1, [Tree::new(2, ())]));
/// let big = Region::new(Tree::<i64>::new(1, [Tree::new(2, ())]));
/// assert_eq!((small.kids[0].id, big.kids[0].id), (2, 2));
/// assert_eq!((size_of::<Tree<i16>>(), size_of::<Tree<i64>>()), (8, 16));
/// ```
pub trait Offset: Flat + Copy + fmt::Debug + Eq + Ord + Hash + Sealed {
  /// The stored form, `NonZero<Self>`.
  #[doc(hidden)]
  type NonZero: Copy + Into<Self> + Send + Sync + UnwindSafe + RefUnwindSafe;
  /// The largest region whose distances fit, a multiple of 8, whatever the target.
  #[doc(hidden)]
  const LIMIT: u64;
  /// `n`, which fits.
  #[doc(hidden)]
  fn from_isize(n: isize) -> Self;
  #[doc(hidden)]
  fn to_isize(self) -> isize;
  /// The first bytes, if there are enough.
  #[doc(hidden)]
  fn read(bytes: &[u8]) -> Option<Self>;
}

/// Keeps [`Offset`] to the three integers.
pub trait Sealed {}

macro_rules! offsets {
  ($($t:ty),*) => {$(
    impl Sealed for $t {}
    // Each converts one way that may truncate: never a distance or a length of a region.
    #[expect(clippy::cast_possible_truncation, reason = "values within a region fit both")]
    impl Offset for $t {
      type NonZero = NonZero<$t>;
      const LIMIT: u64 = <$t>::MAX as u64 & !7;
      #[inline]
      fn from_isize(n: isize) -> Self {
        n as Self
      }
      #[inline]
      fn to_isize(self) -> isize {
        self as isize
      }
      #[inline]
      fn read(bytes: &[u8]) -> Option<Self> {
        bytes.first_chunk().copied().map(Self::from_ne_bytes)
      }
    }
  )*};
}
offsets!(i16, i32, i64);

impl<T: ?Sized, O: Offset> Near<T, O> {
  /// The address of the target, derived from `self` so that it keeps the provenance of the
  /// region.
  fn target(&self) -> *const u8 {
    let off: O = self.off.into();
    ptr::from_ref(self).cast::<u8>().wrapping_offset(off.to_isize())
  }

  /// The same, for writing.
  fn target_mut(&mut self) -> *mut u8 {
    let off: O = self.off.into();
    ptr::from_mut(self).cast::<u8>().wrapping_offset(off.to_isize())
  }

  /// A list's length and elements.
  ///
  /// # Safety
  /// `self` is a `Near<[_], O>` or a `Near<str, O>` in a region.
  unsafe fn list<E>(&self) -> &[E] {
    let p = self.target();
    // SAFETY: the target is a length, an `O`, followed by the elements.
    unsafe {
      let len = p.cast::<O>().read_unaligned().to_isize().cast_unsigned();
      slice::from_raw_parts(p.add(size_of::<O>()).cast::<E>(), len)
    }
  }

  /// The same, for writing.
  ///
  /// # Safety
  /// As for [`list`](Self::list); the elements are reached only through the result.
  unsafe fn list_mut<E>(&mut self) -> &mut [E] {
    let p = self.target_mut();
    // SAFETY: as for `list`.
    unsafe {
      let len = p.cast::<O>().read_unaligned().to_isize().cast_unsigned();
      slice::from_raw_parts_mut(p.add(size_of::<O>()).cast::<E>(), len)
    }
  }
}

impl<T: Flat, O: Offset> Near<T, O> {
  /// Builder of a pointer to a new `T` written from `value`; also accepted where an
  /// `Option<Near<T>>` is expected.
  ///
  /// Constructor parameters for fields written `Near<X>` already take a builder of `X`; this
  /// is for the other places: optional pointers, lists of pointers, and helper functions.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Boxed { inner: Near<u64>, maybe: Option<Near<u64>>, many: Near<[Near<u64>]> }
  ///
  /// let r = Region::new(Boxed::new(7, Near::new(8), [Near::new(9)]));
  /// assert_eq!((*r.inner, r.maybe.as_deref(), *r.many[0]), (7, Some(&8), 9));
  /// ```
  pub fn new(value: impl Emit<T>) -> impl Emit<Self> + Emit<Option<Self>> {
    New(value)
  }

  /// The target, pinned: the way down in a pinned view (see `pin` on handles).
  ///
  /// ```
  /// use core::pin::Pin;
  ///
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Pair { left: Near<u32>, right: Near<u32> }
  ///
  /// let mut r = Region::new(Pair::new(1, 2));
  /// // Both fields at once.
  /// let PairPin { left, right } = r.edit().pin().project();
  /// let mut left: Pin<&mut u32> = left.as_pin_mut();
  /// let right: Pin<&mut u32> = right.as_pin_mut();
  /// *left += *right;
  /// assert_eq!(*r.left, 3);
  /// ```
  #[must_use]
  pub fn as_pin_mut(self: Pin<&mut Self>) -> Pin<&mut T> {
    #[cfg(feature = "nightly")]
    crate::unpin::pinned::<T>();
    // SAFETY: the target is not moved while the pointer is pinned: a pinned view borrows the
    // handle it comes from, so nothing can grow the buffer; and a `Near` owns its target alone.
    unsafe { Pin::new_unchecked(&mut *Pin::get_unchecked_mut(self).target_mut().cast::<T>()) }
  }
}

impl<T: Flat, O: Offset> Near<[T], O> {
  /// Builder of a list of the items of an iterator.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Squares { values: Near<[u32]> }
  ///
  /// let r = Region::new(Squares::new(Near::from_iter((1..4).map(|i| i * i))));
  /// assert_eq!(r.values[..], [1, 4, 9]);
  /// ```
  #[expect(clippy::should_implement_trait, reason = "a builder, not a `Near<[T]>`")]
  pub fn from_iter<I: IntoIterator<Item: Emit<T>>>(items: I) -> impl Emit<Self> {
    Seq(items)
  }

  /// The elements, pinned, all at once.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { id: u32, kids: Near<[Node]> }
  ///
  /// let mut r = Region::new(Node::new(0, [Node::new(1, ()), Node::new(2, ())]));
  /// let NodePin { mut id, kids } = r.edit().pin().project();
  /// for kid in kids.iter_pin_mut() {
  ///   *kid.project().id += 10;
  ///   *id += 1;
  /// }
  /// assert_eq!((r.id, r.kids[0].id, r.kids[1].id), (2, 11, 12));
  /// ```
  #[must_use = "the iterator does nothing unless it is used"]
  pub fn iter_pin_mut(self: Pin<&mut Self>) -> impl ExactSizeIterator<Item = Pin<&mut T>> {
    #[cfg(feature = "nightly")]
    crate::unpin::pinned::<T>();
    // SAFETY: as for `Near::as_pin_mut`; the elements are disjoint and stay where they are.
    unsafe {
      (Pin::get_unchecked_mut(self).list_mut::<T>().iter_mut())
        .map(|element| Pin::new_unchecked(element))
    }
  }
}

impl<T, O: Offset> Deref for Near<T, O> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a `&Near` only exists inside a region, which holds the target at the offset.
    unsafe { &*self.target().cast::<T>() }
  }
}

impl<T, O: Offset> Deref for Near<[T], O> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    // SAFETY: as above; the target is a list.
    unsafe { self.list() }
  }
}

impl<O: Offset> Deref for Near<str, O> {
  type Target = str;
  fn deref(&self) -> &str {
    // SAFETY: as above; the target is a length and UTF-8 bytes.
    unsafe { str::from_utf8_unchecked(self.list()) }
  }
}

/// The elements.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct List { items: Near<[u8]> }
///
/// let r = Region::new(List::new(b"abc".as_slice()));
/// let mut n = 0;
/// for x in &r.items { n += u32::from(*x) }
/// assert_eq!(n, 294);
/// ```
impl<'a, T, O: Offset> IntoIterator for &'a Near<[T], O> {
  type Item = &'a T;
  type IntoIter = slice::Iter<'a, T>;
  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<T: ?Sized + fmt::Debug, O: Offset> fmt::Debug for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

/// Shows the target, as the target does: a string or a number.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Greeting { to: Near<str>, times: Near<u8> }
///
/// let r = Region::new(Greeting::new("world", 3));
/// assert_eq!(format!("hello {} x{}", r.to, r.times), "hello world x3");
/// ```
impl<T: ?Sized + fmt::Display, O: Offset> fmt::Display for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Display::fmt(&**self, f)
  }
}

impl<T: ?Sized + PartialEq, O: Offset> PartialEq for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: ?Sized + Eq, O: Offset> Eq for Near<T, O> where Self: Deref<Target = T> {}

impl<T: ?Sized + PartialOrd, O: Offset> PartialOrd for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    (**self).partial_cmp(&**other)
  }
}

impl<T: ?Sized + Ord, O: Offset> Ord for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn cmp(&self, other: &Self) -> Ordering {
    (**self).cmp(&**other)
  }
}

impl<T: ?Sized + Hash, O: Offset> Hash for Near<T, O>
where
  Self: Deref<Target = T>,
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    (**self).hash(state);
  }
}
