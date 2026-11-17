use core::{
  cmp::Ordering,
  fmt,
  hash::{Hash, Hasher},
  marker::{PhantomData, PhantomPinned},
  num::NonZero,
  ops::Deref,
  pin::Pin,
  ptr, slice, str,
};

use crate::{
  Emit, Flat,
  flat::{New, Seq},
};

/// A 4-byte pointer to a `T` in the same [`Region`](crate::Region), stored as the distance
/// from its own address. It dereferences like a `Box`.
///
/// A `Near` has no constructor and is neither `Clone`, `Copy` nor `Unpin`: it exists only
/// inside a region, as a field of a value there, and a pinned value keeps its pointers.
/// `Near<[T]>` and `Near<str>` point at a `u32` length followed by the elements.
/// `Option<Near<T>>` is also 4 bytes.
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
pub struct Near<T: ?Sized> {
  pub(crate) off: NonZero<i32>,
  _t: PhantomData<T>,
  _pin: PhantomPinned,
}

impl<T: ?Sized> Near<T> {
  /// The address of the target, derived from `self` so that it keeps the provenance of the
  /// region.
  const fn target(&self) -> *const u8 {
    ptr::from_ref(self).cast::<u8>().wrapping_offset(self.off.get() as isize)
  }

  /// The same, for writing.
  const fn target_mut(&mut self) -> *mut u8 {
    ptr::from_mut(self).cast::<u8>().wrapping_offset(self.off.get() as isize)
  }

  /// A list's length and elements.
  ///
  /// # Safety
  /// `self` is a `Near<[_]>` or a `Near<str>` in a region.
  const unsafe fn list<E>(&self) -> &[E] {
    let p = self.target();
    // SAFETY: the target is a `u32` length followed by the elements.
    unsafe {
      slice::from_raw_parts(p.add(4).cast::<E>(), p.cast::<u32>().read_unaligned() as usize)
    }
  }

  /// The same, for writing.
  ///
  /// # Safety
  /// As for [`list`](Self::list); the elements are reached only through the result.
  const unsafe fn list_mut<E>(&mut self) -> &mut [E] {
    let p = self.target_mut();
    // SAFETY: as for `list`.
    unsafe {
      slice::from_raw_parts_mut(p.add(4).cast::<E>(), p.cast::<u32>().read_unaligned() as usize)
    }
  }
}

impl<T: Flat> Near<T> {
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

impl<T: Flat> Near<[T]> {
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

impl<T> Deref for Near<T> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a `&Near` only exists inside a region, which holds the target at the offset.
    unsafe { &*self.target().cast::<T>() }
  }
}

impl<T> Deref for Near<[T]> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    // SAFETY: as above; the target is a list.
    unsafe { self.list() }
  }
}

impl Deref for Near<str> {
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
impl<'a, T> IntoIterator for &'a Near<[T]> {
  type Item = &'a T;
  type IntoIter = slice::Iter<'a, T>;
  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Near<T>
where
  Self: Deref<Target = T>,
{
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

impl<T: ?Sized + PartialEq> PartialEq for Near<T>
where
  Self: Deref<Target = T>,
{
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: ?Sized + Eq> Eq for Near<T> where Self: Deref<Target = T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Near<T>
where
  Self: Deref<Target = T>,
{
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    (**self).partial_cmp(&**other)
  }
}

impl<T: ?Sized + Ord> Ord for Near<T>
where
  Self: Deref<Target = T>,
{
  fn cmp(&self, other: &Self) -> Ordering {
    (**self).cmp(&**other)
  }
}

impl<T: ?Sized + Hash> Hash for Near<T>
where
  Self: Deref<Target = T>,
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    (**self).hash(state);
  }
}
