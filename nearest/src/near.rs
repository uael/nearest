use core::{
  cmp::Ordering,
  fmt,
  hash::{Hash, Hasher},
  iter::Map,
  marker::{PhantomData, PhantomPinned},
  mem,
  num::NonZero,
  ops::Deref,
  pin::Pin,
  ptr, slice,
};

use crate::{
  Emit, Flat,
  flat::{New, Seq, is_list},
};

/// A 4-byte pointer to a `T` in the same [`Region`](crate::Region), stored as the distance
/// from its own address.
///
/// A `Near` has no public constructor and is neither `Clone` nor `Unpin`: it exists only inside
/// a region, and values containing one are reached through `&T` or `Pin<&mut T>`.
/// `Near<[T]>` and `Near<str>` point at a `u32` length followed by the elements.
/// `Option<Near<T>>` is also 4 bytes.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
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
  /// The target, derived from `this` so that it keeps the provenance of the region.
  ///
  /// # Safety
  /// `this` points at a `Near` inside a region.
  const unsafe fn target(this: *const Self) -> *const T {
    // SAFETY: the region holds the target (and, for lists, its length header) at the offset.
    unsafe {
      let p = this.cast::<u8>().offset((*this).off.get() as isize);
      if is_list::<T>() {
        let len = p.cast::<u32>().read_unaligned() as usize;
        ptr::from_raw_parts(p.add(4), mem::transmute_copy(&len))
      } else {
        ptr::from_raw_parts(p, mem::zeroed())
      }
    }
  }

  /// The target, pinned, for edits in place.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
  /// r.edit(|root, _| {
  ///   let Node { next, .. } = root;
  ///   if let Some(next) = next.as_pin_mut().map(Near::as_pin_mut) {
  ///     let Node { value, .. } = next;
  ///     *value.get_mut() = 20;
  ///   }
  /// });
  /// assert_eq!(r.next.as_ref().unwrap().value, 20);
  /// ```
  #[must_use]
  pub fn as_pin_mut(self: Pin<&mut Self>) -> Pin<&mut T> {
    // SAFETY: a region is a tree, so this `Near` is the only way to the target, and the target
    // stays pinned like everything else in the region.
    unsafe { Pin::new_unchecked(&mut *Self::target(Pin::get_unchecked_mut(self)).cast_mut()) }
  }
}

impl<T: Flat> Near<T> {
  /// Builder for a pointer to a new `T`; also accepted where an `Option<Near<T>>` is expected.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Boxed { inner: Near<u64>, maybe: Option<Near<u64>> }
  ///
  /// let r = Region::new(Boxed::new(Near::new(7), Near::new(8)));
  /// assert_eq!((*r.inner, r.maybe.as_deref()), (7, Some(&8)));
  /// ```
  pub fn new<'o>(value: impl Emit<'o, T>) -> impl Emit<'o, Self> + Emit<'o, Option<Self>> {
    New(value)
  }
}

impl<T: Flat> Near<[T]> {
  /// Builder for a list of the items of an iterator.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Squares { values: Near<[u32]> }
  ///
  /// let r = Region::new(Squares::new(Near::from_iter((1..4).map(|i| i * i))));
  /// assert_eq!(r.values[..], [1, 4, 9]);
  /// ```
  #[expect(clippy::should_implement_trait, reason = "a builder, not a `Near<[T]>`")]
  pub fn from_iter<'o, I: IntoIterator<Item: Emit<'o, T>>>(items: I) -> impl Emit<'o, Self> {
    Seq(items)
  }
}

impl<T: ?Sized> Deref for Near<T> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a `&Near` is only ever obtained from inside a region.
    unsafe { &*Self::target(self) }
  }
}

/// The elements, each pinned.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Item { n: u32 }
/// #[flat]
/// struct List { items: Near<[Item]> }
///
/// let mut r = Region::new(List::new([Item { n: 1 }, Item { n: 2 }]));
/// r.edit(|root, _| {
///   let List { items } = root;
///   for item in items {
///     let Item { n } = item;
///     *n.get_mut() *= 10;
///   }
/// });
/// assert_eq!(r.items.iter().map(|i| i.n).collect::<Vec<_>>(), [10, 20]);
/// ```
impl<'a, T> IntoIterator for Pin<&'a mut Near<[T]>> {
  type Item = Pin<&'a mut T>;
  type IntoIter = Map<slice::IterMut<'a, T>, fn(&'a mut T) -> Pin<&'a mut T>>;
  fn into_iter(self) -> Self::IntoIter {
    // SAFETY: the elements are pinned because the list is.
    let pin = |x| unsafe { Pin::new_unchecked(x) };
    // SAFETY: nothing is moved out; each element is handed out pinned.
    unsafe { Pin::get_unchecked_mut(self.as_pin_mut()) }.iter_mut().map(pin)
  }
}

/// The elements.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
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

impl<T: ?Sized + fmt::Debug> fmt::Debug for Near<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

impl<T: ?Sized + PartialEq> PartialEq for Near<T> {
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: ?Sized + Eq> Eq for Near<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Near<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    (**self).partial_cmp(&**other)
  }
}

impl<T: ?Sized + Ord> Ord for Near<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    (**self).cmp(&**other)
  }
}

impl<T: ?Sized + Hash> Hash for Near<T> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    (**self).hash(state);
  }
}
