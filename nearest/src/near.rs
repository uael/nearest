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

/// The integer in which a [`Near`] keeps its distance: `i32`, the default, or `i16` or `i64`,
/// or a type of yours.
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
/// # Implementing it
///
/// `i16`, `i32` and `i64` hold native-endian integers, so the bytes of a region are read back
/// on targets of the same byte order. An `Offset` of yours decides how its bytes encode the
/// integer: the same in every region that uses it, for example little-endian on every target.
/// The bytes of a region hold no other integer of the region's own than its offsets and
/// lengths; the byte order of the stored values is that of their types.
///
/// ```
/// use std::num::NonZero;
///
/// use nearest::{Near, Offset, Region, flat};
///
/// /// An `i32` stored little-endian, whatever the target.
/// #[flat]
/// #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// #[repr(transparent)]
/// struct I32Le(i32);
///
/// /// An `I32Le` that is not zero.
/// #[derive(Clone, Copy)]
/// #[repr(transparent)]
/// struct NonZeroI32Le(NonZero<i32>);
///
/// impl From<NonZeroI32Le> for I32Le {
///   fn from(n: NonZeroI32Le) -> Self {
///     I32Le(n.0.get())
///   }
/// }
///
/// // SAFETY: `I32Le` is an `i32` whose bytes are those of the little-endian integer, and
/// // `NonZeroI32Le` the same bytes when not zero; swapping bytes keeps zero zero, so the
/// // integer 0 is the zero bytes, which are `None` for `Option<NonZero<i32>>`.
/// unsafe impl Offset for I32Le {
///   type NonZero = NonZeroI32Le;
///   fn from_isize(n: isize) -> Self {
///     I32Le((n as i32).to_le())
///   }
///   fn to_isize(self) -> isize {
///     i32::from_le(self.0) as isize
///   }
/// }
///
/// #[flat]
/// struct Table { name: Near<str, I32Le>, rows: Near<[u16], I32Le> }
///
/// let r = Region::new(Table::new("ab", [1, 2]));
/// assert_eq!((&*r.name, &r.rows[..]), ("ab", &[1, 2][..]));
/// // The length of the name, then its bytes: little-endian on every target.
/// let bytes = r.as_bytes().unwrap();
/// assert!(bytes.windows(6).any(|w| w == [2, 0, 0, 0, b'a', b'b']));
/// ```
///
/// # Safety
///
/// A region reads and writes the distances and lengths of an `O` only through `from_isize` and
/// `to_isize`, and reads stored ones as values of `Self` and [`NonZero`](Offset::NonZero) made of
/// bytes it checked or wrote. It trusts that:
///
/// - **Every bit pattern** of `size_of::<Self>()` bytes is a valid `Self`, and `Self` has no
///   padding.
/// - **Round trip.** `from_isize(n).to_isize() == n` for every `n` from `-(LIMIT as isize)` to
///   `LIMIT as isize`.
/// - **Zero.** `from_isize(0)` is all zero bytes, and it is the only `Self` whose `to_isize()` is
///   0.
/// - **`NonZero`** is `Self` without zero: its valid values are the bit patterns of `Self` other
///   than all zero bytes, it has the size and the alignment of `Self`, `Option<Self::NonZero>`
///   has the size of `Self` with `None` all zero bytes (as for a `#[repr(transparent)]` wrapper
///   of a [`core::num::NonZero`]), and `Into<Self>` keeps the bytes.
/// - **Plain functions.** `from_isize` and `to_isize` depend only on their argument, never panic
///   for the values above, and do nothing else.
///
/// What can be checked when compiling is checked where a region uses `O`: `Self` is 2, 4 or 8
/// bytes, `NonZero` and `Option<NonZero>` have its size, `NonZero` has its alignment, and
/// [`LIMIT`](Offset::LIMIT) is a multiple of 8 that the integers of `Self` reach.
pub unsafe trait Offset: Flat + Copy + fmt::Debug + Eq + Ord + Hash {
  /// The stored form: a `Self` that is not zero.
  type NonZero: Copy + Into<Self> + Send + Sync + UnwindSafe + RefUnwindSafe;
  /// The largest region whose distances and lengths fit, a multiple of 8, whatever the target.
  /// By default, the largest multiple of 8 that a signed integer of the size of `Self` holds.
  const LIMIT: u64 = (1 << (8 * size_of::<Self>() - 1)) - 8;
  /// The `Self` of `n`, which is within `LIMIT` of 0.
  fn from_isize(n: isize) -> Self;
  /// The integer of `self`.
  fn to_isize(self) -> isize;
}

/// Rejects, at compile time, an `O` whose layout a region cannot use (see the `# Safety` of
/// [`Offset`]). Every function that reads or writes the bytes of an `O` calls it.
pub const fn offset<O: Offset>() {
  const {
    let w = size_of::<O>();
    assert!(w == 2 || w == 4 || w == 8, "an `Offset` is 2, 4 or 8 bytes");
    assert!(
      size_of::<O::NonZero>() == w && align_of::<O::NonZero>() == align_of::<O>(),
      "the `NonZero` of an `Offset` has its size and its alignment"
    );
    assert!(
      size_of::<Option<O::NonZero>>() == w,
      "`Option` of the `NonZero` of an `Offset` has the size of the `Offset`"
    );
    assert!(
      O::LIMIT % 8 == 0 && O::LIMIT > 0 && O::LIMIT <= (1 << (8 * w - 1)) - 8,
      "the `LIMIT` of an `Offset` is a multiple of 8 that its integers reach"
    );
  };
}

/// The `O` whose bytes start `bytes`, if there are enough.
pub fn read<O: Offset>(bytes: &[u8]) -> Option<O> {
  offset::<O>();
  let bytes = bytes.get(..size_of::<O>())?;
  // SAFETY: `size_of::<O>()` initialized bytes, any of which make an `O` (`Offset`'s contract).
  Some(unsafe { bytes.as_ptr().cast::<O>().read_unaligned() })
}

macro_rules! offsets {
  ($($t:ty),*) => {$(
    // SAFETY: an integer: any bit pattern, `as` round-trips within `MAX`, zero is the zero
    // bytes, and `NonZero` is the integer without zero, `None` being zero.
    // Each converts one way that may truncate: never a distance or a length of a region.
    #[expect(clippy::cast_possible_truncation, reason = "values within a region fit both")]
    unsafe impl Offset for $t {
      type NonZero = NonZero<$t>;
      #[inline]
      fn from_isize(n: isize) -> Self {
        n as Self
      }
      #[inline]
      fn to_isize(self) -> isize {
        self as isize
      }
    }
  )*};
}
offsets!(i16, i32, i64);

impl<T: ?Sized, O: Offset> Near<T, O> {
  /// The address of the target, derived from `self` so that it keeps the provenance of the
  /// region.
  fn target(&self) -> *const u8 {
    offset::<O>();
    let off: O = self.off.into();
    ptr::from_ref(self).cast::<u8>().wrapping_offset(off.to_isize())
  }

  /// The same, for writing.
  fn target_mut(&mut self) -> *mut u8 {
    offset::<O>();
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
