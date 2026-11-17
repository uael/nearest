use alloc::vec::Vec;
use core::{fmt, marker::PhantomData, ops::Deref, ptr, slice};

use crate::{
  Edit, Emit, Flat,
  flat::{Buf, Sink, fits},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// struct Tree { value: u32, children: Near<[Tree]> }
///
/// let r = Region::new(Tree::new(1, [Tree::new(2, ()), Tree::new(3, ())]));
/// let copy = r.clone();
/// assert_eq!(copy.children.iter().map(|c| c.value).sum::<u32>(), 5);
/// assert_eq!(copy, r);
/// ```
pub struct Region<T> {
  buf: Buf,
  _t: PhantomData<T>,
}

impl<T: Flat> Region<T> {
  /// A new region holding `value`: a value, a copy (`&T`, from any region) or a builder.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }
  ///
  /// let r = Region::new(Expr::add(Expr::Lit(1), Expr::Lit(2)));
  /// assert!(matches!(&*r, Expr::Add(a, _) if matches!(**a, Expr::Lit(1))));
  /// // A copy into a new region keeps only what is reachable.
  /// let compact = Region::new(&*r);
  /// assert_eq!(compact.byte_len(), r.byte_len());
  /// ```
  pub fn new(value: impl Emit<T>) -> Self {
    fits::<T>();
    let mut buf = Buf::new();
    // SAFETY: `buf` is live during the call and only reached through the sink.
    let mut sink = unsafe { Sink::new(&raw mut buf) };
    let at = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `at` was just allocated for a `T`.
    unsafe { value.emit(&mut sink, at) };
    Self { buf, _t: PhantomData }
  }

  /// The handle of the root, for edits in place.
  ///
  /// Every field has a method of the same name that returns its handle; plain data is a real
  /// `&mut`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Block { id: u32, insts: Near<[u8]>, name: Near<str> }
  ///
  /// let mut r = Region::new(Block::new(0, [1, 2, 3], "b0"));
  /// let b = r.edit();
  /// *b.id() = 7;
  /// b.insts().retain(|x| *x != 2);
  /// b.insts().push(4);
  /// b.name().set("entry");
  /// assert_eq!((r.id, &r.insts[..], &*r.name), (7, &[1, 3, 4][..], "entry"));
  /// ```
  pub fn edit(&mut self) -> &mut Edit<T> {
    fits::<T>();
    // SAFETY: the root is a `T` at byte 0, and the handle borrows the region.
    unsafe { T::edit(&raw mut self.buf, 0) }
  }

  /// A compact, fully initialized image of the region, for [`from_bytes`](Self::from_bytes).
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// let bytes = r.to_bytes();
  /// // SAFETY: the bytes come from `to_bytes` of a `Region<Msg>` in this program.
  /// let back = unsafe { Region::<Msg>::from_bytes(&bytes) };
  /// assert_eq!(&*back.body, "hi");
  /// ```
  #[must_use]
  pub fn to_bytes(&self) -> Vec<u8> {
    let copy = Self::new(&**self);
    // SAFETY: the bytes of a new region are initialized: zero-filled, then written field by
    // field.
    unsafe { slice::from_raw_parts(copy.buf.base(), copy.buf.len()).to_vec() }
  }

  /// The region whose image is `bytes`.
  ///
  /// ```
  /// use nearest::Region;
  ///
  /// let bytes = Region::new(5u64).to_bytes();
  /// // SAFETY: the bytes come from `to_bytes` of a `Region<u64>` in this program.
  /// assert_eq!(*unsafe { Region::<u64>::from_bytes(&bytes) }, 5);
  /// ```
  ///
  /// # Safety
  /// `bytes` came from [`to_bytes`](Self::to_bytes) of a `Region<T>`, for the same `T` on a
  /// target with the same layout.
  #[must_use]
  pub unsafe fn from_bytes(bytes: &[u8]) -> Self {
    let mut buf = Buf::new();
    let at = buf.alloc(bytes.len(), 8);
    // SAFETY: `bytes.len()` bytes were allocated at `at`.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr().add(at), bytes.len()) };
    Self { buf, _t: PhantomData }
  }

  /// Bytes in use, including those no longer reachable after edits.
  ///
  /// ```
  /// use nearest::Region;
  ///
  /// assert_eq!(Region::new(1u32).byte_len(), 4);
  /// ```
  #[must_use]
  pub const fn byte_len(&self) -> usize {
    self.buf.len()
  }
}

impl<T> Deref for Region<T> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a region holds a `T` at byte 0 of its 8-aligned buffer.
    unsafe { &*self.buf.base().cast() }
  }
}

impl<T> Clone for Region<T> {
  fn clone(&self) -> Self {
    Self { buf: self.buf.to_vec(), _t: PhantomData }
  }
}

impl<T: fmt::Debug> fmt::Debug for Region<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

impl<T: PartialEq> PartialEq for Region<T> {
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: Eq> Eq for Region<T> {}
