use alloc::borrow::Cow;
use core::{fmt, marker::PhantomData, ops::Deref, ptr};

use crate::{
  Edit, Emit, Flat, Invalid,
  bytes::check,
  flat::{Buf, Sink, fits},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place; [`to_bytes`](Self::to_bytes) and [`from_bytes`](Self::from_bytes) save
/// and load it.
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
  /// A new region holding `value`: a value, a copy (`&T`, from any region), an `Old` or a
  /// builder.
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

  /// The handle of the root, for edits in place. `'e`, the borrow of the region, is the edit:
  /// handles and moved values ([`Old`](crate::Old)) of the edit live at most that long.
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
  pub fn edit(&mut self) -> &mut Edit<'_, T> {
    fits::<T>();
    self.buf.compact = false;
    self.buf.me = &raw mut self.buf;
    // SAFETY: the root is a `T` at byte 0, the region is borrowed for the edit, and every
    // access of the edit goes through `me` (I8).
    unsafe { T::edit(self.buf.me, 0) }
  }

  /// The compact image of the region: borrowed if the region is compact (it was not edited
  /// since it was built or loaded), otherwise a compact copy.
  ///
  /// ```
  /// use std::borrow::Cow;
  ///
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let mut r = Region::new(Msg::new(1, "hi"));
  /// assert!(matches!(r.to_bytes(), Cow::Borrowed(_)));
  /// r.edit().body().set("hello");
  /// let bytes = r.to_bytes();
  /// assert!(matches!(bytes, Cow::Owned(_)) && bytes.len() < r.byte_len());
  /// assert_eq!(&*Region::<Msg>::from_bytes(&bytes)?.body, "hello");
  /// # Ok::<(), nearest::Invalid>(())
  /// ```
  #[must_use]
  pub fn to_bytes(&self) -> Cow<'_, [u8]> {
    if self.buf.compact {
      // SAFETY: a compact buffer is initialized.
      Cow::Borrowed(unsafe { self.buf.bytes() })
    } else {
      let copy = Self::new(&**self);
      // SAFETY: as above; a deep copy is compact.
      Cow::Owned(unsafe { copy.buf.bytes() }.to_vec())
    }
  }

  /// The region whose image is `bytes`, which may have any alignment. The bytes are checked,
  /// then copied: they must be exactly what [`to_bytes`](Self::to_bytes) writes for a `T`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let bytes = Region::new(Msg::new(1, "hi")).to_bytes().into_owned();
  /// let back = Region::<Msg>::from_bytes(&bytes)?;
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// assert!(Region::<Msg>::from_bytes(&bytes[1..]).is_err());
  /// # Ok::<(), nearest::Invalid>(())
  /// ```
  ///
  /// # Errors
  /// [`Invalid`] if the bytes are not the compact image of a valid `T`.
  pub fn from_bytes(bytes: &[u8]) -> Result<Self, Invalid> {
    // The checker reads bytes at any alignment, so nothing is copied for a bad image.
    check::<T>(bytes)?;
    // SAFETY: just checked.
    Ok(unsafe { Self::from_bytes_unchecked(bytes) })
  }

  /// [`from_bytes`](Self::from_bytes) without the check, for bytes that are trusted.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let bytes = Region::new(Msg::new(1, "hi")).to_bytes().into_owned();
  /// // SAFETY: the bytes come from `to_bytes` for the same type, in this build.
  /// let back = unsafe { Region::<Msg>::from_bytes_unchecked(&bytes) };
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// ```
  ///
  /// # Safety
  /// `bytes` are what [`to_bytes`](Self::to_bytes) returned for a `Region<T>`, in a build with
  /// the same layout of `T`.
  ///
  /// # Panics
  /// If `bytes` is longer than 2 GiB.
  #[must_use]
  pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self {
    fits::<T>();
    let mut buf = Buf::new();
    let at = buf.alloc(bytes.len(), 8);
    // SAFETY: `bytes.len()` bytes were allocated at `at`.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr().add(at), bytes.len()) };
    Self { buf, _t: PhantomData }
  }

  /// The value whose image is `bytes`, read in place, for example from a memory-mapped file.
  /// The slice must be 8-aligned; the bytes are checked as by [`from_bytes`](Self::from_bytes).
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// let bytes = r.to_bytes(); // the region's own buffer, which is 8-aligned
  /// let msg: &Msg = Region::view(&bytes)?;
  /// assert_eq!(&*msg.body, "hi");
  /// # Ok::<(), nearest::Invalid>(())
  /// ```
  ///
  /// # Errors
  /// [`Invalid`] if the slice is not 8-aligned or the bytes are not the compact image of a
  /// valid `T`.
  pub fn view(bytes: &[u8]) -> Result<&T, Invalid> {
    if !bytes.as_ptr().addr().is_multiple_of(8) {
      return Err(Invalid(0));
    }
    check::<T>(bytes)?;
    // SAFETY: just checked.
    Ok(unsafe { Self::view_unchecked(bytes) })
  }

  /// [`view`](Self::view) without the check, for bytes that are trusted: nothing is read
  /// before the value is used.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// let bytes = r.to_bytes(); // the region's own buffer, which is 8-aligned
  /// // SAFETY: the bytes come from `to_bytes` for the same type, in this build.
  /// let msg: &Msg = unsafe { Region::view_unchecked(&bytes) };
  /// assert_eq!(&*msg.body, "hi");
  /// ```
  ///
  /// # Safety
  /// `bytes` is 8-aligned and holds what [`to_bytes`](Self::to_bytes) returned for a
  /// `Region<T>`, in a build with the same layout of `T`.
  #[must_use]
  pub const unsafe fn view_unchecked(bytes: &[u8]) -> &T {
    fits::<T>();
    // SAFETY: per the caller, the bytes are 8-aligned and the image of a valid `T`, whose
    // pointers all aim inside them.
    unsafe { &*bytes.as_ptr().cast::<T>() }
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

// A region owns its value in a buffer of its own: moving the `Region` moves nothing in it.
impl<T> Unpin for Region<T> {}

const _: () = {
  const fn send_sync_unpin<T: Send + Sync + Unpin>() {}
  send_sync_unpin::<Region<u64>>();
  send_sync_unpin::<Region<crate::Near<[u64]>>>();
};
