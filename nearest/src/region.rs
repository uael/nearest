#[cfg(feature = "alloc")]
use alloc::{borrow::Cow, vec::Vec};
use core::{fmt, marker::PhantomData, ops::Deref, slice};

#[cfg(feature = "alloc")]
use crate::Heap;
use crate::{
  Edit, Emit, Flat, Invalid, Pending, Storage,
  bytes::{Stack, check},
  flat::{Buf, Sink, fits},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place; [`to_bytes`](Self::to_bytes), [`as_bytes`](Self::as_bytes) and
/// [`from_bytes`](Self::from_bytes) save and load it.
///
/// `S` is the [`Storage`] of the buffer: by default the [`Heap`](crate::Heap), which grows as
/// needed up to 2 GiB; [`Inline<N>`](crate::Inline) keeps it in `N` bytes inside the `Region`
/// value, with no heap; and a storage of your own puts it anywhere else. A region in another
/// storage is built with [`new_in`](Self::new_in) or [`from_bytes`](Self::from_bytes);
/// everything else is the same for all.
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
pub struct Region<T, S = crate::Heap> {
  buf: Buf,
  store: S,
  _t: PhantomData<T>,
}

impl<T, S: Storage> Region<T, S> {
  /// An empty region in `store`, not attached: attach it once it is where it is used, since a
  /// storage may move with its region.
  const fn empty(store: S) -> Self {
    Self { buf: Buf::new(), store, _t: PhantomData }
  }

  /// Attaches the storage to the buffer, where it is now: a storage may move with its region.
  /// Every use of the buffer through `&mut self` starts here, after the region last moved.
  fn attach(&mut self) {
    // SAFETY: the storage holds the buffer's bytes (none when given), moves only with the
    // region, and is reached only through the buffer while the region is borrowed.
    unsafe { self.buf.attach(&raw mut self.store) };
  }

  /// Byte 0, for reading, from the storage where the region is now.
  fn base(&self) -> *const u8 {
    self.store.memory().as_ptr().cast()
  }

  /// A region in `store` holding a copy of the `len` bytes at `src`, or `None` if the storage
  /// cannot hold them.
  ///
  /// # Safety
  /// `src` is valid for reading `len` bytes, outside `store`.
  unsafe fn copy_of(store: S, src: *const u8, len: usize) -> Option<Self> {
    let mut r = Self::empty(store);
    r.attach();
    // SAFETY: the buffer was just attached and is empty; `src` is not in it.
    unsafe { r.buf.load(src, len) }.then_some(r)
  }
}

#[cfg(feature = "alloc")]
impl<T: Flat> Region<T> {
  /// A new region holding `value`, on the heap: a value, a copy (`&T`, from any region), an
  /// `Old` or a builder.
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
    Self::new_in(Heap::default(), value)
  }
}

impl<T: Flat, S: Storage> Region<T, S> {
  /// A new region holding `value`, in `storage`: [`new`](Region::new) for any storage, whose
  /// memory may hold anything (see [`Storage`]).
  ///
  /// ```
  /// use nearest::{Inline, Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// // 64 bytes inside the `Region` value.
  /// let mut r = Region::new_in(Inline::<64>::default(), Msg::new(1, "hi"));
  /// r.edit().body().set("hello");
  /// let moved = r; // the bytes move with it
  /// assert_eq!(&*moved.body, "hello");
  /// ```
  ///
  /// # Panics
  /// If the value does not fit the storage.
  pub fn new_in(storage: S, value: impl Emit<T>) -> Self {
    fits::<T>();
    let mut r = Self::empty(storage);
    r.attach();
    // SAFETY: the buffer is attached, live during the call and only reached through the sink.
    let mut sink = unsafe { Sink::new(&raw mut r.buf) };
    let at = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `at` was just allocated for a `T`.
    unsafe { value.emit(&mut sink, at) };
    r
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
    self.attach();
    self.buf.compact = false;
    self.buf.me = &raw mut self.buf;
    // SAFETY: the root is a `T` at byte 0, the buffer is attached, the region is borrowed for
    // the edit, and every access of the edit goes through `me` (I8).
    unsafe { T::edit(self.buf.me, 0) }
  }

  /// The compact image of the region, borrowed, if the region is compact: built by `new`,
  /// `new_in` or `from_bytes`, or cloned from a compact region, and not edited since. `None`
  /// otherwise; a copy, `Region::new_in(storage, &*region)`, is compact. It allocates nothing.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let mut r = Region::new(Msg::new(1, "hi"));
  /// let image = r.as_bytes().unwrap();
  /// assert_eq!(&*Region::<Msg>::view(image)?.body, "hi"); // the buffer is 8-aligned
  /// r.edit().body().set("hello");
  /// assert!(r.as_bytes().is_none());
  /// # Ok::<(), nearest::Invalid>(())
  /// ```
  #[must_use]
  pub fn as_bytes(&self) -> Option<&[u8]> {
    // SAFETY: the bytes in use are in the memory, and initialized since the buffer is compact.
    self.buf.compact.then(|| unsafe { slice::from_raw_parts(self.base(), self.buf.len()) })
  }

  /// The compact image of the region: borrowed if the region is compact (see
  /// [`as_bytes`](Self::as_bytes)), otherwise a compact copy.
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
  #[cfg(feature = "alloc")]
  #[must_use]
  pub fn to_bytes(&self) -> Cow<'_, [u8]> {
    self.as_bytes().map_or_else(
      // A copy is compact.
      || Cow::Owned(Region::<T>::new(&**self).as_bytes().map(<[u8]>::to_vec).unwrap_or_default()),
      Cow::Borrowed,
    )
  }

  /// The region whose image is `bytes`, which may have any alignment, in a new storage. The
  /// bytes are checked, then copied: they must be exactly what [`to_bytes`](Self::to_bytes)
  /// writes for a `T`. For a storage that has no `Default`, copy from a view instead:
  /// `Region::new_in(storage, Region::<T>::view(bytes)?)`. It needs `alloc`: checking takes
  /// memory that grows with the nesting of the data, which
  /// [`from_bytes_with`](Self::from_bytes_with) takes from the caller.
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
  /// [`Invalid`] if the bytes are not the compact image of a valid `T`, or if they are one
  /// larger than the storage gives (then [`Invalid::offset`] is `None`).
  #[cfg(feature = "alloc")]
  pub fn from_bytes(bytes: &[u8]) -> Result<Self, Invalid>
  where
    S: Default,
  {
    // A vector is never full.
    Self::load(bytes, Stack::Heap(Vec::new())).unwrap_or(Err(Invalid(0)))
  }

  /// [`from_bytes`](Self::from_bytes), checking with `scratch` for memory: `None` if it ran out
  /// before the check finished, which says nothing about the bytes; otherwise the result of
  /// `from_bytes`. `bytes.len() / 4` entries are always enough (see [`Pending`]). It needs no
  /// `alloc`.
  ///
  /// ```
  /// use nearest::{Inline, Near, Pending, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// type Small = Region<Msg, Inline<64>>;
  ///
  /// let r = Small::new_in(Inline::default(), Msg::new(1, "hi"));
  /// let bytes = r.as_bytes().unwrap();
  /// let mut scratch = [Pending::default(); 64 / 4]; // enough for anything `Inline<64>` holds
  /// let back = Small::from_bytes_with(bytes, &mut scratch).unwrap()?;
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// assert!(Small::from_bytes_with(&bytes[1..], &mut scratch).unwrap().is_err());
  /// # Ok::<(), nearest::Invalid>(())
  /// ```
  pub fn from_bytes_with(bytes: &[u8], scratch: &mut [Pending]) -> Option<Result<Self, Invalid>>
  where
    S: Default,
  {
    Self::load(bytes, Stack::Scratch(scratch, 0))
  }

  /// `from_bytes`, checking with `todo`.
  fn load(bytes: &[u8], todo: Stack<'_>) -> Option<Result<Self, Invalid>>
  where
    S: Default,
  {
    // The checker reads bytes at any alignment, so nothing is copied for a bad image.
    if let Err(e) = check::<T>(bytes, todo)? {
      return Some(Err(e));
    }
    // SAFETY: `bytes` is not in a new storage.
    Some(unsafe { Self::copy_of(S::default(), bytes.as_ptr(), bytes.len()) }.ok_or(Invalid::LARGE))
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
  /// If the storage cannot hold them (or they exceed 2 GiB).
  #[must_use]
  pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self
  where
    S: Default,
  {
    fits::<T>();
    // SAFETY: `bytes` is not in a new storage.
    let r = unsafe { Self::copy_of(S::default(), bytes.as_ptr(), bytes.len()) };
    r.expect("region exceeds its storage")
  }
}

impl<T, S> Region<T, S> {
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

impl<T: Flat> Region<T> {
  /// The value whose image is `bytes`, read in place, for example from a memory-mapped file.
  /// The slice must be 8-aligned; the bytes are checked as by [`from_bytes`](Self::from_bytes).
  /// It needs `alloc`; [`view_with`](Self::view_with) does not.
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
  #[cfg(feature = "alloc")]
  pub fn view(bytes: &[u8]) -> Result<&T, Invalid> {
    // A vector is never full.
    Self::read(bytes, Stack::Heap(Vec::new())).unwrap_or(Err(Invalid(0)))
  }

  /// [`view`](Self::view), checking with `scratch` for memory: `None` if it ran out before the
  /// check finished, which says nothing about the bytes; otherwise the result of `view`.
  /// `bytes.len() / 4` entries are always enough (see [`Pending`]). It needs no `alloc`.
  ///
  /// ```
  /// use nearest::{Near, Pending, Region, flat};
  ///
  /// #[flat]
  /// #[derive(Debug, PartialEq)]
  /// enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }
  ///
  /// let mut r = Region::new(Expr::Lit(0));
  /// for i in 1..=20 {
  ///   r.edit().update(|sum| Expr::add(sum, Expr::Lit(i))); // deep on the left
  /// }
  /// let compact = Region::new(&*r);
  /// let bytes = compact.as_bytes().unwrap(); // the region's own buffer, which is 8-aligned
  /// let mut scratch = vec![Pending::default(); bytes.len() / 4];
  /// assert_eq!(Region::<Expr>::view_with(&bytes, &mut scratch), Some(Ok(&*r)));
  /// assert_eq!(Region::<Expr>::view_with(&bytes, &mut scratch[..10]), None);
  /// ```
  pub fn view_with<'b>(bytes: &'b [u8], scratch: &mut [Pending]) -> Option<Result<&'b T, Invalid>> {
    Self::read(bytes, Stack::Scratch(scratch, 0))
  }

  /// `view`, checking with `todo`.
  fn read<'b>(bytes: &'b [u8], todo: Stack<'_>) -> Option<Result<&'b T, Invalid>> {
    if !bytes.as_ptr().addr().is_multiple_of(8) {
      return Some(Err(Invalid(0)));
    }
    // SAFETY: just checked.
    Some(check::<T>(bytes, todo)?.map(|()| unsafe { Self::view_unchecked(bytes) }))
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
}

impl<T, S: Storage> Deref for Region<T, S> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a region holds a `T` at byte 0 of its 8-aligned buffer.
    unsafe { &*self.base().cast() }
  }
}

impl<T, S: Storage + Default> Clone for Region<T, S> {
  fn clone(&self) -> Self {
    // SAFETY: the bytes in use are in the memory of `self`, not of the new storage.
    let r = unsafe { Self::copy_of(S::default(), self.base(), self.buf.len()) };
    let mut r = r.expect("region exceeds its storage");
    r.buf.compact = self.buf.compact;
    r
  }
}

impl<T: fmt::Debug, S: Storage> fmt::Debug for Region<T, S> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

impl<T: PartialEq, S: Storage> PartialEq for Region<T, S> {
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: Eq, S: Storage> Eq for Region<T, S> {}

// A region owns its value in a storage of its own. Nothing pinned outlives a borrow of the
// region: moving it (which moves inline storage) is like growing its buffer.
impl<T, S: Unpin> Unpin for Region<T, S> {}

const _: () = {
  const fn send_sync_unpin<T: Send + Sync + Unpin>() {}
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<u64>>();
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<crate::Near<[u64]>>>();
  send_sync_unpin::<Region<crate::Near<[u64]>, crate::Inline<64>>>();
};
