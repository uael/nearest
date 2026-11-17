#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{fmt, marker::PhantomData, ops::Deref, slice};

#[cfg(feature = "alloc")]
use crate::Heap;
use crate::{
  Edit, Emit, Error, Flat, Pending, Storage,
  bytes::{Stack, check},
  flat::{Buf, Sink, fits, limit},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place; [`as_bytes`](Self::as_bytes) and [`from_bytes`](Self::from_bytes) save
/// and load it.
///
/// `S` is the [`Storage`] of the buffer: by default, with `alloc`, the [`Heap`](crate::Heap), which
/// grows as needed up to the limit of `T` (2 GiB with the default pointers; see
/// [`Offset`](crate::Offset)); [`Inline<N>`](crate::Inline) keeps it in `N` bytes inside the
/// `Region` value, with no heap; and a storage of your own puts it anywhere else. A region in
/// another storage is built with [`new_in`](Self::new_in) or [`from_bytes`](Self::from_bytes);
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
pub struct Region<T, #[cfg(feature = "alloc")] S = Heap, #[cfg(not(feature = "alloc"))] S> {
  buf: Buf,
  store: S,
  _t: PhantomData<T>,
}

impl<T, S: Storage> Region<T, S> {
  /// An empty region in `store`, not attached: attach it once it is where it is used, since a
  /// storage may move with its region. It holds at most `max` bytes, the limit of `T`.
  const fn empty(store: S, max: usize) -> Self {
    Self { buf: Buf::new(max), store, _t: PhantomData }
  }

  /// Attaches the storage to the buffer, where it is now: a storage may move with its region.
  /// Every use of the buffer through `&mut self` starts here, after the region last moved.
  fn attach(&mut self) {
    // SAFETY: the storage holds the buffer's bytes (none when given), moves only with the
    // region, and is reached only through the buffer while the region is borrowed.
    unsafe { self.buf.attach(&raw mut self.store, 0) };
  }

  /// Byte 0, for reading, from the storage where the region is now.
  fn base(&self) -> *const u8 {
    self.store.memory().as_ptr().cast()
  }

  /// A region of at most `max` bytes in `store` holding a copy of the `len` bytes at `src`, or
  /// `None` if the storage cannot hold them.
  ///
  /// # Safety
  /// `src` is valid for reading `len` bytes, outside `store`.
  unsafe fn copy_of(store: S, max: usize, src: *const u8, len: usize) -> Option<Self> {
    let mut r = Self::empty(store, max);
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
    let mut r = Self::empty(storage, limit::<T>());
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

  /// The compact image of the region, borrowed: the bytes a deep copy writes, which the loaders
  /// take back. `None` if the region is not compact: it was edited, or a list in it was built
  /// from an iterator whose size hint was wrong. A copy is always compact, so any region is
  /// saved with `Region::new(&*r)` first (or `new_in`, in any storage). It allocates nothing.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let mut r = Region::new(Msg::new(1, "hi"));
  /// let image = r.as_bytes().unwrap();
  /// assert_eq!(&*nearest::view::<Msg>(image)?.body, "hi"); // the buffer is 8-aligned
  /// r.edit().body().set("hello");
  /// assert!(r.as_bytes().is_none());
  /// let copy = Region::new(&*r);
  /// let image = copy.as_bytes().unwrap();
  /// assert!(image.len() < r.byte_len());
  /// assert_eq!(&*Region::<Msg>::from_bytes(image)?.body, "hello");
  /// # Ok::<(), nearest::Error>(())
  /// ```
  #[must_use]
  pub fn as_bytes(&self) -> Option<&[u8]> {
    // SAFETY: the bytes in use are in the memory, and initialized since the buffer is compact.
    self.buf.compact.then(|| unsafe { slice::from_raw_parts(self.base(), self.buf.len()) })
  }

  /// The region whose image is `bytes`, which may have any alignment, in a new storage. The
  /// bytes are checked, then copied: they must be exactly what [`as_bytes`](Self::as_bytes)
  /// returns for a `T`. For a storage that has no `Default`, copy from a view instead:
  /// `Region::new_in(storage, nearest::view::<T>(bytes)?)`. It needs `alloc`: checking takes
  /// memory that grows with the nesting of the data, which
  /// [`from_bytes_with`](Self::from_bytes_with) takes from the caller.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// let bytes = r.as_bytes().unwrap();
  /// let back = Region::<Msg>::from_bytes(bytes)?;
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// assert!(Region::<Msg>::from_bytes(&bytes[1..]).is_err());
  /// # Ok::<(), nearest::Error>(())
  /// ```
  ///
  /// # Errors
  /// [`Error::Invalid`] if the bytes are not the compact image of a valid `T`, and
  /// [`Error::StorageFull`] if they are one, larger than the storage gives.
  #[cfg(feature = "alloc")]
  pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error>
  where
    S: Default,
  {
    Self::load(bytes, Stack::Heap(Vec::new()))
  }

  /// [`from_bytes`](Self::from_bytes), checking with `scratch` for memory.
  /// `bytes.len() / 4` entries are always enough (`/ 2` with 2-byte pointers; see [`Pending`]).
  /// It needs no `alloc`.
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
  /// let back = Small::from_bytes_with(bytes, &mut scratch)?;
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// assert!(Small::from_bytes_with(&bytes[1..], &mut scratch).is_err());
  /// # Ok::<(), nearest::Error>(())
  /// ```
  ///
  /// # Errors
  /// As [`from_bytes`](Self::from_bytes), and [`Error::ScratchFull`] if `scratch` ran out before
  /// the check finished, which says nothing about the bytes.
  pub fn from_bytes_with(bytes: &[u8], scratch: &mut [Pending]) -> Result<Self, Error>
  where
    S: Default,
  {
    Self::load(bytes, Stack::Scratch(scratch, 0))
  }

  /// `from_bytes`, checking with `todo`.
  fn load(bytes: &[u8], todo: Stack<'_>) -> Result<Self, Error>
  where
    S: Default,
  {
    // The checker reads bytes at any alignment, so nothing is copied for a bad image.
    check::<T>(bytes, todo)?;
    // SAFETY: `bytes` is not in a new storage.
    let r = unsafe { Self::copy_of(S::default(), limit::<T>(), bytes.as_ptr(), bytes.len()) };
    r.ok_or(Error::StorageFull)
  }

  /// [`from_bytes`](Self::from_bytes) without the check, for bytes that are trusted.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// // SAFETY: the bytes come from `as_bytes` for the same type, in this build.
  /// let back = unsafe { Region::<Msg>::from_bytes_unchecked(r.as_bytes().unwrap()) };
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// ```
  ///
  /// # Safety
  /// `bytes` are what [`as_bytes`](Self::as_bytes) returned for a region of `T`, in a build
  /// with the same layout of `T`.
  ///
  /// # Panics
  /// If the storage cannot hold them (or they exceed what the pointers of `T` reach).
  #[must_use]
  pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self
  where
    S: Default,
  {
    fits::<T>();
    // SAFETY: `bytes` is not in a new storage.
    let r = unsafe { Self::copy_of(S::default(), limit::<T>(), bytes.as_ptr(), bytes.len()) };
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
    let r = unsafe { Self::copy_of(S::default(), self.buf.max(), self.base(), self.buf.len()) };
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

/// Shows the value, as the value does.
///
/// ```
/// use nearest::Region;
///
/// assert_eq!(Region::new(7u32).to_string(), "7");
/// ```
impl<T: fmt::Display, S: Storage> fmt::Display for Region<T, S> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Display::fmt(&**self, f)
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

/// With the `serde` feature: a region is its bytes, [`as_bytes`](Region::as_bytes) of the
/// region or of a compact copy of it, written with `serialize_bytes`. Like any image, they hold
/// the layout of the target, and distances and lengths in the byte order of their [`Offset`]
/// (native-endian for `i16`, `i32`, `i64`).
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// struct Msg { id: u32, body: Near<str> }
///
/// let mut r = Region::new(Msg::new(1, "hi"));
/// r.edit().body().set("hello");
/// let json = serde_json::to_string(&r)?;
/// let back: Region<Msg> = serde_json::from_str(&json)?;
/// assert_eq!(back, r);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[cfg(feature = "serde")]
impl<T: Flat, S: Storage> ::serde::Serialize for Region<T, S> {
  fn serialize<Z: ::serde::Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
    if let Some(bytes) = self.as_bytes() {
      return serializer.serialize_bytes(bytes);
    }
    // A copy is compact.
    let copy = Region::<T>::new(&**self);
    serializer.serialize_bytes(copy.as_bytes().unwrap_or_default())
  }
}

/// With the `serde` feature: a region from its bytes, checked as
/// [`from_bytes`](Region::from_bytes) checks them; an [`Error`] is a custom error of the
/// deserializer. Bytes are accepted borrowed, owned, or as a sequence of `u8` (as JSON writes
/// them).
#[cfg(feature = "serde")]
impl<'de, T: Flat, S: Storage + Default> ::serde::Deserialize<'de> for Region<T, S> {
  fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use ::serde::de::{self, SeqAccess, Visitor};

    /// Builds the region from what the format gives.
    struct Bytes<T, S>(PhantomData<(T, S)>);

    impl<'de, T: Flat, S: Storage + Default> Visitor<'de> for Bytes<T, S> {
      type Value = Region<T, S>;

      fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the bytes of a region")
      }

      fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
        Region::from_bytes(bytes).map_err(E::custom)
      }

      fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        // The hint comes from the input: trust it only so far.
        let mut bytes = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 16));
        while let Some(byte) = seq.next_element::<u8>()? {
          bytes.push(byte);
        }
        self.visit_bytes(&bytes)
      }
    }

    deserializer.deserialize_bytes(Bytes(PhantomData))
  }
}

const _: () = {
  const fn send_sync_unpin<T: Send + Sync + Unpin>() {}
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<u64>>();
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<crate::Near<[u64]>>>();
  send_sync_unpin::<Region<crate::Near<[u64]>, crate::Inline<64>>>();
};
