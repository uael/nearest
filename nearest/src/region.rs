#[cfg(feature = "alloc")]
use alloc::alloc::Global;
#[cfg(feature = "serde")]
use alloc::vec::Vec;
use core::{alloc::Allocator, fmt, marker::PhantomData, ops::Deref, slice};

use crate::{
  Emit, Error, Flat,
  flat::{Buf, Sink, fits, limit},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place; [`as_bytes`](Self::as_bytes) and [`from_bytes`](Self::from_bytes) save
/// and load it.
///
/// The buffer is one block of the allocator `A`, aligned to 8 bytes on every target: by default,
/// with `alloc`, the global allocator, and [`Fixed`](crate::Fixed) over a buffer of your own,
/// without a heap. The block grows as needed, up to the limit of `T` (2 GiB with the default
/// pointers; see [`Offset`](crate::Offset)), and a growth that the allocator refuses panics. A
/// region in another allocator is built with [`new_in`](Self::new_in) or loaded with
/// [`from_bytes_in`](Self::from_bytes_in); everything else is the same for all.
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
pub struct Region<
  T,
  #[cfg(feature = "alloc")] A: Allocator = Global,
  #[cfg(not(feature = "alloc"))] A: Allocator,
> {
  buf: Buf,
  alloc: A,
  _t: PhantomData<T>,
}

impl<T, A: Allocator> Region<T, A> {
  /// An empty region with no block, of at most `max` bytes, the limit of `T`.
  const fn empty(alloc: A, max: usize) -> Self {
    Self { buf: Buf::new::<A>(max), alloc, _t: PhantomData }
  }

  /// Writes into the buffer where the region is now: the address of its allocator, which moves
  /// with the region, and of the buffer itself, through which an edit reaches it (I8). Every use
  /// of the buffer through `&mut self` that may grow it starts here.
  const fn bind(&mut self) {
    self.buf.alloc = (&raw const self.alloc).cast();
    self.buf.me = &raw mut self.buf;
  }

  /// A region of at most `max` bytes in `alloc` holding a copy of the `len` bytes at `src`, or
  /// `None` if the allocator does not give a block for them.
  ///
  /// # Safety
  /// `src` is valid for reading `len` bytes, outside a new block, and `len <= max`.
  unsafe fn copy_of(alloc: A, max: usize, src: *const u8, len: usize) -> Option<Self> {
    let mut r = Self::empty(alloc, max);
    r.bind();
    // SAFETY: the buffer was just bound and is empty; `src` is not in it.
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
    Self::new_in(value, Global)
  }

  /// The region whose image is `bytes`, on the heap: [`from_bytes_in`](Self::from_bytes_in)
  /// with the global allocator.
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
  /// assert!(Region::<Msg>::from_bytes(&bytes[..bytes.len() - 1]).is_err());
  /// # Ok::<(), nearest::Error>(())
  /// ```
  ///
  /// # Errors
  /// As for [`from_bytes_in`](Self::from_bytes_in).
  pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
    Self::from_bytes_in(bytes, Global)
  }
}

impl<T: Flat, A: Allocator> Region<T, A> {
  /// A new region holding `value`, in a block of `alloc`: [`new`](Region::new) for any
  /// allocator.
  ///
  /// ```
  /// use core::mem::MaybeUninit;
  ///
  /// use nearest::{Fixed, Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// // 64 bytes on the stack.
  /// let mut bytes = [MaybeUninit::uninit(); 64];
  /// let mut r = Region::new_in(Msg::new(1, "hi"), Fixed::new(&mut bytes));
  /// r.edit().body().set("hello");
  /// assert_eq!(&*r.body, "hello");
  /// ```
  ///
  /// # Panics
  /// If the allocator does not give the block.
  pub fn new_in(value: impl Emit<T>, alloc: A) -> Self {
    fits::<T>();
    let mut r = Self::empty(alloc, limit::<T>());
    r.bind();
    // SAFETY: the buffer is bound, live during the call and only reached through the sink.
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
  pub fn edit(&mut self) -> T::Ref<'_, '_> {
    fits::<T>();
    self.bind();
    // SAFETY: the root is a `T` at byte 0, the buffer is bound, the region is borrowed for the
    // edit, and every access of the edit goes through `me` (I8).
    T::reference(unsafe { T::edit(self.buf.me, 0) })
  }

  /// Makes room for `bytes` more bytes: the block does not grow until the region holds more.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Log { entries: Near<[u64]> }
  ///
  /// let mut r = Region::new(Log::new(()));
  /// r.reserve(4096);
  /// for i in 0..100 {
  ///   r.edit().entries().push(i);
  /// }
  /// assert_eq!(r.entries.len(), 100);
  /// ```
  ///
  /// # Panics
  /// If the region would exceed its limit, or the allocator does not give the block.
  pub fn reserve(&mut self, bytes: usize) {
    self.bind();
    let end = self.buf.len().saturating_add(bytes);
    // SAFETY: the buffer is bound, and only reached through it during the call.
    unsafe { self.buf.room(end, end) };
  }

  /// The bytes in use, borrowed: the value, what it reaches, and what edits left behind. `None`
  /// if some of them may not be initialized: after a `&mut` to a type with padding was handed
  /// out (`*e.field() = ..` of such a type, a list of them as a slice, or a pinned view), since
  /// an assignment through it writes the padding as uninitialized bytes. It allocates nothing.
  ///
  /// The bytes load back as a region if their pointees are in the order of a copy, which every
  /// region that was built and not edited has (see [`from_bytes_in`](Self::from_bytes_in)), and
  /// read with [`view`](crate::view) otherwise. A copy, `Region::new(&*r)` (or `new_in`), has
  /// its bytes, in that order, without what edits left behind.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let mut r = Region::new(Msg::new(1, "hi"));
  /// let image = r.as_bytes().unwrap();
  /// assert_eq!(&*nearest::view::<Msg>(image)?.body, "hi"); // aligned as a region is
  /// r.edit().body().set("hello");
  /// let edited = r.as_bytes().unwrap(); // the old text is still there
  /// assert_eq!(&*nearest::view::<Msg>(edited)?.body, "hello");
  /// let copy = Region::new(&*r);
  /// assert!(copy.byte_len() < r.byte_len());
  /// assert_eq!(&*Region::<Msg>::from_bytes(copy.as_bytes().unwrap())?.body, "hello");
  /// # Ok::<(), nearest::Error>(())
  /// ```
  #[must_use]
  pub fn as_bytes(&self) -> Option<&[u8]> {
    // SAFETY: the bytes in use are in the block, initialized while `init` holds.
    self.buf.init.then(|| unsafe { slice::from_raw_parts(self.buf.base(), self.buf.len()) })
  }

  /// The region whose image is `bytes`, which may have any alignment, in a block of `alloc`: the
  /// bytes are copied as they are, to an address aligned as a region is, then checked there as
  /// [`view`](crate::view) checks them. Nothing is rearranged: edits append after the image. The
  /// check of `puns` requires the pointees in the order of a copy, so that each is reached by
  /// one pointer, which the edits of a region rely on: what [`as_bytes`](Self::as_bytes)
  /// returns for a region that was built, with gaps that no pointer reaches too. Bytes whose
  /// pointees are shared, or out of that order, are refused, as `view` refuses them: save a copy
  /// of an edited region, `Region::new(&*r)`, which is in that order.
  ///
  /// # Errors
  /// In this order: [`Error::Invalid`] with [`puns::Error::Size`] for fewer bytes than a `T` or
  /// more than the limit of `T`, [`Error::Alloc`] if the allocator does not give a block for
  /// them, and [`Error::Invalid`] with what `puns` found if the copy is not the image of a valid
  /// `T` in the order of a copy.
  pub fn from_bytes_in(bytes: &[u8], alloc: A) -> Result<Self, Error> {
    fits::<T>();
    if bytes.len() < size_of::<T>() || bytes.len() > limit::<T>() {
      return Err(Error::Invalid(puns::Error::Size));
    }
    // SAFETY: `bytes` is not in a new block, and within the limit.
    let r = unsafe { Self::copy_of(alloc, limit::<T>(), bytes.as_ptr(), bytes.len()) };
    let r = r.ok_or(Error::Alloc)?;
    // SAFETY: the bytes in use are the copy, initialized.
    let image = unsafe { slice::from_raw_parts(r.buf.base(), r.buf.len()) };
    crate::view::<T>(image)?;
    Ok(r)
  }

  /// The region whose bytes are a copy of `bytes`, in a block of `alloc`, which are trusted, or
  /// that the caller checked itself: nothing is checked, and nothing is copied but the bytes.
  ///
  /// ```
  /// use std::alloc::Global;
  ///
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// // SAFETY: the bytes come from `as_bytes` of a region of the same type, in this build, that
  /// // was not edited.
  /// let back = unsafe { Region::<Msg>::from_bytes_unchecked_in(r.as_bytes().unwrap(), Global) };
  /// assert_eq!((back.id, &*back.body), (1, "hi"));
  /// ```
  ///
  /// # Safety
  /// In this build, [`from_bytes_in`](Self::from_bytes_in) would accept `bytes`: every offset
  /// and length is an `O` that `from_isize` makes (see [`Offset`](crate::Offset)), and the
  /// pointees are in the order of a copy, so each is reached by one pointer only, and overlaps no
  /// other value: the bytes are a tree, which the edits of a region rely on. That is what
  /// [`as_bytes`](Self::as_bytes) returned for a region of `T` that was built and not edited, in
  /// a build with the same layout of `T` (this one, for example).
  ///
  /// # Panics
  /// If the allocator does not give a block for them.
  #[must_use]
  pub unsafe fn from_bytes_unchecked_in(bytes: &[u8], alloc: A) -> Self {
    fits::<T>();
    assert!(bytes.len() <= limit::<T>(), "region exceeds what its pointers reach");
    // SAFETY: `bytes` is not in a new block, and within the limit.
    let r = unsafe { Self::copy_of(alloc, limit::<T>(), bytes.as_ptr(), bytes.len()) };
    r.expect("region exceeds what its allocator gives")
  }
}

impl<T, A: Allocator> Region<T, A> {
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

impl<T, A: Allocator> Deref for Region<T, A> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a region holds a `T` at byte 0 of its block, aligned to 8, so for `T`, which is
    // not more aligned (`fits`, on every way into a region). The region upholds what `puns`
    // requires of unsafe code that gives a reference to a value that holds an `InPlace` ("Values
    // made by unsafe code" in `puns::InPlace`): every `Near` of the value leads to a valid
    // pointee in the block, which its writers put there, and which is not written while the
    // region is borrowed, and the reference has the provenance of the block.
    unsafe { &*self.buf.base().cast() }
  }
}

impl<T, A: Allocator> Drop for Region<T, A> {
  fn drop(&mut self) {
    if let Some((block, layout)) = self.buf.block() {
      // SAFETY: the block is of this allocator, and fits `layout`.
      unsafe { self.alloc.deallocate(block, layout) };
    }
  }
}

impl<T, A: Allocator + Clone> Clone for Region<T, A> {
  fn clone(&self) -> Self {
    let (base, len) = (self.buf.base(), self.buf.len());
    // SAFETY: the bytes in use are in the block of `self`, not in a new one, within the limit.
    let r = unsafe { Self::copy_of(self.alloc.clone(), self.buf.max(), base, len) };
    let mut r = r.expect("region exceeds what its allocator gives");
    (r.buf.init, r.buf.loaded) = (self.buf.init, self.buf.loaded);
    r
  }
}

impl<T: fmt::Debug, A: Allocator> fmt::Debug for Region<T, A> {
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
impl<T: fmt::Display, A: Allocator> fmt::Display for Region<T, A> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Display::fmt(&**self, f)
  }
}

impl<T: PartialEq, A: Allocator> PartialEq for Region<T, A> {
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: Eq, A: Allocator> Eq for Region<T, A> {}

// A region owns its value in a block of its own, which does not move with it. Nothing pinned
// outlives a borrow of the region.
impl<T, A: Allocator + Unpin> Unpin for Region<T, A> {}

/// With the `serde` feature: a region is its bytes, written with `serialize_bytes`: those of
/// [`as_bytes`](Region::as_bytes) if they load back as the region, or else those of a copy. Like any image, they hold
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
impl<T: Flat, A: Allocator> ::serde::Serialize for Region<T, A> {
  fn serialize<Z: ::serde::Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
    if let Some(bytes) = self.as_bytes()
      && crate::view::<T>(bytes).is_ok()
    {
      return serializer.serialize_bytes(bytes);
    }
    // A copy has its bytes, in the order of a copy.
    let copy = Region::<T>::new(&**self);
    serializer.serialize_bytes(copy.as_bytes().unwrap_or_default())
  }
}

/// With the `serde` feature: a region from its bytes, checked as
/// [`from_bytes`](Region::from_bytes) checks them; an [`Error`] is a custom error of the
/// deserializer. Bytes are accepted borrowed, owned, or as a sequence of `u8` (as JSON writes
/// them).
#[cfg(feature = "serde")]
impl<'de, T: Flat, A: Allocator + Default> ::serde::Deserialize<'de> for Region<T, A> {
  fn deserialize<D: ::serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    use ::serde::de::{self, SeqAccess, Visitor};

    /// Builds the region from what the format gives.
    struct Bytes<T, A>(PhantomData<(T, A)>);

    impl<'de, T: Flat, A: Allocator + Default> Visitor<'de> for Bytes<T, A> {
      type Value = Region<T, A>;

      fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the bytes of a region")
      }

      fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
        Region::from_bytes_in(bytes, A::default()).map_err(E::custom)
      }

      fn visit_seq<Q: SeqAccess<'de>>(self, mut seq: Q) -> Result<Self::Value, Q::Error> {
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
  #[cfg(feature = "alloc")]
  const fn send_sync_unpin<T: Send + Sync + Unpin>() {}
  const fn send_unpin<T: Send + Unpin>() {}
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<u64>>();
  #[cfg(feature = "alloc")]
  send_sync_unpin::<Region<crate::Near<[u64]>>>();
  send_unpin::<Region<crate::Near<[u64]>, crate::Fixed<'static>>>();
};
