use core::fmt;

use puns::layout::Endian;

use crate::{
  Flat, Offset,
  aim::{ListBlock, TextBlock},
  flat::{ALIGN, fits, limit},
  near::offset,
};

/// Why bytes could not be loaded: the error of every loader ([`view`],
/// [`Region::from_bytes`](crate::Region::from_bytes)).
///
/// ```
/// use nearest::{Error, Inline, Near, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap();
/// let cut = Region::<Msg>::from_bytes(&bytes[..bytes.len() - 1]);
/// assert_eq!(cut.err(), Some(Error::Invalid)); // the text, cut short
/// let small = Region::<Msg, Inline<8>>::from_bytes(bytes);
/// assert_eq!(small.err(), Some(Error::StorageFull)); // larger than 8 bytes
/// assert_eq!(Error::Invalid.to_string(), "invalid region bytes");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
  /// The bytes are not the compact image of a valid value.
  Invalid,
  /// The check stopped before it found the bytes valid or not, which says nothing about them:
  /// `puns` follows pointers at most [`puns::AIM_DEPTH`] deep, one pointer behind another.
  Limit,
  /// The bytes are larger than the storage of `from_bytes` gives. They are not checked then.
  StorageFull,
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::Invalid => "invalid region bytes",
      Self::Limit => "region bytes nested too deep to check",
      Self::StorageFull => "region bytes larger than the storage",
    })
  }
}

impl core::error::Error for Error {}

/// A walk over the image of a valid value, which `puns` checked, that requires it to be compact:
/// the root, then every target where a deep copy writes it.
///
/// `puns` checks every value and every target, wherever the target is in the bytes: targets may be
/// shared, overlap other values, or leave gaps. A region requires more, a tree of values that do
/// not overlap, since an edit writes plain data through a `&mut` and a pointer through its handle:
/// one value written through both would let safe code write a pointer. So the walk follows the
/// pointers in the order that the deep copy writes their targets (depth first, the fields in their
/// order, the elements of a list in order) and requires each to aim where the copy puts its
/// target, the next position aligned for it, with `cap == len` for a list, and no byte after the
/// last target. That gives no sharing, no overlap, no cycle and no gap.
///
/// It reads only what `puns` checked: tags, offsets and lengths, which are valid, and positions
/// within the bytes. It recurses once for each pointer behind another, which `puns` bounds
/// ([`puns::AIM_DEPTH`]), and once for each field holding a pointer, which the type bounds. It is
/// safe code that never panics.
pub struct Compact<'b> {
  bytes: &'b [u8],
  /// Where the next target must start (before alignment).
  next: usize,
}

impl Compact<'_> {
  /// The tag of the enum at `at`, which `puns` found valid.
  ///
  /// # Errors
  /// If `at` is out of bounds, which it is not after `puns` checked the value.
  #[inline]
  pub fn tag(&self, at: usize) -> Result<u8, Error> {
    self.bytes.get(at).copied().ok_or(Error::Invalid)
  }

  /// The integer of the `O` at `at`: two's complement of the width and the byte order of `O`,
  /// which is what `to_isize` reads (the contract of `Offset`), and what `puns` read of the
  /// offset or the length there.
  #[inline]
  fn integer<O: Offset>(&self, at: usize) -> Result<i64, Error> {
    offset::<O>();
    Ok(match size_of::<O>() {
      2 => i64::from(i16::from_ne_bytes(self.get(at, O::ENDIAN)?)),
      4 => i64::from(i32::from_ne_bytes(self.get(at, O::ENDIAN)?)),
      _ => i64::from_ne_bytes(self.get(at, O::ENDIAN)?),
    })
  }

  /// The `N` bytes at `at`, in the byte order of the target, read in the byte order `endian`.
  #[inline]
  fn get<const N: usize>(&self, at: usize, endian: Endian) -> Result<[u8; N], Error> {
    let mut bytes =
      *self.bytes.get(at..).and_then(<[u8]>::first_chunk::<N>).ok_or(Error::Invalid)?;
    if endian != Endian::NATIVE {
      bytes.reverse();
    }
    Ok(bytes)
  }

  /// The next block, of `size` bytes whose byte `inset` is aligned to `align`: the pointer of `O`
  /// at `ptr` must aim `lead` bytes into it, which is where a deep copy puts it. Returns its
  /// position, and takes its bytes.
  #[inline]
  fn claim<O: Offset>(
    &mut self,
    ptr: usize,
    lead: usize,
    inset: usize,
    align: usize,
    size: usize,
  ) -> Result<usize, Error> {
    let off = isize::try_from(self.integer::<O>(ptr)?).map_err(|_| Error::Invalid)?;
    let at = (self.next.checked_add(inset).and_then(|n| n.checked_next_multiple_of(align)))
      .and_then(|n| n.checked_sub(inset))
      .ok_or(Error::Invalid)?;
    let aimed = at.checked_add(lead).is_some_and(|aim| ptr.checked_add_signed(off) == Some(aim));
    let end = (at.checked_add(size).filter(|&end| aimed && end <= self.bytes.len()))
      .ok_or(Error::Invalid)?;
    self.next = end;
    Ok(at)
  }

  /// Takes `size` more bytes after the last block.
  fn extend(&mut self, size: Option<usize>) -> Result<(), Error> {
    let end = size.and_then(|s| self.next.checked_add(s)).filter(|&end| end <= self.bytes.len());
    self.next = end.ok_or(Error::Invalid)?;
    Ok(())
  }

  /// The length of a block, the `O` at `at`, which `puns` found not negative.
  fn len<O: Offset>(&self, at: usize) -> Result<usize, Error> {
    usize::try_from(self.integer::<O>(at)?).map_err(|_| Error::Invalid)
  }

  /// Whether the pointer of `O` at `at` is zero, which is `None` for an `Option<Near>`.
  pub(crate) fn null<O: Offset>(&self, at: usize) -> Result<bool, Error> {
    Ok(self.integer::<O>(at)? == 0)
  }

  /// The target of the `Near<T, O>` at `ptr`, which aims at one value: at the next position
  /// aligned for it.
  pub(crate) fn target<T: Flat, O: Offset>(&mut self, ptr: usize) -> Result<(), Error> {
    let at = self.claim::<O>(ptr, 0, 0, align_of::<T>(), size_of::<T>())?;
    T::compact(self, at)
  }

  /// The list block of the `Near<[T], O>` at `ptr`, as `List::new` places it: its elements at the
  /// next position aligned for them and for the offset, with `cap == len`.
  pub(crate) fn list<T: Flat, O: Offset>(&mut self, ptr: usize) -> Result<(), Error> {
    let (lead, data) = (ListBlock::<T, O>::LEN, ListBlock::<T, O>::DATA);
    let at = self.claim::<O>(ptr, lead, data, align_of::<T>().max(size_of::<O>()), data)?;
    let (cap, len) = (self.len::<O>(at)?, self.len::<O>(at + lead)?);
    self.extend(len.checked_mul(size_of::<T>()).filter(|_| cap == len))?;
    if T::POINTERS {
      for i in 0..len {
        T::compact(self, at + data + i * size_of::<T>())?;
      }
    }
    Ok(())
  }

  /// The text of the `Near<str, O>` at `ptr`: its block at the next position aligned for the
  /// offset.
  pub(crate) fn text<O: Offset>(&mut self, ptr: usize) -> Result<(), Error> {
    let (lead, data) = (TextBlock::<O>::LEN, TextBlock::<O>::DATA);
    let at = self.claim::<O>(ptr, lead, 0, size_of::<O>(), data)?;
    let len = self.len::<O>(at + lead)?;
    self.extend(Some(len))
  }
}

/// Checks that `bytes`, at an address aligned to [`ALIGN`], are the compact image of a valid `T`:
/// every value and every target valid, as `puns` checks it, then the image compact. Like every way
/// into a region, it rejects at compile time the types a region cannot hold.
pub fn check<T: Flat>(bytes: &[u8]) -> Result<&T, Error> {
  fits::<T>();
  if bytes.len() < size_of::<T>() || bytes.len() > limit::<T>() {
    return Err(Error::Invalid);
  }
  let value = match puns::from_head::<T>(bytes) {
    Ok((value, _)) => value,
    Err(puns::Error::Limit) => return Err(Error::Limit),
    Err(_) => return Err(Error::Invalid),
  };
  let mut walk = Compact { bytes, next: size_of::<T>() };
  T::compact(&mut walk, 0)?;
  if walk.next == bytes.len() { Ok(value) } else { Err(Error::Invalid) }
}

/// The value whose image is `bytes`, read in place, for example from a memory-mapped file.
///
/// The slice must be aligned as a region is, to `align_of::<u64>()` (the words of the memory of
/// every [`Storage`](crate::Storage): 8 bytes on x86-64, 4 on `i686`), as the bytes of a region
/// and a mapped file are; the bytes are checked as by
/// [`Region::from_bytes`](crate::Region::from_bytes).
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap(); // the region's own buffer, aligned as a region is
/// let msg: &Msg = nearest::view(bytes)?;
/// assert_eq!(&*msg.body, "hi");
/// # Ok::<(), nearest::Error>(())
/// ```
///
/// # Errors
/// [`Error::Invalid`] if the bytes are not the compact image of a valid `T`, or the slice is
/// not aligned to `align_of::<u64>()`; [`Error::Limit`] if they nest pointers too deep to check.
pub fn view<T: Flat>(bytes: &[u8]) -> Result<&T, Error> {
  if !bytes.as_ptr().addr().is_multiple_of(ALIGN) {
    return Err(Error::Invalid);
  }
  check::<T>(bytes)
}

/// [`view`] without the check, for bytes that are trusted, or that the caller checked itself:
/// nothing is read before the value is used.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap(); // the region's own buffer, aligned as a region is
/// // SAFETY: the bytes come from `as_bytes` for the same type, in this build, and are aligned as
/// // a region is.
/// let msg: &Msg = unsafe { nearest::view_unchecked(bytes) };
/// assert_eq!(&*msg.body, "hi");
/// ```
///
/// # Safety
/// [`view::<T>(bytes)`](view) would return `Ok`, in this build: `bytes` is at an address that is
/// a multiple of `align_of::<u64>()` (the alignment of a region), and is the compact image of a
/// valid `T`. That is what [`Region::as_bytes`](crate::Region::as_bytes) returned for a region of
/// `T` in a build with the same layout of `T` (this one, for example), and what a check of the
/// caller's that accepts no more than [`view`] accepted.
#[must_use]
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T {
  fits::<T>();
  // SAFETY: per the caller, the bytes are aligned as a region is, so for `T` (`fits`: `T` is not
  // more aligned), and the image of a valid `T`, whose pointers all aim inside them, at positions
  // aligned for their targets, so at addresses aligned for them.
  unsafe { &*bytes.as_ptr().cast::<T>() }
}
