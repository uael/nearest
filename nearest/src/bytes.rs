#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{fmt, str};

use puns::layout::Endian;

use crate::{
  Flat, Offset,
  flat::{ALIGN, fits, fits_behind, limit},
  near::offset,
};

/// Why bytes could not be loaded: the error of every loader ([`view`], [`view_with`],
/// [`Region::from_bytes`](crate::Region::from_bytes),
/// [`Region::from_bytes_with`](crate::Region::from_bytes_with)).
///
/// ```
/// use nearest::{Error, Inline, Near, Pending, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap();
/// let cut = Region::<Msg>::from_bytes(&bytes[..bytes.len() - 1]);
/// assert_eq!(cut.err(), Some(Error::Invalid(8))); // the text, cut short
/// let small = Region::<Msg, Inline<8>>::from_bytes(bytes);
/// assert_eq!(small.err(), Some(Error::StorageFull)); // a valid image, larger than 8 bytes
/// let none = Region::<Msg>::from_bytes_with(bytes, &mut []);
/// assert_eq!(none.err(), Some(Error::ScratchFull)); // the check could not finish
/// assert_eq!(Error::Invalid(8).to_string(), "invalid region bytes at offset 8");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
  /// The bytes are not the compact image of a valid value: where they stop being one.
  Invalid(usize),
  /// The scratch given to `view_with` or `from_bytes_with` ran out before the check finished,
  /// which says nothing about the bytes.
  ScratchFull,
  /// The bytes are a valid image, larger than the storage of `from_bytes` or
  /// `from_bytes_with`.
  StorageFull,
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Invalid(at) => write!(f, "invalid region bytes at offset {at}"),
      Self::ScratchFull => f.write_str("scratch too small to check region bytes"),
      Self::StorageFull => f.write_str("region bytes larger than the storage"),
    }
  }
}

impl core::error::Error for Error {}

/// What checks a pointer: given the checker, the position of the pointer (or of the first
/// element, for the rest of a list) and a count.
type Step = fn(&mut Check<'_>, usize, usize) -> Result<(), Error>;

/// One entry of scratch memory for checking bytes: a pointer, or the rest of a list, to check.
///
/// [`view_with`] and [`Region::from_bytes_with`](crate::Region::from_bytes_with) take a slice of
/// them.
///
/// `bytes.len() / 4` entries are enough for any bytes (`bytes.len() / 2` with 2-byte pointers,
/// `Near<_, i16>`): each entry waits for its own bytes of the image, at least as many as a
/// pointer has (a pointer, or the start of the next element of a list, which holds one), so an
/// image that fits a storage of `N` bytes never needs more than `N / 4`. Most data needs far
/// fewer: about as many as the pointers that wait at once while the walk goes down the deepest
/// path. An entry is 24 bytes on a 64-bit target, 12 on a 32-bit one.
///
/// ```
/// use nearest::{Error, Near, Pending, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap();
/// let mut scratch = [Pending::default(); 8];
/// let back = Region::<Msg>::from_bytes_with(bytes, &mut scratch)?;
/// assert_eq!(&*back.body, "hi");
/// // Too little scratch is not an invalid image: the check could not finish.
/// assert_eq!(Region::<Msg>::from_bytes_with(bytes, &mut []).err(), Some(Error::ScratchFull));
/// # Ok::<(), nearest::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Pending(usize, usize, Step);

impl Default for Pending {
  fn default() -> Self {
    Self(0, 0, |_, _, _| Ok(()))
  }
}

/// What is still to check, the last one next: the caller's scratch and how much of it is used,
/// or, with `alloc`, a vector.
pub enum Stack<'s> {
  Scratch(&'s mut [Pending], usize),
  #[cfg(feature = "alloc")]
  Heap(Vec<Pending>),
}

impl Stack<'_> {
  const fn len(&self) -> usize {
    match self {
      Self::Scratch(_, len) => *len,
      #[cfg(feature = "alloc")]
      Self::Heap(items) => items.len(),
    }
  }

  /// Whether there was room.
  fn push(&mut self, item: Pending) -> bool {
    match self {
      Self::Scratch(items, len) => {
        let Some(slot) = items.get_mut(*len) else { return false };
        *slot = item;
        *len += 1;
      }
      #[cfg(feature = "alloc")]
      Self::Heap(items) => items.push(item),
    }
    true
  }

  fn pop(&mut self) -> Option<Pending> {
    match self {
      Self::Scratch(items, len) => {
        *len = len.checked_sub(1)?;
        items.get(*len).copied()
      }
      #[cfg(feature = "alloc")]
      Self::Heap(items) => items.pop(),
    }
  }

  /// The entries from `mark` on.
  fn top(&mut self, mark: usize) -> &mut [Pending] {
    match self {
      Self::Scratch(items, len) => items.get_mut(mark..*len),
      #[cfg(feature = "alloc")]
      Self::Heap(items) => items.get_mut(mark..),
    }
    .unwrap_or_default()
  }
}

/// A walk over bytes that must be the compact image of a value.
///
/// The root, then every target in the order a deep copy writes it. Each value is checked by the
/// `check` of its type ([`Flat::check`]), which queues its pointers on `todo`: `puns` checks what
/// is not a pointer ([`puns::validate_but_never`]), and where the targets of the pointers are, and
/// their blocks, is this crate's.
pub struct Check<'b> {
  bytes: &'b [u8],
  /// Where the next target must start (before alignment).
  next: usize,
  /// Pointers and rests of lists still to check; the last one is next.
  todo: Stack<'b>,
}

impl Check<'_> {
  /// Checks the `T` at `at`, which holds no pointer, as `puns` validates it: every scalar, tag and
  /// rule. Not at all if every value is valid.
  #[inline]
  pub(crate) fn value<T: Flat>(&self, at: usize) -> Result<(), Error> {
    if T::ANY {
      return Ok(());
    }
    let bytes = self.span(at, size_of::<T>())?;
    puns::validate_but_never::<T>(bytes).map_err(|_| Error::Invalid(at))
  }

  /// Checks the rules of the `T` at `at`, whose fields are checked, as `puns` validates it: a
  /// rule that does not hold is refused where the value is. `puns` does not read its pointers.
  ///
  /// # Errors
  /// If a rule does not hold.
  #[inline]
  pub fn rules<T: Flat>(&self, at: usize) -> Result<(), Error> {
    let bytes = self.span(at, size_of::<T>())?;
    puns::validate_but_never::<T>(bytes).map_err(|_| Error::Invalid(at))
  }

  /// Checks the rules of a field (`#[pun(valid(..))]`) at `at`, as `puns` validates `W`, a struct
  /// of that field alone with the same rules: a rule that does not hold is refused where the field
  /// is.
  ///
  /// # Errors
  /// If a rule does not hold.
  #[inline]
  pub fn valid<W: puns::Pun>(&self, at: usize) -> Result<(), Error> {
    let bytes = self.span(at, size_of::<W>())?;
    puns::validate_but_never::<W>(bytes).map_err(|_| Error::Invalid(at))
  }

  /// The `size` bytes at `at`. The slice is made from a checked end, so its length is known to be
  /// `size`, and the size check of `puns` on it folds away.
  #[inline]
  fn span(&self, at: usize, size: usize) -> Result<&[u8], Error> {
    at.checked_add(size).and_then(|end| self.bytes.get(at..end)).ok_or(Error::Invalid(at))
  }

  /// The tag of an enum of `count` variants at `at`.
  ///
  /// # Errors
  /// If it is out of bounds or not below `count`.
  #[inline]
  pub fn tag(&self, at: usize, count: usize) -> Result<u8, Error> {
    match self.bytes.get(at) {
      Some(&tag) if usize::from(tag) < count => Ok(tag),
      _ => Err(Error::Invalid(at)),
    }
  }

  /// A pointer of `O` at `at`: what it aims at waits for the values before it, and `step` checks
  /// it. A pointer of zero bytes is `None` if `nullable`, and refused where it is otherwise (no
  /// `Near` is zero).
  #[inline]
  pub(crate) fn pointer<O: Offset>(
    &mut self,
    at: usize,
    nullable: bool,
    step: Step,
  ) -> Result<(), Error> {
    offset::<O>();
    // One read of the width of `O`, which is 2, 4 or 8 bytes (`offset`): zero in any byte order.
    let zero = match size_of::<O>() {
      2 => self.get::<2>(at, Endian::NATIVE)? == [0; 2],
      4 => self.get::<4>(at, Endian::NATIVE)? == [0; 4],
      _ => self.get::<8>(at, Endian::NATIVE)? == [0; 8],
    };
    if zero {
      return if nullable { Ok(()) } else { Err(Error::Invalid(at)) };
    }
    self.push(at, 0, step)
  }

  /// Queues a pointer, or the rest of a list. If the scratch is full, the walk ends.
  fn push(&mut self, at: usize, count: usize, step: Step) -> Result<(), Error> {
    if self.todo.push(Pending(at, count, step)) { Ok(()) } else { Err(Error::ScratchFull) }
  }

  /// The integer of the offset (or the length) of `O` at `at`: two's complement of the width and
  /// the byte order of `O`, which is what `to_isize` reads (the contract of `Offset`). Valid only
  /// if it fits an `isize` (Valid offsets): it is then also within the `LIMIT` of the offset, as
  /// soon as it aims in the bytes or counts bytes of them, which are no more than the limit of the
  /// root, at most the `LIMIT` of every pointer (`fits_behind`).
  #[inline]
  fn offset<O: Offset>(&self, at: usize) -> Result<isize, Error> {
    offset::<O>();
    let width = size_of::<O>();
    let n = match width {
      2 => i64::from(i16::from_ne_bytes(self.get(at, O::ENDIAN)?)),
      4 => i64::from(i32::from_ne_bytes(self.get(at, O::ENDIAN)?)),
      _ => i64::from_ne_bytes(self.get(at, O::ENDIAN)?),
    };
    // Past the default `LIMIT` of an offset of that width, `iN::MAX & !7` (that of every offset
    // of this crate), no integer is a distance or a length: refused where it is, as `from_isize`
    // never makes it. (An offset of the user's with a smaller `LIMIT` is refused at the block
    // that it cannot reach.)
    let limit = (1u64 << (8 * width - 1)) - 8;
    if n.unsigned_abs() > limit {
      return Err(Error::Invalid(at));
    }
    isize::try_from(n).map_err(|_| Error::Invalid(at))
  }

  /// The `N` bytes at `at`, in the byte order of the target, read in the byte order `endian`.
  #[inline]
  fn get<const N: usize>(&self, at: usize, endian: Endian) -> Result<[u8; N], Error> {
    let mut bytes =
      *self.bytes.get(at..).and_then(<[u8]>::first_chunk::<N>).ok_or(Error::Invalid(at))?;
    if endian != Endian::NATIVE {
      bytes.reverse();
    }
    Ok(bytes)
  }

  /// The length, an integer like the offset of `O`, at `at`; a negative one is as out of bounds
  /// as any too large.
  #[inline]
  fn len<O: Offset>(&self, at: usize) -> Result<usize, Error> {
    Ok(usize::try_from(self.offset::<O>(at)?).unwrap_or(usize::MAX))
  }

  /// The next block, of `size` bytes whose byte `inset` is aligned to `align`: the pointer of
  /// `O` at `ptr` must aim `lead` bytes into it, which is where a deep copy would have put it.
  /// Returns its position.
  #[inline]
  fn claim<O: Offset>(
    &mut self,
    ptr: usize,
    lead: usize,
    inset: usize,
    align: usize,
    size: usize,
  ) -> Result<usize, Error> {
    let off = self.offset::<O>(ptr)?;
    let at = (self.next.checked_add(inset).and_then(|n| n.checked_next_multiple_of(align)))
      .ok_or(Error::Invalid(ptr))?
      - inset;
    let aimed = ptr.checked_add_signed(off) == Some(at + lead);
    let end = (at.checked_add(size).filter(|&end| aimed && end <= self.bytes.len()))
      .ok_or(Error::Invalid(ptr))?;
    self.next = end;
    Ok(at)
  }

  /// Takes `size` more bytes after the last block.
  fn extend(&mut self, at: usize, size: Option<usize>) -> Result<(), Error> {
    let end = size.and_then(|s| self.next.checked_add(s)).filter(|&end| end <= self.bytes.len());
    self.next = end.ok_or(Error::Invalid(at))?;
    Ok(())
  }
}

/// The target of the `Near<T, O>` at `ptr`, which aims at one value: at the next position aligned
/// for it.
pub fn target<T: Flat, O: Offset>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  fits_behind::<T, O>();
  let at = ck.claim::<O>(ptr, 0, 0, align_of::<T>(), size_of::<T>())?;
  T::check(ck, at)
}

/// The list block of the `Near<[T], O>` at `ptr`: `[cap][len]`, integers like the offset, then the
/// elements, aligned for them and for the offset.
pub fn list<T: Flat, O: Offset>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  fits_behind::<T, O>();
  let w = size_of::<O>();
  let at = ck.claim::<O>(ptr, w, 2 * w, align_of::<T>().max(w), 2 * w)?;
  let (cap, len) = (ck.len::<O>(at)?, ck.len::<O>(at + w)?);
  ck.extend(at, len.checked_mul(size_of::<T>()).filter(|_| cap == len))?;
  items::<T>(ck, at + 2 * w, len)
}

/// The text of the `Near<str, O>` at `ptr`: `[len]`, an integer like the offset, then UTF-8.
pub fn text<O: Offset>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  let w = size_of::<O>();
  let at = ck.claim::<O>(ptr, 0, 0, w, w)?;
  let len = ck.len::<O>(at)?;
  ck.extend(at, Some(len))?;
  match ck.bytes.get(at + w..ck.next).map(str::from_utf8) {
    Some(Ok(_)) => Ok(()),
    _ => Err(Error::Invalid(at)),
  }
}

/// The `count` elements from `first`, in order. When one has pointers, they are checked before
/// the remaining elements, which wait on the stack behind them. Elements without pointers are
/// validated all at once, and one by one only to tell where the first invalid one is; not at all
/// if every value is valid.
fn items<T: Flat>(ck: &mut Check<'_>, first: usize, count: usize) -> Result<(), Error> {
  if T::ANY {
    return Ok(());
  }
  let size = size_of::<T>();
  if !T::POINTERS {
    let bytes = ck.bytes.get(first..first.saturating_add(size.saturating_mul(count)));
    if bytes.is_some_and(|b| puns::validate_but_never::<[T]>(b).is_ok()) {
      return Ok(());
    }
  }
  for i in 0..count {
    let mark = ck.todo.len();
    T::check(ck, first + i * size)?;
    if ck.todo.len() > mark {
      if i + 1 < count {
        ck.push(first + (i + 1) * size, count - i - 1, items::<T>)?;
      }
      break;
    }
  }
  Ok(())
}

/// Checks that `bytes` are the compact image of a `T`, without recursion or panics, keeping
/// what is still to check in `todo` ([`Error::ScratchFull`] if it runs out first). Like every way
/// into a region, it rejects at compile time the types a region cannot hold.
pub fn check<T: Flat>(bytes: &[u8], todo: Stack<'_>) -> Result<(), Error> {
  fits::<T>();
  if bytes.len() < size_of::<T>() || bytes.len() > limit::<T>() {
    return Err(Error::Invalid(0));
  }
  walk::<T>(&mut Check { bytes, next: size_of::<T>(), todo })
}

/// The value whose image is `bytes`, read in place, for example from a memory-mapped file.
///
/// The slice must be aligned as a region is, to `align_of::<u64>()` (the words of the memory of
/// every [`Storage`](crate::Storage): 8 bytes on x86-64, 4 on `i686`), as the bytes of a region
/// and a mapped file are; the bytes are checked as by
/// [`Region::from_bytes`](crate::Region::from_bytes). It needs `alloc`; [`view_with`] does not.
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
/// not aligned to `align_of::<u64>()` (at offset 0).
#[cfg(feature = "alloc")]
pub fn view<T: Flat>(bytes: &[u8]) -> Result<&T, Error> {
  read(bytes, Stack::Heap(Vec::new()))
}

/// [`view`], checking with `scratch` for memory. `bytes.len() / 4` entries are always enough
/// (`bytes.len() / 2` with 2-byte pointers; see [`Pending`]). It needs no `alloc`.
///
/// ```
/// use nearest::{Error, Near, Pending, Region, flat};
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
/// let bytes = compact.as_bytes().unwrap(); // the region's own buffer, aligned as a region is
/// let mut scratch = vec![Pending::default(); bytes.len() / 4];
/// assert_eq!(nearest::view_with::<Expr>(bytes, &mut scratch), Ok(&*r));
/// assert_eq!(nearest::view_with::<Expr>(bytes, &mut scratch[..10]), Err(Error::ScratchFull));
/// ```
///
/// # Errors
/// As [`view`], and [`Error::ScratchFull`] if `scratch` ran out before the check finished,
/// which says nothing about the bytes.
pub fn view_with<'b, T: Flat>(bytes: &'b [u8], scratch: &mut [Pending]) -> Result<&'b T, Error> {
  read(bytes, Stack::Scratch(scratch, 0))
}

/// `view`, checking with `todo`.
fn read<'b, T: Flat>(bytes: &'b [u8], todo: Stack<'_>) -> Result<&'b T, Error> {
  if !bytes.as_ptr().addr().is_multiple_of(ALIGN) {
    return Err(Error::Invalid(0));
  }
  check::<T>(bytes, todo)?;
  // SAFETY: just checked.
  Ok(unsafe { view_unchecked(bytes) })
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
/// valid `T`, every offset and length an `O` that `from_isize` makes (see [`Offset`](crate::Offset)). That is what
/// [`Region::as_bytes`](crate::Region::as_bytes) returned for a region of `T` in a build with the
/// same layout of `T` (this one, for example), and what a check of the caller's that accepts no
/// more than [`view`] accepted.
#[must_use]
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T {
  fits::<T>();
  // SAFETY: per the caller, the bytes are aligned as a region is, so for `T` (`fits`: `T` is not
  // more aligned), and the image of a valid `T`, whose pointers all aim inside them, at positions
  // aligned for their targets, so at addresses aligned for them.
  unsafe { &*bytes.as_ptr().cast::<T>() }
}

/// The root, then everything it points to.
fn walk<T: Flat>(ck: &mut Check<'_>) -> Result<(), Error> {
  T::check(ck, 0)?;
  let mut mark = 0;
  loop {
    // What the last step queued is checked first-queued first.
    ck.todo.top(mark).reverse();
    let Some(Pending(at, count, step)) = ck.todo.pop() else { break };
    mark = ck.todo.len();
    step(ck, at, count)?;
  }
  if ck.next == ck.bytes.len() { Ok(()) } else { Err(Error::Invalid(ck.next)) }
}
