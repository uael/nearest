#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;

use puns::{Failure, Pending, Worklist};

use crate::{
  Flat,
  flat::{ALIGN, fits, limit},
};

/// Why bytes could not be loaded: the error of every loader ([`view`], [`view_with`],
/// [`Region::from_bytes`](crate::Region::from_bytes),
/// [`Region::from_bytes_with`](crate::Region::from_bytes_with)).
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
/// assert_eq!(cut.err(), Some(Error::Invalid(8))); // the text, cut short
/// let small = Region::<Msg, Inline<8>>::from_bytes(bytes);
/// assert_eq!(small.err(), Some(Error::StorageFull)); // a valid image, larger than 8 bytes
/// assert_eq!(Error::Invalid(8).to_string(), "invalid region bytes at offset 8");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
  /// The bytes are not the compact image of a valid value: where they stop being one, as `puns`
  /// says ([`puns::Failure`]).
  Invalid(usize),
  /// The scratch given to `view_with` or `from_bytes_with` ran out before the check finished,
  /// which says nothing about the bytes (see [`Pending`]).
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

/// The entries of a check on the heap, as many as it needs.
#[cfg(feature = "alloc")]
pub struct Stack(Vec<Pending>);

#[cfg(feature = "alloc")]
impl Stack {
  pub const fn new() -> Self {
    Self(Vec::new())
  }
}

// SAFETY: `items` is the vector, which only `grow` changes, and `grow` makes it longer, keeping its
// elements.
#[cfg(feature = "alloc")]
unsafe impl Worklist for Stack {
  fn items(&mut self) -> &mut [Pending] {
    &mut self.0
  }

  fn grow(&mut self) -> bool {
    self.0.resize(self.0.len().max(8) * 2, Pending::default());
    true
  }
}

/// Checks that `bytes`, at any address, are the compact image of a valid `T` once at an address
/// aligned as a region is: what `puns` checks without a reference, with a stack that does not grow
/// with the data ([`puns::check_compact`]), every value and every pointee valid, each pointee where a deep
/// copy writes it (depth first, in the order of the fields, at the next position aligned for it),
/// and nothing after the last one. That gives no sharing, no overlap, no cycle and no gap. The
/// pointees that wait are kept in `todo` ([`Error::ScratchFull`] if it runs out first). Like every
/// way into a region, it rejects at compile time the types a region cannot hold, behind pointers
/// too, as writing them does (`Flat::store`).
pub fn check<T: Flat, W: Worklist + ?Sized>(bytes: &[u8], todo: &mut W) -> Result<(), Error> {
  fits::<T>();
  // A load refuses at compile time what a build refuses: the stores of a value of `T`, which this
  // instantiates, assert at every depth where a pointer may lead (`fits_behind`). A constant
  // cannot hold that assertion: the types that a recursive type reaches lead back to it, a cycle
  // of constants, which Rust refuses (E0391), where functions instantiate each other.
  let _: unsafe fn(&T, &mut crate::flat::Sink, usize) = T::store;
  if bytes.len() < size_of::<T>() || bytes.len() > limit::<T>() {
    return Err(Error::Invalid(0));
  }
  match puns::check_compact::<T, ALIGN, W>(bytes, todo) {
    Ok(()) => Ok(()),
    Err(Failure { error: puns::Error::Limit, .. }) => Err(Error::ScratchFull),
    Err(Failure { at, .. }) => Err(Error::Invalid(at)),
  }
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
  read(bytes, &mut Stack::new())
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
  read(bytes, scratch)
}

/// `view`, checking with `todo`.
fn read<'b, T: Flat, W: Worklist + ?Sized>(bytes: &'b [u8], todo: &mut W) -> Result<&'b T, Error> {
  if !bytes.as_ptr().addr().is_multiple_of(ALIGN) {
    return Err(Error::Invalid(0));
  }
  check::<T, W>(bytes, todo)?;
  // SAFETY: just checked, at an address aligned as a region is.
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
/// valid `T`, every offset and length an `O` that `from_isize` makes (see
/// [`Offset`](crate::Offset)). That is what [`Region::as_bytes`](crate::Region::as_bytes) returned
/// for a region of `T` in a build with the same layout of `T` (this one, for example), and what a
/// check of the caller's that accepts no more than [`view`] accepted.
#[must_use]
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T {
  fits::<T>();
  // SAFETY: per the caller, the bytes are aligned as a region is, so for `T` (`fits`: `T` is not
  // more aligned), and the image of a valid `T`, whose pointers all aim inside them, at positions
  // aligned for their pointees, so at addresses aligned for them.
  unsafe { &*bytes.as_ptr().cast::<T>() }
}
