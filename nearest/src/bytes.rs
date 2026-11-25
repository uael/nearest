use core::fmt;

use crate::{
  Flat,
  flat::{Sink, fits, limit},
};

/// Why bytes could not be loaded: the error of [`view`] and
/// [`Region::from_bytes`](crate::Region::from_bytes).
///
/// ```
/// use nearest::{Error, Inline, Near, Region, flat, puns};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let r = Region::new(Msg::new(1, "hi"));
/// let bytes = r.as_bytes().unwrap();
/// let cut = Region::<Msg>::from_bytes(&bytes[..bytes.len() - 1]);
/// assert_eq!(cut.err(), Some(Error::Invalid(puns::Error::Validity))); // the text, cut short
/// let small = Region::<Msg, Inline<8>>::from_bytes(bytes);
/// assert_eq!(small.err(), Some(Error::StorageFull)); // a valid image, larger than 8 bytes
/// assert_eq!(Error::StorageFull.to_string(), "region bytes larger than the storage");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
  /// The bytes are not the image of a valid value: what the check of `puns` found
  /// ([`puns::from_head`]). [`puns::Error::Size`] also for bytes longer than the limit of the
  /// root (see [`Offset`](crate::Offset)).
  Invalid(puns::Error),
  /// The bytes are a valid image, larger than the storage of `from_bytes`.
  StorageFull,
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Invalid(error) => write!(f, "invalid region bytes: {error}"),
      Self::StorageFull => f.write_str("region bytes larger than the storage"),
    }
  }
}

impl core::error::Error for Error {}

/// The value at the start of `bytes`, read in place, for example from a memory-mapped file.
///
/// The check is the one of `puns` ([`puns::from_head`]): the slice aligned for `T`, every value
/// and every pointee valid, each pointee within the bytes and aligned for it, at any depth, the
/// pointees in the order of a copy, each after everything before it. Bytes that no pointer reaches
/// are allowed, after the value or between pointees. So two pointers to one pointee, a pointee
/// that overlaps a value, and a cycle are refused. It is `const`, and takes no memory.
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
/// [`Error::Invalid`] with what `puns` found if the bytes do not start with a valid `T` (an
/// address not aligned for `T` is [`puns::Error::Alignment`]), or with [`puns::Error::Size`] if
/// they are longer than the limit of `T`.
pub const fn view<T: Flat>(bytes: &[u8]) -> Result<&T, Error> {
  fits::<T>();
  // A read refuses at compile time what a build refuses: the stores of a value of `T`, which this
  // instantiates, assert at every depth where a pointer may lead (`fits_behind`). A constant
  // cannot hold that assertion: the types that a recursive type reaches lead back to it, a cycle
  // of constants, which Rust refuses (E0391), where functions instantiate each other.
  let _: unsafe fn(&T, &mut Sink, usize) = T::store;
  if bytes.len() > limit::<T>() {
    return Err(Error::Invalid(puns::Error::Size));
  }
  match puns::from_head::<T>(bytes) {
    Ok((value, _)) => Ok(value),
    Err(error) => Err(Error::Invalid(error)),
  }
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
/// [`view::<T>(bytes)`](view) would return `Ok`, in this build: every offset and length is an `O`
/// that `from_isize` makes (see [`Offset`](crate::Offset)). That is what
/// [`Region::as_bytes`](crate::Region::as_bytes) returned for a region of `T` in a build with the
/// same layout of `T` (this one, for example), at an address aligned as a region is, and what a
/// check of the caller's that accepts no more than [`view`] accepted.
#[must_use]
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T {
  fits::<T>();
  // SAFETY: per the caller, `view` would accept the bytes: they are aligned for `T` and start
  // with a valid `T`, whose pointers all aim inside them, at addresses aligned for their
  // pointees.
  unsafe { &*bytes.as_ptr().cast::<T>() }
}
