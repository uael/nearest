#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{fmt, str};

use crate::{
  Flat,
  flat::{MAX, fits},
};

/// Why bytes could not be loaded: the error of every loader ([`Region::view`](crate::Region::view),
/// [`Region::view_with`](crate::Region::view_with),
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
/// [`Region::view_with`](crate::Region::view_with) and
/// [`Region::from_bytes_with`](crate::Region::from_bytes_with) take a slice of them.
///
/// `bytes.len() / 4` entries are enough for any bytes: each entry waits for its own 4 bytes of
/// the image (a pointer, or the start of the next element of a list), so an image that fits a
/// storage of `N` bytes never needs more than `N / 4`. Most data needs far fewer: about as many
/// as the pointers that wait at once while the walk goes down the deepest path. An entry is 16
/// bytes on a 64-bit target, 12 on a 32-bit one.
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
pub struct Pending(u32, u32, Step);

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

/// A walk over bytes that must be the compact image of a value: the root, then every target in
/// the order a deep copy writes it.
pub struct Check<'b> {
  bytes: &'b [u8],
  /// Where the next target must start (before alignment).
  next: usize,
  /// Pointers and rests of lists still to check; the last one is next.
  todo: Stack<'b>,
}

impl Check<'_> {
  /// The `N` bytes at `at`.
  pub(crate) fn get<const N: usize>(&self, at: usize) -> Result<[u8; N], Error> {
    self.bytes.get(at..).and_then(<[u8]>::first_chunk).copied().ok_or(Error::Invalid(at))
  }

  /// The tag of an enum of `count` variants at `at`.
  ///
  /// # Errors
  /// If it is out of bounds or not below `count`.
  pub fn tag(&self, at: usize, count: usize) -> Result<u8, Error> {
    let [tag] = self.get(at)?;
    if usize::from(tag) < count { Ok(tag) } else { Err(Error::Invalid(at)) }
  }

  /// Queues a pointer, or the rest of a list. If the scratch is full, the walk ends.
  pub(crate) fn push(&mut self, at: usize, count: usize, step: Step) -> Result<(), Error> {
    // Positions and counts are below `MAX`, which the length of the bytes is.
    let item = u32::try_from(at).ok().zip(u32::try_from(count).ok());
    let pushed = item.is_some_and(|(at, count)| self.todo.push(Pending(at, count, step)));
    if pushed { Ok(()) } else { Err(Error::ScratchFull) }
  }

  /// The next block, of `size` bytes aligned to `align`: the pointer at `ptr` must aim `lead`
  /// bytes into it, which is where a deep copy would have put it. Returns its position.
  fn claim(&mut self, ptr: usize, lead: usize, align: usize, size: usize) -> Result<usize, Error> {
    let off = i32::from_ne_bytes(self.get(ptr)?);
    let at = self.next.checked_next_multiple_of(align).ok_or(Error::Invalid(ptr))?;
    let aim = (ptr as u64).wrapping_add_signed(i64::from(off));
    let aimed = at.checked_add(lead).is_some_and(|a| usize::try_from(aim) == Ok(a));
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

  fn u32(&self, at: usize) -> Result<usize, Error> {
    Ok(u32::from_ne_bytes(self.get(at)?) as usize)
  }
}

/// The target of the `Near<T>` at `ptr`.
pub fn target<T: Flat>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  fits::<T>();
  let at = ck.claim(ptr, 0, align_of::<T>(), size_of::<T>())?;
  T::check(ck, at)
}

/// The list block of the `Near<[T]>` at `ptr`: `[len][len]`, then the elements.
pub fn list<T: Flat>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  fits::<T>();
  let at = ck.claim(ptr, 4, align_of::<T>().max(4), 8)?;
  let (cap, len) = (ck.u32(at)?, ck.u32(at + 4)?);
  ck.extend(at, len.checked_mul(size_of::<T>()).filter(|_| cap == len))?;
  items::<T>(ck, at + 8, len)
}

/// The text of the `Near<str>` at `ptr`: `[len]`, then UTF-8.
pub fn text(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Error> {
  let at = ck.claim(ptr, 0, 4, 4)?;
  let len = ck.u32(at)?;
  ck.extend(at, Some(len))?;
  match ck.bytes.get(at + 4..ck.next).map(str::from_utf8) {
    Some(Ok(_)) => Ok(()),
    _ => Err(Error::Invalid(at)),
  }
}

/// The `count` elements from `first`, in order. When one has pointers, they are checked before
/// the remaining elements, which wait on the stack behind them. Elements of which every bit
/// pattern is valid are not walked.
fn items<T: Flat>(ck: &mut Check<'_>, first: usize, count: usize) -> Result<(), Error> {
  if T::ANY {
    return Ok(());
  }
  for i in 0..count {
    let mark = ck.todo.len();
    T::check(ck, first + i * size_of::<T>())?;
    if ck.todo.len() > mark {
      if i + 1 < count {
        ck.push(first + (i + 1) * size_of::<T>(), count - i - 1, items::<T>)?;
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
  if bytes.len() < size_of::<T>() || bytes.len() > MAX {
    return Err(Error::Invalid(0));
  }
  walk::<T>(&mut Check { bytes, next: size_of::<T>(), todo })
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
    step(ck, at as usize, count as usize)?;
  }
  if ck.next == ck.bytes.len() { Ok(()) } else { Err(Error::Invalid(ck.next)) }
}
