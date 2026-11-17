use alloc::vec::Vec;
use core::{error::Error, fmt, str};

use crate::{
  Flat,
  flat::{MAX, fits},
};

/// The bytes given to [`Region::from_bytes`](crate::Region::from_bytes) or
/// [`Region::view`](crate::Region::view) are not the compact image of a valid value.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Msg { id: u8, body: Near<str> }
///
/// let bytes = Region::new(Msg::new(1, "hi")).to_bytes().into_owned();
/// let err = Region::<Msg>::from_bytes(&bytes[..bytes.len() - 1]).err().unwrap();
/// assert!(err.to_string().starts_with("invalid region bytes"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid(pub(crate) usize);

impl fmt::Display for Invalid {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "invalid region bytes at offset {}", self.0)
  }
}

impl Error for Invalid {}

/// What checks a pointer: given the checker, the position of the pointer (or of the first
/// element, for the rest of a list) and a count.
type Step = fn(&mut Check<'_>, usize, usize) -> Result<(), Invalid>;

/// A walk over bytes that must be the compact image of a value: the root, then every target in
/// the order a deep copy writes it.
pub struct Check<'b> {
  bytes: &'b [u8],
  /// Where the next target must start (before alignment).
  next: usize,
  /// Pointers and rests of lists still to check; the last one is next.
  todo: Vec<(usize, usize, Step)>,
}

impl Check<'_> {
  /// The `N` bytes at `at`.
  pub(crate) fn get<const N: usize>(&self, at: usize) -> Result<[u8; N], Invalid> {
    self.bytes.get(at..).and_then(<[u8]>::first_chunk).copied().ok_or(Invalid(at))
  }

  /// The tag of an enum of `count` variants at `at`.
  ///
  /// # Errors
  /// If it is out of bounds or not below `count`.
  pub fn tag(&self, at: usize, count: usize) -> Result<u8, Invalid> {
    let [tag] = self.get(at)?;
    if usize::from(tag) < count { Ok(tag) } else { Err(Invalid(at)) }
  }

  pub(crate) fn push(&mut self, at: usize, count: usize, step: Step) {
    self.todo.push((at, count, step));
  }

  /// The next block, of `size` bytes aligned to `align`: the pointer at `ptr` must aim `lead`
  /// bytes into it, which is where a deep copy would have put it. Returns its position.
  fn claim(
    &mut self,
    ptr: usize,
    lead: usize,
    align: usize,
    size: usize,
  ) -> Result<usize, Invalid> {
    let off = i32::from_ne_bytes(self.get(ptr)?);
    let at = self.next.checked_next_multiple_of(align).ok_or(Invalid(ptr))?;
    let aim = (ptr as u64).wrapping_add_signed(i64::from(off));
    let aimed = at.checked_add(lead).is_some_and(|a| usize::try_from(aim) == Ok(a));
    let end =
      (at.checked_add(size).filter(|&end| aimed && end <= self.bytes.len())).ok_or(Invalid(ptr))?;
    self.next = end;
    Ok(at)
  }

  /// Takes `size` more bytes after the last block.
  fn extend(&mut self, at: usize, size: Option<usize>) -> Result<(), Invalid> {
    let end = size.and_then(|s| self.next.checked_add(s)).filter(|&end| end <= self.bytes.len());
    self.next = end.ok_or(Invalid(at))?;
    Ok(())
  }

  fn u32(&self, at: usize) -> Result<usize, Invalid> {
    Ok(u32::from_ne_bytes(self.get(at)?) as usize)
  }
}

/// The target of the `Near<T>` at `ptr`.
pub fn target<T: Flat>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Invalid> {
  fits::<T>();
  let at = ck.claim(ptr, 0, align_of::<T>(), size_of::<T>())?;
  T::check(ck, at)
}

/// The list block of the `Near<[T]>` at `ptr`: `[len][len]`, then the elements.
pub fn list<T: Flat>(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Invalid> {
  fits::<T>();
  let at = ck.claim(ptr, 4, align_of::<T>().max(4), 8)?;
  let (cap, len) = (ck.u32(at)?, ck.u32(at + 4)?);
  ck.extend(at, len.checked_mul(size_of::<T>()).filter(|_| cap == len))?;
  items::<T>(ck, at + 8, len)
}

/// The text of the `Near<str>` at `ptr`: `[len]`, then UTF-8.
pub fn text(ck: &mut Check<'_>, ptr: usize, _: usize) -> Result<(), Invalid> {
  let at = ck.claim(ptr, 0, 4, 4)?;
  let len = ck.u32(at)?;
  ck.extend(at, Some(len))?;
  match ck.bytes.get(at + 4..ck.next).map(str::from_utf8) {
    Some(Ok(_)) => Ok(()),
    _ => Err(Invalid(at)),
  }
}

/// The `count` elements from `first`, in order. When one has pointers, they are checked before
/// the remaining elements, which wait on the stack behind them. Elements of which every bit
/// pattern is valid are not walked.
fn items<T: Flat>(ck: &mut Check<'_>, first: usize, count: usize) -> Result<(), Invalid> {
  if T::ANY {
    return Ok(());
  }
  for i in 0..count {
    let mark = ck.todo.len();
    T::check(ck, first + i * size_of::<T>())?;
    if ck.todo.len() > mark {
      if i + 1 < count {
        ck.push(first + (i + 1) * size_of::<T>(), count - i - 1, items::<T>);
      }
      break;
    }
  }
  Ok(())
}

/// Checks that `bytes` are the compact image of a `T`, without recursion or panics. Like every
/// way into a region, it rejects at compile time the types a region cannot hold.
pub fn check<T: Flat>(bytes: &[u8]) -> Result<(), Invalid> {
  fits::<T>();
  if bytes.len() < size_of::<T>() || bytes.len() > MAX {
    return Err(Invalid(0));
  }
  let mut ck = Check { bytes, next: size_of::<T>(), todo: Vec::new() };
  T::check(&mut ck, 0)?;
  let mut mark = 0;
  loop {
    // What the last step queued is checked first-queued first.
    if let Some(queued) = ck.todo.get_mut(mark..) {
      queued.reverse();
    }
    let Some((at, count, step)) = ck.todo.pop() else { break };
    mark = ck.todo.len();
    step(&mut ck, at, count)?;
  }
  if ck.next == bytes.len() { Ok(()) } else { Err(Invalid(ck.next)) }
}
