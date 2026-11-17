use alloc::vec::Vec;
use core::{
  mem::{MaybeUninit, needs_drop},
  num::NonZero,
  ptr, slice,
};

use crate::{
  Invalid, Near,
  bytes::{Check, list, target, text},
  edit::{ListEdit, OptionEdit, Raw, StrEdit},
};

/// Largest region, a multiple of 8: every distance inside it fits the `i32` offset. Small in
/// unit tests, so that the limit can be hit cheaply.
pub const MAX: usize = if cfg!(test) { 1 << 16 } else { i32::MAX as usize & !7 };

const fn align_up(n: usize, align: usize) -> usize {
  (n + align - 1) & !(align - 1)
}

/// `to - from` as an offset; positions are below `MAX`, so it fits.
#[expect(clippy::cast_possible_wrap, reason = "positions are below `MAX`, so the distance fits")]
pub const fn distance(from: usize, to: usize) -> i32 {
  to.wrapping_sub(from) as i32
}

/// Rejects, at compile time, types a region cannot hold.
pub const fn fits<T>() {
  const {
    assert!(align_of::<T>() <= 8, "a region holds only types aligned to at most 8 bytes");
    assert!(size_of::<T>() != 0, "a region holds only types of non-zero size");
    assert!(!needs_drop::<T>(), "a region holds only types without drop glue");
  };
}

/// 8-aligned bytes, zero-filled when they grow.
///
/// Invariants, also while unwinding: `len <= words.len() * 8 <= MAX`, so every position fits
/// an `i32`.
pub struct Buf {
  words: Vec<MaybeUninit<u64>>,
  len: usize,
  /// The address of this `Buf`, written by `Region::edit`. Every access of an edit goes
  /// through it, so that all of them have the same provenance (I8).
  pub(crate) me: *mut Buf,
  /// The bytes are the compact image of the value: initialized, and what a deep copy writes.
  pub(crate) compact: bool,
}

// SAFETY: `me` is only the address of the `Buf` itself, used during an edit, which borrows the
// `Buf` mutably; otherwise a `Buf` is a vector of bytes.
unsafe impl Send for Buf {}
// SAFETY: as above; nothing is written through a shared `Buf`.
unsafe impl Sync for Buf {}

impl Buf {
  pub(crate) const fn new() -> Self {
    Self { words: Vec::new(), len: 0, me: ptr::null_mut(), compact: true }
  }

  /// Bytes in use.
  pub(crate) const fn len(&self) -> usize {
    self.len
  }

  /// Byte 0, for writing.
  pub(crate) const fn ptr(&mut self) -> *mut u8 {
    self.words.as_mut_ptr().cast()
  }

  /// Byte 0, for reading.
  pub(crate) const fn base(&self) -> *const u8 {
    self.words.as_ptr().cast()
  }

  /// The bytes in use.
  ///
  /// # Safety
  /// They are initialized, which is true of a compact buffer.
  pub(crate) const unsafe fn bytes(&self) -> &[u8] {
    // SAFETY: per the caller; `len` bytes are allocated.
    unsafe { slice::from_raw_parts(self.base(), self.len) }
  }

  /// `size` zeroed bytes aligned to `align` (bytes past `len` are never written); returns their
  /// position.
  #[inline]
  pub(crate) fn alloc(&mut self, size: usize, align: usize) -> usize {
    let at = align_up(self.len, align);
    let end = at.saturating_add(size);
    if end > self.words.len() * 8 {
      self.grow(end);
    }
    self.len = end;
    at
  }

  /// Grows to hold `end` bytes, keeping `words.len() * 8 <= MAX`: anything past the limit comes
  /// here and panics, before changing anything.
  #[cold]
  fn grow(&mut self, end: usize) {
    assert!(end <= MAX, "region exceeds 2 GiB");
    let words = end.div_ceil(8).max(self.words.len() * 2).clamp(8, MAX / 8);
    self.words.resize(words, MaybeUninit::zeroed());
  }

  /// A copy of the bytes in use.
  pub(crate) fn to_vec(&self) -> Self {
    let words = self.words[..self.len.div_ceil(8)].to_vec();
    Self { words, len: self.len, me: ptr::null_mut(), compact: self.compact }
  }
}

/// Where values are written: a region's buffer, reached through a raw pointer so that no
/// reference to it outlives a statement.
pub struct Sink {
  buf: *mut Buf,
}

impl Sink {
  /// # Safety
  /// `buf` points to a live `Buf` while the sink is used, and every access to that `Buf` meanwhile
  /// goes through `buf` (its `me` during an edit).
  pub(crate) const unsafe fn new(buf: *mut Buf) -> Self {
    Self { buf }
  }

  pub(crate) const fn buf(&self) -> *mut Buf {
    self.buf
  }

  /// `size` bytes aligned to `align`; returns their position.
  #[inline]
  pub(crate) fn alloc(&mut self, size: usize, align: usize) -> usize {
    // SAFETY: `buf` is live per `new`, and this reborrow ends with the statement.
    unsafe { (*self.buf).alloc(size, align) }
  }

  /// Marks the bytes as not the compact image.
  pub(crate) const fn loose(&mut self) {
    // SAFETY: as for `alloc`.
    unsafe { (*self.buf).compact = false }
  }

  /// # Safety
  /// `at + size_of::<P>()` is within the allocated bytes.
  #[inline]
  pub(crate) const unsafe fn write<P: Copy>(&mut self, at: usize, v: P) {
    // SAFETY: in bounds per the caller.
    unsafe { (*self.buf).ptr().add(at).cast::<P>().write_unaligned(v) }
  }

  /// Points the pointer at `at` to `target`.
  ///
  /// # Safety
  /// `at` holds a pointer field.
  #[inline]
  pub(crate) const unsafe fn link(&mut self, at: usize, target: usize) {
    // SAFETY: forwarded; the distance fits an `i32` because the region is at most `MAX` bytes.
    unsafe { self.write(at, distance(at, target)) }
  }
}

/// The position a pointer at `at` points to.
///
/// # Safety
/// `at` holds a non-null pointer in `buf`.
pub unsafe fn aim(buf: *mut Buf, at: usize) -> usize {
  // SAFETY: per the caller.
  let off = unsafe { (*buf).base().add(at).cast::<i32>().read_unaligned() };
  at.wrapping_add_signed(off as isize)
}

/// Types that may live in a region: plain data, [`Near`] pointers, and types made with
/// [`#[flat]`](crate::flat). Implement it with `#[flat]`, not by hand.
///
/// | Type | [`Edit`](Flat::Edit) |
/// |---|---|
/// | `u8`..`u64`, `i8`..`i64`, `f32`, `f64`, `bool`, `char`, `NonZero` integers | itself |
/// | `[T; N]`, `T: Flat + Copy` | itself |
/// | `Near<T>` | `Edit<'e, T>`, the target's handle |
/// | `Near<[T]>` | [`ListEdit<'e, T>`] |
/// | `Near<str>` | [`StrEdit<'e>`] |
/// | `Option<Near<T>>` | [`OptionEdit<'e, T>`] |
/// | `#[flat]` types | generated |
///
/// Every stored type is `'static`, has alignment at most 8, non-zero size and no drop glue;
/// the last three are checked at compile time.
///
/// ```
/// use nearest::{Flat, Near, flat};
///
/// #[flat]
/// struct Point { x: i32, y: i32 }
///
/// fn assert_flat<T: Flat>() {}
/// assert_flat::<Point>();
/// assert_flat::<Near<[Point]>>();
/// assert_flat::<Option<Near<Point>>>();
/// ```
///
/// # Safety
///
/// `Self` has no drop glue and no pointers other than [`Near`]. `store` writes every field of
/// `self` at `at` as described on it; `PLAIN` is `true` only if `Self` has no padding and no
/// pointers, and `ANY` only if it has no pointers and every bit pattern is a valid `Self`;
/// `edit` returns the handle of a valid `Self` at `at`; `check` accepts exactly the compact
/// images of valid values, in the order `store` writes them.
pub unsafe trait Flat: Sized + 'static {
  /// The handle of a place holding a `Self`, during the edit `'e`: what
  /// [`Region::edit`](crate::Region::edit) and field methods return a `&mut` to. See
  /// [`Edit`](crate::Edit).
  type Edit<'e>: ?Sized + 'e;

  /// No pointers and no padding: may be copied as bytes.
  #[doc(hidden)]
  const PLAIN: bool = false;

  /// No pointers, and every bit pattern is a valid value: there is nothing to check, so lists
  /// of such elements are not walked.
  #[doc(hidden)]
  const ANY: bool = false;

  /// Writes `self` at `at`. With `shift == None` every pointer target is copied too; with
  /// `Some(d)` every pointer keeps its target and is re-aimed for a move of `d` bytes, and no
  /// pointer is dereferenced.
  ///
  /// # Safety
  /// `at` is allocated in the sink, aligned and sized for `Self`. With `None`, `self` is not in
  /// the sink's buffer.
  #[doc(hidden)]
  unsafe fn store(&self, sink: &mut Sink, at: usize, shift: Option<i32>);

  /// The handle of the `Self` at `at`.
  ///
  /// # Safety
  /// `buf` is the `me` of a `Buf` holding a valid `Self` at `at`, borrowed for the edit `'e`;
  /// the place is only reached through the returned handle for `'a`.
  #[doc(hidden)]
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self::Edit<'e>;

  /// Checks the `Self` at `at`, whose own bytes are in bounds, and queues its pointers.
  #[doc(hidden)]
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid>;
}

/// A deep copy of `src` at `at`.
///
/// # Safety
/// As for [`Flat::store`] with `None`.
pub unsafe fn write_copy<T: Flat>(src: &T, sink: &mut Sink, at: usize) {
  // SAFETY: forwarded, for one element.
  unsafe { write_slice(slice::from_ref(src), sink, at) }
}

/// A deep copy of the elements of `src` at `at`.
///
/// # Safety
/// As for [`Flat::store`] with `None`, with room for `src.len()` elements.
unsafe fn write_slice<T: Flat>(src: &[T], sink: &mut Sink, at: usize) {
  // SAFETY: plain data is its bytes, and the destination is sized for it; other elements are
  // stored one by one.
  unsafe {
    if T::PLAIN {
      let dst = (*sink.buf).ptr().add(at);
      ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), dst, size_of_val(src));
    } else {
      for (i, x) in src.iter().enumerate() {
        x.store(sink, at + i * size_of::<T>(), None);
      }
    }
  }
}

/// Writes at `to` the value `v` that was read from `from`, re-aiming its pointers.
///
/// # Safety
/// `v` is a bitwise copy of the `T` at `from` in `buf`, and `to` is allocated for a `T`.
pub unsafe fn put_moved<T: Flat>(buf: *mut Buf, v: &T, from: usize, to: usize) {
  // SAFETY: per the caller; a move never dereferences a pointer, so `v` may live anywhere.
  unsafe { v.store(&mut Sink::new(buf), to, Some(distance(to, from))) }
}

/// Moves the `T` at `from` to `to`, re-aiming its pointers. The two may overlap.
///
/// # Safety
/// A `T` is at `from` in `buf`, and `to` is allocated for a `T`.
pub unsafe fn relocate<T: Flat>(buf: *mut Buf, from: usize, to: usize) {
  // SAFETY: per the caller; the value is read before anything is written.
  unsafe {
    let p = (*buf).ptr();
    if T::PLAIN {
      ptr::copy(p.add(from), p.add(to), size_of::<T>());
    } else {
      let v = p.add(from).cast::<T>().read();
      put_moved(buf, &v, from, to);
    }
  }
}

/// Moves the `T` at `at` to fresh bytes, where nothing reaches it; returns their position.
///
/// # Safety
/// A `T` is at `at` in `buf`.
pub unsafe fn detach<T: Flat>(buf: *mut Buf, at: usize) -> usize {
  // SAFETY: per the caller; the fresh bytes are allocated before anything is read.
  unsafe {
    let kept = (*buf).alloc(size_of::<T>(), align_of::<T>());
    relocate::<T>(buf, at, kept);
    kept
  }
}

/// Moves `n` consecutive `T`s from `from` to `to`; the ranges may overlap.
///
/// # Safety
/// `n` `T`s are at `from` in `buf`, and `to` is allocated for `n` of them.
pub unsafe fn move_elems<T: Flat>(buf: *mut Buf, from: usize, to: usize, n: usize) {
  let size = size_of::<T>();
  // SAFETY: per the caller; with overlap, every element is read before it is overwritten.
  unsafe {
    if T::PLAIN {
      let p = (*buf).ptr();
      ptr::copy(p.add(from), p.add(to), n * size);
    } else if to < from {
      for i in 0..n {
        relocate::<T>(buf, from + i * size, to + i * size);
      }
    } else {
      for i in (0..n).rev() {
        relocate::<T>(buf, from + i * size, to + i * size);
      }
    }
  }
}

/// The handle of the plain `T` at `at`.
///
/// # Safety
/// As for [`Flat::edit`].
const unsafe fn plain_edit<'a, T>(buf: *mut Buf, at: usize) -> &'a mut T {
  // SAFETY: per the caller.
  unsafe { &mut *(*buf).ptr().add(at).cast::<T>() }
}

macro_rules! plain {
  ($($any:literal: $($t:ty => |$b:ident: $raw:ty| $ok:expr),*;)*) => {$($(
    // SAFETY: primitive values have no pointers and no padding; `check` accepts the valid ones,
    // which for the `ANY` types is every bit pattern.
    unsafe impl Flat for $t {
      type Edit<'e> = Self;
      const PLAIN: bool = true;
      const ANY: bool = $any;
      unsafe fn store(&self, sink: &mut Sink, at: usize, _: Option<i32>) {
        // SAFETY: the caller allocated `at` for `Self`.
        unsafe { sink.write(at, *self) }
      }
      unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
        // SAFETY: forwarded.
        unsafe { plain_edit(buf, at) }
      }
      fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
        let $b = <$raw>::from_ne_bytes(ck.get(at)?);
        if $ok { Ok(()) } else { Err(Invalid(at)) }
      }
    }
  )*)*};
}
plain! {
  true:
    u8 => |_b: u8| true, u16 => |_b: u16| true, u32 => |_b: u32| true, u64 => |_b: u64| true,
    i8 => |_b: u8| true, i16 => |_b: u16| true, i32 => |_b: u32| true, i64 => |_b: u64| true,
    f32 => |_b: u32| true, f64 => |_b: u64| true;
  false:
    bool => |b: u8| b <= 1,
    char => |b: u32| char::from_u32(b).is_some(),
    NonZero<u8> => |b: u8| b != 0, NonZero<u16> => |b: u16| b != 0,
    NonZero<u32> => |b: u32| b != 0, NonZero<u64> => |b: u64| b != 0,
    NonZero<i8> => |b: u8| b != 0, NonZero<i16> => |b: u16| b != 0,
    NonZero<i32> => |b: u32| b != 0, NonZero<i64> => |b: u64| b != 0;
}

// SAFETY: an array is its elements, without padding between them; `Copy` elements have no
// pointers.
unsafe impl<T: Flat + Copy, const N: usize> Flat for [T; N] {
  type Edit<'e> = Self;
  const PLAIN: bool = {
    fits::<Self>();
    T::PLAIN
  };
  const ANY: bool = T::ANY;
  unsafe fn store(&self, sink: &mut Sink, at: usize, _: Option<i32>) {
    fits::<Self>();
    // SAFETY: the caller allocated `at` for the array.
    unsafe { write_slice(self, sink, at) }
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
    fits::<Self>();
    // SAFETY: forwarded.
    unsafe { plain_edit(buf, at) }
  }
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
    fits::<Self>();
    if T::ANY {
      return Ok(());
    }
    (0..N).try_for_each(|i| T::check(ck, at + i * size_of::<T>()))
  }
}

impl<T: ?Sized> Near<T> {
  /// Writes this pointer at `at` for a value that moved by `d` bytes: it keeps its target.
  ///
  /// # Safety
  /// `at` is allocated in the sink for a `Near`.
  const unsafe fn moved(&self, sink: &mut Sink, at: usize, d: i32) {
    // SAFETY: per the caller.
    unsafe { sink.write(at, self.off.get().wrapping_add(d)) }
  }
}

// SAFETY: copies the target, or re-aims the offset by the distance moved.
unsafe impl<T: Flat> Flat for Near<T> {
  type Edit<'e> = T::Edit<'e>;
  unsafe fn store(&self, sink: &mut Sink, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`; the target is allocated before the
    // source is read.
    unsafe {
      if let Some(d) = shift {
        self.moved(sink, at, d);
      } else {
        fits::<T>();
        let t = sink.alloc(size_of::<T>(), align_of::<T>());
        write_copy(&**self, sink, t);
        sink.link(at, t);
      }
    }
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut T::Edit<'e> {
    fits::<T>();
    // SAFETY: a `Near` at `at` points to a valid `T`, owned by it alone.
    unsafe { T::edit(buf, aim(buf, at)) }
  }
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
    ck.push(at, 0, target::<T>);
    Ok(())
  }
}

// SAFETY: copies the list, or re-aims the offset by the distance moved.
unsafe impl<T: Flat> Flat for Near<[T]> {
  type Edit<'e> = ListEdit<'e, T>;
  unsafe fn store(&self, sink: &mut Sink, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe {
      match shift {
        Some(d) => self.moved(sink, at, d),
        None => Emit::<Self>::emit(&**self, sink, at),
      }
    }
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut ListEdit<'e, T> {
    fits::<T>();
    // SAFETY: forwarded.
    unsafe { ListEdit::new(Raw::new(buf, at)) }
  }
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
    ck.push(at, 0, list::<T>);
    Ok(())
  }
}

// SAFETY: copies the text, or re-aims the offset by the distance moved.
unsafe impl Flat for Near<str> {
  type Edit<'e> = StrEdit<'e>;
  unsafe fn store(&self, sink: &mut Sink, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe {
      match shift {
        Some(d) => self.moved(sink, at, d),
        None => copy_str(sink, at, self),
      }
    }
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut StrEdit<'e> {
    // SAFETY: forwarded.
    unsafe { StrEdit::new(Raw::new(buf, at)) }
  }
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
    ck.push(at, 0, text);
    Ok(())
  }
}

// SAFETY: `None` is a zero offset; `Some` is the `Near`.
unsafe impl<T: Flat> Flat for Option<Near<T>> {
  type Edit<'e> = OptionEdit<'e, T>;
  unsafe fn store(&self, sink: &mut Sink, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe {
      match self {
        Some(near) => near.store(sink, at, shift),
        None => sink.write(at, 0i32),
      }
    }
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut OptionEdit<'e, T> {
    fits::<T>();
    // SAFETY: forwarded.
    unsafe { OptionEdit::new(Raw::new(buf, at)) }
  }
  fn check(ck: &mut Check<'_>, at: usize) -> Result<(), Invalid> {
    if i32::from_ne_bytes(ck.get(at)?) != 0 {
      ck.push(at, 0, target::<T>);
    }
    Ok(())
  }
}

/// Writes at `at` a pointer to a copy of `s`.
///
/// # Safety
/// `at` holds a `Near<str>` in the sink.
unsafe fn copy_str(sink: &mut Sink, at: usize, s: &str) {
  let h = sink.alloc(s.len().saturating_add(4), 4);
  // SAFETY: `h` was just allocated for the length and the bytes.
  unsafe {
    sink.write(h, s.len() as u32);
    ptr::copy_nonoverlapping(s.as_ptr(), (*sink.buf).ptr().add(h + 4), s.len());
    sink.link(at, h);
  }
}

/// Sources that can be written as a `T`: values, copies, moves and builders.
///
/// | Target | Builders |
/// |---|---|
/// | any `T` | `T` by value, `&T` (a deep copy, from any region), [`Old<'_, T>`](crate::Old) (a move) |
/// | `Near<T>`, `Option<Near<T>>` | [`Near::new(e)`](Near::new) |
/// | `Near<[T]>` | `[e; N]`, tuples up to 12 and `()`, `&[T]`, [`Near::from_iter(iter)`](Near::from_iter) |
/// | `Near<str>` | `&str` |
/// | `#[flat]` types | the generated constructors |
///
/// Constructor parameters for fields written `Near<X>` take a builder of `X` and allocate the
/// target themselves.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Doc { title: Near<str>, tags: Near<[u16]>, parent: Option<Near<Doc>> }
///
/// let a = Region::new(Doc::new("a", [1, 2], None));
/// let b = Region::new(Doc::new("b", &a.tags[..], Near::new(&*a)));
/// assert_eq!(b.parent.as_ref().map(|p| &*p.title), Some("a"));
/// assert_eq!(b.tags[..], [1, 2]);
/// ```
///
/// # Safety
///
/// `emit` writes a valid `T` at `at`, allocating whatever it points to.
pub unsafe trait Emit<T: Flat> {
  /// Writes `self` as a `T` at `at`.
  ///
  /// # Safety
  /// `at` is allocated in the sink, aligned and sized for `T`.
  #[doc(hidden)]
  unsafe fn emit(self, sink: &mut Sink, at: usize);

  /// `self` as a `T` that is its own bytes (`PLAIN`), if it is one; `emit` then only copies
  /// those bytes. Lets a list of numbers be filled without going through the sink.
  #[doc(hidden)]
  fn plain(&self) -> Option<&T> {
    None
  }
}

// SAFETY: a value is stored as itself; it is not in any region.
unsafe impl<T: Flat> Emit<T> for T {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { write_copy(&self, sink, at) }
  }
  fn plain(&self) -> Option<&T> {
    T::PLAIN.then_some(self)
  }
}

// SAFETY: a deep copy is a valid `T`. A `&T` into a region cannot coexist with a handle of that
// region, so it is not in the sink's buffer.
unsafe impl<T: Flat> Emit<T> for &T {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { write_copy(self, sink, at) }
  }
  fn plain(&self) -> Option<&T> {
    T::PLAIN.then_some(*self)
  }
}

/// Builder made by [`Near::new`].
pub struct New<E>(pub E);

// SAFETY: allocates and writes the target, then links to it.
unsafe impl<T: Flat, E: Emit<T>> Emit<Near<T>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    fits::<T>();
    let t = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `t` was just allocated for a `T`.
    unsafe {
      self.0.emit(sink, t);
      sink.link(at, t);
    }
  }
}

// SAFETY: `Some(near)` has the bytes of `near`.
unsafe impl<T: Flat, E: Emit<T>> Emit<Option<Near<T>>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { Emit::<Near<T>>::emit(self, sink, at) }
  }
}

// SAFETY: copies the text.
unsafe impl Emit<Near<str>> for &str {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { copy_str(sink, at, self) }
  }
}

// SAFETY: copies the slice, which is not in the sink's buffer (as for `&T`).
unsafe impl<T: Flat> Emit<Near<[T]>> for &[T] {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    let mut list = List::new::<T>(sink, self.len());
    // SAFETY: the list has room for the elements.
    unsafe {
      write_slice(self, sink, list.elem::<T>(0));
      list.len = self.len();
      list.finish(sink, at);
    }
  }
}

/// Builder made by [`Near::from_iter`].
pub struct Seq<I>(pub I);

// SAFETY: writes the elements into a list, then links to it.
unsafe impl<T: Flat, I: IntoIterator<Item: Emit<T>>> Emit<Near<[T]>> for Seq<I> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    let items = self.0.into_iter();
    let mut list = List::new::<T>(sink, items.size_hint().0);
    // SAFETY: a new block, filled by `extend`.
    unsafe {
      list.extend(sink, items);
      list.finish(sink, at);
    }
  }
}

// SAFETY: as for `Seq`.
unsafe impl<T: Flat, E: Emit<T>, const N: usize> Emit<Near<[T]>> for [E; N] {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { Seq(self).emit(sink, at) }
  }
}

/// Implements `Emit` for tuples of every length from the given one down to `()`.
macro_rules! tuples {
  ($($e:ident $x:ident)*) => {
    // SAFETY: as for `Seq`.
    unsafe impl<T: Flat, $($e: Emit<T>),*> Emit<Near<[T]>> for ($($e,)*) {
      unsafe fn emit(self, sink: &mut Sink, at: usize) {
        let ($($x,)*) = self;
        let list = &mut List::new::<T>(sink, 0 $(+ { let _ = &$x; 1 })*);
        // SAFETY: a new block, filled by `push`.
        unsafe {
          $(list.push(sink, $x);)*
          list.finish(sink, at);
        }
      }
    }
    tuples!(@ $($e $x)*);
  };
  (@) => {};
  (@ $e:ident $x:ident $($rest:tt)*) => { tuples!($($rest)*); };
}
tuples!(A a B b C c D d E e F f G g H h I i J j K k L l);

/// A list block `[cap][len][elements]` in a buffer: `head` is the position of `len`, and `len`
/// and `cap` are the values in use (the builder writes them to the header in `finish`).
pub struct List {
  pub head: usize,
  pub len: usize,
  pub cap: usize,
}

impl List {
  /// A new block with room for `cap` elements, its header `[cap][0]`.
  pub fn new<T: Flat>(sink: &mut Sink, cap: usize) -> Self {
    fits::<T>();
    let bytes = size_of::<T>().saturating_mul(cap).saturating_add(8);
    let start = sink.alloc(bytes, align_of::<T>().max(4));
    // SAFETY: the header was just allocated; `cap` fits a `u32` because the block does.
    unsafe { sink.write(start, cap as u32) };
    Self { head: start + 4, len: 0, cap }
  }

  /// The list the pointer at `at` points to.
  ///
  /// # Safety
  /// `at` holds a `Near<[T]>` in `buf`.
  pub unsafe fn read(buf: *mut Buf, at: usize) -> Self {
    // SAFETY: the pointer leads to a list header.
    unsafe {
      let head = aim(buf, at);
      let header = |p: usize| (*buf).base().add(p).cast::<u32>().read_unaligned() as usize;
      Self { head, len: header(head), cap: header(head - 4) }
    }
  }

  /// Position of element `i`.
  pub const fn elem<T>(&self, i: usize) -> usize {
    self.head + 4 + i * size_of::<T>()
  }

  /// Makes room for one more element. A full list moves to a new block of twice the capacity
  /// (at least 4), whose header gets the length; the old block becomes garbage. Returns whether
  /// the list moved.
  ///
  /// # Safety
  /// `self` is a block of `T`s in the sink's buffer whose first `len` elements are valid.
  pub unsafe fn reserve<T: Flat>(&mut self, sink: &mut Sink) -> bool {
    if self.len < self.cap {
      return false;
    }
    let cap = (self.cap * 2).max(4);
    let new = Self::new::<T>(sink, cap);
    sink.loose();
    // SAFETY: per the caller, and the new block has room for the elements.
    unsafe {
      move_elems::<T>(sink.buf, self.elem::<T>(0), new.elem::<T>(0), self.len);
      sink.write(new.head, self.len as u32);
    }
    (self.head, self.cap) = (new.head, cap);
    true
  }

  /// Appends `item`, growing the block if it is full.
  ///
  /// # Safety
  /// As for [`reserve`](Self::reserve).
  unsafe fn push<T: Flat>(&mut self, sink: &mut Sink, item: impl Emit<T>) {
    // SAFETY: per the caller; element `len` is within the room.
    unsafe {
      self.reserve::<T>(sink);
      item.emit(sink, self.elem::<T>(self.len));
    }
    self.len += 1;
  }

  /// Appends the items.
  ///
  /// # Safety
  /// As for [`reserve`](Self::reserve).
  unsafe fn extend<T: Flat>(&mut self, sink: &mut Sink, mut items: impl Iterator<Item: Emit<T>>) {
    loop {
      // Fast path: fill the room in one internal iteration, with no growth check. Plain values
      // are written straight through `base`, which is read again after anything else, since
      // that may have grown the buffer; the iterator itself cannot reach the buffer.
      let mut at = self.elem::<T>(self.len);
      let mut n = 0;
      // SAFETY: the sink's buffer is live.
      let mut base = unsafe { (*sink.buf).ptr() };
      items.by_ref().take(self.cap - self.len).for_each(|item| {
        // SAFETY: `at` is an element of the room, sized for a `T`.
        unsafe {
          if let Some(v) = item.plain() {
            ptr::copy_nonoverlapping(ptr::from_ref(v).cast::<u8>(), base.add(at), size_of::<T>());
          } else {
            item.emit(sink, at);
            base = (*sink.buf).ptr();
          }
        }
        at += size_of::<T>();
        n += 1;
      });
      self.len += n;
      if self.len < self.cap {
        return;
      }
      let Some(item) = items.next() else { return };
      // SAFETY: per the caller.
      unsafe { self.push(sink, item) };
    }
  }

  /// Writes the header of a built list, `[cap][len]`, and links `at` to it. A list with spare
  /// room is not part of a compact image.
  ///
  /// # Safety
  /// `self` is a block of `T`s in the sink's buffer whose first `len` elements are valid, and
  /// `at` holds a `Near<[T]>` there.
  pub const unsafe fn finish(&self, sink: &mut Sink, at: usize) {
    if self.cap != self.len {
      sink.loose();
    }
    // SAFETY: per the caller.
    unsafe {
      sink.write(self.head - 4, self.cap as u32);
      sink.write(self.head, self.len as u32);
      sink.link(at, self.head);
    }
  }
}

/// Wraps a closure that writes a `T`; used by the constructors `#[flat]` generates.
///
/// # Safety
/// `f(sink, at)` writes a valid `T` at `at`, given the guarantees of [`Emit::emit`].
#[doc(hidden)]
pub unsafe fn make<T: Flat>(f: impl FnOnce(&mut Sink, usize)) -> impl Emit<T> {
  struct Make<F>(F);
  // SAFETY: the caller of `make` vouches for `F`.
  unsafe impl<T: Flat, F: FnOnce(&mut Sink, usize)> Emit<T> for Make<F> {
    unsafe fn emit(self, sink: &mut Sink, at: usize) {
      (self.0)(sink, at);
    }
  }
  Make(f)
}
