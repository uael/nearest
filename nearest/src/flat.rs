use core::{
  marker::PhantomData,
  mem::{MaybeUninit, needs_drop},
  num::NonZero,
  ptr,
};

use puns::Pun;

use crate::{
  Near, Offset, Storage,
  edit::{ListEdit, OptionEdit, Raw, StrEdit},
  near::offset,
};

/// The alignment of a region: that of the words of its memory (a [`Storage`] gives
/// `[MaybeUninit<u64>]`), so of byte 0 of its buffer and of the bytes that `as_bytes` returns. It
/// is that of the target, not 8: 8 on x86-64, 4 on `i686`, 2 on `msp430`. A region holds no type
/// more aligned ([`fits`]), and `view` wants an address so aligned. Every typed access of the
/// buffer is at an address `ALIGN`-aligned plus a position aligned for the type, which is then
/// aligned for it because the type is not more aligned than `ALIGN`.
pub const ALIGN: usize = align_of::<u64>();

/// The bytes of a word of memory.
pub const WORD: usize = size_of::<u64>();

/// The largest alignment of a position in a region: that of a stored type (at most [`ALIGN`]) or
/// of the elements of a list (at most the width of its offset, at most 8). The limits of a region
/// and of an `Offset` are multiples of it, so that a position aligned up from below a limit stays
/// below it. It is about positions, never addresses: a position aligned to 8 is at an address
/// aligned to `ALIGN` only.
pub const GRAIN: usize = 8;

const _: () = {
  assert!(ALIGN.is_power_of_two() && ALIGN <= GRAIN, "a word is aligned to at most 8 bytes");
  assert!(WORD == 8, "a word is 8 bytes");
};

/// The largest region with a `T` at its root, a multiple of [`GRAIN`]: what the pointers of the
/// root reach, and at most `isize::MAX` bytes, like any allocation. Every pointer below the root
/// is at least as wide (see [`fits_behind`]), so every distance inside the region fits its
/// pointer.
#[expect(clippy::cast_possible_truncation, reason = "at most `isize::MAX`")]
pub const fn limit<T: Flat>() -> usize {
  let most = isize::MAX as u64 & !(GRAIN as u64 - 1);
  (if T::REACH < most { T::REACH } else { most }) as usize
}

const fn align_up(n: usize, align: usize) -> usize {
  (n + align - 1) & !(align - 1)
}

/// `to - from`; positions are at most `isize::MAX`, so it fits.
#[expect(clippy::cast_possible_wrap, reason = "positions are at most `isize::MAX`")]
pub const fn distance(from: usize, to: usize) -> isize {
  to.wrapping_sub(from) as isize
}

/// Whether a region can hold a `T` at an address aligned for it: `T` is not more aligned than the
/// buffer, [`ALIGN`].
pub const fn aligned<T>() -> bool {
  align_of::<T>() <= ALIGN
}

/// Rejects, at compile time, types a region cannot hold.
pub const fn fits<T>() {
  const {
    assert!(
      aligned::<T>(),
      "a region holds only types aligned to at most `align_of::<u64>()`, the alignment of its words"
    );
    assert!(size_of::<T>() != 0, "a region holds only types of non-zero size");
    assert!(!needs_drop::<T>(), "a region holds only types without drop glue");
  };
}

/// Rejects, at compile time, a `T` that a pointer of `O` cannot lead to: one that a region
/// cannot hold, or that holds a narrower pointer. The pointers of the root bound the region,
/// so pointers may only get wider going down.
pub const fn fits_behind<T: Flat, O: Offset>() {
  fits::<T>();
  offset::<O>();
  const {
    assert!(T::REACH >= O::LIMIT, "a `Near` cannot lead to a type holding a narrower `Near`");
  };
}

/// The bytes of a region: the memory of its storage, of which the first `len` are in use.
///
/// A storage may move with its region, so every use of the region through `&mut` attaches the
/// storage first; the buffer then has its address, and grows it through `regrow`.
///
/// The region zeroes the memory itself, as it uses it, so a storage may hand out memory of any
/// content: `cap` is the zeroed prefix, which `alloc` hands out without a call, and `mem` the
/// whole memory.
///
/// Invariants, also while unwinding: `len <= cap <= mem <= max`, so every distance fits every
/// pointer of the region; the bytes in `[len, cap)` are zero, and those in `[cap, mem)`
/// anything. While attached, `ptr` is valid for `mem` bytes, and `store` is the storage, of the
/// type `regrow` was made for.
pub struct Buf {
  /// Byte 0.
  ptr: *mut u8,
  /// Bytes in use or zero: `alloc` grows nothing below it.
  cap: usize,
  len: usize,
  /// Bytes of memory at `ptr`, at most `max`.
  mem: usize,
  /// The limit of the region (see [`limit`]).
  max: usize,
  /// The storage.
  store: *mut (),
  /// Grows the storage to a number of words, and attaches it again.
  regrow: unsafe fn(&mut Buf, usize),
  /// The address of this `Buf`, written by `Region::edit`. Every access of an edit goes
  /// through it, so that all of them have the same provenance (I8).
  pub(crate) me: *mut Buf,
  /// The bytes are the compact image of the value: initialized, and what a deep copy writes.
  pub(crate) compact: bool,
}

// SAFETY: `ptr` and `store` are the region's own storage and `me` the address of the `Buf`
// itself, all used while the region is borrowed mutably; a region is `Send` or `Sync` as its
// storage is.
unsafe impl Send for Buf {}
// SAFETY: as above; nothing is written through a shared `Buf`.
unsafe impl Sync for Buf {}

/// `regrow` of a buffer that has no storage yet: nothing grows.
const fn unattached(_: &mut Buf, _: usize) {}

/// Asks the storage of `buf`, an `S`, for at least `words` words, and attaches it again: also
/// if that unwinds, since the storage was borrowed anew and pointers into it are stale.
///
/// # Safety
/// `buf` is attached to an `S`.
unsafe fn regrow<S: Storage>(buf: &mut Buf, words: usize) {
  /// Attaches the storage again, as it is, if growing unwinds.
  struct Again<'b, S: Storage>(&'b mut Buf, PhantomData<S>);
  impl<S: Storage> Drop for Again<'_, S> {
    fn drop(&mut self) {
      // SAFETY: the buffer is still attached to the same storage, which is where it was.
      unsafe { self.0.attach(self.0.store.cast::<S>(), 0) }
    }
  }
  let store = buf.store.cast::<S>();
  let again = Again::<S>(buf, PhantomData);
  // SAFETY: per the caller.
  unsafe { again.0.attach(store, words) };
  core::mem::forget(again);
}

impl Buf {
  /// An empty buffer, attached to no storage, for a region of at most `max` bytes, a multiple
  /// of [`GRAIN`] that every distance fits.
  pub(crate) const fn new(max: usize) -> Self {
    Self {
      ptr: ptr::null_mut(),
      cap: 0,
      len: 0,
      mem: 0,
      max,
      store: ptr::null_mut(),
      regrow: unattached,
      me: ptr::null_mut(),
      compact: true,
    }
  }

  /// Attaches the storage at `store`, where it is now, asking it for `words` words (0: its
  /// memory as it is). `cap` and `len` are lengths, so they stay as they are when the storage
  /// moves.
  ///
  /// # Safety
  /// `store` is valid, its memory holds this buffer's `cap` bytes, and it is reached only
  /// through this buffer, and stays where it is, while the buffer is used.
  pub(crate) unsafe fn attach<S: Storage>(&mut self, store: *mut S, words: usize) {
    // SAFETY: per the caller; this borrow of the storage ends with the statement.
    let memory = unsafe { (*store).memory_mut(words) };
    self.ptr = memory.as_mut_ptr().cast();
    self.mem = memory.len().saturating_mul(WORD).min(self.max);
    self.store = store.cast();
    self.regrow = regrow::<S>;
  }

  /// Bytes in use.
  pub(crate) const fn len(&self) -> usize {
    self.len
  }

  /// Byte 0, for writing.
  pub(crate) const fn ptr(&self) -> *mut u8 {
    self.ptr
  }

  /// Byte 0, for reading.
  pub(crate) const fn base(&self) -> *const u8 {
    self.ptr
  }

  /// The limit of the region.
  pub(crate) const fn max(&self) -> usize {
    self.max
  }

  /// `size` zeroed bytes (bytes past `len` are never written) whose byte `inset` is aligned to
  /// `align`; returns their position.
  ///
  /// # Safety
  /// The buffer is attached, and its storage is where it was then.
  #[inline]
  pub(crate) unsafe fn alloc(&mut self, size: usize, align: usize, inset: usize) -> usize {
    let at = align_up(self.len + inset, align) - inset;
    let end = at.saturating_add(size);
    if end > self.cap {
      // SAFETY: forwarded.
      unsafe { self.grow(end) };
    }
    self.len = end;
    at
  }

  /// Makes `end` bytes available: zeroes the memory past `cap`, twice as much as is in use and at
  /// least 64 bytes, asking the storage for more memory first if it is shorter than `end`.
  /// Anything past the limit, or past what the storage gives, comes here and panics, before
  /// anything changes.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc).
  #[cold]
  unsafe fn grow(&mut self, end: usize) {
    assert!(end <= self.max, "region exceeds what its pointers reach");
    // Not `clamp`: a limit below 64 (an `Offset` of the user's) is not a reason to panic.
    let want = end.max(self.cap.saturating_mul(2)).max(64).min(self.max);
    // SAFETY: forwarded.
    assert!(unsafe { self.room(end, want) }, "region exceeds its storage");
    let to = want.min(self.mem);
    // SAFETY: `cap <= to <= mem`, and the buffer is attached.
    unsafe { self.ptr.add(self.cap).write_bytes(0, to - self.cap) };
    self.cap = to;
  }

  /// Whether the memory holds `end` bytes, asking the storage for `want` if it does not.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc).
  unsafe fn room(&mut self, end: usize, want: usize) -> bool {
    if end > self.mem && end <= self.max {
      // SAFETY: forwarded.
      unsafe { (self.regrow)(self, want.div_ceil(WORD)) };
    }
    end <= self.mem
  }

  /// Copies `len` bytes from `src` into the empty buffer, asking the storage for room first:
  /// whether it had it. Nothing is written if not.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc); the buffer is empty, and `src` valid for reading `len` bytes
  /// outside it.
  pub(crate) unsafe fn load(&mut self, src: *const u8, len: usize) -> bool {
    // SAFETY: forwarded; the memory holds `len` bytes once there is room.
    unsafe {
      if !self.room(len, len.min(self.max)) {
        return false;
      }
      ptr::copy_nonoverlapping(src, self.ptr, len);
    }
    // What was zero past `len` still is.
    (self.len, self.cap) = (len, self.cap.max(len));
    true
  }
}

/// Where values are written: a region's buffer, reached through a raw pointer so that no
/// reference to it outlives a statement.
pub struct Sink {
  buf: *mut Buf,
}

impl Sink {
  /// # Safety
  /// `buf` points to a live, attached `Buf` while the sink is used, and every access to that
  /// `Buf` meanwhile goes through `buf` (its `me` during an edit).
  pub(crate) const unsafe fn new(buf: *mut Buf) -> Self {
    Self { buf }
  }

  pub(crate) const fn buf(&self) -> *mut Buf {
    self.buf
  }

  /// `size` bytes aligned to `align`; returns their position.
  #[inline]
  pub(crate) fn alloc(&mut self, size: usize, align: usize) -> usize {
    // SAFETY: `buf` is live and attached per `new`, and this reborrow ends with the statement.
    unsafe { (*self.buf).alloc(size, align, 0) }
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

  /// Points the pointer of `O` at `at` to `target`.
  ///
  /// # Safety
  /// `at` holds a pointer field of `O`.
  #[inline]
  pub(crate) unsafe fn link<O: Offset>(&mut self, at: usize, target: usize) {
    offset::<O>();
    // SAFETY: forwarded; the distance fits `O` because the region is within its limit.
    unsafe { self.write(at, O::from_isize(distance(at, target))) }
  }

  /// Writes the length of a list or a string, an `O`, at `at`.
  ///
  /// # Safety
  /// `at + size_of::<O>()` is within the allocated bytes.
  #[inline]
  pub(crate) unsafe fn write_len<O: Offset>(&mut self, at: usize, len: usize) {
    offset::<O>();
    // SAFETY: forwarded; a length within the region fits `O`.
    unsafe { self.write(at, O::from_isize(len.cast_signed())) }
  }
}

/// The position a pointer of `O` at `at` points to.
///
/// # Safety
/// `at` holds a non-null pointer of `O` in `buf`.
pub unsafe fn aim<O: Offset>(buf: *mut Buf, at: usize) -> usize {
  offset::<O>();
  // SAFETY: per the caller.
  let off = unsafe { (*buf).base().add(at).cast::<O>().read_unaligned() };
  at.wrapping_add_signed(off.to_isize())
}

/// The length of a list or a string, an `O` at `at`.
///
/// # Safety
/// `at` holds that length in `buf`.
unsafe fn read_len<O: Offset>(buf: *mut Buf, at: usize) -> usize {
  offset::<O>();
  // SAFETY: per the caller.
  unsafe { (*buf).base().add(at).cast::<O>().read_unaligned().to_isize().cast_unsigned() }
}

/// Types that may live in a region: plain data, [`Near`] pointers, and types made with
/// [`#[flat]`](crate::flat). Implement it with `#[flat]`, not by hand.
///
/// | Type | [`Edit`](Flat::Edit) |
/// |---|---|
/// | `u8`..`u64`, `i8`..`i64`, `f32`, `f64`, `bool`, `char`, `NonZero` integers | itself |
/// | `[T; N]`, `T: Flat + Copy` | itself |
/// | `Near<T, O>` | `Edit<'e, T>`, the target's handle |
/// | `Near<[T], O>` | [`ListEdit<'e, T, O>`] |
/// | `Near<str, O>` | [`StrEdit<'e, O>`] |
/// | `Option<Near<T, O>>` | [`OptionEdit<'e, T, O>`] |
/// | `#[flat]` types | generated |
///
/// Every stored type is `'static`, is not more aligned than a `u64` (the words of the memory of a
/// region: 8 bytes on most 64-bit targets, 4 on `i686`), has a non-zero size and no drop glue; the
/// last three are checked at compile time, and so is that no pointer leads to a type holding a
/// narrower one (see [`Offset`]).
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
/// `Self` has no drop glue and no pointers other than [`Near`]; its layout as a [`Pun`] type is
/// its own, whose pointers are those of this crate: values are copied, moved and checked from it
/// (`puns::copy_in`, `pointers_in`, `validate_in`), as is whether a value is copied as bytes
/// (`PLAIN`) or not looked at (`ANY`); `REACH` is at most the `REACH` of every field; `edit`
/// returns the handle of a valid `Self` at `at`.
pub unsafe trait Flat: Pun + Sized + 'static {
  /// The handle of a place holding a `Self`, during the edit `'e`: what
  /// [`Region::edit`](crate::Region::edit) and field methods return a `&mut` to. See
  /// [`Edit`](crate::Edit).
  type Edit<'e>: ?Sized + 'e;

  /// No pointers and no padding: may be copied as bytes. From the layout, not to be given.
  #[doc(hidden)]
  const PLAIN: bool = <Self as Pun>::LAYOUT.is_plain();

  /// No pointers, and every bit pattern is a valid value: there is nothing to check, so lists
  /// of such elements are not walked. From the layout, not to be given.
  #[doc(hidden)]
  const ANY: bool = <Self as Pun>::LAYOUT.is_free();

  /// What the narrowest pointer held by value (not through a pointer) reaches: the largest
  /// region with a `Self` at its root (see [`Offset`]).
  #[doc(hidden)]
  const REACH: u64 = u64::MAX;

  /// The handle of the `Self` at `at`.
  ///
  /// # Safety
  /// `buf` is the `me` of a `Buf` holding a valid `Self` at `at`, borrowed for the edit `'e`;
  /// the place is only reached through the returned handle for `'a`.
  #[doc(hidden)]
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self::Edit<'e>;
}

/// A deep copy of `src` at `at`.
///
/// # Safety
/// `at` is allocated in the sink for a `T`, and `src` is not in the sink's buffer.
pub unsafe fn write_copy<T: Flat>(src: &T, sink: &mut Sink, at: usize) {
  // SAFETY: per the caller; a valid value, outside the sink's buffer.
  unsafe { copy_value::<T>(ptr::from_ref(src).cast(), sink, at) }
}

/// A deep copy of the elements of `src` at `at`.
///
/// # Safety
/// As for [`write_copy`], with room for `src.len()` elements.
unsafe fn write_slice<T: Flat>(src: &[T], sink: &mut Sink, at: usize) {
  // SAFETY: per the caller; valid values, outside the sink's buffer.
  unsafe { copy_values::<T>(src.as_ptr().cast(), src.len(), sink, at) }
}

/// The deep copy of the `n` valid `U`s at `src`, `size_of::<U>()` apart, to `at`: the bytes that
/// a valid `U` holds initialized (`puns::copy_in`, so that the padding of a value made on the
/// stack, uninitialized, does not reach the buffer, whose bytes stay the zeros they were), then
/// each pointer re-made to a copy of its target (`puns::pointers_in`), depth first and in the order
/// of their positions, which is the order of the fields: the compact image.
///
/// # Safety
/// `src` holds `n` valid `U`s, not in the sink's buffer, and `at` is allocated in the sink for
/// them, zero.
unsafe fn copy_values<U: Pun>(src: *const u8, n: usize, sink: &mut Sink, at: usize) {
  let size = const { U::LAYOUT.size() };
  // SAFETY: per the caller; each copy writes the bytes of its value only.
  unsafe {
    if const { U::LAYOUT.is_plain() } {
      ptr::copy_nonoverlapping(src, (*sink.buf).ptr().add(at), n * size);
      return;
    }
    for i in 0..n {
      puns::copy_in::<U>(src.add(i * size), (*sink.buf).ptr().add(at + i * size));
    }
    if const { U::LAYOUT.has_pointers() } {
      let mut deep = Deep { src, sink, at };
      let _ = puns::pointers_elems_in::<U, Deep<'_>>(src, 0, n, &mut deep);
    }
  }
}

/// [`copy_values`] for one value, the most frequent: what a pointer to one value aims at.
///
/// # Safety
/// As [`copy_values`], for one value.
unsafe fn copy_value<U: Pun>(src: *const u8, sink: &mut Sink, at: usize) {
  // SAFETY: per the caller.
  unsafe {
    puns::copy_in::<U>(src, (*sink.buf).ptr().add(at));
    if const { U::LAYOUT.has_pointers() } {
      let _ = puns::pointers_in::<U, Deep<'_>>(src, 0, &mut Deep { src, sink, at });
    }
  }
}

/// Copies what each pointer that `puns` hands over aims at, and aims the copy of the pointer at
/// the copy: for a value at `src` copied to `at`. What a pointer aims at is read as its
/// description says (`Pointer`: one value, a sequence after its length, text after its length),
/// with offsets and lengths read and written as the contract of `Offset` says their bytes are,
/// two's complement integers of the width and the byte order of the pointer.
struct Deep<'s> {
  src: *const u8,
  sink: &'s mut Sink,
  at: usize,
}

impl puns::Visit for Deep<'_> {
  type Error = puns::Invalid;

  #[inline]
  fn pointer<P: ?Sized + Pun>(&mut self, pos: usize) -> Result<(), puns::Invalid> {
    let Some(pointer) = (const { P::LAYOUT.as_pointer() }) else { return Ok(()) };
    let (w, endian) = (pointer.width(), pointer.endian());
    let size = const { <puns::Pointee<P> as Pun>::LAYOUT.size() };
    let align = const { <puns::Pointee<P> as Pun>::LAYOUT.align() };
    // SAFETY: a pointer of a valid value (`pointers_in`), not null, aims at a valid target of its
    // kind in the same buffer (S3 of `SelfRelative`), outside the sink's buffer as the value is.
    // The blocks are allocated before they are written, and every position written is in them.
    unsafe {
      let here = self.src.add(pos);
      let to = here.wrapping_offset(read_int(here, w, endian));
      let target = match pointer.aim() {
        puns::layout::Aim::One => {
          let t = self.sink.alloc(size, align);
          copy_value::<puns::Pointee<P>>(to, self.sink, t);
          t
        }
        puns::layout::Aim::Sequence { .. } => {
          let len = read_int(to, w, endian).cast_unsigned();
          // As `List::new`: `[cap][len]`, then the elements, aligned for them and for the width.
          let start = (*self.sink.buf).alloc(
            size.saturating_mul(len).saturating_add(2 * w),
            align.max(w),
            2 * w,
          );
          let buf = (*self.sink.buf).ptr();
          write_int(buf.add(start), w, endian, len.cast_signed());
          write_int(buf.add(start + w), w, endian, len.cast_signed());
          copy_values::<puns::Pointee<P>>(to.add(w), len, self.sink, start + 2 * w);
          start + w
        }
        puns::layout::Aim::Text => {
          let len = read_int(to, w, endian).cast_unsigned();
          // As `copy_str`: the length, then the bytes.
          let h = self.sink.alloc(len + w, w);
          let buf = (*self.sink.buf).ptr();
          write_int(buf.add(h), w, endian, len.cast_signed());
          ptr::copy_nonoverlapping(to.add(w), buf.add(h + w), len);
          h
        }
      };
      let from = self.at + pos;
      write_int((*self.sink.buf).ptr().add(from), w, endian, distance(from, target));
    }
    Ok(())
  }
}

/// The integer of `width` bytes (2, 4 or 8) in the byte order `endian` at `at`, two's complement.
///
/// # Safety
/// `at` is valid for reads of `width` bytes.
#[inline]
unsafe fn read_int(at: *const u8, width: usize, endian: puns::layout::Endian) -> isize {
  /// # Safety
  /// As `read_int`.
  #[inline]
  unsafe fn read<const N: usize>(at: *const u8, endian: puns::layout::Endian) -> [u8; N] {
    // SAFETY: per the caller.
    let mut bytes = unsafe { at.cast::<[u8; N]>().read_unaligned() };
    if endian != puns::layout::Endian::NATIVE {
      bytes.reverse();
    }
    bytes
  }
  // SAFETY: per the caller.
  #[expect(clippy::cast_possible_truncation, reason = "the integers of a region fit `isize`")]
  unsafe {
    match width {
      2 => i16::from_ne_bytes(read(at, endian)) as isize,
      4 => i32::from_ne_bytes(read(at, endian)) as isize,
      _ => i64::from_ne_bytes(read(at, endian)) as isize,
    }
  }
}

/// Writes `n` as an integer of `width` bytes (2, 4 or 8) in the byte order `endian` at `at`.
///
/// # Safety
/// `at` is valid for writes of `width` bytes, and `n` fits them.
#[inline]
unsafe fn write_int(at: *mut u8, width: usize, endian: puns::layout::Endian, n: isize) {
  /// # Safety
  /// As `write_int`.
  #[inline]
  unsafe fn write<const N: usize>(at: *mut u8, endian: puns::layout::Endian, mut bytes: [u8; N]) {
    if endian != puns::layout::Endian::NATIVE {
      bytes.reverse();
    }
    // SAFETY: per the caller.
    unsafe { at.cast::<[u8; N]>().write_unaligned(bytes) }
  }
  // SAFETY: per the caller.
  #[expect(clippy::cast_possible_truncation, reason = "`n` fits the width")]
  unsafe {
    match width {
      2 => write(at, endian, (n as i16).to_ne_bytes()),
      4 => write(at, endian, (n as i32).to_ne_bytes()),
      _ => write(at, endian, (n as i64).to_ne_bytes()),
    }
  }
}

/// Re-aims every pointer that `puns` hands over by `d` bytes: the value that holds them moved
/// by `-d`, its targets did not. Each offset is read and written as the contract of `Offset` says
/// its bytes are, a two's complement integer of its width in its byte order, which the
/// description of the pointer gives.
struct Shift {
  base: *mut u8,
  d: isize,
}

impl puns::Visit for Shift {
  type Error = puns::Invalid;

  #[inline]
  fn pointer<P: ?Sized + Pun>(&mut self, at: usize) -> Result<(), puns::Invalid> {
    if let Some(pointer) = const { P::LAYOUT.as_pointer() } {
      let (w, endian) = (pointer.width(), pointer.endian());
      // SAFETY: `at` holds the offset of a pointer of a valid value (`pointers_in`), in the buffer
      // at `base`; the new distance is within the region, so within the `LIMIT` of the offset.
      unsafe {
        let here = self.base.add(at);
        write_int(here, w, endian, read_int(here, w, endian).wrapping_add(self.d));
      }
    }
    Ok(())
  }
}

/// Re-aims the pointers of the `n` values of `T` at `at`, which moved by `-d` bytes.
///
/// # Safety
/// `n` valid `T`s are at `at` in `buf`, `size_of::<T>()` apart, whose pointers aim where they
/// did before the move.
#[inline]
unsafe fn shift<T: Flat>(buf: *mut Buf, at: usize, n: usize, d: isize) {
  if const { !<T as Pun>::LAYOUT.has_pointers() } {
    return;
  }
  // SAFETY: per the caller; `puns` reads the tags and the offsets of the values, and `Shift`
  // writes only their offsets.
  unsafe {
    let base = (*buf).ptr();
    let _ = puns::pointers_elems_in::<T, Shift>(base, at, n, &mut Shift { base, d });
  }
}

/// The bytes of the `T` at `at`, kept aside (initialized where they are: the buffer is).
///
/// # Safety
/// A `T` is at `at` in `buf`.
pub unsafe fn stash<T: Flat>(buf: *mut Buf, at: usize) -> MaybeUninit<T> {
  let mut kept = MaybeUninit::<T>::uninit();
  // SAFETY: per the caller; the bytes are copied, not read as a `T`, so they stay what they are.
  unsafe {
    ptr::copy_nonoverlapping((*buf).ptr().add(at), kept.as_mut_ptr().cast::<u8>(), size_of::<T>());
  }
  kept
}

/// Writes at `to` the `T` that `stash` took from `from`, re-aiming its pointers.
///
/// # Safety
/// `kept` is the stash of a `T` at `from` in `buf`, whose targets have not moved, and `to` is
/// allocated for a `T`.
pub unsafe fn unstash<T: Flat>(buf: *mut Buf, kept: &MaybeUninit<T>, from: usize, to: usize) {
  // SAFETY: per the caller.
  unsafe {
    ptr::copy_nonoverlapping(kept.as_ptr().cast::<u8>(), (*buf).ptr().add(to), size_of::<T>());
    shift::<T>(buf, to, 1, distance(to, from));
  }
}

/// Moves the `T` at `from` to `to`, re-aiming its pointers. The two may overlap.
///
/// # Safety
/// A `T` is at `from` in `buf`, and `to` is allocated for a `T`.
pub unsafe fn relocate<T: Flat>(buf: *mut Buf, from: usize, to: usize) {
  // SAFETY: forwarded.
  unsafe { move_elems::<T>(buf, from, to, 1) }
}

/// Moves the `T` at `at` to fresh bytes, where nothing reaches it; returns their position.
///
/// # Safety
/// A `T` is at `at` in `buf`.
pub unsafe fn detach<T: Flat>(buf: *mut Buf, at: usize) -> usize {
  // SAFETY: per the caller; the fresh bytes are allocated before anything is read.
  unsafe {
    let kept = (*buf).alloc(size_of::<T>(), align_of::<T>(), 0);
    relocate::<T>(buf, at, kept);
    kept
  }
}

/// Moves `n` consecutive `T`s from `from` to `to`, re-aiming their pointers; the ranges may
/// overlap. The bytes move at once (they are initialized, as every byte of the buffer below its
/// length), then the pointers of the moved values are re-aimed where they are now.
///
/// # Safety
/// `n` `T`s are at `from` in `buf`, and `to` is allocated for `n` of them.
pub unsafe fn move_elems<T: Flat>(buf: *mut Buf, from: usize, to: usize, n: usize) {
  // SAFETY: per the caller; a pointer's target is not one of the moved values (a value owns its
  // targets, elsewhere), so it does not move.
  unsafe {
    let p = (*buf).ptr();
    ptr::copy(p.add(from), p.add(to), n * size_of::<T>());
    shift::<T>(buf, to, n, distance(to, from));
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
  ($($t:ty),*) => {$(
    // SAFETY: primitive values have no pointers and no padding (their layouts say so).
    unsafe impl Flat for $t {
      type Edit<'e> = Self;
      unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
        // SAFETY: forwarded.
        unsafe { plain_edit(buf, at) }
      }
    }
  )*};
}
plain! {
  u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool, char,
  NonZero<u8>, NonZero<u16>, NonZero<u32>, NonZero<u64>,
  NonZero<i8>, NonZero<i16>, NonZero<i32>, NonZero<i64>
}

/// `Flat` for the numbers of [`puns::num`], which are byte arrays: of every byte order, and of
/// given values.
mod num {
  use puns::num::{
    f32be, f32le, f64be, f64le, i16be, i16le, i32be, i32le, i64be, i64le, i128be, i128le, u8c,
    u8in, u16be, u16bec, u16le, u16lec, u24be, u24bec, u24le, u24lec, u32be, u32bec, u32le, u32lec,
    u48be, u48bec, u48le, u48lec, u64be, u64bec, u64le, u64lec, u128be, u128bec, u128le, u128lec,
  };

  use super::{Buf, Flat, plain_edit};

  macro_rules! numbers {
    ($([$($generics:tt)*] $t:ty;)+) => {$(
      // SAFETY: a byte array (`puns::num`): no pointers, no padding; every bit pattern is a value,
      // or only those that its constructors make, which its layout tells and loading checks. A
      // `&mut Self` is given only for a valid `Self`, through which only a `Self` is written.
      unsafe impl<$($generics)*> Flat for $t {
        type Edit<'e> = Self;
        unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
          // SAFETY: forwarded.
          unsafe { plain_edit(buf, at) }
        }
      }
    )+};
  }

  numbers! {
    [] u16le; [] u16be; [] u24le; [] u24be; [] u32le; [] u32be; [] u48le; [] u48be; [] u64le;
    [] u64be; [] u128le; [] u128be; [] i16le; [] i16be; [] i32le; [] i32be; [] i64le; [] i64be;
    [] i128le; [] i128be; [] f32le; [] f32be; [] f64le; [] f64be;
    [const V: u8] u8c<V>;
    [const MIN: u8, const MAX: u8] u8in<MIN, MAX>;
    [const V: u16] u16lec<V>;
    [const V: u16] u16bec<V>;
    [const V: u32] u24lec<V>;
    [const V: u32] u24bec<V>;
    [const V: u32] u32lec<V>;
    [const V: u32] u32bec<V>;
    [const V: u64] u48lec<V>;
    [const V: u64] u48bec<V>;
    [const V: u64] u64lec<V>;
    [const V: u64] u64bec<V>;
    [const V: u128] u128lec<V>;
    [const V: u128] u128bec<V>;
  }
}

// SAFETY: an array is its elements, without padding between them; `Copy` elements have no
// pointers.
unsafe impl<T: Flat + Copy, const N: usize> Flat for [T; N] {
  type Edit<'e> = Self;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
    fits::<Self>();
    // SAFETY: forwarded.
    unsafe { plain_edit(buf, at) }
  }
}

// SAFETY: copies the target.
unsafe impl<T: Flat, O: Offset> Flat for Near<T, O> {
  type Edit<'e> = T::Edit<'e>;
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut T::Edit<'e> {
    fits_behind::<T, O>();
    // SAFETY: a `Near` at `at` points to a valid `T`, owned by it alone.
    unsafe { T::edit(buf, aim::<O>(buf, at)) }
  }
}

// SAFETY: copies the list.
unsafe impl<T: Flat, O: Offset> Flat for Near<[T], O> {
  type Edit<'e> = ListEdit<'e, T, O>;
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut ListEdit<'e, T, O> {
    fits_behind::<T, O>();
    // SAFETY: forwarded.
    unsafe { ListEdit::new(Raw::new(buf, at)) }
  }
}

// SAFETY: copies the text.
unsafe impl<O: Offset> Flat for Near<str, O> {
  type Edit<'e> = StrEdit<'e, O>;
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut StrEdit<'e, O> {
    // SAFETY: forwarded.
    unsafe { StrEdit::new(Raw::new(buf, at)) }
  }
}

// SAFETY: `None` is a zero offset; `Some` is the `Near`.
unsafe impl<T: Flat, O: Offset> Flat for Option<Near<T, O>> {
  type Edit<'e> = OptionEdit<'e, T, O>;
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut OptionEdit<'e, T, O> {
    fits_behind::<T, O>();
    // SAFETY: forwarded.
    unsafe { OptionEdit::new(Raw::new(buf, at)) }
  }
}

/// Writes at `at` a pointer of `O` to a copy of `s`.
///
/// # Safety
/// `at` holds a `Near<str, O>` in the sink.
unsafe fn copy_str<O: Offset>(sink: &mut Sink, at: usize, s: &str) {
  let w = size_of::<O>();
  let h = sink.alloc(s.len().saturating_add(w), w);
  // SAFETY: `h` was just allocated for the length and the bytes.
  unsafe {
    sink.write_len::<O>(h, s.len());
    ptr::copy_nonoverlapping(s.as_ptr(), (*sink.buf).ptr().add(h + w), s.len());
    sink.link::<O>(at, h);
  }
}

/// Sources that can be written as a `T`: values, copies, moves and builders.
///
/// | Target | Builders |
/// |---|---|
/// | any `T` | `T` by value, `&T` (a deep copy, from any region), [`Old<'_, T>`](crate::Old) (a move) |
/// | `Near<T, O>`, `Option<Near<T, O>>` | [`Near::new(e)`](Near::new) |
/// | `Near<[T], O>` | `[e; N]`, tuples up to 12 and `()`, `&[T]`, [`Near::from_iter(iter)`](Near::from_iter) |
/// | `Near<str, O>` | `&str` |
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
unsafe impl<T: Flat, O: Offset, E: Emit<T>> Emit<Near<T, O>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    fits_behind::<T, O>();
    let t = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `t` was just allocated for a `T`.
    unsafe {
      self.0.emit(sink, t);
      sink.link::<O>(at, t);
    }
  }
}

// SAFETY: `Some(near)` has the bytes of `near`.
unsafe impl<T: Flat, O: Offset, E: Emit<T>> Emit<Option<Near<T, O>>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { Emit::<Near<T, O>>::emit(self, sink, at) }
  }
}

// SAFETY: copies the text.
unsafe impl<O: Offset> Emit<Near<str, O>> for &str {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { copy_str::<O>(sink, at, self) }
  }
}

// SAFETY: copies the slice, which is not in the sink's buffer (as for `&T`).
unsafe impl<T: Flat, O: Offset> Emit<Near<[T], O>> for &[T] {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    let mut list = List::<O>::new::<T>(sink, self.len());
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
unsafe impl<T: Flat, O: Offset, I: IntoIterator<Item: Emit<T>>> Emit<Near<[T], O>> for Seq<I> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    let items = self.0.into_iter();
    let mut list = List::<O>::new::<T>(sink, items.size_hint().0);
    // SAFETY: a new block, filled by `extend`.
    unsafe {
      list.extend(sink, items);
      list.finish(sink, at);
    }
  }
}

// SAFETY: as for `Seq`.
unsafe impl<T: Flat, O: Offset, E: Emit<T>, const N: usize> Emit<Near<[T], O>> for [E; N] {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: forwarded.
    unsafe { Emit::<Near<[T], O>>::emit(Seq(self), sink, at) }
  }
}

/// Implements `Emit` for tuples of every length from the given one down to `()`.
macro_rules! tuples {
  ($($e:ident $x:ident)*) => {
    // SAFETY: as for `Seq`.
    unsafe impl<T: Flat, O: Offset, $($e: Emit<T>),*> Emit<Near<[T], O>> for ($($e,)*) {
      unsafe fn emit(self, sink: &mut Sink, at: usize) {
        let ($($x,)*) = self;
        let list = &mut List::<O>::new::<T>(sink, 0 $(+ { let _ = &$x; 1 })*);
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

/// A list block `[cap][len][elements]` in a buffer, its lengths `O`s: `head` is the position
/// of `len`, and `len` and `cap` are the values in use (the builder writes them to the header in
/// `finish`).
pub struct List<O> {
  pub head: usize,
  pub len: usize,
  pub cap: usize,
  _o: PhantomData<O>,
}

impl<O: Offset> List<O> {
  /// A new block with room for `cap` elements, its header `[cap][0]`. The elements are aligned
  /// for `T` and for `O`, and the header takes the two `O`s before them.
  pub fn new<T: Flat>(sink: &mut Sink, cap: usize) -> Self {
    fits_behind::<T, O>();
    let header = 2 * size_of::<O>();
    let bytes = size_of::<T>().saturating_mul(cap).saturating_add(header);
    let align = align_of::<T>().max(size_of::<O>());
    // SAFETY: the sink's buffer is live and attached (`Sink::new`), and this reborrow ends with
    // the statement; then the header was just allocated, and `cap` fits an `O` because the
    // block is within the region.
    let start = unsafe {
      let start = (*sink.buf).alloc(bytes, align, header);
      sink.write_len::<O>(start, cap);
      start
    };
    Self { head: start + size_of::<O>(), len: 0, cap, _o: PhantomData }
  }

  /// The list the pointer at `at` points to.
  ///
  /// # Safety
  /// `at` holds a `Near<[T], O>` in `buf`.
  pub unsafe fn read(buf: *mut Buf, at: usize) -> Self {
    // SAFETY: the pointer leads to a list header.
    unsafe {
      let head = aim::<O>(buf, at);
      let (len, cap) = (read_len::<O>(buf, head), read_len::<O>(buf, head - size_of::<O>()));
      Self { head, len, cap, _o: PhantomData }
    }
  }

  /// Position of element `i`.
  pub const fn elem<T>(&self, i: usize) -> usize {
    self.head + size_of::<O>() + i * size_of::<T>()
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
      sink.write_len::<O>(new.head, self.len);
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
  /// `at` holds a `Near<[T], O>` there.
  pub unsafe fn finish(&self, sink: &mut Sink, at: usize) {
    if self.cap != self.len {
      sink.loose();
    }
    // SAFETY: per the caller.
    unsafe {
      sink.write_len::<O>(self.head - size_of::<O>(), self.cap);
      sink.write_len::<O>(self.head, self.len);
      sink.link::<O>(at, self.head);
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

#[cfg(test)]
mod tests {
  use super::{ALIGN, aligned};

  /// As aligned as an 8-byte word on x86-64; more than a `u64` on `i686` and `msp430`.
  #[repr(C, align(8))]
  struct Eight(u8);

  /// More than a `u64` on every target.
  #[repr(C, align(16))]
  struct Sixteen(u8);

  /// What `fits` accepts: no type more aligned than a `u64`, whatever the target. On `i686`
  /// (Miri runs the library's tests there) a type aligned to 8 is refused at compile time.
  #[test]
  fn a_region_holds_types_aligned_as_its_words() {
    assert_eq!(ALIGN, align_of::<u64>());
    assert!(aligned::<u64>() && aligned::<f64>() && aligned::<i64>() && aligned::<u8>());
    assert_eq!(aligned::<Eight>(), align_of::<u64>() == 8);
    assert!(!aligned::<Sixteen>());
    #[cfg(target_arch = "x86_64")]
    assert!(aligned::<Eight>());
    #[cfg(target_arch = "x86")]
    assert!(!aligned::<Eight>() && ALIGN == 4);
  }
}
