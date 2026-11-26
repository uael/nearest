use core::{
  alloc::{Allocator, Layout},
  marker::PhantomData,
  mem::{MaybeUninit, needs_drop},
  num::NonZero,
  ptr::{self, NonNull},
  slice,
};

use puns::Pun;

use crate::{
  Near, Offset,
  edit::{Handle, ListEdit, OptionEdit, Plain, PlainEdit, Raw, StrEdit},
  near::offset,
};

/// The alignment of a region, 8 on every target: of its block, so of byte 0 and of the bytes
/// that `as_bytes` returns, and the largest alignment of a position in it, that of a stored type
/// or of the elements of a list (at most the width of its offset, 8). Positions and addresses
/// then agree modulo 8, so a position aligned for a type is an address aligned for it, and a
/// check by address (`view`, `puns`) is a check by position. The limits of a region and of an
/// `Offset` are multiples of it, so that a position aligned up from below a limit stays below it.
pub const ALIGN: usize = 8;

/// The largest region with a `T` at its root, a multiple of [`ALIGN`]: what the pointers of the
/// root reach, and at most `isize::MAX` bytes, like any allocation. Every pointer below the root
/// is at least as wide (see [`fits_behind`]), so every distance inside the region fits its
/// pointer.
#[expect(clippy::cast_possible_truncation, reason = "at most `isize::MAX`")]
pub const fn limit<T: Flat>() -> usize {
  let most = isize::MAX as u64 & !(ALIGN as u64 - 1);
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
/// block, [`ALIGN`].
pub const fn aligned<T>() -> bool {
  align_of::<T>() <= ALIGN
}

/// Rejects, at compile time, types a region cannot hold.
pub const fn fits<T>() {
  const {
    assert!(aligned::<T>(), "a region holds only types aligned to at most 8 bytes");
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

/// The layout of a block of `size` bytes: aligned to [`ALIGN`].
const fn block(size: usize) -> Layout {
  // SAFETY: `ALIGN` is a power of two, and `size` is at most the limit of a region, at most
  // `isize::MAX` rounded down to a multiple of `ALIGN`.
  unsafe { Layout::from_size_align_unchecked(size, ALIGN) }
}

/// The bytes of a region: a block of its allocator, of which the first `len` are in use.
///
/// A block does not move when its region moves, so the address of the bytes is kept. The
/// allocator moves with the region, so its address (`alloc`) is written on every use of the
/// region through `&mut` (`Region::bind`), before anything can grow the block.
///
/// The region zeroes the block itself, as it uses it, so an allocator may hand out memory of any
/// content: `cap` is the zeroed prefix, which `alloc` hands out without a call, and `mem` the
/// whole block.
///
/// Invariants, also while unwinding: `len <= cap <= mem <= max`, so every distance fits every
/// pointer of the region; the bytes in `[len, cap)` are zero, and those in `[cap, mem)`
/// anything. If `mem` is not zero, `ptr` is a block of the allocator of the region that fits
/// `block(mem)`; if it is zero, the region has no block.
pub struct Buf {
  /// Byte 0.
  ptr: NonNull<u8>,
  /// Bytes in use or zero: `alloc` grows nothing below it.
  cap: usize,
  len: usize,
  /// Bytes of the block at `ptr`, at most `max`.
  mem: usize,
  /// The limit of the region (see [`limit`]).
  max: usize,
  /// The end of the image that the region was loaded from, 0 if it was built: the capacity of
  /// a list whose block starts before it is not trusted (see `List::read`).
  pub(crate) loaded: usize,
  /// The allocator of the region, where the region was when it was last bound.
  pub(crate) alloc: *const (),
  /// Grows the block to at least a number of bytes, through the allocator at `alloc`.
  regrow: unsafe fn(&mut Buf, usize) -> bool,
  /// The address of this `Buf`, written by `Region::edit`. Every access of an edit goes
  /// through it, so that all of them have the same provenance (I8).
  pub(crate) me: *mut Buf,
  /// Every byte in use is initialized, which `as_bytes` needs. The region writes only
  /// initialized bytes, and keeps the bytes of padding, but an assignment through a `&mut` writes
  /// the padding of its value as uninitialized: handing out a `&mut` that reaches a type with
  /// padding clears it (`Raw::handed`).
  pub(crate) init: bool,
}

// SAFETY: `ptr` is the region's own block, and `alloc` and `me` the addresses of the region's
// allocator and of the `Buf` itself, all used while the region is borrowed mutably; a region is
// `Send` or `Sync` as its allocator is.
unsafe impl Send for Buf {}
// SAFETY: as above; nothing is written through a shared `Buf`.
unsafe impl Sync for Buf {}

/// Grows the block of `buf` to at least `bytes` bytes, at most its limit, through the `A` at
/// `buf.alloc`, or allocates the first block: whether the allocator gave it. If it did not, the
/// block is as before.
///
/// # Safety
/// `buf.alloc` is the allocator of the region, where it is now, an `A`, and `buf` holds a block
/// of it if `mem` is not zero.
unsafe fn regrow<A: Allocator>(buf: &mut Buf, bytes: usize) -> bool {
  // SAFETY: per the caller.
  let alloc = unsafe { &*buf.alloc.cast::<A>() };
  let new = block(bytes);
  let block = if buf.mem == 0 {
    alloc.allocate(new)
  } else {
    // SAFETY: the block is of this allocator and fits `block(mem)`, and `mem < bytes`.
    unsafe { alloc.grow(buf.ptr, block(buf.mem), new) }
  };
  let Ok(block) = block else { return false };
  buf.ptr = block.cast();
  // The block may be larger than asked: what is past the limit is not used, and `block(mem)`
  // still fits it, since `bytes <= mem`.
  buf.mem = block.len().min(buf.max);
  true
}

impl Buf {
  /// An empty buffer, with no block, for a region of at most `max` bytes, a multiple of
  /// [`ALIGN`] that every distance fits, whose allocator is an `A`.
  pub(crate) const fn new<A: Allocator>(max: usize) -> Self {
    Self {
      ptr: NonNull::dangling(),
      cap: 0,
      len: 0,
      mem: 0,
      max,
      loaded: 0,
      alloc: ptr::null(),
      regrow: regrow::<A>,
      me: ptr::null_mut(),
      init: true,
    }
  }

  /// Bytes in use.
  pub(crate) const fn len(&self) -> usize {
    self.len
  }

  /// Byte 0, for writing.
  pub(crate) const fn ptr(&self) -> *mut u8 {
    self.ptr.as_ptr()
  }

  /// Byte 0, for reading.
  pub(crate) const fn base(&self) -> *const u8 {
    self.ptr.as_ptr()
  }

  /// The limit of the region.
  pub(crate) const fn max(&self) -> usize {
    self.max
  }

  /// The block and its layout, to give back to the allocator, if the region has one.
  pub(crate) const fn block(&self) -> Option<(NonNull<u8>, Layout)> {
    if self.mem == 0 { None } else { Some((self.ptr, block(self.mem))) }
  }

  /// `size` zeroed bytes (bytes past `len` are never written) whose byte `inset` is aligned to
  /// `align`; returns their position.
  ///
  /// # Safety
  /// The buffer is bound, and its region is where it was then.
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

  /// Makes `end` bytes available: zeroes the block past `cap`, twice as much as is in use and at
  /// least 64 bytes, asking the allocator for a larger block first if it is shorter than `end`.
  /// Anything past the limit, or past what the allocator gives, comes here and panics, before
  /// anything changes.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc).
  #[cold]
  unsafe fn grow(&mut self, end: usize) {
    // Not `clamp`: a limit below 64 (an `Offset` of the user's) is not a reason to panic.
    let want = end.max(self.cap.saturating_mul(2)).max(64).min(self.max);
    // SAFETY: forwarded.
    unsafe { self.room(end, want) };
    let to = want.min(self.mem);
    // SAFETY: `cap <= to <= mem`, within the block.
    unsafe { self.ptr().add(self.cap).write_bytes(0, to - self.cap) };
    self.cap = to;
  }

  /// Makes the block at least `end` bytes, asking the allocator for `want` if it is shorter, or
  /// for `end` if it does not give `want`: a buffer of a fixed size may hold what is needed, not
  /// what a growth would like.
  ///
  /// # Panics
  /// If `end` is past the limit, or the allocator does not give the block.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc).
  pub(crate) unsafe fn room(&mut self, end: usize, want: usize) {
    assert!(end <= self.max, "region exceeds what its pointers reach");
    // SAFETY: forwarded; a refused growth leaves the block as it was.
    let given = end <= self.mem
      || unsafe { (self.regrow)(self, want) }
      || (end < want && unsafe { (self.regrow)(self, end) });
    assert!(given, "region exceeds what its allocator gives");
  }

  /// Copies `len` bytes from `src` into the empty buffer, the image of the region, asking the
  /// allocator for a block first: whether it gave it. Nothing is written if not.
  ///
  /// # Safety
  /// As for [`alloc`](Self::alloc); the buffer is empty, `len` at most the limit, and `src`
  /// valid for reading `len` bytes outside it.
  pub(crate) unsafe fn load(&mut self, src: *const u8, len: usize) -> bool {
    // SAFETY: forwarded; the block holds `len` bytes once given.
    unsafe {
      if len > self.mem && !(self.regrow)(self, len) {
        return false;
      }
      ptr::copy_nonoverlapping(src, self.ptr(), len);
    }
    // What was zero past `len` still is.
    (self.len, self.cap, self.loaded) = (len, self.cap.max(len), len);
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
  /// `buf` points to a live, bound `Buf` while the sink is used, and every access to that
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
    // SAFETY: `buf` is live and bound per `new`, and this reborrow ends with the statement.
    unsafe { (*self.buf).alloc(size, align, 0) }
  }

  /// # Safety
  /// `at + size_of::<P>()` is within the allocated bytes.
  #[inline]
  pub(crate) const unsafe fn write<P: Copy>(&mut self, at: usize, v: P) {
    // SAFETY: in bounds per the caller.
    unsafe { (*self.buf).ptr().add(at).cast::<P>().write_unaligned(v) }
  }

  /// Points the pointer of `O` at `at` to `pointee`.
  ///
  /// # Safety
  /// `at` holds a pointer field of `O`.
  #[inline]
  pub(crate) unsafe fn link<O: Offset>(&mut self, at: usize, pointee: usize) {
    offset::<O>();
    // SAFETY: forwarded; the distance fits `O` because the region is within its limit.
    unsafe { self.write(at, O::from_isize(distance(at, pointee))) }
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
/// | `u8`..`u64`, `i8`..`i64`, `f32`, `f64`, `bool`, `char`, `NonZero` integers, `puns::num` | [`PlainEdit<'e, Self>`] |
/// | `[T; N]`, `T: Flat + Copy` | [`PlainEdit<'e, Self>`] |
/// | `Near<T, O>` | `Edit<'e, T>`, the pointee's handle |
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
/// its own, whose eager aims are the pointers of this crate. Whether a value is copied as bytes
/// (`PLAIN`) is read from it. `store` writes every field of `self` at `at`, with a copy of what
/// each pointer aims at, depth first in the order of the fields, each pointee at the next place
/// aligned for it, which is the compact image that `puns` checks when loading; `shift` re-aims
/// every pointer of the value at `at` and nothing else.
/// `REACH` is at most the `REACH` of every field; `edit` returns the handle of a valid `Self` at
/// `at`.
pub unsafe trait Flat: Pun + Sized + 'static {
  /// The handle of a place holding a `Self`, during the edit `'e`: what
  /// [`Region::edit`](crate::Region::edit) and field methods return a `&mut` to. See
  /// [`Edit`](crate::Edit).
  type Edit<'e>: ?Sized + Handle<'e, Self::Value> + 'e;

  /// What a handle of a place of `Self` writes: `Self`, but what the pointee writes for
  /// `Near<T, O>`, whose handle is the pointee's.
  type Value: Flat;

  /// What the getters of handles (field methods, `at`, `push`, `as_mut`, `Region::edit`, ..) give
  /// for a place of `Self`: `&'a mut Edit<'e, Self>`, but for plain data a [`Plain`], whose `*`
  /// is the value.
  type Ref<'a, 'e: 'a>;

  /// The `Ref` of a handle.
  #[doc(hidden)]
  fn reference<'a, 'e: 'a>(edit: &'a mut Self::Edit<'e>) -> Self::Ref<'a, 'e>;

  /// No pointers and no padding: may be copied as bytes. From the layout, which tells whether a
  /// field, at any depth, is a pointer (an eager aim), not to be given.
  #[doc(hidden)]
  const PLAIN: bool = <Self as Pun>::LAYOUT.is_plain() && !<Self as Pun>::LAYOUT.holds_eager_aims();

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

  /// Writes a deep copy of `self` at `at`: each field where it is, and for each pointer a copy of
  /// its pointee, allocated depth first in the order of the fields, which is the compact image.
  /// Padding is not written, so it keeps the bytes that the buffer held there, which are
  /// initialized.
  ///
  /// # Safety
  /// `at` is allocated in the sink, aligned and sized for `Self`, and `self` is not in the sink's
  /// buffer.
  #[doc(hidden)]
  unsafe fn store(&self, sink: &mut Sink, at: usize);

  /// Re-aims every pointer of the `Self` at `at` in the buffer at `base` by `d` bytes: the value
  /// moved by `-d`, and what its pointers aim at did not. Nothing else is written.
  ///
  /// # Safety
  /// A valid `Self` is at `at` in the buffer at `base`, its pointers as they were before the move,
  /// and each new distance is within the region.
  #[doc(hidden)]
  unsafe fn shift(base: *mut u8, at: usize, d: isize);
}

/// A deep copy of `src` at `at`.
///
/// # Safety
/// As for [`Flat::store`].
pub unsafe fn write_copy<T: Flat>(src: &T, sink: &mut Sink, at: usize) {
  // SAFETY: forwarded, for one element.
  unsafe { write_slice(slice::from_ref(src), sink, at) }
}

/// A deep copy of the elements of `src` at `at`.
///
/// # Safety
/// As for [`Flat::store`], with room for `src.len()` elements.
unsafe fn write_slice<T: Flat>(src: &[T], sink: &mut Sink, at: usize) {
  // SAFETY: plain data is its bytes, all initialized, and the destination is sized for it; other
  // elements are stored one by one.
  unsafe {
    if T::PLAIN {
      let dst = (*sink.buf).ptr().add(at);
      ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), dst, size_of_val(src));
    } else {
      for (i, x) in src.iter().enumerate() {
        x.store(sink, at + i * size_of::<T>());
      }
    }
  }
}

/// Re-aims the pointers of the `n` values of `T` at `at`, which moved by `-d` bytes.
///
/// # Safety
/// `n` valid `T`s are at `at` in `buf`, `size_of::<T>()` apart, whose pointers aim where they
/// did before the move.
#[inline]
unsafe fn shift<T: Flat>(buf: *mut Buf, at: usize, n: usize, d: isize) {
  if T::PLAIN {
    return;
  }
  // SAFETY: per the caller, each a valid value of the buffer; the pointees did not move, so each
  // new distance is within the region.
  unsafe {
    let base = (*buf).ptr();
    for i in 0..n {
      T::shift(base, at + i * size_of::<T>(), d);
    }
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
/// `kept` is the stash of a `T` at `from` in `buf`, whose pointees have not moved, and `to` is
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
  // SAFETY: per the caller; a pointer's pointee is not one of the moved values (a value owns its
  // pointees, elsewhere), so it does not move.
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
const unsafe fn plain_edit<'a, 'e: 'a, T: Flat + Copy>(
  buf: *mut Buf,
  at: usize,
) -> &'a mut PlainEdit<'e, T> {
  // SAFETY: per the caller.
  unsafe { PlainEdit::new(Raw::new(buf, at)) }
}

/// The items of `Flat` that copy, move and check a value of plain data without padding: it is its
/// bytes, and it holds no pointer.
macro_rules! data {
  () => {
    unsafe fn store(&self, sink: &mut Sink, at: usize) {
      // SAFETY: the caller allocated `at` for `Self`, whose bytes are all initialized.
      unsafe { sink.write(at, *self) }
    }
    unsafe fn shift(_: *mut u8, _: usize, _: isize) {}
  };
}

macro_rules! plain {
  ($($t:ty),*) => {$(
    // SAFETY: primitive values have no pointers and no padding (their layouts say so).
    unsafe impl Flat for $t {
      type Edit<'e> = PlainEdit<'e, Self>;
      type Value = Self;
      type Ref<'a, 'e: 'a> = Plain<'a, 'e, Self>;
      fn reference<'a, 'e: 'a>(edit: &'a mut PlainEdit<'e, Self>) -> Plain<'a, 'e, Self> {
        Plain::new(edit)
      }
      unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut PlainEdit<'e, Self> {
        // SAFETY: forwarded.
        unsafe { plain_edit(buf, at) }
      }
      data!();
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

  use super::{Buf, Flat, Sink, plain_edit};
  use crate::edit::{Plain, PlainEdit};

  macro_rules! numbers {
    ($([$($generics:tt)*] $t:ty;)+) => {$(
      // SAFETY: a byte array (`puns::num`): no pointers, no padding; every bit pattern is a value,
      // or only those that its constructors make, which its layout tells and loading checks. A
      // `&mut Self` is given only for a valid `Self`, through which only a `Self` is written.
      unsafe impl<$($generics)*> Flat for $t {
        type Edit<'e> = PlainEdit<'e, Self>;
        type Value = Self;
        type Ref<'a, 'e: 'a> = Plain<'a, 'e, Self>;
        fn reference<'a, 'e: 'a>(edit: &'a mut PlainEdit<'e, Self>) -> Plain<'a, 'e, Self> {
          Plain::new(edit)
        }
        unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut PlainEdit<'e, Self> {
          // SAFETY: forwarded.
          unsafe { plain_edit(buf, at) }
        }
        data!();
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
  type Edit<'e> = PlainEdit<'e, Self>;
  type Value = Self;
  type Ref<'a, 'e: 'a> = Plain<'a, 'e, Self>;
  fn reference<'a, 'e: 'a>(edit: &'a mut PlainEdit<'e, Self>) -> Plain<'a, 'e, Self> {
    Plain::new(edit)
  }
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut PlainEdit<'e, Self> {
    fits::<Self>();
    // SAFETY: forwarded.
    unsafe { plain_edit(buf, at) }
  }
  unsafe fn store(&self, sink: &mut Sink, at: usize) {
    fits::<Self>();
    // SAFETY: the caller allocated `at` for the array. Element by element, so that the padding of
    // an element is not written.
    unsafe { write_slice(self, sink, at) }
  }
  // `Copy` elements hold no pointer.
  unsafe fn shift(_: *mut u8, _: usize, _: isize) {}
}

// SAFETY: copies the pointee.
unsafe impl<T: Flat, O: Offset> Flat for Near<T, O> {
  type Edit<'e> = T::Edit<'e>;
  type Value = T::Value;
  type Ref<'a, 'e: 'a> = T::Ref<'a, 'e>;
  fn reference<'a, 'e: 'a>(edit: &'a mut T::Edit<'e>) -> T::Ref<'a, 'e> {
    T::reference(edit)
  }
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut T::Edit<'e> {
    fits_behind::<T, O>();
    // SAFETY: a `Near` at `at` points to a valid `T`, owned by it alone.
    unsafe { T::edit(buf, aim::<O>(buf, at)) }
  }
  unsafe fn store(&self, sink: &mut Sink, at: usize) {
    fits_behind::<T, O>();
    // SAFETY: the caller allocated `at` for a `Near`; the pointee is allocated before the source is
    // read, and the source is not in the sink's buffer, nor is its pointee.
    unsafe {
      let t = sink.alloc(size_of::<T>(), align_of::<T>());
      write_copy(&**self, sink, t);
      sink.link::<O>(at, t);
    }
  }
  unsafe fn shift(base: *mut u8, at: usize, d: isize) {
    // SAFETY: forwarded.
    unsafe { moved::<O>(base, at, d) }
  }
}

// SAFETY: copies the list.
unsafe impl<T: Flat, O: Offset> Flat for Near<[T], O> {
  type Edit<'e> = ListEdit<'e, T, O>;
  type Value = Self;
  type Ref<'a, 'e: 'a> = &'a mut ListEdit<'e, T, O>;
  fn reference<'a, 'e: 'a>(edit: &'a mut ListEdit<'e, T, O>) -> &'a mut ListEdit<'e, T, O> {
    edit
  }
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut ListEdit<'e, T, O> {
    fits_behind::<T, O>();
    // SAFETY: forwarded.
    unsafe { ListEdit::new(Raw::new(buf, at)) }
  }
  unsafe fn store(&self, sink: &mut Sink, at: usize) {
    // SAFETY: the caller allocated `at` for a `Near`; the list is copied as `&[T]` builds one.
    unsafe { Emit::<Self>::emit(&**self, sink, at) }
  }
  unsafe fn shift(base: *mut u8, at: usize, d: isize) {
    // SAFETY: forwarded.
    unsafe { moved::<O>(base, at, d) }
  }
}

// SAFETY: copies the text.
unsafe impl<O: Offset> Flat for Near<str, O> {
  type Edit<'e> = StrEdit<'e, O>;
  type Value = Self;
  type Ref<'a, 'e: 'a> = &'a mut StrEdit<'e, O>;
  fn reference<'a, 'e: 'a>(edit: &'a mut StrEdit<'e, O>) -> &'a mut StrEdit<'e, O> {
    edit
  }
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut StrEdit<'e, O> {
    // SAFETY: forwarded.
    unsafe { StrEdit::new(Raw::new(buf, at)) }
  }
  unsafe fn store(&self, sink: &mut Sink, at: usize) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe { copy_str::<O>(sink, at, self) }
  }
  unsafe fn shift(base: *mut u8, at: usize, d: isize) {
    // SAFETY: forwarded.
    unsafe { moved::<O>(base, at, d) }
  }
}

// SAFETY: `None` is a zero offset; `Some` is the `Near`.
unsafe impl<T: Flat, O: Offset> Flat for Option<Near<T, O>> {
  type Edit<'e> = OptionEdit<'e, T, O>;
  type Value = Self;
  type Ref<'a, 'e: 'a> = &'a mut OptionEdit<'e, T, O>;
  fn reference<'a, 'e: 'a>(edit: &'a mut OptionEdit<'e, T, O>) -> &'a mut OptionEdit<'e, T, O> {
    edit
  }
  const REACH: u64 = O::LIMIT;
  unsafe fn edit<'a, 'e: 'a>(buf: *mut Buf, at: usize) -> &'a mut OptionEdit<'e, T, O> {
    fits_behind::<T, O>();
    // SAFETY: forwarded.
    unsafe { OptionEdit::new(Raw::new(buf, at)) }
  }
  unsafe fn store(&self, sink: &mut Sink, at: usize) {
    offset::<O>();
    // SAFETY: the caller allocated `at` for a `Near`; `None` is a zero offset.
    unsafe {
      match self {
        Some(near) => near.store(sink, at),
        None => sink.write(at, O::from_isize(0)),
      }
    }
  }
  unsafe fn shift(base: *mut u8, at: usize, d: isize) {
    offset::<O>();
    // SAFETY: per the caller, an `Option<Near<T, O>>` at `at`: an `O`, zero for `None`, which
    // aims nowhere and stays zero.
    unsafe {
      if base.add(at).cast::<O>().read_unaligned().to_isize() != 0 {
        moved::<O>(base, at, d);
      }
    }
  }
}

/// Re-aims the `O` of the pointer at `at` in the buffer at `base` by `d` bytes.
///
/// # Safety
/// A pointer of `O` is at `at` in the buffer at `base`, and its new distance is within the region,
/// so within the `LIMIT` of `O`.
unsafe fn moved<O: Offset>(base: *mut u8, at: usize, d: isize) {
  offset::<O>();
  // SAFETY: per the caller; offsets are read and written unaligned, as their positions may be
  // aligned more than their addresses.
  unsafe {
    let here = base.add(at).cast::<O>();
    here.write_unaligned(O::from_isize(here.read_unaligned().to_isize().wrapping_add(d)));
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
/// pointee themselves.
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
/// `emit` writes a valid `T` at `at`, allocating whatever it points to. `plain`, where it returns
/// `Some(v)`, returns a valid `T` whose bytes are all initialized and hold no pointer, written as
/// `emit` would write it: a list of `T: PLAIN` copies those bytes in place of calling `emit`.
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

// SAFETY: allocates and writes the pointee, then links to it.
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
    // SAFETY: the sink's buffer is live and bound (`Sink::new`), and this reborrow ends with
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
  /// The capacity of a block in the image that the region was loaded from is not trusted: the
  /// check of the image claims the elements in use only, so the room after them may hold other
  /// values. Such a list is full, and moves on its first push. The room of a block after the
  /// image was made by the region.
  ///
  /// # Safety
  /// `at` holds a `Near<[T], O>` in `buf`.
  pub unsafe fn read(buf: *mut Buf, at: usize) -> Self {
    // SAFETY: the pointer leads to a list header.
    unsafe {
      let head = aim::<O>(buf, at);
      let len = read_len::<O>(buf, head);
      let cap = if head < (*buf).loaded { len } else { read_len::<O>(buf, head - size_of::<O>()) };
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
          if T::PLAIN
            && let Some(v) = item.plain()
          {
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

  /// Writes the header of a built list, `[cap][len]`, and links `at` to it.
  ///
  /// # Safety
  /// `self` is a block of `T`s in the sink's buffer whose first `len` elements are valid, and
  /// `at` holds a `Near<[T], O>` there.
  pub unsafe fn finish(&self, sink: &mut Sink, at: usize) {
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
  use super::aligned;

  /// Aligned to 8, more than a `u64` on `i686` and `msp430`.
  #[repr(C, align(8))]
  struct Eight(u8);

  /// More than 8.
  #[repr(C, align(16))]
  struct Sixteen(u8);

  /// What `fits` accepts: a type aligned to 8 at most, on every target, also where a `u64` is
  /// aligned to less (Miri runs the library's tests on `i686`).
  #[test]
  fn a_region_holds_types_aligned_to_8() {
    assert!(aligned::<u64>() && aligned::<f64>() && aligned::<i64>() && aligned::<u8>());
    assert!(aligned::<Eight>());
    assert!(!aligned::<Sixteen>());
  }
}
