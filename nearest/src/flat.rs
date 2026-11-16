use alloc::vec::Vec;
use core::{
  marker::PhantomData,
  mem::{self, MaybeUninit},
  pin::Pin,
  ptr::{self, Pointee},
  slice,
};

use crate::Near;

/// Largest region, a multiple of 8: every distance inside it fits the `i32` offset. Small in
/// unit tests, so that the limit can be hit cheaply.
const MAX: usize = if cfg!(test) { 1 << 16 } else { i32::MAX as usize & !7 };

pub const fn align_up(n: usize, align: usize) -> usize {
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
    assert!(
      align_of::<T>() <= 8 && size_of::<T>() != 0,
      "a region holds only types aligned to at most 8 bytes and of non-zero size"
    );
  };
}

/// Whether `Near<T>` points at a length header (slices and `str`) rather than a `T`.
pub const fn is_list<T: ?Sized>() -> bool {
  size_of::<<T as Pointee>::Metadata>() != 0
}

/// 8-aligned bytes, zero-filled when they grow.
///
/// Invariants, also while unwinding: `len <= words.len() * 8`, and `words.len() * 8 + base <=
/// MAX` for the `base` the buffer is appended at, so every position fits an `i32`.
pub struct Buf {
  pub words: Vec<MaybeUninit<u64>>,
  pub len: usize,
}

impl Buf {
  pub(crate) const fn new() -> Self {
    Self { words: Vec::new(), len: 0 }
  }

  #[inline]
  pub const fn ptr(&mut self) -> *mut u8 {
    self.words.as_mut_ptr().cast()
  }

  /// `size` zeroed bytes at `at`, which is at least `len`; `base` is where byte 0 of this
  /// buffer will finally live.
  #[inline]
  pub(crate) fn alloc(&mut self, at: usize, size: usize, base: usize) -> usize {
    let end = at.saturating_add(size);
    if end > self.words.len() * 8 {
      self.grow(end, base);
    }
    self.len = end;
    at
  }

  /// Grows to hold `end` bytes, keeping `words.len() * 8 + base <= MAX`: anything past the
  /// limit comes here and panics, before changing anything.
  #[cold]
  fn grow(&mut self, end: usize, base: usize) {
    assert!(end.saturating_add(base) <= MAX, "region exceeds 2 GiB");
    let words = end.div_ceil(8).max(self.words.len() * 2).max(8).min((MAX - base) / 8);
    self.words.resize(words, MaybeUninit::zeroed());
  }
}

/// Where values are written. Only `Flat` and `Emit` impls touch it.
pub struct Sink<'o> {
  pub(crate) buf: Buf,
  /// Where byte 0 of this sink lands in the region.
  pub(crate) base: usize,
  /// Address of byte 0 of the region being edited.
  pub(crate) origin: usize,
  brand: PhantomData<fn(&'o ()) -> &'o ()>,
}

impl Sink<'_> {
  pub(crate) const fn new(base: usize, origin: usize) -> Self {
    Self { buf: Buf::new(), base, origin, brand: PhantomData }
  }

  #[inline]
  pub(crate) fn alloc(&mut self, size: usize, align: usize) -> usize {
    self.buf.alloc(align_up(self.buf.len, align), size, self.base)
  }

  /// A `u32` length header followed by `bytes` aligned to `align`; returns the header position.
  #[inline]
  fn alloc_list(&mut self, align: usize, bytes: usize) -> usize {
    let head = align_up(self.buf.len + 4, align.max(4)) - 4;
    self.buf.alloc(head, bytes.saturating_add(4), self.base)
  }

  /// # Safety
  /// `at + size_of::<P>()` is within the allocated bytes.
  #[inline]
  const unsafe fn put<P: Copy>(&mut self, at: usize, v: P) {
    // SAFETY: in bounds per the caller; the buffer has no alignment guarantee for `P` here.
    unsafe { self.buf.ptr().add(at).cast::<P>().write_unaligned(v) }
  }

  /// # Safety
  /// `at` holds a pointer field.
  #[inline]
  const unsafe fn link(&mut self, at: usize, target: usize) {
    // SAFETY: forwarded; the distance fits an `i32` because the region is at most `MAX` bytes.
    unsafe { self.put(at, distance(at, target)) }
  }
}

/// Types that may live in a region: plain data, [`Near`] pointers, and types made with
/// [`#[flat]`](crate::flat).
///
/// Implemented for `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool char`, arrays, slices, `str`,
/// `Near<T>` and `Option<Near<T>>`. Implement it with `#[flat]`, not by hand.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
/// use nearest::{Flat, Near, flat};
///
/// #[flat]
/// struct Point { x: i32, y: i32 }
///
/// fn assert_flat<T: Flat + ?Sized>() {}
/// assert_flat::<Point>();
/// assert_flat::<Near<[Point]>>();
/// ```
///
/// # Safety
///
/// `T` has no drop glue and no pointers other than [`Near`]. `store` writes every field of
/// `self` at `at` as described on it, and `PLAIN` is `true` only if `T` has no padding and no
/// pointers. An unsized implementor has the metadata of a slice.
pub unsafe trait Flat {
  /// No pointers and no padding: may be copied as bytes.
  #[doc(hidden)]
  const PLAIN: bool = false;

  /// Writes `self` into `sink` at `at`. With `shift == None` every pointer target is copied
  /// into the sink too; with `Some(d)` every pointer keeps its target and is re-aimed for a
  /// move of `d` bytes.
  ///
  /// # Safety
  /// `at` is allocated in the sink, aligned and sized for `self`.
  #[doc(hidden)]
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>);
}

/// [`Flat::store`], or a byte copy for plain data.
///
/// # Safety
/// As for [`Flat::store`].
unsafe fn transfer<T: ?Sized + Flat>(src: &T, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
  // SAFETY: plain data is its bytes, and the destination is sized for it.
  unsafe {
    if T::PLAIN {
      let dst = sink.buf.ptr().add(at);
      ptr::copy_nonoverlapping(ptr::from_ref(src).cast::<u8>(), dst, size_of_val(src));
    } else {
      src.store(sink, at, shift);
    }
  }
}

/// Writes at `at` a pointer to a deep copy of `src`.
///
/// # Safety
/// `at` holds a `Near<T>` in the sink.
unsafe fn copy_to<T: ?Sized + Flat>(sink: &mut Sink<'_>, at: usize, src: &T) {
  let (size, align) = (size_of_val(src), align_of_val(src));
  // SAFETY: each position written was just allocated with the size it is written with.
  unsafe {
    if is_list::<T>() {
      let h = sink.alloc_list(align, size);
      sink.put(h, mem::transmute_copy::<_, usize>(&ptr::metadata(src)) as u32);
      transfer(src, sink, h + 4, None);
      sink.link(at, h);
    } else {
      let t = sink.alloc(size, align);
      transfer(src, sink, t, None);
      sink.link(at, t);
    }
  }
}

macro_rules! plain {
  ($($t:ty)*) => {$(
    // SAFETY: primitive values have no pointers and no padding.
    unsafe impl Flat for $t {
      const PLAIN: bool = true;
      unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, _: Option<i32>) {
        // SAFETY: the caller allocated `at` for `Self`.
        unsafe { sink.put(at, *self) }
      }
    }
  )*};
}
plain!(u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool char);

// SAFETY: an array is its elements, without padding between them.
unsafe impl<T: Flat, const N: usize> Flat for [T; N] {
  const PLAIN: bool = T::PLAIN;
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
    // SAFETY: same bytes as the slice.
    unsafe { transfer(&self[..], sink, at, shift) }
  }
}

// SAFETY: each element is stored at its own offset.
unsafe impl<T: Flat> Flat for [T] {
  const PLAIN: bool = T::PLAIN;
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
    for (i, x) in self.iter().enumerate() {
      // SAFETY: element `i` lives `i * size_of::<T>()` bytes into the allocation.
      unsafe { x.store(sink, at + i * size_of::<T>(), shift) }
    }
  }
}

// SAFETY: a `str` is its bytes.
unsafe impl Flat for str {
  const PLAIN: bool = true;
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
    // SAFETY: same bytes.
    unsafe { transfer(self.as_bytes(), sink, at, shift) }
  }
}

// SAFETY: copies the target, or re-aims the offset by the distance moved.
unsafe impl<T: ?Sized + Flat> Flat for Near<T> {
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe {
      match shift {
        Some(d) => sink.put(at, self.off.get().wrapping_add(d)),
        None => copy_to(sink, at, &**self),
      }
    }
  }
}

// SAFETY: `None` is a zero offset; `Some` is the `Near`.
unsafe impl<T: ?Sized + Flat> Flat for Option<Near<T>> {
  unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
    // SAFETY: the caller allocated `at` for a `Near`.
    unsafe {
      match self {
        Some(near) => near.store(sink, at, shift),
        None => sink.put(at, 0i32),
      }
    }
  }
}

/// Sources that can be written as a `T`: values, copies (`&T`), moves (`Pin<&'o mut T>`) and
/// builders.
///
/// | Source | Writes |
/// |---|---|
/// | `T` | the value |
/// | `&T` | a deep copy, from any region |
/// | `Pin<&'o mut T>` | a move, from the value a [`Slot<'o, _>`](crate::Slot) replaces |
/// | [`Near::new(e)`](Near::new) | `Near<T>` or `Option<Near<T>>` to a new `T` |
/// | `[e; N]`, tuples up to 12, `()` | `Near<[T]>`; tuple elements may be different builders |
/// | `&[T]` | `Near<[T]>`, a copy of the slice |
/// | [`Near::from_iter(iter)`](Near::from_iter) | `Near<[T]>` |
/// | `&str` | `Near<str>` |
/// | `Type::new(..)`, `Enum::variant(..)` | constructors made by [`#[flat]`](crate::flat) |
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
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
/// `emit` writes a valid `T` at `at`, allocating in the sink whatever it points to.
pub unsafe trait Emit<'o, T: Flat> {
  /// Writes `self` as a `T` at `at`.
  ///
  /// # Safety
  /// `at` is allocated in the sink, aligned and sized for `T`.
  #[doc(hidden)]
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize);
}

// SAFETY: a value is stored as itself.
unsafe impl<'o, T: Flat> Emit<'o, T> for T {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    // SAFETY: forwarded.
    unsafe { transfer(&self, sink, at, None) }
  }
}

// SAFETY: a deep copy is a valid `T`.
unsafe impl<'o, T: Flat> Emit<'o, T> for &T {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    // SAFETY: forwarded.
    unsafe { transfer(self, sink, at, None) }
  }
}

// SAFETY: the brand `'o` ties the pin to the value the slot replaces, so its targets stay where
// they are and get exactly one owner: the copy written here, re-aimed for the move.
unsafe impl<'o, T: Flat> Emit<'o, T> for Pin<&'o mut T> {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    let from = ptr::from_ref::<T>(&self).addr().wrapping_sub(sink.origin);
    let shift = distance(sink.base + at, from);
    // SAFETY: forwarded.
    unsafe { transfer(&*self, sink, at, Some(shift)) }
  }
}

/// Builder made by [`Near::new`].
pub struct New<E>(pub E);

// SAFETY: allocates and writes the target, then links to it.
unsafe impl<'o, T: Flat, E: Emit<'o, T>> Emit<'o, Near<T>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
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
unsafe impl<'o, T: Flat, E: Emit<'o, T>> Emit<'o, Option<Near<T>>> for New<E> {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    // SAFETY: forwarded.
    unsafe { Emit::<Near<T>>::emit(self, sink, at) }
  }
}

// SAFETY: copies the text.
unsafe impl<'o> Emit<'o, Near<str>> for &str {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    // SAFETY: forwarded.
    unsafe { copy_to(sink, at, self) }
  }
}

// SAFETY: copies the slice.
unsafe impl<'o, T: Flat> Emit<'o, Near<[T]>> for &[T] {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    fits::<T>();
    // SAFETY: forwarded.
    unsafe { copy_to(sink, at, self) }
  }
}

/// Builder made by [`Near::from_iter`].
pub struct Seq<I>(pub I);

// SAFETY: writes the elements into a list, then links to it.
unsafe impl<'o, T: Flat, I: IntoIterator<Item: Emit<'o, T>>> Emit<'o, Near<[T]>> for Seq<I> {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    let items = self.0.into_iter();
    let mut list = List::new::<T>(sink, items.size_hint().0);
    list.extend(sink, items);
    // SAFETY: forwarded.
    unsafe { list.finish(sink, at) }
  }
}

// SAFETY: as for `Seq`.
unsafe impl<'o, T: Flat, E: Emit<'o, T>, const N: usize> Emit<'o, Near<[T]>> for [E; N] {
  unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
    // SAFETY: forwarded.
    unsafe { Seq(self).emit(sink, at) }
  }
}

/// Implements `Emit` for tuples of every length from the given one down to `()`.
macro_rules! tuples {
  ($($e:ident $x:ident)*) => {
    // SAFETY: as for `Seq`.
    unsafe impl<'o, T: Flat, $($e: Emit<'o, T>),*> Emit<'o, Near<[T]>> for ($($e,)*) {
      unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
        let ($($x,)*) = self;
        let list = &mut List::new::<T>(sink, 0 $(+ { let _ = &$x; 1 })*);
        $(list.push(sink, $x);)*
        // SAFETY: forwarded.
        unsafe { list.finish(sink, at) }
      }
    }
    tuples!(@ $($e $x)*);
  };
  (@) => {};
  (@ $e:ident $x:ident $($rest:tt)*) => { tuples!($($rest)*); };
}
tuples!(A a B b C c D d E e F f G g H h I i J j K k L l);

/// A list being written into a sink: a header at `head`, then room for `cap` elements of which
/// `len` are written. The header is written by `finish`.
pub struct List {
  head: usize,
  len: usize,
  cap: usize,
}

impl List {
  pub(crate) const NONE: Self = Self { head: 0, len: 0, cap: 0 };

  pub(crate) const fn is_some(&self) -> bool {
    self.cap != 0
  }

  pub(crate) fn new<T: Flat>(sink: &mut Sink<'_>, cap: usize) -> Self {
    fits::<T>();
    let head = sink.alloc_list(align_of::<T>(), size_of::<T>().saturating_mul(cap));
    Self { head, len: 0, cap }
  }

  /// Moves the elements to new room for `cap`; the old room becomes garbage.
  fn grow<T: Flat>(&mut self, sink: &mut Sink<'_>, cap: usize) {
    let old = Self::new::<T>(sink, cap);
    let (from, to) = (self.head + 4, old.head + 4);
    // SAFETY: the `len` elements at `from` are valid, and the walk with `Some(_)` only writes
    // the new room, which does not overlap them and was allocated before they were borrowed.
    unsafe {
      let elems = slice::from_raw_parts(sink.buf.ptr().add(from).cast::<T>(), self.len);
      transfer(elems, sink, to, Some(distance(to, from)));
    }
    (self.head, self.cap) = (old.head, cap);
  }

  pub(crate) fn push<'o, T: Flat>(&mut self, sink: &mut Sink<'o>, item: impl Emit<'o, T>) {
    if self.len == self.cap {
      self.grow::<T>(sink, (self.cap * 2).max(4));
    }
    // SAFETY: element `len` is within the room.
    unsafe { item.emit(sink, self.head + 4 + self.len * size_of::<T>()) };
    self.len += 1;
  }

  fn extend<'o, T: Flat>(
    &mut self,
    sink: &mut Sink<'o>,
    mut items: impl Iterator<Item: Emit<'o, T>>,
  ) {
    loop {
      // Fast path: fill the room in one internal iteration, with no growth check and no
      // header write.
      let (room, mut at) = (self.cap - self.len, self.head + 4 + self.len * size_of::<T>());
      items.by_ref().take(room).for_each(|item| {
        // SAFETY: `at` is an element of the room.
        unsafe { item.emit(sink, at) };
        at += size_of::<T>();
      });
      let filled = (at - self.head - 4) / size_of::<T>();
      let full = filled == self.cap;
      self.len = filled;
      let Some(item) = items.next().filter(|_| full) else { return };
      self.push(sink, item);
    }
  }

  /// Writes the length and links `at` to the list.
  ///
  /// # Safety
  /// `at` holds a `Near<[T]>` in the sink.
  pub(crate) const unsafe fn finish(&self, sink: &mut Sink<'_>, at: usize) {
    // SAFETY: the header was allocated by `new`.
    unsafe {
      sink.put(self.head, self.len as u32);
      sink.link(at, self.head);
    }
  }
}

/// Wraps a closure that writes a `T`; used by the constructors `#[flat]` generates.
///
/// # Safety
/// `f(sink, at)` writes a valid `T` at `at`, given the guarantees of [`Emit::emit`].
#[doc(hidden)]
pub unsafe fn make<'o, T: Flat>(f: impl FnOnce(&mut Sink<'o>, usize)) -> impl Emit<'o, T> {
  struct Make<F>(F);
  // SAFETY: the caller of `make` vouches for `F`.
  unsafe impl<'o, T: Flat, F: FnOnce(&mut Sink<'o>, usize)> Emit<'o, T> for Make<F> {
    unsafe fn emit(self, sink: &mut Sink<'o>, at: usize) {
      (self.0)(sink, at);
    }
  }
  Make(f)
}
