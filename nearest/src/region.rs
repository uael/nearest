use alloc::vec::Vec;
use core::{
  fmt,
  hash::{Hash, Hasher},
  marker::PhantomData,
  ops::Deref,
  pin::Pin,
  ptr, slice,
};

use crate::{
  Emit, Flat, Near,
  flat::{Buf, List, Sink, align_up, distance, fits},
};

/// One buffer holding a `T` at byte 0 and everything it points to.
///
/// Reading is plain `&T` through [`Deref`]; [`Clone`] is one `memcpy`; [`edit`](Self::edit)
/// changes it in place.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// #[derive(Debug, PartialEq)]
/// struct Tree { value: u32, children: Near<[Tree]> }
///
/// let r = Region::new(Tree::new(1, [Tree::new(2, ()), Tree::new(3, ())]));
/// let copy = r.clone();
/// assert_eq!(copy.children.iter().map(|c| c.value).sum::<u32>(), 5);
/// assert_eq!(copy, r);
/// ```
pub struct Region<T> {
  buf: Buf,
  _t: PhantomData<T>,
}

impl<T: Flat> Region<T> {
  /// A new region holding `value`.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }
  ///
  /// let r = Region::new(Expr::add(Near::new(Expr::Lit(1)), Near::new(Expr::Lit(2))));
  /// assert!(matches!(&*r, Expr::Add(a, _) if matches!(**a, Expr::Lit(1))));
  /// // A copy into a new region keeps only what is reachable.
  /// let compact = Region::new(&*r);
  /// assert_eq!(compact.byte_len(), r.byte_len());
  /// ```
  pub fn new(value: impl Emit<'static, T>) -> Self {
    fits::<T>();
    let mut sink = Sink::new(0, 0);
    let at = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `at` was just allocated for a `T`.
    unsafe { value.emit(&mut sink, at) };
    Self { buf: sink.buf, _t: PhantomData }
  }

  /// Edits the region: `f` gets the root, pinned, and the [`Slot`] that replaces it.
  ///
  /// Plain fields are changed in place through the pin. Changes that need new memory go
  /// through the slot; they are appended to the region when `f` returns, or unwinds.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Block { id: u32, insts: Near<[u8]> }
  ///
  /// let mut r = Region::new(Block::new(0, [1, 2, 3]));
  /// let old_id = r.edit(|root, slot| {
  ///   let Block { id, insts } = root;
  ///   let old_id = *id;
  ///   *id.get_mut() = 7;
  ///   slot.at(insts, |old, new| {
  ///     for x in old { if *x != 2 { new.push(x) } }
  ///     new.push(4);
  ///   });
  ///   old_id
  /// });
  /// assert_eq!((old_id, r.id, &r.insts[..]), (0, 7, &[1, 3, 4][..]));
  /// ```
  pub fn edit<R>(&mut self, f: impl for<'a> FnOnce(Pin<&'a mut T>, &mut Slot<'a, T>) -> R) -> R {
    /// Appends what the edit built, also when it unwinds.
    struct Graft<'r, 'o> {
      main: &'r mut Buf,
      sink: Sink<'o>,
    }
    impl Drop for Graft<'_, '_> {
      fn drop(&mut self) {
        let (base, n) = (self.sink.base, self.sink.buf.len);
        if n != 0 {
          let at = self.main.alloc(base, n, 0);
          // SAFETY: `n` bytes were allocated at `at`, and the sink holds `n` bytes.
          unsafe { ptr::copy_nonoverlapping(self.sink.buf.ptr(), self.main.ptr().add(at), n) }
        }
      }
    }
    let root = self.buf.ptr().cast::<T>();
    let sink = Sink::new(align_up(self.buf.len, 8), root.addr());
    let mut graft = Graft { main: &mut self.buf, sink };
    // SAFETY: the root is a `T` at byte 0, and the sink outlives the call.
    unsafe { Slot::run(root, &raw mut graft.sink, f) }
  }

  /// A compact, fully initialized image of the region, for [`from_bytes`](Self::from_bytes).
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Msg { id: u8, body: Near<str> }
  ///
  /// let r = Region::new(Msg::new(1, "hi"));
  /// let bytes = r.to_bytes();
  /// // SAFETY: the bytes come from `to_bytes` of a `Region<Msg>` in this program.
  /// let back = unsafe { Region::<Msg>::from_bytes(&bytes) };
  /// assert_eq!(&*back.body, "hi");
  /// ```
  #[must_use]
  pub fn to_bytes(&self) -> Vec<u8> {
    let mut copy = Self::new(&**self);
    // SAFETY: the first `len` bytes of a new region are initialized: zero-filled, then written.
    unsafe { slice::from_raw_parts(copy.buf.ptr(), copy.buf.len).to_vec() }
  }

  /// The region whose image is `bytes`.
  ///
  /// ```
  /// use nearest::Region;
  ///
  /// let bytes = Region::new(5u64).to_bytes();
  /// // SAFETY: the bytes come from `to_bytes` of a `Region<u64>` in this program.
  /// assert_eq!(*unsafe { Region::<u64>::from_bytes(&bytes) }, 5);
  /// ```
  ///
  /// # Safety
  /// `bytes` came from [`to_bytes`](Self::to_bytes) of a `Region<T>`, for the same `T` in the
  /// same program.
  #[must_use]
  pub unsafe fn from_bytes(bytes: &[u8]) -> Self {
    let mut buf = Buf::new();
    let at = buf.alloc(0, bytes.len(), 0);
    // SAFETY: `bytes.len()` bytes were allocated at `at`.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr().add(at), bytes.len()) };
    Self { buf, _t: PhantomData }
  }

  /// Bytes in use, including those no longer reachable after edits.
  ///
  /// ```
  /// use nearest::Region;
  ///
  /// assert_eq!(Region::new(1u32).byte_len(), 4);
  /// ```
  #[must_use]
  pub const fn byte_len(&self) -> usize {
    self.buf.len
  }
}

impl<T> Deref for Region<T> {
  type Target = T;
  fn deref(&self) -> &T {
    // SAFETY: a region holds a `T` at byte 0 of its 8-aligned buffer.
    unsafe { &*self.buf.words.as_ptr().cast() }
  }
}

impl<T> Clone for Region<T> {
  fn clone(&self) -> Self {
    let words = self.buf.words[..self.buf.len.div_ceil(8)].to_vec();
    Self { buf: Buf { words, len: self.buf.len }, _t: PhantomData }
  }
}

impl<T: fmt::Debug> fmt::Debug for Region<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    (**self).fmt(f)
  }
}

impl<T: PartialEq> PartialEq for Region<T> {
  fn eq(&self, other: &Self) -> bool {
    **self == **other
  }
}

impl<T: Eq> Eq for Region<T> {}

impl<T: Hash> Hash for Region<T> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    (**self).hash(state);
  }
}

/// The pending replacement of one place in a region, during [`Region::edit`].
///
/// It accepts values, copies, builders, and moves (`Pin<&'o mut _>`) of parts of the value it
/// replaces. The replacement is written when the closure that received the slot returns.
///
/// ```
/// # #![feature(pin_ergonomics)]
/// # #![allow(incomplete_features)]
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Pair { left: Near<str>, right: Near<str> }
///
/// let mut r = Region::new(Pair::new("a", "b"));
/// // The root's slot accepts moves of the old root's parts: here, a swap.
/// r.edit(|old, slot| {
///   let Pair { left, right } = old;
///   slot.put(Pair::new(right, left));
/// });
/// assert_eq!((&*r.left, &*r.right), ("b", "a"));
/// ```
pub struct Slot<'o, T> {
  place: *mut T,
  sink: *mut Sink<'o>,
  staged: Option<usize>,
  list: List,
}

impl<'o, T: Flat> Slot<'o, T> {
  /// Runs `f` on the place and its slot, then commits the slot.
  ///
  /// # Safety
  /// `place` is a pinned `T` inside the region `sink` appends to, valid for `'o`.
  unsafe fn run<R>(
    place: *mut T,
    sink: *mut Sink<'o>,
    f: impl FnOnce(Pin<&'o mut T>, &mut Self) -> R,
  ) -> R
  where
    T: 'o,
  {
    let mut slot = Self { place, sink, staged: None, list: List::NONE };
    // SAFETY: per the caller.
    let out = f(unsafe { Pin::new_unchecked(&mut *place) }, &mut slot);
    slot.commit();
    out
  }

  /// Writes the staged replacement into the place, its pointers aimed at where the scratch
  /// will be once appended.
  fn commit(&self) {
    // SAFETY: the sink outlives every slot of the edit, and no other reference to it is live.
    let sink = unsafe { &mut *self.sink };
    let Some(at) = self.staged else { return };
    if self.list.is_some() {
      // SAFETY: `at` holds the `Near<[_]>` staged by `push`.
      unsafe { self.list.finish(sink, at) };
    }
    let tmp = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `at` holds a `T`; `tmp` was allocated for one after `staged` was last moved, and
    // the place is written through the pointer the closure's pin was derived from, after the
    // closure returned.
    unsafe {
      let staged = &*sink.buf.ptr().add(at).cast::<T>();
      let shift = distance(self.place.addr().wrapping_sub(sink.origin), sink.base + at);
      staged.store(sink, tmp, Some(shift));
      ptr::copy_nonoverlapping(sink.buf.ptr().add(tmp), self.place.cast(), size_of::<T>());
    }
  }

  /// Stages `value` as the replacement; the last call wins.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// enum Expr { Lit(i64), Neg(Near<Expr>) }
  ///
  /// let mut r = Region::new(Expr::Lit(3));
  /// // Wrap: the old root moves under a new node.
  /// r.edit(|old, slot| slot.put(Expr::neg(Near::new(old))));
  /// assert!(matches!(&*r, Expr::Neg(e) if matches!(**e, Expr::Lit(3))));
  /// // Unwrap: the child moves up.
  /// r.edit(|old, slot| if let Expr::Neg(inner) = old { slot.put(inner.as_pin_mut()) });
  /// assert!(matches!(*r, Expr::Lit(3)));
  /// ```
  pub fn put(&mut self, value: impl Emit<'o, T>) {
    // SAFETY: as in `commit`.
    let sink = unsafe { &mut *self.sink };
    let at = sink.alloc(size_of::<T>(), align_of::<T>());
    // SAFETY: `at` was just allocated for a `T`.
    unsafe { value.emit(sink, at) };
    (self.staged, self.list) = (Some(at), List::NONE);
  }

  /// Works on `place`, a part of the old value: `f` gets it, pinned, and its own slot.
  /// Returns what `f` returns.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Func { name: Near<str>, body: Near<[u32]> }
  ///
  /// let mut r = Region::new(Func::new("f", [1, 2]));
  /// let old_len = r.edit(|root, slot| {
  ///   let Func { name, body } = root;
  ///   slot.at(name, |_, new| new.put("g"));
  ///   slot.at(body, |old, new| {
  ///     new.put(Near::from_iter(old.iter().map(|x| x * 10)));
  ///     old.len()
  ///   })
  /// });
  /// assert_eq!((&*r.name, &r.body[..], old_len), ("g", &[10, 20][..], 2));
  /// ```
  pub fn at<U: Flat, R>(
    &mut self,
    place: Pin<&'o mut U>,
    f: impl for<'p> FnOnce(Pin<&'p mut U>, &mut Slot<'p, U>) -> R,
  ) -> R {
    // SAFETY: the pin has the full lifetime of this level, so the place stays untouched by
    // anything else until the edit ends; the sink is shared by every level of the edit.
    unsafe { Slot::run(Pin::get_unchecked_mut(place), self.sink.cast(), f) }
  }
}

impl<'o, T: Flat> Slot<'o, Near<[T]>> {
  /// Appends `value` to the replacement list, which the first `push` starts empty. A slot that
  /// gets no `push` and no `put` leaves its place unchanged: to empty a list, `put(())`.
  ///
  /// ```
  /// # #![feature(pin_ergonomics)]
  /// # #![allow(incomplete_features)]
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Stack { items: Near<[u16]> }
  ///
  /// let mut r = Region::new(Stack::new([1, 2]));
  /// r.edit(|root, slot| {
  ///   let Stack { items } = root;
  ///   slot.at(items, |old, new| {
  ///     for x in old.into_iter().rev() { new.push(x) }  // elements move
  ///     new.push(0);
  ///   });
  /// });
  /// assert_eq!(r.items[..], [2, 1, 0]);
  /// ```
  pub fn push(&mut self, value: impl Emit<'o, T>) {
    // SAFETY: as in `commit`.
    let sink = unsafe { &mut *self.sink };
    if !self.list.is_some() {
      self.staged = Some(sink.alloc(4, 4));
      self.list = List::new::<T>(sink, 4);
    }
    self.list.push(sink, value);
  }
}
