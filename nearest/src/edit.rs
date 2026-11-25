use core::{
  marker::PhantomData,
  mem::{self, MaybeUninit},
  ops::{Deref, DerefMut},
  pin::Pin,
  ptr, slice,
};

use crate::{
  Emit, Flat, Near, Offset,
  flat::{
    Buf, List, Sink, aim, detach, fits, fits_behind, move_elems, relocate, stash, unstash,
    write_copy,
  },
};

/// The core of every handle: a `&'a mut Raw<'e, T>` is a fat reference whose address is the
/// region's `Buf` and whose length is the position of a `T` in it. It never holds an address
/// inside the buffer, so it stays valid when the buffer grows. `'e` is the edit.
///
/// It is neither `Send` nor `Sync` (the `*mut ()`), and so is every handle: an edit stays on
/// one thread. An [`Old`] reads its region without borrowing a handle, so a handle on another
/// thread could grow the buffer under it.
#[doc(hidden)]
#[repr(transparent)]
pub struct Raw<'e, T>(PhantomData<(&'e mut (), T, *mut ())>, [()]);

impl<'e, T> Raw<'e, T> {
  /// The handle of the `T` at `at`.
  ///
  /// # Safety
  /// As for [`Flat::edit`].
  pub const unsafe fn new<'a>(buf: *mut Buf, at: usize) -> &'a mut Self
  where
    'e: 'a,
  {
    // SAFETY: `Raw` is zero-sized, so any address is valid for it; the length carries `at`.
    unsafe { &mut *(ptr::slice_from_raw_parts_mut(buf.cast::<()>(), at) as *mut Self) }
  }

  const fn at(&self) -> usize {
    self.1.len()
  }

  /// The `me` of the region's buffer, read through this handle. Every method reads it once and
  /// reaches the buffer only through it (I8).
  const fn buf(&self) -> *mut Buf {
    let buf: *const Buf = ptr::from_ref(self).cast();
    // SAFETY: a handle's address is the `Buf` of a region being edited.
    unsafe { (*buf).me }
  }

  /// The handle of the `U` that is `off` bytes into the `T`.
  ///
  /// # Safety
  /// A `U` is at that offset whenever the `T` is valid.
  pub unsafe fn field<U: Flat>(&mut self, off: usize) -> &mut U::Edit<'e> {
    // SAFETY: per the caller; the field is reached only through the returned handle, which
    // borrows this one.
    unsafe { U::edit(self.buf(), self.at() + off) }
  }
}

impl<'e, T: Flat> Raw<'e, T> {
  /// The `T`.
  #[must_use]
  pub const fn get(&self) -> &T {
    // SAFETY: a handle designates a valid `T` in a live buffer, and borrows it.
    unsafe { &*(*self.buf()).base().add(self.at()).cast::<T>() }
  }

  /// The `T`, for writing.
  ///
  /// # Safety
  /// No safe code gets a `&mut T` out of it unless `T` has no pointers.
  unsafe fn place(&mut self) -> &mut T {
    // SAFETY: as for `get`.
    unsafe { &mut *(*self.buf()).ptr().add(self.at()).cast::<T>() }
  }

  /// The `T`, which has no pointers since it is `Copy`.
  pub fn get_mut(&mut self) -> &mut T
  where
    T: Copy,
  {
    // SAFETY: a `Copy` type has no `Near`, so assigning to it is sound.
    unsafe { self.place() }
  }

  /// The `T`, pinned: a view in which several fields can be borrowed at once. It borrows the
  /// handle, so nothing can grow the buffer while it lives, and being pinned it gives no
  /// `&mut` to anything that contains a `Near`.
  pub fn pin(&mut self) -> Pin<&mut T> {
    // SAFETY: the value stays where it is while the handle is borrowed, and pinned it gives a
    // `&mut` only to what is `Unpin`, which has no pointers.
    unsafe { Pin::new_unchecked(self.place()) }
  }

  /// Writes `value` over the `T`. The old bytes wait on the stack and come back if writing
  /// unwinds.
  pub fn set(&mut self, value: impl Emit<T>) {
    /// Puts the old bytes back.
    struct Restore<T> {
      buf: *mut Buf,
      at: usize,
      old: MaybeUninit<T>,
    }
    impl<T> Drop for Restore<T> {
      fn drop(&mut self) {
        // SAFETY: the place still holds a `T`'s room; the old bytes are a valid `T` there.
        unsafe {
          (*self.buf)
            .ptr()
            .add(self.at)
            .cast::<MaybeUninit<T>>()
            .write(ptr::read(&raw const self.old));
        }
      }
    }
    fits::<T>();
    let (buf, at) = (self.buf(), self.at());
    // SAFETY: the place holds a `T`.
    let old = unsafe { (*buf).base().add(at).cast::<MaybeUninit<T>>().read() };
    let restore = Restore { buf, at, old };
    // SAFETY: `at` holds a `T`, which the new one replaces.
    unsafe { value.emit(&mut Sink::new(buf), at) };
    mem::forget(restore);
  }

  /// Writes `value` over the `T` and returns the old one, moved to fresh bytes.
  pub fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T> {
    let (buf, at) = self.swap_in(|_, _| value);
    Old { buf, at, _e: PhantomData }
  }

  /// Writes over the `T` what `f` builds from the old one. That `Old` lives as long as the
  /// borrow of this handle: kept past the call, it freezes every handle of the region, so it
  /// can only move into what `f` returns, and anywhere else it is a copy.
  ///
  /// The old value is used at most once (these three cases are also trybuild programs,
  /// `tests/compile-fail/update_*.rs`, which pin the reason down):
  ///
  /// ```compile_fail
  /// # use nearest::{Near, Region, flat};
  /// # #[flat]
  /// # enum Expr { Lit(i64), Add(Near<Expr>, Near<Expr>) }
  /// let mut r = Region::new(Expr::Lit(1));
  /// r.edit().update(|old| Expr::add(old, old));
  /// ```
  ///
  /// The handle is borrowed while the closure runs:
  ///
  /// ```compile_fail
  /// # use nearest::{Near, Region, flat};
  /// # #[flat]
  /// # enum Expr { Lit(i64), Neg(Near<Expr>) }
  /// let mut r = Region::new(Expr::Lit(1));
  /// let e = r.edit();
  /// e.update(|old| {
  ///   e.set(Expr::Lit(2));
  ///   Expr::neg(old)
  /// });
  /// ```
  ///
  /// And for as long as the old value is kept, so it cannot move back in later:
  ///
  /// ```compile_fail
  /// # use nearest::{Near, Region, flat};
  /// # #[flat]
  /// # enum Expr { Lit(i64), Neg(Near<Expr>) }
  /// let mut r = Region::new(Expr::neg(Expr::Lit(1)));
  /// let e = r.edit();
  /// let mut stash = None;
  /// e.update(|old| {
  ///   stash = Some(old);
  ///   Expr::Lit(2)
  /// });
  /// e.set(Expr::Lit(3));
  /// drop(stash);
  /// ```
  pub fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E) {
    self.swap_in(|buf, at| f(Old { buf, at, _e: PhantomData }));
  }

  /// Moves the old `T` to fresh bytes, writes over it what `f` makes given where the old one
  /// went, and returns that place: the buffer and the position, for exactly one `Old`. If `f`
  /// or the writing unwinds, the old value moves back.
  #[expect(clippy::needless_pass_by_ref_mut, reason = "it writes the place, through the buffer")]
  fn swap_in<E: Emit<T>>(&mut self, f: impl FnOnce(*mut Buf, usize) -> E) -> (*mut Buf, usize) {
    /// Moves the old value back.
    struct Restore<T: Flat> {
      buf: *mut Buf,
      at: usize,
      kept: usize,
      _t: PhantomData<T>,
    }
    impl<T: Flat> Drop for Restore<T> {
      fn drop(&mut self) {
        // SAFETY: the old value is intact at `kept`: moves only write their destination.
        unsafe { relocate::<T>(self.buf, self.kept, self.at) }
      }
    }
    fits::<T>();
    let (buf, at) = (self.buf(), self.at());
    // SAFETY: a `T` is at `at`.
    let kept = unsafe { detach::<T>(buf, at) };
    let restore = Restore::<T> { buf, at, kept, _t: PhantomData };
    let value = f(buf, kept);
    // SAFETY: `at` holds a `T`; the old value is no longer reachable from it.
    unsafe { value.emit(&mut Sink::new(buf), at) };
    mem::forget(restore);
    (buf, kept)
  }
}

/// What every handle does, whatever the type of its place: the trait of
/// [`Edit<'e, T>`](crate::Edit), so that code generic over the stored type writes through a handle.
///
/// Every handle has these three as its own methods too, so that the trait is needed only by such
/// code. They are the same functions: `set` through the trait compiles to the machine code of
/// `set` on the handle.
///
/// ```
/// use nearest::{Edit, Emit, Flat, Handle, Near, Region, flat};
///
/// /// Writes `value` over the place, whatever it holds (`Value` is `T`, but what a pointer's
/// /// pointee is for a pointer, whose handle is the pointee's).
/// fn reset<T: Flat>(place: &mut Edit<'_, T>, value: impl Emit<T::Value>) {
///   place.set(value);
/// }
///
/// #[flat]
/// struct Node { value: u32, next: Option<Near<Node>> }
///
/// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
/// let n = r.edit();
/// reset::<Option<Near<Node>>>(n.next(), None);
/// reset::<Node>(n, Node::new(3, None));
/// assert_eq!((r.value, r.next.is_none()), (3, true));
/// ```
pub trait Handle<'e, T: Flat> {
  /// Writes `value` over the value, like `*place = value`.
  fn set(&mut self, value: impl Emit<T>);

  /// Writes `value` over the value and returns the old one, like `mem::replace`.
  fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T>;

  /// Writes over the value what `f` builds from the old one, which moves into it. See
  /// `update` on the handles of `#[flat]` types.
  fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E);
}

/// Implements [`Handle`] for a handle that wraps the `Raw` of its place in `.0`.
macro_rules! handle {
  ($([$($g:tt)*] $handle:ty => $t:ty;)*) => {$(
    impl<'e, $($g)*> Handle<'e, $t> for $handle {
      fn set(&mut self, value: impl Emit<$t>) {
        self.0.set(value);
      }
      fn replace(&mut self, value: impl Emit<$t>) -> Old<'e, $t> {
        self.0.replace(value)
      }
      fn update<'a, E: Emit<$t>>(&'a mut self, f: impl FnOnce(Old<'a, $t>) -> E) {
        self.0.update(f);
      }
    }
  )*};
}

handle! {
  [T: Flat + Copy] PlainEdit<'e, T> => T;
  [T: Flat, O: Offset] ListEdit<'e, T, O> => Near<[T], O>;
  [T: Flat, O: Offset] OptionEdit<'e, T, O> => Option<Near<T, O>>;
  [O: Offset] StrEdit<'e, O> => Near<str, O>;
}

/// The handle of plain data, `Edit<'e, T>` for a number, a `bool` or an array of plain data.
///
/// That is also a `char`, a `NonZero`, a number of [`puns::num`](crate::puns::num): it reads and
/// writes as the value (`Deref`, `DerefMut`), and has `set`, `replace` and `update` as every
/// handle has. The getters of handles (field methods, `at`, `push`, ..) give it as [`Plain`],
/// whose `*` is the value.
#[repr(transparent)]
pub struct PlainEdit<'e, T>(Raw<'e, T>);

impl<'e, T: Flat + Copy> PlainEdit<'e, T> {
  pub(crate) const fn new<'a>(raw: &'a mut Raw<'e, T>) -> &'a mut Self {
    // SAFETY: `PlainEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// Writes `value` over the value, like `*place = value`.
  pub fn set(&mut self, value: impl Emit<T>) {
    self.0.set(value);
  }

  /// Writes `value` over the value and returns the old one, like `mem::replace`; as for every
  /// handle, it is an [`Old`], which can only be written somewhere.
  pub fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T> {
    self.0.replace(value)
  }

  /// Writes over the value what `f` builds from the old one.
  pub fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E) {
    self.0.update(f);
  }
}

impl<T: Flat + Copy> Deref for PlainEdit<'_, T> {
  type Target = T;
  fn deref(&self) -> &T {
    self.0.get()
  }
}

impl<T: Flat + Copy> DerefMut for PlainEdit<'_, T> {
  fn deref_mut(&mut self) -> &mut T {
    self.0.get_mut()
  }
}

/// A place of plain data, given by value by the getters of handles: `*` is the value.
///
/// So a field reads and writes with one star, as a `&mut` would (`*e.count() += 1`), and it is a
/// handle too (`set`, `replace`, `update`). Bound by a pattern, it is mutable if the binding is
/// (`Some(mut n) => *n += 1`), as any value.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Counter { count: u32, names: Near<[Near<str>]> }
///
/// let mut r = Region::new(Counter::new(1, ()));
/// let c = r.edit();
/// *c.count() += 1;
/// let n = *c.count();
/// c.count().set(n * 10);
/// assert_eq!(r.count, 20);
/// ```
pub struct Plain<'a, 'e, T>(&'a mut PlainEdit<'e, T>);

impl<'a, 'e, T: Flat + Copy> Plain<'a, 'e, T> {
  pub(crate) const fn new(edit: &'a mut PlainEdit<'e, T>) -> Self {
    Self(edit)
  }

  /// Writes `value` over the value, like `*place = value`.
  pub fn set(&mut self, value: impl Emit<T>) {
    self.0.set(value);
  }

  /// Writes `value` over the value and returns the old one, like `mem::replace`.
  pub fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T> {
    self.0.replace(value)
  }

  /// Writes over the value what `f` builds from the old one.
  pub fn update<'b, E: Emit<T>>(&'b mut self, f: impl FnOnce(Old<'b, T>) -> E) {
    self.0.update(f);
  }

  /// The handle of the place, `&mut Edit<'e, T>`, as other places are given.
  #[must_use]
  pub const fn edit(self) -> &'a mut PlainEdit<'e, T> {
    self.0
  }
}

impl<'e, T: Flat + Copy> Handle<'e, T> for Plain<'_, 'e, T> {
  fn set(&mut self, value: impl Emit<T>) {
    self.0.set(value);
  }
  fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T> {
    self.0.replace(value)
  }
  fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E) {
    self.0.update(f);
  }
}

impl<T: Flat + Copy> Deref for Plain<'_, '_, T> {
  type Target = T;
  fn deref(&self) -> &T {
    self.0
  }
}

impl<T: Flat + Copy> DerefMut for Plain<'_, '_, T> {
  fn deref_mut(&mut self) -> &mut T {
    self.0
  }
}

/// A value that left its place, during the edit `'e`: returned by `replace`,
/// [`ListEdit::remove`] and [`OptionEdit::take`].
///
/// It can only be written somewhere: into a place of the same region it moves there in O(1),
/// whatever the size of what it points to; anywhere else (another region, `Region::new`) it is
/// copied. It is neither `Copy` nor `Clone`, and it keeps the region borrowed while it lives.
/// Dropped, it is garbage until the region is compacted. Like handles, it stays on the thread
/// of its edit.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// enum Expr { Lit(i64), Neg(Near<Expr>) }
///
/// let mut r = Region::new(Expr::Lit(3));
/// let e = r.edit();
/// let old = e.replace(Expr::Lit(0)); // like `mem::replace`
/// e.set(Expr::neg(old));              // wrap: the old value moves under the new one
/// assert!(matches!(&*r, Expr::Neg(e) if matches!(**e, Expr::Lit(3))));
/// ```
pub struct Old<'e, T> {
  /// The `me` of its region's buffer.
  buf: *mut Buf,
  at: usize,
  _e: PhantomData<(&'e mut (), T)>,
}

// SAFETY: in its own region the value moves: its bytes are copied with re-aimed pointers, and
// nothing else owns its pointees (I4, I5). Anywhere else it is a deep copy from its own region,
// which is borrowed for `'e` and where nothing writes over it (I5).
unsafe impl<T: Flat> Emit<T> for Old<'_, T> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: `at` is allocated for a `T`; `buf` is live while `'_` is.
    unsafe {
      if ptr::eq(self.buf, sink.buf()) {
        relocate::<T>(self.buf, self.at, at);
      } else {
        write_copy(&*(*self.buf).base().add(self.at).cast::<T>(), sink, at);
      }
    }
  }
}

/// The handle of a list, `Edit<'e, Near<[T], O>>`.
///
/// It reads as `[T]` and, for plain `T`, writes as `[T]` too. The operations below move
/// elements inside the list; a full list moves to a new block of twice the capacity.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Stack { items: Near<[u32]> }
///
/// let mut r = Region::new(Stack::new([3, 1, 2]));
/// let items = r.edit().items();
/// items.sort();                // a real `&mut [u32]`
/// items.push(4);
/// assert_eq!(r.items[..], [1, 2, 3, 4]);
/// ```
#[repr(transparent)]
pub struct ListEdit<'e, T, O: Offset = i32>(Raw<'e, Near<[T], O>>);

impl<'e, T: Flat, O: Offset> ListEdit<'e, T, O> {
  pub(crate) const fn new<'a>(raw: &'a mut Raw<'e, Near<[T], O>>) -> &'a mut Self {
    // SAFETY: `ListEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// The buffer, and the list as it is now.
  fn list(&self) -> (*mut Buf, List<O>) {
    let buf = self.0.buf();
    // SAFETY: the handle designates a `Near<[T]>`.
    (buf, unsafe { List::read(buf, self.0.at()) })
  }

  /// Writes `value` into the spare room after the elements, growing the list if it is full.
  fn stage(&mut self, value: impl Emit<T>) -> (*mut Buf, List<O>) {
    let (buf, mut list) = self.list();
    // SAFETY: `buf` is the edit's `me`.
    let mut sink = unsafe { Sink::new(buf) };
    // SAFETY: `list` is the list the handle points to; if it moved, the handle's pointer is
    // aimed at the new block, which has all the elements.
    unsafe {
      if list.reserve::<T>(&mut sink) {
        sink.link::<O>(self.0.at(), list.head);
      }
    }
    // SAFETY: element `len` is spare room; the length is written only once it is complete.
    unsafe { value.emit(&mut sink, list.elem::<T>(list.len)) };
    (buf, list)
  }

  /// Writes the length `len`.
  ///
  /// # Safety
  /// The first `len` elements of `list` are valid.
  unsafe fn set_len(buf: *mut Buf, list: &List<O>, len: usize) {
    // SAFETY: per the caller.
    unsafe { Sink::new(buf).write_len::<O>(list.head, len) }
  }

  /// Appends `value`; returns the handle of the new element.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Inst { op: u8, args: Near<[u32]> }
  /// #[flat]
  /// struct Block { insts: Near<[Inst]> }
  ///
  /// let mut r = Region::new(Block::new(()));
  /// r.edit().insts().push(Inst::new(1, ())).args().push(10);
  /// assert_eq!((r.insts[0].op, &r.insts[0].args[..]), (1, &[10][..]));
  /// ```
  pub fn push(&mut self, value: impl Emit<T>) -> T::Ref<'_, 'e> {
    let (buf, list) = self.stage(value);
    let at = list.elem::<T>(list.len);
    // SAFETY: the element is complete; the new handle borrows this one.
    unsafe {
      Self::set_len(buf, &list, list.len + 1);
      T::reference(T::edit(buf, at))
    }
  }

  /// Inserts `value` at `index`, moving the later elements up; returns the handle of the new
  /// element.
  ///
  /// # Panics
  /// If `index > len`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Names { names: Near<[Near<str>]> }
  ///
  /// let mut r = Region::new(Names::new(("a", "c")));
  /// r.edit().names().insert(1, "b");
  /// assert_eq!(r.names.iter().map(|s| &**s).collect::<Vec<_>>(), ["a", "b", "c"]);
  /// ```
  pub fn insert(&mut self, index: usize, value: impl Emit<T>) -> T::Ref<'_, 'e> {
    let len = self.len();
    assert!(index <= len, "insertion index (is {index}) should be <= len (is {len})");
    let (buf, list) = self.stage(value);
    let (staged, at) = (list.elem::<T>(len), list.elem::<T>(index));
    // SAFETY: the new element is complete at `staged`; the elements from `index` move up by
    // one into the room it leaves, then it moves to `index`, and only then is the length
    // written. Moves cannot panic.
    unsafe {
      let v = stash::<T>(buf, staged);
      move_elems::<T>(buf, at, list.elem::<T>(index + 1), len - index);
      unstash(buf, &v, staged, at);
      Self::set_len(buf, &list, len + 1);
      T::reference(T::edit(buf, at))
    }
  }

  /// Removes the element at `index` and returns it, moving the later elements down.
  ///
  /// # Panics
  /// If `index >= len`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Lists { a: Near<[Near<str>]>, b: Near<[Near<str>]> }
  ///
  /// let mut r = Region::new(Lists::new(("x", "y"), ()));
  /// let lists = r.edit();
  /// let x = lists.a().remove(0); // like `Vec::remove`
  /// lists.b().push(x);           // the string moves to the other list
  /// assert_eq!((r.a.len(), &*r.b[0]), (1, "x"));
  /// ```
  pub fn remove(&mut self, index: usize) -> Old<'e, T> {
    let (buf, list) = self.list();
    let len = list.len;
    assert!(index < len, "removal index (is {index}) should be < len (is {len})");
    // SAFETY: the element moves out to fresh bytes; the later elements move down over its
    // slot; then the length shrinks.
    unsafe {
      let kept = detach::<T>(buf, list.elem::<T>(index));
      move_elems::<T>(buf, list.elem::<T>(index + 1), list.elem::<T>(index), len - index - 1);
      Self::set_len(buf, &list, len - 1);
      Old { buf, at: kept, _e: PhantomData }
    }
  }

  /// Swaps the elements at `a` and `b`.
  ///
  /// # Panics
  /// If `a` or `b` is out of bounds.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Names { names: Near<[Near<str>]> }
  ///
  /// let mut r = Region::new(Names::new(("a", "b")));
  /// r.edit().names().swap(0, 1);
  /// assert_eq!((&*r.names[0], &*r.names[1]), ("b", "a"));
  /// ```
  pub fn swap(&mut self, a: usize, b: usize) {
    let (buf, list) = self.list();
    let len = list.len;
    assert!(a < len && b < len, "swap indices (are {a} and {b}) should be < len (is {len})");
    if a != b {
      let (pa, pb) = (list.elem::<T>(a), list.elem::<T>(b));
      // SAFETY: both elements are valid; `a` is read first, then each takes the other's place.
      unsafe {
        let v = stash::<T>(buf, pa);
        relocate::<T>(buf, pb, pa);
        unstash(buf, &v, pa, pb);
      }
    }
  }

  /// Shortens the list to `len` elements; does nothing if it is not longer.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Stack { items: Near<[u32]> }
  ///
  /// let mut r = Region::new(Stack::new([1, 2, 3]));
  /// r.edit().items().truncate(1);
  /// assert_eq!(r.items[..], [1]);
  /// ```
  pub fn truncate(&mut self, len: usize) {
    let (buf, list) = self.list();
    if len < list.len {
      // SAFETY: the first `len` elements stay valid.
      unsafe { Self::set_len(buf, &list, len) };
    }
  }

  /// Keeps the elements for which `f` returns `true`, in order. If `f` panics, the elements
  /// it has not visited are kept too.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Inst { op: u8, args: Near<[u32]> }
  /// #[flat]
  /// struct Block { insts: Near<[Inst]> }
  ///
  /// let mut r = Region::new(Block::new((Inst::new(1, [1]), Inst::new(2, ()), Inst::new(3, [3]))));
  /// r.edit().insts().retain(|inst| inst.op != 2);
  /// assert_eq!(r.insts.iter().map(|i| i.op).collect::<Vec<_>>(), [1, 3]);
  /// ```
  pub fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
    /// Moves the unvisited elements down and writes the length, also when `f` unwinds.
    struct Compact<T: Flat, O: Offset> {
      buf: *mut Buf,
      list: List<O>,
      read: usize,
      write: usize,
      _t: PhantomData<T>,
    }
    impl<T: Flat, O: Offset> Drop for Compact<T, O> {
      fn drop(&mut self) {
        let Self { buf, ref list, read, write, .. } = *self;
        let rest = list.len - read;
        // SAFETY: the elements before `write` are kept, those from `read` are unvisited.
        unsafe {
          move_elems::<T>(buf, list.elem::<T>(read), list.elem::<T>(write), rest);
          ListEdit::<T, O>::set_len(buf, list, write + rest);
        }
      }
    }
    let (buf, list) = self.list();
    let mut g = Compact::<T, O> { buf, list, read: 0, write: 0, _t: PhantomData };
    while g.read < g.list.len {
      let at = g.list.elem::<T>(g.read);
      // SAFETY: a valid element; the reference ends before anything is written.
      if f(unsafe { &*(*buf).base().add(at).cast::<T>() }) {
        if g.read != g.write {
          // SAFETY: moves a kept element down over an element already moved or dropped.
          unsafe { relocate::<T>(buf, at, g.list.elem::<T>(g.write)) };
        }
        g.write += 1;
      }
      g.read += 1;
    }
  }

  /// The handle of the element at `index`.
  ///
  /// # Panics
  /// If `index >= len`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Inst { op: u8, args: Near<[u32]> }
  /// #[flat]
  /// struct Block { insts: Near<[Inst]> }
  ///
  /// let mut r = Region::new(Block::new([Inst::new(1, [10])]));
  /// let inst = r.edit().insts().at(0);
  /// *inst.op() = 2;
  /// inst.args().push(11);
  /// assert_eq!((r.insts[0].op, &r.insts[0].args[..]), (2, &[10, 11][..]));
  /// ```
  pub fn at(&mut self, index: usize) -> T::Ref<'_, 'e> {
    let len = self.len();
    self
      .try_at(index)
      .unwrap_or_else(|| panic!("index out of bounds: the len is {len} but the index is {index}"))
  }

  /// The handle of the element at `index`, or `None` out of bounds: [`at`](Self::at) without
  /// the panic.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Names { names: Near<[Near<str>]> }
  ///
  /// let mut r = Region::new(Names::new(["a"]));
  /// let names = r.edit().names();
  /// if let Some(name) = names.try_at(0) {
  ///   name.set("b");
  /// }
  /// assert!(names.try_at(1).is_none());
  /// assert_eq!(&*r.names[0], "b");
  /// ```
  pub fn try_at(&mut self, index: usize) -> Option<T::Ref<'_, 'e>> {
    let (buf, list) = self.list();
    // SAFETY: a valid element; the new handle borrows this one.
    (index < list.len).then(|| unsafe { T::reference(T::edit(buf, list.elem::<T>(index))) })
  }

  /// Replaces the list with `value`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Stack { items: Near<[u32]> }
  ///
  /// let mut r = Region::new(Stack::new([1, 2]));
  /// r.edit().items().set(Near::from_iter(5..8));
  /// assert_eq!(r.items[..], [5, 6, 7]);
  /// ```
  pub fn set(&mut self, value: impl Emit<Near<[T], O>>) {
    self.0.set(value);
  }

  /// Replaces the list with `value` and returns the old list.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Pair { a: Near<[u32]>, b: Near<[u32]> }
  ///
  /// let mut r = Region::new(Pair::new([1, 2], [3]));
  /// let p = r.edit();
  /// let a = p.a().replace(());   // the whole list moves out
  /// let b = p.b().replace(a);    // and into `b`
  /// p.a().set(b);
  /// assert_eq!((&r.a[..], &r.b[..]), (&[3][..], &[1, 2][..]));
  /// ```
  pub fn replace(&mut self, value: impl Emit<Near<[T], O>>) -> Old<'e, Near<[T], O>> {
    self.0.replace(value)
  }

  /// Replaces the list with what `f` builds from the old one, which moves into it.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Tree { id: u32, kids: Near<[Tree]> }
  ///
  /// let mut r = Region::new(Tree::new(0, [Tree::new(1, ()), Tree::new(2, ())]));
  /// // The children go under a new child, without a copy.
  /// r.edit().kids().update(|old| [Tree::new(9, old)]);
  /// assert_eq!((r.kids.len(), r.kids[0].id, r.kids[0].kids.len()), (1, 9, 2));
  /// ```
  pub fn update<'a, E: Emit<Near<[T], O>>>(
    &'a mut self,
    f: impl FnOnce(Old<'a, Near<[T], O>>) -> E,
  ) {
    self.0.update(f);
  }
}

impl<T: Flat, O: Offset> ListEdit<'_, T, O> {
  /// The elements, pinned, all at once: a view for reading and for writing plain data, in
  /// which nothing can be pushed, built or moved. See `pin` on the handles of `#[flat]` types.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { id: u32, kids: Near<[Node]> }
  ///
  /// let mut r = Region::new(Node::new(0, [Node::new(1, ()), Node::new(2, ())]));
  /// let mut kids: Vec<_> = r.edit().kids().iter_pin_mut().map(|kid| kid.project().id).collect();
  /// kids.swap(0, 1); // all the handles are live together
  /// *kids[0] += 10;
  /// assert_eq!((r.kids[0].id, r.kids[1].id), (1, 12));
  /// ```
  pub fn iter_pin_mut(&mut self) -> impl ExactSizeIterator<Item = Pin<&mut T>> {
    self.0.pin().iter_pin_mut()
  }
}

impl<T: Flat, O: Offset> Deref for ListEdit<'_, T, O> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    self.0.get()
  }
}

impl<T: Flat + Copy, O: Offset> DerefMut for ListEdit<'_, T, O> {
  fn deref_mut(&mut self) -> &mut [T] {
    let (buf, list) = self.list();
    // SAFETY: the `len` elements are valid, have no pointers since `T: Copy`, and are borrowed
    // through this handle.
    unsafe { slice::from_raw_parts_mut((*buf).ptr().add(list.elem::<T>(0)).cast(), list.len) }
  }
}

/// The handle of an optional pointer, `Edit<'e, Option<Near<T, O>>>`.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Node { value: u32, next: Option<Near<Node>> }
///
/// let mut r = Region::new(Node::new(1, None));
/// let next = r.edit().next();
/// next.insert(Node::new(2, None));
/// if let Some(n) = next.as_mut() {
///   *n.value() = 3;
/// }
/// assert_eq!(r.next.as_ref().map(|n| n.value), Some(3));
/// ```
#[repr(transparent)]
pub struct OptionEdit<'e, T, O: Offset = i32>(Raw<'e, Option<Near<T, O>>>);

impl<'e, T: Flat, O: Offset> OptionEdit<'e, T, O> {
  pub(crate) const fn new<'a>(raw: &'a mut Raw<'e, Option<Near<T, O>>>) -> &'a mut Self {
    // SAFETY: `OptionEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// The handle of the pointee, if there is one.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
  /// if let Some(next) = r.edit().next().as_mut() {
  ///   *next.value() = 20;
  ///   assert!(next.next().as_mut().is_none());
  /// }
  /// assert_eq!(r.next.as_ref().unwrap().value, 20);
  /// ```
  pub fn as_mut(&mut self) -> Option<T::Ref<'_, 'e>> {
    // SAFETY: a `Some` holds a `Near<T>` at offset 0.
    self.0.get().is_some().then(|| T::reference(unsafe { self.0.field::<Near<T, O>>(0) }))
  }

  /// Points to a new pointee written from `value`, like [`Option::insert`]; returns its handle.
  /// A previous pointee is no longer reachable.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, None));
  /// r.edit().next().insert(Node::new(2, None)).next().insert(Node::new(3, None));
  /// assert_eq!(r.next.as_ref().unwrap().next.as_ref().unwrap().value, 3);
  /// ```
  pub fn insert(&mut self, value: impl Emit<T>) -> T::Ref<'_, 'e> {
    fits_behind::<T, O>();
    let (buf, at) = (self.0.buf(), self.0.at());
    // SAFETY: `buf` is the edit's `me`; the pointee is complete before the pointer is written,
    // and the new handle borrows this one.
    unsafe {
      let mut sink = Sink::new(buf);
      let t = sink.alloc(size_of::<T>(), align_of::<T>());
      value.emit(&mut sink, t);
      sink.link::<O>(at, t);
      T::reference(T::edit(buf, t))
    }
  }

  /// Takes the pointee out, leaving `None`, like [`Option::take`].
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
  /// let n = r.edit();
  /// let next = n.next().take().unwrap();
  /// n.next().insert(Node::new(3, Near::new(next))); // put a node in front
  /// let next = r.next.as_ref().unwrap();
  /// assert_eq!((next.value, next.next.as_ref().unwrap().value), (3, 2));
  /// ```
  pub fn take(&mut self) -> Option<Old<'e, T>> {
    let (buf, at) = (self.0.buf(), self.0.at());
    self.0.get().is_some().then(|| {
      // SAFETY: the place holds a pointer to a `T`, which nothing reaches once it is cleared.
      unsafe {
        let pointee = aim::<O>(buf, at);
        Sink::new(buf).write(at, O::from_isize(0));
        Old { buf, at: pointee, _e: PhantomData }
      }
    })
  }

  /// Replaces the pointer with `value`: `None`, or `Near::new(..)`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
  /// r.edit().next().set(None);
  /// assert!(r.next.is_none());
  /// ```
  pub fn set(&mut self, value: impl Emit<Option<Near<T, O>>>) {
    self.0.set(value);
  }

  /// Replaces the pointer with `value` and returns the old one, like `mem::replace`; `take`
  /// returns the pointee instead.
  pub fn replace(&mut self, value: impl Emit<Option<Near<T, O>>>) -> Old<'e, Option<Near<T, O>>> {
    self.0.replace(value)
  }

  /// Replaces the pointer with what `f` builds from the old one, which moves into it.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(3, None))));
  /// // A node goes in front of the rest of the chain.
  /// r.edit().next().update(|rest| Near::new(Node::new(2, rest)));
  /// let next = r.next.as_ref().unwrap();
  /// assert_eq!((next.value, next.next.as_ref().unwrap().value), (2, 3));
  /// ```
  pub fn update<'a, E: Emit<Option<Near<T, O>>>>(
    &'a mut self,
    f: impl FnOnce(Old<'a, Option<Near<T, O>>>) -> E,
  ) {
    self.0.update(f);
  }
}

impl<T: Flat, O: Offset> Deref for OptionEdit<'_, T, O> {
  type Target = Option<Near<T, O>>;
  fn deref(&self) -> &Option<Near<T, O>> {
    self.0.get()
  }
}

/// The handle of a string, `Edit<'e, Near<str, O>>`.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// struct Func { name: Near<str> }
///
/// let mut r = Region::new(Func::new("f"));
/// let name = r.edit().name();
/// assert_eq!(&**name, "f");
/// name.set("g");
/// assert_eq!(&*r.name, "g");
/// ```
#[repr(transparent)]
pub struct StrEdit<'e, O: Offset = i32>(Raw<'e, Near<str, O>>);

impl<'e, O: Offset> StrEdit<'e, O> {
  pub(crate) const fn new<'a>(raw: &'a mut Raw<'e, Near<str, O>>) -> &'a mut Self {
    // SAFETY: `StrEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// Replaces the string with `value`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Func { name: Near<str> }
  ///
  /// let mut r = Region::new(Func::new("f"));
  /// r.edit().name().set("main");
  /// assert_eq!(&*r.name, "main");
  /// ```
  pub fn set(&mut self, value: impl Emit<Near<str, O>>) {
    self.0.set(value);
  }

  /// Replaces the string with `value` and returns the old one.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Func { name: Near<str>, alias: Near<str> }
  ///
  /// let mut r = Region::new(Func::new("f", "g"));
  /// let f = r.edit();
  /// let name = f.name().replace("main");
  /// f.alias().set(name);
  /// assert_eq!((&*r.name, &*r.alias), ("main", "f"));
  /// ```
  pub fn replace(&mut self, value: impl Emit<Near<str, O>>) -> Old<'e, Near<str, O>> {
    self.0.replace(value)
  }

  /// Replaces the string with what `f` builds from the old one, which moves into it.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Func { name: Near<str> }
  ///
  /// let mut r = Region::new(Func::new("f"));
  /// r.edit().name().update(|old| old); // the old string moves back in
  /// assert_eq!(&*r.name, "f");
  /// ```
  pub fn update<'a, E: Emit<Near<str, O>>>(
    &'a mut self,
    f: impl FnOnce(Old<'a, Near<str, O>>) -> E,
  ) {
    self.0.update(f);
  }
}

impl<O: Offset> Deref for StrEdit<'_, O> {
  type Target = str;
  fn deref(&self) -> &str {
    self.0.get()
  }
}
