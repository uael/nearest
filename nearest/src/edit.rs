use core::{
  marker::PhantomData,
  mem,
  ops::{Deref, DerefMut},
  ptr, slice,
};

use crate::{
  Edit, Emit, Flat, Near,
  flat::{Buf, List, Sink, fits, move_elems, put_moved, relocate, write_copy},
};

/// The core of every handle: a `&mut Raw<T>` is a fat reference whose address is the region's
/// `Buf` and whose length is the position of a `T` in it. It never holds an address inside the
/// buffer, so it stays valid when the buffer grows.
#[doc(hidden)]
#[repr(transparent)]
pub struct Raw<T>(PhantomData<T>, [()]);

impl<T> Raw<T> {
  /// The handle of the `T` at `at`.
  ///
  /// # Safety
  /// As for [`Flat::edit`].
  pub const unsafe fn new<'a>(buf: *mut Buf, at: usize) -> &'a mut Self {
    // SAFETY: `Raw` is zero-sized, so any address is valid for it; the length carries `at`.
    unsafe { &mut *(ptr::slice_from_raw_parts_mut(buf.cast::<()>(), at) as *mut Self) }
  }

  const fn at(&self) -> usize {
    self.1.len()
  }

  /// The region's buffer. Every method takes it once and reaches the buffer only through it.
  const fn buf(&mut self) -> *mut Buf {
    ptr::from_mut(self).cast()
  }

  /// The handle of the `U` that is `off` bytes into the `T`.
  ///
  /// # Safety
  /// A `U` is at that offset whenever the `T` is valid.
  pub unsafe fn field<U: Flat>(&mut self, off: usize) -> &mut U::Edit {
    // SAFETY: per the caller; the field is reached only through the returned handle, which
    // borrows this one.
    unsafe { U::edit(self.buf(), self.at() + off) }
  }
}

impl<T: Flat> Raw<T> {
  /// The `T`.
  #[must_use]
  pub const fn get(&self) -> &T {
    let buf: *const Buf = ptr::from_ref(self).cast();
    // SAFETY: a handle designates a valid `T` in a live buffer, and borrows it.
    unsafe { &*(*buf).base().add(self.at()).cast::<T>() }
  }

  /// The `T`, which has no pointers since it is `Copy`.
  pub fn get_mut(&mut self) -> &mut T
  where
    T: Copy,
  {
    // SAFETY: as for `get`; a `Copy` type has no `Near`, so assigning to it is sound.
    unsafe { &mut *(*self.buf()).ptr().add(self.at()).cast::<T>() }
  }

  /// Replaces the `T` with `value`.
  pub fn set(&mut self, value: impl Emit<T>) {
    self.replace(|_| value);
  }

  /// Replaces the `T` with what `f` makes of the old one.
  pub fn replace<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E) {
    /// Moves the old value back if writing the new one unwinds.
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
    // SAFETY: the buffer is live and only reached through `buf` here. The old value moves to
    // fresh memory first, so the place can be written directly and whatever `Old` designates
    // stays intact.
    let kept = unsafe { (*buf).alloc(size_of::<T>(), align_of::<T>()) };
    // SAFETY: a `T` is at `at`, and `kept` was just allocated for one.
    unsafe { relocate::<T>(buf, at, kept) };
    let restore = Restore::<T> { buf, at, kept, _t: PhantomData };
    let value = f(Old { src: Src::At(buf, kept), _borrow: PhantomData });
    // SAFETY: `at` holds a `T`; the old value is no longer reachable from it.
    unsafe { value.emit(&mut Sink::new(buf), at) };
    mem::forget(restore);
  }
}

enum Src<'a, T> {
  /// A position in a region's buffer.
  At(*mut Buf, usize),
  /// A value outside any region being edited.
  Out(&'a T),
}

/// The value being replaced, in the closure given to `replace`.
///
/// Writing it into the new value moves it there, in O(1), without copying what it points to.
/// It is neither `Copy` nor `Clone`, so it is used at most once; it keeps the handle it came
/// from borrowed for `'a`. Written into another region, it is copied.
///
/// ```
/// use nearest::{Near, Region, flat};
///
/// #[flat]
/// enum Expr { Lit(i64), Neg(Near<Expr>) }
///
/// let mut r = Region::new(Expr::Lit(3));
/// r.edit().replace(|old| Expr::neg(old)); // wrap
/// assert!(matches!(&*r, Expr::Neg(e) if matches!(**e, Expr::Lit(3))));
/// ```
pub struct Old<'a, T> {
  src: Src<'a, T>,
  _borrow: PhantomData<&'a mut T>,
}

impl<'a, T: Flat> Old<'a, T> {
  /// Narrows to a part of the old value, for example a child that takes its parent's place.
  ///
  /// A part that is not inside the old value (a `static`) is copied when written.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// enum Expr { Lit(i64), Neg(Near<Expr>) }
  ///
  /// let mut r = Region::new(Expr::neg(Expr::neg(Expr::Lit(3))));
  /// // Unwrap: the grandchild moves up.
  /// r.edit().replace(|old| {
  ///   old.map(|e| match e {
  ///     Expr::Neg(x) => match &**x { Expr::Neg(y) => &**y, x => x },
  ///     e => e,
  ///   })
  /// });
  /// assert!(matches!(*r, Expr::Lit(3)));
  /// ```
  pub fn map<U: Flat>(self, part: impl FnOnce(&T) -> &U) -> Old<'a, U> {
    fits::<U>();
    let src = match self.src {
      Src::Out(r) => Src::Out(part(r)),
      Src::At(buf, pos) => {
        // SAFETY: the buffer is borrowed for `'a` and holds a valid `T` at `pos`.
        let (base, len) = unsafe { ((*buf).base(), (*buf).len()) };
        // SAFETY: as above; the reference is dropped below unless it is outside the buffer.
        let u = part(unsafe { &*base.add(pos).cast::<T>() });
        let off = ptr::from_ref(u).addr().wrapping_sub(base.addr());
        if off < len { Src::At(buf, off) } else { Src::Out(u) }
      }
    };
    Old { src, _borrow: PhantomData }
  }
}

// SAFETY: in its own region the old value moves: its bytes are copied with re-aimed pointers,
// and nothing else owns its targets. Anywhere else it is a deep copy, from a buffer that is
// still borrowed and unchanged (I5).
unsafe impl<T: Flat> Emit<T> for Old<'_, T> {
  unsafe fn emit(self, sink: &mut Sink, at: usize) {
    // SAFETY: `at` is allocated for a `T`; `buf` is live while `'_` is.
    unsafe {
      match self.src {
        Src::At(buf, pos) if ptr::eq(buf, sink.buf()) => relocate::<T>(buf, pos, at),
        Src::At(buf, pos) => write_copy(&*(*buf).base().add(pos).cast::<T>(), sink, at),
        Src::Out(r) => write_copy(r, sink, at),
      }
    }
  }
}

/// The handle of a list, `Edit<Near<[T]>>`.
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
pub struct ListEdit<T>(Raw<Near<[T]>>);

impl<T: Flat> ListEdit<T> {
  pub(crate) const fn new(raw: &mut Raw<Near<[T]>>) -> &mut Self {
    // SAFETY: `ListEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// The buffer, and the list as it is now.
  fn list(&mut self) -> (*mut Buf, List) {
    let buf = self.0.buf();
    // SAFETY: the handle designates a `Near<[T]>`.
    (buf, unsafe { List::read(buf, self.0.at()) })
  }

  /// Writes `value` into the spare room after the elements, growing the list if it is full.
  fn stage(&mut self, value: impl Emit<T>) -> (*mut Buf, List) {
    let (buf, mut list) = self.list();
    // SAFETY: `buf` is live, and only reached through it.
    let mut sink = unsafe { Sink::new(buf) };
    if list.len == list.cap {
      // SAFETY: `list` is the list the handle points to; then the handle's pointer is aimed at
      // the new block, which has all the elements.
      unsafe {
        list.grow::<T>(&mut sink, (list.cap * 2).max(4));
        sink.link(self.0.at(), list.head);
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
  const unsafe fn set_len(buf: *mut Buf, list: &List, len: usize) {
    // SAFETY: per the caller.
    unsafe { Sink::new(buf).write(list.head, len as u32) }
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
  pub fn push(&mut self, value: impl Emit<T>) -> &mut Edit<T> {
    let (buf, list) = self.stage(value);
    let at = list.elem::<T>(list.len);
    // SAFETY: the element is complete; the new handle borrows this one.
    unsafe {
      Self::set_len(buf, &list, list.len + 1);
      T::edit(buf, at)
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
  pub fn insert(&mut self, index: usize, value: impl Emit<T>) -> &mut Edit<T> {
    let len = self.len();
    assert!(index <= len, "insertion index (is {index}) should be <= len (is {len})");
    let (buf, list) = self.stage(value);
    let (staged, at) = (list.elem::<T>(len), list.elem::<T>(index));
    // SAFETY: the new element is complete at `staged`; the elements from `index` move up by
    // one into the room it leaves, then it moves to `index`, and only then is the length
    // written. Moves cannot panic.
    unsafe {
      let v = (*buf).base().add(staged).cast::<T>().read();
      move_elems::<T>(buf, at, list.elem::<T>(index + 1), len - index);
      put_moved(buf, &v, staged, at);
      Self::set_len(buf, &list, len + 1);
      T::edit(buf, at)
    }
  }

  /// Removes the element at `index`, moving the later elements down.
  ///
  /// # Panics
  /// If `index >= len`.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Stack { items: Near<[u32]> }
  ///
  /// let mut r = Region::new(Stack::new([1, 2, 3]));
  /// r.edit().items().remove(0);
  /// assert_eq!(r.items[..], [2, 3]);
  /// ```
  pub fn remove(&mut self, index: usize) {
    let (buf, list) = self.list();
    let len = list.len;
    assert!(index < len, "removal index (is {index}) should be < len (is {len})");
    // SAFETY: the later elements move down over the removed one; then the length shrinks.
    unsafe {
      move_elems::<T>(buf, list.elem::<T>(index + 1), list.elem::<T>(index), len - index - 1);
      Self::set_len(buf, &list, len - 1);
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
        let v = (*buf).base().add(pa).cast::<T>().read();
        relocate::<T>(buf, pb, pa);
        put_moved(buf, &v, pa, pb);
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
    struct Compact<T: Flat> {
      buf: *mut Buf,
      list: List,
      read: usize,
      write: usize,
      _t: PhantomData<T>,
    }
    impl<T: Flat> Drop for Compact<T> {
      fn drop(&mut self) {
        let Self { buf, ref list, read, write, .. } = *self;
        let rest = list.len - read;
        // SAFETY: the elements before `write` are kept, those from `read` are unvisited.
        unsafe {
          move_elems::<T>(buf, list.elem::<T>(read), list.elem::<T>(write), rest);
          ListEdit::<T>::set_len(buf, list, write + rest);
        }
      }
    }
    let (buf, list) = self.list();
    let mut g = Compact::<T> { buf, list, read: 0, write: 0, _t: PhantomData };
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
  pub fn at(&mut self, index: usize) -> &mut Edit<T> {
    let (buf, list) = self.list();
    let len = list.len;
    assert!(index < len, "index out of bounds: the len is {len} but the index is {index}");
    // SAFETY: a valid element; the new handle borrows this one.
    unsafe { T::edit(buf, list.elem::<T>(index)) }
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
  pub fn set(&mut self, value: impl Emit<Near<[T]>>) {
    self.0.set(value);
  }

  /// Replaces the list with what `f` makes of the old one.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Pair { a: Near<[u32]>, b: Near<[u32]> }
  ///
  /// let mut r = Region::new(Pair::new([1], [2]));
  /// // Each element of the new list moves from the old one.
  /// r.edit().a().replace(|old| (old.map(|l| &l[0]), 3));
  /// assert_eq!(r.a[..], [1, 3]);
  /// ```
  pub fn replace<'a, E: Emit<Near<[T]>>>(&'a mut self, f: impl FnOnce(Old<'a, Near<[T]>>) -> E) {
    self.0.replace(f);
  }
}

impl<T: Flat> Deref for ListEdit<T> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    self.0.get()
  }
}

impl<T: Flat + Copy> DerefMut for ListEdit<T> {
  fn deref_mut(&mut self) -> &mut [T] {
    let (buf, list) = self.list();
    // SAFETY: the `len` elements are valid, have no pointers since `T: Copy`, and are borrowed
    // through this handle.
    unsafe { slice::from_raw_parts_mut((*buf).ptr().add(list.elem::<T>(0)).cast(), list.len) }
  }
}

/// The handle of an optional pointer, `Edit<Option<Near<T>>>`.
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
pub struct OptionEdit<T>(Raw<Option<Near<T>>>);

impl<T: Flat> OptionEdit<T> {
  pub(crate) const fn new(raw: &mut Raw<Option<Near<T>>>) -> &mut Self {
    // SAFETY: `OptionEdit` is a transparent wrapper.
    unsafe { &mut *(ptr::from_mut(raw) as *mut Self) }
  }

  /// The handle of the target, if there is one.
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
  pub fn as_mut(&mut self) -> Option<&mut Edit<T>> {
    // SAFETY: a `Some` holds a `Near<T>` at offset 0.
    self.0.get().is_some().then(|| unsafe { self.0.field::<Near<T>>(0) })
  }

  /// Points to a new target written from `value`, like [`Option::insert`]; returns its handle.
  /// A previous target is no longer reachable.
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
  pub fn insert(&mut self, value: impl Emit<T>) -> &mut Edit<T> {
    fits::<T>();
    let (buf, at) = (self.0.buf(), self.0.at());
    // SAFETY: `buf` is live and only reached through it; the target is complete before the
    // pointer is written, and the new handle borrows this one.
    unsafe {
      let mut sink = Sink::new(buf);
      let t = sink.alloc(size_of::<T>(), align_of::<T>());
      value.emit(&mut sink, t);
      sink.link(at, t);
      T::edit(buf, t)
    }
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
  pub fn set(&mut self, value: impl Emit<Option<Near<T>>>) {
    self.0.set(value);
  }

  /// Replaces the pointer with what `f` makes of the old one.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Node { value: u32, next: Option<Near<Node>> }
  ///
  /// let mut r = Region::new(Node::new(1, Near::new(Node::new(2, None))));
  /// // Push a node in front of the old one, which moves.
  /// r.edit().next().replace(|old| Near::new(Node::new(3, old)));
  /// let next = r.next.as_ref().unwrap();
  /// assert_eq!((next.value, next.next.as_ref().unwrap().value), (3, 2));
  /// ```
  pub fn replace<'a, E: Emit<Option<Near<T>>>>(
    &'a mut self,
    f: impl FnOnce(Old<'a, Option<Near<T>>>) -> E,
  ) {
    self.0.replace(f);
  }
}

impl<T: Flat> Deref for OptionEdit<T> {
  type Target = Option<Near<T>>;
  fn deref(&self) -> &Option<Near<T>> {
    self.0.get()
  }
}

/// The handle of a string, `Edit<Near<str>>`.
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
pub struct StrEdit(Raw<Near<str>>);

impl StrEdit {
  pub(crate) const fn new(raw: &mut Raw<Near<str>>) -> &mut Self {
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
  pub fn set(&mut self, value: impl Emit<Near<str>>) {
    self.0.set(value);
  }

  /// Replaces the string with what `f` makes of the old one.
  ///
  /// ```
  /// use nearest::{Near, Region, flat};
  ///
  /// #[flat]
  /// struct Func { name: Near<str> }
  ///
  /// let mut r = Region::new(Func::new("f"));
  /// let upper = r.name.to_uppercase();
  /// r.edit().name().replace(|_| upper.as_str());
  /// assert_eq!(&*r.name, "F");
  /// ```
  pub fn replace<'a, E: Emit<Near<str>>>(&'a mut self, f: impl FnOnce(Old<'a, Near<str>>) -> E) {
    self.0.replace(f);
  }
}

impl Deref for StrEdit {
  type Target = str;
  fn deref(&self) -> &str {
    self.0.get()
  }
}
