use core::{hash::Hash, marker::PhantomData};

use crate::{Emit, Flat, Near, NearList, Patch, Region, buf::Buf, emitter::Pos, list::Segment};

/// Invariant phantom lifetime brand. Zero-sized.
///
/// Uses `fn(&'id ()) -> &'id ()` to make `'id` **invariant** (not covariant,
/// not contravariant). This prevents the compiler from unifying two different
/// sessions' brands, which is the foundation of the ghost-cell safety
/// guarantee. See the [Nomicon on variance](https://doc.rust-lang.org/nomicon/subtyping.html).
#[derive(Copy, Clone)]
pub(crate) struct Brand<'id>(PhantomData<fn(&'id ()) -> &'id ()>);

impl Brand<'_> {
  pub(crate) const fn new() -> Self {
    Self(PhantomData)
  }
}

/// Typed position branded to a [`Session`]. `Copy`, no borrow.
///
/// `Ref<'id, T>` carries only a byte position (4 bytes) and a phantom brand —
/// it borrows nothing, so multiple Refs can coexist with `&mut Session` without
/// borrow conflicts.
///
/// # Soundness
///
/// The `'id` lifetime is **invariant** (via `Brand`'s
/// `fn(&'id ()) -> &'id ()` phantom). Combined with the `for<'id>` bound on
/// [`Region::session`](crate::Region::session), this gives two compile-time
/// guarantees:
///
/// 1. **No escape**: A `Ref<'id, T>` cannot be returned from the session
///    closure because `'id` is universally quantified and cannot unify with
///    any external lifetime.
/// 2. **No cross-session use**: A `Ref` from session A cannot be passed to
///    session B because their `'id` brands are distinct.
///
/// Both properties are verified by compile-fail tests.
pub struct Ref<'id, T: Flat> {
  pos: Pos,
  #[expect(
    dead_code,
    reason = "phantom field — carries invariant 'id brand for compile-time safety"
  )]
  brand: Brand<'id>,
  _type: PhantomData<T>,
}

// Manual Copy/Clone — Ref is always Copy regardless of T.
impl<T: Flat> Copy for Ref<'_, T> {}
impl<T: Flat> Clone for Ref<'_, T> {
  fn clone(&self) -> Self {
    *self
  }
}

impl<T: Flat> PartialEq for Ref<'_, T> {
  fn eq(&self, other: &Self) -> bool {
    self.pos == other.pos
  }
}

impl<T: Flat> Eq for Ref<'_, T> {}

impl<T: Flat> Hash for Ref<'_, T> {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    self.pos.0.hash(state);
  }
}

impl<'id, T: Flat> Ref<'id, T> {
  const fn new(pos: Pos, brand: Brand<'id>) -> Self {
    Self { pos, brand, _type: PhantomData }
  }
}

// SAFETY:
//   `emit` returns the existing position (for splice/redirect use).
//   `write_at` deep-copies the value:
//     1. Pre-reserves byte_len() to prevent buffer reallocation.
//     2. Reads from [pos, pos+size_of::<T>()) via `with_exposed_provenance`
//        so the `&T` is NOT derived from the `&mut Patch`.
//     3. Writes at [at, at+size_of::<T>()) — non-overlapping because `at` is
//        a freshly allocated position beyond the pre-existing byte_len.
unsafe impl<T: Flat> Emit<T> for Ref<'_, T> {
  fn emit(self, _p: &mut impl Patch) -> Pos {
    self.pos
  }

  unsafe fn write_at(self, p: &mut impl Patch, at: Pos) {
    // Pre-reserve to prevent buffer reallocation during deep-copy.
    // Deep-copy re-emits a subset of existing data, so byte_len()
    // additional bytes is a safe upper bound.
    p.reserve(p.byte_len() as u32);
    // SAFETY: After reservation, buffer won't reallocate. We recover the
    // allocation's provenance via `with_exposed_provenance` so the `&T` is
    // not derived from `p` — avoiding an aliased `&T` / `&mut Patch` pair
    // that Stacked Borrows would reject.
    unsafe {
      let addr = p.raw_ptr().add(self.pos.0 as usize).addr();
      let val = &*core::ptr::with_exposed_provenance::<T>(addr);
      Emit::<T>::write_at(val, p, at);
    }
  }
}

/// Tail cursor for O(1) repeated appends to a [`NearList`] via
/// [`Session::push_back`].
///
/// Stores the position of the last segment appended — enabling the next
/// `push_back` to link directly without walking the segment chain.
///
/// Like [`Ref`], `ListTail` is branded with `'id` and cannot escape the
/// session closure.
pub struct ListTail<'id, U: Flat> {
  seg_pos: Pos,
  len: u32,
  head_abs: Pos,
  #[expect(
    dead_code,
    reason = "phantom field — carries invariant 'id brand for compile-time safety"
  )]
  brand: Brand<'id>,
  _type: PhantomData<U>,
}

// Manual Copy/Clone — ListTail is always Copy regardless of U.
impl<U: Flat> Copy for ListTail<'_, U> {}
impl<U: Flat> Clone for ListTail<'_, U> {
  fn clone(&self) -> Self {
    *self
  }
}

/// Branded mutable session over a [`Region`].
///
/// Opened via [`Region::session`]. The invariant `'id` lifetime ensures that
/// [`Ref`]s created within one session cannot be used with another session
/// or escape the closure — **compile-time safety, zero runtime cost**.
///
/// # Mutation model
///
/// All mutations are **append-only**: new data is written at the end of the
/// buffer, and old list headers / near offsets are patched to point to the new
/// data. The old targets become dead bytes. Call [`Region::trim`] after a batch
/// of mutations to compact the buffer.
///
/// # Aliasing strategy
///
/// Methods that both read and write the buffer (e.g. [`re_splice_list`],
/// [`map_list`], [`filter_list`]) use a two-phase approach:
///
/// 1. **Pre-reserve** capacity so the buffer will not reallocate during writes.
/// 2. **Recover provenance** via [`with_exposed_provenance`] to create a `&T`
///    that is *not* derived from the `&mut Region`, avoiding Stacked Borrows
///    violations.
///
/// This pattern is validated by Miri with no UB detected.
///
/// [`re_splice_list`]: Self::re_splice_list
/// [`map_list`]: Self::map_list
/// [`filter_list`]: Self::filter_list
/// [`with_exposed_provenance`]: core::ptr::with_exposed_provenance
#[cfg(feature = "alloc")]
pub struct Session<'id, 'a, Root: Flat, B: Buf = crate::buf::AlignedBuf<Root>> {
  region: &'a mut Region<Root, B>,
  brand: Brand<'id>,
}

/// See the `#[cfg(feature = "alloc")]` variant above for full documentation.
#[cfg(not(feature = "alloc"))]
pub struct Session<'id, 'a, Root: Flat, B: Buf> {
  region: &'a mut Region<Root, B>,
  brand: Brand<'id>,
}

impl<'id, 'a, Root: Flat, B: Buf> Session<'id, 'a, Root, B> {
  /// Create a session (called by `Region::session`).
  pub(crate) const fn new(region: &'a mut Region<Root, B>, brand: Brand<'id>) -> Self {
    Self { region, brand }
  }

  /// Get a [`Ref`] to the root value.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Node::make(1, empty()));
  /// region.session(|s| {
  ///   let root = s.root();
  ///   assert_eq!(s.at(root).id, 1);
  /// });
  /// ```
  #[must_use]
  pub const fn root(&self) -> Ref<'id, Root> {
    Ref::new(Pos::ZERO, self.brand)
  }

  /// Read the value at a [`Ref`]'s position.
  ///
  /// # Panics
  ///
  /// Panics if the position plus `size_of::<T>()` exceeds the region length.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty, list};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, insts: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(10, list([1u32, 2])));
  /// region.session(|s| {
  ///   let root = s.root();
  ///   let block = s.at(root);
  ///   assert_eq!(block.id, 10);
  ///   assert_eq!(block.insts.len(), 2);
  /// });
  /// ```
  #[must_use]
  pub fn at<T: Flat>(&self, r: Ref<'id, T>) -> &T {
    let base = self.region.deref_raw();
    let start = r.pos.0 as usize;
    assert!(start + size_of::<T>() <= self.region.byte_len(), "session at out of bounds");
    // SAFETY: Bounds checked above. `base` is aligned to `BUF_ALIGN >= align_of::<T>()`,
    // and `start` was produced by `alloc::<T>()` which ensures correct alignment.
    unsafe { &*base.add(start).cast::<T>() }
  }

  /// Navigate from a [`Ref`] to a sub-field, returning a new [`Ref`].
  ///
  /// The closure receives `&T` and returns `&U`. The byte offset of `U` within
  /// the region is computed from pointer arithmetic. Navigation through
  /// [`Near`] pointers is transparent thanks to `Deref`.
  ///
  /// # Panics
  ///
  /// Panics if the navigated field is not within this region's buffer.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, insts: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(1, list([10u32, 20])));
  /// region.session(|s| {
  ///   let root = s.root();
  ///
  ///   // Navigate to a scalar field.
  ///   let id_ref = s.nav(root, |b| &b.id);
  ///   assert_eq!(*s.at(id_ref), 1);
  ///
  ///   // Navigate to a list field.
  ///   let insts_ref = s.nav(root, |b| &b.insts);
  ///   assert_eq!(s.at(insts_ref).len(), 2);
  /// });
  /// ```
  #[must_use]
  pub fn nav<T: Flat, U: Flat>(&self, r: Ref<'id, T>, f: impl FnOnce(&T) -> &U) -> Ref<'id, U> {
    let base = self.region.deref_raw() as usize;
    let val = self.at(r);
    let field_ptr = core::ptr::from_ref::<U>(f(val)) as usize;
    let offset =
      field_ptr.checked_sub(base).expect("navigated field is not within this region's buffer");
    assert!(offset + size_of::<U>() <= self.region.byte_len(), "navigated field out of bounds");
    Ref::new(Pos(offset as u32), self.brand)
  }

  /// Follow a [`Near<U>`] pointer, returning a [`Ref`] to its target.
  ///
  /// # Panics
  ///
  /// Panics if the [`Near`] target lies outside the region.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, Near, Region, near};
  ///
  /// #[derive(Flat)]
  /// struct Wrapper { inner: Near<u32> }
  ///
  /// let mut region = Region::new(Wrapper::make(near(42u32)));
  /// region.session(|s| {
  ///   let inner_near = s.nav(s.root(), |w| &w.inner);
  ///   let inner_ref = s.follow(inner_near);
  ///   assert_eq!(*s.at(inner_ref), 42);
  /// });
  /// ```
  #[must_use]
  pub fn follow<U: Flat>(&self, r: Ref<'id, Near<U>>) -> Ref<'id, U> {
    let base = self.region.deref_raw() as usize;
    let near = self.at(r);
    let target_ptr = core::ptr::from_ref::<U>(near.get()) as usize;
    let offset = target_ptr.checked_sub(base).expect("Near target outside region");
    Ref::new(Pos(offset as u32), self.brand)
  }

  /// Convert a reference obtained from [`at`](Self::at) back into a [`Ref`].
  ///
  /// This lets you destructure a value from `s.at(r)` (e.g. match an enum)
  /// and turn inner field references into [`Ref`]s — without a `nav` closure
  /// that would need to re-destructure.
  ///
  /// # Panics
  ///
  /// Panics if `val` does not point within this session's region.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty, list};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(10, list([1u32, 2])));
  /// region.session(|s| {
  ///   let root = s.at(s.root());
  ///   let id_ref = s.ref_of(&root.id);
  ///   assert_eq!(*s.at(id_ref), 10);
  /// });
  /// ```
  #[must_use]
  pub fn ref_of<U: Flat>(&self, val: &U) -> Ref<'id, U> {
    let base = self.region.deref_raw() as usize;
    let ptr = core::ptr::from_ref(val) as usize;
    let offset = ptr.checked_sub(base).expect("ref_of: value not within region");
    assert!(offset + size_of::<U>() <= self.region.byte_len(), "ref_of: value out of bounds");
    Ref::new(Pos(offset as u32), self.brand)
  }

  /// Overwrite a `Copy` value at a [`Ref`]'s position.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(1, empty()));
  /// region.session(|s| {
  ///   let id = s.nav(s.root(), |b| &b.id);
  ///   s.set(id, 99);
  /// });
  /// assert_eq!(region.id, 99);
  /// ```
  pub fn set<T: Flat + Copy>(&mut self, r: Ref<'id, T>, val: T) {
    // SAFETY: `r.pos` was obtained from a branded `Ref` which guarantees it
    // is a valid position for `T` within this region.
    unsafe { self.region.write_flat_internal(r.pos, val) };
  }

  /// Overwrite the value at a [`Ref`]'s position using a builder.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty, list};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(1, empty()));
  /// region.session(|s| {
  ///   // Overwrite the entire root with a new builder.
  ///   s.write(s.root(), Block::make(2, list([10u32, 20])));
  /// });
  /// assert_eq!(region.id, 2);
  /// assert_eq!(region.items.len(), 2);
  /// ```
  pub fn write<T: Flat>(&mut self, r: Ref<'id, T>, builder: impl Emit<T>) {
    // SAFETY: `r.pos` is a valid position for `T` within the region.
    unsafe { builder.write_at(self.region, r.pos) };
  }

  /// Replace the target of a [`Near<U>`] pointer with freshly emitted data.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, Near, Region, near};
  ///
  /// #[derive(Flat)]
  /// struct Wrapper { inner: Near<u32> }
  ///
  /// let mut region = Region::new(Wrapper::make(near(10u32)));
  /// assert_eq!(*region.inner, 10);
  ///
  /// region.session(|s| {
  ///   let inner = s.nav(s.root(), |w| &w.inner);
  ///   s.splice(inner, 42u32);
  /// });
  /// assert_eq!(*region.inner, 42);
  /// ```
  pub fn splice<U: Flat>(&mut self, r: Ref<'id, Near<U>>, builder: impl Emit<U>) {
    let target = builder.emit(self.region);
    // SAFETY: `r.pos` points to a `Near<U>` field (guaranteed by the branded
    // `Ref<Near<U>>`), and `target` was just allocated by `emit`.
    unsafe { self.region.patch_near_internal(r.pos, target) };
  }

  /// Replace the contents of a [`NearList<U>`] with freshly emitted elements.
  ///
  /// Allocates a single contiguous segment for all elements.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2, 3])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.splice_list(items, [10u32, 20]);
  /// });
  ///
  /// assert_eq!(region.items.len(), 2);
  /// assert_eq!(region.items[0], 10);
  /// assert_eq!(region.items[1], 20);
  ///
  /// // Splice to empty.
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.splice_list(items, core::iter::empty::<u32>());
  /// });
  /// assert_eq!(region.items.len(), 0);
  /// ```
  pub fn splice_list<U: Flat, E: Emit<U>, I>(&mut self, r: Ref<'id, NearList<U>>, items: I)
  where
    I: IntoIterator<Item = E>,
    I::IntoIter: ExactSizeIterator,
  {
    let list_pos = r.pos;

    let iter = items.into_iter();
    let count = iter.len();
    let len = count as u32;

    if len == 0 {
      // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`).
      unsafe { self.region.patch_list_header_internal(list_pos, Pos::ZERO, 0) };
      return;
    }

    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();
    for (i, item) in iter.enumerate() {
      let val_pos = seg_pos.offset(values_offset + i * size_of::<U>());
      // SAFETY: `val_pos` was allocated for `U` by `alloc_segment_internal`.
      unsafe { item.write_at(self.region, val_pos) };
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Replace the contents of a [`NearList<U>`] with deep-copied values referenced
  /// by [`Ref`]s.
  ///
  /// This is the deep-copy variant of `splice_list`. It deep-copies
  /// existing values (including all transitively reachable [`Near`]/[`NearList`]
  /// data) into a single contiguous segment. Zero heap allocation beyond the
  /// region's own growth.
  ///
  /// # Aliasing safety
  ///
  /// Capacity is pre-reserved so the buffer does not reallocate during the
  /// deep-copy writes. Reads use [`with_exposed_provenance`] to recover a `&U`
  /// that is *not* derived from `&mut Region`, avoiding a Stacked Borrows
  /// violation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2, 3])));
  ///
  /// // Deep-copy a subset of list elements (reordered).
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   let refs = s.list_refs(items);
  ///   // Reverse: [3, 2, 1]
  ///   let reversed: Vec<_> = refs.iter().rev().copied().collect();
  ///   s.re_splice_list(items, &reversed);
  /// });
  ///
  /// assert_eq!(region.items[0], 3);
  /// assert_eq!(region.items[1], 2);
  /// assert_eq!(region.items[2], 1);
  /// ```
  ///
  /// [`with_exposed_provenance`]: core::ptr::with_exposed_provenance
  pub fn re_splice_list<U: Flat>(&mut self, r: Ref<'id, NearList<U>>, refs: &[Ref<'id, U>]) {
    let list_pos = r.pos;
    let count = refs.len();
    let len = count as u32;

    if len == 0 {
      // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`).
      unsafe { self.region.patch_list_header_internal(list_pos, Pos::ZERO, 0) };
      return;
    }

    // Pre-reserve capacity so the buffer does not reallocate during the loop.
    // Each deep-copy re-emits a subset of the existing region data, so
    // byte_len() covers all transitive targets. The segment is extra overhead.
    let seg_overhead = size_of::<Segment<U>>() + count * size_of::<U>() + align_of::<Segment<U>>();
    self.region.reserve_internal((self.region.byte_len() + seg_overhead) as u32);

    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();
    let base_addr = self.region.deref_raw().addr();
    for (i, &item_ref) in refs.iter().enumerate() {
      let val_pos = seg_pos.offset(values_offset + i * size_of::<U>());
      // SAFETY: The pre-reserve above guarantees no reallocation occurs.
      // We recover the allocation's provenance via `with_exposed_provenance`
      // so the `&U` is not derived from `self.region` — avoiding an aliased
      // `&U` / `&mut Region` pair that Stacked Borrows would reject.
      unsafe {
        let addr = base_addr.wrapping_add(item_ref.pos.0 as usize);
        let val = &*core::ptr::with_exposed_provenance::<U>(addr);
        Emit::<U>::write_at(val, self.region, val_pos);
      }
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Replace the contents of a [`NearList<U>`] by mapping each element through
  /// a function.
  ///
  /// Walks the old segment chain via raw pointer arithmetic, reading each
  /// element through exposed provenance, applying `f`, and writing the result
  /// into a fresh contiguous segment. No heap allocation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2, 3])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.map_list(items, |x| x * 10);
  /// });
  ///
  /// assert_eq!(region.items[0], 10);
  /// assert_eq!(region.items[1], 20);
  /// assert_eq!(region.items[2], 30);
  /// ```
  #[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "segment offset arithmetic: buffer positions are always non-negative and fit in i64"
  )]
  pub fn map_list<U: Flat + Copy>(&mut self, r: Ref<'id, NearList<U>>, mut f: impl FnMut(U) -> U) {
    let (count, first_seg_off) = self.list_meta(r);
    if count == 0 {
      return;
    }

    // Walk segments and overwrite each element in place. U: Copy guarantees
    // same size, so no new segment or header patch is needed.
    let base = self.region.deref_raw();
    let base_addr = base.addr();
    let mut seg_off = first_seg_off;
    let mut remaining = count;
    while remaining > 0 {
      // SAFETY: We have `&mut self` (exclusive access). Reads use
      // `with_exposed_provenance`; writes go through `write_flat_internal`
      // at the same position — no aliasing conflict. Buffer does not
      // reallocate because `write_flat_internal` writes at existing offsets.
      unsafe {
        let seg_addr = base_addr.wrapping_add(seg_off);
        let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
        let seg_len = seg.len as usize;
        let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
        for j in 0..seg_len {
          let val_addr = vals_base.wrapping_add(j * size_of::<U>());
          let val = core::ptr::with_exposed_provenance::<U>(val_addr).read();
          let mapped = f(val);
          self.region.write_flat_internal(Pos((val_addr - base_addr) as u32), mapped);
        }
        remaining -= seg_len;
        if remaining > 0 {
          seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
        }
      }
    }
  }

  /// Prepend an element to a [`NearList<U>`].
  ///
  /// **O(1)**: allocates one 1-element segment and patches the list header.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([2u32, 3])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.push_front(items, 1u32);
  /// });
  ///
  /// assert_eq!(region.items.len(), 3);
  /// assert_eq!(region.items[0], 1);
  /// assert_eq!(region.items[1], 2);
  /// ```
  pub fn push_front<U: Flat>(&mut self, r: Ref<'id, NearList<U>>, item: impl Emit<U>) {
    let list: &NearList<U> = self.at(r);
    let old_len = list.len() as u32;
    let old_head_offset = list.head_offset();

    let seg_pos = self.region.alloc_segment_internal::<U>(1);
    let val_pos = seg_pos.offset(size_of::<Segment<U>>());
    // SAFETY: `val_pos` was allocated for `U` by `alloc_segment_internal`.
    unsafe { item.write_at(self.region, val_pos) };

    // Link new segment's next to the old first segment (if any).
    if old_len > 0 {
      let head_field_abs = i64::from(r.pos.0);
      let old_first_abs = head_field_abs + i64::from(old_head_offset);
      #[expect(clippy::cast_sign_loss, reason = "absolute position is always non-negative")]
      let old_first_pos = Pos(old_first_abs as u32);
      // SAFETY: `seg_pos` was just allocated by `alloc_segment_internal`,
      // and `old_first_pos` points to the existing first segment.
      unsafe { self.region.patch_segment_next_internal(seg_pos, old_first_pos) };
    }

    // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(r.pos, seg_pos, old_len + 1) };
  }

  /// Append one element to the end of a [`NearList<U>`].
  ///
  /// **O(1)** when a `tail` from a previous `push_back` is provided.
  /// **O(s)** on the first call to a non-empty list without a cached tail
  /// (walks s segments to find the tail).
  ///
  /// Returns a [`ListTail`] cursor for O(1) chaining of subsequent appends.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(empty()));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   let tail = s.push_back(items, None, 1u32);
  ///   let tail = s.push_back(items, Some(tail), 2u32);
  ///   let _tail = s.push_back(items, Some(tail), 3u32);
  /// });
  ///
  /// assert_eq!(region.items.len(), 3);
  /// assert_eq!(region.items[0], 1);
  /// assert_eq!(region.items[1], 2);
  /// assert_eq!(region.items[2], 3);
  /// ```
  pub fn push_back<U: Flat>(
    &mut self,
    r: Ref<'id, NearList<U>>,
    tail: Option<ListTail<'id, U>>,
    item: impl Emit<U>,
  ) -> ListTail<'id, U> {
    // Allocate a 1-element segment, write item.
    let seg_pos = self.region.alloc_segment_internal::<U>(1);
    let val_pos = seg_pos.offset(size_of::<Segment<U>>());
    // SAFETY: `val_pos` was allocated for `U` by `alloc_segment_internal`.
    unsafe { item.write_at(self.region, val_pos) };

    if let Some(t) = tail {
      // Hot path — O(1): link cached tail, patch header with cached len/head.
      // SAFETY: `t.seg_pos` is a previously allocated segment, `seg_pos` was
      // just allocated.
      unsafe { self.region.patch_segment_next_internal(t.seg_pos, seg_pos) };
      let new_len = t.len + 1;
      // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`).
      unsafe { self.region.patch_list_header_internal(r.pos, t.head_abs, new_len) };
      ListTail {
        seg_pos,
        len: new_len,
        head_abs: t.head_abs,
        brand: self.brand,
        _type: PhantomData,
      }
    } else {
      // Cold path: read header to determine empty vs non-empty.
      let list: &NearList<U> = self.at(r);
      let old_len = list.len() as u32;

      if old_len == 0 {
        // First element: point header at new segment.
        // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`), and
        // `seg_pos` was just allocated.
        unsafe { self.region.patch_list_header_internal(r.pos, seg_pos, 1) };
        ListTail { seg_pos, len: 1, head_abs: seg_pos, brand: self.brand, _type: PhantomData }
      } else {
        // O(s): walk segments to find tail, link it.
        let head_off = list.head_offset();
        #[expect(clippy::cast_sign_loss, reason = "absolute positions are always non-negative")]
        let (last_seg_pos, head_abs) = {
          let base = self.region.deref_raw();
          let head_abs = (i64::from(r.pos.0) + i64::from(head_off)) as usize;
          let mut seg_abs = head_abs;
          loop {
            // SAFETY: `seg_abs` points to a valid `Segment<U>` in the region
            // buffer. The `next` field at offset 0 is an `i32`.
            let next_rel = unsafe { core::ptr::read_unaligned(base.add(seg_abs).cast::<i32>()) };
            if next_rel == 0 {
              break;
            }
            #[expect(clippy::cast_possible_wrap, reason = "buffer offsets fit in i64")]
            {
              seg_abs = (seg_abs as i64 + i64::from(next_rel)) as usize;
            }
          }
          (Pos(seg_abs as u32), Pos(head_abs as u32))
        };
        // SAFETY: `last_seg_pos` points to the tail segment (found by walk),
        // `seg_pos` was just allocated.
        unsafe { self.region.patch_segment_next_internal(last_seg_pos, seg_pos) };
        let new_len = old_len + 1;
        // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`).
        unsafe { self.region.patch_list_header_internal(r.pos, head_abs, new_len) };
        ListTail { seg_pos, len: new_len, head_abs, brand: self.brand, _type: PhantomData }
      }
    }
  }

  /// Append elements to a [`NearList<U>`] without copying existing segments.
  ///
  /// **O(s + k)**: walks the existing segment chain (s segments) to find the
  /// tail, then emits k new elements into a single contiguous segment and
  /// links the tail to it.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.extend_list(items, [3u32, 4]);
  /// });
  ///
  /// assert_eq!(region.items.len(), 4);
  /// assert_eq!(region.items[2], 3);
  /// assert_eq!(region.items[3], 4);
  /// ```
  pub fn extend_list<U: Flat, E: Emit<U>, I>(&mut self, r: Ref<'id, NearList<U>>, extra: I)
  where
    I: IntoIterator<Item = E>,
    I::IntoIter: ExactSizeIterator,
  {
    // Read header before any mutation.
    let list = self.at(r);
    let old_len = list.len() as u32;
    let head_off = list.head_offset();

    // Walk segments to find the last one (raw pointer arithmetic).
    #[expect(clippy::cast_sign_loss, reason = "absolute positions are always non-negative")]
    let last_seg_pos = if old_len == 0 {
      None
    } else {
      let base = self.region.deref_raw();
      let mut seg_abs = (i64::from(r.pos.0) + i64::from(head_off)) as usize;
      loop {
        // SAFETY: `seg_abs` points to a valid `Segment<U>` in the region
        // buffer. The `next` field at offset 0 is an `i32`.
        let next_rel = unsafe { core::ptr::read_unaligned(base.add(seg_abs).cast::<i32>()) };
        if next_rel == 0 {
          break;
        }
        #[expect(
          clippy::cast_possible_wrap,
          clippy::cast_sign_loss,
          reason = "buffer offsets fit in i64"
        )]
        {
          seg_abs = (seg_abs as i64 + i64::from(next_rel)) as usize;
        }
      }
      Some(Pos(seg_abs as u32))
    };

    let iter = extra.into_iter();
    let count = iter.len() as u32;
    if count == 0 {
      return;
    }

    let seg_pos = self.region.alloc_segment_internal::<U>(count);
    let values_offset = size_of::<Segment<U>>();
    for (i, item) in iter.enumerate() {
      let val_pos = seg_pos.offset(values_offset + i * size_of::<U>());
      // SAFETY: `val_pos` was allocated for `U` by `alloc_segment_internal`.
      unsafe { item.write_at(self.region, val_pos) };
    }

    // Link last existing segment → new segment, update header.
    if let Some(last) = last_seg_pos {
      // SAFETY: `last` points to the tail segment (found by walk), `seg_pos`
      // was just allocated.
      unsafe { self.region.patch_segment_next_internal(last, seg_pos) };
      #[expect(clippy::cast_sign_loss, reason = "absolute position is always non-negative")]
      let first_abs = Pos((i64::from(r.pos.0) + i64::from(head_off)) as u32);
      // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`).
      unsafe { self.region.patch_list_header_internal(r.pos, first_abs, old_len + count) };
    } else {
      // SAFETY: `r.pos` points to a `NearList<U>` (branded `Ref`), and
      // `seg_pos` was just allocated.
      unsafe { self.region.patch_list_header_internal(r.pos, seg_pos, count) };
    }
  }

  /// Bulk-copy another region's bytes into this session's region, returning a
  /// [`Ref`] to the grafted root.
  ///
  /// All self-relative pointers within the grafted data remain valid because
  /// the entire source is copied as a contiguous block.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty, list};
  ///
  /// #[derive(Flat)]
  /// struct Node { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Node::make(1, empty()));
  /// let other = Region::new(Node::make(2, list([10u32, 20])));
  ///
  /// region.session(|s| {
  ///   let grafted = s.graft(&other);
  ///   let grafted_node = s.at(grafted);
  ///   assert_eq!(grafted_node.id, 2);
  ///   assert_eq!(grafted_node.items.len(), 2);
  /// });
  /// ```
  #[must_use]
  pub fn graft<U: Flat, B2: Buf>(&mut self, src: &Region<U, B2>) -> Ref<'id, U> {
    let pos = self.region.graft_internal(src);
    Ref::new(pos, self.brand)
  }

  /// Collect [`Ref`]s to every element of a [`NearList`].
  ///
  /// **O(n)**: walks through segments. Returns a `Vec<Ref>` with no borrow on
  /// the session, enabling subsequent mutation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([10u32, 20, 30])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   let refs = s.list_refs(items);
  ///   assert_eq!(refs.len(), 3);
  ///   assert_eq!(*s.at(refs[0]), 10);
  ///   assert_eq!(*s.at(refs[2]), 30);
  /// });
  /// ```
  #[cfg(feature = "alloc")]
  #[must_use]
  pub fn list_refs<T: Flat>(&self, list: Ref<'id, NearList<T>>) -> alloc::vec::Vec<Ref<'id, T>> {
    let nl = self.at(list);
    let len = nl.len();
    if len == 0 {
      return alloc::vec::Vec::new();
    }
    let base = self.region.deref_raw() as usize;
    let mut refs = alloc::vec::Vec::with_capacity(len);
    for elem in nl {
      let offset = (core::ptr::from_ref(elem) as usize) - base;
      refs.push(Ref::new(Pos(offset as u32), self.brand));
    }
    refs
  }

  /// Read list metadata (count, head offset) and drop the borrow.
  #[expect(clippy::cast_sign_loss, reason = "absolute positions are always non-negative")]
  fn list_meta<U: Flat>(&self, r: Ref<'id, NearList<U>>) -> (usize, usize) {
    let list: &NearList<U> = self.at(r);
    let count = list.len();
    if count == 0 {
      return (0, 0);
    }
    let first_seg_off = (i64::from(r.pos.0) + i64::from(list.head_offset())) as usize;
    (count, first_seg_off)
  }

  /// Collect element buffer offsets from a list into a `Vec`.
  #[cfg(feature = "alloc")]
  #[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "segment offset arithmetic: buffer positions are always non-negative and fit in i64"
  )]
  fn collect_list_positions<U: Flat>(&self, r: Ref<'id, NearList<U>>) -> alloc::vec::Vec<u32> {
    let (count, first_seg_off) = self.list_meta(r);
    if count == 0 {
      return alloc::vec::Vec::new();
    }

    let mut positions = alloc::vec::Vec::with_capacity(count);
    let base_addr = self.region.deref_raw().addr();
    let mut seg_off = first_seg_off;
    let mut remaining = count;
    while remaining > 0 {
      // SAFETY: Segment reads use `with_exposed_provenance`. Buffer is not
      // mutated, so the cached base address remains valid.
      unsafe {
        let seg_addr = base_addr.wrapping_add(seg_off);
        let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
        let seg_len = seg.len as usize;
        let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
        for j in 0..seg_len {
          let val_off = vals_base.wrapping_add(j * size_of::<U>()) - base_addr;
          positions.push(val_off as u32);
        }
        remaining -= seg_len;
        if remaining > 0 {
          seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
        }
      }
    }
    positions
  }

  /// Remove elements from a [`NearList<U>`] that do not satisfy a predicate.
  ///
  /// Two-pass: first counts matching elements, then walks again to deep-copy
  /// them into a single contiguous segment. No-op when all elements match.
  /// No heap allocation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2, 3, 4, 5])));
  ///
  /// // Keep only even numbers.
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.filter_list(items, |x| *x % 2 == 0);
  /// });
  ///
  /// assert_eq!(region.items.len(), 2);
  /// assert_eq!(region.items[0], 2);
  /// assert_eq!(region.items[1], 4);
  /// ```
  #[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "segment offset arithmetic: buffer positions are always non-negative and fit in i64"
  )]
  pub fn filter_list<U: Flat>(
    &mut self,
    r: Ref<'id, NearList<U>>,
    mut pred: impl FnMut(&U) -> bool,
  ) {
    let (count, first_seg_off) = self.list_meta(r);
    if count == 0 {
      return;
    }

    // First pass: count how many elements match the predicate.
    // If all match, return early with no allocation.
    let kept_count = {
      let base_addr = self.region.deref_raw().addr();
      let mut seg_off = first_seg_off;
      let mut remaining = count;
      let mut kept = 0usize;
      while remaining > 0 {
        // SAFETY: Read-only pass — buffer is not mutated, so cached base
        // address remains valid throughout.
        unsafe {
          let seg_addr = base_addr.wrapping_add(seg_off);
          let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
          let seg_len = seg.len as usize;
          let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
          for j in 0..seg_len {
            let val_addr = vals_base.wrapping_add(j * size_of::<U>());
            let val = &*core::ptr::with_exposed_provenance::<U>(val_addr);
            if pred(val) {
              kept += 1;
            }
          }
          remaining -= seg_len;
          if remaining > 0 {
            seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
          }
        }
      }
      kept
    };

    // If nothing was filtered, no work needed.
    if kept_count == count {
      return;
    }

    let list_pos = r.pos;
    let len = kept_count as u32;

    if kept_count == 0 {
      // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`).
      unsafe { self.region.patch_list_header_internal(list_pos, Pos::ZERO, 0) };
      return;
    }

    // Pre-reserve so the buffer does not reallocate during deep-copy writes.
    let seg_overhead =
      size_of::<Segment<U>>() + kept_count * size_of::<U>() + align_of::<Segment<U>>();
    self.region.reserve_internal((self.region.byte_len() + seg_overhead) as u32);

    // Second pass: walk segments again, deep-copy matching elements directly
    // into a fresh contiguous segment. No scratch needed.
    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();
    let base_addr = self.region.deref_raw().addr();
    let mut seg_off = first_seg_off;
    let mut remaining = count;
    let mut dest_i = 0;
    while remaining > 0 {
      // SAFETY: Pre-reserve guarantees no reallocation. We recover provenance
      // via `with_exposed_provenance` so the read is not derived from
      // `self.region` — avoiding Stacked Borrows violation.
      unsafe {
        let seg_addr = base_addr.wrapping_add(seg_off);
        let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
        let seg_len = seg.len as usize;
        let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
        for j in 0..seg_len {
          let val_addr = vals_base.wrapping_add(j * size_of::<U>());
          let val = &*core::ptr::with_exposed_provenance::<U>(val_addr);
          if pred(val) {
            let val_pos = seg_pos.offset(values_offset + dest_i * size_of::<U>());
            Emit::<U>::write_at(val, self.region, val_pos);
            dest_i += 1;
          }
        }
        remaining -= seg_len;
        if remaining > 0 {
          seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
        }
      }
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Replace the contents of a [`NearList<U>`] with its elements in reverse
  /// order.
  ///
  /// Walks forward through segments, writing elements in reverse order into
  /// a new contiguous segment. No-op on empty or single-element lists.
  /// No heap allocation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 2, 3])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.reverse_list(items);
  /// });
  ///
  /// assert_eq!(region.items[0], 3);
  /// assert_eq!(region.items[1], 2);
  /// assert_eq!(region.items[2], 1);
  /// ```
  #[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "segment offset arithmetic: buffer positions are always non-negative and fit in i64"
  )]
  pub fn reverse_list<U: Flat>(&mut self, r: Ref<'id, NearList<U>>) {
    let list_pos = r.pos;
    let list: &NearList<U> = self.at(r);
    let count = list.len();
    if count <= 1 {
      return;
    }
    let head_offset = list.head_offset();
    let len = count as u32;

    let first_seg_off = (i64::from(list_pos.0) + i64::from(head_offset)) as usize;

    // Pre-reserve so the buffer does not reallocate during writes.
    let seg_overhead = size_of::<Segment<U>>() + count * size_of::<U>() + align_of::<Segment<U>>();
    self.region.reserve_internal((self.region.byte_len() + seg_overhead) as u32);

    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();

    // Walk old segments forward, write into new segment in reverse order.
    let base_addr = self.region.deref_raw().addr();
    let mut seg_off = first_seg_off;
    let mut remaining = count;
    let mut dest_i = count; // counts down from count to 0
    while remaining > 0 {
      // SAFETY: Pre-reserve guarantees no reallocation. We recover provenance
      // via `with_exposed_provenance` so the read is not derived from
      // `self.region` — avoiding Stacked Borrows violation.
      unsafe {
        let seg_addr = base_addr.wrapping_add(seg_off);
        let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
        let seg_len = seg.len as usize;
        let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
        for j in 0..seg_len {
          dest_i -= 1;
          let val_addr = vals_base.wrapping_add(j * size_of::<U>());
          let val = &*core::ptr::with_exposed_provenance::<U>(val_addr);
          let val_pos = seg_pos.offset(values_offset + dest_i * size_of::<U>());
          Emit::<U>::write_at(val, self.region, val_pos);
        }
        remaining -= seg_len;
        if remaining > 0 {
          seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
        }
      }
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Replace the contents of a [`NearList<U>`] with its elements sorted by a
  /// comparator.
  ///
  /// Collects element positions into a `Vec`, sorts by comparing elements
  /// through the comparator, then deep-copies into a single contiguous
  /// segment. No-op on empty or single-element lists.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([3u32, 1, 2])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.sort_list(items, |a, b| a.cmp(b));
  /// });
  ///
  /// assert_eq!(region.items[0], 1);
  /// assert_eq!(region.items[1], 2);
  /// assert_eq!(region.items[2], 3);
  /// ```
  #[cfg(feature = "alloc")]
  pub fn sort_list<U: Flat>(
    &mut self,
    r: Ref<'id, NearList<U>>,
    mut cmp: impl FnMut(&U, &U) -> core::cmp::Ordering,
  ) {
    let mut positions = self.collect_list_positions::<U>(r);
    if positions.len() <= 1 {
      return;
    }

    // Expose provenance once and cache the base pointer to avoid O(n log n)
    // `deref_raw` calls inside the comparator.
    let base = self.region.deref_raw();
    positions.sort_by(|&a_pos, &b_pos| {
      // SAFETY: Buffer is not mutated during the sort. `base` was obtained
      // from `deref_raw` which exposes provenance. Positions are valid offsets
      // collected from the segment walk above.
      unsafe {
        let a = &*core::ptr::with_exposed_provenance::<U>(base.add(a_pos as usize).addr());
        let b = &*core::ptr::with_exposed_provenance::<U>(base.add(b_pos as usize).addr());
        cmp(a, b)
      }
    });

    let list_pos = r.pos;
    let count = positions.len();
    let len = count as u32;

    // Pre-reserve so the buffer does not reallocate during deep-copy writes.
    let seg_overhead = size_of::<Segment<U>>() + count * size_of::<U>() + align_of::<Segment<U>>();
    self.region.reserve_internal((self.region.byte_len() + seg_overhead) as u32);

    // Deep-copy elements in sorted order into a fresh contiguous segment.
    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();
    let base_addr = self.region.deref_raw().addr();
    for (i, &elem_off) in positions.iter().enumerate() {
      // SAFETY: Pre-reserve guarantees no reallocation. We recover provenance
      // via `with_exposed_provenance` — no aliasing with `&mut Region`.
      unsafe {
        let val_pos = seg_pos.offset(values_offset + i * size_of::<U>());
        let addr = base_addr.wrapping_add(elem_off as usize);
        let val = &*core::ptr::with_exposed_provenance::<U>(addr);
        Emit::<U>::write_at(val, self.region, val_pos);
      }
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Remove consecutive duplicate elements from a [`NearList<U>`].
  ///
  /// Two elements are considered equal when `eq` returns `true`. Only the
  /// first element of each run of equal elements is kept — identical to
  /// `[T]::dedup_by` semantics. No-op on empty or single-element lists.
  /// No heap allocation.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([1u32, 1, 2, 3, 3, 3, 2])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.dedup_list(items, |a, b| a == b);
  /// });
  ///
  /// assert_eq!(region.items.len(), 4);
  /// assert_eq!(region.items[0], 1);
  /// assert_eq!(region.items[1], 2);
  /// assert_eq!(region.items[2], 3);
  /// assert_eq!(region.items[3], 2);
  /// ```
  #[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "segment offset arithmetic: buffer positions are always non-negative and fit in i64"
  )]
  pub fn dedup_list<U: Flat>(
    &mut self,
    r: Ref<'id, NearList<U>>,
    mut eq: impl FnMut(&U, &U) -> bool,
  ) {
    let (count, first_seg_off) = self.list_meta(r);
    if count <= 1 {
      return;
    }

    // First pass: count how many elements survive dedup.
    // If no duplicates, return early with no allocation.
    let kept_count = {
      let base_addr = self.region.deref_raw().addr();
      let mut seg_off = first_seg_off;
      let mut remaining = count;
      let mut kept = 0usize;
      let mut prev_off: Option<usize> = None;
      while remaining > 0 {
        // SAFETY: Read-only pass — buffer is not mutated, so cached base
        // address remains valid throughout.
        unsafe {
          let seg_addr = base_addr.wrapping_add(seg_off);
          let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
          let seg_len = seg.len as usize;
          let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
          for j in 0..seg_len {
            let val_addr = vals_base.wrapping_add(j * size_of::<U>());
            let val = &*core::ptr::with_exposed_provenance::<U>(val_addr);
            let is_dup = prev_off.is_some_and(|p| {
              let prev = &*core::ptr::with_exposed_provenance::<U>(base_addr.wrapping_add(p));
              eq(prev, val)
            });
            if !is_dup {
              kept += 1;
            }
            prev_off = Some(val_addr - base_addr);
          }
          remaining -= seg_len;
          if remaining > 0 {
            seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
          }
        }
      }
      kept
    };

    // If nothing was deduped, no work needed.
    if kept_count == count {
      return;
    }

    let list_pos = r.pos;
    let len = kept_count as u32;

    // Pre-reserve so the buffer does not reallocate during deep-copy writes.
    let seg_overhead =
      size_of::<Segment<U>>() + kept_count * size_of::<U>() + align_of::<Segment<U>>();
    self.region.reserve_internal((self.region.byte_len() + seg_overhead) as u32);

    // Second pass: walk segments again, deep-copy first element of each dedup
    // run directly into a fresh contiguous segment. No scratch needed.
    let seg_pos = self.region.alloc_segment_internal::<U>(len);
    let values_offset = size_of::<Segment<U>>();
    let base_addr = self.region.deref_raw().addr();
    let mut seg_off = first_seg_off;
    let mut remaining = count;
    let mut dest_i = 0;
    let mut prev_off: Option<usize> = None;
    while remaining > 0 {
      // SAFETY: Pre-reserve guarantees no reallocation. We recover provenance
      // via `with_exposed_provenance` so the read is not derived from
      // `self.region` — avoiding Stacked Borrows violation.
      unsafe {
        let seg_addr = base_addr.wrapping_add(seg_off);
        let seg = &*core::ptr::with_exposed_provenance::<Segment<U>>(seg_addr);
        let seg_len = seg.len as usize;
        let vals_base = seg_addr.wrapping_add(size_of::<Segment<U>>());
        for j in 0..seg_len {
          let val_addr = vals_base.wrapping_add(j * size_of::<U>());
          let val = &*core::ptr::with_exposed_provenance::<U>(val_addr);
          let is_dup = prev_off.is_some_and(|p| {
            let prev = &*core::ptr::with_exposed_provenance::<U>(base_addr.wrapping_add(p));
            eq(prev, val)
          });
          if !is_dup {
            let val_pos = seg_pos.offset(values_offset + dest_i * size_of::<U>());
            Emit::<U>::write_at(val, self.region, val_pos);
            dest_i += 1;
          }
          prev_off = Some(val_addr - base_addr);
        }
        remaining -= seg_len;
        if remaining > 0 {
          seg_off = (seg_off as i64 + i64::from(seg.next)) as usize;
        }
      }
    }
    // SAFETY: `list_pos` points to a `NearList<U>` (branded `Ref`), and
    // `seg_pos` was just allocated by `alloc_segment_internal`.
    unsafe { self.region.patch_list_header_internal(list_pos, seg_pos, len) };
  }

  /// Get a [`Ref`] to a list element by index.
  ///
  /// **O(1)** when all elements are in a single segment (always true after
  /// [`trim`](Region::trim)). Falls back to O(n) segment walk otherwise.
  ///
  /// # Panics
  ///
  /// Panics if `index >= list.len()`.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make(list([10u32, 20, 30])));
  ///
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   let second = s.list_item(items, 1);
  ///   assert_eq!(*s.at(second), 20);
  /// });
  /// ```
  #[must_use]
  pub fn list_item<T: Flat>(&self, list: Ref<'id, NearList<T>>, index: usize) -> Ref<'id, T> {
    let nl = self.at(list);
    let val = &nl[index];
    self.ref_of(val)
  }

  /// Return a chainable [`Cursor`] starting at the root.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(1, empty()));
  /// region.session(|s| {
  ///   s.cursor().at(|b| &b.id).set(99);
  /// });
  /// assert_eq!(region.id, 99);
  /// ```
  #[must_use]
  pub const fn cursor<'s>(&'s mut self) -> Cursor<'id, 's, 'a, Root, Root, B> {
    let r = self.root();
    Cursor { session: self, r }
  }

  /// Return a chainable [`Cursor`] starting at the given [`Ref`].
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, list};
  ///
  /// #[derive(Flat)]
  /// struct Block { id: u32, items: NearList<u32> }
  ///
  /// let mut region = Region::new(Block::make(1, list([10u32])));
  /// region.session(|s| {
  ///   let items_ref = s.nav(s.root(), |b| &b.items);
  ///   let first = s.list_item(items_ref, 0);
  ///   s.cursor_at(first).set(42);
  /// });
  /// assert_eq!(region.items[0], 42);
  /// ```
  #[must_use]
  pub const fn cursor_at<'s, T: Flat>(
    &'s mut self,
    r: Ref<'id, T>,
  ) -> Cursor<'id, 's, 'a, T, Root, B> {
    Cursor { session: self, r }
  }
}

/// Chainable cursor over a [`Session`].
///
/// Provides a fluent API for navigating and mutating a region. Each method
/// consumes `self` (by move), preventing borrow conflicts. For complex
/// navigation patterns or batch operations, use [`Session`] methods directly.
///
/// ```
/// use nearest::{Flat, Near, Region, near};
///
/// #[derive(Flat, Debug)]
/// struct Outer { inner: Near<Inner> }
///
/// #[derive(Flat, Debug, Clone, Copy)]
/// struct Inner { value: u32 }
///
/// let mut region = Region::new(Outer::make(near(Inner { value: 1 })));
/// assert_eq!(region.inner.get().value, 1);
///
/// region.session(|s| {
///   s.cursor()
///     .at(|root| &root.inner)       // navigate to inner Near<Inner>
///     .follow()                      // dereference the Near pointer
///     .set(Inner { value: 42 });     // overwrite
/// });
///
/// assert_eq!(region.inner.get().value, 42);
/// ```
pub struct Cursor<'id, 's, 'a, T: Flat, Root: Flat, B: Buf> {
  session: &'s mut Session<'id, 'a, Root, B>,
  r: Ref<'id, T>,
}

impl<'id, 's, 'a, T: Flat, Root: Flat, B: Buf> Cursor<'id, 's, 'a, T, Root, B> {
  /// Navigate to a sub-field.
  #[must_use]
  pub fn at<U: Flat>(self, f: impl FnOnce(&T) -> &U) -> Cursor<'id, 's, 'a, U, Root, B> {
    let r = self.session.nav(self.r, f);
    Cursor { session: self.session, r }
  }

  /// Read the value at this cursor's position.
  #[must_use]
  pub fn get(&self) -> &T {
    self.session.at(self.r)
  }

  /// Extract the [`Ref`] at this cursor's position.
  #[must_use]
  pub const fn pin(self) -> Ref<'id, T> {
    self.r
  }
}

impl<T: Flat + Copy, Root: Flat, B: Buf> Cursor<'_, '_, '_, T, Root, B> {
  /// Overwrite the value at this cursor's position.
  pub fn set(self, val: T) {
    self.session.set(self.r, val);
  }
}

impl<T: Flat, Root: Flat, B: Buf> Cursor<'_, '_, '_, T, Root, B> {
  /// Overwrite the value at this cursor's position using a builder.
  pub fn write_with(self, builder: impl Emit<T>) {
    self.session.write(self.r, builder);
  }
}

// --- Near<U>: splice, follow ---

impl<'id, 's, 'a, U: Flat, Root: Flat, B: Buf> Cursor<'id, 's, 'a, Near<U>, Root, B> {
  /// Replace the target of this [`Near<U>`] pointer with freshly emitted data.
  pub fn splice(self, builder: impl Emit<U>) {
    self.session.splice(self.r, builder);
  }

  /// Follow the [`Near`] pointer, returning a cursor at the target.
  #[must_use]
  pub fn follow(self) -> Cursor<'id, 's, 'a, U, Root, B> {
    let r = self.session.follow(self.r);
    Cursor { session: self.session, r }
  }
}

// --- NearList<U>: splice_list, push_front ---

impl<U: Flat, Root: Flat, B: Buf> Cursor<'_, '_, '_, NearList<U>, Root, B> {
  /// Replace the elements of this [`NearList<U>`].
  pub fn splice_list<E: Emit<U>, I>(self, items: I)
  where
    I: IntoIterator<Item = E>,
    I::IntoIter: ExactSizeIterator,
  {
    self.session.splice_list(self.r, items);
  }

  /// Remove elements that do not satisfy a predicate.
  pub fn filter_list(self, pred: impl FnMut(&U) -> bool) {
    self.session.filter_list(self.r, pred);
  }
}
