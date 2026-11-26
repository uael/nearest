use core::{
  alloc::{AllocError, Allocator, Layout},
  cell::Cell,
  marker::PhantomData,
  mem::MaybeUninit,
  ptr::NonNull,
};

/// An allocator over a buffer that it borrows, without a heap.
///
/// It gives one block at a time, from the first address of the buffer aligned as asked to its
/// end, so a region in it grows to the whole buffer, and past it panics. It needs no `alloc`.
///
/// ```
/// use core::mem::MaybeUninit;
///
/// use nearest::{Fixed, Near, Region, flat};
///
/// #[flat]
/// struct Msg { id: u32, body: Near<str> }
///
/// let mut bytes = [MaybeUninit::uninit(); 64]; // any content: the region zeroes what it uses
/// let mut r = Region::new_in(Msg::new(1, "hi"), Fixed::new(&mut bytes));
/// r.edit().body().set("hello");
/// assert_eq!(&*r.body, "hello");
/// ```
///
/// Bytes inside the allocator value cannot be an allocator: a block must stay valid when its
/// allocator moves (the `# Safety` of [`Allocator`]), so the buffer is borrowed.
pub struct Fixed<'a> {
  start: NonNull<u8>,
  len: usize,
  /// A block is out.
  given: Cell<bool>,
  _buf: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

impl<'a> Fixed<'a> {
  /// The allocator over `buf`, whose bytes may hold anything.
  pub const fn new(buf: &'a mut [MaybeUninit<u8>]) -> Self {
    let len = buf.len();
    Self { start: NonNull::from_mut(buf).cast(), len, given: Cell::new(false), _buf: PhantomData }
  }

  /// The block from `at` to the end of the buffer, if it holds `layout`.
  ///
  /// # Safety
  /// `at` is in the buffer or at its end.
  unsafe fn rest(&self, at: NonNull<u8>, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    // SAFETY: per the caller, `at` is at or after the start, in the same buffer.
    let room = self.len - unsafe { at.offset_from_unsigned(self.start) };
    if room < layout.size() || !at.addr().get().is_multiple_of(layout.align()) {
      return Err(AllocError);
    }
    Ok(NonNull::slice_from_raw_parts(at, room))
  }
}

// SAFETY: a `Fixed` is the unique borrow of its buffer, and a `Cell`, which may move to another
// thread with it; it is not `Sync`.
unsafe impl Send for Fixed<'_> {}

// SAFETY: a block is in the buffer, which is borrowed for `'a`, so it stays valid while the
// allocator lives, wherever it moves; one block is out at a time, so blocks do not overlap.
// `grow` and `shrink` keep the block where it is, and so keep its bytes.
unsafe impl Allocator for Fixed<'_> {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    if self.given.get() {
      return Err(AllocError);
    }
    let skip = self.start.align_offset(layout.align());
    if skip > self.len {
      return Err(AllocError);
    }
    // SAFETY: `skip <= len`, so the address is in the buffer or at its end.
    let block = unsafe { self.rest(self.start.add(skip), layout)? };
    self.given.set(true);
    Ok(block)
  }

  unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {
    self.given.set(false);
  }

  unsafe fn grow(
    &self,
    block: NonNull<u8>,
    _: Layout,
    layout: Layout,
  ) -> Result<NonNull<[u8]>, AllocError> {
    // SAFETY: the block is the one out, in the buffer.
    unsafe { self.rest(block, layout) }
  }

  unsafe fn grow_zeroed(
    &self,
    block: NonNull<u8>,
    old: Layout,
    layout: Layout,
  ) -> Result<NonNull<[u8]>, AllocError> {
    // SAFETY: as for `grow`.
    let new = unsafe { self.rest(block, layout)? };
    // SAFETY: the block holds `layout.size()` bytes, at least `old.size()`.
    unsafe { block.add(old.size()).write_bytes(0, layout.size() - old.size()) };
    Ok(new)
  }

  unsafe fn shrink(
    &self,
    block: NonNull<u8>,
    _: Layout,
    layout: Layout,
  ) -> Result<NonNull<[u8]>, AllocError> {
    // SAFETY: as for `grow`.
    unsafe { self.rest(block, layout) }
  }
}
