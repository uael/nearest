#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::{mem::MaybeUninit, slice};

/// Where a [`Region`](crate::Region) keeps its bytes: memory of 8-byte words, which the region
/// asks to grow, through [`memory_mut`](Self::memory_mut), as it needs.
///
/// Two are provided: [`Heap`], the default, and [`Inline<N>`], `N` bytes inside the `Region`
/// value. Write your own for other memory, such as an arena or a memory-mapped file: the
/// region needs nothing else, and a storage that does not grow ignores the length it is
/// asked for.
///
/// ```
/// use core::mem::MaybeUninit;
///
/// use nearest::{Near, Region, Storage, flat};
///
/// /// Words borrowed for as long as the region lives.
/// struct Arena<'a>(&'a mut [MaybeUninit<u64>]);
///
/// // SAFETY: the arena borrows the words, so nothing else reaches them.
/// unsafe impl Storage for Arena<'_> {
///   fn memory(&self) -> &[MaybeUninit<u64>] {
///     self.0
///   }
///   fn memory_mut(&mut self, _words: usize) -> &mut [MaybeUninit<u64>] {
///     self.0 // does not grow
///   }
/// }
///
/// #[flat]
/// struct Msg { id: u32, body: Near<str> }
///
/// let mut words = [MaybeUninit::uninit(); 8]; // any content: the region zeroes what it uses
/// let mut r = Region::new_in(Arena(&mut words), Msg::new(1, "hi"));
/// r.edit().body().set("hello");
/// assert_eq!(&*r.body, "hello");
/// ```
///
/// # Safety
///
/// A region reads and writes the memory through raw pointers, and keeps them while it is
/// borrowed; it calls `memory` only through `&self` of the region, `memory_mut` only through
/// `&mut`. The memory may hold anything, initialized or not: the region zeroes what it uses
/// before it uses it, and reads only that. An implementation guarantees:
///
/// - **Its own.** Nothing but the region reads or writes the memory while the storage is in a
///   region, and the memory stays valid as long as the storage lives.
/// - **Stable.** `memory`, and `memory_mut` asked for no more words than the memory has, return
///   the same memory (address and length) and change none of it. That memory changes only
///   through `memory_mut` asked for more, or, if it is inside the storage value, when that
///   value moves; its bytes are those last written.
/// - **Growth.** `memory_mut` asked for more words than the memory has may make it longer: it
///   keeps the bytes of the memory, wherever it puts them, and never makes it shorter; what it
///   adds may hold anything. If it panics, the memory is as before, at the same address.
///
/// The region checks the rest: it asks for at most 2 GiB, uses at most 2 GiB of any memory,
/// and panics if it gets less than it asked for, except in
/// [`from_bytes`](crate::Region::from_bytes), which returns an error.
pub unsafe trait Storage {
  /// The memory, for reading.
  fn memory(&self) -> &[MaybeUninit<u64>];

  /// The memory, for writing, made at least `words` long first if the storage can grow.
  ///
  /// The region asks for no more than the memory has, except when it needs more. A storage
  /// that does not grow returns its memory as it is: a region that gets less than it asked for
  /// panics, as growing past 2 GiB does, except
  /// [`Region::from_bytes`](crate::Region::from_bytes), which returns an error.
  fn memory_mut(&mut self, words: usize) -> &mut [MaybeUninit<u64>];
}

/// The heap, the default storage: memory that grows as the region needs, up to 2 GiB. It is a
/// storage with the `alloc` feature; without it, `Region<T>` only names [`view_with`] and
/// [`view_unchecked`].
///
/// [`view_with`]: crate::Region::view_with
/// [`view_unchecked`]: crate::Region::view_unchecked
#[derive(Default)]
pub struct Heap(#[cfg(feature = "alloc")] Vec<MaybeUninit<u64>>);

// SAFETY: a vector of words, reached only through `&self` and `&mut self`; `resize` keeps the
// words, and panics only before changing anything.
#[cfg(feature = "alloc")]
unsafe impl Storage for Heap {
  fn memory(&self) -> &[MaybeUninit<u64>] {
    &self.0
  }

  fn memory_mut(&mut self, words: usize) -> &mut [MaybeUninit<u64>] {
    if words > self.0.len() {
      // Nothing is written: the region zeroes what it uses.
      self.0.resize(words, MaybeUninit::uninit());
    }
    &mut self.0
  }
}

/// `N` bytes inside the `Region` value, 8-aligned: a region without a heap.
///
/// It does not grow: a region that needs more panics, and `from_bytes` of a larger image returns
/// an error. `N` is rounded down to whole words.
#[repr(C, align(8))]
pub struct Inline<const N: usize>([MaybeUninit<u8>; N]);

impl<const N: usize> Default for Inline<N> {
  fn default() -> Self {
    Self([MaybeUninit::uninit(); N])
  }
}

// SAFETY: the bytes are the value's own; they move with it, and are reached only through
// `&self` and `&mut self`. The words are the first `N / 8 * 8` bytes, 8-aligned.
unsafe impl<const N: usize> Storage for Inline<N> {
  fn memory(&self) -> &[MaybeUninit<u64>] {
    // SAFETY: as above.
    unsafe { slice::from_raw_parts(self.0.as_ptr().cast(), N / 8) }
  }

  fn memory_mut(&mut self, _words: usize) -> &mut [MaybeUninit<u64>] {
    // SAFETY: as above; the memory does not grow.
    unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), N / 8) }
  }
}
