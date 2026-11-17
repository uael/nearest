//! The integration tests again, with every region in a storage written here, and storages of
//! other kinds: what `Storage` asks of an implementation is all a region needs.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

use std::mem::MaybeUninit;

use nearest::Storage;

/// Words on the heap that move to a new allocation every time they grow.
#[derive(Default)]
struct Moving(Vec<MaybeUninit<u64>>);

// SAFETY: growing copies the words to new memory, and the old memory is freed only once nothing
// can panic. Nothing else reaches the words.
unsafe impl Storage for Moving {
  fn memory(&self) -> &[MaybeUninit<u64>] {
    &self.0
  }
  fn memory_mut(&mut self, words: usize) -> &mut [MaybeUninit<u64>] {
    if words > self.0.len() {
      let mut new = vec![MaybeUninit::uninit(); words];
      new[..self.0.len()].copy_from_slice(&self.0);
      self.0 = new;
    }
    &mut self.0
  }
}

/// The storage of every region: `Moving`.
type Store = Moving;

include!("common/integration.rs");

// ---------------------------------------------------------------------------------------------
// Other storages

/// Words borrowed from elsewhere: they stay where they are when the region moves, and do not
/// grow.
struct Arena<'a>(&'a mut [MaybeUninit<u64>]);

// SAFETY: the storage borrows the words, so nothing else reaches them.
unsafe impl Storage for Arena<'_> {
  fn memory(&self) -> &[MaybeUninit<u64>] {
    self.0
  }
  fn memory_mut(&mut self, _words: usize) -> &mut [MaybeUninit<u64>] {
    self.0
  }
}

#[test]
fn a_borrowed_arena() {
  let mut words = vec![MaybeUninit::uninit(); 32];
  let r = nearest::Region::new_in(Arena(&mut words), Block::new(0, [Inst::new(1, [10])], None));
  assert!(r.as_bytes().is_some());
  let mut moved = Box::new(r);
  moved.edit().insts().push(Inst::new(2, [20]));
  assert!(moved.as_bytes().is_none());
  assert_eq!((ops(&moved), args(&moved)), (vec![1, 2], vec![vec![10], vec![20]]));
  // Full: the region is left as it was.
  let len = moved.byte_len();
  fails(|| {
    for i in 0..u32::MAX {
      moved.edit().insts().at(0).args().push(i);
    }
  });
  assert!(moved.byte_len() >= len && moved.byte_len() <= 256);
  assert_eq!(ops(&moved), [1, 2]);
  // Copied to and from other storages; loaded from a view, for a storage without `Default`.
  let copy = Region::new_in(Store::default(), &**moved);
  assert_eq!(*copy, **moved);
  drop(moved);
  let image = copy.as_bytes().unwrap(); // 8-aligned, as the memory of a storage is
  let mut again = vec![MaybeUninit::uninit(); 32];
  let loaded = nearest::Region::new_in(Arena(&mut again), nearest::view::<Block>(image).unwrap());
  assert_eq!(loaded.as_bytes(), Some(image));
}

/// Words used again, after a region edited them and was dropped, and words never written: the
/// region zeroes what it uses, so its image is the same as on the heap, padding included.
#[test]
fn arena_words_of_any_content() {
  let heap = nearest::Region::new(Block::new(1, [Inst::new(2, [3])], None));
  assert!(heap.as_bytes().is_some());
  let mut words = [MaybeUninit::uninit(); 32];
  {
    let r = nearest::Region::new_in(Arena(&mut words), &*heap);
    assert_eq!(r.as_bytes(), heap.as_bytes());
  }
  {
    let mut r = nearest::Region::<[u32; 16], _>::new_in(Arena(&mut words), [0; 16]);
    *r.edit() = [u32::MAX; 16];
  }
  let r = nearest::Region::new_in(Arena(&mut words), &*heap);
  assert_eq!(r.as_bytes(), heap.as_bytes());
  // Loaded, then edited: the bytes after the image are zeroed as they are used.
  let mut r = Region::<Block>::from_bytes(heap.as_bytes().unwrap()).unwrap();
  r.edit().insts().push(Inst::new(4, [5]));
  let compact = Region::new_in(Store::default(), &*r);
  assert_eq!(compact.as_bytes(), nearest::Region::new(&*r).as_bytes());
  assert_eq!(r.clone(), r);
}
