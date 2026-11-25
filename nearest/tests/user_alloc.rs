//! The integration tests again, with every region in an allocator written here, and regions in
//! `Fixed`: what `Allocator` asks of an implementation is all a region needs.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

use std::{
  alloc::{AllocError, Allocator, Global, Layout},
  mem::MaybeUninit,
  ptr::NonNull,
};

use nearest::{Error, Fixed};

/// The global allocator, but a block that grows always moves to a new one, and new bytes hold
/// `0xa5`, not zeros: the region zeroes what it uses.
#[derive(Clone, Copy, Default)]
struct Moving;

// SAFETY: every block is a block of `Global`; growing copies the bytes to a new block, and frees
// the old one only once the new one was given.
unsafe impl Allocator for Moving {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let block = Global.allocate(layout)?;
    // SAFETY: the block is the caller's, of `block.len()` bytes.
    unsafe { block.cast::<u8>().write_bytes(0xa5, block.len()) };
    Ok(block)
  }
  unsafe fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
    // SAFETY: a block of `Global`, which fits `layout`.
    unsafe { Global.deallocate(block, layout) }
  }
  unsafe fn grow(
    &self,
    block: NonNull<u8>,
    old: Layout,
    layout: Layout,
  ) -> Result<NonNull<[u8]>, AllocError> {
    let new = self.allocate(layout)?;
    // SAFETY: the old block holds `old.size()` bytes, fewer than the new one, and they are
    // distinct blocks; then the old block is of this allocator and fits `old`.
    unsafe {
      block.copy_to_nonoverlapping(new.cast(), old.size());
      self.deallocate(block, old);
    }
    Ok(new)
  }
}

/// The allocator of every region: `Moving`.
type Store = Moving;

include!("common/integration.rs");

// ---------------------------------------------------------------------------------------------
// `Fixed`

/// A region in a borrowed buffer: it moves without its bytes, which stay in the buffer, and
/// grows to the whole buffer; past it, an edit panics and the region is left as it was.
#[test]
fn a_region_in_a_fixed_buffer() {
  let mut bytes = vec![MaybeUninit::uninit(); 256];
  let r =
    nearest::Region::new_in(Block::new(0, [Inst::new(1, [10])], None), Fixed::new(&mut bytes));
  let mut moved = Box::new(r);
  moved.edit().insts().push(Inst::new(2, [20]));
  let mut moved = std::thread::scope(|s| {
    s.spawn(move || {
      moved.edit().insts().at(0).args().push(11);
      moved
    })
    .join()
    .unwrap()
  });
  assert_eq!((ops(&moved), args(&moved)), (vec![1, 2], vec![vec![10, 11], vec![20]]));
  let len = moved.byte_len();
  fails(|| {
    for i in 0..u32::MAX {
      moved.edit().insts().at(0).args().push(i);
    }
  });
  assert!(moved.byte_len() >= len && moved.byte_len() <= 256);
  assert_eq!(ops(&moved), [1, 2]);
  // Copied to the heap, and back into the buffer once the region left it.
  let copy = nearest::Region::new(&**moved);
  assert_eq!(*copy, **moved);
  drop(moved);
  let image = copy.as_bytes().unwrap();
  let loaded = nearest::Region::<Block, _>::from_bytes_in(image, Fixed::new(&mut bytes)).unwrap();
  assert_eq!(loaded.as_bytes(), Some(image));
}

/// One block at a time: a second region in the same `Fixed` finds no room until the first one is
/// dropped; a buffer too small for a value panics, and for an image is `Error::Alloc`.
#[test]
fn a_fixed_buffer_holds_one_region() {
  let mut bytes = [MaybeUninit::uninit(); 64];
  let fixed = Fixed::new(&mut bytes);
  let first = nearest::Region::new_in(Lst::<u8>::new([1, 2]), &fixed);
  fails(|| drop(nearest::Region::new_in(7u32, &fixed)));
  let image = first.as_bytes().unwrap().to_vec();
  assert_eq!(
    nearest::Region::<Lst<u8>, _>::from_bytes_in(&image, &fixed).err(),
    Some(Error::Alloc)
  );
  drop(first);
  assert_eq!(
    nearest::Region::<Lst<u8>, _>::from_bytes_in(&image, &fixed).unwrap().items[..],
    [1, 2]
  );
  let big = nearest::Region::new(Lst::new(Near::from_iter(0..16u64)));
  let mut small = [MaybeUninit::uninit(); 64];
  fails(|| drop(nearest::Region::new_in(&*big, Fixed::new(&mut small))));
  let bytes = big.as_bytes().unwrap();
  let mut small = [MaybeUninit::uninit(); 64];
  assert_eq!(
    nearest::Region::<Lst<u64>, _>::from_bytes_in(bytes, Fixed::new(&mut small)).err(),
    Some(Error::Alloc)
  );
  let mut small = [MaybeUninit::uninit(); 64];
  fails(|| {
    // SAFETY: the bytes come from `as_bytes` of a region of the same type that was not edited.
    let r = unsafe {
      nearest::Region::<Lst<u64>, _>::from_bytes_unchecked_in(bytes, Fixed::new(&mut small))
    };
    drop(r);
  });
}

/// Bytes used again, after a region edited them and was dropped, and bytes never written: the
/// region zeroes what it uses, so its bytes are the same as on the heap, padding included. A
/// buffer at any address: the block starts at the first address aligned to 8.
#[test]
fn fixed_bytes_of_any_content() {
  let heap = nearest::Region::new(Block::new(1, [Inst::new(2, [3])], None));
  let mut bytes = [MaybeUninit::uninit(); 260];
  {
    let r = nearest::Region::new_in(&*heap, Fixed::new(&mut bytes[1..]));
    assert_eq!(r.as_bytes(), heap.as_bytes());
    assert_eq!(r.as_bytes().unwrap().as_ptr().addr() % 8, 0);
  }
  {
    let mut r = nearest::Region::<[u32; 16], _>::new_in([0; 16], Fixed::new(&mut bytes[1..]));
    *r.edit() = [u32::MAX; 16];
  }
  let r = nearest::Region::new_in(&*heap, Fixed::new(&mut bytes[1..]));
  assert_eq!(r.as_bytes(), heap.as_bytes());
}
