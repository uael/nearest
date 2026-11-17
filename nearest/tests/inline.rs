//! The integration tests again, with every region in inline storage, and what is particular to
//! it.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

/// The storage of every region: bytes inside the `Region` value.
type Store = nearest::Inline<{ 1 << 15 }>;

include!("common/integration.rs");

// ---------------------------------------------------------------------------------------------
// Inline storage moves with its region

/// The bytes move with the region; every edit starts from where the region is now.
#[test]
fn a_region_moves_with_its_storage() {
  fn build() -> Region<Block> {
    Region::new_in(Store::default(), Block::new(0, [Inst::new(1, [10])], None))
  }
  let mut r = build();
  r.edit().insts().push(Inst::new(2, [20]));
  let mut boxed = Box::new(r);
  boxed.edit().insts().at(0).args().push(11);
  let mut v = vec![*boxed];
  v.push(v[0].clone());
  let b = v[0].edit();
  let inst = b.insts().remove(1); // moves inside the region
  b.next().insert(Block::new(1, [inst], None));
  let mut r = std::thread::spawn(move || {
    let mut r = v.swap_remove(0);
    *r.edit().id() = 7;
    r
  })
  .join()
  .unwrap();
  assert_eq!((r.id, ops(&r), args(&r)), (7, vec![1], vec![vec![10, 11]]));
  assert_eq!(r.next.as_ref().map(|n| args(n)), Some(vec![vec![20]]));
  // To the heap and back.
  let heap = nearest::Region::new(&*r);
  assert_eq!(*Region::new_in(Store::default(), &*heap), *r);
  // A value moved out of an inline region is copied into another region.
  let mut other = nearest::Region::new(Lst::<Inst>::new(()));
  let e = r.edit();
  let inst = e.insts().remove(0);
  other.edit().items().push(inst);
  assert_eq!((r.insts.len(), &other.items[0].args[..]), (0, &[10, 11][..]));
  // Bytes: borrowed while compact, read in place, loaded, and moved again.
  let c = Region::new_in(Store::default(), &*r);
  let bytes = c.to_bytes();
  assert!(matches!(bytes, Cow::Borrowed(_)));
  assert_eq!(nearest::Region::<Block>::view(&bytes), Ok(&*r));
  let loaded = Box::new(Region::<Block>::from_bytes(&bytes).unwrap());
  assert_eq!(**loaded, *r);
  let mut edited = *loaded;
  edited.edit().insts().push(Inst::new(3, ()));
  assert!(matches!(edited.to_bytes(), Cow::Owned(_)));
  assert_eq!(Region::<Block>::from_bytes(&edited.to_bytes()).unwrap(), edited);
}

/// Inline storage does not grow: past it, building and editing panic, and a region is left as it
/// was; loading returns an error.
#[test]
fn full_inline_storage() {
  type Small = nearest::Region<Lst<u64>, Store64>;
  type Store64 = nearest::Inline<64>;
  type Big = nearest::Region<Lst<u64>, nearest::Inline<256>>;
  let mut r = Small::new_in(Store64::default(), Lst::new([1u64, 2]));
  fails(|| {
    for i in 3..u64::MAX {
      r.edit().items().push(i);
    }
  });
  assert!(r.byte_len() <= 64);
  assert!(r.items.iter().copied().eq(1..=r.items.len() as u64));
  let full = r.items.len();
  fails(|| r.edit().items().set(Near::from_iter(0..8u64)));
  assert_eq!(r.items.len(), full);
  assert_eq!(*Small::new_in(Store64::default(), &*r), *r);
  // 4 + 4 + 4 + 64 bytes do not fit in 64.
  let big = nearest::Region::new(Lst::new(Near::from_iter(0..16u64)));
  assert!(catch_unwind(|| Small::new_in(Store64::default(), &*big)).is_err());
  // Loading does not panic: the image is valid, and larger than the storage.
  assert_eq!(Small::from_bytes(&big.to_bytes()).err().unwrap().offset(), None);
  // SAFETY: the bytes come from `to_bytes` for the same type.
  assert!(catch_unwind(|| unsafe { Small::from_bytes_unchecked(&big.to_bytes()) }).is_err());
  assert_eq!(*Big::from_bytes(&big.to_bytes()).unwrap(), *big);
}
