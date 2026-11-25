//! The `serde` feature: a region is its bytes, checked when loaded.
#![cfg(feature = "serde")]

use std::{
  alloc::{AllocError, Allocator, Global, Layout},
  ptr::NonNull,
};

use nearest::{Near, Region, flat};
use serde_test::{Token, assert_de_tokens, assert_de_tokens_error, assert_ser_tokens};

#[flat]
#[derive(Debug, PartialEq)]
struct Tree {
  id: u32,
  name: Near<str>,
  kids: Near<[Tree]>,
}

fn tree() -> Region<Tree> {
  Region::new(Tree::new(
    1,
    "root",
    (Tree::new(2, "a", ()), Tree::new(3, "b", [Tree::new(4, "c", ())])),
  ))
}

/// The bytes of a region, leaked so that tokens can borrow them for `'static`.
fn image(r: &Region<Tree>) -> &'static [u8] {
  Region::<Tree>::new(&**r).as_bytes().unwrap().to_vec().leak()
}

#[test]
fn serializes_as_its_bytes() {
  let r = tree();
  assert_ser_tokens(&r, &[Token::Bytes(image(&r))]);
}

/// The new name is after the kids of the kid that holds it, out of the order of a copy: the
/// bytes of the region do not load back, so it serializes as a copy.
#[test]
fn an_edited_region_serializes_as_its_compact_copy() {
  let mut r = tree();
  r.edit().kids().at(0).name().set("renamed");
  assert!(nearest::view::<Tree>(r.as_bytes().unwrap()).is_err());
  assert_ser_tokens(&r, &[Token::Bytes(image(&r))]);
}

#[test]
fn deserializes_from_every_form_of_bytes() {
  let r = tree();
  let bytes = image(&r);
  assert_de_tokens(&r, &[Token::Bytes(bytes)]);
  assert_de_tokens(&r, &[Token::BorrowedBytes(bytes)]);
  assert_de_tokens(&r, &[Token::ByteBuf(bytes)]);
  let mut seq = vec![Token::Seq { len: Some(bytes.len()) }];
  seq.extend(bytes.iter().map(|&b| Token::U8(b)));
  seq.push(Token::SeqEnd);
  assert_de_tokens(&r, &seq);
}

/// The global allocator, but for blocks of `N` bytes at most: an allocator with a default value
/// other than `Global`, which a region deserializes into.
#[derive(Default)]
struct AtMost<const N: usize>;

// SAFETY: every block is a block of `Global`, given and freed with the layout it was asked with.
unsafe impl<const N: usize> Allocator for AtMost<N> {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    if layout.size() > N {
      return Err(AllocError);
    }
    Global.allocate(layout)
  }

  unsafe fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
    // SAFETY: a block of `Global`, which fits `layout`.
    unsafe { Global.deallocate(block, layout) }
  }
}

#[test]
fn deserializes_into_any_allocator_with_a_default() {
  let r = tree();
  let bytes = image(&r);
  let loaded: Region<Tree, AtMost<256>> =
    serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
  assert_eq!(*loaded, *r);
  assert_eq!(loaded.as_bytes(), Some(bytes));
  assert_de_tokens_error::<Region<Tree, AtMost<16>>>(
    &[Token::Bytes(bytes)],
    "the allocator did not give the block of the region",
  );
}

#[test]
fn invalid_bytes_are_an_error() {
  let bytes = image(&tree());
  // The same error as `from_bytes`, as a custom error of the format.
  let cut = &bytes[..bytes.len() - 1];
  let err = Region::<Tree>::from_bytes(cut).unwrap_err();
  assert!(matches!(err, nearest::Error::Invalid(_)));
  assert_de_tokens_error::<Region<Tree>>(&[Token::Bytes(cut)], &err.to_string());
  assert_de_tokens_error::<Region<Tree>>(
    &[Token::Bytes(&[])],
    "invalid region bytes: value does not have the size of the target type",
  );
  assert_de_tokens_error::<Region<Tree>>(
    &[Token::Str("x")],
    "invalid type: string \"x\", expected the bytes of a region",
  );
}

#[test]
fn json_round_trip() {
  let r = tree();
  let json = serde_json::to_string(&r).unwrap();
  assert!(json.starts_with('['));
  let back: Region<Tree> = serde_json::from_str(&json).unwrap();
  assert_eq!(back, r);
  assert_eq!(back.kids[1].kids[0].name.as_ref() as &str, "c");
  // Inside other data, as a field.
  let pair = (7u8, r);
  let back: (u8, Region<Tree>) =
    serde_json::from_str(&serde_json::to_string(&pair).unwrap()).unwrap();
  assert_eq!(back, pair);
}
