//! `Fixed` needs no heap: these tests also run with the library built without `alloc`,
//! `cargo test -p nearest --no-default-features --test no_alloc`.

use core::{mem::MaybeUninit, pin::Pin};

use nearest::{Error, Fixed, Near, Region, flat};

#[flat]
#[derive(Debug, PartialEq)]
struct Msg {
  id: u32,
  body: Near<str>,
  tags: Near<[u16]>,
}

#[flat]
#[derive(Debug, PartialEq)]
enum Expr {
  Lit(i64),
  Add(Near<Expr>, Near<Expr>),
}

#[test]
fn build_edit_move_and_load() {
  let mut bytes = [MaybeUninit::uninit(); 128];
  let mut r = Region::new_in(Msg::new(1, "hi", [1, 2]), Fixed::new(&mut bytes));
  let m = r.edit();
  m.body().set("hello");
  m.tags().push(3);
  let mut pinned: Pin<&mut Msg> = m.pin();
  let MsgPin { id, .. } = pinned.as_mut().project();
  *id.get_mut() = 2;
  let moved = r;
  assert_eq!((moved.id, &*moved.body, &moved.tags[..]), (2, "hello", &[1, 2, 3][..]));
  let mut other = [MaybeUninit::uninit(); 128];
  let copy = Region::new_in(&*moved, Fixed::new(&mut other));
  let image = copy.as_bytes().unwrap();
  assert_eq!(nearest::view::<Msg>(image), Ok(&*moved));
  let mut small = [MaybeUninit::uninit(); 64];
  let loaded = Region::<Msg, _>::from_bytes_in(image, Fixed::new(&mut small));
  assert_eq!(*loaded.unwrap(), *moved);
  // A valid image larger than the buffer is an error, not a panic.
  let mut tiny = [MaybeUninit::uninit(); 32];
  assert_eq!(
    Region::<Msg, _>::from_bytes_in(image, Fixed::new(&mut tiny)).err(),
    Some(Error::Alloc)
  );
  let mut tiny = [MaybeUninit::uninit(); 32];
  let fixed = Fixed::new(&mut tiny);
  let build = std::panic::AssertUnwindSafe(|| drop(Region::new_in(&*moved, &fixed)));
  assert!(std::panic::catch_unwind(build).is_err());
}

/// Under Miri everything below is eight times smaller: the same paths, in seconds rather than
/// minutes (the time that Miri reports is not the one that passes).
const SCALE: usize = if cfg!(miri) { 8 } else { 1 };

/// The buffer of the deep sums, and so their largest image.
const SIZE: usize = (1 << 14) / SCALE;

/// The terms of a deep sum: its image nearly fills `SIZE`.
const DEPTH: usize = 500 / SCALE;

/// A left-deep sum of `depth` terms, copied into `bytes`: checking it keeps about `depth` pointers
/// waiting.
fn sum(depth: usize, bytes: &mut [MaybeUninit<u8>]) -> Region<Expr, Fixed<'_>> {
  // Each term leaves the sum it wraps behind: four times the image is room enough.
  let mut room = vec![MaybeUninit::uninit(); 4 * SIZE];
  let mut r = Region::new_in(Expr::Lit(0), Fixed::new(&mut room));
  for i in 1..=depth {
    r.edit().update(|old| Expr::add(old, Expr::Lit(i64::try_from(i).unwrap())));
  }
  Region::new_in(&*r, Fixed::new(bytes))
}

/// Checking takes no memory from the caller, and does not recurse with the depth of the data: a
/// deep sum, which keeps about `DEPTH` pointers waiting, is checked within the frames of the check.
#[test]
fn checking_takes_no_memory_from_the_caller() {
  let mut bytes = vec![MaybeUninit::uninit(); SIZE];
  let deep = sum(DEPTH, &mut bytes);
  let image = deep.as_bytes().unwrap();
  assert!(image.len() > SIZE / 2);
  assert_eq!(nearest::view::<Expr>(image), Ok(&*deep));
  let mut again = vec![MaybeUninit::uninit(); SIZE];
  assert_eq!(*Region::<Expr, _>::from_bytes_in(image, Fixed::new(&mut again)).unwrap(), *deep);
  let cut = &image[..image.len() - 16];
  assert!(matches!(nearest::view::<Expr>(cut), Err(Error::Invalid(_))));
  let mut again = vec![MaybeUninit::uninit(); SIZE];
  assert!(matches!(
    Region::<Expr, _>::from_bytes_in(cut, Fixed::new(&mut again)),
    Err(Error::Invalid(_))
  ));
}
