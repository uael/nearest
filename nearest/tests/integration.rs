#![feature(pin_ergonomics)]
#![expect(incomplete_features, reason = "pin_ergonomics")]

use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
  panic::{AssertUnwindSafe, catch_unwind},
  pin::Pin,
};

use nearest::{Emit, Flat, Near, Region, flat};

#[flat]
#[derive(Debug, PartialEq, Eq, Hash)]
struct Func {
  name: Near<str>,
  entry: Near<Block>,
  flag: bool,
}

#[flat]
#[derive(Debug, PartialEq, Eq, Hash)]
struct Block {
  id: u32,
  insts: Near<[Inst]>,
  next: Option<Near<Block>>,
}

#[flat]
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Inst {
  op: u8,
  args: Near<[u32]>,
}

#[flat]
#[derive(Debug, PartialEq)]
enum Expr {
  Lit(i64),
  Neg(Near<Expr>),
  Add(Near<Expr>, Near<Expr>),
}

/// Plain data with padding.
#[flat]
#[derive(Debug, PartialEq, Clone, Copy)]
struct Pad {
  a: u8,
  b: u32,
}

#[flat]
#[derive(Debug, PartialEq)]
struct Prims {
  a: u8,
  b: u16,
  c: u32,
  d: u64,
  e: i8,
  f: i16,
  g: i32,
  h: i64,
  i: f32,
  j: f64,
  k: bool,
  l: char,
  m: [u16; 3],
}

#[flat]
#[derive(Debug, PartialEq, Clone, Copy)]
enum Color {
  Red,
  Green,
  Blue,
}

/// Plain data without padding.
#[flat]
#[derive(Debug, PartialEq)]
struct Rgb {
  r: u8,
  g: u8,
  b: u8,
  c: Color,
}

#[flat]
#[derive(Debug, PartialEq)]
struct Wrap(u32, Near<[Color]>);

#[flat]
#[derive(Debug, PartialEq)]
struct Unit;

#[flat]
#[derive(Debug, PartialEq)]
enum Stmt {
  Nop,
  Return { value: Option<Near<Expr>> },
  Let(Near<str>, Near<Expr>),
  Block { body: Near<[Stmt]>, unit: Unit },
  IntLit(i64),
  Gen(u8),
}

#[flat]
#[derive(Debug, PartialEq)]
struct Pair<A, B: ?Sized> {
  first: A,
  second: Near<B>,
}

#[flat]
#[derive(Debug, PartialEq)]
enum Tree<T> {
  Leaf(T),
  Node(Near<Tree<T>>, Near<Tree<T>>),
}

#[flat]
#[derive(Debug, PartialEq)]
struct Arr<const N: usize> {
  items: [u64; N],
  rest: Near<[[u8; N]]>,
}

#[flat]
#[derive(Debug, PartialEq)]
struct Aligned {
  small: Near<[u8]>,
  big: Near<[u64]>,
  text: Near<str>,
  wide: Near<[u16]>,
}

/// A recursive type spelled with `Self`.
#[flat]
#[derive(Debug, PartialEq)]
enum Chain {
  End,
  Link(u32, Near<Self>),
}

fn func() -> Region<Func> {
  Region::new(Func::new(
    "main",
    Near::new(Block::new(
      0,
      (Inst::new(1, [10, 20, 30]), Inst::new(2, ())),
      Near::new(Block::new(1, [Inst::new(3, Near::from_iter(0..5))], None)),
    )),
    true,
  ))
}

fn eval(e: &Expr) -> i64 {
  match e {
    Expr::Lit(n) => *n,
    Expr::Neg(a) => -eval(a),
    Expr::Add(a, b) => eval(a) + eval(b),
  }
}

fn ops(b: &Block) -> Vec<u8> {
  b.insts.iter().map(|i| i.op).collect()
}

fn hash<T: Hash + ?Sized>(t: &T) -> u64 {
  let mut h = DefaultHasher::new();
  t.hash(&mut h);
  h.finish()
}

#[test]
fn build_and_read() {
  let f = func();
  assert_eq!(&*f.name, "main");
  assert!(f.flag);
  assert_eq!(f.entry.id, 0);
  assert_eq!(f.entry.insts.len(), 2);
  assert_eq!(f.entry.insts[0].args[..], [10, 20, 30]);
  assert!(f.entry.insts[1].args.is_empty());
  let next = f.entry.next.as_ref().unwrap();
  assert_eq!((next.id, &next.insts[0].args[..]), (1, &[0, 1, 2, 3, 4][..]));
  assert!(next.next.is_none());
  assert_eq!(*Region::new(7u32), 7);
  assert_eq!(*Region::new(Pad { a: 1, b: 2 }), Pad { a: 1, b: 2 });
}

#[test]
fn every_primitive() {
  let p = Prims {
    a: u8::MAX,
    b: u16::MAX - 1,
    c: 0xdead_beef,
    d: u64::MAX - 2,
    e: i8::MIN,
    f: -2,
    g: i32::MIN + 3,
    h: -4,
    i: 1.5,
    j: -2.25,
    k: true,
    l: 'é',
    m: [1, 2, 3],
  };
  let r = Region::new(&p);
  assert_eq!(*r, p);
  // The same through the generated constructor, and as list elements.
  let q = Region::new(Prims::new(
    1u8,
    2u16,
    3u32,
    4u64,
    5i8,
    6i16,
    7i32,
    8i64,
    9.0f32,
    10.0f64,
    false,
    'x',
    [4, 5, 6],
  ));
  assert_eq!((q.a, q.h, q.l, q.m), (1, 8, 'x', [4, 5, 6]));
  const { assert!(Rgb::PLAIN && <[u16; 3]>::PLAIN && Color::PLAIN && Unit::PLAIN) };
  const { assert!(!Prims::PLAIN && !Pad::PLAIN && !Func::PLAIN && !Expr::PLAIN) };
  const { assert!(!<Near<u8>>::PLAIN && !<Option<Near<u8>>>::PLAIN) };
  assert_eq!(
    *Region::new(Rgb { r: 1, g: 2, b: 3, c: Color::Blue }),
    Rgb { r: 1, g: 2, b: 3, c: Color::Blue }
  );
  let lists = Region::new(Pair::<u8, [Prims]>::new(0, [&p, &p]));
  assert_eq!(lists.second[1], p);
}

#[test]
fn every_emit_source() {
  let other = func();
  let v = [5u32, 6, 7];
  let r = Region::new(Block::new(
    9, // a value
    (
      Inst::new(1, &v[..]),                                   // a slice copy
      Inst::new(2, [1, 2]),                                   // an array
      Inst::new(3, (4, 5, 6)),                                // a tuple
      Inst::new(4, ()),                                       // the empty list
      Inst::new(5, Near::from_iter(v.iter().map(|x| x * 2))), // an iterator
    ),
    Near::new(&*other.entry), // a deep copy from another region
  ));
  assert_eq!(ops(&r), [1, 2, 3, 4, 5]);
  assert_eq!(r.insts[0].args[..], [5, 6, 7]);
  assert_eq!(r.insts[2].args[..], [4, 5, 6]);
  assert_eq!(r.insts[4].args[..], [10, 12, 14]);
  assert_eq!(**r.next.as_ref().unwrap(), *other.entry);

  // Tuples mix builders, values and copies; lists of lists.
  let inst = Region::new(Inst::new(8, [9]));
  let mixed = Region::new(Block::new(
    0,
    (Inst::new(1, ()), &*inst, Inst::new(3, Near::from_iter(None::<u32>))),
    None,
  ));
  assert_eq!(ops(&mixed), [1, 8, 3]);
  let t12 = Region::new(Inst::new(0, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)));
  assert_eq!(t12.args.iter().sum::<u32>(), 78);
  let nested = Region::new(Pair::<u8, [Near<[u8]>]>::new(0, ([1u8], (), [2, 3])));
  assert_eq!(nested.second.iter().map(|l| l.len()).collect::<Vec<_>>(), [1, 0, 2]);

  // `Near::new` writes `Near<T>` and `Option<Near<T>>`; `None` is a value.
  let e = Region::new(Stmt::r#return(Near::new(Expr::Lit(1))));
  assert!(matches!(&*e, Stmt::Return { value: Some(v) } if **v == Expr::Lit(1)));
  let e = Region::new(Stmt::r#return(None));
  assert_eq!(*e, Stmt::Return { value: None });
  // Keywords, including those reserved by the edition, become raw identifiers.
  assert_eq!(*Region::new(Stmt::r#gen(3)), Stmt::Gen(3));

  // Text, including non-ASCII.
  let s = Region::new(Stmt::r#let("héllo, 世界 🦀", Near::new(Expr::Lit(0))));
  let Stmt::Let(name, _) = &*s else { unreachable!() };
  assert_eq!(&**name, "héllo, 世界 🦀");
  assert_eq!(name.chars().count(), 11);
  assert_eq!(&*Region::new(Pair::<u8, str>::new(0, "")).second, "");
}

#[test]
fn enums_generics_and_nesting() {
  fn sum(tree: &Tree<u16>) -> u16 {
    match tree {
      Tree::Leaf(x) => *x,
      Tree::Node(l, r) => sum(l) + sum(r),
    }
  }
  let expr =
    Region::new(Expr::add(Near::new(Expr::Lit(2)), Near::new(Expr::neg(Near::new(Expr::lit(5))))));
  assert_eq!(eval(&expr), -3);

  let stmt = Region::new(Stmt::block(
    (Stmt::Nop, Stmt::int_lit(3), Stmt::block([Stmt::IntLit(4)], Unit)),
    Unit,
  ));
  let Stmt::Block { body, .. } = &*stmt else { unreachable!() };
  assert_eq!(body[0], Stmt::Nop);
  assert_eq!(body[1], Stmt::IntLit(3));
  assert!(matches!(&body[2], Stmt::Block { body, .. } if body[0] == Stmt::IntLit(4)));

  let tree = Region::new(Tree::node(
    Near::new(Tree::Leaf(1u16)),
    Near::new(Tree::node(Near::new(Tree::leaf(2)), Near::new(Tree::Leaf(3)))),
  ));
  assert_eq!(sum(&tree), 6);
  let pair = Region::new(Pair::new(
    Color::Blue,
    Near::new(Tree::leaf(Wrap::new(1, [Color::Red, Color::Green]))),
  ));
  assert_eq!(pair.first, Color::Blue);
  assert!(matches!(&*pair.second, Tree::Leaf(Wrap(1, c)) if c[..] == [Color::Red, Color::Green]));

  let arr = Region::new(Arr::<2>::new([1, 2], [[3, 4], [5, 6]]));
  assert_eq!((arr.items, &arr.rest[..]), ([1, 2], &[[3, 4], [5, 6]][..]));
  let chain = Region::new(Chain::link(1, Near::new(Chain::link(2, Near::new(Chain::End)))));
  assert!(
    matches!(&*chain, Chain::Link(1, n) if matches!(&**n, Chain::Link(2, e) if **e == Chain::End))
  );
}

#[test]
fn lists_are_aligned() {
  let r = Region::new(Aligned::new([1u8], [2u64, 3], "abc", [4u16]));
  let r2 = Region::new(Aligned::new((), (), "", ()));
  for r in [&r, &r2, &r.clone()] {
    assert_eq!(r.big.as_ptr().addr() % 8, 0);
    assert_eq!(r.wide.as_ptr().addr() % 2, 0);
  }
  assert_eq!(r.big[..], [2, 3]);
  assert!(r2.small.is_empty() && r2.big.is_empty() && r2.text.is_empty());
  assert_eq!(size_of::<Near<[u64]>>(), 4);
  assert_eq!(size_of::<Option<Near<Expr>>>(), 4);
}

#[test]
fn near_traits() {
  let r = Region::new(Pair::<Inst, [Inst]>::new(
    Inst::new(1, [2]),
    (Inst::new(1, [2]), Inst::new(1, [3])),
  ));
  let (a, b) = (&r.second[0], &r.second[1]);
  assert_eq!(a, &r.first);
  assert!(a < b);
  assert_eq!(a.args.cmp(&b.args), std::cmp::Ordering::Less);
  assert_eq!(a.args.partial_cmp(&b.args), Some(std::cmp::Ordering::Less));
  assert_eq!(hash(&a.args), hash(&r.first.args));
  assert_ne!(a.args, b.args);
  assert_eq!(format!("{:?}", a.args), "[2]");
  assert_eq!(format!("{:?}", &*r.second), "[Inst { op: 1, args: [2] }, Inst { op: 1, args: [3] }]");
  let total: u32 = (&r.second).into_iter().map(|i| i.args[0]).sum();
  assert_eq!(total, 5);
}

#[test]
fn region_traits() {
  let f = func();
  let g = f.clone();
  assert_eq!(f, g);
  assert_eq!(hash(&f), hash(&g));
  assert_eq!(format!("{f:?}"), format!("{:?}", *g));
  assert!(format!("{f:?}").starts_with("Func { name: \"main\""));
  // The clone is independent.
  let mut h = f.clone();
  h.edit(|root, slot| {
    let Func { flag, name, .. } = root;
    *flag.get_mut() = false;
    slot.at(name, |_, new| new.put("other"));
  });
  assert_eq!((&*f.name, f.flag, &*h.name, h.flag), ("main", true, "other", false));
  assert_ne!(f, h);
  drop(f);
  assert_eq!(*g.entry, *h.entry);
}

#[test]
fn edit_in_place_needs_no_api() {
  let mut f = func();
  let before = f.byte_len();
  let out = f.edit(|root, _| {
    let Func { flag, entry, .. } = root;
    *flag.get_mut() = false;
    let Block { id, insts, next } = entry.as_pin_mut();
    *id.get_mut() = 9;
    for inst in insts {
      let Inst { op, args } = inst;
      *op.get_mut() += 10;
      args.as_pin_mut().get_mut().reverse();
    }
    let mut next = next;
    next.set(None);
    "out"
  });
  assert_eq!(out, "out");
  assert_eq!(f.byte_len(), before);
  assert_eq!((f.flag, f.entry.id, f.entry.next.is_none()), (false, 9, true));
  assert_eq!((f.entry.insts[0].op, &f.entry.insts[0].args[..]), (11, &[30, 20, 10][..]));

  let mut e = Region::new(Expr::Lit(1));
  e.edit(|e, _| {
    let mut e = e;
    e.set(Expr::Lit(5));
  });
  assert_eq!(*e, Expr::Lit(5));
  let mut l = Region::new(Inst::new(0, [3, 1, 2]));
  l.edit(|root, _| {
    let Inst { args, .. } = root;
    args.as_pin_mut().get_mut().sort_unstable();
  });
  assert_eq!(l.args[..], [1, 2, 3]);
  let mut c = Region::new(Wrap::new(0, [Color::Red, Color::Blue]));
  c.edit(|root, _| {
    let Wrap(_, colors) = root;
    for c in colors {
      let mut c = c;
      c.set(Color::Green);
    }
  });
  assert_eq!(c.1[..], [Color::Green; 2]);
}

#[test]
fn edits_that_allocate() {
  let mut f = func();
  f.edit(|root, slot| {
    let Func { entry, name, .. } = root;
    slot.at(name, |_, new| new.put("renamed"));
    let Block { insts, next, .. } = entry.as_pin_mut();
    slot.at(insts, |old, new| {
      for inst in old {
        if inst.op == 1 {
          new.push(inst);
        }
      }
      new.push(Inst::new(7, [70, 71]));
    });
    let Block { insts, next, .. } = next.as_pin_mut().unwrap().as_pin_mut();
    slot.at(insts, |_, new| {
      new.put(Near::from_iter((0..3u8).map(|op| Inst::new(op, [u32::from(op)]))));
    });
    slot.at(next, |_, new| new.put(Near::new(Block::new(2, (), None))));
  });
  assert_eq!(&*f.name, "renamed");
  let b0 = &f.entry;
  assert_eq!(ops(b0), [1, 7]);
  assert_eq!((&b0.insts[0].args[..], &b0.insts[1].args[..]), (&[10, 20, 30][..], &[70, 71][..]));
  let b1 = b0.next.as_ref().unwrap();
  assert_eq!(
    b1.insts.iter().map(|i| (i.op, i.args[0])).collect::<Vec<_>>(),
    [(0, 0), (1, 1), (2, 2)]
  );
  assert_eq!(b1.next.as_ref().unwrap().id, 2);

  let compact = Region::new(&*f);
  assert!(compact.byte_len() < f.byte_len());
  assert_eq!(*compact, *f);
}

#[test]
fn put_push_and_at_semantics() {
  let mut r = Region::new(Inst::new(0, [1, 2]));
  // Neither `put` nor `push`: the place is left as is.
  r.edit(|root, slot| {
    let Inst { args, .. } = root;
    slot.at(args, |_, _| {});
  });
  assert_eq!(r.args[..], [1, 2]);
  // The last `put` wins; `push` after `put` starts a new list; `put` after `push` replaces it.
  r.edit(|root, slot| {
    let Inst { args, .. } = root;
    slot.at(args, |_, new| {
      new.put([9]);
      new.put([8]);
      new.push(5);
      new.push(6);
    });
  });
  assert_eq!(r.args[..], [5, 6]);
  r.edit(|root, slot| {
    let Inst { args, .. } = root;
    slot.at(args, |_, new| {
      new.push(5);
      new.put(());
    });
  });
  assert!(r.args.is_empty());
  // Many pushes grow the list, also with elements that allocate.
  let mut b = Region::new(Block::new(0, (), None));
  b.edit(|root, slot| {
    let Block { insts, .. } = root;
    slot.at(insts, |_, new| {
      for i in 0..100u8 {
        new.push(Inst::new(i, Near::from_iter(0..u32::from(i % 7))));
      }
    });
  });
  assert_eq!(b.insts.len(), 100);
  assert!(b.insts.iter().enumerate().all(|(i, x)| x.op as usize == i && x.args.len() == i % 7));
  // The root's own slot, with a value from another region.
  let other = func();
  let mut f = func();
  f.edit(|_, slot| slot.put(Func::new("x", &other.entry, false)));
  assert_eq!((&*f.name, &*f.entry, f.flag), ("x", &*other.entry, false));
}

#[test]
fn nested_and_sibling_ats() {
  // Three levels of chained blocks, each edited at every level.
  let mut r = Region::new(Block::new(
    0,
    [Inst::new(0, [0])],
    Near::new(Block::new(
      1,
      [Inst::new(1, [1])],
      Near::new(Block::new(2, [Inst::new(2, [2])], None)),
    )),
  ));
  let depth = r.edit(|root, slot| {
    let Block { insts, next, .. } = root;
    slot.at(insts, |old, new| {
      for i in old {
        new.push(i);
      }
      new.push(Inst::new(10, ()));
    });
    slot.at(next, |next, slot| {
      let Block { insts, next, .. } = next.as_pin_mut().unwrap().as_pin_mut();
      // Sibling places at the same level.
      let Inst { args: a0, .. } = insts.into_iter().next().unwrap();
      slot.at(a0, |old, new| new.put(Near::from_iter(old.iter().map(|x| x + 100))));
      slot.at(next, |next, slot| {
        let Block { insts, next, .. } = next.as_pin_mut().unwrap().as_pin_mut();
        slot.at(insts, |old, slot| {
          let Inst { args, .. } = old.into_iter().next().unwrap();
          slot.at(args, |_, new| {
            new.push(7);
            new.push(8);
          });
        });
        slot.at(next, |_, new| new.put(Near::new(Block::new(3, (), None))));
        4
      })
    })
  });
  assert_eq!(depth, 4);
  assert_eq!(ops(&r), [0, 10]);
  let b1 = r.next.as_ref().unwrap();
  assert_eq!(b1.insts[0].args[..], [101]);
  let b2 = b1.next.as_ref().unwrap();
  assert_eq!(b2.insts[0].args[..], [7, 8]);
  assert_eq!(b2.next.as_ref().unwrap().id, 3);
  assert_eq!(*Region::new(&*r), *r);

  // Sibling elements of one list, each replaced.
  let mut l =
    Region::new(Block::new(0, (Inst::new(0, [1]), Inst::new(1, [2]), Inst::new(2, [3])), None));
  l.edit(|root, slot| {
    let Block { insts, .. } = root;
    for (k, inst) in insts.into_iter().enumerate() {
      let Inst { args, .. } = inst;
      slot.at(args, |old, new| {
        new.put(Near::from_iter(old.iter().copied().chain([k as u32 * 10])));
      });
    }
  });
  assert_eq!(
    l.insts.iter().map(|i| i.args.to_vec()).collect::<Vec<_>>(),
    [vec![1, 0], vec![2, 10], vec![3, 20]]
  );
}

#[test]
fn moves_wrap_unwrap_rotate() {
  let mut e = Region::new(Expr::add(Near::new(Expr::Lit(2)), Near::new(Expr::Lit(5))));
  // x -> Neg(x)
  e.edit(|old, slot| slot.put(Expr::neg(Near::new(old))));
  assert_eq!(eval(&e), -7);
  // Neg(x) -> x
  e.edit(|old, slot| {
    if let Expr::Neg(inner) = old {
      slot.put(inner.as_pin_mut());
    }
  });
  assert_eq!(eval(&e), 7);
  // Add(a, b) -> Add(b, Neg(a)): pointers are re-linked, nothing is copied.
  let before = e.byte_len();
  e.edit(|old, slot| {
    if let Expr::Add(a, b) = old {
      slot.put(Expr::add(b, Near::new(Expr::neg(a))));
    }
  });
  assert_eq!(eval(&e), 3);
  assert!(e.byte_len() - before <= 64);
  assert!(matches!(&*e, Expr::Add(b, _) if **b == Expr::Lit(5)));
  // Move a grandchild up while replacing a child, inside a nested slot.
  e.edit(|old, slot| {
    if let Expr::Add(_, n) = old {
      slot.at(n, |n, slot| {
        if let Expr::Neg(a) = n.as_pin_mut() {
          slot.put(Near::new(Expr::add(a, Near::new(Expr::Lit(10)))));
        }
      });
    }
  });
  assert_eq!(eval(&e), 17);
  assert_eq!(eval(&Region::new(&*e)), 17);
}

#[test]
fn list_moves() {
  let mut b = Region::new(Block::new(
    0,
    Near::from_iter((0..6u8).map(|i| Inst::new(i, [u32::from(i); 2]))),
    None,
  ));
  // Filter.
  b.edit(|root, slot| {
    let Block { insts, .. } = root;
    slot.at(insts, |old, new| new.put(Near::from_iter(old.into_iter().filter(|i| i.op % 2 == 0))));
  });
  assert_eq!(ops(&b), [0, 2, 4]);
  // Reverse.
  b.edit(|root, slot| {
    let Block { insts, .. } = root;
    slot.at(insts, |old, new| new.put(Near::from_iter(old.into_iter().rev())));
  });
  assert_eq!(ops(&b), [4, 2, 0]);
  // Insert in the middle.
  b.edit(|root, slot| {
    let Block { insts, .. } = root;
    slot.at(insts, |old, new| {
      for (k, i) in old.into_iter().enumerate() {
        if k == 1 {
          new.push(Inst::new(9, [9]));
        }
        new.push(i);
      }
    });
  });
  assert_eq!(ops(&b), [4, 9, 2, 0]);
  assert!(b.insts.iter().all(|i| i.args.iter().all(|&a| a == u32::from(i.op))));
  // Duplicate: copies of old elements may be taken as often as wanted.
  b.edit(|root, slot| {
    let Block { insts, .. } = root;
    slot.at(insts, |old, new| {
      let copies: Vec<&Inst> = old.iter().collect();
      new.put(Near::from_iter(copies.iter().chain(copies.iter()).copied()));
    });
  });
  assert_eq!(ops(&b), [4, 9, 2, 0, 4, 9, 2, 0]);
  assert_eq!(*Region::new(&*b), *b);
}

#[test]
fn copies_between_and_within_regions() {
  let a = func();
  let mut b = func();
  b.edit(|root, slot| {
    let Func { entry, .. } = root;
    let Block { next, .. } = entry.as_pin_mut();
    slot.at(next, |_, new| new.put(Near::new(&*a.entry)));
  });
  assert_eq!(**b.entry.next.as_ref().unwrap(), *a.entry);
  // Within one region: copy the first block into its own tail.
  b.edit(|root, slot| {
    let Func { entry, .. } = root;
    let Block { insts, next, .. } = entry.as_pin_mut();
    let first: Vec<&Inst> = insts.iter().collect();
    slot.at(next, |_, new| new.put(Near::new(Block::new(5, Near::from_iter(first), None))));
  });
  assert_eq!(ops(b.entry.next.as_ref().unwrap()), ops(&a.entry));
  drop(a);
  assert_eq!(&*b.name, "main");
}

#[test]
fn compaction() {
  let mut b = Region::new(Block::new(0, (), None));
  for i in 0..20u8 {
    b.edit(|root, slot| {
      let Block { insts, .. } = root;
      slot.at(insts, |old, new| {
        for x in old {
          new.push(x);
        }
        new.push(Inst::new(i, [u32::from(i)]));
      });
    });
  }
  let c = Region::new(&*b);
  assert!(c.byte_len() * 2 < b.byte_len());
  assert_eq!(*c, *b);
  assert_eq!(Region::new(&*c).byte_len(), c.byte_len());
}

#[test]
fn bytes_round_trip() {
  let mut r = Region::<[Pad; 2]>::new([Pad { a: 1, b: 2 }, Pad { a: 3, b: 4 }]);
  r.edit(|root, _| {
    let [first, _] = root;
    let mut first = first;
    first.set(Pad { a: 9, b: 9 }); // may leave padding uninitialized in the buffer
  });
  let bytes = r.to_bytes();
  assert!(bytes.iter().map(|&b| u32::from(b)).sum::<u32>() > 0); // every byte is initialized
  // SAFETY: from `to_bytes` of the same type.
  let back = unsafe { Region::<[Pad; 2]>::from_bytes(&bytes) };
  assert_eq!(*back, [Pad { a: 9, b: 9 }, Pad { a: 3, b: 4 }]);
  let f = func();
  let bytes = f.to_bytes();
  assert_eq!(bytes.len(), Region::new(&*f).byte_len());
  // SAFETY: from `to_bytes` of the same type.
  let back = unsafe { Region::<Func>::from_bytes(&bytes) };
  assert_eq!(*back, *f);
  assert_eq!(back.to_bytes(), bytes);
}

#[test]
fn panic_mid_edit_leaves_a_valid_region() {
  let mut f = func();
  let r = catch_unwind(AssertUnwindSafe(|| {
    f.edit(|root, slot| {
      let Func { entry, name, .. } = root;
      slot.at(name, |_, new| new.put("committed before the panic"));
      let Block { insts, .. } = entry.as_pin_mut();
      slot.at(insts, |_, new| {
        new.push(Inst::new(1, [1]));
        new.put(Near::from_iter(
          (0..3).map(|i| if i == 2 { panic!("user code panics") } else { Inst::new(i, ()) }),
        ));
      });
    });
  }));
  assert!(r.is_err());
  assert_eq!(&*f.name, "committed before the panic");
  assert_eq!(f.entry.insts.len(), 2);
  assert_eq!(*Region::new(&*f), *f);
  // The region stays usable.
  f.edit(|root, slot| {
    let Func { name, .. } = root;
    slot.at(name, |_, new| new.put("again"));
  });
  assert_eq!(&*f.name, "again");
}

#[flat]
struct Item {
  next: Option<Near<u32>>,
  cells: Near<[u32]>,
}

#[flat]
struct Items {
  items: Near<[Item]>,
}

#[test]
fn a_caught_panic_mid_push_leaves_no_stale_bytes() {
  let mut r = Region::new(Items::new(()));
  r.edit(|root, slot| {
    let Items { items } = root;
    slot.at(items, |_, new| {
      let boom = (0..2u32).map(|i| if i == 1 { panic!("boom") } else { i });
      let item = Item::new(Near::new(7u32), Near::from_iter(boom));
      assert!(catch_unwind(AssertUnwindSafe(|| new.push(item))).is_err());
      new.push(Item::new(None, [1, 2]));
    });
  });
  assert_eq!(r.items.len(), 1);
  assert_eq!(r.items[0].next.as_deref(), None);
  assert_eq!(r.items[0].cells[..], [1, 2]);
}

/// Yields `n` items while claiming `hint` as its lower bound.
struct Liar {
  n: u32,
  hint: usize,
}

impl Iterator for Liar {
  type Item = u32;
  fn next(&mut self) -> Option<u32> {
    self.n = self.n.checked_sub(1)?;
    Some(self.n)
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.hint, Some(self.hint))
  }
}

#[test]
fn size_hint_may_lie() {
  for (n, hint) in [(3, 1000), (1000, 0), (50, 3), (0, 10)] {
    let r = Region::new(Inst::new(0, Near::from_iter(Liar { n, hint })));
    assert_eq!(r.args.to_vec(), (0..n).rev().collect::<Vec<_>>());
    // Elements that allocate while the list grows.
    let b = Region::new(Block::new(
      0,
      Near::from_iter(
        Liar { n, hint }.map(|i| Inst::new((i % 5) as u8, Near::from_iter(0..i % 5))),
      ),
      None,
    ));
    assert_eq!(b.insts.len(), n as usize);
    assert!(b.insts.iter().all(|i| i.args.len() == usize::from(i.op) % 5));
  }
}

/// Anything `impl Emit` can be passed on generically.
fn wrap_twice<'o>(e: impl Emit<'o, Expr>) -> impl Emit<'o, Expr> {
  Expr::neg(Near::new(Expr::neg(Near::new(e))))
}

#[test]
fn builders_compose() {
  let r = Region::new(wrap_twice(Expr::Lit(4)));
  assert_eq!(eval(&r), 4);
  let mut r = r;
  r.edit(|old, slot| slot.put(wrap_twice(old)));
  assert_eq!(eval(&r), 4);
  assert!(matches!(&*r, Expr::Neg(_)));
}

#[test]
fn pins_are_pins() {
  let mut r = Region::new(Inst::new(1, [1, 2]));
  r.edit(|root: Pin<&mut Inst>, _| {
    let Inst { args, .. } = root;
    let total: u32 = args.into_iter().map(|x: Pin<&mut u32>| *x).sum();
    assert_eq!(total, 3);
  });
}

#[test]
fn many_edits_match_a_model() {
  type Model = Vec<(u8, Vec<u32>)>;
  fn check(r: &Block, model: &Model) {
    let got: Model = r.insts.iter().map(|i| (i.op, i.args.to_vec())).collect();
    assert_eq!(&got, model);
  }
  let mut r = Region::new(Block::new(0, (), None));
  let mut model: Model = Vec::new();
  let mut seed = 0x2545_f491u32;
  let mut rnd = move |n: u32| {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    seed % n
  };
  let steps = if cfg!(miri) { 60 } else { 2000 };
  for step in 0..steps {
    let k = if model.is_empty() { 0 } else { rnd(model.len() as u32) as usize };
    match rnd(9) {
      // Append, moving every old element.
      0 => {
        let (op, args) = (step as u8, (0..rnd(5)).collect::<Vec<_>>());
        model.push((op, args.clone()));
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          slot.at(insts, |old, new| {
            for i in old {
              new.push(i);
            }
            new.push(Inst::new(op, Near::from_iter(args.iter())));
          });
        });
      }
      // Remove one.
      1 if !model.is_empty() => {
        model.remove(k);
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          slot.at(insts, |old, new| {
            new.put(Near::from_iter(
              old.into_iter().enumerate().filter(|(i, _)| *i != k).map(|(_, x)| x),
            ));
          });
        });
      }
      // Reverse.
      2 => {
        model.reverse();
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          slot.at(insts, |old, new| new.put(Near::from_iter(old.into_iter().rev())));
        });
      }
      // Append to one element's arguments.
      3 if !model.is_empty() => {
        model[k].1.push(step);
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          let Inst { args, .. } = insts.into_iter().nth(k).unwrap();
          slot.at(args, |old, new| new.put(Near::from_iter(old.iter().copied().chain([step]))));
        });
      }
      // Insert before `k`, copying element `k`.
      4 if !model.is_empty() => {
        model.insert(k, model[k].clone());
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          slot.at(insts, |old, new| {
            let copy = Region::new(&old[k]);
            for (i, x) in old.into_iter().enumerate() {
              if i == k {
                new.push(&*copy);
              }
              new.push(x);
            }
          });
        });
      }
      // Change in place.
      5 if !model.is_empty() => {
        model[k].0 = model[k].0.wrapping_add(1);
        model[k].1.iter_mut().for_each(|a| *a *= 2);
        r.edit(|root, _| {
          let Block { insts, .. } = root;
          let Inst { op, args } = insts.into_iter().nth(k).unwrap();
          *op.get_mut() = op.wrapping_add(1);
          args.as_pin_mut().get_mut().iter_mut().for_each(|a| *a *= 2);
        });
      }
      // Swap two elements by rebuilding the list from moves.
      6 if model.len() > 1 => {
        let j = rnd(model.len() as u32) as usize;
        model.swap(k, j);
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          slot.at(insts, |old, new| {
            let mut v: Vec<_> = old.into_iter().map(Some).collect();
            let order = (0..v.len()).map(|i| {
              if i == k {
                j
              } else if i == j {
                k
              } else {
                i
              }
            });
            new.put(Near::from_iter(
              order.collect::<Vec<_>>().into_iter().map(|i| v[i].take().unwrap()),
            ));
          });
        });
      }
      // Sort each element's arguments in place, and clear one list through a nested slot.
      7 if !model.is_empty() => {
        for (_, args) in &mut model {
          args.sort_unstable();
        }
        model[k].1.clear();
        r.edit(|root, slot| {
          let Block { insts, .. } = root;
          for (i, inst) in insts.into_iter().enumerate() {
            let Inst { args, .. } = inst;
            if i == k {
              slot.at(args, |_, new| new.put(()));
            } else {
              args.as_pin_mut().get_mut().sort_unstable();
            }
          }
        });
      }
      // Compact, clone, or round-trip through bytes.
      _ => {
        r = match step % 3 {
          0 => Region::new(&*r),
          1 => r.clone(),
          // SAFETY: from `to_bytes` of the same type.
          _ => unsafe { Region::from_bytes(&r.to_bytes()) },
        };
      }
    }
    check(&r, &model);
  }
  let compact = Region::new(&*r);
  check(&compact, &model);
  assert!(compact.byte_len() <= r.byte_len());
}
