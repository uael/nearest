#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
  num::NonZero,
  panic::{AssertUnwindSafe, catch_unwind},
};

use nearest::{Edit, Emit, Flat, Near, Region, flat};

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
  n: NonZero<u32>,
  o: NonZero<i8>,
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
#[derive(Debug, PartialEq, Clone, Copy)]
struct Rgb {
  r: u8,
  g: u8,
  b: u8,
  c: Color,
}

/// Plain data with a payload.
#[flat]
#[derive(Debug, PartialEq, Clone, Copy)]
enum Shape {
  Dot,
  Circle(u32),
  Rect { w: u16, h: u16 },
}

#[flat]
#[derive(Debug, PartialEq)]
struct Wrap(u32, Near<[Color]>);

#[flat]
#[derive(Debug, PartialEq)]
enum Stmt {
  Nop,
  Return { value: Option<Near<Expr>> },
  Let(Near<str>, Near<Expr>),
  Block { body: Near<[Stmt]>, tag: u8 },
  IntLit(i64),
  Gen(u8),
}

#[flat]
#[derive(Debug, PartialEq)]
struct Pair<A, B> {
  first: A,
  second: Near<B>,
}

#[flat]
#[derive(Debug, PartialEq)]
struct Lst<T> {
  items: Near<[T]>,
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

/// A recursive type spelled with `Self`, and a qualified pointer path.
#[flat]
#[derive(Debug, PartialEq)]
enum Chain {
  End,
  Link(u32, nearest::Near<Self>),
}

/// Fields whose names collide with the handle's own methods.
#[flat]
#[derive(Debug, PartialEq)]
struct Names {
  set: u32,
  replace: Near<str>,
  r#as_mut: u8,
}

/// An alias of a pointer is not boxed by constructors.
type Ptr<T> = Near<T>;

#[flat]
#[derive(Debug, PartialEq)]
struct Aliased {
  p: Ptr<u64>,
}

mod private {
  use nearest::{Near, flat};

  #[flat]
  pub struct Mixed {
    pub open: u32,
    hidden: Near<str>,
  }

  impl Mixed {
    pub fn hidden(&self) -> &str {
      &self.hidden
    }
  }

  pub fn rename(m: &mut nearest::Edit<Mixed>, to: &str) {
    m.hidden().set(to);
  }
}

fn func() -> Region<Func> {
  Region::new(Func::new(
    "main",
    Block::new(
      0,
      (Inst::new(1, [10, 20, 30]), Inst::new(2, ())),
      Near::new(Block::new(1, [Inst::new(3, Near::from_iter(0..5))], None)),
    ),
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

fn show(e: &Expr) -> String {
  match e {
    Expr::Lit(n) => n.to_string(),
    Expr::Neg(a) => format!("-{}", show(a)),
    Expr::Add(a, b) => format!("({} + {})", show(a), show(b)),
  }
}

fn ops(b: &Block) -> Vec<u8> {
  b.insts.iter().map(|i| i.op).collect()
}

fn args(b: &Block) -> Vec<Vec<u32>> {
  b.insts.iter().map(|i| i.args.to_vec()).collect()
}

fn hash<T: Hash + ?Sized>(t: &T) -> u64 {
  let mut h = DefaultHasher::new();
  t.hash(&mut h);
  h.finish()
}

// ---------------------------------------------------------------------------------------------
// Building and reading

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
    n: NonZero::new(9).unwrap(),
    o: NonZero::new(-1).unwrap(),
  };
  let r = Region::new(&p);
  assert_eq!(*r, p);
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
    NonZero::new(1).unwrap(),
    NonZero::new(2).unwrap(),
  ));
  assert_eq!((q.a, q.h, q.l, q.m, q.n.get()), (1, 8, 'x', [4, 5, 6], 1));
  const { assert!(Rgb::PLAIN && <[u16; 3]>::PLAIN && Color::PLAIN && <NonZero<u64>>::PLAIN) };
  const { assert!(!Prims::PLAIN && !Pad::PLAIN && !Func::PLAIN && !Expr::PLAIN && !Shape::PLAIN) };
  const { assert!(!<Near<u8>>::PLAIN && !<Option<Near<u8>>>::PLAIN && !<Near<str>>::PLAIN) };
  assert_eq!(
    *Region::new(Rgb { r: 1, g: 2, b: 3, c: Color::Blue }),
    Rgb { r: 1, g: 2, b: 3, c: Color::Blue }
  );
  let lists = Region::new(Lst::<Prims>::new([&p, &p]));
  assert_eq!(lists.items[1], p);
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
      Inst::new(6, &other.entry.insts[0].args),               // a copy of a list
    ),
    Near::new(&*other.entry), // a deep copy from another region
  ));
  assert_eq!(ops(&r), [1, 2, 3, 4, 5, 6]);
  assert_eq!(r.insts[0].args[..], [5, 6, 7]);
  assert_eq!(r.insts[2].args[..], [4, 5, 6]);
  assert_eq!(r.insts[4].args[..], [10, 12, 14]);
  assert_eq!(r.insts[5].args[..], [10, 20, 30]);
  assert_eq!(**r.next.as_ref().unwrap(), *other.entry);

  // Tuples mix builders, values and copies; lists of lists; lists of pointers.
  let inst = Region::new(Inst::new(8, [9]));
  let mixed = Region::new(Block::new(
    0,
    (Inst::new(1, ()), &*inst, Inst::new(3, Near::from_iter(None::<u32>))),
    None,
  ));
  assert_eq!(ops(&mixed), [1, 8, 3]);
  let t12 = Region::new(Inst::new(0, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)));
  assert_eq!(t12.args.iter().sum::<u32>(), 78);
  let nested = Region::new(Lst::<Near<[u8]>>::new(([1u8], (), [2, 3])));
  assert_eq!(nested.items.iter().map(|l| l.len()).collect::<Vec<_>>(), [1, 0, 2]);
  let ptrs =
    Region::new(Lst::<Near<Expr>>::new((Near::new(Expr::Lit(1)), Near::new(&*inst_expr()))));
  assert_eq!(ptrs.items.iter().map(|e| eval(e)).collect::<Vec<_>>(), [1, -2]);

  // `Near::new` writes `Option<Near<T>>`; `None` is a value.
  let e = Region::new(Stmt::r#return(Near::new(Expr::Lit(1))));
  assert!(matches!(&*e, Stmt::Return { value: Some(v) } if **v == Expr::Lit(1)));
  let e = Region::new(Stmt::r#return(None));
  assert_eq!(*e, Stmt::Return { value: None });
  // Keywords, including those reserved by the edition, become raw identifiers.
  assert_eq!(*Region::new(Stmt::r#gen(3)), Stmt::Gen(3));

  // Text, including non-ASCII.
  let s = Region::new(Stmt::r#let("héllo, 世界 🦀", Expr::Lit(0)));
  let Stmt::Let(name, _) = &*s else { unreachable!() };
  assert_eq!(&**name, "héllo, 世界 🦀");
  assert_eq!(name.chars().count(), 11);
  assert_eq!(&**Region::new(Pair::<u8, Near<str>>::new(0, "")).second, "");
}

fn inst_expr() -> Region<Expr> {
  Region::new(Expr::neg(Expr::Lit(2)))
}

#[test]
fn constructors_box_pointer_fields() {
  // A `Near<X>` field takes a builder of `X`: a value, a copy, a constructor, or an `Old`.
  let r = Region::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::add(Expr::Lit(2), &*inst_expr()))));
  assert_eq!(show(&r), "(1 + -(2 + -2))");
  let c = Region::new(Chain::link(1, Chain::link(2, Chain::End)));
  assert!(
    matches!(&*c, Chain::Link(1, n) if matches!(&**n, Chain::Link(2, e) if **e == Chain::End))
  );
  let p = Region::new(Pair::new(Color::Red, Tree::node(Tree::Leaf(1u8), Tree::leaf(2))));
  assert!(matches!(&*p.second, Tree::Node(a, _) if **a == Tree::Leaf(1)));
  // An alias of `Near<X>` takes a builder of the pointer.
  let a = Region::new(Aliased::new(Near::new(5)));
  assert_eq!(*a.p, 5);
  // `Pair<_, Near<str>>`: `second` is a pointer to a pointer.
  let s = Region::new(Pair::<u8, Near<str>>::new(1, "text"));
  assert_eq!(&**s.second, "text");
}

#[test]
fn enums_generics_and_nesting() {
  fn sum(tree: &Tree<u16>) -> u16 {
    match tree {
      Tree::Leaf(x) => *x,
      Tree::Node(l, r) => sum(l) + sum(r),
    }
  }
  let stmt =
    Region::new(Stmt::block((Stmt::Nop, Stmt::int_lit(3), Stmt::block([Stmt::IntLit(4)], 1)), 2));
  let Stmt::Block { body, tag } = &*stmt else { unreachable!() };
  assert_eq!((&body[0], &body[1], *tag), (&Stmt::Nop, &Stmt::IntLit(3), 2));
  assert!(matches!(&body[2], Stmt::Block { body, tag: 1 } if body[0] == Stmt::IntLit(4)));

  let tree = Region::new(Tree::node(Tree::Leaf(1u16), Tree::node(Tree::leaf(2), Tree::Leaf(3))));
  assert_eq!(sum(&tree), 6);
  let pair =
    Region::new(Pair::new(Color::Blue, Tree::leaf(Wrap::new(1, [Color::Red, Color::Green]))));
  assert_eq!(pair.first, Color::Blue);
  assert!(matches!(&*pair.second, Tree::Leaf(Wrap(1, c)) if c[..] == [Color::Red, Color::Green]));

  let arr = Region::new(Arr::<2>::new([1, 2], [[3, 4], [5, 6]]));
  assert_eq!((arr.items, &arr.rest[..]), ([1, 2], &[[3, 4], [5, 6]][..]));
  let shapes = Region::new(Lst::new([Shape::Dot, Shape::Circle(3), Shape::Rect { w: 1, h: 2 }]));
  assert_eq!(shapes.items[2], Shape::Rect { w: 1, h: 2 });
}

#[test]
fn layout() {
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
  // `[cap][len][elements]` after the pointer; strings are `[len][bytes]`.
  let l = Region::new(Near::from_iter([7u32, 8]));
  let words =
    |b: &[u8]| b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
  assert_eq!(words(&l.to_bytes()), [8, 2, 2, 7, 8]);
  let s = Region::new(Lst::<Near<str>>::new(["ab"]));
  assert_eq!(s.to_bytes()[..16], [8, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0]);
  // A pushed list has spare capacity; a copy is compact again.
  let mut l = l;
  l.edit().push(9);
  assert_eq!(l.byte_len(), 20 + 8 + 4 * 4);
  assert_eq!(words(&l.to_bytes()), [8, 3, 3, 7, 8, 9]);
}

#[test]
fn near_traits() {
  let r = Region::new(Pair::<Inst, Near<[Inst]>>::new(
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
  assert_eq!(
    format!("{:?}", &**r.second),
    "[Inst { op: 1, args: [2] }, Inst { op: 1, args: [3] }]"
  );
  let total: u32 = (&*r.second).into_iter().map(|i| i.args[0]).sum();
  assert_eq!(total, 5);
  let s = Region::new(Pair::<u8, Near<str>>::new(0, "x"));
  assert_eq!(
    (format!("{:?}", s.second), hash(&*s.second) == hash(&*s.second)),
    ("\"x\"".into(), true)
  );
  let e = Region::new(Expr::neg(Expr::Lit(1)));
  let Expr::Neg(x) = &*e else { unreachable!() };
  assert_eq!((format!("{x:?}"), x == x), ("Lit(1)".into(), true));
}

#[test]
fn region_traits() {
  let f = func();
  let g = f.clone();
  assert_eq!(f, g);
  assert_eq!(format!("{f:?}"), format!("{:?}", *g));
  assert!(format!("{f:?}").starts_with("Func { name: \"main\""));
  // The clone is independent.
  let mut h = f.clone();
  *h.edit().flag() = false;
  h.edit().name().set("other");
  assert_eq!((&*f.name, f.flag, &*h.name, h.flag), ("main", true, "other", false));
  assert_ne!(f, h);
  drop(f);
  assert_eq!(*g.entry, *h.entry);
}

// ---------------------------------------------------------------------------------------------
// Handles

#[test]
fn struct_handles() {
  let mut f = func();
  let before = f.byte_len();
  let root = f.edit();
  *root.flag() = false;
  let entry = root.entry(); // the handle of a `Near<Block>` is the block's
  *entry.id() = 9;
  for i in 0..entry.insts().len() {
    let inst = entry.insts().at(i);
    *inst.op() += 10;
    inst.args().reverse();
  }
  assert_eq!(entry.id, 9); // handles read like the value
  assert_eq!(f.byte_len(), before); // nothing was allocated
  assert_eq!((f.flag, f.entry.id), (false, 9));
  assert_eq!((f.entry.insts[0].op, &f.entry.insts[0].args[..]), (11, &[30, 20, 10][..]));

  // Tuple fields, colliding names, and visibility.
  let mut w = Region::new(Wrap::new(1, [Color::Red]));
  *w.edit()._0() = 2;
  w.edit()._1().push(Color::Blue);
  assert_eq!((w.0, &w.1[..]), (2, &[Color::Red, Color::Blue][..]));
  let mut n = Region::new(Names::new(1, "a", 2));
  *n.edit().set_() = 3;
  n.edit().replace_().set("b");
  *n.edit().r#as_mut() = 4;
  assert_eq!((n.set, &*n.replace, n.r#as_mut), (3, "b", 4));
  let mut m = Region::new(private::Mixed::new(1, "h"));
  *m.edit().open() = 2;
  private::rename(m.edit(), "renamed");
  assert_eq!((m.open, m.hidden()), (2, "renamed"));

  // Handles of generic types.
  let mut p = Region::new(Pair::new(1u16, Lst::new([1u8])));
  *p.edit().first() = 2;
  p.edit().second().items().push(2);
  assert_eq!((p.first, &p.second.items[..]), (2, &[1, 2][..]));
  let mut a = Region::new(Arr::<2>::new([1, 2], ()));
  a.edit().items()[1] = 5;
  a.edit().rest().push([7, 8]);
  assert_eq!((a.items, &a.rest[..]), ([1, 5], &[[7, 8]][..]));
}

#[test]
fn plain_handles_deref_mut() {
  let mut r = Region::new(Rgb { r: 1, g: 2, b: 3, c: Color::Red });
  **r.edit() = Rgb { r: 9, g: 9, b: 9, c: Color::Green }; // `Copy`: the handle is assignable
  r.edit().c = Color::Blue;
  *r.edit().r() = 4;
  assert_eq!(*r, Rgb { r: 4, g: 9, b: 9, c: Color::Blue });
  let mut s = Region::new(Shape::Dot);
  **s.edit() = Shape::Circle(1);
  if let ShapeMut::Circle(radius) = s.edit().as_mut() {
    *radius += 1;
  }
  assert_eq!(*s, Shape::Circle(2));
  **s.edit() = Shape::Rect { w: 1, h: 2 };
  if let ShapeMut::Rect(rect) = s.edit().as_mut() {
    *rect.w() = 3;
    *rect.h() += 1;
  }
  assert_eq!(*s, Shape::Rect { w: 3, h: 3 });
  assert!(matches!(s.edit().set(Shape::Dot), ()));
  assert!(matches!(s.edit().as_mut(), ShapeMut::Dot));
  let mut c = Region::new(Wrap::new(0, [Color::Red, Color::Blue]));
  c.edit()._1().fill(Color::Green); // a list of plain data is a real `&mut [T]`
  assert_eq!(c.1[..], [Color::Green; 2]);
  let mut u = Region::new(Color::Red);
  u.edit().set(Color::Blue);
  **u.edit() = Color::Green;
  assert_eq!(*u, Color::Green);
}

#[test]
fn enum_handles() {
  let mut s =
    Region::new(Stmt::block((Stmt::r#let("x", Expr::Lit(1)), Stmt::r#return(None), Stmt::Nop), 7));
  let StmtMut::Block(block) = s.edit().as_mut() else { unreachable!() };
  *block.tag() = 8;
  let body = block.body();
  if let StmtMut::Let(binding) = body.at(0).as_mut() {
    binding._0().set("y");
    binding._1().set(Expr::neg(Expr::Lit(2)));
  }
  if let StmtMut::Return(value) = body.at(1).as_mut() {
    value.insert(Expr::Lit(3));
  }
  assert!(matches!(body.at(2).as_mut(), StmtMut::Nop));
  body.at(2).set(Stmt::IntLit(5));
  let Stmt::Block { body, tag } = &*s else { unreachable!() };
  assert_eq!(*tag, 8);
  assert!(matches!(&body[0], Stmt::Let(n, e) if &**n == "y" && eval(e) == -2));
  assert!(matches!(&body[1], Stmt::Return { value: Some(e) } if eval(e) == 3));
  assert_eq!(body[2], Stmt::IntLit(5));

  // Generic enums.
  let mut t = Region::new(Tree::node(Tree::Leaf(1u32), Tree::Leaf(2)));
  if let TreeMut::Node(kids) = t.edit().as_mut() {
    if let TreeMut::Leaf(x) = kids._0().as_mut() {
      *x = 10;
    }
    kids._1().set(Tree::node(Tree::Leaf(3), Tree::Leaf(4)));
  }
  let Tree::Node(a, b) = &*t else { unreachable!() };
  assert!(**a == Tree::Leaf(10) && matches!(&**b, Tree::Node(c, _) if **c == Tree::Leaf(3)));
}

#[test]
fn list_handles() {
  let mut b = Region::new(Block::new(0, (), None));
  let insts = b.edit().insts();
  // `push` and `insert` return the new element's handle.
  insts.push(Inst::new(1, ())).args().push(10);
  *insts.push(Inst::new(3, [30])).op() = 4;
  insts.insert(1, Inst::new(2, [20, 21])).args().push(22);
  insts.insert(0, Inst::new(0, ()));
  insts.insert(4, Inst::new(5, [50]));
  assert_eq!(insts.len(), 5);
  assert_eq!(insts.iter().map(|i| i.op).collect::<Vec<_>>(), [0, 1, 2, 4, 5]);
  insts.swap(0, 4);
  insts.swap(2, 2);
  insts.remove(1);
  assert_eq!(ops(&b), [5, 2, 4, 0]);
  assert_eq!(args(&b), [vec![50], vec![20, 21, 22], vec![30], vec![]]);
  let insts = b.edit().insts();
  insts.retain(|i| i.op % 2 == 0);
  assert_eq!(insts.iter().map(|i| i.op).collect::<Vec<_>>(), [2, 4, 0]);
  insts.truncate(5);
  insts.truncate(2);
  insts.at(1).args().push(31);
  assert_eq!(args(&b), [vec![20, 21, 22], vec![30, 31]]);
  // Growth from empty and from a compact copy.
  let mut b = Region::new(&*b);
  for i in 0..100u8 {
    b.edit().insts().push(Inst::new(i, Near::from_iter(0..u32::from(i % 7))));
  }
  assert_eq!(b.insts.len(), 102);
  assert!(
    b.insts[2..].iter().enumerate().all(|(i, x)| x.op as usize == i && x.args.len() == i % 7)
  );
  // `set` and `replace` act on the pointer.
  b.edit().insts().set(());
  assert!(b.insts.is_empty());
  b.edit().insts().set([Inst::new(1, [1])]);
  b.edit().insts().replace(|old| (Inst::new(0, ()), old.map(|l| &l[0])));
  assert_eq!(ops(&b), [0, 1]);
  // Lists of lists, of pointers and of strings.
  let mut l = Region::new(Lst::<Near<[u8]>>::new(([1u8], ())));
  l.edit().items().at(1).push(2);
  l.edit().items().push([3]).push(4);
  l.edit().items().insert(0, ()).push(0);
  assert_eq!(
    l.items.iter().map(|x| x.to_vec()).collect::<Vec<_>>(),
    [vec![0], vec![1], vec![2], vec![3, 4]]
  );
  let mut p = Region::new(Lst::<Near<Expr>>::new([Near::new(Expr::Lit(1))]));
  p.edit().items().push(Near::new(Expr::Lit(2))).set(Expr::Lit(3));
  p.edit().items().at(0).replace(Expr::neg);
  assert_eq!(p.items.iter().map(|e| eval(e)).collect::<Vec<_>>(), [-1, 3]);
  let mut s = Region::new(Lst::<Near<str>>::new(["b"]));
  s.edit().items().insert(0, "a");
  s.edit().items().push("c").set("C");
  assert_eq!(s.items.iter().map(|x| &**x).collect::<String>(), "abC");
}

#[test]
fn list_index_panics() {
  let mut r = Region::new(Near::from_iter([1u32, 2]));
  assert!(catch_unwind(AssertUnwindSafe(|| *r.edit().at(2) = 0)).is_err());
  assert!(catch_unwind(AssertUnwindSafe(|| r.edit().remove(2))).is_err());
  assert!(catch_unwind(AssertUnwindSafe(|| *r.edit().insert(3, 0) = 0)).is_err());
  assert!(catch_unwind(AssertUnwindSafe(|| r.edit().swap(0, 2))).is_err());
  assert_eq!((r[..].to_vec(), r.byte_len()), (vec![1, 2], 4 + 8 + 8));
}

#[test]
fn option_handles() {
  let mut b = Region::new(Block::new(0, (), None));
  assert!(b.edit().next().as_mut().is_none());
  let next = b.edit().next();
  next.insert(Block::new(1, (), None)).insts().push(Inst::new(1, ()));
  assert!(next.is_some()); // reads as `Option<Near<Block>>`
  *next.as_mut().unwrap().id() = 2;
  next.as_mut().unwrap().next().insert(Block::new(3, (), None));
  assert_eq!(b.next.as_ref().map(|n| (n.id, ops(n))), Some((2, vec![1])));
  assert_eq!(b.next.as_ref().unwrap().next.as_ref().unwrap().id, 3);
  // `insert` over a target replaces it.
  b.edit().next().insert(Block::new(4, (), None));
  assert_eq!(b.next.as_ref().map(|n| (n.id, n.next.is_none())), Some((4, true)));
  b.edit().next().set(None);
  assert!(b.next.is_none());
  b.edit().next().set(Near::new(Block::new(5, (), None)));
  // `replace`: push a block in front, the old one moves.
  b.edit().next().replace(|old| Near::new(Block::new(6, (), old)));
  let n = b.next.as_ref().unwrap();
  assert_eq!((n.id, n.next.as_ref().unwrap().id), (6, 5));
  // Optional pointers to plain data are `&mut T`.
  let mut o = Region::new(Pair::new(0u8, None::<Near<u64>>));
  *o.edit().second().insert(1) += 1;
  assert_eq!(o.second.as_deref(), Some(&2));
}

// ---------------------------------------------------------------------------------------------
// Replacing

#[test]
fn set_and_replace() {
  let mut f = func();
  // `set` on a struct, from a value built elsewhere and from a copy.
  let other = func();
  f.edit().entry().set(Block::new(5, &other.entry.insts, None));
  assert_eq!((f.entry.id, ops(&f.entry)), (5, vec![1, 2]));
  f.edit().set(&*other);
  assert_eq!(*f, *other);
  // `replace` on a struct: the old value is moved, field by field through `map`.
  f.edit().entry().replace(|old| Block::new(7, old.map(|b| &b.insts), None));
  assert_eq!((f.entry.id, args(&f.entry)), (7, vec![vec![10, 20, 30], vec![]]));
  f.edit().name().replace(|old| old); // identity
  assert_eq!(&*f.name, "main");
}

#[test]
fn wrap_hoist_and_moves() {
  fn inner(e: &Expr) -> &Expr {
    match e {
      Expr::Neg(x) => x,
      e => e,
    }
  }
  let mut e = Region::new(Expr::add(Expr::Lit(2), Expr::Lit(5)));
  // x -> -x: the old value moves under the new node.
  let before = e.byte_len();
  e.edit().replace(Expr::neg);
  assert_eq!(show(&e), "-(2 + 5)");
  assert!(e.byte_len() - before <= 3 * size_of::<Expr>() + 8); // no copy of the children
  // -x -> x: the child moves up.
  e.edit().replace(|old| old.map(inner));
  assert_eq!(show(&e), "(2 + 5)");
  // (a + b) -> (b + -a), deep inside: children trade places, nothing is copied.
  let mut e = Region::new(Expr::neg(Expr::add(Expr::Lit(2), Expr::Lit(5))));
  let before = e.byte_len();
  let ExprMut::Neg(x) = e.edit().as_mut() else { unreachable!() };
  x.replace(|old| {
    let b = old.map(|e| match e {
      Expr::Add(_, b) => &**b,
      e => e,
    });
    Expr::add(b, Expr::Lit(0))
  });
  assert_eq!(show(&e), "-(5 + 0)");
  assert!(e.byte_len() - before <= 4 * size_of::<Expr>() + 8);
  // Many wraps: the buffer reallocates under the moves.
  let mut r = Region::new(Expr::Lit(1));
  for i in 0..200 {
    r.edit().replace(|old| Expr::add(old, Expr::Lit(i)));
  }
  assert_eq!(eval(&r), 1 + (0..200).sum::<i64>());
  let copy = Region::new(&*r);
  assert_eq!(eval(&copy), eval(&r));
  assert!(copy.byte_len() < r.byte_len());
}

#[test]
fn old_elsewhere_is_a_copy() {
  static FIVE: Expr = Expr::Lit(5);
  let mut a = Region::new(Expr::neg(Expr::Lit(7)));
  let mut b = Region::new(Expr::Lit(0));
  // A part that is not inside the old value is copied.
  a.edit().replace(|old| Expr::add(old.map(|_| &FIVE), Expr::Lit(1)));
  assert_eq!(show(&a), "(5 + 1)");
  // An `Old` taken out of its closure keeps its region borrowed; elsewhere it is a copy.
  let mut stash = None;
  a.edit().replace(|old| {
    stash = Some(old.map(|e| match e {
      Expr::Add(l, _) => &**l,
      e => e,
    }));
    Expr::neg(Expr::Lit(9))
  });
  b.edit().replace(|old| Expr::add(old, stash.take().unwrap()));
  assert_eq!((show(&a), show(&b)), ("-9".into(), "(0 + 5)".into()));
  // The whole old value, while the other replacement is still running.
  a.edit().replace(|old_a| {
    b.edit().replace(|old_b| Expr::add(old_b, old_a));
    Expr::Lit(2)
  });
  assert_eq!((show(&a), show(&b)), ("2".into(), "((0 + 5) + -9)".into()));
  // Into a new region.
  let mut copy = None;
  b.edit().replace(|old| {
    copy = Some(Region::new(old));
    Expr::Lit(3)
  });
  assert_eq!((show(&b), show(&copy.unwrap())), ("3".into(), "((0 + 5) + -9)".into()));
}

/// Anything `impl Emit` can be passed on generically.
fn wrap_twice(e: impl Emit<Expr>) -> impl Emit<Expr> {
  Expr::neg(Expr::neg(e))
}

#[test]
fn builders_compose() {
  let mut r = Region::new(wrap_twice(Expr::Lit(4)));
  assert_eq!(show(&r), "--4");
  r.edit().replace(wrap_twice);
  assert_eq!(show(&r), "----4");
}

// ---------------------------------------------------------------------------------------------
// Panics

/// Writes nothing and panics.
struct Boom;

// SAFETY: never returns.
unsafe impl<T: Flat> Emit<T> for Boom {
  unsafe fn emit(self, _: &mut nearest::__private::Sink, _: usize) {
    panic!("boom")
  }
}

fn fails(f: impl FnOnce()) {
  assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
}

#[test]
fn a_panic_leaves_the_old_value() {
  let mut r = Region::new(Expr::neg(Expr::Lit(7)));
  fails(|| r.edit().replace(|old| Expr::add(old, Boom)));
  assert_eq!(show(&r), "-7");
  fails(|| r.edit().replace(|_| -> &'static Expr { panic!() }));
  assert_eq!(show(&r), "-7");
  fails(|| r.edit().replace(|old| Expr::add(Expr::Lit(1), old.map(|_| -> &Expr { panic!() }))));
  assert_eq!(show(&r), "-7");
  fails(|| r.edit().set(Expr::add(Expr::Lit(1), Boom)));
  assert_eq!(show(&r), "-7");
  let ExprMut::Neg(x) = r.edit().as_mut() else { unreachable!() };
  x.set(Expr::Lit(8));
  assert_eq!(show(&r), "-8");

  let mut f = func();
  fails(|| f.edit().name().set(Boom));
  fails(|| f.edit().entry().next().set(Near::new(Block::new(1, (), Boom))));
  fails(|| {
    f.edit().entry().next().insert(Block::new(1, (Inst::new(0, ()), Boom), None));
  });
  assert_eq!(*f, *func());
}

#[test]
fn a_panic_in_a_list_operation_leaves_a_valid_list() {
  let mut b = Region::new(Block::new(0, (Inst::new(0, [0]), Inst::new(1, [1])), None));
  let expected = args(&b);
  for k in 0..3 {
    fails(|| {
      b.edit().insts().insert(k, Inst::new(9, (1, Boom)));
    });
    assert_eq!(args(&b), expected);
  }
  fails(|| {
    b.edit().insts().push(Inst::new(9, Boom));
  });
  assert_eq!(args(&b), expected);
  // A full list grows before the element is written; the panic leaves it grown and intact.
  for i in 2..4 {
    b.edit().insts().push(Inst::new(i, [u32::from(i)]));
  }
  fails(|| {
    b.edit().insts().push(Inst::new(9, Boom));
  });
  fails(|| {
    b.edit().insts().insert(0, Inst::new(9, Boom));
  });
  assert_eq!(ops(&b), [0, 1, 2, 3]);
  b.edit().insts().push(Inst::new(4, [4]));
  assert_eq!(args(&b), [vec![0], vec![1], vec![2], vec![3], vec![4]]);
  // `retain` keeps the unvisited elements.
  let mut seen = 0;
  fails(|| {
    b.edit().insts().retain(|i| {
      seen += 1;
      assert!(i.op < 3, "stop");
      i.op % 2 == 1
    });
  });
  assert_eq!((seen, ops(&b)), (4, vec![1, 3, 4]));
  assert_eq!(args(&b), [vec![1], vec![3], vec![4]]);
  assert_eq!(*Region::new(&*b), *b);
}

#[test]
fn the_size_limit_leaves_a_valid_region() {
  // The real limit is 2 GiB; check that a request past it fails before changing anything.
  let mut r = Region::new(Lst::new([1u64]));
  let len = r.byte_len();
  fails(|| r.edit().items().set(Near::from_iter(std::iter::repeat_n(0u64, usize::MAX / 16))));
  assert!(r.byte_len() <= len + 8); // only the slot the old value moved to
  assert_eq!(r.items[..], [1]);
}

// ---------------------------------------------------------------------------------------------
// Bytes and copies

#[test]
fn bytes_round_trip() {
  let mut r = Region::<[Pad; 2]>::new([Pad { a: 1, b: 2 }, Pad { a: 3, b: 4 }]);
  r.edit()[0] = Pad { a: 9, b: 9 }; // may leave padding uninitialized in the buffer
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
fn compaction() {
  let mut b = Region::new(Block::new(0, (), None));
  for i in 0..20u8 {
    b.edit().insts().push(Inst::new(i, [u32::from(i)]));
    b.edit().insts().at(0).args().push(u32::from(i));
    b.edit().next().insert(Block::new(u32::from(i), (), None));
  }
  let c = Region::new(&*b);
  assert!(c.byte_len() * 2 < b.byte_len());
  assert_eq!(*c, *b);
  assert_eq!(Region::new(&*c).byte_len(), c.byte_len());
  assert_eq!(c.to_bytes().len(), c.byte_len());
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
    // Pushing to a list whose header says `cap == len` after an over-allocation.
    let mut b = b;
    b.edit().insts().push(Inst::new(0, ()));
    assert_eq!(b.insts.len(), n as usize + 1);
  }
}

// ---------------------------------------------------------------------------------------------
// Model tests: long random edit sequences, compared against plain Rust values.

#[flat]
struct Doc {
  id: u32,
  name: Near<str>,
  items: Near<[Item]>,
  next: Option<Near<Doc>>,
}

#[flat]
struct Item {
  tag: u8,
  vals: Near<[u32]>,
  label: Option<Near<Label>>,
}

#[flat]
struct Label {
  n: u64,
  text: Near<str>,
}

#[derive(Clone, Debug, PartialEq)]
struct MDoc {
  id: u32,
  name: String,
  items: Vec<MItem>,
  next: Option<Box<MDoc>>,
}

#[derive(Clone, Debug, PartialEq)]
struct MItem {
  tag: u8,
  vals: Vec<u32>,
  label: Option<(u64, String)>,
}

fn doc_model(d: &Doc) -> MDoc {
  MDoc {
    id: d.id,
    name: d.name.to_string(),
    items: d.items.iter().map(item_model).collect(),
    next: d.next.as_ref().map(|n| Box::new(doc_model(n))),
  }
}

fn item_model(i: &Item) -> MItem {
  MItem {
    tag: i.tag,
    vals: i.vals.to_vec(),
    label: i.label.as_ref().map(|l| (l.n, l.text.to_string())),
  }
}

/// Writes `m` with a handle: the item, then its label.
fn put_item(h: &mut Edit<Item>, m: &MItem) {
  h.set(Item::new(m.tag, &m.vals[..], None));
  if let Some((n, t)) = &m.label {
    h.label().insert(Label::new(*n, t.as_str()));
  }
}

struct Rng(u64);

impl Rng {
  fn below(&mut self, n: usize) -> usize {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    (self.0 % n.max(1) as u64) as usize
  }
}

/// The edit at `depth` levels down the `next` chain of the model and of the region.
fn walk(m: &mut MDoc, depth: usize) -> &mut MDoc {
  if depth == 0 { m } else { walk(m.next.as_mut().unwrap(), depth - 1) }
}

fn walk_edit(e: &mut Edit<Doc>, depth: usize) -> &mut Edit<Doc> {
  if depth == 0 { e } else { walk_edit(e.next().as_mut().unwrap(), depth - 1) }
}

fn chain_len(m: &MDoc) -> usize {
  1 + m.next.as_ref().map_or(0, |n| chain_len(n))
}

#[test]
fn random_edits_match_a_model() {
  let steps: u32 = if cfg!(miri) { 120 } else { 4000 };
  for seed in [0x2545_f491_u64, 0x9e37_79b9_7f4a_7c15] {
    let mut rng = Rng(seed);
    let mut m = MDoc { id: 0, name: "root".into(), items: vec![], next: None };
    let mut r = Region::new(Doc::new(0, "root", (), None));
    let other = Region::new(Item::new(7, [7, 7], Near::new(Label::new(70, "other"))));
    for step in 0..steps {
      let depth = rng.below(chain_len(&m));
      let (dm, de) = (walk(&mut m, depth), walk_edit(r.edit(), depth));
      let len = dm.items.len();
      let k = rng.below(len);
      let s = step;
      match rng.below(20) {
        0 => {
          dm.id = s;
          *de.id() = s;
        }
        1 => {
          dm.name = format!("n{s}");
          de.name().set(dm.name.as_str());
        }
        2 => {
          let it = MItem { tag: s as u8, vals: (0..s % 4).collect(), label: None };
          put_item(de.items().push(Item::new(0, (), None)), &it);
          dm.items.push(it);
        }
        3 => {
          let k = rng.below(len + 1);
          let it = MItem { tag: s as u8, vals: vec![s], label: Some((u64::from(s), "l".into())) };
          put_item(de.items().insert(k, Item::new(0, (), None)), &it);
          dm.items.insert(k, it);
        }
        4 if len > 0 => {
          dm.items.remove(k);
          de.items().remove(k);
        }
        5 if len > 0 => {
          let j = rng.below(len);
          dm.items.swap(k, j);
          de.items().swap(k, j);
        }
        6 => {
          let n = rng.below(len + 2);
          dm.items.truncate(n);
          de.items().truncate(n);
        }
        7 => {
          let p = rng.below(3) as u8 + 2;
          dm.items.retain(|i| i.tag % p != 0);
          de.items().retain(|i| i.tag % p != 0);
        }
        8 if len > 0 => {
          dm.items[k].vals.push(s);
          de.items().at(k).vals().push(s);
        }
        9 if len > 0 => {
          dm.items[k].vals.sort_unstable_by(|a, b| b.cmp(a));
          de.items().at(k).vals().sort_unstable_by(|a, b| b.cmp(a));
        }
        10 if len > 0 => {
          let v: Vec<u32> = (0..rng.below(5) as u32).map(|x| x * s).collect();
          de.items().at(k).vals().set(&v[..]);
          dm.items[k].vals = v;
        }
        11 if len > 0 => {
          dm.items[k].label = Some((u64::from(s), format!("t{s}")));
          de.items().at(k).label().insert(Label::new(u64::from(s), format!("t{s}").as_str()));
        }
        12 if len > 0 => {
          if let Some(l) = &mut dm.items[k].label {
            l.0 += 1;
          }
          if let Some(l) = de.items().at(k).label().as_mut() {
            *l.n() += 1;
          }
          if rng.below(2) == 0 {
            dm.items[k].label = None;
            de.items().at(k).label().set(None);
          }
        }
        13 if len > 0 => {
          // Replace an item by a new one that moves the old one's values.
          dm.items[k].tag = dm.items[k].tag.wrapping_add(1);
          dm.items[k].label = None;
          let tag = dm.items[k].tag;
          de.items().at(k).replace(|old| Item::new(tag, old.map(|i| &i.vals), None));
        }
        14 if len > 0 => {
          // Keep only item `k`, moved.
          dm.items = vec![dm.items[k].clone()];
          de.items().replace(|old| [old.map(|l| &l[k])]);
        }
        15 if len > 0 => {
          // A copy from another region.
          dm.items[k] = item_model(&other);
          de.items().at(k).set(&*other);
        }
        16 => {
          // Wrap: the doc moves into `next` of a new one.
          let old =
            std::mem::replace(dm, MDoc { id: s, name: "w".into(), items: vec![], next: None });
          dm.next = Some(Box::new(old));
          de.replace(|old| Doc::new(s, "w", (), Near::new(old)));
        }
        17 if dm.next.is_some() => {
          // Hoist: `next` takes its parent's place.
          *dm = *dm.next.take().unwrap();
          de.replace(|old| old.map(|d| &**d.next.as_ref().unwrap()));
        }
        18 => {
          dm.next = None;
          de.next().set(None);
        }
        _ => {
          r = match step % 3 {
            0 => Region::new(&*r),
            1 => r.clone(),
            // SAFETY: from `to_bytes` of the same type.
            _ => unsafe { Region::from_bytes(&r.to_bytes()) },
          };
        }
      }
      assert_eq!(doc_model(&r), m, "step {step}");
    }
    let compact = Region::new(&*r);
    assert_eq!(doc_model(&compact), m);
    assert!(compact.byte_len() <= r.byte_len());
  }
}

#[test]
fn random_expression_rewrites_match_a_model() {
  #[derive(Clone, Debug, PartialEq)]
  enum M {
    Lit(i64),
    Neg(Box<M>),
    Add(Box<M>, Box<M>),
  }
  fn model(e: &Expr) -> M {
    match e {
      Expr::Lit(n) => M::Lit(*n),
      Expr::Neg(x) => M::Neg(Box::new(model(x))),
      Expr::Add(a, b) => M::Add(Box::new(model(a)), Box::new(model(b))),
    }
  }
  let steps: u32 = if cfg!(miri) { 150 } else { 5000 };
  let mut rng = Rng(0x1234_5678);
  let mut m = M::Lit(0);
  let mut r = Region::new(Expr::Lit(0));
  for step in 0..steps {
    let s = i64::from(step);
    // Pick a node by a random path.
    let mut path = Vec::new();
    {
      let mut cur = &m;
      while rng.below(3) != 0 {
        match cur {
          M::Lit(_) => break,
          M::Neg(x) => {
            path.push(0);
            cur = x;
          }
          M::Add(a, b) => {
            let side = rng.below(2);
            path.push(side);
            cur = if side == 0 { a } else { b };
          }
        }
      }
    }
    let mut mm = &mut m;
    for &side in &path {
      mm = match mm {
        M::Neg(x) => x,
        M::Add(a, b) => {
          if side == 0 {
            a
          } else {
            b
          }
        }
        M::Lit(_) => unreachable!(),
      };
    }
    let mut e = r.edit();
    for &side in &path {
      e = match e.as_mut() {
        ExprMut::Neg(x) => x,
        ExprMut::Add(args) => {
          if side == 0 {
            args._0()
          } else {
            args._1()
          }
        }
        ExprMut::Lit(_) => unreachable!(),
      };
    }
    match rng.below(7) {
      0 => {
        *mm = M::Neg(Box::new(mm.clone()));
        e.replace(Expr::neg);
      }
      1 => {
        *mm = M::Add(Box::new(mm.clone()), Box::new(M::Lit(s)));
        e.replace(|old| Expr::add(old, Expr::Lit(s)));
      }
      2 => {
        *mm = M::Add(Box::new(M::Lit(s)), Box::new(mm.clone()));
        e.replace(|old| Expr::add(Expr::Lit(s), old));
      }
      3 => {
        // Hoist a child.
        let side = rng.below(2);
        let child = match mm {
          M::Lit(_) => None,
          M::Neg(x) => Some((**x).clone()),
          M::Add(a, b) => Some(if side == 0 { (**a).clone() } else { (**b).clone() }),
        };
        if let Some(c) = child {
          *mm = c;
          e.replace(|old| {
            old.map(|e| match e {
              Expr::Lit(_) => e,
              Expr::Neg(x) => x,
              Expr::Add(a, b) => {
                if side == 0 {
                  a
                } else {
                  b
                }
              }
            })
          });
        }
      }
      4 => {
        if let M::Lit(n) = mm {
          *n += s;
        }
        if let ExprMut::Lit(n) = e.as_mut() {
          *n += s;
        }
      }
      5 => {
        *mm = M::Lit(s);
        e.set(Expr::Lit(s));
      }
      _ => {
        if step % 2 == 0 {
          r = Region::new(&*r);
        } else {
          // SAFETY: from `to_bytes` of the same type.
          r = unsafe { Region::from_bytes(&r.to_bytes()) };
        }
      }
    }
    assert_eq!(model(&r), m, "step {step}");
  }
}
