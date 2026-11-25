use std::{
  collections::hash_map::DefaultHasher,
  hash::{Hash, Hasher},
  num::NonZero,
  panic::{AssertUnwindSafe, catch_unwind},
};

use nearest::{Edit, Emit, Flat, Near, flat};

/// A region in the storage under test, `Store`: the heap (`integration.rs`), inline
/// (`inline.rs`) or a storage written here (`user_storage.rs`).
type Region<T> = nearest::Region<T, Store>;

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

// `Debug` and `PartialEq` read the pointees, through `Deref`, which reads with `puns`: a parameter
// behind a pointer is `Pun`, which `#[flat]` states on the type.
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
  Region::new_in(Store::default(), Func::new(
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
  assert_eq!(*Region::new_in(Store::default(), 7u32), 7);
  assert_eq!(*Region::new_in(Store::default(), Pad { a: 1, b: 2 }), Pad { a: 1, b: 2 });
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
  let r = Region::new_in(Store::default(), &p);
  assert_eq!(*r, p);
  let q = Region::new_in(Store::default(), Prims::new(
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
    *Region::new_in(Store::default(), Rgb { r: 1, g: 2, b: 3, c: Color::Blue }),
    Rgb { r: 1, g: 2, b: 3, c: Color::Blue }
  );
  let lists = Region::new_in(Store::default(), Lst::<Prims>::new([&p, &p]));
  assert_eq!(lists.items[1], p);
}

/// `XInit` names the fields of a struct, in any order, and writes what `X::new` writes.
#[test]
fn init_structs_name_the_fields() {
  let positional = Region::new_in(Store::default(), Func::new("f", Block::new(1, [Inst::new(2, [3])], None), true));
  let named = Region::new_in(Store::default(), FuncInit {
    flag: true,
    entry: BlockInit { next: None, insts: [InstInit { args: [3], op: 2 }], id: 1 },
    name: "f",
  });
  assert_eq!(positional.as_bytes(), named.as_bytes());
}

#[test]
fn every_emit_source() {
  let other = func();
  let v = [5u32, 6, 7];
  let r = Region::new_in(Store::default(), Block::new(
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
  let inst = Region::new_in(Store::default(), Inst::new(8, [9]));
  let mixed = Region::new_in(Store::default(), Block::new(
    0,
    (Inst::new(1, ()), &*inst, Inst::new(3, Near::from_iter(None::<u32>))),
    None,
  ));
  assert_eq!(ops(&mixed), [1, 8, 3]);
  let t12 = Region::new_in(Store::default(), Inst::new(0, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)));
  assert_eq!(t12.args.iter().sum::<u32>(), 78);
  let nested = Region::new_in(Store::default(), Lst::<Near<[u8]>>::new(([1u8], (), [2, 3])));
  assert_eq!(nested.items.iter().map(|l| l.len()).collect::<Vec<_>>(), [1, 0, 2]);
  let ptrs =
    Region::new_in(Store::default(), Lst::<Near<Expr>>::new((Near::new(Expr::Lit(1)), Near::new(&*inst_expr()))));
  assert_eq!(ptrs.items.iter().map(|e| eval(e)).collect::<Vec<_>>(), [1, -2]);

  // `Near::new` writes `Option<Near<T>>`; `None` is a value.
  let e = Region::new_in(Store::default(), Stmt::r#return(Near::new(Expr::Lit(1))));
  assert!(matches!(&*e, Stmt::Return { value: Some(v) } if **v == Expr::Lit(1)));
  let e = Region::new_in(Store::default(), Stmt::r#return(None));
  assert_eq!(*e, Stmt::Return { value: None });
  // Keywords, including those reserved by the edition, become raw identifiers.
  assert_eq!(*Region::new_in(Store::default(), Stmt::r#gen(3)), Stmt::Gen(3));

  // Text, including non-ASCII.
  let s = Region::new_in(Store::default(), Stmt::r#let("héllo, 世界 🦀", Expr::Lit(0)));
  let Stmt::Let(name, _) = &*s else { unreachable!() };
  assert_eq!(&**name, "héllo, 世界 🦀");
  assert_eq!(name.chars().count(), 11);
  assert_eq!(&**Region::new_in(Store::default(), Pair::<u8, Near<str>>::new(0, "")).second, "");
}

fn inst_expr() -> Region<Expr> {
  Region::new_in(Store::default(), Expr::neg(Expr::Lit(2)))
}

type Bytes = [u8];
type Text = str;
type Sub = Expr;

/// Fields of plain data marked `#[flat(into)]`: their parameters take `impl Into<..>`.
#[flat]
#[derive(Debug, PartialEq)]
struct Sample {
  #[flat(into)]
  id: u64,
  #[flat(into)]
  ratio: f64,
  #[flat(into)]
  count: NonZero<u32>,
  name: Near<str>,
}

/// `into` with `rename`.
#[flat]
#[derive(Debug, PartialEq)]
enum Figure {
  #[flat(rename = "disc")]
  Circle(#[flat(into)] u32),
  Rect {
    #[flat(into)]
    w: u64,
    h: u64,
    next: Option<Near<Figure>>,
  },
}

#[test]
fn into_parameters() {
  let one = NonZero::new(1u8).unwrap();
  let s = Region::new_in(Store::default(), Sample::new(7u8, 1.5f32, one, "s"));
  assert_eq!((s.id, s.ratio, s.count.get(), &*s.name), (7, 1.5, 1, "s"));
  let r = Region::new_in(Store::default(), Figure::rect(2u16, 3, Near::new(Figure::disc(4u8))));
  assert!(matches!(&*r, Figure::Rect { w: 2, h: 3, next: Some(n) } if **n == Figure::Circle(4)));
  // The values themselves, a copy and handles work as before.
  let mut l = Region::new_in(Store::default(), Lst::<Sample>::new([Sample::new(1u64, 2.0, NonZero::<u32>::MIN, "x")]));
  let items = l.edit().items();
  items.push(&*s);
  items.at(0).set(Sample::new(3u32, 4.0f32, one, "y"));
  assert_eq!((l.items[0].id, &*l.items[0].name, &l.items[1]), (3, "y", &*s));
}

/// The pointee of a pointer field may be written through an alias.
#[flat]
struct ViaAlias {
  bytes: Near<Bytes>,
  text: Near<Text>,
  expr: Near<Sub>,
}

#[test]
fn constructors_box_pointer_fields() {
  let a = Region::new_in(Store::default(), ViaAlias::new([1, 2], "t", Expr::neg(Expr::Lit(3))));
  assert_eq!((&a.bytes[..], &*a.text, show(&a.expr)), (&[1, 2][..], "t", "-3".to_owned()));

  // A `Near<X>` field takes a builder of `X`: a value, a copy, a constructor, or an `Old`.
  let r = Region::new_in(Store::default(), Expr::add(Expr::Lit(1), Expr::neg(Expr::add(Expr::Lit(2), &*inst_expr()))));
  assert_eq!(show(&r), "(1 + -(2 + -2))");
  let c = Region::new_in(Store::default(), Chain::link(1, Chain::link(2, Chain::End)));
  assert!(
    matches!(&*c, Chain::Link(1, n) if matches!(&**n, Chain::Link(2, e) if **e == Chain::End))
  );
  let p = Region::new_in(Store::default(), Pair::new(Color::Red, Tree::node(Tree::Leaf(1u8), Tree::leaf(2))));
  assert!(matches!(&*p.second, Tree::Node(a, _) if **a == Tree::Leaf(1)));
  // An alias of `Near<X>` takes a builder of the pointer.
  let a = Region::new_in(Store::default(), Aliased::new(Near::new(5)));
  assert_eq!(*a.p, 5);
  // `Pair<_, Near<str>>`: `second` is a pointer to a pointer.
  let s = Region::new_in(Store::default(), Pair::<u8, Near<str>>::new(1, "text"));
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
    Region::new_in(Store::default(), Stmt::block((Stmt::Nop, Stmt::int_lit(3), Stmt::block([Stmt::IntLit(4)], 1)), 2));
  let Stmt::Block { body, tag } = &*stmt else { unreachable!() };
  assert_eq!((&body[0], &body[1], *tag), (&Stmt::Nop, &Stmt::IntLit(3), 2));
  assert!(matches!(&body[2], Stmt::Block { body, tag: 1 } if body[0] == Stmt::IntLit(4)));

  let tree = Region::new_in(Store::default(), Tree::node(Tree::Leaf(1u16), Tree::node(Tree::leaf(2), Tree::Leaf(3))));
  assert_eq!(sum(&tree), 6);
  let pair =
    Region::new_in(Store::default(), Pair::new(Color::Blue, Tree::leaf(Wrap::new(1, [Color::Red, Color::Green]))));
  assert_eq!(pair.first, Color::Blue);
  assert!(matches!(&*pair.second, Tree::Leaf(Wrap(1, c)) if c[..] == [Color::Red, Color::Green]));

  let arr = Region::new_in(Store::default(), Arr::<2>::new([1, 2], [[3, 4], [5, 6]]));
  assert_eq!((arr.items, &arr.rest[..]), ([1, 2], &[[3, 4], [5, 6]][..]));
  let shapes = Region::new_in(Store::default(), Lst::new([Shape::Dot, Shape::Circle(3), Shape::Rect { w: 1, h: 2 }]));
  assert_eq!(shapes.items[2], Shape::Rect { w: 1, h: 2 });
}

/// Aligned to 4: as much as a `u64` on `i686` (a region holds no type more aligned than that).
#[flat]
#[repr(align(4))]
struct Padded {
  a: u8,
  b: u32,
  c: u8,
  d: u16,
}

#[test]
fn layout() {
  let r = Region::new_in(Store::default(), Aligned::new([1u8], [2u64, 3], "abc", [4u16]));
  let r2 = Region::new_in(Store::default(), Aligned::new((), (), "", ()));
  for r in [&r, &r2, &r.clone()] {
    assert_eq!(r.big.as_ptr().addr() % align_of::<u64>(), 0);
    assert_eq!(r.wide.as_ptr().addr() % 2, 0);
  }
  assert_eq!(r.big[..], [2, 3]);
  assert!(r2.small.is_empty() && r2.big.is_empty() && r2.text.is_empty());
  assert_eq!(size_of::<Near<[u64]>>(), 4);
  assert_eq!(size_of::<Option<Near<Expr>>>(), 4);
  // Fields keep their declaration order also next to another `repr`: the image must not depend
  // on how the compiler orders fields.
  let offsets = [
    std::mem::offset_of!(Padded, a),
    std::mem::offset_of!(Padded, b),
    std::mem::offset_of!(Padded, c),
    std::mem::offset_of!(Padded, d),
  ];
  assert_eq!((offsets, size_of::<Padded>(), align_of::<Padded>()), ([0, 4, 8, 10], 12, 4));
  let padded = [&[1, 0, 0, 0][..], &2u32.to_ne_bytes(), &[3, 0], &4u16.to_ne_bytes()].concat();
  assert_eq!(Region::new_in(Store::default(), Padded::new(1, 2, 3, 4)).as_bytes().unwrap(), padded);
  // `[cap][len][elements]` after the pointer; strings are `[len][bytes]`.
  let l = Region::<Near<[u32]>>::new_in(Store::default(), Near::from_iter([7u32, 8]));
  let words =
    |b: &[u8]| b.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
  assert_eq!(words(l.as_bytes().unwrap()), [8, 2, 2, 7, 8]);
  let s = Region::new_in(Store::default(), Lst::<Near<str>>::new(["ab"]));
  assert_eq!(words(&s.as_bytes().unwrap()[..16]), [8, 1, 1, 4]);
  // A pushed list has spare capacity; a copy is compact again.
  let mut l = l;
  l.edit().push(9);
  assert_eq!(l.byte_len(), 20 + 8 + 4 * 4);
  assert!(l.as_bytes().is_none());
  assert_eq!(words(&compact_bytes(&l)), [8, 3, 3, 7, 8, 9]);
}

#[test]
fn near_traits() {
  let r = Region::new_in(Store::default(), Pair::<Inst, Near<[Inst]>>::new(
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
  let s = Region::new_in(Store::default(), Pair::<u8, Near<str>>::new(0, "x"));
  assert_eq!(
    (format!("{:?}", s.second), hash(&*s.second) == hash(&*s.second)),
    ("\"x\"".into(), true)
  );
  let e = Region::new_in(Store::default(), Expr::neg(Expr::Lit(1)));
  let Expr::Neg(x) = &*e else { unreachable!() };
  assert_eq!((format!("{x:?}"), x == x), ("Lit(1)".into(), true));
  // `Display` shows the pointee.
  let d = Region::new_in(Store::default(), Pair::<Near<u64>, Near<str>>::new(Near::new(42), "text"));
  assert_eq!(format!("{} {:>6}", d.first, d.second), "42   text");
}

#[test]
fn region_traits() {
  // The clones outlive the region they come from.
  let (g, h) = {
    let f = func();
    let g = f.clone();
    assert_eq!(f, g);
    assert_eq!(format!("{f:?}"), format!("{:?}", *g));
    assert!(format!("{f:?}").starts_with("Func { name: \"main\""));
    assert_eq!(Region::new_in(Store::default(), 1.5f64).to_string(), "1.5");
    // The clone is independent.
    let mut h = f.clone();
    *h.edit().flag() = false;
    h.edit().name().set("other");
    assert_eq!((&*f.name, f.flag, &*h.name, h.flag), ("main", true, "other", false));
    assert_ne!(f, h);
    (g, h)
  };
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
  let mut w = Region::new_in(Store::default(), Wrap::new(1, [Color::Red]));
  *w.edit()._0() = 2;
  w.edit()._1().push(Color::Blue);
  assert_eq!((w.0, &w.1[..]), (2, &[Color::Red, Color::Blue][..]));
  let mut n = Region::new_in(Store::default(), Names::new(1, "a", 2));
  *n.edit().set_() = 3;
  n.edit().replace_().set("b");
  *n.edit().r#as_mut() = 4;
  assert_eq!((n.set, &*n.replace, n.r#as_mut), (3, "b", 4));
  let mut m = Region::new_in(Store::default(), private::Mixed::new(1, "h"));
  *m.edit().open() = 2;
  private::rename(m.edit(), "renamed");
  assert_eq!((m.open, m.hidden()), (2, "renamed"));

  // Handles of generic types.
  let mut p = Region::new_in(Store::default(), Pair::new(1u16, Lst::new([1u8])));
  *p.edit().first() = 2;
  p.edit().second().items().push(2);
  assert_eq!((p.first, &p.second.items[..]), (2, &[1, 2][..]));
  let mut a = Region::new_in(Store::default(), Arr::<2>::new([1, 2], ()));
  a.edit().items()[1] = 5;
  a.edit().rest().push([7, 8]);
  assert_eq!((a.items, &a.rest[..]), ([1, 5], &[[7, 8]][..]));
}

#[test]
fn plain_handles_deref_mut() {
  let mut r = Region::new_in(Store::default(), Rgb { r: 1, g: 2, b: 3, c: Color::Red });
  **r.edit() = Rgb { r: 9, g: 9, b: 9, c: Color::Green }; // `Copy`: the handle is assignable
  r.edit().c = Color::Blue;
  *r.edit().r() = 4;
  assert_eq!(*r, Rgb { r: 4, g: 9, b: 9, c: Color::Blue });
  let mut s = Region::new_in(Store::default(), Shape::Dot);
  **s.edit() = Shape::Circle(1);
  if let ShapeMut::Circle(mut radius) = s.edit().as_mut() {
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
  let mut c = Region::new_in(Store::default(), Wrap::new(0, [Color::Red, Color::Blue]));
  c.edit()._1().fill(Color::Green); // a list of plain data is a real `&mut [T]`
  assert_eq!(c.1[..], [Color::Green; 2]);
  let mut u = Region::new_in(Store::default(), Color::Red);
  u.edit().set(Color::Blue);
  **u.edit() = Color::Green;
  assert_eq!(*u, Color::Green);
}

#[test]
fn enum_handles() {
  let mut s =
    Region::new_in(Store::default(), Stmt::block((Stmt::r#let("x", Expr::Lit(1)), Stmt::r#return(None), Stmt::Nop), 7));
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
  let mut t = Region::new_in(Store::default(), Tree::node(Tree::Leaf(1u32), Tree::Leaf(2)));
  if let TreeMut::Node(kids) = t.edit().as_mut() {
    if let TreeMut::Leaf(mut x) = kids._0().as_mut() {
      *x = 10;
    }
    kids._1().set(Tree::node(Tree::Leaf(3), Tree::Leaf(4)));
  }
  let Tree::Node(a, b) = &*t else { unreachable!() };
  assert!(**a == Tree::Leaf(10) && matches!(&**b, Tree::Node(c, _) if **c == Tree::Leaf(3)));
}

/// A value written over another one writes every byte of it, its pointers included: a `None`
/// over a `Some` is zeros again, from a value on the stack and from a copy of a region's.
#[test]
fn a_value_overwrites_every_pointer() {
  let mut b = Region::new_in(Store::default(), Block::new(0, [Inst::new(1, [2])], Near::new(Block::new(5, (), None))));
  b.edit().set(Block::new(1, (), None));
  assert!(b.next.is_none() && b.insts.is_empty() && b.id == 1);
  b.edit().set(Block::new(2, (), Near::new(Block::new(6, (), None))));
  let none = Region::new_in(Store::default(), Block::new(3, (), None));
  b.edit().set(&*none);
  assert!(b.next.is_none() && b.id == 3);
}

#[test]
fn list_handles() {
  let mut b = Region::new_in(Store::default(), Block::new(0, (), None));
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
  let mut b = Region::new_in(Store::default(), &*b);
  for i in 0..100u8 {
    b.edit().insts().push(Inst::new(i, Near::from_iter(0..u32::from(i % 7))));
  }
  assert_eq!(b.insts.len(), 102);
  assert!(
    b.insts[2..].iter().enumerate().all(|(i, x)| x.op as usize == i && x.args.len() == i % 7)
  );
  // `set` and `replace` act on the pointer; a whole list moves.
  b.edit().insts().set(());
  assert!(b.insts.is_empty());
  b.edit().insts().set([Inst::new(1, [1])]);
  let e = b.edit();
  let old = e.insts().replace((Inst::new(0, ()),));
  e.next().insert(Block::new(9, old, None));
  assert_eq!((ops(&b), ops(b.next.as_ref().unwrap())), (vec![0], vec![1]));
  // Lists of lists, of pointers and of strings.
  let mut l = Region::new_in(Store::default(), Lst::<Near<[u8]>>::new(([1u8], ())));
  l.edit().items().at(1).push(2);
  l.edit().items().push([3]).push(4);
  l.edit().items().insert(0, ()).push(0);
  assert_eq!(
    l.items.iter().map(|x| x.to_vec()).collect::<Vec<_>>(),
    [vec![0], vec![1], vec![2], vec![3, 4]]
  );
  let mut p = Region::new_in(Store::default(), Lst::<Near<Expr>>::new([Near::new(Expr::Lit(1))]));
  let items = p.edit().items();
  items.push(Near::new(Expr::Lit(2))).set(Expr::Lit(3));
  let one = items.at(0).replace(Expr::Lit(0));
  items.at(0).set(Expr::neg(one));
  assert_eq!(p.items.iter().map(|e| eval(e)).collect::<Vec<_>>(), [-1, 3]);
  let mut s = Region::new_in(Store::default(), Lst::<Near<str>>::new(["b"]));
  s.edit().items().insert(0, "a");
  s.edit().items().push("c").set("C");
  assert_eq!(s.items.iter().map(|x| &**x).collect::<String>(), "abC");
}

#[test]
fn list_index_panics() {
  let mut r = Region::<Near<[u32]>>::new_in(Store::default(), Near::from_iter([1u32, 2]));
  fails(|| *r.edit().at(2) = 0);
  fails(|| {
    let _ = r.edit().remove(2);
  });
  fails(|| *r.edit().insert(3, 0) = 0);
  fails(|| r.edit().swap(0, 2));
  // `try_at` is `at` without the panic.
  let list = r.edit();
  assert!(list.try_at(2).is_none());
  if let Some(mut x) = list.try_at(1) {
    *x += 1;
  }
  assert_eq!((r[..].to_vec(), r.byte_len()), (vec![1, 3], 4 + 8 + 8));
  let mut exprs = Region::new_in(Store::default(), Lst::<Expr>::new([Expr::Lit(1)]));
  let items = exprs.edit().items();
  assert!(items.try_at(1).is_none());
  items.try_at(0).unwrap().set(Expr::neg(Expr::Lit(2)));
  assert_eq!(show(&exprs.items[0]), "-2");
}

#[test]
fn option_handles() {
  let mut b = Region::new_in(Store::default(), Block::new(0, (), None));
  assert!(b.edit().next().as_mut().is_none());
  assert!(b.edit().next().take().is_none());
  let next = b.edit().next();
  next.insert(Block::new(1, (), None)).insts().push(Inst::new(1, ()));
  assert!(next.is_some()); // reads as `Option<Near<Block>>`
  *next.as_mut().unwrap().id() = 2;
  next.as_mut().unwrap().next().insert(Block::new(3, (), None));
  assert_eq!(b.next.as_ref().map(|n| (n.id, ops(n))), Some((2, vec![1])));
  assert_eq!(b.next.as_ref().unwrap().next.as_ref().unwrap().id, 3);
  // `insert` over a pointee replaces it.
  b.edit().next().insert(Block::new(4, (), None));
  assert_eq!(b.next.as_ref().map(|n| (n.id, n.next.is_none())), Some((4, true)));
  b.edit().next().set(None);
  assert!(b.next.is_none());
  b.edit().next().set(Near::new(Block::new(5, (), None)));
  // `take`: push a block in front, the old one moves; `take` allocates nothing.
  let e = b.edit();
  let before = e.next().as_mut().map(|n| n.id);
  let old = e.next().take().unwrap();
  assert!(e.next().is_none());
  e.next().insert(Block::new(6, (), Near::new(old)));
  let n = b.next.as_ref().unwrap();
  assert_eq!((before, n.id, n.next.as_ref().unwrap().id), (Some(5), 6, 5));
  // Optional pointers to plain data are `&mut T`.
  let mut o = Region::new_in(Store::default(), Pair::new(0u8, None::<Near<u64>>));
  *o.edit().second().insert(1) += 1;
  assert_eq!(o.second.as_deref(), Some(&2));
}

// ---------------------------------------------------------------------------------------------
// Moving

#[test]
fn set_and_replace() {
  let mut f = func();
  // `set` on a struct, from a value built elsewhere and from a copy.
  let other = func();
  f.edit().entry().set(Block::new(5, &other.entry.insts, None));
  assert_eq!((f.entry.id, ops(&f.entry)), (5, vec![1, 2]));
  f.edit().set(&*other);
  assert_eq!(*f, *other);
  // `set` allocates nothing for the old value.
  let len = f.byte_len();
  *f.edit().entry().id() = 1;
  f.edit().entry().next().set(None);
  f.edit().flag().clone_from(&false);
  assert_eq!(f.byte_len(), len);
  // `replace` returns the old value; it moves with everything it points to.
  let e = f.edit();
  let old = e.entry().replace(Block::new(7, (), None));
  e.entry().next().insert(old);
  let _ = e.name().replace("renamed");
  e.entry().insts().push(Inst::new(9, ()));
  let len = f.byte_len();
  let e = f.edit();
  let n = e.entry().next().as_mut().unwrap().insts().at(0).args().replace(());
  e.entry().insts().at(0).args().set(n);
  assert!(f.byte_len() <= len + 4 + 8, "moves copy no elements");
  let entry = &f.entry;
  assert_eq!(
    (&*f.name, entry.id, ops(entry), args(entry)),
    ("renamed", 7, vec![9], vec![vec![10, 20, 30]])
  );
  assert_eq!(args(entry.next.as_ref().unwrap()), [vec![], vec![]]);
}

#[test]
fn moves_in_a_tree() {
  let mut r = Region::new_in(Store::default(), Expr::add(Expr::neg(Expr::neg(Expr::Lit(3))), Expr::Lit(4)));
  let e = r.edit();
  // Wrap, like `mem::replace`.
  let old = e.replace(Expr::Lit(0));
  e.set(Expr::neg(old));
  assert_eq!(show(e), "-(--3 + 4)");
  // Hoist: a child takes the place of its parent.
  if let ExprMut::Neg(x) = e.as_mut() {
    let inner = x.replace(Expr::Lit(0));
    e.set(inner);
  }
  assert_eq!(show(e), "(--3 + 4)");
  // Swap two children: three moves.
  let before = r.byte_len();
  let e = r.edit();
  if let ExprMut::Add(args) = e.as_mut() {
    let a = args._0().replace(Expr::Lit(0));
    let b = args._1().replace(a);
    args._0().set(b);
  }
  assert_eq!(show(e), "(4 + --3)");
  // Rotate with two parts of the old value: (a + b) -> (b + (a + 9)).
  if let ExprMut::Add(args) = e.as_mut() {
    let a = args._0().replace(Expr::Lit(0));
    let b = args._1().replace(Expr::Lit(0));
    e.set(Expr::add(b, Expr::add(a, Expr::Lit(9))));
  }
  assert_eq!(show(&r), "(--3 + (4 + 9))");
  assert!(r.byte_len() - before <= 12 * size_of::<Expr>(), "moves, no deep copies");
  // Many wraps: the buffer reallocates under the moves.
  let e = r.edit();
  let wraps = if cfg!(miri) { 30 } else { 200 };
  for i in 0..wraps {
    let old = e.replace(Expr::Lit(0));
    e.set(Expr::add(old, Expr::Lit(i)));
  }
  assert_eq!(eval(&r), 3 + 13 + (0..wraps).sum::<i64>());
  let copy = Region::new_in(Store::default(), &*r);
  assert_eq!(show(&copy), show(&r));
  assert!(copy.byte_len() < r.byte_len());
}

/// Helper functions name the edit's lifetime to return handles and moved values.
fn first<'a, 'e>(b: &'a mut Edit<'e, Block>) -> &'a mut Edit<'e, Inst> {
  b.insts().at(0)
}

fn pop<'e>(list: &mut Edit<'e, Near<[Inst]>>) -> nearest::Old<'e, Inst> {
  list.remove(0)
}

#[test]
fn moves_between_lists_and_regions() {
  let mut f = func();
  let b = f.edit().entry();
  // An instruction moves to the next block, with its arguments.
  let inst = pop(b.insts());
  b.next().as_mut().unwrap().insts().push(inst);
  assert_eq!(ops(b), [2]);
  assert_eq!(args(b.next.as_ref().unwrap()), [vec![0, 1, 2, 3, 4], vec![10, 20, 30]]);
  // A whole list moves, and a removed element comes back elsewhere.
  let list = b.next().as_mut().unwrap().insts().at(1).args().replace(());
  first(b).args().set(list);
  let inst = b.next().as_mut().unwrap().insts().remove(0);
  b.insts().insert(0, inst).args().push(5);
  assert_eq!(args(b), [vec![0, 1, 2, 3, 4, 5], vec![10, 20, 30]]);
  assert_eq!(args(b.next.as_ref().unwrap()), [vec![]]);
  // A string moves.
  let _ = f.edit().name().replace("x");
  let e = f.edit();
  let name = e.name().replace("y");
  e.entry().next().as_mut().unwrap().insts().at(0).args().push(1);
  e.name().set(name);
  assert_eq!(&*f.name, "x");

  // Into another region, and into a new one, a moved value is a copy.
  let (mut a, mut c) = (func(), func());
  let inst = a.edit().entry().insts().remove(0);
  c.edit().entry().insts().push(inst);
  assert_eq!((ops(&a.entry), ops(&c.entry)), (vec![2], vec![1, 2, 1]));
  let next = c.edit().entry().next().take().unwrap();
  let d = Region::new_in(Store::default(), next);
  assert_eq!((d.id, ops(&d)), (1, vec![3]));
  assert!(c.entry.next.is_none());
  // Two edits at once, of two regions.
  let (ea, ec) = (a.edit(), c.edit());
  let x = ea.entry().insts().remove(0);
  let y = ec.entry().insts().remove(0);
  ea.entry().insts().push(y);
  ec.entry().insts().push(x);
  assert_eq!((ops(&a.entry), ops(&c.entry)), (vec![1], vec![2, 1, 2]));
  assert_eq!(*Region::new_in(Store::default(), &*a), *a);
}

/// Anything `impl Emit` can be passed on generically.
fn wrap_twice(e: impl Emit<Expr>) -> impl Emit<Expr> {
  Expr::neg(Expr::neg(e))
}

#[test]
fn builders_compose() {
  let mut r = Region::new_in(Store::default(), wrap_twice(Expr::Lit(4)));
  assert_eq!(show(&r), "--4");
  let e = r.edit();
  let old = e.replace(Expr::Lit(0));
  e.set(wrap_twice(old));
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
  let mut r = Region::new_in(Store::default(), Expr::neg(Expr::Lit(7)));
  let e = r.edit();
  fails(|| e.set(Expr::add(Expr::Lit(1), Boom)));
  assert_eq!(show(e), "-7");
  fails(|| {
    let _ = e.replace(Expr::add(Expr::Lit(1), Boom));
  });
  assert_eq!(show(e), "-7");
  // A moved value already written into the new one: the old one comes back.
  let ExprMut::Neg(x) = e.as_mut() else { unreachable!() };
  let seven = x.replace(Expr::Lit(1));
  fails(|| e.set(Expr::add(seven, Expr::add(Expr::Lit(2), Boom))));
  assert_eq!(show(e), "-1");
  let ExprMut::Neg(x) = e.as_mut() else { unreachable!() };
  x.set(Expr::Lit(8));
  assert_eq!(show(&r), "-8");

  let mut f = func();
  let e = f.edit();
  fails(|| e.name().set(Boom));
  fails(|| {
    let _ = e.name().replace(Boom);
  });
  fails(|| e.entry().next().set(Near::new(Block::new(1, (), Boom))));
  fails(|| {
    e.entry().next().insert(Block::new(1, (Inst::new(0, ()), Boom), None));
  });
  fails(|| e.entry().insts().at(0).args().set([Boom]));
  fails(|| {
    let _ = e.entry().replace(Block::new(1, (), Boom));
  });
  assert_eq!(*f, *func());
}

#[test]
fn a_panic_in_a_list_operation_leaves_a_valid_list() {
  let mut b = Region::new_in(Store::default(), Block::new(0, (Inst::new(0, [0]), Inst::new(1, [1])), None));
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
  assert_eq!(*Region::new_in(Store::default(), &*b), *b);
}

#[test]
fn the_size_limit_leaves_a_valid_region() {
  // The real limit is 2 GiB; check that a request past it fails before changing anything.
  let mut r = Region::new_in(Store::default(), Lst::new([1u64]));
  let len = r.byte_len();
  fails(|| r.edit().items().set(Near::from_iter(std::iter::repeat_n(0u64, usize::MAX / 16))));
  assert_eq!(r.byte_len(), len);
  assert_eq!(r.items[..], [1]);
}

// ---------------------------------------------------------------------------------------------
// Bytes

/// Bytes at an 8-aligned address, as a memory map would give.
fn aligned(bytes: &[u8]) -> Vec<u64> {
  let mut words = vec![0u64; bytes.len().div_ceil(8)];
  for (w, c) in words.iter_mut().zip(bytes.chunks(8)) {
    let mut word = [0; 8];
    word[..c.len()].copy_from_slice(c);
    *w = u64::from_ne_bytes(word);
  }
  words
}

/// The compact image of any region: a copy is compact.
fn compact_bytes<T: Flat>(r: &Region<T>) -> Vec<u8> {
  Region::new_in(Store::default(), &**r).as_bytes().unwrap().to_vec()
}

/// `nearest::view` of `bytes`, copied to words: an address aligned as a region is.
fn view<T: Flat, R>(bytes: &[u8], f: impl FnOnce(&T) -> R) -> Result<R, nearest::Error> {
  let words = aligned(bytes);
  // SAFETY: `words` holds at least `bytes.len()` initialized bytes.
  let slice = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), bytes.len()) };
  nearest::view::<T>(slice).map(f)
}

#[flat]
#[derive(Debug, PartialEq)]
struct Rec {
  flag: bool,
  letter: char,
  count: NonZero<u16>,
  color: Color,
  name: Near<str>,
  kids: Near<[Rec]>,
  expr: Option<Near<Expr>>,
  wide: Near<[u64]>,
}

fn rec() -> Region<Rec> {
  let leaf = |c: char| Rec::new(false, c, NonZero::new(1).unwrap(), Color::Red, "", (), None, ());
  Region::new_in(Store::default(), Rec::new(
    true,
    'é',
    NonZero::new(300).unwrap(),
    Color::Blue,
    "root",
    (
      leaf('a'),
      Rec::new(
        true,
        'b',
        NonZero::new(2).unwrap(),
        Color::Green,
        "kid",
        [leaf('c')],
        Near::new(Expr::Lit(5)),
        [7],
      ),
    ),
    Near::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2)))),
    [1, 2],
  ))
}

#[test]
fn bytes_round_trip() {
  let mut r = rec();
  let image = r.as_bytes().unwrap().to_vec();
  assert_eq!(image.len(), r.byte_len());
  assert_eq!(view::<Rec, _>(&image, |v| format!("{v:?}")), Ok(format!("{:?}", *r)));
  // In place, from the region's own (8-aligned) buffer.
  assert_eq!(nearest::view::<Rec>(r.as_bytes().unwrap()).map(|v| v.letter), Ok('é'));
  // Any alignment of the input.
  let mut shifted = vec![0u8];
  shifted.extend_from_slice(&image);
  let back = Region::<Rec>::from_bytes(&shifted[1..]).unwrap();
  assert_eq!(*back, *r);
  assert_eq!(back.as_bytes(), Some(&*image));
  // Trusted bytes can skip the check.
  // SAFETY: `image` comes from `as_bytes` for the same type.
  let trusted = unsafe { Region::<Rec>::from_bytes_unchecked(&image) };
  assert_eq!((&*trusted, trusted.byte_len()), (&*r, image.len()));
  let own = trusted.as_bytes().unwrap();
  // SAFETY: as above, and a region's own buffer is 8-aligned.
  assert_eq!(unsafe { nearest::view_unchecked::<Rec>(own) }, &*r);
  // `view` needs an 8-aligned slice.
  let words = aligned(&shifted);
  // SAFETY: `words` holds `shifted.len()` initialized bytes.
  let misaligned =
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), shifted.len()) };
  assert!(nearest::view::<Rec>(&misaligned[1..]).is_err());
  // An edited region has no image of its own; a copy has one, and only what is reachable.
  let e = r.edit();
  let kid = e.kids().remove(0);
  e.kids().at(0).kids().push(kid);
  e.wide().push(3);
  e.name().set("edited");
  assert!(r.as_bytes().is_none());
  let edited = compact_bytes(&r);
  assert!(edited.len() < r.byte_len());
  let mut back = Region::<Rec>::from_bytes(&edited).unwrap();
  assert_eq!(*back, *r);
  assert_eq!(back.as_bytes(), Some(&*edited));
  // A loaded region is editable.
  back.edit().kids().at(0).kids().at(1).wide().push(5);
  assert_eq!(back.kids[0].kids[1].wide[..], [5]);
  // Padding is not part of the value.
  let mut p = Region::<[Pad; 2]>::new_in(Store::default(), [Pad { a: 1, b: 2 }, Pad { a: 3, b: 4 }]);
  p.edit()[0] = Pad { a: 9, b: 9 };
  assert_eq!(
    *Region::<[Pad; 2]>::from_bytes(&compact_bytes(&p)).unwrap(),
    [Pad { a: 9, b: 9 }, Pad { a: 3, b: 4 }]
  );
  let f = func();
  assert_eq!(*Region::<Func>::from_bytes(f.as_bytes().unwrap()).unwrap(), *f);
}

#[test]
fn builders_write_the_compact_image() {
  fn same<T: Flat>(r: &Region<T>) {
    let image = r.as_bytes().unwrap();
    assert_eq!(Some(image), Region::new_in(Store::default(), &**r).as_bytes());
    assert!(Region::<T>::from_bytes(image).is_ok());
  }
  same(&func());
  same(&rec());
  same(&Region::new_in(Store::default(), Stmt::block(
    (Stmt::Nop, Stmt::r#let("x", Expr::Lit(1)), Stmt::r#return(None)),
    1,
  )));
  same(&Region::new_in(Store::default(), Tree::node(Tree::Leaf(1u16), Tree::node(Tree::leaf(2), Tree::Leaf(3)))));
  same(&Region::new_in(Store::default(), Aligned::new([1u8], [2u64, 3], "abc", [4u16])));
  same(&Region::new_in(Store::default(), Arr::<2>::new([1, 2], [[3, 4], [5, 6]])));
  same(&Region::new_in(Store::default(), Chain::link(1, Chain::link(2, Chain::End))));
  same(&Region::new_in(Store::default(), Lst::<Near<Expr>>::new(Near::from_iter(
    (0..5).map(|i| Near::new(Expr::Lit(i))),
  ))));
  same(&Region::new_in(Store::default(), Prims::new(
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
    true,
    'x',
    [4, 5, 6],
    NonZero::new(1).unwrap(),
    NonZero::new(2).unwrap(),
  )));
  // A list whose iterator lied about its length is not compact; its spare room is used.
  for (n, hint) in [(3, 1000), (40, 3)] {
    let mut r = Region::new_in(Store::default(), Inst::new(0, Near::from_iter(Liar { n, hint })));
    assert!(r.as_bytes().is_none());
    assert!(Region::<Inst>::from_bytes(&compact_bytes(&r)).is_ok());
    let len = r.byte_len();
    r.edit().args().push(7);
    if hint > n as usize {
      assert_eq!(r.byte_len(), len, "the spare room takes the element");
    }
  }
}

/// Every image with one byte changed, or cut, or longer, is either rejected or read in full as a
/// value that shows the same and whose copy is no longer than those bytes: its pointees form a tree
/// in them, so a corruption is never read as more than its bytes, nor out of them (Miri). Which corruptions are accepted depends on the bytes of
/// the target (its widths, its alignment), so only that both outcomes happen is asserted: the
/// image itself is accepted, a cut one rejected. Under Miri every 127th byte is changed.
#[test]
fn small_corruptions_are_rejected_or_harmless() {
  let image = rec().as_bytes().unwrap().to_vec();
  // Whether `bytes` are accepted; what is accepted is walked in full and is exactly the image of
  // its value.
  let try_one = |bytes: &[u8]| match view::<Rec, _>(bytes, |v| {
    (format!("{v:?}"), Region::new_in(Store::default(), v))
  }) {
    Ok((shown, again)) => {
      assert_eq!(format!("{:?}", *again), shown);
      assert!(again.byte_len() <= bytes.len(), "a copy is no longer than a tree");
      true
    }
    Err(_) => false,
  };
  assert!(try_one(&image), "the image is accepted");
  assert!(!try_one(&image[..image.len() - 1]), "a cut one is not");
  let step = if cfg!(miri) { 127 } else { 1 };
  for i in (0..image.len()).step_by(step) {
    for v in [0u8, 1, 2, 4, 8, 0x7f, 0x80, 0xff, image[i].wrapping_add(1), image[i] ^ 0x10] {
      let mut bytes = image.clone();
      bytes[i] = v;
      try_one(&bytes);
    }
  }
  for cut in (0..image.len()).step_by(step) {
    try_one(&image[..cut]);
  }
  let mut longer = image;
  longer.extend_from_slice(&[0; 8]);
  try_one(&longer);
}

#[test]
fn malformed_images_are_rejected() {
  let word = |bytes: &[u8], at: usize| i32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
  let put =
    |bytes: &mut [u8], at: usize, v: i32| bytes[at..at + 4].copy_from_slice(&v.to_ne_bytes());
  let ok = |bytes: &[u8]| view::<Expr, _>(bytes, show);
  let image = Region::new_in(Store::default(), Expr::add(Expr::neg(Expr::Lit(1)), Expr::Lit(2))).as_bytes().unwrap().to_vec();
  assert_eq!(ok(&image), Ok("(-1 + 2)".into()));
  let (l, r) = (4, 8); // the two pointers of the root `Add`
  let mut shared = image.clone();
  put(&mut shared, r, word(&image, l) - 4); // both pointers to the same pointee
  assert!(ok(&shared).is_err());
  let mut cycle = image.clone();
  put(&mut cycle, l, -4); // a pointer back to the root
  assert!(ok(&cycle).is_err());
  let mut swapped = image.clone(); // valid pointees in the wrong order
  put(&mut swapped, l, word(&image, r) + 4);
  put(&mut swapped, r, word(&image, l) - 4);
  assert!(ok(&swapped).is_err());
  let mut tag = image.clone();
  tag[0] = 3; // no such variant
  assert!(ok(&tag).is_err());
  assert!(ok(&image[..image.len() - 1]).is_err()); // missing bytes
  assert_eq!(ok(&[&image[..], &[0; 8]].concat()), Ok("(-1 + 2)".into())); // no pointer reaches them
  assert!(ok(&[]).is_err());

  // Lists and strings: absurd lengths, `cap != len`, bad UTF-8.
  let r = Region::new_in(Store::default(), Lst::<Near<str>>::new(["ab", "c"]));
  let image = r.as_bytes().unwrap().to_vec();
  let len_at = usize::try_from(word(&image, 0)).unwrap(); // the list's `len`
  assert_eq!((word(&image, len_at - 4), word(&image, len_at)), (2, 2));
  let ok = |bytes: &[u8]| view::<Lst<Near<str>>, _>(bytes, |l| l.items.len());
  assert_eq!(ok(&image), Ok(2));
  for (at, v) in [(len_at, -1), (len_at - 4, -1), (len_at - 4, 3), (len_at, 1), (len_at, 3)] {
    let mut bad = image.clone();
    put(&mut bad, at, v);
    assert!(ok(&bad).is_err(), "{at} = {v}");
  }
  let text_at = len_at + 4 + usize::try_from(word(&image, len_at + 4)).unwrap();
  let mut utf8 = image.clone();
  utf8[text_at + 4] = 0xff;
  assert!(ok(&utf8).is_err());
  let mut long = image;
  put(&mut long, text_at, 1 << 30);
  assert!(ok(&long).is_err());

  // Values: `bool`, `char`, `NonZero`, enum tags.
  let image = Region::new_in(Store::default(), Prims::new(
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
    true,
    'x',
    [4, 5, 6],
    NonZero::new(1).unwrap(),
    NonZero::new(2).unwrap(),
  ))
  .as_bytes()
  .unwrap()
  .to_vec();
  let ok = |bytes: &[u8]| view::<Prims, _>(bytes, |p| p.l);
  assert_eq!(ok(&image), Ok('x'));
  for (field, bytes) in [
    (std::mem::offset_of!(Prims, k), vec![2u8]),
    (std::mem::offset_of!(Prims, l), 0xd800u32.to_ne_bytes().to_vec()),
    (std::mem::offset_of!(Prims, l), 0x11_0000u32.to_ne_bytes().to_vec()),
    (std::mem::offset_of!(Prims, n), 0u32.to_ne_bytes().to_vec()),
    (std::mem::offset_of!(Prims, o), vec![0u8]),
  ] {
    let mut bad = image.clone();
    bad[field..field + bytes.len()].copy_from_slice(&bytes);
    assert!(ok(&bad).is_err(), "field at {field}");
  }
  let color = Region::new_in(Store::default(), Color::Blue).as_bytes().unwrap().to_vec();
  assert!(view::<Color, _>(&[3], |_| ()).is_err() && view::<Color, _>(&color, |_| ()).is_ok());

  // Elements that can be invalid are checked one by one, inside lists and arrays too; lists
  // of numbers are not walked, and any bytes there are a value.
  let flags = Region::new_in(Store::default(), Lst::<[bool; 2]>::new([[true, false], [false, true]]));
  let numbers = Region::new_in(Store::default(), Lst::<[u16; 2]>::new([[1, 2], [3, 4]]));
  let (flags, numbers) = (flags.as_bytes().unwrap().to_vec(), numbers.as_bytes().unwrap().to_vec());
  for i in 12..flags.len() {
    let (mut f, mut n) = (flags.clone(), numbers.clone());
    (f[i], n[12 + 2 * (i - 12)]) = (2, 0xff);
    assert!(view::<Lst<[bool; 2]>, _>(&f, |_| ()).is_err(), "flag {i}");
    assert_eq!(view::<Lst<[u16; 2]>, _>(&n, |l| l.items.len()), Ok(2));
  }

  // The error.
  let err = Region::<Expr>::from_bytes(&[9]).err().unwrap();
  assert_eq!(err, nearest::Error::Invalid(nearest::puns::Error::Size));
  assert_eq!(err.to_string(), "invalid region bytes: value does not have the size of the target type");
  let _: &dyn std::error::Error = &err;
  assert_eq!(format!("{err:?}"), "Invalid(Size)");
  assert_eq!(nearest::Error::StorageFull.to_string(), "region bytes larger than the storage");
}

#[test]
fn noise_is_rejected_without_panicking() {
  let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
  let mut accepted = 0;
  for _ in 0..if cfg!(miri) { 40 } else { 100_000 } {
    let len = rng.below(96);
    let small = rng.below(2) == 0;
    let bytes: Vec<u8> =
      (0..len).map(|_| if small { rng.below(24) } else { rng.below(256) } as u8).collect();
    accepted += usize::from(view::<Expr, _>(&bytes, show).is_ok());
    accepted += usize::from(view::<Rec, _>(&bytes, |r| format!("{r:?}")).is_ok());
    accepted += usize::from(view::<Lst<Near<str>>, _>(&bytes, |l| l.items.len()).is_ok());
  }
  // Few random images are valid; enough samples find some.
  assert!(cfg!(miri) || accepted > 0);
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn a_very_deep_value_is_checked_without_recursion() {
  const N: u32 = 300_000;
  // Built in a loop; copying it out recurses, so that part gets a large stack. Several MiB:
  // always on the heap.
  let image = std::thread::Builder::new()
    .stack_size(1 << 30)
    .spawn(|| {
      let mut r = nearest::Region::new(Block::new(0, (), None));
      let mut cur = r.edit();
      for id in 1..N {
        cur = cur.next().insert(Block::new(id, [Inst::new(id as u8, [id])], None));
      }
      nearest::Region::new(&*r).as_bytes().unwrap().to_vec()
    })
    .unwrap()
    .join()
    .unwrap();
  std::thread::Builder::new()
    .stack_size(64 << 10)
    .spawn(move || {
      let r = nearest::Region::<Block>::from_bytes(&image).unwrap();
      let (mut b, mut n) = (&*r, 1);
      while let Some(next) = &b.next {
        b = next;
        n += 1;
      }
      assert_eq!((n, b.id, b.insts[0].args[0]), (N, N - 1, N - 1));
    })
    .unwrap()
    .join()
    .unwrap();
}

#[test]
fn compaction() {
  let mut b = Region::new_in(Store::default(), Block::new(0, (), None));
  for i in 0..20u8 {
    b.edit().insts().push(Inst::new(i, [u32::from(i)]));
    b.edit().insts().at(0).args().push(u32::from(i));
    b.edit().next().insert(Block::new(u32::from(i), (), None));
  }
  let c = Region::new_in(Store::default(), &*b);
  assert!(c.byte_len() * 2 < b.byte_len());
  assert_eq!(*c, *b);
  assert_eq!(Region::new_in(Store::default(), &*c).byte_len(), c.byte_len());
  assert_eq!(c.as_bytes().map(<[u8]>::len), Some(c.byte_len()));
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
  // The long list is a tenth of its length under Miri: it grows as many times but four.
  let long = if cfg!(miri) { 100 } else { 1000 };
  for (n, hint) in [(3, 1000), (long, 0), (50, 3), (0, 10)] {
    let r = Region::new_in(Store::default(), Inst::new(0, Near::from_iter(Liar { n, hint })));
    assert_eq!(r.args.to_vec(), (0..n).rev().collect::<Vec<_>>());
    // Elements that allocate while the list grows.
    let b = Region::new_in(Store::default(), Block::new(
      0,
      Near::from_iter(
        Liar { n, hint }.map(|i| Inst::new((i % 5) as u8, Near::from_iter(0..i % 5))),
      ),
      None,
    ));
    assert_eq!(b.insts.len(), n as usize);
    assert!(b.insts.iter().all(|i| i.args.len() == usize::from(i.op) % 5));
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

/// The doc `depth` levels down the `next` chain of the model and of the region.
fn walk(m: &mut MDoc, depth: usize) -> &mut MDoc {
  if depth == 0 { m } else { walk(m.next.as_mut().unwrap(), depth - 1) }
}

fn walk_edit<'a, 'e>(e: &'a mut Edit<'e, Doc>, depth: usize) -> &'a mut Edit<'e, Doc> {
  if depth == 0 { e } else { walk_edit(e.next().as_mut().unwrap(), depth - 1) }
}

fn chain_len(m: &MDoc) -> usize {
  1 + m.next.as_ref().map_or(0, |n| chain_len(n))
}

#[test]
fn random_edits_match_a_model() {
  let steps: u32 = if cfg!(miri) { 40 } else { 4000 };
  for seed in [0x2545_f491_u64, 0x9e37_79b9_7f4a_7c15] {
    let mut rng = Rng(seed);
    let mut m = MDoc { id: 0, name: "root".into(), items: vec![], next: None };
    let mut r = Region::new_in(Store::default(), Doc::new(0, "root", (), None));
    let mut om: Vec<MItem> = vec![];
    let mut other = Region::new_in(Store::default(), Lst::<Item>::new(()));
    let copied = Region::new_in(Store::default(), Item::new(7, [7, 7], Near::new(Label::new(70, "other"))));
    for step in 0..steps {
      let depth = rng.below(chain_len(&m));
      let (dm, de) = (walk(&mut m, depth), walk_edit(r.edit(), depth));
      let len = dm.items.len();
      let k = rng.below(len);
      let s = step;
      match rng.below(24) {
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
          let _ = de.items().remove(k);
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
          // A new item takes over the old one's values, which move.
          dm.items[k].tag = dm.items[k].tag.wrapping_add(1);
          dm.items[k].label = None;
          let tag = dm.items[k].tag;
          let vals = de.items().at(k).vals().replace(());
          let _ = de.items().at(k).replace(Item::new(tag, vals, None));
        }
        14 if len > 0 => {
          // Keep only item `k`, moved.
          dm.items = vec![dm.items[k].clone()];
          let it = de.items().remove(k);
          de.items().set([it]);
        }
        15 if len > 0 => {
          // A copy from another region.
          dm.items[k] = item_model(&copied);
          de.items().at(k).set(&*copied);
        }
        16 => {
          // Wrap: the doc moves into `next` of a new one.
          let old =
            std::mem::replace(dm, MDoc { id: s, name: "w".into(), items: vec![], next: None });
          dm.next = Some(Box::new(old));
          let old = de.replace(Doc::new(0, "", (), None));
          de.set(Doc::new(s, "w", (), Near::new(old)));
        }
        17 if dm.next.is_some() => {
          // Hoist: `next` takes its parent's place.
          *dm = *dm.next.take().unwrap();
          let next = de.next().take().unwrap();
          de.set(next);
        }
        18 => {
          dm.next = None;
          de.next().set(None);
        }
        19 if len > 0 => {
          // An item moves to the end, or to the next doc.
          let it = dm.items.remove(k);
          let old = de.items().remove(k);
          if let Some(next) = &mut dm.next {
            next.items.push(it);
            de.next().as_mut().unwrap().items().push(old);
          } else {
            dm.items.push(it);
            de.items().push(old);
          }
        }
        20 => {
          // `take` then `insert`: a doc is put in front of `next`.
          let old = dm.next.take();
          dm.next = Some(Box::new(MDoc { id: s, name: "f".into(), items: vec![], next: old }));
          let old = de.next().take();
          let new = de.next().insert(Doc::new(s, "f", (), None));
          if let Some(old) = old {
            new.next().set(Near::new(old));
          }
        }
        21 if len > 1 => {
          // Two items trade places through `replace` and `set`.
          let j = (k + 1) % len;
          dm.items.swap(k, j);
          let a = de.items().at(k).replace(Item::new(0, (), None));
          let b = de.items().at(j).replace(a);
          de.items().at(k).set(b);
        }
        22 if len > 0 => {
          // Into the other region: a copy.
          om.push(dm.items.remove(k));
          let it = de.items().remove(k);
          other.edit().items().push(it);
        }
        _ => {
          r = match step % 3 {
            0 => Region::new_in(Store::default(), &*r),
            1 => r.clone(),
            _ => Region::from_bytes(&compact_bytes(&r)).unwrap(),
          };
        }
      }
      assert_eq!(doc_model(&r), m, "step {step}");
      if step % 5 == 0 {
        let back = Region::<Doc>::from_bytes(&compact_bytes(&r)).unwrap();
        assert_eq!(doc_model(&back), m, "step {step}");
        let others: Vec<MItem> = other.items.iter().map(item_model).collect();
        assert_eq!(others, om);
      }
    }
    let compact = Region::new_in(Store::default(), &*r);
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

  let steps: u32 = if cfg!(miri) { 60 } else { 5000 };
  let mut rng = Rng(0x1234_5678);
  let mut m = M::Lit(0);
  let mut r = Region::new_in(Store::default(), Expr::Lit(0));
  let mut om = M::Lit(0);
  let mut other = Region::new_in(Store::default(), Expr::Lit(0));
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
    match rng.below(9) {
      0 => {
        *mm = M::Neg(Box::new(mm.clone()));
        let old = e.replace(Expr::Lit(0));
        e.set(Expr::neg(old));
      }
      1 => {
        *mm = M::Add(Box::new(mm.clone()), Box::new(M::Lit(s)));
        let old = e.replace(Expr::Lit(0));
        e.set(Expr::add(old, Expr::Lit(s)));
      }
      2 => {
        *mm = M::Add(Box::new(M::Lit(s)), Box::new(mm.clone()));
        let old = e.replace(Expr::Lit(0));
        e.set(Expr::add(Expr::Lit(s), old));
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
          let old = match e.as_mut() {
            ExprMut::Lit(_) => unreachable!(),
            ExprMut::Neg(x) => x.replace(Expr::Lit(0)),
            ExprMut::Add(args) => {
              if side == 0 {
                args._0().replace(Expr::Lit(0))
              } else {
                args._1().replace(Expr::Lit(0))
              }
            }
          };
          e.set(old);
        }
      }
      4 => {
        if let M::Lit(n) = mm {
          *n += s;
        }
        if let ExprMut::Lit(mut n) = e.as_mut() {
          *n += s;
        }
      }
      5 => {
        *mm = M::Lit(s);
        e.set(Expr::Lit(s));
      }
      6 => {
        // Swap the children of an `Add`: three moves.
        if let M::Add(a, b) = mm {
          std::mem::swap(a, b);
        }
        if let ExprMut::Add(args) = e.as_mut() {
          let a = args._0().replace(Expr::Lit(0));
          let b = args._1().replace(a);
          args._0().set(b);
        }
      }
      7 => {
        // Into the other region: a copy; the place keeps a literal.
        om = std::mem::replace(mm, M::Lit(s));
        let old = e.replace(Expr::Lit(s));
        other.edit().set(old);
      }
      _ => {
        if step % 2 == 0 {
          r = Region::new_in(Store::default(), &*r);
        } else {
          r = Region::from_bytes(&compact_bytes(&r)).unwrap();
        }
      }
    }
    assert_eq!(model(&r), m, "step {step}");
    assert_eq!(model(&other), om, "step {step}");
  }
}

/// `update`: the new value is built from the old one in a closure.
mod update {
  use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
  };

  use nearest::{Edit, Near, Old, flat};

  use super::{Expr, ExprMut, Region, Store, show};

  #[flat]
  struct Node {
    value: u32,
    name: Near<str>,
    kids: Near<[Node]>,
    next: Option<Near<Node>>,
  }

  fn chain(n: &Node) -> Vec<u32> {
    std::iter::successors(Some(n), |n| n.next.as_deref()).map(|n| n.value).collect()
  }

  /// Wraps every literal in a negation, in place.
  fn negate_literals(e: &mut Edit<Expr>) {
    match e.as_mut() {
      ExprMut::Lit(_) => e.update(Expr::neg),
      ExprMut::Neg(x) => negate_literals(x),
      ExprMut::Add(args) => {
        negate_literals(args._0());
        negate_literals(args._1());
      }
    }
  }

  #[test]
  fn update_builds_from_the_old_value() {
    let mut r = Region::new_in(Store::default(), Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2))));
    let e = r.edit();
    e.update(Expr::neg); // wrap, without a placeholder
    e.update(|old| Expr::add(Expr::Lit(0), old));
    assert_eq!(show(e), "(0 + -(1 + -2))");
    let before = r.byte_len();
    for _ in 0..50 {
      r.edit().update(|old| old); // the old value moves back in: no copy
    }
    assert!(r.byte_len() <= before + 50 * 2 * size_of::<Expr>());
    negate_literals(r.edit());
    assert_eq!(show(&r), "(-0 + -(-1 + --2))");

    let mut n = Region::new_in(Store::default(), Node::new(1, "a", [Node::new(7, "k", (), None)], None));
    let node = n.edit();
    node.next().insert(Node::new(3, "c", (), None));
    node.next().update(|rest| Near::new(Node::new(2, "b", (), rest))); // goes in front
    node.kids().update(|old| [Node::new(8, "group", old, None)]); // the kids go one level down
    node.name().update(|old| old);
    assert_eq!((chain(&n), &*n.name), (vec![1, 2, 3], "a"));
    assert_eq!((n.kids.len(), n.kids[0].value, n.kids[0].kids[0].value), (1, 8, 7));
  }

  #[test]
  fn update_restores_the_old_value_when_the_closure_panics() {
    let mut r = Region::new_in(Store::default(), Expr::neg(Expr::Lit(7)));
    let e = r.edit();
    let boom = catch_unwind(AssertUnwindSafe(|| e.update(|_| -> Expr { panic!("boom") })));
    assert!(boom.is_err());
    assert_eq!(show(e), "-7");
    e.update(Expr::neg);
    assert_eq!(show(&r), "--7");
  }

  /// While a panic unwinds, copies the kept value into another region.
  struct CopyOut<'s, 'a>(&'s mut Region<Expr>, &'s Cell<Option<Old<'a, Expr>>>);

  impl Drop for CopyOut<'_, '_> {
    fn drop(&mut self) {
      if let Some(old) = self.1.take() {
        self.0.edit().set(old);
      }
      assert_eq!(show(self.0), "--7");
    }
  }

  #[test]
  fn an_old_that_leaves_its_closure_is_a_copy_elsewhere() {
    let (mut a, mut b) = (Region::new_in(Store::default(), Expr::neg(Expr::Lit(7))), Region::new_in(Store::default(), Expr::Lit(0)));
    let mut stash = None;
    a.edit().update(|old| {
      stash = Some(old);
      Expr::Lit(1)
    });
    // It keeps `a` borrowed, so it can only go to another region, as a copy.
    b.edit().set(Expr::neg(stash.take().unwrap()));
    assert_eq!((show(&a), show(&b)), ("1".to_owned(), "--7".to_owned()));
    // Also when the closure then panics: `update` puts the old value back, and the kept one,
    // which still borrows its region, is copied out while the panic unwinds.
    let boom = catch_unwind(|| {
      let mut a = Region::new_in(Store::default(), Expr::Lit(0));
      let mut b = Region::new_in(Store::default(), Expr::neg(Expr::neg(Expr::Lit(7))));
      let stash = Cell::new(None);
      let _copy = CopyOut(&mut a, &stash);
      b.edit().update(|old| -> Expr {
        stash.set(Some(old));
        panic!("boom")
      });
    });
    assert!(boom.is_err());
  }
}

/// The pinned view: several fields at once, for reading, writing plain data and assigning
/// values without pointers. The handle is borrowed meanwhile, so nothing grows the buffer.
mod pinned {
  use std::pin::Pin;

  use nearest::{Near, flat};

  use super::{Block, BlockPin, Expr, ExprPin, Inst, InstPin, Region, Store, show};

  #[flat]
  struct Node {
    id: u32,
    kids: Near<[Node]>,
    next: Option<Near<Node>>,
  }

  /// Adds `by` to every literal, visiting both sides of an addition with both borrowed.
  fn bump(e: Pin<&mut Expr>, by: i64) {
    match e.project() {
      ExprPin::Lit(mut n) => *n += by,
      ExprPin::Neg(x) => bump(x.as_pin_mut(), by),
      ExprPin::Add(a, b) => {
        let (a, b) = (a.as_pin_mut(), b.as_pin_mut());
        bump(b, by);
        bump(a, by);
      }
    }
  }

  /// A region loaded as its bytes are, with bytes that no pointer reaches after them, is edited
  /// through its pinned view, both sides of an addition borrowed at once, and through handles: the
  /// check of `puns` found the pointees to form a tree, which these writes rely on (Miri). The
  /// same image with two pointers to one pointee, or a pointer back to the root, is not loaded.
  #[test]
  fn a_loaded_image_is_edited_in_place() {
    let built = Region::new_in(Store::default(), Expr::add(Expr::neg(Expr::Lit(1)), Expr::Lit(2)));
    let image = [built.as_bytes().unwrap(), &[0; 8][..]].concat();
    let mut r = Region::<Expr>::from_bytes(&image).unwrap();
    assert_eq!(r.byte_len(), image.len());
    bump(r.edit().pin(), 10);
    r.edit().update(|old| Expr::add(old, Expr::Lit(5)));
    let mut expected = Region::new_in(Store::default(), Expr::add(Expr::neg(Expr::Lit(11)), Expr::Lit(12)));
    expected.edit().update(|old| Expr::add(old, Expr::Lit(5)));
    assert_eq!(*r, *expected);
    let (left, right) = (4, 8); // the two pointers of the root `Add`
    let word = |at: usize| i32::from_ne_bytes(image[at..at + 4].try_into().unwrap());
    let mut shared = image.clone();
    shared[right..right + 4].copy_from_slice(&(word(left) - 4).to_ne_bytes());
    assert!(Region::<Expr>::from_bytes(&shared).is_err());
    let mut cycle = image;
    cycle[left..left + 4].copy_from_slice(&(-4i32).to_ne_bytes());
    assert!(Region::<Expr>::from_bytes(&cycle).is_err());
  }

  /// Data without pointers stays `Unpin`: a pin of it gives a `&mut`.
  #[test]
  fn plain_data_is_unpin() {
    const fn unpin<T: Unpin>() {}
    unpin::<super::Prims>();
    unpin::<super::Rgb>();
    unpin::<super::Shape>();
    let mut r = Region::new_in(Store::default(), super::Rgb { r: 1, g: 2, b: 3, c: super::Color::Red });
    r.edit().pin().get_mut().r = 9;
    let mut p = Region::new_in(Store::default(), super::Pad { a: 1, b: 2 });
    p.edit().pin().get_mut().b = 7;
    assert_eq!((r.r, p.b), (9, 7));
  }

  #[test]
  fn several_fields_are_borrowed_at_once() {
    let mut r =
      Region::new_in(Store::default(), Expr::add(Expr::neg(Expr::Lit(1)), Expr::add(Expr::Lit(2), Expr::Lit(3))));
    let e = r.edit();
    bump(e.pin(), 10);
    assert_eq!(show(e), "(-11 + (12 + 13))");
    // A value without pointers is assigned in place; what it replaces is garbage.
    if let ExprPin::Add(a, b) = e.pin().project() {
      let (mut a, b) = (a.as_pin_mut(), b.as_pin_mut());
      assert_eq!(show(&b), "(12 + 13)");
      a.set(Expr::Lit(7));
    }
    assert_eq!(show(e), "(7 + (12 + 13))");
    // The view is over: the handle edits again.
    e.set(Expr::neg(Expr::Lit(1)));
    bump(r.edit().pin(), 1);
    assert_eq!(show(&r), "-2");
  }

  #[test]
  fn struct_fields_and_list_elements() {
    let mut r = Region::new_in(Store::default(), Block::new(1, (Inst::new(1, [1, 2]), Inst::new(2, [3])), None));
    let b = r.edit();
    let BlockPin { mut id, insts, next } = b.pin().project();
    assert!(next.as_pin_mut().is_none());
    // Every element at once, each with its fields at once.
    let mut insts: Vec<InstPin<'_>> = insts.iter_pin_mut().map(Inst::project).collect();
    insts.reverse();
    for (i, inst) in insts.iter_mut().enumerate() {
      *inst.op += 10 * (i as u8 + 1);
      *id += u32::from(*inst.op);
    }
    assert_eq!((b.id, b.insts[0].op, b.insts[1].op), (1 + 12 + 21, 21, 12));
    // From a list handle too, and plain elements are plain.
    let total: u32 = b.insts().iter_pin_mut().map(|inst| inst.args.iter().sum::<u32>()).sum();
    for mut arg in b.insts().at(0).args().iter_pin_mut() {
      *arg += total;
    }
    assert_eq!(b.insts[0].args[..], [7, 8]);
    b.insts().push(Inst::new(3, ()));
    assert_eq!(r.insts.len(), 3);

    let mut n =
      Region::new_in(Store::default(), Node::new(0, [Node::new(1, (), None)], Near::new(Node::new(2, (), None))));
    let NodePin { mut id, kids, next } = n.edit().pin().project();
    let mut next = next.as_pin_mut().map(Near::as_pin_mut).unwrap().project();
    for kid in kids.iter_pin_mut() {
      *id += *kid.project().id + *next.id;
    }
    // Clearing an optional pointer is an assignment of a value without pointers.
    next.next.set(None);
    assert_eq!((n.id, n.next.as_ref().unwrap().id), (3, 2));
  }
}

/// Names that the generated code keeps for itself, and `Self` inside a field type.
mod names {
  use nearest::{Near, flat};

  use super::{Lst, Region, Store};

  /// Fields called like a method of the handle get a trailing underscore.
  #[flat]
  struct Led {
    pin: u8,
    update: u32,
    set: bool,
    replace: Near<str>,
  }

  /// A variant called `Project` gets the constructor `project_`: `project` is for pinned values.
  #[flat]
  enum Plan {
    Scan(u32),
    Project(Near<Plan>, u32),
  }

  /// Constructors named with `#[flat(rename = "..")]`: two variants that would both have
  /// `http_server`, and one that would have `project_`.
  #[flat]
  #[derive(Debug, PartialEq)]
  enum Host {
    HTTPServer(u8),
    #[flat(rename = "http_server_v2")]
    HttpServer(u16),
    #[flat(rename = "plan")]
    Project { id: u32, next: Option<Near<Host>> },
    Idle,
  }

  /// A type parameter called like one the generated code could use.
  #[flat]
  struct Wrap<E> {
    value: Near<E>,
    count: u32,
  }

  #[flat]
  struct Tree {
    id: u32,
    kids: Lst<Self>,
  }

  #[test]
  fn reserved_names_and_self() {
    let mut led = Region::new_in(Store::default(), Led::new(1, 2, false, "a"));
    let l = led.edit();
    *l.pin_() += 1;
    *l.update_() += 1;
    *l.set_() = true;
    l.replace_().set("b");
    assert_eq!(*l.pin().project().pin, 2);
    assert_eq!((led.pin, led.update, led.set, &*led.replace), (2, 3, true, "b"));

    let mut plan = Region::new_in(Store::default(), Plan::project_(Plan::Scan(1), 2));
    if let PlanPin::Project(_, mut n) = plan.edit().pin().project() {
      *n += 1;
    }
    assert!(matches!(&*plan, Plan::Project(input, 3) if matches!(**input, Plan::Scan(1))));

    let mut host = Region::new_in(Store::default(), Host::plan(1, Near::new(Host::http_server_v2(2))));
    if let HostMut::Project(p) = host.edit().as_mut() {
      p.next().insert(Host::http_server(3));
    }
    let Host::Project { id: 1, next: Some(next) } = &*host else { unreachable!() };
    assert_eq!(**next, Host::HTTPServer(3));
    assert_ne!(*Region::new_in(Store::default(), Host::Idle), **next);

    let mut wrap = Region::new_in(Store::default(), Wrap::<u8>::new(7, 1));
    *wrap.edit().value() += 1;
    assert_eq!((*wrap.value, wrap.count), (8, 1));

    let mut tree = Region::new_in(Store::default(), Tree::new(0, Lst::new([Tree::new(1, Lst::new(()))])));
    tree.edit().kids().items().push(Tree::new(2, Lst::new(())));
    assert_eq!(tree.kids.items.iter().map(|t| t.id).sum::<u32>(), 3);

    // A region can be moved whatever it holds.
    unpin::<Region<Tree>>();
    unpin::<Region<Plan>>();
  }

  const fn unpin<T: Unpin>() {}
}

// ---------------------------------------------------------------------------------------------
// The alignment of a region: that of a `u64`, whatever the target

/// As aligned as a stored type may be: as a `u64`, 8 bytes on x86-64, 4 on `i686`.
#[flat]
#[derive(Debug, PartialEq, Clone, Copy)]
struct Word {
  value: u64,
}

/// Words behind pointers of every width: positions aligned for them, at addresses of a buffer
/// aligned as a `u64` only. With `i16`, the elements of a list are 4 bytes past a multiple of 8;
/// with `i64`, at a multiple of 8, which is all the alignment a `u64` has on x86-64.
#[flat]
#[derive(Debug, PartialEq)]
struct Words {
  id: u8,
  one: Near<Word>,
  text: Near<str>,
  short: Near<[Word], i16>,
  wide: Near<[Word], i64>,
}

/// The bytes of a region are accepted by `view` on the target that made it, in every storage,
/// and every reference into it is aligned (Miri runs this on `i686`, where a `u64` is aligned to
/// 4). Several regions, so that the heap gives addresses of every kind.
#[test]
fn a_region_is_aligned_as_its_words() {
  assert_eq!(align_of::<Word>(), align_of::<u64>());
  let regions: Vec<_> = (0..8u8)
    .map(|i| {
      let w = |v: u64| Word { value: v + u64::from(i) };
      Region::new_in(Store::default(), Words::new(i, w(1), "x", [w(2), w(3)], [w(4)]))
    })
    .collect();
  for (i, r) in regions.iter().enumerate() {
    let bytes = r.as_bytes().unwrap();
    assert!(bytes.as_ptr().addr().is_multiple_of(align_of::<u64>()));
    let v = nearest::view::<Words>(bytes).unwrap();
    assert_eq!(v, &**r);
    let i = i as u64;
    let values = [v.one.value, v.short[0].value, v.short[1].value, v.wide[0].value];
    assert_eq!(values, [1 + i, 2 + i, 3 + i, 4 + i]);
    // SAFETY: `view` accepted them, at this address.
    assert_eq!(unsafe { nearest::view_unchecked::<Words>(bytes) }, &**r);
    // Read in place where they are, and loaded elsewhere.
    assert_eq!(*Region::<Words>::from_bytes(bytes).unwrap(), **r);
  }
  // A single `Word`, the root, and lists of them edited in place.
  let mut r = Region::new_in(Store::default(), Lst::new([Word { value: 1 }]));
  for v in 2..10 {
    r.edit().items().push(Word { value: v });
  }
  assert!(r.items.iter().map(|w| w.value).eq(1..10));
  let copy = Region::new_in(Store::default(), &*r);
  assert_eq!(nearest::view::<Lst<Word>>(copy.as_bytes().unwrap()).map(|l| l.items.len()), Ok(9));
}
