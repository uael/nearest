#![expect(missing_docs, reason = "criterion generates undocumented items")]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use nearest::{Near, Region, flat};

#[flat]
struct Func {
  name: Near<str>,
  blocks: Near<[Block]>,
}

#[flat]
struct Block {
  id: u32,
  insts: Near<[Inst]>,
}

#[flat]
struct Inst {
  op: u8,
  args: Near<[u32]>,
}

#[flat]
enum Expr {
  Lit(i64),
  Add(Near<Expr>, Near<Expr>),
}

/// A table: numbers, and names behind a flag.
#[flat]
struct Table {
  numbers: Near<[u32]>,
  rows: Near<[Row]>,
}

#[flat]
struct Row {
  public: bool,
  name: Near<str>,
}

/// A balanced tree of `2^depth` leaves.
fn tree(depth: u32) -> Region<Expr> {
  fn build(depth: u32) -> Region<Expr> {
    if depth == 0 {
      return Region::new(Expr::Lit(1));
    }
    let (a, b) = (build(depth - 1), build(depth - 1));
    Region::new(Expr::add(&*a, &*b))
  }
  build(depth)
}

/// `blocks` blocks of `insts` instructions with 3 arguments each.
fn func(blocks: u32, insts: u32) -> Region<Func> {
  Region::new(Func::new(
    "bench",
    Near::from_iter((0..blocks).map(|b| {
      Block::new(b, Near::from_iter((0..insts).map(|i| Inst::new(i as u8, [i, i + 1, i + 2]))))
    })),
  ))
}

/// A left-leaning sum of `depth` additions, built by wrapping the root.
fn expr(depth: u32) -> Region<Expr> {
  let mut r = Region::new(Expr::Lit(1));
  for i in 0..depth {
    let e = r.edit();
    let old = e.replace(Expr::Lit(0));
    e.set(Expr::add(old, Expr::Lit(i64::from(i))));
  }
  Region::new(&*r)
}

fn build(c: &mut Criterion) {
  let mut g = c.benchmark_group("build");
  g.bench_function("list 100k u32", |b| {
    b.iter(|| Region::new(Inst::new(0, Near::from_iter(0..black_box(100_000u32)))));
  });
  g.bench_function("func 100 blocks x 100 insts", |b| b.iter(|| func(black_box(100), 100)));
  g.bench_function("10k small blocks", |b| {
    b.iter(|| {
      Region::new(Func::new(
        "",
        Near::from_iter((0..black_box(10_000u32)).map(|i| Block::new(i, ()))),
      ))
    });
  });
  g.finish();
}

fn read(c: &mut Criterion) {
  let f = func(100, 100);
  let e = expr(1000);
  let mut g = c.benchmark_group("read");
  g.bench_function("sum all args", |b| {
    b.iter(|| {
      black_box(&f)
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .map(|i| i.args.iter().map(|&a| u64::from(a)).sum::<u64>())
        .sum::<u64>()
    });
  });
  g.bench_function("eval expr depth 1000", |b| {
    fn eval(e: &Expr) -> i64 {
      match e {
        Expr::Lit(n) => *n,
        Expr::Add(a, b) => eval(a) + eval(b),
      }
    }
    b.iter(|| eval(black_box(&e)));
  });
  g.finish();
}

fn clone_and_compact(c: &mut Criterion) {
  let mut f = func(100, 100);
  let mut g = c.benchmark_group("copy");
  g.bench_function("clone", |b| b.iter(|| black_box(&f).clone()));
  g.bench_function("compact", |b| b.iter(|| Region::new(&**black_box(&f))));
  reverse(&mut f);
  g.bench_function("compact after edit", |b| b.iter(|| Region::new(&**black_box(&f))));
  g.finish();
}

/// Reverses the blocks, moving each one.
fn reverse(f: &mut Region<Func>) {
  let blocks = f.edit().blocks();
  let n = blocks.len();
  for i in 0..n / 2 {
    blocks.swap(i, n - 1 - i);
  }
}

fn edit(c: &mut Criterion) {
  let mut g = c.benchmark_group("edit");
  let mut f = func(100, 100);
  g.bench_function("in place: bump every op", |b| {
    b.iter(|| {
      let blocks = f.edit().blocks();
      for i in 0..blocks.len() {
        let insts = blocks.at(i).insts();
        for j in 0..insts.len() {
          let mut op = insts.at(j).op();
          *op = op.wrapping_add(1);
        }
      }
    });
  });
  g.bench_function("structural: reverse blocks (moves)", |b| {
    b.iter_batched_ref(|| func(100, 100), reverse, criterion::BatchSize::LargeInput);
  });
  g.bench_function("structural: retain insts in every block", |b| {
    b.iter_batched_ref(
      || func(100, 100),
      |f| {
        let blocks = f.edit().blocks();
        for i in 0..blocks.len() {
          blocks.at(i).insts().retain(|inst| inst.op % 2 == 0);
        }
      },
      criterion::BatchSize::LargeInput,
    );
  });
  g.bench_function("structural: move 100 elements between two lists", |b| {
    b.iter_batched_ref(
      || func(2, 100),
      |f| {
        let blocks = f.edit().blocks();
        for _ in 0..100 {
          let inst = blocks.at(0).insts().remove(0);
          blocks.at(1).insts().push(inst);
        }
      },
      criterion::BatchSize::SmallInput,
    );
  });
  g.bench_function("structural: 100 pushes", |b| {
    b.iter_batched_ref(
      || Region::new(Inst::new(0, ())),
      |r| {
        for i in 0..100u32 {
          r.edit().args().push(i);
        }
      },
      criterion::BatchSize::SmallInput,
    );
  });
  g.bench_function("structural: wrap the root 100 times", |b| {
    b.iter_batched_ref(
      || Region::new(Expr::Lit(0)),
      |r| {
        let e = r.edit();
        for i in 0..100 {
          let old = e.replace(Expr::Lit(0));
          e.set(Expr::add(old, Expr::Lit(i)));
        }
      },
      criterion::BatchSize::SmallInput,
    );
  });
  g.finish();
}

fn bytes(c: &mut Criterion) {
  let f = func(100, 100);
  let image = f.as_bytes().unwrap().to_vec();
  let mut g = c.benchmark_group("bytes");
  g.bench_function("from_bytes 100 blocks x 100 insts", |b| {
    b.iter(|| Region::<Func>::from_bytes(black_box(&image)).unwrap());
  });
  // The region's own buffer is 8-aligned.
  let aligned = f.as_bytes().unwrap();
  g.bench_function("view 100 blocks x 100 insts", |b| {
    b.iter(|| nearest::view::<Func>(black_box(aligned)).unwrap().blocks.len());
  });
  // Enums and pointers, nothing else: what the check of a value costs.
  let t = tree(16);
  let aligned = t.as_bytes().unwrap();
  g.bench_function("view tree of 2^16 leaves", |b| {
    b.iter(|| matches!(nearest::view::<Expr>(black_box(aligned)).unwrap(), Expr::Add(..)));
  });
  // Numbers, which are not looked at, and rows that are.
  let names: Vec<String> = (0..10_000).map(|i| format!("row {i}")).collect();
  let table = Region::new(Table::new(
    Near::from_iter(0..1_000_000u32),
    Near::from_iter(names.iter().enumerate().map(|(i, n)| Row::new(i % 2 == 0, n.as_str()))),
  ));
  let aligned = table.as_bytes().unwrap();
  g.bench_function("view table of 1M numbers, 10k rows", |b| {
    b.iter(|| nearest::view::<Table>(black_box(aligned)).unwrap().rows.len());
  });
  g.finish();
}

criterion_group!(benches, build, read, clone_and_compact, edit, bytes);
criterion_main!(benches);
