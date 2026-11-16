#![feature(pin_ergonomics)]
#![expect(incomplete_features, reason = "pin_ergonomics")]
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
    r.edit(|old, slot| slot.put(Expr::add(Near::new(old), Near::new(Expr::Lit(i64::from(i))))));
  }
  Region::new(&*r)
}

fn build(c: &mut Criterion) {
  let mut g = c.benchmark_group("build");
  g.bench_function("list 100k u32", |b| {
    b.iter(|| Region::new(Inst::new(0, Near::from_iter(0..black_box(100_000u32)))));
  });
  g.bench_function("func 100 blocks x 100 insts", |b| b.iter(|| func(black_box(100), 100)));
  g.bench_function("10k Near::new(u64)", |b| {
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
  f.edit(|root, slot| {
    let Func { blocks, .. } = root;
    slot.at(blocks, |old, new| new.put(Near::from_iter(old.into_iter().rev())));
  });
  g.bench_function("compact after edit", |b| b.iter(|| Region::new(&**black_box(&f))));
  g.bench_function("to_bytes", |b| b.iter(|| black_box(&f).to_bytes()));
  g.finish();
}

fn edit(c: &mut Criterion) {
  let mut g = c.benchmark_group("edit");
  let mut f = func(100, 100);
  g.bench_function("in place: bump every op", |b| {
    b.iter(|| {
      f.edit(|root, _| {
        let Func { blocks, .. } = root;
        for block in blocks {
          let Block { insts, .. } = block;
          for inst in insts {
            let Inst { op, .. } = inst;
            *op.get_mut() = op.wrapping_add(1);
          }
        }
      });
    });
  });
  g.bench_function("structural: reverse blocks (moves)", |b| {
    b.iter_batched_ref(
      || func(100, 100),
      |f| {
        f.edit(|root, slot| {
          let Func { blocks, .. } = root;
          slot.at(blocks, |old, new| new.put(Near::from_iter(old.into_iter().rev())));
        });
      },
      criterion::BatchSize::LargeInput,
    );
  });
  g.bench_function("structural: filter insts in every block", |b| {
    b.iter_batched_ref(
      || func(100, 100),
      |f| {
        f.edit(|root, slot| {
          let Func { blocks, .. } = root;
          for block in blocks {
            let Block { insts, .. } = block;
            slot.at(insts, |old, new| {
              for inst in old {
                if inst.op % 2 == 0 {
                  new.push(inst);
                }
              }
            });
          }
        });
      },
      criterion::BatchSize::LargeInput,
    );
  });
  g.bench_function("structural: 100 single appends", |b| {
    b.iter_batched_ref(
      || Region::new(Inst::new(0, ())),
      |r| {
        for i in 0..100u32 {
          r.edit(|root, slot| {
            let Inst { args, .. } = root;
            slot.at(args, |old, new| {
              for x in old {
                new.push(x);
              }
              new.push(i);
            });
          });
        }
      },
      criterion::BatchSize::SmallInput,
    );
  });
  g.finish();
}

criterion_group!(benches, build, read, clone_and_compact, edit);
criterion_main!(benches);
