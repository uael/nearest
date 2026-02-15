#![feature(offset_of_enum)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nearest::{Flat, Near, NearList, Region, empty};

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
struct Type(u8);

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
struct Symbol(u32);

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum Value {
  Const(u32),
  Type(Type),
}

#[derive(Flat, Debug)]
struct Inst {
  op: u16,
  typ: Type,
  args: NearList<Value>,
}

#[derive(Flat, Debug)]
struct Jmp {
  args: NearList<Value>,
  target: Near<Block>,
}

#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum Term {
  Ret { values: NearList<Value> },
  Jmp(Jmp),
}

#[derive(Flat, Debug)]
struct Block {
  name: Symbol,
  params: NearList<(Symbol, Type)>,
  insts: NearList<Inst>,
  term: Term,
}

#[derive(Flat, Debug)]
struct Func {
  name: Symbol,
  entry: Near<Block>,
}

#[derive(Flat, Debug)]
struct ListBlock {
  name: Symbol,
  items: NearList<Value>,
}

fn build_func() -> Region<Func> {
  Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      [Inst::make(1, Type(0), [Value::Const(42)])],
      Term::make_jmp(Jmp::make(
        [Value::Const(1)],
        Block::make(
          Symbol(0),
          empty(),
          [Inst::make(1, Type(0), [Value::Const(42)])],
          Term::make_ret([Value::Const(42)]),
        ),
      )),
    ),
  ))
}

fn bench_build(c: &mut Criterion) {
  c.bench_function("region_build", |b| {
    b.iter(|| black_box(build_func()));
  });
}

fn bench_clone(c: &mut Criterion) {
  let region = build_func();
  c.bench_function("region_clone", |b| {
    b.iter(|| black_box(region.clone()));
  });
}

fn bench_trim(c: &mut Criterion) {
  c.bench_function("region_trim_noop", |b| {
    b.iter_batched(build_func, |mut r| r.trim(), criterion::BatchSize::SmallInput);
  });
}

fn bench_push_front(c: &mut Criterion) {
  c.bench_function("push_front_100", |b| {
    b.iter_batched(
      || Region::new(ListBlock::make(Symbol(1), empty())),
      |mut region| {
        for i in 0..100u32 {
          region.session(|s| {
            let items = s.nav(s.root(), |b| &b.items);
            s.push_front(items, Value::Const(i));
          });
        }
        region
      },
      criterion::BatchSize::SmallInput,
    );
  });
}

fn bench_extend_list(c: &mut Criterion) {
  c.bench_function("extend_list_100", |b| {
    b.iter_batched(
      || Region::new(ListBlock::make(Symbol(1), empty())),
      |mut region| {
        region.session(|s| {
          let items = s.nav(s.root(), |b| &b.items);
          s.extend_list(items, (0..100u32).map(Value::Const));
        });
        region
      },
      criterion::BatchSize::SmallInput,
    );
  });
}

fn bench_iterate(c: &mut Criterion) {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, (0..1000u32).map(Value::Const));
  });

  c.bench_function("iterate_1000", |b| {
    b.iter(|| {
      let block: &ListBlock = &region;
      let mut sum = 0u32;
      for v in &block.items {
        if let Value::Const(n) = v {
          sum += n;
        }
      }
      black_box(sum)
    });
  });
}

fn bench_graft(c: &mut Criterion) {
  let callee: Region<Block> =
    Region::new(Block::make(Symbol(99), empty(), empty(), Term::make_ret([Value::Const(77)])));

  c.bench_function("graft", |b| {
    b.iter_batched(
      build_func,
      |mut caller| {
        caller.session(|s| {
          black_box(s.graft(&callee));
        });
        caller
      },
      criterion::BatchSize::SmallInput,
    );
  });
}

fn bench_re_splice_list(c: &mut Criterion) {
  c.bench_function("re_splice_list_100", |b| {
    b.iter_batched(
      || {
        let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
        region.session(|s| {
          let items = s.nav(s.root(), |b| &b.items);
          s.extend_list(items, (0..100u32).map(Value::Const));
        });
        region
      },
      |mut region| {
        region.session(|s| {
          let items = s.nav(s.root(), |b| &b.items);
          let refs = s.list_refs(items);
          s.re_splice_list(items, &refs);
        });
        region
      },
      criterion::BatchSize::SmallInput,
    );
  });
}

fn bench_validate(c: &mut Criterion) {
  let region = build_func();
  let bytes = region.as_bytes().to_vec();
  c.bench_function("validate", |b| {
    b.iter(|| {
      black_box(Func::validate(0, black_box(&bytes)).unwrap());
    });
  });
}

fn bench_from_bytes(c: &mut Criterion) {
  let region = build_func();
  let bytes = region.as_bytes().to_vec();
  c.bench_function("from_bytes", |b| {
    b.iter(|| {
      black_box(Region::<Func>::from_bytes(black_box(&bytes)).unwrap());
    });
  });
}

fn bench_as_bytes_roundtrip(c: &mut Criterion) {
  let region = build_func();
  c.bench_function("as_bytes_roundtrip", |b| {
    b.iter(|| {
      let bytes = black_box(&region).as_bytes();
      black_box(Region::<Func>::from_bytes(bytes).unwrap());
    });
  });
}

criterion_group!(
  benches,
  bench_build,
  bench_clone,
  bench_trim,
  bench_push_front,
  bench_extend_list,
  bench_iterate,
  bench_graft,
  bench_re_splice_list,
  bench_validate,
  bench_from_bytes,
  bench_as_bytes_roundtrip,
);
criterion_main!(benches);
