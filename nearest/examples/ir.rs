//! A small IR (a function of blocks of instructions) transformed by compiler-style passes.
//!
//! - constant folding rewrites instructions in place, with no new memory;
//! - dead-code elimination rebuilds each block's instruction list, moving the survivors;
//! - a final pass renames the function and appends a block.
//!
//! An interpreter checks that every pass keeps the function's result.
#![feature(pin_ergonomics)]
#![expect(incomplete_features, reason = "pin_ergonomics")]

use std::collections::{HashMap, HashSet};

use nearest::{Near, Region, flat};

#[flat]
struct Func {
  name: Near<str>,
  blocks: Near<[Block]>,
}

#[flat]
struct Block {
  insts: Near<[Inst]>,
  term: Term,
}

/// `v{dst} = op`
#[flat]
#[derive(Clone, Copy)]
struct Inst {
  dst: u32,
  op: Op,
}

#[flat]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
  Const(i64),
  Add(u32, u32),
  Mul(u32, u32),
  Copy(u32),
}

#[flat]
#[derive(Clone, Copy)]
enum Term {
  /// Jump to a block.
  Jump(u32),
  /// Return a value.
  Ret(u32),
}

fn run(f: &Func) -> i64 {
  let mut vals = vec![0i64; 64];
  let mut block = 0;
  loop {
    let b = &f.blocks[block];
    for inst in &b.insts {
      let v = |r: u32| vals[r as usize];
      vals[inst.dst as usize] = match inst.op {
        Op::Const(c) => c,
        Op::Add(x, y) => v(x) + v(y),
        Op::Mul(x, y) => v(x) * v(y),
        Op::Copy(x) => v(x),
      };
    }
    match b.term {
      Term::Jump(to) => block = to as usize,
      Term::Ret(r) => return vals[r as usize],
    }
  }
}

fn count(f: &Func) -> usize {
  f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// Folds instructions whose operands are constants, in place.
fn fold_constants(r: &mut Region<Func>) -> usize {
  let mut known = HashMap::new();
  let mut folded = 0;
  r.edit(|root, _| {
    let Func { blocks, .. } = root;
    for block in blocks {
      let Block { insts, .. } = block;
      // `Inst` holds no pointer, so the list is plain data: a real `&mut [Inst]`.
      for inst in insts.as_pin_mut().get_mut() {
        let c = |r| known.get(&r).copied();
        let value = match inst.op {
          Op::Const(c) => Some(c),
          Op::Add(x, y) => c(x).zip(c(y)).map(|(x, y)| x + y),
          Op::Mul(x, y) => c(x).zip(c(y)).map(|(x, y)| x * y),
          Op::Copy(x) => c(x),
        };
        if let Some(v) = value {
          known.insert(inst.dst, v);
          if inst.op != Op::Const(v) {
            inst.op = Op::Const(v);
            folded += 1;
          }
        }
      }
    }
  });
  folded
}

/// Removes instructions whose result is never used.
fn eliminate_dead_code(r: &mut Region<Func>) -> usize {
  let mut used = HashSet::new();
  for b in r.blocks.iter() {
    if let Term::Ret(v) = b.term {
      used.insert(v);
    }
    for inst in &b.insts {
      match inst.op {
        Op::Add(x, y) | Op::Mul(x, y) => used.extend([x, y]),
        Op::Copy(x) => {
          used.insert(x);
        }
        Op::Const(_) => {}
      }
    }
  }
  let before = count(r);
  r.edit(|root, slot| {
    let Func { blocks, .. } = root;
    for block in blocks {
      let Block { insts, .. } = block;
      // The survivors move into a new list; it may be empty, which `push` alone cannot say.
      slot.at(insts, |old, new| {
        new.put(Near::from_iter(old.into_iter().filter(|inst| used.contains(&inst.dst))));
      });
    }
  });
  before - count(r)
}

fn main() {
  // entry: v0 = 2; v1 = 3; v2 = v0 * v1; v3 = 100; jump exit
  // exit:  v4 = v2 + v0; v5 = v3 * v3; v6 = copy v4; v7 = v6 + v1; ret v7
  let mut f = Region::new(Func::new(
    "f",
    (
      Block::new(
        [
          Inst { dst: 0, op: Op::Const(2) },
          Inst { dst: 1, op: Op::Const(3) },
          Inst { dst: 2, op: Op::Mul(0, 1) },
          Inst { dst: 3, op: Op::Const(100) },
        ],
        Term::Jump(1),
      ),
      Block::new(
        [
          Inst { dst: 4, op: Op::Add(2, 0) },
          Inst { dst: 5, op: Op::Mul(3, 3) },
          Inst { dst: 6, op: Op::Copy(4) },
          Inst { dst: 7, op: Op::Add(6, 1) },
        ],
        Term::Ret(7),
      ),
    ),
  ));
  let expected = run(&f);
  assert_eq!(expected, 11);
  println!("{}: {} instructions, {} bytes, returns {expected}", &*f.name, count(&f), f.byte_len());

  let folded = fold_constants(&mut f);
  assert_eq!(folded, 5);
  assert_eq!(run(&f), expected);
  let removed = eliminate_dead_code(&mut f);
  assert_eq!(removed, 7);
  assert_eq!(run(&f), expected);
  assert_eq!(f.blocks[1].insts.iter().map(|i| i.op).collect::<Vec<_>>(), [Op::Const(11)]);
  println!(
    "folded {folded}, removed {removed}: {} instructions, {} bytes",
    count(&f),
    f.byte_len()
  );

  // Rename, and route the exit through a new block that doubles the result.
  f.edit(|root, slot| {
    let Func { name, blocks } = root;
    slot.at(name, |_, new| new.put("f.opt"));
    slot.at(blocks, |old, new| {
      for (i, block) in old.into_iter().enumerate() {
        if i == 1 {
          // Change the old block in place, then move it.
          let mut block = block;
          let Block { term, .. } = block.as_mut();
          let Term::Ret(r) = *term else { unreachable!() };
          *term.get_mut() = Term::Jump(2);
          new.push(block);
          new.push(Block::new([Inst { dst: 9, op: Op::Add(r, r) }], Term::Ret(9)));
        } else {
          new.push(block);
        }
      }
    });
  });
  assert_eq!((&*f.name, f.blocks.len(), run(&f)), ("f.opt", 3, 2 * expected));

  let compact = Region::new(&*f);
  println!(
    "{}: returns {}, {} bytes after compaction",
    &*compact.name,
    run(&compact),
    compact.byte_len()
  );
  assert!(compact.byte_len() < f.byte_len());
}
