#![feature(offset_of_enum)]

use nearest::{Flat, Near, NearList, Ref, Region, ValidateError, empty};

// --- IR type definitions using derive(Flat) ---

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
struct Type(u8);

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
struct Symbol(u32);

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, u8)]
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

// Use repr(C, u8) so we can compute variant field offsets portably.
#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via raw byte emission, not Rust constructors")]
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

// Generic struct — the derive must thread CT through the builder with PhantomData.
#[derive(Flat, Debug)]
struct Signature<CT: Flat> {
  params: NearList<Type>,
  returns: NearList<Type>,
  custom: NearList<CT>,
}

// Generic enum — variant emitters must also carry outer generics.
#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum Wrapper<CT: Flat> {
  Empty,
  WithSig { sig: Near<Signature<CT>> },
  Pair(CT, u32),
}

// ===========================================================================
// Builder API tests — zero Pos, zero unsafe, zero Emitter in user code
// ===========================================================================

/// Build a simple function with two blocks: entry jumps to exit, exit returns.
fn build_simple_func(name: u32) -> Region<Func> {
  Region::new(Func::make(
    Symbol(name),
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

#[test]
fn emit_build_and_traverse() {
  let region = build_simple_func(1);
  let func: &Func = &region;
  assert_eq!(func.name, Symbol(1));

  let block = func.entry.get();
  assert_eq!(block.name, Symbol(0));
  assert!(block.params.is_empty());
  assert_eq!(block.insts.len(), 1);

  let inst = &block.insts[0];
  assert_eq!(inst.op, 1);
  assert_eq!(inst.typ, Type(0));
  assert_eq!(inst.args.len(), 1);
  assert_eq!(inst.args[0], Value::Const(42));

  match &block.term {
    Term::Jmp(jmp) => {
      assert_eq!(jmp.args.len(), 1);
      assert_eq!(jmp.args[0], Value::Const(1));
      let exit = jmp.target.get();
      match &exit.term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(42));
        }
        Term::Jmp(_) => panic!("expected Ret in exit block"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp, got Ret"),
  }
}

#[test]
fn emit_multi_block() {
  // Multi-block function built entirely with the emitter API.
  // Entry block jumps to exit block; exit block returns.
  let region: Region<Func> = Region::new(Func::make(
    Symbol(10),
    Block::make(
      Symbol(0),
      empty(),
      empty(),
      Term::make_jmp(Jmp::make(
        empty(),
        Block::make(Symbol(1), empty(), empty(), Term::make_ret([Value::Const(99)])),
      )),
    ),
  ));

  let func: &Func = &region;
  assert_eq!(func.name, Symbol(10));

  let entry = func.entry.get();
  assert_eq!(entry.name, Symbol(0));
  assert!(entry.insts.is_empty());

  match &entry.term {
    Term::Jmp(jmp) => {
      let exit = jmp.target.get();
      assert_eq!(exit.name, Symbol(1));
      assert!(jmp.args.is_empty());

      match &exit.term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(99));
        }
        Term::Jmp(_) => panic!("exit block should have Ret"),
      }
    }
    Term::Ret { .. } => panic!("entry block should have Jmp"),
  }
}

#[test]
fn clone_preserves_offsets() {
  let region = build_simple_func(2);
  let cloned = region.clone();

  let func1: &Func = &region;
  let func2: &Func = &cloned;

  assert_eq!(func1.name, func2.name);

  let block1 = func1.entry.get();
  let block2 = func2.entry.get();
  assert_eq!(block1.name, block2.name);
  assert_eq!(block1.insts.len(), block2.insts.len());
  assert_eq!(block1.insts[0].op, block2.insts[0].op);
  assert_eq!(block1.insts[0].args[0], block2.insts[0].args[0]);

  drop(cloned);
  assert_eq!(func1.name, Symbol(2));
}

#[test]
fn near_list_iteration() {
  let region = build_simple_func(4);
  let func: &Func = &region;
  let block = func.entry.get();

  let args: Vec<&Value> = block.insts[0].args.iter().collect();
  assert_eq!(args.len(), 1);
  assert_eq!(*args[0], Value::Const(42));

  let mut count = 0;
  for val in &block.insts[0].args {
    assert_eq!(*val, Value::Const(42));
    count += 1;
  }
  assert_eq!(count, 1);

  assert_eq!(block.insts[0].args.len(), 1);
  assert_eq!(block.insts[0].args[0], Value::Const(42));
}

#[test]
fn region_debug() {
  let region = build_simple_func(42);
  let debug_str = format!("{region:?}");
  assert!(debug_str.contains("Region"));
}

#[test]
fn empty_slices() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(99),
    Block::make(Symbol(0), empty(), empty(), Term::make_ret(empty())),
  ));

  let func: &Func = &region;
  let block = func.entry.get();
  assert!(block.params.is_empty());
  assert!(block.insts.is_empty());

  match &block.term {
    Term::Ret { values } => assert!(values.is_empty()),
    Term::Jmp(_) => panic!("expected Ret"),
  }
}

#[test]
fn multiple_instructions() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(5),
    Block::make(
      Symbol(0),
      [(Symbol(1), Type(0)), (Symbol(2), Type(1))],
      [
        Inst::make(1, Type(0), [Value::Const(10), Value::Const(11)]),
        Inst::make(2, Type(1), [Value::Const(20), Value::Type(Type(0))]),
        Inst::make(3, Type(0), [Value::Const(30), Value::Const(31)]),
      ],
      Term::make_ret([Value::Const(10), Value::Const(20)]),
    ),
  ));

  let func: &Func = &region;
  assert_eq!(func.name, Symbol(5));

  let block = func.entry.get();
  assert_eq!(block.params.len(), 2);
  assert_eq!(block.params[0], (Symbol(1), Type(0)));
  assert_eq!(block.params[1], (Symbol(2), Type(1)));

  assert_eq!(block.insts.len(), 3);
  assert_eq!(block.insts[0].op, 1);
  assert_eq!(block.insts[0].args.len(), 2);
  assert_eq!(block.insts[0].args[0], Value::Const(10));
  assert_eq!(block.insts[1].op, 2);
  assert_eq!(block.insts[1].args.len(), 2);
  assert_eq!(block.insts[1].args[1], Value::Type(Type(0)));
  assert_eq!(block.insts[2].op, 3);
  assert_eq!(block.insts[2].args.len(), 2);
}

// ===========================================================================
// Session API tests — navigate with immutable closures, mutate via Ref
// ===========================================================================

#[test]
fn session_roundtrip() {
  let mut region = build_simple_func(7);
  region.session(|_s| {
    // Just open and drop — no mutation.
  });

  let func: &Func = &region;
  assert_eq!(func.name, Symbol(7));
  let block = func.entry.get();
  assert_eq!(block.insts[0].op, 1);
  assert_eq!(block.insts[0].args[0], Value::Const(42));
}

// ===========================================================================
// Splice API tests — structural mutations: redirect Near/NearList
// ===========================================================================

#[test]
fn session_splice_near_replaces_jmp_target() {
  // Build: entry → jmp([Const(1)]) → exit(ret [Const(42)])
  let mut region = build_simple_func(1);

  // Replace the exit block with a completely new block.
  region.session(|s| {
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(
      jmp_target,
      Block::make(
        Symbol(99),
        empty(),
        [Inst::make(7, Type(1), [Value::Const(77)])],
        Term::make_ret([Value::Const(99)]),
      ),
    );
  });

  let func: &Func = &region;
  assert_eq!(func.name, Symbol(1)); // root unchanged

  let entry = func.entry.get();
  match &entry.term {
    Term::Jmp(jmp) => {
      // jmp args unchanged
      assert_eq!(jmp.args.len(), 1);
      assert_eq!(jmp.args[0], Value::Const(1));

      let exit = jmp.target.get();
      assert_eq!(exit.name, Symbol(99));
      assert_eq!(exit.insts.len(), 1);
      assert_eq!(exit.insts[0].op, 7);
      assert_eq!(exit.insts[0].typ, Type(1));
      assert_eq!(exit.insts[0].args[0], Value::Const(77));
      match &exit.term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(99));
        }
        Term::Jmp(_) => panic!("expected Ret in new exit block"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

#[test]
fn session_splice_list_replaces_insts() {
  // Build a function with 1 instruction.
  let mut region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      [Inst::make(1, Type(0), [Value::Const(42)])],
      Term::make_ret([Value::Const(42)]),
    ),
  ));

  // Replace with 3 instructions (growing the buffer).
  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.entry.insts);
    s.splice_list(
      insts,
      [
        Inst::make(10, Type(0), vec![Value::Const(1)]),
        Inst::make(20, Type(1), vec![Value::Const(2), Value::Const(3)]),
        Inst::make(30, Type(0), vec![]),
      ],
    );
  });

  let func: &Func = &region;
  let block = func.entry.get();

  assert_eq!(block.insts.len(), 3);

  assert_eq!(block.insts[0].op, 10);
  assert_eq!(block.insts[0].args.len(), 1);
  assert_eq!(block.insts[0].args[0], Value::Const(1));

  assert_eq!(block.insts[1].op, 20);
  assert_eq!(block.insts[1].args.len(), 2);
  assert_eq!(block.insts[1].args[0], Value::Const(2));
  assert_eq!(block.insts[1].args[1], Value::Const(3));

  assert_eq!(block.insts[2].op, 30);
  assert!(block.insts[2].args.is_empty());

  // Term unchanged.
  match &block.term {
    Term::Ret { values } => {
      assert_eq!(values.len(), 1);
      assert_eq!(values[0], Value::Const(42));
    }
    Term::Jmp(_) => panic!("expected Ret"),
  }
}

#[test]
fn session_splice_list_to_empty() {
  let mut region = build_simple_func(1);

  // Replace entry block instructions with an empty slice.
  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.entry.insts);
    s.splice_list(insts, empty());
  });

  let func: &Func = &region;
  assert!(func.entry.get().insts.is_empty());
}

#[test]
fn session_splice_list_params() {
  // Build a block with 2 params, then replace them with 3.
  let mut region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      [(Symbol(1), Type(0)), (Symbol(2), Type(1))],
      empty(),
      Term::make_ret(empty()),
    ),
  ));

  region.session(|s| {
    let params = s.nav(s.root(), |f| &f.entry.params);
    s.splice_list(params, [(Symbol(10), Type(0)), (Symbol(11), Type(1)), (Symbol(12), Type(2))]);
  });

  let func: &Func = &region;
  let block = func.entry.get();
  assert_eq!(block.params.len(), 3);
  assert_eq!(block.params[0], (Symbol(10), Type(0)));
  assert_eq!(block.params[1], (Symbol(11), Type(1)));
  assert_eq!(block.params[2], (Symbol(12), Type(2)));
}

#[test]
fn session_inline_callee_block() {
  // Simulate function inlining: replace a jmp target with a new block
  // that contains the "callee's" instructions.

  // Caller: entry(empty) → jmp → exit(ret 0)
  let mut region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      empty(),
      Term::make_jmp(Jmp::make(
        empty(),
        Block::make(Symbol(1), empty(), empty(), Term::make_ret([Value::Const(0)])),
      )),
    ),
  ));

  // "Inline" the callee: replace exit block with callee's body.
  region.session(|s| {
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(
      jmp_target,
      Block::make(
        Symbol(100),
        [(Symbol(10), Type(0))],
        [
          Inst::make(1, Type(0), vec![Value::Const(10)]),
          Inst::make(2, Type(0), vec![Value::Const(20), Value::Type(Type(1))]),
        ],
        Term::make_ret([Value::Const(42)]),
      ),
    );
  });

  let func: &Func = &region;
  assert_eq!(func.name, Symbol(1));

  let entry = func.entry.get();
  assert_eq!(entry.name, Symbol(0));
  assert!(entry.insts.is_empty());

  match &entry.term {
    Term::Jmp(jmp) => {
      let inlined = jmp.target.get();
      assert_eq!(inlined.name, Symbol(100));
      assert_eq!(inlined.params.len(), 1);
      assert_eq!(inlined.params[0], (Symbol(10), Type(0)));
      assert_eq!(inlined.insts.len(), 2);
      assert_eq!(inlined.insts[0].op, 1);
      assert_eq!(inlined.insts[0].args[0], Value::Const(10));
      assert_eq!(inlined.insts[1].op, 2);
      assert_eq!(inlined.insts[1].args.len(), 2);
      assert_eq!(inlined.insts[1].args[0], Value::Const(20));
      assert_eq!(inlined.insts[1].args[1], Value::Type(Type(1)));
      match &inlined.term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(42));
        }
        Term::Jmp(_) => panic!("expected Ret in inlined block"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

#[test]
fn session_splice_then_set() {
  // Splice a new block, then use set to modify a scalar within it.
  let mut region = build_simple_func(1);

  // Replace exit block.
  region.session(|s| {
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(
      jmp_target,
      Block::make(Symbol(50), empty(), empty(), Term::make_ret([Value::Const(0)])),
    );
  });

  // Now modify a scalar in the spliced block.
  region.session(|s| {
    let name = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target.name,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.set(name, Symbol(77));
  });

  let func: &Func = &region;
  match &func.entry.get().term {
    Term::Jmp(jmp) => {
      assert_eq!(jmp.target.get().name, Symbol(77));
      match &jmp.target.get().term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(0));
        }
        Term::Jmp(_) => panic!("expected Ret"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

// ===========================================================================
// Cursor chain tests — session::Cursor
// ===========================================================================

#[test]
fn cursor_chain_set() {
  let mut region = build_simple_func(1);
  region.session(|s| {
    s.cursor().at(|f| &f.name).set(Symbol(42));
  });
  let func: &Func = &region;
  assert_eq!(func.name, Symbol(42));
}

#[test]
fn cursor_chain_splice() {
  let mut region = build_simple_func(1);
  region.session(|s| {
    s.cursor()
      .at(|f| &f.entry)
      .follow()
      .at(|b| match &b.term {
        Term::Jmp(jmp) => &jmp.target,
        Term::Ret { .. } => panic!("expected Jmp"),
      })
      .splice(Block::make(Symbol(99), empty(), empty(), Term::make_ret([Value::Const(99)])));
  });

  let func: &Func = &region;
  let entry = func.entry.get();
  match &entry.term {
    Term::Jmp(jmp) => {
      assert_eq!(jmp.target.get().name, Symbol(99));
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

// ===========================================================================
// Generic type tests — derive(Flat) with type parameters
// ===========================================================================

#[test]
fn generic_struct_empty_custom() {
  // Signature<u8> with an empty custom slice.
  let region: Region<Signature<u8>> =
    Region::new(Signature::<u8>::make([Type(1), Type(2)], [Type(3)], empty()));
  let sig: &Signature<u8> = &region;
  assert_eq!(sig.params.len(), 2);
  assert_eq!(sig.params[0], Type(1));
  assert_eq!(sig.params[1], Type(2));
  assert_eq!(sig.returns.len(), 1);
  assert_eq!(sig.returns[0], Type(3));
  assert!(sig.custom.is_empty());
}

#[test]
fn generic_struct_concrete_param() {
  // Signature<u32> — custom slice holds u32 values.
  let region: Region<Signature<u32>> =
    Region::new(Signature::<u32>::make([Type(0)], empty(), [10u32, 20, 30]));
  let sig: &Signature<u32> = &region;
  assert_eq!(sig.params.len(), 1);
  assert_eq!(sig.params[0], Type(0));
  assert!(sig.returns.is_empty());
  assert_eq!(sig.custom.len(), 3);
  assert_eq!(sig.custom[0], 10);
  assert_eq!(sig.custom[1], 20);
  assert_eq!(sig.custom[2], 30);
}

#[test]
fn generic_enum_unit_variant() {
  let region: Region<Wrapper<u8>> = Region::new(Wrapper::<u8>::make_empty());
  let w: &Wrapper<u8> = &region;
  assert!(matches!(w, Wrapper::Empty));
}

#[test]
fn generic_enum_named_variant() {
  let region: Region<Wrapper<u8>> =
    Region::new(Wrapper::<u8>::make_with_sig(Signature::<u8>::make([Type(1)], [Type(2)], empty())));
  let w: &Wrapper<u8> = &region;
  match w {
    Wrapper::WithSig { sig } => {
      let s = sig.get();
      assert_eq!(s.params.len(), 1);
      assert_eq!(s.params[0], Type(1));
      assert_eq!(s.returns.len(), 1);
      assert_eq!(s.returns[0], Type(2));
    }
    _ => panic!("expected WithSig"),
  }
}

#[test]
fn generic_enum_unnamed_variant() {
  let region: Region<Wrapper<u32>> = Region::new(Wrapper::<u32>::make_pair(42u32, 7));
  let w: &Wrapper<u32> = &region;
  match w {
    Wrapper::Pair(a, b) => {
      assert_eq!(*a, 42);
      assert_eq!(*b, 7);
    }
    _ => panic!("expected Pair"),
  }
}

#[test]
fn generic_struct_clone() {
  let region: Region<Signature<u32>> =
    Region::new(Signature::<u32>::make([Type(5)], [Type(6), Type(7)], [100u32]));
  let cloned = region.clone();
  let s1: &Signature<u32> = &region;
  let s2: &Signature<u32> = &cloned;
  assert_eq!(s1.params[0], s2.params[0]);
  assert_eq!(s1.returns.len(), s2.returns.len());
  assert_eq!(s1.custom[0], s2.custom[0]);
}

// ===========================================================================
// Trim tests — compact regions after splice mutations
// ===========================================================================

#[test]
fn trim_after_splice_shrinks_region() {
  // Build a function, splice in a new exit block (leaving old data as dead bytes),
  // then trim.
  let mut region = build_simple_func(1);
  let before = region.byte_len();

  region.session(|s| {
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(
      jmp_target,
      Block::make(Symbol(50), empty(), empty(), Term::make_ret([Value::Const(0)])),
    );
  });

  let after_splice = region.byte_len();

  // Splice appended data → region grew.
  assert!(after_splice > before, "splice should grow the buffer");

  region.trim();
  let after_trim = region.byte_len();

  // Trim removed dead bytes → region shrunk (at most as large as a fresh build).
  assert!(after_trim < after_splice, "trim should shrink: {after_trim} < {after_splice}");

  // Verify the data is still correct.
  let func: &Func = &region;
  assert_eq!(func.name, Symbol(1));
  let entry = func.entry.get();
  assert_eq!(entry.insts.len(), 1);
  assert_eq!(entry.insts[0].op, 1);
  match &entry.term {
    Term::Jmp(jmp) => {
      assert_eq!(jmp.target.get().name, Symbol(50));
      match &jmp.target.get().term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(0));
        }
        Term::Jmp(_) => panic!("expected Ret"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

#[test]
fn trim_preserves_fresh_region() {
  // A freshly built region has no dead data — trim should preserve all content.
  let mut region = build_simple_func(42);
  let before = region.byte_len();

  region.trim();

  // Size is the same (fresh build already minimal).
  assert_eq!(region.byte_len(), before);

  // All data intact.
  let func: &Func = &region;
  assert_eq!(func.name, Symbol(42));
  let block = func.entry.get();
  assert_eq!(block.insts.len(), 1);
  assert_eq!(block.insts[0].args[0], Value::Const(42));
  match &block.term {
    Term::Jmp(jmp) => {
      let exit = jmp.target.get();
      match &exit.term {
        Term::Ret { values } => assert_eq!(values[0], Value::Const(42)),
        Term::Jmp(_) => panic!("expected Ret"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

#[test]
fn trim_after_splice_list() {
  // Replace instructions with a different set, then trim.
  let mut region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      [
        Inst::make(1, Type(0), vec![Value::Const(10), Value::Const(11)]),
        Inst::make(2, Type(1), vec![Value::Const(20)]),
      ],
      Term::make_ret([Value::Const(42)]),
    ),
  ));

  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.entry.insts);
    s.splice_list(insts, [Inst::make(9, Type(0), vec![Value::Const(99)])]);
  });

  region.trim();

  let func: &Func = &region;
  let block = func.entry.get();
  assert_eq!(block.insts.len(), 1);
  assert_eq!(block.insts[0].op, 9);
  assert_eq!(block.insts[0].args[0], Value::Const(99));
  match &block.term {
    Term::Ret { values } => {
      assert_eq!(values.len(), 1);
      assert_eq!(values[0], Value::Const(42));
    }
    Term::Jmp(_) => panic!("expected Ret"),
  }
}

#[test]
fn trim_generic_type() {
  // Trim on a generic type (Signature<u32>).
  let mut region: Region<Signature<u32>> =
    Region::new(Signature::<u32>::make([Type(1), Type(2)], [Type(3)], [100u32, 200]));
  let before = region.byte_len();
  region.trim();
  assert_eq!(region.byte_len(), before);

  let sig: &Signature<u32> = &region;
  assert_eq!(sig.params.len(), 2);
  assert_eq!(sig.params[0], Type(1));
  assert_eq!(sig.returns[0], Type(3));
  assert_eq!(sig.custom[0], 100);
  assert_eq!(sig.custom[1], 200);
}

// ===========================================================================
// NearList tests
// ===========================================================================

/// A struct with a `NearList` field for testing.
#[derive(Flat, Debug)]
struct ListBlock {
  name: Symbol,
  items: NearList<Value>,
}

/// A struct using `NearList` for nested data (instructions with pointer fields).
#[derive(Flat, Debug)]
struct ListFunc {
  name: Symbol,
  insts: NearList<Inst>,
}

#[test]
fn near_list_empty() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  let block: &ListBlock = &region;
  assert_eq!(block.name, Symbol(1));
  assert!(block.items.is_empty());
  assert_eq!(block.items.len(), 0);
  assert!(block.items.first().is_none());
  assert_eq!(block.items.iter().count(), 0);
}

#[test]
fn near_list_single_element() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(2), [Value::Const(42)]));
  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 1);
  assert_eq!(*block.items.first().unwrap(), Value::Const(42));

  let collected: Vec<&Value> = block.items.iter().collect();
  assert_eq!(collected.len(), 1);
  assert_eq!(*collected[0], Value::Const(42));
}

#[test]
fn near_list_multiple_elements() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(
    Symbol(3),
    [Value::Const(10), Value::Type(Type(1)), Value::Const(30)],
  ));
  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);

  let vals: Vec<&Value> = block.items.iter().collect();
  assert_eq!(*vals[0], Value::Const(10));
  assert_eq!(*vals[1], Value::Type(Type(1)));
  assert_eq!(*vals[2], Value::Const(30));
}

#[test]
fn near_list_into_iter() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(4), [Value::Const(1), Value::Const(2)]));
  let block: &ListBlock = &region;

  let mut count = 0;
  for val in &block.items {
    assert!(matches!(val, Value::Const(1 | 2)));
    count += 1;
  }
  assert_eq!(count, 2);
}

#[test]
fn near_list_with_pointer_fields() {
  // NearList<Inst> where Inst contains NearList<Value> (nested linked lists)
  let region: Region<ListFunc> = Region::new(ListFunc::make(
    Symbol(5),
    [
      Inst::make(1, Type(0), [Value::Const(10), Value::Const(11)].as_slice()),
      Inst::make(2, Type(1), [Value::Const(20)].as_slice()),
      Inst::make(3, Type(0), [].as_slice()),
    ],
  ));
  let func: &ListFunc = &region;
  assert_eq!(func.name, Symbol(5));
  assert_eq!(func.insts.len(), 3);

  let insts: Vec<&Inst> = func.insts.iter().collect();
  assert_eq!(insts[0].op, 1);
  assert_eq!(insts[0].args.len(), 2);
  assert_eq!(insts[0].args[0], Value::Const(10));
  assert_eq!(insts[0].args[1], Value::Const(11));

  assert_eq!(insts[1].op, 2);
  assert_eq!(insts[1].args.len(), 1);
  assert_eq!(insts[1].args[0], Value::Const(20));

  assert_eq!(insts[2].op, 3);
  assert!(insts[2].args.is_empty());
}

#[test]
fn near_list_clone_preserves() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(6), [Value::Const(1), Value::Const(2), Value::Const(3)]));
  let cloned = region.clone();

  let b1: &ListBlock = &region;
  let b2: &ListBlock = &cloned;

  assert_eq!(b1.name, b2.name);
  assert_eq!(b1.items.len(), b2.items.len());

  let v1: Vec<&Value> = b1.items.iter().collect();
  let v2: Vec<&Value> = b2.items.iter().collect();
  for (a, b) in v1.iter().zip(v2.iter()) {
    assert_eq!(**a, **b);
  }
}

#[test]
fn near_list_debug() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(7), [Value::Const(42)]));
  let debug_str = format!("{:?}", &*region);
  assert!(debug_str.contains("ListBlock"));
}

#[test]
fn near_list_splice_list() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.splice_list(items, [Value::Const(10), Value::Const(20), Value::Const(30)]);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);

  let vals: Vec<&Value> = block.items.iter().collect();
  assert_eq!(*vals[0], Value::Const(10));
  assert_eq!(*vals[1], Value::Const(20));
  assert_eq!(*vals[2], Value::Const(30));
}

#[test]
fn near_list_splice_list_to_empty() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.splice_list(items, empty());
  });

  let block: &ListBlock = &region;
  assert!(block.items.is_empty());
}

#[test]
fn near_list_push_front() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);

  let vals: Vec<&Value> = block.items.iter().collect();
  assert_eq!(*vals[0], Value::Const(1));
  assert_eq!(*vals[1], Value::Const(2));
  assert_eq!(*vals[2], Value::Const(3));
}

#[test]
fn near_list_push_front_empty() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(42));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 1);
  assert_eq!(*block.items.first().unwrap(), Value::Const(42));
}

#[test]
fn near_list_trim() {
  // Build, splice, then trim — verify deep-copy works for NearList.
  let mut region: Region<ListFunc> = Region::new(ListFunc::make(
    Symbol(1),
    [
      Inst::make(1, Type(0), [Value::Const(10)].as_slice()),
      Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)].as_slice()),
    ],
  ));
  let before = region.byte_len();

  // Splice to replace with smaller list
  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.insts);
    s.splice_list(insts, [Inst::make(9, Type(0), [Value::Const(99)])]);
  });

  let after_splice = region.byte_len();
  assert!(after_splice >= before, "splice should not shrink");

  region.trim();
  let after_trim = region.byte_len();
  assert!(after_trim <= after_splice, "trim should not grow");

  // Verify data integrity after trim.
  let func: &ListFunc = &region;
  assert_eq!(func.name, Symbol(1));
  assert_eq!(func.insts.len(), 1);
  let insts: Vec<&Inst> = func.insts.iter().collect();
  assert_eq!(insts[0].op, 9);
  assert_eq!(insts[0].args.len(), 1);
  assert_eq!(insts[0].args[0], Value::Const(99));
}

#[test]
fn near_list_trim_preserves_fresh() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));
  let before = region.byte_len();

  region.trim();
  assert_eq!(region.byte_len(), before);

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  let vals: Vec<&Value> = block.items.iter().collect();
  assert_eq!(*vals[0], Value::Const(1));
  assert_eq!(*vals[1], Value::Const(2));
  assert_eq!(*vals[2], Value::Const(3));
}

#[test]
fn near_list_exact_size_iterator() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));
  let block: &ListBlock = &region;
  let iter = block.items.iter();
  assert_eq!(iter.len(), 3);
}

#[test]
fn near_list_fused_iterator() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2)]));
  let block: &ListBlock = &region;
  let mut iter = block.items.iter();

  // Exhaust the iterator.
  assert!(iter.next().is_some());
  assert!(iter.next().is_some());
  assert!(iter.next().is_none());

  // FusedIterator guarantee: keeps returning None.
  assert!(iter.next().is_none());
  assert!(iter.next().is_none());
}

#[test]
fn near_list_fused_iterator_empty() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  let block: &ListBlock = &region;
  let mut iter = block.items.iter();

  // Empty list: returns None immediately and repeatedly.
  assert!(iter.next().is_none());
  assert!(iter.next().is_none());
}

#[test]
fn near_list_last_single_segment() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));
  let block: &ListBlock = &region;
  assert_eq!(block.items.last(), Some(&Value::Const(30)));
}

#[test]
fn near_list_last_empty() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  let block: &ListBlock = &region;
  assert_eq!(block.items.last(), None);
}

#[test]
fn near_list_last_single_element() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(42)]));
  let block: &ListBlock = &region;
  assert_eq!(block.items.last(), Some(&Value::Const(42)));
  // first and last should be the same for a single-element list.
  assert_eq!(block.items.first(), block.items.last());
}

#[test]
fn near_list_last_multi_segment() {
  // push_front creates a second segment, so last() must walk segments.
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 2);
  // Last element should still be 3 (from the original segment).
  assert_eq!(block.items.last(), Some(&Value::Const(3)));
}

#[test]
fn near_list_last_multi_segment_after_trim() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  region.trim();
  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items.last(), Some(&Value::Const(3)));
}

#[test]
fn near_list_fused_iterator_multi_segment() {
  // FusedIterator should hold across segment boundaries.
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 2);

  let mut iter = block.items.iter();
  assert_eq!(iter.len(), 3);
  assert_eq!(iter.next(), Some(&Value::Const(1)));
  assert_eq!(iter.next(), Some(&Value::Const(2)));
  assert_eq!(iter.next(), Some(&Value::Const(3)));
  assert!(iter.next().is_none());
  // Fused: stays None.
  assert!(iter.next().is_none());
}

// ===========================================================================
// extend_list tests
// ===========================================================================

#[test]
fn extend_list_appends_to_existing() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, [Value::Const(3), Value::Const(4)]);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 4);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
  assert_eq!(block.items[3], Value::Const(4));
}

#[test]
fn extend_list_on_empty() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, [Value::Const(10), Value::Const(20)]);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 2);
  assert_eq!(block.items[0], Value::Const(10));
  assert_eq!(block.items[1], Value::Const(20));
}

#[test]
fn extend_list_with_empty_extra() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, empty());
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 1);
  assert_eq!(block.items[0], Value::Const(1));
}

#[test]
fn extend_list_single_element() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, [Value::Const(4)]);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 4);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
  assert_eq!(block.items[3], Value::Const(4));
}

#[test]
fn extend_list_with_iterator() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, (0..3).map(|k| Value::Const(10 + k)));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 4);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(10));
  assert_eq!(block.items[2], Value::Const(11));
  assert_eq!(block.items[3], Value::Const(12));
}

#[test]
fn extend_list_preserves_after_trim() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, [Value::Const(2), Value::Const(3)]);
  });

  region.trim();

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

// ===========================================================================
// Ref<T> deep-copy tests — push_front / splice_list with Ref items
// ===========================================================================

#[test]
fn push_front_with_ref_deep_copies() {
  // Verify that push_front with a Ref<Inst> deep-copies the value
  // (including nested NearList<Value>) into the new node.
  let mut region: Region<ListFunc> =
    Region::new(ListFunc::make(Symbol(1), [Inst::make(1, Type(0), [Value::Const(42)].as_slice())]));

  region.session(|s| {
    let first_inst = s.nav(s.root(), |f| &f.insts[0]);
    let insts = s.nav(s.root(), |f| &f.insts);
    s.push_front(insts, first_inst);
  });

  let func: &ListFunc = &region;
  assert_eq!(func.insts.len(), 2);

  let insts: Vec<&Inst> = func.insts.iter().collect();
  // The pushed copy should be identical to the original.
  assert_eq!(insts[0].op, 1);
  assert_eq!(insts[0].typ, Type(0));
  assert_eq!(insts[0].args.len(), 1);
  assert_eq!(insts[0].args[0], Value::Const(42));
  // Original still intact.
  assert_eq!(insts[1].op, 1);
  assert_eq!(insts[1].args[0], Value::Const(42));
}

#[test]
fn splice_list_with_refs_deep_copies() {
  // Verify that splice_list with Ref items deep-copies values
  // and can reverse the order.
  let mut region: Region<ListFunc> = Region::new(ListFunc::make(
    Symbol(1),
    [
      Inst::make(1, Type(0), [Value::Const(10)].as_slice()),
      Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)].as_slice()),
    ],
  ));

  region.session(|s| {
    let ref0 = s.nav(s.root(), |f| &f.insts[0]);
    let ref1 = s.nav(s.root(), |f| &f.insts[1]);
    let insts = s.nav(s.root(), |f| &f.insts);
    s.splice_list(insts, [ref1, ref0]);
  });

  let func: &ListFunc = &region;
  assert_eq!(func.insts.len(), 2);

  let insts: Vec<&Inst> = func.insts.iter().collect();
  // Reversed order — inst[1] is now first.
  assert_eq!(insts[0].op, 2);
  assert_eq!(insts[0].args.len(), 2);
  assert_eq!(insts[0].args[0], Value::Const(20));
  assert_eq!(insts[0].args[1], Value::Const(21));
  assert_eq!(insts[1].op, 1);
  assert_eq!(insts[1].args.len(), 1);
  assert_eq!(insts[1].args[0], Value::Const(10));
}

#[test]
fn push_front_ref_primitive_list() {
  // Ref deep-copy on a list of primitive values.
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20)]));

  region.session(|s| {
    let first_val = s.nav(s.root(), |b| &b.items[0]);
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, first_val);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(10));
  assert_eq!(block.items[1], Value::Const(10));
  assert_eq!(block.items[2], Value::Const(20));
}

// ===========================================================================
// Option<Near<T>> tests
// ===========================================================================

/// A struct with an `Option<Near<T>>` field for testing the `OptionNear` derive.
#[derive(Flat, Debug)]
struct OptBlock {
  name: Symbol,
  next: Option<Near<Block>>,
}

#[test]
fn option_near_some_roundtrip() {
  let region: Region<OptBlock> = Region::new(OptBlock::make(
    Symbol(1),
    Some(Block::make(Symbol(2), empty(), empty(), Term::make_ret([Value::Const(42)]))),
  ));

  let ob: &OptBlock = &region;
  assert_eq!(ob.name, Symbol(1));
  let next = ob.next.as_ref().expect("should be Some");
  let block = next.get();
  assert_eq!(block.name, Symbol(2));
  match &block.term {
    Term::Ret { values } => {
      assert_eq!(values.len(), 1);
      assert_eq!(values[0], Value::Const(42));
    }
    Term::Jmp(_) => panic!("expected Ret"),
  }
}

#[test]
fn option_near_none_roundtrip() {
  let region: Region<OptBlock> =
    Region::new(OptBlock::make(Symbol(1), None::<std::convert::Infallible>));

  let ob: &OptBlock = &region;
  assert_eq!(ob.name, Symbol(1));
  assert!(ob.next.is_none());
}

#[test]
fn option_near_trim() {
  // Build with Some, verify trim preserves data.
  let mut region: Region<OptBlock> = Region::new(OptBlock::make(
    Symbol(1),
    Some(Block::make(
      Symbol(2),
      empty(),
      [Inst::make(1, Type(0), [Value::Const(10)])],
      Term::make_ret([Value::Const(42)]),
    )),
  ));

  let before = region.byte_len();
  region.trim();
  assert_eq!(region.byte_len(), before);

  let ob: &OptBlock = &region;
  let next = ob.next.as_ref().expect("should be Some");
  let block = next.get();
  assert_eq!(block.name, Symbol(2));
  assert_eq!(block.insts.len(), 1);
  assert_eq!(block.insts[0].args[0], Value::Const(10));
}

#[test]
fn extend_list_nested_pointer_fields() {
  let mut region: Region<ListFunc> =
    Region::new(ListFunc::make(Symbol(1), [Inst::make(1, Type(0), [Value::Const(10)].as_slice())]));

  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.insts);
    s.extend_list(
      insts,
      [
        Inst::make(2, Type(1), vec![Value::Const(20), Value::Const(21)]),
        Inst::make(3, Type(0), vec![]),
      ],
    );
  });

  let func: &ListFunc = &region;
  assert_eq!(func.insts.len(), 3);

  let insts: Vec<&Inst> = func.insts.iter().collect();
  assert_eq!(insts[0].op, 1);
  assert_eq!(insts[0].args.len(), 1);
  assert_eq!(insts[0].args[0], Value::Const(10));

  assert_eq!(insts[1].op, 2);
  assert_eq!(insts[1].args.len(), 2);
  assert_eq!(insts[1].args[0], Value::Const(20));
  assert_eq!(insts[1].args[1], Value::Const(21));

  assert_eq!(insts[2].op, 3);
  assert!(insts[2].args.is_empty());
}

// ===========================================================================
// map_list tests
// ===========================================================================

#[test]
fn session_map_list_identity() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.map_list(items, |v| v);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(10));
  assert_eq!(block.items[1], Value::Const(20));
  assert_eq!(block.items[2], Value::Const(30));
}

#[test]
fn session_map_list_transform() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.map_list(items, |v| match v {
      Value::Const(n) => Value::Const(n * 10),
      other @ Value::Type(_) => other,
    });
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(10));
  assert_eq!(block.items[1], Value::Const(20));
  assert_eq!(block.items[2], Value::Const(30));
}

#[test]
fn session_map_list_empty() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.map_list(items, |v| v);
  });

  let block: &ListBlock = &region;
  assert!(block.items.is_empty());
}

// ===========================================================================
// graft tests
// ===========================================================================

#[test]
fn session_graft_copies_region() {
  let grafted_block: Region<Block> =
    Region::new(Block::make(Symbol(99), empty(), empty(), Term::make_ret([Value::Const(77)])));

  let mut caller: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      empty(),
      Term::make_jmp(Jmp::make(
        empty(),
        Block::make(Symbol(1), empty(), empty(), Term::make_ret(empty())),
      )),
    ),
  ));

  caller.session(|s| {
    let grafted: Ref<'_, Block> = s.graft(&grafted_block);
    // Redirect the jmp target to the grafted block.
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(jmp_target, grafted);
  });

  let func: &Func = &caller;
  match &func.entry.term {
    Term::Jmp(jmp) => {
      let target = jmp.target.get();
      assert_eq!(target.name, Symbol(99));
      match &target.term {
        Term::Ret { values } => {
          assert_eq!(values.len(), 1);
          assert_eq!(values[0], Value::Const(77));
        }
        Term::Jmp(_) => panic!("expected Ret in grafted block"),
      }
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

// ===========================================================================
// list_refs tests
// ===========================================================================

#[test]
fn session_list_refs_collects() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    let refs = s.list_refs(items);
    assert_eq!(refs.len(), 3);
    assert_eq!(*s.at(refs[0]), Value::Const(10));
    assert_eq!(*s.at(refs[1]), Value::Const(20));
    assert_eq!(*s.at(refs[2]), Value::Const(30));
  });
}

#[test]
fn session_list_refs_empty() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    let refs = s.list_refs(items);
    assert!(refs.is_empty());
  });
}

// ===========================================================================
// Cursor extended tests
// ===========================================================================

#[test]
fn cursor_pin_extracts_ref() {
  let mut region = build_simple_func(1);
  region.session(|s| {
    let r = s.cursor().at(|f| &f.name).pin();
    assert_eq!(*s.at(r), Symbol(1));
  });
}

#[test]
fn cursor_write_with_builder() {
  let mut region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(Symbol(0), empty(), empty(), Term::make_ret([Value::Const(42)])),
  ));

  region.session(|s| {
    s.cursor()
      .at(|f| &f.entry)
      .follow()
      .at(|b| &b.term)
      .write_with(Term::make_ret([Value::Const(99)]));
  });

  let func: &Func = &region;
  match &func.entry.term {
    Term::Ret { values } => {
      assert_eq!(values.len(), 1);
      assert_eq!(values[0], Value::Const(99));
    }
    Term::Jmp(_) => panic!("expected Ret"),
  }
}

// ===========================================================================
// Panic-path tests (#[should_panic])
// ===========================================================================

// ===========================================================================
// ZST tests
// ===========================================================================

#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
struct Marker;

#[derive(Flat, Debug)]
struct TaggedList {
  tag: Marker,
  items: NearList<u32>,
}

#[test]
fn zst_struct_in_region() {
  let region: Region<Marker> = Region::new(Marker);
  let _m: &Marker = &region;
}

#[test]
fn zst_field_in_struct() {
  let region: Region<TaggedList> = Region::new(TaggedList::make(Marker, [1u32, 2, 3]));
  let tl: &TaggedList = &region;
  assert_eq!(tl.tag, Marker);
  assert_eq!(tl.items.len(), 3);
  assert_eq!(tl.items[0], 1);
  assert_eq!(tl.items[1], 2);
  assert_eq!(tl.items[2], 3);
}

#[derive(Flat, Debug)]
struct MarkerList {
  items: NearList<Marker>,
}

#[test]
fn zst_list_elements() {
  let region: Region<MarkerList> = Region::new(MarkerList::make([Marker, Marker, Marker]));
  let ml: &MarkerList = &region;
  assert_eq!(ml.items.len(), 3);
  for m in &ml.items {
    assert_eq!(*m, Marker);
  }
}

// ===========================================================================
// Segment contiguity tests
// ===========================================================================

#[test]
fn fresh_build_single_segment() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));
  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
}

#[test]
fn splice_list_produces_single_segment() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.splice_list(items, [Value::Const(10), Value::Const(20), Value::Const(30)]);
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items.len(), 3);
}

#[test]
fn push_front_creates_multi_segment() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  // push_front creates a new 1-element segment → 2 segments total
  assert_eq!(block.items.segment_count(), 2);
  assert_eq!(block.items.len(), 3);
}

#[test]
fn trim_compacts_to_single_segment() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  // push_front creates multi-segment chain
  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 2);

  // trim compacts to single segment
  region.trim();

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

#[test]
fn extend_then_trim_single_segment() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(1)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.extend_list(items, [Value::Const(2), Value::Const(3)]);
  });

  let block: &ListBlock = &region;
  // extend_list appends a new segment → 2 segments
  assert_eq!(block.items.segment_count(), 2);

  region.trim();

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

#[test]
fn empty_list_zero_segments() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 0);
}

#[test]
fn nested_list_trim_all_single_segment() {
  let mut region: Region<ListFunc> =
    Region::new(ListFunc::make(Symbol(1), [Inst::make(1, Type(0), [Value::Const(10)].as_slice())]));

  // extend creates multi-segment for outer list
  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.insts);
    s.extend_list(insts, [Inst::make(2, Type(1), vec![Value::Const(20), Value::Const(21)])]);
  });

  let func: &ListFunc = &region;
  assert_eq!(func.insts.segment_count(), 2);

  region.trim();

  let func: &ListFunc = &region;
  // After trim, both outer and inner lists are single-segment
  assert_eq!(func.insts.segment_count(), 1);
  assert_eq!(func.insts.len(), 2);
  let insts: Vec<&Inst> = func.insts.iter().collect();
  assert_eq!(insts[0].args.segment_count(), 1);
  assert_eq!(insts[1].args.segment_count(), 1);
}

// ===========================================================================
// NearList::get tests
// ===========================================================================

#[test]
fn near_list_get_returns_some() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));
  let block: &ListBlock = &region;
  assert_eq!(*block.items.get(0).unwrap(), Value::Const(10));
  assert_eq!(*block.items.get(1).unwrap(), Value::Const(20));
  assert_eq!(*block.items.get(2).unwrap(), Value::Const(30));
}

#[test]
fn near_list_get_returns_none_oob() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(10)]));
  let block: &ListBlock = &region;
  assert!(block.items.get(1).is_none());
  assert!(block.items.get(100).is_none());
}

#[test]
fn near_list_get_empty_returns_none() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  let block: &ListBlock = &region;
  assert!(block.items.get(0).is_none());
}

#[test]
fn index_single_segment_o1() {
  // After fresh build, all elements are in a single segment → O(1) fast path.
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));
  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items[0], Value::Const(10));
  assert_eq!(block.items[1], Value::Const(20));
  assert_eq!(block.items[2], Value::Const(30));
}

#[test]
fn index_multi_segment_fallback() {
  // After push_front, we have 2 segments → index falls back to iterator.
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(2), Value::Const(3)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.push_front(items, Value::Const(1));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.segment_count(), 2);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

// ===========================================================================
// filter_list tests
// ===========================================================================

#[test]
fn filter_list_keeps_matching() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(
    Symbol(1),
    [Value::Const(1), Value::Const(2), Value::Const(3), Value::Const(4)],
  ));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.filter_list(items, |v| matches!(v, Value::Const(n) if *n % 2 == 0));
  });

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 2);
  assert_eq!(block.items[0], Value::Const(2));
  assert_eq!(block.items[1], Value::Const(4));
}

#[test]
fn filter_list_removes_all() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.filter_list(items, |_| false);
  });

  let block: &ListBlock = &region;
  assert!(block.items.is_empty());
}

#[test]
fn filter_list_keeps_all_noop() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(1), Value::Const(2), Value::Const(3)]));
  let before = region.byte_len();

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.filter_list(items, |_| true);
  });

  // No-op: nothing filtered, byte_len unchanged.
  assert_eq!(region.byte_len(), before);
  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

#[test]
fn filter_list_empty() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.filter_list(items, |_| true);
  });

  let block: &ListBlock = &region;
  assert!(block.items.is_empty());
}

#[test]
fn filter_list_nested_pointers() {
  // Filter on NearList<Inst> where Inst has NearList<Value> fields.
  let mut region: Region<ListFunc> = Region::new(ListFunc::make(
    Symbol(1),
    [
      Inst::make(1, Type(0), [Value::Const(10)].as_slice()),
      Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)].as_slice()),
      Inst::make(3, Type(0), [Value::Const(30)].as_slice()),
    ],
  ));

  // Keep only instructions with op != 2.
  region.session(|s| {
    let insts = s.nav(s.root(), |f| &f.insts);
    s.filter_list(insts, |inst| inst.op != 2);
  });

  let func: &ListFunc = &region;
  assert_eq!(func.insts.len(), 2);
  let insts: Vec<&Inst> = func.insts.iter().collect();
  assert_eq!(insts[0].op, 1);
  assert_eq!(insts[0].args.len(), 1);
  assert_eq!(insts[0].args[0], Value::Const(10));
  assert_eq!(insts[1].op, 3);
  assert_eq!(insts[1].args.len(), 1);
  assert_eq!(insts[1].args[0], Value::Const(30));
}

#[test]
fn filter_list_then_trim() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(
    Symbol(1),
    [Value::Const(1), Value::Const(2), Value::Const(3), Value::Const(4), Value::Const(5)],
  ));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    s.filter_list(items, |v| matches!(v, Value::Const(n) if *n <= 3));
  });

  region.trim();

  let block: &ListBlock = &region;
  assert_eq!(block.items.len(), 3);
  assert_eq!(block.items.segment_count(), 1);
  assert_eq!(block.items[0], Value::Const(1));
  assert_eq!(block.items[1], Value::Const(2));
  assert_eq!(block.items[2], Value::Const(3));
}

// ===========================================================================
// list_item tests
// ===========================================================================

#[test]
fn list_item_returns_correct_ref() {
  let mut region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20), Value::Const(30)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    let r0 = s.list_item(items, 0);
    let r1 = s.list_item(items, 1);
    let r2 = s.list_item(items, 2);
    assert_eq!(*s.at(r0), Value::Const(10));
    assert_eq!(*s.at(r1), Value::Const(20));
    assert_eq!(*s.at(r2), Value::Const(30));
  });
}

#[test]
#[should_panic(expected = "NearList index out of bounds")]
fn list_item_oob_panics() {
  let mut region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(10)]));

  region.session(|s| {
    let items = s.nav(s.root(), |b| &b.items);
    let _ = s.list_item(items, 5);
  });
}

// ===========================================================================
// Panic-path tests (#[should_panic])
// ===========================================================================

#[test]
#[should_panic(expected = "NearList index out of bounds")]
fn near_list_index_out_of_bounds_panics() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(10)]));
  let block: &ListBlock = &region;
  let _ = block.items[5]; // only 1 element, index 5 is OOB
}

// ===========================================================================
// Validation success tests
// ===========================================================================

#[test]
fn validate_primitive_region() {
  let region: Region<u32> = Region::new(42u32);
  let bytes = region.as_bytes();
  u32::validate(0, bytes).unwrap();
}

#[test]
fn validate_struct_with_near() {
  let region = build_simple_func(1);
  let bytes = region.as_bytes();
  Func::validate(0, bytes).unwrap();
}

#[test]
fn validate_struct_with_near_list() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20)]));
  let bytes = region.as_bytes();
  ListBlock::validate(0, bytes).unwrap();
}

#[test]
fn validate_struct_with_option_near_some() {
  let region: Region<OptBlock> = Region::new(OptBlock::make(
    Symbol(1),
    Some(Block::make(Symbol(2), empty(), empty(), Term::make_ret([Value::Const(42)]))),
  ));
  let bytes = region.as_bytes();
  OptBlock::validate(0, bytes).unwrap();
}

#[test]
fn validate_struct_with_option_near_none() {
  let region: Region<OptBlock> =
    Region::new(OptBlock::make(Symbol(1), None::<std::convert::Infallible>));
  let bytes = region.as_bytes();
  OptBlock::validate(0, bytes).unwrap();
}

#[test]
fn validate_nested_structs() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      [(Symbol(1), Type(0))],
      [
        Inst::make(1, Type(0), [Value::Const(10), Value::Const(11)]),
        Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)]),
      ],
      Term::make_jmp(Jmp::make(
        [Value::Const(1)],
        Block::make(Symbol(1), empty(), empty(), Term::make_ret([Value::Const(42)])),
      )),
    ),
  ));
  let bytes = region.as_bytes();
  Func::validate(0, bytes).unwrap();
}

#[test]
fn validate_enum_all_variants() {
  // Test Ret variant
  let region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(Symbol(0), empty(), empty(), Term::make_ret([Value::Const(1)])),
  ));
  Func::validate(0, region.as_bytes()).unwrap();

  // Test Jmp variant
  let region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(
      Symbol(0),
      empty(),
      empty(),
      Term::make_jmp(Jmp::make(
        empty(),
        Block::make(Symbol(1), empty(), empty(), Term::make_ret(empty())),
      )),
    ),
  ));
  Func::validate(0, region.as_bytes()).unwrap();
}

#[test]
fn validate_generic_struct() {
  let region: Region<Signature<u32>> =
    Region::new(Signature::<u32>::make([Type(1)], [Type(2)], [10u32, 20]));
  Signature::<u32>::validate(0, region.as_bytes()).unwrap();
}

#[test]
fn validate_after_mutation() {
  let mut region = build_simple_func(1);
  region.session(|s| {
    let name = s.nav(s.root(), |f| &f.name);
    s.set(name, Symbol(99));
  });
  Func::validate(0, region.as_bytes()).unwrap();
}

#[test]
fn validate_after_trim() {
  let mut region = build_simple_func(1);
  region.session(|s| {
    let jmp_target = s.nav(s.root(), |f| match &f.entry.term {
      Term::Jmp(jmp) => &jmp.target,
      Term::Ret { .. } => panic!("expected Jmp"),
    });
    s.splice(
      jmp_target,
      Block::make(Symbol(50), empty(), empty(), Term::make_ret([Value::Const(0)])),
    );
  });
  region.trim();
  Func::validate(0, region.as_bytes()).unwrap();
}

#[test]
fn validate_empty_list() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), empty()));
  ListBlock::validate(0, region.as_bytes()).unwrap();
}

// ===========================================================================
// Validation failure tests
// ===========================================================================

#[test]
fn validate_empty_buffer() {
  let result = u32::validate(0, &[]);
  assert!(matches!(result, Err(ValidateError::OutOfBounds { .. })));
}

#[test]
fn validate_truncated_buffer() {
  let region = build_simple_func(1);
  let bytes = region.as_bytes();
  // Truncate to just 2 bytes — too small for Func root.
  let result = Func::validate(0, &bytes[..2]);
  assert!(matches!(result, Err(ValidateError::OutOfBounds { .. })));
}

#[test]
fn validate_bad_near_offset_oob() {
  let region = build_simple_func(1);
  let mut bytes = region.as_bytes().to_vec();
  // The Near<Block> field (entry) is at offset_of!(Func, entry) which is
  // right after the Symbol(u32) name field = 4 bytes.
  let near_offset = core::mem::offset_of!(Func, entry);
  // Write an offset that points way past the buffer.
  let bad_off: i32 = i32::try_from(bytes.len()).unwrap() * 2;
  bytes[near_offset..near_offset + 4].copy_from_slice(&bad_off.to_ne_bytes());
  let result = Func::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::OutOfBounds { .. })));
}

#[test]
fn validate_bad_near_offset_zero() {
  let region = build_simple_func(1);
  let mut bytes = region.as_bytes().to_vec();
  let near_offset = core::mem::offset_of!(Func, entry);
  // Zero out the Near offset → NullNear.
  bytes[near_offset..near_offset + 4].copy_from_slice(&0i32.to_ne_bytes());
  let result = Func::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::NullNear { .. })));
}

#[test]
fn validate_bad_near_alignment() {
  let region = build_simple_func(1);
  let mut bytes = region.as_bytes().to_vec();
  let near_offset = core::mem::offset_of!(Func, entry);
  // Set offset to point to an odd address (misaligned for Block which needs
  // at least align_of::<i32>() = 4).
  let bad_off: i32 = 1; // points to addr (near_offset + 1) which is odd.
  bytes[near_offset..near_offset + 4].copy_from_slice(&bad_off.to_ne_bytes());
  let result = Func::validate(0, &bytes);
  // Should fail with either Misaligned or OutOfBounds depending on position.
  assert!(result.is_err());
}

#[test]
fn validate_bad_list_header_inconsistent() {
  let region: Region<ListBlock> = Region::new(ListBlock::make(Symbol(1), [Value::Const(10)]));
  let mut bytes = region.as_bytes().to_vec();
  let list_offset = core::mem::offset_of!(ListBlock, items);
  // Set head to non-zero but len to 0 → inconsistent.
  bytes[list_offset..list_offset + 4].copy_from_slice(&42i32.to_ne_bytes());
  bytes[list_offset + 4..list_offset + 8].copy_from_slice(&0u32.to_ne_bytes());
  let result = ListBlock::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::InvalidListHeader { .. })));
}

#[test]
fn validate_bad_list_len_mismatch() {
  let region: Region<ListBlock> =
    Region::new(ListBlock::make(Symbol(1), [Value::Const(10), Value::Const(20)]));
  let mut bytes = region.as_bytes().to_vec();
  let list_offset = core::mem::offset_of!(ListBlock, items);
  // Change the total len to 99 while segment has only 2 elements.
  bytes[list_offset + 4..list_offset + 8].copy_from_slice(&99u32.to_ne_bytes());
  let result = ListBlock::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::ListLenMismatch { .. })));
}

#[test]
fn validate_bad_enum_discriminant() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(1),
    Block::make(Symbol(0), empty(), empty(), Term::make_ret([Value::Const(42)])),
  ));
  let mut bytes = region.as_bytes().to_vec();
  // Find the Term's address: it's at the entry block's term field offset.
  // We need to follow the Near pointer first.
  let near_offset = core::mem::offset_of!(Func, entry);
  let near_off = i32::from_ne_bytes(bytes[near_offset..near_offset + 4].try_into().unwrap());
  let block_addr = near_offset.cast_signed().wrapping_add(near_off as isize).cast_unsigned();
  let term_offset = core::mem::offset_of!(Block, term);
  let disc_addr = block_addr + term_offset;
  // Set discriminant to 255 (only 2 valid variants: 0 and 1).
  bytes[disc_addr] = 255;
  let result = Func::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::InvalidDiscriminant { .. })));
}

#[derive(Flat, Debug)]
struct BoolStruct {
  flag: bool,
  id: u32,
}

#[test]
fn validate_bad_bool() {
  let region: Region<BoolStruct> = Region::new(BoolStruct { flag: true, id: 42 });
  let mut bytes = region.as_bytes().to_vec();
  // Set the bool byte to 2 (invalid).
  let bool_offset = core::mem::offset_of!(BoolStruct, flag);
  bytes[bool_offset] = 2;
  let result = BoolStruct::validate(0, &bytes);
  assert!(matches!(result, Err(ValidateError::InvalidBool { value: 2, .. })));
}

// ===========================================================================
// Roundtrip tests (as_bytes / from_bytes)
// ===========================================================================

#[test]
fn region_as_bytes_from_bytes_roundtrip() {
  let region = build_simple_func(42);
  let bytes = region.as_bytes();
  let restored: Region<Func> = Region::from_bytes(bytes).unwrap();

  assert_eq!(restored.name, Symbol(42));
  let block = restored.entry.get();
  assert_eq!(block.insts.len(), 1);
  assert_eq!(block.insts[0].args[0], Value::Const(42));
}

#[test]
fn region_from_bytes_unchecked_roundtrip() {
  let region = build_simple_func(7);
  let bytes = region.as_bytes();
  // SAFETY: `bytes` came from `region.as_bytes()` on a valid `Region<Func>`.
  let restored: Region<Func> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(restored.name, Symbol(7));
  let block = restored.entry.get();
  assert_eq!(block.insts[0].op, 1);
}

#[test]
fn region_from_bytes_complex() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(100),
    Block::make(
      Symbol(0),
      [(Symbol(1), Type(0)), (Symbol(2), Type(1))],
      [
        Inst::make(1, Type(0), [Value::Const(10), Value::Const(11)]),
        Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)]),
      ],
      Term::make_jmp(Jmp::make(
        [Value::Const(1)],
        Block::make(Symbol(1), empty(), empty(), Term::make_ret([Value::Const(42)])),
      )),
    ),
  ));

  let bytes = region.as_bytes();
  let restored: Region<Func> = Region::from_bytes(bytes).unwrap();

  assert_eq!(restored.name, Symbol(100));
  let block = restored.entry.get();
  assert_eq!(block.params.len(), 2);
  assert_eq!(block.insts.len(), 2);
  assert_eq!(block.insts[0].args[0], Value::Const(10));
  match &block.term {
    Term::Jmp(jmp) => {
      assert_eq!(jmp.target.get().name, Symbol(1));
    }
    Term::Ret { .. } => panic!("expected Jmp"),
  }
}

#[test]
fn region_from_bytes_clone_equivalence() {
  let region = build_simple_func(5);
  let cloned = region.clone();
  let from_bytes: Region<Func> = Region::from_bytes(region.as_bytes()).unwrap();

  // Both should produce equivalent regions.
  assert_eq!(cloned.name, from_bytes.name);
  assert_eq!(cloned.entry.get().name, from_bytes.entry.get().name);
  assert_eq!(cloned.entry.get().insts.len(), from_bytes.entry.get().insts.len());
}

#[test]
fn region_from_bytes_fixed_buf() {
  use nearest::FixedBuf;

  let region: Region<u32, FixedBuf<64>> = Region::new_in(42u32);
  let bytes = region.as_bytes();
  let restored: Region<u32, FixedBuf<64>> = Region::from_bytes(bytes).unwrap();
  assert_eq!(*restored, 42);
}

// ===========================================================================
// from_bytes_unchecked / from_buf_unchecked tests
// ===========================================================================

#[test]
fn from_bytes_unchecked_matches_checked_simple() {
  let region: Region<Func> = build_simple_func(99);
  let bytes = region.as_bytes();

  let checked: Region<Func> = Region::from_bytes(bytes).unwrap();
  // SAFETY: `bytes` came from `as_bytes()` on a valid `Region<Func>`.
  let unchecked: Region<Func> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(checked.name, unchecked.name);
  assert_eq!(checked.entry.get().name, unchecked.entry.get().name);
  assert_eq!(checked.entry.get().insts.len(), unchecked.entry.get().insts.len());
  assert_eq!(checked.entry.get().insts[0].op, unchecked.entry.get().insts[0].op);
  // Note: raw byte comparison is omitted for Func because Value enums have
  // uninitialized padding bytes, which would trip Miri.
  assert_eq!(checked.byte_len(), unchecked.byte_len());
}

#[test]
fn from_bytes_unchecked_matches_checked_complex() {
  let region: Region<Func> = Region::new(Func::make(
    Symbol(200),
    Block::make(
      Symbol(0),
      [(Symbol(1), Type(0)), (Symbol(2), Type(1))],
      [
        Inst::make(1, Type(0), [Value::Const(10), Value::Const(11)]),
        Inst::make(2, Type(1), [Value::Const(20), Value::Const(21)]),
      ],
      Term::make_jmp(Jmp::make(
        [Value::Const(1)],
        Block::make(Symbol(1), empty(), empty(), Term::make_ret([Value::Const(42)])),
      )),
    ),
  ));

  let bytes = region.as_bytes();
  let checked: Region<Func> = Region::from_bytes(bytes).unwrap();
  // SAFETY: `bytes` came from `as_bytes()` on a valid `Region<Func>`.
  let unchecked: Region<Func> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(checked.name, unchecked.name);
  let cb = checked.entry.get();
  let ub = unchecked.entry.get();
  assert_eq!(cb.params.len(), ub.params.len());
  assert_eq!(cb.insts.len(), ub.insts.len());
  for i in 0..cb.insts.len() {
    assert_eq!(cb.insts[i].op, ub.insts[i].op);
    assert_eq!(cb.insts[i].typ, ub.insts[i].typ);
    assert_eq!(cb.insts[i].args.len(), ub.insts[i].args.len());
  }
  assert_eq!(checked.byte_len(), unchecked.byte_len());
}

#[test]
fn from_bytes_unchecked_primitive() {
  let region: Region<u32> = Region::new(42u32);
  let bytes = region.as_bytes();

  let checked: Region<u32> = Region::from_bytes(bytes).unwrap();
  // SAFETY: `bytes` came from `as_bytes()` on a valid `Region<u32>`.
  let unchecked: Region<u32> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(*checked, *unchecked);
  assert_eq!(checked.as_bytes(), unchecked.as_bytes());
}

#[test]
fn from_bytes_unchecked_generic_type() {
  let region: Region<Signature<u32>> =
    Region::new(Signature::<u32>::make([Type(0), Type(1)], [Type(2)], [10u32, 20, 30]));
  let bytes = region.as_bytes();

  let checked: Region<Signature<u32>> = Region::from_bytes(bytes).unwrap();
  // SAFETY: `bytes` came from `as_bytes()` on a valid `Region<Signature<u32>>`.
  let unchecked: Region<Signature<u32>> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(checked.params.len(), unchecked.params.len());
  assert_eq!(checked.returns.len(), unchecked.returns.len());
  assert_eq!(checked.custom.len(), unchecked.custom.len());
  for i in 0..checked.custom.len() {
    assert_eq!(checked.custom[i], unchecked.custom[i]);
  }
  assert_eq!(checked.as_bytes(), unchecked.as_bytes());
}

#[test]
fn from_buf_unchecked_fixed_buf() {
  use nearest::{Buf, FixedBuf};

  let region: Region<u32, FixedBuf<64>> = Region::new_in(42u32);
  let bytes = region.as_bytes();

  let mut buf = FixedBuf::<64>::new();
  buf.extend_from_slice(bytes);

  // SAFETY: `buf` contains bytes from a valid `Region<u32>`.
  let restored: Region<u32, FixedBuf<64>> = unsafe { Region::from_buf_unchecked(buf) };
  assert_eq!(*restored, 42);
}

#[test]
fn from_buf_unchecked_matches_checked() {
  use nearest::Buf;

  let region: Region<Func> = build_simple_func(77);
  let bytes = region.as_bytes();

  let checked: Region<Func> = Region::from_bytes(bytes).unwrap();

  let mut buf = nearest::AlignedBuf::<Func>::with_capacity(bytes.len() as u32);
  buf.extend_from_slice(bytes);
  // SAFETY: `buf` contains bytes from a valid `Region<Func>`.
  let unchecked: Region<Func> = unsafe { Region::from_buf_unchecked(buf) };

  assert_eq!(checked.name, unchecked.name);
  assert_eq!(checked.entry.get().name, unchecked.entry.get().name);
  assert_eq!(checked.byte_len(), unchecked.byte_len());
}

#[test]
fn from_bytes_unchecked_after_mutation_and_trim() {
  let mut region: Region<Func> = build_simple_func(50);

  region.session(|s| {
    let root = s.root();
    let name = s.nav(root, |f| &f.name);
    s.set(name, Symbol(999));
  });
  region.trim();

  let bytes = region.as_bytes();
  let checked: Region<Func> = Region::from_bytes(bytes).unwrap();
  // SAFETY: `bytes` came from `as_bytes()` on a valid (trimmed) `Region<Func>`.
  let unchecked: Region<Func> = unsafe { Region::from_bytes_unchecked(bytes) };

  assert_eq!(checked.name, Symbol(999));
  assert_eq!(unchecked.name, Symbol(999));
  assert_eq!(checked.byte_len(), unchecked.byte_len());
}

// ===========================================================================
// serde roundtrip tests
// ===========================================================================

#[cfg(feature = "serde")]
mod serde_tests {
  use nearest::{Flat, Near, NearList, Region};

  // Serde test types — use only 4-byte-aligned fields so there is no internal
  // padding. Types with enum padding (e.g. `Value`) cause uninitialized bytes
  // in the buffer due to a pre-existing issue in `gen_emit_self` (whole-value
  // `write_flat` copies uninit padding from the stack). Avoiding padding here
  // keeps serde tests Miri-clean.

  #[derive(Flat, Debug)]
  struct SNode {
    id: u32,
    items: NearList<u32>,
  }

  #[derive(Flat, Debug)]
  struct SFunc {
    name: u32,
    entry: Near<SNode>,
  }

  #[test]
  fn serde_json_roundtrip_simple() {
    let region: Region<SNode> = Region::new(SNode::make(42, [1u32, 2, 3]));
    let json = serde_json::to_string(&region).unwrap();
    let restored: Region<SNode> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.id, 42);
    assert_eq!(restored.items.len(), 3);
    assert_eq!(restored.items[0], 1);
    assert_eq!(restored.items[1], 2);
    assert_eq!(restored.items[2], 3);
  }

  #[test]
  fn serde_json_roundtrip_nested() {
    let region: Region<SFunc> = Region::new(SFunc::make(100, SNode::make(7, [10u32, 20, 30])));
    let json = serde_json::to_string(&region).unwrap();
    let restored: Region<SFunc> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.name, 100);
    assert_eq!(restored.entry.get().id, 7);
    assert_eq!(restored.entry.get().items.len(), 3);
    assert_eq!(restored.entry.get().items[0], 10);
    assert_eq!(restored.entry.get().items[1], 20);
    assert_eq!(restored.entry.get().items[2], 30);
  }

  #[test]
  fn serde_json_roundtrip_fixed_buf() {
    use nearest::FixedBuf;

    let region: Region<SNode, FixedBuf<256>> = Region::new_in(SNode::make(7, [10u32, 20]));
    let json = serde_json::to_string(&region).unwrap();
    let restored: Region<SNode, FixedBuf<256>> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.id, 7);
    assert_eq!(restored.items.len(), 2);
    assert_eq!(restored.items[0], 10);
    assert_eq!(restored.items[1], 20);
  }

  #[test]
  fn serde_json_invalid_bytes_rejected() {
    let bad_json = serde_json::to_string(&[0u8; 2]).unwrap();
    let result: Result<Region<SNode>, _> = serde_json::from_str(&bad_json);
    assert!(result.is_err());
  }

  #[test]
  fn serde_roundtrip_preserves_byte_equality() {
    let region: Region<SNode> = Region::new(SNode::make(99, [5u32, 6, 7, 8]));
    let original_bytes = region.as_bytes().to_vec();
    let json = serde_json::to_string(&region).unwrap();
    let restored: Region<SNode> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.as_bytes(), &original_bytes[..]);
  }
}

// ===========================================================================
// PartialEq / Eq tests
// ===========================================================================

#[derive(Flat, Debug, PartialEq, Eq)]
struct EqNode {
  id: u32,
  items: NearList<u32>,
}

#[derive(Flat, Debug, PartialEq, Eq)]
struct EqNested {
  label: u32,
  child: Near<EqNode>,
}

#[test]
fn region_eq_identical_builds() {
  let a = Region::new(EqNode::make(1, [10u32, 20, 30]));
  let b = Region::new(EqNode::make(1, [10u32, 20, 30]));
  assert_eq!(a, b);
}

#[test]
fn region_eq_clone() {
  let a = Region::new(EqNode::make(42, [1u32, 2, 3]));
  let b = a.clone();
  assert_eq!(a, b);
}

#[test]
fn region_ne_different_scalar() {
  let a = Region::new(EqNode::make(1, [10u32, 20]));
  let b = Region::new(EqNode::make(2, [10u32, 20]));
  assert_ne!(a, b);
}

#[test]
fn region_ne_different_list_len() {
  let a = Region::new(EqNode::make(1, [10u32, 20]));
  let b = Region::new(EqNode::make(1, [10u32, 20, 30]));
  assert_ne!(a, b);
}

#[test]
fn region_ne_different_list_values() {
  let a = Region::new(EqNode::make(1, [10u32, 20, 30]));
  let b = Region::new(EqNode::make(1, [10u32, 20, 99]));
  assert_ne!(a, b);
}

#[test]
fn region_eq_empty_lists() {
  let a: Region<EqNode> = Region::new(EqNode::make(5, empty()));
  let b: Region<EqNode> = Region::new(EqNode::make(5, empty()));
  assert_eq!(a, b);
}

#[test]
fn region_eq_after_trim_same_logical_content() {
  // Build two regions with the same logical content but different buffer layouts.
  // Region `a` is built fresh; region `b` is mutated then trimmed.
  let a = Region::new(EqNode::make(1, [100u32]));

  let mut b = Region::new(EqNode::make(1, [10u32, 20, 30]));
  b.session(|s| {
    let items = s.nav(s.root(), |n| &n.items);
    s.splice_list(items, [100u32]);
  });
  // Before trim, buffer has dead bytes — byte_len differs.
  assert!(b.byte_len() > a.byte_len());
  b.trim();
  // After trim, logical content matches.
  assert_eq!(a, b);
}

#[test]
fn region_eq_nested_near() {
  let a = Region::new(EqNested::make(1, EqNode::make(2, [3u32, 4])));
  let b = Region::new(EqNested::make(1, EqNode::make(2, [3u32, 4])));
  assert_eq!(a, b);
}

#[test]
fn region_ne_nested_near_different_child() {
  let a = Region::new(EqNested::make(1, EqNode::make(2, [3u32, 4])));
  let b = Region::new(EqNested::make(1, EqNode::make(9, [3u32, 4])));
  assert_ne!(a, b);
}

#[test]
fn region_eq_different_buffer_layouts() {
  // Two regions with identical logical content but different buffer layouts
  // due to mutation history. They should be equal when compared via root deref.
  let mut a = Region::new(EqNode::make(7, [1u32, 2]));
  a.session(|s| {
    let items = s.nav(s.root(), |n| &n.items);
    s.push_front(items, 0u32);
  });
  // a now has items [0, 1, 2] across 2 segments

  let b = Region::new(EqNode::make(7, [0u32, 1, 2]));
  // b has items [0, 1, 2] in 1 segment

  // Different buffer layouts but same logical content.
  assert_ne!(a.byte_len(), b.byte_len());
  assert_eq!(a, b);
}

// ===========================================================================
// Display tests
// ===========================================================================

#[derive(Flat, Debug)]
struct DisplayNode {
  value: u32,
  child: Near<u32>,
}

#[derive(Flat, Debug)]
struct DisplayList {
  items: NearList<u32>,
}

#[test]
fn display_near_delegates_to_target() {
  let region = Region::new(DisplayNode::make(1, 42u32));
  let output = format!("{}", region.child);
  assert_eq!(output, "42");
}

#[test]
fn display_near_with_format_spec() {
  let region = Region::new(DisplayNode::make(1, 42u32));
  let output = format!("{:>5}", region.child);
  assert_eq!(output, "   42");
}

#[test]
fn display_nearlist_empty() {
  let region = Region::new(DisplayList::make(empty()));
  let output = format!("{}", region.items);
  assert_eq!(output, "[]");
}

#[test]
fn display_nearlist_single() {
  let region = Region::new(DisplayList::make([7u32]));
  let output = format!("{}", region.items);
  assert_eq!(output, "[7]");
}

#[test]
fn display_nearlist_multiple() {
  let region = Region::new(DisplayList::make([1u32, 2, 3]));
  let output = format!("{}", region.items);
  assert_eq!(output, "[1, 2, 3]");
}

#[test]
fn display_region_delegates_to_root() {
  let region: Region<u32> = Region::new(42u32);
  let output = format!("{region}");
  assert_eq!(output, "42");
}
