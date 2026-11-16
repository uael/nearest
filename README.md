# nearest

[![CI](https://github.com/uael/nearest/workflows/CI/badge.svg)](https://github.com/uael/nearest/actions)
[![docs.rs](https://img.shields.io/docsrs/nearest)](https://docs.rs/nearest)
[![crates.io](https://img.shields.io/crates/v/nearest)](https://crates.io/crates/nearest)
[![license](https://img.shields.io/crates/l/nearest)](LICENSE-MIT)

**Self-relative pointers in one buffer. Clone is `memcpy`.**

A `Region<T>` holds a `T` and everything it points to in one contiguous buffer. Every pointer
is a `Near<T>`: 4 bytes holding the distance from its own address to its target. Copying the
buffer copies the whole graph, reading is plain `&T`, and edits are checked by the borrow
checker with no runtime checks.

```rust
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

// Build. Constructors take anything that can be written as the field's type.
let mut r = Region::new(Block::new(0, (Inst::new(1, [10, 20, 30]), Inst::new(2, ())), None));

// Read. It is just `&Block`.
assert_eq!(r.insts[0].args[1], 20);

// Clone. One memcpy.
let snapshot = r.clone();

// Edit. `root` is `Pin<&mut Block>`; ordinary patterns project it.
r.edit(|root, slot| {
  let Block { id, insts, .. } = root;
  *id.get_mut() = 7;                                         // in place
  slot.at(insts, |old, new| {                                // a change that needs new memory
    for inst in old { if inst.op == 1 { new.push(inst) } }   // old elements move
    new.push(Inst::new(9, [1, 2]));
  });
});

// Compact. A copy into a new region keeps only what is reachable.
r = Region::new(&*r);
```

## The API

| Item | |
|---|---|
| `#[flat]` | makes a struct or enum storable; generates `Type::new(..)` and `Enum::variant(..)` |
| `Near<T>` | the pointer: `Near::new(e)`, `Near::from_iter(iter)`, `as_pin_mut` |
| `Region<T>` | the buffer: `new`, `edit`, `to_bytes`, `from_bytes` (unsafe), `byte_len` |
| `Slot<'o, T>` | a pending change during `edit`: `put`, `at`, `push` |
| `Flat`, `Emit<'o, T>` | "may live in a region", "can be written as a `T`" |

Anything that can be written as a `T` is accepted wherever a `T` is built: a value, a copy
(`&T`, from any region), a move (`Pin<&mut T>`, from the value being replaced), a list literal
(`[a, b]`, `(a, b)`, `()`), a slice, an iterator, a string, or a generated constructor.

## Safety model

- A `Near` has no constructor and is not `Clone`: it only exists inside a region.
- `Near` is `!Unpin`, so values containing one are reached as `&T` or `Pin<&mut T>`;
  `#[flat]` makes types `#[pin_v2]` so that patterns project the pin. Plain data stays
  `Unpin` and gets a real `&mut`.
- A region is a tree: builders make fresh targets, `&T` copies, and a `Pin<&mut T>` moves. A
  slot accepts moves only from the value it replaces, which a lifetime brand enforces.
- New data is built in a scratch buffer and appended when the edit ends, also on panic.
- No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
  provenance.

Types must have alignment at most 8, non-zero size and no drop glue (checked at compile
time). A region holds at most 2 GiB.

## Getting started

Requires nightly Rust (`nightly-2026-02-10`). Crates that use `#[flat]` enable
`#![feature(pin_ergonomics)]`.

```toml
[dependencies]
nearest = "0.4"
```

Examples: [`expr.rs`](nearest/examples/expr.rs) simplifies an expression tree by rewriting it
in place; [`ir.rs`](nearest/examples/ir.rs) runs constant folding and dead-code elimination on
a small IR.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
