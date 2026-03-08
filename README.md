# nearest

[![CI](https://github.com/uael/nearest/workflows/CI/badge.svg)](https://github.com/uael/nearest/actions)
[![docs.rs](https://img.shields.io/docsrs/nearest)](https://docs.rs/nearest)
[![crates.io](https://img.shields.io/crates/v/nearest)](https://crates.io/crates/nearest)
[![license](https://img.shields.io/crates/l/nearest)](LICENSE-MIT)

**Self-relative pointers for Rust. Clone is memcpy.**

Store entire data graphs — trees, DAGs, linked lists — in a single contiguous
byte buffer where every internal pointer is a 4-byte `i32` offset relative to
its own address. Cloning a region is a plain `memcpy` with zero fixup.
Serialization is `as_bytes()`. Mutation is compile-time safe.

## Quick look

```rust
use nearest::{Flat, Near, NearList, Region, near, list};

// Define recursive types with derive — no boilerplate.
#[derive(Flat, Debug)]
#[repr(C, u8)]
enum Expr {
  Lit(i64),
  Add { lhs: Near<Expr>, rhs: Near<Expr> },
}

// Build: (1 + 2)
let region = Region::new(Expr::make_add(
  near(Expr::make_lit(1)),
  near(Expr::make_lit(2)),
));

// Read — Region<T>: Deref<Target = T>
match &*region {
  Expr::Add { lhs, rhs } => {
    assert!(matches!(&**lhs, Expr::Lit(1)));
    assert!(matches!(&**rhs, Expr::Lit(2)));
  }
  _ => unreachable!(),
}

// Clone is memcpy — self-relative offsets just work.
let cloned = region.clone();
assert_eq!(region.as_bytes(), cloned.as_bytes());
```

## Why nearest?

Traditional Rust data structures use heap pointers. Cloning walks the entire
graph. Serialization requires a framework. Moving data between threads means
`Arc` overhead or deep copies.

`nearest` sidesteps all of this. A `Region<T>` is a flat byte buffer that you
can:

- **Clone** with `memcpy` (no pointer fixup)
- **Serialize** with `as_bytes()` / **deserialize** with `from_bytes()` (no schema, no codegen)
- **Send** across threads or processes (it's just bytes)
- **Mutate** safely through branded sessions (ghost-cell pattern, zero runtime cost)
- **Compact** with `trim()` to reclaim dead bytes after mutations

## Features at a glance

| Feature | How |
|---------|-----|
| Derive-driven construction | `#[derive(Flat)]` generates `make()` builders with zero boilerplate |
| Self-relative pointers | `Near<T>` — 4-byte `NonZero<i32>` offset, `Deref<Target = T>` |
| Inline linked lists | `NearList<T>` — segmented list with `O(1)` prepend/append, indexing, iteration |
| Compile-time safe mutation | Branded `Session` + `Ref` tokens — no `Ref` can escape or cross sessions |
| Region compaction | `trim()` deep-copies only reachable data, reclaiming dead bytes |
| `no_std` support | Works without `alloc`; use `FixedBuf<N>` for fully stack-based regions |
| Serialization | `as_bytes()` / `from_bytes()` with validation; optional `serde` feature |
| Miri-validated | All unsafe code tested under Miri with permissive provenance |

## Examples

### Expression trees

Recursive types work naturally — `Near<Expr>` is a 4-byte self-relative pointer:

```rust
use nearest::{Flat, Near, Region, near};

#[derive(Flat, Debug)]
#[repr(C, u8)]
enum Expr {
  Lit(i64),
  Bin { op: u8, lhs: Near<Expr>, rhs: Near<Expr> },
}

fn eval(e: &Expr) -> i64 {
  match e {
    Expr::Lit(n) => *n,
    Expr::Bin { op, lhs, rhs } => match op {
      b'+' => eval(lhs) + eval(rhs),
      b'*' => eval(lhs) * eval(rhs),
      _ => unreachable!(),
    },
  }
}

// (2 + 3) * 10 = 50
let region = Region::new(Expr::make_bin(
  b'*',
  near(Expr::make_bin(b'+', near(Expr::make_lit(2)), near(Expr::make_lit(3)))),
  near(Expr::make_lit(10)),
));
assert_eq!(eval(&region), 50);
```

### Mutation via branded sessions

Sessions use a ghost-cell pattern — `Ref` tokens are branded with a unique
lifetime so they can't escape the closure or be used in another session.
All checked at compile time, zero runtime cost:

```rust
use nearest::{Flat, NearList, Region, list};

#[derive(Flat)]
struct World { tick: u32, items: NearList<u32> }

let mut region = Region::new(World::make(0, list([1u32, 2, 3])));

region.session(|s| {
  let items = s.nav(s.root(), |w| &w.items);

  // Map: double every element (edits in place, zero allocation).
  s.map_list(items, |x| x * 2);

  // Filter: keep only values > 2 (no heap allocation).
  s.filter_list(items, |x| *x > 2);

  // Advance tick.
  s.set(s.nav(s.root(), |w| &w.tick), 1);
});

assert_eq!(region.tick, 1);
assert_eq!(region.items.len(), 2);  // [4, 6]
assert_eq!(region.items[0], 4);
```

### Stack-only regions (`no_std`)

Use `FixedBuf<N>` for zero-heap construction — ideal for embedded or real-time:

```rust
use nearest::{Flat, FixedBuf, NearList, Region, list};

#[derive(Flat)]
struct Packet { id: u32, payload: NearList<u8> }

let region: Region<Packet, FixedBuf<128>> =
  Region::new_in(Packet::make(42, list(b"hello".as_slice())));

assert_eq!(region.id, 42);
assert_eq!(region.payload.len(), 5);
```

### Composing regions

Build sub-trees as separate regions, then compose them with references:

```rust
use nearest::{Flat, Near, NearList, Region, near, list, empty};

#[derive(Flat)]
struct Node { label: u32, children: NearList<Node> }

// Build children independently.
let child_a = Region::new(Node::make(1, empty()));
let child_b = Region::new(Node::make(2, empty()));

// Compose: &*Region<T> implements Emit<T>.
let tree = Region::new(Node::make(0, list([&*child_a, &*child_b])));
assert_eq!(tree.children.len(), 2);
assert_eq!(tree.children[0].label, 1);

// The entire tree is one contiguous buffer.
// Clone is still just memcpy.
let cloned = tree.clone();
assert_eq!(cloned.children[1].label, 2);
```

## How it works

```text
Region<T> buffer layout:

  [ Root T fields | Near<U> targets | NearList segments | ... ]
  ^                     ^
  byte 0                |
              offset ───┘  (i32, relative to the Near<U> field's own address)
```

1. **Construction**: `Region::new(T::make(...))` writes values into a contiguous
   buffer. `Near<T>` fields store `i32` offsets pointing forward to their
   targets. `NearList<T>` stores a head offset + length.

2. **Reading**: `Region<T>: Deref<Target = T>`. `Near<T>: Deref<Target = T>` —
   resolves by adding the stored offset to its own address.

3. **Mutation**: `region.session(|s| { ... })` opens a branded session. All
   writes are append-only — new data goes at the end, old offsets are patched.
   Dead bytes accumulate but are never read.

4. **Compaction**: `region.trim()` deep-copies only reachable data into a fresh
   buffer, eliminating dead bytes.

## Comparison

| | nearest | rkyv | FlatBuffers | bumpalo |
|---|---|---|---|---|
| Self-relative pointers | `i32` offsets | relative pointers | `u32` offsets | no (heap ptrs) |
| Clone = memcpy | yes | no (absolute ptrs when deserialized) | n/a | no |
| Safe mutation | branded sessions | read-only archived data | read-only | `&mut` |
| Compaction | `trim()` | no | no | reset only |
| Construction | `#[derive(Flat)]` | `#[derive(Archive)]` | schema codegen | manual |
| `no_std` | yes | yes | yes | yes |

## Getting started

Requires **nightly** Rust (`nightly-2026-02-10`) for `#![feature(offset_of_enum)]`.

```toml
[dependencies]
nearest = "0.4"
```

See the [API documentation](https://docs.rs/nearest) for full reference, and
the [`examples/`](nearest/examples/) directory for complete runnable programs:

- **[`expr.rs`](nearest/examples/expr.rs)** — Expression tree evaluator with recursive `Near<Expr>`
- **[`json.rs`](nearest/examples/json.rs)** — JSON document model with heterogeneous enums
- **[`ecs.rs`](nearest/examples/ecs.rs)** — Entity-component system with physics, filtering, spawning
- **[`fsm.rs`](nearest/examples/fsm.rs)** — Finite state machine on a stack-only `FixedBuf`
- **[`fs.rs`](nearest/examples/fs.rs)** — Virtual file system with region composition and grafting

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
