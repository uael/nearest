# nearest

[![CI](https://github.com/uael/nearest/workflows/CI/badge.svg)](https://github.com/uael/nearest/actions)
[![docs.rs](https://img.shields.io/docsrs/nearest)](https://docs.rs/nearest)
[![crates.io](https://img.shields.io/crates/v/nearest)](https://crates.io/crates/nearest)
[![license](https://img.shields.io/crates/l/nearest)](LICENSE-MIT)

**Self-relative pointers in one buffer. Clone is `memcpy`.**

A `Region<T>` keeps a `T` and everything it points to in one buffer. A pointer is a `Near<T>`:
4 bytes holding the distance from its own address to its target. So copying the buffer copies
the whole value, reading is plain `&T` with `Near` dereferencing like `Box`, and the bytes mean
the same at any address, so a region can be written out and read back.

```rust
use nearest::{Edit, Near, Region, flat};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

// Build. Constructors take anything that can be written as the field's type.
let mut r = Region::new(Block::new(0, (Inst::new(1, [10, 20, 30]), Inst::new(2, ())), None));

// Read. It is `&Block`.
assert_eq!(r.insts[0].args[1], 20);

// Edit. `r.edit()` is `&mut Edit<Block>`; every field has a method of the same name.
let b = r.edit();
*b.id() = 7;
b.insts().push(Inst::new(9, [1, 2])).args().push(3);   // `push` returns the new element
b.insts().at(0).args()[1] = 21;                         // plain data is a real slice
b.insts().retain(|inst| inst.op != 2);
b.next().insert(Block::new(1, (), None)).insts().push(Inst::new(4, ()));
b.next().set(None);

// Compact. A copy into a new region keeps only what is reachable.
r = Region::new(&*r);
```

Enums get a view of their variant, and `replace` hands over the old value, which moves into
the new one instead of being copied:

```rust
#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

fn inner(e: &Expr) -> &Expr { match e { Expr::Neg(x) => x, e => e } }

/// `--x` becomes `x`, bottom up.
fn simplify(e: &mut Edit<Expr>) {
  match e.as_mut() {
    ExprMut::Lit(_) => {}
    ExprMut::Neg(x) => simplify(x),
    ExprMut::Add(args) => { simplify(args._0()); simplify(args._1()); }
  }
  if let Expr::Neg(x) = &**e && let Expr::Neg(_) = &**x {
    e.replace(|old| old.map(|e| inner(inner(e))));   // the inner value moves up, O(1)
  }
}

let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2))));
simplify(r.edit());
r.edit().replace(|old| Expr::neg(old));              // wrap: the old value moves, O(1)
```

## The API

| Item | |
|---|---|
| `#[flat]` | makes a struct or enum storable; generates constructors and the handle |
| `Region<T>` | the buffer: `new`, `edit`, `to_bytes`, `from_bytes` (unsafe), `byte_len` |
| `Near<T>` | the pointer; builders `Near::new(e)` and `Near::from_iter(iter)` |
| `Edit<T>` | the handle of a place holding a `T`, always behind `&mut` |
| `ListEdit<T>` | the handle of a `Near<[T]>`: `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `set`, `replace` |
| `OptionEdit<T>` | the handle of an `Option<Near<T>>`: `as_mut`, `insert`, `set`, `replace` |
| `StrEdit` | the handle of a `Near<str>`: `set`, `replace` |
| `Old<'a, T>` | the value being replaced: write it to move it, `map` it to a part |
| `Flat`, `Emit<T>` | "may live in a region", "can be written as a `T`" |

Anything that can be written as a `T` is accepted wherever a `T` is built: a value, a copy
(`&T`, from any region), a move (`Old`), a list literal (`[a, b]`, `(a, b)`, `()`), a slice, an
iterator, a string, or a generated constructor. A constructor parameter for a field written
`Near<X>` takes a builder of `X` and allocates the target; `Near::new(x)` builds a pointer
where one is expected (optional pointers, lists of pointers).

Handles reach every place in a region. Plain data (`T: Copy`) is a real `&mut T` or `&mut [T]`.
A handle borrows its parent handle, so one handle is usable at a time, and no reference into
the buffer outlives a change that may grow it.

## Safety model

- A `Near` has no constructor and is not `Clone`: it only exists inside a region.
- No safe API gives a `&mut T` for a `T` that contains a `Near`; such places are reached
  through handles, which are unsized, so they cannot be swapped or assigned through.
- A region is a tree: builders make fresh targets, `&T` copies, and the only moves are `Old`
  (used at most once) and list operations.
- A handle holds a position, not an address, so it stays valid when the buffer grows.
- A panic leaves every place either unchanged or holding its complete new value.
- No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
  provenance.

Stored types must have alignment at most 8, non-zero size and no drop glue (checked at compile
time). A region holds at most 2 GiB. [`DESIGN.md`](DESIGN.md) is the specification.

## Getting started

Stable Rust (1.95 or later), `#![no_std]` + `alloc`, no dependencies besides the macro.

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
