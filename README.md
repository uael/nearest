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

// Move. A value that leaves its place can be written anywhere else in the region, in O(1).
let inst = b.insts().remove(0);                                 // like `Vec::remove`
b.next().insert(Block::new(1, (), None)).insts().push(inst);    // the instruction changes block
let tail = b.next().take();                                     // like `Option::take`
drop(tail);

// Compact. A copy into a new region keeps only what is reachable.
r = Region::new(&*r);

// Save and load. Loading checks the bytes.
let bytes = r.to_bytes();
let copy = Region::<Block>::from_bytes(&bytes)?;
// A memory-mapped file, or any 8-aligned slice, is read in place.
let block: &Block = Region::view(map)?;
```

Enums get a view of their variant, and `replace` returns the old value, like `mem::replace`;
written somewhere else in the region, it moves there:

```rust
#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

/// `--x` becomes `x`, bottom up.
fn simplify(e: &mut Edit<Expr>) {
  match e.as_mut() {
    ExprMut::Lit(_) => {}
    ExprMut::Neg(x) => simplify(x),
    ExprMut::Add(args) => { simplify(args._0()); simplify(args._1()); }
  }
  if let ExprMut::Neg(x) = e.as_mut() && let ExprMut::Neg(y) = x.as_mut() {
    let inner = y.replace(Expr::Lit(0));     // like `mem::replace`: the old value comes back
    e.set(inner);                            // and moves up, O(1)
  }
}

let mut r = Region::new(Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2))));
let e = r.edit();
simplify(e);
let old = e.replace(Expr::Lit(0));
e.set(Expr::neg(old));                       // wrap: the old value moves, O(1)
if let ExprMut::Neg(x) = e.as_mut() && let ExprMut::Add(args) = x.as_mut() {
  let a = args._0().replace(Expr::Lit(0));   // swap two children: three moves
  let b = args._1().replace(a);
  args._0().set(b);
}
```

## The API

| Item | |
|---|---|
| `#[flat]` | makes a struct or enum storable; generates constructors and the handle |
| `Region<T>` | the buffer: `new`, `edit`, `to_bytes`, `from_bytes`, `view`, `byte_len`; `unsafe` `from_bytes_unchecked`, `view_unchecked` |
| `Near<T>` | the pointer; builders `Near::new(e)` and `Near::from_iter(iter)`; `as_pin_mut`, `iter_pin_mut` in a pinned view |
| `Edit<'e, T>` | the handle of a place holding a `T` during the edit `'e`, always behind `&mut` |
| `ListEdit<'e, T>` | the handle of a `Near<[T]>`: `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `get_mut`, `set`, `replace`, `iter_pin_mut` |
| `OptionEdit<'e, T>` | the handle of an `Option<Near<T>>`: `as_mut`, `insert`, `take`, `set` |
| `StrEdit<'e>` | the handle of a `Near<str>`: `set`, `replace` |
| `Old<'e, T>` | a value that left its place: write it anywhere to move it there |
| `Invalid` | the error of `from_bytes` and `view` |
| `Flat`, `Emit<T>` | "may live in a region", "can be written as a `T`" |

Anything that can be written as a `T` is accepted wherever a `T` is built: a value, a copy
(`&T`, from any region), a move (`Old`), a list literal (`[a, b]`, `(a, b)`, `()`), a slice, an
iterator, a string, or a generated constructor. A constructor parameter for a field written
`Near<X>` takes a builder of `X` and allocates the target; `Near::new(x)` builds a pointer
where one is expected (optional pointers, lists of pointers).

Handles reach every place in a region. Plain data (`T: Copy`) is a real `&mut T` or `&mut [T]`.
A handle borrows its parent handle, so one handle is usable at a time, and no reference into
the buffer outlives a change that may grow it. Every handle has `set` (assign); `replace`
(assign and return the old value), `ListEdit::remove` and `OptionEdit::take` return the value
that left. Such an `Old` keeps the region borrowed for the whole edit and can be written
into any place of it in O(1): wrapping, hoisting, swapping and moving elements between lists
are ordinary statements.

## Several fields at once

One handle is usable at a time, because any handle may grow the buffer. Where an edit only
reads, writes plain data or assigns values without pointers, `pin()` on a handle gives a
`Pin<&mut T>` in which several places are borrowed together:

```rust
/// Adds `by` to every literal.
fn bump(e: Pin<&mut Expr>, by: i64) {
  match e.project() {                        // generated: the fields, pinned
    ExprPin::Lit(mut n) => *n += by,
    ExprPin::Neg(x) => bump(x.as_pin_mut(), by),
    ExprPin::Add(a, b) => {                  // both at once
      bump(a.as_pin_mut(), by);
      bump(b.as_pin_mut(), by);
    }
  }
}

bump(r.edit().pin(), 10);
```

Nothing is pushed, built or moved through a pin; the handle is borrowed meanwhile and edits
again afterwards. With the `nightly` feature (nightly Rust and `#![feature(pin_ergonomics)]`
in your crate) a pattern does what `project` does: `match e { Expr::Add(a, b) => .. }`.

## Bytes

`to_bytes` returns the compact image of a region, the bytes a deep copy writes; it borrows the
buffer when the region has not been edited since it was built or loaded. `from_bytes` checks
bytes and copies them into a new region; `view` checks an 8-aligned slice, such as a
memory-mapped file, and reads it in place. Checking accepts exactly the compact images of valid
values, in one pass, in safe code, without recursion or panics. For bytes you trust,
`from_bytes_unchecked` and `view_unchecked` skip the check and are `unsafe`. The bytes hold
native-endian integers and the layout of the target.

## The `unstable` feature

`nearest = { version = "..", features = ["unstable"] }` adds `update` to every handle: the
closure form of `replace`, so that wrapping a value needs no placeholder.

```rust
e.update(Expr::neg);                                         // wrap
e.update(|old| Expr::add(old, Expr::Lit(1)));
node.next().update(|rest| Near::new(Node::new(2, rest)));    // put a node in front
```

It works on stable Rust; "unstable" is about the API, which may change in any release.

## Safety model

- A `Near` has no constructor and is not `Clone`: it only exists inside a region.
- No safe API gives a `&mut T` for a `T` that contains a `Near`; such places are reached
  through handles, which are unsized, so they cannot be swapped or assigned through.
- A region is a tree: builders make fresh targets, `&T` copies, and the only moves are `Old`
  (used at most once) and list operations. Bytes that are no longer reachable are never
  written again, so an `Old` stays intact for as long as the edit lasts.
- A handle holds a position, not an address, so it stays valid when the buffer grows. Every
  access of an edit goes through one pointer stored in the region's buffer.
- Handles and `Old` are neither `Send` nor `Sync`: an edit stays on one thread.
- A panic leaves every place either unchanged or holding its complete new value.
- No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
  provenance.

Stored types must have alignment at most 8, non-zero size and no drop glue (checked at compile
time). A region holds at most 2 GiB. [`DESIGN.md`](DESIGN.md) is the specification.

## Getting started

Stable Rust (1.95 or later), `#![no_std]` + `alloc`, no dependencies besides the macro.

```sh
cargo add nearest
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
