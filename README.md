# nearest

[![CI](https://github.com/uael/nearest/workflows/CI/badge.svg)](https://github.com/uael/nearest/actions)
[![docs.rs](https://img.shields.io/docsrs/nearest)](https://docs.rs/nearest)
[![crates.io](https://img.shields.io/crates/v/nearest)](https://crates.io/crates/nearest)
[![license](https://img.shields.io/crates/l/nearest)](LICENSE-MIT)

**Self-relative pointers in one buffer. Clone is `memcpy`.**

A `Region<T>` keeps a `T` and everything it points to in one buffer. A pointer is a `Near<T>`:
4 bytes (or 2, or 8) holding the distance from its own address to its pointee. So copying the
buffer copies the whole value, reading is plain `&T` with `Near` dereferencing like `Box`, and
the bytes mean the same at any address, so a region can be written out and read back.

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

// Save and load. A compact region is its own image; loading checks the bytes.
let bytes = r.as_bytes().unwrap();
let copy = Region::<Block>::from_bytes(bytes)?;
// A memory-mapped file, or any slice aligned as a region is (as a `u64`), is read in place.
let block: &Block = nearest::view(map)?;
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
| `Region<T, S = Heap>` | the buffer, in a storage: `new`, `new_in`, `edit`, `as_bytes`, `from_bytes`, `byte_len`; `unsafe` `from_bytes_unchecked` |
| `view` | a `const` function: the value at the start of a slice, checked and read in place; `unsafe` `view_unchecked` |
| `Storage`, `Heap`, `Inline<N>` | where a region keeps its bytes (an `unsafe` trait), the heap (`Heap::with_capacity`), `N` bytes inside the region |
| `Near<T, O = i32>` | the pointer; builders `Near::new(e)` and `Near::from_iter(iter)`; `as_pin_mut`, `iter_pin_mut` in a pinned view |
| `Offset` | the integer a `Near` keeps its distance in: `i16`, `i32`, `i64`, `i32le`.. of `puns::num`, or a type of yours (an `unsafe` trait) |
| `puns` | the crate [`puns`](https://docs.rs/puns), re-exported: its layouts check the values of a region; `num` has numbers of a fixed byte order |
| `Edit<'e, T>` | the handle of a place holding a `T` during the edit `'e`, always behind `&mut` |
| `ListEdit<'e, T>` | the handle of a `Near<[T]>`: `push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `try_at`, `set`, `replace`, `update`, `iter_pin_mut` |
| `OptionEdit<'e, T>` | the handle of an `Option<Near<T>>`: `as_mut`, `insert`, `take`, `set`, `replace`, `update` |
| `StrEdit<'e>` | the handle of a `Near<str>`: `set`, `replace`, `update` |
| `Old<'e, T>` | a value that left its place: write it anywhere to move it there |
| `Error` | the error of every loader: `Invalid(puns::Error)`, `StorageFull` |
| `Flat`, `Emit<T>` | "may live in a region", "can be written as a `T`" |

Anything that can be written as a `T` is accepted wherever a `T` is built: a value, a copy
(`&T`, from any region), a move (`Old`), a list literal (`[a, b]`, `(a, b)`, `()`), a slice, an
iterator, a string, or a generated constructor. A constructor parameter for a field written
`Near<X>` takes a builder of `X` and allocates the pointee; `Near::new(x)` builds a pointer
where one is expected (optional pointers, lists of pointers). For a struct, `XInit` takes the same
builders by name, in any order: `FuncInit { name: "f", blocks: () }` is `Func::new("f", ())`.

Handles reach every place in a region. A field of plain data reads and writes with one star,
`*e.count() += 1`, and a list of it is a real `&mut [T]`.
A handle borrows its parent handle, so one handle is usable at a time, and no reference into
the buffer outlives a change that may grow it. Every handle has `set` (assign); `replace`
(assign and return the old value), `ListEdit::remove` and `OptionEdit::take` return the value
that left. Such an `Old` keeps the region borrowed for the whole edit and can be written
into any place of it in O(1): wrapping, hoisting, swapping and moving elements between lists
are ordinary statements. `update` builds the new value from the old one in a closure, so that
wrapping needs no placeholder:

```rust
e.update(Expr::neg);                                         // wrap
e.update(|old| Expr::add(old, Expr::Lit(1)));
node.next().update(|rest| Near::new(Node::new(2, rest)));    // put a node in front
```

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
again afterwards. A `#[flat]` type is `Unpin` exactly when its fields are (an impl that the macro
writes, so a hand-written one conflicts), and cannot be `#[pin_v2]`.

## Pointer widths

A `Near<T>` keeps its distance in an `i32`, so a region holds at most 2 GiB. `Near<T, O>` picks
another width: `i16`, 2 bytes, for regions of at most 32 KiB, or `i64`, 8 bytes, for regions as
large as memory. List and string lengths follow. Each field picks its width, and a type can take
it as a parameter:

```rust
#[flat]
struct Table<O: Offset> { name: Near<str, O>, rows: Near<[u64], O> }

let small = Region::new(Table::<i16>::new("small", [1, 2, 3]));   // 4 bytes at the root
let large = Region::new(Table::<i64>::new("large", [1, 2, 3]));   // 16 bytes at the root
```

The pointers of the root bound the region, and a pointer may only lead to types whose pointers
are at least as wide (checked at compile time), so every distance fits. Building or editing past
the bound panics, and bytes longer than it do not load.

`i16`, `i32` (the default) and `i64`, like the numbers of Rust, are in the byte order of the
machine: a region of them loads as the same value only on a machine of the same byte order.
**Bytes that cross machines** (a file, the network) use the offsets of `puns::num` (`i16le`,
`i32le`, `i64le` and their big-endian counterparts) and its numbers (`u32le`, `f64be`,
`u16bec<MAGIC>`, ...), which make the bytes of a region the same on every machine. Where the
order is the machine's they cost nothing (the same instructions on x86-64); elsewhere, a byte
swap:

```rust
use nearest::puns::num::{i32le, u16le};

#[flat]
struct Table { name: Near<str, i32le>, rows: Near<[u16le], i32le> }
```

## One version read as another

Every `#[flat]` type is a [`puns`](https://docs.rs/puns) type too, and a `Near` holds an eager
aim of `puns`: its offset, and its pointee at the address of the pointer plus the offset, which
`puns` checks with the value that holds the pointer. The loaders check bytes with `puns`, pointees
included, and that they are compact. So the bytes saved by a program load in its next version,
in place, when they are a value of it:

```rust
#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>) }

#[flat]
enum NewExpr { Lit(i64), Neg(Near<NewExpr>), Abs(Near<NewExpr>) }   // one more variant

let old = Region::new(Expr::neg(Expr::Lit(2)));
let new: &NewExpr = nearest::view(old.as_bytes().unwrap())?;      // checked, the same bytes
```

The other way, an `Abs` is refused where its tag is. `#[pun(valid(..))]` and
`#[pun(assert(..))]` can be written on a `#[flat]` type: requirements that the loaders check
(regions do not check them when they build or edit a value: one that breaks a rule is saved,
and refused when loaded).

## Bytes

`as_bytes` returns the compact image of a region, the bytes a deep copy writes, borrowed: a
region is compact until it is edited, and a copy, `Region::new(&*r)`, always is. `from_bytes`
copies bytes into a new region and checks them there; `nearest::view` checks a slice, such as a
memory-mapped file, and reads it in place, in a `const` too. Checking accepts the images of valid
values whose pointees are in the order of a copy, with bytes that no pointer reaches between
them, in one pass, in safe code, without recursion with the depth of the data, panics or memory
from you: each value as its layout says (`puns`), its rules included. So the pointees of a loaded
image form a tree, which editing it in place relies on: two pointers to one pointee, a pointee
that overlaps a value, and a cycle are refused. Every loader returns `Result<_, Error>`:
`Error::Invalid` says what `puns` found, and `Error::StorageFull` that a valid image is larger
than the storage of `from_bytes`. For bytes you
trust, or check yourself, `from_bytes_unchecked` and `view_unchecked` skip the check and are
`unsafe`. An offset or a length is valid only as its
`Offset` writes it: an `i64` that does not fit `isize` on a 32-bit target is invalid. The bytes
hold the layout of the target, its values in the byte order of their types, and distances and
lengths as their `Offset` writes them: native-endian for `i16`, `i32` and `i64`, or as their
names say for the offsets of `puns::num`.

## Storage

A region keeps its buffer in a storage. `Region<T>` is on the `Heap`, where it grows as
needed; `Inline<N>` keeps it in `N` bytes inside the `Region` value, with no heap, and growing
past them panics (`from_bytes` returns an error). Other memory, an arena or a memory-mapped
file, is a storage of your own: the `unsafe` trait `Storage` asks for two accessors to its
memory, the second of which grows it when asked for more, if it can. A region in any storage
is built with `new_in`, or with `from_bytes`; everything else is the same.

```rust
let mut r = Region::new_in(Inline::<256>::default(), Block::new(0, (), None));
r.edit().insts().push(Inst::new(1, ()));
let moved = r;                                       // the bytes move with it
```

## Safety model

- A `Near` has no constructor and is not `Clone`: it only exists inside a region.
- No safe API gives a `&mut T` for a `T` that contains a `Near`; such places are reached
  through handles, which are unsized, so they cannot be swapped or assigned through.
- A region is a tree: builders make fresh pointees, `&T` copies, and the only moves are `Old`
  (used at most once) and list operations. Bytes that are no longer reachable are never
  written again, so an `Old` stays intact for as long as the edit lasts.
- A handle holds a position, not an address, so it stays valid when the buffer grows. Every
  access of an edit goes through one pointer stored in the region's buffer.
- Handles and `Old` are neither `Send` nor `Sync`: an edit stays on one thread.
- A panic leaves every place either unchanged or holding its complete new value.
- No pointer is made from an integer. Miri runs the tests with Tree Borrows and strict
  provenance.

Stored types must be no more aligned than a `u64` (the words of a region's memory: 8 bytes on
x86-64, 4 on `i686`), and have non-zero size and no drop glue (checked at compile time). Miri runs
the tests on x86-64, on `i686` (32 bits) and on `s390x` (big-endian). A region holds at most what
the pointers of its root reach (2 GiB by default), or what its storage gives. The specification is
[`nearest/CLAUDE.md`](nearest/CLAUDE.md).

## Getting started

Stable Rust (1.99 or later, as `puns`), `#![no_std]`, no dependencies besides the macro and
[`puns`](https://docs.rs/puns), here at a commit of its repository, whose aims are not released
yet.

```sh
cargo add nearest
```

The default feature `alloc` provides the heap. Without it (`default-features = false`) the
crate needs no allocator: regions live in other storage, and `Heap` (so `Region<T>`) is not
available. Every loader is, since checking takes no memory.

The opt-in feature `serde` (with `alloc`) makes a `Region` `Serialize` and `Deserialize` as its
bytes, checked when loaded, like `as_bytes` and `from_bytes`.

Examples: [`expr.rs`](nearest/examples/expr.rs) simplifies an expression tree by rewriting it
in place; [`passes.rs`](nearest/examples/passes.rs) runs constant folding and dead-code
elimination on a small IR; [`ir.rs`](nearest/examples/ir.rs) is a compiler IR that is its own
file format, the same bytes on every machine, saved, loaded and run in place.

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
