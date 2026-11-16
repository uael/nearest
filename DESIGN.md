# nearest — design

One idea: a value and everything it points to live in **one buffer**, and every
pointer is a 4-byte offset **relative to its own address**. Copying the buffer
copies the whole graph, so `Clone` is a `memcpy`, and reading is plain `&T`.

Everything else in this document exists to make that idea safe to build, read
and edit with nothing but the borrow checker: no runtime checks, no tokens, no
sessions.

## Public surface

```text
#[flat]                    attribute: makes a struct or enum storable
trait Flat                 "may live in a region"        (implemented by #[flat])
trait Emit<'o, T>          "can be written as a T"       (values, copies, moves, builders)

struct Near<T: ?Sized>     the pointer        new · from_iter · as_pin_mut
struct Region<T>           the buffer         new · edit · to_bytes · from_bytes (unsafe) · byte_len
struct Slot<'o, T>         a pending change   put · at · push
```

Two traits, three types, one attribute, eleven methods. Nothing else is public
except the hidden plumbing the attribute expands to (`Sink`, `make`, and one
hidden method on each trait).

```rust
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use nearest::{flat, Near, Region};

#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }
#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }

// Build. Constructors take anything that can be written as the field's type.
let mut r = Region::new(Block::new(0, (Inst::new(1, [10, 20, 30]), Inst::new(2, ())), None));

// Read. It is just `&Block`.
assert_eq!(r.insts[0].args[1], 20);

// Clone. One memcpy.
let snapshot = r.clone();

// Edit. `root` is `Pin<&mut Block>`; ordinary patterns project it.
r.edit(|root, slot| {
    let Block { id, insts, .. } = root;
    *id.get_mut() = 7;                               // in place, no API involved
    slot.at(insts, |old, new| {                      // a change that needs new memory
        for inst in old { if inst.op == 1 { new.push(inst) } }   // old elements are moved
        new.push(Inst::new(9, [1, 2]));
    });
});

// Compact. A copy into a new region keeps only what is reachable.
r = Region::new(&*r);
```

## Why it is safe

**I1. A `Near` never exists outside a region.** It has no constructor, and is
not `Clone`, `Copy` or `Default`. Types that contain one cannot be built by
value, so nothing can be dereferenced at the wrong address.

**I2. Values that contain a `Near` are pinned.** `Near` is `!Unpin`, so every
type containing one is too, and the only mutable access `edit` hands out is
`Pin<&mut T>`. Safe code cannot `swap`, `replace` or move such a value. Plain
data is `Unpin`, so `Pin::get_mut` yields a real `&mut` and all of `std`
applies (`sort`, `swap`, assignment). `#[flat]` marks types `#[pin_v2]`, so
field, variant, array and slice patterns project a pin, and the compiler
rejects a manual `Unpin` impl.

**I3. A region is a tree.** Each allocation has one owning pointer, as with
`Box`. This is what makes `Near::as_pin_mut` sound. It holds by construction:
builders create fresh targets, `&T` copies, and a `Pin<&mut T>` *moves*.

**I4. A slot accepts moves only from the value it replaces.** `Slot<'o, T>`
and `Emit<'o, _>` carry a brand. `Slot::at` hands its closure the old value as
`Pin<&'p mut U>` for a fresh, universally quantified `'p`, and only
`Pin<&'p mut _>` implements `Emit<'p, _>`. A pin from anywhere else cannot be
coerced to `'p`. Since the old value is destroyed by the replacement, whatever
was moved out of it has exactly one owner afterwards. A pin is consumed by value
when used, so nothing is moved twice.

**I5. A place that received a replacement cannot be observed until the edit
ends.** `Slot::at` takes the place as `Pin<&'o mut U>` with the *full* lifetime
of its level; a reborrow is too short and is rejected. New data is built in a
scratch buffer and appended to the region when `edit` returns (also when it
unwinds), so no reference ever sees the buffer move.

**I6. No pointer is made from an integer.** `Near` resolves by offsetting a
pointer derived from `&self`. This is valid under Tree Borrows and rejected by
Stacked Borrows (the same position rkyv takes). CI runs Miri with
`-Zmiri-tree-borrows -Zmiri-strict-provenance`.

Compile-time limits, checked with `const` assertions when a type is first
stored: alignment at most 8, non-zero size, no drop glue. The only runtime
failure is running out of room: a region is at most `i32::MAX` bytes and
growing past that panics, as `Vec` does at its limit.

## Layout

- The buffer is 8-aligned and zero-filled when it grows. The root `T` is at byte 0.
- `Near<T>` is `#[repr(transparent)]` over `NonZero<i32>`: the distance from the
  pointer's own address to its target. `Option<Near<T>>` is the same 4 bytes,
  zero meaning `None`.
- `Near<T>`, `T: Sized`: the target is the `T`.
- `Near<[T]>` and `Near<str>`: the target is a `u32` length followed directly by
  the elements. The header is placed so the elements are aligned for `T`.
  Pointing at the header (never at the elements) keeps the offset non-zero even
  for an empty list.
- `#[flat]` gives structs `#[repr(C)]` (unless a `repr` is written) and enums
  `#[repr(u8)]`. Enum variants are numbered in order; explicit discriminants are
  rejected.

## Traits

```rust
pub unsafe trait Flat {
    #[doc(hidden)] const PLAIN: bool = false;      // no pointers, no padding: may be copied as bytes
    #[doc(hidden)] unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>);
}
pub unsafe trait Emit<'o, T: Flat> {
    #[doc(hidden)] unsafe fn emit(self, sink: &mut Sink<'o>, at: usize);
}
```

`Flat::store` writes `self` at position `at`, field by field. For a pointer
field it does one of two things: with `shift == None` it copies the target into
the sink and links to the copy (deep copy); with `shift == Some(d)` it writes the
same offset plus `d` (the value moved by `d` bytes, its targets did not). Every
algorithm in the crate is one of those two walks. Padding is never copied,
because writes are per field onto zeroed memory; `PLAIN` types skip the walk
and are copied as bytes.

`Flat` is implemented for `u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 bool char`,
`[T; N]`, `[T]`, `str`, `Near<T>`, `Option<Near<T>>`, and by `#[flat]`.

`Emit<'o, T>` is implemented by:

| Source | Writes | Meaning |
|---|---|---|
| `T` | `T` | a value (`7u32`, `Expr::Lit(1)`, `None`, `Point { x, y }`) |
| `&T` | `T` | deep copy, from any region |
| `Pin<&'o mut T>` | `T` | move, from the value the slot replaces |
| `Near::new(e)` | `Near<T>`, `Option<Near<T>>` | pointer to a new `T` |
| `[e; N]`, tuples up to 12, `()` | `Near<[T]>` | list literal; tuples may mix builders |
| `&[T]` | `Near<[T]>` | copy of a slice |
| `Near::from_iter(iter)` | `Near<[T]>` | list from an iterator |
| `&str` | `Near<str>` | text |
| `Type::new(..)`, `Enum::variant(..)` | the type | generated constructors |

There is deliberately no blanket impl over `IntoIterator`: it would conflict
with every other impl for `Near<[T]>` (tuples, moves, copies), and coherence
cannot be argued with.

## Editing

```rust
impl<T: Flat> Region<T> {
    pub fn edit<R>(&mut self, f: impl for<'a> FnOnce(Pin<&'a mut T>, &mut Slot<'a, T>) -> R) -> R;
}
impl<'o, T: Flat> Slot<'o, T> {
    pub fn put(&mut self, value: impl Emit<'o, T>);
    pub fn at<U: Flat, R>(&mut self, place: Pin<&'o mut U>,
                          f: impl for<'p> FnOnce(Pin<&'p mut U>, &mut Slot<'p, U>) -> R) -> R;
}
impl<'o, T: Flat> Slot<'o, Near<[T]>> {
    pub fn push(&mut self, value: impl Emit<'o, T>);
}
```

- A closure always has the same shape: `|old, slot|`. `edit` gives the root and
  the root's slot; `slot.at(place, ..)` narrows to a place inside.
- `put` stages a replacement for the slot's place. `push` (list places) appends
  to a replacement list, starting it empty. The replacement is written into the
  place when the closure returns. If neither is called the place is left as is.
  If the closure panics nothing is written.
- Everything a replacement needs is appended to a scratch buffer. The staged
  value is then written into its place with one `memcpy`, with its pointers
  aimed at where the scratch will be once appended. Nothing in between runs
  user code.
- What `put` cannot express is not offered: there is no way to make two pointers
  share a target, to move a value between unrelated places, or to keep a
  reference across a change.

Idioms that need no method of ours:

| To do this | Write |
|---|---|
| assign a plain field | `*id.get_mut() = 7` |
| overwrite with a plain value | `let mut e = e; e.set(Expr::Lit(0))` |
| clear an optional pointer | `let mut next = next; next.set(None)` |
| follow an optional pointer | `next.as_pin_mut().map(Near::as_pin_mut)` |
| sort a list of plain data | `list.as_pin_mut().get_mut().sort()` |
| visit list elements | `for x in list { .. }` |
| compact | `region = Region::new(&*region)` |

## Bytes

`to_bytes` returns a compact, fully initialized image: it is a deep copy into a
fresh region, so garbage and padding never leave the process. `from_bytes` is
`unsafe`: the bytes must come from `to_bytes` for the same type in the same
program. Checking untrusted bytes is a runtime concern and is not part of this
crate.

## What the attribute generates

For `#[flat] struct Inst { op: u8, args: Near<[u32]> }`:

```rust
#[pin_v2] #[repr(C)] struct Inst { op: u8, args: Near<[u32]> }

unsafe impl Flat for Inst {
    const PLAIN: bool = <u8 as Flat>::PLAIN && <Near<[u32]> as Flat>::PLAIN
        && size_of::<Self>() == size_of::<u8>() + size_of::<Near<[u32]>>();
    unsafe fn store(&self, sink: &mut Sink<'_>, at: usize, shift: Option<i32>) {
        const { assert!(!core::mem::needs_drop::<Self>()) };
        unsafe {
            Flat::store(&self.op, sink, at + offset_of!(Self, op), shift);
            Flat::store(&self.args, sink, at + offset_of!(Self, args), shift);
        }
    }
}
impl Inst {
    fn new<'o>(op: impl Emit<'o, u8>, args: impl Emit<'o, Near<[u32]>>) -> impl Emit<'o, Self> {
        unsafe { make(move |sink, at| {
            Emit::emit(op, sink, at + offset_of!(Self, op));
            Emit::emit(args, sink, at + offset_of!(Self, args));
        }) }
    }
}
```

Enums get `#[repr(u8)]`, a `store` that matches on `self` and writes the variant
index then the fields, and one constructor per non-unit variant named after it
in snake case (`Expr::add`). Field offsets for constructors come from hidden
`#[repr(C)]` mirror structs `(u8, fields..)`, which the language defines to match
a `#[repr(u8)]` enum, so user crates need no `offset_of_enum`.

## Toolchain

`nightly-2026-02-10` (already pinned). The library uses `ptr_metadata`; code that
edits a region uses `pin_ergonomics` for `#[pin_v2]` pattern projection.

Do not move the pin past `nightly-2026-03-31`. Every nightly from 2026-04-02 to
at least 2026-11-13 hits an internal compiler error when a pinned binding is
used inside a closure, which is how every edit is written:

```rust
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]
use core::pin::Pin;
#[pin_v2] struct S { a: u32, b: u32 }
fn outer(_: impl FnOnce(Pin<&mut S>)) {}
fn main() { outer(|root| { let S { a, .. } = root; *a.get_mut() = 1; }); }
// error: internal compiler error: explicit deref of non-derefable type
```

No upstream report was found for it. If it is never fixed, the fallback is to
have `#[flat]` generate a `project()` method per type; nothing else in the
design changes.

## Evidence

A 350-line prototype of this design was built before the implementation
(`scratchpad/v3`): 8 runtime tests including a 400-step randomized comparison
against a `Vec` model pass natively and under Miri with strict provenance and
Tree Borrows, and 14 misuse programs (moving from an unrelated place, using a
place after replacing it, reborrowed places, escaping pins or slots, swapping
pinned values, cross-region moves, double moves) are all compile errors.
