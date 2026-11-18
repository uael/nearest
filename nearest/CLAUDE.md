# nearest — specification

This file is the specification of the crate: code, tests and documentation follow it. Commands,
toolchain, releases and style are in the root `CLAUDE.md`. The examples are the crate docs
(`src/lib.rs`, also in `README.md`), tested as doc tests, and `examples/` (`expr`, `ir`).

## Idea

A `Region<T>` keeps a `T` and everything it points to in one buffer. A pointer is a `Near<T, O>`:
the distance from its own address to its target, an `O` (`i32`, 4 bytes, by default; `i16` or
`i64`). So copying the buffer copies the value (`Clone` is a `memcpy`), reading is plain `&T`
(`Near` derefs like `Box`), and the bytes mean the same at any address (a region is saved and loaded
as bytes). Stable Rust, `#![no_std]`, no dependency but the macro (and `serde`, optional). Aliasing,
lifetimes, moves and the tree shape are enforced by types; see Runtime checks for the rest.

## Public surface

- `#[flat]`: attribute for structs and enums (see Generated code).
- `trait Flat`: `type Edit<'e>: ?Sized`; the rest is hidden. `trait Emit<T>`: all hidden; "can
  be written as a `T`". `type Edit<'e, T> = <T as Flat>::Edit<'e>`: the handle of a place
  holding a `T`.
- `Near<T, O: Offset = i32>`: builders `new`, `from_iter`; pinned view `as_pin_mut`,
  `iter_pin_mut`.
- `unsafe trait Offset`: the integer of a `Near`, implemented for `i16`, `i32`, `i64`, and open
  to the user's (an integer of a fixed byte order, say): `type NonZero`, `LIMIT` (defaulted from
  the size), `from_isize`, `to_isize`; its `# Safety` is the contract (see Offsets). Its
  supertraits `Flat + Copy + Debug + Eq + Ord + Hash` let a type generic over the width be
  `#[flat]` (which bounds every type parameter by `Flat`) and derive the usual traits.
- `Region<T, S: Storage = Heap>`: `new`, `new_in`, `edit`, `as_bytes`, `from_bytes`,
  `from_bytes_with`, `byte_len`; `unsafe` `from_bytes_unchecked`.
- Functions `view::<T>(bytes)`, `view_with::<T>(bytes, scratch)`; `unsafe`
  `view_unchecked::<T>(bytes)`.
- `unsafe trait Storage`: `memory`, `memory_mut(words)`, where a region keeps its bytes.
  Storages: `Heap` (the default; `with_capacity(bytes)`), `Inline<N>` (`N` bytes inside the
  region).
- Handles: `ListEdit<'e, T, O = i32>` (`push`, `insert`, `remove`, `swap`, `truncate`,
  `retain`, `at`, `try_at`, `set`, `replace`, `update`, `iter_pin_mut`),
  `OptionEdit<'e, T, O = i32>` (`as_mut`, `insert`, `take`, `set`, `update`),
  `StrEdit<'e, O = i32>` (`set`, `replace`, `update`).
- `Old<'e, T>`: no methods; a value that left its place, which can only be written somewhere.
- `enum Error`: `Invalid(usize)`, `ScratchFull`, `StorageFull`; the error of every loader.
- `Pending`: no methods; an entry of the scratch that `from_bytes_with` and `view_with` take.

Trait impls: `Region`: `Deref`, `Clone` (`S: Default`), `Debug`, `Display`, `PartialEq`, `Eq`;
with `serde`, `Serialize` and `Deserialize` (`S: Default`). `Near`: `Deref`, `Debug`,
`Display`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`, and
`IntoIterator` for `&Near<[T]>`; `Debug`, `Display` and the comparisons forward to the value or
the target. Handles: `Deref`, and `DerefMut` for plain data. `Error`: `Debug`, `Display`,
`core::error::Error`, `Clone`, `Copy`, `PartialEq`, `Eq`. `Pending`: `Default`, `Clone`, `Copy`,
`Debug`.

Hidden: `nearest::__private::{Raw, Buf, Sink, Check, fits, make, Variant, Pointer}` (and `unpin`
with `nightly`), the plumbing of the macro; the hidden items of `Flat` (`PLAIN`, `ANY`, `REACH`,
`store`, `edit`, `check`; `UNPIN` with `nightly`) and `Emit` (`emit`, `plain`).
Nothing else is public; do not add public items.

Features:

- `alloc` (default) gives `Heap`, the default `S = Heap` (so `Region<T>` and `Region::new`; a
  `#[cfg]` on the generic parameter), and `from_bytes` and `view`, whose check takes memory from
  the heap. Without it nothing allocates: a region names its storage, and `as_bytes`,
  `from_bytes_with`, `view_with` and the unchecked loaders remain. The feature is additive: code
  that builds without it names every storage, so it builds with it.
- `nightly` (forwarded to `nearest-derive`; a nightly compiler for the whole build) makes
  `#[flat]` mark every type `#[pin_v2]` (see The pinned view). It adds no public item. It uses
  `specialization` (`nearest`) and `allow_internal_unstable` (`nearest-derive`) inside the two
  crates; users need no `#![feature]`, except `pin_ergonomics` to write patterns on pinned
  values. Without it nothing changes.
- `serde` (implies `alloc`; the optional `serde` dependency, without its `std`) makes a `Region`
  `Serialize` and `Deserialize` as its bytes (see Bytes). It adds no public item.

## Stored types

| Type | `Edit` |
|---|---|
| `u8`..`u64`, `i8`..`i64`, `f32`, `f64`, `bool`, `char`, `NonZero<_>` of those integers | itself |
| `[T; N]`, `T: Flat + Copy`, `N > 0` | itself |
| `Near<T, O>`, `T: Flat` | `Edit<'e, T>`: the handle is the target's |
| `Near<[T], O>`, `T: Flat` | `ListEdit<'e, T, O>` |
| `Near<str, O>` | `StrEdit<'e, O>` |
| `Option<Near<T, O>>`, `T: Flat` | `OptionEdit<'e, T, O>` (an `O`, `None` is 0) |
| `#[flat]` types | generated |

`Flat: 'static`: a stored type has no lifetime. Alignment at most `ALIGN` (that of a `u64`, see
Layout), non-zero size and no drop glue are checked at compile time by `fits`, a `const` block
that every use instantiates (building, editing, checking bytes, the unchecked loaders); the
alignment through `aligned::<T>()`, which a unit test of the library checks on each Miri target. Every use of a pointer also instantiates `fits_behind`: its target holds
no narrower pointer (see Layout). Plain data means `T: Copy`; a `Copy` type cannot contain a `Near`,
so it is safe to hand out as `&mut T`.

## Layout

- The root is at position 0. Byte 0 of the buffer is aligned to `ALIGN = align_of::<u64>()`, the
  alignment of the words of the memory (`flat.rs`): 8 on x86-64 and most 64-bit targets, 4 on
  `i686`, 2 on `msp430`. Not 8 everywhere: a word is 8 bytes (`WORD`), its alignment is the
  target's. **Positions and addresses differ**: a position is aligned for what is there (up to
  `GRAIN = 8`, for the elements of a list behind `i64`), its address only to
  `min(ALIGN, alignment of the position)`. Every typed access is at a position aligned for its
  type, and no stored type is more aligned than `ALIGN` (`fits`), so its address is aligned for
  it; offsets and lengths, whose positions may be aligned more than their addresses, are read
  and written unaligned (`read_unaligned`, `Sink::write`).
- `Near<T, O>`: `NonZero<O>`, target address minus own address. Below, `w` is `size_of::<O>()`.
- `Near<str, O>` points at an `O` length, `w`-aligned; the bytes follow.
- `Near<[T], O>` points at an `O` length; the elements follow at `+w`, an `O` capacity sits at
  `-w`. The elements are aligned to `max(w, align_of::<T>())` and the header `[cap][len]` takes
  the `2w` bytes before them (`Buf::alloc` places a block by its byte `inset = 2w`). With `i32`
  and `i64` the block itself starts so aligned; with `i16`, before 8-aligned elements, it starts
  4 bytes past a multiple of 8. On a target where a `u64` is 4-aligned, a block of `i64` lengths
  is still at a position that is a multiple of 8, at an address that may not be: its lengths are
  read unaligned, and its elements need no more than `ALIGN`.
- `#[flat]` adds `#[repr(C)]` to structs (unless already `C` or `transparent`) and `#[repr(u8)]`
  to enums: the layout does not depend on the compiler's field order. Variant field offsets
  come from hidden `#[repr(C)]` mirror structs `(u8, fields..)`.
- A region holds at most the limit of its root type, `limit::<T>()`: `T::REACH` (hidden `Flat`
  const: the `Offset::LIMIT`, `iN::MAX & !7`, of the narrowest pointer `T` holds by value, not
  through a pointer; `u64::MAX` without one) capped at `isize::MAX & !7`. So 32760 bytes with
  `i16` at the root, `i32::MAX & !7` with `i32`, `isize::MAX & !7` with `i64`. Limits are
  multiples of `GRAIN` (8), the largest alignment of a position, so that a position aligned up
  from below a limit stays below it; this is about positions, not addresses, and not about
  words. On a 16-bit target every limit is `isize::MAX & !7` = 32760. A pointer only
  leads to types whose `REACH` is at least its own `LIMIT` (`fits_behind`): widths grow, or stay,
  going down, so every distance and length in the region fits its `O`. `REACH` and `LIMIT` are
  `u64` so the rule is the same on every target (an `i64` pointer to an `i32` one is rejected on
  32-bit targets too, where both reach 2 GiB).

## Offsets

An `O: Offset` is how a region reads and writes the distances of its pointers and the lengths and
capacities of its lists and strings: only through `O::from_isize` and `O::to_isize`, and by
reading stored bytes as `O` (`near::read`, `read_unaligned`) or as `O::NonZero` (the field of a
`Near`). So the byte order of those integers is the offset's: `i16`, `i32`, `i64` are
native-endian; a user's offset can fix one for every target. The contract (the trait's
`# Safety`, keep both the same):

- every bit pattern of `size_of::<O>()` bytes is a valid `O`, which has no padding;
- `from_isize(n).to_isize() == n` for every `n: isize` with `|n| <= LIMIT` (on a target whose
  `isize` is narrower, every `n`);
- `from_isize(0)` is all zero bytes, and the only `O` whose `to_isize()` is 0: zeroed memory
  reads as 0 (null, an empty list), and a `None` is never read as a `Near`;
- `O::NonZero` is `O` without zero: same size and alignment, valid for every pattern but zero,
  `Option<O::NonZero>` of `O`'s size with `None` zero, `Into<O>` keeping the bytes;
- `from_isize` and `to_isize` depend only on their argument, never panic for those values (a
  panic between allocating and linking would leave a zero `Near`, I7), and do nothing else.
  `from_isize` is only called within `LIMIT`.

**Valid offsets.** A region holds only the `O`s that `from_isize` makes: `o` is valid when
`from_isize(o.to_isize())` has the very bytes of `o` (and `|o.to_isize()| <= LIMIT`). The checker
reads every offset and length through `near::read`, which refuses any other (`None`, so
`Error::Invalid` at its position), for every `Offset`, the user's included. So an integer that
does not fit `isize` (an `i64` of 2^32 + 8 on a 32-bit target, an `i32` of 2^16 on a 16-bit one),
which `to_isize` truncates, is invalid: in 0.7.1 it was read as the truncated value (2^32 + 8 as
8). `to_isize` is trusted only where it round-trips; elsewhere it may truncate (`as`), and should
not panic (the checker calls it on bytes from outside: a panic is not unsound, but a loader would
panic instead of failing). Truncation cannot be reached otherwise: a dereference, an edit and a
copy read only offsets the region wrote (`from_isize` of a distance or a length within `LIMIT`)
or that a loader checked; the unchecked loaders ask for bytes that the checker would accept.

`near::offset::<O>()`, a `const` block, checks the layout half at compile time (2, 4 or 8
bytes, so at most `GRAIN`; `NonZero` and `Option<NonZero>` of that size, `NonZero` of that alignment; `LIMIT` a
multiple of 8 that the integers of that size reach). Every function that reads or writes the
bytes of an `O` calls it: `fits_behind`, `aim`, `read_len`, `Sink::link`, `Sink::write_len`,
`Near::target`, `Near::target_mut`, `moved`, the `None` of `Option<Near>`, and `near::read`, which
the checker reads offsets with. `LIMIT` defaults to `2^(8 * size - 1) - 8`, the largest multiple
of 8 of a signed integer of that size (`iN::MAX & !7` for the three integers).

## Runtime checks

Only what cannot be known before:

- **Size.** Growing past the limit (2 GiB with `i32` at the root), or past what the storage
  gives, panics before anything changes (the capacity overflow of `Vec`); it is only looked at
  when the zeroed capacity is used up. `from_bytes*` return `Error::StorageFull` instead (and
  `Error::Invalid(0)` for bytes longer than the limit). The growth is `max(end, 2 * cap, 64)` capped at the limit (not `clamp`, which
  panicked on a limit below 64, possible with an `Offset` of the user's).
- **Indices.** `at`, `insert`, `remove`, `swap` panic out of bounds; `try_at` returns `None`.
- **Loaded bytes.** The loaders check bytes from outside (see Bytes). A region built or edited
  in the program is never checked; the `unsafe` unchecked loaders skip the check.

## Invariants

- **I1 — only in a region.** `Near` has no constructor and is not `Clone`/`Copy`: one exists
  only inside a region, and a `&Near` always has its target at the stored offset.
- **I2 — never `&mut` to a pointer.** No safe API yields `&mut T` for a `T` containing a `Near`.
  Such places are reached through handles, which are unsized: they cannot be swapped, moved out
  of or assigned through.
- **I3 — one handle chain.** A handle borrows its parent handle mutably, the root handle the
  region. So one handle is usable at a time, no `&T` into the buffer is live while a handle can
  allocate, and a `&mut` to plain data from a handle is unique. This makes growing the buffer
  during an edit sound.
- **I4 — a tree.** Every target has exactly one owning pointer. Builders make fresh targets,
  `&T` copies deeply, and the only moves are `Old` (affine, one use) and list operations.
- **I5 — append only.** Nothing in the buffer is freed, and bytes no longer reachable from the
  root are never written again; positions stay valid when the buffer grows. So an `Old` stays
  intact, with all it points to, for the whole edit.
- **I6 — copies come from elsewhere, moves do not allocate.** A deep copy (`store` with `None`)
  never reads the buffer it writes: a `&T` into a region cannot coexist with a handle of it, and
  an `Old` of the same region is moved, not copied. A move (`Some(d)`) only writes the
  destination.
- **I7 — panics leave a valid value.** A place is unchanged or holds the complete new value:
  list lengths are written after the element, `retain` keeps unvisited elements, `set`,
  `replace` and `update` restore the old value.
- **I8 — one provenance per edit.** `Region::edit` stores `&raw mut self.buf` in `Buf::me`, with
  the storage attached where the region is now. A handle reads `me` only through its own
  reference; every other access of the edit (handles, `Sink`, `Old`) goes through that
  pointer, each access a reborrow that ends with the statement. So no access invalidates a
  pointer kept by another part of the edit (Tree Borrows). Never keep a `&mut Buf`/`&Buf` across
  a call that touches the buffer; allocate before taking references into it.
- **I9 — an edit stays on one thread.** `Raw` holds a `PhantomData<*mut ()>` and `Old` a
  `*mut Buf`: neither is `Send` or `Sync`. An `Old` reads its region without borrowing a handle
  (when copied into another region), so a handle on another thread could grow the buffer under
  it. References from a handle (`&T`, `&mut` plain data) and the `Region` are `Send`/`Sync` as
  usual.
- **Buffer.** `len <= cap <= mem <= max` (the limit), also while unwinding; `[len, cap)` is
  zero.
- **Unsafe code.** `pub(crate)` never relaxes a safety precondition: a function that has one is
  an `unsafe fn`, whatever its visibility.

## Target assumptions

0.7.1 was released with two bugs that no check could find: "memory is `u64` words, so
8-aligned" (false where a `u64` is aligned to 4 or 2: `fits` let an 8-aligned type in, references
to it were misaligned, and `view` refused a region's own bytes) and `i64::to_isize` truncating on
32-bit targets (an offset of 2^32 + 8 loaded as 8). What let them through:

- **Nothing ran where the assumptions are false.** Miri ran on x86-64 only; `thumbv7em` and
  `msp430` were only built, and a build finds no undefined behavior.
- **An assumption about the target was written as a fact**, in prose, and never asserted.
- **No test used a type that stresses it**: an over-aligned type, an integer wider than `isize`.

The rule since: every assumption about the target that `unsafe` code relies on (the alignment of
a primitive, the width of `usize` and `isize`, the byte order) is a `const` assertion where it
is used or a constant derived from the target, never a literal justified by prose; the code runs
under Miri on a target where the assumption does not hold (`i686-unknown-linux-gnu`: 32 bits, a
`u64` aligned to 4; `s390x-unknown-linux-gnu`: big-endian), with the tests that run on x86-64;
and a new `unsafe` access comes with the test that stresses it. A 16-bit target cannot run under
Miri: assertions carry it. The assumptions, and what holds each:

| Assumption (where) | Held by |
|---|---|
| The alignment of a region is that of a `u64`: `ALIGN = align_of::<u64>()`, not 8 (`flat.rs`; `Region::deref`, `Raw::get`/`place`, `relocate`, `plain_edit`, `set`'s `Restore`, `insert`, `swap`, `retain`, `Old::emit`, `ListEdit::deref_mut`, `Near::deref`/`list`/`as_pin_mut`/`iter_pin_mut`, `view_unchecked`) | a constant of the target; `fits` refuses a type more aligned (`aligned`, unit test on each Miri target; trybuild for 16); `view` checks the address against `ALIGN`; Miri on `i686`; `a_region_is_aligned_as_its_words` in every storage |
| `ALIGN` is a power of two, at most `GRAIN` = 8, the largest alignment of a position (`limit`, `Buf::alloc`) | `const` assertion in `flat.rs` |
| A word is 8 bytes (`Buf::attach`, `room`, `Heap::with_capacity`, `Inline<N>`) | `WORD = size_of::<u64>()`, asserted 8 |
| The memory of `Inline<N>` is aligned as a `u64` (`Inline::memory*`) | `#[repr(C, align(8))]`, `const` assertion against `align_of::<u64>()` |
| An offset is 2, 4 or 8 bytes, so its block is at most `GRAIN`-aligned (`List::new`, `list`) | `near::offset` (`const`) |
| Offsets and lengths may sit at addresses less aligned than their positions (`i64` lengths on `i686`) | read and written unaligned only (`near::read`, `aim`, `read_len`, `Near::list`, `Sink::write`); Miri on `i686` |
| The integer of an offset fits `isize` (`Near::target`, `aim`, `read_len`, `Near::list`) | Valid offsets: `near::read` refuses an `O` that does not round-trip, for every `Offset`; `integers_wider_than_isize_are_invalid` (Miri on `i686`), `offsets_that_from_isize_does_not_make_are_invalid` (every target); 16 bits: the same code, not run |
| A region is at most `isize::MAX` bytes, so a distance fits `isize` (`distance`, `wrapping_offset`) | `limit` caps at `isize::MAX & !7`, from `isize::MAX` of the target |
| `usize` is at least 16 bits (`limit`, positions) | Rust supports no narrower `usize`; built for `msp430` |
| The byte order of `i16`, `i32`, `i64` and of stored numbers is the target's (`plain!`'s `from_ne_bytes`, the images) | no `unsafe` code depends on it; Miri on `s390x` runs the tests of the library, of offsets and of every storage big-endian; the tests write expected bytes with `to_ne_bytes` |

## Handles

```rust
#[repr(transparent)]
pub struct Raw<'e, T>(PhantomData<(&'e mut (), T, *mut ())>, [()]);   // __private
// &'a mut Raw<'e, T> == (*mut Buf, position)
```

A handle is `&'a mut` to an unsized `#[repr(transparent)]` wrapper of `Raw`: its address is the
region's `Buf` (which cannot move while the region is borrowed), its slice length the position
of the place. It holds no address into the buffer, so it survives reallocation; real references
(`&T` from `Deref`, `&mut` plain data) borrow the handle and so end before the next allocation
(I3). `ListEdit`, `OptionEdit`, `StrEdit` and generated handles wrap `Raw`. `Flat::edit(buf, at)`
(hidden) builds a place's handle; for `Near<T>` it reads the offset and returns the target's.

Lifetimes: `'e` is the edit, the borrow taken by
`Region::edit(&'e mut self) -> &'e mut Edit<'e, T>`, the same for every handle of it; `'a`
borrows the parent handle. Both are elided when a
handle is only passed down (`fn pass(e: &mut Edit<Expr>)`); a function returning a handle or an
`Old` names them:

```rust
fn first<'a, 'e>(e: &'a mut Edit<'e, Block>) -> &'a mut Edit<'e, Inst> { e.insts().at(0) }
fn pop<'e>(list: &mut Edit<'e, Near<[Inst]>>) -> Old<'e, Inst> { list.remove(0) }
```

Every handle derefs to its value (`[T]`, `str`, `Option<Near<T>>` for the three below), and
mutably for plain data (`T: Copy`):

- plain `T`: `&mut T`, ordinary Rust;
- `#[flat]` struct: generated; a method per field, `set`, `replace`, `update`, `pin`;
- `#[flat]` enum: generated; `as_mut`, `set`, `replace`, `update`, `pin`;
- `Near<[T]>`: `ListEdit<'e, T>`; list operations, `set`, `replace`, `update`, `iter_pin_mut`;
- `Option<Near<T>>`: `OptionEdit<'e, T>`; `as_mut() -> Option<&mut Edit<'e, T>>`,
  `insert(x) -> &mut Edit<'e, T>`, `take() -> Option<Old<'e, T>>`, `set`, `update`;
- `Near<str>`: `StrEdit<'e>`; `set`, `replace`, `update`.

The three take the width as a last parameter, `i32` by default (`ListEdit<'e, T, O>`).

## Generated code

For a type `X`, `#[flat]` emits, inside `const _: () = { .. }`, the `Flat` impl (`Edit<'e>`,
`PLAIN`, `ANY`, `REACH` (the least of the fields', left at the default without fields),
`store`, `edit`, `check`, each calling `fits`), the constructors, the handle
`__NearestEdit<'e, ..>` (`set`, `replace`, `update`, `pin`, a method per struct field, `as_mut`
for enums, `Deref`, `DerefMut` under `for<'a> X: Copy`), the mirror structs, and `X::project`;
next to `X`, with its visibility, `XPin<'a>` and, for an enum with fields, `XMut<'a, 'e>` (an
item of the user's with one of those names is a duplicate definition).

- Constructors: `X::new(fields..)` for a struct; one function per non-unit variant, in snake
  case. They return `impl Emit<X>` (see Building).
- Field methods are named after the field (`_0`, `_1`, .. for tuple fields), have its
  visibility and return `&mut Edit<'e, FieldType>`. Names the code uses itself get a trailing
  underscore: fields `set`, `replace`, `pin`, `update`; the constructor of a variant `Project`
  is `project_`.
- `#[flat(rename = "name")]` on a variant with fields names its constructor: not `project` and
  not another constructor's name; two variants whose constructors collide are an error until one
  is renamed. A unit variant, a field or the type take no `rename`.
- `#[flat(into)]` on a field of plain data makes its parameter `impl Into<FieldType>`, converted
  when written; rejected on a field written `Near<..>` or `Option<Near<..>>`.
- `#[flat]` reads and strips every `#[flat(..)]` on the type, variants and fields (`take_args`);
  unknown, repeated or misplaced arguments are errors.
- `XMut<'a, 'e>` has the variants of `X`, all tuple-like: a unit stays a unit; one field holds
  `&'a mut Edit<'e, Field>`; several fields hold one `&'a mut` payload handle with a method per
  field (two handles cannot be live at once, I3). The payload handle is a `Raw` of the mirror,
  named through `__private::Variant<I>`, with no `set` or `replace`: the variant cannot change
  under it. An enum with only unit variants gets no `as_mut`.
- Generic type parameters must be `Flat` (an `O: Offset` is). A field written `Near<X>` or
  `Near<X, O>` is a pointer; with a type parameter as `X` the constructor writes
  `<FieldType>::new(x)`. Rejected: lifetime parameters, unions,
  `repr(packed)`, enum `repr`s other than `u8`, explicit discriminants, empty enums, more than 256
  variants, and `#[pin_v2]` without `nightly`.

## Moving values

```rust
pub fn set(&mut self, value: impl Emit<T>)                    // `*place = value`
pub fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T>  // `mem::replace(place, value)`
pub fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E)
```

`OptionEdit` has `take` in place of `replace`. `replace` and `update` share `Raw::swap_in`.

- `set` keeps the old bytes of the place on the stack while the new value is written and puts
  them back if that unwinds (I7). It allocates nothing for the old value.
- `replace` moves the old value to `size_of::<T>()` fresh bytes (`detach`: pointers re-aimed,
  targets stay), writes the new value (a guard moves the old one back on unwind), and returns
  an `Old` designating the moved value.
- `Old<'e, T>` comes from `replace`, `update`, `ListEdit::remove`, `OptionEdit::take`. It is
  `Emit<T>`: written into its own region (same `me`) it moves (`size_of::<T>()` bytes, pointers
  re-aimed, whatever it points to); into another region, or `Region::new`, it is deep-copied;
  dropped, it is garbage until compaction. Not `Copy`/`Clone`: used at most once (I4). It holds
  `(me, position)`, so it survives growth. Its lifetime `'e` keeps the region mutably borrowed:
  the region cannot be read, replaced, dropped or edited by a second `Region::edit` meanwhile,
  and an `Old` never reaches a later edit. It cannot be read: a reference would not survive the
  next allocation, and nothing ties an `Old` to the handle that allocates; read through the
  handle before moving.
- `update` moves the old value aside, calls `f` and writes its result (`e.update(Expr::neg)`);
  if `f` or the writing unwinds, the old value moves back. Its `Old` lives for `'a`, the borrow
  of the handle, not `'e`: an `Old` that leaves the closure keeps every handle of the region
  borrowed, so it can never be written into its region after the old value moved back; into
  another region it is a copy.

Every rearrangement is then a sequence of statements: wrap (`update`, or `replace` with a
placeholder then `set` the wrapper), hoist (`replace` the child, `set` the parent), swap,
rotate, and moving elements or lists between distant places.

## Lists

`push` and `insert` return the new element's handle. `push` writes into spare capacity, then
bumps the length; a full list moves to a new block of twice the capacity (at least 4) and its
pointer is re-aimed. `insert`, `remove`, `swap`, `retain` move elements inside the block (each
move re-aims pointers; plain elements are moved with `memmove`). `remove` first moves the element
to fresh bytes and returns it as an `Old`. `truncate` only writes the length. `at(i)` panics out
of bounds, `try_at(i)` returns `None`. `retain` keeps the unvisited elements if its closure
panics.

## Building

`Emit<T>` is implemented for:

- any `T`: `T` by value, `&T` (deep copy), `Old<'_, T>` (move; for a list or string place, `T`
  is `Near<[U]>` or `Near<str>`);
- `Near<T>` and `Option<Near<T>>`: `Near::new(e)`;
- `Near<[T]>`: `[E; N]`, tuples up to 12 and `()`, `&[T]`, `Near::from_iter(iter)`;
- `Near<str>`: `&str`;
- `#[flat]` types: the generated constructors.

Constructors are lazy: nothing is written until the builder reaches a region. Types without
pointers can also be built with a plain literal. Pointers are transparent where unambiguous: a
constructor parameter for a field written `Near<X>` takes a builder of `X` and allocates the
target (`Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2)))`); if `X` is a slice or `str` it takes
a list or string builder. The same for `Near<X, O>`: builders are generic over the width, which
the place gives. The macro recognizes the field by the name `Near`; the hidden trait
`Pointer` (`<FieldType as Pointer>::Arg`) then decides from the type, so `X` may be an alias; a
type parameter as `X` is used as written, so the argument infers it. A field whose whole type is
an alias of a pointer takes a builder of the pointer, `Near::new(x)`. Likewise the handle of a
`Near<X>` place is the `X`'s handle, so `set` on it and `OptionEdit::insert` take a builder of
`X`. An `Option<Near<X>>` parameter takes `None` or `Near::new(x)`.

A list built from an iterator reserves its size hint's lower bound; if the iterator yields
another count, the list keeps its real capacity (or grows) and the region is no longer compact.

## The pinned view

One handle is usable at a time (I3). Where an edit only reads, writes plain data or assigns
values without pointers, several places are borrowed together through `Pin`:

- `pin(&mut self) -> Pin<&mut X>` (`Raw::pin`, on every generated handle) borrows the handle, so
  nothing grows the buffer while a pin, which holds an address, lives. Nothing is pushed, built
  or moved through a pin.
- `X::project(self: Pin<&mut X>) -> XPin<'_>`: a struct with the fields of `X`, or an enum with
  its variants, each field `Pin<&'a mut Field>`. `Near::as_pin_mut` pins a target,
  `Near::<[T]>::iter_pin_mut` (also on `ListEdit`) the elements; `core` has
  `Option::as_pin_mut`.
- `Near` holds a `PhantomPinned`. A stored type is `Unpin` exactly when its fields are: without
  `nightly` the macro emits `impl Unpin for X where __NearestFields<'a, ..>: Unpin` (all field
  types), so a hand-written impl conflicts. A `Pin<&mut T>` thus gives `&mut T` only for data
  without pointers (I2). A `Region` is `Unpin` as its storage is (it owns its buffer).
- Without `nightly`, `#[flat]` rejects a `#[pin_v2]` it sees, since the type gets that impl; a
  `#[pin_v2]` added by another macro after `#[flat]` meets the impl, so soundness does not
  depend on the macro seeing it.
- With `nightly`, `#[flat]` adds `#[pin_v2]` (through `#[allow_internal_unstable(pin_ergonomics)]`
  on the macro; one written by hand is kept, not repeated), so in a crate with `pin_ergonomics`
  patterns project: `match e.pin() { Expr::Add(a, b) => .. }`. A `#[pin_v2]` type cannot have
  the bounded impl: with `pin_ergonomics` the compiler rejects every `Unpin` impl of it (the
  macro's too), so the macro writes none and the compiler keeps the rule there. Without
  `pin_ergonomics` the compiler accepts a hand-written impl, so the library checks:
  `nearest/src/unpin.rs` (built only with the feature; `specialization`) defines
  `__private::unpin` and `pinned`. The hidden `Flat::UNPIN` ("every field is `Unpin`": `true`
  for plain types, `false` for pointers without looking at the target, so recursive types do
  not cycle; `__private::unpin::<Self>(true & F1::UNPIN & ..)` for generated types, `&` so every
  field is evaluated) fails the build if a type is `Unpin` while a field is not. `Raw::pin`,
  `Near::as_pin_mut` and `iter_pin_mut` evaluate it (`unpin::pinned::<T>()`), so a hand-written
  impl on a type holding a `Near` fails where it, or a type holding it by value, is pinned. Like
  `fits`, it fails a build that generates code, not `cargo check`. An impl on data without
  pointers is accepted, and harmless. `tests/pin_v2.rs` is built only with the feature; its
  `compile_fail` doc tests are on `unpin.rs`, including the compiler's rule.
- `Pin::set` assigns a value held by value, which has no pointer (I1): the place then holds a
  complete value, and what it pointed to is garbage, as after `set`.

## Storage

```rust
pub unsafe trait Storage {
  fn memory(&self) -> &[MaybeUninit<u64>];
  /// The memory, made at least `words` long first if the storage can grow.
  fn memory_mut(&mut self, words: usize) -> &mut [MaybeUninit<u64>];
}
```

`Region<T, S: Storage>` keeps its bytes in the memory of `S`, which is words, so aligned as a
`u64` is (`ALIGN`; 8 bytes only where a `u64` is 8-aligned).
`Heap` (`alloc`), the default, is a `Vec` that grows as asked; `Heap::with_capacity(bytes)`
makes room up front (nothing written). `Inline<N>` is `N` bytes (rounded down to words) inside
the `Region` value, and does not grow. Any other memory (an arena, a borrowed buffer, a mapped
file) is a storage of the user's. `new_in(storage, value)` builds in any storage; `new` is
`new_in(Heap::default(), ..)`. `from_bytes*`, the unchecked loaders and `clone` make a storage
with `Default`; a storage without one loads with `Region::new_in(storage, view::<T>(bytes)?)`.
Everything else is the same for every storage.

The memory may hold anything, initialized or not, when given and where growing adds to it: the
region zeroes what it uses. The contract (also the trait's `# Safety`, keep both the same):

- **Its own.** Nothing but the region reads or writes the memory while the storage is in a
  region, and the memory stays valid as long as the storage lives.
- **Stable.** `memory`, and `memory_mut` asked for no more words than the memory has, return
  the same memory (address and length) and change none of it. It changes only through
  `memory_mut` asked for more, or, if inside the storage value, when that value moves; its bytes
  are those last written.
- **Growth.** `memory_mut` asked for more may make it longer: it keeps the bytes (wherever it
  puts them), never shortens the memory, and what it adds may hold anything. If it panics, the
  memory is as before, at the same address.

The region checks the rest: it asks for no more than its limit, uses no more of any memory, and
panics if it gets less than it asked for (`from_bytes*`: `Error::StorageFull`). A storage that
does not grow returns its memory as it is, whatever it is asked for.

Handles never see the storage; they reach one concrete `Buf` (`flat.rs`): `ptr` (byte 0), `len`,
`cap` (the zeroed prefix: `[len, cap)` zero, `[cap, mem)` anything), `mem` (bytes of memory, at
most `max`), `max` (the limit, given at creation: `limit::<T>()`, or the source's for `clone`),
`store` and `regrow` (a function pointer made for `S`), `me` (I8) and the compact
mark. `Buf` is `Send`/`Sync` by an explicit impl; a region is `Send`, `Sync`, `Unpin` as its
storage is (the provided ones are all three).

- Allocating below `cap` is the hot path and calls nothing. Past it, the cold `Buf::grow` zeroes
  the next part, doubling (`max(end, 2 * cap)`, clamped to `[64, max]`, at most `mem`), and
  calls `regrow` first only if `mem` is too short (asking for the same, so the heap doubles).
  Zeroing costs the bytes used, never the memory's size: a large arena or mapping costs nothing
  up front. Loading copies the image and zeroes nothing.
- A storage may move with its region, so its address is not kept between uses. Through `&self`
  (`Deref`, `as_bytes`, `clone`) the region calls `memory` (`Region::base`), never the `Buf`'s
  pointer. Through `&mut self` (`new_in`, `edit`, loading, `clone`'s new region) it first
  attaches (`Region::attach`): `memory_mut(0)` and its address and size written into the
  `Buf`, after the region's last move (attach, then never return the region by value before
  using it). `attach` keeps `cap`, a length. `regrow` attaches with `memory_mut(words)`, and
  again with 0 if that unwinds, since the storage was borrowed anew.
- A region cannot move while a handle, an `Old` or a reference into it lives: all borrow it. A
  pin into it ends with the handle's borrow, as before the heap grows.

Soundness for any storage keeping the contract: the region reads only below `len`, and every
byte there was written by it (a value, or the zeros of `cap`) and kept since (Stable, Growth),
so it is initialized whatever the memory held; pointers into the memory are taken after the
region's last move and after every growth (Stable, Growth), and nothing else touches it (Its
own); a growth that unwinds leaves the memory where the restoring guards of I7 write (Growth).
Memory is words, so aligned to `ALIGN`, and no stored type is more aligned (`fits`), so every
typed access at a position aligned for its type is at an address aligned for it; no more than
the limit is used, so every offset fits. `Inline<N>` is `#[repr(C, align(8))]`, at least `ALIGN`
(asserted).

A storage that cannot hold the region panics where the heap would grow: `new_in`,
`from_bytes_unchecked`, `clone`, edits. `from_bytes*` load bytes from outside and return an
error: they check, ask the storage for room (`Buf::load`), and copy only if they got it.

## Bytes

The compact image of a value is what a deep copy writes: the root at 0, then every target in the
order the copy reaches it (depth first, fields in declaration order, list elements in order),
each at the next position aligned for it; lists have `cap == len`; nothing follows the last
target. Builders write the same image.

```rust
pub fn as_bytes(&self) -> Option<&[u8]>                                      // Region
pub fn from_bytes(bytes: &[u8]) -> Result<Region<T, S>, Error>               // S: Default; alloc
pub fn from_bytes_with(bytes: &[u8], scratch: &mut [Pending]) -> Result<Region<T, S>, Error>
pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Region<T, S>              // panics if full
pub fn view<T: Flat>(bytes: &[u8]) -> Result<&T, Error>                       // crate root; alloc
pub fn view_with<'b, T: Flat>(bytes: &'b [u8], scratch: &mut [Pending]) -> Result<&'b T, Error>
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T
```

- `as_bytes` borrows the buffer if the region is compact, else `None`; it never allocates. A
  region is compact when built (`new`, `new_in`), loaded or cloned from a compact one; `edit`
  clears the mark, and so does a list built from an iterator whose count differed from its size
  hint (`List::finish` with `cap != len`, or a builder list that had to grow). A deep copy never
  clears it: it writes every list with `cap == len` and grows none, so `Region::new(&*r)` is
  always compact and is how any region is saved.
- `from_bytes` checks the bytes (any alignment) and copies them into a new region in any storage
  with `Default`. A valid image the storage cannot hold is `Error::StorageFull`, not a panic.
- `view` checks a slice aligned as a region is, to `ALIGN` (a mapped file, a region's own buffer,
  on every target), and reads it in place;
  a `&T` needs nothing else, since `Near` resolves relative to itself. It is a function, not a
  method: it makes no region and has no storage.
- The `_with` forms check with the caller's scratch; `Error::ScratchFull` says it ran out before
  the check finished, which says nothing about the bytes.
- The unchecked forms are `unsafe`: the checked form would accept the bytes, in this build (for
  `view_unchecked`, at an address aligned to `ALIGN`, and the compact image of a valid `T`; for
  the loaders, the image at any address). That is what `as_bytes` returned for the same `T` in a
  build with the same layout, and what a check of the caller's that accepts no more (a crate
  that checks images itself) accepted. Loading is then one `memcpy`, and viewing reads nothing.
  `from_bytes_unchecked` panics if the storage is full.
- `Error` (exhaustive): `Invalid(at)`, where the bytes stop being the image of a valid value (0
  for a slice not aligned to `ALIGN` in `view*`, or bytes shorter than a `T` or longer than its
  limit);
  `ScratchFull`; `StorageFull`.

Checking (`bytes.rs`) accepts exactly the compact images of valid values. It walks the image
once, in the order the copy wrote it, and requires every pointer to aim at the very position
where the copy put its target: that one rule gives bounds, alignment, no sharing, no cycles and
no gaps. Values are checked on the way: `bool`, `char`, `NonZero`, enum tags, UTF-8,
`cap == len`; a negative length is as out of bounds as one too long, and an
offset or a length that `from_isize` does not make (see Valid offsets) is invalid. Padding is
not looked at.

- It is safe code that never panics (checked arithmetic, `get` instead of indexing):
  `Flat::check(&mut Check, at) -> Result<(), Error>` (hidden, generated). The only unsafe step
  of a loader is the cast of checked bytes to `&T`.
- It does not recurse. Pointers and rests of lists wait on a stack (`Stack`), a pointer of a list
  element before the remaining elements; time is linear in the bytes. The stack grows with the
  nesting of the data (no fixed size suffices; nothing points back): a `Vec` for `from_bytes`
  and `view`, the caller's `&mut [Pending]` for the `_with` forms (`Check::push` returns
  `Error::ScratchFull` when it is full, and the walk ends).
- `bytes.len() / 4` entries always suffice (`/ 2` with `i16` pointers): each waits for its own
  bytes of the image, at least as many as a pointer has (a pointer inside a value already
  checked, or the start of the next element of a list, whose type holds a pointer); checked
  values are disjoint, each pointer is queued once, a list has at most one rest waiting. So an
  image fitting `Inline<N>` needs at most `N / 4`; most data needs about as many as the pointers
  waiting along the deepest path. An entry (`Pending(usize, usize, Step)`, as positions may pass
  4 GiB) is 24 bytes on 64-bit targets, 12 on 32-bit.
- Elements with `Flat::ANY` (no pointers, every bit pattern valid: lists and arrays of numbers)
  are not walked, so checking costs pointers, text and values that can be invalid, not bulk
  numbers: a large mapped table is ready at once.
- The bytes hold the layout of the target, its values in the byte order of their types, and
  distances and lengths as their `Offset` writes them (native-endian for `i16`, `i32`, `i64`);
  no header, no version in the bytes. A build with the same layout reads back what `as_bytes`
  gave.
- With `serde`, a region serializes as its bytes (`serialize_bytes` of `as_bytes`, or of a
  compact copy, `Region::new(&*r)`, if it was edited), and deserializes through `from_bytes`
  (any storage with `Default`); its `Error` becomes `de::Error::custom`. The visitor takes bytes
  borrowed or owned, or a sequence of `u8` (as JSON writes them), whose size hint it trusts up
  to 64 KiB. It needs `alloc`: `from_bytes` checks with the heap, and a sequence is collected
  into a `Vec`.

## Internals

Files of `src/`:

- `lib.rs`: crate docs, exports, `Edit<'e, T>`, `__private`.
- `flat.rs`: `ALIGN`, `WORD`, `GRAIN` (see Target assumptions), `limit`, `aligned`, `fits`,
  `fits_behind`, `Buf` (`attach`, `alloc`, `grow`, `room`, `load`), `Sink` (`link`,
  `write_len`), `aim`, `Flat` and `Emit` with their impls, moves (`relocate`,
  `detach`, `move_elems`), the list block `List<O>`, `make`.
- `edit.rs`: `Raw` (the core of every handle: `set`, `replace`, `update`, `swap_in`, `pin`),
  `Old`, `ListEdit`, `OptionEdit`, `StrEdit`.
- `bytes.rs`: `Error`, `Pending`, `Stack`, the checker `Check` (`get`, `offset`) and its steps (`target`, `list`,
  `text`, `items`), `check`, `view`, `view_with`, `view_unchecked`.
- `near.rs`: `Near<T, O>`: `Deref` for `T`, `[T]` and `str`, builders, trait impls; `Offset`,
  its three impls (`offsets!`), `offset` (its layout, checked) and `read` (valid offsets only).
- `region.rs`: `Region<T, S>`: `attach`, `base`, `copy_of`, constructors, `edit`, `as_bytes`,
  `from_bytes*`, trait impls, `serde` impls.
- `storage.rs`: `Storage` and its contract, `Heap`, `Inline<N>`.
- `unpin.rs`: `nightly` only, `unpin` and `pinned` (see The pinned view).

Mechanics:

- `Sink`: a `*mut Buf` (`alloc`, `write`, `link`, `loose` clears the compact mark), never a
  `&mut Buf`.
- `Flat::store(&self, sink, at, shift: Option<isize>)`: `None` deep-copies targets; `Some(d)` keeps
  targets, re-aims offsets for a move of `d` bytes and never dereferences a `Near` (moves call it on
  stack copies). `PLAIN` types (no pointers, no padding) are copied as bytes.
- Self-relative pointers are resolved from `&self`: valid under Tree Borrows, not Stacked
  Borrows. No pointer is made from an integer. Miri runs with `-Zmiri-tree-borrows
  -Zmiri-strict-provenance`.
- Tests: `tests/common/integration.rs` runs in three storages (`integration`: heap, `inline`,
  `user_storage`); `width` runs tests generic over the width with `i16`, `i32` and `i64`, the
  limit (reached cheaply with `i16`), mixed widths, and `a_region_past_2_gib` (`#[ignore]`d;
  CI runs it in release), offsets that do not round-trip; `no_alloc` also
  without `alloc`; `serde` with that feature; `compile_fail` (trybuild) holds the compile-time
  guarantees. Miri runs `--lib` and the five tests above (not `serde`, `compile_fail`,
  `pin_v2` but with `nightly`) on x86-64, and on `i686` and `s390x` what the target can change
  (`miri-targets` in `ci.yml`): the tests assume nothing of the target (sizes and alignments are
  written from `size_of`, `align_of` and `to_ne_bytes`).

## What is not here, and why

- No sibling handles that can grow the buffer. Several live handles, references into the
  buffer and a buffer that grows in place cannot hold together: growing may move the buffer and
  dangle a reference from another handle. Only a buffer that never moves would allow it, which a
  growable heap allocation is not. So handles grow, one at a time, and the pinned view borrows
  several places without growing. A value moves between places through an `Old`.
- No sibling swap in the pinned view: two pins do not say whether they are in the same region,
  and a pointer cannot be re-aimed across regions.
- No pin back to a handle: a pin has an address, not a position.
- No single-lifetime handles: a moved value must outlive the borrow of the handle it came
  through, or it could not be written elsewhere; that longer borrow is `'e`. Both lifetimes are
  elided wherever a handle is only passed down.
- No reading of an `Old`, and no `Old` across two calls of `Region::edit`.
- No handle or `Old` sent to another thread (I9): send the `&mut Region` and edit there, or the
  `&mut [T]` of plain data.
- No in-place reuse of freed space: compact with `Region::new(&*region)`.
- No storage type on handles: it is a parameter of `Region` alone; handles reach one `Buf`.
- No adopting of memory that already holds an image: a region starts empty in any storage. Load
  with `from_bytes`, or `new_in` from a `view`; a mapped image is read in place with `view`.
- No spilling of `Inline` to the heap: a full inline region panics. Spilling would make the place
  of the bytes a question every read asks at run time.
- No `to_bytes`: `as_bytes` borrows the image of a compact region and `Region::new(&*r)`
  compacts any other; a method fusing the two needs a `Cow` and exists only with `alloc`.
- No views on `Region`: viewing builds no region and needs no storage. As methods of
  `Region<T>` (the heap) they only served inference, and kept `Heap` alive without `alloc`.
- No `from_bytes` or `view` without `alloc`, and no fixed bound on checking: a check that never
  rejects a valid image needs memory that grows with the data. Without a heap the caller gives
  it (`_with` forms); a missing method is better than a bound that rejects valid images.
- No `#[non_exhaustive]` on `Error`: its three cases are every way a loader fails (the bytes,
  the scratch, the storage), and an exhaustive match lets callers handle each one, checked by
  the compiler. A new case would be a breaking release anyway.
- Copies recurse once per level of nesting (`Region::new(&*region)`, so saving an edited
  region), like `Clone` for a `Box`. Checking bytes does not.
- No `isize` or `usize` width: the bytes would change with the target, for no reach that a
  fixed width lacks (`i64` already reaches `isize::MAX`). No unsigned width: after a move a
  pointer may aim backward. No cargo feature for the width: two crates of one build would
  disagree.
- No narrower pointer below a wider one: the limit comes from the root alone, as a const that
  followed pointers would cycle on recursive types; so widths may only grow going down.
- No fallback to `i32` in inference: Rust does not default type parameters in expressions, so a
  region whose root is a bare pointer names its type (`Region::<Near<[u32]>>::new(..)`).
- No test-only limit: `i16` pointers reach the real one at 32 KiB.
- No 8-aligned words on every target (yet). The memory of a storage is `[MaybeUninit<u64>]`,
  aligned as a `u64`; making a region 8-aligned everywhere would let the same types (an
  `align(8)` type) be stored on every target and make `view` want the same address everywhere.
  It needs memory of a type aligned to 8 (`#[repr(align(8))] struct Word(MaybeUninit<u64>)`, or
  `[MaybeUninit<u8>]` with a stated alignment) in `Storage`, which changes the trait (breaking:
  every user storage), or a region that panics on a storage whose memory is not 8-aligned
  (breaking at run time where `i686` storages work today). A candidate for 0.8; 0.7 keeps the
  alignment of the target.
