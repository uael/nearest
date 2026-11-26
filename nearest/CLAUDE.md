# nearest — specification

This file is the specification of the crate: code, tests and documentation follow it. Commands,
toolchain, releases and style are in the root `CLAUDE.md`. The examples are the crate docs
(`src/lib.rs`, also in `README.md`), tested as doc tests, and `examples/` (`expr`, `passes`,
`ir`).

## Idea

A `Region<T>` keeps a `T` and everything it points to in one buffer. A pointer is a `Near<T, O>`:
the distance from its own address to its pointee, an `O` (`i32`, 4 bytes, by default; `i16` or
`i64`). So copying the buffer copies the value (`Clone` is a `memcpy`), reading is plain `&T`
(`Near` derefs like `Box`), and the bytes mean the same at any address (a region is saved and loaded
as bytes). Stable Rust, `#![no_std]`, no dependency but the macro and `puns` (and `serde`,
optional). Aliasing, lifetimes, moves and the tree shape are enforced by types; see Runtime checks
for the rest.

`puns` (type punning proven at compile time, same owner) knows the layout and the valid values of
every stored type: `#[flat]` derives `puns::Pun`, and so does `Near`, with an eager aim of `puns` at
its pointee (see Pointers and `puns`). So the loaders check bytes with `puns` alone, pointees
included (`puns::from_head`), and the numbers of a fixed byte order of `puns::num` make the
bytes of a region the same on every machine. `puns` is a git dependency, pinned to a commit: its
aims are not released.

## Public surface

- `#[flat]`: attribute for structs and enums (see Generated code); every `#[flat]` type is a
  `puns::Pun` type.
- `nearest::puns`: the crate `puns`, re-exported, so that `puns::num` and the derive are those of
  the version that `nearest` uses.
- `trait Flat: puns::Pun + Sized + 'static`: `type Edit<'e>: ?Sized + Handle<'e, Self::Value>`,
  `type Value: Flat` (what its handle writes: `Self`, the pointee's for `Near<T>`), `type Ref<'a,
  'e>` (what getters give: `&'a mut Edit<'e, Self>`, a `Plain` for plain data); the rest is
  hidden. `trait Emit<T>`: all hidden; "can
  be written as a `T`". `type Edit<'e, T> = <T as Flat>::Edit<'e>`: the handle of a place
  holding a `T`.
- `Near<T, O: Offset = i32>`: builders `new`, `from_iter`; pinned view `as_pin_mut`,
  `iter_pin_mut`.
- `unsafe trait Offset`: the integer of a `Near`, implemented for `i16`, `i32`, `i64`, for
  `puns::num::{i16le, i32le, i64le, i16be, i32be, i64be}`, and open to the user's: `type
  NonZero` (a `puns::Pun` type), `LIMIT` (defaulted from the size), `from_isize`, `to_isize`;
  its `# Safety` is the contract (see Offsets), and the layouts of `Self` and `NonZero` say the
  byte order of its integer. Its
  supertraits `Flat + Copy + Debug + Eq + Ord + Hash` let a type generic over the width be
  `#[flat]` (which bounds every type parameter by `Flat`) and derive the usual traits.
- `Region<T, A: Allocator = Global>`: `new`, `from_bytes` (`Global`); `new_in`, `from_bytes_in`,
  `edit`, `reserve`, `as_bytes`, `byte_len`; `unsafe` `from_bytes_unchecked_in`.
- Functions `const fn view::<T>(bytes)`; `unsafe` `view_unchecked::<T>(bytes)`.
- `Fixed<'a>`: `new(&'a mut [MaybeUninit<u8>])`, an `Allocator` over a borrowed buffer.
- `trait Handle<'e, T: Flat>`: `set`, `replace`, `update`, implemented by every handle (each also
  has them as its own methods): the bound of `Flat::Edit`, for code generic over the stored type.
- Handles: `ListEdit<'e, T, O = i32>` (`push`, `insert`, `remove`, `swap`, `truncate`,
  `retain`, `at`, `try_at`, `set`, `replace`, `update`, `iter_pin_mut`),
  `OptionEdit<'e, T, O = i32>` (`as_mut`, `insert`, `take`, `set`, `replace`, `update`),
  `StrEdit<'e, O = i32>` (`set`, `replace`, `update`), `PlainEdit<'e, T>` (`set`, `replace`,
  `update`; the handle of plain data), and `Plain<'a, 'e, T>` (the same by value: `edit` gives
  the `&mut PlainEdit`).
- `Old<'e, T>`: no methods; a value that left its place, which can only be written somewhere.
- `enum Error`: `Invalid(puns::Error)`, `Alloc`; the error of every loader.

Trait impls: `Region`: `Deref`, `Drop` (gives the block back), `Clone` (`A: Clone`), `Debug`,
`Display`, `PartialEq`, `Eq`; with `serde`, `Serialize` and `Deserialize` (`A: Default`).
`Fixed`: `Allocator`, `Send`. `Near`: `Deref`, `Debug`,
`Display`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`, and
`IntoIterator` for `&Near<[T]>`; `Debug`, `Display` and the comparisons forward to the value or
the pointee. Handles: `Deref`, and `DerefMut` for plain data. `Error`: `Debug`, `Display`,
`core::error::Error`, `Clone`, `Copy`, `PartialEq`, `Eq`.

Hidden: `nearest::__private::{Raw, Buf, Sink, fits, make, Variant, Pointer}`, the plumbing of the
macro; `Near::AIMS`, which the derive of `puns` writes, and which names the aim of a `Near` for
`puns::aim_mut`; the hidden items of `Flat` (`PLAIN`, `REACH`, `edit`, `store`, `shift`;
`PLAIN` is not given: it is `Pun::LAYOUT.is_plain()` and no eager aim,
`!Pun::LAYOUT.holds_eager_aims()`) and `Emit` (`emit`, `plain`). The trait `aim::Pointee<O>`,
which maps the `T` of a `Near<T, O>` to the pointee of its aim, is public in a private module: a
bound of `Near` and its `Deref` that a user cannot name or implement.
Nothing else is public; do not add public items.

Features:

- `alloc` (default) gives the default `A = Global` (so `Region<T>`, `Region::new` and
  `Region::from_bytes`; a `#[cfg]` on the generic parameter). Without it nothing allocates but
  the allocator a region names, such as `Fixed`, and every loader remains, since a check takes no
  memory. The feature is additive: code that builds without it names every allocator, so it
  builds with it.
- `serde` (implies `alloc`; the optional `serde` dependency, without its `std`) makes a `Region`
  `Serialize` and `Deserialize` as its bytes (see Bytes). It adds no public item.

## Stored types

| Type | `Edit` |
|---|---|
| `u8`..`u64`, `i8`..`i64`, `f32`, `f64`, `bool`, `char`, `NonZero<_>` of those integers | `PlainEdit<'e, Self>` |
| the numbers of `puns::num`: `u16le`..`u128be`, `i16le`..`i128be`, `f32le`..`f64be` (byte arrays, every value valid), `u8c<V>`, `u8in<MIN, MAX>`, `u16lec<V>`..`u128bec<V>` (checked as `puns::try_pun` checks them) | `PlainEdit<'e, Self>` |
| `[T; N]`, `T: Flat + Copy`, `N > 0` | `PlainEdit<'e, Self>` |
| `Near<T, O>`, `T: Flat` | `Edit<'e, T>`: the handle is the pointee's |
| `Near<[T], O>`, `T: Flat` | `ListEdit<'e, T, O>` |
| `Near<str, O>` | `StrEdit<'e, O>` |
| `Option<Near<T, O>>`, `T: Flat` | `OptionEdit<'e, T, O>` (an `O`, `None` is 0) |
| `#[flat]` types | generated |

`Flat: 'static`: a stored type has no lifetime. Alignment at most `ALIGN` (8, see Layout), non-zero
size and no drop glue are checked at compile time by `fits`, a `const` block that every use
instantiates (building, editing, checking bytes, the unchecked loaders); the alignment through
`aligned::<T>()`, which a unit test of the library checks on each Miri target, and
`a_region_is_aligned_to_8` with a type aligned to 8. Every use of a pointer also instantiates
`fits_behind`: its pointee holds no narrower pointer (see Layout). Plain data means `T: Copy`; a
`Copy` type cannot contain a `Near`, so it is safe to hand out as `&mut T`.

## Layout

- The root is at position 0. Byte 0 of the buffer is aligned to `ALIGN = 8` on every target
  (`flat.rs`): the block is allocated with that alignment. It is also the largest alignment of a
  position (a stored type, or the elements of a list behind `i64`), so positions and addresses
  agree modulo 8: a position aligned for a type is an address aligned for it, and a check of
  `puns` by address is a check by position. Offsets and lengths are read and written unaligned
  (`read_unaligned`, `Sink::write`), as they always were.
- `Near<T, O>`: `NonZero<O>`, pointee address minus own address. Below, `w` is `size_of::<O>()`.
- `Near<str, O>` points at an `O` length, `w`-aligned; the bytes follow.
- `Near<[T], O>` points at an `O` length; the elements follow at `+w`, an `O` capacity sits at
  `-w`. The elements are aligned to `max(w, align_of::<T>())` and the header `[cap][len]` takes
  the `2w` bytes before them (`Buf::alloc` places a block by its byte `inset = 2w`). With `i32`
  and `i64` the block itself starts so aligned; with `i16`, before 8-aligned elements, it starts
  4 bytes past a multiple of 8.
- `#[flat]` adds `#[repr(C)]` to structs (unless already `C` or `transparent`) and `#[repr(u8)]`
  to enums: the layout does not depend on the compiler's field order. Variant field offsets
  come from hidden `#[repr(C)]` mirror structs `(u8, fields..)`.
- A region holds at most the limit of its root type, `limit::<T>()`: `T::REACH` (hidden `Flat`
  const: the `Offset::LIMIT`, `iN::MAX & !7`, of the narrowest pointer `T` holds by value, not
  through a pointer; `u64::MAX` without one) capped at `isize::MAX & !7`. So 32760 bytes with
  `i16` at the root, `i32::MAX & !7` with `i32`, `isize::MAX & !7` with `i64`. Limits are
  multiples of `ALIGN` (8), the largest alignment of a position, so that a position aligned up
  from below a limit stays below it; this is about positions, not addresses, and not about
  words. On a 16-bit target every limit is `isize::MAX & !7` = 32760. A pointer only
  leads to types whose `REACH` is at least its own `LIMIT` (`fits_behind`): widths grow, or stay,
  going down, so every distance and length in the region fits its `O`. `REACH` and `LIMIT` are
  `u64` so the rule is the same on every target (an `i64` pointer to an `i32` one is rejected on
  32-bit targets too, where both reach 2 GiB).

## Offsets

An `O: Offset` is how a region reads and writes the distances of its pointers and the lengths and
capacities of its lists and strings: only through `O::from_isize` and `O::to_isize`, and by
reading stored bytes as `O` (`read_unaligned`) or as `O::NonZero` (the field of a
`Near`). Its bytes are a two's complement integer in the byte order that its layout says, a
`puns` scalar (`Spec`) or a `#[repr(transparent)]` struct around one: `i16`, `i32`, `i64` are
native-endian; `i16le`, `i32le`, `i64le`, `i16be`, `i32be`, `i64be` of `puns::num` (byte arrays,
of alignment 1, implemented in `near.rs` with `puns::num::NonZero<O>` as `NonZero`, whose
`Scalar` says the byte order of `O`: packing gives alignment 1 and keeps the niche) and a
user's offset fix one for every target. `puns` reads the offset of a `Near` as the layout of
`O::NonZero` says, and a length or a capacity as the layout of `O` says, so the layout is the
byte order: there is no other statement of it to keep in step. The image of a type with one
offset loads as the type with another of the same width and byte order (`view`), which finds the
same pointees only because both read the same integer from the same bytes: hence the first item.
The contract (the trait's `# Safety`, keep both the same):

- two's complement: the bytes of every `O` are a signed integer, `size_of::<O>()` bytes, which
  the layout of `O` is, and `to_isize` is that integer converted as `as` converts (truncated if
  wider than `isize`); so every bit pattern is a valid `O`, which has no padding;
- round trip: `from_isize(n)` is the `O` whose integer is `n`, for every `n: isize` with
  `|n| <= LIMIT`; so `from_isize(0)` is all zero bytes, and the only valid offset whose integer
  is 0: zeroed memory reads as 0 (null, an empty list), and a `None` is never read as a `Near`;
- `O::NonZero` is `O` without zero: same size and alignment, valid for every pattern but zero,
  its layout the integer of `O` without zero (the same byte order and sign),
  `Option<O::NonZero>` of `O`'s size with `None` zero, `Into<O>` keeping the bytes;
- `from_isize` and `to_isize` depend only on their argument, never panic (a panic between
  allocating and linking would leave a zero `Near`, I7), and do nothing else. `from_isize` is
  only called within `LIMIT`.

0.7 asked less of `to_isize` (a round trip within `LIMIT`, anything elsewhere): an offset whose
`to_isize` read past `LIMIT` as something else was one. It no longer is, since an image
written with one offset loads with another. `contract` (`tests/width.rs`) checks the items on values for
every offset of the tests, on every Miri target.

**Native and fixed byte orders.** The default offset is `i32`, native-endian, like `i16`, `i64`
and the numbers of Rust: a region of them is the bytes of the machine that wrote it, and loads as
the same value only on a machine of the same byte order. Bytes that cross machines (files, the
network) use the offsets and the numbers of `puns::num`, of a fixed byte order; the crate docs and
the README say so where widths are explained. They read with the same instructions where the
order is the machine's (`eval` of a tree: 25 instructions with `i32` and with `i32le` on x86-64)
and with a byte swap elsewhere; they are aligned to 1, so the layout of a type that holds them
may be tighter (`Expr::Add`'s pointers at 1 and 5, not 4 and 8). The default stays native: a
fixed order is a choice of the type, made where its bytes leave the machine.

**Valid offsets.** A region holds only the `O`s that `from_isize` makes: `o` is valid when
`from_isize(o.to_isize())` has the very bytes of `o` (and `|o.to_isize()| <= LIMIT`). By the
contract, that is the two's complement integer of the width and the byte order of the offset,
which `puns` reads as the layout of the offset says, as an integer of any width: a distance that
leads out of the bytes, or a length that does not fit in them, is refused, and no distance or
length past `LIMIT` is in bounds (the bytes are at most the limit of the root, at most the `LIMIT`
of every pointer, `fits_behind`). So an integer that does not fit `isize` (an `i64` of 2^32 + 8
on a 32-bit target), which `to_isize` truncates, is invalid: in 0.7.1 it was read as the
truncated value (2^32 + 8 as 8); so is one past `LIMIT`.
Truncation cannot be reached otherwise: a dereference, an edit and a copy read only offsets the
region wrote (`from_isize` of a distance or a length within `LIMIT`) or that a loader checked;
the unchecked loaders ask for bytes that the checker would accept.

`near::offset::<O>()`, a `const` block, checks the layout half at compile time (2, 4 or 8 bytes, so
at most `ALIGN`; `NonZero` and `Option<NonZero>` of that size, `NonZero` of that alignment; `LIMIT`
a multiple of 8 that the integers of that size reach). Every function that reads or writes the bytes
of an `O` calls it: `fits_behind`, `aim`, `read_len`, `Sink::link`, `Sink::write_len`, `moved`, the
`None` of `Option<Near>`, and the loaders, through the `store` of the type they load
(`bytes::check`), which reaches `fits_behind` for every pointer of the type, at every depth. `LIMIT`
defaults to `2^(8 * size - 1) - 8`, the largest multiple of 8 of a signed integer of that size
(`iN::MAX & !7` for the three integers).

## Pointers and `puns`

`Near<T, O>` derives `puns::Pun` (`near.rs`): a `repr(transparent)` struct of its offset, an
`InPlace<O::NonZero>` (so zero is not a `Near`, and `puns` makes no `Near`), with a named eager
aim, `#[pun(pointer(mut pointee: <T as Pointee<O>>::Pointee))]`: the pointee starts
at the address of the pointer plus the offset, read as the layout of `O::NonZero` says.
`aim::Pointee<O>` maps the `T` of the three pointers to their pointees, `puns::Pointee` types:

| Pointer | Pointee | Points at |
|---|---|---|
| `Near<T, O>` | `T` | the `T` |
| `Near<[T], O>` | `puns::Thin<[T], ListLen<O>>`: the capacity, the length, the elements | the length |
| `Near<str, O>` | `puns::Thin<str, O>`: the length, the UTF-8 bytes | the length |

A `Thin<T, M>` stores the metadata of a pointer to `T` before it: for a list or a text, an `M`
stored right before the elements (its last integer is the length, read as its layout says; a
negative one is refused), the pointer aimed at the length, and the elements aligned to their
alignment and to the size of the length; nothing for a sized `T`. That is the format of 0.8 (see
Layout), byte for byte. `ListLen<O>` is a `#[repr(C)]` struct of the capacity and the length, whose
rule `len <= cap` an image keeps, with a capacity within `LIMIT` of 0, and whose length is within
`LIMIT` of 0 (`#[pun(valid(..))]`), so that a check refuses a length past it where the length is.
An image may hold spare room, as a region does; a region does not trust the capacity of a list
of a loaded image (see Bytes). `aim::Pointee<O>` exists because the format puts a
capacity before the length of a list and none before the length of a text: a `Thin<T, M>` of every
`T` with one `M` cannot say both. The `Pun` of a `Near` does not evaluate `fits_behind`, which a
read of `puns` does not need: the loaders instantiate `Flat::store`, a function pointer to it (see
Bytes), as every other way into a region does. A constant of `Flat` cannot hold the assertion: the
types that a recursive type reaches lead back to it, a cycle of constants that Rust refuses (E0391).

`puns` reads the pointee: the aim is named, so the derive writes `Near::pointee`, safe, which
makes the `&T`, the `&[T]` or the `&str`, and `Near::AIMS.pointee`, with which `Near::pointee_mut`,
`unsafe`, calls `puns::aim_mut` (the derive of `puns` writes no method that writes a pointee: it
would be `unsafe`, and call a function that a crate path names). `Deref` calls
`pointee` with no `unsafe` block: a `&Near` comes from a check of `puns` or from the region, whose
`Deref` takes the obligation that `puns::InPlace` states for unsafe code that gives a reference to
a value that holds one (a region holds each pointee at its offset, and the reference has the
provenance of the buffer under Tree Borrows). The pinned view (`as_pin_mut`, `iter_pin_mut`)
calls `pointee_mut`, one `unsafe` block each, whose `// SAFETY:` is the contract of "Reading a
pointee" in `aim.rs` (a `Near` owns its pointee alone).
`Deref` requires only what the read needs, `T: Pun`; `#[flat]` bounds every type parameter of the
type by `Pun`, so a generic type that derives `Debug` or `PartialEq` over a `Near<T>` reads it
(`Pair`, `Lst`, `Tree` in `tests/common/integration.rs`, `Boxed` in `tests/layouts.rs`).

`Option<Near<..>>` is the `Option` of the pointer, which `puns` lays out as the `Option` of its
`NonZero`: of a native offset, the niche that `core` documents; of an offset of `puns::num`, the
one of the `NonZero<iN>` that the `Scalar` of `puns::num::NonZero` says its bytes are
(`Spec::stored_as`), which `puns` assumes and checks when compiling, without its `unstable`
feature.

What `puns` does with a value that holds a pointer:

- It reads it as plain data: a pointer is initialized bytes (`as_bytes`, `head_ref`, a pun to a
  type without pointers: `pointers_are_read_as_bytes`, `tests/layouts.rs`).
- It checks it with its pointee, and the pointees of the pointee, in the bytes that it checks:
  `from_bytes`, `from_head`, `from_tail` make a reference to a value that holds a pointer only
  once every pointee is a valid value within the bytes, aligned for it, so a `&Near` from `puns`
  is one whose pointee `Deref` may read
  (`puns_reads_pointers_from_bytes_once_it_checked_their_pointees`, `tests/layouts.rs`). Those
  follow pointers at any depth, with a stack that the types bound (`puns::AIM_PARTS`).
- It requires the pointees in the order of a copy, each after everything before it, with gaps
  allowed: so no pointee is shared, overlaps another value, or is reached by a cycle, and the
  bytes form a tree, which the edits of a region rely on (I4). The reads of `puns` give `&T` only,
  never a region or a handle.
- It makes no value that holds one (`pun`, `try_pun`, `zeroed` do not compile; a `Near` is not
  `Copy` either), and no `&mut` to or from one (`pun_mut`, `from_bytes_mut`:
  `tests/compile-fail/puns_mut_bytes_as_pointer.rs`).

So with `puns` in the build, I1 to I9 hold: `puns` makes no pointer (I1, I4), is never handed a
`&mut` to a value that holds one (no safe API gives it, I2), makes no handle and takes nothing
mutably from a region (I3), writes nothing to a region and never reaches its `Buf` (I5 to I8,
Buffer), and makes a `&T` that holds a pointer only when its pointees are valid (I9).

Whether a type holds a pointer is whether its layout holds an eager aim
(`Layout::holds_eager_aims`). `PLAIN` (copied as bytes) is a layout without padding
(`Layout::is_plain`) and without an eager aim.

**The alignment of the elements of a list.** The position of the first element of a list is
aligned to `max(align_of::<T>(), w)` (`List::new`), and so is its address, since the block is
aligned to `ALIGN`, at least that; `T` is never more aligned than `ALIGN` (`fits`). Every way a list block comes to exist is `List::new`
(builders, a push past the capacity, a deep copy, so compaction) or a loader's check, both at a
position aligned to `max(align_of::<T>(), w)`; `insert`, `remove`, `swap`, `retain`, `truncate`
move elements inside a block, and a move of a list (an `Old`, `set`) re-aims its pointer, not the
block; `clone` and loading copy the image to another buffer aligned to `ALIGN`, at the same
positions. `the_elements_of_a_list_are_aligned_as_declared` (`tests/layouts.rs`, Miri on x86-64,
`i686` and `s390x`) builds and edits lists of elements of alignment 1 every such way, and asserts
the positions of their elements.

**Rules.** `#[pun(valid(..))]` on a field and `#[pun(assert(..))]` on a struct (and the other
rules of the derive) are requirements of the type, in `puns`'s terms (its crate docs),
never relied upon for soundness, since safe code can write any value to a field. `nearest` treats
them so: builders, edits (`&mut` to plain data, `set`, the pinned view) and moves do not check
them, so a region can hold a value that breaks a rule of its type, as a struct of safe code can.
Checking at construction would not make them a guarantee: an edit can break one right after,
through a `&mut u8`. Copies and moves do not need them: `store` and `shift` read valid tags and
the fields of the active variant, not rules (`a_value_that_breaks_a_rule_is_copied_and_moved`,
under Miri). The loaders check them (see Bytes): a value that breaks a rule is saved by
`as_bytes` and refused when loaded (`a_value_that_breaks_a_rule_is_saved_not_loaded`), where the
field is for a rule of a field, where the value is for a rule of the type.

## Runtime checks

Only what cannot be known before:

- **Size.** Growing past the limit (2 GiB with `i32` at the root), or past what the allocator
  gives, panics before anything changes (the capacity overflow of `Vec`); it is only looked at
  when the zeroed capacity is used up. `from_bytes_in` returns `Error::Alloc` instead (and
  `Error::Invalid(puns::Error::Size)` for bytes shorter than a `T` or longer than the limit). The
  growth is `max(end, 2 * cap, 64)` capped at the limit (not `clamp`, which panicked on a limit
  below 64, possible with an `Offset` of the user's).
- **Indices.** `at`, `insert`, `remove`, `swap` panic out of bounds; `try_at` returns `None`.
- **Loaded bytes.** The loaders check bytes from outside (see Bytes). A region built or edited
  in the program is never checked; the `unsafe` unchecked loaders skip the check.

## Invariants

- **I1 — only in a region.** `Near` has no constructor and is not `Clone`/`Copy`: one exists
  only inside a region, and a `&Near` always has its pointee at the stored offset.
- **I2 — never `&mut` to a pointer.** No safe API yields `&mut T` for a `T` containing a `Near`.
  Such places are reached through handles, which are unsized: they cannot be swapped, moved out
  of or assigned through.
- **I3 — one handle chain.** A handle borrows its parent handle mutably, the root handle the
  region. So one handle is usable at a time, no `&T` into the buffer is live while a handle can
  allocate, and a `&mut` to plain data from a handle is unique. This makes growing the buffer
  during an edit sound.
- **I4 — a tree.** Every pointee has exactly one owning pointer. Builders make fresh pointees,
  `&T` copies deeply, and the only moves are `Old` (affine, one use) and list operations.
- **I5 — append only.** Nothing in the buffer is freed, and bytes no longer reachable from the
  root are never written again; positions stay valid when the buffer grows. So an `Old` stays
  intact, with all it points to, for the whole edit.
- **I6 — copies come from elsewhere, moves do not allocate.** A deep copy (`write_copy`) never
  reads the buffer it writes: a `&T` into a region cannot coexist with a handle of it, and an
  `Old` of the same region is moved, not copied. A move (`move_elems`) only writes the
  destination, and allocates nothing.
- **I7 — panics leave a valid value.** A place is unchanged or holds the complete new value:
  list lengths are written after the element, `retain` keeps unvisited elements, `set`,
  `replace` and `update` restore the old value.
- **I8 — one provenance per edit.** `Region::edit` stores `&raw mut self.buf` in `Buf::me`, and
  the address of the allocator in `Buf::alloc`, where the region is now (`Region::bind`). A handle reads `me` only through its own
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
  zero; `[0, len)` is initialized while `init` holds, which only a `&mut` to a type with padding
  clears (see Bytes).
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
| A region is aligned to 8, on every target, and so is every position that a type needs: `ALIGN = 8` (`flat.rs`; `Region::deref`, `Raw::get`/`place`, `relocate`, `plain_edit`, `set`'s `Restore`, `insert`, `swap`, `retain`, `Old::emit`, `ListEdit::deref_mut`, `Near::deref`/`list`/`as_pin_mut`/`iter_pin_mut`, `view_unchecked`) | the block is allocated with `Layout` of alignment `ALIGN` (`flat::block`), `Fixed` aligns its block as asked; `fits` refuses a type more aligned (`aligned`, unit test on each Miri target; trybuild for 16); `a_region_is_aligned_to_8` with a type aligned to 8, in both allocators, Miri on `i686` |
| An offset is 2, 4 or 8 bytes, so its block needs at most `ALIGN` (`List::new`, `list`) | `near::offset` (`const`) |
| The integer of an offset fits `isize` (`aim`, `read_len`, and the address that `Near::pointee` computes) | Valid offsets: the loaders refuse an `O` that leads out of the bytes or counts past them, so past `LIMIT`, for every `Offset`; `integers_wider_than_isize_are_invalid` (Miri on `i686`), `offsets_that_from_isize_does_not_make_are_invalid` (every target); 16 bits: the same code, not run |
| A region is at most `isize::MAX` bytes, so a distance fits `isize` (`distance`, `wrapping_offset`) | `limit` caps at `isize::MAX & !7`, from `isize::MAX` of the target |
| `usize` is at least 16 bits (`limit`, positions) | Rust supports no narrower `usize`; built for `msp430` (and `puns`, whose assumptions of `usize` are in its crate docs) |
| The bytes of an offset are a two's complement integer in the byte order of its layout, which `to_isize` reads (`puns` reads it so) | the contract of `Offset` (`unsafe`); the layouts of `i16`, `i32`, `i64`, from the target; `contract` (`tests/width.rs`) for every offset of the tests, Miri on `i686` and `s390x` |
| The first element of a list is at a position that is a multiple of `max(align_of::<T>(), w)` (the slice that `Near::pointee` makes, aligned for `T`) | `List::new`, and the alignment of the aim of a `Thin`, which `puns` checks at load; `tests/layouts.rs` under Miri on `i686` |
| An image written with the offsets and the numbers of a fixed byte order is the same on every machine (`examples/ir.rs`, `ir.image`) | `tests/ir.rs` under Miri on `s390x` compares it byte for byte |
| The byte order of `i16`, `i32`, `i64` and of stored numbers is the target's (`plain!`'s `from_ne_bytes`, the images) | no `unsafe` code depends on it; Miri on `s390x` runs the tests of the library, of offsets and of both allocators big-endian; the tests write expected bytes with `to_ne_bytes` |

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
(I3). `ListEdit`, `OptionEdit`, `StrEdit`, `PlainEdit` and generated handles wrap `Raw`.
`Flat::edit(buf, at)` (hidden) builds a place's handle; for `Near<T>` it reads the offset and
returns the pointee's. Every handle implements `Handle<'e, Value>` (`set`, `replace`, `update`,
the same functions as its own methods: `set` through the trait compiles to the machine code of
`set` on the handle), the bound of `Flat::Edit`.

Getters (field methods, `ListEdit::{at, try_at, push, insert}`, `OptionEdit::{as_mut, insert}`,
the variants of `XMut`, `Region::edit`) give `T::Ref<'a, 'e>`, through the hidden
`Flat::reference`: `&'a mut Edit<'e, T>`, but for plain data a `Plain<'a, 'e, T>`, the handle by
value, whose `Deref`/`DerefMut` are the value, so that one star reaches it as with a `&mut`
(`*e.count() += 1`, `*list.at(0) = 5`; a field `bump` is `inc byte [..]`, as with `&mut u8`).
Bound by a pattern, it is mutable as a value is (`Some(mut n) => *n += 1`). A `#[flat]` type
that is `Copy` keeps `&mut` to its generated handle, so two stars reach its value
(`**e.term() = ..`): the macro cannot know that a type is `Copy` (an impl can be written apart
from its derive), and a by-value `Ref` derefs to one target, the value, which would hide the
handle's field methods and `as_mut`.

Lifetimes: `'e` is the edit, the borrow taken by
`Region::edit(&'e mut self) -> T::Ref<'e, 'e>`, the same for every handle of it; `'a`
borrows the parent handle. Both are elided when a
handle is only passed down (`fn pass(e: &mut Edit<Expr>)`); a function returning a handle or an
`Old` names them:

```rust
fn first<'a, 'e>(e: &'a mut Edit<'e, Block>) -> &'a mut Edit<'e, Inst> { e.insts().at(0) }
fn count<'a, 'e>(e: &'a mut Edit<'e, Block>) -> Plain<'a, 'e, u32> { e.count() }
fn pop<'e>(list: &mut Edit<'e, Near<[Inst]>>) -> Old<'e, Inst> { list.remove(0) }
```

Every handle derefs to its value (`[T]`, `str`, `Option<Near<T>>` for the three below), and
mutably for data without pointers (`T: Copy`):

- plain `T` (numbers, `bool`, `char`, `NonZero`, `puns::num`, arrays of them): `PlainEdit<'e, T>`,
  given as `Plain`; `set`, `replace`, `update`;
- `#[flat]` struct: generated; a method per field, `set`, `replace`, `update`, `pin`;
- `#[flat]` enum: generated; `as_mut`, `set`, `replace`, `update`, `pin`;
- `Near<[T]>`: `ListEdit<'e, T>`; list operations, `set`, `replace`, `update`, `iter_pin_mut`;
- `Option<Near<T>>`: `OptionEdit<'e, T>`; `as_mut() -> Option<T::Ref<'_, 'e>>`,
  `insert(x) -> T::Ref<'_, 'e>`, `take() -> Option<Old<'e, T>>`, `set`, `replace`, `update`;
- `Near<str>`: `StrEdit<'e>`; `set`, `replace`, `update`.

The three take the width as a last parameter, `i32` by default (`ListEdit<'e, T, O>`).

## Generated code

On the type itself, `#[flat]` adds `::nearest::puns::Pun` to the bounds of each type parameter, and
puts `#[derive(::nearest::puns::Pun)]` and `#[pun(crate = ::nearest::puns, bound())]` (without
`bound` for a type without type parameters, which `puns` refuses on an enum without fields) first
(so that the user's `#[pun(..)]` attributes, which it leaves as they are, come after the derive
that reads them). This is the derive of `puns`, not code of this crate's: the layout of the type is
what `puns` finds (`#[repr(C)]` for a struct, `#[repr(u8)]` for an enum, as `#[flat]` makes them),
and every field type must be `Pun` (every `Flat` type is). The bounds of `Pun` are those of the
type, every type parameter `Pun` among them, which a layout needs, and none that the derive would
find from the fields (`bound()`): through pointers, the bounds of types that hold each other would
go round in a circle (a user's `#[pun(bound(..))]` adds to them). The `Pun` bound is on the type,
not only on the impl of `Pun`, so that the impls that the derives of `core` write (`Debug`, which
reads the pointees) have it, and the user does not write it: written twice, clippy reports it. Not
`Flat`: the `Pun` of a type that reads its pointees in such an impl must hold under the bounds that
the impl has. The impls of `#[flat]` add `::nearest::Flat` to the bounds of each type parameter.
A derive rather than an
implementation written by `#[flat]`: the items of `Pun` are `puns`'s own, which it may change,
and the derive is how it writes them for a struct or an enum. Consequences for a user: a type that
derives `Pun` itself conflicts, and the field types of a `pub` type must be as visible as it is
(the derive names them in public associated types). Deriving `Pun` exposes no more than `#[flat]`
did: a loader already made any value whose fields are valid from bytes.

For a type `X`, `#[flat]` emits, inside `const _: () = { .. }`, the `Flat` impl (`Edit<'e>`,
`REACH` (the least of the fields', left at the default without fields), `edit`, calling `fits`,
and `store` and `shift`: see Mechanics), the constructors, the handle
`__NearestEdit<'e, ..>` (`set`, `replace`, `update`, `pin`, `Handle`, a method per struct field
returning the field's `Ref`, `as_mut`
for enums, `Deref`, `DerefMut` under `for<'a> X: Copy`), the mirror structs, and `X::project`;
next to `X`, with its visibility, `XPin<'a>`, for a struct with fields `XInit<..>`, and for an
enum with fields `XMut<'a, 'e>` (an item of the user's with one of those names is a duplicate
definition).

- Constructors: `X::new(fields..)` for a struct; one function per non-unit variant, in snake
  case. They return `impl Emit<X>` (see Building).
- `XInit<__F0, ..>`: for a struct with fields, a struct (tuple-like for a tuple struct) with the
  fields of `X`, each of a type parameter bounded as the parameter of `new` is (a builder of the
  field, of the pointee of a pointer, `impl Into` with `#[flat(into)]`), all with the visibility of
  `X`, as `new` takes every field. It is `Emit<X>`, writing each field at its offset as `new`
  does: the same machine code (an `XInit` and `X::new` of the same fields compile to identical
  functions), and the same bytes (`init_structs_name_the_fields`). The names are the point: a
  struct of seven fields (`Function` of `examples/ir.rs`) reads as a literal.
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
  `Field::Ref<'a, 'e>` (a `Plain` for plain data); several fields hold one `&'a mut` payload
  handle with a method per field, returning its `Ref` (two handles cannot be live at once, I3). The payload handle is a `Raw` of the mirror,
  named through `__private::Variant<I>`, with no `set` or `replace`: the variant cannot change
  under it. An enum with only unit variants gets no `as_mut`.
- Generic type parameters must be `Flat` (an `O: Offset` is). A field written `Near<X>` or
  `Near<X, O>` is a pointer; with a type parameter as `X` the constructor writes
  `<FieldType>::new(x)`. Rejected: lifetime parameters, unions,
  `repr(packed)`, enum `repr`s other than `u8`, explicit discriminants, empty enums, more than 256
  variants, and `#[pin_v2]`.

## Moving values

```rust
pub fn set(&mut self, value: impl Emit<T>)                    // `*place = value`
pub fn replace(&mut self, value: impl Emit<T>) -> Old<'e, T>  // `mem::replace(place, value)`
pub fn update<'a, E: Emit<T>>(&'a mut self, f: impl FnOnce(Old<'a, T>) -> E)
```

`OptionEdit` also has `take`, which returns the pointee instead of the pointer (`replace` returns
the old `Option<Near<T>>`). `replace` and `update` share `Raw::swap_in`.

- `set` keeps the old bytes of the place on the stack while the new value is written and puts
  them back if that unwinds (I7). It allocates nothing for the old value.
- `replace` moves the old value to `size_of::<T>()` fresh bytes (`detach`: pointers re-aimed,
  pointees stay), writes the new value (a guard moves the old one back on unwind), and returns
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
- `#[flat]` types: the generated constructors, and `XInit { .. }` for a struct.

Constructors are lazy: nothing is written until the builder reaches a region. Types without
pointers can also be built with a plain literal. Pointers are transparent where unambiguous: a
constructor parameter for a field written `Near<X>` takes a builder of `X` and allocates the
pointee (`Expr::add(Expr::Lit(1), Expr::neg(Expr::Lit(2)))`); if `X` is a slice or `str` it takes
a list or string builder. The same for `Near<X, O>`: builders are generic over the width, which
the place gives. The macro recognizes the field by the name `Near`; the hidden trait
`Pointer` (`<FieldType as Pointer>::Arg`) then decides from the type, so `X` may be an alias; a
type parameter as `X` is used as written, so the argument infers it. A field whose whole type is
an alias of a pointer takes a builder of the pointer, `Near::new(x)`. Likewise the handle of a
`Near<X>` place is the `X`'s handle, so `set` on it and `OptionEdit::insert` take a builder of
`X`. An `Option<Near<X>>` parameter takes `None` or `Near::new(x)`.

A list built from an iterator reserves its size hint's lower bound; if the iterator yields
another count, the list keeps its real capacity (or grows) and the region holds it as it is.

## The pinned view

One handle is usable at a time (I3). Where an edit only reads, writes plain data or assigns
values without pointers, several places are borrowed together through `Pin`:

- `pin(&mut self) -> Pin<&mut X>` (`Raw::pin`, on every generated handle) borrows the handle, so
  nothing grows the buffer while a pin, which holds an address, lives. Nothing is pushed, built
  or moved through a pin.
- `X::project(self: Pin<&mut X>) -> XPin<'_>`: a struct with the fields of `X`, or an enum with
  its variants, each field `Pin<&'a mut Field>`. `Near::as_pin_mut` pins a pointee,
  `Near::<[T]>::iter_pin_mut` (also on `ListEdit`) the elements; `core` has
  `Option::as_pin_mut`.
- `Near` holds a `PhantomPinned`. A stored type is `Unpin` exactly when its fields are: the macro
  emits `impl Unpin for X where __NearestFields<'a, ..>: Unpin` (all field types), so a hand-written
  impl conflicts. A `Pin<&mut T>` thus gives `&mut T` only for data without pointers (I2). A
  `Region` is `Unpin` as its allocator is (it owns its block, which does not move with it).
- `#[flat]` rejects a `#[pin_v2]` it sees, since the type gets that impl, which a `#[pin_v2]`
  type cannot have; a `#[pin_v2]` added by another macro after `#[flat]` meets the impl, and the
  compiler rejects the two, so soundness does not depend on the macro seeing it. Patterns that
  project a pinned value (`pin_ergonomics`, nightly) wait for that feature to be stable: until
  then `project` does what they would. 0.7 had a feature `nightly` for them, which took
  `specialization` to check by hand what the compiler does for a `#[pin_v2]` type; 0.8 has none.
- `Pin::set` assigns a value held by value, which has no pointer (I1): the place then holds a
  complete value, and what it pointed to is garbage, as after `set`.

## Allocators

`Region<T, A: Allocator>` keeps its bytes in one block of `A` (`core::alloc::Allocator`, stable
since 1.100), of the layout `block(size)`: aligned to `ALIGN` (8). `Global` (`alloc`) is the
default; `reserve(bytes)` makes room up front (nothing written). `Fixed<'a>` borrows a buffer of
`MaybeUninit<u8>` and gives one block at a time, from the first address of the buffer aligned as
asked to its end, so a region in it grows to the whole buffer; `grow` and `shrink` keep the block
where it is. Bytes inside the allocator value cannot be the allocator: the contract of
`Allocator` requires a block to stay valid when its allocator moves, so `Fixed` borrows. Any other
allocator works the same. `new_in(value, alloc)` builds in any allocator, the value first as in
`Box::new_in`; `new` is `new_in(value, Global)`, `from_bytes` is `from_bytes_in(bytes, Global)`.
Everything else is the same for every allocator.

The block may hold anything, initialized or not, when given and where growing adds to it: the
region zeroes what it uses. A block that the allocator gives larger than asked is used up to the
limit only.

Handles never see the allocator; they reach one concrete `Buf` (`flat.rs`): `ptr` (byte 0),
`len`, `cap` (the zeroed prefix: `[len, cap)` zero, `[cap, mem)` anything), `mem` (bytes of the
block, at most `max`; 0 when there is no block), `max` (the limit, given at creation:
`limit::<T>()`, or the source's for `clone`), `loaded` (the end of the loaded image), `alloc` and
`regrow` (the address of the allocator, and a function pointer made for `A` that grows the block
through it), `me` (I8) and `init`. `Buf` is `Send`/`Sync` by an explicit impl; a region is `Send`,
`Sync`, `Unpin` as its allocator is (`Global` is all three, `Fixed` is `Send` and `Unpin`).

- Allocating below `cap` is the hot path and calls nothing. Past it, the cold `Buf::grow` zeroes
  the next part, doubling (`max(end, 2 * cap)`, clamped to `[64, max]`, at most `mem`), and
  calls `regrow` first only if `mem` is too short: `Allocator::allocate` for the first block,
  `Allocator::grow` after, which may move the block. Zeroing costs the bytes used, never the
  block's size. Loading copies the image and zeroes nothing.
- A block does not move when its region moves, so reads through `&self` (`Deref`, `as_bytes`,
  `clone`) use `ptr`. The allocator moves with the region, so every use through `&mut self` that
  may grow the block (`new_in`, `edit`, `reserve`, loading) first writes its address into the
  `Buf` (`Region::bind`), after the region's last move.
- A region cannot move while a handle, an `Old` or a reference into it lives: all borrow it.
- `Drop` gives the block back with `deallocate`, with the layout `block(mem)`, which fits it.

Soundness for any allocator keeping its contract: the region reads only below `len`, and every
byte there was written by it (a value, or the zeros of `cap`) and kept since, so it is
initialized but for the padding that an assignment through a `&mut` wrote (`init`); the block is
the region's alone until it is deallocated, and stays valid while the allocator lives, wherever
it moves; a growth that the allocator refuses leaves the block as it was, and panics. The block is
aligned to `ALIGN`, and no stored type is more aligned (`fits`), so every typed access at a
position aligned for its type is at an address aligned for it; no more than the limit is used, so
every offset fits.

A block that the allocator does not give panics where the heap would grow: `new_in`,
`from_bytes_unchecked_in`, `clone`, `reserve`, edits. `from_bytes_in` loads bytes from outside and
returns an error: it asks the allocator for a block (`Buf::load`), copies, and checks the copy.

## Bytes

The compact image of a value is what a deep copy writes: the root at 0, then every pointee in the
order the copy reaches it (depth first, fields in declaration order, list elements in order),
each at the next position aligned for it; lists have `cap == len`; nothing follows the last
pointee. Builders write the same image, but for a list built from an iterator whose size hint was
wrong, which keeps spare room, or moved as it grew.

```rust
pub fn as_bytes(&self) -> Option<&[u8]>                                          // Region
pub fn from_bytes_in(bytes: &[u8], alloc: A) -> Result<Region<T, A>, Error>
pub unsafe fn from_bytes_unchecked_in(bytes: &[u8], alloc: A) -> Region<T, A>    // panics if refused
pub const fn view<T: Flat>(bytes: &[u8]) -> Result<&T, Error>                     // crate root
pub const unsafe fn view_unchecked<T: Flat>(bytes: &[u8]) -> &T
```

- `as_bytes` borrows the bytes in use: the value, what it reaches, and what edits left behind.
  It never allocates. It is `None` once `init` is cleared: an assignment through a `&mut T`
  writes the padding of `T` as uninitialized bytes, so handing out a `&mut` that reaches a type
  that is not `PLAIN` clears it (`Raw::handed`): `DerefMut` of `PlainEdit`, `Plain`, a generated
  handle and `ListEdit`, and the pinned view (`Raw::pin`, which `iter_pin_mut` of a `ListEdit`
  calls). The region's own writers write initialized bytes only and keep the bytes of padding.
  Edits may leave pointees out of the order of a copy (a value set after the pointees of later
  fields, a list that moved after its later siblings), so the bytes of an edited region may not
  load as a region: before saving, `view` tells, and `Region::new(&*r)` is the copy that does, in
  the order of a copy, with only what is reachable.
- `view` is `puns::from_head` (`const`, no memory): the slice aligned for `T`, every value and
  pointee valid, the pointees in the order of a copy, bytes that no pointer reaches allowed. A
  `&T` needs nothing else, since `Near` resolves relative to itself. It is a function, not a
  method: it makes no region and has no allocator.
- `from_bytes_in` copies the bytes (any alignment) into a new block of `alloc`, then checks the
  copy as `view` does, so the region holds the image as it is, and edits append after it. Bytes
  shorter than a `T` or longer than the limit are `Invalid(Size)` before anything is allocated; a
  block the allocator does not give is `Error::Alloc`, not a panic.
- The capacity of a list whose block is in the loaded image is not trusted (`List::read`, from
  `Buf::loaded`): the check claims the elements in use only, so the room after them may hold other
  values. Such a list is full, and its first push moves it. The room of a list after the image
  was made by the region.
- The unchecked forms are `unsafe`: `from_bytes_in` (or `view`) would accept the bytes, in this
  build, which makes them a tree. That is what `as_bytes` returned for the same `T`, in a build
  with the same layout, of a region that was built and not edited. Loading is then one `memcpy`,
  and viewing reads nothing. `from_bytes_unchecked_in` panics if the allocator does not give the
  block.
- `Error` (exhaustive): `Invalid(puns::Error)`, what `puns` found (`Size` also for bytes shorter
  than a `T` or longer than the limit of `T`); `Alloc`.

Checking is `puns::from_head::<T>`, after `fits::<T>()` and a function pointer to `Flat::store`
of `T`, so that a load refuses at compile time, at every depth, the types that a build refuses
(`fits_behind`). Each value is checked as its layout says: its scalars (`bool`, `char`,
`NonZero`, the numbers of given values of `puns::num`), the tag of an enum then the fields of its
variant, its rules (`valid(..)` of a field, `assert(..)` of the type), each `Near` (an offset of
zero bytes is not one, and is `None` for an `Option<Near>`), UTF-8 and `len <= cap` (the rule of
`ListLen`). Padding is not looked at. The pointees are claimed in the order of a copy, each after
everything before it: that gives bounds, alignment, no sharing, no overlap and no cycles, with
gaps allowed. `tests/corpus.rs` compares, on every image of a corpus and every corruption, which
ones are compact images of valid values (accepted, and copied to the same bytes) with what the
loaders of 0.8.1 accepted, whose results were those of 0.7.

- The only `unsafe` steps of a loader are the copy of `from_bytes_in` and the cast of checked
  bytes to `&T`. It never panics on any input.
- It does not recurse with the depth of the data, and takes no memory from the caller: `puns`
  checks the pointees of a type that points at itself in a loop of that type, with 64 entries for
  the pointees that wait, in its own frame (`puns::AIM_PARTS`).
- Elements of which every value is valid (no pointers, no rules, every bit pattern a value: lists
  and arrays of numbers) are not walked, so checking costs pointers, text and values that can be
  invalid, not bulk numbers: a large mapped table is ready at once.
- The bytes hold the layout of the target, its values in the byte order of their types, and
  distances and lengths as their `Offset` writes them (native-endian for `i16`, `i32`, `i64`);
  no header, no version in the bytes. A build with the same layout reads back what `as_bytes`
  gave.
- With `serde`, a region serializes as its bytes (`serialize_bytes` of `as_bytes` if `view`
  accepts them, or else of a copy, `Region::new(&*r)`), and deserializes through `from_bytes_in`
  (any allocator with `Default`); its `Error` becomes `de::Error::custom`. The visitor takes bytes
  borrowed or owned, or a sequence of `u8` (as JSON writes them), whose size hint it trusts up
  to 64 KiB. It needs `alloc`: a sequence is collected into a `Vec`.

## Internals

Files of `src/`:

- `lib.rs`: crate docs, exports, `Edit<'e, T>`, `__private`.
- `flat.rs`: `ALIGN` (see Layout), `limit`, `aligned`, `fits`, `fits_behind`, `block`, `Buf`
  (`regrow`, `alloc`, `grow`, `room`, `load`, `block`), `Sink` (`link`,
  `write_len`), `aim`, `Flat` and `Emit` with their impls (`data!` for numbers; `mod num`: the
  numbers of `puns::num`; `moved` for pointers), the deep copy (`write_copy`, `write_slice`),
  moves (`relocate`, `detach`, `move_elems`, `stash`, `unstash`, `shift`), the list block
  `List<O>`, `make`.
- `edit.rs`: `Raw` (the core of every handle: `set`, `replace`, `update`, `swap_in`, `pin`),
  `Old`, `ListEdit`, `OptionEdit`, `StrEdit`.
- `aim.rs`: `Pointee<O>`, the pointee of the aim of a `Near<T, O>`, `ListLen<O>`, and the
  contract of a read of a pointee.
- `bytes.rs`: `Error`, `view` (`puns::from_head`), `view_unchecked`.
- `near.rs`: `Near<T, O>` (its `Pun` and its aim): `Deref`, builders, trait impls, the pinned
  view, `pointee_mut`; `Offset`, its impls (`offsets!` for the integers of the target, `fixed!`
  for those of `puns::num`, whose `NonZero` is the one of `puns::num`), `offset` (its layout,
  checked).
- `region.rs`: `Region<T, A>`: `bind`, `copy_of`, constructors, `edit`, `reserve`, `as_bytes`,
  the loaders, `Drop`, trait impls, `serde` impls.
- `fixed.rs`: `Fixed`, its `Allocator`.

Mechanics:

- `Sink`: a `*mut Buf` (`alloc`, `write`, `link`), never a `&mut Buf`.
- Copies and moves are the hidden methods of `Flat`, which `#[flat]` writes per type and
  `flat.rs` for numbers and pointers. A deep copy (`write_copy`) copies `PLAIN` values as bytes
  and calls `store` on others: each field at its offset (the tag of an enum, then the fields of
  its variant), and for a pointer, the pointee allocated as the pointer says (one value, a list
  block `[cap][len]` as `List::new` makes it, text), copied the same way, then the offset;
  depth first and in the order of the fields, which is the compact image. Padding is not written,
  so it keeps what the buffer held there: zeros in new capacity (so a compact copy has zero
  padding), the old bytes under `set`, the bytes of the image after a load. They are initialized
  but for what an assignment through a `&mut` wrote (`init`, see Bytes). A move
  (`move_elems`, and `relocate`, `detach` for one value) copies the bytes at once (`ptr::copy`:
  the buffer's bytes are initialized) and re-aims the pointers where they are now (`shift`, which
  calls `Flat::shift` on each value: the pointers of its fields, of its variant for an enum);
  `insert` and `swap` keep a value's bytes aside (`stash`) and put them back re-aimed
  (`unstash`).
- Self-relative pointers are resolved from `&self`: valid under Tree Borrows, not Stacked
  Borrows. No pointer is made from an integer. Miri runs with `-Zmiri-tree-borrows
  -Zmiri-strict-provenance`.
- Tests: `tests/common/integration.rs` runs in two allocators (`integration`: `Global`,
  `user_alloc`: one written there, which moves every block that grows, with the tests of
  `Fixed`); `width` runs tests generic over the width with `i16`, `i32` and `i64`, the
  limit (reached cheaply with `i16`), mixed widths, and `a_region_past_2_gib` (`#[ignore]`d;
  CI runs it in release), offsets that do not round-trip, the contract of every offset, and the
  same tests with the offsets of `puns::num`; `layouts` covers what `nearest` takes from `puns`:
  generic types, numbers of a fixed byte order, pointers as bytes, optional pointers, the
  alignment of the elements of a list, and rules; `ir` runs
  `examples/ir.rs` (an image the same on every machine, `-Zmiri-disable-isolation` under Miri:
  it writes a file); `corpus` compares the loaders with recorded results (`corpus.expected`)
  and checks the rules at load; `no_alloc` also
  without `alloc`; `serde` with that feature; `compile_fail` (trybuild) holds the compile-time
  guarantees (`puns_*.rs`: what `puns` refuses of the pointers of a region). Miri runs `--lib`
  and the tests above (not `serde`, `compile_fail`) on x86-64, and
  on `i686` and `s390x` what the target can change (the `miri` job of `ci.yml`): the tests assume nothing of the target (sizes and alignments are
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
- No allocator type on handles: it is a parameter of `Region` alone; handles reach one `Buf`.
- No region over memory that already holds an image, edited in place: an assignment through a
  `&mut` to a type with padding writes uninitialized bytes, which would reach the `&mut [u8]` of
  the caller once the region is dropped. Load with `from_bytes_in`, a copy; a mapped image is read
  in place with `view`.
- No bytes inside the `Region` value: an allocator whose blocks are in its own value breaks the
  contract of `Allocator` (a block stays valid when its allocator moves), so `Fixed` borrows its
  buffer.
- No `to_bytes`: `as_bytes` borrows the bytes of a region whose bytes are all initialized and
  `Region::new(&*r)` copies any; a method fusing the two needs a `Cow` and exists only with
  `alloc`.
- No views on `Region`: viewing builds no region and needs no allocator. As methods of
  `Region<T>` they only served inference.
- No `#[non_exhaustive]` on `Error`: its two cases are every way a loader fails (the bytes, the
  allocator), and an exhaustive match lets callers handle each one, checked by
  the compiler. A new case would be a breaking release anyway.
- Copies recurse once per level of nesting (`Region::new(&*region)`, so saving an edited
  region), like `Clone` for a `Box`. Checking bytes does not.
- No check of the rules at construction: an edit can break one right after (a `&mut u8`), so it
  would not make a rule a guarantee; nothing relies on one (see Rules).
- No `isize` or `usize` width: the bytes would change with the target, for no reach that a
  fixed width lacks (`i64` already reaches `isize::MAX`). No unsigned width: after a move a
  pointer may aim backward. No cargo feature for the width: two crates of one build would
  disagree.
- No narrower pointer below a wider one: the limit comes from the root alone, as a const that
  followed pointers would cycle on recursive types; so widths may only grow going down.
- No fallback to `i32` in inference: Rust does not default type parameters in expressions, so a
  region whose root is a bare pointer names its type (`Region::<Near<[u32]>>::new(..)`).
- No test-only limit: `i16` pointers reach the real one at 32 KiB.
