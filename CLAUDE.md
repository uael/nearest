# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```sh
cargo build                          # Build all crates
cargo test --all                     # All tests: unit, integration, compile-fail, doc tests
cargo test --test integration        # Run only integration tests (regions on the heap)
cargo test --test integration TESTNAME  # Run a single integration test
cargo test --test inline             # The same tests with `Inline` storage, and its own tests
cargo test --test user_storage       # The same tests with a storage written in the test, and others
cargo test --test compile_fail       # Compile-fail tests (trybuild); TRYBUILD=overwrite to regenerate .stderr
cargo run --example expr             # Examples (also: ir); they assert their results
cargo bench --bench region           # Benchmarks (criterion)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --lib --test integration --test inline --test user_storage --test no_alloc  # Miri
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --no-default-features --test no_alloc  # Miri, no `alloc`
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --features nightly --test pin_v2  # Miri, `nightly` feature
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --features nightly --lib --test integration --test pin_v2  # The `nightly` feature
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --features nightly --doc  # Its doc tests
cargo +nightly-2026-02-10 clippy --workspace --features nearest/nightly --tests --examples --benches -- -D warnings  # Its clippy
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --no-default-features --features nightly --lib --test no_alloc  # The `nightly` feature without `alloc`
cargo test -p nearest --no-default-features --lib --test no_alloc  # Without `alloc`: no heap
cargo clippy -p nearest --no-default-features --lib --test no_alloc -- -D warnings  # Its clippy (in `cargo xtask lint`)
cargo build -p nearest --no-default-features --target thumbv7em-none-eabihf  # A target without std or allocator
cargo xtask fmt                      # Format code (cargo fmt on the pinned nightly)
cargo xtask fmt --check              # Check formatting
cargo xtask lint                     # Clippy (workspace, tests, examples, benches; and without `alloc`; -D warnings) + format check
cargo xtask lint --fix               # Auto-fix clippy + formatting
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all  # Build documentation
cargo deny check                     # License/advisory/ban auditing
```

The crates build on **stable** Rust (`1.100.0`, pinned in `rust-toolchain`; `rust-version` is
1.95). Without `alloc`, CI also builds for `thumbv7em-none-eabihf`, a target without `std`. A
pinned nightly (`nightly-2026-02-10`) is used by `cargo xtask fmt` (`rustfmt.toml` uses unstable
options), by Miri, and to build and test the `nightly` feature. Generate trybuild `.stderr`
files with the pinned stable; the trybuild test runs on stable only.

## Architecture

**nearest** keeps a value and everything it points to in one buffer, on the heap or inline.
Every pointer is a 4-byte offset relative to its own address, so `Clone` is a `memcpy` and
reading is plain `&T`. Edits go through handles checked by the borrow checker; the only runtime
checks are the size limit (2 GiB, or the storage), index bounds and the check of loaded
bytes. `DESIGN.md` is the specification.

### Workspace Crates

- **`nearest/`** — the library (`#![no_std]`; the default feature `alloc` makes `Heap` a storage;
  one opt-in feature, `nightly`)
- **`nearest-derive/`** — the `#[flat]` attribute, re-exported as `nearest::flat`
- **`xtask/`** — dev automation (fmt, lint)

### Public surface

`#[flat]`, `trait Flat` (`type Edit<'e>`), `trait Emit<T>`, `type Edit<'e, T>`, `Near<T>` (`new`,
`from_iter`, `as_pin_mut`, `iter_pin_mut`), `Region<T, S = Heap>` (`new`, `new_in`, `edit`, `as_bytes`, `to_bytes`,
`from_bytes`, `view` (both with `alloc`), `from_bytes_with`, `view_with`, `byte_len`, and the
`unsafe` `from_bytes_unchecked`, `view_unchecked`),
`unsafe trait Storage` (`memory`, `memory_mut`, `grow`), `Heap`, `Inline<N>`,
`ListEdit<'e, T>` (`push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `try_at`,
`set`, `replace`, `update`, `iter_pin_mut`), `OptionEdit<'e, T>` (`as_mut`, `insert`, `take`,
`set`, `update`), `StrEdit<'e>` (`set`, `replace`, `update`), `Old<'e, T>` (no methods),
`Invalid` (`offset`), `Pending` (no methods), plus the trait impls listed in `DESIGN.md`.
Hidden plumbing for the macro: `nearest::__private::{Raw, Buf, Sink, Check, fits, make,
Variant, Pointer}`, and `unpin` with the feature; the hidden items of the two traits (with the
feature, also `Flat::UNPIN`). Do not add public items.

Every handle has `update`, the closure form of `replace`: `Raw::update` (which shares
`Raw::swap_in` with `replace`), its wrappers on `ListEdit`, `OptionEdit`, `StrEdit`, and the
generated one. Its `Old<'a, T>` has the lifetime of the handle's borrow, not of the edit.

The pinned view: `pin()` on generated handles (`Raw::pin`) gives `Pin<&mut X>`; the macro
generates `X::project` and `XPin<'a>` (a struct or an enum with the shape of `X`, every field
`Pin<&'a mut Field>`); `Near::as_pin_mut` and `iter_pin_mut` go down. `Near` holds a
`PhantomPinned`. A stored type must be `Unpin` exactly when its fields are: the macro emits
`impl Unpin for X where __NearestFields<'a, ..>: Unpin` (all field types), which makes a
hand-written impl a conflict. Without the feature it always writes that impl, and rejects a
`#[pin_v2]` it sees on the input (`#[pin_v2]` needs the feature, which adds it); a `#[pin_v2]`
added by another macro after `#[flat]` ran meets the impl, so soundness does not depend on the
macro seeing the attribute.

The feature `nightly` (`nearest/nightly` forwards to `nearest-derive/nightly`; nightly compiler
only) makes `#[flat]` add `#[pin_v2]` to every type, through
`#[allow_internal_unstable(pin_ergonomics)]` on the macro, so crates using `#[flat]` need no
gate. The bounded `Unpin` impl cannot stay under it: in a crate with `pin_ergonomics` the
compiler rejects every `Unpin` impl of a `#[pin_v2]` type, the macro's too, while in a crate
without it the compiler accepts a hand-written one. So the macro writes none, and the library
checks instead (`nearest/src/unpin.rs`, built only with the feature, as it uses
`specialization`): the hidden `Flat::UNPIN` ("every field is `Unpin`": `true` for plain
types, `false` for pointers without looking at the target, `__private::unpin::<Self>(true &
F1::UNPIN & ..)` for generated types) asserts that a type is not `Unpin` unless its fields
are. `Raw::pin`, `Near::as_pin_mut` and `Near::<[T]>::iter_pin_mut` evaluate it
(`unpin::pinned::<T>()`), so a hand-written impl on a type holding a `Near` fails the build
where it, or a type holding it by value, is pinned (in a build that generates code, not
`cargo check`). A `#[pin_v2]` written by hand is kept, not repeated. Without the feature nothing
changes. `tests/pin_v2.rs` is built only with the feature; the feature's own `compile_fail` doc
tests are on `nearest/src/unpin.rs`, including the compiler's rule in a crate with
`pin_ergonomics`. The trybuild test runs on stable only.
Names the generated code keeps: a field called `set`, `replace`, `pin` or `update` gets a
trailing underscore, and a variant called `Project` gets the constructor `project_`.

### Files (nearest/src/)

| File | Contents |
|------|----------|
| `lib.rs` | crate docs, exports, `Edit<'e, T>`, `__private`, the size-limit unit test |
| `flat.rs` | `Buf` (the attached storage: byte 0, zeroed capacity, memory size, `regrow`; `me`, compact mark; `load`), `Sink`, `Flat` + impls, `Emit` + impls, moves (`relocate`, `detach`, `move_elems`), the list block `List`, `make` |
| `edit.rs` | `Raw` (the core of every handle, with `set`/`replace`), `Old`, `ListEdit`, `OptionEdit`, `StrEdit` |
| `bytes.rs` | `Invalid` (`offset`; `LARGE`, a valid image larger than the storage), `Pending` (an entry of the checker's stack), the stack `Stack` (the caller's `&mut [Pending]`, or a `Vec` with `alloc`), the checker `Check` (`full` when the scratch ran out) and its steps (`target`, `list`, `text`, `items`), `check` (`None` if the scratch ran out) |
| `near.rs` | `Near<T>`: three `Deref` impls (`T`, `[T]`, `str`), builders, trait impls |
| `region.rs` | `Region<T, S>` (the storage attached before every `&mut` use), including `as_bytes`, `to_bytes`, `from_bytes`, `view` |
| `storage.rs` | `Storage` (the one public trait added for storage, `unsafe`, its contract), `Heap`, `Inline<N>` |
| `unpin.rs` | `nightly` feature only (`specialization`): `unpin` and `pinned`, the guard against a hand-written `Unpin` |

`nearest-derive/src/lib.rs` generates, inside `const _: () = { .. }`: the `Flat` impl (`Edit<'e>`,
`PLAIN`, `ANY`, `store`, `edit`, `check`, each checking `fits`), constructors (`new`, or one snake-case
function per non-unit variant; a parameter for a field written `Near<..>` takes
`<FieldType as __private::Pointer>::Arg`: a builder of `X` for `Near<X>`), the handle `__NearestEdit<'e, ..>` (`set`, `replace`, a method per struct field, `as_mut`
for enums, `Deref`, `DerefMut` under `for<'a> X: Copy`), hidden `#[repr(C)]` mirror structs
`(u8, fields..)` for variant field offsets, and for multi-field variants a payload handle (a
`Raw` of the mirror, so it has no `set`/`replace`) named through `__private::Variant<I>`.
`XMut<'a, 'e>` is emitted at module level with the visibility of `X`.

### Mechanics

- A handle is `&'a mut` to an unsized `#[repr(transparent)]` wrapper of `Raw<'e, T>`: the address
  is the region's `Buf`, the slice length is the position. `Flat::edit(buf, at)` builds it; for
  `Near<T>` it is the target's handle. `'e` is the edit, the borrow taken by `Region::edit`.
- `Flat::store(&self, sink, at, shift)`: `None` deep-copies targets; `Some(d)` re-aims offsets
  for a move of `d` bytes and never dereferences a `Near` (moves call it on stack copies).
  `PLAIN` types are copied as bytes.
- `Near<str>` points at `[len][bytes]`; `Near<[T]>` points at `len` in a block
  `[cap][len][elements]`. Deep copies and builders write `cap == len`; a builder whose iterator
  did not match its size hint writes its real capacity and clears the compact mark.
- `set` keeps the old bytes on the stack and restores them on unwind. `replace` moves the old
  value to fresh bytes (`detach`), writes the new one (moving the old one back on unwind) and
  returns an `Old` (`Raw::swap_in`, shared with `update`); `ListEdit::remove` and `OptionEdit::take` also return one. An `Old` written
  into its own region (same `me`) is a move, anywhere else a deep copy.
- Bytes: a region's compact image is what a deep copy writes. `Flat::check` (safe, generated)
  walks an image in that order without recursion; each pointer must aim exactly where the copy
  would have put its target. Elements with `Flat::ANY` (no pointers, every bit pattern valid)
  are not walked. `to_bytes` borrows the buffer when the compact mark is set. What waits to be
  checked is on a stack that grows with nesting: a `Vec` for `from_bytes`/`view` (`alloc`
  only), the caller's `&mut [Pending]` for `from_bytes_with`/`view_with` (`None` when full;
  `bytes.len() / 4` entries always suffice). No fixed bound anywhere. `from_bytes` asks for
  room in the storage (`Buf::load`) after the check and returns `Invalid::LARGE` if the
  storage gives too little; `Storage::grow` does nothing by default.

### Safety Model

- `Near` has no constructor and is not `Clone`/`Copy`. No safe API yields `&mut T` for a `T`
  containing a `Near` (plain data means `T: Copy`).
- One handle chain: a handle borrows its parent mutably, so no `&T` into the buffer is live
  while a handle can allocate. An `Old<'e, T>` keeps the region borrowed for the edit; bytes
  no longer reachable are never written again (I5), so it stays intact.
- Storage: `Region<T, S>` keeps its bytes in `S: Storage` (contract in `storage.rs` and
  DESIGN.md), which may move with the region (`Inline`). Handles never see the storage:
  `Region::attach` writes its memory, address and `regrow` into the `Buf` before every use
  through `&mut self` (`new_in`, `edit`, loading, `clone`), always after the region's last move
  (attach, then never return the region by value before using it); uses through `&self` call
  `memory` (`Region::base`), never the `Buf`'s pointer. `regrow` attaches again after `grow`,
  also when it unwinds. A storage's memory may hold anything: `Buf::cap` is the zeroed prefix
  (`[len, cap)` zero, `[cap, mem)` anything), raised by the cold `Buf::grow`, which zeroes the
  next part and calls `regrow` only when `mem` is too short; `attach` keeps `cap` (a length).
  Reads touch only `[0, len)`.
- One provenance per edit (I8): `Region::edit` stores `&raw mut self.buf` in `Buf::me`; a handle
  reads `me` through its own reference, and every other access of the edit (handles, `Sink`,
  `Old`) goes through that pointer, each access a reborrow that ends with the statement. Never
  keep a `&mut Buf`/`&Buf` across a call that touches the buffer. Allocate before taking
  references into the buffer.
- An edit stays on one thread (I9): `Raw` holds a `PhantomData<*mut ()>`, so handles are
  neither `Send` nor `Sync`, and neither is `Old`. An `Old` reads its region without borrowing
  a handle, so a handle on another thread could grow the buffer under it.
- Self-relative pointers are resolved from `&self` (Tree Borrows; Stacked Borrows is not
  supported). No integer-to-pointer casts. Miri runs with `-Zmiri-tree-borrows
  -Zmiri-strict-provenance`.
- Panics leave every place unchanged or complete: list lengths are written after the element,
  `retain` keeps unvisited elements, `set` and `replace` restore the old value.
- The checker is safe code that never panics (checked arithmetic, `get` instead of indexing);
  the only unsafe step of `view`/`from_bytes` is the cast of checked bytes to `&T`.
- `Buf` invariants hold also while unwinding: `len <= cap <= MAX`; anything past 2 GiB or past
  what the storage gives panics in `grow` before the length changes. Unit tests use a 64 KiB
  `MAX`.
- Compile-time checks (`fits`): alignment at most 8, non-zero size, no drop glue.
- **`pub(crate)` never relaxes safety invariants**: if a function has unsafe preconditions, it must
  be `unsafe fn` regardless of visibility.

## Releasing

Releases are automated with [release-plz](https://release-plz.dev/). On every push to `main`, the GitHub Action opens (or updates) a release PR that bumps versions and updates `CHANGELOG.md`. Merging that PR publishes to crates.io and creates a GitHub release with a `v<version>` tag.

- Config: `release-plz.toml`
- Workflow: `.github/workflows/release-plz.yml`
- Only `nearest` and `nearest-derive` are published; `xtask` is excluded
- `nearest` owns the changelog and git tags (`v{{ version }}`); `nearest-derive` has no changelog or tags of its own
- Both crates are released under the same version (`version_group` in `release-plz.toml`): the higher of their next versions

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) — release-plz uses them to determine version bumps and generate changelog entries.

```
<type>[optional scope]: <description>

[optional body]
```

| Type | Bump | Example |
|------|------|---------|
| `fix:` | patch | `fix: handle zero-length regions in trim` |
| `feat:` | minor | `feat: add Region::into_vec conversion` |
| `feat!:` / `BREAKING CHANGE:` | major | `feat!: rename Emitter to Builder` |
| `docs:`, `ci:`, `chore:`, `refactor:`, `test:` | none | `ci: pin release-plz action version` |

## Code Style

- 2-space indentation, max 100 columns
- Imports: reordered, grouped by `StdExternalCrate`, granularity `Crate`
- Clippy: `pedantic` + `nursery` at warn; `dbg_macro`, `allow_attributes`, `missing_safety_doc`, `undocumented_unsafe_blocks` denied
- Dual-licensed: Apache-2.0 OR MIT
