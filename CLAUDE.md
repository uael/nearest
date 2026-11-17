# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```sh
cargo build                          # Build all crates
cargo test --all                     # All tests: unit, integration, compile-fail, doc tests
cargo test --test integration        # Run only integration tests
cargo test --test integration TESTNAME  # Run a single integration test
cargo test --test compile_fail       # Compile-fail tests (trybuild); TRYBUILD=overwrite to regenerate .stderr
cargo run --example expr             # Examples (also: ir); they assert their results
cargo bench --bench region           # Benchmarks (criterion)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --lib --test integration  # Miri
cargo xtask fmt                      # Format code (cargo fmt on the pinned nightly)
cargo xtask fmt --check              # Check formatting
cargo xtask lint                     # Clippy (workspace, tests, examples, benches, -D warnings) + format check
cargo xtask lint --fix               # Auto-fix clippy + formatting
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all  # Build documentation
cargo deny check                     # License/advisory/ban auditing
```

The crates build on **stable** Rust (`1.100.0`, pinned in `rust-toolchain`; `rust-version` is
1.95). A pinned nightly (`nightly-2026-02-10`) is used only by `cargo xtask fmt` (`rustfmt.toml`
uses unstable options) and by Miri. Generate trybuild `.stderr` files with the pinned stable.

## Architecture

**nearest** keeps a value and everything it points to in one buffer. Every pointer is a 4-byte
offset relative to its own address, so `Clone` is a `memcpy` and reading is plain `&T`. Edits
go through handles checked by the borrow checker; the only runtime checks are the 2 GiB size
limit and index bounds. `DESIGN.md` is the specification.

### Workspace Crates

- **`nearest/`** — the library (`#![no_std]` + `alloc`, no features)
- **`nearest-derive/`** — the `#[flat]` attribute, re-exported as `nearest::flat`
- **`xtask/`** — dev automation (fmt, lint)

### Public surface

`#[flat]`, `trait Flat` (`type Edit`), `trait Emit<T>`, `type Edit<T>`, `Near<T>` (`new`,
`from_iter`), `Region<T>` (`new`, `edit`, `to_bytes`, `from_bytes`, `byte_len`), `ListEdit<T>`
(`push`, `insert`, `remove`, `swap`, `truncate`, `retain`, `at`, `set`, `replace`),
`OptionEdit<T>` (`as_mut`, `insert`, `set`, `replace`), `StrEdit` (`set`, `replace`),
`Old<'a, T>` (`map`), plus the trait impls listed in `DESIGN.md`. Hidden plumbing for the macro:
`nearest::__private::{Raw, Buf, Sink, fits, make, Variant}` and the hidden items of the two
traits. Do not add public items.

### Files (nearest/src/)

| File | Contents |
|------|----------|
| `lib.rs` | crate docs, exports, `Edit<T>`, `__private`, the size-limit unit test |
| `flat.rs` | `Buf` (8-aligned, zero-filled growth), `Sink`, `Flat` + impls, `Emit` + impls, moves (`relocate`, `move_elems`), the list block `List`, `make` |
| `edit.rs` | `Raw` (the core of every handle, with `set`/`replace`), `Old`, `ListEdit`, `OptionEdit`, `StrEdit` |
| `near.rs` | `Near<T>`: three `Deref` impls (`T`, `[T]`, `str`), builders, trait impls |
| `region.rs` | `Region<T>` |

`nearest-derive/src/lib.rs` generates, inside `const _: () = { .. }`: the `Flat` impl (`Edit`,
`PLAIN`, `store`, `edit`, each checking `fits`), constructors (`new`, or one snake-case function
per non-unit variant; a parameter for a field written `Near<X>` takes a builder of `X`), the
handle `__NearestEdit` (`set`, `replace`, a method per struct field, `as_mut` for enums, `Deref`,
`DerefMut` under `for<'a> X: Copy`), hidden `#[repr(C)]` mirror structs `(u8, fields..)` for
variant field offsets, and for multi-field variants a payload handle named through
`__private::Variant<I>`. `XMut<'a>` is emitted at module level with the visibility of `X`.

### Mechanics

- A handle is `&mut` to an unsized `#[repr(transparent)]` wrapper of `Raw<T>(PhantomData<T>,
  [()])`: the address is the region's `Buf`, the slice length is the position. `Flat::edit(buf,
  at)` builds it; for `Near<T>` it is the target's handle.
- `Flat::store(&self, sink, at, shift)`: `None` deep-copies targets; `Some(d)` re-aims offsets
  for a move of `d` bytes and never dereferences a `Near` (moves call it on stack copies).
  `PLAIN` types are copied as bytes.
- `Near<str>` points at `[len][bytes]`; `Near<[T]>` points at `len` in a block
  `[cap][len][elements]`. Builders and deep copies write `cap == len`.
- `replace` moves the old value to fresh memory, calls the closure with an `Old` at that
  position, writes the result into the place, and moves the old value back if that unwinds.
  `Old` written into its own region is a move, anywhere else a deep copy.

### Safety Model

- `Near` has no constructor and is not `Clone`/`Copy`. No safe API yields `&mut T` for a `T`
  containing a `Near` (plain data means `T: Copy`).
- One handle chain: a handle borrows its parent mutably, so no `&T` into the buffer is live
  while a handle can allocate. `Old<'a, T>` keeps the handle borrowed for `'a`.
- Tree Borrows discipline: a method takes its `*mut Buf` from the handle once and does
  everything through it, each access a reborrow that ends with the statement; `Sink` holds the
  raw pointer, never a `&mut Buf`. Allocate before taking references into the buffer.
- Self-relative pointers are resolved from `&self` (Tree Borrows; Stacked Borrows is not
  supported). No integer-to-pointer casts. Miri runs with `-Zmiri-tree-borrows
  -Zmiri-strict-provenance`.
- Panics leave every place unchanged or complete: list lengths are written after the element,
  `retain` keeps unvisited elements, `replace` restores the old value.
- `Buf` invariants hold also while unwinding: `len <= capacity <= MAX`; anything past the limit
  panics in `grow` before changing state. Unit tests use a 64 KiB limit.
- Compile-time checks (`fits`): alignment at most 8, non-zero size, no drop glue.
- **`pub(crate)` never relaxes safety invariants**: if a function has unsafe preconditions, it must
  be `unsafe fn` regardless of visibility.

## Releasing

Releases are automated with [release-plz](https://release-plz.dev/). On every push to `main`, the GitHub Action opens (or updates) a release PR that bumps versions and updates `CHANGELOG.md`. Merging that PR publishes to crates.io and creates a GitHub release with a `v<version>` tag.

- Config: `release-plz.toml`
- Workflow: `.github/workflows/release-plz.yml`
- Only `nearest` and `nearest-derive` are published; `xtask` is excluded
- `nearest` owns the changelog and git tags (`v{{ version }}`); `nearest-derive` is bumped in lockstep without its own changelog or tags

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
