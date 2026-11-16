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
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo miri test -p nearest --lib --test integration  # Miri
cargo xtask fmt                      # Format code (cargo fmt on the pinned toolchain)
cargo xtask fmt --check              # Check formatting
cargo xtask lint                     # Clippy (workspace, tests, examples, benches, -D warnings) + format check
cargo xtask lint --fix               # Auto-fix clippy + formatting
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all  # Build documentation
cargo deny check                     # License/advisory/ban auditing
```

Requires **nightly** Rust (`nightly-2026-02-10`, pinned in `rust-toolchain`; do not change it).
The library uses `ptr_metadata`; code that uses `#[flat]` needs `#![feature(pin_ergonomics)]`.

## Architecture

**nearest** keeps a value and everything it points to in one buffer. Every pointer is a 4-byte
offset relative to its own address, so `Clone` is a `memcpy` and reading is plain `&T`. Edits
are checked by the borrow checker; there are no runtime checks besides the 2 GiB size limit.
`DESIGN.md` is the specification.

### Workspace Crates

- **`nearest/`** — the library (`#![no_std]` + `alloc`, no features)
- **`nearest-derive/`** — the `#[flat]` attribute, re-exported as `nearest::flat`
- **`xtask/`** — dev automation (fmt, lint)

### Public surface

`#[flat]`, `trait Flat`, `trait Emit<'o, T>`, `Near<T>` (`new`, `from_iter`, `as_pin_mut`),
`Region<T>` (`new`, `edit`, `to_bytes`, `from_bytes`, `byte_len`), `Slot<'o, T>` (`put`, `at`,
`push`), plus trait impls. Hidden plumbing for the macro: `nearest::__private::{Sink, make}` and
the hidden items of the two traits. Do not add public items.

### Files (nearest/src/)

| File | Contents |
|------|----------|
| `lib.rs` | crate docs, exports, `__private`, the size-limit unit test |
| `flat.rs` | `Buf` (8-aligned, zero-filled growth), `Sink`, `Flat` + impls, `Emit` + impls, the list builder `List`, `make` |
| `near.rs` | `Near<T>`: resolution via `ptr_metadata`, `as_pin_mut`, builders, trait impls |
| `region.rs` | `Region<T>` (`edit` with the graft-on-drop guard) and `Slot` (deferred commit) |

`nearest-derive/src/lib.rs` generates, inside `const _: () = { .. }`: the `Flat` impl (`PLAIN`
and `store`, with a `needs_drop` assertion), constructors (`new`, or one snake-case function per
non-unit variant), and for enums hidden `#[repr(C)]` mirror structs `(u8, fields..)` that give
variant field offsets without `offset_of_enum`.

### Mechanics

- `Flat::store(&self, sink, at, shift)` writes a value field by field: `shift == None` deep-copies
  targets; `Some(d)` keeps targets and re-aims offsets for a move of `d` bytes. `PLAIN` types are
  copied as bytes.
- `Near<[T]>`/`Near<str>` point at a `u32` length header followed by the elements.
- `Region::edit` builds new data in a scratch `Sink` whose positions map to `base + pos` in the
  region; each `Slot` commits its staged value into its place when its closure returns, and the
  scratch is appended when `edit` returns or unwinds.

### Safety Model

- `Near` has no constructor, is not `Clone`, and is `!Unpin`; `#[flat]` adds `#[pin_v2]` so
  patterns project `Pin<&mut T>`.
- A region is a tree; `Slot<'o, T>`/`Emit<'o, T>` carry an invariant brand so a slot only accepts
  moves (`Pin<&'o mut T>`) from the value it replaces; `Slot::at` takes the place for the full
  `'o`, so it cannot be observed after being replaced.
- Self-relative pointers are resolved from `&self` (Tree Borrows; Stacked Borrows is not
  supported). No integer-to-pointer casts. Miri runs with `-Zmiri-tree-borrows
  -Zmiri-strict-provenance`.
- Commit writes into a place go through the pointer the closure's pin was derived from, after the
  closure returned, never through the buffer base while user pins are live.
- `Buf` invariants hold also while unwinding: `len <= capacity`, and `capacity + base <= MAX`, so
  every position fits an `i32` and anything past the limit panics in `grow` before changing state.
- Compile-time checks: alignment at most 8 and non-zero size (`fits`), no drop glue (in `store`).
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
