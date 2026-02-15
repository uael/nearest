# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```sh
cargo build                          # Build all crates
cargo test                           # Run all tests (unit + integration + compile-fail)
cargo test --test integration        # Run only integration tests
cargo test --test integration TESTNAME  # Run a single integration test
cargo test --test compile_fail       # Run compile-fail tests (trybuild)
cargo bench --bench region           # Run benchmarks (criterion)
cargo +nightly miri test -Zmiri-permissive-provenance  # Run tests under Miri
cargo xtask fmt                      # Format code
cargo xtask fmt --check              # Check formatting
cargo xtask lint                     # Clippy + format check (-D warnings)
cargo xtask lint --fix               # Auto-fix clippy + formatting
cargo deny check                     # License/advisory/ban auditing
```

Requires **nightly** Rust (`nightly-2026-02-10` pinned in `rust-toolchain`) for `#![feature(offset_of_enum)]`.

## Architecture

**nearest** is a self-relative pointer library for region-based allocation. The entire data graph lives in a single contiguous byte buffer; all internal pointers are stored as `i32` offsets relative to their own address, making `Clone` a plain `memcpy` with no pointer fixup.

### Workspace Crates

- **`nearest/`** — Main library: `Near<T>`, `NearList<T>`, `Region<T>`, `Session`, traits (`Flat`, `Emit`, `Patch`)
- **`nearest-derive/`** — Proc-macro crate: `#[derive(Flat)]` and `#[derive(Emit)]`
- **`xtask/`** — Dev automation (fmt, lint)

### Core Types (nearest/src/)

| Type | File | Role |
|------|------|------|
| `Flat` (unsafe trait) | `flat.rs` | Marker for types storable in a region — no `Drop`, no heap pointers, correct `deep_copy` |
| `Emit<T>` (unsafe trait) | `emit.rs` | Builder pattern for constructing values in regions; derive generates `T::make(...)` factories |
| `Patch` (trait) | `patch.rs` | Unifies `Emitter` (construction) and `Region` (mutation) for position-based writes |
| `Near<T>` | `near.rs` | Self-relative pointer: 4-byte `NonZero<i32>` offset, `Deref<Target=T>` |
| `NearList<T>` | `list.rs` | Segmented linked list: `i32` head offset + `u32` length, iteration + indexing |
| `Region<T>` | `region.rs` | Owning contiguous byte buffer with root `T` at byte 0 |
| `Session` | `session.rs` | Ghost-cell branded mutable session API (`Ref<'id, T>`, `Cursor`, `Brand`) |
| `AlignedBuf<T>` | `buf.rs` | Low-level growable byte buffer with alignment guarantees |
| `Emitter<T>` | `emitter.rs` | Internal builder that writes values into an `AlignedBuf` during construction |

### Data Flow

1. **Construction**: `Region::new(T::make(...))` → derive-generated `Emit` writes values and patches self-relative offsets
2. **Reading**: `Region<T>: Deref<Target=T>`, `Near<T>: Deref<Target=T>` (computes `self_addr + offset`)
3. **Mutation**: `region.session(|s| { ... })` opens a branded session; mutations are append-only (old data becomes dead bytes)
4. **Compaction**: `region.trim()` deep-copies only reachable data into a fresh buffer

### Derive Macro (nearest-derive/src/)

- `flat.rs` — Generates `Flat` impls: enum repr validation, `deep_copy` body for structs/enums
- `emit.rs` — Generates `Emit` impls: `make()` factories, `__Builder` structs, `NearList` codegen
- `emit_proxy.rs` — `Emit` for proxy enums dispatching to inner builders
- `util.rs` — `FieldKind` classification (Primitive/Near/NearList/OptionNear/Other)

### Safety Model

- **Provenance**: Uses `expose_provenance`/`with_exposed_provenance` for self-relative pointer resolution
- **Branding**: Ghost-cell pattern with `for<'id>` lifetime prevents `Ref` escape from session and cross-session use (enforced at compile time)
- **No-Drop**: Derive macro emits `const { assert!(!needs_drop::<T>()) }` to reject types with `Drop`
- **Alignment**: Buffer aligned to `max(align_of::<Root>(), 8)`

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
