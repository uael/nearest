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
cargo test --test width              # Pointers of every width (i16, i32, i64), their limits
cargo test --release --test width -- --ignored  # A region past 2 GiB (about 4 GB of memory)
cargo test -p nearest --features serde --test serde  # The `serde` feature
cargo test --test compile_fail       # Compile-fail tests (trybuild); TRYBUILD=overwrite to regenerate .stderr
cargo run --example expr             # Examples (also: ir); they assert their results
cargo bench --bench region           # Benchmarks (criterion)
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --lib --test integration --test inline --test user_storage --test no_alloc --test width  # Miri
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --no-default-features --test no_alloc  # Miri, no `alloc`
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-strict-provenance" cargo +nightly-2026-02-10 miri test -p nearest --features nightly --test pin_v2  # Miri, `nightly` feature
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --features nightly --lib --test integration --test pin_v2  # The `nightly` feature
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --features nightly --doc  # Its doc tests
cargo +nightly-2026-02-10 clippy --workspace --features nearest/nightly --tests --examples --benches -- -D warnings  # Its clippy
RUSTFLAGS="-D warnings" cargo +nightly-2026-02-10 test -p nearest --no-default-features --features nightly --lib --test no_alloc  # The `nightly` feature without `alloc`
cargo test -p nearest --no-default-features --lib --test no_alloc  # Without `alloc`: no heap
cargo clippy -p nearest --no-default-features --lib --test no_alloc -- -D warnings  # Its clippy (in `cargo xtask lint`)
cargo build -p nearest --no-default-features --target thumbv7em-none-eabihf  # A target without std or allocator
cargo +nightly-2026-02-10 build -p nearest --no-default-features -Zbuild-std=core --target msp430-none-elf  # A 16-bit target
cargo xtask fmt                      # Format code (cargo fmt on the pinned nightly)
cargo xtask fmt --check              # Check formatting
cargo xtask lint                     # Clippy (workspace, tests, examples, benches; and without `alloc`; -D warnings) + format check
cargo xtask lint --fix               # Auto-fix clippy + formatting
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all  # Build documentation
cargo deny check                     # License/advisory/ban auditing
```

The crates build on **stable** Rust (`1.100.0`, pinned in `rust-toolchain`; `rust-version` is
1.95). Without `alloc`, CI also builds for `thumbv7em-none-eabihf`, a target without `std`, and
for `msp430-none-elf`, a 16-bit one. A pinned nightly (`nightly-2026-02-10`) is used by
`cargo xtask fmt` (`rustfmt.toml` uses unstable options), by Miri, to build `core` for the
16-bit target, and to build and test the `nightly` feature. Generate trybuild `.stderr`
files with the pinned stable; the trybuild test runs on stable only.

## Architecture

**nearest** keeps a value and everything it points to in one buffer, on the heap, inline or in
any storage. Every pointer is a 4-byte offset relative to its own address, so `Clone` is a
`memcpy` and reading is plain `&T`; edits go through handles checked by the borrow checker.
**`nearest/CLAUDE.md` is the specification** (public surface, invariants, mechanics, files,
what is left out and why): read it before changing the library, and keep it, the crate docs
and `README.md` in step with the code.

- **`nearest/`** — the library (`#![no_std]`; default feature `alloc`, opt-in `nightly`)
- **`nearest-derive/`** — the `#[flat]` attribute, re-exported as `nearest::flat`
- **`xtask/`** — dev automation (fmt, lint)

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
