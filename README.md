# nearest

[![Crates.io](https://img.shields.io/crates/v/nearest.svg)](https://crates.io/crates/nearest)
[![Documentation](https://docs.rs/nearest/badge.svg)](https://docs.rs/nearest)
[![CI](https://github.com/uael/nearest/actions/workflows/ci.yml/badge.svg)](https://github.com/uael/nearest/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/nearest.svg)](LICENSE-APACHE)

Self-relative pointer library for region-based allocation.

`nearest` provides self-relative pointers (`Near<T>`) and segmented lists
(`NearList<T>`) stored in a contiguous `Region<T>` buffer. All pointers are
4-byte `i32` offsets relative to their own address -- cloning a region is a
plain `memcpy` with no fixup.

## Key types

| Type | Description |
|------|-------------|
| `Region<T>` | Owning contiguous buffer; root `T` at byte 0 |
| `Near<T>` | Self-relative pointer (4-byte `NonZero<i32>` offset) |
| `NearList<T>` | Segmented list of `T` values (8-byte header) |
| `Session` | Branded mutable session (ghost-cell pattern) |
| `Ref<'id, T>` | Branded position token (4-byte, `Copy`, no borrow) |
| `Flat` | Marker trait for region-storable types |
| `Emit<T>` | Builder trait for declarative region construction |

## Features

- **Zero-cost cloning**: `Region::clone` is a `memcpy` -- all self-relative
  offsets remain valid without fixup.
- **Compile-time safety**: The `Session` API uses the ghost-cell pattern
  (`for<'id>`) to prevent `Ref` tokens from escaping the session closure or
  being used across sessions.
- **Declarative construction**: `#[derive(Flat)]` generates builder types
  (`T::make(...)`) that implement `Emit<T>`, enabling tree-shaped region
  construction in a single expression.
- **Compaction**: `Region::trim` re-emits only reachable data, eliminating
  dead bytes left by mutations.
- **Miri-validated**: All unsafe code is tested under Miri with permissive
  provenance (`-Zmiri-permissive-provenance`).

## Quick start

```rust
use nearest::{Flat, NearList, Region, empty};

#[derive(Flat, Debug)]
struct Block {
  id: u32,
  items: NearList<u32>,
}

let mut region = Region::new(Block::make(1, [10u32, 20, 30]));
assert_eq!(region.items.len(), 3);
assert_eq!(region.items[0], 10);

// Mutate via a branded session.
region.session(|s| {
  let items = s.nav(s.root(), |b| &b.items);
  s.splice_list(items, [40u32, 50]);
});

assert_eq!(region.items.len(), 2);
assert_eq!(region.items[0], 40);
assert_eq!(region.items[1], 50);
```

## Minimum Supported Rust Version

This crate requires **nightly** Rust (`rust-version = "1.93.0"`) due to
the use of `#![feature(offset_of_enum)]`.

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
