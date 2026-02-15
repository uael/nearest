# nearest

[![CI](https://github.com/uael/nearest/workflows/CI/badge.svg)](https://github.com/uael/nearest/actions)
[![docs.rs](https://img.shields.io/docsrs/nearest)](https://docs.rs/nearest)
[![crates.io](https://img.shields.io/crates/v/nearest)](https://crates.io/crates/nearest)
[![license](https://img.shields.io/crates/l/nearest)](LICENSE-MIT)

Self-relative pointers and region-based allocation for Rust.

Store entire data graphs in a single contiguous byte buffer where all internal
pointers are 4-byte `i32` offsets relative to their own address — cloning a
region is a plain `memcpy` with no fixup.

## Example

```rust
use nearest::{Flat, NearList, Region, empty};

#[derive(Flat, Debug)]
struct Block {
  id: u32,
  items: NearList<u32>,
}

// Build
let mut region = Region::new(Block::make(1, [10u32, 20, 30]));
assert_eq!(region.items.len(), 3);

// Read (Region<T>: Deref<Target = T>)
assert_eq!(region.id, 1);
assert_eq!(region.items[0], 10);

// Mutate via branded session
region.session(|s| {
  let items = s.nav(s.root(), |b| &b.items);
  s.splice_list(items, [40u32, 50]);
});
assert_eq!(region.items.len(), 2);

// Clone is memcpy
let cloned = region.clone();
assert_eq!(cloned.items[0], 40);
```

## Highlights

- **Zero-cost clone** — `Region::clone` is a `memcpy`; self-relative offsets
  need no fixup
- **Compile-time safety** — ghost-cell branded sessions prevent `Ref` escape
  or cross-session use
- **Declarative construction** — `#[derive(Flat)]` generates `make()` builders
  for tree-shaped data
- **Compaction** — `Region::trim` reclaims dead bytes left by append-only
  mutations
- **Serialization** — `as_bytes` / `from_bytes` round-trip with validation;
  optional `serde` feature for `Serialize`/`Deserialize`
- **Miri-validated** — all unsafe code tested under Miri with permissive
  provenance

## Getting started

Requires **nightly** Rust due to `#![feature(offset_of_enum)]`.

```toml
[dependencies]
nearest = "0.2"
```

See the [API documentation](https://docs.rs/nearest) for comprehensive guides
and examples.

## Minimum Supported Rust Version

Nightly Rust (`rust-version = "1.93.0"`), pinned to `nightly-2026-02-10`.

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
