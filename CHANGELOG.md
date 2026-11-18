# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2](https://github.com/uael/nearest/compare/v0.7.1...v0.7.2) - 2026-11-18

### Fixed

- a region is aligned as its words are, and an offset must fit `isize` ([#59](https://github.com/uael/nearest/pull/59))

## [0.7.1](https://github.com/uael/nearest/compare/v0.7.0...v0.7.1) - 2026-11-17

### Added

- open `Offset` to types of the user's, as an `unsafe trait` ([#57](https://github.com/uael/nearest/pull/57))

## [0.7.0](https://github.com/uael/nearest/compare/v0.6.2...v0.7.0) - 2026-11-17

### Added

- [**breaking**] pointer widths, `Near<T, O>` with `i16`, `i32` or `i64` ([#55](https://github.com/uael/nearest/pull/55))

## [0.6.2](https://github.com/uael/nearest/compare/v0.6.1...v0.6.2) - 2026-11-17

### Other

- the serde feature in the crate docs ([#53](https://github.com/uael/nearest/pull/53))

## [0.6.1](https://github.com/uael/nearest/compare/v0.6.0...v0.6.1) - 2026-11-17

### Added

- serde feature, a region is its bytes ([#51](https://github.com/uael/nearest/pull/51))

## [0.6.0](https://github.com/uael/nearest/compare/v0.5.5...v0.6.0) - 2026-11-17

### Added

- [**breaking**] one error enum, no to_bytes, views as functions ([#44](https://github.com/uael/nearest/pull/44))

## [0.5.5](https://github.com/uael/nearest/compare/v0.5.4...v0.5.5) - 2026-11-17

### Added

- *(derive)* #[flat(into)] on fields ([#43](https://github.com/uael/nearest/pull/43))

## [0.5.4](https://github.com/uael/nearest/compare/v0.5.3...v0.5.4) - 2026-11-17

### Added

- *(derive)* #[flat(rename = "..")] on variants ([#42](https://github.com/uael/nearest/pull/42))

## [0.5.3](https://github.com/uael/nearest/compare/v0.5.2...v0.5.3) - 2026-11-17

### Added

- Heap::with_capacity ([#41](https://github.com/uael/nearest/pull/41))

## [0.5.2](https://github.com/uael/nearest/compare/v0.5.1...v0.5.2) - 2026-11-17

### Added

- Display for Region and Near ([#40](https://github.com/uael/nearest/pull/40))

## [0.5.1](https://github.com/uael/nearest/compare/v0.5.0...v0.5.1) - 2026-11-17

### Added

- custom storage, and no_std without alloc ([#39](https://github.com/uael/nearest/pull/39))

## [0.5.0](https://github.com/uael/nearest/compare/v0.4.5...v0.5.0) - 2026-11-17

### Added

- [**breaking**] one buffer, plain `&mut` handles, O(1) moves, checked bytes, stable Rust ([#35](https://github.com/uael/nearest/pull/35)) ([#36](https://github.com/uael/nearest/pull/36))

## [0.4.5](https://github.com/uael/nearest/compare/v0.4.4...v0.4.5) - 2026-03-09

### Fixed

- *(nearest-derive)* add Flat bounds to unbounded type params in emit impl ([#32](https://github.com/uael/nearest/pull/32))

## [0.4.4](https://github.com/uael/nearest/compare/v0.4.3...v0.4.4) - 2026-03-09

### Added

- add `array` emitter for `[T; N]` fields ([#30](https://github.com/uael/nearest/pull/30))

## [0.4.3](https://github.com/uael/nearest/compare/v0.4.2...v0.4.3) - 2026-03-09

### Other

- deduplicate buffer operations between Emitter and Region ([#28](https://github.com/uael/nearest/pull/28))

## [0.4.2](https://github.com/uael/nearest/compare/v0.4.1...v0.4.2) - 2026-03-08

### Other

- rewrite README with comprehensive examples and comparison ([#26](https://github.com/uael/nearest/pull/26))

## [0.4.1](https://github.com/uael/nearest/compare/v0.4.0...v0.4.1) - 2026-03-08

### Other

- remove alloc dependency from Session list operations ([#24](https://github.com/uael/nearest/pull/24))

## [0.4.0](https://github.com/uael/nearest/compare/v0.3.2...v0.4.0) - 2026-03-08

### Other

- use explicit near()/list()/maybe() wrappers for Emit builders ([#21](https://github.com/uael/nearest/pull/21))

## [0.3.2](https://github.com/uael/nearest/compare/v0.3.1...v0.3.2) - 2026-03-08

### Added

- self-sufficient Near<T> and NearList<T> Flat impls ([#18](https://github.com/uael/nearest/pull/18))

## [0.3.1](https://github.com/uael/nearest/compare/v0.3.0...v0.3.1) - 2026-02-15

### Added

- add consuming conversions for Region ([#17](https://github.com/uael/nearest/pull/17))
- add #[flat(into)] and #[flat(rename)] derive macro attributes ([#16](https://github.com/uael/nearest/pull/16))
- add collection methods to NearList and Session ([#15](https://github.com/uael/nearest/pull/15))
- implement Display for Near<T>, NearList<T>, and Region<T>
- add unsafe Region::from_bytes_unchecked() and from_buf_unchecked()
- implement FusedIterator for NearListIter and add NearList::last
- implement PartialEq and Eq for Region<T>, Near<T>, and NearList<T>

### Fixed

- avoid raw byte comparison on types with enum padding in Miri tests

### Other

- fix crate version in README.md

## [0.3.0](https://github.com/uael/nearest/compare/v0.2.1...v0.3.0) - 2026-02-15

### Added

- add optional `serde` feature with Serialize/Deserialize for Region
- [**breaking**] add `Flat::validate` with `Region::as_bytes` / `Region::from_bytes`

### Fixed

- mark pub(crate) functions with unsafe preconditions as unsafe fn
- copy before validate in `from_bytes` to avoid double read

### Other

- document serialization, features, no_std and serde in lib.rs and README

## [0.2.1](https://github.com/uael/nearest/compare/v0.2.0...v0.2.1) - 2026-02-15

### Fixed

- correct swapped arrows in Region memory layout diagram

### Other

- add 5 real-world examples (expr, json, ecs, fs, fsm)

## [0.2.0](https://github.com/uael/nearest/compare/v0.1.2...v0.2.0) - 2026-02-15

### Added

- [**breaking**] add `no_std` support with `Buf` trait and `FixedBuf<N>`

### Fixed

- resolve broken doc links for no_std compatibility

### Other

- make README badges clickable links

## [0.1.2](https://github.com/uael/nearest/compare/v0.1.1...v0.1.2) - 2026-02-15

### Other

- rewrite lib.rs crate docs and README

## [0.1.1](https://github.com/uael/nearest/compare/v0.1.0...v0.1.1) - 2026-02-15

### Other

- update README badges for consistency and clarity
- update README license badge for clarity
- Fix CI badge
- Update README badges: improve clarity and remove redundant links

## [0.1.0] - 2026-02-15

Initial release.

### Added

- `Region<T>`: owning contiguous byte buffer with root `T` at byte 0.
- `Near<T>`: self-relative pointer stored as a 4-byte `NonZero<i32>` offset.
- `NearList<T>`: segmented linked list with 8-byte header (offset + length).
- `Session` API: branded mutable session using the ghost-cell pattern
  (`for<'id>`) preventing `Ref` tokens from escaping or crossing sessions.
- `Flat` marker trait with `#[derive(Flat)]` for safe region-storable types.
- `Emit<T>` builder trait with declarative tree-shaped region construction.
- `Region::trim` for compaction of dead bytes after mutations.
- Session operations: `splice`, `splice_list`, `re_splice_list`, `map_list`,
  `filter_list`, `push_front`, `push_back`, `extend_list`, `graft`.
- `Cursor` API for fluent chained navigation and mutation.
- Compile-fail tests verifying `Ref` escape prevention, cross-session safety,
  `Drop` rejection, enum repr requirements, and `Flat` field enforcement.
- Miri validation with `-Zmiri-permissive-provenance`.

[0.1.0]: https://github.com/uael/nearest/releases/tag/v0.1.0
