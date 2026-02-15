# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
