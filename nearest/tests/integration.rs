//! The integration tests, with every region on the heap; `inline.rs` runs them again with
//! inline storage.
#![expect(clippy::many_single_char_names, reason = "short names for regions and models")]

/// The storage of every region: the heap.
type Store = nearest::Heap;

include!("common/integration.rs");
