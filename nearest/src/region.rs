use core::{fmt, mem, num::NonZero, ops::Deref};

use crate::{
  Emit, Flat,
  buf::Buf,
  emitter::{Emitter, Pos},
  list::Segment,
  session::{Brand, Session},
};

/// An owning, contiguous byte buffer whose root value `T` starts at byte 0.
///
/// All [`Near`](crate::Near) and [`NearList`](crate::NearList) pointers
/// inside the region are self-relative offsets, so `Clone` is a plain memcpy
/// — no fixup needed.
///
/// # Memory layout
///
/// ```text
/// ┌──────────────────────────────────────────────┐
/// │ Root T (starts at byte 0)                    │
/// │  ├── scalar fields (inline)                  │
/// │  ├── Near<U> → i32 offset ───────┐           │
/// │  └── NearList<V> → i32 + u32 ──┐ │           │
/// │                                │ │           │
/// │ [padding]                      │ │           │
/// │ U value ◄──────────────────────│─┘           │
/// │ [padding]                      │             │
/// │ Segment<V> header ◄────────────┘             │
/// │ V values...                                  │
/// └──────────────────────────────────────────────┘
/// ```
///
/// # Soundness
///
/// **Ownership**: A `Region` exclusively owns its buffer. There is no
/// shared mutable state. `Clone` performs a byte-for-byte copy of the buffer;
/// all self-relative offsets remain valid because they are position-independent.
///
/// **Alignment**: The buffer base is aligned to `max(align_of::<T>(), 8)`.
/// Every sub-allocation is padded to the target type's alignment. A
/// compile-time assertion ensures no type exceeds the buffer's base alignment.
///
/// **Mutation safety**: All mutations go through [`Session`](crate::Session),
/// which holds `&mut Region`. The branded `'id` lifetime on [`Ref`](crate::Ref)
/// prevents refs from escaping or crossing sessions.
///
/// **`Send`/`Sync`**: Implemented with `T: Send + Sync` bounds as
/// defense-in-depth. All `Flat` types are `Send + Sync` by construction
/// (no heap pointers, no interior mutability), but the bounds let the
/// compiler verify this.
#[cfg(feature = "alloc")]
#[must_use]
pub struct Region<T: Flat, B: Buf = crate::buf::AlignedBuf<T>> {
  buf: B,
  _type: core::marker::PhantomData<T>,
}

/// See the `#[cfg(feature = "alloc")]` variant above for full documentation.
#[cfg(not(feature = "alloc"))]
#[must_use]
pub struct Region<T: Flat, B: Buf> {
  buf: B,
  _type: core::marker::PhantomData<T>,
}

#[cfg(feature = "alloc")]
impl<T: Flat> Region<T> {
  /// Construct a region from a builder using the default [`AlignedBuf`](crate::AlignedBuf).
  ///
  /// The builder emits the root `T` (and any nested data) into a fresh
  /// emitter, producing an immutable `Region`.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node {
  ///   id: u32,
  ///   children: NearList<u32>,
  /// }
  ///
  /// // Build with the derive-generated `Node::make(id, children)` builder.
  /// let region = Region::new(Node::make(1, [10u32, 20, 30]));
  /// assert_eq!(region.id, 1);
  /// assert_eq!(region.children.len(), 3);
  ///
  /// // Build with an empty list.
  /// let region = Region::new(Node::make(2, empty()));
  /// assert_eq!(region.children.len(), 0);
  /// ```
  pub fn new(builder: impl Emit<T>) -> Self {
    Self::new_in(builder)
  }

  /// Construct a region with a pre-allocated buffer of at least `capacity` bytes.
  ///
  /// Avoids repeated reallocations when the final size is approximately known.
  pub fn with_capacity(capacity: u32, builder: impl Emit<T>) -> Self {
    Self::with_capacity_in(capacity, builder)
  }
}

impl<T: Flat, B: Buf> Region<T, B> {
  /// Construct a region from a builder using an explicit buffer type `B`.
  ///
  /// For the default heap-backed buffer, use [`Region::new`] instead.
  pub fn new_in(builder: impl Emit<T>) -> Self {
    let mut em = Emitter::<T, B>::new();
    builder.emit(&mut em);
    em.finish()
  }

  /// Construct a region with a pre-allocated buffer of at least `capacity` bytes.
  pub fn with_capacity_in(capacity: u32, builder: impl Emit<T>) -> Self {
    let mut em = Emitter::<T, B>::with_capacity(capacity);
    builder.emit(&mut em);
    em.finish()
  }

  /// Create a region from a buffer.
  ///
  /// # Safety
  ///
  /// The buffer must contain a valid representation of `T` at byte 0 and
  /// all transitively reachable data must be valid. Constructing a region
  /// from an invalid buffer causes UB on `Deref`.
  pub(crate) unsafe fn from_buf(buf: B) -> Self {
    debug_assert!(buf.len() as usize >= mem::size_of::<T>(), "buffer too small for root type");
    Self { buf, _type: core::marker::PhantomData }
  }

  /// Open a branded session. [`Ref`](crate::Ref)s created inside the closure
  /// cannot escape or be used with a different session — **compile-time safety,
  /// zero runtime cost**.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node {
  ///   id: u32,
  ///   children: NearList<u32>,
  /// }
  ///
  /// let mut region = Region::new(Node::make(1, [10u32, 20]));
  ///
  /// // Read and mutate inside a session.
  /// region.session(|s| {
  ///   let root = s.root();
  ///   assert_eq!(s.at(root).id, 1);
  ///
  ///   let children = s.nav(root, |n| &n.children);
  ///   s.splice_list(children, [99u32]);
  /// });
  ///
  /// assert_eq!(region.children.len(), 1);
  /// assert_eq!(region.children[0], 99);
  /// ```
  pub fn session<R>(&mut self, f: impl for<'id> FnOnce(&mut Session<'id, '_, T, B>) -> R) -> R {
    self.buf.expose_provenance();
    let brand = Brand::new();
    let mut session = Session::new(self, brand);
    f(&mut session)
  }

  /// Bulk-copy another region's bytes into this region, returning the position
  /// of the grafted root.
  ///
  /// All self-relative pointers within the grafted data remain valid because
  /// the entire source is copied as a contiguous block. The graft position is
  /// aligned to the source region's maximum internal alignment so that all
  /// transitively referenced data maintains correct alignment.
  pub(crate) fn graft_internal<U: Flat, B2: Buf>(&mut self, src: &Region<U, B2>) -> Pos {
    // All types within `src` have alignment ≤ BUF_ALIGN (enforced by
    // Buf::alloc). Both regions share alignment ≥ 8, so aligning the graft
    // offset to B2::ALIGN preserves alignment for all data.
    self.buf.align_to(B2::ALIGN);
    let pos = Pos(self.buf.len());
    self.buf.extend_from_slice(src.buf.as_bytes());
    pos
  }

  /// Returns the total byte length of the region.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Node { id: u32, items: NearList<u32> }
  ///
  /// let region = Region::new(Node::make(1, empty()));
  /// assert!(region.byte_len() >= core::mem::size_of::<Node>());
  /// ```
  #[must_use]
  pub fn byte_len(&self) -> usize {
    self.buf.len() as usize
  }

  /// Raw const pointer to the buffer (for Cursor reads).
  pub(crate) fn deref_raw(&self) -> *const u8 {
    self.buf.expose_provenance();
    self.buf.as_ptr()
  }

  /// Compact this region by re-emitting only reachable data.
  ///
  /// After mutations (e.g. [`splice_list`](Session::splice_list),
  /// [`push_front`](Session::push_front)), old targets of redirected
  /// [`Near`](crate::Near)/[`NearList`](crate::NearList) pointers become
  /// dead bytes. `trim` walks the root `T` and all transitively reachable
  /// data, emitting a fresh compact buffer via `Emit<T> for &T` deep-copy.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Node::make([1u32, 2, 3]));
  /// let before = region.byte_len();
  ///
  /// // Mutation leaves dead bytes (old list data).
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |n| &n.items);
  ///   s.splice_list(items, [42u32]);
  /// });
  /// assert!(region.byte_len() > before);
  ///
  /// // Trim compacts the region.
  /// region.trim();
  /// assert!(region.byte_len() <= before);
  /// assert_eq!(region.items[0], 42);
  /// ```
  pub fn trim(&mut self) {
    let new_buf = {
      let root: &T = self;
      let mut em = Emitter::<T, B>::new();
      Emit::<T>::emit(root, &mut em);
      em.into_buf()
    };
    self.buf = new_buf;
  }

  // --- pub(crate) buffer mutation methods for Session/Patch ---

  /// Ensure at least `additional` bytes of spare capacity.
  pub(crate) fn reserve_internal(&mut self, additional: u32) {
    self.buf.reserve(additional);
  }

  /// Allocate aligned space for one `U`, returning its position.
  pub(crate) fn alloc_internal<U: Flat>(&mut self) -> Pos {
    self.buf.alloc::<U>()
  }

  /// Write a [`Flat`] value at position `at`.
  ///
  /// # Safety
  ///
  /// `at` must have been allocated for `U` via `alloc_internal::<U>()`,
  /// ensuring correct alignment.
  pub(crate) unsafe fn write_flat_internal<U: Flat>(&mut self, at: Pos, val: U) {
    let start = at.0 as usize;
    let size = mem::size_of::<U>();
    assert!(
      start + size <= self.buf.len() as usize,
      "write_flat out of bounds: {}..{} but len is {}",
      start,
      start + size,
      self.buf.len()
    );
    // SAFETY: Bounds checked above. `at` was allocated for `U` via
    // `alloc_internal::<U>()`, ensuring correct alignment. `mem::forget`
    // prevents double-drop.
    unsafe {
      let src = core::ptr::from_ref(&val).cast::<u8>();
      core::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), size);
    }
    mem::forget(val);
  }

  /// Patch a [`Near<U>`](crate::Near) at position `at` to point to `target`.
  ///
  /// # Safety
  ///
  /// `at` must point to a `Near<U>` field within a previously allocated
  /// value, and `target` must be a position allocated for `U`.
  pub(crate) unsafe fn patch_near_internal(&mut self, at: Pos, target: Pos) {
    let rel = i64::from(target.0) - i64::from(at.0);
    let rel_i32: i32 = rel.try_into().expect("near offset overflow");
    let nz = NonZero::new(rel_i32).expect("near offset must be non-zero");

    let start = at.0 as usize;
    let size = mem::size_of::<NonZero<i32>>();
    assert!(start + size <= self.buf.len() as usize, "patch_near out of bounds");
    // SAFETY: Bounds checked above. `at` points to the `Near<U>` field whose
    // first 4 bytes hold a `NonZero<i32>` offset.
    unsafe {
      let src = core::ptr::from_ref(&nz).cast::<u8>();
      core::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), size);
    }
  }

  /// Patch a [`NearList<U>`](crate::NearList) header at position `at`.
  ///
  /// # Safety
  ///
  /// `at` must point to a `NearList<U>` field within a previously allocated
  /// value, and `target` must be a position of a `Segment<U>` (or
  /// `Pos::ZERO` when `len == 0`).
  pub(crate) unsafe fn patch_list_header_internal(&mut self, at: Pos, target: Pos, len: u32) {
    let off_pos = at.0 as usize;
    let len_pos = off_pos + mem::size_of::<i32>();

    assert!(
      len_pos + mem::size_of::<u32>() <= self.buf.len() as usize,
      "patch_list_header out of bounds"
    );

    let rel: i32 = if len == 0 {
      0
    } else {
      let r = i64::from(target.0) - i64::from(at.0);
      r.try_into().expect("list header offset overflow")
    };

    // SAFETY: Bounds checked above. The list header at `at` has layout
    // `[i32 offset, u32 len]`, and both writes are within bounds.
    unsafe {
      let buf = self.buf.as_mut_ptr();
      core::ptr::copy_nonoverlapping(
        core::ptr::from_ref(&rel).cast::<u8>(),
        buf.add(off_pos),
        mem::size_of::<i32>(),
      );
      core::ptr::copy_nonoverlapping(
        core::ptr::from_ref(&len).cast::<u8>(),
        buf.add(len_pos),
        mem::size_of::<u32>(),
      );
    }
  }

  /// Allocate a segment header plus `count` contiguous values of type `U`.
  ///
  /// Returns the position of the segment header. The segment's `len` field
  /// is initialized to `count`; `next` is 0 (end of chain, from zero-fill).
  pub(crate) fn alloc_segment_internal<U: Flat>(&mut self, count: u32) -> Pos {
    self.buf.align_to(align_of::<Segment<U>>());
    let pos = Pos(self.buf.len());
    let values_size =
      count.checked_mul(mem::size_of::<U>() as u32).expect("segment values overflow");
    let total = (mem::size_of::<Segment<U>>() as u32)
      .checked_add(values_size)
      .expect("segment total size overflow");
    self.buf.resize(self.buf.len() + total, 0);
    // Write segment len at offset 4 (next is already 0 from zero-fill).
    let len_offset = pos.0 as usize + mem::size_of::<i32>();
    // SAFETY: `resize` just allocated `total` bytes at `pos`. The `len` field
    // is at `pos + 4`, within the freshly allocated region.
    unsafe {
      core::ptr::copy_nonoverlapping(
        core::ptr::from_ref(&count).cast::<u8>(),
        self.buf.as_mut_ptr().add(len_offset),
        mem::size_of::<u32>(),
      );
    }
    pos
  }

  /// Patch the `next` pointer of a segment at `seg_pos`.
  ///
  /// # Safety
  ///
  /// `seg_pos` must be a position of a previously allocated `Segment<T>`.
  pub(crate) unsafe fn patch_segment_next_internal(&mut self, seg_pos: Pos, next_seg_pos: Pos) {
    // next field is at offset 0 of Segment<T>
    let rel = i64::from(next_seg_pos.0) - i64::from(seg_pos.0);
    let rel_i32: i32 = rel.try_into().expect("segment next offset overflow");
    let start = seg_pos.0 as usize;
    assert!(
      start + mem::size_of::<i32>() <= self.buf.len() as usize,
      "patch_segment_next out of bounds"
    );
    // SAFETY: Bounds checked above. The `next` field is at offset 0 of
    // `Segment<T>`, and we write exactly `size_of::<i32>()` bytes.
    unsafe {
      core::ptr::copy_nonoverlapping(
        core::ptr::from_ref(&rel_i32).cast::<u8>(),
        self.buf.as_mut_ptr().add(start),
        mem::size_of::<i32>(),
      );
    }
  }

  /// Copy raw bytes to position `at`.
  ///
  /// # Safety
  ///
  /// `src` must be valid for reading `len` bytes. `at` must be a valid
  /// position with at least `len` bytes available.
  pub(crate) unsafe fn write_bytes_internal(&mut self, at: Pos, src: *const u8, len: usize) {
    let start = at.0 as usize;
    assert!(
      start + len <= self.buf.len() as usize,
      "write_bytes out of bounds: {}..{} but len is {}",
      start,
      start + len,
      self.buf.len()
    );
    // SAFETY: Bounds checked above. `src` is valid for `len` bytes (caller
    // contract). The destination does not overlap the source.
    unsafe {
      core::ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr().add(start), len);
    }
  }

  /// Return the raw byte contents of this region.
  ///
  /// The returned slice can be persisted (e.g. written to a file) and later
  /// restored via [`from_bytes`](Self::from_bytes).
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node {
  ///   id: u32,
  ///   children: NearList<u32>,
  /// }
  ///
  /// let region = Region::new(Node::make(1, [10u32, 20, 30]));
  /// let bytes = region.as_bytes();
  /// assert!(bytes.len() >= core::mem::size_of::<Node>());
  /// ```
  #[must_use]
  pub fn as_bytes(&self) -> &[u8] {
    self.buf.as_bytes()
  }

  /// Compute a CRC32 checksum of the region's byte contents.
  ///
  /// Returns the CRC-32/ISO-HDLC checksum of [`as_bytes()`](Self::as_bytes).
  /// Useful for verifying data integrity after persistence or transmission.
  ///
  /// The checksum is computed over the raw buffer, so two `Region`s with
  /// identical byte contents always produce the same value. Mutations that
  /// change the buffer (including dead bytes from append-only updates) will
  /// change the checksum; call [`trim`](Self::trim) first if you want a
  /// canonical form.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node { id: u32, items: NearList<u32> }
  ///
  /// let region = Region::new(Node::make(1, [10u32, 20, 30]));
  /// let c1 = region.checksum();
  /// let c2 = region.checksum();
  /// assert_eq!(c1, c2);
  ///
  /// // A clone has the same bytes, so the same checksum.
  /// let cloned = region.clone();
  /// assert_eq!(region.checksum(), cloned.checksum());
  /// ```
  #[cfg(feature = "checksum")]
  #[must_use]
  pub fn checksum(&self) -> u32 {
    crate::crc32::checksum(self.as_bytes())
  }

  /// Validate and reconstruct a region from raw bytes.
  ///
  /// Copies the bytes into an aligned buffer first, then runs
  /// [`T::validate`](Flat::validate) on the copy. This copy-then-validate
  /// order means the input is read only once (during the copy), and
  /// validation reads the cache-hot aligned buffer instead.
  ///
  /// # Errors
  ///
  /// Returns [`ValidateError`](crate::ValidateError) if the bytes do not
  /// form a valid representation of `T` and its transitively reachable data.
  ///
  /// # Examples
  ///
  /// Round-trip through bytes:
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Node {
  ///   id: u32,
  ///   children: NearList<u32>,
  /// }
  ///
  /// let original = Region::new(Node::make(1, [10u32, 20, 30]));
  /// let bytes = original.as_bytes();
  /// let restored: Region<Node> = Region::from_bytes(bytes).unwrap();
  /// assert_eq!(restored.id, 1);
  /// assert_eq!(restored.children.len(), 3);
  /// ```
  ///
  /// Validation catches invalid data — here a `bool` field with value `2`:
  ///
  /// ```
  /// use nearest::{Flat, Near, Region, ValidateError};
  ///
  /// #[derive(Flat, Debug)]
  /// struct Flags {
  ///   active: bool,
  ///   label: Near<u32>,
  /// }
  ///
  /// let region = Region::new(Flags::make(true, 42u32));
  /// let mut bytes = region.as_bytes().to_vec();
  /// // Corrupt the bool — its offset is computed from the struct layout.
  /// let bool_offset = core::mem::offset_of!(Flags, active);
  /// bytes[bool_offset] = 2;
  /// assert!(matches!(
  ///   Region::<Flags>::from_bytes(&bytes),
  ///   Err(ValidateError::InvalidBool { .. })
  /// ));
  /// ```
  pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::ValidateError> {
    let mut buf = B::empty();
    buf.extend_from_slice(bytes);
    T::validate(0, buf.as_bytes())?;
    // SAFETY: `T::validate` just verified the buffer contents are valid.
    Ok(unsafe { Self::from_buf(buf) })
  }

  /// Reconstruct a region from raw bytes **without validation**.
  ///
  /// # Safety
  ///
  /// The caller must guarantee that `bytes` is a valid region buffer for `T`
  /// (e.g. produced by [`as_bytes`](Self::as_bytes) on a valid `Region<T>`).
  /// Passing arbitrary bytes is immediate UB.
  pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> Self {
    let mut buf = B::empty();
    buf.extend_from_slice(bytes);
    // SAFETY: Caller guarantees the bytes form a valid region buffer.
    unsafe { Self::from_buf(buf) }
  }
}

impl<T: Flat, B: Buf> Deref for Region<T, B> {
  type Target = T;

  fn deref(&self) -> &T {
    self.buf.expose_provenance();
    // SAFETY: The buffer is aligned to `align_of::<T>()` and at least
    // `size_of::<T>()` bytes. The root `T` starts at byte 0.
    unsafe { &*self.buf.as_ptr().cast::<T>() }
  }
}

impl<T: Flat, B: Buf + Clone> Clone for Region<T, B> {
  fn clone(&self) -> Self {
    Self { buf: self.buf.clone(), _type: core::marker::PhantomData }
  }
}

impl<T: Flat + fmt::Debug, B: Buf> fmt::Debug for Region<T, B> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Region").field("root", &**self).finish()
  }
}

// SAFETY: Region owns its buffer exclusively via Buf which is Send+Sync.
// The `Send + Sync` bounds on `T` are defense-in-depth: all `Flat` types are
// `Send + Sync` by construction (no heap pointers, no interior mutability),
// but we add the bounds explicitly so the compiler checks this invariant.
unsafe impl<T: Flat + Send + Sync, B: Buf + Send> Send for Region<T, B> {}
// SAFETY: See above — Region owns its buffer exclusively.
unsafe impl<T: Flat + Send + Sync, B: Buf + Sync> Sync for Region<T, B> {}

// ---------------------------------------------------------------------------
// serde support
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
impl<T: Flat, B: Buf> serde::Serialize for Region<T, B> {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(self.as_bytes())
  }
}

#[cfg(feature = "serde")]
impl<'de, T: Flat, B: Buf> serde::Deserialize<'de> for Region<T, B> {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct RegionVisitor<T, B>(core::marker::PhantomData<(T, B)>);

    impl<'de, T: Flat, B: Buf> serde::de::Visitor<'de> for RegionVisitor<T, B> {
      type Value = Region<T, B>;

      fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str("a valid nearest region byte buffer")
      }

      fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Region::from_bytes(v).map_err(E::custom)
      }

      fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut seq: A,
      ) -> Result<Self::Value, A::Error> {
        let mut bytes = alloc::vec::Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(b) = seq.next_element::<u8>()? {
          bytes.push(b);
        }
        self.visit_byte_buf(bytes)
      }
    }

    deserializer.deserialize_byte_buf(RegionVisitor(core::marker::PhantomData))
  }
}
