use core::{fmt, marker::PhantomData};

use crate::{Flat, Patch, emitter::Pos};

/// A linked list of `T` values stored in a [`Region`](crate::Region).
///
/// # Layout
///
/// `#[repr(C)]` — 8 bytes, `align_of::<i32>()` alignment:
///
/// | Offset | Field  | Type  | Description                             |
/// |--------|--------|-------|-----------------------------------------|
/// | 0      | `head` | `i32` | Self-relative offset to first segment   |
/// | 4      | `len`  | `u32` | TOTAL element count across all segments |
///
/// When `len == 0`, `head` is `0` (no valid segment). When non-empty, `head` is
/// a self-relative offset from the `head` field itself to the first `Segment<T>`.
///
/// `NearList<T>` is **not** `Copy` or `Clone` — moving it would invalidate the
/// self-relative offset.
///
/// After [`trim`](crate::Region::trim), all elements live in a single contiguous
/// segment for cache-friendly iteration.
///
/// # Segment chain
///
/// A non-empty list consists of one or more `Segment<T>` nodes linked by
/// self-relative `next` pointers (`next == 0` terminates the chain). Each
/// segment stores a contiguous array of `T` values immediately after its
/// header.
///
/// Mutation operations ([`splice_list`], [`push_front`], [`extend_list`]) may
/// produce multi-segment lists. [`trim`](crate::Region::trim) compacts all
/// segments into one, restoring O(1) indexing and cache-friendly iteration.
///
/// # Soundness
///
/// **Provenance**: Segment walking uses `with_exposed_provenance` to
/// follow self-relative `next` pointers, recovering the allocation's
/// provenance exposed by `AlignedBuf::grow`. Same pattern as `Near<T>`.
///
/// **Invariant**: `len == 0` ⟺ `head == 0`. Enforced by all list-patching
/// methods which write both fields atomically.
///
/// [`splice_list`]: crate::Session::splice_list
/// [`push_front`]: crate::Session::push_front
/// [`extend_list`]: crate::Session::extend_list
#[repr(C)]
pub struct NearList<T> {
  head: i32,
  len: u32,
  _type: PhantomData<T>,
}

// SAFETY: NearList contains only i32, u32, and PhantomData — no Drop, no heap.
unsafe impl<T> Flat for NearList<T> {
  unsafe fn deep_copy(&self, p: &mut impl Patch, at: Pos) {
    // SAFETY: Caller guarantees `at` was allocated for `NearList<T>`.
    // Byte-copy the 8-byte header (head offset + len). Containing struct's
    // deep_copy handles walking and deep-copying list elements.
    unsafe {
      p.write_bytes(at, core::ptr::from_ref(self).cast(), size_of::<Self>());
    }
  }

  fn validate(addr: usize, buf: &[u8]) -> Result<(), crate::ValidateError> {
    crate::ValidateError::check::<Self>(addr, buf)?;
    let head = i32::from_ne_bytes(buf[addr..addr + 4].try_into().unwrap());
    let len = u32::from_ne_bytes(buf[addr + 4..addr + 8].try_into().unwrap());
    // Invariant: len == 0 ⟺ head == 0
    if (len == 0) != (head == 0) {
      return Err(crate::ValidateError::InvalidListHeader { addr });
    }
    // Does NOT walk segments — derive code calls __private::validate_list::<T>()
    // for that (mirrors the deep_copy pattern).
    Ok(())
  }
}

/// A contiguous segment of values in a [`NearList`]. Stored in the Region buffer.
///
/// # Layout
///
/// `#[repr(C)]` — 8 bytes header + `len * size_of::<T>()` values:
///
/// | Offset                       | Field     | Type    | Description                          |
/// |------------------------------|-----------|---------|--------------------------------------|
/// | 0                            | `next`    | `i32`   | Self-relative offset to next segment |
/// | 4                            | `len`     | `u32`   | Number of values in this segment     |
/// | `size_of::<Segment<T>>()`    | values... | `[T]`   | Contiguous values                    |
///
/// `next == 0` indicates the end of the segment chain. Values are accessed by
/// pointer arithmetic from the segment address: the first value is at
/// `segment_addr + size_of::<Segment<T>>()`, subsequent values are contiguous
/// at `+ i * size_of::<T>()`.
#[repr(C)]
pub struct Segment<T> {
  pub(crate) next: i32,
  pub(crate) len: u32,
  pub(crate) _values: [T; 0],
}

impl<T: Flat> NearList<T> {
  /// Returns the number of elements.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let region = Region::new(Root::make([1u32, 2, 3]));
  /// assert_eq!(region.items.len(), 3);
  ///
  /// let empty_region = Region::new(Root::make(empty()));
  /// assert_eq!(empty_region.items.len(), 0);
  /// ```
  #[must_use]
  pub const fn len(&self) -> usize {
    self.len as usize
  }

  /// Returns `true` if the list has no elements.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let region = Region::new(Root::make(empty()));
  /// assert!(region.items.is_empty());
  ///
  /// let region = Region::new(Root::make([1u32]));
  /// assert!(!region.items.is_empty());
  /// ```
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Returns the raw head offset (for internal Session use).
  pub(crate) const fn head_offset(&self) -> i32 {
    self.head
  }

  /// Returns the number of segments in the chain.
  ///
  /// After [`trim`](crate::Region::trim), this is always 0 (empty) or 1.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let mut region = Region::new(Root::make([1u32, 2, 3]));
  /// assert_eq!(region.items.segment_count(), 1);
  ///
  /// // push_front adds a new segment.
  /// region.session(|s| {
  ///   let items = s.nav(s.root(), |r| &r.items);
  ///   s.push_front(items, 0u32);
  /// });
  /// assert_eq!(region.items.segment_count(), 2);
  ///
  /// // trim consolidates into one segment.
  /// region.trim();
  /// assert_eq!(region.items.segment_count(), 1);
  /// ```
  #[must_use]
  pub fn segment_count(&self) -> usize {
    if self.len == 0 {
      return 0;
    }
    let mut count = 1usize;
    let mut seg_addr =
      core::ptr::from_ref(&self.head).cast::<u8>().addr().wrapping_add_signed(self.head as isize);
    loop {
      // SAFETY: seg_addr points to a valid Segment<T> in the region buffer.
      let seg = unsafe { &*core::ptr::with_exposed_provenance::<Segment<T>>(seg_addr) };
      if seg.next == 0 {
        break;
      }
      seg_addr =
        core::ptr::from_ref(&seg.next).cast::<u8>().addr().wrapping_add_signed(seg.next as isize);
      count += 1;
    }
    count
  }

  /// Returns a reference to the first element, or `None` if empty.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region, empty};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let region = Region::new(Root::make([10u32, 20]));
  /// assert_eq!(region.items.first(), Some(&10));
  ///
  /// let region = Region::new(Root::make(empty()));
  /// assert_eq!(region.items.first(), None);
  /// ```
  #[must_use]
  pub fn first(&self) -> Option<&T> {
    if self.len == 0 {
      return None;
    }
    // SAFETY: head offset was written by the emitter, pointing to a valid Segment<T>.
    // The first value is at seg_addr + size_of::<Segment<T>>().
    unsafe {
      let addr =
        core::ptr::from_ref(&self.head).cast::<u8>().addr().wrapping_add_signed(self.head as isize);
      let val_ptr =
        core::ptr::with_exposed_provenance::<T>(addr.wrapping_add(size_of::<Segment<T>>()));
      Some(&*val_ptr)
    }
  }

  /// Returns an iterator over the elements.
  ///
  /// Construction is O(1). Within a segment, iteration is contiguous (`ptr.add(1)`).
  /// At segment boundaries, follows `next` pointers to the next segment.
  /// After [`trim`](crate::Region::trim), iteration is pure array traversal.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let region = Region::new(Root::make([10u32, 20, 30]));
  /// let sum: u32 = region.items.iter().sum();
  /// assert_eq!(sum, 60);
  ///
  /// // Also works with for-in (via IntoIterator for &NearList<T>).
  /// let mut count = 0;
  /// for item in &region.items {
  ///   count += 1;
  ///   assert!(*item >= 10);
  /// }
  /// assert_eq!(count, 3);
  /// ```
  #[must_use]
  pub fn iter(&self) -> NearListIter<'_, T> {
    if self.len == 0 {
      return NearListIter {
        current_value: core::ptr::null(),
        remaining_in_seg: 0,
        current_seg: core::ptr::null(),
        remaining_total: 0,
        _type: PhantomData,
      };
    }
    // SAFETY: head offset was written by the emitter, pointing to a valid Segment<T>.
    let seg_addr =
      core::ptr::from_ref(&self.head).cast::<u8>().addr().wrapping_add_signed(self.head as isize);
    let seg_ptr = core::ptr::with_exposed_provenance::<Segment<T>>(seg_addr);
    // SAFETY: `seg_ptr` points to a valid `Segment<T>` in the region buffer,
    // reached via the head offset written by the emitter.
    let seg = unsafe { &*seg_ptr };
    let val_ptr =
      core::ptr::with_exposed_provenance::<T>(seg_addr.wrapping_add(size_of::<Segment<T>>()));
    NearListIter {
      current_value: val_ptr,
      remaining_in_seg: seg.len,
      current_seg: seg_ptr,
      remaining_total: self.len,
      _type: PhantomData,
    }
  }
}

/// Iterator over the elements of a [`NearList`].
///
/// Within a segment, values are contiguous (pointer increment). At segment
/// boundaries, follows the `next` pointer to the next segment.
pub struct NearListIter<'a, T: Flat> {
  current_value: *const T,
  remaining_in_seg: u32,
  current_seg: *const Segment<T>,
  remaining_total: u32,
  _type: PhantomData<&'a T>,
}

impl<'a, T: Flat> Iterator for NearListIter<'a, T> {
  type Item = &'a T;

  fn next(&mut self) -> Option<&'a T> {
    if self.remaining_total == 0 {
      return None;
    }

    // If current segment is exhausted, advance to the next segment.
    if self.remaining_in_seg == 0 {
      // SAFETY: remaining_total > 0 guarantees a next segment exists.
      unsafe {
        let seg = &*self.current_seg;
        let next_addr =
          core::ptr::from_ref(&seg.next).cast::<u8>().addr().wrapping_add_signed(seg.next as isize);
        self.current_seg = core::ptr::with_exposed_provenance::<Segment<T>>(next_addr);
        let new_seg = &*self.current_seg;
        self.remaining_in_seg = new_seg.len;
        self.current_value =
          core::ptr::with_exposed_provenance::<T>(next_addr.wrapping_add(size_of::<Segment<T>>()));
      }
    }

    // SAFETY: current_value points to a valid T within the current segment.
    unsafe {
      let val = &*self.current_value;
      self.remaining_total -= 1;
      self.remaining_in_seg -= 1;
      // Advance to the next value in the segment (contiguous).
      // For ZSTs, add(1) is a no-op on the address, which is correct.
      self.current_value = self.current_value.add(1);
      Some(val)
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let r = self.remaining_total as usize;
    (r, Some(r))
  }
}

impl<T: Flat> ExactSizeIterator for NearListIter<'_, T> {}

impl<'a, T: Flat> IntoIterator for &'a NearList<T> {
  type Item = &'a T;
  type IntoIter = NearListIter<'a, T>;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<T: Flat + fmt::Debug> fmt::Debug for NearList<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_list().entries(self.iter()).finish()
  }
}

impl<T: Flat> NearList<T> {
  /// Returns a reference to the element at `index`, or `None` if out of bounds.
  ///
  /// **O(1)** when all elements are in a single segment (always true after
  /// [`trim`](crate::Region::trim)). Falls back to O(n) segment walk otherwise.
  ///
  /// # Examples
  ///
  /// ```
  /// use nearest::{Flat, NearList, Region};
  ///
  /// #[derive(Flat)]
  /// struct Root { items: NearList<u32> }
  ///
  /// let region = Region::new(Root::make([10u32, 20, 30]));
  /// assert_eq!(region.items.get(0), Some(&10));
  /// assert_eq!(region.items.get(2), Some(&30));
  /// assert_eq!(region.items.get(3), None);
  /// ```
  #[must_use]
  pub fn get(&self, index: usize) -> Option<&T> {
    if index >= self.len as usize {
      return None;
    }
    // SAFETY: head offset was written by the emitter, pointing to a valid Segment<T>.
    unsafe {
      let seg_addr =
        core::ptr::from_ref(&self.head).cast::<u8>().addr().wrapping_add_signed(self.head as isize);
      let seg = &*core::ptr::with_exposed_provenance::<Segment<T>>(seg_addr);
      // Fast path: all elements in first segment (always true after trim).
      if (seg.len as usize) > index {
        let val_addr =
          seg_addr.wrapping_add(size_of::<Segment<T>>()).wrapping_add(index * size_of::<T>());
        return Some(&*core::ptr::with_exposed_provenance::<T>(val_addr));
      }
    }
    // Slow path: multi-segment walk.
    self.iter().nth(index)
  }
}

impl<T: Flat> core::ops::Index<usize> for NearList<T> {
  type Output = T;

  /// Access the element at `index`.
  ///
  /// **O(1)** when all elements are in a single segment (always true after
  /// [`trim`](crate::Region::trim)). Falls back to O(n) segment walk otherwise.
  ///
  /// # Panics
  ///
  /// Panics if `index >= self.len()`.
  fn index(&self, index: usize) -> &T {
    self.get(index).expect("NearList index out of bounds")
  }
}
