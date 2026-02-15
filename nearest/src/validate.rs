use core::fmt;

use crate::{Flat, list::Segment};

/// Error returned by [`Flat::validate`] and [`Region::from_bytes`](crate::Region::from_bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidateError {
  /// Buffer is too small to hold the type at the given address.
  OutOfBounds {
    /// Address of the value in the buffer.
    addr: usize,
    /// Number of bytes needed.
    need: usize,
    /// Actual buffer length.
    buf_len: usize,
  },
  /// Address is not correctly aligned for the type.
  Misaligned {
    /// Address that failed the alignment check.
    addr: usize,
    /// Required alignment.
    align: usize,
  },
  /// A `Near<T>` offset is zero (null pointer).
  NullNear {
    /// Address of the `Near<T>` field.
    addr: usize,
  },
  /// `NearList` header has inconsistent head/len (e.g. non-zero head with zero len).
  InvalidListHeader {
    /// Address of the `NearList` header.
    addr: usize,
  },
  /// Segment lengths don't sum to the total `NearList` length.
  ListLenMismatch {
    /// Address of the `NearList` header.
    addr: usize,
    /// Expected total length from the header.
    expected: u32,
    /// Actual sum of segment lengths.
    actual: u32,
  },
  /// Enum discriminant is not a valid variant.
  InvalidDiscriminant {
    /// Address of the discriminant byte.
    addr: usize,
    /// The discriminant value found.
    value: u8,
    /// Maximum valid discriminant (`variant_count - 1`).
    max: u8,
  },
  /// A `bool` value is not 0 or 1.
  InvalidBool {
    /// Address of the bool.
    addr: usize,
    /// The invalid value found.
    value: u8,
  },
  /// An uninhabited type (`Infallible`) can never be valid.
  Uninhabited,
}

impl fmt::Display for ValidateError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::OutOfBounds { addr, need, buf_len } => {
        write!(f, "out of bounds: addr {addr}, need {need} bytes, buf_len {buf_len}")
      }
      Self::Misaligned { addr, align } => {
        write!(f, "misaligned: addr {addr} not aligned to {align}")
      }
      Self::NullNear { addr } => write!(f, "null Near offset at addr {addr}"),
      Self::InvalidListHeader { addr } => {
        write!(f, "invalid NearList header at addr {addr}")
      }
      Self::ListLenMismatch { addr, expected, actual } => {
        write!(f, "list len mismatch at addr {addr}: expected {expected}, got {actual}")
      }
      Self::InvalidDiscriminant { addr, value, max } => {
        write!(f, "invalid discriminant at addr {addr}: value {value}, max {max}")
      }
      Self::InvalidBool { addr, value } => {
        write!(f, "invalid bool at addr {addr}: value {value}")
      }
      Self::Uninhabited => write!(f, "uninhabited type can never be valid"),
    }
  }
}

impl core::error::Error for ValidateError {}

impl ValidateError {
  /// Check that `buf` has at least `size_of::<T>()` bytes starting at `addr`.
  ///
  /// # Errors
  ///
  /// Returns [`OutOfBounds`](Self::OutOfBounds) if the buffer is too small.
  #[inline]
  pub fn check_bounds<T>(addr: usize, buf: &[u8]) -> Result<(), Self> {
    let need = size_of::<T>();
    if addr.checked_add(need).is_none_or(|end| end > buf.len()) {
      return Err(Self::OutOfBounds { addr, need, buf_len: buf.len() });
    }
    Ok(())
  }

  /// Check that `addr` is aligned for `T`.
  ///
  /// # Errors
  ///
  /// Returns [`Misaligned`](Self::Misaligned) if the address is not aligned.
  #[inline]
  pub const fn check_align<T>(addr: usize) -> Result<(), Self> {
    let align = align_of::<T>();
    if align > 1 && !addr.is_multiple_of(align) {
      return Err(Self::Misaligned { addr, align });
    }
    Ok(())
  }

  /// Combined bounds + alignment check.
  ///
  /// # Errors
  ///
  /// Returns [`OutOfBounds`](Self::OutOfBounds) or [`Misaligned`](Self::Misaligned).
  #[inline]
  pub fn check<T>(addr: usize, buf: &[u8]) -> Result<(), Self> {
    Self::check_bounds::<T>(addr, buf)?;
    Self::check_align::<T>(addr)
  }
}

/// Validate a `NearList<T>` segment chain starting at `hdr_addr`.
///
/// Walks the segment chain, validating each segment header and element,
/// and checks that segment lengths sum to the total list length.
///
/// # Errors
///
/// Returns a [`ValidateError`] if any part of the list structure is invalid.
pub fn validate_list_impl<T: Flat>(hdr_addr: usize, buf: &[u8]) -> Result<(), ValidateError> {
  // Read the NearList header: i32 head + u32 len
  ValidateError::check_bounds::<i32>(hdr_addr, buf)?;
  ValidateError::check_align::<i32>(hdr_addr)?;
  let head = i32::from_ne_bytes(buf[hdr_addr..hdr_addr + 4].try_into().unwrap());
  let total_len = u32::from_ne_bytes(buf[hdr_addr + 4..hdr_addr + 8].try_into().unwrap());

  // Invariant: len == 0 ⟺ head == 0
  if (total_len == 0) != (head == 0) {
    return Err(ValidateError::InvalidListHeader { addr: hdr_addr });
  }

  if total_len == 0 {
    return Ok(());
  }

  let seg_hdr_size = size_of::<Segment<T>>();
  let elem_size = size_of::<T>();
  let max_iters = buf.len().checked_div(seg_hdr_size).unwrap_or(total_len as usize) + 1;
  let mut counted: u32 = 0;
  let mut seg_addr = hdr_addr.cast_signed().wrapping_add(head as isize).cast_unsigned();
  let mut iters = 0usize;

  loop {
    if iters >= max_iters {
      return Err(ValidateError::InvalidListHeader { addr: hdr_addr });
    }
    iters += 1;

    // Validate segment header bounds
    ValidateError::check_bounds::<Segment<T>>(seg_addr, buf)?;
    ValidateError::check_align::<Segment<T>>(seg_addr)?;

    let seg_next = i32::from_ne_bytes(buf[seg_addr..seg_addr + 4].try_into().unwrap());
    let seg_len = u32::from_ne_bytes(buf[seg_addr + 4..seg_addr + 8].try_into().unwrap());

    // Validate values in this segment
    let values_start = seg_addr + seg_hdr_size;
    for i in 0..seg_len as usize {
      let elem_addr = values_start + i * elem_size;
      T::validate(elem_addr, buf)?;
    }

    counted = counted.checked_add(seg_len).ok_or(ValidateError::ListLenMismatch {
      addr: hdr_addr,
      expected: total_len,
      actual: u32::MAX,
    })?;

    // Early cycle/corruption detection
    if counted > total_len {
      return Err(ValidateError::ListLenMismatch {
        addr: hdr_addr,
        expected: total_len,
        actual: counted,
      });
    }

    if seg_next == 0 {
      break;
    }

    seg_addr = seg_addr.cast_signed().wrapping_add(seg_next as isize).cast_unsigned();
  }

  if counted != total_len {
    return Err(ValidateError::ListLenMismatch {
      addr: hdr_addr,
      expected: total_len,
      actual: counted,
    });
  }

  Ok(())
}
