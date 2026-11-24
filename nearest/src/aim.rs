//! A pointer as `puns` sees it: an eager aim at its target ([`puns::layout::Aim`]).
//!
//! The layout of a `Near<T, O>` is a struct of one field, its offset, whose rule is an aim at its
//! target, at the address of the pointer plus the offset, minus the bytes of the target before
//! the place where the pointer aims. A check of a value by `puns` follows every pointer and checks
//! the target there, in the bytes that it checks, so a reference that `puns` makes to a value that
//! holds a pointer is a reference to a valid target, which `Deref` reads without a check.
//!
//! The target of an aim holds its own metadata (the aim is thin), so the target of a pointer to a
//! list or a text is a [`Block`], which holds its length:
//!
//! | Pointer | Target | Aims at |
//! |---|---|---|
//! | `Near<T, O>` | `T` | the `T` |
//! | `Near<[T], O>` | `ListBlock<T, O>`: the capacity, the length, the elements | the length |
//! | `Near<str, O>` | `TextBlock<O>`: the length, the UTF-8 bytes | the length |
//!
//! The integers of the offset and of the length are read in the byte order of the `Offset`
//! ([`Offset::ENDIAN`]), which is what `to_isize` reads, and not as the layout of `O` says: an
//! `Offset` of the user's may be a struct whose layout is that of a native integer.

use core::{marker::PhantomData, ptr};

use puns::{
  At, Fields, Path, Pun, RuleAt,
  layout::{Aim, Check, Endian, Field, Id, Layout, Metadata, Op, Rule, Start},
  steps::{One, Two},
};

use crate::{
  Flat, Near, Offset,
  flat::fits_behind,
  near::{Packed, offset},
};

/// The steps of seven operations, `0` to `6`.
type Seven = Two<Two<One<0>, Two<One<1>, One<2>>>, Two<Two<One<3>, One<4>>, Two<One<5>, One<6>>>>;

/// The steps of the start of the target of a pointer ([`start`]): nine operations.
type StartSteps = Two<Seven, Two<One<7>, One<8>>>;

/// The steps of the `len` rule of a [`Block`] ([`len`]): seven operations.
type LenSteps = Seven;

/// The operation that loads the integer of an `O` at `at` in a struct: `size_of::<O>()` bytes in
/// the byte order of `O`.
const fn load<O: Offset>(at: usize) -> Op {
  offset::<O>();
  #[expect(clippy::cast_possible_truncation, reason = "2, 4 or 8 bytes (`offset`)")]
  let size = size_of::<O>() as u8;
  Op::Load { offset: at, size, big: matches!(O::ENDIAN, Endian::Big) }
}

/// Where the target of a pointer of `O` starts, from the address of the pointer: its offset,
/// sign-extended to 128 bits (`off | (off >> (bits - 1)) * (u128::MAX << bits)`, which `puns`
/// reads as an `i128`), minus `lead`, the bytes of the target before the place where the
/// pointer aims. The subtraction fails, so the value is invalid, where the offset is positive and
/// below `lead`: the target would then hold the pointer.
const fn start<O: Offset>(lead: usize) -> [Op; 9] {
  let bits = 8 * size_of::<O>() as u32;
  let off = load::<O>(0);
  [
    off,
    off,
    Op::Lit((bits - 1) as u128),
    Op::Shr,
    Op::Lit(u128::MAX << bits),
    Op::Mul,
    Op::Or,
    Op::Lit(lead as u128),
    Op::Sub,
  ]
}

/// The `len` rule of a block whose length is the `O` at `at`: that integer, whose highest bit is
/// clear, as that of an `O` that is not negative (`len - (len >> (bits - 1)) * 2^bits`, which
/// fails where that bit is set, since a difference below zero fails).
const fn len<O: Offset>(at: usize) -> [Op; 7] {
  let bits = 8 * size_of::<O>() as u32;
  let n = load::<O>(at);
  [n, n, Op::Lit((bits - 1) as u128), Op::Shr, Op::Lit(1 << bits), Op::Mul, Op::Sub]
}

/// What a pointer to a list or a text aims at, as `puns` checks it.
///
/// A `head` (the capacity of a list, and nothing for a text), the length of the `data`, then the
/// data, a slice or a `str`, aligned for it. The pointer aims at the length. The capacity and the
/// length are `O`s of alignment 1 ([`Packed`]), as the image places them: the block is then
/// aligned as its data, and ends where its data ends (`size_of_val`, which `puns` requires of a
/// target, has no padding after the data).
///
/// No value of it is made: it is the type of the target of the aim, whose layout `puns` checks,
/// and whose offsets are those of the image (`ListBlock`, `TextBlock`).
#[repr(C)]
pub struct Block<H, D: ?Sized, O: Copy> {
  head: H,
  len: Packed<O>,
  data: D,
}

/// The block of a list of `T`: its capacity, its length, then the elements.
pub type ListBlock<T, O> = Block<Packed<O>, [T], O>;

/// The block of a text: its length, then the UTF-8 bytes.
pub type TextBlock<O> = Block<(), str, O>;

/// A [`Block`] whose data is `[E]`, or a `str` for `E = u8`, without its data: a sized `repr(C)`
/// struct whose fields are those of the block, the data an empty array of the elements. The
/// fields of a `repr(C)` struct are in the order of their declarations, each at the first offset
/// after the one before that is aligned for it, sized or not, and `[E; 0]` and `[E]` are aligned
/// as `E` ([`r-layout.repr.c.struct`]). So the offsets of its fields are those of the block, which
/// `offset_of!` gives without a `Pun` type.
///
/// [`r-layout.repr.c.struct`]: https://doc.rust-lang.org/1.99.0/reference/type-layout.html#r-layout.repr.c.struct
#[repr(C)]
struct Shape<H, E, O: Copy> {
  head: H,
  len: Packed<O>,
  data: [E; 0],
}

/// The offsets of a block, and its `Pun`, for a list and for a text.
macro_rules! blocks {
  ($([$($any:tt)*] [$($generics:tt)*] $data:ty, $head:ty, $elem:ty;)+) => {$(
    impl<$($any)*> Block<$head, $data, O> {
      /// The offset of the length, where the pointer aims.
      pub const LEN: usize = core::mem::offset_of!(Shape<$head, $elem, O>, len);
      /// The offset of the data.
      pub const DATA: usize = core::mem::offset_of!(Shape<$head, $elem, O>, data);
      /// The bytes from the length to the data.
      pub const SKIP: usize = Self::DATA - Self::LEN;
    }

    // SAFETY: The layout is that of a `repr(C)` struct of the fields `head`, `len` and `data`, at
    // the offsets of `Shape` (0 for the first field), which are those of the block, each with the
    // layout of its type. `Layout::structure` takes the size and the alignment from the compiler,
    // and checks that the fields do not overlap and that the struct has the size that the layout
    // computes. A block is valid if its fields are and if its `len` rule holds: the length of
    // `data` is the integer of `len`, read as `Offset::to_isize` reads it (the contract of
    // `Offset`), and not negative. `Fields` is the list `BlockHead`, `BlockLen`, `BlockData`,
    // `Rules` the `len` rule, and `Metadata` that of the data, a slice or a `str`, since its tail
    // is that of the block. No byte is in an `UnsafeCell`, since the fields are `Pun`. No value of
    // it is made: safe code cannot write one.
    unsafe impl<$($generics)*> Pun for Block<$head, $data, O> {
      type Elem = ();
      type Fields = BlockHead<$head, $data, O>;
      type Variants = ();
      type Rules = RuleAt<0, LenSteps, ()>;
      type Metadata = <$data as Pun>::Metadata;
      const PROBE: *const Self = <$data as Pun>::PROBE as *const Self;
      const ID: Id = Id::new(
        puns::__site!(Block, 0),
        const { &[<$head as Pun>::ID, <$data as Pun>::ID, <O as Pun>::ID] },
      );
      // SAFETY: `Block` is a struct whose last field is a slice or a `str`, and `Self::PROBE` is
      // the probe of the data cast with an `as` cast, as `Layout::structure` requires.
      const LAYOUT: Layout = unsafe {
        Layout::structure::<Self>(
          Self::PROBE,
          const {
            &Layout::sort_fields([
              Field::new(0, &<$head as Pun>::LAYOUT),
              Field::new(Self::LEN, &<Packed<O> as Pun>::LAYOUT),
              Field::new(Self::DATA, &<$data as Pun>::LAYOUT),
            ])
          },
          const { &[Rule::Len(const { &len::<O>(Self::LEN) })] },
        )
        .identified(Self::ID)
      };
    }
  )+};
}

blocks! {
  [T, O: Offset] [T: Flat, O: Offset] [T], Packed<O>, T;
  [O: Offset] [O: Offset] str, (), u8;
}

/// The field `head` of a [`Block`], then `BlockLen`.
#[doc(hidden)]
pub struct BlockHead<H, D: ?Sized, O: Copy>(PhantomData<(H, O)>, PhantomData<D>);

/// The field `len` of a [`Block`], then `BlockData`.
#[doc(hidden)]
pub struct BlockLen<H, D: ?Sized, O: Copy>(PhantomData<(H, O)>, PhantomData<D>);

/// The field `data` of a [`Block`], the last one.
#[doc(hidden)]
pub struct BlockData<H, D: ?Sized, O: Copy>(PhantomData<(H, O)>, PhantomData<D>);

// SAFETY: `FIELD` is the field of the layout of `Block` declared first, `head`, of type `H`, and
// `Next` the one declared after it. So the list is the fields of the layout in the order of their
// declarations.
unsafe impl<H: Pun, D: ?Sized + Pun, O: Offset> Fields for BlockHead<H, D, O>
where
  Block<H, D, O>: Pun,
{
  type Type = H;
  type Next = BlockLen<H, D, O>;
  type Last = BlockData<H, D, O>;
  type Path<P: Path> = At<P, Self>;
  const FIELD: Option<Field> = Some(*<Block<H, D, O> as Pun>::LAYOUT.field(0));
}

// SAFETY: `FIELD` is the second field of the layout of `Block`, `len`, of type `Packed<O>`, and
// `Next` the one declared after it.
unsafe impl<H: Pun, D: ?Sized + Pun, O: Offset> Fields for BlockLen<H, D, O>
where
  Block<H, D, O>: Pun,
{
  type Type = Packed<O>;
  type Next = BlockData<H, D, O>;
  type Last = BlockData<H, D, O>;
  type Path<P: Path> = At<P, Self>;
  const FIELD: Option<Field> = Some(*<Block<H, D, O> as Pun>::LAYOUT.field(1));
}

// SAFETY: `FIELD` is the field `data` of the layout of `Block`, of type `D`, declared last, and
// `Next` is the end of the list.
unsafe impl<H: Pun, D: ?Sized + Pun, O: Offset> Fields for BlockData<H, D, O>
where
  Block<H, D, O>: Pun,
{
  type Type = D;
  type Next = ();
  type Last = Self;
  type Path<P: Path> = At<P, Self>;
  const FIELD: Option<Field> = Some(*<Block<H, D, O> as Pun>::LAYOUT.field(2));
}

/// The field `off` of a `Near`, its only field with bytes.
#[doc(hidden)]
pub struct NearOff<T: ?Sized, O>(PhantomData<O>, PhantomData<T>);

/// The `Pun` of the three pointers, each with its target and the bytes of the target before the
/// place where it aims.
macro_rules! pointers {
  ($([$($generics:tt)*] $t:ty => $target:ty, $lead:expr;)+) => {$(
    // SAFETY: `FIELD` is the one field of the layout of `Near`, `off`, of type `O::NonZero`, and
    // the list ends after it. The other fields have no bytes and no invariant, and are not in the
    // layout.
    unsafe impl<$($generics)*> Fields for NearOff<$t, O> {
      type Type = O::NonZero;
      type Next = ();
      type Last = Self;
      type Path<P: Path> = At<P, Self>;
      const FIELD: Option<Field> = Some(*<Near<$t, O> as Pun>::LAYOUT.field(0));
    }

    // SAFETY: `Near` is `repr(transparent)` over its `O::NonZero`, whose layout is the one field,
    // at 0, so it has its size and its alignment; `PhantomData` and `PhantomPinned` have no bytes
    // and no invariant. The rule is an eager, thin aim at the target, which `Rules` names, where
    // the offset says, minus the bytes of the target before the place where the pointer aims. A
    // `Near` has no constructor and is neither `Clone` nor `Copy`, and `puns` makes no value of
    // it and no `&mut` to it (an eager aim), so one exists only in a region, which made it or
    // checked it with `puns`, or behind a reference that `puns` made once it checked its target
    // in the same bytes: that is what `Deref` reads. `identified` changes neither its size, its
    // alignment nor its values. `Option` of it
    // is null-optimized where it is for `O::NonZero` (a `NonZero` integer, or a
    // `repr(transparent)` struct around one), since `Near` is `repr(transparent)` around it.
    unsafe impl<$($generics)*> Pun for Near<$t, O> {
      type Elem = ();
      type Fields = NearOff<$t, O>;
      type Variants = ();
      type Rules = RuleAt<0, StartSteps, (), $target>;
      type Metadata = ();
      const PROBE: *const Self = ptr::null();
      // SAFETY: `Near` is `Sized`.
      const LAYOUT: Layout = unsafe {
        Layout::structure::<Self>(
          ptr::null(),
          const { &Layout::sort_fields([Field::new(0, &<O::NonZero as Pun>::LAYOUT)]) },
          const {
            &[Rule::Aim(Aim::new::<$target>(
              Start::Stored(const { &start::<O>($lead) }),
              Metadata::Thin,
              Check::Eager,
            ))]
          },
        )
        .identified(Self::ID)
      };
      // Where `Near` is declared, and the identities of the target and of `O`.
      const ID: Id = Id::new(puns::__site!(Near, 0), const { &[<$t as Pun>::ID, <O as Pun>::ID] });
      const NULL_OPTIMIZED: bool = <O::NonZero as Pun>::NULL_OPTIMIZED;
    }
  )+};
}

pointers! {
  [T: Flat, O: Offset] T => T, { fits_behind::<T, O>(); 0 };
  [T: Flat, O: Offset] [T] => ListBlock<T, O>, { fits_behind::<T, O>(); ListBlock::<T, O>::LEN };
  [O: Offset] str => TextBlock<O>, TextBlock::<O>::LEN;
}
