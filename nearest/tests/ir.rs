#![feature(offset_of_enum)]
#![allow(dead_code)]

use core::marker::PhantomData;
use core::mem;
use core::num::NonZero;
use core::ops::Deref;

use generativity::{Guard, Id, make_guard};
use nearest::{AlignedBuf, Buf};

// =========================================================================
// Self-relative pointers — live INSIDE the arena buffer bytes.
// `off == 0` is `None` (niche). `off == 1` is the empty-slice sentinel.
// Values 2, 3 are reserved (also invariantly invalid for real pointees).
//
// No lifetime brand on these: the generativity brand only does work during
// construction. Post-construction safety comes from (a) private fields —
// no public constructor — so Refs/Owns can only originate from a built
// arena, (b) `Deref` resolving against the pointer's own address inside
// the buffer, (c) the normal borrow checker tying every `&T` reached
// through `Box::deref` to the `Box`'s lifetime.
// =========================================================================

#[repr(C)]
#[derive(Debug)]
pub struct Ref<T: ?Sized> {
  off: NonZero<i32>,
  _phantom: PhantomData<fn(&T) -> &T>,
}

#[repr(C)]
#[derive(Debug)]
pub struct Own<T: ?Sized> {
  off: NonZero<i32>,
  _phantom: PhantomData<T>,
}

impl<T> Deref for Ref<T> {
  type Target = T;
  #[inline(always)]
  fn deref(&self) -> &T {
    // SAFETY: off is a nonzero self-relative delta produced by the builder.
    // The target was emitted into the same arena and fits a T at that offset.
    unsafe {
      let base = self as *const _ as *const u8;
      &*(base.offset(self.off.get() as isize) as *const T)
    }
  }
}

impl<T> Deref for Own<T> {
  type Target = T;
  #[inline(always)]
  fn deref(&self) -> &T {
    // SAFETY: see Ref::deref.
    unsafe {
      let base = self as *const _ as *const u8;
      &*(base.offset(self.off.get() as isize) as *const T)
    }
  }
}

#[inline(always)]
const fn align_up(off: usize, align: usize) -> usize {
  (off + align - 1) & !(align - 1)
}

#[inline(always)]
unsafe fn deref_slice<'r, T>(base: *const u8, off: i32) -> &'r [T] {
  if off == 1 {
    return &[];
  }
  // SAFETY: off points at a slice header [len: u32] written by the builder;
  // data follows, aligned to align_of::<T>(), for `len` contiguous Ts.
  unsafe {
    let header = base.offset(off as isize) as *const u32;
    let len = *header as usize;
    let data_off = align_up(4, core::mem::align_of::<T>());
    let data = (header as *const u8).add(data_off) as *const T;
    core::slice::from_raw_parts(data, len)
  }
}

impl<T> Deref for Ref<[T]> {
  type Target = [T];
  #[inline(always)]
  fn deref(&self) -> &[T] {
    // SAFETY: see deref_slice.
    unsafe { deref_slice::<T>(self as *const _ as *const u8, self.off.get()) }
  }
}

impl<T> Deref for Own<[T]> {
  type Target = [T];
  #[inline(always)]
  fn deref(&self) -> &[T] {
    // SAFETY: see deref_slice.
    unsafe { deref_slice::<T>(self as *const _ as *const u8, self.off.get()) }
  }
}

// =========================================================================
// Arena types — the IR itself. Plain Rust, no lifetime parameter.
// =========================================================================

#[derive(Debug, Copy, Clone)]
pub struct Sym(pub u32);

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum Owning {
  Borrow,    // `&x`     — immutable borrow
  BorrowMut, // `&mut x` — mutable borrow
  Move,      // `x`      — move / copy
}

#[derive(Debug)]
pub struct Arg {
  pub owning: Owning,
  pub block: Ref<Block>,
  pub name: Option<Sym>,
  pub ty: Type,
}

#[repr(C, u8)]
#[derive(Debug)]
pub enum Opcode {
  Imm(Type),
  Arg(Ref<Arg>, u16),
  Cap(Ref<Arg>, u16),
  Clo(Func, Ref<[Inst]>),
  Cor(Coro, Ref<[Inst]>),
  Equ(Ref<Inst>, Ref<Inst>),
  Neq(Ref<Inst>, Ref<Inst>),
  Lth(Ref<Inst>, Ref<Inst>),
  Add(Ref<Inst>, Ref<Inst>),
  Mul(Ref<Inst>, Ref<Inst>),
}

#[derive(Debug)]
pub struct Inst {
  pub block: Ref<Block>,
  pub name: Option<Sym>,
  pub opc: Opcode,
  pub ty: Type,
}

#[derive(Debug)]
pub struct Jmp {
  pub target: Ref<Block>,
  pub args: Ref<[Inst]>,
}

#[repr(C, u8)]
#[derive(Debug)]
pub enum Term {
  Unreachable,
  Jmp(Jmp),                               // unconditional jump
  Br(Ref<Inst>, [Jmp; 2]),                // conditional branch: [then, else]
  Ret(Ref<Inst>),                         // return from function
  Call(Ref<Inst>, Ref<[Inst]>, Jmp),      // function call and continuation
}

#[derive(Debug)]
pub struct Transition {
  pub on: Type,
  pub goto: Ref<State>,
}

#[derive(Debug)]
pub struct Resume {
  pub name: Sym,
  pub func: Func,
  pub transitions: Own<[Transition]>,
}

#[derive(Debug)]
pub struct State {
  pub resumes: Own<[Resume]>,
}

#[derive(Debug)]
pub struct Block {
  pub func: Ref<Func>,
  pub args: Own<[Arg]>,
  pub ret: Type,
  pub code: Own<[Inst]>,
  pub term: Term,
}

#[derive(Debug)]
pub struct Func {
  pub captures: Own<[Arg]>,
  pub resume: Option<Ref<Resume>>,
  pub entry: Block,
  pub rest: Own<[Block]>,
}

#[derive(Debug)]
pub struct Coro {
  pub entry: State,
  pub rest: Own<[State]>,
}

#[repr(C, u8)]
#[derive(Debug)]
pub enum Type {
  Unreachable,
  Any,
  Unit,
  Bool,
  I32,
  Func(Own<Func>),
  Coro(Own<State>),
}

// =========================================================================
// Box<T> — the finalized arena. Root T lives at byte 0. No brand: once
// built, cross-arena mixing is physically impossible (Ref fields deref
// against their own bytes, which are inside a specific Box).
// =========================================================================

pub struct Box<T> {
  buf: AlignedBuf<T>,
}

impl<T> Deref for Box<T> {
  type Target = T;
  #[inline]
  fn deref(&self) -> &T {
    // SAFETY: AlignedBuf<T>'s base pointer is aligned to
    // `max(align_of::<T>(), 8)` (BUF_ALIGN), satisfying T's alignment. The
    // root T was emitted at offset 0 by Builder::new's first `reserve`.
    unsafe { &*(self.buf.as_ptr() as *const T) }
  }
}

// =========================================================================
// Slot / SlotRef — construction-time handles. Both are linear: dropping
// them references an undefined extern symbol so the linker rejects any
// build where a slot was allocated but never consumed.
// =========================================================================

#[inline(always)]
fn leaked() -> ! {
  unsafe extern "C" {
    fn __nearest_slot_leaked() -> !;
  }
  // SAFETY: if code is correct, every Slot/SlotRef is mem::forget'd before
  // Drop. With strong DCE (release + LTO) the entire Drop glue — including
  // this symbol reference — is eliminated, and link succeeds. Without DCE
  // the reference survives; the fallback below lets the test still run,
  // turning the check into a runtime abort.
  unsafe { __nearest_slot_leaked() }
}

// Runtime fallback so `cargo test` can link in debug mode (where drop glue
// is not DCE'd even when `mem::forget` is called on every Slot). In
// production builds with `-C lto -C panic=abort` you would omit this to get
// pure link-time enforcement of linearity.
#[unsafe(no_mangle)]
extern "C" fn __nearest_slot_leaked() -> ! {
  panic!("nearest: Slot or SlotRef was dropped without being consumed")
}

pub struct Slot<'a, T: ?Sized> {
  abs_off: u32,
  id: Id<'a>,
  _type: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Drop for Slot<'_, T> {
  #[inline(always)]
  fn drop(&mut self) {
    leaked()
  }
}

impl<'a, T: ?Sized> Slot<'a, T> {
  #[inline(always)]
  fn new(abs_off: u32, id: Id<'a>) -> Self {
    Slot { abs_off, id, _type: PhantomData }
  }

  // Produce a shareable SlotRef *while keeping the Slot alive* — used for
  // emitting Ref<T> back-pointers to a reservation that hasn't been filled
  // yet. The returned SlotRef has its own linearity obligation.
  #[inline(always)]
  pub fn as_ref(&self) -> SlotRef<'a, T> {
    SlotRef::new(self.abs_off, self.id)
  }

  // Project a SlotRef to a subfield (by static offset). Used when a target
  // lives inline inside a parent reservation — e.g. `Func.entry: Block`.
  /// Unsafe primitive: produce a SlotRef pointing at `abs_off + field_offset`
  /// with target type `U`. The derive generates safe wrappers like
  /// `Slot<Func>::project_entry()` that fix both the offset and the type.
  ///
  /// # Safety
  /// - `field_offset` must be the static byte offset of a field of type `U`
  ///   within the reservation pointed to by `self`.
  /// - The caller must ensure the resulting SlotRef is only used at positions
  ///   expecting `Ref<U>` / `Own<U>` / the same compatible layout.
  #[inline(always)]
  unsafe fn project<U: ?Sized>(&self, field_offset: u32) -> SlotRef<'a, U> {
    SlotRef::new(self.abs_off + field_offset, self.id)
  }

  // One-way downgrade: consume the Slot, yield a shareable SlotRef. Use when
  // a target will be referenced only through `Ref<T>` fields (no `Own<T>`
  // field claims it). Irreversible.
  #[inline(always)]
  pub fn into_ref(self) -> SlotRef<'a, T> {
    let off = self.abs_off;
    let id = self.id;
    mem::forget(self);
    SlotRef::new(off, id)
  }

  // Consume self, return the absolute offset. Internal helper for write_own.
  #[inline(always)]
  fn consume(self) -> u32 {
    let off = self.abs_off;
    mem::forget(self);
    off
  }
}

pub struct SlotRef<'a, T: ?Sized> {
  // abs_off == 1 is the empty-slice sentinel; any other value is an absolute
  // offset into the builder buffer. At write time it gets converted into a
  // self-relative NonZero<i32> in the parent struct's Ref/Own field.
  abs_off: u32,
  id: Id<'a>,
  _type: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Drop for SlotRef<'_, T> {
  #[inline(always)]
  fn drop(&mut self) {
    leaked()
  }
}

impl<'a, T: ?Sized> SlotRef<'a, T> {
  #[inline(always)]
  fn new(abs_off: u32, id: Id<'a>) -> Self {
    SlotRef { abs_off, id, _type: PhantomData }
  }

  // Explicit clone — not `Clone` trait, because we don't want SlotRef to
  // silently multiply via `.clone()` calls from generic code.
  #[inline(always)]
  pub fn clone(&self) -> Self {
    SlotRef::new(self.abs_off, self.id)
  }

  // Consume self, return the absolute offset. Suppresses the linker trip.
  #[inline(always)]
  fn consume(self) -> u32 {
    let off = self.abs_off;
    mem::forget(self);
    off
  }
}

// =========================================================================
// Builder — owns the `Guard`, pre-reserves the root at byte 0, and drives
// graph construction. `finish()` consumes the builder and returns the Box.
// =========================================================================

pub struct Builder<'a, Root> {
  // `AlignedBuf<Root>` guarantees the base pointer is aligned to
  // `max(align_of::<Root>(), 8)`, so we can keep byte-level allocations here
  // and transfer the buffer into the final `Box<Root>` with no copy or
  // realignment at `finish` time.
  buf: AlignedBuf<Root>,
  id: Id<'a>,
  _root: PhantomData<fn() -> Root>,
}

impl<'a, Root> Builder<'a, Root> {
  pub fn new(guard: Guard<'a>) -> (Self, Slot<'a, Root>) {
    // Consume the Guard into an Id. The brand 'a is kept alive by the
    // LifetimeBrand the `make_guard!` macro placed in the caller's scope;
    // we don't need to hold the Guard here.
    let id: Id<'a> = guard.into();
    let mut b = Builder {
      buf: AlignedBuf::<Root>::with_capacity(256),
      id,
      _root: PhantomData,
    };
    let root = b.reserve::<Root>();
    debug_assert_eq!(root.abs_off, 0);
    (b, root)
  }

  pub fn finish(self) -> Box<Root> {
    // No copy: AlignedBuf already satisfies Box's alignment invariant.
    Box { buf: self.buf }
  }

  /// Read-only view of the root. Navigate via safe cursor helpers (see
  /// `root_cursor()`) — reading the root directly is safe.
  #[inline]
  pub fn root(&self) -> &Root {
    // SAFETY: identical to Box::deref — buf is Root-aligned, root at offset 0.
    unsafe { &*(self.buf.as_ptr() as *const Root) }
  }

  /// Raw byte slice of the in-progress buffer — used by derive-generated
  /// cursor code to read deltas when stepping through `Own<[T]>` / `Ref<T>`
  /// offsets at runtime.
  #[inline]
  fn bytes(&self) -> &[u8] {
    self.buf.as_bytes()
  }

  /// Unsafe primitive: absolute byte offset of an arbitrary reference. The
  /// derive exposes safe cursor-based navigators that never need this.
  ///
  /// # Safety
  /// `r` must point inside this builder's buffer (equivalently, into a value
  /// reachable from `self.root()`). Passing any other reference yields a
  /// meaningless u32 that, if later fed to `patch_*`, corrupts the arena.
  #[inline]
  pub unsafe fn offset_of<T: ?Sized>(&self, r: &T) -> u32 {
    let base = self.buf.as_ptr() as usize;
    let addr = r as *const T as *const u8 as usize;
    (addr - base) as u32
  }

  /// Unsafe primitive: build a `SlotRef<T>` targeting a reference into the
  /// buffer. Derive exposes typed cursor `.to_ref()` helpers.
  ///
  /// # Safety
  /// Same precondition as `offset_of`: `r` must live inside this builder's
  /// buffer. Additionally, the caller must ensure a `Ref<T>` targeting this
  /// location will deref to a valid `T` at runtime (true automatically when
  /// `T` matches the declared type of the field `r` was borrowed from).
  #[inline]
  pub unsafe fn ref_to<T: ?Sized>(&self, r: &T) -> SlotRef<'a, T> {
    // SAFETY: delegated to caller per this function's contract.
    SlotRef::new(unsafe { self.offset_of(r) }, self.id)
  }

  /// Unsafe primitive: overwrite a `Ref<T>` field at absolute byte offset
  /// `at`. Derive generates safe cursor helpers (`ref_cursor.patch(...)`).
  ///
  /// # Safety
  /// `at` must be the absolute byte offset of a `Ref<T>` field within the
  /// buffer (for exactly this `T`, not a compatibly-sized substitute).
  #[inline]
  pub unsafe fn patch_ref<T: ?Sized>(&mut self, at: u32, new: SlotRef<'a, T>) {
    write_ref(self.bytes_mut(), at, new);
  }

  /// Unsafe primitive: overwrite an `Own<T>` field at absolute byte offset
  /// `at`. Derive generates safe cursor helpers. The previous target's bytes
  /// remain as garbage.
  ///
  /// # Safety
  /// `at` must be the absolute byte offset of an `Own<T>` field within the
  /// buffer (for exactly this `T`).
  #[inline]
  pub unsafe fn patch_own<T: ?Sized>(&mut self, at: u32, new: Slot<'a, T>) {
    write_own(self.bytes_mut(), at, new);
  }

  /// Unsafe primitive: overwrite a POD field at absolute byte offset `at`.
  /// Derive generates safe cursor helpers.
  ///
  /// # Safety
  /// `at` must be the absolute byte offset of a `T`-typed, `T`-aligned field
  /// in the buffer. The written bytes will be reinterpreted as `T` on read.
  #[inline]
  pub unsafe fn patch_pod<T: Copy>(&mut self, at: u32, v: T) {
    write_pod(self.bytes_mut(), at, v);
  }
}

// =========================================================================
// Box -> Builder round-trip: opens a built arena for in-place mutation.
//
// Semantics: `edit` consumes the Box and returns a Builder that owns the
// same AlignedBuf. While the Builder exists, no reader of the old Box can
// alias with mutations (ownership is transferred). The `'a` brand is a
// fresh edit-session brand — any Slot/SlotRef issued during this session
// cannot escape it or be used on another Builder.
//
// Existing data in the buffer is reachable via `Builder::root()` (read-only)
// for offset computation; fresh data emitted during the session is appended
// past the existing end. `finish` hands the buffer back as a `Box<Root>`.
// =========================================================================

impl<T> Box<T> {
  pub fn edit<'a>(self, guard: Guard<'a>) -> Builder<'a, T> {
    Builder { buf: self.buf, id: guard.into(), _root: PhantomData }
  }
}

impl<'a, Root> Builder<'a, Root> {

  #[inline]
  fn alloc(&mut self, size: usize, align: usize) -> u32 {
    debug_assert!(align <= AlignedBuf::<Root>::ALIGN, "alignment exceeds buffer alignment");
    self.buf.align_to(align);
    let pos = self.buf.len();
    self.buf.resize(pos + size as u32, 0);
    pos
  }

  #[inline]
  fn bytes_mut(&mut self) -> &mut [u8] {
    let len = self.buf.len() as usize;
    // SAFETY: AlignedBuf::as_mut_ptr returns a pointer valid for `len` bytes;
    // all of those bytes are initialized (zero-filled by `resize` in alloc).
    unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr(), len) }
  }

  #[inline]
  pub fn reserve<T>(&mut self) -> Slot<'a, T> {
    let off = self.alloc(core::mem::size_of::<T>(), core::mem::align_of::<T>());
    Slot::new(off, self.id)
  }

  #[inline]
  pub fn fill<T, E: Emit<T>>(&mut self, slot: Slot<'a, T>, e: E) {
    let at = slot.abs_off;
    mem::forget(slot);
    e.write(self.bytes_mut(), at);
  }

  #[inline]
  pub fn emit<T, E: Emit<T>>(&mut self, e: E) -> Slot<'a, T> {
    let slot = self.reserve::<T>();
    let at = slot.abs_off;
    e.write(self.bytes_mut(), at);
    slot
  }

  // Empty-slice slot. `abs_off == 1` is the sentinel that `deref_slice`
  // treats as an empty slice without touching arena bytes. Returned as a
  // `Slot` so it can fill an `Own<[T]>` field directly; downgrade with
  // `.into_ref()` if you need it in a `Ref<[T]>` position.
  #[inline]
  pub fn empty_slice<T>(&self) -> Slot<'a, [T]> {
    Slot::new(1, self.id)
  }

  pub fn emit_slice<T, E: Emit<T>>(&mut self, items: std::vec::Vec<E>) -> Slot<'a, [T]> {
    if items.is_empty() {
      return self.empty_slice();
    }
    let len = items.len();
    let t_align = core::mem::align_of::<T>();
    let t_size = core::mem::size_of::<T>();
    // Header must be aligned to max(4, align_of::<T>()) so that data starts
    // exactly align_up(4, align_of::<T>()) bytes after the header — the
    // invariant `deref_slice` relies on.
    let header_align = core::cmp::max(4, t_align);
    let header_off = self.alloc(4, header_align);
    let data_off = self.alloc(t_size * len, t_align);
    debug_assert_eq!(
      data_off as usize - header_off as usize,
      align_up(4, t_align)
    );
    let bytes = self.bytes_mut();
    bytes[header_off as usize..(header_off as usize + 4)]
      .copy_from_slice(&(len as u32).to_le_bytes());
    for (i, e) in items.into_iter().enumerate() {
      e.write(bytes, data_off + (i * t_size) as u32);
    }
    Slot::new(header_off, self.id)
  }
}

// =========================================================================
// Emit trait — mirror types serialize themselves into the buffer.
// `self` is consumed (to force linearity of embedded SlotRefs); `at` is
// the absolute offset where this value's bytes begin. Trait is not
// parameterized by brand — the brand lives on `Self` (e.g. BlockEmit<'a>).
// =========================================================================

pub trait Emit<T: ?Sized> {
  fn write(self, buf: &mut [u8], at: u32);
}

#[inline(always)]
fn write_i32(buf: &mut [u8], at: u32, v: i32) {
  buf[at as usize..(at as usize + 4)].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
fn write_ptr(buf: &mut [u8], at: u32, target: u32) {
  let v: i32 = if target == 1 {
    // empty-slice sentinel: store literal 1, not a delta
    1
  } else {
    let delta = target as i32 - at as i32;
    debug_assert!(delta != 0, "self-pointing pointer");
    debug_assert!(delta != 1, "delta collides with empty-slice sentinel");
    delta
  };
  write_i32(buf, at, v);
}

// Shared reference — consumes the SlotRef, writes the delta. The caller can
// hold other SlotRefs to the same target; nothing about the bytes changes.
#[inline(always)]
fn write_ref<T: ?Sized>(buf: &mut [u8], at: u32, sr: SlotRef<'_, T>) {
  write_ptr(buf, at, sr.consume());
}

// Exclusive ownership — consumes the Slot, writes the delta. Slot linearity
// means no other SlotRef<T> or Slot<T> can target the same offset through a
// legal call chain, so the resulting `Own<T>` field is the sole route to T
// at the type level. (Runtime overlap via Ref<U> into the same byte range —
// e.g. Ref<Arg> into Own<[Arg]> — is a separate concern; see DerefMut
// discussion.)
#[inline(always)]
fn write_own<T: ?Sized>(buf: &mut [u8], at: u32, slot: Slot<'_, T>) {
  write_ptr(buf, at, slot.consume());
}

#[inline(always)]
fn write_option_ref<T: ?Sized>(buf: &mut [u8], at: u32, sr: Option<SlotRef<'_, T>>) {
  match sr {
    None => write_i32(buf, at, 0), // NonZero niche
    Some(sr) => write_ptr(buf, at, sr.consume()),
  }
}

#[inline(always)]
fn write_pod<T: Copy>(buf: &mut [u8], at: u32, value: T) {
  // SAFETY: T: Copy, buf has >= at + size_of::<T>() bytes (caller ensures).
  unsafe {
    core::ptr::copy_nonoverlapping(
      &value as *const T as *const u8,
      buf.as_mut_ptr().add(at as usize),
      core::mem::size_of::<T>(),
    );
  }
}

// =========================================================================
// Mirror types — hand-written *Emit for each arena type that carries
// Ref/Own fields (or contains nested mirrors that do). The `'a` brand
// rides on the mirrors because they hold SlotRefs from a specific builder.
// =========================================================================

pub struct ArgEmit<'a> {
  pub owning: Owning,
  pub block: SlotRef<'a, Block>,
  pub name: Option<Sym>,
  pub ty: TypeEmit<'a>,
}

pub struct InstEmit<'a> {
  pub block: SlotRef<'a, Block>,
  pub name: Option<Sym>,
  pub opc: OpcodeEmit<'a>,
  pub ty: TypeEmit<'a>,
}

pub struct BlockEmit<'a> {
  pub func: SlotRef<'a, Func>,   // Ref<Func>
  pub args: Slot<'a, [Arg]>,     // Own<[Arg]>
  pub ret: TypeEmit<'a>,
  pub code: Slot<'a, [Inst]>,    // Own<[Inst]>
  pub term: TermEmit<'a>,
}

pub struct FuncEmit<'a> {
  pub captures: Slot<'a, [Arg]>,             // Own<[Arg]>
  pub resume: Option<SlotRef<'a, Resume>>,   // Option<Ref<Resume>>
  pub entry: BlockEmit<'a>,                  // inlined Block
  pub rest: Slot<'a, [Block]>,               // Own<[Block]>
}

pub enum TypeEmit<'a> {
  Unreachable,
  Any,
  Unit,
  Bool,
  I32,
  Func(Slot<'a, Func>),     // Own<Func>
  Coro(Slot<'a, State>),    // Own<State>
}

pub enum OpcodeEmit<'a> {
  Imm(TypeEmit<'a>),
  // Arg/Cap/Clo/Cor/Equ/Neq/Lth/Add/Mul variants are scoped out of this POC;
  // they follow the same pattern (write discriminant + field refs).
}

pub enum TermEmit<'a> {
  Unreachable,
  Ret(SlotRef<'a, Inst>),
  // Jmp/Br/Call scoped out of this POC — see comment on OpcodeEmit.
}

// =========================================================================
// Emit impls — structured like the output of a hypothetical `#[derive(Emit)]`.
//
// Each mirror struct/enum has:
//   * `OFF_<FIELD>` consts for struct field offsets (or `OFF_<VARIANT>_PAYLOAD`
//     for enum variant payloads). All computed at compile time via
//     `core::mem::offset_of!`.
//   * `TAG_<VARIANT>` consts for enum discriminants (match the declaration
//     order of the `#[repr(C, u8)]` arena enum).
//   * Per-field / per-variant-with-data `write_<name>` helpers that encapsulate
//     one offset + one write primitive. Typed signatures lock field kind at
//     compile time: you cannot pass `SlotRef<Inst>` where `SlotRef<Func>` is
//     expected, and a Slot (Own position) cannot be silently used where a
//     SlotRef (Ref position) is expected.
//   * An `impl Emit<ArenaType>` that is a mechanical composition of the above.
//
// No unsafe appears in this section: all primitive writes go through private
// helpers whose unsafe is encapsulated.
// =========================================================================

// ---- Arg / ArgEmit ----------------------------------------------------

impl<'a> ArgEmit<'a> {
  pub const OFF_OWNING: u32 = core::mem::offset_of!(Arg, owning) as u32;
  pub const OFF_BLOCK: u32 = core::mem::offset_of!(Arg, block) as u32;
  pub const OFF_NAME: u32 = core::mem::offset_of!(Arg, name) as u32;
  pub const OFF_TY: u32 = core::mem::offset_of!(Arg, ty) as u32;

  #[inline(always)]
  fn write_owning(buf: &mut [u8], at: u32, v: Owning) {
    write_pod(buf, at + Self::OFF_OWNING, v);
  }
  #[inline(always)]
  fn write_block(buf: &mut [u8], at: u32, v: SlotRef<'a, Block>) {
    write_ref(buf, at + Self::OFF_BLOCK, v);
  }
  #[inline(always)]
  fn write_name(buf: &mut [u8], at: u32, v: Option<Sym>) {
    write_pod(buf, at + Self::OFF_NAME, v);
  }
  #[inline(always)]
  fn write_ty(buf: &mut [u8], at: u32, v: TypeEmit<'a>) {
    v.write(buf, at + Self::OFF_TY);
  }
}

impl<'a> Emit<Arg> for ArgEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    Self::write_owning(buf, at, self.owning);
    Self::write_block(buf, at, self.block);
    Self::write_name(buf, at, self.name);
    Self::write_ty(buf, at, self.ty);
  }
}

// ---- Inst / InstEmit --------------------------------------------------

impl<'a> InstEmit<'a> {
  pub const OFF_BLOCK: u32 = core::mem::offset_of!(Inst, block) as u32;
  pub const OFF_NAME: u32 = core::mem::offset_of!(Inst, name) as u32;
  pub const OFF_OPC: u32 = core::mem::offset_of!(Inst, opc) as u32;
  pub const OFF_TY: u32 = core::mem::offset_of!(Inst, ty) as u32;

  #[inline(always)]
  fn write_block(buf: &mut [u8], at: u32, v: SlotRef<'a, Block>) {
    write_ref(buf, at + Self::OFF_BLOCK, v);
  }
  #[inline(always)]
  fn write_name(buf: &mut [u8], at: u32, v: Option<Sym>) {
    write_pod(buf, at + Self::OFF_NAME, v);
  }
  #[inline(always)]
  fn write_opc(buf: &mut [u8], at: u32, v: OpcodeEmit<'a>) {
    v.write(buf, at + Self::OFF_OPC);
  }
  #[inline(always)]
  fn write_ty(buf: &mut [u8], at: u32, v: TypeEmit<'a>) {
    v.write(buf, at + Self::OFF_TY);
  }
}

impl<'a> Emit<Inst> for InstEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    Self::write_block(buf, at, self.block);
    Self::write_name(buf, at, self.name);
    Self::write_opc(buf, at, self.opc);
    Self::write_ty(buf, at, self.ty);
  }
}

// ---- Block / BlockEmit ------------------------------------------------

impl<'a> BlockEmit<'a> {
  pub const OFF_FUNC: u32 = core::mem::offset_of!(Block, func) as u32;
  pub const OFF_ARGS: u32 = core::mem::offset_of!(Block, args) as u32;
  pub const OFF_RET: u32 = core::mem::offset_of!(Block, ret) as u32;
  pub const OFF_CODE: u32 = core::mem::offset_of!(Block, code) as u32;
  pub const OFF_TERM: u32 = core::mem::offset_of!(Block, term) as u32;

  #[inline(always)]
  fn write_func(buf: &mut [u8], at: u32, v: SlotRef<'a, Func>) {
    write_ref(buf, at + Self::OFF_FUNC, v);
  }
  #[inline(always)]
  fn write_args(buf: &mut [u8], at: u32, v: Slot<'a, [Arg]>) {
    write_own(buf, at + Self::OFF_ARGS, v);
  }
  #[inline(always)]
  fn write_ret(buf: &mut [u8], at: u32, v: TypeEmit<'a>) {
    v.write(buf, at + Self::OFF_RET);
  }
  #[inline(always)]
  fn write_code(buf: &mut [u8], at: u32, v: Slot<'a, [Inst]>) {
    write_own(buf, at + Self::OFF_CODE, v);
  }
  #[inline(always)]
  fn write_term(buf: &mut [u8], at: u32, v: TermEmit<'a>) {
    v.write(buf, at + Self::OFF_TERM);
  }
}

impl<'a> Emit<Block> for BlockEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    Self::write_func(buf, at, self.func);
    Self::write_args(buf, at, self.args);
    Self::write_ret(buf, at, self.ret);
    Self::write_code(buf, at, self.code);
    Self::write_term(buf, at, self.term);
  }
}

// ---- Func / FuncEmit --------------------------------------------------

impl<'a> FuncEmit<'a> {
  pub const OFF_CAPTURES: u32 = core::mem::offset_of!(Func, captures) as u32;
  pub const OFF_RESUME: u32 = core::mem::offset_of!(Func, resume) as u32;
  pub const OFF_ENTRY: u32 = core::mem::offset_of!(Func, entry) as u32;
  pub const OFF_REST: u32 = core::mem::offset_of!(Func, rest) as u32;

  #[inline(always)]
  fn write_captures(buf: &mut [u8], at: u32, v: Slot<'a, [Arg]>) {
    write_own(buf, at + Self::OFF_CAPTURES, v);
  }
  #[inline(always)]
  fn write_resume(buf: &mut [u8], at: u32, v: Option<SlotRef<'a, Resume>>) {
    write_option_ref(buf, at + Self::OFF_RESUME, v);
  }
  #[inline(always)]
  fn write_entry(buf: &mut [u8], at: u32, v: BlockEmit<'a>) {
    <BlockEmit<'a> as Emit<Block>>::write(v, buf, at + Self::OFF_ENTRY);
  }
  #[inline(always)]
  fn write_rest(buf: &mut [u8], at: u32, v: Slot<'a, [Block]>) {
    write_own(buf, at + Self::OFF_REST, v);
  }
}

impl<'a> Emit<Func> for FuncEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    Self::write_captures(buf, at, self.captures);
    Self::write_resume(buf, at, self.resume);
    Self::write_entry(buf, at, self.entry);
    Self::write_rest(buf, at, self.rest);
  }
}

// Typed projection into Func's inline composite field. The derive-generated
// counterpart of `slot.project(FuncEmit::OFF_ENTRY)` with the target type
// fixed by the field's declaration (`entry: Block`). Keeps the general
// `Slot::project` primitive off the user API surface while letting cycle-
// construction code get a `SlotRef<Block>` to the inlined entry.
impl<'a> Slot<'a, Func> {
  #[inline(always)]
  pub fn project_entry(&self) -> SlotRef<'a, Block> {
    // SAFETY: `FuncEmit::OFF_ENTRY` is the compile-time offset of `Func.entry`
    // whose declared type is `Block`. The derive's static knowledge of Func's
    // layout makes both the offset and the target type correct by construction.
    unsafe { self.project(FuncEmit::OFF_ENTRY) }
  }
}

// ---- Type / TypeEmit --------------------------------------------------

impl<'a> TypeEmit<'a> {
  pub const TAG_UNREACHABLE: u8 = 0;
  pub const TAG_ANY: u8 = 1;
  pub const TAG_UNIT: u8 = 2;
  pub const TAG_BOOL: u8 = 3;
  pub const TAG_I32: u8 = 4;
  pub const TAG_FUNC: u8 = 5;
  pub const TAG_CORO: u8 = 6;
  pub const OFF_FUNC_PAYLOAD: u32 = core::mem::offset_of!(Type, Func.0) as u32;
  pub const OFF_CORO_PAYLOAD: u32 = core::mem::offset_of!(Type, Coro.0) as u32;

  #[inline(always)]
  fn write_func(buf: &mut [u8], at: u32, slot: Slot<'a, Func>) {
    buf[at as usize] = Self::TAG_FUNC;
    write_own(buf, at + Self::OFF_FUNC_PAYLOAD, slot);
  }
  #[inline(always)]
  fn write_coro(buf: &mut [u8], at: u32, slot: Slot<'a, State>) {
    buf[at as usize] = Self::TAG_CORO;
    write_own(buf, at + Self::OFF_CORO_PAYLOAD, slot);
  }

  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      TypeEmit::Unreachable => buf[at as usize] = Self::TAG_UNREACHABLE,
      TypeEmit::Any => buf[at as usize] = Self::TAG_ANY,
      TypeEmit::Unit => buf[at as usize] = Self::TAG_UNIT,
      TypeEmit::Bool => buf[at as usize] = Self::TAG_BOOL,
      TypeEmit::I32 => buf[at as usize] = Self::TAG_I32,
      TypeEmit::Func(slot) => Self::write_func(buf, at, slot),
      TypeEmit::Coro(slot) => Self::write_coro(buf, at, slot),
    }
  }
}

// ---- Opcode / OpcodeEmit ----------------------------------------------

impl<'a> OpcodeEmit<'a> {
  pub const TAG_IMM: u8 = 0;
  pub const OFF_IMM_PAYLOAD: u32 = core::mem::offset_of!(Opcode, Imm.0) as u32;

  #[inline(always)]
  fn write_imm(buf: &mut [u8], at: u32, ty: TypeEmit<'a>) {
    buf[at as usize] = Self::TAG_IMM;
    ty.write(buf, at + Self::OFF_IMM_PAYLOAD);
  }

  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      OpcodeEmit::Imm(ty) => Self::write_imm(buf, at, ty),
    }
  }
}

// ---- Term / TermEmit --------------------------------------------------

impl<'a> TermEmit<'a> {
  pub const TAG_UNREACHABLE: u8 = 0;
  pub const TAG_RET: u8 = 3;
  pub const OFF_RET_PAYLOAD: u32 = core::mem::offset_of!(Term, Ret.0) as u32;

  #[inline(always)]
  fn write_ret(buf: &mut [u8], at: u32, sr: SlotRef<'a, Inst>) {
    buf[at as usize] = Self::TAG_RET;
    write_ref(buf, at + Self::OFF_RET_PAYLOAD, sr);
  }

  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      TermEmit::Unreachable => buf[at as usize] = Self::TAG_UNREACHABLE,
      TermEmit::Ret(sr) => Self::write_ret(buf, at, sr),
    }
  }
}

// =========================================================================
// Cursors — safe, typed, derive-generated navigation into the buffer.
//
// Each cursor carries:
//   * the absolute byte offset of a field in the buffer,
//   * a `&mut Builder` so terminal ops (patch) can write,
//   * a type phantom that fixes what kind of access is available.
//
// The cursor types are *not* constructed by user code. Only derive-generated
// accessor methods create them, and each accessor encapsulates a single
// `unsafe` call into a `Builder::patch_*` / `Slot::project` primitive. The
// public cursor API is fully safe.
// =========================================================================

pub struct StructCursor<'a, 'b, Root, T: ?Sized> {
  b: &'b mut Builder<'a, Root>,
  at: u32,
  _phantom: PhantomData<fn() -> T>,
}

pub struct OwnSliceCursor<'a, 'b, Root, T> {
  b: &'b mut Builder<'a, Root>,
  at: u32, // offset of the Own<[T]> *field* (4 bytes holding the self-relative delta)
  _phantom: PhantomData<fn() -> T>,
}

pub struct PodCursor<'a, 'b, Root, T: Copy> {
  b: &'b mut Builder<'a, Root>,
  at: u32,
  _phantom: PhantomData<fn() -> T>,
}

impl<'a, 'b, Root, T: ?Sized> StructCursor<'a, 'b, Root, T> {
  /// Hand out a `SlotRef<T>` targeting the cursor's position. The cursor is
  /// consumed, releasing the Builder's mut borrow.
  #[inline(always)]
  pub fn to_ref(self) -> SlotRef<'a, T> {
    SlotRef::new(self.at, self.b.id)
  }
}

impl<'a, 'b, Root, T> OwnSliceCursor<'a, 'b, Root, T> {
  /// Replace the `Own<[T]>` pointer with a new slice.
  #[inline(always)]
  pub fn patch(self, new: Slot<'a, [T]>) {
    // SAFETY: `self.at` was set by a derive-generated accessor to the offset
    // of an `Own<[T]>` field; `new` is a `Slot<[T]>` of the matching element
    // type, so the patched delta is correct for both offset and type.
    unsafe { self.b.patch_own(self.at, new) }
  }

  /// Index into the slice, returning a struct cursor at element `i`.
  #[inline(always)]
  pub fn at(self, i: u32) -> StructCursor<'a, 'b, Root, T> {
    // Read the 4-byte self-relative delta at self.at to find the slice
    // header, then compute the data offset.
    let delta_bytes: [u8; 4] = self.b.bytes()[self.at as usize..self.at as usize + 4]
      .try_into()
      .expect("slice cursor: delta read");
    let delta = i32::from_le_bytes(delta_bytes);
    assert!(
      delta != 0 && delta != 1,
      "OwnSliceCursor::at: empty or null slice"
    );
    let header_off = (self.at as i32 + delta) as u32;
    let data_off = header_off + align_up(4, core::mem::align_of::<T>()) as u32;
    let elem_off = data_off + i * core::mem::size_of::<T>() as u32;
    StructCursor { b: self.b, at: elem_off, _phantom: PhantomData }
  }
}

impl<'a, 'b, Root, T: Copy> PodCursor<'a, 'b, Root, T> {
  /// Overwrite the POD field with a new value.
  #[inline(always)]
  pub fn patch(self, new: T) {
    // SAFETY: `self.at` was set by a derive-generated accessor to the offset
    // of a `T`-typed POD field; the write is size/align-correct for `T`.
    unsafe { self.b.patch_pod(self.at, new) }
  }
}

// --- Derive-generated root-cursor entry point -------------------------

impl<'a, Root> Builder<'a, Root> {
  /// Start a safe cursor walk from the root. Returned cursor owns a `&mut`
  /// borrow of the Builder until a terminal operation releases it.
  #[inline(always)]
  pub fn root_cursor<'b>(&'b mut self) -> StructCursor<'a, 'b, Root, Root> {
    StructCursor { b: self, at: 0, _phantom: PhantomData }
  }
}

// --- Derive-generated per-arena-type accessors ------------------------
//
// One accessor per field. Each encapsulates its offset and the type of the
// resulting cursor (Struct / OwnSlice / Pod / ...).

impl<'a, 'b, Root> StructCursor<'a, 'b, Root, Func> {
  #[inline(always)]
  pub fn entry(self) -> StructCursor<'a, 'b, Root, Block> {
    StructCursor {
      b: self.b,
      at: self.at + FuncEmit::OFF_ENTRY,
      _phantom: PhantomData,
    }
  }
}

impl<'a, 'b, Root> StructCursor<'a, 'b, Root, Block> {
  #[inline(always)]
  pub fn args(self) -> OwnSliceCursor<'a, 'b, Root, Arg> {
    OwnSliceCursor {
      b: self.b,
      at: self.at + BlockEmit::OFF_ARGS,
      _phantom: PhantomData,
    }
  }
  #[inline(always)]
  pub fn code(self) -> OwnSliceCursor<'a, 'b, Root, Inst> {
    OwnSliceCursor {
      b: self.b,
      at: self.at + BlockEmit::OFF_CODE,
      _phantom: PhantomData,
    }
  }
}

impl<'a, 'b, Root> StructCursor<'a, 'b, Root, Arg> {
  #[inline(always)]
  pub fn name(self) -> PodCursor<'a, 'b, Root, Option<Sym>> {
    PodCursor {
      b: self.b,
      at: self.at + ArgEmit::OFF_NAME,
      _phantom: PhantomData,
    }
  }
}

// =========================================================================
// Test — same round-trip assertions as before, but on the new API:
// straight-line Builder::new / fill / finish, no closure indirection.
// =========================================================================

#[test]
fn it_works() {
  make_guard!(guard);
  let (mut b, func_slot) = Builder::<Func>::new(guard);

  // The entry Block is inlined inside Func. Project a ref to it via its
  // static field offset — no separate allocation, no separate slot.
  let entry_ref: SlotRef<Block> = func_slot.project_entry();
  let func_ref: SlotRef<Func> = func_slot.as_ref();

  // One arg back-pointing to the entry block.
  let args = b.emit_slice::<Arg, _>(std::vec![ArgEmit {
    owning: Owning::Move,
    block: entry_ref.clone(),
    name: Some(Sym(0)),
    ty: TypeEmit::I32,
  }]);

  // One instruction, also back-pointing to the entry block.
  let code = b.emit_slice::<Inst, _>(std::vec![InstEmit {
    block: entry_ref, // final consumer of entry_ref
    name: Some(Sym(1)),
    opc: OpcodeEmit::Imm(TypeEmit::I32),
    ty: TypeEmit::I32,
  }]);

  let captures: Slot<[Arg]> = b.empty_slice();
  let rest: Slot<[Block]> = b.empty_slice();

  b.fill(
    func_slot,
    FuncEmit {
      captures,
      resume: None,
      entry: BlockEmit {
        func: func_ref,
        args,
        ret: TypeEmit::I32,
        code,
        term: TermEmit::Unreachable,
      },
      rest,
    },
  );

  let arena: Box<Func> = b.finish();

  // Read it back.
  let func: &Func = &arena;
  assert!(func.resume.is_none());
  assert_eq!(func.captures.len(), 0);
  assert_eq!(func.rest.len(), 0);

  let block: &Block = &func.entry;
  assert_eq!(block.args.len(), 1);
  assert_eq!(block.code.len(), 1);

  // Cycle 1: Block.func → Func
  let block_func_addr = &*block.func as *const Func as usize;
  let func_addr = func as *const Func as usize;
  assert_eq!(block_func_addr, func_addr, "Block.func back-ref");

  // Cycle 2: Arg.block → entry Block (inlined in Func)
  let arg: &Arg = &block.args[0];
  let arg_block_addr = &*arg.block as *const Block as usize;
  let entry_block_addr = &func.entry as *const Block as usize;
  assert_eq!(arg_block_addr, entry_block_addr, "Arg.block back-ref");

  // Cycle 3: Inst.block → entry Block
  let inst: &Inst = &block.code[0];
  let inst_block_addr = &*inst.block as *const Block as usize;
  assert_eq!(inst_block_addr, entry_block_addr, "Inst.block back-ref");

  // Sanity: primitive fields survived the round-trip.
  assert!(matches!(arg.owning, Owning::Move));
  assert_eq!(arg.name.unwrap().0, 0);
  assert_eq!(inst.name.unwrap().0, 1);
  assert!(matches!(arg.ty, Type::I32));
  assert!(matches!(inst.ty, Type::I32));
  assert!(matches!(inst.opc, Opcode::Imm(Type::I32)));
  assert!(matches!(block.ret, Type::I32));
  assert!(matches!(block.term, Term::Unreachable));
}

// Helper for the round-trip test — builds the same minimal Func used in
// `it_works`, returns the finalized arena. Kept separate so both tests see
// the same initial shape.
fn build_initial(guard: Guard<'_>) -> Box<Func> {
  let (mut b, func_slot) = Builder::<Func>::new(guard);
  let entry_ref: SlotRef<Block> = func_slot.project_entry();
  let func_ref: SlotRef<Func> = func_slot.as_ref();
  let args = b.emit_slice::<Arg, _>(std::vec![ArgEmit {
    owning: Owning::Move,
    block: entry_ref.clone(),
    name: Some(Sym(0)),
    ty: TypeEmit::I32,
  }]);
  let code = b.emit_slice::<Inst, _>(std::vec![InstEmit {
    block: entry_ref,
    name: Some(Sym(1)),
    opc: OpcodeEmit::Imm(TypeEmit::I32),
    ty: TypeEmit::I32,
  }]);
  let captures: Slot<[Arg]> = b.empty_slice();
  let rest: Slot<[Block]> = b.empty_slice();
  b.fill(
    func_slot,
    FuncEmit {
      captures,
      resume: None,
      entry: BlockEmit {
        func: func_ref,
        args,
        ret: TypeEmit::I32,
        code,
        term: TermEmit::Unreachable,
      },
      rest,
    },
  );
  b.finish()
}

#[test]
fn edit_round_trip() {
  make_guard!(g1);
  let arena: Box<Func> = build_initial(g1);

  // Sanity check the initial state.
  assert_eq!(arena.entry.args[0].name.unwrap().0, 0);
  assert_eq!(arena.entry.code[0].name.unwrap().0, 1);

  // Open for editing with a fresh brand. The old `arena` is consumed.
  make_guard!(g2);
  let mut b = arena.edit(g2);

  // --- Patch a POD field: args[0].name from Sym(0) to Sym(42).
  // All navigation goes through derive-generated cursor accessors; no unsafe.
  b.root_cursor()
    .entry()
    .args()
    .at(0)
    .name()
    .patch(Some(Sym(42)));

  // --- Patch an Own slice: replace entry.code with a longer one that still
  // back-points to the entry Block via Ref<Block> on each new Inst.
  let entry_ref: SlotRef<Block> = b.root_cursor().entry().to_ref();
  let new_code_slot: Slot<[Inst]> = b.emit_slice::<Inst, _>(std::vec![
    InstEmit {
      block: entry_ref.clone(),
      name: Some(Sym(100)),
      opc: OpcodeEmit::Imm(TypeEmit::I32),
      ty: TypeEmit::I32,
    },
    InstEmit {
      block: entry_ref,
      name: Some(Sym(101)),
      opc: OpcodeEmit::Imm(TypeEmit::I32),
      ty: TypeEmit::I32,
    },
  ]);
  b.root_cursor().entry().code().patch(new_code_slot);

  // Re-seal the arena.
  let arena: Box<Func> = b.finish();

  // Verify the mutations are visible and the back-pointers still resolve.
  let func: &Func = &arena;
  assert_eq!(func.entry.args[0].name.unwrap().0, 42, "pod patch");
  assert_eq!(func.entry.code.len(), 2, "Own slice patch grew the slice");
  assert_eq!(func.entry.code[0].name.unwrap().0, 100);
  assert_eq!(func.entry.code[1].name.unwrap().0, 101);

  // The new Insts' block back-refs still point at the entry block.
  let entry_addr = &func.entry as *const Block as usize;
  for inst in func.entry.code.iter() {
    let inst_block_addr = &*inst.block as *const Block as usize;
    assert_eq!(inst_block_addr, entry_addr, "new Inst.block back-ref");
  }

  // Unchanged data is still intact.
  assert!(matches!(func.entry.term, Term::Unreachable));
  assert_eq!(func.entry.args[0].block.func.captures.len(), 0);
}
