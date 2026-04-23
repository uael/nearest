#![feature(offset_of_enum)]
#![allow(dead_code)]

use core::marker::PhantomData;
use core::mem;
use core::num::NonZero;
use core::ops::Deref;

use generativity::{Guard, Id, make_guard};

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

pub struct Box<T: ?Sized> {
  data: std::vec::Vec<u64>, // u64 backing for guaranteed 8-byte alignment
  _phantom: PhantomData<T>,
}

impl<T> Deref for Box<T> {
  type Target = T;
  #[inline]
  fn deref(&self) -> &T {
    // SAFETY: data is u64-aligned => satisfies any T with align <= 8 (checked
    // by const assertion in Builder::new). Root T was emitted at offset 0.
    unsafe { &*(self.data.as_ptr() as *const T) }
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
  #[inline(always)]
  pub fn project<U: ?Sized>(&self, field_offset: u32) -> SlotRef<'a, U> {
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

pub struct Builder<'a, Root: ?Sized> {
  // u64 backing storage — directly transferred to `Box` at `finish` time
  // with no copy. Allocations are tracked in bytes via `used`; the Vec's
  // length is the number of u64 slots (always `used.div_ceil(8)`).
  data: std::vec::Vec<u64>,
  used: u32,
  id: Id<'a>,
  _root: PhantomData<fn() -> Root>,
}

impl<'a, Root> Builder<'a, Root> {
  pub fn new(guard: Guard<'a>) -> (Self, Slot<'a, Root>) {
    const { assert!(core::mem::align_of::<Root>() <= 8, "root alignment must be <= 8") };
    // Consume the Guard into an Id. The brand 'a is kept alive by the
    // LifetimeBrand the `make_guard!` macro placed in the caller's scope;
    // we don't need to hold the Guard here.
    let id: Id<'a> = guard.into();
    let mut b = Builder {
      data: std::vec::Vec::with_capacity(32), // 32 u64 = 256 bytes initial
      used: 0,
      id,
      _root: PhantomData,
    };
    let root = b.reserve::<Root>();
    debug_assert_eq!(root.abs_off, 0);
    (b, root)
  }

  pub fn finish(self) -> Box<Root> {
    // No copy: the u64-backed storage is already 8-byte aligned. Any trailing
    // padding bytes inside the final u64 slot are unreachable via `Deref`
    // (only `size_of::<Root>()` bytes are read from offset 0).
    Box { data: self.data, _phantom: PhantomData }
  }
}

impl<'a, Root: ?Sized> Builder<'a, Root> {
  #[inline]
  fn alloc(&mut self, size: usize, align: usize) -> u32 {
    debug_assert!(align <= 8, "type alignment > 8 not supported");
    let off = align_up(self.used as usize, align);
    let new_used = off + size;
    let u64_needed = new_used.div_ceil(8);
    if u64_needed > self.data.len() {
      self.data.resize(u64_needed, 0);
    }
    self.used = new_used as u32;
    off as u32
  }

  #[inline]
  fn bytes_mut(&mut self) -> &mut [u8] {
    let len_bytes = self.data.len() * 8;
    // SAFETY: Vec<u64> is contiguous; `len_bytes` covers `data.len()` slots
    // worth of initialized (zeroed) u8s produced by `resize(_, 0)` in alloc.
    unsafe {
      core::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut u8, len_bytes)
    }
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
// Emit impls
// =========================================================================

impl<'a> TypeEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      TypeEmit::Unreachable => buf[at as usize] = 0,
      TypeEmit::Any => buf[at as usize] = 1,
      TypeEmit::Unit => buf[at as usize] = 2,
      TypeEmit::Bool => buf[at as usize] = 3,
      TypeEmit::I32 => buf[at as usize] = 4,
      TypeEmit::Func(slot) => {
        buf[at as usize] = 5;
        let payload = at + core::mem::offset_of!(Type, Func.0) as u32;
        write_own(buf, payload, slot);
      }
      TypeEmit::Coro(slot) => {
        buf[at as usize] = 6;
        let payload = at + core::mem::offset_of!(Type, Coro.0) as u32;
        write_own(buf, payload, slot);
      }
    }
  }
}

impl<'a> OpcodeEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      OpcodeEmit::Imm(ty) => {
        buf[at as usize] = 0;
        ty.write(buf, at + core::mem::offset_of!(Opcode, Imm.0) as u32);
      }
    }
  }
}

impl<'a> TermEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    match self {
      TermEmit::Unreachable => buf[at as usize] = 0,
      TermEmit::Ret(sr) => {
        buf[at as usize] = 3;
        let payload = at + core::mem::offset_of!(Term, Ret.0) as u32;
        write_ref(buf, payload, sr);
      }
    }
  }
}

impl<'a> Emit<Arg> for ArgEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    write_pod(buf, at + core::mem::offset_of!(Arg, owning) as u32, self.owning);
    write_ref(buf, at + core::mem::offset_of!(Arg, block) as u32, self.block);
    write_pod(buf, at + core::mem::offset_of!(Arg, name) as u32, self.name);
    self.ty.write(buf, at + core::mem::offset_of!(Arg, ty) as u32);
  }
}

impl<'a> Emit<Inst> for InstEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    write_ref(buf, at + core::mem::offset_of!(Inst, block) as u32, self.block);
    write_pod(buf, at + core::mem::offset_of!(Inst, name) as u32, self.name);
    self.opc.write(buf, at + core::mem::offset_of!(Inst, opc) as u32);
    self.ty.write(buf, at + core::mem::offset_of!(Inst, ty) as u32);
  }
}

impl<'a> Emit<Block> for BlockEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    write_ref(buf, at + core::mem::offset_of!(Block, func) as u32, self.func);
    write_own(buf, at + core::mem::offset_of!(Block, args) as u32, self.args);
    self.ret.write(buf, at + core::mem::offset_of!(Block, ret) as u32);
    write_own(buf, at + core::mem::offset_of!(Block, code) as u32, self.code);
    self.term.write(buf, at + core::mem::offset_of!(Block, term) as u32);
  }
}

impl<'a> Emit<Func> for FuncEmit<'a> {
  #[inline(always)]
  fn write(self, buf: &mut [u8], at: u32) {
    write_own(buf, at + core::mem::offset_of!(Func, captures) as u32, self.captures);
    write_option_ref(buf, at + core::mem::offset_of!(Func, resume) as u32, self.resume);
    self.entry.write(buf, at + core::mem::offset_of!(Func, entry) as u32);
    write_own(buf, at + core::mem::offset_of!(Func, rest) as u32, self.rest);
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
  let entry_ref: SlotRef<Block> = func_slot.project(core::mem::offset_of!(Func, entry) as u32);
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
