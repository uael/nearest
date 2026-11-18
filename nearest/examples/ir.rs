//! A compiler IR that is its own file format, in safe Rust: a function, its blocks, their
//! instructions, trees of expressions and a table of strings, declared once with `#[flat]`.
//!
//! A region keeps a value and everything it points to in one buffer, where a pointer is the
//! distance from its own address to its target: the bytes of the region are the value, wherever
//! they are. So the IR is saved by writing those bytes to a file, and read back by looking at
//! them in place (a mapped file): no parsing, no allocation, nothing built node by node. The
//! numbers of `puns` (re-exported as `nearest::puns`) make the bytes the same on every machine,
//! and the bytes are checked when they come back:
//!
//! | Specificity                                 | Told by                                        |
//! |---------------------------------------------|------------------------------------------------|
//! | pointers that are the same on every machine | `Near<T, i32le>`: a little-endian offset       |
//! | numbers that are the same on every machine  | `u16le`, `u32le`, `i64le`                      |
//! | no padding, whose bytes would be anything   | all of them are byte arrays: no alignment      |
//! | a magic number, and a version               | `u32lec<0x3152_4950>`, `valid(1)`              |
//! | no more parameters than registers           | `assert(params <= registers)`                  |
//! | a tree of expressions                       | `Near<Expr, Off>` in `Expr`                    |
//! | a table of strings                          | `Near<[Near<str, Off>], Off>`                  |
//! | what a loaded file must be                  | all of the above: `nearest::view`              |
//!
//! When a file is loaded, what is checked is what bytes do not prove: that every pointer aims
//! where a copy of the value would have put its target (hence in bounds, aligned, a tree, without
//! a gap), tags of enums, UTF-8, lengths, the magic number and the rules. What cannot be invalid
//! (a list of numbers) is not looked at, however long.
//!
//! The first half of this file is type declarations only: no `unsafe`, no `impl`, no code that
//! reads or writes bytes. The second half builds a function, saves it, loads it and runs it.
//!
//! What is not covered, and why:
//!
//! - A reference by number (a jump to a block, a register, a string of the table) is an index,
//!   not a pointer: it is checked where it is used, as any index is, not when the file is loaded.
//!   A rule of the function cannot tell that a block jumps within the blocks: rules read the
//!   fields of their struct, not what its pointers aim at.
//! - The file is read into memory at an address that is a multiple of 8, where a compiler would
//!   map it (`mmap` gives pages, which are): mapping needs a dependency or `unsafe`, and shows
//!   nothing more. `view` takes bytes aligned as a region is, wherever they come from.
//! - Editing a region in place (pushing a block, replacing an instruction): see `passes.rs`.
//! - Reading the file of an older version of the IR as the new one, without converting it, is
//!   `puns::view_ref`: see the documentation of the crate.

use std::{
  fmt::Write as _,
  fs,
  io::{Read as _, Write as _},
};

use nearest::{
  Emit, Error, Near, Region, flat,
  puns::{
    self,
    num::{i32le, i64le, u16le, u32le, u32lec},
  },
};

// -------------------------------------------------------------------------------------------------
// The IR

/// The offset of every pointer, and the length of every list and string: 4 bytes, little-endian
/// on every machine.
pub type Off = i32le;

/// A function: registers (the first ones hold its parameters), blocks (the first one is where
/// it starts), and the strings that its instructions name by their index.
#[flat]
#[derive(Debug, PartialEq, Eq)]
#[pun(assert(params <= registers))]
pub struct Function {
  /// `PIR1`.
  pub magic: u32lec<0x3152_4950>,
  #[pun(valid(1))]
  pub version: u16le,
  pub params: u16le,
  pub registers: u16le,
  pub name: Near<str, Off>,
  pub blocks: Near<[Block], Off>,
  pub strings: Near<[Near<str, Off>], Off>,
}

/// Instructions, then what comes next.
#[flat]
#[derive(Debug, PartialEq, Eq)]
pub struct Block {
  pub label: Near<str, Off>,
  pub body: Near<[Inst], Off>,
  pub end: End,
}

#[flat]
#[derive(Debug, PartialEq, Eq)]
pub enum Inst {
  /// A register takes a value.
  Set(u16le, Operand),
  /// A register takes what an operation makes of two values.
  Bin(u16le, Op, Operand, Operand),
  /// A register takes the value of an expression.
  Eval(u16le, Near<Expr, Off>),
  /// A string of the table (by its index) is printed, then values.
  Print(u32le, Near<[Operand], Off>),
}

/// A tree: a value, or an operation on trees.
#[flat]
#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
  Leaf(Operand),
  Neg(Near<Expr, Off>),
  Bin(Op, Near<Expr, Off>, Near<Expr, Off>),
}

/// A register, by its number, or an integer.
#[flat]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Operand {
  Reg(u16le),
  Int(i64le),
}

#[flat]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Op {
  Add,
  Sub,
  Mul,
  Less,
}

/// How a block ends: blocks are named by their index.
#[flat]
#[derive(Debug, PartialEq, Eq)]
pub enum End {
  Return(Operand),
  Jump(u32le),
  /// To the first block if the value is not 0, else to the second.
  Branch(Operand, u32le, u32le),
}

// -------------------------------------------------------------------------------------------------
// Building one

const fn reg(number: u16) -> Operand {
  Operand::Reg(u16le::new(number))
}

const fn int(value: i64) -> Operand {
  Operand::Int(i64le::new(value))
}

const fn block(index: u32) -> u32le {
  u32le::new(index)
}

fn leaf(operand: Operand) -> impl Emit<Expr> {
  Expr::Leaf(operand)
}

/// The sum of the squares of the integers below its parameter, which it prints.
///
/// ```text
/// entry:  total = 0; i = 0; jump test
/// test:   more = i < n; branch more, body, done
/// body:   total = total + -(-(i * i)); i = i + 1; jump test
/// done:   print "sum of squares below", n, total; return total
/// ```
#[must_use]
pub fn sum_of_squares() -> Region<Function> {
  const N: u16 = 0;
  const TOTAL: u16 = 1;
  const I: u16 = 2;
  const MORE: u16 = 3;
  let set = |register, operand| Inst::Set(u16le::new(register), operand);
  let square = Expr::bin(Op::Mul, leaf(reg(I)), leaf(reg(I)));
  let sum = Expr::bin(Op::Add, leaf(reg(TOTAL)), Expr::neg(Expr::neg(square)));
  Region::new(Function::new(
    u32lec::new(),
    u16le::new(1),
    u16le::new(1),
    u16le::new(4),
    "sum_of_squares",
    (
      Block::new("entry", [set(TOTAL, int(0)), set(I, int(0))], End::Jump(block(1))),
      Block::new(
        "test",
        [Inst::Bin(u16le::new(MORE), Op::Less, reg(I), reg(N))],
        End::Branch(reg(MORE), block(2), block(3)),
      ),
      Block::new(
        "body",
        (Inst::eval(u16le::new(TOTAL), sum), Inst::Bin(u16le::new(I), Op::Add, reg(I), int(1))),
        End::Jump(block(1)),
      ),
      Block::new(
        "done",
        [Inst::print(u32le::new(1), [reg(N), reg(TOTAL)])],
        End::Return(reg(TOTAL)),
      ),
    ),
    ["unused", "sum of squares below"],
  ))
}

// -------------------------------------------------------------------------------------------------
// Running one: the IR is read as it is, wherever its bytes are

fn operand(value: &Operand, registers: &[i64]) -> Option<i64> {
  match value {
    Operand::Reg(number) => registers.get(usize::from(number.get())).copied(),
    Operand::Int(value) => Some(value.get()),
  }
}

fn apply(op: Op, left: i64, right: i64) -> i64 {
  match op {
    Op::Add => left.wrapping_add(right),
    Op::Sub => left.wrapping_sub(right),
    Op::Mul => left.wrapping_mul(right),
    Op::Less => i64::from(left < right),
  }
}

fn eval(expr: &Expr, registers: &[i64]) -> Option<i64> {
  Some(match expr {
    Expr::Leaf(value) => operand(value, registers)?,
    Expr::Neg(inner) => eval(inner, registers)?.wrapping_neg(),
    Expr::Bin(op, left, right) => apply(*op, eval(left, registers)?, eval(right, registers)?),
  })
}

/// What the function returns for `arguments`, with what it prints added to `output`. `None` if it
/// names a register, a block or a string that it does not have: an index is checked where it is
/// used.
pub fn run(function: &Function, arguments: &[i64], output: &mut String) -> Option<i64> {
  let mut registers = vec![0; usize::from(function.registers.get())];
  registers.get_mut(..arguments.len())?.copy_from_slice(arguments);
  let jump = |to: &u32le| function.blocks.get(usize::try_from(to.get()).ok()?);
  let mut block = function.blocks.first()?;
  loop {
    for inst in &block.body {
      let (register, value) = match inst {
        Inst::Set(register, value) => (register, operand(value, &registers)?),
        Inst::Bin(register, op, left, right) => {
          (register, apply(*op, operand(left, &registers)?, operand(right, &registers)?))
        }
        Inst::Eval(register, expr) => (register, eval(expr, &registers)?),
        Inst::Print(text, values) => {
          output.push_str(function.strings.get(usize::try_from(text.get()).ok()?)?);
          for value in values {
            write!(output, " {}", operand(value, &registers)?).ok()?;
          }
          output.push('\n');
          continue;
        }
      };
      *registers.get_mut(usize::from(register.get()))? = value;
    }
    block = match &block.end {
      End::Return(value) => return operand(value, &registers),
      End::Jump(to) => jump(to)?,
      End::Branch(value, then, otherwise) => {
        jump(if operand(value, &registers)? == 0 { otherwise } else { then })?
      }
    };
  }
}

// -------------------------------------------------------------------------------------------------
// Saving and loading

/// The file of [`sum_of_squares`], as any machine writes it.
pub const IMAGE: &[u8] = include_bytes!("ir.image");

/// Writes the function to a file, reads the file, and runs what it holds as it is.
///
/// # Panics
/// If something is not what it should be.
pub fn main() {
  let function = sum_of_squares();
  let mut output = String::new();
  assert_eq!(run(&function, &[4], &mut output), Some(14));
  assert_eq!(output, "sum of squares below 4 14\n");

  // The bytes of the region are the file. They are the same on every machine: offsets, lengths
  // and numbers are little-endian, and nothing is aligned but lists, to the size of an offset.
  let image = function.as_bytes().expect("a region that was just built is compact");
  if std::env::var_os("NEAREST_WRITE_IMAGE").is_some() {
    fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/ir.image"), image).unwrap();
  }
  assert_eq!(image, IMAGE, "the image is not the one of `examples/ir.image`");
  assert_eq!(image[..4], *b"PIR1");
  let path = std::env::temp_dir().join(format!("nearest-ir-{}.image", std::process::id()));
  fs::File::create(&path).and_then(|mut file| file.write_all(image)).unwrap();

  // Read back, at an address that is a multiple of 8 as a mapped file is: here, in a buffer
  // that has room for it.
  let size = usize::try_from(fs::metadata(&path).unwrap().len()).unwrap();
  let mut buffer = vec![0u8; size + 8];
  let start = buffer.as_ptr().addr().wrapping_neg() % 8;
  fs::File::open(&path)
    .and_then(|mut file| file.read_exact(&mut buffer[start..start + size]))
    .unwrap();
  fs::remove_file(&path).unwrap();
  let bytes = &buffer[start..start + size];

  // Checked, and that is all: the function is these bytes.
  let loaded: &Function = nearest::view(bytes).unwrap();
  assert!(core::ptr::addr_eq(loaded, bytes.as_ptr()));
  assert_eq!(*loaded, *function);
  assert_eq!(
    (&*loaded.name, loaded.blocks.len(), &*loaded.blocks[2].label),
    ("sum_of_squares", 4, "body")
  );
  let mut output = String::new();
  assert_eq!(run(loaded, &[10], &mut output), Some(285));
  assert_eq!(output, "sum of squares below 10 285\n");

  // A copy of it in a region of its own, to keep or to edit, from bytes at any address.
  let unaligned = [&[0][..], bytes].concat();
  let owned = Region::<Function>::from_bytes(&unaligned[1..]).unwrap();
  assert_eq!(*owned, *loaded);

  // Bytes that are not a function are refused, where they stop being one.
  let invalid = |at: usize, value: u8| {
    let mut bad = bytes.to_vec();
    bad[at] = value;
    Region::<Function>::from_bytes(&bad).err()
  };
  let place = |part: &u8| (&raw const *part).addr() - bytes.as_ptr().addr();
  // The magic number, the version, and the rule of the function.
  assert_eq!(invalid(0, b'Q'), Some(Error::Invalid(0)));
  assert_eq!(invalid(4, 2), Some(Error::Invalid(4)));
  assert_eq!(invalid(6, 5), Some(Error::Invalid(0)));
  // A pointer that aims elsewhere: the name, one byte further.
  assert_eq!(invalid(10, bytes[10] + 1), Some(Error::Invalid(10)));
  // A tag that is not one: the end of the first block, an operation of the second.
  let end = place(puns::head_ref::<End, u8>(&loaded.blocks[0].end));
  assert_eq!(invalid(end, 3), Some(Error::Invalid(end)));
  let Inst::Bin(_, op, ..) = &loaded.blocks[1].body[0] else { panic!("not what was written") };
  let op = place(puns::pun_ref::<Op, u8>(op));
  assert_eq!(invalid(op, 4), Some(Error::Invalid(op)));
  // A label that is not UTF-8: the error is at its length.
  let label = place(&loaded.blocks[3].label.as_bytes()[0]);
  assert_eq!(invalid(label, 0xff), Some(Error::Invalid(label - size_of::<Off>())));
  // Cut short, or with something after it.
  assert!(Region::<Function>::from_bytes(&bytes[..size - 1]).is_err());
  assert_eq!(invalid_extended(bytes), Some(Error::Invalid(size)));

  // What a loaded function names by number is checked where it is used.
  let mut wild = bytes.to_vec();
  let Inst::Print(text, _) = &loaded.blocks[3].body[0] else { panic!("not what was written") };
  wild[place(&puns::as_bytes(text)[0])] = 9;
  let wild = Region::<Function>::from_bytes(&wild).unwrap();
  assert_eq!(run(&wild, &[3], &mut String::new()), None);

  println!("ok");
}

/// What loading `bytes` with one more gives.
fn invalid_extended(bytes: &[u8]) -> Option<Error> {
  let mut long = bytes.to_vec();
  long.push(0);
  Region::<Function>::from_bytes(&long).err()
}
