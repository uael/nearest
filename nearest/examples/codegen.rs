//! Functions whose machine code is worth looking at:
//!
//! ```text
//! cargo rustc -p nearest --release --example codegen -- --emit asm -C llvm-args=-x86-asm-syntax=intel
//! ```
//!
//! and look for their names in the `codegen.s` that it writes under `target/release` (`find target
//! -name codegen.s`). The check of a region is what `#[flat]` writes for the type.
#![expect(
  clippy::no_mangle_with_rust_abi,
  reason = "the code of the Rust ABI is what is measured; `no_mangle` only names it in the assembly"
)]

use nearest::{
  Near, Pending, flat,
  puns::num::{i32le, u16le, u32le},
};

/// An expression: a tag, then pointers to its parts.
#[flat]
pub enum Expr {
  Leaf(bool),
  Neg(Near<Expr, i32le>),
  Call(Near<str, i32le>, Near<[Expr], i32le>),
}

/// A table: a name, rows that hold a pointer and a `bool`, and numbers.
#[flat]
pub struct Table {
  pub version: u32le,
  pub name: Near<str, i32le>,
  pub rows: Near<[Row], i32le>,
  pub numbers: Near<[u16le], i32le>,
}

#[flat]
pub struct Row {
  pub public: bool,
  pub label: Near<str, i32le>,
}

/// Numbers, and only numbers: nothing in them can be invalid.
#[flat]
pub struct Numbers {
  pub count: u32le,
  pub values: Near<[u32le], i32le>,
}

// The check of a recursive enum: a tag, then the parts of its variant.
#[unsafe(no_mangle)]
pub fn load_expr<'b>(bytes: &'b [u8], scratch: &mut [Pending]) -> Option<&'b Expr> {
  nearest::view_with(bytes, scratch).ok()
}

// The check of a region, with the memory of the caller.
#[unsafe(no_mangle)]
pub fn load_table<'b>(bytes: &'b [u8], scratch: &mut [Pending]) -> Option<&'b Table> {
  nearest::view_with(bytes, scratch).ok()
}

// A list of numbers is not walked: its pointer, its length, and that is all.
#[unsafe(no_mangle)]
pub fn load_numbers<'b>(bytes: &'b [u8], scratch: &mut [Pending]) -> Option<&'b Numbers> {
  nearest::view_with(bytes, scratch).ok()
}

fn main() {}
