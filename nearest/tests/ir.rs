//! Runs `examples/ir.rs`: a compiler IR in a region, saved to a file, loaded and run, whose bytes
//! are the ones of `examples/ir.image` on every machine.
//!
//! Under Miri as well, where it needs `-Zmiri-disable-isolation` (it writes a file and reads it
//! back): with `--target s390x-unknown-linux-gnu`, this is what shows that a big-endian machine
//! writes the same file.

#[path = "../examples/ir.rs"]
mod ir;

#[test]
fn ir() {
  ir::main();
}
