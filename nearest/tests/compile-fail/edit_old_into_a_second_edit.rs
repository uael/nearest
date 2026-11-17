//! An `Old` cannot reach a second `Region::edit` of its region.
use nearest::{Near, Region, flat};

#[flat]
enum Expr { Lit(i64), Neg(Near<Expr>), Add(Near<Expr>, Near<Expr>) }
#[flat]
struct Block { id: u32, insts: Near<[Inst]>, next: Option<Near<Block>> }
#[flat]
struct Inst { op: u8, args: Near<[u32]> }

fn tree() -> Region<Expr> { Region::new(Expr::add(Expr::neg(Expr::Lit(1)), Expr::Lit(2))) }
fn blocks() -> Region<Block> { Region::new(Block::new(0, [Inst::new(1, [10])], Near::new(Block::new(1, (), None)))) }

fn main() {
  let mut r = tree();
  let old = r.edit().replace(Expr::Lit(0));
  r.edit().set(Expr::neg(old));
  let _ = blocks();
}
