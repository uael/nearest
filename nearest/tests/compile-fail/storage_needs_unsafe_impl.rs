//! A storage promises what `Storage` documents: implementing it is `unsafe`.
use std::mem::MaybeUninit;

use nearest::Storage;

struct Words(Vec<MaybeUninit<u64>>);

impl Storage for Words {
  fn memory(&self) -> &[MaybeUninit<u64>] { &self.0 }
  fn memory_mut(&mut self, _words: usize) -> &mut [MaybeUninit<u64>] { &mut self.0 }
}

fn main() {}
