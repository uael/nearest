//! A `Near` has no constructor.
use nearest::Near;

fn main() {
  let _ = Near::<u32> { off: std::num::NonZero::new(4).unwrap(), _t: std::marker::PhantomData };
}
