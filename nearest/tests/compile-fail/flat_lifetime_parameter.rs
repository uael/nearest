//! Stored types are `'static`: a lifetime parameter is rejected.
use nearest::flat;

#[flat]
struct Tagged<'a> { id: u32, tag: core::marker::PhantomData<&'a ()> }

fn main() {}
