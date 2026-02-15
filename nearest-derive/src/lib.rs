//! Derive macros for the `nearest` crate.

mod emit;
mod emit_proxy;
mod flat;
mod util;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, parse_macro_input};

/// Derive `Flat` and `Emit` for region-based storage.
///
/// Generates implementations for both the `Flat` marker trait and the `Emit`
/// builder trait, enabling declarative region construction.
#[proc_macro_derive(Flat)]
pub fn derive_flat(input: TokenStream) -> TokenStream {
  let input = parse_macro_input!(input as syn::DeriveInput);
  if let Data::Union(ref u) = input.data {
    return TokenStream::from(
      syn::Error::new_spanned(u.union_token, "Flat cannot be derived for unions")
        .to_compile_error(),
    );
  }
  if let Data::Enum(ref data) = input.data
    && let Some(err) = flat::validate_enum(&input, data)
  {
    return TokenStream::from(err);
  }
  let flat_impl = flat::gen_flat_impl(&input);
  let emit_impl = emit::gen_emit_impl(&input);
  TokenStream::from(quote! { #flat_impl #emit_impl })
}

/// Derive [`Emit<T>`] for a proxy enum that dispatches to inner builders.
///
/// Each variant must be a single-field newtype. The target type is specified
/// via `#[emit(TargetType)]`. All generic type parameters receive an
/// `Emit<TargetType>` bound.
///
/// # Example
///
/// ```ignore
/// #[derive(Emit)]
/// #[emit(Block)]
/// enum BlockEmit<A, B, C> {
///   Br(A),
///   Brif(B),
///   Call(C),
/// }
/// ```
///
/// This generates `unsafe impl<A: Emit<Block>, B: Emit<Block>, C: Emit<Block>> Emit<Block> for BlockEmit<A, B, C>`,
/// where each variant delegates `write_at` to its inner value.
#[proc_macro_derive(Emit, attributes(emit))]
pub fn derive_emit(input: TokenStream) -> TokenStream {
  let input = parse_macro_input!(input as syn::DeriveInput);
  TokenStream::from(emit_proxy::gen_emit_proxy(&input))
}
