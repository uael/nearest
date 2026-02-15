//! Derive macros for the `nearest` crate.

mod attrs;
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
///
/// # Field attributes
///
/// ## `#[flat(into)]`
///
/// On a primitive or `Other`-classified field, the generated `make()` builder
/// accepts `impl Into<T>` instead of `T`. This allows callers to pass values
/// that can be cheaply converted.
///
/// ```ignore
/// #[derive(Flat)]
/// struct Inst {
///   #[flat(into)]
///   op: u16,
///   typ: Type,
/// }
///
/// // Now `op` accepts any `impl Into<u16>`:
/// Inst::make(42u8, Type(0))
/// ```
///
/// # Variant attributes
///
/// ## `#[flat(rename = "name")]`
///
/// On an enum variant, overrides the generated `make_*` method name.
///
/// ```ignore
/// #[derive(Flat)]
/// #[repr(C, u8)]
/// enum Term {
///   #[flat(rename = "ret")]
///   Return { values: NearList<Value> },
///   Jmp(Jmp),
/// }
///
/// // Instead of `Term::make_return(...)`:
/// Term::ret(...)
/// ```
#[proc_macro_derive(Flat, attributes(flat))]
pub fn derive_flat(input: TokenStream) -> TokenStream {
  let input = parse_macro_input!(input as syn::DeriveInput);
  if let Data::Union(ref u) = input.data {
    return TokenStream::from(
      syn::Error::new_spanned(u.union_token, "Flat cannot be derived for unions")
        .to_compile_error(),
    );
  }
  if let Err(err) = attrs::validate_all_attrs(&input) {
    return TokenStream::from(err.to_compile_error());
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
