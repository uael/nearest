use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Fields, Meta};

/// Generate an `Emit<Target>` proxy impl for an enum with `#[emit(Target)]`.
///
/// Each variant must have exactly one unnamed field. The generated impl
/// dispatches `write_at` to the inner value's `Emit<Target>` implementation.
pub fn gen_emit_proxy(input: &DeriveInput) -> TokenStream {
  let target = match parse_emit_target(input) {
    Ok(t) => t,
    Err(e) => return e.to_compile_error(),
  };

  let syn::Data::Enum(data) = &input.data else {
    return syn::Error::new_spanned(&input.ident, "#[derive(Emit)] is only supported on enums")
      .to_compile_error();
  };

  let enum_name = &input.ident;

  // Validate variants and collect match arms.
  let mut arms = Vec::new();
  for variant in &data.variants {
    let vname = &variant.ident;
    match &variant.fields {
      Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
        arms.push(quote! {
          Self::#vname(inner) => unsafe { inner.write_at(nearest_p, nearest_at) }
        });
      }
      _ => {
        return syn::Error::new_spanned(
          variant,
          "#[derive(Emit)] variants must have exactly one unnamed field",
        )
        .to_compile_error();
      }
    }
  }

  // Add Emit<Target> bounds to all generic type parameters.
  let (_, ty_generics, where_clause) = input.generics.split_for_impl();
  let mut generics = input.generics.clone();
  for param in &mut generics.params {
    if let syn::GenericParam::Type(tp) = param {
      tp.bounds.push(syn::parse_quote!(::nearest::Emit<#target>));
    }
  }
  let (impl_generics, _, _) = generics.split_for_impl();

  // Merge with existing where clause.
  let where_clause = where_clause.cloned();

  quote! {
    unsafe impl #impl_generics ::nearest::Emit<#target> for #enum_name #ty_generics
    #where_clause
    {
      unsafe fn write_at(
        self,
        nearest_p: &mut impl ::nearest::Patch,
        nearest_at: ::nearest::__private::Pos,
      ) {
        match self {
          #(#arms),*
        }
      }
    }
  }
}

/// Parse the `#[emit(TargetType)]` attribute from the derive input.
fn parse_emit_target(input: &DeriveInput) -> syn::Result<syn::Type> {
  for attr in &input.attrs {
    if attr.path().is_ident("emit") {
      if let Meta::List(list) = &attr.meta {
        let target: syn::Type = syn::parse2(list.tokens.clone())?;
        return Ok(target);
      }
      return Err(syn::Error::new_spanned(attr, "expected #[emit(TargetType)]"));
    }
  }
  Err(syn::Error::new_spanned(
    &input.ident,
    "#[derive(Emit)] requires #[emit(TargetType)] attribute",
  ))
}
