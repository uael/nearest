use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DataEnum, DataStruct, DeriveInput, Fields};

use crate::util::{combine_where, flat_bounded_param_names, is_bool_type, is_primitive_type};

// ---------------------------------------------------------------------------
// Enum validation
// ---------------------------------------------------------------------------

/// Validate that an enum has the correct repr and no explicit discriminants.
///
/// - All enums must have `u8` in their `#[repr(...)]` (either `repr(u8)` or
///   `repr(C, u8)`) so the derive can write the discriminant as a `u8` at byte 0.
/// - Explicit discriminant values (e.g. `A = 5`) are rejected because the
///   derive uses the variant index as the discriminant.
pub fn validate_enum(input: &DeriveInput, data: &DataEnum) -> Option<TokenStream> {
  let name = &input.ident;

  if data.variants.len() > 255 {
    return Some(
      syn::Error::new_spanned(name, "Flat derive: enum has more than 255 variants (u8 overflow)")
        .to_compile_error(),
    );
  }

  for variant in &data.variants {
    if variant.discriminant.is_some() {
      return Some(
        syn::Error::new_spanned(
          &variant.ident,
          "Flat derive: explicit discriminant values are not supported",
        )
        .to_compile_error(),
      );
    }
  }

  let mut has_u8 = false;
  let mut has_c = false;
  for attr in &input.attrs {
    if attr.path().is_ident("repr") {
      let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("u8") {
          has_u8 = true;
        }
        if meta.path.is_ident("C") {
          has_c = true;
        }
        Ok(())
      });
    }
  }

  if !has_u8 {
    return Some(
      syn::Error::new_spanned(name, "Flat derive: enum requires #[repr(u8)] or #[repr(C, u8)]")
        .to_compile_error(),
    );
  }

  let has_data_variants = data.variants.iter().any(|v| !matches!(v.fields, Fields::Unit));
  if has_data_variants && !has_c {
    return Some(
      syn::Error::new_spanned(
        name,
        "Flat derive: enums with data fields require #[repr(C, u8)] to guarantee discriminant layout",
      )
      .to_compile_error(),
    );
  }

  None
}

// ---------------------------------------------------------------------------
// Flat impl generation (with deep_copy)
// ---------------------------------------------------------------------------

pub fn gen_flat_impl(input: &DeriveInput) -> TokenStream {
  let name = &input.ident;
  let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
  let already_bounded = flat_bounded_param_names(&input.generics);

  // Only generate `Flat` bounds for type parameters, not for concrete field types.
  // Concrete types have their own `impl Flat` blocks that the compiler can verify
  // from the impl graph. Including concrete types in the where clause creates
  // unsolvable cycles in recursive type graphs (e.g. Block → Term → Jmp → Block).
  let mut where_predicates = Vec::new();
  for tp in input.generics.type_params() {
    let ident = &tp.ident;
    let name = ident.to_string();
    if !already_bounded.contains(&name) {
      where_predicates.push(quote! { #ident: ::nearest::Flat });
    }
  }

  let deep_copy_body = gen_deep_copy_body(input);
  let validate_body = gen_validate_body(input);
  let combined_where = combine_where(where_clause, &where_predicates);

  quote! {
    // SAFETY: All fields are bounded by `Flat`, and the const assert ensures no `Drop` impl.
    unsafe impl #impl_generics ::nearest::Flat for #name #ty_generics #combined_where {
      const _ASSERT_NO_DROP: () = {
        const { assert!(!::core::mem::needs_drop::<#name #ty_generics>()) };
      };

      unsafe fn deep_copy(&self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
        #deep_copy_body
      }

      fn validate(nearest_addr: usize, nearest_buf: &[u8]) -> ::core::result::Result<(), ::nearest::ValidateError> {
        #validate_body
      }
    }
  }
}

// ---------------------------------------------------------------------------
// deep_copy body generation
// ---------------------------------------------------------------------------

fn gen_deep_copy_body(input: &DeriveInput) -> TokenStream {
  match &input.data {
    Data::Struct(s) => gen_deep_copy_struct(input, s),
    Data::Enum(e) => gen_deep_copy_enum(input, e),
    Data::Union(_) => quote! {},
  }
}

fn gen_deep_copy_struct(input: &DeriveInput, data: &DataStruct) -> TokenStream {
  let name = &input.ident;
  let (_, ty_generics, _) = input.generics.split_for_impl();

  let field_codes: Vec<_> = match &data.fields {
    Fields::Named(named) => named
      .named
      .iter()
      .map(|f| {
        let field_name = f.ident.as_ref().unwrap();
        let field_ty = &f.ty;
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #field_name) };
        let ref_expr = quote! { &self.#field_name };
        gen_deep_copy_field(&ref_expr, field_ty, &offset_expr)
      })
      .collect(),
    Fields::Unnamed(unnamed) => unnamed
      .unnamed
      .iter()
      .enumerate()
      .map(|(i, f)| {
        let idx = syn::Index::from(i);
        let field_ty = &f.ty;
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #idx) };
        let ref_expr = quote! { &self.#idx };
        gen_deep_copy_field(&ref_expr, field_ty, &offset_expr)
      })
      .collect(),
    Fields::Unit => vec![],
  };

  quote! { #(#field_codes)* }
}

fn gen_deep_copy_enum(input: &DeriveInput, data: &DataEnum) -> TokenStream {
  let name = &input.ident;
  let (_, ty_generics, _) = input.generics.split_for_impl();

  let match_arms: Vec<_> = data
    .variants
    .iter()
    .enumerate()
    .map(|(idx, variant)| {
      let vname = &variant.ident;
      let disc = idx as u8;

      match &variant.fields {
        Fields::Named(named) => {
          let field_names: Vec<_> = named.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
          let field_codes: Vec<_> = named
            .named
            .iter()
            .map(|f| {
              let field_name = f.ident.as_ref().unwrap();
              let field_ty = &f.ty;
              let offset_expr =
                quote! { ::core::mem::offset_of!(#name #ty_generics, #vname.#field_name) };
              let ref_expr = quote! { #field_name };
              gen_deep_copy_field(&ref_expr, field_ty, &offset_expr)
            })
            .collect();

          quote! {
            #name::#vname { #(#field_names),* } => {
              unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
              #(#field_codes)*
            }
          }
        }
        Fields::Unnamed(unnamed) => {
          let field_idents: Vec<_> =
            (0..unnamed.unnamed.len()).map(|i| quote::format_ident!("f{}", i)).collect();
          let field_codes: Vec<_> = unnamed
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| {
              let field_ident = quote::format_ident!("f{}", i);
              let field_ty = &f.ty;
              let idx = syn::Index::from(i);
              let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #vname.#idx) };
              let ref_expr = quote! { #field_ident };
              gen_deep_copy_field(&ref_expr, field_ty, &offset_expr)
            })
            .collect();

          quote! {
            #name::#vname(#(#field_idents),*) => {
              unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
              #(#field_codes)*
            }
          }
        }
        Fields::Unit => quote! {
          #name::#vname => {
            unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
          }
        },
      }
    })
    .collect();

  quote! {
    match self {
      #(#match_arms)*
    }
  }
}

// ---------------------------------------------------------------------------
// validate body generation
// ---------------------------------------------------------------------------

fn gen_validate_body(input: &DeriveInput) -> TokenStream {
  match &input.data {
    Data::Struct(s) => gen_validate_struct(input, s),
    Data::Enum(e) => gen_validate_enum(input, e),
    Data::Union(_) => quote! { Ok(()) },
  }
}

fn gen_validate_struct(input: &DeriveInput, data: &DataStruct) -> TokenStream {
  let name = &input.ident;
  let (_, ty_generics, _) = input.generics.split_for_impl();

  let field_codes: Vec<_> = match &data.fields {
    Fields::Named(named) => named
      .named
      .iter()
      .map(|f| {
        let field_name = f.ident.as_ref().unwrap();
        let field_ty = &f.ty;
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #field_name) };
        gen_validate_field(field_ty, &offset_expr)
      })
      .collect(),
    Fields::Unnamed(unnamed) => unnamed
      .unnamed
      .iter()
      .enumerate()
      .map(|(i, f)| {
        let idx = syn::Index::from(i);
        let field_ty = &f.ty;
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #idx) };
        gen_validate_field(field_ty, &offset_expr)
      })
      .collect(),
    Fields::Unit => vec![],
  };

  quote! {
    ::nearest::ValidateError::check::<Self>(nearest_addr, nearest_buf)?;
    #(#field_codes)*
    Ok(())
  }
}

fn gen_validate_enum(input: &DeriveInput, data: &DataEnum) -> TokenStream {
  let name = &input.ident;
  let (_, ty_generics, _) = input.generics.split_for_impl();
  let variant_count = data.variants.len();
  let max_disc = if variant_count == 0 { 0u8 } else { (variant_count - 1) as u8 };

  let match_arms: Vec<_> = data
    .variants
    .iter()
    .enumerate()
    .map(|(idx, variant)| {
      let vname = &variant.ident;
      let disc = idx as u8;

      let field_codes: Vec<_> = match &variant.fields {
        Fields::Named(named) => named
          .named
          .iter()
          .map(|f| {
            let field_name = f.ident.as_ref().unwrap();
            let field_ty = &f.ty;
            let offset_expr =
              quote! { ::core::mem::offset_of!(#name #ty_generics, #vname.#field_name) };
            gen_validate_field(field_ty, &offset_expr)
          })
          .collect(),
        Fields::Unnamed(unnamed) => unnamed
          .unnamed
          .iter()
          .enumerate()
          .map(|(i, f)| {
            let idx = syn::Index::from(i);
            let field_ty = &f.ty;
            let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #vname.#idx) };
            gen_validate_field(field_ty, &offset_expr)
          })
          .collect(),
        Fields::Unit => vec![],
      };

      quote! {
        #disc => { #(#field_codes)* }
      }
    })
    .collect();

  quote! {
    ::nearest::ValidateError::check::<Self>(nearest_addr, nearest_buf)?;
    let nearest_disc = nearest_buf[nearest_addr];
    if nearest_disc > #max_disc {
      return Err(::nearest::ValidateError::InvalidDiscriminant {
        addr: nearest_addr,
        value: nearest_disc,
        max: #max_disc,
      });
    }
    match nearest_disc {
      #(#match_arms)*
      _ => unreachable!(),
    }
    Ok(())
  }
}

/// Generate validate code for a single field.
///
/// Each type's `Flat::validate` handles its own semantics (`Near<T>` follows offsets,
/// `NearList<T>` walks segments, etc.), so the derive macro just delegates uniformly.
fn gen_validate_field(field_ty: &syn::Type, offset_expr: &TokenStream) -> TokenStream {
  if is_primitive_type(field_ty) && !is_bool_type(field_ty) {
    // Covered by the struct/enum-level bounds check — no extra validation needed.
    quote! {}
  } else {
    quote! { <#field_ty as ::nearest::Flat>::validate(nearest_addr + #offset_expr, nearest_buf)?; }
  }
}

// ---------------------------------------------------------------------------
// Deep-copy field code generation
// ---------------------------------------------------------------------------

/// Generate `deep_copy` code for a single field.
///
/// `ref_expr` is an expression of type `&FieldType`.
///
/// Each type's `Flat::deep_copy` handles its own semantics (`Near<T>` emits target
/// and patches offset, `NearList<T>` allocates segments and copies elements, etc.),
/// so the derive macro delegates uniformly via the blanket `Emit<T> for &T` impl.
fn gen_deep_copy_field(
  ref_expr: &TokenStream,
  field_ty: &syn::Type,
  offset_expr: &TokenStream,
) -> TokenStream {
  quote! {
    unsafe {
      ::nearest::Emit::<#field_ty>::write_at(
        #ref_expr,
        nearest_p,
        nearest_at.offset(#offset_expr),
      );
    }
  }
}
