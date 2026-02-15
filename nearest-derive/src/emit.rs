use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DataEnum, DataStruct, DeriveInput, Fields, Variant};

use crate::{
  attrs::{parse_field_attrs, parse_variant_attrs},
  util::{
    FieldKind, capitalize, classify_field, collect_field_types, combine_where,
    flat_bounded_param_names, has_flat_bound, has_no_pointer_fields, is_all_primitive,
    is_type_param_ident, opt_where_clause, to_snake_case,
  },
};

// ---------------------------------------------------------------------------
// Emit impl generation (builders)
// ---------------------------------------------------------------------------

pub fn gen_emit_impl(input: &DeriveInput) -> TokenStream {
  match &input.data {
    Data::Struct(s) => gen_emit_struct(input, s),
    Data::Enum(e) => gen_emit_enum(input, e),
    Data::Union(_) => quote! {},
  }
}

// ---------------------------------------------------------------------------
// Shared field analysis
// ---------------------------------------------------------------------------

/// Info about a single emitter field.
struct EmitterField {
  /// The type for the function parameter (e.g. `u16` or `impl Emit<Block>`).
  fn_param_type: TokenStream,
  /// The type for the internal builder struct field.
  builder_type: TokenStream,
  /// If Some, a generic param name was introduced.
  generic_param: Option<proc_macro2::Ident>,
  /// The where-clause predicate for this generic, if any.
  where_pred: Option<TokenStream>,
  /// Code to emit in `write_at` for this field.
  write_at_code: TokenStream,
  /// If the `fn_param_type` uses a named generic (e.g. `Option<__P>`), this
  /// provides the declaration with bounds for the `fn make` signature.
  fn_generic_decl: Option<TokenStream>,
}

fn analyze_field(
  value_expr: &TokenStream,
  field_ty: &syn::Type,
  param_name: &proc_macro2::Ident,
  offset_expr: &TokenStream,
  use_into: bool,
) -> EmitterField {
  match classify_field(field_ty) {
    FieldKind::Primitive if use_into => EmitterField {
      fn_param_type: quote! { impl Into<#field_ty> },
      builder_type: quote! { #param_name },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! { #param_name: Into<#field_ty> }),
      write_at_code: quote! {
        unsafe {
          ::nearest::Emit::<#field_ty>::write_at(
            (#value_expr).into(),
            nearest_p,
            nearest_at.offset(#offset_expr),
          );
        }
      },
      fn_generic_decl: None,
    },
    FieldKind::Primitive => EmitterField {
      fn_param_type: quote! { #field_ty },
      builder_type: quote! { #field_ty },
      generic_param: None,
      where_pred: None,
      write_at_code: quote! {
        unsafe {
          ::nearest::Emit::<#field_ty>::write_at(
            #value_expr,
            nearest_p,
            nearest_at.offset(#offset_expr),
          );
        }
      },
      fn_generic_decl: None,
    },
    FieldKind::Near { inner } => EmitterField {
      fn_param_type: quote! { impl ::nearest::Emit<#inner> },
      builder_type: quote! { #param_name },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! { #param_name: ::nearest::Emit<#inner> }),
      write_at_code: quote! {
        {
          let nearest_target = ::nearest::Emit::<#inner>::emit(#value_expr, nearest_p);
          unsafe {
            nearest_p.patch_near::<#inner>(nearest_at.offset(#offset_expr), nearest_target);
          }
        }
      },
      fn_generic_decl: None,
    },
    FieldKind::NearList { inner } => EmitterField {
      fn_param_type: quote! {
        impl IntoIterator<IntoIter: ExactSizeIterator, Item: ::nearest::Emit<#inner>>
      },
      builder_type: quote! { #param_name },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! {
        #param_name: IntoIterator,
        #param_name::IntoIter: ExactSizeIterator,
        #param_name::Item: ::nearest::Emit<#inner>
      }),
      write_at_code: gen_near_list_write_at(
        value_expr,
        &inner,
        &quote! { nearest_at.offset(#offset_expr) },
      ),
      fn_generic_decl: None,
    },
    FieldKind::OptionNear { inner } => EmitterField {
      fn_param_type: quote! { Option<#param_name> },
      builder_type: quote! { Option<#param_name> },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! { #param_name: ::nearest::Emit<#inner> }),
      write_at_code: quote! {
        match #value_expr {
          Some(nearest_inner) => {
            let nearest_target = ::nearest::Emit::<#inner>::emit(nearest_inner, nearest_p);
            unsafe {
              nearest_p.patch_near::<#inner>(nearest_at.offset(#offset_expr), nearest_target);
            }
          }
          None => {
            unsafe { nearest_p.write_flat::<i32>(nearest_at.offset(#offset_expr), 0) };
          }
        }
      },
      fn_generic_decl: Some(quote! { #param_name: ::nearest::Emit<#inner> }),
    },
    FieldKind::Other if use_into => EmitterField {
      fn_param_type: quote! { impl Into<#field_ty> },
      builder_type: quote! { #param_name },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! { #param_name: Into<#field_ty> }),
      write_at_code: quote! {
        unsafe {
          let nearest_val: #field_ty = (#value_expr).into();
          ::nearest::Emit::<#field_ty>::write_at(
            nearest_val,
            nearest_p,
            nearest_at.offset(#offset_expr),
          );
        }
      },
      fn_generic_decl: None,
    },
    FieldKind::Other => EmitterField {
      fn_param_type: quote! { impl ::nearest::Emit<#field_ty> },
      builder_type: quote! { #param_name },
      generic_param: Some(param_name.clone()),
      where_pred: Some(quote! { #param_name: ::nearest::Emit<#field_ty> }),
      write_at_code: quote! {
        unsafe {
          ::nearest::Emit::<#field_ty>::write_at(
            #value_expr,
            nearest_p,
            nearest_at.offset(#offset_expr),
          );
        }
      },
      fn_generic_decl: None,
    },
  }
}

// ---------------------------------------------------------------------------
// Builder generics helper
// ---------------------------------------------------------------------------

/// Tokens needed for a `__Builder` struct that must carry the outer type's
/// generic parameters (since nested items can't capture them from the parent).
struct BuilderGen {
  /// Generic params for the struct definition (no bounds): `<CT, __Params, __Returns>`
  struct_generics: TokenStream,
  /// Generic params for the `Emit` impl (with bounds): `<CT: Flat, __Params, __Returns>`
  impl_generics: TokenStream,
  /// `PhantomData` field in struct body (empty if no outer generics).
  phantom_field: TokenStream,
  /// `PhantomData` initializer in construction (empty if no outer generics).
  phantom_init: TokenStream,
  /// Combined where clause: input where-preds + builder where-preds.
  where_clause: TokenStream,
}

fn builder_generics(
  input_generics: &syn::Generics,
  builder_params: &[proc_macro2::Ident],
  builder_where_preds: &[TokenStream],
) -> BuilderGen {
  let outer_type_idents: Vec<_> = input_generics.type_params().map(|tp| &tp.ident).collect();

  let struct_generics = {
    let all: Vec<TokenStream> = outer_type_idents
      .iter()
      .map(|id| quote! { #id })
      .chain(builder_params.iter().map(|id| quote! { #id }))
      .collect();
    if all.is_empty() {
      quote! {}
    } else {
      quote! { <#(#all),*> }
    }
  };

  let impl_generics = {
    let all: Vec<TokenStream> = input_generics
      .type_params()
      .map(|tp| {
        let ident = &tp.ident;
        let bounds = &tp.bounds;
        if bounds.is_empty() {
          quote! { #ident: ::nearest::Flat }
        } else if has_flat_bound(tp) {
          quote! { #ident: #bounds }
        } else {
          quote! { #ident: #bounds + ::nearest::Flat }
        }
      })
      .chain(builder_params.iter().map(|id| quote! { #id }))
      .collect();
    if all.is_empty() {
      quote! {}
    } else {
      quote! { <#(#all),*> }
    }
  };

  let phantom_field = if outer_type_idents.is_empty() {
    quote! {}
  } else {
    quote! { _phantom: ::core::marker::PhantomData<fn() -> (#(#outer_type_idents,)*)>, }
  };

  let phantom_init = if outer_type_idents.is_empty() {
    quote! {}
  } else {
    quote! { _phantom: ::core::marker::PhantomData, }
  };

  let mut all_preds: Vec<TokenStream> = input_generics
    .where_clause
    .iter()
    .flat_map(|wc| wc.predicates.iter().map(|p| quote! { #p }))
    .collect();
  all_preds.extend(builder_where_preds.iter().cloned());
  let where_clause = opt_where_clause(&all_preds);

  BuilderGen { struct_generics, impl_generics, phantom_field, phantom_init, where_clause }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

fn gen_emit_struct(input: &DeriveInput, data: &DataStruct) -> TokenStream {
  if is_all_primitive(&input.data) {
    return gen_emit_self(input);
  }

  let name = &input.ident;
  let vis = &input.vis;
  let (impl_generics, ty_generics, input_where) = input.generics.split_for_impl();

  match &data.fields {
    Fields::Named(named) => {
      let mut fn_params = Vec::new();
      let mut struct_fields = Vec::new();
      let mut generic_params = Vec::new();
      let mut where_preds = Vec::new();
      let mut write_at_codes = Vec::new();
      let mut field_names = Vec::new();
      let mut fn_generic_decls = Vec::new();

      for f in &named.named {
        let field_name = f.ident.as_ref().unwrap();
        let field_ty = &f.ty;
        let field_attrs = match parse_field_attrs(&f.attrs) {
          Ok(a) => a,
          Err(e) => return e.to_compile_error(),
        };
        let param_name = format_ident!("__{}", capitalize(&field_name.to_string()));
        let value_expr = quote! { self.#field_name };
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #field_name) };
        let info =
          analyze_field(&value_expr, field_ty, &param_name, &offset_expr, field_attrs.into);

        let fn_pt = &info.fn_param_type;
        fn_params.push(quote! { #field_name: #fn_pt });
        let bt = &info.builder_type;
        struct_fields.push(quote! { #field_name: #bt });
        if let Some(gp) = info.generic_param {
          generic_params.push(gp);
        }
        if let Some(wp) = info.where_pred {
          where_preds.push(wp);
        }
        if let Some(fg) = info.fn_generic_decl {
          fn_generic_decls.push(fg);
        }
        write_at_codes.push(info.write_at_code);
        field_names.push(field_name.clone());
      }

      let bg = builder_generics(&input.generics, &generic_params, &where_preds);
      let bsg = &bg.struct_generics;
      let big = &bg.impl_generics;
      let bpf = &bg.phantom_field;
      let bpi = &bg.phantom_init;
      let bwc = &bg.where_clause;
      let fn_make_generics = if fn_generic_decls.is_empty() {
        quote! {}
      } else {
        quote! { <#(#fn_generic_decls),*> }
      };

      quote! {
        impl #impl_generics #name #ty_generics #input_where {
          #vis fn make #fn_make_generics (#(#fn_params),*) -> impl ::nearest::Emit<Self> {
            struct __Builder #bsg {
              #bpf
              #(#struct_fields,)*
            }

            unsafe impl #big ::nearest::Emit<#name #ty_generics>
              for __Builder #bsg
            #bwc
            {
              unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
                #(#write_at_codes)*
              }
            }

            __Builder { #bpi #(#field_names,)* }
          }
        }
      }
    }
    Fields::Unnamed(unnamed) => {
      let mut fn_params = Vec::new();
      let mut struct_fields = Vec::new();
      let mut generic_params = Vec::new();
      let mut where_preds = Vec::new();
      let mut write_at_codes = Vec::new();
      let mut field_idents = Vec::new();
      let mut fn_generic_decls = Vec::new();

      for (i, f) in unnamed.unnamed.iter().enumerate() {
        let field_ty = &f.ty;
        let field_attrs = match parse_field_attrs(&f.attrs) {
          Ok(a) => a,
          Err(e) => return e.to_compile_error(),
        };
        let field_ident = format_ident!("f{}", i);
        let param_name = format_ident!("__F{}", i);
        let idx = syn::Index::from(i);
        let value_expr = quote! { self.#field_ident };
        let offset_expr = quote! { ::core::mem::offset_of!(#name #ty_generics, #idx) };
        let info =
          analyze_field(&value_expr, field_ty, &param_name, &offset_expr, field_attrs.into);

        let fn_pt = &info.fn_param_type;
        fn_params.push(quote! { #field_ident: #fn_pt });
        let bt = &info.builder_type;
        struct_fields.push(quote! { #field_ident: #bt });
        if let Some(gp) = info.generic_param {
          generic_params.push(gp);
        }
        if let Some(wp) = info.where_pred {
          where_preds.push(wp);
        }
        if let Some(fg) = info.fn_generic_decl {
          fn_generic_decls.push(fg);
        }
        write_at_codes.push(info.write_at_code);
        field_idents.push(field_ident);
      }

      let bg = builder_generics(&input.generics, &generic_params, &where_preds);
      let bsg = &bg.struct_generics;
      let big = &bg.impl_generics;
      let bpf = &bg.phantom_field;
      let bpi = &bg.phantom_init;
      let bwc = &bg.where_clause;
      let fn_make_generics = if fn_generic_decls.is_empty() {
        quote! {}
      } else {
        quote! { <#(#fn_generic_decls),*> }
      };

      quote! {
        impl #impl_generics #name #ty_generics #input_where {
          #vis fn make #fn_make_generics (#(#fn_params),*) -> impl ::nearest::Emit<Self> {
            struct __Builder #bsg {
              #bpf
              #(#struct_fields,)*
            }

            unsafe impl #big ::nearest::Emit<#name #ty_generics>
              for __Builder #bsg
            #bwc
            {
              unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
                #(#write_at_codes)*
              }
            }

            __Builder { #bpi #(#field_idents,)* }
          }
        }
      }
    }
    Fields::Unit => gen_emit_self(input),
  }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

fn gen_emit_enum(input: &DeriveInput, data: &DataEnum) -> TokenStream {
  if has_no_pointer_fields(&input.data) {
    return gen_emit_self(input);
  }

  let name = &input.ident;
  let vis = &input.vis;
  let (impl_generics, ty_generics, input_where) = input.generics.split_for_impl();

  let methods: Vec<_> = data
    .variants
    .iter()
    .enumerate()
    .map(|(idx, variant)| gen_variant_emitter(name, vis, &input.generics, idx, variant))
    .collect();

  quote! {
    impl #impl_generics #name #ty_generics #input_where {
      #(#methods)*
    }
  }
}

fn gen_variant_emitter(
  enum_name: &syn::Ident,
  vis: &syn::Visibility,
  generics: &syn::Generics,
  variant_idx: usize,
  variant: &Variant,
) -> TokenStream {
  let (_, ty_generics, _) = generics.split_for_impl();
  let vname = &variant.ident;
  let variant_attrs = match parse_variant_attrs(&variant.attrs) {
    Ok(a) => a,
    Err(e) => return e.to_compile_error(),
  };
  let method_name = variant_attrs
    .rename
    .unwrap_or_else(|| format_ident!("make_{}", to_snake_case(&vname.to_string())));
  let disc = variant_idx as u8;

  match &variant.fields {
    Fields::Named(named) => {
      let mut fn_params = Vec::new();
      let mut struct_fields = Vec::new();
      let mut generic_params = Vec::new();
      let mut where_preds = Vec::new();
      let mut write_at_codes = Vec::new();
      let mut field_names = Vec::new();
      let mut fn_generic_decls = Vec::new();

      for f in &named.named {
        let field_name = f.ident.as_ref().unwrap();
        let field_ty = &f.ty;
        let field_attrs = match parse_field_attrs(&f.attrs) {
          Ok(a) => a,
          Err(e) => return e.to_compile_error(),
        };
        let param_name = format_ident!("__{}", capitalize(&field_name.to_string()));
        let value_expr = quote! { self.#field_name };
        let offset_expr =
          quote! { ::core::mem::offset_of!(#enum_name #ty_generics, #vname.#field_name) };
        let info =
          analyze_field(&value_expr, field_ty, &param_name, &offset_expr, field_attrs.into);

        let fn_pt = &info.fn_param_type;
        fn_params.push(quote! { #field_name: #fn_pt });
        let bt = &info.builder_type;
        struct_fields.push(quote! { #field_name: #bt });
        if let Some(gp) = info.generic_param {
          generic_params.push(gp);
        }
        if let Some(wp) = info.where_pred {
          where_preds.push(wp);
        }
        if let Some(fg) = info.fn_generic_decl {
          fn_generic_decls.push(fg);
        }
        write_at_codes.push(info.write_at_code);
        field_names.push(field_name.clone());
      }

      let bg = builder_generics(generics, &generic_params, &where_preds);
      let bsg = &bg.struct_generics;
      let big = &bg.impl_generics;
      let bpf = &bg.phantom_field;
      let bpi = &bg.phantom_init;
      let bwc = &bg.where_clause;
      let fn_make_generics = if fn_generic_decls.is_empty() {
        quote! {}
      } else {
        quote! { <#(#fn_generic_decls),*> }
      };

      quote! {
        #vis fn #method_name #fn_make_generics (#(#fn_params),*) -> impl ::nearest::Emit<Self> {
          struct __Builder #bsg {
            #bpf
            #(#struct_fields,)*
          }

          unsafe impl #big ::nearest::Emit<#enum_name #ty_generics>
            for __Builder #bsg
          #bwc
          {
            unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
              unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
              #(#write_at_codes)*
            }
          }

          __Builder { #bpi #(#field_names,)* }
        }
      }
    }
    Fields::Unnamed(unnamed) => {
      let mut fn_params = Vec::new();
      let mut struct_fields = Vec::new();
      let mut generic_params = Vec::new();
      let mut where_preds = Vec::new();
      let mut write_at_codes = Vec::new();
      let mut field_idents = Vec::new();
      let mut fn_generic_decls = Vec::new();

      for (i, f) in unnamed.unnamed.iter().enumerate() {
        let field_ty = &f.ty;
        let field_attrs = match parse_field_attrs(&f.attrs) {
          Ok(a) => a,
          Err(e) => return e.to_compile_error(),
        };
        let field_ident = format_ident!("f{}", i);
        let param_name = format_ident!("__F{}", i);
        let idx = syn::Index::from(i);
        let value_expr = quote! { self.#field_ident };
        let offset_expr = quote! { ::core::mem::offset_of!(#enum_name #ty_generics, #vname.#idx) };
        let info =
          analyze_field(&value_expr, field_ty, &param_name, &offset_expr, field_attrs.into);

        let fn_pt = &info.fn_param_type;
        fn_params.push(quote! { #field_ident: #fn_pt });
        let bt = &info.builder_type;
        struct_fields.push(quote! { #field_ident: #bt });
        if let Some(gp) = info.generic_param {
          generic_params.push(gp);
        }
        if let Some(wp) = info.where_pred {
          where_preds.push(wp);
        }
        if let Some(fg) = info.fn_generic_decl {
          fn_generic_decls.push(fg);
        }
        write_at_codes.push(info.write_at_code);
        field_idents.push(field_ident);
      }

      let bg = builder_generics(generics, &generic_params, &where_preds);
      let bsg = &bg.struct_generics;
      let big = &bg.impl_generics;
      let bpf = &bg.phantom_field;
      let bpi = &bg.phantom_init;
      let bwc = &bg.where_clause;
      let fn_make_generics = if fn_generic_decls.is_empty() {
        quote! {}
      } else {
        quote! { <#(#fn_generic_decls),*> }
      };

      quote! {
        #vis fn #method_name #fn_make_generics (#(#fn_params),*) -> impl ::nearest::Emit<Self> {
          struct __Builder #bsg {
            #bpf
            #(#struct_fields,)*
          }

          unsafe impl #big ::nearest::Emit<#enum_name #ty_generics>
            for __Builder #bsg
          #bwc
          {
            unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
              unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
              #(#write_at_codes)*
            }
          }

          __Builder { #bpi #(#field_idents,)* }
        }
      }
    }
    Fields::Unit => {
      let bg = builder_generics(generics, &[], &[]);
      let bsg = &bg.struct_generics;
      let big = &bg.impl_generics;
      let bpf = &bg.phantom_field;
      let bpi = &bg.phantom_init;

      quote! {
        #vis fn #method_name() -> impl ::nearest::Emit<Self> {
          struct __Builder #bsg { #bpf }

          unsafe impl #big ::nearest::Emit<#enum_name #ty_generics>
            for __Builder #bsg
          {
            unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
              unsafe { nearest_p.write_flat(nearest_at, #disc as u8) };
            }
          }

          __Builder { #bpi }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// NearList write_at code generation
// ---------------------------------------------------------------------------

/// Generate the `write_at` code for a `NearList` field.
///
/// Allocates a single contiguous segment for all elements.
fn gen_near_list_write_at(
  value_expr: &TokenStream,
  inner: &syn::Type,
  at_expr: &TokenStream,
) -> TokenStream {
  quote! {
    {
      let mut nearest_iter = (#value_expr).into_iter();
      let nearest_len = nearest_iter.len() as u32;
      if nearest_len == 0 {
        unsafe {
          nearest_p.patch_list_header::<#inner>(
            #at_expr,
            ::nearest::__private::Pos::ZERO,
            0,
          );
        }
      } else {
        let nearest_seg_pos = nearest_p.alloc_segment::<#inner>(nearest_len);
        let nearest_values_offset = ::nearest::__private::segment_values_offset::<#inner>();
        for nearest_i in 0..nearest_len as usize {
          let nearest_item = nearest_iter.next().expect("ExactSizeIterator lied about length");
          unsafe {
            nearest_item.write_at(
              nearest_p,
              nearest_seg_pos.offset(nearest_values_offset + nearest_i * ::core::mem::size_of::<#inner>()),
            );
          }
        }
        unsafe {
          nearest_p.patch_list_header::<#inner>(
            #at_expr,
            nearest_seg_pos,
            nearest_len,
          );
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Self-emit for all-primitive types
// ---------------------------------------------------------------------------

fn gen_emit_self(input: &DeriveInput) -> TokenStream {
  let name = &input.ident;
  let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

  let field_types = collect_field_types(&input.data);
  let already_bounded = flat_bounded_param_names(&input.generics);
  let flat_predicates: Vec<_> = field_types
    .iter()
    .filter(|ty| !already_bounded.iter().any(|name| is_type_param_ident(ty, name)))
    .map(|ty| quote! { #ty: ::nearest::Flat })
    .collect();
  let combined_where = combine_where(where_clause, &flat_predicates);

  quote! {
    unsafe impl #impl_generics ::nearest::Emit<#name #ty_generics> for #name #ty_generics #combined_where {
      unsafe fn write_at(self, nearest_p: &mut impl ::nearest::Patch, nearest_at: ::nearest::__private::Pos) {
        unsafe { nearest_p.write_flat(nearest_at, self) };
      }
    }
  }
}
