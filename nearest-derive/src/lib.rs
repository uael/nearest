//! The `#[flat]` attribute of the [`nearest`](https://docs.rs/nearest) crate, re-exported as
//! `nearest::flat`. Use it from there.

use proc_macro2::{Group, Ident, Literal, Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Index, Lifetime, parse_macro_input, parse_quote};

/// Makes a struct or enum storable in a `nearest` region. See `nearest::flat`.
#[proc_macro_attribute]
pub fn flat(
  attr: proc_macro::TokenStream,
  item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
  let input = parse_macro_input!(item as DeriveInput);
  expand(attr.into(), input).unwrap_or_else(Error::into_compile_error).into()
}

fn expand(attr: TokenStream, mut input: DeriveInput) -> syn::Result<TokenStream> {
  if !attr.is_empty() {
    return Err(Error::new_spanned(attr, "`#[flat]` takes no arguments"));
  }
  let is_enum = matches!(input.data, Data::Enum(_));
  let mut has_repr = false;
  for a in input.attrs.iter().filter(|a| a.path().is_ident("repr")) {
    has_repr = true;
    let tokens = a.meta.require_list()?.tokens.clone();
    if is_enum && tokens.to_string() != "u8" {
      return Err(Error::new_spanned(a, "`#[flat]` enums are `#[repr(u8)]`, the default"));
    }
    if tokens.into_iter().any(|t| matches!(&t, TokenTree::Ident(i) if i == "packed")) {
      return Err(Error::new_spanned(a, "`#[flat]` types cannot be packed"));
    }
  }

  let name = &input.ident;
  let vis = &input.vis;
  let mut generics = input.generics.clone();
  for p in input.generics.type_params() {
    let id = &p.ident;
    generics.make_where_clause().predicates.push(parse_quote!(#id: ::nearest::Flat));
  }
  let (impl_g, ty_g, where_c) = generics.split_for_impl();
  let mut o = String::from("o");
  while input.generics.lifetimes().any(|l| l.lifetime.ident == o) {
    o.push('_');
  }
  let o = Lifetime::new(&format!("'{o}"), Span::call_site());
  let [sink, at, shift] = ["sink", "at", "shift"].map(|s| Ident::new(s, Span::mixed_site()));

  // A constructor: `make` around one `emit` per field, after an optional tag.
  let ctor =
    |fname: Ident, doc: String, fields: &Fields, tag: TokenStream, offs: Vec<TokenStream>| {
      let params = fields.iter().enumerate().map(|(i, f)| {
        f.ident.clone().unwrap_or_else(|| Ident::new(&format!("f{i}"), Span::mixed_site()))
      });
      let params: Vec<_> = params.collect();
      let tys = fields.iter().map(|f| &f.ty);
      let tys2 = tys.clone();
      // One parameter per field is the point of a constructor.
      let lint = (params.len() > 7).then(|| quote!(#[allow(clippy::too_many_arguments)]));
      quote! {
        #[doc = #doc]
        #lint
        #vis fn #fname<#o>(#(#params: impl ::nearest::Emit<#o, #tys>),*)
          -> impl ::nearest::Emit<#o, Self> {
          unsafe {
            ::nearest::__private::make(
              move |#sink: &mut ::nearest::__private::Sink<#o>, #at: usize| {
                #tag
                #(::nearest::Emit::<#o, #tys2>::emit(#params, #sink, #at + #offs);)*
              },
            )
          }
        }
      }
    };
  let store = |field: TokenStream, off: TokenStream| {
    let call = quote!(::nearest::Flat::store(#field, #sink, #at + #off, #shift));
    quote!(#call;)
  };

  let (mut mirrors, mut ctors) = (Vec::new(), Vec::new());
  let (plain, body) = match &input.data {
    Data::Union(u) => {
      return Err(Error::new(u.union_token.span, "`#[flat]` does not support unions"));
    }
    Data::Struct(s) => {
      let tys: Vec<_> = s.fields.iter().map(|f| &f.ty).collect();
      let offs: Vec<_> =
        s.fields.members().map(|m| quote!(::core::mem::offset_of!(Self, #m))).collect();
      if !s.fields.is_empty() {
        let doc = format!("Builds a `{name}` from its fields, in declaration order.");
        ctors.push(ctor(format_ident!("new"), doc, &s.fields, quote!(), offs.clone()));
      }
      let stores = s.fields.members().zip(offs).map(|(m, off)| store(quote!(&self.#m), off));
      let plain = quote! {
        true #(&& <#tys as ::nearest::Flat>::PLAIN)*
          && ::core::mem::size_of::<Self>() == 0 #(+ ::core::mem::size_of::<#tys>())*
      };
      (plain, quote!(#(#stores)*))
    }
    Data::Enum(e) => {
      if let Some(v) = e.variants.iter().nth(256) {
        return Err(Error::new_spanned(v, "`#[flat]` enums have at most 256 variants"));
      }
      if e.variants.is_empty() {
        return Err(Error::new_spanned(name, "`#[flat]` enums need at least one variant"));
      }
      let self_ty = quote!(#name #ty_g);
      let (mut arms, mut built) = (Vec::new(), Vec::new());
      for (i, v) in e.variants.iter().enumerate() {
        if let Some((eq, expr)) = &v.discriminant {
          return Err(Error::new_spanned(quote!(#eq #expr), "`#[flat]` numbers variants in order"));
        }
        let (vname, tag) = (&v.ident, Literal::u8_suffixed(i as u8));
        let binds: Vec<_> =
          (0..v.fields.len()).map(|k| format_ident!("f{k}", span = Span::mixed_site())).collect();
        let pat = match &v.fields {
          Fields::Named(f) => {
            let ids = f.named.iter().map(|f| &f.ident);
            quote!({ #(#ids: #binds),* })
          }
          Fields::Unnamed(_) => quote!((#(#binds),*)),
          Fields::Unit => quote!(),
        };
        let tag_store = store(quote!(&#tag), quote!(0));
        if matches!(v.fields, Fields::Unit) {
          arms.push(quote!(Self::#vname => { #tag_store }));
          continue;
        }
        // `#[repr(u8)]` lays each variant out as this `#[repr(C)]` struct.
        let mirror = format_ident!("__NearestMirror{}", vname);
        let tys = v.fields.iter().map(|f| unself(f.ty.to_token_stream(), &self_ty));
        mirrors.push(quote! {
          #[repr(C)]
          #[allow(dead_code)]
          struct #mirror #impl_g (u8, #(#tys,)* ::core::marker::PhantomData<fn() -> #self_ty>)
            #where_c;
        });
        let offs: Vec<_> = (1..=v.fields.len())
          .map(|k| {
            let k = Index::from(k);
            quote!(::core::mem::offset_of!(#mirror #ty_g, #k))
          })
          .collect();
        let doc = format!("Builds `{name}::{vname}` from its fields, in declaration order.");
        let tag_emit = quote!(::nearest::Emit::<#o, u8>::emit(#tag, #sink, #at););
        ctors.push(ctor(snake(vname), doc, &v.fields, tag_emit, offs.clone()));
        let stores = binds.iter().zip(offs).map(|(b, off)| store(quote!(#b), off));
        arms.push(quote!(Self::#vname #pat => { #tag_store #(#stores)* }));
        // Tells the dead-code lint that the variant is built, by the constructor.
        built.push(quote!(let _ = |#(#binds),*| Self::#vname #pat;));
      }
      let unit = e.variants.iter().all(|v| matches!(v.fields, Fields::Unit));
      (quote!(#unit), quote!(#(#built)* match self { #(#arms)* }))
    }
  };

  input.attrs.insert(0, parse_quote!(#[pin_v2]));
  if !has_repr {
    input
      .attrs
      .insert(0, if is_enum { parse_quote!(#[repr(u8)]) } else { parse_quote!(#[repr(C)]) });
  }
  Ok(quote! {
    #input
    const _: () = {
      #(#mirrors)*
      unsafe impl #impl_g ::nearest::Flat for #name #ty_g #where_c {
        const PLAIN: bool = #plain;
        unsafe fn store(
          &self,
          #sink: &mut ::nearest::__private::Sink<'_>,
          #at: usize,
          #shift: ::core::option::Option<i32>,
        ) {
          const {
            let drops = ::core::mem::needs_drop::<Self>();
            ::core::assert!(!drops, "`#[flat]` types cannot have drop glue");
          };
          unsafe { #body }
        }
      }
      impl #impl_g #name #ty_g #where_c {
        #(#ctors)*
      }
    };
  })
}

/// `Self` in a field type, spelled out for use in a mirror struct.
fn unself(tokens: TokenStream, self_ty: &TokenStream) -> TokenStream {
  let tree = |t| match t {
    TokenTree::Ident(i) if i == "Self" => self_ty.clone(),
    TokenTree::Group(g) => {
      let mut n = Group::new(g.delimiter(), unself(g.stream(), self_ty));
      n.set_span(g.span());
      TokenTree::Group(n).into()
    }
    t => t.into(),
  };
  tokens.into_iter().map(tree).collect()
}

/// `IntLit` -> `int_lit`; keywords become raw identifiers.
fn snake(variant: &Ident) -> Ident {
  let cs: Vec<char> = variant.to_string().chars().collect();
  let mut s = String::new();
  for (i, &c) in cs.iter().enumerate() {
    let next_lower = cs.get(i + 1).is_some_and(|n| n.is_lowercase());
    if i > 0 && c.is_uppercase() && (!cs[i - 1].is_uppercase() && cs[i - 1] != '_' || next_lower) {
      s.push('_');
    }
    s.extend(c.to_lowercase());
  }
  match &*s {
    "self" | "super" | "crate" => format_ident!("{s}_", span = variant.span()),
    // syn accepts `gen`, which edition 2024 reserves.
    _ if s == "gen" || syn::parse_str::<Ident>(&s).is_err() => Ident::new_raw(&s, variant.span()),
    _ => Ident::new(&s, variant.span()),
  }
}
