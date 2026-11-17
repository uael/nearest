//! The `#[flat]` attribute of the [`nearest`](https://docs.rs/nearest) crate, re-exported as
//! `nearest::flat`. Use it from there.

use proc_macro2::{Group, Ident, Literal, Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{
  Data, DeriveInput, Error, Field, Fields, GenericArgument, GenericParam, Index, Lifetime, Member,
  PathArguments, Type, parse_macro_input, parse_quote,
};

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
  let self_ty = quote!(#name #ty_g);
  let mut a = String::from("a");
  while input.generics.lifetimes().any(|l| l.lifetime.ident == a) {
    a.push('_');
  }
  let lt = Lifetime::new(&format!("'{a}"), Span::call_site());
  let [sink, at, shift, buf] =
    ["sink", "at", "shift", "buf"].map(|s| Ident::new(s, Span::mixed_site()));
  let np = quote!(::nearest::__private);
  let handle = format_ident!("__NearestEdit");

  // A constructor: `make` around one `emit` per field, after an optional tag. A field written
  // `Near<X>` takes a builder of `X` and allocates the target.
  let ctor =
    |fname: Ident, doc: String, fields: &Fields, tag: TokenStream, offs: Vec<TokenStream>| {
      let params: Vec<_> = (fields.iter().enumerate())
        .map(|(i, f)| {
          f.ident.clone().unwrap_or_else(|| Ident::new(&format!("f{i}"), Span::mixed_site()))
        })
        .collect();
      let (mut ptys, mut vals) = (Vec::new(), Vec::new());
      for (f, p) in fields.iter().zip(&params) {
        if let Some(x) = near_target(&f.ty) {
          ptys.push(x.to_token_stream());
          vals.push(quote!(::nearest::Near::<#x>::new(#p)));
        } else {
          ptys.push(f.ty.to_token_stream());
          vals.push(p.to_token_stream());
        }
      }
      let tys = fields.iter().map(|f| &f.ty);
      // One parameter per field is the point of a constructor.
      let lint = (params.len() > 7).then(|| quote!(#[allow(clippy::too_many_arguments)]));
      quote! {
        #[doc = #doc]
        #lint
        #vis fn #fname(#(#params: impl ::nearest::Emit<#ptys>),*) -> impl ::nearest::Emit<Self> {
          #np::fits::<Self>();
          unsafe {
            #np::make(move |#sink: &mut #np::Sink, #at: usize| {
              #tag
              #(::nearest::Emit::<#tys>::emit(#vals, #sink, #at + #offs);)*
            })
          }
        }
      }
    };
  let store = |field: TokenStream, off: &TokenStream| quote!(::nearest::Flat::store(#field, #sink, #at + #off, #shift););
  // A method of a handle whose `.0` is the `Raw` of the value, returning a field's handle.
  let field_method = |f: &Field, i: usize, fvis: TokenStream, off: &TokenStream| {
    let mname = match &f.ident {
      Some(id) if id == "set" || id == "replace" => format_ident!("{id}_"),
      Some(id) => id.clone(),
      None => format_ident!("_{i}"),
    };
    let fty = unself(f.ty.to_token_stream(), &self_ty);
    let label = f.ident.as_ref().map_or_else(|| i.to_string(), Ident::to_string);
    let doc = format!("The handle of `{label}`.");
    quote! {
      #[doc = #doc]
      #fvis fn #mname(&mut self) -> &mut ::nearest::Edit<#fty> {
        unsafe { self.0.field::<#fty>(#off) }
      }
    }
  };

  // What differs between structs and enums.
  let (mut mirrors, mut ctors, mut methods, mut extra) = (vec![], vec![], vec![], vec![]);
  let mut view = None;
  let (plain, body) = match &input.data {
    Data::Union(u) => {
      return Err(Error::new(u.union_token.span, "`#[flat]` does not support unions"));
    }
    Data::Struct(s) => {
      let tys: Vec<_> = s.fields.iter().map(|f| &f.ty).collect();
      let members: Vec<Member> = s.fields.members().collect();
      let offs: Vec<_> =
        members.iter().map(|m| quote!(::core::mem::offset_of!(Self, #m))).collect();
      if !s.fields.is_empty() {
        let doc = format!("Builds a `{name}` from its fields, in declaration order.");
        ctors.push(ctor(format_ident!("new"), doc, &s.fields, quote!(), offs.clone()));
      }
      for (i, (f, m)) in s.fields.iter().zip(&members).enumerate() {
        let off = quote!(::core::mem::offset_of!(#self_ty, #m));
        methods.push(field_method(f, i, f.vis.to_token_stream(), &off));
      }
      let stores = members.iter().zip(&offs).map(|(m, off)| store(quote!(&self.#m), off));
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
      let mut_name = format_ident!("{name}Mut");
      let (mut arms, mut built, mut views, mut view_arms) = (vec![], vec![], vec![], vec![]);
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
        let tag_store = store(quote!(&#tag), &quote!(0));
        let vdoc = format!("`{name}::{vname}`.");
        let view_arm = quote!(#name::#vname { .. } => #mut_name::#vname);
        if matches!(v.fields, Fields::Unit) {
          arms.push(quote!(Self::#vname => { #tag_store }));
          views.push(quote!(#[doc = #vdoc] #vname));
          view_arms.push(view_arm);
          continue;
        }
        // `#[repr(u8)]` lays each variant out as this `#[repr(C)]` struct.
        let mirror = format_ident!("__NearestMirror{}", vname);
        let tys: Vec<_> =
          v.fields.iter().map(|f| unself(f.ty.to_token_stream(), &self_ty)).collect();
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
        let tag_emit = quote!(::nearest::Emit::<u8>::emit(#tag, #sink, #at););
        ctors.push(ctor(snake(vname), doc, &v.fields, tag_emit, offs.clone()));
        let stores = binds.iter().zip(&offs).map(|(b, off)| store(quote!(#b), off));
        arms.push(quote!(Self::#vname #pat => { #tag_store #(#stores)* }));
        // Tells the dead-code lint that the variant is built, by the constructor.
        built.push(quote!(let _ = |#(#binds),*| Self::#vname #pat;));
        if let ([ty], [off]) = (&tys[..], &offs[..]) {
          views.push(quote!(#[doc = #vdoc] #vname(&#lt mut ::nearest::Edit<#ty>)));
          view_arms.push(quote!(#view_arm(unsafe { self.0.field::<#ty>(#off) })));
          continue;
        }
        // Several fields: one handle with a method per field, named through `Variant`. It wraps
        // the `Raw` of the mirror, which is not `Flat`: even code that can reach its field gets
        // no `set` or `replace` that could change the variant under the handle.
        let payload = format_ident!("__NearestEdit{}", vname);
        let fmethods = (v.fields.iter().zip(&offs).enumerate())
          .map(|(k, (f, off))| field_method(f, k, vis.to_token_stream(), off));
        let pdoc = format!("The fields of `{name}::{vname}`.");
        extra.push(quote! {
          #[doc = #pdoc]
          #[repr(transparent)]
          #vis struct #payload #impl_g (#np::Raw<#mirror #ty_g>) #where_c;
          impl #impl_g #np::Variant<#tag> for #self_ty #where_c {
            type Edit = #payload #ty_g;
          }
          impl #impl_g #payload #ty_g #where_c {
            #(#fmethods)*
          }
        });
        views.push(quote!(#[doc = #vdoc] #vname(&#lt mut <#self_ty as #np::Variant<#tag>>::Edit)));
        view_arms.push(quote! {
          #view_arm(unsafe { &mut *(::core::ptr::from_mut(&mut self.0) as *mut #payload #ty_g) })
        });
      }
      if !built.is_empty() {
        let mut mg = generics.clone();
        mg.params.insert(0, GenericParam::Lifetime(parse_quote!(#lt)));
        for p in input.generics.type_params() {
          let id = &p.ident;
          mg.make_where_clause().predicates.push(parse_quote!(#id: #lt));
        }
        let (m_impl, m_ty, m_where) = mg.split_for_impl();
        let doc = format!(
          "The variant of a `{name}` in a region, holding the handles of its fields. Returned \
           by `as_mut` on `Edit<{name}>`."
        );
        methods.push(quote! {
          /// The variant, holding the handles of its fields.
          #vis fn as_mut<#lt>(&#lt mut self) -> #mut_name #m_ty {
            match &**self {
              #(#view_arms,)*
            }
          }
        });
        view = Some(quote! {
          #[doc = #doc]
          #vis enum #mut_name #m_impl #m_where {
            #(#views,)*
          }
        });
      }
      let unit = e.variants.iter().all(|v| matches!(v.fields, Fields::Unit));
      (quote!(#unit), quote!(#(#built)* match self { #(#arms)* }))
    }
  };

  let mut copy_g = generics.clone();
  copy_g.make_where_clause().predicates.push(parse_quote!(for<#lt> #self_ty: ::core::marker::Copy));
  let copy_where = copy_g.split_for_impl().2;
  let handle_doc = format!("The handle of a `{name}` in a region: `Edit<{name}>`.");
  if !has_repr {
    let repr = if is_enum { parse_quote!(#[repr(u8)]) } else { parse_quote!(#[repr(C)]) };
    input.attrs.insert(0, repr);
  }
  Ok(quote! {
    #input
    #view
    const _: () = {
      #(#mirrors)*
      unsafe impl #impl_g ::nearest::Flat for #self_ty #where_c {
        type Edit = #handle #ty_g;
        const PLAIN: bool = {
          #np::fits::<Self>();
          #plain
        };
        unsafe fn store(
          &self,
          #sink: &mut #np::Sink,
          #at: usize,
          #shift: ::core::option::Option<i32>,
        ) {
          #np::fits::<Self>();
          unsafe { #body }
        }
        unsafe fn edit<#lt>(#buf: *mut #np::Buf, #at: usize) -> &#lt mut Self::Edit {
          #np::fits::<Self>();
          unsafe {
            &mut *(::core::ptr::from_mut(#np::Raw::<Self>::new(#buf, #at)) as *mut #handle #ty_g)
          }
        }
      }
      impl #impl_g #self_ty #where_c {
        #(#ctors)*
      }
      #[doc = #handle_doc]
      #[repr(transparent)]
      #vis struct #handle #impl_g (#np::Raw<#self_ty>) #where_c;
      impl #impl_g #handle #ty_g #where_c {
        /// Replaces the value with `value`.
        #vis fn set(&mut self, value: impl ::nearest::Emit<#self_ty>) {
          self.0.set(value);
        }
        /// Replaces the value with what `f` makes of the old one.
        #vis fn replace<#lt, __NearestE: ::nearest::Emit<#self_ty>>(
          &#lt mut self,
          f: impl ::core::ops::FnOnce(::nearest::Old<#lt, #self_ty>) -> __NearestE,
        ) {
          self.0.replace(f);
        }
        #(#methods)*
      }
      impl #impl_g ::core::ops::Deref for #handle #ty_g #where_c {
        type Target = #self_ty;
        fn deref(&self) -> &#self_ty {
          self.0.get()
        }
      }
      impl #impl_g ::core::ops::DerefMut for #handle #ty_g #copy_where {
        fn deref_mut(&mut self) -> &mut #self_ty {
          self.0.get_mut()
        }
      }
      #(#extra)*
    };
  })
}

/// `X` if `ty` is written `Near<X>` (any path ending in `Near` with one type argument) and `X`
/// is not a slice or `str`.
fn near_target(ty: &Type) -> Option<&Type> {
  let ty = match ty {
    Type::Group(g) => &*g.elem,
    Type::Paren(p) => &*p.elem,
    ty => ty,
  };
  let Type::Path(p) = ty else { return None };
  let last = p.path.segments.last()?;
  let PathArguments::AngleBracketed(args) = &last.arguments else { return None };
  if last.ident != "Near" || args.args.len() != 1 {
    return None;
  }
  let Some(GenericArgument::Type(x)) = args.args.first() else { return None };
  match x {
    Type::Slice(_) => None,
    Type::Path(s) if s.qself.is_none() && s.path.is_ident("str") => None,
    x => Some(x),
  }
}

/// `Self` in a field type, spelled out for use outside the type's own `impl`.
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
