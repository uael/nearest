//! The `#[flat]` attribute of the [`nearest`](https://docs.rs/nearest) crate, re-exported as
//! `nearest::flat`. Use it from there.

use std::mem;

use proc_macro2::{Group, Ident, Literal, Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{
  Attribute, Data, DeriveInput, Error, Field, Fields, GenericArgument, GenericParam, Index,
  Lifetime, LitStr, Member, PathArguments, Type, ext::IdentExt, meta::ParseNestedMeta,
  parse::Parser, parse_macro_input, parse_quote, spanned::Spanned,
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
  // `#[flat(..)]` is read here and left out of the output: on the type it says nothing, on a
  // variant it may rename the constructor, on a field it may make its parameter `impl Into`.
  syn::meta::parser(read_args(&mut Args::default(), Place::Type)).parse2(attr)?;
  take_args(&mut input.attrs, Place::Type)?;
  let intos = |fields: &mut Fields| -> syn::Result<Vec<bool>> {
    (fields.iter_mut())
      .map(|f| match take_args(&mut f.attrs, Place::Field)?.into {
        Some(span) if is_pointer(&f.ty) => {
          let msg = "`into` is for plain data: the parameter of a pointer takes a builder of \
                     its pointee";
          Err(Error::new(span, msg))
        }
        into => Ok(into.is_some()),
      })
      .collect()
  };
  // Per variant (one entry for a struct): the new name of the constructor, and which of its
  // parameters take `impl Into`.
  let (mut renames, mut into) = (vec![], vec![]);
  match &mut input.data {
    Data::Enum(e) => {
      for v in &mut e.variants {
        into.push(intos(&mut v.fields)?);
        renames.push(take_args(&mut v.attrs, Place::Variant)?.rename);
      }
    }
    Data::Struct(s) => into.push(intos(&mut s.fields)?),
    Data::Union(_) => {}
  }
  // A type marked `#[pin_v2]` cannot have the bounded `Unpin` impl written below, which a pinned
  // value of a type that holds a pointer needs. One that another macro adds after this one runs
  // meets the impl instead, and the compiler rejects the two.
  if let Some(a) = input.attrs.iter().find(|a| a.path().is_ident("pin_v2")) {
    let msg = "a `#[flat]` type cannot be `#[pin_v2]`: it is `Unpin` exactly when its fields are, \
               by an impl that such a type cannot have";
    return Err(Error::new_spanned(a, msg));
  }
  let is_enum = matches!(input.data, Data::Enum(_));
  // Whether the layout is already fixed: `u8` for an enum, `C` or `transparent` for a struct.
  let mut has_repr = false;
  for a in input.attrs.iter().filter(|a| a.path().is_ident("repr")) {
    let tokens = a.meta.require_list()?.tokens.clone();
    if is_enum && tokens.to_string() != "u8" {
      return Err(Error::new_spanned(a, "`#[flat]` enums are `#[repr(u8)]`, the default"));
    }
    for t in tokens {
      match &t {
        TokenTree::Ident(i) if i == "packed" => {
          return Err(Error::new_spanned(a, "`#[flat]` types cannot be packed"));
        }
        TokenTree::Ident(i) if is_enum || i == "C" || i == "transparent" => has_repr = true,
        _ => {}
      }
    }
  }

  if let Some(l) = input.generics.lifetimes().next() {
    return Err(Error::new_spanned(l, "`#[flat]` types are `'static`: no lifetime parameters"));
  }

  // Every type parameter of the type is `Pun`, added to the bounds that the user wrote on it: the
  // type has a layout only then. On the type, every impl has the bound, those that the derives of
  // `core` write too (`Debug` reads the pointees, and the `Deref` of a pointer requires its pointee
  // to be `Pun`), so the user does not write it.
  for param in input.generics.type_params_mut() {
    param.bounds.push(parse_quote!(::nearest::puns::Pun));
  }
  let name = &input.ident;
  let vis = &input.vis;
  let mut generics = input.generics.clone();
  // Every type parameter is `Flat` in the impls of this crate, added to the bounds of the type: a
  // type bounded in two places is one that clippy sees twice.
  for param in generics.type_params_mut() {
    param.bounds.push(parse_quote!(::nearest::Flat));
  }
  let (impl_g, ty_g, where_c) = generics.split_for_impl();
  let self_ty = quote!(#name #ty_g);
  let type_params: Vec<_> = input.generics.type_params().map(|p| &p.ident).collect();
  // `'a`: the borrow of a handle; `'e`: the edit.
  let [lt, e] = ["'a", "'e"].map(|name| Lifetime::new(name, Span::call_site()));
  let mut handle_g = generics.clone();
  handle_g.params.insert(0, GenericParam::Lifetime(parse_quote!(#e)));
  let (h_impl, h_ty, h_where) = handle_g.split_for_impl();
  let [sink, at, buf, base, d] =
    ["sink", "at", "buf", "base", "d"].map(|s| Ident::new(s, Span::mixed_site()));
  let np = quote!(::nearest::__private);
  let pin = quote!(::core::pin::Pin);
  let handle = format_ident!("__NearestEdit");
  let pin_name = format_ident!("{name}Pin");
  let mut pin_g = generics.clone();
  pin_g.params.insert(0, GenericParam::Lifetime(parse_quote!(#lt)));
  let (p_impl, p_ty, p_where) = pin_g.split_for_impl();

  // A constructor: `make` around one `emit` per field, after an optional tag. A field written
  // `Near<..>` takes what `Pointer` says: for `Near<X>` (or `Near<X, O>`) a builder of `X`, whose
  // pointee the constructor allocates. A field marked `#[flat(into)]` takes `impl Into<FieldType>`.
  // What the parameter of the field `f` takes, and how its value `p` is written: for a field
  // written `Near<X>` (or `Near<X, O>`) a builder of `X`, whose pointee is allocated; for one
  // marked `#[flat(into)]` anything that converts to the field; else a builder of the field.
  let param = |f: &Field, into: bool, p: &Ident| {
    let ty = &f.ty;
    match near_pointee(ty) {
      _ if into => {
        (quote!(::core::convert::Into<#ty>), quote!(::core::convert::Into::<#ty>::into(#p)))
      }
      // A type parameter is a sized pointee; naming it lets the argument infer it.
      Some(Type::Path(x)) if type_params.iter().any(|t| x.path.is_ident(*t)) => {
        (quote!(::nearest::Emit<#x>), quote!(<#ty>::new(#p)))
      }
      Some(_) => (
        quote!(::nearest::Emit<<#ty as #np::Pointer>::Arg>),
        quote!(<#ty as #np::Pointer>::wrap(#p)),
      ),
      None => (quote!(::nearest::Emit<#ty>), p.to_token_stream()),
    }
  };
  let ctor = |fname: Ident,
              doc: String,
              fields: &Fields,
              into: &[bool],
              tag: TokenStream,
              offs: Vec<TokenStream>| {
    let params: Vec<_> = (fields.iter().enumerate())
      .map(|(i, f)| {
        f.ident.clone().unwrap_or_else(|| Ident::new(&format!("f{i}"), Span::mixed_site()))
      })
      .collect();
    let (mut ptys, mut vals) = (Vec::new(), Vec::new());
    for ((f, p), into) in fields.iter().zip(&params).zip(into) {
      let (bound, val) = param(f, *into, p);
      ptys.push(quote!(impl #bound));
      vals.push(val);
    }
    let tys = fields.iter().map(|f| &f.ty);
    // One parameter per field is the point of a constructor.
    let lint = (params.len() > 7).then(|| quote!(#[allow(clippy::too_many_arguments)]));
    quote! {
      #[doc = #doc]
      #lint
      #vis fn #fname(#(#params: #ptys),*) -> impl ::nearest::Emit<Self> {
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
  // A method of a handle whose `.0` is the `Raw` of the value, returning a field's handle.
  let field_method = |f: &Field, i: usize, fvis: TokenStream, off: &TokenStream| {
    let mname = match &f.ident {
      // The handle's own methods keep their names.
      Some(id) if ["set", "replace", "pin", "update"].iter().any(|m| id == m) => {
        format_ident!("{id}_")
      }
      Some(id) => id.clone(),
      None => format_ident!("_{i}"),
    };
    let fty = unself(f.ty.to_token_stream(), &self_ty);
    let label = f.ident.as_ref().map_or_else(|| i.to_string(), Ident::to_string);
    let doc = format!("The handle of `{label}`.");
    quote! {
      #[doc = #doc]
      #fvis fn #mname(&mut self) -> <#fty as ::nearest::Flat>::Ref<'_, #e> {
        <#fty as ::nearest::Flat>::reference(unsafe { self.0.field::<#fty>(#off) })
      }
    }
  };

  // What differs between structs and enums.
  // `XInit`: the fields of a struct as builders, named, in any order; written as `X::new` writes
  // them, which it calls (the same machine code). Its type parameters are the builders.
  let init_name = format_ident!("{name}Init");
  let init_struct = |fields: &Fields, into: &[bool]| {
    let members: Vec<Member> = fields.members().collect();
    let binds: Vec<Ident> =
      (0..fields.len()).map(|k| format_ident!("f{k}", span = Span::mixed_site())).collect();
    let gs: Vec<Ident> = (0..fields.len()).map(|k| format_ident!("__F{k}")).collect();
    let mut init_g = generics.clone();
    let (mut vals, mut tys) = (vec![], vec![]);
    for ((f, g), (into, b)) in fields.iter().zip(&gs).zip(into.iter().zip(&binds)) {
      // Outside the impl of `X`, `Self` in a field type is spelled out.
      let (bound, val) = param(f, *into, b);
      let bound = unself(bound, &self_ty);
      init_g.params.push(parse_quote!(#g: #bound));
      vals.push(unself(val, &self_ty));
      tys.push(unself(f.ty.to_token_stream(), &self_ty));
    }
    let (i_impl, _, i_where) = init_g.split_for_impl();
    let doc = format!(
      "The fields of a `{name}`, as builders, by name and in any order: written as \
       `{name}::new` writes them."
    );
    let decl = if let Fields::Named(_) = fields {
      let docs = members.iter().map(|m| format!("`{}`.", m.to_token_stream()));
      quote!(#vis struct #init_name<#(#gs),*> { #(#[doc = #docs] #vis #members: #gs,)* })
    } else {
      quote!(#vis struct #init_name<#(#gs),*>(#(#vis #gs),*);)
    };
    quote! {
      #[doc = #doc]
      #decl
      // SAFETY: every field written at its offset, as `X::new` writes them.
      unsafe impl #i_impl ::nearest::Emit<#self_ty> for #init_name<#(#gs),*> #i_where {
        unsafe fn emit(self, #sink: &mut #np::Sink, #at: usize) {
          #np::fits::<#self_ty>();
          let Self { #(#members: #binds),* } = self;
          // SAFETY: each field at its offset, as `new` writes it.
          unsafe {
            #(::nearest::Emit::<#tys>::emit(
              #vals,
              #sink,
              #at + ::core::mem::offset_of!(#self_ty, #members),
            );)*
          }
        }
      }
    }
  };
  let mut init = None;
  let (mut mirrors, mut ctors, mut methods, mut extra) = (vec![], vec![], vec![], vec![]);
  let mut view = None;
  // The declaration of `XPin` and the body of `project`, and every field type of the type.
  let (mut pinned, mut field_tys) = (None, vec![]);
  // What `edit` holds besides building the handle: for an enum, a closure per variant with
  // fields that builds it, never called, which tells the dead-code lint that the variant is
  // built (by its constructor, which it cannot see through `make`).
  // The bodies of `store` and `shift`, and the types of all fields.
  let (built, store, shift) = match &input.data {
    Data::Union(u) => {
      return Err(Error::new(u.union_token.span, "`#[flat]` does not support unions"));
    }
    Data::Struct(s) => {
      let tys: Vec<_> = s.fields.iter().map(|f| f.ty.to_token_stream()).collect();
      let members: Vec<Member> = s.fields.members().collect();
      let offs: Vec<_> =
        members.iter().map(|m| quote!(::core::mem::offset_of!(Self, #m))).collect();
      if !s.fields.is_empty() {
        let doc = format!("Builds a `{name}` from its fields, in declaration order.");
        ctors.push(ctor(format_ident!("new"), doc, &s.fields, &into[0], quote!(), offs));
        init = Some(init_struct(&s.fields, &into[0]));
      }
      for (i, (f, m)) in s.fields.iter().zip(&members).enumerate() {
        let off = quote!(::core::mem::offset_of!(#self_ty, #m));
        methods.push(field_method(f, i, f.vis.to_token_stream(), &off));
      }
      if !s.fields.is_empty() {
        let binds: Vec<_> =
          (0..tys.len()).map(|k| format_ident!("f{k}", span = Span::mixed_site())).collect();
        let (fvis, ftys): (Vec<_>, Vec<_>) =
          (s.fields.iter().map(|f| (&f.vis, unself(f.ty.to_token_stream(), &self_ty)))).unzip();
        let fdocs = members.iter().map(|m| format!("`{}`, pinned.", m.to_token_stream()));
        let (decl, make) = match &s.fields {
          Fields::Named(_) => (
            quote!(#p_where { #(#[doc = #fdocs] #fvis #members: #pin<&#lt mut #ftys>,)* }),
            quote!(#pin_name { #(#members: #pin::new_unchecked(#binds),)* }),
          ),
          _ => (
            quote!((#(#[doc = #fdocs] #fvis #pin<&#lt mut #ftys>,)*) #p_where;),
            quote!(#pin_name(#(#pin::new_unchecked(#binds),)*)),
          ),
        };
        let doc = format!("The fields of a pinned `{name}`, pinned. Returned by `project`.");
        pinned = Some((
          quote!(#[doc = #doc] #vis struct #pin_name #p_impl #decl),
          quote!(let Self { #(#members: #binds,)* } = self.get_unchecked_mut(); #make),
        ));
      }
      field_tys.extend(tys.iter().map(|ty| unself(ty.clone(), &self_ty)));
      let offs: Vec<_> =
        members.iter().map(|m| quote!(::core::mem::offset_of!(Self, #m))).collect();
      (
        quote!(),
        quote!(#(::nearest::Flat::store(&self.#members, #sink, #at + #offs);)*),
        quote!(#(<#tys as ::nearest::Flat>::shift(#base, #at + #offs, #d);)*),
      )
    }
    Data::Enum(data) => {
      if let Some(v) = data.variants.iter().nth(256) {
        return Err(Error::new_spanned(v, "`#[flat]` enums have at most 256 variants"));
      }
      if data.variants.is_empty() {
        return Err(Error::new_spanned(name, "`#[flat]` enums need at least one variant"));
      }
      let mut_name = format_ident!("{name}Mut");
      let (mut built, mut views, mut view_arms) = (vec![], vec![], vec![]);
      let (mut store_arms, mut shift_arms) = (vec![], vec![]);
      let mut ctor_names = vec![];
      let (mut pin_variants, mut pin_arms) = (vec![], vec![]);
      for (i, v) in data.variants.iter().enumerate() {
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
        let vdoc = format!("`{name}::{vname}`.");
        let view_arm = quote!(#name::#vname { .. } => #mut_name::#vname);
        let ftys = v.fields.iter().map(|f| unself(f.ty.to_token_stream(), &self_ty));
        let pins = quote!(#(#pin::new_unchecked(#binds)),*);
        let (pin_decl, pin_make) = match &v.fields {
          Fields::Named(f) => {
            let ids: Vec<_> = f.named.iter().map(|f| &f.ident).collect();
            let fdocs = ids.iter().map(|id| format!("`{}`, pinned.", id.to_token_stream()));
            (
              quote!({ #(#[doc = #fdocs] #ids: #pin<&#lt mut #ftys>),* }),
              quote!({ #(#ids: #pin::new_unchecked(#binds)),* }),
            )
          }
          Fields::Unnamed(_) => (quote!((#(#pin<&#lt mut #ftys>),*)), quote!((#pins))),
          Fields::Unit => (quote!(), quote!()),
        };
        pin_variants.push(quote!(#[doc = #vdoc] #vname #pin_decl));
        pin_arms.push(quote!(Self::#vname #pat => #pin_name::#vname #pin_make));
        let tag_store = quote!(::nearest::Flat::store(&#tag, #sink, #at););
        if matches!(v.fields, Fields::Unit) {
          store_arms.push(quote!(Self::#vname => { #tag_store }));
          if let Some(r) = &renames[i] {
            return Err(Error::new_spanned(r, "a unit variant has no constructor to rename"));
          }
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
        field_tys.extend(tys.iter().cloned());
        let offs: Vec<_> = (1..=v.fields.len())
          .map(|k| {
            let k = Index::from(k);
            quote!(::core::mem::offset_of!(#mirror #ty_g, #k))
          })
          .collect();
        store_arms.push(quote! {
          Self::#vname #pat => { #tag_store #(::nearest::Flat::store(#binds, #sink, #at + #offs);)* }
        });
        shift_arms
          .push(quote!(#tag => { #(<#tys as ::nearest::Flat>::shift(#base, #at + #offs, #d);)* }));
        let doc = format!("Builds `{name}::{vname}` from its fields, in declaration order.");
        let tag_emit = quote!(::nearest::Emit::<u8>::emit(#tag, #sink, #at););
        let cname = renames[i].clone().unwrap_or_else(|| snake(vname));
        let at = renames[i].as_ref().unwrap_or(vname);
        if cname.unraw() == "project" {
          let msg = "`project` is the method `#[flat]` generates for pinned values";
          return Err(Error::new_spanned(at, msg));
        }
        if ctor_names.iter().any(|c: &Ident| c.unraw() == cname.unraw()) {
          let msg = format!(
            "another variant already has the constructor `{cname}`: name one with \
             `#[flat(rename = \"..\")]`"
          );
          return Err(Error::new_spanned(at, msg));
        }
        ctor_names.push(cname.clone());
        ctors.push(ctor(cname, doc, &v.fields, &into[i], tag_emit, offs.clone()));
        // Tells the dead-code lint that the variant is built, by the constructor.
        built.push(quote!(let _ = |#(#binds),*| Self::#vname #pat;));
        if let ([ty], [off]) = (&tys[..], &offs[..]) {
          views.push(quote!(#[doc = #vdoc] #vname(<#ty as ::nearest::Flat>::Ref<#lt, #e>)));
          view_arms.push(quote! {
            #view_arm(<#ty as ::nearest::Flat>::reference(unsafe { self.0.field::<#ty>(#off) }))
          });
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
          #vis struct #payload #h_impl (#np::Raw<#e, #mirror #ty_g>) #h_where;
          impl #impl_g #np::Variant<#tag> for #self_ty #where_c {
            type Edit<#e> = #payload #h_ty;
          }
          impl #h_impl #payload #h_ty #h_where {
            #(#fmethods)*
          }
        });
        views.push(quote! {
          #[doc = #vdoc]
          #vname(&#lt mut <#self_ty as #np::Variant<#tag>>::Edit<#e>)
        });
        view_arms.push(quote! {
          #view_arm(unsafe { &mut *(::core::ptr::from_mut(&mut self.0) as *mut #payload #h_ty) })
        });
      }
      if !built.is_empty() {
        let mut mg = generics.clone();
        mg.params.insert(0, GenericParam::Lifetime(parse_quote!(#e: #lt)));
        mg.params.insert(0, GenericParam::Lifetime(parse_quote!(#lt)));
        let (m_impl, m_ty, m_where) = mg.split_for_impl();
        let doc = format!(
          "The variant of a `{name}` in a region, holding the handles of its fields. Returned \
           by `as_mut` on `Edit<'e, {name}>`."
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
        let doc = format!(
          "The variant of a pinned `{name}`, with its fields pinned. Returned by `project`."
        );
        pinned = Some((
          quote!(#[doc = #doc] #vis enum #pin_name #p_impl #p_where { #(#pin_variants,)* }),
          quote!(match self.get_unchecked_mut() { #(#pin_arms,)* }),
        ));
      }
      (
        quote!(#(#built)*),
        quote!(match self { #(#store_arms)* }),
        // The tag of a valid value, read from the buffer: a `u8` at the start (`repr(u8)`).
        quote! {
          match unsafe { #base.add(#at).read() } {
            #(#shift_arms)*
            _ => {}
          }
        },
      )
    }
  };
  let mut copy_g = handle_g.clone();
  copy_g.make_where_clause().predicates.push(parse_quote!(for<#lt> #self_ty: ::core::marker::Copy));
  let copy_where = copy_g.split_for_impl().2;
  let handle_doc = format!("The handle of a `{name}` in a region: `Edit<'e, {name}>`.");
  // The closure form of `replace`.
  let update = quote! {
    /// Writes over the value what `f` builds from the old one, which moves into it.
    #vis fn update<#lt, __NearestNew: ::nearest::Emit<#self_ty>>(
      &#lt mut self,
      f: impl ::core::ops::FnOnce(::nearest::Old<#lt, #self_ty>) -> __NearestNew,
    ) {
      self.0.update(f);
    }
  };
  if !has_repr {
    let repr = if is_enum { parse_quote!(#[repr(u8)]) } else { parse_quote!(#[repr(C)]) };
    input.attrs.insert(0, repr);
  }
  // The type is a `Pun` type: its layout and its valid values, for `puns` (the crate that
  // `nearest` re-exports), which checks loaded bytes, reads a value as another type, and reads
  // the `#[pun(..)]` attributes of the user. First, so that those come after the derive. Bounded
  // by the bounds of the type, every type parameter `Pun` among them, which is what a layout
  // needs, and what gives the type an identity (`puns` reads a pointer to it as the same pointer
  // only then), and by nothing that the derive would find from the fields (`bound()`): through
  // pointers, the bounds of types that hold each other would go round in a circle. Not `Flat`,
  // so that the impls of `core` that the user derives, bounded by what the type states, read the
  // pointees. `puns` takes `bound` only where there are bounds to write: an enum without fields
  // has no type parameter.
  let bound = input.generics.type_params().next().map(|_| quote!(, bound()));
  input.attrs.insert(0, parse_quote!(#[pun(crate = ::nearest::puns #bound)]));
  input.attrs.insert(0, parse_quote!(#[derive(::nearest::puns::Pun)]));
  // A pinned value must keep its pointers, so the type is `Unpin` exactly when its fields are.
  // An impl with a bound on all the fields says so, and makes any impl written by hand a
  // conflict. With the bounds of the type, and no others.
  let unpin = {
    let mut fields_g = input.generics.clone();
    fields_g.params.insert(0, GenericParam::Lifetime(parse_quote!(#lt)));
    let mut unpin_g = fields_g.clone();
    let (f_impl, f_ty, f_where) = fields_g.split_for_impl();
    (unpin_g.make_where_clause().predicates)
      .push(parse_quote!(__NearestFields #f_ty: ::core::marker::Unpin));
    let unpin_where = unpin_g.split_for_impl().2;
    quote! {
      #[allow(dead_code)]
      struct __NearestFields #f_impl (::core::marker::PhantomData<&#lt ()>, #(#field_tys,)*)
        #f_where;
      impl #f_impl ::core::marker::Unpin for #self_ty #unpin_where {}
    }
  };
  // What the narrowest pointer held by value reaches: the least `REACH` of the fields (without
  // fields, the default).
  let reach = (!field_tys.is_empty()).then(|| {
    quote! {
      const REACH: u64 = {
        let mut reach = u64::MAX;
        #(if <#field_tys as ::nearest::Flat>::REACH < reach {
          reach = <#field_tys as ::nearest::Flat>::REACH;
        })*
        reach
      };
    }
  });
  let (pin_decl, project) = pinned.map_or((None, None), |(decl, body)| {
    let project = quote! {
      /// The fields of a pinned value, pinned, all at once.
      #vis fn project<#lt>(self: #pin<&#lt mut Self>) -> #pin_name #p_ty {
        // Fields of a pinned value are pinned, and nothing moves out of them.
        unsafe { #body }
      }
    };
    (Some(decl), Some(project))
  });
  Ok(quote! {
    #input
    #view
    #pin_decl
    #init
    const _: () = {
      #unpin
      #(#mirrors)*
      unsafe impl #impl_g ::nearest::Flat for #self_ty #where_c {
        type Edit<#e> = #handle #h_ty;
        type Value = Self;
        type Ref<#lt, #e: #lt> = &#lt mut #handle #h_ty;
        fn reference<#lt, #e: #lt>(edit: &#lt mut #handle #h_ty) -> &#lt mut #handle #h_ty {
          edit
        }
        #reach
        unsafe fn edit<#lt, #e: #lt>(
          #buf: *mut #np::Buf,
          #at: usize,
        ) -> &#lt mut Self::Edit<#e> {
          #np::fits::<Self>();
          #built
          let raw = unsafe { #np::Raw::<#e, Self>::new(#buf, #at) };
          unsafe { &mut *(::core::ptr::from_mut(raw) as *mut #handle #h_ty) }
        }
        unsafe fn store(&self, #sink: &mut #np::Sink, #at: usize) {
          #np::fits::<Self>();
          unsafe { #store }
        }
        unsafe fn shift(#base: *mut u8, #at: usize, #d: isize) {
          unsafe { #shift }
        }
      }
      impl #impl_g #self_ty #where_c {
        #(#ctors)*
        #project
      }
      #[doc = #handle_doc]
      #[repr(transparent)]
      #vis struct #handle #h_impl (#np::Raw<#e, #self_ty>) #h_where;
      impl #h_impl #handle #h_ty #h_where {
        /// Writes `value` over the value, like `*place = value`.
        #vis fn set(&mut self, value: impl ::nearest::Emit<#self_ty>) {
          self.0.set(value);
        }
        /// Writes `value` over the value and returns the old one, like `mem::replace`.
        #vis fn replace(
          &mut self,
          value: impl ::nearest::Emit<#self_ty>,
        ) -> ::nearest::Old<#e, #self_ty> {
          self.0.replace(value)
        }
        /// The value, pinned: a view in which several fields can be borrowed at once (see
        /// `project`). Nothing can be pushed, built or moved through it.
        #vis fn pin(&mut self) -> #pin<&mut #self_ty> {
          self.0.pin()
        }
        #update
        #(#methods)*
      }
      impl #h_impl ::nearest::Handle<#e, #self_ty> for #handle #h_ty #h_where {
        fn set(&mut self, value: impl ::nearest::Emit<#self_ty>) {
          self.0.set(value);
        }
        fn replace(
          &mut self,
          value: impl ::nearest::Emit<#self_ty>,
        ) -> ::nearest::Old<#e, #self_ty> {
          self.0.replace(value)
        }
        fn update<#lt, __NearestNew: ::nearest::Emit<#self_ty>>(
          &#lt mut self,
          f: impl ::core::ops::FnOnce(::nearest::Old<#lt, #self_ty>) -> __NearestNew,
        ) {
          self.0.update(f);
        }
      }
      impl #h_impl ::core::ops::Deref for #handle #h_ty #h_where {
        type Target = #self_ty;
        fn deref(&self) -> &#self_ty {
          self.0.get()
        }
      }
      impl #h_impl ::core::ops::DerefMut for #handle #h_ty #copy_where {
        fn deref_mut(&mut self) -> &mut #self_ty {
          self.0.get_mut()
        }
      }
      #(#extra)*
    };
  })
}

/// Where `#[flat(..)]` is written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
  Type,
  Variant,
  Field,
}

/// What `#[flat(..)]` says: on a variant, the name of its constructor; on a field, that its
/// parameter takes `impl Into<..>` (where `into` is written).
#[derive(Default)]
struct Args {
  rename: Option<Ident>,
  into: Option<Span>,
}

/// Reads one argument of `#[flat(..)]` at `place` into `args`, rejecting what does not belong
/// there.
fn read_args(args: &mut Args, place: Place) -> impl FnMut(ParseNestedMeta) -> syn::Result<()> {
  move |meta| {
    if meta.path.is_ident("into") {
      if place != Place::Field {
        return Err(meta.error("`into` is for the parameter of a field: write it there"));
      }
      if args.into.is_some() {
        return Err(meta.error("`into` is given twice"));
      }
      if !meta.input.is_empty() && !meta.input.peek(syn::Token![,]) {
        return Err(meta.error("`into` takes no value"));
      }
      args.into = Some(meta.path.span());
      return Ok(());
    }
    if !meta.path.is_ident("rename") {
      return Err(meta.error(match place {
        Place::Type => "`#[flat]` takes no arguments",
        Place::Variant => "unknown argument: a variant takes `#[flat(rename = \"name\")]`",
        Place::Field => "unknown argument: a field takes `#[flat(into)]`",
      }));
    }
    if place != Place::Variant {
      return Err(meta.error("`rename` names the constructor of an enum variant: write it there"));
    }
    if args.rename.is_some() {
      return Err(meta.error("`rename` is given twice"));
    }
    let name: LitStr = meta.value()?.parse()?;
    let msg = "`rename` takes the name of a function";
    args.rename = Some(name.parse().map_err(|_| Error::new(name.span(), msg))?);
    Ok(())
  }
}

/// Takes the `#[flat(..)]` attributes out of `attrs` and reads them.
fn take_args(attrs: &mut Vec<Attribute>, place: Place) -> syn::Result<Args> {
  let mut args = Args::default();
  let (ours, others) = mem::take(attrs).into_iter().partition(|a| a.path().is_ident("flat"));
  *attrs = others;
  for a in ours {
    a.parse_nested_meta(read_args(&mut args, place))?;
  }
  Ok(args)
}

/// `X` if `ty` is written `Near<X>` or `Near<X, O>`: any path ending in `Near` with one or two
/// type arguments.
fn near_pointee(ty: &Type) -> Option<&Type> {
  type_arg(ty, "Near", 2)
}

/// Whether `ty` is written as a pointer, `Near<..>` (a list and a string too) or
/// `Option<Near<..>>`.
fn is_pointer(ty: &Type) -> bool {
  near_pointee(ty).or_else(|| type_arg(ty, "Option", 1).and_then(near_pointee)).is_some()
}

/// `X` if `ty` is written `name<X, ..>`: any path ending in `name` with at least one and at most
/// `most` type arguments.
fn type_arg<'t>(ty: &'t Type, name: &str, most: usize) -> Option<&'t Type> {
  let ty = match ty {
    Type::Group(g) => &*g.elem,
    Type::Paren(p) => &*p.elem,
    ty => ty,
  };
  let Type::Path(p) = ty else { return None };
  let last = p.path.segments.last()?;
  match &last.arguments {
    PathArguments::AngleBracketed(a)
      if last.ident == name && (1..=most).contains(&a.args.len()) =>
    {
      match a.args.first() {
        Some(GenericArgument::Type(x)) => Some(x),
        _ => None,
      }
    }
    _ => None,
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
    // `project` is the generated method on pinned values.
    "self" | "super" | "crate" | "project" => format_ident!("{s}_", span = variant.span()),
    // syn accepts `gen`, which edition 2024 reserves.
    _ if s == "gen" || syn::parse_str::<Ident>(&s).is_err() => Ident::new_raw(&s, variant.span()),
    _ => Ident::new(&s, variant.span()),
  }
}
