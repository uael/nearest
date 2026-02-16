use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, Fields};

/// Classifies how a field type maps to an emitter parameter.
pub enum FieldKind {
  /// Primitive type (u8, u16, etc.) — concrete parameter, uses Emit<T> directly.
  Primitive,
  /// `Near<T>` — accepts `impl Emit<T>`, generates alloc+patch.
  Near { inner: syn::Type },
  /// `NearList<T>` — accepts `impl IntoIterator<Item: Emit<T>>`.
  NearList { inner: syn::Type },
  /// Any other user type — accepts `impl Emit<FieldType>`.
  Other,
}

pub fn classify_field(ty: &syn::Type) -> FieldKind {
  if is_primitive_type(ty) {
    return FieldKind::Primitive;
  }
  if let syn::Type::Path(p) = ty
    && let Some(seg) = p.path.segments.last()
  {
    let name = seg.ident.to_string();
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
      && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
    {
      if name == "Near" {
        return FieldKind::Near { inner: inner.clone() };
      }
      if name == "NearList" {
        return FieldKind::NearList { inner: inner.clone() };
      }
    }
  }
  FieldKind::Other
}

pub fn is_bool_type(ty: &syn::Type) -> bool {
  if let syn::Type::Path(p) = ty
    && let Some(ident) = p.path.get_ident()
  {
    return ident == "bool";
  }
  false
}

pub fn is_primitive_type(ty: &syn::Type) -> bool {
  if let syn::Type::Path(p) = ty
    && let Some(ident) = p.path.get_ident()
  {
    return matches!(
      ident.to_string().as_str(),
      "u8" | "u16" | "u32" | "i32" | "u64" | "i64" | "bool"
    );
  }
  false
}

/// Returns true if all field types are primitive.
///
/// Types where every field is primitive can safely self-emit (byte-copy).
/// Types with non-primitive fields (even if not Near/NearList directly)
/// need builders because those fields may transitively contain self-relative
/// pointers that require proper allocation and patching.
///
/// Used for **structs** only — a struct like `Func { entry: Block }` needs a
/// builder because `Block` transitively contains `Near`/`NearList`.
pub fn is_all_primitive(data: &Data) -> bool {
  collect_field_types(data).iter().all(is_primitive_type)
}

/// Returns true if `ty` is `Option<Near<T>>` or `Option<NearList<T>>`.
pub fn is_option_of_pointer(ty: &syn::Type) -> bool {
  if let syn::Type::Path(p) = ty
    && let Some(seg) = p.path.segments.last()
    && seg.ident == "Option"
    && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
    && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
  {
    return matches!(classify_field(inner), FieldKind::Near { .. } | FieldKind::NearList { .. });
  }
  false
}

/// Unwrap `Option<T>` to get the inner `T`, if the outer type is `Option`.
pub fn unwrap_option(ty: &syn::Type) -> Option<&syn::Type> {
  if let syn::Type::Path(p) = ty
    && let Some(seg) = p.path.segments.last()
    && seg.ident == "Option"
    && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
    && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
  {
    return Some(inner);
  }
  None
}

/// Returns true if none of the fields are `Near<T>`, `NearList<T>`,
/// `Option<Near<T>>`, or `Option<NearList<T>>`.
///
/// Used for **enums** — an enum like `Value { Const(u32), Type(Type) }` has no
/// pointer fields and can safely self-emit, even though `Type` is not primitive.
pub fn has_no_pointer_fields(data: &Data) -> bool {
  collect_field_types(data).iter().all(|ty| {
    !matches!(classify_field(ty), FieldKind::Near { .. } | FieldKind::NearList { .. })
      && !is_option_of_pointer(ty)
  })
}

/// Returns true if a type parameter already has `Flat` (or `::nearest::Flat`) in
/// its bounds list.
pub fn has_flat_bound(tp: &syn::TypeParam) -> bool {
  tp.bounds.iter().any(|bound| {
    if let syn::TypeParamBound::Trait(tb) = bound
      && let Some(seg) = tb.path.segments.last()
    {
      seg.ident == "Flat"
    } else {
      false
    }
  })
}

/// Returns true if `ty` is a simple ident path matching `name`.
pub fn is_type_param_ident(ty: &syn::Type, name: &str) -> bool {
  if let syn::Type::Path(p) = ty
    && let Some(ident) = p.path.get_ident()
  {
    return ident == name;
  }
  false
}

/// Collect the names of type parameters that already have a `Flat` bound.
pub fn flat_bounded_param_names(generics: &syn::Generics) -> Vec<String> {
  generics.type_params().filter(|tp| has_flat_bound(tp)).map(|tp| tp.ident.to_string()).collect()
}

pub fn collect_field_types(data: &Data) -> Vec<syn::Type> {
  match data {
    Data::Struct(s) => fields_types(&s.fields),
    Data::Enum(e) => e.variants.iter().flat_map(|v| fields_types(&v.fields)).collect(),
    Data::Union(_) => panic!("Flat cannot be derived for unions"),
  }
}

fn fields_types(fields: &Fields) -> Vec<syn::Type> {
  fields.iter().map(|f| f.ty.clone()).collect()
}

pub fn capitalize(s: &str) -> String {
  s.chars()
    .enumerate()
    .map(|(i, c)| if i == 0 { c.to_uppercase().next().unwrap() } else { c })
    .collect()
}

pub fn to_snake_case(s: &str) -> String {
  let mut result = String::new();
  for (i, c) in s.chars().enumerate() {
    if c.is_uppercase() {
      if i > 0 {
        result.push('_');
      }
      result.extend(c.to_lowercase());
    } else {
      result.push(c);
    }
  }
  result
}

pub fn opt_where_clause(preds: &[TokenStream]) -> TokenStream {
  if preds.is_empty() {
    quote! {}
  } else {
    quote! { where #(#preds),* }
  }
}

pub fn combine_where(
  existing: Option<&syn::WhereClause>,
  predicates: &[TokenStream],
) -> TokenStream {
  match existing {
    Some(existing) => quote! { #existing #(, #predicates)* },
    None if predicates.is_empty() => quote! {},
    None => quote! { where #(#predicates),* },
  }
}
