use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, Fields};

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

/// Returns true if a type syntactically names a self-relative pointer type.
fn is_pointer_type(ty: &syn::Type) -> bool {
  if let syn::Type::Path(p) = ty
    && let Some(seg) = p.path.segments.last()
  {
    let name = seg.ident.to_string();
    if name == "Near" || name == "NearList" {
      return true;
    }
    // Check Option<Near<T>>
    if name == "Option"
      && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
      && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
    {
      return is_pointer_type(inner);
    }
  }
  false
}

/// Returns true if none of the fields are `Near<T>`, `NearList<T>`, or
/// `Option<Near<T>>`.
///
/// Used for **enums** — an enum like `Value { Const(u32), Type(Type) }` has no
/// pointer fields and can safely self-emit, even though `Type` is not primitive.
pub fn has_no_pointer_fields(data: &Data) -> bool {
  collect_field_types(data).iter().all(|ty| !is_pointer_type(ty))
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
