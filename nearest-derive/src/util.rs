use proc_macro2::TokenStream;
use quote::quote;

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
