use syn::{Attribute, Data, DeriveInput, Fields};

/// Parsed `#[flat(...)]` attributes on a struct/enum field.
#[derive(Default)]
pub struct FieldAttrs {
  /// If `true`, the builder parameter accepts `impl Into<T>` instead of `T`.
  pub into: bool,
}

/// Parsed `#[flat(...)]` attributes on an enum variant.
#[derive(Default)]
pub struct VariantAttrs {
  /// Custom name for the generated `make_*` method (`None` = use default).
  pub rename: Option<syn::Ident>,
}

/// Parse `#[flat(...)]` attributes from a field.
pub fn parse_field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
  let mut result = FieldAttrs::default();
  for attr in attrs {
    if !attr.path().is_ident("flat") {
      continue;
    }
    attr.parse_nested_meta(|meta| {
      if meta.path.is_ident("into") {
        result.into = true;
        return Ok(());
      }
      Err(meta.error("expected `into`"))
    })?;
  }
  Ok(result)
}

/// Parse `#[flat(...)]` attributes from an enum variant.
pub fn parse_variant_attrs(attrs: &[Attribute]) -> syn::Result<VariantAttrs> {
  let mut result = VariantAttrs::default();
  for attr in attrs {
    if !attr.path().is_ident("flat") {
      continue;
    }
    attr.parse_nested_meta(|meta| {
      if meta.path.is_ident("rename") {
        let value = meta.value()?;
        let s: syn::LitStr = value.parse()?;
        result.rename = Some(s.parse()?);
        return Ok(());
      }
      Err(meta.error("expected `rename = \"...\"`"))
    })?;
  }
  Ok(result)
}

/// Validate all `#[flat(...)]` attributes in the input, returning the first
/// error encountered (if any). This ensures that invalid attributes are
/// rejected even when the emit code generator takes a fast path that doesn't
/// iterate individual fields.
pub fn validate_all_attrs(input: &DeriveInput) -> syn::Result<()> {
  let validate_fields = |fields: &Fields| -> syn::Result<()> {
    for f in fields {
      parse_field_attrs(&f.attrs)?;
    }
    Ok(())
  };
  match &input.data {
    Data::Struct(s) => validate_fields(&s.fields)?,
    Data::Enum(e) => {
      for v in &e.variants {
        parse_variant_attrs(&v.attrs)?;
        validate_fields(&v.fields)?;
      }
    }
    Data::Union(_) => {}
  }
  Ok(())
}
