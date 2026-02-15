#![feature(offset_of_enum)]

//! JSON document model using self-relative pointers.
//!
//! Demonstrates heterogeneous enums, `NearList<u8>` as inline strings,
//! nested construction via pre-built sub-regions, recursive pretty-printing,
//! session `extend_list`, and `trim`.

use nearest::{Flat, Near, NearList, Region};

/// A key-value pair for JSON objects.
#[derive(Flat, Debug)]
struct Entry {
  key: NearList<u8>,
  val: Near<Json>,
}

/// A JSON-like value type stored entirely in a region.
#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum Json {
  Null,
  Bool(bool),
  Num(i64),
  Str { chars: NearList<u8> },
  Arr { items: NearList<Json> },
  Obj { fields: NearList<Entry> },
}

fn str_of(list: &NearList<u8>) -> String {
  list.iter().map(|&b| b as char).collect()
}

fn pretty(json: &Json, indent: usize) -> String {
  let pad = "  ".repeat(indent);
  match json {
    Json::Null => "null".into(),
    Json::Bool(b) => b.to_string(),
    Json::Num(n) => n.to_string(),
    Json::Str { chars } => format!("\"{}\"", str_of(chars)),
    Json::Arr { items } => {
      if items.is_empty() {
        return "[]".into();
      }
      let inner: Vec<String> =
        items.iter().map(|v| format!("{}  {}", pad, pretty(v, indent + 1))).collect();
      format!("[\n{}\n{}]", inner.join(",\n"), pad)
    }
    Json::Obj { fields } => {
      if fields.is_empty() {
        return "{}".into();
      }
      let inner: Vec<String> = fields
        .iter()
        .map(|e| format!("{}  \"{}\": {}", pad, str_of(&e.key), pretty(&e.val, indent + 1)))
        .collect();
      format!("{{\n{}\n{}}}", inner.join(",\n"), pad)
    }
  }
}

fn main() {
  // Build individual JSON values as separate regions.
  let name_val = Region::new(Json::make_str(b"Alice".as_slice()));
  let age_val = Region::new(Json::make_num(30));
  // Homogeneous array: all items use the same make_num builder type.
  let scores_val =
    Region::new(Json::make_arr([Json::make_num(95), Json::make_num(87), Json::make_num(92)]));

  // Build the object using &*Region references — all entries share
  // the same emitter type (&Json), so they can go in one array.
  let mut region = Region::new(Json::make_obj([
    Entry::make(b"name".as_slice(), &*name_val),
    Entry::make(b"age".as_slice(), &*age_val),
    Entry::make(b"scores".as_slice(), &*scores_val),
  ]));

  println!("=== original ===");
  println!("{}", pretty(&region, 0));

  // Add a new field via session: "active": true.
  // Each extend_list call adds a single entry, avoiding mixed emitter types.
  region.session(|s| {
    let fields = s.nav(s.root(), |j| match j {
      Json::Obj { fields } => fields,
      _ => panic!("expected Obj"),
    });
    s.extend_list(fields, [Entry::make(b"active".as_slice(), Json::make_bool(true))]);
  });

  println!("\n=== after adding \"active\" ===");
  println!("{}", pretty(&region, 0));

  // Verify the new field.
  match &*region {
    Json::Obj { fields } => {
      assert_eq!(fields.len(), 4);
      let last = &fields[3];
      assert_eq!(str_of(&last.key), "active");
      assert!(matches!(&*last.val, Json::Bool(true)));
    }
    _ => panic!("expected Obj"),
  }

  // Trim reclaims dead bytes from append-only mutations.
  let before = region.byte_len();
  region.trim();
  println!("\ntrim: {} -> {} bytes", before, region.byte_len());
}
