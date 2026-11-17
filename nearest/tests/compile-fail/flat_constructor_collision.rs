//! Two variants whose constructors would have the same name.
use nearest::flat;

#[flat]
enum E { HTTPServer(u8), HttpServer(u16) }

fn main() {}
