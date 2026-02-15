#![feature(offset_of_enum)]

//! Virtual file system using self-relative pointers.
//!
//! Demonstrates tree structures via `Near<FsNode>` + `NearList<Child>`,
//! building sub-regions and composing via `&*region` emitters,
//! recursive printing, session `extend_list`, `graft`, and `trim`.

use nearest::{Flat, Near, NearList, Region};

/// A named child entry in a directory.
#[derive(Flat, Debug)]
struct Child {
  name: NearList<u8>,
  node: Near<FsNode>,
}

/// A file system node: either a file with a size, or a directory with children.
#[derive(Flat, Debug)]
#[repr(C, u8)]
#[expect(dead_code, reason = "variants constructed via emitters")]
enum FsNode {
  File { size: u32 },
  Dir { children: NearList<Child> },
}

fn name_of(list: &NearList<u8>) -> String {
  list.iter().map(|&b| b as char).collect()
}

fn print_tree(node: &FsNode, prefix: &str, name: &str) {
  match node {
    FsNode::File { size } => println!("{}{} ({} bytes)", prefix, name, size),
    FsNode::Dir { children } => {
      println!("{}{}/", prefix, name);
      for (i, child) in children.iter().enumerate() {
        let is_last = i == children.len() - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let next_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
        print_tree(&child.node, &next_prefix, &format!("{}{}", connector, name_of(&child.name)));
      }
    }
  }
}

fn main() {
  // Build leaf directories containing only files (homogeneous children).
  let src = Region::new(FsNode::make_dir([
    Child::make(b"main.rs".as_slice(), FsNode::make_file(1200)),
    Child::make(b"lib.rs".as_slice(), FsNode::make_file(800)),
  ]));
  let docs =
    Region::new(FsNode::make_dir([Child::make(b"readme.md".as_slice(), FsNode::make_file(2400))]));

  // Compose into root using &*Region references — both entries share
  // the same emitter type (&FsNode), so they can go in one array.
  let mut region = Region::new(FsNode::make_dir([
    Child::make(b"src".as_slice(), &*src),
    Child::make(b"docs".as_slice(), &*docs),
  ]));

  println!("=== initial tree ===");
  print_tree(&region, "", "root");

  // Add test.rs to src/ via session.
  region.session(|s| {
    let src_children = s.nav(s.root(), |n| match n {
      FsNode::Dir { children } => match &*children[0].node {
        FsNode::Dir { children } => children,
        _ => panic!("expected Dir"),
      },
      _ => panic!("expected Dir"),
    });
    s.extend_list(src_children, [Child::make(b"test.rs".as_slice(), FsNode::make_file(500))]);
  });

  println!("\n=== after adding test.rs ===");
  print_tree(&region, "", "root");

  // Build a separate /build directory, then graft it in.
  let build = Region::new(FsNode::make_dir([
    Child::make(b"output.bin".as_slice(), FsNode::make_file(4096)),
    Child::make(b"deps.lock".as_slice(), FsNode::make_file(128)),
  ]));

  region.session(|s| {
    let grafted = s.graft(&build);
    let root_children = s.nav(s.root(), |n| match n {
      FsNode::Dir { children } => children,
      _ => panic!("expected Dir"),
    });
    s.extend_list(root_children, [Child::make(b"build".as_slice(), grafted)]);
  });

  println!("\n=== after grafting build/ ===");
  print_tree(&region, "", "root");

  // Verify structure.
  match &*region {
    FsNode::Dir { children } => {
      assert_eq!(children.len(), 3);
      assert_eq!(name_of(&children[2].name), "build");
    }
    _ => panic!("expected Dir"),
  }

  // Trim and report.
  let before = region.byte_len();
  region.trim();
  println!("\ntrim: {} -> {} bytes", before, region.byte_len());
}
