#![feature(offset_of_enum)]

//! Finite state machine built on a stack-only `FixedBuf`.
//!
//! Demonstrates `FixedBuf<N>` for no-heap construction,
//! `Region::<T, FixedBuf<N>>::new_in()`, enum match conditions,
//! `NearList` indexing for state lookup, session `extend_list`
//! to add transitions, and a simulation loop.

use nearest::{FixedBuf, Flat, NearList, Region, empty};

/// A match condition on a single byte.
#[derive(Flat, Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, u8)]
enum Match {
  Char(u8),
  Any,
}

/// A transition: match condition + target state index.
#[derive(Flat, Debug)]
struct Transition {
  cond: Match,
  target: u32,
}

/// A state in the FSM.
#[derive(Flat, Debug)]
struct State {
  accept: bool,
  edges: NearList<Transition>,
}

/// The FSM: a list of states; state 0 is the start state.
#[derive(Flat, Debug)]
struct Fsm {
  states: NearList<State>,
}

fn simulate(fsm: &Fsm, input: &[u8]) -> bool {
  let mut current: u32 = 0;
  for &byte in input {
    let state = &fsm.states[current as usize];
    let mut matched = false;
    for edge in &state.edges {
      let ok = match edge.cond {
        Match::Char(c) => c == byte,
        Match::Any => true,
      };
      if ok {
        current = edge.target;
        matched = true;
        break;
      }
    }
    if !matched {
      return false; // dead state
    }
  }
  fsm.states[current as usize].accept
}

fn main() {
  // Build FSM for pattern "a.b" (matches 'a', any char, 'b').
  //
  // States: 0 --a--> 1 --any--> 2 --b--> 3(accept)
  //
  // Pre-build each state as a separate Region, then compose them
  // using &State references (uniform type for the states array).
  let s0 = Region::new(State::make(false, [Transition::make(Match::Char(b'a'), 1)]));
  let s1 = Region::new(State::make(false, [Transition::make(Match::Any, 2)]));
  let s2 = Region::new(State::make(false, [Transition::make(Match::Char(b'b'), 3)]));
  let s3 = Region::new(State::make(true, empty()));

  // Final FSM lives on the stack — no heap allocation for the region itself.
  let mut region: Region<Fsm, FixedBuf<512>> = Region::new_in(Fsm::make([&*s0, &*s1, &*s2, &*s3]));

  println!("=== FSM for pattern 'a.b' ===");
  println!("states: {}", region.states.len());
  for (i, state) in region.states.iter().enumerate() {
    println!("  state {}: accept={}, edges={}", i, state.accept, state.edges.len());
  }

  // Test cases.
  let tests: &[(&[u8], bool)] = &[
    (b"axb", true),
    (b"a1b", true),
    (b"aab", true),
    (b"ab", false),  // only 2 chars, need 3
    (b"axc", false), // third char must be 'b'
    (b"bxa", false), // must start with 'a'
    (b"", false),    // empty
  ];

  println!("\n=== simulation ===");
  for (input, expected) in tests {
    let input_str = core::str::from_utf8(input).unwrap_or("?");
    let result = simulate(&region, input);
    println!(
      "  \"{}\" -> {} {}",
      input_str,
      result,
      if result == *expected { "ok" } else { "FAIL" }
    );
    assert_eq!(result, *expected);
  }

  // Add a self-loop on state 3 for 'b', making it accept "a.b+" patterns.
  region.session(|s| {
    let state3_edges = s.nav(s.root(), |fsm| &fsm.states[3].edges);
    s.extend_list(state3_edges, [Transition::make(Match::Char(b'b'), 3)]);
  });

  println!("\n=== after adding 'b' loop on accept state ===");
  let extra_tests: &[(&[u8], bool)] = &[
    (b"axb", true),
    (b"axbb", true),
    (b"axbbb", true),
    (b"axbc", false), // 'c' has no transition from state 3
  ];

  for (input, expected) in extra_tests {
    let input_str = core::str::from_utf8(input).unwrap_or("?");
    let result = simulate(&region, input);
    println!(
      "  \"{}\" -> {} {}",
      input_str,
      result,
      if result == *expected { "ok" } else { "FAIL" }
    );
    assert_eq!(result, *expected);
  }

  println!("\nbyte_len: {}", region.byte_len());
}
