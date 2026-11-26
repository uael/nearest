//! Each case of `tests/compile-fail/` fails to compile with the error of its `.stderr` file, and
//! each case of `tests/compile-pass/` compiles. The cases are in two tests of about the same time,
//! which CI runs as two jobs: `a_to_o_but_f` (names that start with `a` to `o`, but not `f`) and
//! `the_others`. trybuild runs one at a time, under a lock.

/// The cases whose names start with `a` to `o`, but not `f`.
#[test]
fn a_to_o_but_f() {
  cases("tests/compile-fail/[a-eg-o]*.rs");
}

/// The other cases.
#[test]
fn the_others() {
  cases("tests/compile-fail/[!a-eg-o]*.rs");
}

/// Checks that the cases that `pattern` names fail to compile.
fn cases(pattern: &str) {
  let t = trybuild::TestCases::new();
  t.compile_fail(pattern);
  // A passing case makes trybuild build instead of check, so that errors from constants
  // evaluated at monomorphization (alignment, drop glue) are reported too.
  t.pass("tests/compile-pass/*.rs");
}
