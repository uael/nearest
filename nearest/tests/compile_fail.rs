#[test]
fn compile_fail() {
  let t = trybuild::TestCases::new();
  t.compile_fail("tests/compile-fail/*.rs");
  // A passing case makes trybuild build instead of check, so that errors from constants
  // evaluated at monomorphization (alignment, drop glue) are reported too.
  t.pass("tests/compile-pass/*.rs");
}
