//! stage0 against contracts/opt-passes.md, the release passes' contract. stage0 has the base
//! passes only (rule P3), so this checks what it can:
//!
//! - cases of the base passes beside the examples: the parts of params placed after the slots
//!   and a part whose anchor is deleted in the same step (B1, B3).

use stave_driver::ir_cmd;

fn load(name: &str, text: &str) -> ir_cmd::LoadedIr {
    ir_cmd::load(name, text.as_bytes()).unwrap_or_else(|e| panic!("{name}: {:?}", e.lines()))
}

/// `stave ir opt` on `text`.
fn opt(name: &str, text: &str) -> String {
    ir_cmd::opt(&load(name, text)).unwrap_or_else(|e| panic!("{name}: ir opt: {:?}", e.lines()))
}

/// `text` with the level of its header replaced.
fn at_level(text: &str, from: &str, to: &str) -> String {
    let (head, rest) = text.split_once('\n').unwrap();
    let head = head.strip_suffix(from).unwrap_or_else(|| panic!("header `{head}` is not at level {from}"));
    format!("{head}{to}\n{rest}")
}

/// B1: a part of a param of `b0` goes after the slots, which stay a prefix of `b0`; a part of a
/// param of another block goes to the block's start.
#[test]
fn parts_of_params_go_after_the_slots() {
    let types = "type t0 int u size
type t1 ref t3
type t2 ref mut t0
type t3 slice t0
fn @0 main.f(t1, t1) -> t0 {
b0(%0: t1, %1: t1):
  %2 = slot t2
  %3 = slot t2
";
    let input = format!(
        "stave.ir 4 \"x86_64-linux-v2\" release checked
{types}  %4 = slice.len t0 %1
  store () %2 %4
  %5 = slice.len t0 %1
  store () %3 %5
  br b1(%0)
b1(%6: t1):
  %7 = load t0 %2
  %8 = slice.len t0 %6
  %9 = slice.len t0 %6
  %10 = add.wrap t0 %8 %9
  %11 = add.wrap t0 %10 %7
  ret %11
}}
"
    );
    let want = format!(
        "stave.ir 4 \"x86_64-linux-v2\" release opt
{types}  %4 = slice.len t0 %1
  store () %2 %4
  store () %3 %4
  br b1(%0)
b1(%5: t1):
  %6 = slice.len t0 %5
  %7 = load t0 %2
  %8 = add.wrap t0 %6 %6
  %9 = add.wrap t0 %8 %7
  ret %9
}}
"
    );
    assert_eq!(opt("params.ir", &input), want);
}

/// B3 in one step: the group `(%1, 1)` moves `%2` and deletes `%4`; the group `(%4, 2)`, one part
/// in the loop, goes to the place after `%4`, which is deleted: its old position, not after `%2`.
#[test]
fn a_part_whose_anchor_is_deleted_goes_to_the_anchors_old_position() {
    let head = "stave.ir 4 \"x86_64-linux-v2\" release checked
type t0 int u size
type t1 bool
fn @0 main.f(t0) -> t0 {
";
    let input = format!(
        "{head}b0(%0: t0):
  %1 = add.wrap t0 %0 5
  %2 = sub.sat t0 %1 1
  %3 = mul.wrap t0 %1 3
  %4 = sub.sat t0 %1 1
  %5 = add.wrap t0 %3 7
  br b1(0, %2)
b1(%6: t0, %7: t0):
  %8 = lt t1 %6 %5
  cbr %8 b2() b3()
b2():
  %9 = sub.sat t0 %4 2
  %10 = add.wrap t0 %6 1
  %11 = add.wrap t0 %7 %9
  br b1(%10, %11)
b3():
  ret %7
}}
"
    );
    let want = format!(
        "{}b0(%0: t0):
  %1 = add.wrap t0 %0 5
  %2 = sub.sat t0 %1 1
  %3 = mul.wrap t0 %1 3
  %4 = sub.sat t0 %2 2
  %5 = add.wrap t0 %3 7
  br b1(0, %2)
b1(%6: t0, %7: t0):
  %8 = lt t1 %6 %5
  cbr %8 b2() b3()
b2():
  %9 = add.wrap t0 %6 1
  %10 = add.wrap t0 %7 %4
  br b1(%9, %10)
b3():
  ret %7
}}
",
        at_level(head, "checked", "opt")
    );
    assert_eq!(opt("m1.ir", &input), want);
}