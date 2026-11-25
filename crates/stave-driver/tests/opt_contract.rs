//! stage0 against contracts/opt-passes.md, the release passes' contract. stage0 has the base
//! passes only (rule P3), so this checks what it can:
//!
//! - the golden cases, one directory per case under `tests/opt/` in the layout of the contract's
//!   section 9: `input.ir` (the `checked` input), `inline.txt` (the decision dump,
//!   `--dump=inline`), `inline.ir` (the module after the expansion, `--dump-phase=inline`),
//!   `clean.ir` (after the clean-up, when it differs from the output), `output.ir` (the `opt`
//!   output), `limits` (the `--limit` overrides of the case), `passes` (the line `base` for a
//!   `--passes=base` case). The worked examples of section 10 are `10.<k>`; the further goldens
//!   are `<phase>-<topic>` with a one-line `README` naming the rules the case exercises. Every
//!   `.ir` text loads (`stave ir check`) and is its own `stave ir fmt`. `stave ir opt` gives the
//!   output byte for byte: from `input.ir` for a `--passes=base` case; else from the clean-up's
//!   text (`clean.ir`, or `output.ir` itself) read at level `checked`;
//! - cases of the base passes beside the goldens: the parts of params placed after the slots
//!   and a part whose anchor is deleted in the same step (B1, B3), dead code whose value has 255
//!   uses and more (B5);
//! - a `switch` on an int constant, legal at level `opt` only: every consumer of an `opt` text
//!   (`stave ir run`, `stave ir build` for x86-64 and AArch64) takes the case whose value is the
//!   constant, and the base passes never see one (they take `lowered` or `checked` text);
//! - `stave ir opt` keeps the stack section of an emit-form file.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../tests/support/legs.rs"]
mod legs;

use stave_driver::dump::emit_text;
use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::ir_cmd::{self, IrError};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn examples() -> PathBuf {
    repo().join("tests/opt")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

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

/// Every case directory under `tests/opt/`: the worked examples `10.<k>`, and the further goldens,
/// named `<phase>-<topic>` with the phase `d` (decision), `x` (expansion), `c` (clean-up), `b`
/// (base passes), or `p` (pipeline).
fn case_dirs() -> Vec<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(examples())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    dirs.sort();
    dirs
}

fn worked_examples() -> Vec<String> {
    (1..=10).map(|k| format!("10.{k}")).collect()
}

/// The goldens beside the worked examples.
fn further_cases() -> Vec<String> {
    let worked = worked_examples();
    case_dirs().into_iter().filter(|d| !worked.contains(d)).collect()
}

/// Whether case `d` runs with `--passes=base`: its `passes` file, whose one line is `base`.
fn base_case(d: &str) -> bool {
    let p = examples().join(d).join("passes");
    if !p.is_file() {
        return false;
    }
    assert_eq!(read(&p), "base\n", "{d}/passes: the one line `base`");
    true
}

/// The files a case directory may hold.
const CASE_FILES: [&str; 8] =
    ["input.ir", "inline.txt", "inline.ir", "clean.ir", "output.ir", "limits", "passes", "README"];

#[test]
fn every_case_directory_holds_the_layout_of_the_contract() {
    let dirs = case_dirs();
    for d in worked_examples() {
        assert!(dirs.contains(&d), "{d}: no directory for the worked example");
    }
    let further = further_cases();
    let phases = ["d-", "x-", "c-", "b-", "p-"];
    for d in &dirs {
        let dir = examples().join(d);
        for e in std::fs::read_dir(&dir).unwrap() {
            let f = e.unwrap().file_name().to_string_lossy().into_owned();
            assert!(CASE_FILES.contains(&f.as_str()), "{d}/{f}: not a file of a case");
        }
        assert!(dir.join("input.ir").is_file(), "{d}: no input.ir");
        if further.contains(d) {
            assert!(
                phases.iter().any(|p| d.starts_with(p))
                    && d.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                "{d}: a golden is named <phase>-<topic>"
            );
            assert!(dir.join("README").is_file(), "{d}: no README");
        } else {
            assert!(!dir.join("README").is_file(), "{d}: a worked example has no README");
        }
        if base_case(d) {
            for f in ["inline.txt", "inline.ir", "clean.ir", "limits"] {
                assert!(!dir.join(f).is_file(), "{d}: a `--passes=base` case has no {f}");
            }
            assert!(dir.join("output.ir").is_file(), "{d}: a `--passes=base` case has an output.ir");
        } else {
            assert!(dir.join("inline.txt").is_file(), "{d}: no inline.txt");
        }
        if dir.join("clean.ir").is_file() {
            assert_ne!(
                read(&dir.join("clean.ir")),
                read(&dir.join("output.ir")),
                "{d}: clean.ir equals output.ir"
            );
        }
    }
    assert!(base_case("10.8") && base_case("10.9"), "10.8 and 10.9 run with `--passes=base`");
}

#[test]
fn every_case_text_loads_and_is_canonical() {
    let worked = worked_examples();
    let mut example_texts = 0;
    for d in &case_dirs() {
        let dir = examples().join(d);
        for f in ["input.ir", "inline.ir", "clean.ir", "output.ir"] {
            let p = dir.join(f);
            if !p.is_file() {
                continue;
            }
            let name = format!("{d}/{f}");
            let text = read(&p);
            let level = if f == "input.ir" { "checked" } else { "opt" };
            assert!(text.lines().next().unwrap().ends_with(&format!(" release {level}")), "{name}: header");
            let l = load(&name, &text);
            let printed = ir_cmd::fmt(&l).unwrap_or_else(|e| panic!("{name}: ir fmt: {:?}", e.lines()));
            assert_eq!(printed, text, "{name}: not its own `stave ir fmt`");
            if worked.contains(d) {
                example_texts += 1;
            }
        }
    }
    assert_eq!(example_texts, 20);
}

/// The fault in a `limits` file's text (opt-passes.md K1), if any: each line `NAME=N`, NAME a key
/// of `[limits]` at most once, N an unsigned decimal in that key's range.
fn limits_fault(text: &str) -> Option<String> {
    if text.is_empty() || !text.ends_with('\n') {
        return Some("not `NAME=N` lines".into());
    }
    let call = stave_ir::opt_table::row("call").expect("the `call` row").weight as u64;
    let mut seen = Vec::new();
    for line in text.lines() {
        let Some((name, n)) = line.split_once('=') else { return Some(format!("`{line}`: no `=`")) };
        if !n.bytes().all(|b| b.is_ascii_digit()) || n.is_empty() {
            return Some(format!("`{line}`: not a decimal integer without sign"));
        }
        let Ok(n) = n.parse::<u64>() else { return Some(format!("`{line}`: out of range")) };
        let ok = match name {
            "threshold" | "cap" | "slot_callee" | "slot_caller" => true,
            "call_saved" => n <= call,
            "rounds" => n >= 1,
            _ => return Some(format!("`{line}`: unknown limit `{name}`")),
        };
        if !ok {
            return Some(format!("`{line}`: out of range"));
        }
        if seen.contains(&name) {
            return Some(format!("`{name}` twice"));
        }
        seen.push(name);
    }
    None
}

#[test]
fn a_limits_file_names_known_limits_in_their_ranges() {
    assert_eq!(limits_fault("cap=20\nrounds=2\ncall_saved=3\nthreshold=0\n"), None);
    for bad in [
        "",
        "cap=20",
        "cap\n",
        "depth=3\n",
        "CAP=20\n",
        "cap=-1\n",
        "cap=+1\n",
        "cap=0x10\n",
        "rounds=0\n",
        "call_saved=4\n",
        "cap=1\ncap=2\n",
    ] {
        assert!(limits_fault(bad).is_some(), "`{bad}` is refused");
    }
}

/// A case's `limits` file: `NAME=N` lines over the `[limits]` keys of opt.toml. Of the worked
/// examples only 10.10, the budget cut on `CAP`, runs under a limit of its own.
#[test]
fn a_case_keeps_its_limit_overrides_in_its_directory() {
    for d in case_dirs() {
        let p = examples().join(&d).join("limits");
        if !p.is_file() {
            assert_ne!(d, "10.10", "10.10: no limits file");
            continue;
        }
        let text = read(&p);
        if let Some(fault) = limits_fault(&text) {
            panic!("{d}/limits: {fault}");
        }
        if worked_examples().contains(&d) {
            assert_eq!(d, "10.10", "{d}: a limits file");
            assert_eq!(text, "cap=20\n");
        }
    }
}

/// A `--passes=base` case: `stave ir opt input.ir` is its output. Every other case with an output:
/// the output is `canon(B(clean))` (rule P1), so the base passes on the clean-up's text (`clean.ir`,
/// else the output itself) read at level `checked` give the output.
#[test]
fn base_passes_give_every_cases_output() {
    for d in case_dirs() {
        let dir = examples().join(&d);
        let out = dir.join("output.ir");
        if !out.is_file() {
            continue;
        }
        let want = read(&out);
        if base_case(&d) {
            let got = opt(&format!("{d}/input.ir"), &read(&dir.join("input.ir")));
            assert_eq!(got, want, "{d}: `stave ir opt input.ir`");
        } else {
            let (f, clean) = match dir.join("clean.ir") {
                p if p.is_file() => ("clean.ir", read(&p)),
                _ => ("output.ir", want.clone()),
            };
            let got = opt(&format!("{d}/{f} as checked"), &at_level(&clean, "opt", "checked"));
            assert_eq!(got, want, "{d}: the base passes on {f}");
        }
    }
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

/// `%1` has `n` uses, all by dead instructions: B5 deletes them and then `%1`, whatever `n`.
#[test]
fn dead_code_goes_however_many_uses_it_had() {
    for n in [1, 254, 255, 256, 300, 600] {
        let mut text = String::from(
            "stave.ir 4 \"x86_64-linux-v2\" release checked
type t0 int u 64
fn @0 main.f(t0) -> t0 {
b0(%0: t0):
  %1 = add.wrap t0 %0 1
",
        );
        for k in 0..n {
            text.push_str(&format!("  %{} = xor t0 %1 {k}\n", k + 2));
        }
        text.push_str("  ret %0\n}\n");
        let want = "stave.ir 4 \"x86_64-linux-v2\" release opt
type t0 int u 64
fn @0 main.f(t0) -> t0 {
b0(%0: t0):
  ret %0
}
";
        assert_eq!(opt(&format!("uses{n}.ir"), &text), want, "{n} uses");
    }
}

/// `pick` lowers to `switch %0 0:.. 1:.. 7:.. _:..`; `main` exits with `pick(0)`.
const SWITCH_PROGRAM: &str = "fn pick(x: i32) -> i32 {
    return match x { 0 => 10, 1 => 20, 7 => 40, _ => 30 };
}

fn main() -> i32 {
    return pick(0);
}
";

/// The `opt` text of [`SWITCH_PROGRAM`] for `target` in the emit form (`stave ir emit --release`,
/// then `stave ir opt`), with the `checked` text it came from.
fn switch_program(dir: &Path, target: Triple) -> (String, String) {
    let src = dir.join("main.stv");
    std::fs::write(&src, SWITCH_PROGRAM).unwrap();
    let mut o = Options::new(Goal::Check).threads(Some(2));
    o.lower = true;
    o.emit_ir = true;
    o.release = true;
    o.target = Some(target);
    let out = pipeline::run(&src, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors() && out.failure.is_none(), "{target}: {:?} {:?}", out.diagnostics, out.failure);
    let checked = emit_text(&out).unwrap();
    let optimized = opt("main.ir", &checked);
    (checked, optimized)
}

/// `text` with the scrutinee of `pick`'s `switch` replaced by `k`.
fn switch_on(text: &str, k: &str) -> String {
    let start = text.find(" main.pick(").expect("fn main.pick");
    let at = start + text[start..].find("\n  switch %0 ").expect("pick switches on its param") + 1;
    format!("{}  switch {k} {}", &text[..at], &text[at + "  switch %0 ".len()..])
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// How binaries for `target` run here: `Some(None)` natively, `Some(Some(cmd))` through `cmd`.
fn runner(target: Triple) -> Option<Option<&'static str>> {
    let how = if exec::can_run(target) {
        Some(None)
    } else if target.name() == "aarch64-linux" && cfg!(target_os = "linux") && have("qemu-aarch64") {
        Some(Some("qemu-aarch64"))
    } else {
        None
    };
    legs::gate(target.name(), how.is_some());
    how
}

/// The scrutinee constants and the exit status `pick` gives with them: a case value, no case
/// value, and constants whose low 64 (or 32) bits equal a case value but whose value does not.
const SCRUTINEES: [(&str, i32); 7] = [
    ("0", 10),
    ("1", 20),
    ("7", 40),
    ("2", 30),
    ("4294967297", 30),
    ("18446744073709551617", 30),
    ("-18446744073709551615", 30),
];

#[test]
fn a_constant_switch_scrutinee_takes_the_case_of_its_value() {
    let tmp = TempDir::new("optsw").unwrap();
    for target in ["x86_64-linux", "aarch64-linux"].map(|t| Triple::parse(t).unwrap()) {
        let (checked, optimized) = switch_program(tmp.path(), target);
        let how = runner(target);
        for (k, want) in SCRUTINEES {
            let name = format!("{target} switch {k}");
            let text = switch_on(&optimized, k);
            let l = load(&name, &text);
            assert_eq!(ir_cmd::fmt(&l).unwrap(), text, "{name}: not canonical");
            let ran = ir_cmd::run(&l, &[]).unwrap_or_else(|e| panic!("{name}: ir run: {:?}", e.lines()));
            assert_eq!(ran.ran.exit, want, "{name}: `stave ir run`");
            let built =
                ir_cmd::build(&l, Some(2)).unwrap_or_else(|e| panic!("{name}: ir build: {:?}", e.lines()));
            if let Some(runner) = how {
                let bin = tmp.path().join(format!("sw-{target}-{}", k.replace('-', "m")));
                built.image.write_file(&bin).unwrap();
                let spec = RunSpec { args: &[], runner, cwd: None, inherit: false };
                let got = exec::run(&bin, &spec).unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
                assert_eq!(got.exit, want, "{name}: the `stave ir build` binary");
            }
            // Below level `opt` the scrutinee must be a value: the base passes never see a constant.
            let low = switch_on(&checked, k);
            match ir_cmd::load(&name, low.as_bytes()) {
                Err(IrError::Input(es)) => {
                    assert!(
                        es.iter().any(|e| e.contains("switch scrutinee must be a value")),
                        "{name}: {es:?}"
                    )
                }
                r => panic!("{name}: a `checked` text with a constant scrutinee: {:?}", r.map(|_| ())),
            }
        }
        let refused = ir_cmd::opt(&load("opt", &switch_on(&optimized, "1")));
        assert!(matches!(refused, Err(IrError::Input(_))), "{target}: `stave ir opt` of an `opt` text");
    }
}

fn stack_lines(text: &str) -> Vec<&str> {
    let stack = ["runstack ", "truncated", "node ", "edge ", "nodes "];
    text.lines().filter(|l| stack.iter().any(|s| l.starts_with(s))).collect()
}

#[test]
fn ir_opt_keeps_the_stack_section() {
    let tmp = TempDir::new("optstack").unwrap();
    let (checked, optimized) = switch_program(tmp.path(), Triple::parse("x86_64-linux").unwrap());
    let lines = stack_lines(&checked);
    assert!(lines.iter().any(|l| l.starts_with("nodes ")), "the emit form has a stack section");
    assert_eq!(stack_lines(&optimized), lines);
    let gate: String = checked.lines().filter(|l| !lines.contains(l)).map(|l| format!("{l}\n")).collect();
    assert!(stack_lines(&opt("gate.ir", &gate)).is_empty(), "the gate form stays without one");
}
