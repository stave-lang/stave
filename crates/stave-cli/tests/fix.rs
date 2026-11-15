//! `stave fix [--dry-run]` (contracts/cli.md): every `exact` fix, applied once, removes its
//! diagnostic; a second run changes nothing; fixes that are not `exact` are left; edits go to the
//! root package only; rounds repeat (at most 8); overlapping edits wait for the next round;
//! `--dry-run` writes nothing and prints a unified diff.

use std::path::{Path, PathBuf};
use std::process::Command;

use stave_cli::fix::{MAX_ROUNDS, apply_edits, plan_round};
use stave_diag::{Applicability, Diagnostic, FileId, Span};

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fix").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

fn stave(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_stave"))
        .args(args)
        .env_remove("STAVE_THREADS")
        .env_remove("STAVE_STD_DIR")
        .env_remove("STAVE_SEMA_SHUFFLE")
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

/// The diagnostics of `stave check --diagnostics json`.
fn check_json(path: &str) -> Vec<serde_json::Value> {
    let (_, _, err) = stave(&["check", path, "--diagnostics", "json"]);
    err.lines().filter(|l| l.starts_with('{')).map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn codes(diags: &[serde_json::Value]) -> Vec<&str> {
    diags.iter().map(|d| d["code"].as_str().unwrap()).collect()
}

const MAIN: &str = "fn main() -> i32 {\n    return 0;\n}\n";

/// (code, the faulty item, the fixed item, `line:col` of the diagnostic).
const EXACT: [(&str, &str, &str, &str); 5] = [
    ("E0023", "import std::io;\n", "import std.io;\n", "1:11"),
    (
        "E0024",
        "fn total(xs: &'a [u8]) -> usize {\n    return xs.len();\n}\n",
        "fn total(xs: & [u8]) -> usize {\n    return xs.len();\n}\n",
        "1:15",
    ),
    ("E0026", "struct Marker;\n", "struct Marker {}\n", "1:14"),
    ("E0027", "fn halt() -> ! {\n    loop { }\n}\n", "fn halt() -> never {\n    loop { }\n}\n", "1:14"),
    (
        "E0210",
        "fn seven() -> u32 {\n    return 7;\n}\n\nfn user() {\n    seven();\n}\n",
        "fn seven() -> u32 {\n    return 7;\n}\n\nfn user() {\n    let _ = seven();\n}\n",
        "6:5",
    ),
];

#[test]
fn each_exact_fix_removes_its_code_and_a_second_run_is_idle() {
    let dir = tmp("exact");
    for (code, faulty, fixed, at) in EXACT {
        let name = format!("{}.stv", code.to_lowercase());
        let file = dir.join(&name);
        let path = file.to_str().unwrap();
        write(&file, &format!("{faulty}\n{MAIN}"));

        let before = check_json(path);
        let own: Vec<_> = before.iter().filter(|d| d["code"] == code).collect();
        assert_eq!(own.len(), 1, "{code}: {before:?}");
        assert_eq!(own[0]["fix"]["applicability"], "exact", "{code}");
        assert_eq!(own[0]["fix"]["edits"].as_array().map(Vec::len), Some(1), "{code}");

        let (c, out, err) = stave(&["fix", path]);
        assert_eq!(c, 0, "{code}: {err}");
        assert_eq!(out, format!("{name}:{at}: {code} fixed\n"), "{code}");
        assert_eq!(err, "", "{code}: nothing remains");
        assert_eq!(read(&file), format!("{fixed}\n{MAIN}"), "{code}");
        assert_eq!(codes(&check_json(path)), Vec::<&str>::new(), "{code}");
        assert_eq!(stave(&["check", path]).0, 0, "{code}");

        let (c, out, err) = stave(&["fix", path]);
        assert_eq!((c, out.as_str(), err.as_str()), (0, "", ""), "{code}: second run");
        assert_eq!(read(&file), format!("{fixed}\n{MAIN}"), "{code}: second run");
    }
}

#[test]
fn all_exact_fixes_in_one_file() {
    let dir = tmp("all");
    let file = dir.join("all.stv");
    let path = file.to_str().unwrap();
    let faulty: String = EXACT.iter().map(|e| format!("{}\n", e.1)).collect();
    let fixed: String = EXACT.iter().map(|e| format!("{}\n", e.2)).collect();
    write(&file, &format!("{faulty}{MAIN}"));
    let (c, out, err) = stave(&["fix", path]);
    assert_eq!(c, 0, "{err}");
    let fixed_codes: Vec<&str> = out.lines().map(|l| l.split(' ').nth(1).unwrap()).collect();
    let mut sorted = fixed_codes.clone();
    sorted.sort();
    assert_eq!(sorted, ["E0023", "E0024", "E0026", "E0027", "E0210"], "{out}");
    assert!(out.lines().all(|l| l.starts_with("all.stv:") && l.ends_with(" fixed")), "{out}");
    assert_eq!(read(&file), format!("{fixed}{MAIN}"));
    assert_eq!(stave(&["fix", path]), (0, String::new(), String::new()));
}

#[test]
fn other_fixes_are_left_to_the_user() {
    let dir = tmp("left");
    let file = dir.join("left.stv");
    let path = file.to_str().unwrap();
    // E0201 carries an edit, E0024 outside `&` and E0027 outside `->` too: none is exact.
    // E0026 for a tuple struct has no edit at all.
    let text = "struct Pair(u32, u32);\n\n\
                fn generic<'a>(x: u32) -> u32 {\n    return x;\n}\n\n\
                struct Never { f: ! }\n\n\
                fn widen(x: i32) -> i64 {\n    return x;\n}\n\n\
                fn main() -> i32 {\n    return 0;\n}\n";
    write(&file, text);
    let before = check_json(path);
    assert_eq!(codes(&before), ["E0026", "E0024", "E0027", "E0201"]);
    let kinds: Vec<&str> = before.iter().map(|d| d["fix"]["applicability"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["placeholder", "maybe", "maybe", "maybe"]);
    for d in &before {
        let edits = d["fix"]["edits"].as_array().unwrap();
        assert_eq!(edits.is_empty(), d["fix"]["applicability"] == "placeholder", "{d}");
    }

    let (c, out, err) = stave(&["fix", path, "--diagnostics", "json"]);
    assert_eq!(c, 1, "{err}");
    assert_eq!(out, "", "nothing applied");
    assert_eq!(read(&file), text);
    let after: Vec<serde_json::Value> = err.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(after, before, "the remaining diagnostics are those of `stave check`");
    // the human form is the default
    let (c, _, err) = stave(&["fix", path]);
    assert_eq!(c, 1);
    assert!(err.starts_with("left.stv:1:12: error[E0026]: "), "{err}");
}

#[test]
fn rounds_repeat_and_stop_after_eight() {
    assert_eq!(MAX_ROUNDS, 8);
    let dir = tmp("rounds");
    // One syntax error is reported per item, so each `::` of an item takes a round.
    let file = dir.join("two.stv");
    let path = file.to_str().unwrap();
    write(&file, &format!("fn two() -> usize {{\n    return a::b::c;\n}}\n\n{MAIN}"));
    let (c, out, err) = stave(&["fix", path]);
    assert_eq!(out, "two.stv:2:13: E0023 fixed\ntwo.stv:2:15: E0023 fixed\n", "{err}");
    assert_eq!(read(&file), format!("fn two() -> usize {{\n    return a.b.c;\n}}\n\n{MAIN}"));
    // `a` is no name: the fixed path now reaches name resolution
    assert_eq!(c, 1);
    assert!(err.contains("error[E0101]") && !err.contains("E0023"), "{err}");

    let file = dir.join("ten.stv");
    let path = file.to_str().unwrap();
    let chain = |sep: &dyn Fn(usize) -> &'static str| {
        let mut s = String::from("a");
        for i in 0..10 {
            s.push_str(sep(i));
            s.push('b');
        }
        format!("fn ten() -> usize {{\n    return {s};\n}}\n\n{MAIN}")
    };
    write(&file, &chain(&|_| "::"));
    let (c, out, err) = stave(&["fix", path]);
    assert_eq!(c, 1, "{err}");
    assert_eq!(out.lines().count(), 8, "{out}");
    assert_eq!(read(&file), chain(&|i| if i < 8 { "." } else { "::" }));
    assert_eq!(err.matches("error[E0023]").count(), 1, "the ninth `::` remains: {err}");
}

#[test]
fn only_the_root_package_is_edited() {
    let dir = tmp("root_only");
    let app = dir.join("app");
    let manifest = |name: &str, kind: &str, deps: &str| {
        format!("name = \"{name}\"\nkind = \"{kind}\"\nversion = \"0.1.0\"\n{deps}")
    };
    let lib = |name: &str| format!("pub fn {name}() -> u32 {{\n    1u32 + 1u32;\n    return 1;\n}}\n");
    write(
        &app.join("stave.toml"),
        &manifest(
            "app",
            "bin",
            "\n[dependencies]\ninner = { path = \"inner\" }\nouter = { path = \"../outer\" }\n",
        ),
    );
    write(&app.join("inner/stave.toml"), &manifest("inner", "lib", ""));
    write(&app.join("inner/src/lib.stv"), &lib("one"));
    write(&dir.join("outer/stave.toml"), &manifest("outer", "lib", ""));
    write(&dir.join("outer/src/lib.stv"), &lib("two"));
    let main = |discard: &str| {
        format!(
            "import inner;\nimport outer;\nimport util;\n\n\
             fn main() -> i32 {{\n    {discard}inner.one();\n    {discard}outer.two();\n    \
             {discard}util.three();\n    return 0;\n}}\n"
        )
    };
    let util =
        |discard: &str| format!("pub fn three() -> u32 {{\n    {discard}inner.one();\n    return 3;\n}}\n");
    write(&app.join("src/main.stv"), &main(""));
    write(&app.join("src/util.stv"), &format!("import inner;\n\n{}", util("")));

    let (c, out, err) = stave(&["fix", app.to_str().unwrap()]);
    assert_eq!(c, 1, "the dependencies keep their errors: {err}");
    assert_eq!(
        out,
        "src/main.stv:6:5: E0210 fixed\nsrc/main.stv:7:5: E0210 fixed\nsrc/main.stv:8:5: E0210 fixed\n\
         src/util.stv:4:5: E0210 fixed\n"
    );
    assert_eq!(read(&app.join("src/main.stv")), main("let _ = "));
    assert_eq!(read(&app.join("src/util.stv")), format!("import inner;\n\n{}", util("let _ = ")));
    assert_eq!(read(&app.join("inner/src/lib.stv")), lib("one"), "a dependency inside the root directory");
    assert_eq!(read(&dir.join("outer/src/lib.stv")), lib("two"), "a dependency outside it");
    assert_eq!(err.matches("error[E0210]").count(), 2, "{err}");
    assert!(!err.contains("src/main.stv") && !err.contains("src/util.stv"), "{err}");
}

#[test]
fn dry_run_prints_a_diff_and_writes_nothing() {
    let dir = tmp("dry");
    let file = dir.join("both.stv");
    let path = file.to_str().unwrap();
    let text = "struct Marker;\n\nfn seven() -> u32 {\n    return 7;\n}\n\n\
                fn main() -> i32 {\n    seven();\n    return 0;\n}\n";
    write(&file, text);
    let (c, out, err) = stave(&["fix", path, "--dry-run"]);
    assert_eq!(c, 0, "no diagnostic would remain: {err}");
    assert_eq!(err, "");
    assert_eq!(
        out,
        "--- a/both.stv\n+++ b/both.stv\n@@ -1,10 +1,10 @@\n\
         -struct Marker;\n+struct Marker {}\n \n fn seven() -> u32 {\n     return 7;\n }\n \n \
         fn main() -> i32 {\n-    seven();\n+    let _ = seven();\n     return 0;\n }\n"
    );
    assert_eq!(read(&file), text, "nothing written");
    // the same call again: the file is still unfixed, the diff is the same
    assert_eq!(stave(&["fix", path, "--dry-run"]).1, out);

    let (c, fixed, _) = stave(&["fix", path]);
    assert_eq!(c, 0);
    assert_eq!(fixed, "both.stv:1:14: E0026 fixed\nboth.stv:8:5: E0210 fixed\n");
    assert_eq!(
        read(&file),
        text.replace("Marker;", "Marker {}").replace("    seven();", "    let _ = seven();")
    );
    assert_eq!(stave(&["fix", path, "--dry-run"]), (0, String::new(), String::new()));
}

#[test]
fn usage_errors() {
    let (c, _, err) = stave(&["fix", "/nonexistent/stave-pkg"]);
    assert_eq!(c, 2);
    assert!(err.starts_with("stave: /nonexistent/stave-pkg: "), "{err}");
    let (c, _, err) = stave(&["fix", "--release"]);
    assert_eq!(c, 2);
    assert!(err.contains("`stave fix` does not take --release"), "{err}");
}

// ---------- the plan of one round ----------

fn lifetime(offset: u32) -> Diagnostic {
    let span = Span::new(FileId(0), offset, 2);
    Diagnostic::new("E0024", Some("ref"), &[("text", "'a")], Some(span), &[]).unwrap()
}

fn path_sep(file: u32, offset: u32) -> Diagnostic {
    Diagnostic::new("E0023", None, &[], Some(Span::new(FileId(file), offset, 2)), &[]).unwrap()
}

fn discard(offset: u32) -> Diagnostic {
    let at = Span::new(FileId(0), offset, 4);
    let start = Span::new(FileId(0), offset, 0);
    Diagnostic::new("E0210", None, &[("type", "u32")], Some(at), &[("start", start)]).unwrap()
}

#[test]
fn overlapping_edits_wait_for_the_next_round() {
    let mismatch = Diagnostic::new(
        "E0201",
        None,
        &[("expected", "i64"), ("found", "i32"), ("operand", "x")],
        Some(Span::new(FileId(0), 40, 1)),
        &[],
    )
    .unwrap();
    assert_eq!(mismatch.fix().map(|f| f.applicability), Some(Applicability::Maybe));
    let diags = [
        lifetime(9),     // bytes 9..11
        path_sep(0, 10), // bytes 10..12: shares byte 10 with the fix before it
        path_sep(1, 10), // the same bytes of another file
        path_sep(0, 11), // bytes 11..13: free again, the overlapping fix was not applied
        discard(20),     // an insertion at 20
        discard(20),     // a second insertion at the same offset
        path_sep(0, 20), // bytes 20..22: an insertion at their start is no overlap
        discard(21),     // an insertion inside bytes 20..22
        mismatch,        // not exact
    ];
    let all = |_: Span| true;
    assert_eq!(plan_round(&diags, all), [0, 2, 3, 4, 6]);
    // the next round's check reports the skipped ones again: now nothing is in their way
    let deferred = [diags[1].clone(), diags[5].clone(), diags[7].clone()];
    assert_eq!(plan_round(&deferred, all), [0, 1, 2]);
    // a fix with an edit outside the fixable files is never applied
    assert_eq!(plan_round(&diags, |s| s.file == FileId(1)), [2]);
    assert_eq!(plan_round(&diags, |_| false), Vec::<usize>::new());
}

#[test]
fn a_round_applies_its_edits_at_the_offsets_of_its_check() {
    let text = b"fn f(x: &'a u8) { g::h(); k(); }";
    let diags = [lifetime(9), path_sep(0, 19), discard(26)];
    let chosen = plan_round(&diags, |_| true);
    assert_eq!(chosen, [0, 1, 2]);
    let edits: Vec<(usize, usize, &str)> = chosen
        .iter()
        .flat_map(|&i| &diags[i].fix().unwrap().edits)
        .map(|e| (e.span.offset as usize, (e.span.offset + e.span.len) as usize, e.replacement.as_str()))
        .collect();
    assert_eq!(apply_edits(text, &edits), b"fn f(x: & u8) { g.h(); let _ = k(); }");
}
