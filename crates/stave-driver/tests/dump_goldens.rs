//! Dump goldens: `tests/conformance/dumps/<case>/input` is a package; the expected `tokens.txt
//! items.txt ast.txt strings.txt names.txt sigs.txt kinds.txt types.txt diags.txt` sit next to it
//! (formats: contracts/stage-gates.md). `stats.txt` must be written and is never compared. Dumps
//! cover the whole program, so the cases build against the pinned stand-in library
//! `tests/conformance/dumps/_lib` (`std/`, an empty `runtime/`; the CLI test uses it through
//! `STAVE_STD_DIR`), the `kinds_` cases against `_lib_kinds` (`_lib` plus the marker traits,
//! `std.cell`, and `std.sync`): the goldens do not move with the shipped std/.
//!
//! IR text goldens (contracts/checked-ir.md): every `ir_<construct>/` case is a package `input`
//! plus the expected `ir.txt` of a debug lowering (`stave check --dump=ir`), its `lir.txt`
//! (`--dump=lir`: the module before the removal of proven checks) and, when present,
//! `ir.release.txt` of a `--release` lowering (instance root sets; the `ir_bounds_*` cases: the
//! bounds checks left in each mode, which must be the same ones) and `opt.release.txt` (its
//! `--dump=opt`). A case whose input has an error diagnostic has none of them, and none may be
//! written for it. Every `ir_` case also has the `sigs.txt` and `kinds.txt` of a check.
//!
//! `STAVE_BLESS=1` rewrites the expected files from the current output (review the diff!).

use std::path::{Path, PathBuf};

#[path = "../../../tests/support/checked_ir.rs"]
mod checked_ir;

use stave_driver::dump::{DumpStage, emit_text, render_to_vec, write_dumps};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;

const CASES: [&str; 9] = [
    "valid_small",
    "dropped_item",
    "unbalanced",
    "multi_file",
    "intern_collision",
    "sigs_decls",
    "sigs_errors",
    "kinds_layout",
    "kinds_errors",
];
/// The `ir_` cases that also have a golden of the release lowering.
const RELEASE_CASES: [&str; 3] = ["ir_bounds_kept", "ir_bounds_removed", "ir_roots"];
/// The `ir_` cases that also have the golden of the release passes' output (`opt.release.txt`).
const OPT_CASES: [&str; 5] =
    ["ir_bounds_kept", "ir_bounds_removed", "ir_roots", "ir_for_range", "ir_for_chunk_scalar"];
const COMPARED: [DumpStage; 9] = [
    DumpStage::Tokens,
    DumpStage::Items,
    DumpStage::Ast,
    DumpStage::Strings,
    DumpStage::Names,
    DumpStage::Sigs,
    DumpStage::Kinds,
    DumpStage::Types,
    DumpStage::Diags,
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dumps_dir(case: &str) -> PathBuf {
    repo().join("tests/conformance/dumps").join(case)
}

/// The pinned library of `case`: `_lib_kinds` for the `kinds_` cases, else `_lib`.
fn golden_std(case: &str) -> Library {
    let lib = if case.starts_with("kinds_") { "_lib_kinds" } else { "_lib" };
    Library::in_dir(&repo().join("tests/conformance/dumps").join(lib))
}

fn output(case: &str, threads: usize) -> Output {
    let opts = Options::new(Goal::Check).threads(Some(threads));
    pipeline::run(&dumps_dir(case).join("input"), &golden_std(case), &opts).unwrap()
}

/// Writes every dump of `case` into a directory of its own per (`test`, case, threads): tests
/// run in parallel.
fn run_case(test: &str, case: &str, threads: usize) -> PathBuf {
    let out = output(case, threads);
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join("dump_goldens").join(format!("{test}_{case}_{threads}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut stages = COMPARED.to_vec();
    stages.push(DumpStage::Stats);
    write_dumps(&out, &stages, &dir).unwrap();
    dir
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Line discipline of every dump: `\n`-terminated, no trailing spaces, no empty lines.
fn check_lines(name: &str, text: &str) {
    assert!(text.is_empty() || text.ends_with('\n'), "{name}: not newline-terminated");
    for (i, l) in text.lines().enumerate() {
        assert!(!l.is_empty() && !l.ends_with(' '), "{name}:{}: {l:?}", i + 1);
    }
}

#[test]
fn goldens() {
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    let mut failures = Vec::new();
    for case in CASES {
        let dir = run_case("goldens", case, 2);
        let stats = read(&dir.join("stats.txt"));
        let stages: Vec<&str> = stats.lines().map(|l| l.split(' ').next().unwrap_or("")).collect();
        assert_eq!(
            stages,
            ["read", "lex", "items", "parse", "intern", "resolve", "typeck", "borrowck"],
            "{case}: stats.txt"
        );
        for l in stats.lines() {
            assert_eq!(l.split(' ').count(), 6, "{case}: stats line {l:?}");
            assert!(l.ends_with(" - - -"), "{case}: stage0 has no rounds/fallbacks/peak_bytes: {l:?}");
        }
        for s in COMPARED {
            let got = read(&dir.join(s.file_name()));
            check_lines(&format!("{case}/{}", s.file_name()), &got);
            let want_path = dumps_dir(case).join(s.file_name());
            if bless {
                std::fs::write(&want_path, &got).unwrap();
                continue;
            }
            match std::fs::read_to_string(&want_path) {
                Ok(want) if want == got => {}
                Ok(want) => failures.push(format!(
                    "{case}/{}: differs from the golden\n--- want\n{want}--- got\n{got}",
                    s.file_name()
                )),
                Err(e) => failures.push(format!("{}: {e}", want_path.display())),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Contract properties the goldens must show, checked independently of the golden text.
#[test]
fn golden_properties() {
    // tokens: never EOF; ordered by (F, O).
    for case in CASES {
        let dir = run_case("properties", case, 1);
        let tokens = read(&dir.join("tokens.txt"));
        let keys: Vec<(u32, u32)> = tokens
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split(' ').collect();
                assert_eq!(f.len(), 3, "{l:?}");
                assert_ne!(f[2], "EOF");
                (f[0].parse().unwrap(), f[1].parse().unwrap())
            })
            .collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "{case}: tokens not in (F, O) order");
        // diags: schema lines in canonical order.
        for l in read(&dir.join("diags.txt")).lines() {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert_eq!(v["kind"], "diagnostic");
            assert!(v["fix"]["description"].is_string(), "errors carry a fix: {l}");
        }
        // items: I is 0.. in order; ast lines only for ok items.
        let items = read(&dir.join("items.txt"));
        let mut ok = std::collections::BTreeSet::new();
        for (k, l) in items.lines().enumerate() {
            let f: Vec<&str> = l.split(' ').collect();
            assert_eq!(f.len(), 5, "{l:?}");
            assert_eq!(f[0], k.to_string());
            assert!(f[2].parse::<u32>().unwrap() < f[3].parse::<u32>().unwrap());
            if f[4] == "ok" {
                ok.insert(f[0].to_string());
            } else {
                assert_eq!(f[4], "dropped");
            }
        }
        for l in read(&dir.join("ast.txt")).lines() {
            let f: Vec<&str> = l.split(' ').collect();
            assert_eq!(f.len(), 5, "{l:?}");
            assert!(ok.contains(f[0]), "{case}: ast line for a dropped item: {l}");
        }
        match case {
            "unbalanced" => {
                // F 0..3: the stand-in runtime and std; F 4: the package's src/main.stv.
                assert!(items.lines().all(|l| l.split(' ').nth(1) != Some("4")), "E0016: no items");
                assert!(tokens.lines().any(|l| l.starts_with("4 ")), "tokens before the stop are kept");
                assert!(read(&dir.join("diags.txt")).contains("\"code\":\"E0016\""));
            }
            "dropped_item" => assert_eq!(items.lines().filter(|l| l.ends_with(" dropped")).count(), 1),
            "valid_small" | "intern_collision" | "sigs_decls" | "kinds_layout" => {
                assert!(read(&dir.join("diags.txt")).is_empty())
            }
            "sigs_errors" => {
                let d = read(&dir.join("diags.txt"));
                assert!(d.contains("\"code\":\"E0101\"") && d.contains("\"code\":\"E0203\""), "{d}");
            }
            "kinds_errors" => {
                let d = read(&dir.join("diags.txt"));
                assert!(d.contains("\"code\":\"E0903\"") && d.contains("\"code\":\"E0421\""), "{d}");
            }
            "multi_file" => {
                // F = rank of (package, path): multi_file's src/main.stv, src/util.stv,
                // src/util/math.stv, then runtime's empty.stv (no tokens), std's core.stv, io.stv,
                // string.stv.
                let files: std::collections::BTreeSet<&str> =
                    tokens.lines().filter_map(|l| l.split(' ').next()).collect();
                assert_eq!(files.into_iter().collect::<Vec<_>>(), ["0", "1", "2", "4", "5", "6"]);
                let d = read(&dir.join("diags.txt"));
                assert!(d.contains("\"file\":\"src/util/math.stv\",\"line\":6"), "{d}");
            }
            _ => unreachable!(),
        }
    }
}

/// `strings`, `names`, `types` field rules, checked independently of the golden text.
#[test]
fn semantic_dump_properties() {
    for case in CASES {
        let dir = run_case("semantic", case, 1);
        // strings: S = 0.. in order, ranked by (hash32, bytes), quoted.
        let mut prev: Option<(u32, String)> = None;
        for (k, l) in read(&dir.join("strings.txt")).lines().enumerate() {
            let (s, rest) = l.split_once(' ').unwrap();
            assert_eq!(s, k.to_string(), "{case}: {l}");
            let (h, q) = rest.split_once(' ').unwrap();
            assert_eq!(h.len(), 8, "{l}");
            assert!(q.starts_with('"') && q.ends_with('"') && q.len() >= 2, "{l}");
            let key = (u32::from_str_radix(h, 16).unwrap(), q.to_string());
            if let Some(p) = &prev {
                assert!(p.0 <= key.0, "{case}: strings not ranked by hash: {l}");
            }
            prev = Some(key);
        }
        // names: (I, N) non-decreasing; two forms.
        let mut last = (0u32, 0u32);
        for l in read(&dir.join("names.txt")).lines() {
            let f: Vec<&str> = l.split(' ').collect();
            let key: (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
            assert!(key >= last, "{case}: names out of order: {l}");
            last = key;
            match f[3] {
                "decl" => assert_eq!(f.len(), 5, "{l}"),
                "local" => {
                    assert_eq!(f.len(), 6, "{l}");
                    assert_eq!(f[4], f[0], "a local of the same item: {l}");
                }
                _ => panic!("{case}: bad names line {l}"),
            }
            assert_eq!(f[2], "->");
        }
        // types: the table (`t<T> key`, by T) first, then `I N t<T>|-` by (I, N); every T a
        // line names is in the table.
        let types = read(&dir.join("types.txt"));
        let mut table = std::collections::BTreeSet::new();
        let mut in_table = true;
        let mut last_t: Option<u32> = None;
        let mut last_n: Option<(u32, u32)> = None;
        for l in types.lines() {
            if l.starts_with('t') {
                assert!(in_table, "{case}: table line after node lines: {l}");
                let (t, key) = l.split_once(' ').unwrap();
                let t: u32 = t[1..].parse().unwrap();
                assert!(last_t.is_none_or(|p| p < t), "{case}: table not by T: {l}");
                assert!(!key.is_empty());
                last_t = Some(t);
                table.insert(t);
            } else {
                in_table = false;
                let f: Vec<&str> = l.split(' ').collect();
                assert_eq!(f.len(), 3, "{l}");
                let key: (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
                assert!(last_n.is_none_or(|p| p < key), "{case}: type lines out of order: {l}");
                last_n = Some(key);
                if f[2] != "-" {
                    let t: u32 = f[2].strip_prefix('t').unwrap().parse().unwrap();
                    assert!(table.contains(&t), "{case}: t{t} not in the table");
                }
            }
        }
        if case == "valid_small" {
            assert!(types.lines().any(|l| l.ends_with(" int s 32")), "{case}: i32 in the table");
            let names = read(&dir.join("names.txt"));
            assert!(names.contains(" -> local "), "{case}: locals resolve");
            assert!(names.contains(" -> decl "), "{case}: items resolve");
        }
    }
}

/// A dump's type table (`t<T> key` lines) with every key expanded: each `t<T>` component is
/// replaced by its own expanded key in parentheses, so two tables can be compared modulo their T
/// numbering. The error type (no line) expands to `error`.
fn deep_keys(text: &str) -> Vec<(u32, String)> {
    let table: std::collections::BTreeMap<u32, &str> = text
        .lines()
        .filter_map(|l| {
            let (t, key) = l.split_once(' ')?;
            Some((t.strip_prefix('t')?.parse().ok()?, key))
        })
        .collect();
    fn expand(
        t: u32,
        table: &std::collections::BTreeMap<u32, &str>,
        memo: &mut std::collections::BTreeMap<u32, String>,
    ) -> String {
        if let Some(s) = memo.get(&t) {
            return s.clone();
        }
        let Some(key) = table.get(&t) else { return "error".to_string() };
        let parts: Vec<String> = key
            .split(' ')
            .map(|w| match w.strip_prefix('t').and_then(|n| n.parse::<u32>().ok()) {
                Some(c) => format!("({})", expand(c, table, memo)),
                None => w.to_string(),
            })
            .collect();
        let s = parts.join(" ");
        memo.insert(t, s.clone());
        s
    }
    let mut memo = std::collections::BTreeMap::new();
    table.keys().map(|&t| (t, expand(t, &table, &mut memo))).collect()
}

/// The numbers of the `<prefix><n>` words of a line.
fn refs(line: &str, prefix: char) -> Vec<u32> {
    line.split([' ', '[', ']', ','])
        .filter_map(|w| w.strip_prefix(prefix).and_then(|n| n.parse().ok()))
        .collect()
}

const DECL_KINDS: [&str; 14] = [
    "module",
    "fn",
    "struct",
    "enum",
    "trait",
    "const",
    "static",
    "alias",
    "impl",
    "field",
    "variant",
    "method",
    "trait-method",
    "error",
];

/// `sigs` field rules, and its table against the `types` table of the same build: the `sigs`
/// table is dense from t0, its keys are distinct, and the keys both tables hold are in the same
/// order in both (a renumbering of a subset of one canonical order). Same bytes at 1 and 8
/// threads.
#[test]
fn sigs_table_is_a_renumbered_subset_of_types() {
    let mut cases: Vec<String> = CASES.iter().map(|c| c.to_string()).collect();
    cases.extend(ir_cases());
    for case in &cases {
        let out = output(case, 1);
        let sigs = String::from_utf8(render_to_vec(DumpStage::Sigs, &out).unwrap()).unwrap();
        let types = String::from_utf8(render_to_vec(DumpStage::Types, &out).unwrap()).unwrap();
        let out8 = output(case, 8);
        assert_eq!(sigs.as_bytes(), render_to_vec(DumpStage::Sigs, &out8).unwrap(), "{case}: 1 vs 8 threads");
        check_lines(&format!("{case}/sigs.txt"), &sigs);
        let n_types = sigs.lines().take_while(|l| l.starts_with('t')).count() as u32;
        let decls: Vec<&str> = sigs.lines().skip(n_types as usize).collect();
        for (k, l) in sigs.lines().take(n_types as usize).enumerate() {
            assert!(l.starts_with(&format!("t{k} ")), "{case}: table not dense from t0: {l}");
        }
        assert!(!decls.is_empty(), "{case}: no decl lines");
        for (k, l) in decls.iter().enumerate() {
            let f: Vec<&str> = l.split(' ').collect();
            assert!(f.len() >= 9, "{case}: {l}");
            assert_eq!(f[0], format!("d{k}"), "{case}: decl lines not by D from d0: {l}");
            assert!(DECL_KINDS.contains(&f[1]), "{case}: kind: {l}");
            let sig = match f[1] {
                "fn" | "method" | "trait-method" => Some(" params["),
                "struct" | "enum" | "trait" => Some(" gen["),
                "impl" => Some(" trait "),
                "const" | "static" | "field" => Some(" ty "),
                "alias" => Some(" target "),
                "module" => Some(" path["),
                _ => None,
            };
            if let Some(s) = sig {
                assert!(l.contains(s), "{case}: no `{}` field: {l}", s.trim());
            }
            if f[1] == "error" {
                assert_eq!(f.len(), 9, "{case}: an error decl has no signature: {l}");
            }
            for t in refs(l, 't') {
                assert!(t <= n_types, "{case}: t{t} not in the table: {l}");
            }
            for d in refs(l, 'd') {
                assert!((d as usize) < decls.len(), "{case}: d{d} is no decl: {l}");
            }
        }
        for l in sigs.lines().take(n_types as usize) {
            for t in refs(l.split_once(' ').unwrap().1, 't') {
                assert!(t <= n_types, "{case}: t{t} not in the table: {l}");
            }
        }
        // A `fntype` key does not print the deny clause, so two fn types that differ only there
        // share a text; they are adjacent in the order, and the first of each text stands for it.
        let mut local = deep_keys(&sigs);
        let mut seen = std::collections::BTreeSet::new();
        for (t, k) in &local {
            assert!(seen.insert(k.clone()) || k.contains("fntype"), "{case}: t{t} repeats the key {k}");
        }
        let mut seen = std::collections::BTreeSet::new();
        local.retain(|(_, k)| seen.insert(k.clone()));
        let mut global: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        for (t, k) in deep_keys(&types) {
            global.entry(k).or_insert(t);
        }
        let common: Vec<(u32, u32)> =
            local.iter().filter_map(|(t, k)| global.get(k).map(|&g| (*t, g))).collect();
        assert!(
            common.windows(2).all(|w| w[0].1 < w[1].1),
            "{case}: sigs and types disagree on the order: {common:?}"
        );
        if !out.has_errors() {
            assert!(!common.is_empty(), "{case}: no type in both tables");
        }
    }
}

/// The kind flags of `kinds`, in order.
const KIND_FLAGS: [&str; 10] =
    ["copy", "view", "send", "sync", "sized", "flat", "frozen", "drop-impl", "drop", "lock"];

/// `kinds` field rules against `sigs` of the same build, and its layouts against the back end's:
/// one `t<T>` line per `sigs` table type, by T, then `impl` lines (one per impl decl), then
/// `apply` lines by (T, trait); every `t<T>` / `d<D>` named exists; same bytes at 1 and 8
/// threads. On every case without errors that lowers (each `ir_` case and `kinds_layout` must),
/// each type both the `sigs` table and the IR type table hold (same expanded key) has the size,
/// alignment, and field offsets stave-ir's layout gives it, and no layout where stave-ir has
/// none.
#[test]
fn kinds_lines_and_ir_layouts() {
    let mut cases: Vec<String> = CASES.iter().map(|c| c.to_string()).collect();
    cases.extend(ir_cases());
    let mut compared = 0;
    let mut compared_records = 0;
    for case in &cases {
        let out = output(case, 1);
        let kinds = String::from_utf8(render_to_vec(DumpStage::Kinds, &out).unwrap()).unwrap();
        let sigs = String::from_utf8(render_to_vec(DumpStage::Sigs, &out).unwrap()).unwrap();
        let out8 = output(case, 8);
        assert_eq!(
            kinds.as_bytes(),
            render_to_vec(DumpStage::Kinds, &out8).unwrap(),
            "{case}: 1 vs 8 threads"
        );
        check_lines(&format!("{case}/kinds.txt"), &kinds);
        let n_types = sigs.lines().take_while(|l| l.starts_with('t')).count();
        let decls: Vec<&str> = sigs.lines().skip(n_types).collect();
        let impl_decls: std::collections::BTreeSet<String> = decls
            .iter()
            .filter(|l| l.split(' ').nth(1) == Some("impl"))
            .map(|l| l.split(' ').next().unwrap().to_string())
            .collect();
        let tlines: Vec<&str> = kinds.lines().take_while(|l| l.starts_with('t')).collect();
        assert_eq!(tlines.len(), n_types, "{case}: one kinds line per sigs table type");
        let mut sizes: Vec<String> = Vec::new();
        for (k, l) in tlines.iter().enumerate() {
            let f: Vec<&str> = l.split(' ').collect();
            assert_eq!(f[0], format!("t{k}"), "{case}: not by T from t0: {l}");
            if f[1] != "-" {
                let flags: Vec<usize> =
                    f[1].split(',').map(|x| KIND_FLAGS.iter().position(|k| *k == x).expect(l)).collect();
                assert!(flags.windows(2).all(|w| w[0] < w[1]), "{case}: flag order: {l}");
            }
            assert_eq!(f[2], "depth", "{l}");
            assert!(f[3] == "-" || f[3] == "over" || f[3].parse::<u32>().is_ok_and(|d| d >= 1), "{l}");
            assert_eq!(f[4], "size", "{l}");
            let rest = f[5..].join(" ");
            assert!(
                ["param", "unsized", "overflow", "none"].contains(&rest.as_str()) || f[6] == "align",
                "{case}: size form: {l}"
            );
            sizes.push(rest);
        }
        let mut impls_seen = std::collections::BTreeSet::new();
        let mut last_apply: Option<(u32, u32)> = None;
        for l in kinds.lines().skip(n_types) {
            let f: Vec<&str> = l.split(' ').collect();
            match f[0] {
                "impl" => {
                    assert!(last_apply.is_none(), "{case}: impl line after apply lines: {l}");
                    assert_eq!((f[2], f[4], f[6]), ("trait", "head", "gen"), "{l}");
                    assert!(impl_decls.contains(f[1]), "{case}: {} is no impl decl: {l}", f[1]);
                    assert!(impls_seen.insert(f[1].to_string()), "{case}: impl listed twice: {l}");
                }
                "apply" => {
                    let key = (refs(f[1], 't')[0], refs(f[2], 'd')[0]);
                    assert!(last_apply.is_none_or(|p| p < key), "{case}: apply lines not by (T, trait): {l}");
                    last_apply = Some(key);
                    for d in refs(f[3..].join(" ").as_str(), 'd') {
                        assert!(impl_decls.contains(&format!("d{d}")), "{case}: d{d} is no impl: {l}");
                    }
                }
                _ => panic!("{case}: bad kinds line {l}"),
            }
            for t in refs(l, 't') {
                assert!((t as usize) < n_types, "{case}: t{t} not in the sigs table: {l}");
            }
            for d in refs(l, 'd') {
                assert!((d as usize) < decls.len(), "{case}: d{d} is no decl: {l}");
            }
        }
        assert_eq!(impls_seen, impl_decls, "{case}: every impl decl has one impl line");

        if out.has_errors() {
            continue;
        }
        let ir_out = lowered(case, false, 1);
        let Some(m) = ir_out.module.as_ref() else {
            let f = ir_out.failure.as_ref().map(|f| f.to_string());
            assert!(!case.starts_with("ir_") && case != "kinds_layout", "{case}: no module ({f:?})");
            continue;
        };
        let layouts = stave_ir::layout::Layouts::compute(&m.types);
        let ir_table: String = (0..m.types.len())
            .map(|i| format!("t{i} {}\n", m.types.key_text(stave_ir::TypeId(i as u32))))
            .collect();
        let mut ir_of: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        for (t, k) in deep_keys(&ir_table) {
            ir_of.entry(k).or_insert(t);
        }
        for (t, k) in deep_keys(&sigs) {
            let Some(&i) = ir_of.get(&k) else { continue };
            let got = &sizes[t as usize];
            let want = match layouts.get_layout(stave_ir::TypeId(i)) {
                Err(_) => {
                    assert!(
                        !got.contains(" align "),
                        "{case}: t{t} ({k}): stave-ir has no layout, kinds: {got}"
                    );
                    continue;
                }
                Ok(l) => {
                    let list = |v: &[(u64, stave_ir::TypeId)]| {
                        v.iter().map(|(o, _)| o.to_string()).collect::<Vec<_>>().join(" ")
                    };
                    let record = k.starts_with("adt ") || k.starts_with("tuple");
                    let shape = match &l.shape {
                        stave_ir::layout::Shape::Fields(fs) if record => format!(" fields[{}]", list(fs)),
                        stave_ir::layout::Shape::Enum(vs) if record => format!(
                            " tag 4 variants[{}]",
                            vs.iter().map(|v| format!("[{}]", list(v))).collect::<Vec<_>>().join(" ")
                        ),
                        _ => String::new(),
                    };
                    if record {
                        compared_records += 1;
                    }
                    format!("{} align {}{shape}", l.size, l.align)
                }
            };
            assert_eq!(got, &want, "{case}: t{t} ({k}): kinds vs stave-ir layout");
            compared += 1;
        }
    }
    assert!(compared > 150 && compared_records > 20, "{compared} types, {compared_records} records compared");
}

/// `sigs.txt` and `kinds.txt` of every `ir_` case (a check, the pinned library).
#[test]
fn sigs_and_kinds_goldens_of_ir_cases() {
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    let mut failures = Vec::new();
    for case in ir_cases() {
        let out = output(&case, 2);
        for stage in [DumpStage::Sigs, DumpStage::Kinds] {
            let got = String::from_utf8(render_to_vec(stage, &out).unwrap()).unwrap();
            let file = stage.file_name();
            let want_path = dumps_dir(&case).join(&file);
            if bless {
                std::fs::write(&want_path, &got).unwrap();
                continue;
            }
            match std::fs::read_to_string(&want_path) {
                Ok(want) if want == got => {}
                Ok(want) => failures
                    .push(format!("{case}/{file}: differs from the golden\n--- want\n{want}--- got\n{got}")),
                Err(e) => failures.push(format!("{}: {e}", want_path.display())),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn ir_dump_of_a_check_is_refused() {
    let out = output("valid_small", 1);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dump_refused");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.borrowck.is_some());
    // `ir` needs a lowered module: a plain check has none.
    let e = write_dumps(&out, &[DumpStage::Ir], &dir).unwrap_err();
    assert!(e.to_string().contains("stopped before its stage"), "{e}");
}

/// The `ir_<construct>` cases, in name order.
fn ir_cases() -> Vec<String> {
    let root = repo().join("tests/conformance/dumps");
    let mut v: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("ir_"))
        .collect();
    v.sort();
    v
}

/// `case` lowered in debug or release mode.
fn lowered(case: &str, release: bool, threads: usize) -> Output {
    let mut opts = Options::new(Goal::Check).threads(Some(threads));
    opts.lower = true;
    opts.keep_lowered = true;
    opts.release = release;
    // Fixed target: the dispatch arms dropped for another arch must not depend on the host.
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    pipeline::run(&dumps_dir(case).join("input"), &golden_std(case), &opts).unwrap()
}

/// Writes `lir.txt`, `ir.txt`, with `--release` `opt.txt` (plus `diags.txt`) of `case` lowered in
/// debug or release mode.
fn run_ir_case(case: &str, release: bool, threads: usize) -> (PathBuf, Output) {
    let out = lowered(case, release, threads);
    // one directory per call: tests run in parallel and lower the same case with equal options
    static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("dump_goldens")
        .join(format!("ir_{case}_{}_{threads}_{call}", if release { "release" } else { "debug" }));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for stale in ["lir.txt", "ir.txt", "opt.txt"] {
        std::fs::write(dir.join(stale), "stale").unwrap();
    }
    let mut stages = vec![DumpStage::Lir, DumpStage::Ir, DumpStage::Diags];
    if release {
        stages.insert(2, DumpStage::Opt);
    }
    write_dumps(&out, &stages, &dir).unwrap();
    (dir, out)
}

/// The goldens of one run: (file in the case directory, file in the run directory).
fn ir_files(case: &str, release: bool) -> Vec<(&'static str, &'static str)> {
    if release {
        let mut v = Vec::new();
        if RELEASE_CASES.contains(&case) {
            v.push(("ir.release.txt", "ir.txt"));
        }
        if OPT_CASES.contains(&case) {
            v.push(("opt.release.txt", "opt.txt"));
        }
        v
    } else {
        vec![("ir.txt", "ir.txt"), ("lir.txt", "lir.txt")]
    }
}

#[test]
fn ir_goldens() {
    let bless = std::env::var_os("STAVE_BLESS").is_some();
    let cases = ir_cases();
    assert!(cases.len() >= 15, "ir cases: {cases:?}");
    let mut failures = Vec::new();
    for case in &cases {
        for release in [false, true] {
            let files = ir_files(case, release);
            if files.is_empty() {
                continue;
            }
            let (dir, out) = run_ir_case(case, release, 2);
            for (file, made) in files {
                let want_path = dumps_dir(case).join(file);
                let got_path = dir.join(made);
                if out.has_errors() {
                    // No IR for a build with errors: the stale file is removed, none is expected.
                    assert!(!got_path.exists(), "{case}: {made} written for a build with errors");
                    assert!(!want_path.exists(), "{case}: a golden {file} for a build with errors");
                    assert!(!read(&dir.join("diags.txt")).is_empty());
                    continue;
                }
                assert!(
                    out.module.is_some(),
                    "{case}: no module ({:?})",
                    out.failure.as_ref().map(|f| f.to_string())
                );
                let got = read(&got_path);
                check_lines(&format!("{case}/{file}"), &got);
                let level = match made {
                    "lir.txt" => "lowered",
                    "ir.txt" => "checked",
                    _ => "opt",
                };
                let header = format!(
                    "stave.ir 4 \"x86_64-linux-v2\" {} {level}\n",
                    if release { "release" } else { "debug" }
                );
                assert!(got.starts_with(&header), "{case}/{file}: {got}");
                if bless {
                    std::fs::write(&want_path, &got).unwrap();
                    continue;
                }
                match std::fs::read_to_string(&want_path) {
                    Ok(want) if want == got => {}
                    Ok(want) => failures.push(format!(
                        "{case}/{file}: differs from the golden\n--- want\n{want}--- got\n{got}"
                    )),
                    Err(e) => failures.push(format!("{}: {e}", want_path.display())),
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The `ir` dump does not depend on the thread count; release prints a subset of debug's fns,
/// each with the debug build's ops.
#[test]
fn ir_dump_properties() {
    for case in ir_cases() {
        let (d1, o1) = run_ir_case(&case, false, 1);
        if o1.has_errors() {
            continue;
        }
        let (d8, o8) = run_ir_case(&case, false, 8);
        let a = read(&d1.join("ir.txt"));
        assert_eq!(a, read(&d8.join("ir.txt")), "{case}: ir differs between 1 and 8 threads");
        assert_eq!(
            read(&d1.join("lir.txt")),
            read(&d8.join("lir.txt")),
            "{case}: lir differs between 1 and 8 threads"
        );
        assert_eq!(
            emit_text(&o1).unwrap(),
            emit_text(&o8).unwrap(),
            "{case}: the emit form differs between 1 and 8 threads"
        );
        // type lines dense from t0, before every fn; fns by instance id.
        let mut next_t = 0u32;
        let mut seen_fn = false;
        let mut last_inst: Option<Vec<u32>> = None;
        for l in a.lines() {
            if let Some(rest) = l.strip_prefix("type t") {
                assert!(!seen_fn, "{case}: type line after a fn: {l}");
                let t: u32 = rest.split(' ').next().unwrap().parse().unwrap();
                assert_eq!(t, next_t, "{case}: type ids not dense: {l}");
                next_t += 1;
            } else if let Some(rest) = l.strip_prefix("fn @") {
                seen_fn = true;
                let inst: Vec<u32> =
                    rest.split(' ').next().unwrap().split('.').map(|n| n.parse().unwrap()).collect();
                assert!(last_inst.as_ref().is_none_or(|p| *p < inst), "{case}: fns not by instance id: {l}");
                last_inst = Some(inst);
            }
        }
        let (dr, or) = run_ir_case(&case, true, 1);
        // Every fn the release build lowers has the ops of the debug build's, and so its checks.
        if let (Some(d), Some(r)) = (&o1.module, &or.module) {
            let differ = checked_ir::not_in(r, d);
            assert!(differ.is_empty(), "{case}: release differs from debug in {differ:?}");
            let (kd, kr) = (checked_ir::kept_checks(d), checked_ir::kept_checks(r));
            assert!(kr.iter().all(|(path, n)| kd.get(path) == Some(n)), "{case}: {kr:?} vs {kd:?}");
        }
        let rel = read(&dr.join("ir.txt"));
        let paths = |t: &str| -> std::collections::BTreeSet<String> {
            t.lines()
                .filter_map(|l| l.strip_prefix("fn @"))
                .map(|r| r.split(' ').nth(1).unwrap().split(['(', '<']).next().unwrap().to_string())
                .collect()
        };
        assert!(paths(&rel).is_subset(&paths(&a)), "{case}: release prints a fn debug does not");
    }
}

/// The emit form is the gate form plus the stack section: dropping the stack lines gives the
/// `ir` dump, byte for byte, and the stack lines stand together before the first fn.
#[test]
fn emit_form_is_the_gate_form_plus_the_stack_section() {
    let is_stack = |l: &str| {
        l == "truncated" || ["runstack ", "node ", "edge ", "nodes "].iter().any(|p| l.starts_with(p))
    };
    let mut with_stack = 0;
    for case in ir_cases() {
        let (dir, out) = run_ir_case(&case, false, 2);
        if out.has_errors() {
            assert!(emit_text(&out).is_err(), "{case}: an emit form for a build with errors");
            continue;
        }
        let emit = emit_text(&out).unwrap();
        let gate = read(&dir.join("ir.txt"));
        let filtered: String = emit.lines().filter(|l| !is_stack(l)).flat_map(|l| [l, "\n"]).collect();
        assert_eq!(filtered, gate, "{case}: emit minus the stack lines is not the ir dump");
        let at: Vec<usize> = emit.lines().enumerate().filter(|(_, l)| is_stack(l)).map(|(i, _)| i).collect();
        if let (Some(&first), Some(&last)) = (at.first(), at.last()) {
            with_stack += 1;
            assert_eq!(last - first + 1, at.len(), "{case}: the stack section is not contiguous");
            let after = emit.lines().nth(last + 1).unwrap_or("");
            assert!(after.starts_with("fn @"), "{case}: the stack section is not followed by a fn: {after}");
        }
    }
    assert!(with_stack > 10, "stack sections: {with_stack}");
}
