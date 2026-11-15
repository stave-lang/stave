//! gen-corpus output checks: every file parses with a minimal table-driven PDA over the
//! generated `stave-syntax` LL(1) tables; output is byte-identical per seed; line counts are
//! exact; the module tree, imports and module-alias uses are consistent; the construct mix
//! covers the constructs the generator promises.

#[path = "support/pda.rs"]
mod pda;

use gen_corpus::{Mix, Options, Summary, generate};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("gen-corpus-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gen_into(dir: &Path, lines: u64, seed: u64, threads: usize) -> Summary {
    let mix = Mix::default_mix().expect("default mix");
    generate(&Options { lines, seed, out: dir.to_path_buf(), mix, threads, force: true }).expect("generate")
}

/// All files under `dir`, relative path → bytes.
fn read_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, d: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let mut es: Vec<_> =
            std::fs::read_dir(d).expect("read_dir").map(|e| e.expect("entry").path()).collect();
        es.sort();
        for p in es {
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                let rel = p.strip_prefix(root).expect("prefix").to_string_lossy().replace('\\', "/");
                out.insert(rel, std::fs::read(&p).expect("read"));
            }
        }
    }
    let mut m = BTreeMap::new();
    walk(dir, dir, &mut m);
    m
}

/// Module path of `src/a/b.stv` → `a.b`.
fn module_of(rel: &str) -> Option<String> {
    let s = rel.strip_prefix("src/")?.strip_suffix(".stv")?;
    Some(s.replace('/', "."))
}

/// `import a.b as mK;` lines → (path, alias).
fn imports(src: &str) -> Vec<(String, Option<String>)> {
    src.lines()
        .filter_map(|l| l.strip_prefix("import ")?.strip_suffix(';').map(str::to_string))
        .map(|l| match l.split_once(" as ") {
            Some((p, a)) => (p.to_string(), Some(a.to_string())),
            None => (l, None),
        })
        .collect()
}

/// Identifiers of the form `m<digits>` directly followed by `.` outside strings and comments.
fn alias_uses(src: &str) -> BTreeSet<String> {
    let b = src.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += if b[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
        } else if pda::is_idstart(c) {
            let s = i;
            while i < b.len() && pda::is_idcont(b[i]) {
                i += 1;
            }
            let w = &src[s..i];
            let prev_ok = s == 0 || b[s - 1] != b'.';
            if prev_ok
                && w.len() > 1
                && w.starts_with('m')
                && w[1..].bytes().all(|x| x.is_ascii_digit())
                && b.get(i) == Some(&b'.')
            {
                out.insert(w.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn check_corpus(dir: &Path, lines: u64, s: &Summary) -> BTreeMap<String, Vec<u8>> {
    let tree = read_tree(dir);
    assert!(tree.contains_key("stave.toml"), "stave.toml missing");
    assert!(tree.contains_key("src/main.stv"), "src/main.stv missing");
    let mut total = 0u64;
    let modules: BTreeSet<String> = tree.keys().filter_map(|k| module_of(k)).collect();
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (rel, bytes) in &tree {
        let Some(module) = module_of(rel) else { continue };
        let src = std::str::from_utf8(bytes).unwrap_or_else(|_| panic!("{rel}: not UTF-8"));
        assert!(src.ends_with('\n'), "{rel}: no final newline");
        total += src.bytes().filter(|&b| b == b'\n').count() as u64;
        if let Err(e) = pda::parse(src) {
            panic!("{rel}: {e}\n{}", pda::locate(src, &e));
        }
        assert!(module.split('.').count() <= 3, "{rel}: depth > 3");
        let ims = imports(src);
        let mut aliases = BTreeSet::new();
        for (p, a) in &ims {
            if p == "std.io" {
                continue;
            }
            assert!(modules.contains(p), "{rel}: import of missing module {p}");
            let a = a.clone().unwrap_or_else(|| panic!("{rel}: import {p} without alias"));
            assert!(aliases.insert(a), "{rel}: duplicate alias");
            edges.entry(module.clone()).or_default().push(p.clone());
        }
        for u in alias_uses(src) {
            assert!(aliases.contains(&u), "{rel}: alias {u} used but not imported");
        }
        let uses_io = src.contains("io.print(");
        assert!(!uses_io || ims.iter().any(|(p, _)| p == "std.io"), "{rel}: prints without import std.io");
    }
    // Acyclic imports.
    fn visit(m: &str, edges: &BTreeMap<String, Vec<String>>, state: &mut BTreeMap<String, u8>) {
        match state.get(m) {
            Some(1) => panic!("import cycle through {m}"),
            Some(2) => return,
            _ => {}
        }
        state.insert(m.to_string(), 1);
        for n in edges.get(m).into_iter().flatten() {
            visit(n, edges, state);
        }
        state.insert(m.to_string(), 2);
    }
    let mut state = BTreeMap::new();
    for m in &modules {
        visit(m, &edges, &mut state);
    }
    assert_eq!(total, s.lines, "summary line count");
    if lines >= 1000 {
        assert_eq!(total, lines, "line count");
    } else {
        let tol = (lines / 100).max(10);
        assert!(total.abs_diff(lines) <= tol, "{total} lines for --lines {lines}");
    }
    tree
}

#[test]
fn corpora_parse_and_have_exact_line_counts() {
    for &(lines, seed) in &[(40u64, 3u64), (500, 9), (3_000, 1), (12_000, 2), (20_000, 7), (25_000, 42)] {
        let d = TempDir::new(&format!("p{lines}-{seed}"));
        let s = gen_into(&d.0, lines, seed, 0);
        check_corpus(&d.0, lines, &s);
        if lines >= 10_000 {
            assert!(s.modules >= 8, "{lines}: {} modules", s.modules);
        }
    }
}

#[test]
fn output_is_deterministic_across_runs_and_threads() {
    for &seed in &[1u64, 5, 77] {
        let a = TempDir::new(&format!("da{seed}"));
        let b = TempDir::new(&format!("db{seed}"));
        let sa = gen_into(&a.0, 15_000, seed, 1);
        let sb = gen_into(&b.0, 15_000, seed, 7);
        assert_eq!(sa, sb);
        let (ta, tb) = (read_tree(&a.0), read_tree(&b.0));
        assert_eq!(ta.keys().collect::<Vec<_>>(), tb.keys().collect::<Vec<_>>());
        for (k, v) in &ta {
            assert!(tb.get(k) == Some(v), "seed {seed}: {k} differs");
        }
    }
    let a = TempDir::new("dx1");
    let b = TempDir::new("dx2");
    gen_into(&a.0, 15_000, 1, 0);
    gen_into(&b.0, 15_000, 2, 0);
    assert_ne!(read_tree(&a.0), read_tree(&b.0), "different seeds gave the same corpus");
}

#[test]
fn construct_mix_is_covered() {
    let d = TempDir::new("mix");
    let s = gen_into(&d.0, 60_000, 11, 0);
    let tree = check_corpus(&d.0, 60_000, &s);
    let all: String = tree
        .iter()
        .filter(|(k, _)| k.ends_with(".stv"))
        .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    let need = [
        "pub struct ",
        "pub enum ",
        "impl<T> ",
        "pub trait ",
        " for ",
        "<T: ",
        "<const N: u32>",
        "::<",
        "pub type ",
        "pub const ",
        "match ",
        " then ",
        " else ",
        "while ",
        "for k",
        "loop {",
        "break;",
        "continue;",
        "let mut ",
        "+% ",
        "+| ",
        " as% ",
        " as| ",
        " as~ ",
        " as ",
        "f\"",
        "io.print(",
        "?;",
        "Option.Some(",
        "Option.None",
        "Result.Ok(",
        "Result.Err(",
        "&mut ",
        "from p0",
        "from p0 + p1",
        "from self",
        ".{ ",
        "(&self",
        "(&mut self",
        "(self)",
        "-> Self",
        "&Self",
        "return ",
        "//",
        "usize]",
        "..=",
    ];
    for n in need {
        assert!(all.contains(n), "construct {n:?} never generated");
    }
    // Proportions: items 60/20/10/10, generics in 15% of items, fn body median 12,
    // about 8 modules per 10k lines.
    let (mut items, mut fns, mut adts, mut impls, mut generic) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut bodies = Vec::new();
    for v in tree.iter().filter(|(k, _)| k.ends_with(".stv")).map(|(_, v)| String::from_utf8_lossy(v)) {
        let ls: Vec<&str> = v.lines().collect();
        for (i, l) in ls.iter().enumerate() {
            let top = !l.is_empty() && !l.starts_with([' ', '/', '}']) && !l.starts_with("import ");
            if !top || l.starts_with("pub const PAD_") {
                continue;
            }
            items += 1;
            let fn_line = l.starts_with("fn ") || l.starts_with("pub fn ");
            if fn_line {
                fns += 1;
                let end = ls[i..].iter().position(|x| *x == "}").unwrap_or(0);
                bodies.push(end.saturating_sub(1));
            } else if l.starts_with("pub struct ") || l.starts_with("pub enum ") {
                adts += 1;
            } else if l.starts_with("impl") {
                impls += 1;
            }
            let name_end = l.find('(').unwrap_or(l.len());
            if (fn_line && l[..name_end].contains('<')) || l.starts_with("impl<") || l.contains("<T> {") {
                generic += 1;
            }
        }
    }
    let pct = |x: u32| f64::from(x) * 100.0 / f64::from(items);
    assert!((55.0..65.0).contains(&pct(fns)), "fns {:.1}%", pct(fns));
    assert!((16.0..24.0).contains(&pct(adts)), "structs+enums {:.1}%", pct(adts));
    assert!((7.0..14.0).contains(&pct(impls)), "impls {:.1}%", pct(impls));
    assert!((12.0..18.0).contains(&pct(generic)), "generic items {:.1}%", pct(generic));
    bodies.sort_unstable();
    let median = bodies[bodies.len() / 2];
    assert!((10..=14).contains(&median), "fn body median {median}");
    assert!(s.fns * 2 > s.items, "fns {} of {} items", s.fns, s.items);
    assert!((40..=56).contains(&s.modules), "{} modules for 60k lines", s.modules);
}

#[test]
fn refuses_non_empty_output_without_force() {
    let d = TempDir::new("force");
    std::fs::create_dir_all(&d.0).expect("mkdir");
    std::fs::write(d.0.join("keep.txt"), "x").expect("write");
    let mix = Mix::default_mix().expect("mix");
    let r = generate(&Options { lines: 1000, seed: 1, out: d.0.clone(), mix, threads: 1, force: false });
    assert!(r.is_err());
}

#[test]
fn mix_rejects_unimplemented_tiers_and_unknown_keys() {
    let base = gen_corpus::mix::DEFAULT_MIX;
    assert!(Mix::parse(&base.replace("m3 = false", "m3 = true")).is_err());
    assert!(Mix::parse(&base.replace("m6 = false", "m6 = true")).is_err());
    assert!(Mix::parse(&format!("{base}\n[extra]\nx = 1\n")).is_err());
    let no_m2 = Mix::parse(&base.replace("m2 = true", "m2 = false")).expect("m2 off");
    assert_eq!(no_m2.fns.ref_field + no_m2.fns.struct_ref + no_m2.stmt.ref_block + no_m2.stmt.closure, 0);
}

#[test]
fn m1_only_mix_has_no_from_clauses_in_free_fns() {
    let d = TempDir::new("m1");
    let mix = Mix::parse(&gen_corpus::mix::DEFAULT_MIX.replace("m2 = true", "m2 = false")).expect("mix");
    let s = generate(&Options { lines: 8_000, seed: 3, out: d.0.clone(), mix, threads: 0, force: true })
        .expect("generate");
    let tree = check_corpus(&d.0, 8_000, &s);
    for (k, v) in &tree {
        let t = String::from_utf8_lossy(v);
        assert!(!t.contains("from p0"), "{k}: ref return in m1-only corpus");
    }
}
