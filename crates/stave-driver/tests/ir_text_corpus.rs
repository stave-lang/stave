//! The IR text of real programs: every program of `tests/conformance/run` (every fifth, or all
//! with `STAVE_IR_TEXT_ALL=1`), in debug and release, at the levels `lowered`, `checked`, and
//! `opt`, and in the emit form, obeys the structure rules of contracts/checked-ir.md "Loading"
//! that need no verifier: section order, dense ids, ids that name a line, the canonical spelling
//! of paths and strings, the order of fns, blocks and values. The gate form is the emit form
//! without its stack lines, and `known` names the fn's own path.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use stave_driver::dump::{DumpStage, emit_text, render_to_vec};
use stave_driver::pipeline::{self, Goal, Options, Output};
use stave_driver::shipped::Library;

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn programs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(repo().join("tests/conformance/run"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "stv"))
        .collect();
    v.sort();
    if std::env::var_os("STAVE_IR_TEXT_ALL").is_some() {
        return v;
    }
    // Every fifth program, plus the ones a line kind depends on: a new file in the directory
    // shifts the stride, and the kinds asserted below must not depend on where it lands.
    let named = |p: &PathBuf| ALWAYS.iter().any(|n| p.file_name().is_some_and(|f| f == *n));
    v.into_iter().enumerate().filter(|(i, p)| i % 5 == 0 || named(p)).map(|(_, p)| p).collect()
}

/// Programs in every sample: `static_str.stv` has the data blobs with address words (`ref`).
const ALWAYS: [&str; 1] = ["static_str.stv"];

fn lowered(root: &Path, release: bool) -> Output {
    let mut opts = Options::new(Goal::Check).threads(Some(2));
    opts.lower = true;
    opts.keep_lowered = true;
    opts.release = release;
    opts.target = Some(stave_target::Triple::parse("x86_64-linux").unwrap());
    pipeline::run(root, &Library::in_dir(repo()), &opts).unwrap()
}

fn is_word(p: &str) -> bool {
    let b = p.as_bytes();
    b.first().is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        && b.iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.'))
}

/// An instance id as a sort key: numbers, `u` last.
fn inst_key(s: &str) -> Vec<u64> {
    s.split('.').map(|n| if n == "u" { u64::MAX } else { n.parse().unwrap() }).collect()
}

/// Splits a line into tokens the way the grammar does for the shapes the printer makes: strings
/// are one token.
fn tokens(l: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = l.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b' ' => i += 1,
            b'"' => {
                let mut j = i + 1;
                while b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                out.push(l[i..=j].to_string());
                i = j + 1;
            }
            c if c.is_ascii_alphanumeric() || matches!(c, b'%' | b'@' | b'_' | b'-' | b'.') => {
                let mut j = i + 1;
                while j < b.len() && (b[j].is_ascii_alphanumeric() || matches!(b[j], b'_' | b'.')) {
                    j += 1;
                }
                out.push(l[i..j].to_string());
                i = j;
            }
            _ => {
                if l[i..].starts_with("->") || l[i..].starts_with("<=") {
                    out.push(l[i..i + 2].to_string());
                    i += 2;
                } else {
                    out.push(l[i..=i].to_string());
                    i += 1;
                }
            }
        }
    }
    out
}

/// Checks the structure rules on one text; returns the count per line kind.
fn validate(text: &str, level: &str, release: bool, what: &str) -> BTreeMap<String, usize> {
    let fail = |n: usize, l: &str, msg: &str| -> ! { panic!("{what}:{}: {msg}: {l}", n + 1) };
    let lines: Vec<&str> = text.lines().collect();
    assert!(text.ends_with('\n') && text.is_ascii(), "{what}: not ASCII lines");
    let header =
        format!("stave.ir 4 \"x86_64-linux-v2\" {} {level}", if release { "release" } else { "debug" });
    assert_eq!(lines[0], header, "{what}");
    let rank = |kind: &str| -> u32 {
        match kind {
            "decl" => 1,
            "type" => 2,
            "shape" => 3,
            "str" => 4,
            "site" => 5,
            "static" | "data" | "ref" => 6,
            "main" => 7,
            "cell" => 8,
            "runstack" => 9,
            "truncated" => 10,
            "node" => 11,
            "edge" => 12,
            "nodes" => 13,
            _ => 14,
        }
    };
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let (mut ntypes, mut nstr, mut nsites, mut nstatics) = (0usize, 0usize, 0usize, 0usize);
    let mut last_rank = 0;
    let mut decls: BTreeMap<u32, String> = BTreeMap::new();
    let mut adt_decl: BTreeMap<u32, u32> = BTreeMap::new();
    let mut shapes: BTreeMap<u32, String> = BTreeMap::new();
    let mut fns: Vec<(Vec<u64>, String, bool)> = Vec::new();
    let mut insts_named: BTreeSet<String> = BTreeSet::new();
    let mut deferred: Vec<(usize, &str)> = Vec::new();
    let mut last_data: Option<u64> = None;
    let mut last_ref: Option<u64> = None;
    let (mut in_body, mut cur_fn_has_body, mut cur_fn_inst) = (false, false, String::new());
    let mut seen_term = true;
    let mut next_block = 0usize;
    let mut next_val = 0u32;
    for (n, l) in lines.iter().enumerate().skip(1) {
        if l.contains('?') && !l.contains('"') {
            fail(n, l, "a placeholder token");
        }
        let toks = tokens(l);
        let kind = toks[0].as_str();
        if l.starts_with("  ") {
            if !in_body {
                fail(n, l, "an instruction outside a body");
            }
            if seen_term {
                fail(n, l, "an instruction after the terminator");
            }
            let is_term = matches!(kind, "br" | "cbr" | "switch" | "ret" | "unreachable");
            if is_term {
                seen_term = true;
                *counts.entry(kind.to_string()).or_default() += 1;
            } else {
                *counts.entry("inst".into()).or_default() += 1;
            }
            for (k, t) in toks.iter().enumerate() {
                if let Some(v) = t.strip_prefix('%') {
                    let v: u32 = v.parse().unwrap();
                    if !is_term && k == 0 {
                        if v != next_val {
                            fail(n, l, "a value out of print order");
                        }
                        next_val += 1;
                    }
                }
                if let Some(i) = t.strip_prefix('@') {
                    insts_named.insert(i.to_string());
                }
                if t == "!" && toks[k + 1].parse::<usize>().unwrap() >= nsites {
                    fail(n, l, "a site row out of range");
                }
                if t.starts_with('t') && t[1..].chars().all(|c| c.is_ascii_digit()) && t.len() > 1 {
                    deferred.push((n, l));
                }
            }
            continue;
        }
        if kind.starts_with('b') && l.ends_with(':') && in_body {
            if !seen_term {
                fail(n, l, "a block before the terminator");
            }
            if toks[0] != format!("b{next_block}") {
                fail(n, l, "blocks not dense in order");
            }
            next_block += 1;
            seen_term = false;
            *counts.entry("block".into()).or_default() += 1;
            // params: `%v : tT` triples after `(`
            for (k, t) in toks.iter().enumerate() {
                if let Some(v) = t.strip_prefix('%') {
                    let v: u32 = v.parse().unwrap();
                    if toks.get(k + 1).map(String::as_str) != Some(":") {
                        fail(n, l, "a block param without a type");
                    }
                    if v != next_val {
                        fail(n, l, "a value out of print order");
                    }
                    next_val += 1;
                }
            }
            continue;
        }
        if *l == "}" {
            if !in_body || !seen_term {
                fail(n, l, "a stray }");
            }
            in_body = false;
            continue;
        }
        if in_body {
            fail(n, l, "a section line inside a body");
        }
        let r = rank(kind);
        if r < last_rank && r != 14 {
            fail(n, l, "sections out of order");
        }
        if r == 14 && kind != "fn" && kind != "claim" {
            fail(n, l, "an unknown line");
        }
        if r != 14 {
            last_rank = r;
        }
        *counts.entry(kind.to_string()).or_default() += 1;
        match kind {
            "decl" => {
                let d: u32 = toks[1][1..].parse().unwrap();
                if decls.keys().next_back().is_some_and(|&p| p >= d) {
                    fail(n, l, "decls not ascending");
                }
                decls.insert(d, toks[2].clone());
                let quoted = toks[3].starts_with('"');
                if quoted == is_word(toks[3].trim_matches('"')) {
                    fail(n, l, "path spelling");
                }
            }
            "type" => {
                if toks[1] != format!("t{ntypes}") {
                    fail(n, l, "types not dense");
                }
                if matches!(toks[2].as_str(), "param" | "constparam" | "never") {
                    fail(n, l, "a key the IR text has none of");
                }
                if toks[2] == "adt" {
                    adt_decl.insert(ntypes as u32, toks[3][1..].parse().unwrap());
                }
                ntypes += 1;
            }
            "shape" => {
                let t: u32 = toks[1][1..].parse().unwrap();
                if (t as usize) >= ntypes || shapes.keys().next_back().is_some_and(|&p| p >= t) {
                    fail(n, l, "shape of an unknown or repeated type");
                }
                shapes.insert(t, toks[2].clone());
            }
            "str" => {
                if toks[1] != format!("s{nstr}") {
                    fail(n, l, "strings not dense");
                }
                nstr += 1;
            }
            "site" => {
                if toks[1] != nsites.to_string() {
                    fail(n, l, "site rows not dense");
                }
                nsites += 1;
            }
            "static" => {
                if toks[1] != nstatics.to_string() {
                    fail(n, l, "statics not dense");
                }
                nstatics += 1;
                last_data = None;
                last_ref = None;
            }
            "data" => {
                let (i, off): (usize, u64) = (toks[1].parse().unwrap(), toks[2].parse().unwrap());
                let bytes = &toks[3][1..];
                if i + 1 != nstatics || !bytes.len().is_multiple_of(2) || bytes.len() > 128 || bytes.len() < 2
                {
                    fail(n, l, "data of another static, or not 1..=64 bytes");
                }
                if last_ref.is_some() || last_data.is_some_and(|o| o / 64 >= off / 64) {
                    fail(n, l, "data after a ref, or two data lines in one block");
                }
                if off / 64 != (off + (bytes.len() as u64 / 2) - 1) / 64 {
                    fail(n, l, "data across a 64-byte block");
                }
                if bytes.starts_with("00") || bytes.ends_with("00") {
                    fail(n, l, "data not trimmed to its nonzero bytes");
                }
                last_data = Some(off);
            }
            "ref" => {
                let (i, off): (usize, u64) = (toks[1].parse().unwrap(), toks[2].parse().unwrap());
                if i + 1 != nstatics || off % 8 != 0 || last_ref.is_some_and(|o| o >= off) {
                    fail(n, l, "a ref of another static, unaligned, or out of order");
                }
                last_ref = Some(off);
            }
            "fn" => {
                let inst = toks[1][1..].to_string();
                let key = inst_key(&inst);
                if fns.last().is_some_and(|(k, _, _)| *k >= key) {
                    fail(n, l, "fns not in instance order");
                }
                let has_body = l.ends_with(" {");
                let path_tok = &toks[2];
                if !path_tok.starts_with('"') && !is_word(path_tok) {
                    fail(n, l, "a bare path that is not a Word");
                }
                if path_tok.starts_with('"') && is_word(&path_tok[1..path_tok.len() - 1]) {
                    fail(n, l, "a quoted Word path");
                }
                if toks.contains(&"known".to_string()) {
                    let path = path_tok.trim_matches('"');
                    let known: BTreeSet<String> = stave_ir::Known::ALL.iter().map(|k| k.path()).collect();
                    if !known.contains(path) {
                        fail(n, l, "known: no known item has this path");
                    }
                }
                fns.push((key, inst.clone(), has_body));
                (in_body, cur_fn_has_body, cur_fn_inst) = (has_body, has_body, inst);
                seen_term = true;
                next_block = 0;
                next_val = 0;
            }
            "claim" if !cur_fn_has_body || toks[1] != format!("@{cur_fn_inst}") || level == "lowered" => {
                fail(n, l, "a claim not after the body of its fn, or at level lowered");
            }
            _ => {}
        }
    }
    // Ids that name a line.
    let known_insts: BTreeSet<&str> = fns.iter().map(|(_, i, _)| i.as_str()).collect();
    for i in &insts_named {
        assert!(known_insts.contains(i.as_str()), "{what}: @{i} names no fn");
    }
    for (n, l) in deferred {
        for t in tokens(l) {
            if let Some(id) = t.strip_prefix('t')
                && let Ok(id) = id.parse::<usize>()
            {
                assert!(id < ntypes, "{what}:{}: t{id} out of range", n + 1);
            }
        }
    }
    for l in &lines {
        for t in tokens(l) {
            if let Some(id) = t.strip_prefix('d')
                && let Ok(id) = id.parse::<u32>()
                && !l.starts_with("decl")
                && (l.starts_with("type") || l.starts_with("cell"))
            {
                assert!(decls.contains_key(&id), "{what}: d{id} has no decl line");
            }
        }
    }
    for (t, d) in &adt_decl {
        if let Some(shape) = shapes.get(t) {
            assert_eq!(shape, &decls[d], "{what}: shape kind of t{t} differs from its decl's");
        }
    }
    counts
}

fn merge(into: &mut BTreeMap<String, usize>, from: BTreeMap<String, usize>) {
    for (k, v) in from {
        *into.entry(k).or_default() += v;
    }
}

#[test]
fn the_text_of_real_programs_obeys_the_structure_rules() {
    let mut all: BTreeMap<String, usize> = BTreeMap::new();
    let mut flags: BTreeMap<&str, usize> = BTreeMap::new();
    let progs = programs();
    assert!(progs.len() >= 20, "{}", progs.len());
    for p in &progs {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        for release in [false, true] {
            let out = lowered(p, release);
            if out.has_errors() || out.module.is_none() {
                continue;
            }
            let mode = if release { "release" } else { "debug" };
            let ir = String::from_utf8(render_to_vec(DumpStage::Ir, &out).unwrap()).unwrap();
            let lir = String::from_utf8(render_to_vec(DumpStage::Lir, &out).unwrap()).unwrap();
            merge(&mut all, validate(&lir, "lowered", release, &format!("{name} {mode} lir")));
            if release {
                let opt = String::from_utf8(render_to_vec(DumpStage::Opt, &out).unwrap()).unwrap();
                merge(&mut all, validate(&opt, "opt", true, &format!("{name} opt")));
            }
            let emit = emit_text(&out).unwrap();
            merge(&mut all, validate(&emit, "checked", release, &format!("{name} {mode} emit")));
            let is_stack = |l: &str| {
                l == "truncated" || ["runstack ", "node ", "edge ", "nodes "].iter().any(|p| l.starts_with(p))
            };
            let gate: String = emit.lines().filter(|l| !is_stack(l)).flat_map(|l| [l, "\n"]).collect();
            assert_eq!(gate, ir, "{name} {mode}: emit minus the stack lines is not the ir dump");
            for l in ir.lines().filter(|l| l.starts_with("fn ")) {
                for (flag, key) in [
                    (" known", "known"),
                    (" altstack", "altstack"),
                    (" symbol ", "symbol"),
                    (" variadic", "variadic"),
                    (" unbuilt", "unbuilt"),
                    (" needs(", "needs"),
                ] {
                    if l.contains(flag) {
                        *flags.entry(key).or_default() += 1;
                    }
                }
            }
        }
    }
    println!("line kinds over {} programs: {all:?}", progs.len());
    println!("fn attributes: {flags:?}");
    for kind in ["decl", "shape", "str", "static", "data", "ref", "main", "claim", "node", "edge", "nodes"] {
        assert!(all.get(kind).copied().unwrap_or(0) > 0, "no {kind} line in the sample: {all:?}");
    }
    for key in ["known", "altstack"] {
        assert!(flags.get(key).copied().unwrap_or(0) > 0, "no {key} fn in the sample: {flags:?}");
    }
}
