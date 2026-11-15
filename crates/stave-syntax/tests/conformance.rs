//! Runs the syntax conformance suite `tests/conformance/syntax/` at the
//! syntax layer: lex + parse every file and compare its diagnostics with its `//~` annotations
//! (contracts/conformance-annotations.md). This is a stand-in until `stave test` (stave-driver) runs
//! the suite end to end; it understands the annotation subset the syntax files use:
//! `//~ ERROR`, `//~ WARNING`, `//~^…` shifts, `//~ NOLOC …`, and ignores the run/build
//! directives (`CHECK-ONLY` and `PARSE-ONLY` included: this suite only lexes and parses). A package directory (with `stave.toml`) is one test over its `src/**.stv` files.
//!
//! Pass rule (negative and positive tests alike): the multiset of (severity, code, file, line)
//! of the diagnostics equals the annotated multiset. Every pretty-printed ok item must also
//! re-parse to the same AST (`print_file` has no fallbacks).

mod common;

use common::{repo, stave_files};
use stave_diag::{FileId, LineIndex, SourceFiles, canonical_order};
use stave_syntax::{parse_file, print_file};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Expect {
    severity: &'static str,
    code: String,
    file: String,
    /// `None` for NOLOC.
    line: Option<u32>,
}

/// The text after the `//` of the line's first comment (strings and chars skipped).
fn comment_body(line: &str) -> Option<&str> {
    #[derive(PartialEq)]
    enum St {
        Code,
        Str,
        Char,
    }
    let b = line.as_bytes();
    let mut st = St::Code;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match st {
            St::Code => match c {
                b'/' if b.get(i + 1) == Some(&b'/') => return line.get(i + 2..),
                b'"' => st = St::Str,
                b'\'' => st = St::Char,
                _ => {}
            },
            St::Str => match c {
                b'\\' => i += 1,
                b'"' => st = St::Code,
                _ => {}
            },
            St::Char => match c {
                b'\\' => i += 1,
                b'\'' => st = St::Code,
                _ => {}
            },
        }
        i += 1;
    }
    None
}

fn annotations(text: &str, file: &str, out: &mut Vec<Expect>) -> Result<(), String> {
    for (idx, raw) in text.split('\n').enumerate() {
        let line_no = idx as u32 + 1;
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let Some(body) = comment_body(line) else { continue };
        let Some(rest) = body.strip_prefix('~') else { continue };
        let carets = rest.bytes().take_while(|&b| b == b'^').count() as u32;
        let after = &rest[carets as usize..];
        let body =
            after.strip_prefix(' ').ok_or(format!("{file}:{line_no}: malformed annotation"))?.trim_end();
        let (kw, args) = body.split_once(' ').map_or((body, ""), |(k, a)| (k, a.trim()));
        let code_ok = |c: &str| {
            if stave_diag::entry(c).is_some() {
                Ok(c.to_string())
            } else {
                Err(format!("{file}:{line_no}: unknown code {c}"))
            }
        };
        match kw {
            "ERROR" | "WARNING" => {
                let line = line_no.checked_sub(carets).filter(|&l| l >= 1).ok_or("shift above line 1")?;
                let severity = if kw == "ERROR" { "error" } else { "warning" };
                out.push(Expect { severity, code: code_ok(args)?, file: file.into(), line: Some(line) });
            }
            "NOLOC" => {
                let (sev, code) = args.split_once(' ').ok_or(format!("{file}:{line_no}: NOLOC"))?;
                let severity = if sev == "ERROR" { "error" } else { "warning" };
                out.push(Expect { severity, code: code_ok(code.trim())?, file: file.into(), line: None });
            }
            "STDOUT" | "STDERR-CONTAINS" | "EXIT" | "TARGET" | "ARGS" | "CHECK-ONLY" | "PARSE-ONLY"
            | "RESOURCE-LIMIT" => {}
            other => return Err(format!("{file}:{line_no}: unknown annotation {other}")),
        }
    }
    Ok(())
}

/// One test: a single file, or the `.stv` files of a package. Returns an error description.
fn run_test(root: &Path, files: &[PathBuf]) -> Result<bool, String> {
    let mut want = Vec::new();
    let mut got = Vec::new();
    let mut table = SourceFiles::new();
    let mut all = Vec::new();
    for (k, path) in files.iter().enumerate() {
        let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().into_owned();
        let bytes = std::fs::read(path).map_err(|e| format!("{rel}: {e}"))?;
        annotations(&String::from_utf8_lossy(&bytes), &rel, &mut want)?;
        let pf = parse_file(&bytes, FileId(k as u32));
        let li = LineIndex::new(&bytes);
        for d in &pf.diagnostics {
            let line = d.primary().and_then(|s| li.line_col(s.offset)).map(|(l, _)| l);
            got.push(Expect {
                severity: d.severity().as_str(),
                code: d.code().to_string(),
                file: rel.clone(),
                line,
            });
        }
        let printed = print_file(&pf, &bytes);
        if pf.stop.is_none() && printed.fallbacks != 0 {
            return Err(format!("{rel}: printer fell back on {} item(s)", printed.fallbacks));
        }
        if pf.stop.is_none() {
            let again = parse_file(printed.text.as_bytes(), FileId(k as u32));
            let same = again.items.len() == pf.items.len()
                && again.items.iter().zip(&pf.items).all(|(a, b)| a.ok == b.ok)
                && again.ast == pf.ast;
            if !same {
                return Err(format!("{rel}: printed text parses differently:\n{}", printed.text));
            }
        }
        table.add("syntax", &rel, &bytes).map_err(|e| e.to_string())?;
        all.extend(pf.diagnostics);
    }
    // canonical order across files: by path, then offset
    let ordered = canonical_order(all, &table);
    let paths: Vec<&str> = ordered
        .iter()
        .filter_map(|d| d.primary())
        .filter_map(|s| table.get(s.file))
        .map(|f| f.path.as_str())
        .collect();
    let mut sorted = paths.clone();
    sorted.sort();
    if paths != sorted {
        return Err(format!("diagnostics not in canonical file order: {paths:?}"));
    }
    want.sort();
    got.sort();
    if want != got {
        return Err(format!("expected {want:#?}\ngot {got:#?}"));
    }
    Ok(want.iter().any(|e| e.severity == "error"))
}

#[test]
fn syntax_conformance_suite() {
    let root = repo().join("tests/conformance/syntax");
    let files = stave_files(&root);
    let is_pkg =
        |p: &Path| p.ancestors().skip(1).take_while(|a| *a != root).any(|a| a.join("stave.toml").exists());
    let mut tests: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for f in &files {
        if !is_pkg(f) {
            tests.push((f.strip_prefix(&root).unwrap().display().to_string(), vec![f.clone()]));
        }
    }
    let mut pkgs: Vec<PathBuf> = files
        .iter()
        .filter(|f| is_pkg(f))
        .filter_map(|f| f.ancestors().find(|a| a.join("stave.toml").exists()).map(Path::to_path_buf))
        .collect();
    pkgs.dedup();
    for p in pkgs {
        let mut fs: Vec<PathBuf> = files.iter().filter(|f| f.starts_with(&p)).cloned().collect();
        fs.sort();
        tests.push((p.strip_prefix(&root).unwrap().display().to_string(), fs));
    }
    let (mut neg, mut pos, mut failures) = (0, 0, Vec::new());
    for (name, fs) in &tests {
        match run_test(&root, fs) {
            Ok(true) => neg += 1,
            Ok(false) => pos += 1,
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    eprintln!("syntax conformance: {neg} negative, {pos} positive, {} failed", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(neg + pos >= 50, "suite too small: {}", neg + pos);
}
