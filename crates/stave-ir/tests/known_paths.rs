//! Compiler-known items are found by identity ([`stave_ir::Known`], resolved once in the front
//! end), never by the text of their paths: no compiler source outside the one table spells a
//! known item's path or module as a string literal, in any of the forms a lookup would use.

use std::path::{Path, PathBuf};

use stave_ir::Known;

/// The table itself, and the files allowed to spell a path for output only.
const ALLOWED: &[(&str, &str)] = &[
    ("stave-ir/src/known.rs", "the table"),
    ("stave-obj/src/stub.rs", "a symbol name for images the object writer's tests build by hand"),
];

/// Every literal form a lookup of `k` could be written with.
fn spellings(k: Known) -> Vec<String> {
    let segs: Vec<&str> = k.module().split('.').collect();
    let mut out = vec![
        format!("\"{}\"", k.path()),
        format!("\"{}\"", k.module()),
        format!("\"{}.{}\"", k.module(), k.name()),
        segs.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", "),
    ];
    out.sort();
    out.dedup();
    out
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            // unit-test modules and generated tables build modules by hand or are data
            if name != "tests" && name != "generated" {
                sources(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// The text of `src` without comment lines and without its unit-test module (from
/// `#[cfg(test)] mod` to the end of the file, where unit tests sit).
fn code(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let end = (0..lines.len())
        .find(|&i| {
            lines[i].trim() == "#[cfg(test)]"
                && lines.get(i + 1).is_some_and(|l| l.trim_start().starts_with("mod "))
        })
        .unwrap_or(lines.len());
    lines[..end].iter().filter(|l| !l.trim_start().starts_with("//")).copied().collect::<Vec<_>>().join("\n")
}

#[test]
fn no_known_path_literal_outside_the_table() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates dir").to_path_buf();
    let mut files = Vec::new();
    for e in std::fs::read_dir(&crates).expect("read crates").flatten() {
        sources(&e.path().join("src"), &mut files);
    }
    files.sort();
    assert!(files.len() > 50, "found only {} source files under {}", files.len(), crates.display());
    let words: Vec<(Known, String)> =
        Known::ALL.iter().flat_map(|&k| spellings(k).into_iter().map(move |s| (k, s))).collect();
    let mut found = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&crates).unwrap_or(f).to_string_lossy().replace('\\', "/");
        if ALLOWED.iter().any(|(p, _)| *p == rel) {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(f) else { continue };
        let text = code(&src);
        for (k, w) in &words {
            if text.contains(w.as_str()) {
                found.push(format!("{rel}: {w} (`{k}`)"));
            }
        }
    }
    found.sort();
    found.dedup();
    assert!(
        found.is_empty(),
        "known items must be looked up through stave_ir::Known, not by path text:\n{}",
        found.join("\n")
    );
}

#[test]
fn the_scan_sees_a_lookup() {
    let src = "fn f(m: &Module) { m.find_func(\"runtime.panic.trap\"); item_at(p, \"std.thread\", n); }";
    let hits: Vec<String> = [Known::Trap, Known::ThreadJoin]
        .iter()
        .flat_map(|&k| spellings(k))
        .filter(|w| code(src).contains(w.as_str()))
        .collect();
    assert_eq!(hits, ["\"runtime.panic.trap\"", "\"std.thread\""]);
}

#[test]
fn entries_are_distinct() {
    let mut paths: Vec<String> = Known::ALL.iter().map(|k| k.path()).collect();
    let n = paths.len();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), n, "two entries name the same item");
}
