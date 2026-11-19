//! A short run of every class: the generator is deterministic, and a handful of seeds per class
//! compile and run without a compiler crash, a hang, invalid generated Stave, a trap, or a
//! native run that differs from the interpreter. Verdict findings (`hole`, `precision`) are
//! left to the scheduled run.

use gen_corpus::classes::{Class, generate};
use stave_driver::exec::TempDir;
use std::process::Command;

/// Findings of compiler bugs that are not fixed yet: kind, texts the saved program's head
/// comment holds, the bug. An entry is removed with its fix.
const KNOWN: &[(&str, &[&str], &str)] = &[];

#[test]
fn same_seed_same_text() {
    for c in Class::ALL {
        let texts: Vec<String> = (0..16).map(|s| generate(c, s).text).collect();
        for (s, t) in texts.iter().enumerate() {
            assert_eq!(*t, generate(c, s as u64).text, "{} seed {s}", c.name());
        }
        let mut distinct = texts.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > 8, "{}: {} distinct programs of 16", c.name(), distinct.len());
    }
}

#[test]
fn a_few_seeds_per_class() {
    let tmp = TempDir::new("fuzz-classes-smoke").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_fuzz-classes"))
        .args(["--seed", "0", "--count", "3", "--jobs", "4", "--timeout", "60", "--out"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let summary = String::from_utf8_lossy(&out.stdout);
    let found = String::from_utf8_lossy(&out.stderr);
    assert!(summary.contains(" programs in "), "no summary:\n{summary}\n{found}");
    for kind in ["ice", "hang", "generator", "mismatch", "trap", "oracle"] {
        let dir = tmp.path().join("findings").join(kind);
        let files: Vec<String> = std::fs::read_dir(&dir)
            .map(|d| d.map(|e| e.unwrap().path().display().to_string()).collect())
            .unwrap_or_default();
        let files: Vec<String> = files
            .into_iter()
            .filter(|f| {
                let text = std::fs::read_to_string(f).unwrap_or_default();
                !KNOWN.iter().any(|(k, marks, _)| *k == kind && marks.iter().all(|m| text.contains(m)))
            })
            .collect();
        assert!(files.is_empty(), "{kind}: {files:?}\n{summary}\n{found}");
    }
}
