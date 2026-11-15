//! touch-fn on generated corpora: exactly one fn body changes, no signature line changes, the
//! edited file still parses, the choice is seeded and deterministic, and re-touching the same
//! body rewrites its touch statement in place.

#[path = "../../gen-corpus/tests/support/pda.rs"]
mod pda;

use gen_corpus::{Mix, Options, generate};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("touch-fn-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn corpus(dir: &Path, lines: u64, seed: u64) {
    let mix = Mix::default_mix().expect("mix");
    generate(&Options { lines, seed, out: dir.to_path_buf(), mix, threads: 0, force: true })
        .expect("generate");
}

fn snapshot(dir: &Path) -> BTreeMap<PathBuf, String> {
    touch_fn::stave_files(dir)
        .expect("files")
        .into_iter()
        .map(|f| {
            let t = std::fs::read_to_string(&f).expect("read");
            (f, t)
        })
        .collect()
}

fn signature_lines(src: &str) -> Vec<&str> {
    src.lines().filter(|l| l.contains("fn ") && l.trim_end().ends_with('{')).collect()
}

#[test]
fn touches_one_body_and_no_signature() {
    let d = TempDir::new("one");
    corpus(&d.0, 6_000, 4);
    for seed in 0..12u64 {
        let before = snapshot(&d.0);
        let t = touch_fn::touch(&d.0, seed).expect("touch");
        let after = snapshot(&d.0);
        let changed: Vec<&PathBuf> = before.keys().filter(|k| before[*k] != after[*k]).collect();
        assert_eq!(changed, vec![&t.file], "seed {seed}");
        let (a, b) = (&before[&t.file], &after[&t.file]);
        assert_eq!(signature_lines(a), signature_lines(b), "seed {seed}: signature changed");
        let line = b.lines().nth(t.line - 1).expect("reported line exists");
        assert!(line.contains("let touch_"), "seed {seed}: line {}: {line}", t.line);
        if t.in_place {
            assert_eq!(a.lines().count(), b.lines().count());
        } else {
            // Inserted: removing the reported line gives the original back.
            let mut ls: Vec<&str> = b.lines().collect();
            let removed = ls.remove(t.line - 1);
            if removed.trim_start().starts_with("let touch_") {
                assert_eq!(ls, a.lines().collect::<Vec<_>>(), "seed {seed}");
            }
        }
        if let Err(e) = pda::parse(b) {
            panic!("seed {seed}: {}: {e}\n{}", t.file.display(), pda::locate(b, &e));
        }
    }
}

#[test]
fn same_seed_same_body_and_in_place_second_time() {
    let a = TempDir::new("da");
    let b = TempDir::new("db");
    corpus(&a.0, 4_000, 9);
    corpus(&b.0, 4_000, 9);
    let ta = touch_fn::touch(&a.0, 17).expect("touch a");
    let tb = touch_fn::touch(&b.0, 17).expect("touch b");
    assert_eq!(ta.file.strip_prefix(&a.0).ok(), tb.file.strip_prefix(&b.0).ok());
    assert_eq!(ta.line, tb.line);
    let lines1 = std::fs::read_to_string(&ta.file).expect("read").lines().count();
    let again = touch_fn::touch(&a.0, 17).expect("touch again");
    assert_eq!((again.file.clone(), again.line), (ta.file.clone(), ta.line));
    assert!(again.in_place);
    assert_eq!(std::fs::read_to_string(&again.file).expect("read").lines().count(), lines1);
}

#[test]
fn seeds_spread_over_bodies() {
    let d = TempDir::new("spread");
    corpus(&d.0, 4_000, 2);
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..20u64 {
        let t = touch_fn::touch(&d.0, seed).expect("touch");
        seen.insert((t.file, t.line));
    }
    assert!(seen.len() >= 10, "only {} distinct bodies for 20 seeds", seen.len());
}

#[test]
fn empty_package_is_an_error() {
    let d = TempDir::new("empty");
    std::fs::create_dir_all(d.0.join("src")).expect("mkdir");
    std::fs::write(d.0.join("src/main.stv"), "import std.io;\n").expect("write");
    assert!(touch_fn::touch(&d.0, 0).is_err());
}
