//! `--pair` output: byte-identical per seed, the Stave and C texts define the same fns, and the
//! Stave text passes the stage0 parser. No C compiler is needed.

use gen_corpus::{PairOptions, PairTexts, pair_texts};
use stave_diag::FileId;

fn opts(lines: u64, seed: u64) -> PairOptions {
    PairOptions { lines, seed, units: 0, reps: 3 }
}

/// Names of the fns a C text defines: unindented lines `[static ]<type> <name>(...) {`.
fn c_defs(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.ends_with(") {") && !l.starts_with(' ') && !l.starts_with("static inline"))
        .filter_map(|l| {
            let head = l.strip_prefix("static ").unwrap_or(l);
            let (_, rest) = head.split_once(' ')?;
            Some(rest.split_once('(')?.0.to_string())
        })
        .collect()
}

/// Names of the fns a Stave text defines: unindented `[pub ]fn <name>(`.
fn stave_defs(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| l.strip_prefix("pub fn ").or_else(|| l.strip_prefix("fn ")))
        .filter_map(|l| Some(l.split_once('(')?.0.to_string()))
        .collect()
}

fn all_stave_defs(t: &PairTexts) -> Vec<String> {
    let mut v: Vec<String> = t.stave.iter().flat_map(|(_, s)| stave_defs(s)).collect();
    v.sort();
    v
}

#[test]
fn c_output_is_deterministic() {
    let a = pair_texts(&opts(6_000, 11));
    let b = pair_texts(&opts(6_000, 11));
    assert_eq!(a.c_single, b.c_single);
    assert_eq!(a.c_split, b.c_split);
    assert_eq!(a.stave, b.stave);
    let c = pair_texts(&opts(6_000, 12));
    assert_ne!(a.c_single, c.c_single, "another seed gives another program");
}

#[test]
fn both_languages_define_the_same_fns() {
    for &(lines, seed) in &[(3_000u64, 1u64), (20_000, 7)] {
        let t = pair_texts(&opts(lines, seed));
        let stave = all_stave_defs(&t);
        let mut single = c_defs(&t.c_single);
        single.sort();
        let mut split: Vec<String> =
            t.c_split.iter().filter(|(p, _)| p.ends_with(".c")).flat_map(|(_, s)| c_defs(s)).collect();
        split.sort();
        // fns, one driver per unit, main
        let want = t.fns as usize + t.units as usize + 1;
        assert_eq!(stave.len(), want, "Stave fn count");
        assert_eq!(single, stave, "single-file C defines the Stave fns");
        assert_eq!(split, stave, "split C defines the Stave fns");
        let listed: Vec<&str> = t
            .c_split
            .iter()
            .find(|(p, _)| p == "files.txt")
            .map(|(_, s)| s.lines().collect())
            .unwrap_or_default();
        assert_eq!(listed.len(), t.units as usize + 1, "files.txt lists every unit and main.c");
        let stave_lines: usize =
            t.stave.iter().filter(|(p, _)| p.ends_with(".stv")).map(|(_, s)| s.lines().count()).sum();
        assert!(
            stave_lines as u64 >= lines * 9 / 10 && (stave_lines as u64) < lines * 13 / 10,
            "{stave_lines} lines"
        );
    }
}

#[test]
fn module_names_do_not_shadow_builtin_types() {
    const BUILTIN: [&str; 15] = [
        "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "usize", "isize", "f32", "f64", "bool", "char",
        "str",
    ];
    let t = pair_texts(&PairOptions { lines: 120_000, seed: 5, units: 100, reps: 1 });
    for (p, text) in &t.stave {
        for l in text.lines() {
            if let Some(m) = l.strip_prefix("import ").and_then(|r| r.strip_suffix(';')) {
                assert!(!BUILTIN.contains(&m), "{p}: `{l}` shadows a built-in type");
            }
        }
    }
}

#[test]
fn stave_output_parses() {
    let t = pair_texts(&opts(20_000, 3));
    for (i, (p, text)) in t.stave.iter().filter(|(p, _)| p.ends_with(".stv")).enumerate() {
        let pf = stave_syntax::parse_file(text.as_bytes(), FileId(i as u32));
        assert!(pf.stop.is_none(), "{p}: stopped: {:?}", pf.stop);
        if let Some(d) = pf.diagnostics.first() {
            panic!("{p}: {} diagnostics, first: {d:?}", pf.diagnostics.len());
        }
        assert!(pf.items.iter().all(|it| it.ok), "{p}: dropped item");
    }
}
