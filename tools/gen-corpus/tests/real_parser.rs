//! gen-corpus output through the stage0 front end (`stave_syntax::parse_file`): no lexical or
//! syntax diagnostic, every item parsed.

use gen_corpus::{Mix, Options, generate};
use stave_diag::FileId;
use std::path::{Path, PathBuf};

fn stave_files(d: &Path, out: &mut Vec<PathBuf>) {
    let mut es: Vec<PathBuf> =
        std::fs::read_dir(d).expect("read_dir").map(|e| e.expect("entry").path()).collect();
    es.sort();
    for p in es {
        if p.is_dir() {
            stave_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "stv") {
            out.push(p);
        }
    }
}

#[test]
fn stage0_parser_accepts_every_file() {
    for &(lines, seed) in &[(5_000u64, 1u64), (30_000, 2), (30_000, 3)] {
        let dir = std::env::temp_dir().join(format!("gen-corpus-real-{}-{lines}-{seed}", std::process::id()));
        let mix = Mix::default_mix().expect("mix");
        generate(&Options { lines, seed, out: dir.clone(), mix, threads: 0, force: true }).expect("generate");
        let mut files = Vec::new();
        stave_files(&dir.join("src"), &mut files);
        assert!(!files.is_empty());
        for (i, f) in files.iter().enumerate() {
            let src = std::fs::read(f).expect("read");
            let pf = stave_syntax::parse_file(&src, FileId(i as u32));
            assert!(pf.stop.is_none(), "{}: stopped: {:?}", f.display(), pf.stop);
            if let Some(d) = pf.diagnostics.first() {
                panic!("{}: {} diagnostics, first: {d:?}", f.display(), pf.diagnostics.len());
            }
            assert!(pf.items.iter().all(|it| it.ok), "{}: dropped item", f.display());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
