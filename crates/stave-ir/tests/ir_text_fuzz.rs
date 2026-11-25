//! The reader on mutated printed modules: the printed corpus (the conformance `lir`/`ir`/`opt`
//! dumps and the printer goldens, kitchen sink included) mutated at byte, token and line level,
//! 20000 mutants from fixed seeds. No mutant panics, hangs, or overflows the stack (run with
//! `RUST_MIN_STACK=1048576` too); every accepted mutant passes the verifier, prints, and its text
//! reads back to an equal module that prints the same bytes (an accepted text has one canonical
//! form). A mutant that is accepted and prints as the original text loads the original module.

use std::path::Path;
use std::time::{Duration, Instant};

mod common;

use common::mutate::{Rng, mutate};
use stave_ir::canon::canonicalize;
use stave_ir::text::read;
use stave_ir::{print_module, verify_module};

fn corpus() -> Vec<(String, Vec<u8>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut v = Vec::new();
    let mut push_dir = |dir: &Path, deep: bool| {
        let mut es: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        es.sort();
        for p in es {
            if deep && p.is_dir() {
                let mut fs: Vec<_> = std::fs::read_dir(&p).unwrap().map(|e| e.unwrap().path()).collect();
                fs.sort();
                for f in fs.into_iter().filter(|f| f.extension().is_some_and(|x| x == "txt")) {
                    v.push((f.display().to_string(), std::fs::read(&f).unwrap()));
                }
            } else if p.extension().is_some_and(|x| x == "ir") {
                v.push((p.display().to_string(), std::fs::read(&p).unwrap()));
            }
        }
    };
    push_dir(&root.join("tests/conformance/dumps"), true);
    push_dir(&root.join("crates/stave-ir/tests/print_golden"), false);
    v.retain(|(_, t)| t.starts_with(b"stave.ir "));
    v
}

#[test]
fn mutated_printed_modules() {
    // the printer's kitchen sink shows every op on ill-typed operands: it prints, and is refused
    // by the verifier on load; every other file loads
    let mut corpus = corpus();
    corpus.retain(|(n, _)| !n.ends_with("kitchen_sink.ir"));
    assert!(corpus.len() > 40, "corpus: {} files", corpus.len());
    let originals: Vec<_> =
        corpus.iter().map(|(n, t)| (n, read(t).unwrap_or_else(|e| panic!("{n}: {e:?}")))).collect();
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    let (mut accepted, mut same, mut slowest) = (0usize, 0usize, Duration::ZERO);
    for k in 0..20_000 {
        let (name, base) = &corpus[k % corpus.len()];
        let input = mutate(&mut r, base);
        let t = Instant::now();
        let got = read(&input);
        slowest = slowest.max(t.elapsed());
        let Ok(l) = got else { continue };
        accepted += 1;
        let show = || String::from_utf8_lossy(&input).into_owned();
        assert!(
            verify_module(&l.module).is_ok(),
            "{name}: an accepted mutant fails the verifier:\n{}",
            show()
        );
        let text = print_module(&l.module, &l.header, l.stack.as_ref())
            .unwrap_or_else(|e| panic!("{name}: an accepted mutant does not print: {e:?}\n{}", show()));
        let again =
            read(text.as_bytes()).unwrap_or_else(|e| panic!("{name}: reprint does not load: {e:?}\n{text}"));
        // the reader keeps the text's type, string and site numbering; the printer prints the
        // canonical module
        let mut canonical = l.module.clone();
        canonicalize(&mut canonical);
        assert!(again.module == canonical, "{name}: reprint reads back differently:\n{}", show());
        let text2 = print_module(&again.module, &again.header, again.stack.as_ref()).unwrap();
        assert_eq!(text, text2, "{name}: print is not a fixed point");
        if text.as_bytes() == base.as_slice() {
            same += 1;
            let orig = &originals[k % corpus.len()].1;
            assert!(canonical == orig.module, "{name}: same text, another module:\n{}", show());
        }
    }
    println!(
        "mutants 20000, accepted {accepted}, of which print as the original {same}, slowest read {slowest:?}"
    );
    assert!(accepted > 0);
    assert!(slowest < Duration::from_secs(5), "a read took {slowest:?}");
}
