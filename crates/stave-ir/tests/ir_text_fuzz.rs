//! The reader on mutated printed modules: the printed corpus (the conformance `lir`/`ir`/`opt`
//! dumps and the printer goldens, kitchen sink included) mutated at byte, token and line level,
//! 20000 mutants from fixed seeds. No mutant panics, hangs, or overflows the stack (run with
//! `RUST_MIN_STACK=1048576` too); every accepted mutant passes the verifier, prints, and its text
//! reads back to an equal module that prints the same bytes (an accepted text has one canonical
//! form). A mutant that is accepted and prints as the original text loads the original module.

use std::path::Path;
use std::time::{Duration, Instant};

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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

/// Tokens other inputs hold: ids, numbers at their limits, keywords, punctuation.
const TOKENS: &[&str] = &[
    "%0",
    "%4294967295",
    "%4294967296",
    "b0",
    "b9",
    "t0",
    "t99999",
    "s0",
    "d0",
    "@0",
    "@0.u",
    "@1.0",
    "0",
    "-1",
    "1",
    "255",
    "256",
    "4294967295",
    "18446744073709551615",
    "18446744073709551616",
    "170141183460469231731687303715884105727",
    "-170141183460469231731687303715884105728",
    "170141183460469231731687303715884105728",
    "0x0000",
    "0xffff",
    "0x7fc00001",
    "0x80000000",
    "0x0000000000000001",
    "0x7ff8000000000001",
    "bf16",
    "x00",
    "xff",
    "()",
    "(",
    ")",
    ",",
    ":",
    "=",
    "!",
    "|",
    "{",
    "}",
    "->",
    "<=",
    "\"\"",
    "\"\\x00\"",
    "\"\\xff\"",
    "fn",
    "type",
    "shape",
    "claim",
    "pre",
    "post",
    "checked",
    "main",
    "cell",
    "decl",
    "static",
    "data",
    "ref",
    "site",
    "str",
    "node",
    "edge",
    "nodes",
    "runstack",
    "truncated",
    "br",
    "cbr",
    "switch",
    "ret",
    "unreachable",
    "_:",
    "call",
    "intrinsic.syscall3",
    "intrinsic.x",
    "add",
    "slot",
    "load",
    "store",
    "known",
    "unbuilt",
    "rec",
    "#",
    "\t",
    "\r",
];

fn mutate(r: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut v = base.to_vec();
    for _ in 0..1 + r.below(3) {
        match r.below(10) {
            // byte level
            0 if !v.is_empty() => {
                let i = r.below(v.len());
                v[i] = r.next() as u8;
            }
            1 if !v.is_empty() => {
                let i = r.below(v.len());
                v.remove(i);
            }
            2 => {
                let i = r.below(v.len() + 1);
                v.insert(i, r.next() as u8);
            }
            3 if !v.is_empty() => {
                let i = r.below(v.len());
                let n = r.below(64).min(v.len() - i);
                let dup: Vec<u8> = v[i..i + n].to_vec();
                v.splice(i..i, dup);
            }
            // token level: the tokens of one line
            4..=7 => {
                let lines: Vec<&[u8]> = v.split(|&c| c == b'\n').collect();
                let k = r.below(lines.len());
                let mut toks: Vec<Vec<u8>> = lines[k].split(|&c| c == b' ').map(<[u8]>::to_vec).collect();
                let j = r.below(toks.len());
                match r.below(5) {
                    0 => toks[j] = TOKENS[r.below(TOKENS.len())].as_bytes().to_vec(),
                    1 => {
                        toks.remove(j);
                    }
                    2 => toks.insert(j, TOKENS[r.below(TOKENS.len())].as_bytes().to_vec()),
                    3 => {
                        let i = r.below(toks.len());
                        toks.swap(i, j);
                    }
                    _ => {
                        // a number: change it by one, or to a limit
                        if let Some(d) = toks[j].iter().position(u8::is_ascii_digit) {
                            let (head, digits) = toks[j].split_at(d);
                            let n: u128 =
                                std::str::from_utf8(digits).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                            let m = match r.below(4) {
                                0 => n.wrapping_add(1),
                                1 => n.wrapping_sub(1),
                                2 => u128::from(u32::MAX),
                                _ => u128::MAX,
                            };
                            toks[j] = [head, m.to_string().as_bytes()].concat();
                        }
                    }
                }
                let line = toks.join(&b' ');
                let mut out: Vec<Vec<u8>> = lines.iter().map(|l| l.to_vec()).collect();
                out[k] = line;
                v = out.join(&b'\n');
            }
            // line level
            _ => {
                let mut lines: Vec<Vec<u8>> = v.split(|&c| c == b'\n').map(<[u8]>::to_vec).collect();
                let (i, j) = (r.below(lines.len()), r.below(lines.len()));
                match r.below(3) {
                    0 => {
                        lines.remove(i);
                    }
                    1 => {
                        let l = lines[i].clone();
                        lines.insert(j, l);
                    }
                    _ => lines.swap(i, j),
                }
                v = lines.join(&b'\n');
            }
        }
    }
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
