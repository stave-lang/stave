//! Fuzz the lexer: no panic on any bytes, and the token list is well formed.
#![no_main]

use stave_diag::FileId;
use stave_syntax::{Term, lex, token_end};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let l = lex(data, FileId(0));
    let toks = &l.tokens;
    assert_eq!(toks.last().map(|t| t.kind), Some(Term::Eof as u8), "ends with EOF");
    assert_eq!(toks.iter().filter(|t| t.kind == Term::Eof as u8).count(), 1);
    for i in 0..toks.len() {
        let e = token_end(data, toks, i);
        assert!(e >= toks[i].start && e as usize <= data.len(), "token {i} end in range");
        if let Some(next) = toks.get(i + 1) {
            assert!(e <= next.start, "tokens {i} and {} overlap", i + 1);
        }
    }
    for d in &l.diagnostics {
        let s = d.primary().expect("lexical diagnostics are located");
        assert!((s.offset as usize) <= data.len() && (s.offset + s.len) as usize <= data.len());
    }
});
