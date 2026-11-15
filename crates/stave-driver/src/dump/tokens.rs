//! `tokens`: `F O KIND` per token, ordered by (F, O) (stage-gates.md). `KIND` is the terminal
//! name (`Ident`, `IntLit`, `FloatLit`, `StrLit`, `FmtLit`, `CharLit`), a keyword or punctuation
//! by its text, or `ERROR` for a lexical error token. `EOF` (and comments, whitespace) has no
//! line; a file with E0001/E0002 lists only the tokens before the stop point (the lexer keeps
//! only those).

use stave_syntax::{Term, kind_name};

use super::push_u32;
use crate::pipeline::Output;

pub(super) fn file(out: &Output, f: usize, buf: &mut Vec<u8>) {
    let Some(file) = out.program.files.get(f) else { return };
    let eof = Term::Eof as u8;
    buf.reserve(file.parsed.tokens.len() * 12);
    for t in &file.parsed.tokens {
        if t.kind == eof {
            continue;
        }
        push_u32(buf, f as u32);
        buf.push(b' ');
        push_u32(buf, t.start);
        buf.push(b' ');
        buf.extend_from_slice(kind_name(t.kind).as_bytes());
        buf.push(b'\n');
    }
}
