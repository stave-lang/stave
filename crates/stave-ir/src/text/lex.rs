//! The lexer of one IR text line (contracts/ir-text.ebnf, lexical part). It works on bytes, never
//! on `str`, so any input is a sequence of tokens or one error at a byte offset.

use crate::generated::ir_text_terminals::{TERM_IS_CLASS, TERM_TEXT, Term};

/// A token: its terminal and its byte range in the line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tok {
    pub term: Term,
    pub start: u32,
    pub end: u32,
}

/// A lexical error at byte offset `at` of the line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LexError {
    pub at: usize,
    pub msg: String,
}

fn err<T>(at: usize, msg: impl Into<String>) -> Result<T, LexError> {
    Err(LexError { at, msg: msg.into() })
}

/// A byte that continues a `Word`, `Num`, `Val`, `FnRef` or `Hex` token.
fn is_word_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.'
}

pub(crate) fn is_lower_hex(c: u8) -> bool {
    c.is_ascii_digit() || (b'a'..=b'f').contains(&c)
}

/// `Num`: `0` or a digit run without a leading zero.
pub(crate) fn is_num(w: &[u8]) -> bool {
    !w.is_empty() && w.iter().all(u8::is_ascii_digit) && (w == b"0" || w[0] != b'0')
}

/// The `Word` shape `[A-Za-z_][A-Za-z0-9_.]*`.
pub(crate) fn is_word(w: &[u8]) -> bool {
    match w.split_first() {
        Some((&c, rest)) => (c.is_ascii_alphabetic() || c == b'_') && rest.iter().all(|&c| is_word_byte(c)),
        None => false,
    }
}

/// The end of the `Num` starting at `i`.
fn num_end(b: &[u8], i: usize) -> Result<usize, LexError> {
    let mut j = i;
    while b.get(j).is_some_and(u8::is_ascii_digit) {
        j += 1;
    }
    if j == i {
        return err(i, "a decimal number is expected here");
    }
    if b[i] == b'0' && j - i > 1 {
        return err(i, "a number has no leading zero");
    }
    Ok(j)
}

/// The end of the `FnRef` starting at `i` (the `@`).
fn fnref_end(b: &[u8], i: usize) -> Result<usize, LexError> {
    let mut j = num_end(b, i + 1)?;
    while b.get(j) == Some(&b'.') && b.get(j + 1).is_some_and(u8::is_ascii_digit) {
        j = num_end(b, j + 1)?;
    }
    if b.get(j) == Some(&b'.') && b.get(j + 1) == Some(&b'u') {
        j += 2;
    }
    Ok(j)
}

/// The end of the string starting at `i` (the opening quote), after checking every byte and
/// escape.
fn string_end(b: &[u8], i: usize) -> Result<usize, LexError> {
    let mut j = i + 1;
    loop {
        match b.get(j) {
            None => return err(i, "unterminated string"),
            Some(b'"') => return Ok(j + 1),
            Some(b'\\') => match b.get(j + 1) {
                Some(b'\\' | b'"' | b'n' | b't') => j += 2,
                Some(b'x') => {
                    let hex = b.get(j + 2..j + 4).filter(|h| h.iter().all(|&d| is_lower_hex(d)));
                    let Some(h) = hex else {
                        return err(j, "`\\x` takes two lowercase hex digits");
                    };
                    let byte = hex_val(h[0]) * 16 + hex_val(h[1]);
                    if let Some(short) = shorter_escape(byte) {
                        return err(j, format!("`\\x{:02x}` has the shorter form `{short}`", byte));
                    }
                    j += 4;
                }
                _ => return err(j, "unknown escape (the escapes are `\\\\ \\\" \\n \\t \\xHH`)"),
            },
            Some(0x20..=0x7e) => j += 1,
            Some(&c) => return err(j, format!("byte 0x{c:02x} in a string is written `\\x{c:02x}`")),
        }
    }
}

/// The form a byte must be written in when it has one other than `\xHH`.
fn shorter_escape(byte: u8) -> Option<String> {
    match byte {
        b'\\' => Some("\\\\".into()),
        b'"' => Some("\\\"".into()),
        b'\n' => Some("\\n".into()),
        b'\t' => Some("\\t".into()),
        0x20..=0x7e => Some((byte as char).to_string()),
        _ => None,
    }
}

fn hex_val(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => 0,
    }
}

/// The bytes a string token denotes (`raw` includes the quotes; the lexer has checked it).
pub(crate) fn unescape(raw: &[u8]) -> Vec<u8> {
    let inner = raw.get(1..raw.len().saturating_sub(1)).unwrap_or(&[]);
    let mut out = Vec::with_capacity(inner.len());
    let mut i = 0;
    while i < inner.len() {
        if inner[i] == b'\\' && i + 1 < inner.len() {
            match inner[i + 1] {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'x' if i + 3 < inner.len() => {
                    out.push(hex_val(inner[i + 2]) * 16 + hex_val(inner[i + 3]));
                    i += 2;
                }
                c => out.push(c),
            }
            i += 2;
        } else {
            out.push(inner[i]);
            i += 1;
        }
    }
    out
}

/// Whether `name` can follow `intrinsic.` in the text: `[a-z0-9_]+` (contracts/ir-text.ebnf
/// `Intr`). Whether it names an intrinsic is the verifier's question ([`crate::intrinsics::kind`]).
pub(crate) fn is_intrinsic_word(name: &[u8]) -> bool {
    !name.is_empty() && name.iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
}

/// The terminal of a `Word`, classified by shape first, then by keyword (contracts/ir-text.ebnf
/// `Word`).
pub(crate) fn classify(w: &[u8]) -> Term {
    if let Some(r) = w.strip_prefix(b"intrinsic.")
        && is_intrinsic_word(r)
    {
        return Term::Intr;
    }
    let shaped = |p: u8| w.first() == Some(&p) && is_num(&w[1..]);
    if shaped(b'b') {
        return Term::Blk;
    }
    if shaped(b't') {
        return Term::TId;
    }
    if shaped(b's') {
        return Term::SId;
    }
    if shaped(b'd') {
        return Term::DId;
    }
    if let Some(h) = w.strip_prefix(b"x")
        && (2..=128).contains(&h.len())
        && h.len().is_multiple_of(2)
        && h.iter().all(|&c| is_lower_hex(c))
    {
        return Term::Bytes;
    }
    keyword(w).unwrap_or(Term::AnyWord)
}

/// The keyword terminal spelled `w` (never a class name).
fn keyword(w: &[u8]) -> Option<Term> {
    let i = TERM_TEXT.binary_search_by(|t| t.as_bytes().cmp(w)).ok()?;
    if TERM_IS_CLASS[i] { None } else { Term::from_id(i as u16) }
}

/// The tokens of one line (without `EOF`); empty for a blank or comment-only line.
pub(crate) fn lex(b: &[u8]) -> Result<Vec<Tok>, LexError> {
    if let Some(at) = b.iter().position(|&c| c == b'\r') {
        return err(at, "a carriage return is not allowed in IR text");
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let s = i;
        let (term, delimited) = match c {
            b' ' | b'\t' => {
                i += 1;
                continue;
            }
            b'#' => break,
            b'"' => {
                i = string_end(b, i)?;
                (Term::String, false)
            }
            b'%' => {
                i = num_end(b, i + 1)?;
                (Term::Val, true)
            }
            b'@' => {
                i = fnref_end(b, i)?;
                (Term::FnRef, true)
            }
            b'-' => match b.get(i + 1) {
                Some(b'>') => {
                    i += 2;
                    (Term::MinusGt, false)
                }
                Some(b'0'..=b'9') => {
                    if b[i + 1] == b'0' && !b.get(i + 2).is_some_and(u8::is_ascii_digit) {
                        return err(i, "`-0` is not an integer: write `0`");
                    }
                    i = num_end(b, i + 1)?;
                    (Term::Int, true)
                }
                _ => return err(i, "`-` begins `->` or a negative integer"),
            },
            b'0' if b.get(i + 1) == Some(&b'x') => {
                let mut j = i + 2;
                while b.get(j).is_some_and(|&d| d.is_ascii_alphanumeric()) {
                    j += 1;
                }
                let digits = &b[i + 2..j];
                if !digits.iter().all(|&d| is_lower_hex(d)) || ![4, 8, 16].contains(&digits.len()) {
                    return err(i, "a float constant is `0x` and 4, 8 or 16 lowercase hex digits");
                }
                i = j;
                (Term::Hex, true)
            }
            b'0'..=b'9' => {
                i = num_end(b, i)?;
                (Term::Int, true)
            }
            b'<' if b.get(i + 1) == Some(&b'=') => {
                i += 2;
                (Term::LtEq, false)
            }
            b'(' | b')' | b'<' | b'>' | b',' | b':' | b'=' | b'!' | b'|' | b'{' | b'}' => {
                i += 1;
                let t = match c {
                    b'(' => Term::LParen,
                    b')' => Term::RParen,
                    b'<' => Term::Lt,
                    b'>' => Term::Gt,
                    b',' => Term::Comma,
                    b':' => Term::Colon,
                    b'=' => Term::Eq,
                    b'!' => Term::Bang,
                    b'|' => Term::Pipe,
                    b'{' => Term::LBrace,
                    _ => Term::RBrace,
                };
                (t, false)
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                while b.get(i).is_some_and(|&c| is_word_byte(c)) {
                    i += 1;
                }
                (classify(&b[s..i]), false)
            }
            _ if c.is_ascii_graphic() => return err(i, format!("unexpected `{}`", c as char)),
            _ => return err(i, format!("unexpected byte 0x{c:02x}")),
        };
        if delimited && b.get(i).is_some_and(|&c| is_word_byte(c)) {
            return err(i, format!("`{}` must be followed by a space or a punctuation mark", show(&b[s..i])));
        }
        out.push(Tok { term, start: s as u32, end: i as u32 });
    }
    Ok(out)
}

/// Printable text of token bytes for a message.
pub(crate) fn show(b: &[u8]) -> String {
    b.iter()
        .map(|&c| if (0x20..=0x7e).contains(&c) { (c as char).to_string() } else { format!("\\x{c:02x}") })
        .collect()
}
