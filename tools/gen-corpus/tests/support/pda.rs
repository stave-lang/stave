//! Minimal Stave lexer and table-driven LL(1) PDA over the generated `stave-syntax` tables (same
//! rules as tools/ll1gen/tests/pda.rs, split actions included). Shared by the gen-corpus and
//! touch-fn tests.

#![allow(dead_code)]

use stave_syntax::generated::ll1_table as lt;
use stave_syntax::generated::terminals::{SPLIT_FIRST, SPLIT_REST, TERM_IS_CLASS, TERM_TEXT, Term};

fn term(text: &str) -> u16 {
    TERM_TEXT.iter().position(|t| *t == text).unwrap_or_else(|| panic!("no terminal {text:?}")) as u16
}

pub fn is_idstart(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

pub fn is_idcont(b: u8) -> bool {
    is_idstart(b) || b.is_ascii_digit()
}

/// Minimal Stave lexer producing terminal ids (same rules as tools/ll1gen/tests/pda.rs).
pub fn lex(src: &str) -> Result<Vec<u16>, String> {
    let b = src.as_bytes();
    let n = b.len();
    let at = |j: usize| b.get(j).copied().unwrap_or(0);
    let mut ops: Vec<(&str, u16)> = TERM_TEXT
        .iter()
        .enumerate()
        .filter(|&(i, t)| !TERM_IS_CLASS[i] && !t.bytes().all(is_idcont))
        .map(|(i, t)| (*t, i as u16))
        .collect();
    ops.sort_by_key(|&(t, _)| std::cmp::Reverse(t.len()));
    let mut out: Vec<u16> = Vec::new();
    let mut i = 0;
    let skip_quoted = |mut j: usize, q: u8| -> Result<usize, String> {
        while j < n && b[j] != q && b[j] != b'\n' {
            j += if b[j] == b'\\' { 2 } else { 1 };
        }
        if j < n && b[j] == q { Ok(j + 1) } else { Err(format!("unterminated literal at {j}")) }
    };
    let dot = term(".");
    while i < n {
        let c = b[i];
        let prev_dot = out.last() == Some(&dot);
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && at(i + 1) == b'/' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'f' && at(i + 1) == b'"' {
            i = skip_quoted(i + 2, b'"')?;
            out.push(Term::FmtLit.id());
        } else if is_idstart(c) {
            let s = i;
            while i < n && is_idcont(b[i]) {
                i += 1;
            }
            let word = &src[s..i];
            if word == "as" && matches!(at(i), b'%' | b'|' | b'?' | b'~') {
                out.push(term(&src[s..i + 1]));
                i += 1;
            } else if let Some(k) = TERM_TEXT.iter().position(|t| *t == word) {
                out.push(if TERM_IS_CLASS[k] { Term::Ident.id() } else { k as u16 });
            } else {
                out.push(Term::Ident.id());
            }
        } else if c.is_ascii_digit() {
            let mut float = false;
            if prev_dot {
                while at(i).is_ascii_digit() {
                    i += 1;
                }
            } else if c == b'0' && matches!(at(i + 1), b'x' | b'b' | b'o') {
                i += 2;
                while is_idcont(at(i)) {
                    i += 1;
                }
            } else {
                while at(i).is_ascii_digit() || at(i) == b'_' {
                    i += 1;
                }
                if at(i) == b'.' && at(i + 1).is_ascii_digit() {
                    float = true;
                    i += 1;
                    while at(i).is_ascii_digit() || at(i) == b'_' {
                        i += 1;
                    }
                }
                if matches!(at(i), b'e' | b'E') {
                    let d = if matches!(at(i + 1), b'+' | b'-') { 2 } else { 1 };
                    if at(i + d).is_ascii_digit() {
                        float = true;
                        i += d;
                        while at(i).is_ascii_digit() || at(i) == b'_' {
                            i += 1;
                        }
                    }
                }
                while i < n && is_idcont(b[i]) {
                    i += 1;
                }
            }
            out.push(if float { Term::FloatLit.id() } else { Term::IntLit.id() });
        } else if c == b'"' {
            i = skip_quoted(i + 1, b'"')?;
            out.push(Term::StrLit.id());
        } else if c == b'\'' {
            i = skip_quoted(i + 1, b'\'')?;
            out.push(Term::CharLit.id());
        } else if let Some(&(t, id)) = ops.iter().find(|(t, _)| b[i..].starts_with(t.as_bytes())) {
            out.push(id);
            i += t.len();
        } else {
            return Err(format!("unexpected byte {c:#x} at {i}"));
        }
    }
    Ok(out)
}

/// Table-driven PDA over the generated tables (split actions included).
pub fn parse(src: &str) -> Result<(), String> {
    let mut toks = lex(src)?;
    toks.push(Term::Eof.id());
    let mut stack = vec![lt::NT_BIT | lt::START];
    let mut i = 0;
    let name = |t: u16| TERM_TEXT[t as usize];
    while let Some(top) = stack.pop() {
        let tok = *toks.get(i).ok_or("input ended before the stack")?;
        let first = SPLIT_FIRST[tok as usize];
        if top & lt::NT_BIT == 0 {
            if top == tok {
                i += 1;
            } else if first >= 0 && top == first as u16 {
                toks[i] = SPLIT_REST[tok as usize] as u16;
            } else {
                return Err(format!("token {i}: expected {:?}, got {:?}", name(top), name(tok)));
            }
            continue;
        }
        let nt = top & !lt::NT_BIT;
        let mut p = lt::action(nt, tok);
        if p < 0 && first >= 0 {
            p = lt::action(nt, first as u16);
        }
        if p < 0 {
            return Err(format!("token {i}: {} cannot start with {:?}", lt::NT_NAME[nt as usize], name(tok)));
        }
        stack.extend_from_slice(lt::rhs_rev(p as u16));
    }
    if i != toks.len() {
        return Err(format!("trailing tokens at {i}"));
    }
    Ok(())
}

/// Error position as `line: text` for a failed parse (re-lexes line by line).
pub fn locate(src: &str, err: &str) -> String {
    let tok: usize = err
        .strip_prefix("token ")
        .and_then(|r| r.split(':').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    let mut count = 0;
    for (ln, line) in src.lines().enumerate() {
        let n = lex(line).map(|t| t.len()).unwrap_or(0);
        if count + n > tok {
            return format!("line {}: {line}", ln + 1);
        }
        count += n;
    }
    String::from("?")
}
