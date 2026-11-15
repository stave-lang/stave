//! Helpers shared by the stave-syntax integration tests.
#![allow(dead_code)]

use stave_diag::{Diagnostic, FileId, LineIndex};
use stave_syntax::{ParsedFile, kind_name, lex, parse_file, token_text};
use std::path::{Path, PathBuf};

pub const F: FileId = FileId(0);

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Token kinds (dump spelling) of `src`, without the final `EOF`.
pub fn kinds(src: &str) -> Vec<&'static str> {
    let l = lex(src.as_bytes(), F);
    let mut v: Vec<&'static str> = l.tokens.iter().map(|t| kind_name(t.kind)).collect();
    assert_eq!(v.pop(), Some("EOF"));
    v
}

/// `kind:text` per token, without `EOF`.
pub fn toks(src: &str) -> Vec<String> {
    let l = lex(src.as_bytes(), F);
    let n = l.tokens.len() - 1;
    (0..n)
        .map(|i| {
            let t = String::from_utf8_lossy(token_text(src.as_bytes(), &l.tokens, i)).into_owned();
            format!("{}:{}", kind_name(l.tokens[i].kind), t)
        })
        .collect()
}

/// Lexical diagnostic codes with their offsets.
pub fn lex_diags(src: &[u8]) -> Vec<(&'static str, u32)> {
    lex(src, F).diagnostics.iter().map(|d| (d.code(), d.primary().map_or(u32::MAX, |s| s.offset))).collect()
}

/// `(code, line)` of every diagnostic of the parsed file.
pub fn code_lines(src: &[u8], pf: &ParsedFile) -> Vec<(&'static str, u32)> {
    let li = LineIndex::new(src);
    pf.diagnostics
        .iter()
        .map(|d| {
            let line = d.primary().and_then(|s| li.line_col(s.offset)).map_or(0, |(l, _)| l);
            (d.code(), line)
        })
        .collect()
}

/// Codes of every diagnostic of `parse_file(src)`.
pub fn codes(src: &str) -> Vec<&'static str> {
    parse_file(src.as_bytes(), F).diagnostics.iter().map(Diagnostic::code).collect()
}

/// The single diagnostic of `parse_file(src)`.
pub fn one(src: &str) -> Diagnostic {
    let pf = parse_file(src.as_bytes(), F);
    assert_eq!(pf.diagnostics.len(), 1, "{src:?}: {:#?}", pf.diagnostics);
    pf.diagnostics[0].clone()
}

/// `(code, source text at the primary span)` of the single diagnostic.
pub fn one_at(src: &str) -> (&'static str, String) {
    let d = one(src);
    let s = d.primary().expect("located");
    let text = src.get(s.offset as usize..(s.offset + s.len) as usize).unwrap_or("?").to_string();
    (d.code(), text)
}

/// Every ```stave block of LANG.md.
pub fn lang_md_blocks() -> Vec<String> {
    let md = std::fs::read_to_string(repo().join("LANG.md")).expect("LANG.md");
    let mut out = Vec::new();
    let mut rest = md.as_str();
    while let Some(s) = rest.find("```stave\n") {
        let body = &rest[s + "```stave\n".len()..];
        let e = body.find("```").expect("unclosed block");
        out.push(body[..e].to_string());
        rest = &body[e + 3..];
    }
    assert!(!out.is_empty());
    out
}

/// Every `.stv` file under `dir`, sorted.
pub fn stave_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                todo.push(p);
            } else if p.extension().is_some_and(|x| x == "stv") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}
