//! Token-level helpers. The AST stores no identifier nodes, so names are read from the token
//! stream at positions fixed by the grammar (node `a` = the node's first token).

use stave_diag::{FileId, Span};
use stave_syntax::{Term, Token};

pub(crate) const T_IDENT: u8 = Term::Ident as u8;
pub(crate) const T_DOT: u8 = Term::Dot as u8;
pub(crate) const T_COLONCOLON: u8 = Term::ColonColon as u8;

#[inline]
pub(crate) fn kind(toks: &[Token], i: usize) -> u8 {
    toks.get(i).map_or(Term::Eof as u8, |t| t.kind)
}

#[inline]
pub(crate) fn is(toks: &[Token], i: usize, t: Term) -> bool {
    kind(toks, i) == t as u8
}

/// The span of token `i`.
pub(crate) fn tok_span(file: FileId, src: &[u8], toks: &[Token], i: usize) -> Span {
    let start = toks.get(i).map_or(0, |t| t.start);
    let end = stave_syntax::token_end(src, toks, i);
    Span::new(file, start, end.saturating_sub(start))
}

/// The span from token `a` through token `b` (inclusive).
pub(crate) fn range_span(file: FileId, src: &[u8], toks: &[Token], a: usize, b: usize) -> Span {
    let start = toks.get(a).map_or(0, |t| t.start);
    let end = stave_syntax::token_end(src, toks, b.max(a));
    Span::new(file, start, end.saturating_sub(start))
}

/// The identifier tokens of a chain `Ident ("." Ident)*` starting at `i` (empty if `i` is not
/// an identifier).
pub(crate) fn ident_chain(toks: &[Token], i: usize, out: &mut Vec<u32>) {
    out.clear();
    if kind(toks, i) != T_IDENT {
        return;
    }
    out.push(i as u32);
    let mut p = i + 1;
    while kind(toks, p) == T_DOT && kind(toks, p + 1) == T_IDENT {
        out.push((p + 1) as u32);
        p += 2;
    }
}

/// The end of the generic argument list opening at token `lt` (a `<`): `(next, partial)`, where
/// `next` is the index of the first token after the list and `partial` says the list ended
/// inside a split token (`>>`, `>=`, `>>=`, D3), so `next - 1` still has a remainder.
pub(crate) fn angle_end(toks: &[Token], lt: usize) -> (usize, bool) {
    let mut angle: i64 = 0;
    let mut bracket: i64 = 0;
    let mut i = lt;
    while i < toks.len() {
        let k = toks[i].kind;
        if k == Term::Eof as u8 {
            return (i, false);
        }
        if k == Term::LParen as u8
            || k == Term::LBracket as u8
            || k == Term::LBrace as u8
            || k == Term::DotLBrace as u8
        {
            bracket += 1;
        } else if k == Term::RParen as u8 || k == Term::RBracket as u8 || k == Term::RBrace as u8 {
            bracket -= 1;
            if bracket < 0 {
                return (i, false);
            }
        } else if bracket == 0 {
            let (delta, whole): (i64, i64) = if k == Term::Lt as u8 {
                (1, 0)
            } else if k == Term::Gt as u8 {
                (-1, 1)
            } else if k == Term::GtGt as u8 {
                (-2, 2)
            } else if k == Term::GtEq as u8 {
                (-1, 2)
            } else if k == Term::GtGtEq as u8 {
                (-2, 3)
            } else {
                (0, 0)
            };
            if delta < 0 {
                let before = angle;
                angle += delta;
                if angle <= 0 {
                    // The token's first `before` closers end the list; it has `whole` closers
                    // or characters in all.
                    return (i + 1, before < whole);
                }
            } else {
                angle += delta;
            }
        }
        i += 1;
    }
    (i, false)
}

/// One f-string hole: the identifier segments (byte offset, text), head first.
pub(crate) struct Hole<'a> {
    pub segs: Vec<(u32, &'a str)>,
}

/// The holes of an `FmtLit` token whose text starts at byte `base` (same scan as the lexer:
/// `\` escapes the next byte; a hole is `{` (Ident | self) ("." Ident)* `}`).
pub(crate) fn fmt_holes(text: &[u8], base: u32) -> Vec<Hole<'_>> {
    let mut out = Vec::new();
    let mut j = 2; // after `f"`
    while j < text.len() {
        match text[j] {
            b'"' => break,
            b'\\' => j += 2,
            b'{' => {
                let mut k = j + 1;
                let mut segs = Vec::new();
                let mut ok = false;
                loop {
                    let s = k;
                    while k < text.len() && (text[k].is_ascii_alphanumeric() || text[k] == b'_') {
                        k += 1;
                    }
                    if k == s {
                        break;
                    }
                    let seg = std::str::from_utf8(&text[s..k]).unwrap_or("");
                    segs.push((base + s as u32, seg));
                    match text.get(k) {
                        Some(b'.') => k += 1,
                        Some(b'}') => {
                            ok = true;
                            k += 1;
                            break;
                        }
                        _ => break,
                    }
                }
                if ok {
                    out.push(Hole { segs });
                }
                j = k.max(j + 1);
            }
            _ => j += 1,
        }
    }
    out
}

/// Names declared by a dropped item: `(name token, is_import_binding)`, plus the
/// identifiers of a dropped `impl` header (its methods are unknown).
pub(crate) struct DroppedNames {
    pub names: Vec<u32>,
    pub impl_header_idents: Vec<u32>,
}

/// Scans the tokens `[start, end)` of a dropped item for the names it would declare.
pub(crate) fn dropped_names(toks: &[Token], start: usize, end: usize) -> DroppedNames {
    let mut out = DroppedNames { names: Vec::new(), impl_header_idents: Vec::new() };
    let end = end.min(toks.len());
    let mut i = start;
    // Prefixes: pub, test, unsafe("..").
    loop {
        if is(toks, i, Term::KwPub) || is(toks, i, Term::KwTest) {
            i += 1;
        } else if is(toks, i, Term::KwUnsafe) {
            i += 1;
            if is(toks, i, Term::LParen) {
                while i < end && !is(toks, i, Term::RParen) {
                    i += 1;
                }
                i += 1;
            }
        } else {
            break;
        }
    }
    let name_at = |j: usize, out: &mut DroppedNames| {
        if j < end && kind(toks, j) == T_IDENT {
            out.names.push(j as u32);
        }
    };
    match toks.get(i).and_then(|t| t.term()) {
        Some(Term::KwImport) => {
            let mut last = None;
            let mut j = i + 1;
            while j < end && !is(toks, j, Term::Semi) {
                if is(toks, j, Term::KwAs) {
                    last = None;
                    if kind(toks, j + 1) == T_IDENT {
                        last = Some(j + 1);
                    }
                    break;
                }
                if kind(toks, j) == T_IDENT {
                    last = Some(j);
                }
                j += 1;
            }
            if let Some(l) = last {
                out.names.push(l as u32);
            }
        }
        Some(Term::KwFn) | Some(Term::KwStruct) | Some(Term::KwEnum) | Some(Term::KwTrait)
        | Some(Term::KwConst) | Some(Term::KwType) => name_at(i + 1, &mut out),
        Some(Term::KwView) => name_at(i + 2, &mut out),
        Some(Term::KwDispatch) | Some(Term::KwKernel) | Some(Term::KwStatic) => name_at(i + 2, &mut out),
        Some(Term::KwExtern) => {
            if is(toks, i + 2, Term::KwFn) {
                name_at(i + 3, &mut out);
            } else {
                let mut depth = 0i32;
                for (j, t) in toks.iter().enumerate().take(end).skip(i + 1) {
                    match t.term() {
                        Some(Term::LBrace)
                        | Some(Term::LParen)
                        | Some(Term::LBracket)
                        | Some(Term::DotLBrace) => depth += 1,
                        Some(Term::RBrace) | Some(Term::RParen) | Some(Term::RBracket) => depth -= 1,
                        Some(Term::KwFn) if depth == 1 => name_at(j + 1, &mut out),
                        _ => {}
                    }
                }
            }
        }
        Some(Term::KwImpl) => {
            let mut depth = 0i32;
            for (j, t) in toks.iter().enumerate().take(end).skip(i + 1) {
                match t.term() {
                    Some(Term::LBrace) if depth == 0 => break,
                    Some(Term::LParen)
                    | Some(Term::LBracket)
                    | Some(Term::LBrace)
                    | Some(Term::DotLBrace) => depth += 1,
                    Some(Term::RParen) | Some(Term::RBracket) | Some(Term::RBrace) => depth -= 1,
                    Some(Term::Ident) => out.impl_header_idents.push(j as u32),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Token> {
        stave_syntax::lex(src.as_bytes(), FileId(0)).tokens
    }

    #[test]
    fn angles() {
        let t = toks("f::<Vec<u8>>(x)");
        // tokens: f :: < Vec < u8 >> ( x ) EOF
        assert_eq!(angle_end(&t, 2), (7, false));
        assert_eq!(angle_end(&t, 4), (7, true));
        let t = toks("a::<T>>= b");
        assert_eq!(angle_end(&t, 2), (5, true));
        let t = toks("g::<{ a < b }, [u8; 4]>()");
        let close = t.iter().position(|x| x.kind == Term::Gt as u8).unwrap();
        assert_eq!(angle_end(&t, 2), (close + 1, false));
    }

    #[test]
    fn holes() {
        let s = br#"f"a {x} \{y} {self.p.q}{bad-} {}""#;
        let h = fmt_holes(s, 100);
        let segs: Vec<Vec<&str>> = h.iter().map(|h| h.segs.iter().map(|s| s.1).collect()).collect();
        assert_eq!(segs, vec![vec!["x"], vec!["self", "p", "q"]]);
        assert_eq!(h[0].segs[0].0, 105);
    }

    #[test]
    fn dropped() {
        let t = toks("pub unsafe(\"x\") fn f(;");
        let d = dropped_names(&t, 0, t.len() - 1);
        assert_eq!(d.names, vec![6]);
        let t = toks("import a.b as c x;");
        assert_eq!(dropped_names(&t, 0, t.len() - 1).names, vec![5]);
        let t = toks("extern \"C\" { fn a(); fn b(; }");
        assert_eq!(dropped_names(&t, 0, t.len() - 1).names.len(), 2);
        let t = toks("static mut S: u8 = ;");
        assert_eq!(dropped_names(&t, 0, t.len() - 1).names, vec![2]);
        let t = toks("view struct V { x: }");
        assert_eq!(dropped_names(&t, 0, t.len() - 1).names, vec![2]);
        let t = toks("impl<T> Tr for W<T> { fn ( }");
        assert_eq!(dropped_names(&t, 0, t.len() - 1).impl_header_idents.len(), 4);
    }
}
