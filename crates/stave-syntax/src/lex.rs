//! The lexer.
//!
//! The lexical grammar is regular: a token is recognised by a finite automaton with at most three
//! bytes of lookahead, and no lexer state crosses a newline (string, f-string, and char literals
//! end at a raw newline, E0015). Every newline is therefore a restart point for a parallel lexer
//! (stage1); this sequential scanner is the reference.
//!
//! Rules (contracts/grammar.ebnf §lexical, and the header's "single tokens" list):
//! - whitespace is ` \t\r\n`; comments are `//` up to the newline. Neither makes a token. There
//!   are no block comments: `/*` starts one [`KIND_ERROR`] token (E0029) that ends after the first
//!   `*/` on its line, else at the end of the line (`a / *p` needs the space).
//! - a file whose first two bytes are `#!` starts with a shebang line: a comment up to the first
//!   newline ([`shebang_end`]). Only offset 0 has this rule, so it adds no state to a restart
//!   point: `#!` anywhere else (after a byte-order mark too) is E0018.
//! - identifiers are ASCII only: `[A-Za-z_][A-Za-z0-9_]*`; the text decides keyword, `_` (the
//!   wildcard token, while `_x` is an `Ident`), or `Ident`. `as` immediately followed by `%`, `|`,
//!   `?` or `~` is the single token `as%`/`as|`/`as?`/`as~` (`as %` is two tokens).
//! - numbers: directly after a `.` token (adjacent), `[0-9] idcont*` is an `IntLit`, so `t.0.1` is
//!   two tuple steps. `0x`/`0b`/`0o` + `idcont*` is an `IntLit`. Otherwise
//!   `[0-9][0-9_]* ('.' [0-9][0-9_]*)? ([eE] [+-]? [0-9][0-9_]*)? idcont*`, a `FloatLit` iff it
//!   has a fraction or an exponent; `.` needs a digit after it (`1.` is `1` `.`, `0..n` is
//!   `0` `..` `n`). Digit and suffix validity are checked by later stages.
//! - `"…"`, `'…'`, and `f"…"` with `\` escapes (a `\` never escapes a newline). An f-string is one
//!   `FmtLit` token; each `{…}` hole must be `(Ident | self) ('.' Ident)* }` (E0006 per bad hole).
//!   A literal reaching a raw newline is E0015 and ends at the newline; reaching the end of the
//!   file first is E0003. Both are reported at the literal's start.
//! - a `'` followed by an identifier with no `'` directly after it is a lifetime, not a char
//!   literal: one [`KIND_ERROR`] token (the `'` and the identifier) with E0024.
//! - operators: longest match over the grammar's punctuation terminals.
//! - any other byte (`@ $ \ # \``, a control byte, DEL, any byte >= 0x80) starts no token: a run of
//!   consecutive such bytes is one [`KIND_ERROR`] token with one E0018 at its start. Non-ASCII text
//!   is allowed only inside string, char, and f-string literals and comments.
//!
//! File-level rules: a file over `u32::MAX` bytes is E0002 at offset 0 and is
//! not lexed; invalid UTF-8 is E0001 at the first invalid byte, and only tokens that end before it
//! are kept; a byte-order mark at offset 0 is E0017 and lexing resumes after it.

use stave_diag::{Diagnostic, FileId, Span};

use crate::generated::terminals::{NTERM, TERM_TEXT, Term};
use crate::make_diag;

/// A token: its kind and the byte offset of its first byte (8 bytes). The end
/// is not stored; [`token_end`] recomputes it by re-lexing the one token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Token {
    /// A terminal id (`Term as u8`) or [`KIND_ERROR`].
    pub kind: u8,
    /// Byte offset of the token's first byte.
    pub start: u32,
}

/// A token with a lexical diagnostic: a literal with E0003, E0006, or E0015, a run of bytes that
/// start no token (E0018), a lifetime (E0024), or a block comment (E0029). An item that contains
/// one is dropped with its lexical diagnostics only (no E0004).
pub const KIND_ERROR: u8 = NTERM as u8;

const _: () = assert!(NTERM < 256, "token kinds must fit a u8");

impl Token {
    /// The terminal, or `None` for an error token.
    pub fn term(self) -> Option<Term> {
        Term::from_id(self.kind as u16)
    }

    /// True for [`KIND_ERROR`].
    pub fn is_error(self) -> bool {
        self.kind >= KIND_ERROR
    }

    /// The `tokens` dump spelling of the kind (contracts/stage-gates.md).
    pub fn kind_name(self) -> &'static str {
        kind_name(self.kind)
    }
}

/// The `tokens` dump spelling of a token kind: the terminal name (`Ident`, `IntLit`, …, `EOF`) or
/// text (`fn`, `.{`, `as%`), and `ERROR` for an error token.
pub fn kind_name(kind: u8) -> &'static str {
    TERM_TEXT.get(kind as usize).copied().unwrap_or("ERROR")
}

/// Why lexing stopped early (no items are cut or parsed for such a file).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LexStop {
    /// E0002: the file is larger than `u32::MAX` bytes. No tokens.
    TooLarge,
    /// E0001: invalid UTF-8 at `offset`. Only tokens ending at or before it are kept.
    InvalidUtf8 { offset: u32 },
}

/// The result of [`lex`].
#[derive(Clone, Debug)]
pub struct Lexed {
    /// The tokens, ending with exactly one `EOF` token (at the end of the file, or at the stop
    /// point: offset 0 for E0002, the invalid byte for E0001).
    pub tokens: Vec<Token>,
    /// Lexical diagnostics in offset order: E0001, E0002, E0003, E0006, E0015, E0017, E0018,
    /// E0024, E0029.
    pub diagnostics: Vec<Diagnostic>,
    /// Set for E0001 / E0002: the file was not lexed to its end.
    pub stop: Option<LexStop>,
}

/// Largest accepted file: offsets (and the `EOF` token's start) must fit a `u32`.
pub const MAX_FILE_LEN: usize = u32::MAX as usize;

const BOM: &[u8] = b"\xEF\xBB\xBF";

// Terminal ids used by the scanner.
const T_EOF: u8 = Term::Eof as u8;
const T_DOT: u8 = Term::Dot as u8;
const T_IDENT: u8 = Term::Ident as u8;
const T_INT: u8 = Term::IntLit as u8;
const T_FLOAT: u8 = Term::FloatLit as u8;
const T_STR: u8 = Term::StrLit as u8;
const T_FMT: u8 = Term::FmtLit as u8;
const T_CHAR: u8 = Term::CharLit as u8;
const T_UNDERSCORE: u8 = Term::Underscore as u8;

// Byte classes.
const C_IDCONT: u8 = 1;
const C_IDSTART: u8 = 2;
const C_DIGIT: u8 = 4;
const C_WS: u8 = 8;
/// Starts no token (E0018): not whitespace, an identifier or digit byte, a quote, or a
/// punctuation byte of the grammar.
const C_BAD: u8 = 16;

static CLASS: [u8; 256] = build_class();

const fn build_class() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        let mut c = 0;
        if b.is_ascii_alphabetic() || b == b'_' {
            c |= C_IDSTART | C_IDCONT;
        }
        if b.is_ascii_digit() {
            c |= C_DIGIT | C_IDCONT;
        }
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            c |= C_WS;
        }
        let punct = matches!(
            b,
            b'!' | b'%'
                | b'&'
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b'-'
                | b'.'
                | b'/'
                | b':'
                | b';'
                | b'<'
                | b'='
                | b'>'
                | b'?'
                | b'['
                | b']'
                | b'^'
                | b'{'
                | b'|'
                | b'}'
                | b'~'
        );
        if c == 0 && !punct && b != b'"' && b != b'\'' {
            c |= C_BAD;
        }
        t[i] = c;
        i += 1;
    }
    t
}

#[inline(always)]
fn class(b: u8) -> u8 {
    CLASS[b as usize]
}

#[inline(always)]
fn is_idcont(b: u8) -> bool {
    class(b) & C_IDCONT != 0
}

#[inline(always)]
fn is_idstart(b: u8) -> bool {
    class(b) & C_IDSTART != 0
}

#[inline(always)]
fn is_bad(b: u8) -> bool {
    class(b) & C_BAD != 0
}

#[inline(always)]
fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

/// The byte at `j`, or 0 past the end (0 is in no byte class).
#[inline(always)]
fn at(src: &[u8], j: usize) -> u8 {
    src.get(j).copied().unwrap_or(0)
}

/// The keyword terminal spelled `s`, if any (`_` is not a keyword).
pub(crate) fn keyword(s: &[u8]) -> Option<u8> {
    use Term::*;
    let &first = s.first()?;
    let t = match first {
        b'a' => match s {
            b"as" => KwAs,
            b"align" => KwAlign,
            b"allocator" => KwAllocator,
            _ => return None,
        },
        b'b' => match s {
            b"break" => KwBreak,
            _ => return None,
        },
        b'c' => match s {
            b"const" => KwConst,
            b"continue" => KwContinue,
            b"copy" => KwCopy,
            _ => return None,
        },
        b'd' => match s {
            b"dispatch" => KwDispatch,
            b"decreases" => KwDecreases,
            b"derive" => KwDerive,
            _ => return None,
        },
        b'e' => match s {
            b"else" => KwElse,
            b"enum" => KwEnum,
            b"extern" => KwExtern,
            b"exact" => KwExact,
            b"emulate" => KwEmulate,
            b"ensures" => KwEnsures,
            b"exists" => KwExists,
            _ => return None,
        },
        b'f' => match s {
            b"fn" => KwFn,
            b"for" => KwFor,
            b"from" => KwFrom,
            b"false" => KwFalse,
            b"forall" => KwForall,
            _ => return None,
        },
        b'g' => match s {
            b"ghost" => KwGhost,
            _ => return None,
        },
        b'i' => match s {
            b"if" => KwIf,
            b"in" => KwIn,
            b"impl" => KwImpl,
            b"import" => KwImport,
            b"invariant" => KwInvariant,
            _ => return None,
        },
        b'k' => match s {
            b"kernel" => KwKernel,
            _ => return None,
        },
        b'l' => match s {
            b"let" => KwLet,
            b"loop" => KwLoop,
            _ => return None,
        },
        b'm' => match s {
            b"mut" => KwMut,
            b"match" => KwMatch,
            b"masked" => KwMasked,
            _ => return None,
        },
        b'n' => match s {
            b"needs" => KwNeeds,
            _ => return None,
        },
        b'p' => match s {
            b"pub" => KwPub,
            b"pure" => KwPure,
            _ => return None,
        },
        b'r' => match s {
            b"return" => KwReturn,
            b"requires" => KwRequires,
            b"rec" => KwRec,
            _ => return None,
        },
        b's' => match s {
            b"struct" => KwStruct,
            b"self" => KwSelf,
            b"scalar" => KwScalar,
            b"scope" => KwScope,
            b"spawn" => KwSpawn,
            b"static" => KwStatic,
            b"strict" => KwStrict,
            b"spec" => KwSpec,
            _ => return None,
        },
        b'S' => match s {
            b"Self" => KwSelfType,
            _ => return None,
        },
        b't' => match s {
            b"trait" => KwTrait,
            b"then" => KwThen,
            b"true" => KwTrue,
            b"tail" => KwTail,
            b"type" => KwType,
            b"test" => KwTest,
            _ => return None,
        },
        b'u' => match s {
            b"unsafe" => KwUnsafe,
            _ => return None,
        },
        b'v' => match s {
            b"view" => KwView,
            _ => return None,
        },
        b'w' => match s {
            b"while" => KwWhile,
            b"without" => KwWithout,
            _ => return None,
        },
        _ => return None,
    };
    Some(t as u8)
}

/// Literal kind, for the `what` hole of E0003 / E0015.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum What {
    Str,
    Fmt,
    Char,
}

impl What {
    fn text(self) -> &'static str {
        match self {
            What::Str => "string",
            What::Fmt => "f-string",
            What::Char => "char",
        }
    }
}

/// A lexical error found while scanning one token; turned into a [`Diagnostic`] at the end.
#[derive(Clone, Copy, Debug)]
enum Pending {
    /// E0003: literal `[start, end)` reaches the end of the file.
    Eof { what: What, start: usize, end: usize },
    /// E0015: literal `[start, end)` reaches a raw newline at `end`.
    Newline { what: What, start: usize, end: usize },
    /// E0006: hole `[start, end)` (from its `{`), inner text `[text, text_end)`.
    Hole { start: usize, end: usize, text: usize, text_end: usize },
    /// E0018: the run `[start, end)` of bytes that start no token.
    BadChar { start: usize, end: usize },
    /// E0024: the lifetime `[start, end)` (`'` and an identifier).
    Lifetime { start: usize, end: usize },
    /// E0029: `/*` at `start`.
    BlockComment { start: usize },
}

/// Where the scanner reports lexical errors; [`NoErrs`] when only the token's end is wanted.
trait ErrSink {
    fn push(&mut self, p: Pending);
}

impl ErrSink for Vec<Pending> {
    #[inline]
    fn push(&mut self, p: Pending) {
        Vec::push(self, p);
    }
}

struct NoErrs;

impl ErrSink for NoErrs {
    #[inline]
    fn push(&mut self, _: Pending) {}
}

/// Lexes one file. Never panics; every input gives a token list ending
/// in `EOF`.
pub fn lex(src: &[u8], file: FileId) -> Lexed {
    let mut diagnostics = Vec::new();
    if src.len() > MAX_FILE_LEN {
        let size = src.len().to_string();
        diagnostics.extend(make_diag("E0002", None, &[("size", &size)], Span::new(file, 0, 0), &[]));
        return Lexed {
            tokens: vec![Token { kind: T_EOF, start: 0 }],
            diagnostics,
            stop: Some(LexStop::TooLarge),
        };
    }
    let (limit, bad_len) = match std::str::from_utf8(src) {
        Ok(_) => (src.len(), 0),
        Err(e) => (e.valid_up_to(), e.error_len().unwrap_or(src.len() - e.valid_up_to())),
    };
    let stopped = limit < src.len();
    // past `limit` when the shebang line holds the invalid byte: nothing is lexed then
    let mut i = shebang_end(src);
    if src.starts_with(BOM) {
        diagnostics.extend(make_diag("E0017", None, &[], Span::new(file, 0, 3), &[]));
        i = BOM.len();
    }
    let mut tokens: Vec<Token> = Vec::with_capacity(limit / 3 + 16);
    let mut errs: Vec<Pending> = Vec::new();
    let mut dot_end = usize::MAX;
    while i < limit {
        let c = src[i];
        if class(c) & C_WS != 0 {
            i += 1;
            continue;
        }
        if c == b'/' && at(src, i + 1) == b'/' {
            let e = line_end(src, i + 2);
            if e > limit {
                break; // the comment holds the invalid byte
            }
            i = e;
            continue;
        }
        let mark = errs.len();
        let (kind, end) = scan(src, i, dot_end == i, &mut errs);
        if end > limit {
            errs.truncate(mark); // the token holds the invalid byte: not listed
            if is_bad(c) {
                // a run of bad bytes ends at the invalid byte (which is not a character)
                errs.push(Pending::BadChar { start: i, end: limit });
                tokens.push(Token { kind, start: i as u32 });
            }
            break;
        }
        tokens.push(Token { kind, start: i as u32 });
        dot_end = if kind == T_DOT { end } else { usize::MAX };
        i = end;
    }
    let eof = if stopped { limit } else { src.len() };
    tokens.push(Token { kind: T_EOF, start: eof as u32 });
    for p in errs {
        diagnostics.extend(pending_diag(src, file, p));
    }
    diagnostics.sort_by_key(|d| d.primary().map(|s| s.offset));
    let stop = if stopped {
        let span = Span::new(file, limit as u32, bad_len as u32);
        diagnostics.extend(make_diag("E0001", None, &[], span, &[]));
        Some(LexStop::InvalidUtf8 { offset: limit as u32 })
    } else {
        None
    };
    Lexed { tokens, diagnostics, stop }
}

fn pending_diag(src: &[u8], file: FileId, p: Pending) -> Option<Diagnostic> {
    let span = |s: usize, e: usize| Span::new(file, s as u32, (e - s) as u32);
    match p {
        Pending::Eof { what, start, end } => {
            make_diag("E0003", None, &[("what", what.text())], span(start, end), &[])
        }
        Pending::Newline { what, start, end } => {
            make_diag("E0015", None, &[("what", what.text())], span(start, end), &[])
        }
        Pending::Hole { start, end, text, text_end } => {
            let hole = String::from_utf8_lossy(src.get(text..text_end).unwrap_or_default());
            make_diag("E0006", None, &[("hole", &hole)], span(start, end), &[])
        }
        Pending::BadChar { start, end } => make_diag("E0018", None, &[], span(start, end), &[]),
        Pending::Lifetime { start, end } => {
            let text = String::from_utf8_lossy(src.get(start..end).unwrap_or_default());
            // directly after `&` the lifetime can be removed with nothing else to change
            let head = src.get(..start).unwrap_or_default();
            let before = head.iter().rev().copied().find(|c| !matches!(c, b' ' | b'\t' | b'\n' | b'\r'));
            let case = (before == Some(b'&')).then_some("ref");
            // the edit takes the blanks after the lifetime too: `&'a T` becomes `&T`
            let rest = src.get(end..).unwrap_or_default();
            let blanks = rest.iter().take_while(|c| matches!(c, b' ' | b'\t')).count();
            let cut = [("cut", span(start, end + blanks))];
            let roles: &[(&str, Span)] = if case.is_some() { &cut } else { &[] };
            make_diag("E0024", case, &[("text", &text)], span(start, end), roles)
        }
        Pending::BlockComment { start } => make_diag("E0029", None, &[], span(start, start + 2), &[]),
    }
}

/// Offset of the first `\n` at or after `j`, or the end of `src`.
#[inline]
fn line_end(src: &[u8], j: usize) -> usize {
    match src.get(j..) {
        Some(rest) => rest.iter().position(|&b| b == b'\n').map_or(src.len(), |p| j + p),
        None => src.len(),
    }
}

/// The end of the shebang line of `src`: the offset of the first `\n` (or the end of `src`) when
/// `src` starts with `#!`, else 0. The line is a comment.
#[inline]
pub fn shebang_end(src: &[u8]) -> usize {
    if src.starts_with(b"#!") { line_end(src, 2) } else { 0 }
}

/// The byte offset just after token `i` (the end is recomputed by re-lexing). The
/// `EOF` token, and an index or offset out of range, give the token's own start (or the end of
/// `src`).
///
/// A token never extends past the start of the next one: the run of bad bytes that [`lex`] cut
/// at the first invalid UTF-8 byte re-scans further (the invalid byte is a bad byte too), and
/// ends where the `EOF` token starts.
pub fn token_end(src: &[u8], tokens: &[Token], i: usize) -> u32 {
    let Some(t) = tokens.get(i) else { return src.len().min(MAX_FILE_LEN) as u32 };
    let s = t.start as usize;
    if t.kind == T_EOF || s >= src.len() {
        return t.start;
    }
    let after_dot = i > 0 && tokens[i - 1].kind == T_DOT && tokens[i - 1].start as usize + 1 == s;
    let end = scan(src, s, after_dot, &mut NoErrs).1;
    match tokens.get(i + 1) {
        Some(next) if (next.start as usize) >= s => end.min(next.start as usize) as u32,
        _ => end as u32,
    }
}

/// The source bytes of token `i` (empty for `EOF`).
pub fn token_text<'a>(src: &'a [u8], tokens: &[Token], i: usize) -> &'a [u8] {
    let Some(t) = tokens.get(i) else { return &[] };
    let s = (t.start as usize).min(src.len());
    let e = (token_end(src, tokens, i) as usize).clamp(s, src.len());
    &src[s..e]
}

/// Lexes `text` (no BOM or shebang handling, no UTF-8 check) into `(kind, start)` pairs, without
/// `EOF`. `after_dot`: the text directly follows a `.` token. Used by the printer to check that
/// its output re-lexes to the same tokens.
pub(crate) fn lex_kinds(text: &[u8], after_dot: bool, out: &mut Vec<(u8, usize)>) {
    let mut i = 0;
    let mut dot_end = if after_dot { 0 } else { usize::MAX };
    while i < text.len() {
        let c = text[i];
        if class(c) & C_WS != 0 {
            i += 1;
            continue;
        }
        if c == b'/' && at(text, i + 1) == b'/' {
            i = line_end(text, i + 2);
            continue;
        }
        let (kind, end) = scan(text, i, dot_end == i, &mut NoErrs);
        out.push((kind, i));
        dot_end = if kind == T_DOT { end } else { usize::MAX };
        i = end;
    }
}

/// Scans the token starting at `i` (not whitespace, not a comment). Returns its kind and end.
#[inline(always)]
fn scan<E: ErrSink>(src: &[u8], i: usize, after_dot: bool, errs: &mut E) -> (u8, usize) {
    let c = src[i];
    let cls = class(c);
    if cls & C_DIGIT != 0 {
        return scan_number(src, i, after_dot);
    }
    if cls & C_IDSTART != 0 {
        if c == b'f' && at(src, i + 1) == b'"' {
            return scan_fmt(src, i, errs);
        }
        return scan_word(src, i);
    }
    if cls & C_BAD != 0 {
        let mut j = i + 1;
        while j < src.len() && is_bad(src[j]) {
            j += 1;
        }
        errs.push(Pending::BadChar { start: i, end: j });
        return (KIND_ERROR, j);
    }
    match c {
        b'"' => scan_quoted(src, i, What::Str, errs),
        b'\'' => {
            if is_idstart(at(src, i + 1)) {
                let j = skip_idcont(src, i + 2);
                if at(src, j) != b'\'' {
                    errs.push(Pending::Lifetime { start: i, end: j });
                    return (KIND_ERROR, j);
                }
            }
            scan_quoted(src, i, What::Char, errs)
        }
        b'/' if at(src, i + 1) == b'*' => {
            errs.push(Pending::BlockComment { start: i });
            (KIND_ERROR, block_comment_end(src, i))
        }
        _ => scan_punct(src, i),
    }
}

/// The end of the `/*` error token at `i`: after the first `*/` on its line, else the end of the
/// line.
fn block_comment_end(src: &[u8], i: usize) -> usize {
    let e = line_end(src, i + 2);
    match src[i + 2..e].windows(2).position(|w| w == b"*/") {
        Some(p) => i + 2 + p + 2,
        None => e,
    }
}

#[inline(always)]
fn skip_idcont(src: &[u8], mut j: usize) -> usize {
    while j < src.len() && is_idcont(src[j]) {
        j += 1;
    }
    j
}

#[inline(always)]
fn skip_digits(src: &[u8], mut j: usize) -> usize {
    while j < src.len() && (is_digit(src[j]) || src[j] == b'_') {
        j += 1;
    }
    j
}

#[inline]
fn scan_word(src: &[u8], i: usize) -> (u8, usize) {
    let j = skip_idcont(src, i + 1);
    let s = &src[i..j];
    if s.len() == 1 && s[0] == b'_' {
        return (T_UNDERSCORE, j);
    }
    if s == b"as" {
        let op = match at(src, j) {
            b'%' => Some(Term::AsPercent),
            b'|' => Some(Term::AsPipe),
            b'?' => Some(Term::AsQuestion),
            b'~' => Some(Term::AsTilde),
            _ => None,
        };
        if let Some(op) = op {
            return (op as u8, j + 1);
        }
    }
    (keyword(s).unwrap_or(T_IDENT), j)
}

#[inline]
fn scan_number(src: &[u8], i: usize, after_dot: bool) -> (u8, usize) {
    if after_dot {
        return (T_INT, skip_idcont(src, i + 1));
    }
    if src[i] == b'0' && matches!(at(src, i + 1), b'x' | b'b' | b'o') {
        return (T_INT, skip_idcont(src, i + 2));
    }
    let mut j = skip_digits(src, i + 1);
    let mut float = false;
    if at(src, j) == b'.' && is_digit(at(src, j + 1)) {
        float = true;
        j = skip_digits(src, j + 2);
    }
    if matches!(at(src, j), b'e' | b'E') {
        if is_digit(at(src, j + 1)) {
            float = true;
            j = skip_digits(src, j + 2);
        } else if matches!(at(src, j + 1), b'+' | b'-') && is_digit(at(src, j + 2)) {
            float = true;
            j = skip_digits(src, j + 3);
        }
    }
    let j = skip_idcont(src, j);
    (if float { T_FLOAT } else { T_INT }, j)
}

/// `"…"` or `'…'` starting at `i`.
#[inline]
fn scan_quoted<E: ErrSink>(src: &[u8], i: usize, what: What, errs: &mut E) -> (u8, usize) {
    let (quote, ok) = if what == What::Char { (b'\'', T_CHAR) } else { (b'"', T_STR) };
    let n = src.len();
    let mut j = i + 1;
    loop {
        if j >= n {
            errs.push(Pending::Eof { what, start: i, end: n });
            return (KIND_ERROR, n);
        }
        let b = src[j];
        if b == quote {
            return (ok, j + 1);
        }
        if b == b'\n' {
            errs.push(Pending::Newline { what, start: i, end: j });
            return (KIND_ERROR, j);
        }
        j += if b == b'\\' && j + 1 < n && src[j + 1] != b'\n' { 2 } else { 1 };
    }
}

/// `f"…"` starting at `i` (the `f`).
fn scan_fmt<E: ErrSink>(src: &[u8], i: usize, errs: &mut E) -> (u8, usize) {
    let n = src.len();
    let mut j = i + 2;
    let mut bad = false;
    loop {
        if j >= n {
            errs.push(Pending::Eof { what: What::Fmt, start: i, end: n });
            return (KIND_ERROR, n);
        }
        match src[j] {
            b'"' => return (if bad { KIND_ERROR } else { T_FMT }, j + 1),
            b'\n' => {
                errs.push(Pending::Newline { what: What::Fmt, start: i, end: j });
                return (KIND_ERROR, j);
            }
            b'\\' => j += if j + 1 < n && src[j + 1] != b'\n' { 2 } else { 1 },
            b'{' => match valid_hole(src, j) {
                Some(after) => j = after,
                None => {
                    bad = true;
                    j = bad_hole(src, j, errs);
                }
            },
            _ => j += 1,
        }
    }
}

/// A hole `{` at `h`: `(Ident | self) ('.' Ident)* '}'`. Returns the offset after the `}`.
fn valid_hole(src: &[u8], h: usize) -> Option<usize> {
    let mut k = h + 1;
    let mut head = true;
    loop {
        if !is_idstart(at(src, k)) {
            return None;
        }
        let s = k;
        k = skip_idcont(src, k + 1);
        let seg = &src[s..k];
        let ok = if seg == b"_" {
            false
        } else {
            match keyword(seg) {
                None => true,
                Some(kw) => head && kw == Term::KwSelf as u8,
            }
        };
        if !ok {
            return None;
        }
        head = false;
        match at(src, k) {
            b'.' => k += 1,
            b'}' => return Some(k + 1),
            _ => return None,
        }
    }
}

/// Reports the bad hole at `h` (E0006) and returns where the literal scan resumes: after the
/// hole's `}`, or at the `"` / newline / end of file that ends the hole first.
fn bad_hole<E: ErrSink>(src: &[u8], h: usize, errs: &mut E) -> usize {
    let n = src.len();
    let mut e = h + 1;
    let (text_end, resume) = loop {
        if e >= n {
            break (n, n);
        }
        match src[e] {
            b'}' => break (e, e + 1),
            b'"' | b'\n' => break (e, e),
            b'\\' if e + 1 < n && src[e + 1] != b'\n' => e += 2,
            _ => e += 1,
        }
    };
    let text_end = text_end.min(n);
    errs.push(Pending::Hole { start: h, end: resume, text: h + 1, text_end });
    resume
}

/// Punctuation (longest match) at `i` (a byte of class [`C_BAD`] never reaches here).
#[inline]
fn scan_punct(src: &[u8], i: usize) -> (u8, usize) {
    use Term::*;
    let b1 = at(src, i + 1);
    let b2 = at(src, i + 2);
    let (t, len) = match src[i] {
        b'!' => {
            if b1 == b'=' {
                (BangEq, 2)
            } else {
                (Bang, 1)
            }
        }
        b'%' => {
            if b1 == b'=' {
                (PercentEq, 2)
            } else {
                (Percent, 1)
            }
        }
        b'&' => match b1 {
            b'&' => (AmpAmp, 2),
            b'=' => (AmpEq, 2),
            _ => (Amp, 1),
        },
        b'(' => (LParen, 1),
        b')' => (RParen, 1),
        b',' => (Comma, 1),
        b';' => (Semi, 1),
        b'?' => (Question, 1),
        b'[' => (LBracket, 1),
        b']' => (RBracket, 1),
        b'{' => (LBrace, 1),
        b'}' => (RBrace, 1),
        b'~' => (Tilde, 1),
        b'*' => match (b1, b2) {
            (b'%', b'=') => (StarPercentEq, 3),
            (b'%', _) => (StarPercent, 2),
            (b'=', _) => (StarEq, 2),
            (b'|', b'=') => (StarPipeEq, 3),
            (b'|', _) => (StarPipe, 2),
            _ => (Star, 1),
        },
        b'+' => match (b1, b2) {
            (b'%', b'=') => (PlusPercentEq, 3),
            (b'%', _) => (PlusPercent, 2),
            (b'=', _) => (PlusEq, 2),
            (b'|', b'=') => (PlusPipeEq, 3),
            (b'|', _) => (PlusPipe, 2),
            _ => (Plus, 1),
        },
        b'-' => match (b1, b2) {
            (b'%', b'=') => (MinusPercentEq, 3),
            (b'%', _) => (MinusPercent, 2),
            (b'=', _) => (MinusEq, 2),
            (b'>', _) => (MinusGt, 2),
            (b'|', b'=') => (MinusPipeEq, 3),
            (b'|', _) => (MinusPipe, 2),
            _ => (Minus, 1),
        },
        b'.' => match (b1, b2) {
            (b'.', b'.') => (DotDotDot, 3),
            (b'.', b'=') => (DotDotEq, 3),
            (b'.', _) => (DotDot, 2),
            (b'{', _) => (DotLBrace, 2),
            _ => (Dot, 1),
        },
        b'/' => {
            if b1 == b'=' {
                (SlashEq, 2)
            } else {
                (Slash, 1)
            }
        }
        b':' => {
            if b1 == b':' {
                (ColonColon, 2)
            } else {
                (Colon, 1)
            }
        }
        b'<' => match (b1, b2) {
            (b'<', b'%') => {
                if at(src, i + 3) == b'=' {
                    (LtLtPercentEq, 4)
                } else {
                    (LtLtPercent, 3)
                }
            }
            (b'<', b'=') => (LtLtEq, 3),
            (b'<', _) => (LtLt, 2),
            (b'=', _) => (LtEq, 2),
            _ => (Lt, 1),
        },
        b'=' => match b1 {
            b'=' => (EqEq, 2),
            b'>' => (EqGt, 2),
            _ => (Eq, 1),
        },
        b'>' => match (b1, b2) {
            (b'>', b'=') => (GtGtEq, 3),
            (b'>', _) => (GtGt, 2),
            (b'=', _) => (GtEq, 2),
            _ => (Gt, 1),
        },
        b'^' => {
            if b1 == b'=' {
                (CaretEq, 2)
            } else {
                (Caret, 1)
            }
        }
        b'|' => match b1 {
            b'=' => (PipeEq, 2),
            b'|' => (PipePipe, 2),
            _ => (Pipe, 1),
        },
        _ => return (KIND_ERROR, i + 1),
    };
    (t as u8, i + len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated::terminals::TERM_IS_CLASS;

    #[test]
    fn keywords_match_the_generated_terminals() {
        for (id, text) in TERM_TEXT.iter().enumerate() {
            let word = text.bytes().all(|b| b.is_ascii_alphabetic()) && !TERM_IS_CLASS[id];
            if word {
                assert_eq!(keyword(text.as_bytes()), Some(id as u8), "{text}");
            } else {
                assert_eq!(keyword(text.as_bytes()), None, "{text}");
            }
        }
    }

    #[test]
    fn every_punctuation_terminal_lexes_as_itself() {
        for (id, text) in TERM_TEXT.iter().enumerate() {
            if TERM_IS_CLASS[id] || text.bytes().all(is_idcont) {
                continue;
            }
            let (k, end) = scan(text.as_bytes(), 0, false, &mut NoErrs);
            assert_eq!((k, end), (id as u8, text.len()), "{text}");
        }
    }

    #[test]
    fn bad_run_cut_at_invalid_utf8_does_not_overlap_eof() {
        // `é` (valid, a bad byte run) then 0xFF (the first invalid byte, offset 4).
        let src = b"a \xc3\xa9\xff b";
        let l = lex(src, FileId(0));
        let starts: Vec<(u8, u32)> = l.tokens.iter().map(|t| (t.kind, t.start)).collect();
        assert_eq!(starts, [(Term::Ident as u8, 0), (KIND_ERROR, 2), (T_EOF, 4)]);
        assert_eq!(l.stop, Some(LexStop::InvalidUtf8 { offset: 4 }));
        assert_eq!(token_end(src, &l.tokens, 1), 4);
        assert_eq!(token_text(src, &l.tokens, 1), b"\xc3\xa9");
        for i in 0..l.tokens.len() {
            let e = token_end(src, &l.tokens, i);
            assert!(e >= l.tokens[i].start && e as usize <= src.len(), "token {i}");
            if let Some(next) = l.tokens.get(i + 1) {
                assert!(e <= next.start, "tokens {i} and {} overlap", i + 1);
            }
        }
        let bad: Vec<(&str, u32, u32)> = l
            .diagnostics
            .iter()
            .map(|d| (d.code(), d.primary().map_or(0, |s| s.offset), d.primary().map_or(0, |s| s.len)))
            .collect();
        assert_eq!(bad, [("E0018", 2, 2), ("E0001", 4, 1)]);
    }
}
