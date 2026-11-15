//! A small TOML subset parser for `stave.toml`, keeping byte spans of keys and values (E1002 is
//! located in the manifest).
//!
//! Hand-written instead of the `toml` + `serde` crates: the workspace keeps those out of the
//! compiler binary, and E1002 needs the span of every key and value, which serde does not give
//! for keys. The subset is what manifests use:
//! - comments `#`, bare keys `[A-Za-z0-9_-]+`, quoted keys, dotted keys (`a.b = 1`);
//! - tables `[a.b]` (implicitly creating parents), inline tables `{ k = v, ... }` (one line,
//!   trailing comma accepted);
//! - values: basic strings `"…"` (escapes `\" \\ \n \t \r \b \f \e \uXXXX \UXXXXXXXX`), literal
//!   strings `'…'`, integers (decimal with `_`, `0x`/`0o`/`0b`), floats, `true`/`false`, arrays
//!   (may span lines).
//!
//! Not supported (a syntax error): multi-line strings, dates and times, arrays of tables
//! `[[x]]`. Redefining a key (or a table header) is an error too, marked
//! [`TomlError::duplicate`] (E1002 `duplicate-key`).

/// A byte range `[lo, hi)` of the manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TSpan {
    pub lo: u32,
    pub hi: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TValue {
    Str(String, TSpan),
    Int(i64, TSpan),
    Float(TSpan),
    Bool(bool, TSpan),
    Array(Vec<TValue>, TSpan),
    Table(TTable),
}

impl TValue {
    pub fn span(&self) -> TSpan {
        match self {
            TValue::Str(_, s)
            | TValue::Int(_, s)
            | TValue::Float(s)
            | TValue::Bool(_, s)
            | TValue::Array(_, s) => *s,
            TValue::Table(t) => t.span,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TEntry {
    pub key: String,
    pub key_span: TSpan,
    pub value: TValue,
}

/// How a table came to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    Root,
    /// `[a.b]` header (span: the header's last key).
    Header,
    /// Created as the parent of a header or a dotted key.
    Implicit,
    Inline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TTable {
    pub entries: Vec<TEntry>,
    pub span: TSpan,
    pub kind: TableKind,
}

impl TTable {
    fn new(kind: TableKind, span: TSpan) -> TTable {
        TTable { entries: Vec::new(), span, kind }
    }

    pub fn get(&self, key: &str) -> Option<&TEntry> {
        self.entries.iter().find(|e| e.key == key)
    }
}

/// A syntax error at `offset`, or (`duplicate`) a key defined a second time, at that key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TomlError {
    pub offset: u32,
    pub duplicate: bool,
}

/// Maximum nesting of arrays / inline tables (keeps the recursive value parser bounded).
const MAX_DEPTH: usize = 64;

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

type R<T> = Result<T, TomlError>;

impl<'a> P<'a> {
    fn err<T>(&self) -> R<T> {
        Err(TomlError { offset: self.i.min(u32::MAX as usize) as u32, duplicate: false })
    }

    fn peek(&self) -> u8 {
        self.s.get(self.i).copied().unwrap_or(0)
    }

    fn at_end(&self) -> bool {
        self.i >= self.s.len()
    }

    fn span(&self, lo: usize) -> TSpan {
        TSpan { lo: lo.min(u32::MAX as usize) as u32, hi: self.i.min(u32::MAX as usize) as u32 }
    }

    /// Spaces and tabs.
    fn ws(&mut self) {
        while matches!(self.peek(), b' ' | b'\t') {
            self.i += 1;
        }
    }

    fn comment(&mut self) {
        if self.peek() == b'#' {
            while !self.at_end() && self.peek() != b'\n' {
                self.i += 1;
            }
        }
    }

    /// Whitespace, newlines, and comments (inside arrays).
    fn ws_nl(&mut self) {
        loop {
            self.ws();
            self.comment();
            match self.peek() {
                b'\n' => self.i += 1,
                b'\r' if self.s.get(self.i + 1) == Some(&b'\n') => self.i += 2,
                _ => return,
            }
        }
    }

    /// End of a line: optional whitespace and comment, then newline or end of input.
    fn eol(&mut self) -> R<()> {
        self.ws();
        self.comment();
        match self.peek() {
            b'\n' => {
                self.i += 1;
                Ok(())
            }
            b'\r' if self.s.get(self.i + 1) == Some(&b'\n') => {
                self.i += 2;
                Ok(())
            }
            _ if self.at_end() => Ok(()),
            _ => self.err(),
        }
    }

    fn simple_key(&mut self) -> R<(String, TSpan)> {
        let lo = self.i;
        match self.peek() {
            b'"' => {
                let s = self.basic_string()?;
                Ok((s, self.span(lo)))
            }
            b'\'' => {
                let s = self.literal_string()?;
                Ok((s, self.span(lo)))
            }
            _ => {
                while matches!(self.peek(), b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-') {
                    self.i += 1;
                }
                if self.i == lo {
                    return self.err();
                }
                let k = String::from_utf8_lossy(&self.s[lo..self.i]).into_owned();
                Ok((k, self.span(lo)))
            }
        }
    }

    fn key(&mut self) -> R<Vec<(String, TSpan)>> {
        let mut parts = vec![self.simple_key()?];
        loop {
            let save = self.i;
            self.ws();
            if self.peek() == b'.' {
                self.i += 1;
                self.ws();
                parts.push(self.simple_key()?);
            } else {
                self.i = save;
                return Ok(parts);
            }
        }
    }

    fn hex_char(&mut self, digits: usize) -> R<char> {
        let mut v: u32 = 0;
        for _ in 0..digits {
            let d = match self.peek() {
                c @ b'0'..=b'9' => c - b'0',
                c @ b'a'..=b'f' => c - b'a' + 10,
                c @ b'A'..=b'F' => c - b'A' + 10,
                _ => return self.err(),
            };
            v = v * 16 + d as u32;
            self.i += 1;
        }
        match char::from_u32(v) {
            Some(c) => Ok(c),
            None => self.err(),
        }
    }

    fn basic_string(&mut self) -> R<String> {
        self.i += 1; // opening quote
        if self.peek() == b'"' && self.s.get(self.i + 1) == Some(&b'"') {
            return self.err(); // multi-line strings are not supported
        }
        let mut out: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                b'"' => {
                    self.i += 1;
                    break;
                }
                b'\\' => {
                    self.i += 1;
                    let c = self.peek();
                    self.i += 1;
                    let ch = match c {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'e' => '\u{1b}',
                        b'u' => self.hex_char(4)?,
                        b'U' => self.hex_char(8)?,
                        _ => {
                            self.i -= 1;
                            return self.err();
                        }
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                b'\n' | b'\r' => return self.err(),
                _ if self.at_end() => return self.err(),
                c => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
        match String::from_utf8(out) {
            Ok(s) => Ok(s),
            Err(_) => self.err(),
        }
    }

    fn literal_string(&mut self) -> R<String> {
        self.i += 1;
        let lo = self.i;
        loop {
            match self.peek() {
                b'\'' => break,
                b'\n' | b'\r' => return self.err(),
                _ if self.at_end() => return self.err(),
                _ => self.i += 1,
            }
        }
        let s = &self.s[lo..self.i];
        self.i += 1;
        match std::str::from_utf8(s) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => self.err(),
        }
    }

    fn number(&mut self) -> R<TValue> {
        let lo = self.i;
        while matches!(self.peek(), b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'+' | b'-' | b'.') {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[lo..self.i]).unwrap_or("");
        let span = self.span(lo);
        if let Some(v) = parse_int(text) {
            return Ok(TValue::Int(v, span));
        }
        if is_float(text) {
            return Ok(TValue::Float(span));
        }
        self.i = lo;
        self.err()
    }

    fn value(&mut self, depth: usize) -> R<TValue> {
        if depth > MAX_DEPTH {
            return self.err();
        }
        let lo = self.i;
        match self.peek() {
            b'"' => {
                let s = self.basic_string()?;
                Ok(TValue::Str(s, self.span(lo)))
            }
            b'\'' => {
                let s = self.literal_string()?;
                Ok(TValue::Str(s, self.span(lo)))
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws_nl();
                    if self.peek() == b']' {
                        self.i += 1;
                        break;
                    }
                    items.push(self.value(depth + 1)?);
                    self.ws_nl();
                    match self.peek() {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err(),
                    }
                }
                Ok(TValue::Array(items, self.span(lo)))
            }
            b'{' => {
                self.i += 1;
                let mut t = TTable::new(TableKind::Inline, self.span(lo));
                loop {
                    self.ws();
                    if self.peek() == b'}' {
                        self.i += 1;
                        break;
                    }
                    let key = self.key()?;
                    self.ws();
                    if self.peek() != b'=' {
                        return self.err();
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    insert_dotted(&mut t, key, v).map_err(|o| TomlError { offset: o, duplicate: true })?;
                    self.ws();
                    match self.peek() {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err(),
                    }
                }
                t.span = self.span(lo);
                Ok(TValue::Table(t))
            }
            b't' if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(TValue::Bool(true, self.span(lo)))
            }
            b'f' if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(TValue::Bool(false, self.span(lo)))
            }
            b'0'..=b'9' | b'+' | b'-' | b'i' | b'n' => self.number(),
            _ => self.err(),
        }
    }
}

fn parse_int(t: &str) -> Option<i64> {
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, digits) = if let Some(d) = body.strip_prefix("0x") {
        (16, d)
    } else if let Some(d) = body.strip_prefix("0o") {
        (8, d)
    } else if let Some(d) = body.strip_prefix("0b") {
        (2, d)
    } else {
        if body.len() > 1 && body.starts_with('0') {
            return None;
        }
        (10, body)
    };
    if radix != 10 && t.len() != body.len() {
        return None; // no sign on prefixed integers
    }
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    let v = i64::from_str_radix(&clean, radix).ok()?;
    Some(if neg { -v } else { v })
}

fn is_float(t: &str) -> bool {
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    if body == "inf" || body == "nan" {
        return true;
    }
    let clean: String = body.chars().filter(|&c| c != '_').collect();
    (clean.contains('.') || clean.contains(['e', 'E']))
        && clean.bytes().next().is_some_and(|b| b.is_ascii_digit())
        && clean.parse::<f64>().is_ok()
}

/// Inserts `v` under the dotted `key` into `t`. `Err(offset)` on a redefinition.
fn insert_dotted(t: &mut TTable, key: Vec<(String, TSpan)>, v: TValue) -> Result<(), u32> {
    let n = key.len();
    let mut cur = t;
    for (k, (name, span)) in key.into_iter().enumerate() {
        if k + 1 == n {
            if cur.get(&name).is_some() {
                return Err(span.lo);
            }
            cur.entries.push(TEntry { key: name, key_span: span, value: v });
            return Ok(());
        }
        let pos = match cur.entries.iter().position(|e| e.key == name) {
            Some(p) => p,
            None => {
                cur.entries.push(TEntry {
                    key: name,
                    key_span: span,
                    value: TValue::Table(TTable::new(TableKind::Implicit, span)),
                });
                cur.entries.len() - 1
            }
        };
        cur = match &mut cur.entries[pos].value {
            TValue::Table(tt) if tt.kind != TableKind::Inline => tt,
            _ => return Err(span.lo),
        };
    }
    Ok(())
}

/// The table at `path` below `root`, created as needed (`[a.b]`). `Err(offset)` if a path
/// element is not a table or the table was already defined by a header.
fn header_table(root: &mut TTable, path: Vec<(String, TSpan)>) -> Result<&mut TTable, u32> {
    let n = path.len();
    let mut cur = root;
    for (k, (name, span)) in path.into_iter().enumerate() {
        let last = k + 1 == n;
        let pos = match cur.entries.iter().position(|e| e.key == name) {
            Some(p) => {
                match &mut cur.entries[p].value {
                    TValue::Table(tt) if tt.kind == TableKind::Implicit && last => {
                        tt.kind = TableKind::Header;
                        tt.span = span;
                    }
                    TValue::Table(tt) if tt.kind != TableKind::Inline && !last => {}
                    _ => return Err(span.lo),
                }
                p
            }
            None => {
                let kind = if last { TableKind::Header } else { TableKind::Implicit };
                cur.entries.push(TEntry {
                    key: name,
                    key_span: span,
                    value: TValue::Table(TTable::new(kind, span)),
                });
                cur.entries.len() - 1
            }
        };
        cur = match &mut cur.entries[pos].value {
            TValue::Table(tt) => tt,
            _ => return Err(span.lo),
        };
    }
    Ok(cur)
}

/// Parses a manifest. On a syntax error, everything before the error line is still returned
/// together with the error.
pub fn parse(text: &[u8]) -> (TTable, Option<TomlError>) {
    let mut root = TTable::new(TableKind::Root, TSpan { lo: 0, hi: 0 });
    let mut p = P { s: text, i: 0 };
    let mut current: Vec<(String, TSpan)> = Vec::new();
    loop {
        p.ws_nl();
        if p.at_end() {
            return (root, None);
        }
        let r: R<()> = (|| {
            if p.peek() == b'[' {
                p.i += 1;
                if p.peek() == b'[' {
                    return p.err(); // arrays of tables are not supported
                }
                p.ws();
                let path = p.key()?;
                p.ws();
                if p.peek() != b']' {
                    return p.err();
                }
                p.i += 1;
                p.eol()?;
                header_table(&mut root, path.clone())
                    .map_err(|o| TomlError { offset: o, duplicate: true })?;
                current = path;
                Ok(())
            } else {
                let key = p.key()?;
                p.ws();
                if p.peek() != b'=' {
                    return p.err();
                }
                p.i += 1;
                p.ws();
                let v = p.value(0)?;
                p.eol()?;
                let lo = key.first().map_or(0, |k| k.1.lo);
                let t = if current.is_empty() {
                    &mut root
                } else {
                    header_table_existing(&mut root, &current)
                        .ok_or(TomlError { offset: lo, duplicate: false })?
                };
                insert_dotted(t, key, v).map_err(|o| TomlError { offset: o, duplicate: true })
            }
        })();
        if let Err(e) = r {
            return (root, Some(e));
        }
    }
}

fn header_table_existing<'t>(root: &'t mut TTable, path: &[(String, TSpan)]) -> Option<&'t mut TTable> {
    let mut cur = root;
    for (name, _) in path {
        let pos = cur.entries.iter().position(|e| &e.key == name)?;
        cur = match &mut cur.entries[pos].value {
            TValue::Table(t) => t,
            _ => return None,
        };
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> TTable {
        let (t, e) = parse(s.as_bytes());
        assert_eq!(e, None, "{s}");
        t
    }

    #[test]
    fn manifest_shapes() {
        let t = ok("name = \"app\" # c\nkind = 'bin'\n\n[dependencies]\nprims = { path = \"prims/cpu\" }\n\
                    [target.\"aarch64-linux+sve\".dependencies]\nx = { path = \"x\", }\n[run]\nstack = \"8MiB\"\n\
                    n = 1_000\nf = 1.5\nb = true\na = [1, 2,\n 3]\n");
        assert_eq!(
            t.get("name").map(|e| &e.value),
            Some(&TValue::Str("app".into(), TSpan { lo: 7, hi: 12 }))
        );
        let deps = match &t.get("dependencies").unwrap().value {
            TValue::Table(d) => d,
            _ => panic!(),
        };
        assert!(matches!(&deps.get("prims").unwrap().value, TValue::Table(i) if i.kind == TableKind::Inline));
        let target = match &t.get("target").unwrap().value {
            TValue::Table(d) => d,
            _ => panic!(),
        };
        let e = target.get("aarch64-linux+sve").unwrap();
        let src = "[target.\"aarch64-linux+sve\"";
        assert_eq!(e.key_span.hi - e.key_span.lo, (src.len() - "[target.".len()) as u32);
        let run = match &t.get("run").unwrap().value {
            TValue::Table(d) => d,
            _ => panic!(),
        };
        assert!(matches!(run.get("n").unwrap().value, TValue::Int(1000, _)));
        assert!(matches!(run.get("f").unwrap().value, TValue::Float(_)));
        assert!(matches!(run.get("a").unwrap().value, TValue::Array(ref v, _) if v.len() == 3));
    }

    #[test]
    fn errors_are_located() {
        for (s, at, duplicate) in [
            ("name = \n", 7, false),
            ("a = 1\na = 2\n", 6, true),
            ("[d]\nx = 1\n[d]\n", 11, true),
            ("t = { a = 1, a = 2 }\n", 13, true),
            ("[[x]]\n", 1, false),
            ("x = \"open\n", 9, false),
            ("x = 1 y\n", 6, false),
            ("x = 09\n", 4, false),
            ("x = \"\"\"a\"\"\"\n", 5, false),
        ] {
            let (_, e) = parse(s.as_bytes());
            assert_eq!(e, Some(TomlError { offset: at, duplicate }), "{s:?}");
        }
    }

    #[test]
    fn nesting_is_bounded() {
        let s = format!("x = {}{}\n", "[".repeat(100), "]".repeat(100));
        let (_, e) = parse(s.as_bytes());
        assert_eq!(e, Some(TomlError { offset: 4 + 65, duplicate: false }));
    }

    #[test]
    fn never_panics_on_bytes() {
        let samples: [&[u8]; 6] = [b"\xff\xfe = 1", b"[", b"a.", b"a = {", b"a = \"\\u12\"", b"a = 'x"];
        for s in samples {
            let _ = parse(s);
        }
    }
}
