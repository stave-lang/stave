//! Annotation scanner (contracts/conformance-annotations.md §Syntax).
//!
//! An annotation is a line comment `//~ KEYWORD args` in a `.stv` file, or `#~ KEYWORD args` in a
//! `stave.toml`. `//~^` (`#~^`) shifts a located annotation up one line per `^`. The scanner finds
//! the comment with a small per-line lexer (string and char literals cannot span lines in either
//! language), so `//~` inside a string literal is not an annotation, which is why `//~^` exists.

use std::fmt;

use stave_target::Triple;

/// Diagnostic severity as the runner sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    pub fn name(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A diagnostic code `EXXXX`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Code(pub u16);

impl Code {
    /// `E` followed by exactly four ASCII digits.
    pub fn parse(s: &str) -> Option<Code> {
        let digits = s.strip_prefix('E')?;
        if digits.len() != 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        digits.parse().ok().map(Code)
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:04}", self.0)
    }
}

/// A run or build directive (the non-diagnostic annotations).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Directive {
    /// `STDOUT <file>`: path as written, relative to the annotated file's directory.
    Stdout(String),
    /// `STDERR-CONTAINS "<text>"`, escapes resolved.
    StderrContains(String),
    /// `EXIT <n>`, 0..=255.
    Exit(u8),
    Target(Triple),
    /// `ARGS <flags>`, split on whitespace.
    Args(Vec<String>),
    CheckOnly,
    /// Lex and parse only (no semantic stages).
    ParseOnly,
    ResourceLimit,
    /// `INTERP-REPORT <class>`: the run in the interpreter's checking mode ends with a report of
    /// this class (a program with unsafe of its own).
    InterpReport(String),
}

impl Directive {
    pub fn keyword(&self) -> &'static str {
        match self {
            Directive::Stdout(_) => "STDOUT",
            Directive::StderrContains(_) => "STDERR-CONTAINS",
            Directive::Exit(_) => "EXIT",
            Directive::Target(_) => "TARGET",
            Directive::Args(_) => "ARGS",
            Directive::CheckOnly => "CHECK-ONLY",
            Directive::ParseOnly => "PARSE-ONLY",
            Directive::ResourceLimit => "RESOURCE-LIMIT",
            Directive::InterpReport(_) => "INTERP-REPORT",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Annot {
    /// An expected diagnostic. `line` is the (shifted) 1-based line of its primary span; `None`
    /// for `NOLOC`.
    Diag {
        severity: Severity,
        code: Code,
        line: Option<u32>,
    },
    Directive(Directive),
}

/// One annotation and the line its comment is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnotAt {
    pub comment_line: u32,
    pub annot: Annot,
}

/// A malformed annotation (the test fails as malformed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnotError {
    pub line: u32,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Syntax {
    /// `.stv`: `//` comments, `"…"` strings (also `f"…"`), `'…'` chars, `\` escapes.
    Stave,
    /// `stave.toml`: `#` comments, `"…"` basic strings with escapes, `'…'` literal strings.
    Toml,
}

/// The text after the comment marker of `line`'s first comment, if any.
fn comment_body(line: &str, syntax: Syntax) -> Option<&str> {
    #[derive(PartialEq)]
    enum St {
        Code,
        Str,
        Char,
    }
    let b = line.as_bytes();
    let mut st = St::Code;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        match st {
            St::Code => match (syntax, c) {
                (Syntax::Stave, b'/') if b.get(i + 1) == Some(&b'/') => return line.get(i + 2..),
                (Syntax::Toml, b'#') => return line.get(i + 1..),
                (_, b'"') => st = St::Str,
                (_, b'\'') => st = St::Char,
                _ => {}
            },
            St::Str => match c {
                b'\\' => i += 1,
                b'"' => st = St::Code,
                _ => {}
            },
            St::Char => match c {
                // TOML literal strings have no escapes; Stave chars do.
                b'\\' if syntax == Syntax::Stave => i += 1,
                b'\'' => st = St::Code,
                _ => {}
            },
        }
        i += 1;
    }
    None
}

/// Scan `text` for annotations. Line numbers are 1-based; lines end at `\n` (a trailing `\r` is
/// dropped).
pub fn scan(text: &str, syntax: Syntax) -> (Vec<AnnotAt>, Vec<AnnotError>) {
    let mut out = Vec::new();
    let mut errs = Vec::new();
    for (idx, raw) in text.split('\n').enumerate() {
        let line_no = u32::try_from(idx + 1).unwrap_or(u32::MAX);
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let Some(body) = comment_body(line, syntax) else { continue };
        let Some(rest) = body.strip_prefix('~') else { continue };
        match parse_annotation(rest, line_no, syntax) {
            Ok(annot) => out.push(AnnotAt { comment_line: line_no, annot }),
            Err(message) => errs.push(AnnotError { line: line_no, message }),
        }
    }
    (out, errs)
}

/// Parse what follows `//~` / `#~`.
fn parse_annotation(rest: &str, line_no: u32, syntax: Syntax) -> Result<Annot, String> {
    let carets = rest.bytes().take_while(|&b| b == b'^').count();
    let after = rest.get(carets..).unwrap_or("");
    let Some(body) = after.strip_prefix(' ') else {
        return Err("an annotation is `//~ KEYWORD ...` (one space after `~` or `^`)".to_string());
    };
    let body = body.trim_end();
    let (kw, args) = match body.split_once(' ') {
        Some((k, a)) => (k, a.trim_start()),
        None => (body, ""),
    };
    let shifted = || -> Result<Option<u32>, String> {
        let shift = u32::try_from(carets).unwrap_or(u32::MAX);
        match line_no.checked_sub(shift) {
            Some(l) if l >= 1 => Ok(Some(l)),
            _ => Err(format!("`{}` shifts above line 1", "^".repeat(carets))),
        }
    };
    let no_shift = |what: &str| -> Result<(), String> {
        if carets > 0 { Err(format!("`^` applies only to located annotations, not {what}")) } else { Ok(()) }
    };
    let annot = match kw {
        "ERROR" | "WARNING" => {
            let severity = if kw == "ERROR" { Severity::Error } else { Severity::Warning };
            Annot::Diag { severity, code: one_code(args)?, line: shifted()? }
        }
        "NOLOC" => {
            no_shift("NOLOC")?;
            let (sev, code) = args.split_once(' ').unwrap_or((args, ""));
            let severity = match sev {
                "ERROR" => Severity::Error,
                "WARNING" => Severity::Warning,
                _ => return Err(format!("NOLOC takes ERROR or WARNING, found `{sev}`")),
            };
            Annot::Diag { severity, code: one_code(code.trim_start())?, line: None }
        }
        "STDOUT" | "STDERR-CONTAINS" | "EXIT" | "TARGET" | "ARGS" | "CHECK-ONLY" | "PARSE-ONLY"
        | "RESOURCE-LIMIT" | "INTERP-REPORT" => {
            no_shift(kw)?;
            Annot::Directive(parse_directive(kw, args)?)
        }
        "" => return Err("missing annotation keyword".to_string()),
        other => return Err(format!("unknown annotation `{other}`")),
    };
    if syntax == Syntax::Toml {
        match annot {
            Annot::Diag { line: Some(_), .. } => {}
            _ => return Err("stave.toml takes only `#~ ERROR EXXXX` and `#~ WARNING EXXXX`".to_string()),
        }
    }
    Ok(annot)
}

fn one_code(args: &str) -> Result<Code, String> {
    let mut it = args.split_whitespace();
    let (Some(c), None) = (it.next(), it.next()) else {
        return Err(format!("expected one diagnostic code `EXXXX`, found `{args}`"));
    };
    Code::parse(c).ok_or_else(|| format!("`{c}` is not a diagnostic code `EXXXX`"))
}

fn no_args(kw: &str, args: &str) -> Result<(), String> {
    if args.is_empty() { Ok(()) } else { Err(format!("{kw} takes no arguments, found `{args}`")) }
}

fn parse_directive(kw: &str, args: &str) -> Result<Directive, String> {
    Ok(match kw {
        "STDOUT" => {
            if args.is_empty() || args.contains(char::is_whitespace) {
                return Err(format!("STDOUT takes one file path, found `{args}`"));
            }
            if args.starts_with('/') {
                return Err(format!("STDOUT path `{args}` must be relative to the test file's directory"));
            }
            Directive::Stdout(args.to_string())
        }
        "STDERR-CONTAINS" => Directive::StderrContains(quoted(args)?),
        "EXIT" => {
            let ok = !args.is_empty() && args.bytes().all(|b| b.is_ascii_digit());
            match args.parse::<u8>() {
                Ok(n) if ok => Directive::Exit(n),
                _ => return Err(format!("EXIT takes a decimal exit status 0..=255, found `{args}`")),
            }
        }
        "TARGET" => Directive::Target(Triple::parse(args).map_err(|e| format!("TARGET: {e}"))?),
        "ARGS" => {
            let flags: Vec<String> = args.split_whitespace().map(str::to_string).collect();
            if flags.is_empty() {
                return Err("ARGS takes at least one flag".to_string());
            }
            Directive::Args(flags)
        }
        "CHECK-ONLY" => {
            no_args(kw, args)?;
            Directive::CheckOnly
        }
        "PARSE-ONLY" => {
            no_args(kw, args)?;
            Directive::ParseOnly
        }
        "INTERP-REPORT" => {
            let class = !args.is_empty() && args.bytes().all(|b| b.is_ascii_lowercase() || b == b'-');
            if !class {
                return Err(format!(
                    "INTERP-REPORT takes one report class (`use-after-free`), found `{args}`"
                ));
            }
            Directive::InterpReport(args.to_string())
        }
        _ => {
            no_args(kw, args)?;
            Directive::ResourceLimit
        }
    })
}

/// `"<text>"` with escapes `\\ \" \n \t`, and nothing after the closing quote.
fn quoted(args: &str) -> Result<String, String> {
    let err =
        || format!("STDERR-CONTAINS takes one \"quoted\" string (escapes \\\\ \\\" \\n \\t), found `{args}`");
    let inner = args.strip_prefix('"').ok_or_else(err)?;
    let mut out = String::new();
    let mut chars = inner.chars();
    loop {
        match chars.next() {
            None => return Err(err()),
            Some('"') => break,
            Some('\\') => match chars.next() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                _ => return Err(err()),
            },
            Some(c) => out.push(c),
        }
    }
    if chars.as_str().is_empty() { Ok(out) } else { Err(err()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(src: &str) -> Result<Annot, String> {
        let (a, e) = scan(src, Syntax::Stave);
        match (a.as_slice(), e.as_slice()) {
            ([x], []) => Ok(x.annot.clone()),
            ([], [x]) => Err(x.message.clone()),
            _ => panic!("expected one annotation in {src:?}: {a:?} {e:?}"),
        }
    }

    #[test]
    fn diag_annotations() {
        let e = |c, l| Annot::Diag { severity: Severity::Error, code: Code(c), line: l };
        assert_eq!(one("let x = a + b; //~ ERROR E0201"), Ok(e(201, Some(1))));
        assert_eq!(one("\n\n//~^^ ERROR E0015"), Ok(e(15, Some(1))));
        assert_eq!(one("//~ NOLOC ERROR E1302"), Ok(e(1302, None)));
        assert_eq!(
            one("//~ WARNING E1205  "),
            Ok(Annot::Diag { severity: Severity::Warning, code: Code(1205), line: Some(1) })
        );
        assert_eq!(
            one("//~ NOLOC WARNING E1205"),
            Ok(Annot::Diag { severity: Severity::Warning, code: Code(1205), line: None })
        );
    }

    #[test]
    fn directives() {
        let d = |s: &str| {
            one(s).map(|a| match a {
                Annot::Directive(d) => d,
                other => panic!("{other:?}"),
            })
        };
        assert_eq!(d("//~ STDOUT out.txt"), Ok(Directive::Stdout("out.txt".into())));
        assert_eq!(
            d(r#"//~ STDERR-CONTAINS "a \"b\"\n\t\\""#),
            Ok(Directive::StderrContains("a \"b\"\n\t\\".into()))
        );
        assert_eq!(d("//~ EXIT 101"), Ok(Directive::Exit(101)));
        assert_eq!(
            d("//~ TARGET x86_64-linux"),
            Ok(Directive::Target(Triple::parse("x86_64-linux-v2").unwrap()))
        );
        assert_eq!(
            d("//~ ARGS --release  --compiler cpu"),
            Ok(Directive::Args(vec!["--release".into(), "--compiler".into(), "cpu".into()]))
        );
        assert_eq!(d("//~ CHECK-ONLY"), Ok(Directive::CheckOnly));
        assert_eq!(d("//~ PARSE-ONLY"), Ok(Directive::ParseOnly));
        assert_eq!(d("//~ RESOURCE-LIMIT"), Ok(Directive::ResourceLimit));
        assert_eq!(
            d("//~ INTERP-REPORT write-under-shared"),
            Ok(Directive::InterpReport("write-under-shared".into()))
        );
    }

    #[test]
    fn malformed() {
        for src in [
            "//~ EROR E0201",
            "//~ERROR E0201",
            "//~",
            "//~ ",
            "//~ ERROR",
            "//~ ERROR 0201",
            "//~ ERROR E201",
            "//~ ERROR E02011",
            "//~ ERROR E0201 E0202",
            "//~ ERROR E0201 trailing words",
            "//~^ ERROR E0201",
            "//~ NOLOC E0201",
            "//~^ NOLOC ERROR E1302",
            "//~ EXIT",
            "//~ EXIT 256",
            "//~ EXIT -1",
            "//~ EXIT +1",
            "//~ EXIT 7 8",
            "//~ STDOUT",
            "//~ STDOUT a b",
            "//~ STDOUT /abs/path",
            "//~ STDERR-CONTAINS unquoted",
            "//~ STDERR-CONTAINS \"open",
            "//~ STDERR-CONTAINS \"x\" extra",
            "//~ STDERR-CONTAINS \"bad \\q escape\"",
            "//~ TARGET x86_64-linux-v5",
            "//~ ARGS",
            "//~ CHECK-ONLY now",
            "//~ PARSE-ONLY yes",
            "//~^ PARSE-ONLY",
            "//~ RESOURCE-LIMIT x",
            "//~ INTERP-REPORT",
            "//~ INTERP-REPORT two-exclusive x",
            "//~ INTERP-REPORT Use_After_Free",
            "//~^ INTERP-REPORT two-exclusive",
            "//~^ EXIT 1",
            "//~ error E0201",
        ] {
            assert!(one(src).is_err(), "{src:?} should be malformed: {:?}", one(src));
        }
    }

    #[test]
    fn comments_inside_literals_are_not_annotations() {
        let (a, e) = scan(r#"let s = "//~ ERROR E0201";"#, Syntax::Stave);
        assert!(a.is_empty() && e.is_empty());
        let (a, _) = scan(r#"let s = "a\"//~ x"; //~ ERROR E0201"#, Syntax::Stave);
        assert_eq!(a.len(), 1);
        let (a, _) = scan(r#"let c = '"'; //~ ERROR E0201"#, Syntax::Stave);
        assert_eq!(a.len(), 1);
        let (a, e) = scan("let s = \"unterminated //~ ERROR E0015\n//~^ ERROR E0015", Syntax::Stave);
        assert!(e.is_empty());
        assert_eq!(
            a,
            [AnnotAt {
                comment_line: 2,
                annot: Annot::Diag { severity: Severity::Error, code: Code(15), line: Some(1) }
            }]
        );
        // A plain comment mentioning `//~` later is not an annotation.
        let (a, e) = scan("// see //~ ERROR E0201", Syntax::Stave);
        assert!(a.is_empty() && e.is_empty());
    }

    #[test]
    fn toml_annotations() {
        let src = "name = \"a#~ ERROR E1002\"\nbogus = 1 #~ ERROR E1002\nx = 'lit\\' #~ WARNING E1205\n#~^ ERROR E1002\n";
        let (a, e) = scan(src, Syntax::Toml);
        assert!(e.is_empty(), "{e:?}");
        let lines: Vec<_> = a
            .iter()
            .map(|x| match x.annot {
                Annot::Diag { severity, line, .. } => (x.comment_line, severity, line),
                _ => panic!(),
            })
            .collect();
        assert_eq!(
            lines,
            [(2, Severity::Error, Some(2)), (3, Severity::Warning, Some(3)), (4, Severity::Error, Some(3))]
        );
        for bad in ["#~ NOLOC ERROR E1002", "#~ EXIT 1", "#~ STDOUT x"] {
            let (a, e) = scan(bad, Syntax::Toml);
            assert!(a.is_empty() && e.len() == 1, "{bad}");
        }
        // `//~` means nothing in TOML, `#~` nothing in Stave.
        assert_eq!(scan("//~ ERROR E1002", Syntax::Toml).0.len(), 0);
        assert_eq!(scan("#~ ERROR E1002", Syntax::Stave).0.len(), 0);
    }

    #[test]
    fn codes() {
        assert_eq!(Code::parse("E0001"), Some(Code(1)));
        assert_eq!(Code(1302).to_string(), "E1302");
        assert_eq!(Code::parse("e0001"), None);
        assert_eq!(Code::parse("E00a1"), None);
        assert_eq!(Code::parse("E+001"), None);
    }
}
