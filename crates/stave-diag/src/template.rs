//! The diagnostic templates (generated from `contracts/diagnostics.toml` by `tools/diaggen`) and
//! the renderer that fills their holes. Every message, label, and fix text comes from here.

use crate::diagnostic::{Applicability, Severity};

#[rustfmt::skip]
#[path = "generated/templates.rs"]
mod generated;

/// Kind of a template hole: what canonical data the compiler may put in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HoleKind {
    Name,
    Path,
    Type,
    TokenKinds,
    Int,
    Place,
    /// A destructuring pattern spelled from the declarations of a type along a place.
    Pattern,
    Source,
    TextLiteral,
}

impl HoleKind {
    /// The kind's spelling in `contracts/diagnostics.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            HoleKind::Name => "name",
            HoleKind::Path => "path",
            HoleKind::Type => "type",
            HoleKind::TokenKinds => "token-kinds",
            HoleKind::Int => "int",
            HoleKind::Place => "place",
            HoleKind::Pattern => "pattern",
            HoleKind::Source => "source",
            HoleKind::TextLiteral => "text-literal",
        }
    }
}

/// A declared hole of an entry.
#[derive(Debug, PartialEq, Eq)]
pub struct Hole {
    pub name: &'static str,
    pub kind: HoleKind,
    /// The allowed values of a `TextLiteral` hole; empty for other kinds.
    pub choices: &'static [&'static str],
}

/// A label template: `at` is `"primary"` or a role naming related spans.
#[derive(Debug, PartialEq, Eq)]
pub struct LabelTemplate {
    pub at: &'static str,
    pub text: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct EditTemplate {
    pub at: &'static str,
    pub replacement: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FixTemplate {
    pub description: &'static str,
    /// [`Applicability::Placeholder`] whenever `edits` is empty.
    pub applicability: Applicability,
    pub edits: &'static [EditTemplate],
}

/// One shape of a code: the base shape (`case == ""`) or a case, complete (keys a case does not
/// give are already copied from the base by diaggen).
#[derive(Debug, PartialEq, Eq)]
pub struct Shape {
    pub case: &'static str,
    pub message: &'static str,
    pub labels: &'static [LabelTemplate],
    pub fix: Option<FixTemplate>,
}

/// One registry entry. A retired entry has no holes and no shapes; its `severity` is a
/// placeholder.
#[derive(Debug, PartialEq, Eq)]
pub struct Entry {
    pub code: &'static str,
    pub severity: Severity,
    pub area: &'static str,
    pub rule: &'static str,
    pub retired: bool,
    pub located: bool,
    pub holes: &'static [Hole],
    pub shapes: &'static [Shape],
}

impl Entry {
    /// The shape for `case` (`None`: the base shape).
    pub fn shape(&self, case: Option<&str>) -> Option<&'static Shape> {
        let want = case.unwrap_or("");
        self.shapes.iter().find(|s| s.case == want)
    }

    pub fn hole(&self, name: &str) -> Option<&'static Hole> {
        self.holes.iter().find(|h| h.name == name)
    }
}

/// Every registry entry, sorted by code.
pub fn entries() -> &'static [Entry] {
    generated::ENTRIES
}

/// The entry of `code`, if registered.
pub fn entry(code: &str) -> Option<&'static Entry> {
    let all = generated::ENTRIES;
    all.binary_search_by(|e| e.code.cmp(code)).ok().and_then(|i| all.get(i))
}

/// Why rendering a template failed. Each is a compiler bug at the call site, reported as an error
/// rather than a panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderError {
    UnknownCode(String),
    Retired(&'static str),
    UnknownCase {
        code: &'static str,
        case: String,
    },
    UnknownHole {
        code: &'static str,
        hole: String,
    },
    DuplicateHole {
        code: &'static str,
        hole: String,
    },
    MissingHole {
        code: &'static str,
        hole: &'static str,
    },
    BadValue {
        code: &'static str,
        hole: &'static str,
        kind: HoleKind,
        value: String,
    },
    /// A rendered message, label, or fix description would span several lines (a hole value other
    /// than `source` with a newline; `source` values are collapsed to one line instead).
    NotOneLine {
        code: &'static str,
        hole: &'static str,
    },
    /// The generated template is malformed (diaggen rejects this; kept for robustness).
    BadTemplate {
        code: &'static str,
        template: &'static str,
    },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::UnknownCode(c) => write!(f, "unknown diagnostic code {c:?}"),
            RenderError::Retired(c) => write!(f, "diagnostic code {c} is retired"),
            RenderError::UnknownCase { code, case } => write!(f, "{code}: unknown case {case:?}"),
            RenderError::UnknownHole { code, hole } => write!(f, "{code}: undeclared hole {hole:?}"),
            RenderError::DuplicateHole { code, hole } => write!(f, "{code}: hole {hole:?} given twice"),
            RenderError::MissingHole { code, hole } => write!(f, "{code}: hole {hole:?} not given"),
            RenderError::BadValue { code, hole, kind, value } => {
                write!(f, "{code}: value {value:?} is not a valid {} for hole {hole:?}", kind.as_str())
            }
            RenderError::NotOneLine { code, hole } => {
                write!(f, "{code}: hole {hole:?} has a newline where the text must be one line")
            }
            RenderError::BadTemplate { code, template } => {
                write!(f, "{code}: malformed template {template:?}")
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// A rendered label: `at` is `"primary"` or a role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedLabel {
    pub at: &'static str,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedEdit {
    pub at: &'static str,
    pub replacement: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedFix {
    pub description: String,
    pub applicability: Applicability,
    pub edits: Vec<RenderedEdit>,
}

/// The texts of one diagnostic, before spans are attached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    pub entry: &'static Entry,
    pub message: String,
    pub labels: Vec<RenderedLabel>,
    pub fix: Option<RenderedFix>,
}

/// Fills the templates of `code` (shape `case`, `None` for the base) with `holes`
/// (`(name, value)` pairs). Every hole the shape uses must be given; extra declared holes are
/// allowed; undeclared ones are an error. Values are checked against their hole kind.
pub fn render(code: &str, case: Option<&str>, holes: &[(&str, &str)]) -> Result<Rendered, RenderError> {
    let entry = entry(code).ok_or_else(|| RenderError::UnknownCode(code.to_string()))?;
    if entry.retired {
        return Err(RenderError::Retired(entry.code));
    }
    let shape = entry
        .shape(case)
        .ok_or_else(|| RenderError::UnknownCase { code: entry.code, case: case.unwrap_or("").to_string() })?;
    for (i, (name, value)) in holes.iter().enumerate() {
        let hole = entry
            .hole(name)
            .ok_or_else(|| RenderError::UnknownHole { code: entry.code, hole: name.to_string() })?;
        if holes[..i].iter().any(|(n, _)| n == name) {
            return Err(RenderError::DuplicateHole { code: entry.code, hole: name.to_string() });
        }
        check_value(entry.code, hole, value)?;
    }
    let one_line = |t: &'static str| expand(entry, t, holes, true);
    let message = one_line(shape.message)?;
    let labels = shape
        .labels
        .iter()
        .map(|l| Ok(RenderedLabel { at: l.at, text: one_line(l.text)? }))
        .collect::<Result<Vec<_>, RenderError>>()?;
    let fix = match &shape.fix {
        None => None,
        Some(f) => Some(RenderedFix {
            description: one_line(f.description)?,
            applicability: f.applicability,
            edits: f
                .edits
                .iter()
                .map(|e| {
                    Ok(RenderedEdit { at: e.at, replacement: expand(entry, e.replacement, holes, false)? })
                })
                .collect::<Result<Vec<_>, RenderError>>()?,
        }),
    };
    Ok(Rendered { entry, message, labels, fix })
}

fn check_value(code: &'static str, hole: &'static Hole, value: &str) -> Result<(), RenderError> {
    let ok = match hole.kind {
        HoleKind::Int => {
            let digits = value.strip_prefix('-').unwrap_or(value);
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        }
        HoleKind::TextLiteral => hole.choices.contains(&value),
        HoleKind::Name | HoleKind::Path | HoleKind::Place | HoleKind::Pattern => {
            !value.is_empty() && !value.contains('\n')
        }
        HoleKind::Type | HoleKind::TokenKinds => !value.contains('\n'),
        HoleKind::Source => true,
    };
    if ok {
        Ok(())
    } else {
        Err(RenderError::BadValue { code, hole: hole.name, kind: hole.kind, value: value.to_string() })
    }
}

/// Expands one template: `{hole}` inserts the hole's value, `{{` and `}}` are literal braces.
fn expand(
    entry: &'static Entry,
    template: &'static str,
    holes: &[(&str, &str)],
    one_line: bool,
) -> Result<String, RenderError> {
    let bad = || RenderError::BadTemplate { code: entry.code, template };
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        if let Some(after) = tail.strip_prefix("{{") {
            out.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}") {
            out.push('}');
            rest = after;
        } else if let Some(after) = tail.strip_prefix('{') {
            let end = after.find('}').ok_or_else(bad)?;
            let name = &after[..end];
            let hole = entry.hole(name).ok_or_else(bad)?;
            let value = holes
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .ok_or(RenderError::MissingHole { code: entry.code, hole: hole.name })?;
            if one_line && hole.kind == HoleKind::Source {
                push_collapsed(&mut out, value);
            } else if one_line && value.contains('\n') {
                return Err(RenderError::NotOneLine { code: entry.code, hole: hole.name });
            } else {
                out.push_str(value);
            }
            rest = &after[end + 1..];
        } else {
            return Err(bad());
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Whitespace of a `source` hole in one-line text: space, tab, LF, VT, FF, CR
/// (contracts/diagnostics.toml header).
fn is_collapsed_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r')
}

/// Appends `value` with every whitespace run replaced by one space and the ends trimmed, so a
/// `source` hole renders on one line.
fn push_collapsed(out: &mut String, value: &str) {
    let mut first = true;
    for word in value.split(is_collapsed_space).filter(|w| !w.is_empty()) {
        if !first {
            out.push(' ');
        }
        out.push_str(word);
        first = false;
    }
}

/// Source text as the value of a hole whose kind is not `source` (a place, name, path, or type
/// spelled as written): every whitespace run becomes one space and the ends are trimmed, as a
/// `source` hole is in one-line text, so an expression or type that spans lines is a valid value.
/// Text that is empty after this stays empty: the builder rejects it as a value, which is then a
/// compiler bug.
pub fn spelled(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    push_collapsed(&mut out, text);
    out
}
