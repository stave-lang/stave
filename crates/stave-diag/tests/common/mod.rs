//! Shared fixtures: one diagnostic per (code, shape) of the registry, with sample hole values.

#![allow(dead_code)]

use stave_diag::{Diagnostic, Entry, FileId, HoleKind, Shape, SourceFiles, Span};

/// A sample file with a multi-byte character on line 2.
pub const SAMPLE: &str = "fn main() -> i32 {\n    let é: i32 = 1;\n    return é;\n}\n";

pub fn sample_files() -> (SourceFiles, FileId) {
    let mut files = SourceFiles::new();
    let id = files.add("app", "src/main.stv", SAMPLE.as_bytes()).expect("one file");
    (files, id)
}

/// A canonical-looking value for each hole kind.
pub fn sample_value(kind: HoleKind, choices: &'static [&'static str]) -> &'static str {
    match kind {
        HoleKind::Name => "x",
        HoleKind::Path => "app.util",
        HoleKind::Type => "&mut [u8]",
        HoleKind::TokenKinds => "SEMI, RBRACE",
        HoleKind::Int => "3",
        HoleKind::Place => "*v.f",
        HoleKind::Pattern => "S.{ a, b: (e0, e1) }",
        HoleKind::Source => "x + 1",
        HoleKind::TextLiteral => choices.first().copied().unwrap_or(""),
    }
}

/// The roles a shape's labels and edits use (without "primary"), first occurrence order.
pub fn roles(shape: &Shape) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    let ats =
        shape.labels.iter().map(|l| l.at).chain(shape.fix.iter().flat_map(|f| f.edits.iter().map(|e| e.at)));
    for at in ats {
        if at != "primary" && !out.contains(&at) {
            out.push(at);
        }
    }
    out
}

/// One diagnostic for `shape` of `entry`, every hole filled, every role given twice.
pub fn build(entry: &Entry, shape: &Shape, file: FileId) -> Diagnostic {
    let holes: Vec<(&str, &str)> =
        entry.holes.iter().map(|h| (h.name, sample_value(h.kind, h.choices))).collect();
    let primary = entry.located.then_some(Span::new(file, 23, 2));
    let mut role_spans = Vec::new();
    for (i, r) in roles(shape).into_iter().enumerate() {
        let i = i as u32;
        role_spans.push((r, Span::new(file, 4 + i, 1)));
        role_spans.push((r, Span::new(file, 40 + i, 1)));
    }
    let case = if shape.case.is_empty() { None } else { Some(shape.case) };
    Diagnostic::new(entry.code, case, &holes, primary, &role_spans)
        .unwrap_or_else(|e| panic!("{} case {:?}: {e}", entry.code, shape.case))
}

/// Every (code, shape) of the registry, built against `file`.
pub fn all_diagnostics(file: FileId) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for e in stave_diag::entries().iter().filter(|e| !e.retired) {
        for s in e.shapes {
            out.push(build(e, s, file));
        }
    }
    out
}
