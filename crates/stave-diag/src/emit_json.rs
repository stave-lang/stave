//! JSON form (`--diagnostics json`): one object per line, per `contracts/diagnostics.schema.json`
//! (`$defs/diagnostic`). Keys in a fixed order: kind, code, severity, message, primary, related,
//! fix (description, applicability, edits). `primary` is absent only for location-less codes;
//! `related` is always present.

use std::fmt::Write;

use crate::diagnostic::Diagnostic;
use crate::span::{LocateError, Location, SourceFiles};

/// Appends `s` as a JSON string literal: `"` and `\` backslash-escaped, newline, carriage return,
/// and tab as `\n` `\r` `\t`, every other control character and DEL as `\uXXXX`. The one JSON
/// string escaper of the compiler's output formats.
pub fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `s` as a JSON string literal ([`push_json_string`]).
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    push_json_string(&mut out, s);
    out
}

fn push_span(out: &mut String, loc: &Location<'_>) {
    out.push_str("{\"file\":");
    push_json_string(out, loc.path);
    let _ = write!(out, ",\"line\":{},\"col\":{},\"len\":{}}}", loc.line, loc.col, loc.len);
}

/// Appends one diagnostic as a JSON object followed by a newline.
pub fn write_json(out: &mut String, d: &Diagnostic, files: &SourceFiles) -> Result<(), LocateError> {
    let primary = d.primary().map(|s| files.locate(s)).transpose()?;
    let related = d.related().iter().map(|r| files.locate(r.span)).collect::<Result<Vec<_>, _>>()?;
    let edits = match d.fix() {
        Some(f) => f.edits.iter().map(|e| files.locate(e.span)).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    out.push_str("{\"kind\":\"diagnostic\",\"code\":");
    push_json_string(out, d.code());
    out.push_str(",\"severity\":");
    push_json_string(out, d.severity().as_str());
    out.push_str(",\"message\":");
    push_json_string(out, d.message());
    if let Some(loc) = &primary {
        out.push_str(",\"primary\":");
        push_span(out, loc);
    }
    out.push_str(",\"related\":[");
    for (i, (r, loc)) in d.related().iter().zip(&related).enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"span\":");
        push_span(out, loc);
        out.push_str(",\"label\":");
        push_json_string(out, &r.label);
        out.push('}');
    }
    out.push(']');
    if let Some(f) = d.fix() {
        out.push_str(",\"fix\":{\"description\":");
        push_json_string(out, &f.description);
        out.push_str(",\"applicability\":");
        push_json_string(out, f.applicability.as_str());
        out.push_str(",\"edits\":[");
        for (i, (e, loc)) in f.edits.iter().zip(&edits).enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"span\":");
            push_span(out, loc);
            out.push_str(",\"replacement\":");
            push_json_string(out, &e.replacement);
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push_str("}\n");
    Ok(())
}

/// All diagnostics as JSON lines, in the given order (sort them with
/// [`crate::canonical_order`] first).
pub fn emit_json(diags: &[Diagnostic], files: &SourceFiles) -> Result<String, LocateError> {
    let mut out = String::new();
    for d in diags {
        write_json(&mut out, d, files)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::json_string;

    #[test]
    fn string_escapes() {
        assert_eq!(json_string("a\"b\\c\nd\r\t\u{1}\u{7f}é"), r#""a\"b\\c\nd\r\t\u0001\u007fé""#);
        assert_eq!(json_string(""), "\"\"");
    }
}
