//! Human-readable form:
//!
//! ```text
//! src/a.stv:3:5: error[E0201]: mismatched types: expected `i64`, found `i32`
//!   src/a.stv:3:5: this is `i32`
//!   src/a.stv:2:12: `i64` required by this
//!   fix: convert explicitly: `x as i64` (or `as%`, `as|`, `as?`, `as~` when lossy)
//!   edit src/a.stv:3:5 (1 byte): "x as i64"
//! ```
//!
//! A location-less diagnostic starts with `error[EXXXX]: `. Edit replacements are printed as
//! JSON-style quoted strings, so the output keeps one line per item.

use std::fmt::Write;

use crate::diagnostic::Diagnostic;
use crate::emit_json::push_json_string;
use crate::span::{LocateError, Location, SourceFiles};

fn push_loc(out: &mut String, loc: &Location<'_>) {
    // Writing to a String never fails.
    let _ = write!(out, "{}:{}:{}", loc.path, loc.line, loc.col);
}

/// Appends one diagnostic in human form (ending with a newline).
pub fn write_human(out: &mut String, d: &Diagnostic, files: &SourceFiles) -> Result<(), LocateError> {
    // Resolve every span first, so a failure leaves `out` unchanged.
    let primary = d.primary().map(|s| files.locate(s)).transpose()?;
    let related = d.related().iter().map(|r| files.locate(r.span)).collect::<Result<Vec<_>, _>>()?;
    let edits = match d.fix() {
        Some(f) => f.edits.iter().map(|e| files.locate(e.span)).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    if let Some(loc) = &primary {
        push_loc(out, loc);
        out.push_str(": ");
    }
    let _ = writeln!(out, "{}[{}]: {}", d.severity().as_str(), d.code(), d.message());
    for (r, loc) in d.related().iter().zip(&related) {
        out.push_str("  ");
        push_loc(out, loc);
        let _ = writeln!(out, ": {}", r.label);
    }
    if let Some(f) = d.fix() {
        let _ = writeln!(out, "  fix: {}", f.description);
        for (e, loc) in f.edits.iter().zip(&edits) {
            out.push_str("  edit ");
            push_loc(out, loc);
            let unit = if loc.len == 1 { "byte" } else { "bytes" };
            let _ = write!(out, " ({} {unit}): ", loc.len);
            push_json_string(out, &e.replacement);
            out.push('\n');
        }
    }
    Ok(())
}

/// All diagnostics in human form, in the given order (sort them with
/// [`crate::canonical_order`] first).
pub fn emit_human(diags: &[Diagnostic], files: &SourceFiles) -> Result<String, LocateError> {
    let mut out = String::new();
    for d in diags {
        write_human(&mut out, d, files)?;
    }
    Ok(out)
}
