//! Diagnostics: spans, registry templates, canonical order, human and JSON emitters.
//!
//! Every message, label, and fix text comes from the templates generated from
//! `contracts/diagnostics.toml` (`tools/diaggen`). The compiler builds a [`Diagnostic`] from a code,
//! a case, hole values, and spans, collects them in a [`Sink`], and emits them once, in canonical
//! order, at the end of the build.

mod diagnostic;
mod emit_human;
mod emit_json;
mod order;
mod span;
mod template;

pub use diagnostic::{DiagError, Diagnostic, Edit, Fix, Related, Severity};
pub use emit_human::{emit_human, write_human};
pub use emit_json::{emit_json, json_string, push_json_string, write_json};
pub use order::{Sink, canonical_order};
pub use span::{FileId, FileTableError, LineIndex, LocateError, Location, SourceFile, SourceFiles, Span};
pub use template::{
    EditTemplate, Entry, FixTemplate, Hole, HoleKind, LabelTemplate, RenderError, Rendered, RenderedEdit,
    RenderedFix, RenderedLabel, Shape, entries, entry, render,
};
