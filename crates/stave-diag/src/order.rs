//! Canonical order and deduplication of diagnostics, done once at the end of a build.
//!
//! Order: location-less diagnostics first, by (code, message); then by (package, file path,
//! primary byte offset, code, message). Diagnostics with identical (file, offset, code, message)
//! are reduced to one. Remaining ties are broken by every other field, so the output never
//! depends on insertion order or thread count.

use std::cmp::Ordering;
use std::sync::Mutex;

use crate::diagnostic::{Diagnostic, Fix, Related};
use crate::span::{FileId, SourceFiles, Span};

/// Sort key of a file: known files by (package, path); files missing from the table after them,
/// by id.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum FileKey<'a> {
    Known(&'a str, &'a str),
    Unknown(FileId),
}

fn file_key(files: &SourceFiles, id: FileId) -> FileKey<'_> {
    match files.get(id) {
        Some(f) => FileKey::Known(&f.package, &f.path),
        None => FileKey::Unknown(id),
    }
}

fn cmp_span(files: &SourceFiles, a: &Span, b: &Span) -> Ordering {
    file_key(files, a.file)
        .cmp(&file_key(files, b.file))
        .then(a.offset.cmp(&b.offset))
        .then(a.len.cmp(&b.len))
}

fn cmp_related(files: &SourceFiles, a: &[Related], b: &[Related]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let o = cmp_span(files, &x.span, &y.span).then_with(|| x.label.cmp(&y.label));
        if o != Ordering::Equal {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

fn cmp_fix(files: &SourceFiles, a: Option<&Fix>, b: Option<&Fix>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(a), Some(b)) => a.description.cmp(&b.description).then_with(|| {
            for (x, y) in a.edits.iter().zip(&b.edits) {
                let o = cmp_span(files, &x.span, &y.span).then_with(|| x.replacement.cmp(&y.replacement));
                if o != Ordering::Equal {
                    return o;
                }
            }
            a.edits.len().cmp(&b.edits.len())
        }),
    }
}

/// The canonical sort-key comparison; `Equal` means the two are duplicates.
fn cmp_key(files: &SourceFiles, a: &Diagnostic, b: &Diagnostic) -> Ordering {
    let loc = match (a.primary(), b.primary()) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => {
            file_key(files, x.file).cmp(&file_key(files, y.file)).then(x.offset.cmp(&y.offset))
        }
    };
    loc.then_with(|| a.code().cmp(b.code())).then_with(|| a.message().cmp(b.message()))
}

/// Total order: the sort key, then every other field.
fn cmp_full(files: &SourceFiles, a: &Diagnostic, b: &Diagnostic) -> Ordering {
    cmp_key(files, a, b)
        .then_with(|| a.primary().map(|s| s.len).cmp(&b.primary().map(|s| s.len)))
        .then_with(|| a.severity().cmp(&b.severity()))
        .then_with(|| cmp_related(files, a.related(), b.related()))
        .then_with(|| cmp_fix(files, a.fix(), b.fix()))
}

/// Sorts `diags` into the canonical order and removes duplicates (the smallest of each group
/// in the total order is kept).
pub fn canonical_order(mut diags: Vec<Diagnostic>, files: &SourceFiles) -> Vec<Diagnostic> {
    diags.sort_by(|a, b| cmp_full(files, a, b));
    diags.dedup_by(|later, kept| cmp_key(files, later, kept) == Ordering::Equal);
    diags
}

/// Collects diagnostics from any number of threads; [`Sink::finish`] returns them in canonical
/// order. Nothing is emitted in discovery order.
#[derive(Debug, Default)]
pub struct Sink {
    diags: Mutex<Vec<Diagnostic>>,
}

impl Sink {
    pub fn new() -> Sink {
        Sink::default()
    }

    pub fn push(&self, d: Diagnostic) {
        self.lock().push(d);
    }

    pub fn extend<I: IntoIterator<Item = Diagnostic>>(&self, ds: I) {
        self.lock().extend(ds);
    }

    /// Number of diagnostics collected so far (duplicates included).
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn finish(self, files: &SourceFiles) -> Vec<Diagnostic> {
        let diags = self.diags.into_inner().unwrap_or_else(|e| e.into_inner());
        canonical_order(diags, files)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Diagnostic>> {
        // A panic in another pushing thread leaves the vector intact: keep using it.
        self.diags.lock().unwrap_or_else(|e| e.into_inner())
    }
}
