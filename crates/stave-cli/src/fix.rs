//! `stave fix [PKG_DIR] [--dry-run]` (contracts/cli.md): applies the fixes whose applicability
//! is `exact`, re-checks, and repeats.
//!
//! A round checks the program as `stave check` does and walks its diagnostics in canonical
//! order. A fix is applied when it is `exact`, all its edits lie in source files of the root
//! package, and none of them overlaps an edit already applied in the round ([`plan_round`]); a
//! fix skipped for an overlap is reported again by the next round's check. At most
//! [`MAX_ROUNDS`] rounds apply edits, and the run ends earlier when a round applies nothing, so
//! the result depends only on the sources.
//!
//! Rounds check an in-memory copy of the changed files: the files are written once, after the
//! last round, and `--dry-run` runs the same rounds without writing.
//!
//! - stdout: one `<file>:<line>:<col>: <code> fixed` line per applied fix, at the diagnostic's
//!   location in the round that reported it. With `--dry-run`: a unified diff of the changed
//!   files instead (`--- a/<file>`, `+++ b/<file>`, three lines of context).
//! - stderr: the diagnostics that remain, in the `--diagnostics` format.
//! - exit status: 0 when no error remains, 1 otherwise.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use stave_diag::{Applicability, Diagnostic, Span};
use stave_driver::build::Failure;
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options, Output};

use crate::args::{Args, DiagFormat};
use crate::{EXIT_ICE, EXIT_OK, EXIT_USAGE, EXIT_USER, internal_error, library, pkg_path, threads_from_env};

/// Rounds that may apply edits.
pub const MAX_ROUNDS: usize = 8;

/// Whether two edited byte ranges `(start, end)` of one file conflict: they share a byte, or one
/// is an insertion strictly inside the other, or both are insertions at the same offset (their
/// order would be arbitrary). An insertion at either end of a replaced range does not conflict.
fn overlaps(a: (u64, u64), b: (u64, u64)) -> bool {
    (a.0 < b.1 && b.0 < a.1) || (a.0 == a.1 && b.0 == b.1 && a.0 == b.0)
}

fn range(s: Span) -> (u64, u64) {
    (u64::from(s.offset), u64::from(s.offset) + u64::from(s.len))
}

/// The fixes one round applies: indices into `diags` (in canonical order), in that order. A
/// diagnostic is chosen when its fix is `exact`, every edit span passes `fixable`, and no edit
/// overlaps an edit of a fix chosen before it (or another edit of the same fix).
pub fn plan_round(diags: &[Diagnostic], fixable: impl Fn(Span) -> bool) -> Vec<usize> {
    let mut taken: Vec<Span> = Vec::new();
    let mut chosen = Vec::new();
    for (i, d) in diags.iter().enumerate() {
        let Some(fix) = d.fix() else { continue };
        if fix.applicability != Applicability::Exact || fix.edits.is_empty() {
            continue;
        }
        let first = taken.len();
        let ok = fix.edits.iter().all(|e| {
            let free = fixable(e.span)
                && !taken.iter().any(|t| t.file == e.span.file && overlaps(range(*t), range(e.span)));
            taken.push(e.span);
            free
        });
        if ok {
            chosen.push(i);
        } else {
            taken.truncate(first);
        }
    }
    chosen
}

/// `text` with `edits` (`(start, end, replacement)`, in bounds, pairwise not overlapping)
/// applied. An insertion at the start of a replaced range goes before the replacement.
pub fn apply_edits(text: &[u8], edits: &[(usize, usize, &str)]) -> Vec<u8> {
    let mut order: Vec<&(usize, usize, &str)> = edits.iter().collect();
    order.sort_by_key(|e| (e.0, e.1));
    let mut out = Vec::with_capacity(text.len() + edits.iter().map(|e| e.2.len()).sum::<usize>());
    let mut at = 0;
    for &&(start, end, replacement) in &order {
        out.extend_from_slice(text.get(at..start).unwrap_or_default());
        out.extend_from_slice(replacement.as_bytes());
        at = end.max(at);
    }
    out.extend_from_slice(text.get(at..).unwrap_or_default());
    out
}

/// A file the run changed.
struct Changed {
    /// The path diagnostics print.
    display: String,
    original: Vec<u8>,
    current: Vec<u8>,
}

/// Whether `span` lies in a source file of the root package (not a manifest, not `std`,
/// `runtime`, or another dependency).
fn in_root_package(output: &Output, span: Span) -> bool {
    let p = &output.program;
    p.files.get(span.file.0 as usize).is_some_and(|f| {
        p.packages.get(f.package.0 as usize).is_some_and(|pkg| pkg.is_root)
            && range(span).1 <= f.bytes.len() as u64
    })
}

/// Applies the round's fixes to `changed` and appends their `fixed` lines to `log`. Returns
/// whether anything was applied.
fn apply_round(
    output: &Output,
    changed: &mut BTreeMap<PathBuf, Changed>,
    log: &mut String,
) -> Result<bool, stave_diag::LocateError> {
    let chosen = plan_round(&output.diagnostics, |s| in_root_package(output, s));
    let mut by_file: BTreeMap<u32, Vec<(usize, usize, &str)>> = BTreeMap::new();
    for &i in &chosen {
        let d = &output.diagnostics[i];
        match d.primary() {
            Some(span) => {
                let loc = output.files().locate(span)?;
                log.push_str(&format!("{}:{}:{}: {} fixed\n", loc.path, loc.line, loc.col, d.code()));
            }
            None => log.push_str(&format!("{} fixed\n", d.code())),
        }
        for e in d.fix().into_iter().flat_map(|f| &f.edits) {
            let (start, end) = range(e.span);
            by_file.entry(e.span.file.0).or_default().push((
                start as usize,
                end as usize,
                e.replacement.as_str(),
            ));
        }
    }
    for (file, edits) in &by_file {
        let Some(f) = output.program.files.get(*file as usize) else { continue };
        let current = apply_edits(&f.bytes, edits);
        match changed.get_mut(&f.abs_path) {
            Some(c) => c.current = current,
            None => {
                let c = Changed { display: f.display_path.clone(), original: f.bytes.to_vec(), current };
                changed.insert(f.abs_path.clone(), c);
            }
        }
    }
    Ok(!chosen.is_empty())
}

pub(crate) fn run(a: &Args, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let path = pkg_path(a);
    if let Err(e) = package::check_root(&path) {
        let _ = writeln!(err, "stave: {e}");
        return EXIT_USAGE;
    }
    let threads = match threads_from_env(std::env::var("STAVE_THREADS").ok().as_deref()) {
        Ok(t) => t,
        Err(msg) => {
            let _ = writeln!(err, "stave: {msg}\nrun `stave --help` for usage");
            return EXIT_USAGE;
        }
    };
    let lib = match library(err) {
        Ok(l) => l,
        Err(code) => return code,
    };
    let pool = match pipeline::thread_pool(threads) {
        Ok(p) => p,
        Err(e) => {
            let _ = writeln!(err, "stave: {e}");
            return EXIT_USAGE;
        }
    };
    let mut opts = Options::new(Goal::Check).threads(threads);
    opts.target = Some(a.target.triple);

    let mut changed: BTreeMap<PathBuf, Changed> = BTreeMap::new();
    let mut log = String::new();
    let mut rounds = 0;
    let output = loop {
        opts.sources = changed.iter().map(|(p, c)| (p.clone(), c.current.clone())).collect();
        let output = pipeline::run_on(&pool, &path, &lib, &opts);
        if rounds == MAX_ROUNDS {
            break output;
        }
        match apply_round(&output, &mut changed, &mut log) {
            Ok(true) => rounds += 1,
            Ok(false) => break output,
            Err(e) => return internal_error(&e.to_string(), err),
        }
    };

    if a.dry_run {
        let mut files: Vec<&Changed> = changed.values().collect();
        files.sort_by(|x, y| x.display.cmp(&y.display));
        for c in files {
            let _ = out.write_all(&unified_diff(&c.display, &c.original, &c.current));
        }
    } else {
        for (p, c) in &changed {
            if let Err(e) = std::fs::write(p, &c.current) {
                let _ = writeln!(err, "stave: cannot write {}: {e}", p.display());
                return EXIT_USAGE;
            }
        }
        let _ = out.write_all(log.as_bytes());
    }

    let text = match a.diagnostics {
        DiagFormat::Human => stave_diag::emit_human(&output.diagnostics, output.files()),
        DiagFormat::Json => stave_diag::emit_json(&output.diagnostics, output.files()),
    };
    match text {
        Ok(t) => {
            let _ = err.write_all(t.as_bytes());
        }
        Err(e) => return internal_error(&e.to_string(), err),
    }
    if output.has_errors() {
        return EXIT_USER;
    }
    match &output.failure {
        Some(f @ Failure::Internal { .. }) => {
            let _ = internal_error(&f.to_string(), err);
            EXIT_ICE
        }
        Some(f) => {
            let _ = writeln!(err, "stave: {f}");
            EXIT_USAGE
        }
        None => EXIT_OK,
    }
}

// ---------- unified diff ----------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Keep,
    Delete,
    Insert,
}

/// The shortest edit script from `a` to `b` (Myers' O(ND) algorithm), over lines.
fn edit_script(a: &[&[u8]], b: &[&[u8]]) -> Vec<Op> {
    let (n, m) = (a.len() as isize, b.len() as isize);
    let max = (n + m) as usize;
    // v[k + off]: the furthest x reached on diagonal k = x - y
    let off = max as isize + 1;
    let mut v = vec![0isize; 2 * max + 3];
    // per d: v's window for diagonals -d-1..=d+1 as it was before round d
    let mut trace: Vec<Vec<isize>> = Vec::new();
    let down = |v: &[isize], base: isize, k: isize, d: isize| {
        k == -d || (k != d && v[(k - 1 + base) as usize] < v[(k + 1 + base) as usize])
    };
    let mut found = None;
    'search: for d in 0..=max as isize {
        trace.push(v[(off - d - 1) as usize..=(off + d + 1) as usize].to_vec());
        let mut k = -d;
        while k <= d {
            let mut x =
                if down(&v, off, k, d) { v[(k + 1 + off) as usize] } else { v[(k - 1 + off) as usize] + 1 };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[(k + off) as usize] = x;
            if x >= n && y >= m {
                found = Some(d);
                break 'search;
            }
            k += 2;
        }
    }
    let mut ops = Vec::new();
    let (mut x, mut y) = (n, m);
    for d in (0..=found.unwrap_or(0)).rev() {
        let w = &trace[d as usize];
        // w[j] is diagonal j - d - 1
        let base = d + 1;
        let k = x - y;
        let prev_k = if down(w, base, k, d) { k + 1 } else { k - 1 };
        let prev_x = w[(prev_k + base) as usize];
        let prev_y = prev_x - prev_k;
        while x > prev_x && y > prev_y {
            ops.push(Op::Keep);
            x -= 1;
            y -= 1;
        }
        if d > 0 {
            ops.push(if x == prev_x { Op::Insert } else { Op::Delete });
        }
        x = prev_x;
        y = prev_y;
    }
    ops.reverse();
    ops
}

/// Lines of unchanged text printed around a change.
const CONTEXT: usize = 3;

fn push_line(out: &mut Vec<u8>, mark: u8, line: &[u8]) {
    out.push(mark);
    out.extend_from_slice(line);
    if line.last() != Some(&b'\n') {
        out.extend_from_slice(b"\n\\ No newline at end of file\n");
    }
}

/// The unified diff from `old` to `new`, empty when they are equal. Lines end at `\n`.
pub fn unified_diff(path: &str, old: &[u8], new: &[u8]) -> Vec<u8> {
    let a: Vec<&[u8]> = old.split_inclusive(|&c| c == b'\n').collect();
    let b: Vec<&[u8]> = new.split_inclusive(|&c| c == b'\n').collect();
    // the common ends need no search
    let head = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let tail = a[head..].iter().rev().zip(b[head..].iter().rev()).take_while(|(x, y)| x == y).count();
    let mut ops = vec![Op::Keep; head];
    ops.extend(edit_script(&a[head..a.len() - tail], &b[head..b.len() - tail]));
    ops.extend(std::iter::repeat_n(Op::Keep, tail));

    let mut out = Vec::new();
    if !ops.iter().any(|&o| o != Op::Keep) {
        return out;
    }
    out.extend_from_slice(format!("--- a/{path}\n+++ b/{path}\n").as_bytes());
    // (ai, bi): the line of `a` and of `b` that ops[i] is at
    let mut pos = Vec::with_capacity(ops.len() + 1);
    let (mut ai, mut bi) = (0usize, 0usize);
    for &o in &ops {
        pos.push((ai, bi));
        ai += usize::from(o != Op::Insert);
        bi += usize::from(o != Op::Delete);
    }
    pos.push((ai, bi));
    let mut i = 0;
    while let Some(first) = (i..ops.len()).find(|&j| ops[j] != Op::Keep) {
        // the hunk grows while the next change is within two contexts of the last one
        let mut last = first;
        while let Some(next) = (last + 1..ops.len()).find(|&j| ops[j] != Op::Keep) {
            if next - last - 1 > 2 * CONTEXT {
                break;
            }
            last = next;
        }
        let start = first.saturating_sub(CONTEXT).max(i);
        let end = (last + 1 + CONTEXT).min(ops.len());
        let (a0, b0) = pos[start];
        let (a1, b1) = pos[end];
        // an empty range is named by the line before it
        let from = |at: usize, len: usize| if len == 0 { at } else { at + 1 };
        out.extend_from_slice(
            format!("@@ -{},{} +{},{} @@\n", from(a0, a1 - a0), a1 - a0, from(b0, b1 - b0), b1 - b0)
                .as_bytes(),
        );
        for (o, &(la, lb)) in ops[start..end].iter().zip(&pos[start..end]) {
            match o {
                Op::Keep => push_line(&mut out, b' ', a[la]),
                Op::Delete => push_line(&mut out, b'-', a[la]),
                Op::Insert => push_line(&mut out, b'+', b[lb]),
            }
        }
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diff(old: &str, new: &str) -> String {
        String::from_utf8(unified_diff("src/a.stv", old.as_bytes(), new.as_bytes())).unwrap()
    }

    #[test]
    fn overlap_rule() {
        assert!(overlaps((2, 5), (4, 6)));
        assert!(overlaps((2, 5), (3, 3)));
        assert!(overlaps((3, 3), (3, 3)));
        assert!(!overlaps((2, 5), (5, 7)));
        assert!(!overlaps((2, 5), (2, 2)));
        assert!(!overlaps((2, 5), (5, 5)));
    }

    #[test]
    fn edits_apply_in_offset_order() {
        let text = b"a::b(x); c;";
        let edits = [(9, 9, "let _ = "), (1, 3, "."), (0, 0, "<"), (0, 1, "A")];
        assert_eq!(apply_edits(text, &edits), b"<A.b(x); let _ = c;");
        assert_eq!(apply_edits(text, &[]), text);
    }

    #[test]
    fn diff_of_equal_texts_is_empty() {
        assert_eq!(diff("a\nb\n", "a\nb\n"), "");
        assert_eq!(diff("", ""), "");
    }

    #[test]
    fn diff_hunks() {
        let old: String = (1..=20).map(|i| format!("l{i}\n")).collect();
        let new = old.replace("l2\n", "L2\n").replace("l17\n", "").replace("l19\n", "l19\nnew\n");
        assert_eq!(
            diff(&old, &new),
            "--- a/src/a.stv\n+++ b/src/a.stv\n\
             @@ -1,5 +1,5 @@\n l1\n-l2\n+L2\n l3\n l4\n l5\n\
             @@ -14,7 +14,7 @@\n l14\n l15\n l16\n-l17\n l18\n l19\n+new\n l20\n"
        );
    }

    #[test]
    fn diff_without_final_newline() {
        assert_eq!(
            diff("a\nb", "a\nc"),
            "--- a/src/a.stv\n+++ b/src/a.stv\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n\\ No newline at end of file\n"
        );
        assert_eq!(diff("", "x\n"), "--- a/src/a.stv\n+++ b/src/a.stv\n@@ -0,0 +1,1 @@\n+x\n");
    }
}
