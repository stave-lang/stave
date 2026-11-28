//! `stave loops [PKG_DIR] [--json] [--all]` (contracts/cli.md): the loops of the root package
//! (`--all`: of every package), one per line in canonical (file, offset) order, then a summary
//! line. Always stage0.
//!
//! - human: `<file>:<line>:<col> <fn> <form> <class>[ <reason>]`
//! - `--json`: one object per line, `{"file":"src/main.stv","line":3,"col":5,"fn":"main.f",
//!   "form":"range","class":"indexed","reason":"","sources":["xs","&mut out"],"folds":[]}`
//!
//! `file` as in `unsafe-report`; `line` and `col` locate the loop's keyword. A package with error
//! diagnostics prints them to stderr, nothing to stdout, exit 1.

use std::io::Write;

use stave_diag::{LocateError, json_string as quote};
use stave_driver::build::Failure;
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options};
use stave_sema::loops::{LoopForm, LoopInfo};

use crate::args::Args;
use crate::effects_report::report_file;
use crate::{EXIT_OK, EXIT_USAGE, EXIT_USER, internal_error, library, pkg_path, threads_from_env};

/// The classes, in the order of the summary line.
#[derive(Clone, Copy)]
enum Class {
    Chunk,
    Indexed,
    Sequential,
    Unread,
}

const CLASSES: [Class; 4] = [Class::Chunk, Class::Indexed, Class::Sequential, Class::Unread];

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Chunk => "chunk",
            Class::Indexed => "indexed",
            Class::Sequential => "sequential",
            Class::Unread => "unread",
        }
    }
}

/// A loop's class and its reason (empty: none).
fn classify(l: &LoopInfo) -> (Class, &'static str) {
    match l.form {
        LoopForm::Chunk => (Class::Chunk, ""),
        LoopForm::Range if l.access.indexed => (Class::Indexed, ""),
        // an element loop has no index variable
        LoopForm::Range | LoopForm::Element => (Class::Sequential, "not-indexed"),
        LoopForm::Iter | LoopForm::While | LoopForm::Loop => (Class::Unread, ""),
    }
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
    let opts = Options::new(Goal::Check).threads(threads);
    let output = match pipeline::run(&path, &lib, &opts) {
        Ok(o) => o,
        Err(e) => {
            let _ = writeln!(err, "stave: {e}");
            return EXIT_USAGE;
        }
    };
    if output.has_errors() {
        let text = if a.json {
            stave_diag::emit_json(&output.diagnostics, output.files())
        } else {
            stave_diag::emit_human(&output.diagnostics, output.files())
        };
        return match text {
            Ok(t) => {
                let _ = err.write_all(t.as_bytes());
                EXIT_USER
            }
            Err(e) => internal_error(&e.to_string(), err),
        };
    }
    if let Some(f) = &output.failure {
        if matches!(f, Failure::Internal { .. }) {
            return internal_error(&f.to_string(), err);
        }
        let _ = writeln!(err, "stave: {f}");
        return EXIT_USAGE;
    }
    let Some(tc) = &output.typeck else { return internal_error("loops: no checked program", err) };
    match render(&output, tc, a) {
        Ok(t) => {
            let _ = out.write_all(t.as_bytes());
            EXIT_OK
        }
        Err(e) => internal_error(&e.to_string(), err),
    }
}

/// The report text.
fn render(output: &pipeline::Output, tc: &stave_sema::Typeck, a: &Args) -> Result<String, LocateError> {
    let prog = &output.program;
    let loops = stave_sema::loops::report(prog, tc, |f| {
        a.all || prog.packages.get(f.package.idx()).is_some_and(|p| p.is_root)
    });
    let mut counts = [0usize; CLASSES.len()];
    let mut s = String::new();
    for l in &loops {
        let loc = output.files().locate(l.span)?;
        let (line, col) = (loc.line, loc.col);
        let file = report_file(prog, l.span.file);
        let (class, reason) = classify(l);
        counts[class as usize] += 1;
        let (form, class) = (l.form.name(), class.name());
        if a.json {
            let sources: Vec<String> = l
                .access
                .sources
                .iter()
                .map(|x| quote(&format!("{}{}", if x.written { "&mut " } else { "" }, x.place)))
                .collect();
            s.push_str(&format!(
                "{{\"file\":{},\"line\":{line},\"col\":{col},\"fn\":{},\"form\":\"{form}\",\
                 \"class\":\"{class}\",\"reason\":\"{reason}\",\"sources\":[{}],\"folds\":[]}}\n",
                quote(&file),
                quote(&l.func),
                sources.join(",")
            ));
        } else {
            let gap = if reason.is_empty() { "" } else { " " };
            s.push_str(&format!("{file}:{line}:{col} {} {form} {class}{gap}{reason}\n", l.func));
        }
    }
    let totals = CLASSES.iter().zip(counts);
    if a.json {
        let fields: Vec<String> = totals.map(|(c, n)| format!("\"{}\":{n}", c.name())).collect();
        s.push_str(&format!("{{\"summary\":true,{}}}\n", fields.join(",")));
    } else {
        let fields: Vec<String> = totals.map(|(c, n)| format!("{n} {}", c.name())).collect();
        s.push_str(&format!("loops: {}\n", fields.join(", ")));
    }
    Ok(s)
}
