//! `stave effects [PKG_DIR] [--json]` (contracts/cli.md): the effect set of every fn and closure
//! of the program's own packages (the root package and its path dependencies; generic fns per
//! instance; a fn the call graph specializes on the fn values it is called with is one line, the
//! union over the nodes the program uses), one per line in canonical (file, offset) order, then
//! by name. Always stage0.
//!
//! - human: `<file>:<line>:<col>: <fn>: <effects>` (effects space-separated in the order
//!   `alloc trap recursive lock unsafe io block park`, or `none`)
//! - `--json`: one object per line,
//!   `{"file":"src/main.stv","line":3,"col":4,"fn":"main.g","effects":["alloc","trap"]}`
//!
//! `file`, `line`, `col` as in `unsafe-report`, at the fn's name (a closure: its `fn` token).
//! Error diagnostics before the effect pass go to stderr, nothing to stdout, exit 1. Deny-clause
//! violations (E1401, E1403) go to stderr too, after the report, and the exit status is 1.

use std::io::Write;

use stave_diag::json_string as quote;
use stave_driver::build::Failure;
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options};
use stave_sema::callgraph::NodeKind;
use stave_sema::effects::Effects;

use crate::args::Args;
use crate::{EXIT_OK, EXIT_USAGE, EXIT_USER, internal_error, library, pkg_path, threads_from_env};

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
    let mut opts = Options::new(Goal::Check).threads(threads);
    opts.effects_all = true;
    let output = match pipeline::run(&path, &lib, &opts) {
        Ok(o) => o,
        Err(e) => {
            let _ = writeln!(err, "stave: {e}");
            return EXIT_USAGE;
        }
    };
    if let Some(f) = &output.failure
        && !output.has_errors()
    {
        if matches!(f, Failure::Internal { .. }) {
            return internal_error(&f.to_string(), err);
        }
        let _ = writeln!(err, "stave: {f}");
        return EXIT_USAGE;
    }
    let report = match (&output.typeck, &output.effects) {
        (Some(tc), Some(sets)) => match render(&output, tc, sets, a.json) {
            Ok(t) => Some(t),
            Err(e) => return internal_error(&e.to_string(), err),
        },
        _ => None,
    };
    if let Some(r) = &report {
        let _ = out.write_all(r.as_bytes());
    }
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
    if report.is_none() {
        return internal_error("effects: no effect sets", err);
    }
    EXIT_OK
}

/// A report's name of file `f`: the diagnostics path, or `<package>/<path in the package>` for a
/// file the diagnostics name by an install path.
pub(crate) fn report_file(prog: &stave_sema::Program, f: stave_diag::FileId) -> String {
    prog.file(f).map_or(String::new(), |f| {
        if f.display_path.starts_with('/') {
            let pkg = prog.packages.get(f.package.0 as usize).map_or("", |p| p.name.as_str());
            format!("{pkg}/{}", f.rel_path)
        } else {
            f.display_path.clone()
        }
    })
}

/// The report text.
fn render(
    output: &pipeline::Output,
    tc: &stave_sema::Typeck,
    sets: &stave_sema::effects::EffectSets,
    json: bool,
) -> Result<String, stave_diag::LocateError> {
    let prog = &output.program;
    let g = &tc.call_graph;
    let own = |n: u32| -> bool {
        let Some(f) = g.fn_of(n) else { return false };
        let NodeKind::Fn { inst, .. } = &g.nodes[f as usize].kind else { return false };
        prog.decl(inst.decl)
            .and_then(|d| prog.module(d.module))
            .and_then(|m| prog.packages.get(m.package.idx()))
            .is_some_and(|p| p.kind != stave_sema::PackageKind::Builtin)
    };
    // one row per source fn or closure: the union over the call-graph nodes the program uses
    let groups: Vec<(u32, Effects)> = stave_sema::effects::by_source(g, sets)
        .into_iter()
        .filter(|(ns, _)| own(ns[0]))
        .map(|(ns, e)| (ns[0], e))
        .collect();
    let nodes: Vec<u32> = groups.iter().map(|x| x.0).collect();
    let names = g.source_names(prog, tc, &nodes);
    let mut rows = Vec::with_capacity(nodes.len());
    for (k, &(n, effects)) in groups.iter().enumerate() {
        let span = g.nodes[n as usize].span;
        let loc = output.files().locate(span)?;
        let file = report_file(prog, span.file);
        rows.push(((span.file, span.offset), names[k].clone(), file, loc.line, loc.col, effects));
    }
    rows.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    let mut s = String::new();
    for (_, name, file, line, col, e) in rows {
        if json {
            let list: Vec<String> = e.names().iter().map(|x| format!("\"{x}\"")).collect();
            s.push_str(&format!(
                "{{\"file\":{},\"line\":{line},\"col\":{col},\"fn\":{},\"effects\":[{}]}}\n",
                quote(&file),
                quote(&name),
                list.join(",")
            ));
        } else {
            let list = if e == Effects::NONE { "none".to_string() } else { e.names().join(" ") };
            s.push_str(&format!("{file}:{line}:{col}: {name}: {list}\n"));
        }
    }
    Ok(s)
}
