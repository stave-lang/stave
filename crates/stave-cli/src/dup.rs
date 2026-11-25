//! `stave dup [PKG_DIR] [--target T] [--all] [--min-size N] [--json | --baseline FILE |
//! --write-baseline FILE]` (contracts/cli.md): checks the program as `stave check` does (always
//! stage0; the target selects the arch packages and which fns are built, default the host
//! baseline),
//! then prints the classes of fns with the same code and the re-declared consts
//! ([`stave_driver::dup`]). `--baseline FILE` is the gate (contracts/cli.md `dup-gate`):
//! one line per new duplicate on stdout and exit 1 when a class is not in FILE; `--write-baseline
//! FILE` writes the current classes to FILE.

use std::io::Write;

use stave_driver::build::Failure;
use stave_driver::dup;
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options};

use crate::args::Args;
use crate::{EXIT_OK, EXIT_USAGE, EXIT_USER, internal_error, library, pkg_path, threads_from_env};

pub(crate) fn run(a: &Args, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let path = pkg_path(a);
    if let Err(e) = package::check_root(&path) {
        let _ = writeln!(err, "stave: {e}");
        return EXIT_USAGE;
    }
    let baseline = match &a.baseline {
        Some(p) => {
            match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| dup::parse_baseline(&t))
            {
                Ok(b) => Some(b),
                Err(e) => {
                    let _ = writeln!(err, "stave: baseline {}: {e}", p.display());
                    return EXIT_USAGE;
                }
            }
        }
        None => None,
    };
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
    opts.target = Some(a.target.triple);
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
    let dopts = dup::Options { all: a.all, min_size: a.min_size.unwrap_or(dup::DEFAULT_MIN_SIZE) };
    let report = match dup::report(&output, &dopts) {
        Ok(r) => r,
        Err(f @ Failure::Internal { .. }) => return internal_error(&f.to_string(), err),
        Err(f) => {
            let _ = writeln!(err, "stave: {f}");
            return EXIT_USAGE;
        }
    };
    if let Some(p) = &a.write_baseline {
        if let Err(e) = std::fs::write(p, dup::baseline_text(&report)) {
            let _ = writeln!(err, "stave: cannot write {}: {e}", p.display());
            return EXIT_USAGE;
        }
        return EXIT_OK;
    }
    if let Some(b) = &baseline {
        let v = dup::check(&report, b);
        for l in &v.new {
            let _ = writeln!(out, "{l}");
        }
        if v.stale > 0 {
            let _ = writeln!(
                err,
                "dup: {} baseline classes no longer occur; removing them is allowed (--write-baseline)",
                v.stale
            );
        }
        if !v.new.is_empty() {
            let _ = writeln!(out, "dup: {} new duplicates not in the baseline", v.new.len());
            return EXIT_USER;
        }
        return EXIT_OK;
    }
    let text = if a.json { dup::render_json(&report) } else { dup::render_human(&report) };
    let _ = out.write_all(text.as_bytes());
    EXIT_OK
}
