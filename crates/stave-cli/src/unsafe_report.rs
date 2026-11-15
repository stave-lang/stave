//! `stave unsafe-report [PKG_DIR] [--json]` (contracts/cli.md): the unsafe
//! inventory of the whole program ([`stave_sema::unsafety::inventory`]) on stdout, one site per line
//! in canonical (file, offset) order. Always stage0 (`--compiler` is accepted and ignored).
//!
//! - human: `<file>:<line>:<col>: <kind> in <fn>: "<obligation>"`
//! - `--json`: one object per line,
//!   `{"file":"src/main.stv","line":3,"col":17,"fn":"main.read_raw","kind":"expr","obligation":"..."}`
//!
//! `file` is the path diagnostics use (relative to the root package directory) for files inside the
//! root package's directory, else `<package>/<path in the package>` (`std/core.stv`,
//! `runtime/alloc.stv`: independent of the install location); `line`/`col` are 1-based
//! (col in bytes) at the `unsafe` keyword (an `extern-call` site: the callee name), `kind` is
//! `block expr fn trait impl extern-call`, strings are JSON-escaped in both forms. The program is
//! checked through type checking first: with error diagnostics they are printed on stderr (`--json`:
//! as JSON lines), nothing goes to stdout, and the exit status is 1.

use std::io::Write;

use stave_diag::json_string as quote;
use stave_driver::package;
use stave_driver::pipeline::{self, Goal, Options};
use stave_sema::unsafety::{self, UnsafeSite};

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
    let Some(tc) = &output.typeck else {
        return internal_error("unsafe-report: no type-checked program", err);
    };
    let sites = unsafety::inventory(&output.program, tc);
    match render(&sites, output.files(), a.json) {
        Ok(text) => {
            let _ = out.write_all(text.as_bytes());
            EXIT_OK
        }
        Err(e) => internal_error(&e.to_string(), err),
    }
}

/// The report text.
pub fn render(
    sites: &[UnsafeSite],
    files: &stave_diag::SourceFiles,
    json: bool,
) -> Result<String, stave_diag::LocateError> {
    let mut s = String::new();
    for site in sites {
        let loc = files.locate(site.span)?;
        if json {
            s.push_str(&format!(
                "{{\"file\":{},\"line\":{},\"col\":{},\"fn\":{},\"kind\":\"{}\",\"obligation\":{}}}\n",
                quote(&site.file),
                loc.line,
                loc.col,
                quote(&site.func),
                site.kind.name(),
                quote(&site.obligation)
            ));
        } else {
            s.push_str(&format!(
                "{}:{}:{}: {} in {}: {}\n",
                site.file,
                loc.line,
                loc.col,
                site.kind.name(),
                site.func,
                quote(&site.obligation)
            ));
        }
    }
    Ok(s)
}
