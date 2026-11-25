//! `stave ir check|fmt|opt|run|build FILE` (contracts/cli.md): the verbs that load IR text
//! ([`stave_driver::ir_cmd`]). Every verb loads FILE first; a rejected file is exit 1, a file
//! that cannot be read exit 2. No verb prints a summary, stack, or `emulated:` line.

use std::io::Write;
use std::path::{Path, PathBuf};

use stave_driver::ir_cmd::{self, IrError, LoadedIr};

use crate::args::{Args, IrVerb};
use crate::{EXIT_OK, EXIT_USAGE, output_file_name, threads_from_env};

pub fn run(a: &Args, verb: IrVerb, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let Some(file) = a.pkg_dir.as_deref() else {
        let _ = writeln!(err, "stave: `stave ir {}` needs a FILE", verb.name());
        return EXIT_USAGE;
    };
    match verb_on(a, verb, file, out, err) {
        Ok(code) => code,
        Err(e) => {
            for l in e.lines() {
                let _ = writeln!(err, "{l}");
            }
            e.exit()
        }
    }
}

fn verb_on(
    a: &Args,
    verb: IrVerb,
    file: &Path,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32, IrError> {
    let src = std::fs::read(file).map_err(|e| IrError::Io(format!("cannot read {}: {e}", file.display())))?;
    let name = file.display().to_string();
    let l = ir_cmd::load(&name, &src)?;
    let print = |out: &mut dyn Write, text: &str| {
        out.write_all(text.as_bytes()).map_err(|e| IrError::Io(format!("cannot write the output: {e}")))
    };
    match verb {
        IrVerb::Check => Ok(EXIT_OK),
        IrVerb::Fmt => print(out, &ir_cmd::fmt(&l)?).map(|()| EXIT_OK),
        IrVerb::Opt => print(out, &ir_cmd::opt(&l)?).map(|()| EXIT_OK),
        IrVerb::Build => build(a, file, &l),
        IrVerb::Run => {
            let args: Vec<&[u8]> = a.run_args.iter().map(|s| s.as_bytes()).collect();
            let r = ir_cmd::run(&l, &args)?;
            out.write_all(&r.ran.stdout).map_err(|e| IrError::Io(format!("cannot write the output: {e}")))?;
            let _ = err.write_all(&r.ran.stderr);
            Ok(r.ran.exit)
        }
        IrVerb::Emit => Err(IrError::Io("`stave ir emit` compiles a package, not IR text".to_string())),
    }
}

/// `build [-o OUT]`: the image to OUT, default `./<file stem>` (refused as `stave build` refuses a
/// default name that is not one plain path component).
fn build(a: &Args, file: &Path, l: &LoadedIr) -> Result<i32, IrError> {
    let threads = threads_from_env(std::env::var("STAVE_THREADS").ok().as_deref()).map_err(IrError::Io)?;
    let path = match &a.out {
        Some(p) => p.clone(),
        None => {
            let stem =
                file.file_stem().map_or_else(|| "a.out".to_string(), |s| s.to_string_lossy().into_owned());
            PathBuf::from(output_file_name(&stem).map_err(|e| IrError::Io(format!("file stem {e}")))?)
        }
    };
    let built = ir_cmd::build(l, threads)?;
    built
        .image
        .write_file(&path)
        .map_err(|e| IrError::Io(format!("cannot write {}: {e}", path.display())))?;
    Ok(EXIT_OK)
}
