//! `lir`, `ir`, `opt`: the IR text of the program at level `lowered`, `checked`, and `opt`
//! (contracts/checked-ir.md, stage-gates.md): [`stave_ir::print_module`], the one printer, on the
//! canonical module of that point; and the emit form of `stave ir emit`.
//!
//! Printed only for a build with no error diagnostics: with errors there is no IR, this refuses
//! ([`DumpError::NotRun`]), and [`super::write_dumps`] writes no file (an old one is removed). A
//! run that did not lower (a plain `stave check` without `Options::lower`) or whose lowering
//! failed has no module and refuses too. `lir` also needs [`crate::pipeline::Options::keep_lowered`],
//! and `opt` a `--release` run.

use std::io::Write;
use std::path::PathBuf;

use stave_ir::text::{Header, Level, Mode, StackInput};

use super::{DumpError, DumpStage};
use crate::pipeline::Output;

/// The stage of `level`'s dump.
fn stage_of(level: Level) -> DumpStage {
    match level {
        Level::Lowered => DumpStage::Lir,
        Level::Checked => DumpStage::Ir,
        Level::Opt => DumpStage::Opt,
    }
}

pub(super) fn write(level: Level, out: &Output, w: &mut dyn Write) -> Result<(), DumpError> {
    let stage = stage_of(level);
    let text = text(level, out, false)?;
    w.write_all(text.as_bytes()).map_err(|e| DumpError::Io(PathBuf::from(stage.file_name()), e))
}

/// The IR text of `out` at `level`: the gate form, or with `emit` the emit form (level `checked`
/// only: the stack section is the front end's data of the checked module).
pub fn text(level: Level, out: &Output, emit: bool) -> Result<String, DumpError> {
    let stage = stage_of(level);
    if out.has_errors() {
        return Err(DumpError::NotRun(stage));
    }
    let header = Header { target: out.program.target, mode: mode(out), level };
    let stack = (emit && level == Level::Checked).then(|| stack_input(out)).flatten();
    let print = |m: &stave_ir::Module, stack: Option<&StackInput>| {
        stave_ir::print_module(m, &header, stack).map_err(|es| {
            let first = es.first().map_or_else(String::new, ToString::to_string);
            DumpError::Internal(format!("IR verifier: {first}"))
        })
    };
    match (level, &out.module) {
        (Level::Lowered, Some(_)) => print(out.lowered.as_ref().ok_or(DumpError::NotRun(stage))?, None),
        (Level::Checked, Some(m)) => print(m, stack.as_ref()),
        (Level::Opt, Some(m)) => {
            if !out.release {
                return Err(DumpError::NotRun(stage));
            }
            let m = crate::build::release_passes(m).map_err(|f| DumpError::Internal(f.to_string()))?;
            print(&m, None)
        }
        _ => Err(DumpError::NotRun(stage)),
    }
}

fn mode(out: &Output) -> Mode {
    if out.release { Mode::Release } else { Mode::Debug }
}

/// The stack section of `out` ([`crate::stack::input`]): what its build's stack bound read.
pub fn stack_input(out: &Output) -> Option<StackInput> {
    let g = &out.typeck.as_ref()?.call_graph;
    let run_stack = out.program.packages.iter().find(|p| p.is_root).and_then(|p| p.run_stack);
    Some(crate::stack::input(g, &out.nodes_of_func, run_stack))
}
