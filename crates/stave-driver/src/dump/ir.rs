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

use stave_ir::text::{EdgeKind, Header, Level, Mode, StackEdge, StackInput};
use stave_sema::callgraph::EdgeKind as GraphEdge;

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
    match (level, &out.module) {
        (Level::Lowered, Some(_)) => {
            let m = out.lowered.as_ref().ok_or(DumpError::NotRun(stage))?;
            Ok(stave_ir::print_module(m, &header, find_main(out, m), None))
        }
        (Level::Checked, Some(m)) => Ok(stave_ir::print_module(m, &header, out.main, stack.as_ref())),
        (Level::Opt, Some(m)) => {
            if !out.release {
                return Err(DumpError::NotRun(stage));
            }
            let m = crate::build::release_passes(m).map_err(|f| DumpError::Internal(f.to_string()))?;
            Ok(stave_ir::print_module(&m, &header, out.main, None))
        }
        _ => Err(DumpError::NotRun(stage)),
    }
}

fn mode(out: &Output) -> Mode {
    if out.release { Mode::Release } else { Mode::Debug }
}

/// The entry of the module at level `lowered`: the fn `main` is, by path, as in the checked one.
fn find_main(out: &Output, m: &stave_ir::Module) -> Option<stave_ir::FuncId> {
    let path = crate::build::main_path(&out.program)?;
    m.find_func(&path)
}

/// The stack section of `out`: the call graph and the run stack of the front end, the nodes each
/// fn stands for (those that are nodes of the graph, ascending).
pub fn stack_input(out: &Output) -> Option<StackInput> {
    let g = &out.typeck.as_ref()?.call_graph;
    let n = g.nodes.len();
    let mut edges = Vec::new();
    for from in 0..n as u32 {
        for e in g.succ(from) {
            let kind = match e.kind {
                GraphEdge::Spawn => EdgeKind::Spawn,
                GraphEdge::Indirect => EdgeKind::Indirect,
                _ => EdgeKind::Call,
            };
            edges.push(StackEdge { from, to: e.to, kind });
        }
    }
    let nodes_of_func = out
        .nodes_of_func
        .iter()
        .map(|v| {
            let mut v: Vec<u32> = v.iter().copied().filter(|&x| (x as usize) < n).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    Some(StackInput {
        run_stack: out.program.packages.iter().find(|p| p.is_root).and_then(|p| p.run_stack),
        truncated: g.truncated,
        nodes: g.nodes.iter().map(|x| x.rec()).collect(),
        edges,
        nodes_of_func,
    })
}
