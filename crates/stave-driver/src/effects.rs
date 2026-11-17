//! The effect pass in the pipeline ([`stave_sema::effects`]): it runs after borrowck on every
//! program without error diagnostics, in `stave check` and in builds alike, before lowering.
//!
//! The `trap` effect is read from the checked IR after the removal of proven bounds checks,
//! which does not depend on the build mode. The pass builds that IR itself, only when a check
//! denies `trap` (or for the full report, `stave effects`), and only for the nodes the checks
//! reach: [`stave_sema::lower_graph_nodes`], the IR verifier, and the removal step of
//! [`crate::build`], the same code a build runs. Lowering an instance does not depend on the
//! build mode or on which other instances are lowered with it, so `stave check`, a debug build,
//! and a release build see the same trapping instructions.

use rayon::prelude::*;
use stave_diag::{Diagnostic, Span};
use stave_ir::FuncId;
use stave_ir::traps;
use stave_sema::effects::{self, EffectSets, NodeIr};
use stave_sema::{Program, Typeck};

use crate::build::{self, Failure};
use crate::pipeline::Stage;

/// The effect sets of a program and the violations of its deny clauses.
pub struct EffectRun {
    pub sets: EffectSets,
    /// E1401, E1403 (case `made`).
    pub diagnostics: Vec<Diagnostic>,
    /// Functions lowered for the `trap` effect (0: none were needed).
    pub lowered: usize,
}

/// Runs the effect pass on a checked program without error diagnostics. `all`: compute `trap`
/// from the IR of every node (the report), not only of what the checks reach.
pub fn run(prog: &Program, tc: &Typeck, all: bool) -> Result<EffectRun, Failure> {
    let plan = effects::plan(prog, tc);
    let mut lowered = 0;
    let ir = if all || plan.needs_ir() {
        let nodes = plan.ir_nodes(&tc.call_graph, all);
        let l = node_ir(prog, tc, &nodes)?;
        lowered = l.funcs;
        Some(l.ir)
    } else {
        None
    };
    let sets = effects::solve(prog, tc, ir.as_deref());
    let diagnostics = effects::check(prog, tc, &plan, &sets);
    Ok(EffectRun { sets, diagnostics, lowered })
}

/// The checked IR of a set of call-graph nodes, read for the effect pass.
pub struct NodeIrs {
    /// Per graph node.
    pub ir: Vec<NodeIr>,
    /// Functions lowered.
    pub funcs: usize,
    /// Nodes of the set with a body that no lowered function maps back to (none is expected);
    /// each traps at its name, so the answer stays sound.
    pub unmapped: Vec<u32>,
}

/// What the checked IR says about each graph node, from lowering `nodes`.
pub fn node_ir(prog: &Program, tc: &Typeck, nodes: &[u32]) -> Result<NodeIrs, Failure> {
    let g = &tc.call_graph;
    let (r, _) = stave_sema::lower_graph_nodes(prog, tc, nodes);
    let mut gl = r.map_err(|es| build::lower_failure(&es))?;
    build::verified(&gl.module, Stage::Lower)?;
    build::remove_proven_checks(&mut gl.module)?;
    let m = &gl.module;
    let scans: Vec<traps::Traps> =
        (0..m.funcs.len() as u32).into_par_iter().map(|f| traps::scan(m, FuncId(f))).collect();
    let span_of = |s: stave_ir::Site, fallback: Span| -> Span {
        let Some(&(fid, off)) = gl.site_pos.get(s.0 as usize) else { return fallback };
        let Some(f) = prog.file(fid) else { return fallback };
        let toks = &f.parsed.tokens;
        let i = toks.partition_point(|t| t.start < off);
        let end = stave_syntax::token_end(&f.bytes, toks, i);
        Span::new(fid, off, end.saturating_sub(off))
    };
    let mut out = vec![NodeIr::default(); g.nodes.len()];
    let mut covered = vec![false; g.nodes.len()];
    for (f, t) in scans.iter().enumerate() {
        let node = gl.node_of_func.get(f).copied().unwrap_or(u32::MAX);
        let Some(slot) = out.get_mut(node as usize) else { continue };
        covered[node as usize] = true;
        let at = g.nodes[node as usize].span;
        slot.trap = t.trap.map(|(s, _)| span_of(s, at));
        slot.out_of_memory = t.out_of_memory.map(|s| span_of(s, at));
    }
    let mut unmapped = Vec::new();
    for &n in nodes {
        let has_body = match &g.nodes[n as usize].kind {
            stave_sema::callgraph::NodeKind::Fn { body, .. } => *body,
            stave_sema::callgraph::NodeKind::Closure { .. } => true,
            stave_sema::callgraph::NodeKind::Table { .. } | stave_sema::callgraph::NodeKind::Glue { .. } => {
                false
            }
        };
        if has_body && !covered[n as usize] {
            out[n as usize].trap = Some(g.nodes[n as usize].span);
            unmapped.push(n);
        }
    }
    Ok(NodeIrs { ir: out, funcs: m.funcs.len(), unmapped })
}
