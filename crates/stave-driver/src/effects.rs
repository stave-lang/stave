//! The effect pass in the pipeline ([`stave_sema::effects`]): it runs after borrowck on every
//! program without error diagnostics, in `stave check` and in builds alike, before lowering.
//!
//! The `trap` effect is read from the checked IR after the removal of proven bounds and contract
//! checks, which does not depend on the build mode. The pass builds that IR itself, only when a
//! check denies `trap` (or for the full report, `stave effects`) or a fn has a contract check,
//! and only for the nodes the checks reach and the fns with a contract check:
//! [`stave_sema::lower_graph_nodes`], the IR verifier, and the removal step of [`crate::build`],
//! the same code a build runs. Lowering an instance does not depend on the build mode or on which
//! other instances are lowered with it, so `stave check`, a debug build, and a release build see
//! the same trapping instructions.
//!
//! The same removal step decides the contract checks, so their diagnostics come from here, once
//! per program and the same in every mode: E0215 (a `requires` that is false whenever reached),
//! E0217 (an `ensures` false whenever a return is reached), and the warning E0216 (a `requires`
//! that the clauses before it imply; not in a generic fn, where it could hold for some instances
//! only). Each names the facts of its proof.

use rayon::prelude::*;
use stave_diag::{Diagnostic, Span};
use stave_ir::FuncId;
use stave_ir::bounds::{Clause, Finding};
use stave_ir::traps;
use stave_sema::effects::{self, EffectSets, NodeIr};
use stave_sema::{Program, Typeck};

use crate::build::{self, Failure};
use crate::pipeline::Stage;

/// The effect sets of a program and the violations of its deny clauses.
pub struct EffectRun {
    pub sets: EffectSets,
    /// E1401, E1403 (case `made`), E0211 (cases `impure-call`, `static`), E0215, E0216
    /// (a warning), E0217.
    pub diagnostics: Vec<Diagnostic>,
    /// Functions lowered for the `trap` effect and the contract checks (0: none were needed).
    pub lowered: usize,
}

/// Runs the effect pass on a checked program without error diagnostics. `all`: compute `trap`
/// from the IR of every node (the report), not only of what the checks reach.
pub fn run(prog: &Program, tc: &Typeck, all: bool) -> Result<EffectRun, Failure> {
    let plan = effects::plan(prog, tc);
    let contracts = effects::contract_nodes(tc);
    let mut lowered = 0;
    let mut diagnostics = Vec::new();
    let ir = if all || plan.needs_ir() || !contracts.is_empty() {
        let mut nodes = plan.ir_nodes(&tc.call_graph, all);
        nodes.extend(contracts);
        nodes.sort_unstable();
        nodes.dedup();
        let l = node_ir(prog, tc, &nodes)?;
        lowered = l.funcs;
        diagnostics = l.contracts;
        Some(l.ir)
    } else {
        None
    };
    let sets = effects::solve(prog, tc, ir.as_deref());
    diagnostics.extend(effects::check(prog, tc, &plan, &sets));
    diagnostics.extend(stave_sema::contracts::purity(prog, tc, &sets));
    diagnostics.extend(stave_sema::contracts::shared_statics(prog, tc));
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
    /// The diagnostics of the contract checks the removal step proved or refuted.
    pub contracts: Vec<Diagnostic>,
}

/// What the checked IR says about each graph node, from lowering `nodes`.
pub fn node_ir(prog: &Program, tc: &Typeck, nodes: &[u32]) -> Result<NodeIrs, Failure> {
    let g = &tc.call_graph;
    let (r, _) = stave_sema::lower_graph_nodes(prog, tc, nodes);
    let mut gl = r.map_err(|es| build::lower_failure(&es))?;
    build::verified(&gl.module, Stage::Lower)?;
    let (_, found) = build::remove_proven_checks(&mut gl.module)?;
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
        for &node in gl.nodes_of_func.get(f).map_or(&[][..], |v| v) {
            let Some(slot) = out.get_mut(node as usize) else { continue };
            covered[node as usize] = true;
            let at = g.nodes[node as usize].span;
            slot.trap = t.trap.map(|(s, _)| span_of(s, at));
            slot.out_of_memory = t.out_of_memory.map(|s| span_of(s, at));
        }
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
    let mut contracts = Vec::new();
    let mut seen: std::collections::HashSet<(&str, u32, u32, u32)> = std::collections::HashSet::new();
    for x in &found {
        let Some(func) = m.func(x.func) else { continue };
        let node = gl.nodes_of_func.get(x.func.index()).and_then(|ns| ns.first().copied());
        let fallback = node.map_or(Span::new(stave_diag::FileId(0), 0, 0), |n| g.nodes[n as usize].span);
        let name = match node {
            Some(n) => g.source_names(prog, tc, &[n]).swap_remove(0),
            None => func.path.clone(),
        };
        let generic = !func.type_args.is_empty();
        let Some((code, primary, clause)) = contract_diag(x, generic, |s| span_of(s, fallback)) else {
            continue;
        };
        if !seen.insert((code, primary.file.0, primary.offset, clause.map_or(u32::MAX, |c| c.offset))) {
            continue;
        }
        let mut roles: Vec<(&str, Span)> = clause.map(|c| ("clause", c)).into_iter().collect();
        roles.extend(x.facts.iter().map(|&s| ("fact", span_of(s, fallback))));
        let d = Diagnostic::new(code, None, &[("fn", name.as_str())], Some(primary), &roles)
            .map_err(|e| Failure::Internal { stage: Stage::Lower, message: e.to_string() })?;
        contracts.push(d);
    }
    Ok(NodeIrs { ir: out, funcs: m.funcs.len(), unmapped, contracts })
}

/// The code, primary span, and (E0217) clause span of the diagnostic of contract finding `x`, if
/// it has one: a proved `ensures` and a proved `requires` of a generic fn have none.
fn contract_diag(
    x: &Finding,
    generic: bool,
    span: impl Fn(stave_ir::Site) -> Span,
) -> Option<(&'static str, Span, Option<Span>)> {
    match (x.clause, x.refuted) {
        (Clause::Requires, true) => Some(("E0215", span(x.site), None)),
        (Clause::Requires, false) if !generic => Some(("E0216", span(x.site), None)),
        (Clause::Ensures, true) => {
            let ret = x.ret.map_or(span(x.site), &span);
            Some(("E0217", ret, Some(span(x.site))))
        }
        _ => None,
    }
}
