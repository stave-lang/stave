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
//! E0217 (an `ensures` false whenever a return is reached), E0218 (a call that always fails a
//! `requires` of its callee), and the warning E0216 (a `requires` that the clauses before it
//! imply). None in an instance of a generic fn, where the answer may depend on the instance. Each
//! names the facts of its proof. For the calls, the pass also lowers every node that calls a fn
//! with a checked `requires`.
//!
//! A `requires` check that stays is the entry effect of its fn, not a trap of its body: the pass
//! hands [`stave_sema::effects::solve_with`] the call edges whose every call the witness checker
//! retargeted to the callee's unchecked entry, and only the other edges carry it.

use rayon::prelude::*;
use stave_diag::{Diagnostic, Span};
use stave_ir::bounds::{CallFinding, Clause, Finding};
use stave_ir::traps;
use stave_ir::{FuncId, Module};
use stave_sema::effects::{self, EffectSets, NodeIr};
use stave_sema::{Program, Typeck};

use crate::build::{self, Failure};
use crate::pipeline::Stage;

/// The effect sets of a program and the violations of its deny clauses.
pub struct EffectRun {
    pub sets: EffectSets,
    /// E1401, E1403 (case `made`), E0211 (cases `impure-call`, `static`), E0215, E0216
    /// (a warning), E0217, E0218.
    pub diagnostics: Vec<Diagnostic>,
    /// Functions lowered for the `trap` effect and the contract checks (0: none were needed).
    pub lowered: usize,
}

/// Runs the effect pass on a checked program without error diagnostics. `all`: compute `trap`
/// from the IR of every node (the report), not only of what the checks reach.
pub fn run(
    prog: &Program,
    tc: &Typeck,
    all: bool,
    drops: Option<&stave_sema::drops::Schedules>,
) -> Result<EffectRun, Failure> {
    let plan = effects::plan(prog, tc);
    let contracts = effects::contract_nodes(tc);
    let mut lowered = 0;
    let mut diagnostics = Vec::new();
    let mut proved = Vec::new();
    let traps = all || plan.needs_ir();
    let ir = if traps || !contracts.is_empty() {
        let planned = plan.ir_nodes(&tc.call_graph, all);
        let mut nodes = planned.clone();
        nodes.extend(&contracts);
        nodes.sort_unstable();
        nodes.dedup();
        match node_ir(prog, tc, &nodes, drops) {
            Ok(l) => {
                lowered = l.funcs;
                diagnostics = l.contracts;
                proved = l.proved;
                Some(l.ir)
            }
            // A fn lowered only for its contract checks may use what stage0 cannot lower yet (or
            // hit a lowering bug): `check` stays as it was without those checks, and a build
            // reports the failure where it lowers the fn. Each such fn alone still gets its
            // contract diagnostics when it lowers.
            Err(_) if !contracts.is_empty() => {
                let mut seen: Vec<(&'static str, Option<Span>, String)> = Vec::new();
                for &c in &contracts {
                    let Ok(l) = node_ir(prog, tc, &[c], drops) else { continue };
                    lowered += l.funcs;
                    for d in l.contracts {
                        let key = (d.code(), d.primary(), d.message().to_string());
                        if !seen.contains(&key) {
                            seen.push(key);
                            diagnostics.push(d);
                        }
                    }
                }
                if traps {
                    let l = node_ir(prog, tc, &planned, drops)?;
                    lowered += l.funcs;
                    proved = l.proved;
                    Some(l.ir)
                } else {
                    None
                }
            }
            Err(f) => return Err(f),
        }
    } else {
        None
    };
    let sets = effects::solve_with(prog, tc, ir.as_deref(), &proved);
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
    /// The diagnostics of the contract checks the removal step proved or refuted, and of the
    /// calls it refuted.
    pub contracts: Vec<Diagnostic>,
    /// (caller, callee) graph-node pairs whose every call proves the callee's `requires`
    /// ([`effects::solve_with`]). Sorted.
    pub proved: Vec<(u32, u32)>,
}

/// What the checked IR says about each graph node, from lowering `nodes`.
pub fn node_ir(
    prog: &Program,
    tc: &Typeck,
    nodes: &[u32],
    drops: Option<&stave_sema::drops::Schedules>,
) -> Result<NodeIrs, Failure> {
    let g = &tc.call_graph;
    let (r, _) = stave_sema::lower_graph_nodes(prog, tc, nodes, drops);
    let mut gl = r.map_err(|es| build::lower_failure(&es))?;
    build::verified(&gl.module, Stage::Lower)?;
    // The removal reads canonical modules, as the build's does ([`build::lower`]).
    let remap = build::canonical(&mut gl.module);
    stave_ir::canon::permute_by_func(&mut gl.nodes_of_func, &remap);
    // The module is a part of the program: every function keeps its checked entry, so the entry
    // effect read below does not depend on which callers were lowered with it.
    let removed = build::remove_proven_checks(&mut gl.module, false)?;
    let (found, removal) = (&removed.found, &removed.removal);
    let m = &gl.module;
    // Per function: its entry instructions, which a call that proves its `requires` cannot trap
    // at: the `requires` checks that stay, and what the witness checker listed.
    let mut quiet: Vec<Vec<u32>> =
        (0..m.funcs.len() as u32).into_par_iter().map(|f| traps::requires_checks(m, FuncId(f))).collect();
    for (f, insts) in &removal.entry {
        if let Some(q) = quiet.get_mut(f.index()) {
            q.extend(insts);
            q.sort_unstable();
            q.dedup();
        }
    }
    let scans: Vec<traps::Traps> = (0..m.funcs.len() as u32)
        .into_par_iter()
        .map(|f| traps::scan_without(m, FuncId(f), &quiet[f as usize]))
        .collect();
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
            slot.entry = traps::first_trap(m, FuncId(f as u32), &quiet[f]).map(|s| span_of(s, at));
        }
    }
    let nodes_of = |f: FuncId| gl.nodes_of_func.get(f.index()).map_or(&[][..], |v| v);
    let mut proved: Vec<(u32, u32)> = Vec::new();
    for &(caller, callee) in &removal.proved {
        for &x in nodes_of(caller) {
            proved.extend(nodes_of(callee).iter().map(|&y| (x, y)));
        }
    }
    proved.sort_unstable();
    proved.dedup();
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
    for x in found {
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
    // E0218: a call that always fails a `requires` of its callee, at the call, with the
    // conjunct. None in an instance of a generic fn (as E0215), nor in a closure or `spawn` body
    // of one: such a body has no generic arguments of its own, its instance id starts with the
    // enclosing fn's.
    // None either for a call of a fn that already has E0215: no call of it can pass.
    let uncallable = uncallable(m, found);
    for x in &removed.calls {
        if !call_reported(m, &uncallable, x) {
            continue;
        }
        let fallback = nodes_of(x.func)
            .first()
            .map_or(Span::new(stave_diag::FileId(0), 0, 0), |&n| g.nodes[n as usize].span);
        let name = match nodes_of(x.callee).first() {
            Some(&n) => g.source_names(prog, tc, &[n]).swap_remove(0),
            None => m.func(x.callee).map_or(String::new(), |f| f.path.clone()),
        };
        let primary = span_of(x.site, fallback);
        let clause = (x.clause != stave_ir::Site::NONE).then(|| span_of(x.clause, fallback));
        if !seen.insert(("E0218", primary.file.0, primary.offset, clause.map_or(u32::MAX, |c| c.offset))) {
            continue;
        }
        let mut roles: Vec<(&str, Span)> = clause.map(|c| ("clause", c)).into_iter().collect();
        roles.extend(x.facts.iter().map(|&s| ("fact", span_of(s, fallback))));
        let d = Diagnostic::new("E0218", None, &[("fn", name.as_str())], Some(primary), &roles)
            .map_err(|e| Failure::Internal { stage: Stage::Lower, message: e.to_string() })?;
        contracts.push(d);
    }
    Ok(NodeIrs { ir: out, funcs: m.funcs.len(), unmapped, contracts, proved })
}

/// The fns with a `requires` that fails whenever it is reached (E0215): no call of them can pass.
pub(crate) fn uncallable(m: &Module, found: &[Finding]) -> Vec<FuncId> {
    found
        .iter()
        .filter(|x| x.refuted && x.clause == Clause::Requires)
        .map(|x| x.func)
        .filter(|&f| m.func(f).is_some_and(|f| f.type_args.is_empty()))
        .collect()
}

/// Whether refuted call `x` is reported (E0218): not in an instance of a generic fn (as E0215),
/// nor in a closure or `spawn` body of one (such a body has no generic arguments of its own, its
/// instance id starts with the enclosing fn's), nor a call of an `uncallable` fn.
pub(crate) fn call_reported(m: &Module, uncallable: &[FuncId], x: &CallFinding) -> bool {
    let Some(func) = m.func(x.func) else { return false };
    let root = func.instance.0.first();
    let encloser_generic = m
        .funcs
        .iter()
        .any(|g| g.instance.0.len() == 1 && g.instance.0.first() == root && !g.type_args.is_empty());
    !uncallable.contains(&x.callee) && func.type_args.is_empty() && !encloser_generic
}

/// The code, primary span, and (E0217) clause span of the diagnostic of contract finding `x`, if
/// it has one. None in an instance of a generic fn: what holds there may depend on the instance,
/// and the instance may be called only where its caller knows the check holds. A proved `ensures`
/// has none, nor a proved `requires` whose own evaluation can trap or calls a fn (removing it
/// from the source would remove that trap).
pub(crate) fn contract_diag(
    x: &Finding,
    generic: bool,
    span: impl Fn(stave_ir::Site) -> Span,
) -> Option<(&'static str, Span, Option<Span>)> {
    if generic {
        return None;
    }
    match (x.clause, x.refuted) {
        (Clause::Requires, true) => Some(("E0215", span(x.site), None)),
        (Clause::Requires, false) if !x.traps => Some(("E0216", span(x.site), None)),
        (Clause::Ensures, true) => {
            let ret = x.ret.map_or(span(x.site), &span);
            Some(("E0217", ret, Some(span(x.site))))
        }
        _ => None,
    }
}
