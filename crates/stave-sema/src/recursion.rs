//! The recursion check on the call graph ([`crate::callgraph`]): every fn of a call cycle is a
//! `rec fn` (E1402), and no cycle goes through a closure literal (E1409).
//!
//! A cycle is a strongly connected component with more than one node or a self-edge. Exempt: a
//! drop cycle ([`CallGraph::is_drop_cycle`]), the recursion of dropping a recursive value. A
//! component with a closure body is E1409; else one with a fn instance not declared `rec` is
//! E1402, where on a component through drop glue only the user-written fns need the marker (the
//! built-in fns there, `Vec.drop` and `Vec.clear` of a user type, cannot carry it). One
//! diagnostic per component, at the component's first closure (smallest node id) or its unmarked
//! fn with the smallest name; the message lists a shortest cycle from there (breadth first within
//! the component, out-edges in node order), tables and the `Member` step hidden: a call through a
//! fn value shows as a call of the value it reaches. Instances of one generic that repeat a cycle
//! at the same call sites are reported once. Names are definition paths, with the type args of a
//! generic instance.

use rustc_hash::FxHashSet;
use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::callgraph::{CallGraph, NodeKind};
use crate::diag::mk;
use crate::sigs::{NameCx, Sigs};
use crate::types::{GArg, TypeTable};

/// The E1402 / E1409 diagnostics of `g`.
pub(crate) fn check(prog: &Program, sigs: &Sigs, table: &TypeTable, g: &CallGraph) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let mut reported: FxHashSet<(&'static str, Span)> = FxHashSet::default();
    let cx = g.types(table);
    let names = NameCx { prog, generics: &[] };
    let name = |n: u32| -> String {
        match &g.nodes[n as usize].kind {
            NodeKind::Fn { inst, .. } => {
                let mut s = crate::lower::fn_path(prog, sigs, inst.decl);
                if !inst.args.is_empty() {
                    let args: Vec<String> = inst
                        .args
                        .iter()
                        .map(|a| match a {
                            GArg::Type(t) => crate::types::display(&cx, &names, *t),
                            GArg::Const(c) => c.value().map_or_else(|| "_".to_string(), |v| v.to_string()),
                        })
                        .collect();
                    s.push('<');
                    s.push_str(&args.join(", "));
                    s.push('>');
                }
                s
            }
            NodeKind::Closure { parent, .. } => {
                let p = g.fn_of(*parent).map_or(String::new(), |p| def_name(prog, sigs, g, p));
                format!("closure in {p}")
            }
            NodeKind::Glue { ty } => format!("drop of {}", crate::types::display(&cx, &names, *ty)),
            NodeKind::Table { .. } => "fn value".to_string(),
        }
    };
    for s in 0..g.scc_count() as u32 {
        if !g.is_cycle(s) || g.is_drop_cycle(s) {
            continue;
        }
        let members = g.scc(s);
        let closure =
            members.iter().copied().find(|&n| matches!(g.nodes[n as usize].kind, NodeKind::Closure { .. }));
        // on a cycle through drop glue, a fn of a built-in package is there only because a user
        // type's drop reaches it again: the user fns of the cycle carry the marker
        let glue = members.iter().any(|&n| matches!(g.nodes[n as usize].kind, NodeKind::Glue { .. }));
        let needs_rec = |n: u32| match &g.nodes[n as usize].kind {
            NodeKind::Fn { rec: false, inst, .. } => {
                !(glue && crate::callgraph::in_builtin_package(prog, inst.decl))
            }
            _ => false,
        };
        let unmarked = members.iter().copied().filter(|&n| needs_rec(n)).map(|n| (name(n), n)).min();
        let (code, start) = match (closure, unmarked) {
            (Some(c), _) => ("E1409", c),
            (None, Some((_, n))) => ("E1402", n),
            (None, None) => continue,
        };
        let Some(path) = shortest_cycle(g, s, start) else { continue };
        // shown steps: every node but tables; a step's site is the edge leaving the shown node
        let mut shown: Vec<u32> = Vec::new();
        let mut sites: Vec<Span> = Vec::new();
        for &(from, e) in &path {
            if matches!(g.nodes[from as usize].kind, NodeKind::Table { .. }) {
                continue;
            }
            shown.push(from);
            sites.push(g.out[e as usize].site);
        }
        let (Some(&first), Some(&site0)) = (shown.first(), sites.first()) else { continue };
        let primary = if code == "E1409" { g.nodes[first as usize].span } else { site0 };
        if !reported.insert((code, primary)) {
            continue;
        }
        let mut cycle: Vec<String> = shown.iter().map(|&n| name(n)).collect();
        cycle.push(name(first));
        let cycle = cycle.join(" -> ");
        let who = match &g.nodes[first as usize].kind {
            NodeKind::Closure { parent, .. } => {
                g.fn_of(*parent).map_or(String::new(), |p| def_name(prog, sigs, g, p))
            }
            _ => def_name(prog, sigs, g, first),
        };
        let skip = if code == "E1409" { 0 } else { 1 };
        let mut roles: Vec<(&str, Span)> = sites.iter().skip(skip).map(|&sp| ("call", sp)).collect();
        if code == "E1402" {
            // the other unmarked fns of the component, once per definition, by name
            let mut others: Vec<(String, Span)> = members
                .iter()
                .filter(|&&n| n != first && needs_rec(n))
                .map(|&n| (def_name(prog, sigs, g, n), g.nodes[n as usize].span))
                .filter(|(d, _)| *d != who)
                .collect();
            others.sort();
            others.dedup();
            roles.extend(others.into_iter().map(|(_, sp)| ("unmarked", sp)));
        }
        out.extend(mk(code, None, &[("name", &who), ("cycle", &cycle)], primary, &roles));
    }
    out.extend(stale_rec(prog, sigs, g));
    out
}

/// E1411 (warning): a `rec fn` with instances, none of them on a call cycle. Not when the graph
/// is truncated (a missing edge could hide a cycle) nor for a generic never instantiated.
fn stale_rec(prog: &Program, sigs: &Sigs, g: &CallGraph) -> Vec<Diagnostic> {
    if g.truncated {
        return Vec::new();
    }
    let mut cyclic: FxHashSet<crate::ids::DeclId> = FxHashSet::default();
    for s in 0..g.scc_count() as u32 {
        if !g.is_cycle(s) {
            continue;
        }
        for &n in g.scc(s) {
            if let NodeKind::Fn { inst, .. } = &g.nodes[n as usize].kind {
                cyclic.insert(inst.decl);
            }
        }
    }
    let mut stale: Vec<crate::ids::DeclId> = g
        .nodes
        .iter()
        .filter_map(|nd| match &nd.kind {
            NodeKind::Fn { inst, rec: true, .. } if !cyclic.contains(&inst.decl) => Some(inst.decl),
            _ => None,
        })
        .collect();
    stale.sort();
    stale.dedup();
    let mut out = Vec::new();
    for d in stale {
        let Some(decl) = prog.decl(d) else { continue };
        let name = crate::lower::fn_path(prog, sigs, d);
        out.extend(mk("E1411", None, &[("name", &name)], decl.name_span, &[]));
    }
    out
}

/// The definition path of fn node `n` (no type args).
fn def_name(prog: &Program, sigs: &Sigs, g: &CallGraph, n: u32) -> String {
    match &g.nodes[n as usize].kind {
        NodeKind::Fn { inst, .. } => crate::lower::fn_path(prog, sigs, inst.decl),
        _ => String::new(),
    }
}

/// A shortest cycle from `start` within SCC `s`: its steps (node, index of the edge leaving it),
/// in order. Breadth first over out-edges in target order, so the result is deterministic.
fn shortest_cycle(g: &CallGraph, s: u32, start: u32) -> Option<Vec<(u32, u32)>> {
    let edge_index = |n: u32, k: usize| g.out_start[n as usize] + k as u32;
    // a self-edge is the shortest cycle
    if let Some(k) = g.succ(start).iter().position(|e| e.to == start) {
        return Some(vec![(start, edge_index(start, k))]);
    }
    let mut prev: rustc_hash::FxHashMap<u32, (u32, u32)> = rustc_hash::FxHashMap::default();
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(v) = queue.pop_front() {
        for (k, e) in g.succ(v).iter().enumerate() {
            if g.scc_of[e.to as usize] != s {
                continue;
            }
            if e.to == start {
                let mut steps = vec![(v, edge_index(v, k))];
                let mut cur = v;
                while cur != start {
                    let &(p, pe) = prev.get(&cur)?;
                    steps.push((p, pe));
                    cur = p;
                }
                steps.reverse();
                return Some(steps);
            }
            if e.to != start && !prev.contains_key(&e.to) {
                prev.insert(e.to, (v, edge_index(v, k)));
                queue.push_back(e.to);
            }
        }
    }
    None
}
