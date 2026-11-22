//! The locks a body holds at its calls, joins, and drops, for the rank check
//! ([`stave_sema::ranks`]).
//!
//! A local may hold a lock while its value may contain a `MutexGuard` (or a value of a `T: view`
//! param) and it may be initialized: forward dataflow over the CFG, union at joins. A local
//! becomes initialized when it or a part of it is assigned (params and captures on entry), and
//! stops being so when it is moved out, dropped as a whole, or dies (a value moves whole: a
//! pattern that takes it apart moves its source). At a call, the locals the call's arguments move out as a whole are
//! not counted: the callee holds them, and its own calls are checked in its body. A drop runs the
//! type's drop glue: it is checked like a call, without the local it drops, and, when the dropped
//! value itself may hold a lock, against the guards inside it ([`HeldAt::own`]). The drops are
//! the CFG's drop statements, which are the body's drop schedule ([`stave_sema::drops`]): a
//! scope or statement end, an exit, a part a pattern ignores, the old value of an assignment
//! (behind a dereference the assignment itself stands for it). The types a body really drops (a
//! drop of a value that may be initialized) are reported for every body: the call graph's drop
//! edges (every type a body makes or owns) narrowed to them, a constructor that returns its value
//! drops nothing.

use stave_sema::ranks::{At, Held, HeldAt, Part};

use crate::bits::Bits;
use crate::cfg::{Body, LocalId, OpKind, Rvalue, StmtKind};
use crate::loans::{Flow, for_each_op};
use crate::ty::Tys;

/// The locks of one body (nested bodies are scanned separately).
#[derive(Default)]
pub struct Scan {
    pub sites: Vec<HeldAt>,
    pub any: Vec<Held>,
    pub calls: Vec<u32>,
    /// Whether any local of the body may hold a lock.
    pub holds: bool,
    /// The local types of the body's type overlay (the [`At::Drop`] types index them).
    pub types: Vec<stave_sema::types::TypeKind>,
    /// The types of the values the body may drop (typeck's table); `None`: one is a type of the
    /// body's overlay only.
    pub drops: Option<Vec<stave_sema::TypeId>>,
}

/// The locals a statement moves out or ends as a whole.
fn ended(k: &StmtKind, mut f: impl FnMut(LocalId)) {
    let whole = |o: &crate::cfg::Operand| {
        (o.kind == OpKind::Move && o.place.proj.is_empty()).then_some(o.place.local)
    };
    match k {
        StmtKind::Assign { rv, .. } => for_each_op(rv, |o| {
            if let Some(l) = whole(o) {
                f(l);
            }
        }),
        StmtKind::Read(o) | StmtKind::Return(o) => {
            if let Some(l) = whole(o) {
                f(l);
            }
        }
        StmtKind::Drop(p) if p.proj.is_empty() => f(p.local),
        StmtKind::StorageDead(l) => f(*l),
        _ => {}
    }
}

/// Scans `flow`'s body.
pub fn scan(flow: &Flow<'_>, tys: &mut Tys<'_>) -> Scan {
    let body: &Body = flow.body;
    let mut out = Scan { drops: Some(Vec::new()), ..Scan::default() };
    let mut drops_any = false;
    for b in &body.blocks {
        for s in &b.stmts {
            match &s.kind {
                StmtKind::Assign { rv: Rvalue::Call { tok, .. }, .. } => out.calls.push(*tok),
                _ => {}
            }
            drops_any |= matches!(&s.kind, StmtKind::Drop(_) | StmtKind::Assign { init: false, .. });
        }
    }
    // the locals that may hold a lock, with their parts (each type walked once), first
    let mut parts: Vec<Vec<Part>> = Vec::new();
    let mut idx: Vec<u32> = vec![u32::MAX; body.locals.len()];
    let mut memo: Vec<(stave_sema::TypeId, Vec<Part>)> = Vec::new();
    for (i, l) in body.locals.iter().enumerate() {
        if l.ty.is_none() || !flow.is_ref_like(l.ty) {
            continue;
        }
        let ps = match memo.iter().find(|m| m.0 == l.ty) {
            Some(m) => m.1.clone(),
            None => {
                let ps = stave_sema::ranks::lock_parts(&mut tys.ov, tys.prog, tys.sigs, tys.env, l.ty);
                memo.push((l.ty, ps.clone()));
                ps
            }
        };
        if !ps.is_empty() {
            idx[i] = parts.len() as u32;
            parts.push(ps);
        }
    }
    let ncand = parts.len();
    out.holds = ncand > 0;
    if !out.holds && !drops_any {
        return out;
    }
    // then the locals with drop glue: a drop runs only on a value that may be initialized
    for (i, l) in body.locals.iter().enumerate() {
        if idx[i] == u32::MAX && !l.ty.is_none() && tys.needs_drop(l.ty) {
            idx[i] = parts.len() as u32;
            parts.push(Vec::new());
        }
    }
    // tracked index -> local (candidates were numbered first, the locals with drop glue after)
    let n = parts.len();
    let mut local_of: Vec<LocalId> = vec![0; n];
    for (l, &i) in idx.iter().enumerate() {
        if i != u32::MAX {
            local_of[i as usize] = l as LocalId;
        }
    }
    let mut entry: Vec<Option<Bits>> = vec![None; body.blocks.len()];
    let mut init = Bits::new(n);
    for (i, l) in body.locals.iter().enumerate() {
        if l.param.is_some() && idx[i] != u32::MAX {
            init.insert(idx[i] as usize);
        }
    }
    if !body.blocks.is_empty() {
        entry[0] = Some(init);
    }
    let step = |k: &StmtKind, st: &mut Bits| {
        ended(k, |l| {
            if let Some(&i) = idx.get(l as usize)
                && i != u32::MAX
            {
                st.remove(i as usize);
            }
        });
        if let StmtKind::Assign { dest, .. } = k
            && let Some(&i) = idx.get(dest.local as usize)
            && i != u32::MAX
        {
            st.insert(i as usize);
        }
    };
    loop {
        let mut changed = false;
        for &b in &flow.rpo {
            let Some(mut st) = entry[b as usize].clone() else { continue };
            for s in &body.blocks[b as usize].stmts {
                step(&s.kind, &mut st);
            }
            for &succ in body.blocks[b as usize].term.succs() {
                match &mut entry[succ as usize] {
                    Some(e) => changed |= e.union_with(&st),
                    slot @ None => {
                        *slot = Some(st.clone());
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let holding = |st: &Bits| st.iter().any(|i| i < ncand);
    let live = |st: &Bits, l: LocalId| match idx.get(l as usize) {
        Some(&i) if i != u32::MAX => st.contains(i as usize),
        _ => true,
    };
    let held = |st: &Bits| -> Vec<Held> {
        st.iter()
            .filter(|&i| i < ncand)
            .map(|i| Held { parts: parts[i].clone(), span: body.locals[local_of[i] as usize].span })
            .collect()
    };
    // whether a value of a type may hold a lock itself
    let mut own = |t: stave_sema::TypeId, tys: &mut Tys<'_>| -> bool {
        if t.is_none() {
            return false;
        }
        match memo.iter().find(|m| m.0 == t) {
            Some(m) => !m.1.is_empty(),
            None => {
                let ps = stave_sema::ranks::lock_parts(&mut tys.ov, tys.prog, tys.sigs, tys.env, t);
                let r = !ps.is_empty();
                memo.push((t, ps));
                r
            }
        }
    };
    let mut dropped: Vec<stave_sema::TypeId> = Vec::new();
    let mut any = Bits::new(n);
    for &b in &flow.rpo {
        let Some(mut st) = entry[b as usize].clone() else { continue };
        for s in &body.blocks[b as usize].stmts {
            any.union_with(&st);
            // the drops that may run, and what they drop (an assignment through a dereference
            // drops the old value; any other has a drop statement before it)
            let drop = match &s.kind {
                StmtKind::Drop(p) if live(&st, p.local) => Some((p.ty, Some(p))),
                StmtKind::Assign { dest, init: false, .. }
                    if dest.has_deref() && live(&st, dest.local) && tys.needs_drop(dest.ty) =>
                {
                    Some((dest.ty, None))
                }
                _ => None,
            };
            if let Some((t, _)) = drop {
                dropped.push(t);
            }
            match &s.kind {
                StmtKind::Assign { rv: Rvalue::Call { tok, args, .. }, .. } if holding(&st) => {
                    let mut at = st.clone();
                    for a in args {
                        if a.kind == OpKind::Move
                            && a.place.proj.is_empty()
                            && let Some(&i) = idx.get(a.place.local as usize)
                            && i != u32::MAX
                        {
                            at.remove(i as usize);
                        }
                    }
                    if holding(&at) {
                        out.sites.push(HeldAt {
                            at: At::Call(*tok),
                            span: s.span,
                            held: held(&at),
                            own: None,
                        });
                    }
                }
                StmtKind::Read(o) if holding(&st) => {
                    if let Some(&(_, scope)) = body.joins.iter().find(|j| j.0 == o.place.local) {
                        out.sites.push(HeldAt {
                            at: At::Join(scope),
                            span: s.span,
                            held: held(&st),
                            own: None,
                        });
                    }
                }
                _ => {}
            }
            // a drop runs the type's drop glue: a call, holding all but what it drops, and
            // what the dropped value holds itself until the glue reaches it
            if let Some((t, whole)) = drop {
                let inner = own(t, tys);
                let mut at = st.clone();
                if let Some(p) = whole
                    && p.proj.is_empty()
                    && let Some(&i) = idx.get(p.local as usize)
                    && i != u32::MAX
                {
                    at.remove(i as usize);
                }
                if holding(&at) || inner {
                    let own = inner.then(|| match whole {
                        Some(p) => body.locals[p.local as usize].span,
                        None => s.span,
                    });
                    out.sites.push(HeldAt { at: At::Drop(t), span: s.span, held: held(&at), own });
                    out.holds = true;
                }
            }
            step(&s.kind, &mut st);
        }
    }
    dropped.sort_unstable();
    dropped.dedup();
    out.drops = if dropped.iter().any(|t| t.is_local()) { None } else { Some(dropped) };
    out.any = held(&any);
    out
}
