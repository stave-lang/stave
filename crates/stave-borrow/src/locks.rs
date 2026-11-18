//! The locks a body holds at its calls and joins, for the rank check ([`stave_sema::ranks`]).
//!
//! A local may hold a lock while its value may contain a `MutexGuard` (or a value of a `T: view`
//! param) and it may be initialized: forward dataflow over the CFG, union at joins. A local
//! becomes initialized when it or a part of it is assigned (params and captures on entry), and
//! stops being so when it is moved out as a whole, dropped as a whole, or dies. A partial move
//! keeps it (conservative). At a call, the locals the call's arguments move out as a whole are
//! not counted: the callee holds them, and its own calls are checked in its body.

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
    let mut out = Scan::default();
    for b in &body.blocks {
        for s in &b.stmts {
            if let StmtKind::Assign { rv: Rvalue::Call { tok, .. }, .. } = &s.kind {
                out.calls.push(*tok);
            }
        }
    }
    // the locals that may hold a lock, with their parts (each type walked once)
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
    if parts.is_empty() {
        return out;
    }
    out.holds = true;
    let local_of: Vec<LocalId> =
        (0..body.locals.len() as u32).filter(|&l| idx[l as usize] != u32::MAX).collect();
    let n = parts.len();
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
    let held = |st: &Bits| -> Vec<Held> {
        st.iter()
            .map(|i| Held { parts: parts[i].clone(), span: body.locals[local_of[i] as usize].span })
            .collect()
    };
    let mut any = Bits::new(n);
    for &b in &flow.rpo {
        let Some(mut st) = entry[b as usize].clone() else { continue };
        for s in &body.blocks[b as usize].stmts {
            any.union_with(&st);
            match &s.kind {
                StmtKind::Assign { rv: Rvalue::Call { tok, args, .. }, .. } if !st.is_empty() => {
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
                    if !at.is_empty() {
                        out.sites.push(HeldAt { at: At::Call(*tok), span: s.span, held: held(&at) });
                    }
                }
                StmtKind::Read(o) if !st.is_empty() => {
                    if let Some(&(_, scope)) = body.joins.iter().find(|j| j.0 == o.place.local) {
                        out.sites.push(HeldAt { at: At::Join(scope), span: s.span, held: held(&st) });
                    }
                }
                _ => {}
            }
            step(&s.kind, &mut st);
        }
    }
    out.any = held(&any);
    out
}
