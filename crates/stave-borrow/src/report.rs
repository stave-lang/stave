//! Borrow diagnostics: messages, labels, and fixes only from the registry templates
//! (`contracts/diagnostics.toml`, through `stave_diag::Diagnostic::new`).
//!
//! **E0501 loan choice**: of the live loans an access conflicts with, the one with the
//! smallest creation offset; ties (one closure capturing several places) by kind (`&mut` before
//! `&`), then by place path. "Borrowed here" is that loan's creation; "later use" is the first
//! use (smallest offset) reachable after the access of a value whose origin set holds the loan
//! ([`crate::check::later_use`]).
//!
//! **Place text** (`place` hole kind): the local's name, then `.field` (`.0` for tuple
//! elements), `[_]` for an index, and a `*` prefix per dereference. A temporary is named by the
//! source text of its expression.

use rustc_hash::FxHashMap;
use stave_diag::{Diagnostic, Span};
use stave_sema::types::{GArg, Interner, TypeId, TypeKind};

use crate::cfg::{Body, GuardEffect, LoanId, MoveOut, OpKind, PartFix, Place, Proj, StmtKind, TempEnd};
use crate::check::{Access, Finding, Unfit};
use crate::ty::Tys;

/// The loan an E0501 names.
pub fn pick_loan(body: &Body, loans: &[LoanId]) -> Option<LoanId> {
    loans.iter().copied().min_by(|&a, &b| {
        let (x, y) = (&body.loans[a as usize], &body.loans[b as usize]);
        let kx = !(x.mutable && !x.reserved);
        let ky = !(y.mutable && !y.reserved);
        (x.span.offset, kx, place_key(&x.place)).cmp(&(y.span.offset, ky, place_key(&y.place)))
    })
}

/// A structural place-path key: the local (in binding order), then the projections in order
/// (fields in declaration order).
fn place_key(p: &Place) -> (u32, Vec<(u8, u32, u32)>) {
    let proj = p
        .proj
        .iter()
        .map(|x| match x {
            Proj::Deref(_) => (0, 0, 0),
            Proj::Field(i) => (1, *i, 0),
            Proj::EnumField(v, f) => (1, *f, *v),
            Proj::Index => (2, 0, 0),
            Proj::Column(i) => (3, *i, 0),
        })
        .collect();
    (p.local, proj)
}

/// The name of a local: its identifier; a temporary's is the source text of its expression
/// (whitespace runs collapsed, at most 60 bytes).
pub fn local_name(body: &Body, tys: &Tys<'_>, l: u32) -> String {
    let Some(loc) = body.locals.get(l as usize) else { return String::from("_") };
    if !loc.name.is_empty() {
        return loc.name.clone();
    }
    let sp = loc.span;
    let text = tys
        .prog
        .file(sp.file)
        .and_then(|f| f.bytes.get(sp.offset as usize..(sp.offset + sp.len) as usize))
        .map(|b| String::from_utf8_lossy(b).split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    if text.len() > 60 {
        let mut cut = 57;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        return format!("{}...", &text[..cut]);
    }
    text
}

/// The `place` text of `p`.
pub fn place_text(body: &Body, tys: &mut Tys<'_>, p: &Place) -> String {
    let Some(l) = body.locals.get(p.local as usize) else { return String::from("_") };
    let mut s = local_name(body, tys, p.local);
    let mut ty = l.ty;
    for pr in &p.proj {
        match *pr {
            Proj::Deref(_) => {
                s.insert(0, '*');
                ty = tys.deref(ty).map_or(stave_sema::types::TypeId::NONE, |x| x.0);
            }
            Proj::Field(i) => {
                match tys.field_name(ty, 0, i) {
                    Some(n) => {
                        s.push('.');
                        s.push_str(&n);
                    }
                    None => s.push_str(&format!(".{i}")),
                }
                ty = tys.field_ty(ty, 0, i);
            }
            Proj::EnumField(v, f) => {
                match tys.field_name(ty, v, f) {
                    Some(n) => {
                        s.push('.');
                        s.push_str(&n);
                    }
                    None => s.push_str(&format!(".{f}")),
                }
                ty = tys.field_ty(ty, v, f);
            }
            Proj::Index => {
                s.push_str("[_]");
                ty = tys.elem(ty);
            }
            Proj::Column(i) => {
                let row = match tys.kind(ty) {
                    TypeKind::Adt { args, .. } => match args.first() {
                        Some(GArg::Type(t)) => *t,
                        _ => TypeId::NONE,
                    },
                    _ => TypeId::NONE,
                };
                match tys.field_name(row, 0, i) {
                    Some(n) => {
                        s.push('.');
                        s.push_str(&n);
                    }
                    None => s.push_str(&format!(".{i}")),
                }
                let et = tys.field_ty(row, 0, i);
                ty = tys.ov.intern(TypeKind::Slice { elem: et, align: 0 });
            }
        }
    }
    s
}

fn mk(
    code: &str,
    case: Option<&str>,
    holes: &[(&str, &str)],
    primary: Span,
    roles: &[(&str, Span)],
) -> Option<Diagnostic> {
    match Diagnostic::new(code, case, holes, Some(primary), roles) {
        Ok(d) => Some(d),
        Err(_) => {
            FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

static FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Diagnostics that could not be built from their templates since process start (tests assert 0).
pub fn failures() -> u64 {
    FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The E0501 case of a temporary that ends with live loans, and the word for what it ends with.
fn temp_case(end: Option<TempEnd>) -> (&'static str, &'static str) {
    use stave_sema::temps::Bound;
    match end {
        Some(TempEnd::Loop) => ("temporary-loop", "loop"),
        Some(TempEnd::Stmt(Bound::Branch)) => ("temporary-branch", "branch"),
        Some(TempEnd::Stmt(Bound::Arm)) => ("temporary-arm", "arm"),
        Some(TempEnd::Stmt(Bound::Clause)) => ("temporary-clause", "check"),
        Some(TempEnd::Stmt(Bound::Old)) => ("temporary-old", "check"),
        Some(TempEnd::Stmt(Bound::Cond)) => ("temporary", "condition"),
        Some(TempEnd::Stmt(Bound::Guard)) => ("temporary", "guard"),
        Some(TempEnd::Stmt(Bound::Stmt)) | None => ("temporary", "statement"),
    }
}

/// The statement the drop of `place` at statement `stmt` of `block` belongs to: the assignment
/// that overwrites `place` (`true`), or the move of the value a pattern takes apart, which
/// follows the drops of the parts the pattern ignores. Its index in the block.
fn paired(body: &Body, block: u32, stmt: u32, place: &Place) -> Option<(u32, bool)> {
    let stmts = &body.blocks.get(block as usize)?.stmts;
    let rest = stmts.get(stmt as usize + 1..)?;
    let skip = rest
        .iter()
        .take_while(|s| matches!(&s.kind, StmtKind::Drop(p) if p.local == place.local && !p.proj.is_empty()))
        .count();
    let j = stmt + 1 + skip as u32;
    match &rest.get(skip)?.kind {
        StmtKind::Assign { dest, init: false, .. } if skip == 0 && dest == place => Some((j, true)),
        StmtKind::Read(o)
            if o.kind == OpKind::Move && o.place.local == place.local && !place.proj.is_empty() =>
        {
            Some((j, false))
        }
        _ => None,
    }
}

/// The diagnostics of one body's findings. `later_use`: the first use after a statement (block,
/// index) of a value that holds a loan, and whether it is in a later iteration
/// ([`crate::check::later_use`]).
pub fn diagnostics(
    body: &Body,
    tys: &mut Tys<'_>,
    findings: Vec<Finding>,
    later_use: &mut dyn FnMut((u32, u32), LoanId) -> Option<(Span, bool)>,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    // E0502: one per move site, at its first (smallest offset) use
    let mut moved: FxHashMap<u32, (Span, Place)> = FxHashMap::default();
    for f in &findings {
        if let Finding::Moved { site, place, span } = f {
            let e = moved.entry(*site).or_insert((*span, place.clone()));
            if span.offset < e.0.offset {
                *e = (*span, place.clone());
            }
        }
    }
    let mut ms: Vec<(u32, (Span, Place))> = moved.into_iter().collect();
    ms.sort_by_key(|(s, (sp, _))| (sp.offset, *s));
    let mut used_spans: Vec<u32> = Vec::new();
    for (site, (span, used)) in ms {
        if used_spans.contains(&span.offset) {
            continue;
        }
        used_spans.push(span.offset);
        let (mp, mspan) = &body.move_sites[site as usize];
        let text = place_text(body, tys, mp);
        // a later operand of the expression whose earlier operand moved the place
        let operand = body.operand_moves.iter().any(|&(s, w)| {
            s == site
                && w.file == span.file
                && span.offset > mspan.offset
                && span.offset + span.len <= w.offset + w.len
        });
        if operand {
            let u = place_text(body, tys, &used);
            out.extend(mk(
                "E0502",
                Some("operand"),
                &[("place", &u), ("moved", &text)],
                span,
                &[("move", *mspan)],
            ));
        } else {
            out.extend(mk("E0502", None, &[("place", &text)], span, &[("move", *mspan)]));
        }
    }
    // An assignment drops the value it overwrites, then writes; a pattern that takes a value
    // apart drops the parts it ignores, then moves the value. Each pair is one access to the
    // reader: the drop is reported only when the write or move is not, an old value's as the
    // assignment it belongs to.
    let hit: Vec<(u32, u32)> = findings
        .iter()
        .filter_map(|f| match f {
            Finding::Conflict { access: Access::Write | Access::Move, block, stmt, .. } => {
                Some((*block, *stmt))
            }
            _ => None,
        })
        .collect();
    for f in findings {
        match f {
            Finding::Conflict { block, stmt, access, place, span, loans } => {
                let access = match (access, paired(body, block, stmt, &place)) {
                    (Access::Drop, Some((j, _))) if hit.contains(&(block, j)) => continue,
                    (Access::Drop, Some((_, true))) => Access::Write,
                    _ => access,
                };
                let Some(l) = pick_loan(body, &loans) else { continue };
                let loan = &body.loans[l as usize];
                let kind = if loan.mutable && !loan.reserved { "`&mut`" } else { "`&`" };
                let text = place_text(body, tys, &place);
                let loaned = place_text(body, tys, &loan.place);
                let mut roles = vec![("loan", loan.span)];
                let later = later_use((block, stmt), l);
                if let Some((u, _)) = later {
                    roles.push(("use", u));
                }
                let again = later.is_some_and(|x| x.1);
                let holes = [
                    ("access", access.verb()),
                    ("place", &text),
                    ("loan", kind),
                    ("loaned", &loaned),
                    ("later", if again { "here, in the next iteration" } else { "later here" }),
                ];
                if crate::check::temp_end(body, access, &place) {
                    // the temporary is where it was made, also when a `break` / `continue` ends it
                    let t = &body.locals[place.local as usize];
                    let (case, stmt) = temp_case(t.end);
                    let m = loan.mutable && !loan.reserved;
                    let more = [
                        ("stmt", stmt),
                        ("bind", if m { "let mut t" } else { "let t" }),
                        ("tref", if m { "&mut t" } else { "&t" }),
                    ];
                    let all: Vec<(&str, &str)> = holes.iter().chain(&more).copied().collect();
                    out.extend(mk("E0501", Some(case), &all, t.span, &roles));
                } else if later.is_some_and(|(u, _)| body.joins.iter().any(|j| j.1 == u)) {
                    // the later use stands for a thread of the scope, which runs until the scope
                    // is left ([`crate::cfg`] `hold_spawns`): it is named at the spawn
                    out.extend(mk("E0501", Some("thread"), &holes, span, &[("loan", loan.span)]));
                } else {
                    out.extend(mk("E0501", None, &holes, span, &roles));
                }
            }
            Finding::Moved { .. } => {}
            Finding::Immutable { place, span, shared, blame, cast } => {
                let text = place_text(body, tys, &place);
                let bspan = body.locals[blame as usize].span;
                let case = match (shared, cast) {
                    (true, true) => Some("shared-ref-cast"),
                    (true, false) => Some("shared-ref"),
                    (false, _) => None,
                };
                out.extend(mk("E0503", case, &[("place", &text)], span, &[("binding", bspan)]));
            }
            Finding::Escape { span, loan } => {
                let l = &body.loans[loan as usize];
                let text = place_text(body, tys, &l.place);
                let root = &body.locals[l.place.local as usize];
                match root.capture {
                    Some(c) => out.extend(mk(
                        "E0512",
                        Some("mut-capture"),
                        &[("place", &text)],
                        span,
                        &[("capture", c)],
                    )),
                    None if root.env => {
                        out.extend(mk("E0512", Some("env"), &[], span, &[("closure", root.span)]))
                    }
                    None => out.extend(mk("E0512", None, &[("place", &text)], span, &[("local", root.span)])),
                }
            }
            Finding::Uncovered { span, param, from, clause, copy } => {
                let name = match body.locals.iter().position(|x| x.param == Some(param)) {
                    Some(i) => local_name(body, tys, i as u32),
                    None => "_".to_string(),
                };
                let case = match (clause, copy) {
                    (false, _) => Some("no-from"),
                    (true, true) => Some("copy"),
                    (true, false) => None,
                };
                out.extend(mk("E0510", case, &[("origin", &name)], span, &[("from", from)]));
            }
            Finding::MoveOut { place, span, why } => {
                let text = place_text(body, tys, &place);
                match why {
                    MoveOut::Capture(c) => {
                        out.extend(mk("E0504", Some("capture"), &[("place", &text)], span, &[("capture", c)]))
                    }
                    MoveOut::Drop(owner) => {
                        let o = place_text(body, tys, &owner);
                        out.extend(mk("E0504", Some("drop"), &[("place", &text), ("owner", &o)], span, &[]))
                    }
                    MoveOut::Static => {
                        out.extend(mk("E0504", Some("static"), &[("place", &text)], span, &[]))
                    }
                    MoveOut::Part(owner, fix) => {
                        let o = place_text(body, tys, &owner);
                        let holes = [("place", text.as_str()), ("owner", o.as_str())];
                        out.extend(match fix {
                            PartFix::Reborrow => mk("E0504", Some("part-reborrow"), &holes, span, &[]),
                            PartFix::Replace => mk("E0504", Some("part-replace"), &holes, span, &[]),
                            PartFix::Pattern(pat) => {
                                let holes = [holes[0], holes[1], ("pattern", pat.as_str())];
                                mk("E0504", Some("part"), &holes, span, &[])
                            }
                        })
                    }
                    MoveOut::Behind => out.extend(mk("E0504", None, &[("place", &text)], span, &[])),
                }
            }
            Finding::TempWrite { span, text } => {
                out.extend(mk("E0503", Some("temporary"), &[("place", &text)], span, &[]));
            }
            Finding::GuardEffect { span, what } => {
                let (case, place) = match what {
                    GuardEffect::Move(p) => ("move", Some(p)),
                    GuardEffect::Assign(p) => ("assign", Some(p)),
                    GuardEffect::MutBorrow(p) => ("mut-borrow", Some(p)),
                    GuardEffect::Try => ("try", None),
                };
                match place {
                    Some(p) => {
                        let text = place_text(body, tys, &p);
                        out.extend(mk("E0505", Some(case), &[("place", &text)], span, &[]))
                    }
                    None => out.extend(mk("E0505", Some(case), &[], span, &[])),
                }
            }
            Finding::ViewWrite { place, span, why } => {
                let text = place_text(body, tys, &place);
                let name = |p: u32| match body.locals.iter().position(|x| x.param == Some(p)) {
                    Some(i) => local_name(body, tys, i as u32),
                    None => "_".to_string(),
                };
                match why {
                    Unfit::Origins => out.extend(mk("E0513", None, &[("place", &text)], span, &[])),
                    Unfit::Several => {
                        out.extend(mk("E0513", Some("several"), &[("place", &text)], span, &[]))
                    }
                    Unfit::Clause { param, add } => {
                        // the clause as it is written: `x + copy y`
                        let entries: Vec<String> = add
                            .iter()
                            .map(|&(q, copy)| if copy { format!("copy {}", name(q)) } else { name(q) })
                            .collect();
                        let (pname, clause) = (name(param), entries.join(" + "));
                        let holes = [
                            ("place", text.as_str()),
                            ("param", pname.as_str()),
                            ("clause", clause.as_str()),
                        ];
                        // a finding is never lost to a case the templates lack: the base shape
                        // of the code, which needs the place only, reports it then
                        let d = mk("E0513", Some("param"), &holes, span, &[])
                            .or_else(|| mk("E0513", None, &holes[..1], span, &[]));
                        out.extend(d);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Block, Loan, Local, RetCheck, Terminator};
    use stave_diag::FileId;
    use stave_sema::types::TypeId;

    fn sp(offset: u32) -> Span {
        Span::new(FileId(0), offset, 1)
    }

    fn body(loans: Vec<Loan>) -> Body {
        let local = |node| Local {
            ty: TypeId::NONE,
            mutable: true,
            span: sp(0),
            name: format!("l{node}"),
            param: None,
            capture: None,
            capture_mut: false,
            temp: false,
            end: None,
            guard: false,
            static_root: false,
            static_item: false,
            stmt: u32::MAX,
            made: None,
            env: false,
            node,
        };
        Body {
            locals: vec![local(0), local(1)],
            blocks: vec![Block { stmts: Vec::new(), term: Terminator::Return }],
            loans,
            move_sites: Vec::new(),
            bad_moves: Vec::new(),
            guard_effects: Vec::new(),
            temp_writes: Vec::new(),
            operand_moves: Vec::new(),
            ret: RetCheck::None,
            declared: Vec::new(),
            is_nested: false,
            at: None,
            nested: Vec::new(),
            joins: Vec::new(),
            unsupported: None,
            internal: None,
            loops: Vec::new(),
            schedule: Default::default(),
        }
    }

    fn loan(local: u32, proj: Vec<Proj>, mutable: bool, offset: u32) -> Loan {
        Loan {
            place: Place { local, proj, ty: TypeId::NONE },
            mutable,
            span: sp(offset),
            node: offset,
            reserved: false,
            capture: true,
            whole: None,
            cast: false,
        }
    }

    /// Several live loans → the smallest creation offset; the same offset (one
    /// closure's capture loans, all created at its `fn` token) → `&mut` before `&`, then by place
    /// path (`s.a` before `s.b`, binding order of the locals first). Whole-local captures
    /// cannot make one access conflict with two loans of one offset from source, so the
    /// tie-break is checked here on the loan list.
    #[test]
    fn e0501_pick_ties() {
        // offset first, even against `&mut` and a smaller path
        let b = body(vec![loan(0, vec![Proj::Field(1)], false, 10), loan(0, vec![Proj::Field(0)], true, 20)]);
        assert_eq!(pick_loan(&b, &[1, 0]), Some(0));
        // same offset: `&mut` before `&`
        let b = body(vec![loan(0, vec![Proj::Field(0)], false, 10), loan(0, vec![Proj::Field(1)], true, 10)]);
        assert_eq!(pick_loan(&b, &[0, 1]), Some(1));
        // a two-phase reservation counts as `&`
        let mut r = loan(0, vec![Proj::Field(0)], true, 10);
        r.reserved = true;
        let b = body(vec![r, loan(0, vec![Proj::Field(1)], true, 10)]);
        assert_eq!(pick_loan(&b, &[0, 1]), Some(1));
        // same offset and kind: by place path, `s.a` (field 0) before `s.b` (field 1)
        let b = body(vec![loan(0, vec![Proj::Field(1)], true, 10), loan(0, vec![Proj::Field(0)], true, 10)]);
        assert_eq!(pick_loan(&b, &[0, 1]), Some(1));
        // a shorter path (the whole local) before its fields; earlier locals first
        let b = body(vec![loan(0, vec![Proj::Field(0)], false, 10), loan(0, vec![], false, 10)]);
        assert_eq!(pick_loan(&b, &[0, 1]), Some(1));
        let b = body(vec![loan(1, vec![], true, 10), loan(0, vec![Proj::Field(3)], true, 10)]);
        assert_eq!(pick_loan(&b, &[0, 1]), Some(1));
        assert_eq!(pick_loan(&b, &[]), None);
    }
}
