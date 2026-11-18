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

use crate::cfg::{Body, LoanId, Place, Proj};
use crate::check::{Finding, later_use};
use crate::loans::Flow;
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

/// The diagnostics of one body's findings.
pub fn diagnostics(flow: &mut Flow<'_>, tys: &mut Tys<'_>, findings: Vec<Finding>) -> Vec<Diagnostic> {
    let body = flow.body;
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
    for (site, (span, _)) in ms {
        if used_spans.contains(&span.offset) {
            continue;
        }
        used_spans.push(span.offset);
        let (mp, mspan) = &body.move_sites[site as usize];
        let text = place_text(body, tys, mp);
        out.extend(mk("E0502", None, &[("place", &text)], span, &[("move", *mspan)]));
    }
    for f in findings {
        match f {
            Finding::Conflict { block, stmt, access, place, span, loans } => {
                let Some(l) = pick_loan(body, &loans) else { continue };
                let loan = &body.loans[l as usize];
                let kind = if loan.mutable && !loan.reserved { "`&mut`" } else { "`&`" };
                let text = place_text(body, tys, &place);
                let loaned = place_text(body, tys, &loan.place);
                let mut roles = vec![("loan", loan.span)];
                if let Some(u) = later_use(flow, (block, stmt), l) {
                    roles.push(("use", u));
                }
                out.extend(mk(
                    "E0501",
                    None,
                    &[("access", access.verb()), ("place", &text), ("loan", kind), ("loaned", &loaned)],
                    span,
                    &roles,
                ));
            }
            Finding::Moved { .. } => {}
            Finding::Immutable { place, span, shared, blame } => {
                let text = place_text(body, tys, &place);
                let bspan = body.locals[blame as usize].span;
                let case = if shared { Some("shared-ref") } else { None };
                out.extend(mk("E0503", case, &[("place", &text)], span, &[("binding", bspan)]));
            }
            Finding::Escape { span, loan } => {
                let l = &body.loans[loan as usize];
                let text = place_text(body, tys, &l.place);
                let lspan = body.locals[l.place.local as usize].span;
                out.extend(mk("E0512", None, &[("place", &text)], span, &[("local", lspan)]));
            }
            Finding::Uncovered { span, param, from, clause } => {
                let name = match body.locals.iter().position(|x| x.param == Some(param)) {
                    Some(i) => local_name(body, tys, i as u32),
                    None => "_".to_string(),
                };
                let case = if clause { None } else { Some("no-from") };
                out.extend(mk("E0510", case, &[("origin", &name)], span, &[("from", from)]));
            }
            Finding::MoveOut { place, span, capture } => {
                let text = place_text(body, tys, &place);
                match capture {
                    Some(c) => {
                        out.extend(mk("E0504", Some("capture"), &[("place", &text)], span, &[("capture", c)]))
                    }
                    None => out.extend(mk("E0504", None, &[("place", &text)], span, &[])),
                }
            }
            Finding::ViewWrite { place, span } => {
                let text = place_text(body, tys, &place);
                out.extend(mk("E0513", None, &[("place", &text)], span, &[]));
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
            temp: false,
            guard: false,
            node,
        };
        Body {
            locals: vec![local(0), local(1)],
            blocks: vec![Block { stmts: Vec::new(), term: Terminator::Return }],
            loans,
            move_sites: Vec::new(),
            bad_moves: Vec::new(),
            ret: RetCheck::None,
            nested: Vec::new(),
            joins: Vec::new(),
            unsupported: None,
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
