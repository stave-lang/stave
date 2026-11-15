//! Thread-crossing capture checks: E0802 and E0806.
//!
//! Run once per type-checked body without errors, over the closure list of
//! [`crate::closures`]. Fn types do not record captures, so the rules apply where a closure body
//! crosses threads:
//!
//! - **sent**: a `spawn` body, and the `a` / `b` arguments of `std.thread.join`. A captured `&T`
//!   ([`CaptureMode::Ref`]) needs `T: Sync`; a `&mut T` capture needs `T: Send`
//!   (E0802, cases default / `sync`).
//! - **shared**: the `body` of `std.thread.par_for` / `par_for_mut` (called from many threads at
//!   once): every capture needs `T: Sync`, and a `&mut` capture is E0802 case `par-mut`.
//! - the argument itself (after single-operand wrappers) must be a closure literal or a fn item:
//!   any other expression (a fn-typed param passed on, a local, a value `if` or `match` choosing
//!   between fn values, a call returning one) is E0806.
//! - the three functions are never fn values (E0807 case `thread`, at the coercion in
//!   [`crate::expr`]): a call through a fn value would skip these checks.
//!
//! The three functions are fixed by path (contracts/intrinsics.md "Paths fixed by the
//! compiler"). The primary span of E0802 is the capture's first use in the body (its first
//! mutating use for a `&mut` capture). Overlapping `&mut` captures of the two `join` closures are
//! the borrow checker's E0501 (both closures are live during the call).

use stave_diag::Diagnostic;
use stave_syntax::{NodeKind, Term};

use crate::closures::{CaptureMode, ClosureInfo, ClosureKind};
use crate::diag::mk;
use crate::ids::{BindingRef, DeclId};
use crate::sigs::TraitRef;
use crate::typeck::{Ck, Coercion, Target, TypedBody};

/// A checked fn: its decl, its name for E0806, and (argument index, crossing) per fn-value arg.
type ThreadFn<'s> = (Option<DeclId>, &'s str, &'s [(usize, Cross)]);

/// The `std.thread` functions whose calls check their fn arguments' captures.
const CHECKED: [&str; 3] = ["join", "par_for", "par_for_mut"];

/// Whether `d` is one of [`CHECKED`].
pub(crate) fn is_checked_thread_fn(prog: &crate::Program, d: DeclId) -> bool {
    let Some(x) = prog.decl(d) else { return false };
    if !CHECKED.contains(&prog.strings.get(x.name)) {
        return false;
    }
    crate::mono::item_at(prog, "std.thread", prog.strings.get(x.name)) == Some(d)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cross {
    Sent,
    Shared,
}

impl Ck<'_> {
    /// The E0802 / E0806 diagnostics of `body` (no diagnostics for a body with errors).
    pub(crate) fn thread_checks(&mut self, body: &TypedBody) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if body.has_error {
            return out;
        }
        let icx = self.icx;
        let tys = &self.tys;
        let infos = crate::closures::closures(&icx, body, |t| crate::closures::is_mut_ref(tys, t), body.node);
        let find = |name: &str| crate::mono::item_at(self.prog, "std.thread", name);
        let fns: [ThreadFn<'_>; 3] = [
            (find("join"), "`thread.join`", &[(0, Cross::Sent), (1, Cross::Sent)]),
            (find("par_for"), "`thread.par_for`", &[(2, Cross::Shared)]),
            (find("par_for_mut"), "`thread.par_for_mut`", &[(1, Cross::Shared)]),
        ];
        for c in infos.iter().filter(|c| c.kind == ClosureKind::Spawn) {
            self.check_captures(c, body, Cross::Sent, &mut out);
        }
        if fns.iter().all(|f| f.0.is_none()) {
            return out;
        }
        let lo = icx.ast.descendants(body.node).start;
        for n in lo..=body.node {
            if icx.kind(n) != Some(NodeKind::Postfix) {
                continue;
            }
            let kids = icx.kids_vec(n);
            for &k in kids.iter().skip(1) {
                let lp = icx.a(k).saturating_sub(1);
                if !icx.is(lp, Term::LParen) {
                    continue;
                }
                let Some(Target::Fn { decl, .. }) = body.target(lp) else { continue };
                let Some(&(_, callee, which)) = fns.iter().find(|f| f.0 == Some(*decl)) else { continue };
                let args = icx.kids_vec(k);
                for &(i, cross) in which {
                    let Some(&arg) = args.get(i) else { continue };
                    let chain = wrapper_chain(&icx, arg);
                    let core = chain.last().copied().unwrap_or(arg);
                    if let Some(c) = infos.iter().find(|c| c.node == core) {
                        self.check_captures(c, body, cross, &mut out);
                    } else if !chain.iter().any(|&n| body.coercion(n) == Some(&Coercion::FnItem)) {
                        let at = icx.node_span(arg);
                        out.extend(mk("E0806", None, &[("callee", callee)], at, &[]));
                    }
                }
            }
        }
        out
    }

    fn check_captures(&mut self, c: &ClosureInfo, body: &TypedBody, cross: Cross, out: &mut Vec<Diagnostic>) {
        let (send_d, sync_d) = (self.sigs.lang.send, self.sigs.lang.sync);
        for cap in &c.captures {
            let Some(t) = body.local(BindingRef { node: cap.binding }) else { continue };
            if t.is_none() {
                continue;
            }
            let a = self.icx.a(cap.binding);
            let a = if self.icx.is(a, Term::KwMut) { a + 1 } else { a };
            let name = String::from_utf8_lossy(self.icx.tok_text(a)).into_owned();
            let genv = self.genv();
            let send = TraitRef { trait_decl: send_d, args: Vec::new() };
            let sync = TraitRef { trait_decl: sync_d, args: Vec::new() };
            let (case, ok, at) = match (cross, cap.mode) {
                (Cross::Shared, CaptureMode::RefMut) => (Some("par-mut"), false, cap.first_mut),
                (Cross::Shared, _) | (Cross::Sent, CaptureMode::Ref) => {
                    (Some("sync"), self.w.implements(&mut self.tys, t, &sync, genv), cap.first_use)
                }
                (Cross::Sent, _) => (None, self.w.implements(&mut self.tys, t, &send, genv), cap.first_mut),
            };
            if ok {
                continue;
            }
            let tn = self.show(t);
            let span = self.icx.tok_span(at);
            out.extend(mk("E0802", case, &[("name", &name), ("type", &tn)], span, &[]));
        }
    }
}

/// `n` and the single-operand expression wrappers below it, outermost first; the last node is
/// the argument's core. A fn item argument has its `FnItem` coercion on one of these nodes; a
/// coercion deeper inside (an operand of `if`, `match`, or a call) does not make the argument a
/// fn item.
fn wrapper_chain(icx: &crate::itemcx::ItemCx<'_>, n: u32) -> Vec<u32> {
    let mut cur = n;
    let mut out = vec![n];
    for _ in 0..1_000_000 {
        match icx.kind(cur) {
            Some(
                NodeKind::Expr
                | NodeKind::RangeExpr
                | NodeKind::OrExpr
                | NodeKind::AndExpr
                | NodeKind::CmpExpr
                | NodeKind::BitOr
                | NodeKind::BitXor
                | NodeKind::BitAnd
                | NodeKind::Shift
                | NodeKind::Add
                | NodeKind::Mul
                | NodeKind::Cast
                | NodeKind::Unary
                | NodeKind::Postfix,
            ) => {
                let kids = icx.kids_vec(cur);
                if kids.len() != 1 || icx.a(kids[0]) != icx.a(cur) {
                    break;
                }
                cur = kids[0];
                out.push(cur);
            }
            _ => break,
        }
    }
    out
}
