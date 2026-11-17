//! `dispatch fn` arms:
//! - E0708: an arm feature name no target owns;
//! - E0705: an arm target whose type is not the dispatch's signature with needs = the
//!   dispatch's needs plus the arm's features (the `_` arm: plus nothing);
//! - E0706 case `extra-needs`: a `_` target needing a feature the dispatch does not have.
//!
//! A dispatch missing its final `_` arm does not parse. Arms whose features (with the
//! dispatch's) the target's arch does not own are dropped from the build unchecked, and so is a
//! dispatch that needs such features itself.
//!
//! [`plan`] gives the arm chain lowering builds a dispatch body from (crate::lower_stmt).

use stave_diag::{Diagnostic, Span};
use stave_syntax::NodeKind;
use stave_target::{Feature, FeatureSet};

use crate::Program;
use crate::decls::{DeclFlags, DeclKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::mono::InstKey;
use crate::resolve::Res;
use crate::sigs::{FnSig, NameCx, Sigs};
use crate::typeck::Typeck;
use crate::types::{self, Interner, TypeKind, TypeTable};

/// One arm of a dispatch: its features (closed; `None` for `_`) and its target's `Name` node.
struct Arm {
    features: Option<FeatureSet>,
    name: u32,
}

pub(crate) fn check_dispatches(
    prog: &Program,
    sigs: &Sigs,
    tys: &mut TypeTable,
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let arch = prog.target.arch();
    for (k, d) in prog.decls.iter().enumerate() {
        if d.kind != DeclKind::Fn || !d.flags.has(DeclFlags::DISPATCH) || d.node == u32::MAX {
            continue;
        }
        let Some(sig) = sigs.fn_sig(DeclId(k as u32)) else { continue };
        let Some(icx) = brackets.get(d.item.idx()).and_then(|b| ItemCx::new(prog, d.item, b)) else {
            continue;
        };
        let arms = arms_of(&icx, d.node, &mut out);
        let own = FeatureSet::from_bits(sig.features).unwrap_or_default();
        if !own.is_owned_by(arch) {
            continue;
        }
        for arm in &arms {
            let want = match arm.features {
                Some(f) => own.union(f),
                None => own,
            };
            if !want.is_owned_by(arch) {
                continue;
            }
            check_arm(prog, sigs, tys, &icx, sig, d.name_span, arm, want, &mut out);
        }
    }
    out
}

/// The arms of dispatch node `n` in order (the `_` arm last); E0708 for unknown feature names.
fn arms_of(icx: &ItemCx<'_>, n: u32, out: &mut Vec<Diagnostic>) -> Vec<Arm> {
    let mut arms = Vec::new();
    for c in icx.kids_vec(n) {
        match icx.kind(c) {
            Some(NodeKind::DispArm) => {
                let mut fs = FeatureSet::EMPTY;
                let mut name = None;
                for x in icx.kids_vec(c) {
                    match icx.kind(x) {
                        Some(NodeKind::Feature) => {
                            let t = icx.a(x);
                            let s = icx.tok_str(t);
                            match Feature::from_name(s) {
                                Some(f) => fs.insert(f),
                                None => out.extend(mk("E0708", None, &[("name", s)], icx.tok_span(t), &[])),
                            }
                        }
                        Some(NodeKind::Name) => name = Some(x),
                        _ => {}
                    }
                }
                if let Some(name) = name {
                    arms.push(Arm { features: Some(fs), name });
                }
            }
            Some(NodeKind::Name) => arms.push(Arm { features: None, name: c }),
            _ => {}
        }
    }
    arms
}

#[allow(clippy::too_many_arguments)]
fn check_arm(
    prog: &Program,
    sigs: &Sigs,
    tys: &mut TypeTable,
    icx: &ItemCx<'_>,
    disp: &FnSig,
    disp_span: Span,
    arm: &Arm,
    want: FeatureSet,
    out: &mut Vec<Diagnostic>,
) {
    let Some(u) = icx.uses(arm.name).last() else { return };
    let Res::Decl(t) = u.res else { return };
    let at = icx.node_span(arm.name);
    let roles = [("dispatch", disp_span)];
    let expected = fn_type_text(prog, tys, disp, want.bits());
    let target = prog.decl(t).filter(|x| matches!(x.kind, DeclKind::Fn | DeclKind::Method));
    let Some(ts) = target.and_then(|_| sigs.fn_sig(t)) else {
        let found = prog.decl_path(t);
        out.extend(mk("E0705", None, &[("expected", &expected), ("found", &found)], at, &roles));
        return;
    };
    let same_shape = ts.generics.is_empty()
        && ts.params == disp.params
        && ts.ret == disp.ret
        && ts.origins == disp.origins
        && ts.is_unsafe == disp.is_unsafe
        && !ts.variadic;
    let have = FeatureSet::from_bits(ts.features).unwrap_or_default();
    match arm.features {
        Some(_) if same_shape && have == want => {}
        None if same_shape => {
            if !have.is_subset(want) {
                let extra = types::feature_names(have.bits() & !want.bits());
                let f = extra.first().copied().unwrap_or("?");
                out.extend(mk("E0706", Some("extra-needs"), &[("feature", f)], at, &roles[..0]));
            }
        }
        _ => {
            let found = fn_type_text(prog, tys, ts, ts.features);
            out.extend(mk("E0705", None, &[("expected", &expected), ("found", &found)], at, &roles));
        }
    }
}

/// How a reachable `dispatch fn` is lowered on the build target.
///
/// The live arms are those whose features (with the dispatch's own) the target's arch owns. With
/// K = target baseline ∪ the dispatch's needs, the first live arm whose features are ⊆ K always
/// matches when reached: it ends the chain (`last`), and the arms below it are not lowered. The
/// `_` arm has no features, so a chain always ends.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// Arms tested at run time, top-down: (feature mask, target). The mask is the arm's closed
    /// feature set, so `(word & mask) == mask` is a complete test of the closed feature word.
    pub tests: Vec<(u64, InstKey)>,
    pub last: InstKey,
}

impl Plan {
    /// The first live arm: its features and target.
    pub fn first(&self) -> (u64, &InstKey) {
        match self.tests.first() {
            Some((m, k)) => (*m, k),
            None => (0, &self.last),
        }
    }
}

/// The [`Plan`] of dispatch `d` (`None`: `d` is not a dispatch with a resolved `_` arm and live arm
/// targets with checked, non-generic bodies; the type checker rules that out).
pub(crate) fn plan(prog: &Program, tc: &Typeck, icx: &ItemCx<'_>, d: DeclId) -> Option<Plan> {
    let decl = prog.decl(d)?;
    if decl.kind != DeclKind::Fn || !decl.flags.has(DeclFlags::DISPATCH) || decl.node == u32::MAX {
        return None;
    }
    let own = FeatureSet::from_bits(tc.sigs.fn_sig(d)?.features)?;
    let arch = prog.target.arch();
    let known = prog.target.baseline().union(own);
    let mut tests = Vec::new();
    for arm in arms_of(icx, decl.node, &mut Vec::new()) {
        let fs = arm.features.unwrap_or(FeatureSet::EMPTY);
        if !own.union(fs).is_owned_by(arch) {
            continue;
        }
        let Res::Decl(t) = icx.uses(arm.name).last()?.res else { return None };
        let ok = prog.decl(t).is_some_and(|x| matches!(x.kind, DeclKind::Fn | DeclKind::Method))
            && tc.body(t).is_some_and(|b| !b.has_error)
            && tc.sigs.fn_sig(t).is_some_and(|s| s.generics.is_empty());
        if !ok {
            return None;
        }
        let k = InstKey { decl: t, args: Vec::new() };
        if fs.is_subset(known) {
            return Some(Plan { tests, last: k });
        }
        tests.push((fs.bits(), k));
    }
    None
}

/// The text of the fn type of `sig` with features `features`.
fn fn_type_text(prog: &Program, tys: &mut TypeTable, sig: &FnSig, features: u64) -> String {
    let t =
        tys.intern(TypeKind::Fn { params: sig.params.clone().into_boxed_slice(), ret: sig.ret, features });
    types::display(&*tys, &NameCx { prog, generics: &sig.generics }, t)
}
