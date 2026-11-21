//! Patterns: typing of `let`, `for`, and `match` patterns (bindings get their types),
//! and exhaustiveness (E0300) and reachability (E0301) by the usefulness algorithm over
//! constructors: enum variants (declaration order), the single constructor of structs and tuples,
//! `false`/`true`, and int and char ranges split at the patterns' boundaries (so a match on `u8`
//! can be covered by ranges). Floats and strings have no finite constructor set: only a binding or
//! `_` covers them.
//!
//! **Witness rule** (stage1 follows it too): constructors are tried smallest first, in declaration order
//! for variants, ascending for ranges, `false` before `true`, columns left to right; the first
//! uncovered one is the witness. The E0300 message prints only the scrutinee type.
//!
//! **References** (not in LANG.md): a constructor or literal pattern matched against `&T` or
//! `&mut T` peels the reference, and the bindings below bind by reference (`&F` / `&mut F`).
//! A `let`/`for` pattern must be irrefutable (E0300 at the pattern).
//!
//! **Or-patterns** (`P1 | P2 =>`, match arms only): every alternative is one row of the match.
//! The alternatives are tried in order and the first that matches binds the names, so within
//! its arm an alternative is unreachable when the earlier ones cover it, guard or not. Only the
//! first alternative's bindings are locals; the later alternatives must bind the same names with
//! the same types and modes (E0302) and bind those locals.

use stave_diag::Span;
use stave_syntax::{NodeKind, Term};

use crate::decls::DeclKind;
use crate::diag::mk;
use crate::resolve::{Binding, Res};
use crate::sigs::VariantShape;
use crate::typeck::{Ck, Ty};
use crate::types::{GArg, Interner, TypeId, TypeKind, ids};

/// How bindings bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Move,
    /// Through a peeled reference (`true`: `&mut`).
    Ref(bool),
}

/// A constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctor {
    Variant(u32),
    /// Structs and tuples.
    Single,
    Bool(bool),
    /// Inclusive int or char range.
    Range(i128, i128),
    /// A float or string literal (equal by source text hash).
    Opaque(u64),
}

/// A lowered pattern.
#[derive(Clone, Debug, PartialEq)]
pub enum Pat {
    Wild,
    Ctor(Ctor, Vec<Pat>),
    /// Ill-typed (diagnosed): exhaustiveness is skipped.
    Err,
}

fn bind_ty(ck: &mut Ck<'_>, t: TypeId, mode: Mode) -> TypeId {
    match mode {
        Mode::Move => t,
        Mode::Ref(m) => ck.tys.mk_ref(m, t),
    }
}

/// Gives binding node `node` its type: `t` bound through `mode`.
fn bind(ck: &mut Ck<'_>, node: u32, t: TypeId, mode: Mode) {
    let bt = bind_ty(ck, t, mode);
    ck.locals.insert(node, bt);
    if mode == Mode::Move {
        let genv = ck.genv();
        if !ck.w.is_copy(&mut ck.tys, t, genv) {
            ck.moves.push(node);
        }
    }
    if let Some(binds) = &mut ck.or_binds {
        binds.push((node, bt, mode));
    }
}

fn peel(ck: &Ck<'_>, t: TypeId, mode: Mode) -> (TypeId, Mode) {
    let mut t = t;
    let mut mode = mode;
    for _ in 0..1_000_000 {
        match ck.tys.kind(t) {
            TypeKind::Ref { mutable, inner } => {
                mode = match mode {
                    Mode::Move => Mode::Ref(*mutable),
                    Mode::Ref(m) => Mode::Ref(m && *mutable),
                };
                t = *inner;
            }
            _ => break,
        }
    }
    (t, mode)
}

fn hash_text(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Types pattern `p` against `t` and binds its names; returns the lowered pattern.
pub(crate) fn check_pattern(ck: &mut Ck<'_>, p: u32, t: TypeId, mode: Mode) -> Pat {
    if !ck.enter(p) {
        return Pat::Err;
    }
    let r = pattern_inner(ck, p, t, mode);
    ck.leave();
    r
}

fn pattern_inner(ck: &mut Ck<'_>, p: u32, t: TypeId, mode: Mode) -> Pat {
    let icx = ck.icx;
    let a = icx.a(p);
    let tk = icx.tk(a);
    if tk == Term::Underscore as u8 {
        return Pat::Wild;
    }
    if tk == Term::KwMut as u8 {
        bind(ck, p, t, mode);
        return Pat::Wild;
    }
    let kids = icx.kids_vec(p);
    if icx.kind(kids.first().copied().unwrap_or(u32::MAX)) == Some(NodeKind::Literal) {
        return literal_pattern(ck, p, &kids, t, mode);
    }
    if tk == Term::LParen as u8 {
        let close = icx.close(a);
        if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
            return check_pattern(ck, kids[0], t, mode);
        }
        if t == ids::ERROR {
            for &k in &kids {
                check_pattern(ck, k, ids::ERROR, mode);
            }
            return Pat::Err;
        }
        let (pt, pm) = peel(ck, t, mode);
        let elems = match ck.tys.kind(pt).clone() {
            TypeKind::Tuple(ts) if ts.len() == kids.len() => ts.to_vec(),
            _ => {
                let found = tuple_placeholder(ck, kids.len());
                mismatch_pat(ck, p, found, t);
                for &k in &kids {
                    check_pattern(ck, k, ids::ERROR, mode);
                }
                return Pat::Err;
            }
        };
        let mut subs = Vec::new();
        for (k, &e) in kids.iter().zip(&elems) {
            subs.push(check_pattern(ck, *k, e, pm));
        }
        return Pat::Ctor(Ctor::Single, subs);
    }
    if tk != Term::Ident as u8 {
        return Pat::Err;
    }
    let e = icx.chain_end(a);
    let has_paren = icx.is(e + 1, Term::LParen);
    let has_rec = icx.is(e + 1, Term::DotLBrace);
    if e == a && !has_paren && !has_rec {
        bind(ck, p, t, mode);
        return Pat::Wild;
    }
    let Some(u) = icx.uses(p).last() else { return sub_err(ck, &kids) };
    let Res::Decl(d) = u.res else { return sub_err(ck, &kids) };
    let Some(decl) = ck.prog.decl(d).copied() else { return sub_err(ck, &kids) };
    let (pt, pm) = peel(ck, t, mode);
    match decl.kind {
        DeclKind::Variant | DeclKind::Struct => {
            let adt = if decl.kind == DeclKind::Variant { decl.parent } else { d };
            let Some(adef) = ck.sigs.adt(adt) else { return sub_err(ck, &kids) };
            let vi = if decl.kind == DeclKind::Variant { decl.index as usize } else { 0 };
            let Some(v) = adef.variants.get(vi).cloned() else { return sub_err(ck, &kids) };
            let args: Vec<GArg> = match ck.tys.kind(pt).clone() {
                TypeKind::Adt { def, args } if def == adt => args.to_vec(),
                TypeKind::Error => return sub_err(ck, &kids),
                _ => {
                    let found = adt_generic(ck, adt);
                    mismatch_pat(ck, p, found, t);
                    return sub_err(ck, &kids);
                }
            };
            let ftys: Vec<TypeId> = v.fields.iter().map(|f| ck.tys.subst(f.ty, &args, None)).collect();
            let ctor = if decl.kind == DeclKind::Variant { Ctor::Variant(vi as u32) } else { Ctor::Single };
            if has_rec {
                if v.shape != VariantShape::Record {
                    return arity_pat(ck, p, ftys.len(), kids.len(), &kids);
                }
                let mut subs = vec![Pat::Wild; ftys.len()];
                let mut seen: Vec<(usize, u32, bool)> = Vec::new();
                for fp in kids {
                    if icx.kind(fp) != Some(NodeKind::FieldPat) {
                        continue;
                    }
                    let fa = icx.a(fp);
                    let name = icx.sym(fa);
                    let Some(fi) = v.fields.iter().position(|f| f.name == name) else {
                        let tn = ck.show(pt);
                        let nm = ck.prog.strings.get(name).to_string();
                        let dg = mk("E0208", None, &[("name", &nm), ("type", &tn)], icx.tok_span(fa), &[]);
                        ck.report(dg);
                        if let Some(sp) = icx.child(fp, NodeKind::Pattern) {
                            check_pattern(ck, sp, ids::ERROR, pm);
                        } else {
                            bind(ck, fp, ids::ERROR, Mode::Move);
                        }
                        continue;
                    };
                    // a pattern names a field once: each part of the value has one sub-pattern,
                    // which alone says whether the part is bound, tested, or ignored
                    if let Some(&(_, first, first_binds)) = seen.iter().find(|x| x.0 == fi) {
                        let sub = icx.child(fp, NodeKind::Pattern);
                        // `P.{ a, a }` binds the name `a` twice, which is reported as that
                        if !(first_binds && sub.is_none()) {
                            let nm = ck.prog.strings.get(name).to_string();
                            let scope = ck.prog.decl_path(v.decl);
                            let dg = mk(
                                "E0105",
                                None,
                                &[("name", &nm), ("scope", &scope)],
                                icx.tok_span(fa),
                                &[("first", icx.tok_span(first))],
                            );
                            ck.report(dg);
                        }
                        ck.has_error = true;
                        match sub {
                            Some(sp) => {
                                check_pattern(ck, sp, ids::ERROR, pm);
                            }
                            None => bind(ck, fp, ids::ERROR, Mode::Move),
                        }
                        continue;
                    }
                    seen.push((fi, fa, icx.child(fp, NodeKind::Pattern).is_none()));
                    match icx.child(fp, NodeKind::Pattern) {
                        Some(sp) => subs[fi] = check_pattern(ck, sp, ftys[fi], pm),
                        None => bind(ck, fp, ftys[fi], pm),
                    }
                }
                return Pat::Ctor(ctor, subs);
            }
            if has_paren {
                if v.shape != VariantShape::Tuple || kids.len() != ftys.len() {
                    return arity_pat(ck, p, ftys.len(), kids.len(), &kids);
                }
                let mut subs = Vec::new();
                for (k, ft) in kids.iter().zip(&ftys) {
                    subs.push(check_pattern(ck, *k, *ft, pm));
                }
                return Pat::Ctor(ctor, subs);
            }
            if v.shape != VariantShape::Unit {
                return arity_pat(ck, p, ftys.len(), 0, &kids);
            }
            Pat::Ctor(ctor, Vec::new())
        }
        DeclKind::Const => {
            let Some(c) = ck.sigs.konst(d) else { return Pat::Err };
            let ct = c.ty;
            if pt != ct && ct != ids::ERROR && pt != ids::ERROR {
                mismatch_pat(ck, p, ct, t);
                return Pat::Err;
            }
            match &c.value {
                Some(crate::consteval::Val::Int(v)) => Pat::Ctor(Ctor::Range(*v, *v), Vec::new()),
                Some(crate::consteval::Val::Bool(b)) => Pat::Ctor(Ctor::Bool(*b), Vec::new()),
                Some(crate::consteval::Val::Char(v)) => {
                    Pat::Ctor(Ctor::Range(*v as i128, *v as i128), Vec::new())
                }
                Some(_) => Pat::Ctor(Ctor::Opaque(hash_text(icx.node_text(p))), Vec::new()),
                None => Pat::Err,
            }
        }
        DeclKind::Error => sub_err(ck, &kids),
        _ => {
            let text = icx.node_text(p).to_string();
            let dg = mk("E0201", Some("not-value"), &[("operand", &text)], icx.node_span(p), &[]);
            ck.report(dg);
            sub_err(ck, &kids)
        }
    }
}

fn tuple_placeholder(ck: &mut Ck<'_>, n: usize) -> TypeId {
    let ts: Vec<TypeId> = (0..n).map(|i| ck.tys.mk_param(1000 + i as u32)).collect();
    ck.tys.intern(TypeKind::Tuple(ts.into_boxed_slice()))
}

fn adt_generic(ck: &mut Ck<'_>, adt: crate::ids::DeclId) -> TypeId {
    let n = ck.sigs.adt(adt).map_or(0, |a| a.generics.len());
    let args: Vec<GArg> = (0..n).map(|i| GArg::Type(ck.tys.mk_param(1000 + i as u32))).collect();
    ck.tys.mk_adt(adt, args)
}

fn sub_err(ck: &mut Ck<'_>, kids: &[u32]) -> Pat {
    for &k in kids {
        match ck.icx.kind(k) {
            Some(NodeKind::Pattern) => {
                check_pattern(ck, k, ids::ERROR, Mode::Move);
            }
            Some(NodeKind::FieldPat) => match ck.icx.child(k, NodeKind::Pattern) {
                Some(sp) => {
                    check_pattern(ck, sp, ids::ERROR, Mode::Move);
                }
                None => bind(ck, k, ids::ERROR, Mode::Move),
            },
            _ => {}
        }
    }
    ck.has_error = true;
    Pat::Err
}

fn arity_pat(ck: &mut Ck<'_>, p: u32, want: usize, got: usize, kids: &[u32]) -> Pat {
    let text = ck.icx.node_text(p).to_string();
    let d = mk(
        "E0201",
        Some("arity"),
        &[("operand", &text), ("count", &want.to_string()), ("given", &got.to_string())],
        ck.icx.node_span(p),
        &[],
    );
    ck.report(d);
    sub_err(ck, kids)
}

fn mismatch_pat(ck: &mut Ck<'_>, p: u32, found: TypeId, want: TypeId) {
    if want == ids::ERROR || ck.tys.has_error(want) {
        return;
    }
    let (e, f) = (ck.show(want), ck.show(found));
    let text = ck.icx.node_text(p).to_string();
    let d =
        mk("E0201", None, &[("expected", &e), ("found", &f), ("operand", &text)], ck.icx.node_span(p), &[]);
    ck.report(d);
}

/// Value of a `Literal` node: (kind, int value) where kind: 0 int, 1 float, 2 str, 3 char, 4 bool.
fn lit_value(ck: &mut Ck<'_>, l: u32) -> Option<(u8, i128, u128, bool)> {
    let icx = ck.icx;
    let a = icx.a(l);
    let (neg, t) = if icx.is(a, Term::Minus) { (true, a + 1) } else { (false, a) };
    let text = icx.tok_str(t);
    let k = icx.tk(t);
    if k == Term::IntLit as u8 {
        let (mag, _) = crate::literals::parse_int(text)?;
        let v = if mag > i128::MAX as u128 { i128::MAX } else { mag as i128 };
        return Some((0, if neg { -v } else { v }, mag, neg));
    }
    if k == Term::FloatLit as u8 {
        return Some((1, 0, 0, neg));
    }
    if k == Term::StrLit as u8 {
        return Some((2, 0, 0, false));
    }
    if k == Term::CharLit as u8 {
        return Some((3, crate::literals::parse_char(text)? as i128, 0, false));
    }
    if k == Term::KwTrue as u8 {
        return Some((4, 1, 0, false));
    }
    if k == Term::KwFalse as u8 {
        return Some((4, 0, 0, false));
    }
    None
}

fn literal_pattern(ck: &mut Ck<'_>, p: u32, kids: &[u32], t: TypeId, mode: Mode) -> Pat {
    let icx = ck.icx;
    let (pt, _) = if icx.is(icx.a(kids[0]), Term::StrLit) { (t, mode) } else { peel(ck, t, mode) };
    let mut vals = Vec::new();
    for &l in kids {
        let Some(v) = lit_value(ck, l) else {
            let text = icx.node_text(l).to_string();
            let d = mk("E0203", Some("malformed"), &[("literal", &text)], icx.node_span(l), &[]);
            ck.report(d);
            return Pat::Err;
        };
        vals.push((l, v));
    }
    if pt == ids::ERROR {
        return Pat::Err;
    }
    let (l0, (kind, v0, _, _)) = vals[0];
    let ok = match kind {
        0 => ck.tys.int(pt).is_some(),
        1 => ck.tys.is_float(pt),
        2 => pt == ids::REF_STR || pt == ids::STR,
        3 => pt == ids::CHAR,
        _ => pt == ids::BOOL,
    };
    if !ok {
        let found = match kind {
            0 => Ty::Lit(ck.new_lit(l0, 0, false, false)),
            1 => Ty::Lit(ck.new_lit(l0, 0, false, true)),
            2 => Ty::T(ids::REF_STR),
            3 => Ty::T(ids::CHAR),
            _ => Ty::T(ids::BOOL),
        };
        ck.mismatch(l0, found, pt, None);
        if let Ty::Lit(g) = found {
            let _ = ck.fix(g, ids::ERROR);
        }
        return Pat::Err;
    }
    // ranges of int literals: range-check each end
    if kind == 0 {
        for &(l, (_, _, mag, neg)) in &vals {
            if !crate::literals::int_fits(mag, neg, pt) {
                let g = ck.new_lit(l, mag, neg, false);
                let _ = ck.fix(g, pt);
            }
        }
    }
    let inclusive = vals.len() == 2 && icx.is(icx.end_tok(vals[0].0) + 1, Term::DotDotEq);
    match kind {
        0 | 3 => {
            let lo = v0;
            let hi = match vals.get(1) {
                Some(&(_, (_, v1, _, _))) => {
                    if inclusive {
                        v1
                    } else {
                        v1 - 1
                    }
                }
                None => v0,
            };
            if hi < lo {
                return Pat::Ctor(Ctor::Opaque(u64::MAX), Vec::new());
            }
            Pat::Ctor(Ctor::Range(lo, hi), Vec::new())
        }
        4 => Pat::Ctor(Ctor::Bool(v0 == 1), Vec::new()),
        _ => Pat::Ctor(Ctor::Opaque(hash_text(icx.node_text(p))), Vec::new()),
    }
}

// ---------- exhaustiveness ----------

#[derive(Clone, Debug)]
enum Desc {
    Enum(Vec<Vec<TypeId>>),
    Single(Vec<TypeId>),
    Bool,
    /// Domain as disjoint inclusive intervals.
    Ints(Vec<(i128, i128)>),
    Opaque,
}

fn desc(ck: &mut Ck<'_>, t: TypeId) -> Desc {
    let (t, _) = peel(ck, t, Mode::Move);
    match ck.tys.kind(t).clone() {
        TypeKind::Adt { def, args } => {
            match ck.sigs.adt(def) {
                Some(a) if a.is_enum => Desc::Enum(
                    a.variants
                        .iter()
                        .map(|v| v.fields.iter().map(|f| ck.tys.subst(f.ty, &args, None)).collect())
                        .collect(),
                ),
                Some(a) => Desc::Single(a.variants.first().map_or(Vec::new(), |v| {
                    v.fields.iter().map(|f| ck.tys.subst(f.ty, &args, None)).collect()
                })),
                None => Desc::Opaque,
            }
        }
        TypeKind::Tuple(ts) => Desc::Single(ts.to_vec()),
        TypeKind::Bool => Desc::Bool,
        TypeKind::Int { .. } => match crate::literals::int_range(t) {
            Some(r) => Desc::Ints(vec![r]),
            None => Desc::Opaque,
        },
        TypeKind::Char => Desc::Ints(vec![(0, 0xD7FF), (0xE000, 0x10FFFF)]),
        _ => Desc::Opaque,
    }
}

fn ctor_fields(d: &Desc, c: &Ctor) -> Vec<TypeId> {
    match (d, c) {
        (Desc::Enum(vs), Ctor::Variant(i)) => vs.get(*i as usize).cloned().unwrap_or_default(),
        (Desc::Single(fs), Ctor::Single) => fs.clone(),
        _ => Vec::new(),
    }
}

fn covers(row: &Ctor, c: &Ctor) -> bool {
    match (row, c) {
        (Ctor::Range(a, b), Ctor::Range(x, y)) => a <= x && y <= b,
        (a, b) => a == b,
    }
}

fn specialize(rows: &[Vec<Pat>], c: &Ctor, arity: usize) -> Vec<Vec<Pat>> {
    let mut out = Vec::new();
    for r in rows {
        let Some(h) = r.first() else { continue };
        match h {
            Pat::Wild | Pat::Err => {
                let mut nr = vec![Pat::Wild; arity];
                nr.extend_from_slice(&r[1..]);
                out.push(nr);
            }
            Pat::Ctor(rc, fields) => {
                if covers(rc, c) {
                    let mut nr = fields.clone();
                    nr.resize(arity, Pat::Wild);
                    nr.extend_from_slice(&r[1..]);
                    out.push(nr);
                }
            }
        }
    }
    out
}

/// Split intervals of `dom` restricted to `[lo, hi]` at the boundaries of the rows' head ranges.
fn split(rows: &[Vec<Pat>], dom: &[(i128, i128)], lo: i128, hi: i128) -> Vec<(i128, i128)> {
    let mut cuts: Vec<i128> = Vec::new();
    for r in rows {
        if let Some(Pat::Ctor(Ctor::Range(a, b), _)) = r.first() {
            cuts.push(*a);
            cuts.push(b.saturating_add(1));
        }
    }
    cuts.sort();
    cuts.dedup();
    let mut out = Vec::new();
    for &(dlo, dhi) in dom {
        let (s, e) = (dlo.max(lo), dhi.min(hi));
        if s > e {
            continue;
        }
        let mut cur = s;
        for &c in &cuts {
            if c > cur && c <= e {
                out.push((cur, c - 1));
                cur = c;
            }
        }
        out.push((cur, e));
    }
    out
}

/// Most nested columns and most [`useful`] calls of one match check (E0300 case `limit`).
const SEARCH_DEPTH: u32 = 2000;
const SEARCH_STEPS: u32 = 1_000_000;

/// The search budget of one match check. Past it [`useful`] answers "useful": the match counts
/// as not exhaustive and no arm as unreachable.
#[derive(Default)]
struct Search {
    steps: u32,
    exceeded: bool,
}

fn useful(ck: &mut Ck<'_>, rows: &[Vec<Pat>], v: &[Pat], tys: &[TypeId], depth: u32, s: &mut Search) -> bool {
    s.steps += 1;
    if s.exceeded || depth > SEARCH_DEPTH || s.steps > SEARCH_STEPS {
        s.exceeded = true;
        return true;
    }
    let Some(first) = v.first() else { return rows.is_empty() };
    let t0 = tys.first().copied().unwrap_or(ids::ERROR);
    let d = desc(ck, t0);
    let rest_tys = tys.get(1..).unwrap_or(&[]);
    let try_ctor = |ck: &mut Ck<'_>, s: &mut Search, c: Ctor, sub: Option<&[Pat]>| -> bool {
        let ft = ctor_fields(&d, &c);
        let spec = specialize(rows, &c, ft.len());
        let mut nv: Vec<Pat> = match sub {
            Some(s) => {
                let mut x = s.to_vec();
                x.resize(ft.len(), Pat::Wild);
                x
            }
            None => vec![Pat::Wild; ft.len()],
        };
        nv.extend_from_slice(&v[1..]);
        let mut nt = ft.clone();
        nt.extend_from_slice(rest_tys);
        useful(ck, &spec, &nv, &nt, depth + 1, s)
    };
    match first {
        Pat::Ctor(Ctor::Range(x, y), _) => {
            let dom = match &d {
                Desc::Ints(dom) => dom.clone(),
                _ => vec![(*x, *y)],
            };
            for (a, b) in split(rows, &dom, *x, *y) {
                if try_ctor(ck, s, Ctor::Range(a, b), None) {
                    return true;
                }
            }
            false
        }
        Pat::Ctor(c, fields) => {
            let c = *c;
            let f = fields.clone();
            try_ctor(ck, s, c, Some(&f))
        }
        Pat::Wild | Pat::Err => match &d {
            Desc::Enum(vs) => {
                let n = vs.len() as u32;
                (0..n).any(|i| try_ctor(ck, s, Ctor::Variant(i), None))
            }
            Desc::Single(_) => try_ctor(ck, s, Ctor::Single, None),
            Desc::Bool => try_ctor(ck, s, Ctor::Bool(false), None) || try_ctor(ck, s, Ctor::Bool(true), None),
            Desc::Ints(dom) => {
                let dom = dom.clone();
                for (a, b) in split(rows, &dom, i128::MIN, i128::MAX) {
                    if try_ctor(ck, s, Ctor::Range(a, b), None) {
                        return true;
                    }
                }
                false
            }
            Desc::Opaque => {
                let def: Vec<Vec<Pat>> = rows
                    .iter()
                    .filter(|r| matches!(r.first(), Some(Pat::Wild | Pat::Err)))
                    .map(|r| r[1..].to_vec())
                    .collect();
                useful(ck, &def, &v[1..], rest_tys, depth + 1, s)
            }
        },
    }
}

fn has_err(p: &Pat) -> bool {
    match p {
        Pat::Err => true,
        Pat::Wild => false,
        Pat::Ctor(_, fs) => fs.iter().any(has_err),
    }
}

/// E0300 case `limit` at `node` for a match (or `let`/`for` pattern) on `t`.
fn too_large(ck: &mut Ck<'_>, node: u32, t: TypeId) {
    let tn = ck.show(t);
    let d = mk("E0300", Some("limit"), &[("type", &tn)], ck.icx.node_span(node), &[]);
    ck.report(d);
}

/// One alternative of a match arm (an arm without `|` has one).
pub(crate) struct Row {
    pub pat: Pat,
    /// The arm has a guard.
    pub guarded: bool,
    /// The alternative's pattern node.
    pub node: u32,
    /// The arm's index in its match.
    pub arm: u32,
    /// The arm has several alternatives.
    pub alt: bool,
}

/// The bindings of the pattern rooted at `p`, in source order.
fn bindings_in<'p>(ck: &Ck<'p>, p: u32) -> &'p [Binding] {
    let icx = ck.icx;
    let lo = icx.ast.descendants(p).start;
    let b = &icx.res.bindings;
    let i = b.partition_point(|x| x.at.node < lo);
    let j = b.partition_point(|x| x.at.node <= p);
    &b[i..j]
}

/// E0302 for every name that alternative `alt` (typed bindings `binds`) does not bind the way
/// the first alternative `first_alt` (typed bindings `first`) does: a name only one of them
/// binds, or one bound with another type, by value in one and by reference in the other, or
/// `mut` in only one.
fn same_bindings(
    ck: &mut Ck<'_>,
    first_alt: u32,
    first: &[(u32, TypeId, Mode)],
    alt: u32,
    binds: &[(u32, TypeId, Mode)],
) {
    let icx = ck.icx;
    let typed =
        |list: &[(u32, TypeId, Mode)], node: u32| list.iter().find(|x| x.0 == node).map(|x| (x.1, x.2));
    let fb = bindings_in(ck, first_alt);
    let mut seen = vec![false; fb.len()];
    let mut bad: Vec<u32> = Vec::new();
    for b in bindings_in(ck, alt) {
        let pair = icx.res.alias(b.at.node).and_then(|c| fb.iter().position(|f| f.at.node == c));
        let Some(i) = pair else {
            bad.push(b.token);
            continue;
        };
        seen[i] = true;
        let f = &fb[i];
        let differs = match (typed(binds, b.at.node), typed(first, f.at.node)) {
            (Some((bt, bm)), Some((ft, fm))) => {
                bm != fm || (bt != ft && !ck.tys.has_error(bt) && !ck.tys.has_error(ft))
            }
            _ => false,
        };
        if differs || b.mutable != f.mutable {
            bad.push(b.token);
        }
    }
    bad.extend(fb.iter().zip(&seen).filter(|x| !*x.1).map(|x| x.0.token));
    let mut reported: Vec<&str> = Vec::new();
    for tok in bad {
        let name = icx.tok_str(tok);
        if reported.contains(&name) {
            continue;
        }
        reported.push(name);
        let roles = [("first", icx.node_span(first_alt))];
        let d = mk("E0302", None, &[("name", name)], icx.node_span(alt), &roles);
        ck.report(d);
    }
}

/// Types the alternatives of arm pattern `ap` (`P1 | P2 | ..`) of arm number `arm` against `t`,
/// binds the names of the first one, and appends a row per alternative to `rows`.
pub(crate) fn check_arm(ck: &mut Ck<'_>, ap: u32, t: TypeId, guarded: bool, arm: u32, rows: &mut Vec<Row>) {
    let icx = ck.icx;
    if let Some(p) = icx.only_child(ap) {
        let pat = check_pattern(ck, p, t, Mode::Move);
        rows.push(Row { pat, guarded, node: p, arm, alt: false });
        return;
    }
    let alts = icx.kids_vec(ap);
    let mut first: Vec<(u32, TypeId, Mode)> = Vec::new();
    for (k, &p) in alts.iter().enumerate() {
        ck.or_binds = Some(Vec::new());
        let pat = check_pattern(ck, p, t, Mode::Move);
        rows.push(Row { pat, guarded, node: p, arm, alt: true });
        let binds = ck.or_binds.take().unwrap_or_default();
        if k == 0 {
            first = binds;
            continue;
        }
        // a later alternative's bindings are occurrences of the first one's locals
        for b in &binds {
            ck.locals.remove(&b.0);
        }
        same_bindings(ck, alts[0], &first, p, &binds);
    }
}

/// E0301 for unreachable alternatives, E0300 when not exhaustive; returns whether the match is
/// exhaustive. `rows`: the alternatives of every arm, in source order.
pub(crate) fn check_match(ck: &mut Ck<'_>, scrut: u32, t: TypeId, rows: &[Row]) -> bool {
    if t == ids::ERROR || ck.tys.has_error(t) || rows.iter().any(|r| has_err(&r.pat)) {
        return true;
    }
    let tys = [t];
    let mut prev: Vec<Vec<Pat>> = Vec::new();
    let mut prev_nodes: Vec<u32> = Vec::new();
    let s = &mut Search::default();
    // where the rows of the current arm start in `prev`
    let mut arm_start = 0;
    for (i, r) in rows.iter().enumerate() {
        if i > 0 && rows[i - 1].arm != r.arm {
            if rows[i - 1].guarded {
                prev.truncate(arm_start);
                prev_nodes.truncate(arm_start);
            }
            arm_start = prev.len();
        }
        let p = &r.pat;
        if !useful(ck, &prev, std::slice::from_ref(p), &tys, 0, s) {
            // the covering row: the first k such that rows 0..=k cover it
            let mut cover = prev_nodes.last().copied();
            for k in 0..prev.len() {
                if !useful(ck, &prev[..=k], std::slice::from_ref(p), &tys, 0, s) {
                    cover = Some(prev_nodes[k]);
                    break;
                }
            }
            let roles: Vec<(&str, Span)> =
                cover.map(|c| ("covering", ck.icx.node_span(c))).into_iter().collect();
            let case = if r.alt { Some("alternative") } else { None };
            let d = mk("E0301", case, &[], ck.icx.node_span(r.node), &roles);
            ck.report(d);
        }
        // A guarded arm covers nothing for the later arms. Inside the arm its alternatives still
        // count: the guard runs once, after the first alternative that matches, so a later
        // alternative is not tried for that value.
        if !r.guarded || r.alt {
            prev.push(vec![p.clone()]);
            prev_nodes.push(r.node);
        }
    }
    if rows.last().is_some_and(|r| r.guarded) {
        prev.truncate(arm_start);
    }
    let open = useful(ck, &prev, &[Pat::Wild], &tys, 0, s);
    if s.exceeded {
        too_large(ck, scrut, t);
        return false;
    }
    if open {
        let tn = ck.show(t);
        let d = mk("E0300", None, &[("type", &tn)], ck.icx.node_span(scrut), &[]);
        ck.report(d);
        return false;
    }
    true
}

/// A `let`/`for` pattern must cover every value (E0300 at the pattern).
pub(crate) fn irrefutable(ck: &mut Ck<'_>, node: u32, p: &Pat, t: TypeId) {
    // A plain binding or `_` covers every value. The search is still run for ADTs and tuples:
    // describing their constructors interns the field types.
    if matches!(p, Pat::Wild) && !matches!(ck.tys.kind(t), TypeKind::Adt { .. } | TypeKind::Tuple(_)) {
        return;
    }
    if t == ids::ERROR || ck.tys.has_error(t) || has_err(p) {
        return;
    }
    let s = &mut Search::default();
    let open = useful(ck, &[vec![p.clone()]], &[Pat::Wild], &[t], 0, s);
    if s.exceeded {
        too_large(ck, node, t);
    } else if open {
        let tn = ck.show(t);
        let d = mk("E0300", None, &[("type", &tn)], ck.icx.node_span(node), &[]);
        ck.report(d);
    }
}
