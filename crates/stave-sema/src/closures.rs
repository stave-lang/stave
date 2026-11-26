//! Closure capture analysis (contracts/checked-ir.md "Closures").
//!
//! A closure literal `fn(x: T) -> U { .. }` or a `spawn { .. }` body is a *closure body*: checked
//! with the outer environment (resolve and typeck see the enclosing fn's locals), lowered as its
//! own function `@<enclosing>.<k>`, `k` = its index among the enclosing body's closure bodies in
//! source order (the position of the `fn` / `spawn` token; nested bodies count in the same
//! sequence). [`closures`] lists them with their captures; lowering, the borrow checker (stave-borrow),
//! and the thread rules (E0802/E0806) all read this one list.
//!
//! **Captures.** A capture is an outer local (a binding of the enclosing fn or of an enclosing
//! closure body, including `self`) used anywhere inside the body, nested bodies included: a body
//! captures everything its nested bodies capture from outside it. Captures are sorted by binding
//! node (declaration order). Each has a [`CaptureMode`]:
//!
//! - [`CaptureMode::Ref`]: every use of the local in the body (nested bodies included) is a
//!   proven read: a shared loan.
//! - [`CaptureMode::RefMut`]: any other capture: a unique loan. The local has a `&mut T` type, or
//!   some use is not proven a read ([`use_reads`]). Fail-closed: a use counts as a read only in a
//!   context listed there, whatever dereferences lie between the local and the context.
//!
//! A use is the head of a place `x`, `x.f`, `x.0`, `x[i]`, `*x`, `(x)`, `*x.p[i]` (any mix of
//! these steps, auto-deref through `&` and `&mut` included). The place is read when it is:
//! the receiver of a built-in method or of a `&self` or by-value `self` method; called as a fn
//! value; the operand of `&` (not `&mut`); an f-string hole; used by value (an operand, an
//! argument, an initializer, a returned value, a condition, an index, a field or element of a
//! literal) where its type is not `&mut T` (a `&mut T` passed as `&T` is a read); the scrutinee
//! of a `match` or a destructuring `let` whose type holds no `&mut` (patterns bind through
//! peeled references); or the sequence of a `for` over a range, an array value, or `&[T]` /
//! `&[T; N]` whose bindings hold no `&mut` (an array or `[T]` place is no source: E0219). A by-value use of a non-Copy place is a move
//! out of the capture, E0504 in the borrow checker. Everything else (an assignment target, the
//! operand of `&mut`, a `&mut self` receiver, a `&mut` chunk or element binding, a `for` over an
//! iterator) mutates.
//!
//! The thread rules ([`crate::threads`]) and the borrow checker (stave-borrow) read these modes;
//! the borrow checker fails the body as unsupported where its CFG writes, mutably borrows, or
//! activates a two-phase loan of a capture held by `&` (the two must agree). Lowering takes
//! only the capture list: every environment entry is the captured slot's address.
//!
//! There is no copy or move capture (LANG.md §8).
//!
//! Every capture is a loan created at the closure literal (its `fn` / `spawn` token) that lives
//! while the fn value lives (a `spawn` body: until its `scope`'s `}`). A by-value use of a
//! non-Copy capture (a move out of the environment) is not detected here: the borrow checker
//! reports it from the use site ([`Capture::first_use`] and the body's own uses).
//!
//! **Environment** (lowering, stave-ir `fn.make`): a body with captures takes one extra trailing
//! param, a pointer to a tuple of `&mut T` (one per capture, in capture order) pointing at the
//! captured locals' slots. A body without captures
//! has no environment param and is used like a fn item.

use stave_syntax::{NodeKind, Term};

use crate::itemcx::ItemCx;
use crate::loops::ForKind;
use crate::resolve::Res;
use crate::typeck::{Coercion, Target, TypedBody};
use crate::types::{Interner, TypeId, TypeKind};

/// Whether `t` is `&mut T` in `types`.
pub fn is_mut_ref<I: Interner + ?Sized>(types: &I, t: TypeId) -> bool {
    matches!(types.kind(t), TypeKind::Ref { mutable: true, .. })
}

/// What introduced a closure body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosureKind {
    /// `fn(..) -> R { .. }` (an expression: a `Primary` node starting with `fn`).
    Literal,
    /// `spawn { .. }` (a `Stmt` node starting with `spawn`).
    Spawn,
}

/// How a closure body holds one outer local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CaptureMode {
    /// A shared reference (`&T`).
    Ref,
    /// A unique reference (`&mut T`).
    RefMut,
}

/// One captured outer local.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capture {
    /// The binding node of the local (`Param`, `SelfParam`'s param, or a pattern binding).
    pub binding: u32,
    pub mode: CaptureMode,
    /// The token of the first use inside the body (source order).
    pub first_use: u32,
    /// The token of the first mutating use, for [`CaptureMode::RefMut`] (else `first_use`).
    pub first_mut: u32,
}

/// One closure body of an item's body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureInfo {
    /// The `Primary` node (literal) or `Stmt` node (`spawn`).
    pub node: u32,
    pub kind: ClosureKind,
    /// `k` of `@<enclosing>.<k>`.
    pub index: u32,
    /// The `fn` / `spawn` token (where the capture loans are created).
    pub token: u32,
    /// The directly enclosing closure body (index into the list), `None` for the fn itself.
    pub parent: Option<u32>,
    /// Sorted by binding node.
    pub captures: Vec<Capture>,
}

/// Whether node `m` starts a closure body, and which kind.
pub fn closure_kind(icx: &ItemCx<'_>, m: u32) -> Option<ClosureKind> {
    match icx.kind(m)? {
        NodeKind::Primary if icx.is(icx.a(m), Term::KwFn) => Some(ClosureKind::Literal),
        NodeKind::Stmt if icx.is(icx.a(m), Term::KwSpawn) => Some(ClosureKind::Spawn),
        _ => None,
    }
}

/// The environment of a closure capturing locals of types `caps`, each with its mode: the tuple
/// `(&T0, &mut T1, ..)` (`&` for a [`CaptureMode::Ref`] capture, which the body only reads, `&mut`
/// for a [`CaptureMode::RefMut`] one) and the `&mut` to it the body receives. The checker builds
/// it for every closure ([`crate::typeck::TypedBody::envs`]), so a capture too tall for the type
/// table is E0005 there, and lowering builds the same type.
pub fn env_type<I: Interner + ?Sized>(cx: &mut I, caps: &[(TypeId, CaptureMode)]) -> (TypeId, TypeId) {
    let elems: Vec<TypeId> = caps.iter().map(|&(t, m)| cx.mk_ref(m == CaptureMode::RefMut, t)).collect();
    let tup = cx.intern(crate::types::TypeKind::Tuple(elems.into_boxed_slice()));
    (tup, cx.mk_ref(true, tup))
}

/// The closure bodies inside the body at `root` (a fn decl node), in `@<enclosing>.<k>` order,
/// with their captures. `tys`: the table `tb`'s types live in.
pub fn closures<I: Interner + ?Sized>(
    icx: &ItemCx<'_>,
    tb: &TypedBody,
    tys: &I,
    root: u32,
) -> Vec<ClosureInfo> {
    let lo = icx.ast.descendants(root).start;
    let mut out: Vec<ClosureInfo> = closure_nodes(icx, root)
        .into_iter()
        .enumerate()
        .map(|(k, (node, kind))| ClosureInfo {
            node,
            kind,
            index: k as u32,
            token: icx.a(node),
            parent: None,
            captures: Vec::new(),
        })
        .collect();
    if out.is_empty() {
        return out;
    }
    // parents: the smallest enclosing body
    let ranges: Vec<(u32, u32)> = out.iter().map(|c| (icx.ast.descendants(c.node).start, c.node)).collect();
    for i in 0..out.len() {
        let (s, e) = ranges[i];
        let mut best: Option<(u32, u32)> = None;
        for (j, &(s2, e2)) in ranges.iter().enumerate() {
            if j != i && s2 <= s && e < e2 && best.is_none_or(|(_, size)| e2 - s2 < size) {
                best = Some((j as u32, e2 - s2));
            }
        }
        out[i].parent = best.map(|b| b.0);
    }
    let cx = UseCx { icx, tb, tys, parents: parents(icx, lo, root) };
    // per (key node, token) of a use of an outer local: `None` read, `Some(t)` mutating at `t`
    let mut verdicts: rustc_hash::FxHashMap<(u32, u32), Option<u32>> = rustc_hash::FxHashMap::default();
    for (i, c) in out.iter_mut().enumerate() {
        let (s, e) = ranges[i];
        let mut caps: Vec<Capture> = Vec::new();
        for m in s..e {
            for u in icx.uses(m) {
                let (Res::Local(b) | Res::SelfValue(b)) = u.res else { continue };
                if (s..=e).contains(&b.node) {
                    continue; // declared inside the body
                }
                let v = *verdicts.entry((m, u.token)).or_insert_with(|| use_reads(&cx, m, u.token));
                let x = match caps.iter_mut().position(|x| x.binding == b.node) {
                    Some(i) => &mut caps[i],
                    None => {
                        caps.push(Capture {
                            binding: b.node,
                            mode: CaptureMode::Ref,
                            first_use: u.token,
                            first_mut: u32::MAX,
                        });
                        caps.last_mut().expect("just pushed")
                    }
                };
                x.first_use = x.first_use.min(u.token);
                if let Some(t) = v {
                    x.mode = CaptureMode::RefMut;
                    x.first_mut = x.first_mut.min(t);
                }
            }
        }
        for x in &mut caps {
            let by_type = tb
                .local(crate::ids::BindingRef { node: x.binding })
                .is_some_and(|t| !t.is_none() && is_mut_ref(tys, t));
            if by_type {
                x.mode = CaptureMode::RefMut;
            }
            if x.first_mut == u32::MAX {
                x.first_mut = x.first_use;
            }
        }
        caps.sort_by_key(|x| x.binding);
        c.captures = caps;
    }
    out
}

/// The closure-body nodes inside the body at `root` (a fn decl node) in `@<enclosing>.<k>` order
/// (by their `fn` / `spawn` token), with their kinds.
pub fn closure_nodes(icx: &ItemCx<'_>, root: u32) -> Vec<(u32, ClosureKind)> {
    let lo = icx.ast.descendants(root).start;
    let mut out: Vec<(u32, ClosureKind)> = Vec::new();
    // A closure body starts at a `fn` or `spawn` token. The tokens of the fn's block are far fewer
    // than its nodes: a block without such a token has no closure body, and its nodes are skipped.
    let skip = icx.child(root, NodeKind::Block).and_then(|b| {
        let open = icx.a(b);
        let close = icx.close(open);
        let toks = icx.toks.get(open as usize..=close as usize)?;
        let none =
            close > open && !toks.iter().any(|t| t.kind == Term::KwFn as u8 || t.kind == Term::KwSpawn as u8);
        none.then(|| icx.ast.descendants(b).start..b + 1)
    });
    let skip = skip.unwrap_or(lo..lo);
    for m in (lo..skip.start).chain(skip.end..root + 1) {
        if let Some(kind) = closure_kind(icx, m) {
            out.push((m, kind));
        }
    }
    out.sort_by_key(|&(m, _)| icx.a(m));
    out
}

/// What [`use_reads`] needs: the body, its types, and the parent of every node of the body.
struct UseCx<'c, 'p, I: Interner + ?Sized> {
    icx: &'c ItemCx<'p>,
    tb: &'c TypedBody,
    tys: &'c I,
    /// (`lo`, `v`): `v[n - lo]` is the parent of node `n` (`u32::MAX`: none in the body).
    parents: (u32, Vec<u32>),
}

impl<I: Interner + ?Sized> UseCx<'_, '_, I> {
    fn parent(&self, n: u32) -> Option<u32> {
        let (lo, v) = &self.parents;
        n.checked_sub(*lo).and_then(|i| v.get(i as usize)).copied().filter(|&p| p != u32::MAX)
    }
}

fn parents(icx: &ItemCx<'_>, lo: u32, root: u32) -> (u32, Vec<u32>) {
    let mut v = vec![u32::MAX; (root + 1 - lo) as usize];
    for n in lo..=root {
        for k in icx.kids_vec(n) {
            if let Some(slot) = k.checked_sub(lo).and_then(|i| v.get_mut(i as usize)) {
                *slot = n;
            }
        }
    }
    (lo, v)
}

/// Whether a value of type `t` may hold a `&mut` that a pattern or a copy reaches without going
/// through a `&`: a `&mut`, a view ADT, a generic param, `Self`, or an unknown type, at any depth
/// outside a shared reference, a raw pointer, or a fn type.
fn may_hold_mut<I: Interner + ?Sized>(prog: &crate::Program, tys: &I, t: TypeId) -> bool {
    let mut stack = vec![t];
    // each distinct component once (exact; types share components)
    let mut seen: rustc_hash::FxHashSet<TypeId> = rustc_hash::FxHashSet::default();
    let mut kids = Vec::new();
    while let Some(x) = stack.pop() {
        if x.is_none() {
            return true;
        }
        if !seen.insert(x) {
            continue;
        }
        match tys.kind(x) {
            TypeKind::Ref { mutable: true, .. } | TypeKind::Param(_) | TypeKind::SelfTy | TypeKind::Error => {
                return true;
            }
            TypeKind::Ref { mutable: false, .. } | TypeKind::Ptr { .. } | TypeKind::Fn { .. } => {}
            TypeKind::Adt { def, .. } if crate::kinds::adt_is_view(prog, *def) => return true,
            k => {
                kids.clear();
                k.children(&mut kids);
                stack.extend_from_slice(&kids);
            }
        }
    }
    false
}

/// Whether the use of an outer local at token `tok` (keyed at node `key`) is a proven read
/// (`None`); else the token of the construct that may mutate it (module docs).
fn use_reads<I: Interner + ?Sized>(cx: &UseCx<'_, '_, I>, key: u32, tok: u32) -> Option<u32> {
    let icx = cx.icx;
    if icx.kind(key) != Some(NodeKind::Primary) {
        return Some(tok);
    }
    let a = icx.a(key);
    if icx.is(a, Term::FmtLit) {
        // an f-string hole is formatted through `&`
        return None;
    }
    if a != tok || !(icx.is(a, Term::Ident) || icx.is(a, Term::KwSelf)) || icx.primary_end(key) != a {
        return Some(tok);
    }
    let mut cur = key;
    for _ in 0..1_000_000 {
        let Some(p) = cx.parent(cur) else { return Some(icx.a(cur)) };
        let pa = icx.a(p);
        match icx.kind(p) {
            Some(NodeKind::Postfix) if icx.first_child(p) == Some(cur) => match chain_reads(cx, p, cur) {
                Chain::Place => cur = p,
                Chain::Read => return None,
                Chain::Mut(t) => return Some(t),
            },
            // an index operand
            Some(NodeKind::Postfix) => return by_value(cx, cur),
            Some(NodeKind::Unary) => {
                if icx.is(pa, Term::Star) {
                    cur = p;
                } else if icx.is(pa, Term::Amp) || icx.is(pa, Term::AmpAmp) {
                    // `&&x` = `& &x`: the outer half when the child Unary starts at the same token
                    let outer = icx.is(pa, Term::AmpAmp)
                        && icx.kind(cur) == Some(NodeKind::Unary)
                        && icx.a(cur) == pa;
                    return if !outer && icx.is(pa + 1, Term::KwMut) { Some(pa) } else { None };
                } else if icx.is(pa, Term::KwUnsafe) {
                    if icx.last_child(p) != Some(cur) {
                        return Some(icx.a(cur));
                    }
                    cur = p;
                } else {
                    return by_value(cx, cur);
                }
            }
            Some(NodeKind::Primary) => {
                let kids = icx.kids_vec(p);
                if icx.is(pa, Term::LParen) && kids.len() == 1 && !icx.is(icx.close(pa) - 1, Term::Comma) {
                    cur = p;
                } else if icx.is(pa, Term::LParen) || icx.is(pa, Term::LBracket) {
                    return by_value(cx, cur);
                } else {
                    return Some(icx.a(cur));
                }
            }
            Some(
                NodeKind::Args
                | NodeKind::StructLit
                | NodeKind::Add
                | NodeKind::Mul
                | NodeKind::Shift
                | NodeKind::BitAnd
                | NodeKind::BitOr
                | NodeKind::BitXor
                | NodeKind::CmpExpr
                | NodeKind::AndExpr
                | NodeKind::OrExpr
                | NodeKind::RangeExpr
                | NodeKind::Cast
                | NodeKind::IfExpr
                | NodeKind::IfStmt
                | NodeKind::ExprArm
                | NodeKind::StmtArm,
            ) => return by_value(cx, cur),
            Some(NodeKind::MatchExpr) if icx.first_child(p) == Some(cur) => return scrutinee(cx, cur),
            Some(NodeKind::ExprStmt) => {
                let kids = icx.kids_vec(p);
                if kids.len() >= 3 && kids.first() == Some(&cur) {
                    return Some(pa);
                }
                return by_value(cx, cur);
            }
            Some(NodeKind::Stmt) => {
                if icx.is(pa, Term::KwLet) {
                    let plain = icx.first_child(p).is_some_and(|pat| binds_whole(icx, pat));
                    return if plain { by_value(cx, cur) } else { scrutinee(cx, cur) };
                }
                if icx.is(pa, Term::KwReturn) || icx.is(pa, Term::KwWhile) {
                    return by_value(cx, cur);
                }
                if icx.is(pa, Term::KwMatch) {
                    return scrutinee(cx, cur);
                }
                return Some(icx.a(cur));
            }
            Some(NodeKind::ForStmt) => return for_reads(cx, p, cur),
            _ => return Some(icx.a(cur)),
        }
    }
    Some(tok)
}

enum Chain {
    /// Every step is a field, element, or sub-slice: the whole chain is a place of the local.
    Place,
    Read,
    Mut(u32),
}

/// The steps of postfix chain `n` after its head `head` (a place of the local).
fn chain_reads<I: Interner + ?Sized>(cx: &UseCx<'_, '_, I>, n: u32, head: u32) -> Chain {
    let (icx, tb) = (cx.icx, cx.tb);
    let at = icx.a(n);
    let end = icx.postfix_end(n);
    let mut p = icx.primary_end(head) + 1;
    while p <= end {
        let t = icx.tk(p);
        if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
            let seg = p + 1;
            let mut q = p + 2;
            if icx.is(q, Term::ColonColon) {
                let (e, partial) = icx.angle_end(q + 1);
                q = if partial { e - 1 } else { e };
            }
            if icx.is(q, Term::LParen) {
                return match tb.target(seg) {
                    // a built-in takes its receiver by value (a scalar, vector, or pointer) or by `&`
                    Some(Target::Builtin { .. }) => Chain::Read,
                    Some(Target::Method { adjust, .. }) => {
                        if adjust.autoref == Some(true) || adjust.reborrow {
                            Chain::Mut(at)
                        } else {
                            Chain::Read
                        }
                    }
                    Some(Target::Field { .. }) if matches!(tb.target(q), Some(Target::FnValue { .. })) => {
                        Chain::Read
                    }
                    _ => Chain::Mut(at),
                };
            }
            match tb.target(seg) {
                Some(Target::Field { .. } | Target::Column { .. }) => {}
                _ => return Chain::Mut(at),
            }
            p = q;
        } else if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
            if !matches!(tb.target(p + 1), Some(Target::TupleField { .. })) {
                return Chain::Mut(at);
            }
            p += 2;
        } else if t == Term::LBracket as u8 {
            if !matches!(tb.target(p), Some(Target::Index { .. } | Target::Slice { .. })) {
                return Chain::Mut(at);
            }
            p = icx.close(p) + 1;
        } else if t == Term::LParen as u8 {
            return if matches!(tb.target(p), Some(Target::FnValue { .. })) {
                Chain::Read
            } else {
                Chain::Mut(at)
            };
        } else if t == Term::Question as u8 {
            // by value: a copy, or a move out of the capture (E0504)
            return Chain::Read;
        } else {
            return Chain::Mut(at);
        }
    }
    Chain::Place
}

/// Place `n` used by value: a read unless it is a `&mut T` (a reborrow or a move of it) not
/// passed as `&T`. A non-Copy place is moved out of the capture: E0504.
fn by_value<I: Interner + ?Sized>(cx: &UseCx<'_, '_, I>, n: u32) -> Option<u32> {
    let t = cx.tb.ty(n);
    if t.is_none() {
        return Some(cx.icx.a(n));
    }
    match cx.tys.kind(t) {
        TypeKind::Ref { mutable: true, .. } if cx.tb.coercion(n) != Some(&Coercion::MutToShared) => {
            Some(cx.icx.a(n))
        }
        TypeKind::Error => Some(cx.icx.a(n)),
        _ => None,
    }
}

/// Place `n` matched against patterns (a `match`, a destructuring `let`): patterns bind through
/// the references they peel.
fn scrutinee<I: Interner + ?Sized>(cx: &UseCx<'_, '_, I>, n: u32) -> Option<u32> {
    let t = cx.tb.ty(n);
    if may_hold_mut(cx.icx.prog, cx.tys, t) { Some(cx.icx.a(n)) } else { None }
}

/// Whether pattern `pat` binds the whole value to one name (or to `_`).
fn binds_whole(icx: &ItemCx<'_>, pat: u32) -> bool {
    if icx.kind(pat) != Some(NodeKind::Pattern) || !icx.ast.descendants(pat).is_empty() {
        return false;
    }
    let a = icx.a(pat);
    if icx.is(a, Term::Underscore) || icx.is(a, Term::KwMut) {
        return true;
    }
    icx.is(a, Term::Ident)
        && icx.chain_end(a) == a
        && !icx.is(a + 1, Term::LParen)
        && !icx.is(a + 1, Term::DotLBrace)
}

/// Place `n` as the sequence of `for` loop `f`.
fn for_reads<I: Interner + ?Sized>(cx: &UseCx<'_, '_, I>, f: u32, n: u32) -> Option<u32> {
    let (icx, tb) = (cx.icx, cx.tb);
    let at = Some(icx.a(f));
    let kids = icx.kids_vec(f);
    if kids.iter().copied().find(|&c| icx.is_expr(c)) != Some(n) {
        return at;
    }
    // the loop's bindings (pattern and tail): a `&mut` element or chunk writes the sequence
    let mut binds: Vec<u32> = Vec::new();
    if let Some(&pat) = kids.first() {
        binds.extend(icx.ast.descendants(pat));
        binds.push(pat);
    }
    if let Some(tc) = kids.iter().copied().find(|&c| icx.kind(c) == Some(NodeKind::TailClause)) {
        binds.extend(icx.kids_vec(tc).into_iter().filter(|&c| icx.kind(c) == Some(NodeKind::TailBind)));
    }
    for b in binds {
        if let Some(t) = tb.local(crate::ids::BindingRef { node: b })
            && may_hold_mut(icx.prog, cx.tys, t)
        {
            return at;
        }
    }
    // the loop's kind, as the type checker decided it (crate::loops)
    match tb.for_kind(f) {
        Some(ForKind::Range | ForKind::Array | ForKind::Seq { mutable: false }) => None,
        // `&mut` elements, an iterator (`next(&mut self)`), or no source
        _ => at,
    }
}
