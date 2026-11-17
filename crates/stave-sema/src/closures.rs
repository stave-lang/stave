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
//! - [`CaptureMode::RefMut`]: the body may mutate the local: it is the root of an assignment
//!   target (`x = ..`, `x.f = ..`, `x[i] += ..`), the operand root of `&mut`, the root of a method
//!   receiver auto-borrowed as `&mut self`, or the local has a `&mut T` type (using a `&mut`
//!   through a capture needs it uniquely; conservative). The closure holds a unique loan on it.
//! - [`CaptureMode::Ref`]: every other use (reads, `&x`, `&self` receivers). A shared loan.
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
use crate::resolve::Res;
use crate::typeck::{Target, TypedBody};
use crate::types::{Interner, TypeId, TypeKind};

/// Whether `t` is `&mut T` in `types`.
pub fn is_mut_ref(types: &impl Interner, t: TypeId) -> bool {
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

/// The closure bodies inside the body at `root` (a fn decl node), in `@<enclosing>.<k>` order,
/// with their captures. `is_mut_ref(t)`: whether `tb`'s type `t` is `&mut T`
/// ([`is_mut_ref`] over the table `tb`'s types live in).
pub fn closures(
    icx: &ItemCx<'_>,
    tb: &TypedBody,
    is_mut_ref: impl Fn(TypeId) -> bool,
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
    // mutating uses: binding node → first mutating token
    let muts = mutations(icx, tb, lo, root);
    for (i, c) in out.iter_mut().enumerate() {
        let (s, e) = ranges[i];
        let mut caps: Vec<Capture> = Vec::new();
        for m in s..e {
            for u in icx.uses(m) {
                let (Res::Local(b) | Res::SelfValue(b)) = u.res else { continue };
                if (s..=e).contains(&b.node) {
                    continue; // declared inside the body
                }
                match caps.iter_mut().find(|x| x.binding == b.node) {
                    Some(x) => x.first_use = x.first_use.min(u.token),
                    None => caps.push(Capture {
                        binding: b.node,
                        mode: CaptureMode::Ref,
                        first_use: u.token,
                        first_mut: u.token,
                    }),
                }
            }
        }
        for x in &mut caps {
            let by_type = tb
                .local(crate::ids::BindingRef { node: x.binding })
                .is_some_and(|t| !t.is_none() && is_mut_ref(t));
            let inner =
                muts.iter().filter(|&&(b, n)| b == x.binding && (s..e).contains(&n)).map(|&(_, n)| n).min();
            if let Some(n) = inner {
                x.mode = CaptureMode::RefMut;
                x.first_mut = icx.a(n);
            } else if by_type {
                x.mode = CaptureMode::RefMut;
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

/// Mutating uses in `lo..=root`: (binding node, node of the mutating expression).
fn mutations(icx: &ItemCx<'_>, tb: &TypedBody, lo: u32, root: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for m in lo..=root {
        match icx.kind(m) {
            Some(NodeKind::ExprStmt) => {
                let kids = icx.kids_vec(m);
                if kids.len() >= 3
                    && let Some(b) = root_binding(icx, tb, kids[0])
                {
                    out.push((b, m));
                }
            }
            Some(NodeKind::Unary) => {
                let a = icx.a(m);
                if (icx.is(a, Term::Amp) || icx.is(a, Term::AmpAmp))
                    && icx.is(a + 1, Term::KwMut)
                    && let Some(c) = icx.last_child(m)
                    && let Some(b) = root_binding(icx, tb, c)
                {
                    out.push((b, m));
                }
            }
            Some(NodeKind::Postfix) => {
                if let Some(b) = chain_binding(icx, tb, m, true) {
                    out.push((b, m));
                }
            }
            _ => {}
        }
    }
    out
}

/// The binding a place expression is rooted at without a dereference.
fn root_binding(icx: &ItemCx<'_>, tb: &TypedBody, n: u32) -> Option<u32> {
    let c = n;
    match icx.kind(c)? {
        NodeKind::Unary if icx.is(icx.a(c), Term::KwUnsafe) => root_binding(icx, tb, icx.last_child(c)?),
        NodeKind::Postfix => chain_binding(icx, tb, c, false),
        NodeKind::Primary => {
            let a = icx.a(c);
            if icx.is(a, Term::KwSelf) || icx.is(a, Term::Ident) {
                return local_use(icx, c, a);
            }
            if icx.is(a, Term::LParen) {
                let kids = icx.kids_vec(c);
                if kids.len() == 1 && !icx.is(icx.close(a) - 1, Term::Comma) {
                    return root_binding(icx, tb, kids[0]);
                }
            }
            None
        }
        _ => None,
    }
}

fn local_use(icx: &ItemCx<'_>, n: u32, tok: u32) -> Option<u32> {
    let u = icx.use_at(n, tok).or_else(|| icx.uses(n).first())?;
    match u.res {
        Res::Local(b) | Res::SelfValue(b) => Some(b.node),
        _ => None,
    }
}

/// A postfix chain rooted at a local through field / tuple / index steps with no auto-deref:
/// `recv_mut`: the binding when some method call on that place auto-borrows it as `&mut self`;
/// else the binding when the whole chain is such a place.
fn chain_binding(icx: &ItemCx<'_>, tb: &TypedBody, n: u32, recv_mut: bool) -> Option<u32> {
    let kids = icx.kids_vec(n);
    let prim = *kids.first()?;
    let pa = icx.a(prim);
    let base = if icx.is(pa, Term::Ident) || icx.is(pa, Term::KwSelf) {
        local_use(icx, prim, pa)
    } else if icx.is(pa, Term::LParen) {
        root_binding(icx, tb, prim)
    } else {
        None
    };
    let mut root = base;
    let mut p = icx.primary_end(prim) + 1;
    for _ in 0..10_000_000 {
        let t = icx.tk(p);
        if t == Term::Dot as u8 && icx.is(p + 1, Term::Ident) {
            let seg = p + 1;
            let mut q = p + 2;
            if icx.is(q, Term::ColonColon) {
                let (e, partial) = icx.angle_end(q + 1);
                q = if partial { e - 1 } else { e };
            }
            if icx.use_at(prim, seg).is_some() {
                // a path step (module item), not a field
                root = None;
                p = q;
                continue;
            }
            if icx.is(q, Term::LParen) {
                let adjust = match tb.target(seg) {
                    Some(Target::Method { adjust, .. } | Target::Builtin { adjust, .. }) => Some(adjust),
                    _ => None,
                };
                if recv_mut
                    && let (Some(r), Some(a)) = (root, adjust)
                    && a.autoref == Some(true)
                    && a.derefs == 0
                {
                    return Some(r);
                }
                root = None;
                p = icx.close(q) + 1;
                continue;
            }
            match tb.target(seg) {
                Some(Target::Field { derefs, .. }) if *derefs == 0 => {}
                _ => root = None,
            }
            p = q;
            continue;
        }
        if t == Term::Dot as u8 && icx.is(p + 1, Term::IntLit) {
            match tb.target(p + 1) {
                Some(Target::TupleField { derefs, .. }) if *derefs == 0 => {}
                _ => root = None,
            }
            p += 2;
            continue;
        }
        if t == Term::LBracket as u8 {
            match tb.target(p) {
                Some(Target::Index { derefs } | Target::Slice { derefs }) if *derefs == 0 => {}
                _ => root = None,
            }
            p = icx.close(p) + 1;
            continue;
        }
        if t == Term::LParen as u8 || t == Term::DotLBrace as u8 {
            root = None;
            p = icx.close(p) + 1;
            continue;
        }
        if t == Term::Question as u8 {
            root = None;
            p += 1;
            continue;
        }
        break;
    }
    if recv_mut { None } else { root }
}
