//! The kinds of `for` sources (contracts/lang-details.md §4 "`for` sources"): one closed
//! classification of the source's type, read by the type checker (which reports every kind that
//! is not a source), the borrow checker, and the lowering. A type this does not recognise is no
//! source.
//!
//! Whether the source is an array or `[T]` *place* (E0219) is a matter of syntax, checked by the
//! type checker on top of this.
//!
//! Also the rows of the loop report ([`report`], contracts/cli.md `stave loops`): the form of
//! each loop statement, recorded by the type checker, and for a range loop the places it indexes
//! with its variable, read from the typed body.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::Span;
use stave_syntax::{NodeKind, Term};

use crate::decls::{DeclFlags, DeclKind};
use crate::ids::{BindingRef, DeclId};
use crate::itemcx::{Brackets, ItemCx};
use crate::known::builtin_module;
use crate::resolve::{Res, TypeHead, Use};
use crate::traits::{Adjust, MethodRef};
use crate::typeck::{Coercion, Target, Typeck, TypedBody};
use crate::types::{Interner, TypeId, TypeKind};
use crate::{File, Program};

/// What a `for` loop does with a source of a given type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForSource {
    /// `a..b`, `a..=b` of an integer type: the values.
    Range { elem: TypeId },
    /// An array value (a place is E0219): the loop's temporary, its elements by value.
    Array { elem: TypeId },
    /// `[T]`: only a place (a sub-slice, a column), so always E0219.
    SlicePlace,
    /// `&[T]`, `&mut [T]`, `&[T; N]`, `&mut [T; N]`: element `i` of `*s` borrowed per iteration.
    Seq { mutable: bool, elem: TypeId },
    /// An iterator value (a struct, enum, or generic type: `next(&mut self)` must exist): moved
    /// into the loop, which owns it.
    Iter,
    /// `&mut It`, `It` an iterator type: advanced in place through the reference.
    IterMut { inner: TypeId },
    /// `&It`: no source (E0219 `shared-iter`).
    SharedIter,
    /// A reference to a reference (`&mut &mut It`, `& &[T]`, `&mut &[T; N]`, ..): no source
    /// (E0219 `ref-ref`).
    RefRef,
    /// Any other type (a scalar, vector, mask, `str`, tuple, raw pointer, fn value, a range of
    /// a non-integer type such as `char`, `&` of one): no source (E0219 `not-source`).
    NotSource,
    /// The error type (already reported).
    Error,
}

impl ForSource {
    /// Whether a `for` loop may iterate a source of this kind.
    pub fn is_source(self) -> bool {
        self.kind().is_some()
    }

    /// The loop kind of a source, `None` for a kind that is no source.
    pub fn kind(self) -> Option<ForKind> {
        match self {
            ForSource::Range { .. } => Some(ForKind::Range),
            ForSource::Array { .. } => Some(ForKind::Array),
            ForSource::Seq { mutable, .. } => Some(ForKind::Seq { mutable }),
            ForSource::Iter => Some(ForKind::Iter),
            ForSource::IterMut { .. } => Some(ForKind::IterMut),
            _ => None,
        }
    }
}

/// The kind of a checked `for` loop, decided once from the source's type as written in the
/// definition ([`for_source`]) and stored with the body
/// ([`crate::typeck::TypedBody::for_kind`]): the borrow checker, the capture analysis, and the
/// lowering of every instance read it, never an instance type. An iterator kind calls the `next`
/// the type checker resolved (for a generic source: the bound's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForKind {
    Range,
    Array,
    Seq { mutable: bool },
    Iter,
    IterMut,
}

/// The kind of a `for` source of type `t` in `tys`.
pub fn for_source<I: Interner + ?Sized>(tys: &I, t: TypeId) -> ForSource {
    if t.is_none() || tys.is_error_ty(t) {
        return ForSource::Error;
    }
    match tys.kind(t) {
        // a counted loop steps an integer; a range of `char` (or of any other type) is none
        TypeKind::Range { elem, .. } if matches!(tys.kind(*elem), TypeKind::Int { .. }) => {
            ForSource::Range { elem: *elem }
        }
        TypeKind::Range { elem, .. } if elem.is_none() || tys.is_error_ty(*elem) => ForSource::Error,
        TypeKind::Array { elem, .. } => ForSource::Array { elem: *elem },
        TypeKind::Slice { .. } => ForSource::SlicePlace,
        TypeKind::Adt { .. } | TypeKind::Param(_) => ForSource::Iter,
        TypeKind::Ref { mutable, inner } => match tys.kind(*inner) {
            TypeKind::Array { elem, .. } | TypeKind::Slice { elem, .. } => {
                ForSource::Seq { mutable: *mutable, elem: *elem }
            }
            TypeKind::Ref { .. } => ForSource::RefRef,
            TypeKind::Adt { .. } | TypeKind::Param(_) if *mutable => ForSource::IterMut { inner: *inner },
            TypeKind::Adt { .. } | TypeKind::Param(_) => ForSource::SharedIter,
            TypeKind::Error => ForSource::Error,
            _ => ForSource::NotSource,
        },
        TypeKind::Error => ForSource::Error,
        _ => ForSource::NotSource,
    }
}

/// The form of a loop statement, as written in its definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopForm {
    /// A chunk loop ([`TypedBody::chunk_loops`]).
    Chunk,
    /// A `for` over a range.
    Range,
    /// A `for` over the elements of an array or `[T]`.
    Element,
    /// A `for` over an iterator.
    Iter,
    While,
    Loop,
}

impl LoopForm {
    /// The form of a `for` loop: `chunk` as [`TypedBody::chunk_loops`] decides it, and the kind
    /// of its source.
    pub fn of_for(chunk: bool, kind: ForKind) -> LoopForm {
        if chunk {
            return LoopForm::Chunk;
        }
        match kind {
            ForKind::Range => LoopForm::Range,
            ForKind::Array | ForKind::Seq { .. } => LoopForm::Element,
            ForKind::Iter | ForKind::IterMut => LoopForm::Iter,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LoopForm::Chunk => "chunk",
            LoopForm::Range => "range",
            LoopForm::Element => "element",
            LoopForm::Iter => "iter",
            LoopForm::While => "while",
            LoopForm::Loop => "loop",
        }
    }
}

/// One loop statement of a checked body: its node (`for`: the `ForStmt`; `while` and `loop`: the
/// statement) and its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopRow {
    pub node: u32,
    pub form: LoopForm,
}

/// A slice, array, or column place that does not change in a range loop and that its body uses
/// only as element `[i]`, `i` the loop variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopSource {
    /// The tokens of the place, joined (`xs`, `self.cols`, `grid[row]`, `v.as_slice()`): as in
    /// its first use, or its first writing use if it has one.
    pub place: String,
    /// Some use writes the element or borrows it `&mut`.
    pub written: bool,
}

/// The element accesses of a range loop at its variable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexedAccess {
    /// Some element of a slice or array is read or written at exactly the loop variable.
    pub indexed: bool,
    /// The read places, then the written ones, each group in order of first element use.
    pub sources: Vec<LoopSource>,
}

/// One loop of the report.
#[derive(Clone, Debug)]
pub struct LoopInfo {
    /// The loop's keyword.
    pub span: Span,
    /// The checked-IR path of the enclosing fn (a closure's loops: of the fn that holds it).
    pub func: String,
    pub form: LoopForm,
    /// Empty unless the form is [`LoopForm::Range`].
    pub access: IndexedAccess,
}

/// The loops of the checked bodies in the files `include` accepts, one per loop statement (a
/// generic fn's loops once, whatever its instances), in canonical (file, offset) order.
pub fn report(prog: &Program, tc: &Typeck, include: impl Fn(&File) -> bool) -> Vec<LoopInfo> {
    let views = slice_views(prog);
    // per item: an impl holds many bodies
    let mut brackets: FxHashMap<u32, Brackets> = FxHashMap::default();
    let mut out = Vec::new();
    for body in &tc.bodies {
        if body.loop_rows.is_empty() {
            continue;
        }
        let Some(item) = prog.item(body.item) else { continue };
        let Some(file) = prog.file(item.file).filter(|f| include(f)) else { continue };
        let br = brackets.entry(body.item.0).or_insert_with(|| {
            file.parsed
                .items
                .get(item.index as usize)
                .map_or_else(Brackets::default, |x| Brackets::new(&file.parsed.tokens, x.start, x.end))
        });
        let Some(icx) = ItemCx::new(prog, body.item, br) else { continue };
        let func = crate::lower::fn_path(prog, &tc.sigs, body.decl);
        for row in &body.loop_rows {
            let access = match row.form {
                LoopForm::Range => indexed_access(&icx, body, row.node, &views),
                _ => IndexedAccess::default(),
            };
            let span = icx.tok_span(icx.a(row.node));
            out.push(LoopInfo { span, func: func.clone(), form: row.form, access });
        }
    }
    out.sort_by_key(|l| (l.span.file, l.span.offset));
    out
}

/// `vec.Vec`'s `as_slice` and `as_mut_slice`, found by identity: the calls a place may go
/// through, since each gives the vector's own elements.
fn slice_views(prog: &Program) -> Vec<DeclId> {
    let sym = |s: &str| prog.strings.lookup(s);
    let Some(vec) = builtin_module(&prog.modules, &prog.packages, &prog.strings, "std.vec")
        .and_then(|m| prog.module_item(m, sym("Vec")?))
    else {
        return Vec::new();
    };
    ["as_slice", "as_mut_slice"]
        .into_iter()
        .filter_map(|n| prog.methods.inherent.get(&(TypeHead::Adt(vec), sym(n)?))?.first().copied())
        .collect()
}

/// Where a place starts: a binding, or a const or `static`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Root {
    Local(BindingRef),
    Global(DeclId),
}

fn root_of(prog: &Program, res: Res) -> Option<Root> {
    match res {
        Res::Local(b) | Res::SelfValue(b) => Some(Root::Local(b)),
        Res::Decl(d) => prog
            .decl(d)
            .is_some_and(|x| matches!(x.kind, DeclKind::Const | DeclKind::Static))
            .then_some(Root::Global(d)),
        _ => None,
    }
}

/// A const, or a `static` that is not `mut`.
fn is_fixed_global(prog: &Program, d: DeclId) -> bool {
    prog.decl(d).is_some_and(|x| match x.kind {
        DeclKind::Const => true,
        DeclKind::Static => !x.flags.has(DeclFlags::MUT),
        _ => false,
    })
}

/// One step of a postfix chain, as written.
#[derive(Clone, Copy)]
enum RawStep {
    /// `.name` or `.0`: the token after the dot.
    Field(u32),
    /// `.name(..)`: the name token.
    Method(u32),
    /// `[..]`: the `[` token.
    Index(u32),
    /// A call, struct literal, `?`, or turbofish step.
    Other,
}

/// The steps of a chain from token `p` through token `end`.
fn chain_steps(icx: &ItemCx<'_>, mut p: u32, end: u32) -> Vec<RawStep> {
    let mut out = Vec::new();
    while p <= end {
        if icx.is(p, Term::Dot) && icx.is(p + 1, Term::Ident) {
            let mut q = p + 2;
            let turbofish = icx.is(q, Term::ColonColon);
            if turbofish {
                let (e, partial) = icx.angle_end(q + 1);
                q = if partial { e - 1 } else { e };
            }
            if icx.is(q, Term::LParen) {
                out.push(RawStep::Method(p + 1));
                p = icx.close(q) + 1;
            } else {
                out.push(if turbofish { RawStep::Other } else { RawStep::Field(p + 1) });
                p = q;
            }
        } else if icx.is(p, Term::Dot) && icx.is(p + 1, Term::IntLit) {
            out.push(RawStep::Field(p + 1));
            p += 2;
        } else if icx.is(p, Term::LBracket) {
            out.push(RawStep::Index(p));
            p = icx.close(p) + 1;
        } else if icx.is(p, Term::LParen) || icx.is(p, Term::DotLBrace) {
            out.push(RawStep::Other);
            p = icx.close(p) + 1;
        } else {
            out.push(RawStep::Other);
            p += 1;
        }
    }
    out
}

/// A step of a place, as far as two places can be told apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Step<'p> {
    /// Different fields are disjoint places.
    Field(&'p str),
    /// An index with one value through the loop: a literal, an immutable local bound outside
    /// the body, a const or a `static` that is not `mut`.
    AtLit(&'p str),
    AtLocal(BindingRef),
    AtGlobal(DeclId),
    /// The elements of a vector, through either of its [`slice_views`].
    View,
    /// Element `[i]`, `i` the loop variable.
    Var,
    /// Anything else: it may reach any part of the place before it.
    Other,
}

/// A place a body names: a root, or a step below another place.
#[derive(Default)]
struct Place<'p> {
    /// A chain ends here: the place is used whole.
    whole: bool,
    children: FxHashMap<Step<'p>, usize>,
    /// How many of `children` are not a [`Step::Field`].
    non_fields: usize,
    /// For a place with a [`Step::Var`] child: the first `[` token of that step and the first
    /// token of the chain there.
    element: Option<(u32, u32)>,
    /// The same of the first use of that step that writes.
    written: Option<(u32, u32)>,
}

fn child<'p>(places: &mut Vec<Place<'p>>, at: usize, step: Step<'p>) -> usize {
    if let Some(&c) = places[at].children.get(&step) {
        return c;
    }
    let c = places.len();
    places.push(Place::default());
    places[at].children.insert(step, c);
    places[at].non_fields += usize::from(!matches!(step, Step::Field(_)));
    c
}

/// Adds to `written` the chains that an assignment to `n`, or `&mut n`, writes through: `n`
/// itself, inside parentheses, `*`, and `unsafe(..)`, and under the fields and elements of a
/// parenthesized place.
fn mark_written(icx: &ItemCx<'_>, mut n: u32, written: &mut FxHashSet<u32>) {
    loop {
        written.insert(n);
        let a = icx.a(n);
        let inner = match icx.kind(n) {
            Some(NodeKind::Primary) if icx.is(a, Term::LParen) => icx.only_child(n),
            Some(NodeKind::Unary) if icx.is(a, Term::Star) || icx.is(a, Term::KwUnsafe) => icx.last_child(n),
            Some(NodeKind::Postfix) => paren_head(icx, n).and_then(|(head, steps)| {
                steps.iter().all(|s| matches!(s, RawStep::Field(_) | RawStep::Index(_))).then_some(head)
            }),
            _ => None,
        };
        match inner {
            Some(c) => n = c,
            None => return,
        }
    }
}

/// The head of postfix chain `n` and the steps after it, if the head is parenthesized.
fn paren_head(icx: &ItemCx<'_>, n: u32) -> Option<(u32, Vec<RawStep>)> {
    let head = icx.first_child(n).filter(|&h| icx.is(icx.a(h), Term::LParen))?;
    Some((head, chain_steps(icx, icx.primary_end(head) + 1, icx.postfix_end(n))))
}

/// The element accesses of range loop `f` ([`IndexedAccess`]).
///
/// A place `s` is a source when every chain of the body that names it, a part of it, or a place
/// holding it either goes on with `[i]` after `s` or leaves the path to `s` at another field.
/// Whatever this cannot read (an f-string hole, a call on the path other than one of `views`,
/// an index that may change) uses the place whole or reaches any part of it.
fn indexed_access<'p>(icx: &ItemCx<'p>, body: &TypedBody, f: u32, views: &[DeclId]) -> IndexedAccess {
    let mut out = IndexedAccess::default();
    let kids = icx.kids_vec(f);
    let Some(var) = kids.first().and_then(|&pat| icx.res.binding(BindingRef { node: pat })) else {
        return out;
    };
    let Some(block) = icx.child(f, NodeKind::Block) else { return out };
    let nodes = icx.ast.descendants(block).start..block + 1;
    let inside = |b: BindingRef| nodes.contains(&b.node);
    let uses: &[Use] = &icx.res.uses[icx.res.uses.partition_point(|u| u.node < nodes.start)..];
    let uses = &uses[..uses.partition_point(|u| u.node < nodes.end)];
    let by_token: FxHashMap<u32, Res> = uses.iter().filter(|u| !u.in_fmt).map(|u| (u.token, u.res)).collect();

    let mut heads: FxHashSet<u32> = FxHashSet::default();
    let mut written: FxHashSet<u32> = FxHashSet::default();
    for n in nodes.clone() {
        let a = icx.a(n);
        match icx.kind(n) {
            Some(NodeKind::Postfix) => {
                heads.extend(icx.first_child(n));
                if let Some((head, steps)) = paren_head(icx, n)
                    && calls_mut_method(body, &steps)
                {
                    mark_written(icx, head, &mut written);
                }
            }
            Some(NodeKind::ExprStmt) => {
                let k = icx.kids_vec(n);
                if k.len() >= 3 {
                    mark_written(icx, k[0], &mut written);
                }
            }
            Some(NodeKind::Unary)
                if (icx.is(a, Term::Amp) || icx.is(a, Term::AmpAmp)) && icx.is(a + 1, Term::KwMut) =>
            {
                if let Some(c) = icx.last_child(n) {
                    mark_written(icx, c, &mut written);
                }
            }
            _ => {}
        }
    }

    let mut places: Vec<Place<'p>> = Vec::new();
    let mut roots: FxHashMap<Root, usize> = FxHashMap::default();
    let mut root_at = |places: &mut Vec<Place<'p>>, r: Root| {
        *roots.entry(r).or_insert_with(|| {
            places.push(Place::default());
            places.len() - 1
        })
    };
    // A name that heads no chain uses its place whole.
    for u in uses {
        if let Some(r) = root_of(icx.prog, u.res)
            && (u.in_fmt || icx.kind(u.node) != Some(NodeKind::Primary) || !heads.contains(&u.node))
        {
            let at = root_at(&mut places, r);
            places[at].whole = true;
        }
    }
    let step_at = |open: u32| -> Step<'p> {
        if !matches!(body.target(open), Some(Target::Index { .. })) {
            return Step::Other;
        }
        let (mut lo, mut hi) = (open + 1, icx.close(open).saturating_sub(1));
        while lo < hi && icx.is(lo, Term::LParen) && icx.close(lo) == hi {
            (lo, hi) = (lo + 1, hi - 1);
        }
        if lo != hi {
            return Step::Other;
        }
        if icx.is(lo, Term::IntLit) {
            return Step::AtLit(icx.tok_str(lo));
        }
        match by_token.get(&lo) {
            Some(&Res::Local(b)) if b == var.at => Step::Var,
            Some(&Res::Local(b)) if !inside(b) && icx.res.binding(b).is_some_and(|x| !x.mutable) => {
                Step::AtLocal(b)
            }
            Some(&Res::Decl(d)) if is_fixed_global(icx.prog, d) => Step::AtGlobal(d),
            _ => Step::Other,
        }
    };
    for n in nodes.clone() {
        if icx.kind(n) != Some(NodeKind::Postfix) {
            continue;
        }
        let Some(prim) = icx.first_child(n) else { continue };
        let head = icx
            .uses(prim)
            .iter()
            .rev()
            .filter(|u| !u.in_fmt)
            .find_map(|u| Some((u.token, root_of(icx.prog, u.res)?)));
        let start = head.map_or_else(|| icx.primary_end(prim), |(t, _)| t) + 1;
        let steps = chain_steps(icx, start, icx.postfix_end(n));
        let classes: Vec<Step<'p>> = steps
            .iter()
            .map(|&s| match s {
                RawStep::Field(t) => Step::Field(icx.tok_str(t)),
                RawStep::Index(open) => step_at(open),
                RawStep::Method(name)
                    if matches!(
                        body.target(name),
                        Some(Target::Method { method: MethodRef::Impl { method, .. }, .. })
                            if views.contains(method)
                    ) =>
                {
                    Step::View
                }
                RawStep::Method(_) | RawStep::Other => Step::Other,
            })
            .collect();
        out.indexed |= classes.contains(&Step::Var);
        let Some((_, root)) = head else { continue };
        let mut at = root_at(&mut places, root);
        let mut ends_here = true;
        for (k, (&step, &class)) in steps.iter().zip(&classes).enumerate() {
            let next = child(&mut places, at, class);
            if let (Step::Var, RawStep::Index(open)) = (class, step) {
                let writes = written.contains(&n)
                    || body.coercion(n) == Some(&Coercion::Reborrow)
                    || calls_mut_method(body, &steps[k + 1..]);
                let here = (open, icx.a(prim));
                let place = &mut places[at];
                place.element = Some(place.element.map_or(here, |first| first.min(here)));
                if writes {
                    place.written = Some(place.written.map_or(here, |first| first.min(here)));
                }
            }
            at = next;
            if matches!(class, Step::Var | Step::Other) {
                ends_here = false;
                break;
            }
        }
        places[at].whole |= ends_here;
    }
    // a `mut` loop variable need not be the index of the iteration
    if var.mutable {
        return out;
    }

    // Down from each root the body cannot replace, while no place on the way is used whole or by
    // a step that may reach further down.
    let mut stack: Vec<usize> = roots
        .iter()
        .filter(|&(&r, _)| match r {
            Root::Local(b) => b != var.at && !inside(b),
            Root::Global(d) => is_fixed_global(icx.prog, d),
        })
        .map(|(_, &at)| at)
        .collect();
    // (written, first element use, the `[` and the first token of the use the text is taken from)
    let mut found: Vec<(bool, u32, (u32, u32))> = Vec::new();
    while let Some(at) = stack.pop() {
        let place = &places[at];
        if place.whole {
            continue;
        }
        let only = place.children.len() == 1;
        for (&step, &next) in &place.children {
            match step {
                Step::Var => {
                    if only && let Some(first) = place.element {
                        let shown = place.written.unwrap_or(first);
                        found.push((place.written.is_some(), first.0, shown));
                    }
                }
                Step::Field(_) if only || place.non_fields == 0 => stack.push(next),
                Step::AtLit(_) | Step::AtLocal(_) | Step::AtGlobal(_) | Step::View if only => {
                    stack.push(next);
                }
                _ => {}
            }
        }
    }
    found.sort_unstable_by_key(|x| (x.0, x.1));
    out.sources = found
        .into_iter()
        .map(|(written, _, (open, from))| LoopSource {
            place: (from..open).map(|t| icx.tok_str(t)).collect(),
            written,
        })
        .collect();
    out
}

/// Whether the steps after an element lead, through fields and elements, to a method that borrows
/// its receiver `&mut`.
fn calls_mut_method(body: &TypedBody, after: &[RawStep]) -> bool {
    for &s in after {
        match s {
            RawStep::Field(_) | RawStep::Index(_) => {}
            RawStep::Method(name) => {
                return matches!(
                    body.target(name),
                    Some(
                        Target::Method { adjust: Adjust { autoref: Some(true), .. }, .. }
                            | Target::Builtin { adjust: Adjust { autoref: Some(true), .. }, .. }
                    )
                );
            }
            RawStep::Other => return false,
        }
    }
    false
}
