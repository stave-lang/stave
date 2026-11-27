//! Type nesting depth (E0005 case `type`).
//!
//! **Depth.** A type without components (a scalar, `str`, a mask, a generic parameter) is 1
//! level deep; a reference, raw pointer, array, slice, tuple, fn type, vector, or range is one
//! level above its deepest component, and a struct or enum one level above its deepest type
//! argument and its deepest field (after substitution). A field or element reached through a
//! reference, pointer, slice, or fn type adds no levels to its holder: the pointee is a type of
//! its own and is measured by itself.
//!
//! **What is measured.** Every type the program writes or builds: fn signatures, fields, alias
//! targets, impl self types, every value type of every checked body (generic bodies with their
//! parameters as leaves), the type args of every instance and the value types of its body after
//! substitution, the types of drop glue, and, closed over components and fields, every type any
//! of these reaches. A type deeper than [`MAX_TYPE_DEPTH`] whose parts and components are not is
//! reported once, at the first place (in that order) that reaches it; the types holding it are
//! not reported again.
//!
//! So no type that lowering, layout, or the IR sees nests deeper than the limit, and every
//! recursion over type structure in the checker (trait queries, `size_of`, drop analysis,
//! substitution) bounds itself by the same limit and answers the rejecting side past it.
//!
//! **Types that hold themselves by value** (E0423). The parts held by value (struct fields, enum
//! payloads, tuple, array, range, and vector elements, after substitution) form a graph; the walk
//! that measures depth finds its strongly connected components (Tarjan's algorithm on the same
//! explicit stack). A type in a component with a cycle holds itself by value: it has no depth and
//! no layout. Every struct and enum declaration is a root with its params as args, used or not.
//! Every component with a cycle that any walk closes is reported ([`holds_itself`]): at the
//! first declaration in source order that writes, in a field, a type of the component (`D<u32>`
//! in `struct D<T> { n: Option<D<u32>> }`; not `Option`, whose field is its bare param), else at
//! the first declaration with an instance in it. Components are taken in that order of reports;
//! each is first cut at the fields of every declaration reported before that lie on its reported
//! cycle (a fix there fixes every instance), and is reported on the cycles it has left, if any:
//! the same cycle at other args is reported once, a separate cycle through an instance of a
//! reported declaration is reported too. A component with no struct or enum in it (none
//! exists) is E0005 where the walk met it. So a type that only
//! reaches a cycle (a struct with a field of it) is not reported itself, nor is any type on or
//! above a cycle too deep (E0005): it has no depth, and its cycle has its report.

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, Span};

use crate::Program;
use crate::callgraph::{CallGraph, NodeKind};
use crate::diag::mk;
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::sigs::{DeclSig, Sigs};
use crate::typeck::TypedBody;
use crate::types::{ConstArg, GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};

/// The deepest type nesting a program may use (module docs), shared by every recursion over
/// type structure in the checker, lowering, layout, and the IR.
pub const MAX_TYPE_DEPTH: u32 = stave_ir::types::MAX_TYPE_DEPTH as u32;

struct Walk<'a, 'g> {
    cx: Overlay<'g>,
    sigs: &'a Sigs,
    /// Height of the type term (components only, no fields), and whether a generic parameter
    /// occurs in it.
    term: FxHashMap<TypeId, (u32, bool)>,
    /// Depth (module docs).
    depth: FxHashMap<TypeId, u32>,
    /// Types whose depth is being computed (a type holding itself through a pointer).
    active: FxHashSet<TypeId>,
    /// Types measured, their components queued.
    seen: FxHashSet<TypeId>,
    queue: Vec<TypeId>,
    /// Types too deep whose parts and components are not.
    causes: FxHashSet<TypeId>,
    /// [`Walk::parts`] left out a field whose type after substitution would be taller than the
    /// type table holds ([`crate::types::MAX_TYPE_HEIGHT`]): the holder is too deep.
    tall_part: bool,
    /// Types whose depth met a type in progress, or such a type: each reaches, through parts, a
    /// type that holds itself by value.
    cyclic: FxHashSet<TypeId>,
    /// The depth being computed met a type in progress or a cyclic type.
    met_cycle: bool,
    /// Tarjan's algorithm over the parts graph: the preorder number of each type measured by a
    /// [`Measure`] frame, the types whose component is still open (in preorder), the next number.
    index: FxHashMap<TypeId, u32>,
    open: Vec<TypeId>,
    is_open: FxHashSet<TypeId>,
    next_index: u32,
    /// The components with a cycle, as closed: the members and the place whose walk closed it
    /// ([`Walk::site`]).
    components: Vec<(Vec<TypeId>, Option<Span>)>,
    /// The place being measured (a declaration, a body node, a call site).
    site: Option<Span>,
}

/// One type of [`Walk::depth`] whose parts are being measured.
struct Measure {
    t: TypeId,
    /// Depth so far: the term height, the parts measured, past the limit for a tall part.
    d: u32,
    /// A part measured so far is too deep.
    part_too_deep: bool,
    /// [`Walk::met_cycle`] when the measure began (restored and joined at its end).
    outer_met: bool,
    parts: Vec<TypeId>,
    /// The next part to measure.
    next: usize,
    /// Preorder number of `t` and the smallest one reachable from it through open types.
    index: u32,
    low: u32,
    /// A part of `t` is `t`.
    self_part: bool,
}

impl Measure {
    fn add_part(&mut self, pd: u32) {
        self.part_too_deep |= pd > MAX_TYPE_DEPTH;
        self.d = self.d.max(pd.saturating_add(1));
    }
}

impl<'a, 'g> Walk<'a, 'g> {
    fn new(cx: Overlay<'g>, sigs: &'a Sigs) -> Self {
        Walk {
            cx,
            sigs,
            term: FxHashMap::default(),
            depth: FxHashMap::default(),
            active: FxHashSet::default(),
            seen: FxHashSet::default(),
            queue: Vec::new(),
            causes: FxHashSet::default(),
            tall_part: false,
            cyclic: FxHashSet::default(),
            met_cycle: false,
            index: FxHashMap::default(),
            open: Vec::new(),
            is_open: FxHashSet::default(),
            next_index: 0,
            components: Vec::new(),
            site: None,
        }
    }

    /// Term height and parameter occurrence, by an explicit post-order walk (term graphs are
    /// acyclic: components are interned before the types holding them).
    fn term(&mut self, t: TypeId) -> (u32, bool) {
        if let Some(&r) = self.term.get(&t) {
            return r;
        }
        let mut stack = vec![(t, false)];
        let mut kids = Vec::new();
        while let Some((u, expanded)) = stack.pop() {
            if self.term.contains_key(&u) {
                continue;
            }
            kids.clear();
            let k = self.cx.kind(u);
            k.children(&mut kids);
            let param = matches!(k, TypeKind::Param(_) | TypeKind::ConstParam(_) | TypeKind::SelfTy)
                || matches!(k, TypeKind::Adt { args, .. }
                    if args.iter().any(|a| matches!(a, GArg::Const(ConstArg::Param(_)))))
                || matches!(k, TypeKind::Array { len: ConstArg::Param(_), .. });
            if !expanded {
                stack.push((u, true));
                stack.extend(kids.iter().filter(|c| !self.term.contains_key(c)).map(|&c| (c, false)));
                continue;
            }
            let (mut h, mut p) = (0u32, param);
            for c in &kids {
                let (ch, cp) = self.term.get(c).copied().unwrap_or((u32::MAX, false));
                h = h.max(ch);
                p |= cp;
            }
            self.term.insert(u, (h.saturating_add(1), p));
        }
        self.term.get(&t).copied().unwrap_or((u32::MAX, false))
    }

    /// The parts of `t` held by value: array, tuple, range, and vector elements, ADT fields.
    fn parts(&mut self, t: TypeId) -> Vec<TypeId> {
        match self.cx.kind(t).clone() {
            TypeKind::Array { elem, .. } | TypeKind::Range { elem, .. } => vec![elem],
            TypeKind::Vector { lane, .. } => vec![lane],
            TypeKind::Tuple(ts) => ts.to_vec(),
            // a pointee, slice element, or fn signature is not held by value; the rest have no
            // components (listed one by one: a new kind must decide what it holds)
            TypeKind::Ref { .. }
            | TypeKind::Ptr { .. }
            | TypeKind::Slice { .. }
            | TypeKind::Fn { .. }
            | TypeKind::Int { .. }
            | TypeKind::Float(_)
            | TypeKind::Bool
            | TypeKind::Char
            | TypeKind::Str
            | TypeKind::Mask { .. }
            | TypeKind::Param(_)
            | TypeKind::ConstParam(_)
            | TypeKind::Never
            | TypeKind::SelfTy
            | TypeKind::Error => Vec::new(),
            TypeKind::Adt { def, args } => {
                let Some(adt) = self.sigs.adt(def) else { return Vec::new() };
                let fields: Vec<TypeId> =
                    adt.variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect();
                let mut out = Vec::with_capacity(fields.len());
                for f in fields {
                    if crate::types::subst_height(&self.cx, f, &args) > MAX_TYPE_DEPTH {
                        self.tall_part = true;
                    } else {
                        out.push(self.cx.subst(f, &args, None));
                    }
                }
                out
            }
        }
    }

    /// Depth of `t` (module docs). Components that are not parts are queued to be measured by
    /// themselves. A chain of parts ends: a type holding itself by value is cut (it has no
    /// layout, diagnosed elsewhere), and an endless family of distinct parts grows its type
    /// arguments, so its term height passes the limit. The chain is walked with an explicit
    /// stack of [`Measure`] frames: a chain of parts as long as a program can write takes no
    /// stack in proportion to it.
    fn depth(&mut self, t: TypeId) -> u32 {
        let mut frames = match self.enter(t) {
            Ok(d) => return d,
            Err(f) => vec![f],
        };
        loop {
            let Some(top) = frames.last_mut() else { return 0 };
            if let Some(&p) = top.parts.get(top.next) {
                top.next += 1;
                if self.is_open.contains(&p)
                    && let Some(&pi) = self.index.get(&p)
                {
                    top.low = top.low.min(pi);
                    top.self_part |= p == top.t;
                }
                match self.enter(p) {
                    Ok(pd) => top.add_part(pd),
                    Err(f) => frames.push(f),
                }
                continue;
            }
            let Some(f) = frames.pop() else { return 0 };
            let low = f.low;
            let d = self.leave(f);
            match frames.last_mut() {
                Some(top) => {
                    top.low = top.low.min(low);
                    top.add_part(d);
                }
                None => return d,
            }
        }
    }

    /// Starts measuring `t`: its depth if known without its parts, else the frame that
    /// measures them.
    fn enter(&mut self, t: TypeId) -> Result<u32, Measure> {
        if let Some(&d) = self.depth.get(&t) {
            self.met_cycle |= self.cyclic.contains(&t);
            return Ok(d);
        }
        let (k, _) = self.term(t);
        if k > MAX_TYPE_DEPTH {
            self.depth.insert(t, k);
            let c = self.term_cause(t);
            self.causes.insert(c);
            return Ok(k);
        }
        if !self.active.insert(t) {
            self.met_cycle = true;
            return Ok(k);
        }
        let outer_met = std::mem::take(&mut self.met_cycle);
        let mut d = k;
        self.tall_part = false;
        let parts = self.parts(t);
        if std::mem::take(&mut self.tall_part) {
            d = MAX_TYPE_DEPTH + 1;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.index.insert(t, index);
        self.open.push(t);
        self.is_open.insert(t);
        Err(Measure {
            t,
            d,
            part_too_deep: false,
            outer_met,
            parts,
            next: 0,
            index,
            low: index,
            self_part: false,
        })
    }

    /// Ends the measure of `f.t` once every part is measured: records its depth, and whether it
    /// is too deep by itself or reaches a type holding itself by value.
    fn leave(&mut self, f: Measure) -> u32 {
        let Measure { t, d, part_too_deep, outer_met, index, low, self_part, .. } = f;
        self.active.remove(&t);
        if low == index {
            // `t` roots a component: the open types from `t` on
            let at = self.open.iter().rposition(|&u| u == t).unwrap_or(self.open.len());
            let members = self.open.split_off(at);
            let cycle = self_part || members.len() > 1;
            for &u in &members {
                self.is_open.remove(&u);
            }
            if cycle {
                self.components.push((members, self.site));
            }
        }
        self.depth.insert(t, d);
        if self.met_cycle {
            self.cyclic.insert(t);
        }
        let cyclic = self.met_cycle;
        self.met_cycle |= outer_met;
        // a type that reaches a by-value cycle has no depth, whatever its chain of parts
        // measured on the way: the walk closed that cycle's component, and every closed
        // component is reported ([`holds_itself`])
        if d > MAX_TYPE_DEPTH && !cyclic {
            if !part_too_deep {
                self.causes.insert(t);
            }
        } else {
            let mut kids = Vec::new();
            self.cx.kind(t).children(&mut kids);
            self.queue.extend(kids.into_iter().filter(|c| !self.seen.contains(c)));
        }
        d
    }

    /// The component of `t` (or `t`) whose term height is just past the limit, along the first
    /// component that is past it.
    fn term_cause(&mut self, mut t: TypeId) -> TypeId {
        let mut kids = Vec::new();
        loop {
            kids.clear();
            self.cx.kind(t).children(&mut kids);
            match kids.iter().copied().find(|&c| self.term(c).0 > MAX_TYPE_DEPTH) {
                Some(c) => t = c,
                None => return t,
            }
        }
    }

    /// Measures `t` and everything it reaches, from the place `at`. True if it reaches a type
    /// too deep whose parts and components are not, and no earlier check found that type.
    fn check(&mut self, t: TypeId, at: Span) -> bool {
        self.site = Some(at);
        let before = self.causes.len();
        self.queue.push(t);
        while let Some(u) = self.queue.pop() {
            if !self.seen.insert(u) {
                continue;
            }
            self.depth(u);
        }
        self.causes.len() > before
    }

    fn has_params(&mut self, t: TypeId) -> bool {
        self.term(t).1
    }
}

/// The depth of each of `ts` (module docs), `None` for a type that reaches, through parts, a
/// type holding itself by value (it has no depth and no layout). Whatever the order of `ts`,
/// the walk of such a type meets either a type in progress or one already found cyclic, and the
/// walk of any other type meets neither, so the answer is per type.
pub(crate) fn depths(sigs: &Sigs, table: &TypeTable, ts: &[TypeId]) -> Vec<Option<u32>> {
    let mut w = Walk::new(Overlay::new(table), sigs);
    ts.iter()
        .map(|&t| {
            let d = w.depth(t);
            w.met_cycle = false;
            (!w.cyclic.contains(&t)).then_some(d)
        })
        .collect()
}

/// The written types of a field held by value that name types directly: the field type, then
/// through tuple, array, and range elements and ADT type arguments (not through a pointer,
/// reference, slice, or fn type).
fn written_parts(w: &mut Walk<'_, '_>, ft: TypeId) -> Vec<TypeId> {
    let mut out = Vec::new();
    let mut stack = vec![ft];
    while let Some(x) = stack.pop() {
        if out.contains(&x) {
            continue;
        }
        out.push(x);
        match w.cx.kind(x).clone() {
            TypeKind::Array { elem, .. } | TypeKind::Range { elem, .. } => stack.push(elem),
            TypeKind::Tuple(ts) => stack.extend(ts.iter().copied()),
            TypeKind::Adt { args, .. } => stack.extend(args.iter().filter_map(|g| match g {
                GArg::Type(t) => Some(*t),
                GArg::Const(_) => None,
            })),
            _ => {}
        }
    }
    out
}

/// The fields of `m`, if it is an ADT instance: its declaration and, per field (variants in
/// order), its index, span, written type, and type after substitution (`None` for one taller
/// than the type table holds: [`Walk::parts`] leaves it out too).
type InstFields = (DeclId, Vec<(usize, Span, TypeId, Option<TypeId>)>);

fn inst_fields(sigs: &Sigs, w: &mut Walk<'_, '_>, m: TypeId) -> Option<InstFields> {
    let TypeKind::Adt { def, args } = w.cx.kind(m).clone() else { return None };
    let adt = sigs.adt(def)?;
    let fields: Vec<(Span, TypeId)> =
        adt.variants.iter().flat_map(|v| v.fields.iter().map(|f| (f.span, f.ty))).collect();
    let out = fields
        .into_iter()
        .enumerate()
        .map(|(i, (span, ft))| {
            let st = (crate::types::subst_height(&w.cx, ft, &args) <= MAX_TYPE_DEPTH)
                .then(|| w.cx.subst(ft, &args, None));
            (i, span, ft, st)
        })
        .collect();
    Some((def, out))
}

/// The cycles of a graph on `0..adj.len()`: per node, the number of its strongly connected
/// component if that has a cycle (two nodes or more, or an edge to itself). Tarjan's algorithm
/// on an explicit stack.
fn cyclic_sccs(adj: &[Vec<u32>]) -> Vec<Option<u32>> {
    let n = adj.len();
    let (mut index, mut low, mut on) = (vec![u32::MAX; n], vec![0u32; n], vec![false; n]);
    let (mut open, mut out) = (Vec::new(), vec![None; n]);
    let (mut next, mut comps) = (0u32, 0u32);
    for s in 0..n {
        if index[s] != u32::MAX {
            continue;
        }
        index[s] = next;
        low[s] = next;
        next += 1;
        open.push(s);
        on[s] = true;
        let mut frames: Vec<(usize, usize)> = vec![(s, 0)];
        while let Some(top) = frames.last_mut() {
            let v = top.0;
            if let Some(&u) = adj[v].get(top.1) {
                top.1 += 1;
                let u = u as usize;
                if index[u] == u32::MAX {
                    index[u] = next;
                    low[u] = next;
                    next += 1;
                    open.push(u);
                    on[u] = true;
                    frames.push((u, 0));
                } else if on[u] {
                    low[v] = low[v].min(index[u]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(p, _)) = frames.last() {
                low[p] = low[p].min(low[v]);
            }
            if low[v] == index[v] {
                let at = open.iter().rposition(|&x| x == v).unwrap_or(open.len());
                let members = open.split_off(at);
                let cycle = members.len() > 1 || adj[v].contains(&(v as u32));
                for &x in &members {
                    on[x] = false;
                    if cycle {
                        out[x] = Some(comps);
                    }
                }
                comps += cycle as u32;
            }
        }
    }
    out
}

/// The declaration a cycle is reported at: whether a field of it writes a type of the cycle
/// (it is the one to change), its name, the declaration, the field (index and span), and the
/// cycle's number.
type Pick = (bool, Span, DeclId, Option<(usize, Span)>, u32);

/// The declaration to report among `members` (module docs). `sub` numbers the cycles left once
/// the fields in `cut` hold nothing; a field counts if its type is on its holder's cycle.
fn pick(
    prog: &Program,
    sigs: &Sigs,
    w: &mut Walk<'_, '_>,
    members: &[TypeId],
    sub: &FxHashMap<TypeId, u32>,
    cut: &FxHashSet<(DeclId, usize)>,
) -> Option<Pick> {
    let mut best: Option<Pick> = None;
    for &m in members {
        let Some(&s) = sub.get(&m) else { continue };
        let Some((def, fields)) = inst_fields(sigs, w, m) else { continue };
        let Some(decl) = prog.decl(def) else { continue };
        let (mut writes, mut field) = (false, None);
        for (i, span, ft, st) in fields {
            if cut.contains(&(def, i)) || st.and_then(|st| sub.get(&st)) != Some(&s) {
                continue;
            }
            let named = written_parts(w, ft).iter().any(|x| sub.get(x) == Some(&s));
            if field.is_none() || (named && !writes) {
                field = Some((i, span));
                writes |= named;
            }
            if writes {
                break;
            }
        }
        let better = match &best {
            None => true,
            Some((bw, bs, bd, ..)) => (!writes, decl.name_span, def.0) < (!*bw, *bs, bd.0),
        };
        if better {
            best = Some((writes, decl.name_span, def, field, s));
        }
    }
    best
}

/// E0423 for every component with a cycle the walk closed (module docs), once per cycle;
/// E0005 case `type` where the walk met a component without a struct or enum.
fn holds_itself(prog: &Program, sigs: &Sigs, w: &mut Walk<'_, '_>) -> Vec<Diagnostic> {
    let comps = std::mem::take(&mut w.components);
    let none = FxHashSet::default();
    // components a declaration writes first, then by the place of the report
    let picks: Vec<Option<Pick>> = comps
        .iter()
        .map(|(members, _)| {
            let all: FxHashMap<TypeId, u32> = members.iter().map(|&m| (m, 0)).collect();
            pick(prog, sigs, w, members, &all, &none)
        })
        .collect();
    let mut order: Vec<usize> = (0..picks.len()).collect();
    order.sort_by_key(|&i| picks[i].map(|(writes, at, d, ..)| (!writes, at, d.0)));
    // The fields of each reported declaration that are on its cycle. A later component whose
    // every cycle passes through one of them is that cycle at other args (fixing the field
    // fixes it); one with a cycle left is reported on what is left.
    let mut cut: FxHashSet<(DeclId, usize)> = FxHashSet::default();
    let mut out = Vec::new();
    for i in order {
        let (members, site) = &comps[i];
        if picks[i].is_none() {
            if let Some(at) = site {
                out.extend(mk("E0005", Some("type"), &[], *at, &[]));
            }
            continue;
        }
        let at_of: FxHashMap<TypeId, u32> = members.iter().enumerate().map(|(k, &m)| (m, k as u32)).collect();
        let mut fields_of = Vec::with_capacity(members.len());
        let mut adj: Vec<Vec<u32>> = Vec::with_capacity(members.len());
        for &m in members {
            let f = inst_fields(sigs, w, m);
            let edges: Vec<u32> = match &f {
                Some((def, fs)) => fs
                    .iter()
                    .filter(|(k, ..)| !cut.contains(&(*def, *k)))
                    .filter_map(|&(.., st)| st.and_then(|st| at_of.get(&st).copied()))
                    .collect(),
                None => w.parts(m).iter().filter_map(|p| at_of.get(p).copied()).collect(),
            };
            w.tall_part = false;
            fields_of.push(f);
            adj.push(edges);
        }
        let sub: FxHashMap<TypeId, u32> =
            cyclic_sccs(&adj).into_iter().zip(members).filter_map(|(s, &m)| s.map(|s| (m, s))).collect();
        let Some((_, at, def, field, s)) = pick(prog, sigs, w, members, &sub, &cut) else { continue };
        for (m, f) in members.iter().zip(&fields_of) {
            let Some((d, fs)) = f else { continue };
            if *d != def || sub.get(m) != Some(&s) {
                continue;
            }
            for &(k, .., st) in fs {
                if st.and_then(|st| sub.get(&st)) == Some(&s) {
                    cut.insert((def, k));
                }
            }
        }
        let name = prog.decl_path(def);
        let roles: Vec<(&str, Span)> = field.map(|(_, f)| ("field", f)).into_iter().collect();
        out.extend(mk("E0423", None, &[("name", &name)], at, &roles));
    }
    out
}

/// E0005 case `type` and E0423 diagnostics (module docs). `bodies` sorted by decl.
pub(crate) fn check(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    graph: &CallGraph,
    brackets: &[Brackets],
) -> Vec<Diagnostic> {
    let mut w = Walk::new(graph.types(table), sigs);
    let mut out = Vec::new();
    let mut report = |at: Span| out.extend(mk("E0005", Some("type"), &[], at, &[]));

    // Declarations, in source order; one report per declaration. The type of each struct and
    // enum declaration (its params as args) is its first root.
    let mut roots: Vec<(Span, u32, TypeId)> = Vec::new();
    for (k, d) in prog.decls.iter().enumerate() {
        if !prog.item(d.item).is_some_and(|it| it.ok) {
            continue;
        }
        let k = k as u32;
        match sigs.get(DeclId(k)) {
            DeclSig::Fn(f) => {
                roots.extend(f.params.iter().zip(&f.param_spans).map(|(&t, &s)| (s, k, t)));
                roots.push((f.ret_span.unwrap_or(d.name_span), k, f.ret));
            }
            DeclSig::Adt(a) => {
                // the type itself, with its parameters as arguments: used or not
                let args: Box<[GArg]> = (0..a.generics.len() as u32)
                    .zip(&a.generics)
                    .map(|(i, g)| {
                        if g.is_const {
                            GArg::Const(ConstArg::Param(i))
                        } else {
                            GArg::Type(w.cx.mk_param(i))
                        }
                    })
                    .collect();
                let own = w.cx.intern(TypeKind::Adt { def: DeclId(k), args });
                roots.push((d.name_span, k, own));
                roots.extend(a.variants.iter().flat_map(|v| v.fields.iter().map(|f| (f.span, k, f.ty))));
            }
            DeclSig::Alias(a) => roots.push((d.name_span, k, a.target)),
            DeclSig::Impl(i) => roots.push((d.name_span, k, i.self_ty)),
            DeclSig::Const(c) | DeclSig::Static(c) => roots.push((d.name_span, k, c.ty)),
            _ => {}
        }
    }
    roots.sort();
    let mut reported: FxHashSet<u32> = FxHashSet::default();
    for (at, k, t) in roots {
        if !t.is_none() && !reported.contains(&k) && w.check(t, at) {
            reported.insert(k);
            report(at);
        }
    }

    // Bodies, generic ones with their parameters as leaves; one report per body.
    let body_types = |b: &TypedBody| -> Vec<(TypeId, u32)> {
        let mut v: Vec<(TypeId, u32)> =
            b.node_types.iter().enumerate().map(|(i, &t)| (t, b.lo + i as u32)).collect();
        v.extend(b.locals.iter().map(|&(n, t)| (t, n)));
        v.extend(b.envs.iter().map(|&(n, t)| (t, n)));
        for s in &b.sites {
            v.extend(s.inst.args.iter().filter_map(|a| match a {
                GArg::Type(t) => Some((*t, s.node)),
                GArg::Const(_) => None,
            }));
        }
        v.retain(|(t, _)| !t.is_none());
        v
    };
    // A body with other errors is measured too: its types are what it wrote.
    for b in bodies {
        let Some(icx) = brackets.get(b.item.idx()).and_then(|br| ItemCx::new(prog, b.item, br)) else {
            continue;
        };
        for (t, n) in body_types(b) {
            if w.check(t, icx.node_span(n)) {
                report(icx.node_span(n));
                break;
            }
        }
    }

    // Instances: their type args and their bodies' types after substitution; drop glue.
    let body_of = |d: DeclId| {
        bodies.binary_search_by_key(&d, |b| b.decl).ok().map(|i| &bodies[i]).filter(|b| !b.has_error)
    };
    let mut generic_types: FxHashMap<DeclId, Vec<TypeId>> = FxHashMap::default();
    for (n, node) in graph.nodes.iter().enumerate() {
        let at = graph.site_of(n);
        match &node.kind {
            NodeKind::Fn { inst, .. } if !inst.args.is_empty() => {
                let mut roots: Vec<TypeId> = inst
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        GArg::Type(t) => Some(*t),
                        GArg::Const(_) => None,
                    })
                    .collect();
                if let Some(b) = body_of(inst.decl) {
                    let ts = match generic_types.get(&inst.decl) {
                        Some(ts) => ts.clone(),
                        None => {
                            let mut ts: Vec<TypeId> = body_types(b).into_iter().map(|(t, _)| t).collect();
                            ts.sort();
                            ts.dedup();
                            ts.retain(|&t| w.has_params(t));
                            generic_types.insert(inst.decl, ts.clone());
                            ts
                        }
                    };
                    let mut tall = false;
                    for t in ts {
                        // a type the table could not hold is too deep by itself
                        if crate::types::subst_height(&w.cx, t, &inst.args) > MAX_TYPE_DEPTH {
                            tall = true;
                            break;
                        }
                        let s = w.cx.subst(t, &inst.args, None);
                        roots.push(s);
                    }
                    if tall {
                        report(at);
                        continue;
                    }
                }
                for t in roots {
                    if w.check(t, at) {
                        report(at);
                        break;
                    }
                }
            }
            NodeKind::Glue { ty } if w.check(*ty, at) => report(at),
            _ => {}
        }
    }
    out.extend(holds_itself(prog, sigs, &mut w));
    out
}
