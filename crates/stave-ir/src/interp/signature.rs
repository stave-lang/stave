//! Borrow tracking: the check of `from` clauses against what one call did (M24 of
//! contracts/borrow-model.md). At the entry of a fn with clause rows the interpreter keeps the
//! tags its arguments reach; at the return it asks of each reference in the result, and of
//! each reference that went into a view a param can write, whether its tag derives from a
//! source the clauses name. A reference that does not is a signature report: not a violation,
//! the run goes on.

use std::collections::HashSet;
use std::rc::Rc;

use super::{Interp, Val};
use crate::borrowtrack::{ROOT, Tag};
use crate::host::Host;
use crate::ir::{ClauseKind, FuncId};
use crate::layout::Shape;
use crate::memcheck::{At, MemReport, Trace};
use crate::types::{MAX_TYPE_DEPTH, TypeId, TypeKey};

/// How many reference words one walk of a value looks at; a value with more is not judged.
const MAX_WORDS: usize = 4096;

/// What a reference word derives from: a tag, or the root of an allocation (a word without a
/// tag acts as the root of the allocation it points to).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Src {
    Tag(Tag),
    Root(u32),
}

/// A reference word: where it points, its tag, how many pointees (a slice: its length), and
/// its type.
#[derive(Clone, Copy, Debug)]
struct Word {
    addr: u64,
    tag: Tag,
    len: u64,
    ty: TypeId,
}

/// What the check needs of a fn, by its clause rows and types.
#[derive(Clone, Debug, Default)]
pub(super) struct Plan {
    closure: bool,
    /// The result holds a reference.
    result: bool,
    /// Per param: its words are a source of some clause.
    source: Vec<bool>,
    /// Per param: it has a `&mut` word whose pointee holds a reference.
    target: Vec<bool>,
}

impl Plan {
    fn idle(&self) -> bool {
        !self.result && !self.target.iter().any(|&t| t)
    }
}

/// The entry state of one call.
#[derive(Clone, Debug)]
pub(super) struct Entry {
    plan: Rc<Plan>,
    args: Vec<Val>,
    /// Per source param: the sources of its own words, and of the words below them.
    top: Vec<Vec<Src>>,
    below: Vec<HashSet<Src>>,
    /// Per target param: the sources of the words writable through it.
    writable: Vec<HashSet<Src>>,
    /// A walk at the entry was cut short: the call is not judged.
    cut: bool,
}

/// A `from` clause one call did not keep.
#[derive(Clone, Debug, PartialEq)]
pub struct SignatureReport {
    pub func: FuncId,
    /// The param whose view took the reference; `None`: the result holds it.
    pub param: Option<u32>,
    /// The return, innermost frame first.
    pub at: Trace,
    /// Where the tag of the reference was made.
    pub made_at: Option<At>,
}

impl SignatureReport {
    /// Multi-line text: `borrow signature [signature] M24: ...`, then the two sites.
    pub fn render(&self, m: &crate::ir::Module) -> String {
        let path = m.func(self.func).map_or("?", |f| f.path.as_str());
        let what = match self.param {
            None => format!("the result of {path} holds a reference its `from` clause does not give"),
            Some(p) => {
                format!("param {p} of {path} took a reference the `from` clauses do not give")
            }
        };
        let made = match &self.made_at {
            Some(a) => MemReport::trace_text(m, &vec![*a]),
            None => "?".into(),
        };
        format!(
            "borrow signature [signature] M24: {what}\n  at {}\n  tag made at {made}",
            MemReport::trace_text(m, &self.at)
        )
    }
}

impl<'m, H: Host> Interp<'m, H> {
    /// Whether a value of type `t` holds a reference word (not behind a pointer, not in a fn
    /// value).
    fn holds_refs(&mut self, t: TypeId, depth: usize) -> bool {
        if let Some(&r) = self.bt.as_ref().and_then(|bt| bt.refs.get(&t)) {
            return r;
        }
        let r = match self.m.types.key(t) {
            Some(TypeKey::Ref { .. }) => true,
            Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. })
                if depth < MAX_TYPE_DEPTH =>
            {
                let parts: Vec<TypeId> = match self.layout(t).map(|l| &l.shape) {
                    Ok(Shape::Fields(fs)) => fs.iter().map(|f| f.1).collect(),
                    Ok(Shape::Array { elem, .. }) => vec![*elem],
                    Ok(Shape::Enum(vs)) => vs.iter().flatten().map(|f| f.1).collect(),
                    _ => Vec::new(),
                };
                parts.into_iter().any(|p| self.holds_refs(p, depth + 1))
            }
            _ => false,
        };
        if let Some(bt) = &mut self.bt {
            bt.refs.insert(t, r);
        }
        r
    }

    /// The pointee of reference type `rt` as (element type, whether `rt` is `&mut`): of a
    /// slice reference its element; `None` for `&str`.
    fn pointee_of(&self, rt: TypeId) -> Option<(TypeId, bool)> {
        let Some(&TypeKey::Ref { mutable, .. }) = self.m.types.key(rt) else { return None };
        let inner = self.pointee.get(rt.index()).copied().flatten()?;
        match self.m.types.key(inner) {
            Some(TypeKey::Slice { elem, .. }) => Some((*elem, mutable)),
            Some(TypeKey::Str) => None,
            _ => Some((inner, mutable)),
        }
    }

    /// Whether a value of type `t` has a `&mut` word whose pointee holds a reference.
    fn has_target(&mut self, t: TypeId, depth: usize) -> bool {
        if depth > MAX_TYPE_DEPTH {
            return false;
        }
        match self.m.types.key(t) {
            Some(TypeKey::Ref { .. }) => match self.pointee_of(t) {
                Some((e, true)) => self.holds_refs(e, 0),
                _ => false,
            },
            Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => {
                let parts: Vec<TypeId> = match self.layout(t).map(|l| &l.shape) {
                    Ok(Shape::Fields(fs)) => fs.iter().map(|f| f.1).collect(),
                    Ok(Shape::Array { elem, .. }) => vec![*elem],
                    Ok(Shape::Enum(vs)) => vs.iter().flatten().map(|f| f.1).collect(),
                    _ => Vec::new(),
                };
                parts.into_iter().any(|p| self.has_target(p, depth + 1))
            }
            _ => false,
        }
    }

    /// The plan of fn `f`; `None` when its calls need no check.
    fn plan(&mut self, f: FuncId) -> Option<Rc<Plan>> {
        if let Some(p) = self.bt.as_ref()?.plans.get(f.index())? {
            return (!p.idle()).then(|| p.clone());
        }
        let m = self.m;
        let func = m.func(f)?;
        let mut plan = Plan::default();
        if let Some(c) = m.clauses.get(f.index()).filter(|c| c.kind != ClauseKind::Unchecked) {
            plan.closure = c.kind == ClauseKind::Closure;
            plan.result = func.ret.is_some_and(|t| self.holds_refs(t, 0));
            for &t in &func.params {
                plan.target.push(self.has_target(t, 0));
            }
            plan.source = vec![false; func.params.len()];
            if plan.closure {
                for (k, &t) in func.params.iter().enumerate() {
                    plan.source[k] = plan.result && self.holds_refs(t, 0);
                }
            } else {
                let named = c.result.iter().map(|r| r.0).chain(c.params.iter().map(|p| p.1));
                for x in named {
                    if let Some(s) = plan.source.get_mut(x as usize) {
                        *s = true;
                    }
                }
            }
        }
        let plan = Rc::new(plan);
        if let Some(slot) = self.bt.as_mut().and_then(|bt| bt.plans.get_mut(f.index())) {
            *slot = Some(plan.clone());
        }
        (!plan.idle()).then_some(plan)
    }

    /// The reference words of `v`, a value of type `t` (of an enum the live variant).
    fn value_words(&self, t: TypeId, v: &Val, out: &mut Vec<Word>, depth: usize) {
        if depth > MAX_TYPE_DEPTH || out.len() > MAX_WORDS {
            return;
        }
        match self.m.types.key(t) {
            Some(TypeKey::Ref { .. }) => {
                let len = match v {
                    Val::Pair(_, n) | Val::PairT(_, n, _) => *n,
                    _ => 1,
                };
                if let Some((addr, tag)) = v.word() {
                    out.push(Word { addr, tag, len, ty: t });
                }
                return;
            }
            Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => {}
            _ => return,
        }
        let Val::Agg(vs) = v else { return };
        let Ok(l) = self.layout(t) else { return };
        match &l.shape {
            Shape::Fields(fs) => {
                for (&(_, ft), fv) in fs.iter().zip(vs.iter()) {
                    self.value_words(ft, fv, out, depth + 1);
                }
            }
            Shape::Array { elem, .. } => {
                for fv in vs.iter() {
                    self.value_words(*elem, fv, out, depth + 1);
                }
            }
            Shape::Enum(vars) => {
                let live = vs.first().and_then(Val::bits).and_then(|k| vars.get(k as usize));
                for (&(_, ft), fv) in live.into_iter().flatten().zip(vs.iter().skip(1)) {
                    self.value_words(ft, fv, out, depth + 1);
                }
            }
            Shape::Scalar | Shape::Fat | Shape::Bytes => {}
        }
    }

    /// The reference words of the value of type `t` stored at `addr`. Bytes that are not live
    /// and initialised hold none. `false`: more than [`MAX_WORDS`].
    fn stored_words(&self, t: TypeId, addr: u64, out: &mut Vec<Word>, depth: usize) -> bool {
        if out.len() > MAX_WORDS {
            return false;
        }
        if depth > MAX_TYPE_DEPTH {
            return true;
        }
        let Some(bt) = &self.bt else { return true };
        match self.m.types.key(t) {
            Some(TypeKey::Ref { .. }) => {
                let Some(bits) = self.mem.peek_uint(addr, 8) else { return true };
                let fat = matches!(self.layout(t).map(|l| &l.shape), Ok(Shape::Fat));
                let len = if fat { self.mem.peek_uint(addr.wrapping_add(8), 8).unwrap_or(0) } else { 1 };
                out.push(Word { addr: bits, tag: bt.loaded(addr, bits), len, ty: t });
                return true;
            }
            Some(TypeKey::Tuple(_) | TypeKey::Array { .. } | TypeKey::Adt { .. }) => {}
            _ => return true,
        }
        if !bt.refs.get(&t).copied().unwrap_or(true) {
            return true;
        }
        let Ok(l) = self.layout(t) else { return true };
        match &l.shape {
            Shape::Fields(fs) => {
                fs.iter().all(|&(off, ft)| self.stored_words(ft, addr.wrapping_add(off), out, depth + 1))
            }
            Shape::Array { elem, stride, len } => (0..*len).all(|k| {
                self.stored_words(*elem, addr.wrapping_add(k.wrapping_mul(*stride)), out, depth + 1)
            }),
            Shape::Enum(vars) => {
                let live = self.mem.peek_uint(addr, 4).and_then(|k| vars.get(k as usize));
                live.into_iter()
                    .flatten()
                    .all(|&(off, ft)| self.stored_words(ft, addr.wrapping_add(off), out, depth + 1))
            }
            Shape::Scalar | Shape::Fat | Shape::Bytes => true,
        }
    }

    /// What `w` derives from by itself; `None`: it points to no allocation.
    fn src(&self, w: &Word) -> Option<Src> {
        if w.tag != ROOT {
            return Some(Src::Tag(w.tag));
        }
        self.mem.alloc_span(w.addr).map(|a| Src::Root(a.0))
    }

    /// The reference words in the pointees of `w`. `None`: too many.
    fn below(&mut self, w: &Word) -> Option<Vec<Word>> {
        let mut out = Vec::new();
        let Some((elem, _)) = self.pointee_of(w.ty) else { return Some(out) };
        if !self.holds_refs(elem, 0) {
            return Some(out);
        }
        let stride = self.layout(elem).map_or(0, |l| l.size);
        for k in 0..w.len {
            if !self.stored_words(elem, w.addr.wrapping_add(k.wrapping_mul(stride)), &mut out, 0) {
                return None;
            }
        }
        Some(out)
    }

    /// The sources of the words reachable from `top` below its own words: those in the views
    /// the words point to, through every reference. `None`: too many.
    fn reach(&mut self, top: &[Word]) -> Option<HashSet<Src>> {
        let mut set = HashSet::new();
        let mut seen: HashSet<(u64, TypeId)> = HashSet::new();
        let mut work: Vec<Word> = top.to_vec();
        while let Some(w) = work.pop() {
            if !seen.insert((w.addr, w.ty)) {
                continue;
            }
            if seen.len() > MAX_WORDS {
                return None;
            }
            for f in self.below(&w)? {
                set.extend(self.src(&f));
                work.push(f);
            }
        }
        Some(set)
    }

    /// The words writable through `top`: those reachable from its words behind `&mut` words
    /// only. `None`: too many.
    fn writable(&mut self, top: &[Word]) -> Option<Vec<Word>> {
        let mut out = Vec::new();
        let mut seen: HashSet<(u64, TypeId)> = HashSet::new();
        let exclusive = |it: &Self, w: &Word| it.pointee_of(w.ty).is_some_and(|p| p.1);
        let mut work: Vec<Word> = top.iter().copied().filter(|w| exclusive(self, w)).collect();
        while let Some(w) = work.pop() {
            if !seen.insert((w.addr, w.ty)) {
                continue;
            }
            if seen.len() > MAX_WORDS || out.len() > MAX_WORDS {
                return None;
            }
            for f in self.below(&w)? {
                out.push(f);
                if exclusive(self, &f) {
                    work.push(f);
                }
            }
        }
        Some(out)
    }

    /// The entry state of a call of `f` with `args`; `None` when the call needs no check.
    pub(super) fn signature_entry(&mut self, f: FuncId, args: &[Val]) -> Option<Box<Entry>> {
        let plan = self.plan(f)?;
        let m = self.m;
        let params = &m.func(f)?.params;
        let n = params.len();
        let mut e = Entry {
            plan: plan.clone(),
            args: args.to_vec(),
            top: vec![Vec::new(); n],
            below: vec![HashSet::new(); n],
            writable: vec![HashSet::new(); n],
            cut: false,
        };
        for k in 0..n {
            let (source, target) = (plan.source.get(k) == Some(&true), plan.target.get(k) == Some(&true));
            if !source && !target {
                continue;
            }
            let (Some(&t), Some(v)) = (params.get(k), args.get(k)) else { continue };
            let mut top = Vec::new();
            self.value_words(t, v, &mut top, 0);
            if source {
                e.top[k] = top.iter().filter_map(|w| self.src(w)).collect();
                match self.reach(&top) {
                    Some(s) => e.below[k] = s,
                    None => e.cut = true,
                }
            }
            if target {
                match self.writable(&top) {
                    Some(ws) => e.writable[k] = ws.iter().filter_map(|w| self.src(w)).collect(),
                    None => e.cut = true,
                }
            }
        }
        Some(Box::new(e))
    }

    /// Adds the sources of clause entry (`x`, `copy`) to `out`: the words reachable from
    /// argument `x` at the entry or now, without its own words when `copy`. `false`: too many.
    fn add_sources(&mut self, e: &Entry, f: FuncId, x: u32, copy: bool, out: &mut HashSet<Src>) -> bool {
        let k = x as usize;
        let m = self.m;
        let (Some(&t), Some(v)) = (m.func(f).and_then(|g| g.params.get(k)), e.args.get(k)) else {
            return true;
        };
        if let Some(b) = e.below.get(k) {
            out.extend(b.iter().copied());
        }
        let mut top = Vec::new();
        self.value_words(t, v, &mut top, 0);
        if !copy {
            out.extend(top.iter().filter_map(|w| self.src(w)));
            out.extend(e.top.get(k).into_iter().flatten().copied());
        }
        match self.reach(&top) {
            Some(s) => {
                out.extend(s);
                true
            }
            None => false,
        }
    }

    /// Whether the tag of `w` derives from a source in `from`; `None`: the word is not judged
    /// (it points to no allocation or to a `static`, or its tag was made through a raw pointer
    /// or below such a tag).
    fn derives(&self, w: &Word, from: &HashSet<Src>, chain: &mut Vec<Tag>) -> Option<bool> {
        let bt = self.bt.as_ref()?;
        if w.tag == ROOT {
            let id = self.mem.alloc_span(w.addr)?.0;
            return (!bt.static_ids.contains(&id)).then(|| from.contains(&Src::Root(id)));
        }
        let (alloc, raw) = bt.tracker.lineage(w.tag, chain)?;
        if raw || bt.static_ids.contains(&alloc) {
            return None;
        }
        Some(chain.iter().any(|&t| from.contains(&Src::Tag(t))) || from.contains(&Src::Root(alloc)))
    }

    /// M24 at the return of `f` with result `ret`: a report for the first reference word of
    /// the result, and for the first one each param's views took, that no source gives.
    pub(super) fn signature_return(&mut self, f: FuncId, e: &Entry, ret: Option<&Val>, at: &Trace) {
        if e.cut {
            return;
        }
        let m = self.m;
        let Some(func) = m.func(f) else { return };
        let none = crate::ir::Clauses::default();
        let clauses = m.clauses.get(f.index()).unwrap_or(&none);
        let n = func.params.len() as u32;
        let mut chain = Vec::new();
        let mut found: Vec<(Option<u32>, Tag)> = Vec::new();
        if let (true, Some(t), Some(v)) = (e.plan.result, func.ret, ret) {
            let mut from = HashSet::new();
            let named: Vec<(u32, bool)> =
                if e.plan.closure { (0..n).map(|x| (x, false)).collect() } else { clauses.result.clone() };
            let whole = named.iter().all(|&(x, copy)| self.add_sources(e, f, x, copy, &mut from));
            let mut words = Vec::new();
            self.value_words(t, v, &mut words, 0);
            if whole && let Some(w) = words.iter().find(|w| self.derives(w, &from, &mut chain) == Some(false))
            {
                found.push((None, w.tag));
            }
        }
        for p in 0..n {
            if e.plan.target.get(p as usize) != Some(&true) {
                continue;
            }
            let (Some(&t), Some(v)) = (func.params.get(p as usize), e.args.get(p as usize)) else { continue };
            let mut top = Vec::new();
            self.value_words(t, v, &mut top, 0);
            let Some(now) = self.writable(&top) else { continue };
            let before = &e.writable[p as usize];
            let fresh: Vec<Word> =
                now.into_iter().filter(|w| self.src(w).is_some_and(|s| !before.contains(&s))).collect();
            if fresh.is_empty() {
                continue;
            }
            // the clause of `p`, and the clauses that name `p` without `copy`
            let mut from: HashSet<Src> = before.clone();
            let mut whole = true;
            for &(target, x, copy) in &clauses.params {
                let names_p = clauses.params.iter().any(|c| c.0 == target && c.1 == p && !c.2);
                if target == p || names_p {
                    whole &= self.add_sources(e, f, x, copy, &mut from);
                }
            }
            if whole && let Some(w) = fresh.iter().find(|w| self.derives(w, &from, &mut chain) == Some(false))
            {
                found.push((Some(p), w.tag));
            }
        }
        let Some(bt) = &mut self.bt else { return };
        for (param, tag) in found {
            if bt.signatures.len() >= 16 {
                break;
            }
            let made = bt.tracker.made_at(tag);
            let made_at = (|| {
                let g = FuncId(u32::try_from((made >> 32).checked_sub(1)?).ok()?);
                Some(At::of(g, m.func(g)?.body.as_ref()?, made as u32))
            })();
            bt.signatures.push(SignatureReport { func: f, param, at: at.clone(), made_at });
        }
    }

    /// The signature reports of the run so far (M24), as text.
    pub fn signature_reports(&self) -> Vec<String> {
        self.bt.as_ref().map_or(Vec::new(), |bt| bt.signatures.iter().map(|s| s.render(self.m)).collect())
    }
}
