//! The dynamic borrow model (`contracts/borrow-model.md`) as a state machine: per allocation a
//! tree of tags, per tag one permission per byte, kept as runs. The caller feeds it the events
//! of one execution (allocation begin and end, borrow, read, write, value end) and gets a
//! [`Report`] for the first one a rule forbids.
//!
//! The tracker knows no IR and no addresses: an allocation is the id the caller gives it, bytes
//! are offsets into it, and a site is an opaque [`Ev`] handle the caller resolves when it renders
//! a report.

use std::collections::HashMap;

/// A tag: made by one event, never reused. [`ROOT`] is the untagged word.
pub type Tag = u32;

/// The tag of a word that carries none: it acts as the root of the allocation it is used on
/// (M20).
pub const ROOT: Tag = 0;

/// The caller's handle of an event's site. [`NO_EV`] is "no site".
pub type Ev = u64;
pub const NO_EV: Ev = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Root,
    Shared,
    Mut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Perm {
    Reserved,
    Active,
    Frozen,
    Cell,
    Dead,
}

/// What the violating event was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Use {
    Read,
    Write,
    Borrow,
    End,
}

impl Use {
    /// The rule the event falls under.
    pub fn rule(self) -> &'static str {
        match self {
            Use::Read => "M1",
            Use::Write => "M2",
            Use::End => "M3",
            Use::Borrow => "M4",
        }
    }
}

/// What made a permission `Frozen` or `Dead`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cause {
    /// The permission is the one the tag was made with: the cause is the borrow itself.
    Made,
    Read,
    Write,
    End,
}

/// The report forms. [`Class::name`] is the stable, machine-readable spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// (a) a use through a `&` that a write made `Dead`; a write through a `Frozen` `&`.
    WriteUnderShared,
    /// (b) a use through a `&mut` that a foreign access made `Dead` or `Frozen`.
    TwoExclusive,
    /// (c) a use through a tag made `Dead` by an end.
    UseAfterReferent,
    /// (d) a read or drop of a moved-out place.
    SecondOwner,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::WriteUnderShared => "write-under-shared",
            Class::TwoExclusive => "two-exclusive",
            Class::UseAfterReferent => "use-after-referent",
            Class::SecondOwner => "second-owner",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub class: Class,
    pub event: Use,
    pub alloc: u32,
    /// The first byte the event may not touch, and the bytes of the event.
    pub offset: u64,
    pub lo: u64,
    pub hi: u64,
    /// The failing tag: the first from the accessing tag towards the root whose permission
    /// forbids the event.
    pub tag: Tag,
    pub kind: Kind,
    pub perm: Perm,
    pub at: Ev,
    pub cause: Cause,
    pub cause_at: Ev,
    pub made_at: Ev,
}

/// Permission of the bytes from the end of the run before (or the tag's first byte) to `hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    hi: u64,
    perm: Perm,
    cause: Cause,
    cause_at: Ev,
}

#[derive(Clone, Debug)]
struct Node {
    tag: Tag,
    parent: u32,
    kind: Kind,
    lo: u64,
    hi: u64,
    made_at: Ev,
    runs: Vec<Run>,
}

const NO_PARENT: u32 = u32::MAX;

impl Node {
    fn holds(&self, lo: u64, hi: u64) -> bool {
        self.lo <= lo && hi <= self.hi
    }

    /// The first run among the bytes `lo..hi` whose permission `bad` rejects, with its first
    /// byte inside the range.
    fn find(&self, lo: u64, hi: u64, bad: impl Fn(Perm) -> bool) -> Option<(u64, Run)> {
        let mut start = self.lo;
        for r in &self.runs {
            if r.hi > lo && start < hi && bad(r.perm) {
                return Some((start.max(lo), *r));
            }
            start = r.hi;
        }
        None
    }

    /// Replaces the permission of each byte in `lo..hi` by `f` of it; a changed byte takes the
    /// cause given.
    fn update(&mut self, lo: u64, hi: u64, cause: Cause, at: Ev, f: impl Fn(Perm) -> Perm) {
        let (lo, hi) = (lo.max(self.lo), hi.min(self.hi));
        if lo >= hi {
            return;
        }
        let mut start = self.lo;
        let mut touched = false;
        for r in &self.runs {
            if r.hi > lo && start < hi && f(r.perm) != r.perm {
                touched = true;
                break;
            }
            start = r.hi;
        }
        if !touched {
            return;
        }
        let mut out: Vec<Run> = Vec::with_capacity(self.runs.len() + 2);
        let mut push = |r: Run| match out.last_mut() {
            Some(l) if l.perm == r.perm && l.cause == r.cause && l.cause_at == r.cause_at => l.hi = r.hi,
            _ => out.push(r),
        };
        let mut start = self.lo;
        for &r in &self.runs {
            let (s, e) = (start, r.hi);
            start = e;
            let new = f(r.perm);
            if e <= lo || s >= hi || new == r.perm {
                push(r);
                continue;
            }
            if s < lo {
                push(Run { hi: lo, ..r });
            }
            push(Run { hi: e.min(hi), perm: new, cause, cause_at: at });
            if e > hi {
                push(r);
            }
        }
        self.runs = out;
    }
}

#[derive(Clone, Debug, Default)]
struct Tree {
    size: u64,
    nodes: Vec<Node>,
}

/// The state of every allocation the caller named.
#[derive(Clone, Debug)]
pub struct Tracker {
    trees: HashMap<u32, Tree>,
    /// Allocation and node of each tag but the roots.
    tags: HashMap<Tag, (u32, u32)>,
    next: Tag,
}

impl Default for Tracker {
    fn default() -> Tracker {
        Tracker::new()
    }
}

impl Tracker {
    pub fn new() -> Tracker {
        Tracker { trees: HashMap::new(), tags: HashMap::new(), next: ROOT + 1 }
    }

    /// Begins allocation `alloc` of `size` bytes. Its root is `Active`, or `Frozen` for a
    /// `static` that is not `mut`.
    pub fn begin(&mut self, alloc: u32, size: u64, frozen: bool, at: Ev) {
        let (kind, perm) = if frozen { (Kind::Shared, Perm::Frozen) } else { (Kind::Root, Perm::Active) };
        let root = Node {
            tag: ROOT,
            parent: NO_PARENT,
            kind,
            lo: 0,
            hi: size,
            made_at: at,
            runs: vec![Run { hi: size, perm, cause: Cause::Made, cause_at: at }],
        };
        self.trees.insert(alloc, Tree { size, nodes: vec![root] });
    }

    /// The node an event on `alloc` through `tag` starts from: the tag's own, or the root for an
    /// untagged word and for a tag of another allocation.
    fn node_of(&self, alloc: u32, tag: Tag) -> u32 {
        match self.tags.get(&tag) {
            Some(&(a, n)) if a == alloc => n,
            _ => 0,
        }
    }

    /// Whether allocation `alloc` has begun.
    pub fn has(&self, alloc: u32) -> bool {
        self.trees.contains_key(&alloc)
    }

    /// M4: a new tag of `kind` for `bytes` of `alloc`, reached through `parent`.
    /// A `&` is `Cell` on the byte ranges in `cells`. An allocation the tracker does not know
    /// gives [`ROOT`].
    pub fn borrow(
        &mut self,
        alloc: u32,
        parent: Tag,
        kind: Kind,
        bytes: std::ops::Range<u64>,
        cells: &[(u64, u64)],
        at: Ev,
    ) -> Result<Tag, Box<Report>> {
        let (lo, hi) = (bytes.start, bytes.end);
        let from = self.node_of(alloc, parent);
        let Some(tree) = self.trees.get_mut(&alloc) else { return Ok(ROOT) };
        let (lo, hi) = (lo.min(tree.size), hi.min(tree.size).max(lo.min(tree.size)));
        let mut p = from;
        while !tree.nodes[p as usize].holds(lo, hi) && tree.nodes[p as usize].parent != NO_PARENT {
            p = tree.nodes[p as usize].parent;
        }
        if let Some(r) = check(tree, alloc, p, Use::Borrow, lo, hi, at) {
            return Err(r);
        }
        let first = if kind == Kind::Mut { Perm::Reserved } else { Perm::Frozen };
        let mut node = Node {
            tag: self.next,
            parent: p,
            kind,
            lo,
            hi,
            made_at: at,
            runs: vec![Run { hi, perm: first, cause: Cause::Made, cause_at: at }],
        };
        if kind == Kind::Shared {
            for &(a, b) in cells {
                node.update(a, b, Cause::Made, at, |_| Perm::Cell);
            }
        }
        tree.nodes.push(node);
        self.tags.insert(self.next, (alloc, tree.nodes.len() as u32 - 1));
        self.next += 1;
        Ok(self.next - 1)
    }

    /// M1: a read of the bytes `lo..hi` of `alloc` through `tag`.
    pub fn read(&mut self, alloc: u32, tag: Tag, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        self.event(alloc, tag, Use::Read, lo, hi, at)
    }

    /// M2: a write of the bytes `lo..hi` of `alloc` through `tag`.
    pub fn write(&mut self, alloc: u32, tag: Tag, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        self.event(alloc, tag, Use::Write, lo, hi, at)
    }

    /// M3: the value in the bytes `lo..hi` of `alloc` ends through `tag`.
    pub fn end(&mut self, alloc: u32, tag: Tag, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        self.event(alloc, tag, Use::End, lo, hi, at)
    }

    /// The allocation ends: every tag of it is `Dead` on every byte, the root included.
    pub fn end_alloc(&mut self, alloc: u32, at: Ev) {
        if let Some(tree) = self.trees.get_mut(&alloc) {
            for n in &mut tree.nodes {
                let (lo, hi) = (n.lo, n.hi);
                n.update(lo, hi, Cause::End, at, |_| Perm::Dead);
            }
        }
    }

    /// The permission of `tag` at byte `offset` of `alloc`, if the byte is in its range.
    pub fn perm(&self, alloc: u32, tag: Tag, offset: u64) -> Option<Perm> {
        let n = &self.trees.get(&alloc)?.nodes[self.node_of(alloc, tag) as usize];
        n.find(offset, offset + 1, |_| true).map(|(_, r)| r.perm)
    }

    fn event(&mut self, alloc: u32, tag: Tag, ev: Use, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        let from = self.node_of(alloc, tag);
        let Some(tree) = self.trees.get_mut(&alloc) else { return Ok(()) };
        let hi = hi.min(tree.size);
        // A byte outside the tag's range is used through the nearest ancestor that holds it
        // (M20): the range splits into the part of the tag and the parts of its ancestors.
        let mut pieces: Vec<(u32, u64, u64)> = Vec::new();
        let mut rest: Vec<(u64, u64)> = vec![(lo, hi)];
        let mut n = from;
        loop {
            let node = &tree.nodes[n as usize];
            let mut out = Vec::new();
            for (a, b) in rest {
                let (ia, ib) = (a.max(node.lo), b.min(node.hi));
                if ia >= ib {
                    out.push((a, b));
                    continue;
                }
                pieces.push((n, ia, ib));
                if a < ia {
                    out.push((a, ia));
                }
                if ib < b {
                    out.push((ib, b));
                }
            }
            rest = out;
            rest.retain(|&(a, b)| a < b);
            if rest.is_empty() || node.parent == NO_PARENT {
                break;
            }
            n = node.parent;
        }
        for &(n, a, b) in &pieces {
            if let Some(r) = check(tree, alloc, n, ev, a, b, at) {
                return Err(r);
            }
        }
        for &(n, a, b) in &pieces {
            apply(tree, n, ev, a, b, at);
        }
        Ok(())
    }
}

/// The report of event `ev` on the bytes `lo..hi` through node `n`, if `n` or an ancestor
/// forbids it.
fn check(tree: &Tree, alloc: u32, n: u32, ev: Use, lo: u64, hi: u64, at: Ev) -> Option<Box<Report>> {
    if lo >= hi {
        return None;
    }
    let mut cur = n;
    loop {
        let node = &tree.nodes[cur as usize];
        let bad = |p: Perm| p == Perm::Dead || (ev == Use::Write && p == Perm::Frozen);
        if let Some((offset, run)) = node.find(lo, hi, bad) {
            let class = match (run.cause, node.kind) {
                (Cause::End, _) => Class::UseAfterReferent,
                (_, Kind::Mut) => Class::TwoExclusive,
                _ => Class::WriteUnderShared,
            };
            return Some(Box::new(Report {
                class,
                event: ev,
                alloc,
                offset,
                lo,
                hi,
                tag: node.tag,
                kind: node.kind,
                perm: run.perm,
                at,
                cause: run.cause,
                cause_at: run.cause_at,
                made_at: node.made_at,
            }));
        }
        if node.parent == NO_PARENT {
            return None;
        }
        cur = node.parent;
    }
}

/// The permission changes of an allowed event on the bytes `lo..hi` through node `n`: a child
/// event for `n` and its ancestors, a foreign one for every other tag.
fn apply(tree: &mut Tree, n: u32, ev: Use, lo: u64, hi: u64, at: Ev) {
    let mut path: Vec<u32> = Vec::new();
    let mut cur = n;
    loop {
        path.push(cur);
        let p = tree.nodes[cur as usize].parent;
        if p == NO_PARENT {
            break;
        }
        cur = p;
    }
    for (i, node) in tree.nodes.iter_mut().enumerate() {
        let child = path.contains(&(i as u32));
        match (ev, child) {
            (Use::Write, true) => node.update(lo, hi, Cause::Write, at, |p| match p {
                Perm::Reserved => Perm::Active,
                p => p,
            }),
            (Use::Read | Use::End | Use::Borrow, true) => {}
            (Use::Read, false) => node.update(lo, hi, Cause::Read, at, |p| match p {
                Perm::Active => Perm::Frozen,
                p => p,
            }),
            (Use::Write, false) => node.update(lo, hi, Cause::Write, at, |p| match p {
                Perm::Cell => Perm::Cell,
                _ => Perm::Dead,
            }),
            (Use::End, false) => node.update(lo, hi, Cause::End, at, |_| Perm::Dead),
            (Use::Borrow, false) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: u32 = 1;

    fn local(size: u64) -> Tracker {
        let mut t = Tracker::new();
        t.begin(X, size, false, 1);
        t
    }

    fn shared(t: &mut Tracker, parent: Tag, lo: u64, hi: u64, at: Ev) -> Tag {
        t.borrow(X, parent, Kind::Shared, lo..hi, &[], at).unwrap()
    }

    fn excl(t: &mut Tracker, parent: Tag, lo: u64, hi: u64, at: Ev) -> Tag {
        t.borrow(X, parent, Kind::Mut, lo..hi, &[], at).unwrap()
    }

    #[test]
    fn write_under_shared_used_later() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.write(X, ROOT, 0, 4, 3).unwrap();
        let e = t.read(X, r, 0, 4, 4).unwrap_err();
        assert_eq!((e.class, e.event, e.tag, e.kind), (Class::WriteUnderShared, Use::Read, r, Kind::Shared));
        assert_eq!((e.at, e.cause, e.cause_at, e.made_at), (4, Cause::Write, 3, 2));
        assert_eq!(e.event.rule(), "M1");
    }

    #[test]
    fn write_through_shared() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        let e = t.write(X, r, 0, 4, 3).unwrap_err();
        assert_eq!(
            (e.class, e.perm, e.cause, e.cause_at),
            (Class::WriteUnderShared, Perm::Frozen, Cause::Made, 2)
        );
    }

    #[test]
    fn exclusive_frozen_by_foreign_read() {
        let mut t = local(4);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        t.write(X, m, 0, 4, 3).unwrap();
        assert_eq!(t.perm(X, m, 0), Some(Perm::Active));
        t.read(X, ROOT, 0, 4, 4).unwrap();
        t.read(X, m, 0, 4, 5).unwrap();
        let e = t.write(X, m, 0, 4, 6).unwrap_err();
        assert_eq!(
            (e.class, e.event, e.perm, e.cause, e.cause_at),
            (Class::TwoExclusive, Use::Write, Perm::Frozen, Cause::Read, 4)
        );
    }

    #[test]
    fn two_exclusive_both_used() {
        let mut t = local(4);
        let a = excl(&mut t, ROOT, 0, 4, 2);
        let b = excl(&mut t, ROOT, 0, 4, 3);
        t.write(X, a, 0, 4, 4).unwrap();
        // The write through `a` was a foreign write for the `Reserved` `b`.
        let e = t.write(X, b, 0, 4, 5).unwrap_err();
        assert_eq!(
            (e.class, e.tag, e.cause, e.cause_at, e.made_at),
            (Class::TwoExclusive, b, Cause::Write, 4, 3)
        );
        let mut t = local(4);
        let a = excl(&mut t, ROOT, 0, 4, 2);
        let b = excl(&mut t, ROOT, 0, 4, 3);
        t.write(X, b, 0, 4, 4).unwrap();
        let e = t.read(X, a, 0, 4, 5).unwrap_err();
        assert_eq!(
            (e.class, e.tag, e.cause, e.cause_at, e.made_at),
            (Class::TwoExclusive, a, Cause::Write, 4, 2)
        );
    }

    #[test]
    fn borrow_is_not_an_access() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        let m = excl(&mut t, ROOT, 0, 4, 3);
        t.read(X, r, 0, 4, 4).unwrap();
        t.read(X, m, 0, 4, 5).unwrap();
        assert_eq!(t.perm(X, m, 0), Some(Perm::Reserved));
        t.read(X, r, 0, 4, 6).unwrap();
        t.write(X, m, 0, 4, 7).unwrap();
        assert_eq!(t.read(X, r, 0, 4, 8).unwrap_err().class, Class::WriteUnderShared);
    }

    #[test]
    fn borrow_through_dead_tag() {
        let mut t = local(4);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        t.write(X, ROOT, 0, 4, 3).unwrap();
        let e = t.borrow(X, m, Kind::Shared, 0..4, &[], 4).unwrap_err();
        assert_eq!((e.class, e.event, e.tag, e.cause_at), (Class::TwoExclusive, Use::Borrow, m, 3));
        assert_eq!(e.event.rule(), "M4");
    }

    #[test]
    fn disjoint_bytes_never_meet() {
        let mut t = local(12);
        let a = excl(&mut t, ROOT, 0, 8, 2);
        let b = excl(&mut t, ROOT, 4, 12, 3);
        t.write(X, b, 4, 8, 4).unwrap();
        t.read(X, a, 0, 4, 5).unwrap();
        let e = t.read(X, a, 0, 8, 6).unwrap_err();
        assert_eq!((e.class, e.offset, e.cause_at), (Class::TwoExclusive, 4, 4));
        let mut t = local(8);
        let f = excl(&mut t, ROOT, 0, 4, 2);
        let g = shared(&mut t, ROOT, 4, 8, 3);
        t.write(X, f, 0, 4, 4).unwrap();
        t.read(X, g, 4, 8, 5).unwrap();
        t.write(X, f, 0, 4, 6).unwrap();
    }

    #[test]
    fn parent_write_kills_child() {
        let mut t = local(4);
        let p = excl(&mut t, ROOT, 0, 4, 2);
        let c = excl(&mut t, p, 0, 4, 3);
        t.write(X, c, 0, 4, 4).unwrap();
        assert_eq!(t.perm(X, p, 0), Some(Perm::Active));
        t.write(X, p, 0, 4, 5).unwrap();
        let e = t.write(X, c, 0, 4, 6).unwrap_err();
        assert_eq!((e.class, e.tag, e.perm, e.cause_at), (Class::TwoExclusive, c, Perm::Dead, 5));
        assert_eq!(e.event.rule(), "M2");
    }

    #[test]
    fn parent_read_keeps_shared_child() {
        let mut t = local(4);
        let p = excl(&mut t, ROOT, 0, 4, 2);
        let c = shared(&mut t, p, 0, 4, 3);
        t.read(X, p, 0, 4, 4).unwrap();
        t.read(X, c, 0, 4, 5).unwrap();
        assert_eq!(t.perm(X, p, 0), Some(Perm::Reserved));
    }

    #[test]
    fn siblings_of_one_capture() {
        let mut t = local(4);
        let cap = excl(&mut t, ROOT, 0, 4, 2);
        let a = excl(&mut t, cap, 0, 4, 3);
        t.write(X, a, 0, 4, 4).unwrap();
        let b = excl(&mut t, cap, 0, 4, 5);
        t.write(X, b, 0, 4, 6).unwrap();
        let e = t.read(X, a, 0, 4, 7).unwrap_err();
        assert_eq!((e.class, e.tag, e.cause_at), (Class::TwoExclusive, a, 6));
    }

    #[test]
    fn end_of_the_referent() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.end_alloc(X, 3);
        let e = t.read(X, r, 0, 4, 4).unwrap_err();
        assert_eq!((e.class, e.cause, e.cause_at), (Class::UseAfterReferent, Cause::End, 3));
        let e = t.read(X, ROOT, 0, 4, 5).unwrap_err();
        assert_eq!((e.class, e.kind), (Class::UseAfterReferent, Kind::Root));
    }

    #[test]
    fn move_ends_the_borrows_of_the_value() {
        let mut t = local(8);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        let s = shared(&mut t, ROOT, 4, 8, 3);
        t.read(X, ROOT, 0, 4, 4).unwrap();
        t.end(X, ROOT, 0, 4, 4).unwrap();
        t.read(X, s, 4, 8, 5).unwrap();
        let e = t.read(X, r, 0, 4, 6).unwrap_err();
        assert_eq!((e.class, e.cause, e.cause_at), (Class::UseAfterReferent, Cause::End, 4));
        t.write(X, ROOT, 0, 4, 7).unwrap();
    }

    #[test]
    fn cell_bytes_are_shared() {
        let mut t = local(8);
        let c = t.borrow(X, ROOT, Kind::Shared, 0..8, &[(0, 4)], 2).unwrap();
        let d = t.borrow(X, ROOT, Kind::Shared, 0..8, &[(0, 4)], 3).unwrap();
        let inner = shared(&mut t, c, 0, 4, 4);
        t.write(X, c, 0, 4, 5).unwrap();
        t.read(X, d, 0, 8, 6).unwrap();
        t.write(X, d, 0, 4, 7).unwrap();
        let e = t.read(X, inner, 0, 4, 8).unwrap_err();
        assert_eq!((e.class, e.cause_at), (Class::WriteUnderShared, 5));
        let e = t.write(X, c, 4, 8, 9).unwrap_err();
        assert_eq!((e.class, e.perm, e.offset), (Class::WriteUnderShared, Perm::Frozen, 4));
    }

    #[test]
    fn zero_bytes_and_bytes_outside_the_tag() {
        let mut t = local(16);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        let r = shared(&mut t, ROOT, 4, 8, 3);
        t.write(X, ROOT, 0, 4, 4).unwrap();
        t.read(X, m, 0, 0, 5).unwrap();
        t.write(X, m, 2, 2, 6).unwrap();
        t.write(X, r, 8, 12, 7).unwrap();
        t.read(X, r, 4, 8, 8).unwrap();
        let mut t = local(16);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        let r = shared(&mut t, ROOT, 4, 8, 3);
        // Bytes 4..8 are outside `m`: the write goes through the root and kills `r`.
        t.write(X, m, 4, 8, 4).unwrap();
        assert_eq!(t.read(X, r, 4, 8, 5).unwrap_err().cause_at, 4);
        t.write(X, m, 0, 4, 6).unwrap();
    }

    #[test]
    fn frozen_root_of_a_static() {
        let mut t = Tracker::new();
        t.begin(X, 4, true, 1);
        t.read(X, ROOT, 0, 4, 2).unwrap();
        let e = t.write(X, ROOT, 0, 4, 3).unwrap_err();
        assert_eq!((e.class, e.perm), (Class::WriteUnderShared, Perm::Frozen));
    }

    #[test]
    fn unknown_allocation_and_foreign_tag() {
        let mut t = local(4);
        t.begin(2, 4, false, 1);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.read(9, r, 0, 4, 3).unwrap();
        // A tag of another allocation acts as the root of the one accessed.
        t.write(2, r, 0, 4, 4).unwrap();
        t.read(X, r, 0, 4, 5).unwrap();
    }
}
