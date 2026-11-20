//! The dynamic borrow model (`contracts/borrow-model.md`) as a state machine: per allocation a
//! tree of tags, per tag one permission per byte, kept as runs. The caller feeds it the events
//! of one execution (allocation begin and end, borrow, read, write, value end) and gets a
//! [`Report`] for the first one a rule forbids.
//!
//! The tracker knows no IR and no addresses: an allocation is the id the caller gives it, bytes
//! are offsets into it, and a site is an opaque [`Ev`] handle the caller resolves when it renders
//! a report.

use std::collections::{BTreeSet, HashMap, HashSet};

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
    /// The rule the event falls under (a protection violation falls under M23,
    /// [`Report::rule`]).
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
    /// The value was moved out of the bytes (M9): the cause is the move.
    Move,
    /// The tag is protected by a call that has not returned (M23): the cause is the call.
    Call,
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

impl Report {
    /// The rule the report is under.
    pub fn rule(&self) -> &'static str {
        match (self.cause, self.class) {
            (Cause::Call, _) => "M23",
            (Cause::Made, Class::SecondOwner) => "M10",
            (_, Class::SecondOwner) => "M9",
            _ => self.event.rule(),
        }
    }
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
    /// The sets of the tree that list the node (bit `s` for `Tree::sets[s]`).
    listed: u8,
    /// How many nodes have this one as their parent.
    children: u32,
    /// `Tree::round` of the last event the node was on the path of.
    round: u32,
    /// How many calls that have not returned protect the tag (M23), and the last of them.
    guards: u32,
    guard_at: Ev,
}

const NO_PARENT: u32 = u32::MAX;

/// The tag of a node slot that holds no node.
const FREE: Tag = Tag::MAX;

/// `Tree::sets`: the nodes with an `Active` byte (a foreign read changes them), with a
/// `Reserved`, `Active`, or `Frozen` byte (a foreign write does), with a byte that is not
/// `Dead` (a foreign end does).
const ACT: usize = 0;
const KILL: usize = 1;
const LIVE: usize = 2;

impl Node {
    fn holds(&self, lo: u64, hi: u64) -> bool {
        self.lo <= lo && hi <= self.hi
    }

    /// The first run among the bytes `lo..hi` whose permission `bad` rejects, with its first
    /// byte inside the range.
    fn find(&self, lo: u64, hi: u64, bad: impl Fn(Perm) -> bool) -> Option<(u64, Run)> {
        let (lo, hi) = (lo.max(self.lo), hi.min(self.hi));
        if lo >= hi {
            return None;
        }
        let k = self.runs.partition_point(|r| r.hi <= lo);
        let mut start = if k == 0 { self.lo } else { self.runs[k - 1].hi };
        for r in &self.runs[k..] {
            if start >= hi {
                break;
            }
            if bad(r.perm) {
                return Some((start.max(lo), *r));
            }
            start = r.hi;
        }
        None
    }

    /// Replaces the permission of each byte in `lo..hi` by `f` of it; a changed byte takes the
    /// cause given. Whether a byte changed.
    fn update(&mut self, lo: u64, hi: u64, cause: Cause, at: Ev, f: impl Fn(Perm) -> Perm) -> bool {
        let (lo, hi) = (lo.max(self.lo), hi.min(self.hi));
        if self.find(lo, hi, |p| f(p) != p).is_none() {
            return false;
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
        true
    }
}

/// The size class of a range of `len` bytes (`len > 0`): the ranges of class `c` are shorter
/// than `2^c` bytes.
fn class(len: u64) -> u8 {
    (64 - len.leading_zeros()) as u8
}

#[derive(Clone, Debug, Default)]
struct Tree {
    size: u64,
    /// Slot 0 is the root. A slot whose tag is [`FREE`] holds no node.
    nodes: Vec<Node>,
    free: Vec<u32>,
    /// The nodes an event can change, as (size class, first byte, slot): [`ACT`], [`KILL`],
    /// [`LIVE`]. The root is in none: every event is a child event for it.
    sets: [BTreeSet<(u8, u64, u32)>; 3],
    /// Nodes but the root: now, and after the last collection of this tree.
    count: u32,
    kept: u32,
    round: u32,
    ended: bool,
    /// The nodes a call protects.
    guarded: Vec<u32>,
    /// The moved-out bytes (M9) as sorted disjoint ranges, each with the move that emptied it.
    moved: Vec<(u64, u64, Ev)>,
}

impl Tree {
    /// The first moved-out byte among `lo..hi`, with the move that emptied it.
    fn moved_in(&self, lo: u64, hi: u64) -> Option<(u64, Ev)> {
        self.moved.iter().find(|m| m.0 < hi && lo < m.1).map(|m| (m.0.max(lo), m.2))
    }

    /// The bytes `lo..hi` hold a value again.
    fn fill(&mut self, lo: u64, hi: u64) {
        if self.moved_in(lo, hi).is_none() {
            return;
        }
        let mut out = Vec::with_capacity(self.moved.len() + 1);
        for &(a, b, at) in &self.moved {
            if b <= lo || hi <= a {
                out.push((a, b, at));
                continue;
            }
            if a < lo {
                out.push((a, lo, at));
            }
            if hi < b {
                out.push((hi, b, at));
            }
        }
        self.moved = out;
    }

    /// The value left the bytes `lo..hi` at `at`.
    fn empty(&mut self, lo: u64, hi: u64, at: Ev) {
        self.fill(lo, hi);
        let k = self.moved.partition_point(|m| m.0 < lo);
        self.moved.insert(k, (lo, hi, at));
    }

    /// M9 for an event the tags allow: the report of a read of a moved-out byte among `lo..hi`;
    /// a write fills the bytes.
    fn second_owner(&mut self, alloc: u32, ev: Use, lo: u64, hi: u64, at: Ev) -> Option<Box<Report>> {
        if self.moved.is_empty() {
            return None;
        }
        match ev {
            Use::Read => {
                let (offset, moved_at) = self.moved_in(lo, hi)?;
                let root = &self.nodes[0];
                Some(Box::new(Report {
                    class: Class::SecondOwner,
                    event: ev,
                    alloc,
                    offset,
                    lo,
                    hi,
                    tag: ROOT,
                    kind: root.kind,
                    perm: Perm::Dead,
                    at,
                    cause: Cause::Move,
                    cause_at: moved_at,
                    made_at: root.made_at,
                }))
            }
            Use::Write => {
                self.fill(lo, hi);
                None
            }
            Use::Borrow | Use::End => None,
        }
    }

    /// Every tag but the root is `Dead` on every byte: the allocation that was in the bytes
    /// ended at `at`.
    fn kill(&mut self, at: Ev) {
        for n in self.nodes.iter_mut().skip(1).filter(|n| n.tag != FREE) {
            let (lo, hi) = (n.lo, n.hi);
            n.update(lo, hi, Cause::End, at, |_| Perm::Dead);
            n.listed = 0;
        }
        self.sets = Default::default();
    }

    /// Puts node `i` in the sets its permissions ask for.
    fn relist(&mut self, i: u32) {
        let n = &mut self.nodes[i as usize];
        if i == 0 || n.lo >= n.hi {
            return;
        }
        let mut want = 0u8;
        for r in &n.runs {
            want |= match r.perm {
                Perm::Active => 0b111,
                Perm::Reserved | Perm::Frozen => 0b110,
                Perm::Cell => 0b100,
                Perm::Dead => 0,
            };
        }
        let diff = want ^ n.listed;
        if diff == 0 {
            return;
        }
        n.listed = want;
        let key = (class(n.hi - n.lo), n.lo, i);
        for s in [ACT, KILL, LIVE] {
            if diff >> s & 1 == 0 {
                continue;
            }
            if want >> s & 1 != 0 {
                self.sets[s].insert(key);
            } else {
                self.sets[s].remove(&key);
            }
        }
    }

    /// The nodes of set `s` with a byte in `lo..hi`.
    fn overlapping(&self, s: usize, lo: u64, hi: u64, out: &mut Vec<u32>) {
        let set = &self.sets[s];
        let mut from = 0u8;
        while let Some(&(c, _, _)) = set.range((from, 0, 0)..).next() {
            // a range of class `c` that holds byte `lo` starts less than `2^c - 1` bytes before
            let reach = if c >= 64 { u64::MAX } else { (1u64 << c) - 1 };
            for &(_, _, i) in set.range((c, lo.saturating_sub(reach - 1), 0)..(c, hi, 0)) {
                if self.nodes[i as usize].hi > lo {
                    out.push(i);
                }
            }
            from = c + 1;
        }
    }

    /// The report of event `ev` on the bytes `lo..hi` through node `n`, if `n` or an ancestor
    /// forbids it.
    fn check(&self, alloc: u32, n: u32, ev: Use, lo: u64, hi: u64, at: Ev) -> Option<Box<Report>> {
        if lo >= hi {
            return None;
        }
        let mut cur = n;
        loop {
            let node = &self.nodes[cur as usize];
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

    /// Begins an event through node `n`: `n` and its ancestors are its path.
    fn enter(&mut self, n: u32) {
        self.round = self.round.wrapping_add(1);
        let mut cur = n;
        while cur != 0 {
            let node = &mut self.nodes[cur as usize];
            node.round = self.round;
            cur = node.parent;
        }
    }

    /// M23: the report of event `ev` on the bytes `lo..hi` through node `n`, if it is foreign
    /// for a protected tag and would make a byte of it `Dead`, or reads a byte a protected
    /// `&mut` may still write.
    fn guard(&mut self, alloc: u32, n: u32, ev: Use, lo: u64, hi: u64, at: Ev) -> Option<Box<Report>> {
        if self.guarded.is_empty() {
            return None;
        }
        self.enter(n);
        for &g in &self.guarded {
            let node = &self.nodes[g as usize];
            if node.round == self.round {
                continue;
            }
            let bad = |p: Perm| match ev {
                Use::Read => node.kind == Kind::Mut && matches!(p, Perm::Reserved | Perm::Active),
                Use::Write => matches!(p, Perm::Reserved | Perm::Active | Perm::Frozen),
                Use::End => p != Perm::Dead,
                Use::Borrow => false,
            };
            if let Some((offset, run)) = node.find(lo, hi, bad) {
                let class = match (ev, node.kind) {
                    (Use::End, _) => Class::UseAfterReferent,
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
                    cause: Cause::Call,
                    cause_at: node.guard_at,
                    made_at: node.made_at,
                }));
            }
        }
        None
    }

    /// The permission changes of an allowed event on the bytes `lo..hi` through node `n`: a
    /// child event for `n` and its ancestors, a foreign one for every other tag.
    fn apply(&mut self, n: u32, ev: Use, lo: u64, hi: u64, at: Ev, hits: &mut Vec<u32>) {
        let set = match ev {
            Use::Read => ACT,
            Use::Write => KILL,
            Use::End => LIVE,
            Use::Borrow => return,
        };
        self.enter(n);
        let mut cur = n;
        while cur != 0 && ev == Use::Write {
            let node = &mut self.nodes[cur as usize];
            let up = node.parent;
            let active = |p| if p == Perm::Reserved { Perm::Active } else { p };
            if node.update(lo, hi, Cause::Write, at, active) {
                self.relist(cur);
            }
            cur = up;
        }
        hits.clear();
        self.overlapping(set, lo, hi, hits);
        for &i in hits.iter() {
            let node = &mut self.nodes[i as usize];
            if node.round == self.round {
                continue;
            }
            let changed = match ev {
                Use::Read => node.update(lo, hi, Cause::Read, at, |p| match p {
                    Perm::Active => Perm::Frozen,
                    p => p,
                }),
                Use::Write => node.update(lo, hi, Cause::Write, at, |p| match p {
                    Perm::Cell => Perm::Cell,
                    _ => Perm::Dead,
                }),
                _ => node.update(lo, hi, Cause::End, at, |_| Perm::Dead),
            };
            if changed {
                self.relist(i);
            }
        }
    }

    /// An event on bytes the tag's range does not hold all of. A byte outside the range is
    /// used through the nearest ancestor that holds it (M20): the range splits into the part
    /// of the tag and the parts of its ancestors.
    fn split(
        &mut self,
        alloc: u32,
        from: u32,
        ev: Use,
        bytes: std::ops::Range<u64>,
        at: Ev,
        hits: &mut Vec<u32>,
    ) -> Result<(), Box<Report>> {
        let tree = self;
        let (lo, hi) = (bytes.start, bytes.end);
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
            let r = tree.check(alloc, n, ev, a, b, at).or_else(|| tree.guard(alloc, n, ev, a, b, at));
            if let Some(r) = r {
                return Err(r);
            }
        }
        if !tree.ended {
            for &(n, a, b) in &pieces {
                tree.apply(n, ev, a, b, at, hits);
            }
        }
        Ok(())
    }
}

/// What [`Tracker::collect`] is to sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Due {
    One(u32),
    All,
}

/// The state of every allocation the caller named.
#[derive(Clone, Debug)]
pub struct Tracker {
    trees: HashMap<u32, Tree>,
    /// The allocations that ended and have no tag left, with the site of the end.
    ended: HashMap<u32, Ev>,
    /// The allocations with a tag besides the root.
    tagged: HashSet<u32>,
    /// Allocation and node of each tag but the roots.
    tags: HashMap<Tag, (u32, u32)>,
    next: Tag,
    /// Nodes but the roots: now, and after the last collection of every tree.
    count: usize,
    kept: usize,
    /// Tags made since the last collection, and the words the caller looked at for it.
    made: usize,
    scanned: usize,
    due: Option<Due>,
    hits: Vec<u32>,
}

impl Default for Tracker {
    fn default() -> Tracker {
        Tracker::new()
    }
}

impl Tracker {
    pub fn new() -> Tracker {
        Tracker {
            trees: HashMap::new(),
            ended: HashMap::new(),
            tagged: HashSet::new(),
            tags: HashMap::new(),
            next: ROOT + 1,
            count: 0,
            kept: 0,
            made: 0,
            scanned: 0,
            due: None,
            hits: Vec::new(),
        }
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
            listed: 0,
            children: 0,
            round: 0,
            guards: 0,
            guard_at: NO_EV,
        };
        self.ended.remove(&alloc);
        self.tagged.remove(&alloc);
        if let Some(old) = self.trees.insert(alloc, Tree { size, nodes: vec![root], ..Tree::default() }) {
            self.count -= old.count as usize;
            for n in old.nodes.iter().skip(1).filter(|n| n.tag != FREE) {
                self.tags.remove(&n.tag);
            }
        }
    }

    /// The node an event on `alloc` through `tag` starts from: the tag's own, or the root for an
    /// untagged word and for a tag of another allocation.
    fn node_of(&self, alloc: u32, tag: Tag) -> u32 {
        if tag == ROOT {
            return 0;
        }
        match self.tags.get(&tag) {
            Some(&(a, n)) if a == alloc => n,
            _ => 0,
        }
    }

    /// Whether allocation `alloc` has begun.
    pub fn has(&self, alloc: u32) -> bool {
        self.trees.contains_key(&alloc) || self.ended.contains_key(&alloc)
    }

    /// The report of an event on an allocation that ended and has no tag left.
    fn after_end(alloc: u32, ev: Use, lo: u64, hi: u64, at: Ev, end: Ev) -> Box<Report> {
        Box::new(Report {
            class: Class::UseAfterReferent,
            event: ev,
            alloc,
            offset: lo,
            lo,
            hi,
            tag: ROOT,
            kind: Kind::Root,
            perm: Perm::Dead,
            at,
            cause: Cause::End,
            cause_at: end,
            made_at: NO_EV,
        })
    }

    /// M4: a new tag of `kind` for `bytes` of `alloc`, reached through `parent`.
    /// A `&` is `Cell` on the byte ranges in `cells`. An allocation the tracker does not know
    /// gives [`ROOT`]; so does a borrow after the last tag is used up ([`Tracker::exhausted`]).
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
        let Some(tree) = self.trees.get_mut(&alloc) else {
            return match self.ended.get(&alloc) {
                Some(&end) if lo < hi => Err(Self::after_end(alloc, Use::Borrow, lo, hi, at, end)),
                _ => Ok(ROOT),
            };
        };
        if self.next == FREE {
            return Ok(ROOT);
        }
        let (lo, hi) = (lo.min(tree.size), hi.min(tree.size).max(lo.min(tree.size)));
        let mut p = from;
        while !tree.nodes[p as usize].holds(lo, hi) && tree.nodes[p as usize].parent != NO_PARENT {
            p = tree.nodes[p as usize].parent;
        }
        if let Some(r) = tree.check(alloc, p, Use::Borrow, lo, hi, at) {
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
            listed: 0,
            children: 0,
            round: 0,
            guards: 0,
            guard_at: NO_EV,
        };
        if kind == Kind::Shared {
            for &(a, b) in cells {
                node.update(a, b, Cause::Made, at, |_| Perm::Cell);
            }
        }
        tree.nodes[p as usize].children += 1;
        let i = match tree.free.pop() {
            Some(i) => {
                tree.nodes[i as usize] = node;
                i
            }
            None => {
                tree.nodes.push(node);
                tree.nodes.len() as u32 - 1
            }
        };
        if !tree.ended {
            tree.relist(i);
        }
        tree.count += 1;
        self.count += 1;
        self.made += 1;
        if tree.count == 1 {
            self.tagged.insert(alloc);
        }
        // A collection costs the caller a look at every word that may hold a tag, and the
        // tracker a look at every node: both are paid for by the tags made since the last one.
        if self.count >= 2 * self.kept + 1024 {
            self.due = Some(Due::All);
        } else if self.due.is_none()
            && tree.count >= 2 * tree.kept + 32
            && self.made.saturating_mul(4) >= self.scanned
        {
            self.due = Some(Due::One(alloc));
        }
        self.tags.insert(self.next, (alloc, i));
        self.next += 1;
        Ok(self.next - 1)
    }

    /// Whether every tag is used up: a later borrow makes no tag.
    pub fn exhausted(&self) -> bool {
        self.next == FREE
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

    /// The allocation ends: every tag of it is `Dead` on every byte, the root included. An end
    /// under a protected tag that is not `Dead` yet is a violation (M23); the allocation ends
    /// all the same.
    pub fn end_alloc(&mut self, alloc: u32, at: Ev) -> Result<(), Box<Report>> {
        let Some(tree) = self.trees.get_mut(&alloc) else { return Ok(()) };
        if tree.count == 0 {
            self.trees.remove(&alloc);
            self.ended.insert(alloc, at);
            return Ok(());
        }
        let size = tree.size;
        let report = if tree.ended { None } else { tree.guard(alloc, 0, Use::End, 0, size, at) };
        tree.kill(at);
        tree.nodes[0].update(0, size, Cause::End, at, |_| Perm::Dead);
        tree.moved = Vec::new();
        tree.ended = true;
        report.map_or(Ok(()), Err)
    }

    /// A local or temporary begins in the bytes of `alloc` (M13, M14): the allocation that was
    /// there ends with every tag of it, and the root stands for the new one. An end under a
    /// protected tag that is not `Dead` yet is a violation (M23). An allocation no event named
    /// yet has nothing to end.
    pub fn restart(&mut self, alloc: u32, at: Ev) -> Result<(), Box<Report>> {
        if self.ended.remove(&alloc).is_some() {
            return Ok(());
        }
        let Some(tree) = self.trees.get_mut(&alloc) else { return Ok(()) };
        let size = tree.size;
        let report = if tree.ended || tree.count == 0 {
            None
        } else {
            let r = tree.guard(alloc, 0, Use::End, 0, size, at);
            tree.kill(at);
            r
        };
        tree.nodes[0].runs = vec![Run { hi: size, perm: Perm::Active, cause: Cause::Made, cause_at: at }];
        tree.moved = Vec::new();
        tree.ended = false;
        report.map_or(Ok(()), Err)
    }

    /// M9: the value in the bytes `lo..hi` of `alloc` leaves through `tag`: the bytes end (M3)
    /// and are moved-out until the next write.
    pub fn gone(&mut self, alloc: u32, tag: Tag, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        self.event(alloc, tag, Use::End, lo, hi, at)?;
        if let Some(tree) = self.trees.get_mut(&alloc)
            && !tree.ended
        {
            let hi = hi.min(tree.size);
            if lo < hi {
                tree.empty(lo, hi, at);
            }
        }
        Ok(())
    }

    /// The moved-out bytes of `alloc`.
    pub fn moved(&self, alloc: u32) -> Vec<(u64, u64, Ev)> {
        self.trees.get(&alloc).map_or(Vec::new(), |t| t.moved.clone())
    }

    /// Bytes of `alloc` that are moved-out from its begin on (a block that took the bytes of
    /// another, with their initialization).
    pub fn set_moved(&mut self, alloc: u32, moved: &[(u64, u64, Ev)]) {
        if let Some(tree) = self.trees.get_mut(&alloc) {
            for &(lo, hi, at) in moved {
                let hi = hi.min(tree.size);
                if lo < hi {
                    tree.empty(lo, hi, at);
                }
            }
        }
    }

    /// M10: the report of an array repeat literal at `at` that makes a second owner of a value
    /// that is not Copy.
    pub fn repeat(at: Ev) -> Box<Report> {
        Box::new(Report {
            class: Class::SecondOwner,
            event: Use::Read,
            alloc: 0,
            offset: 0,
            lo: 0,
            hi: 0,
            tag: ROOT,
            kind: Kind::Root,
            perm: Perm::Dead,
            at,
            cause: Cause::Made,
            cause_at: at,
            made_at: NO_EV,
        })
    }

    /// M23: a call protects `tag` from `at` until [`Tracker::unprotect`]. No effect on a root.
    pub fn protect(&mut self, tag: Tag, at: Ev) {
        let Some(&(a, i)) = self.tags.get(&tag) else { return };
        let Some(tree) = self.trees.get_mut(&a) else { return };
        let n = &mut tree.nodes[i as usize];
        n.guards += 1;
        n.guard_at = at;
        if n.guards == 1 {
            tree.guarded.push(i);
        }
    }

    /// No call protects a tag any more.
    pub fn unprotect_all(&mut self) {
        for tree in self.trees.values_mut() {
            for g in std::mem::take(&mut tree.guarded) {
                tree.nodes[g as usize].guards = 0;
            }
        }
    }

    /// The call that protected `tag` returned.
    pub fn unprotect(&mut self, tag: Tag) {
        let Some(&(a, i)) = self.tags.get(&tag) else { return };
        let Some(tree) = self.trees.get_mut(&a) else { return };
        let n = &mut tree.nodes[i as usize];
        n.guards = n.guards.saturating_sub(1);
        if n.guards == 0 {
            tree.guarded.retain(|&g| g != i);
        }
    }

    /// The permission of `tag` at byte `offset` of `alloc`, if the byte is in its range.
    pub fn perm(&self, alloc: u32, tag: Tag, offset: u64) -> Option<Perm> {
        let n = &self.trees.get(&alloc)?.nodes[self.node_of(alloc, tag) as usize];
        n.find(offset, offset + 1, |_| true).map(|(_, r)| r.perm)
    }

    /// Tags but the roots that the tracker holds a node for.
    pub fn tags(&self) -> usize {
        self.count
    }

    /// Whether [`Tracker::collect`] is worth its cost now.
    pub fn due(&self) -> bool {
        self.due.is_some()
    }

    /// Frees the tags that no word carries and no kept tag descends from: no later event can
    /// name them, and no other tag's permission depends on them. `held` holds every tag a word
    /// of the execution carries; `scanned` is how many words the caller looked at to build it.
    pub fn collect(&mut self, held: &HashSet<Tag>, scanned: usize) {
        let all = !matches!(self.due, Some(Due::One(_)));
        let allocs: Vec<u32> = match self.due.take() {
            Some(Due::One(a)) => vec![a],
            _ => self.tagged.iter().copied().collect(),
        };
        for a in allocs {
            let Some(tree) = self.trees.get_mut(&a) else { continue };
            let gone = |n: &Node| n.tag != FREE && n.children == 0 && n.guards == 0 && !held.contains(&n.tag);
            let mut work: Vec<u32> =
                (1..tree.nodes.len() as u32).filter(|&i| gone(&tree.nodes[i as usize])).collect();
            while let Some(i) = work.pop() {
                let n = &mut tree.nodes[i as usize];
                let (key, listed, up) = ((class((n.hi - n.lo).max(1)), n.lo, i), n.listed, n.parent);
                self.tags.remove(&n.tag);
                n.tag = FREE;
                n.runs = Vec::new();
                n.listed = 0;
                for s in [ACT, KILL, LIVE] {
                    if listed >> s & 1 != 0 {
                        tree.sets[s].remove(&key);
                    }
                }
                tree.free.push(i);
                tree.count -= 1;
                self.count -= 1;
                let p = &mut tree.nodes[up as usize];
                p.children -= 1;
                if up != 0 && gone(p) {
                    work.push(up);
                }
            }
            tree.kept = tree.count;
            if tree.count == 0 {
                self.tagged.remove(&a);
                tree.nodes.truncate(1);
                tree.free = Vec::new();
                if tree.ended {
                    let end = tree.nodes[0].runs.first().map_or(NO_EV, |r| r.cause_at);
                    self.trees.remove(&a);
                    self.ended.insert(a, end);
                }
            }
        }
        if all {
            self.kept = self.count;
        }
        self.made = 0;
        self.scanned = scanned;
    }

    fn event(&mut self, alloc: u32, tag: Tag, ev: Use, lo: u64, hi: u64, at: Ev) -> Result<(), Box<Report>> {
        let from = self.node_of(alloc, tag);
        let Some(tree) = self.trees.get_mut(&alloc) else {
            return match self.ended.get(&alloc) {
                Some(&end) if lo < hi => Err(Self::after_end(alloc, ev, lo, hi, at, end)),
                _ => Ok(()),
            };
        };
        let hi = hi.min(tree.size);
        if lo >= hi {
            return Ok(());
        }
        let mut hits = std::mem::take(&mut self.hits);
        let r = if tree.nodes[from as usize].holds(lo, hi) {
            let r = tree.check(alloc, from, ev, lo, hi, at);
            let r = r.or_else(|| tree.second_owner(alloc, ev, lo, hi, at));
            match r.or_else(|| tree.guard(alloc, from, ev, lo, hi, at)) {
                Some(r) => Err(r),
                None => {
                    if !tree.ended {
                        tree.apply(from, ev, lo, hi, at, &mut hits);
                    }
                    Ok(())
                }
            }
        } else {
            match tree.second_owner(alloc, ev, lo, hi, at) {
                Some(r) => Err(r),
                None => tree.split(alloc, from, ev, lo..hi, at, &mut hits),
            }
        };
        self.hits = hits;
        r
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
        t.end_alloc(X, 3).unwrap();
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
    fn a_dead_tag_keeps_its_cause() {
        let mut t = local(4);
        let first = excl(&mut t, ROOT, 0, 4, 2);
        t.write(X, first, 0, 4, 3).unwrap();
        let mut last = first;
        for k in 0..100u64 {
            last = excl(&mut t, ROOT, 0, 4, 10 + 2 * k);
            t.write(X, last, 0, 4, 11 + 2 * k).unwrap();
        }
        let e = t.read(X, first, 0, 4, 500).unwrap_err();
        assert_eq!((e.class, e.cause_at), (Class::TwoExclusive, 11));
        t.read(X, last, 0, 4, 501).unwrap();
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

    #[test]
    fn collection_frees_the_tags_no_word_holds() {
        let mut t = local(8);
        let a = excl(&mut t, ROOT, 0, 8, 2);
        let b = shared(&mut t, a, 0, 4, 3);
        let c = shared(&mut t, a, 4, 8, 4);
        let d = shared(&mut t, ROOT, 0, 8, 5);
        assert_eq!(t.tags(), 4);
        // `b` keeps its parent `a`; `c` and `d` go
        t.collect(&HashSet::from([b]), 0);
        assert_eq!(t.tags(), 2);
        assert_eq!(t.perm(X, a, 0), Some(Perm::Reserved));
        assert_eq!(t.perm(X, b, 0), Some(Perm::Frozen));
        // a freed tag names the root from then on
        assert_eq!(t.perm(X, c, 4), Some(Perm::Active));
        assert_eq!(t.perm(X, d, 0), Some(Perm::Active));
        // the kept tags follow the rules as before: a write through the root kills both
        t.write(X, ROOT, 0, 1, 6).unwrap();
        let e = t.read(X, b, 0, 4, 7).unwrap_err();
        assert_eq!((e.tag, e.cause, e.cause_at, e.made_at), (b, Cause::Write, 6, 3));
        // a freed slot is used again
        let f = shared(&mut t, ROOT, 4, 8, 8);
        assert_eq!(t.tags(), 3);
        t.read(X, f, 4, 8, 9).unwrap();
        t.collect(&HashSet::new(), 0);
        assert_eq!(t.tags(), 0);
    }

    #[test]
    fn an_ended_allocation_reports_after_its_tags_are_freed() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.end_alloc(X, 3).unwrap();
        let e = t.read(X, r, 0, 4, 4).unwrap_err();
        assert_eq!((e.class, e.tag, e.cause_at, e.made_at), (Class::UseAfterReferent, r, 3, 2));
        t.collect(&HashSet::new(), 0);
        assert_eq!(t.tags(), 0);
        assert!(t.has(X));
        let e = t.read(X, ROOT, 0, 4, 5).unwrap_err();
        assert_eq!(
            (e.class, e.kind, e.cause, e.cause_at),
            (Class::UseAfterReferent, Kind::Root, Cause::End, 3)
        );
        let e = t.borrow(X, ROOT, Kind::Shared, 0..4, &[], 6).unwrap_err();
        assert_eq!((e.class, e.event), (Class::UseAfterReferent, Use::Borrow));
        // an allocation that ended without a tag
        let mut t = local(4);
        t.end_alloc(X, 3).unwrap();
        assert_eq!(t.write(X, ROOT, 0, 4, 4).unwrap_err().cause_at, 3);
    }

    #[test]
    fn a_tag_made_in_each_round_asks_for_a_collection() {
        let mut t = local(4);
        let mut rounds = 0;
        while !t.due() {
            let r = shared(&mut t, ROOT, 0, 4, 2);
            t.read(X, r, 0, 4, 3).unwrap();
            rounds += 1;
            assert!(rounds < 100);
        }
        t.collect(&HashSet::new(), 0);
        assert!(!t.due());
        assert_eq!(t.tags(), 0);
    }

    #[test]
    fn events_meet_only_the_tags_on_their_bytes() {
        // one `&mut` per element, each written through: an event on one element changes the
        // tags of that element alone, whatever their size class
        let mut t = local(4000);
        let whole = shared(&mut t, ROOT, 0, 4000, 2);
        let each: Vec<Tag> = (0..1000u64).map(|i| excl(&mut t, ROOT, 4 * i, 4 * i + 4, 3)).collect();
        t.write(X, each[7], 28, 32, 4).unwrap();
        assert_eq!(t.perm(X, each[7], 28), Some(Perm::Active));
        assert_eq!(t.perm(X, whole, 28), Some(Perm::Dead));
        assert_eq!(t.perm(X, whole, 32), Some(Perm::Frozen));
        assert_eq!(t.perm(X, each[6], 24), Some(Perm::Reserved));
        assert_eq!(t.perm(X, each[8], 32), Some(Perm::Reserved));
        t.read(X, whole, 32, 36, 5).unwrap();
        t.read(X, ROOT, 26, 30, 6).unwrap();
        assert_eq!(t.perm(X, each[6], 26), Some(Perm::Reserved));
        assert_eq!(t.perm(X, each[7], 28), Some(Perm::Frozen));
        assert_eq!(t.perm(X, each[7], 30), Some(Perm::Active));
        t.end(X, ROOT, 0, 4000, 7).unwrap();
        assert!(each.iter().all(|&e| t.perm(X, e, 0).is_none_or(|p| p == Perm::Dead)));
        assert_eq!(t.read(X, each[999], 3996, 4000, 8).unwrap_err().class, Class::UseAfterReferent);
    }

    #[test]
    fn a_protected_tag_may_not_die() {
        // a `&` param: a foreign write under it is a violation at the write
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.protect(r, 3);
        let e = t.write(X, ROOT, 0, 4, 4).unwrap_err();
        assert_eq!(
            (e.class, e.tag, e.cause, e.cause_at, e.rule()),
            (Class::WriteUnderShared, r, Cause::Call, 3, "M23")
        );
        // after the return the write is allowed, and the tag is Dead
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.protect(r, 3);
        t.read(X, ROOT, 0, 4, 4).unwrap();
        t.unprotect(r);
        t.write(X, ROOT, 0, 4, 5).unwrap();
        assert_eq!(t.perm(X, r, 0), Some(Perm::Dead));
    }

    #[test]
    fn a_protected_exclusive_tag_is_not_read_by_others() {
        let mut t = local(4);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        t.protect(m, 3);
        // its own uses and those of its children are child accesses
        let c = shared(&mut t, m, 0, 4, 4);
        t.read(X, c, 0, 4, 5).unwrap();
        t.write(X, m, 0, 4, 6).unwrap();
        let e = t.read(X, ROOT, 0, 4, 7).unwrap_err();
        assert_eq!(
            (e.class, e.event, e.tag, e.cause_at, e.rule()),
            (Class::TwoExclusive, Use::Read, m, 3, "M23")
        );
        // a `Reserved` one too
        let mut t = local(8);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        t.protect(m, 3);
        t.read(X, ROOT, 4, 8, 4).unwrap();
        assert_eq!(t.read(X, ROOT, 0, 4, 5).unwrap_err().cause, Cause::Call);
    }

    #[test]
    fn the_end_of_an_allocation_under_a_protected_tag() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.protect(r, 3);
        let e = t.end_alloc(X, 4).unwrap_err();
        assert_eq!(
            (e.class, e.event, e.tag, e.cause, e.cause_at),
            (Class::UseAfterReferent, Use::End, r, Cause::Call, 3)
        );
        // a `Cell` tag is not made Dead by a write, but by an end
        let mut t = local(4);
        let c = t.borrow(X, ROOT, Kind::Shared, 0..4, &[(0, 4)], 2).unwrap();
        t.protect(c, 3);
        t.write(X, ROOT, 0, 4, 4).unwrap();
        assert_eq!(t.end(X, ROOT, 0, 4, 5).unwrap_err().class, Class::UseAfterReferent);
        // a protected tag is kept by a collection
        t.collect(&HashSet::new(), 0);
        assert_eq!(t.tags(), 1);
    }

    #[test]
    fn a_local_that_begins_again_ends_the_tags_of_the_one_before() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.restart(X, 3).unwrap();
        // the root is the new local's
        t.write(X, ROOT, 0, 4, 4).unwrap();
        t.read(X, ROOT, 0, 4, 5).unwrap();
        let e = t.read(X, r, 0, 4, 6).unwrap_err();
        assert_eq!((e.class, e.cause, e.cause_at, e.made_at), (Class::UseAfterReferent, Cause::End, 3, 2));
        assert_eq!(e.rule(), "M1");
        // a borrow of the new local is a tag of its own
        let s = shared(&mut t, ROOT, 0, 4, 7);
        t.read(X, s, 0, 4, 8).unwrap();
    }

    #[test]
    fn a_local_between_its_end_and_its_next_begin() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.end_alloc(X, 3).unwrap();
        let e = t.read(X, ROOT, 0, 4, 4).unwrap_err();
        assert_eq!((e.class, e.cause_at), (Class::UseAfterReferent, 3));
        t.restart(X, 5).unwrap();
        t.write(X, ROOT, 0, 4, 6).unwrap();
        // the tag died at the end, not at the begin after it
        assert_eq!(t.read(X, r, 0, 4, 7).unwrap_err().cause_at, 3);
        // without a tag the ended local is one map entry, and begins again all the same
        t.collect(&HashSet::new(), 0);
        t.end_alloc(X, 8).unwrap();
        t.collect(&HashSet::new(), 0);
        assert!(t.has(X));
        assert_eq!(t.read(X, ROOT, 0, 4, 9).unwrap_err().cause_at, 8);
        t.restart(X, 10).unwrap();
        assert!(!t.has(X));
    }

    #[test]
    fn a_begin_under_a_protected_tag() {
        let mut t = local(4);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        t.protect(r, 3);
        let e = t.restart(X, 4).unwrap_err();
        assert_eq!(
            (e.class, e.event, e.cause, e.rule()),
            (Class::UseAfterReferent, Use::End, Cause::Call, "M23")
        );
    }

    #[test]
    fn a_move_ends_the_bytes_and_empties_them() {
        let mut t = local(8);
        let r = shared(&mut t, ROOT, 0, 4, 2);
        let other = shared(&mut t, ROOT, 4, 8, 3);
        t.gone(X, ROOT, 0, 4, 4).unwrap();
        // a reference into the moved value: its referent ended
        let e = t.read(X, r, 0, 4, 5).unwrap_err();
        assert_eq!((e.class, e.cause, e.cause_at), (Class::UseAfterReferent, Cause::End, 4));
        // the place itself: a second owner
        let e = t.read(X, ROOT, 0, 8, 6).unwrap_err();
        assert_eq!(
            (e.class, e.event, e.offset, e.cause_at, e.rule()),
            (Class::SecondOwner, Use::Read, 0, 4, "M9")
        );
        // the bytes beside it are untouched
        t.read(X, other, 4, 8, 7).unwrap();
        t.read(X, ROOT, 4, 8, 8).unwrap();
        // a write fills the place again; the old reference stays dead, by the end
        t.write(X, ROOT, 0, 4, 9).unwrap();
        t.read(X, ROOT, 0, 8, 10).unwrap();
        let e = t.read(X, r, 0, 4, 11).unwrap_err();
        assert_eq!((e.class, e.cause, e.cause_at), (Class::UseAfterReferent, Cause::End, 4));
    }

    #[test]
    fn moved_bytes_are_kept_as_ranges() {
        let mut t = local(16);
        t.gone(X, ROOT, 0, 16, 2).unwrap();
        t.write(X, ROOT, 4, 8, 3).unwrap();
        assert_eq!(t.moved(X), vec![(0, 4, 2), (8, 16, 2)]);
        t.gone(X, ROOT, 2, 10, 4).unwrap();
        assert_eq!(t.moved(X), vec![(0, 2, 2), (2, 10, 4), (10, 16, 2)]);
        t.read(X, ROOT, 0, 0, 5).unwrap();
        assert_eq!(t.read(X, ROOT, 9, 12, 6).unwrap_err().cause_at, 4);
        let mut u = Tracker::new();
        u.begin(2, 8, false, 1);
        u.set_moved(2, &t.moved(X));
        assert_eq!(u.moved(2), vec![(0, 2, 2), (2, 8, 4)]);
        // a begin and an end leave no moved byte
        t.restart(X, 7).unwrap();
        assert!(t.moved(X).is_empty());
    }

    #[test]
    fn a_move_through_a_dead_tag() {
        let mut t = local(4);
        let m = excl(&mut t, ROOT, 0, 4, 2);
        t.write(X, ROOT, 0, 4, 3).unwrap();
        let e = t.gone(X, m, 0, 4, 4).unwrap_err();
        assert_eq!((e.class, e.event, e.cause_at), (Class::TwoExclusive, Use::End, 3));
        assert!(t.moved(X).is_empty());
    }

    #[test]
    fn an_array_repeat_of_a_value_that_is_not_copy() {
        let e = Tracker::repeat(7);
        assert_eq!((e.class, e.at, e.cause_at, e.rule()), (Class::SecondOwner, 7, 7, "M10"));
    }
}
