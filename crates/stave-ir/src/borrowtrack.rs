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
    /// The sets of the tree that list the node (bit `s` for `Tree::sets[s]`).
    listed: u8,
    /// How many nodes have this one as their parent.
    children: u32,
    /// `Tree::round` of the last event the node was on the path of.
    round: u32,
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
}

impl Tree {
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

    /// The permission changes of an allowed event on the bytes `lo..hi` through node `n`: a
    /// child event for `n` and its ancestors, a foreign one for every other tag.
    fn apply(&mut self, n: u32, ev: Use, lo: u64, hi: u64, at: Ev, hits: &mut Vec<u32>) {
        let set = match ev {
            Use::Read => ACT,
            Use::Write => KILL,
            Use::End => LIVE,
            Use::Borrow => return,
        };
        self.round = self.round.wrapping_add(1);
        let mut cur = n;
        while cur != 0 {
            let node = &mut self.nodes[cur as usize];
            node.round = self.round;
            let up = node.parent;
            if ev == Use::Write
                && node.update(
                    lo,
                    hi,
                    Cause::Write,
                    at,
                    |p| if p == Perm::Reserved { Perm::Active } else { p },
                )
            {
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
            if let Some(r) = tree.check(alloc, n, ev, a, b, at) {
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

    /// The allocation ends: every tag of it is `Dead` on every byte, the root included.
    pub fn end_alloc(&mut self, alloc: u32, at: Ev) {
        let Some(tree) = self.trees.get_mut(&alloc) else { return };
        if tree.count == 0 {
            self.trees.remove(&alloc);
            self.ended.insert(alloc, at);
            return;
        }
        for n in tree.nodes.iter_mut().filter(|n| n.tag != FREE) {
            let (lo, hi) = (n.lo, n.hi);
            n.update(lo, hi, Cause::End, at, |_| Perm::Dead);
            n.listed = 0;
        }
        tree.sets = Default::default();
        tree.ended = true;
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
            let gone = |n: &Node| n.tag != FREE && n.children == 0 && !held.contains(&n.tag);
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
            match tree.check(alloc, from, ev, lo, hi, at) {
                Some(r) => Err(r),
                None => {
                    if !tree.ended {
                        tree.apply(from, ev, lo, hi, at, &mut hits);
                    }
                    Ok(())
                }
            }
        } else {
            tree.split(alloc, from, ev, lo..hi, at, &mut hits)
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
        t.end_alloc(X, 3);
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
        t.end_alloc(X, 3);
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
}
