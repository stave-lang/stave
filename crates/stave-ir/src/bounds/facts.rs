//! The difference-constraint graph behind the `LenEq` and `Offset` proofs and the contract-check
//! goals.
//!
//! Values are ints, each read as the mathematical integer it is under its own type (unsigned
//! plain, signed two's complement), so a fact between a signed and an unsigned value is an
//! ordinary inequality; a source is read only in the signedness its op computes in (a compare's
//! or an `add`'s operands share one int type; bit ops, `sub.sat`, and lengths are unsigned). A
//! term is `node + k`: `k` a constant, `node` the constant zero, an SSA value, or the length of an
//! SSA slice value. Trapping `add x c` / `sub x c` (constant `c`) are folded into the term of `x`:
//! a defined result is exactly `x ± c` (overflow traps), and the check's own trap stays in the
//! code, so no wrap reasoning is needed. `slice.len (slice.make p n)` is the term of `n`. A
//! `cast`, `cast.wrap`, or `cast.sat` of an int value to an int type that holds every value of
//! the source type (same signedness and at least as wide, or unsigned to a wider signed type) is
//! the term of its operand: such a cast keeps the value under all three. A `load` of an int field
//! behind a `&` parameter (T not `UnsafeCell`), or a call of a getter of one, is the field's node:
//! it holds one value while the function runs ([`Node::Field`]), so the `len` of a `&Soa` that
//! `len()` returns and every column reads is one node. Wrapping arithmetic, other casts, other
//! loads, and everything else are opaque nodes.
//!
//! A fact `x <= y + w` is an edge `x → y` of weight `w`, valid in a region of the function:
//!
//! - a branch: `cbr c T F` where `T` (or `F`) is the target's only predecessor edge; `c` (or `!c`)
//!   holds in every block that target dominates. `c` is an int `lt le gt ge eq ne`, or `not`
//!   of one. A strict unsigned comparison with `n -| c` (`sub.sat`, constant `c`) also gives
//!   `x + c < n` (`x < n -| c`) or `x + c >= n` (its negation): `n -| c` is `max(n - c, 0)`.
//!   A contract check's branch is such a branch: its true side is the only way past it.
//! - a check, after it: `check.bounds x (n -| c)` gives `x + c < n`, as the strict comparison
//!   with `n -| c` above (a check on another length is read by the `Dominated` proof alone).
//! - an asserting call: `call @f .. c ..` where every `ret` of `f` is reached only through a branch
//!   on that parameter being true (`std.debug.assert`); `c` holds after the call.
//! - a bit op on unsigned values, after it: `v = and x y` gives `v <= x` and `v <= y`;
//!   `v = rem x y` gives `v <= x` and `v <= y - 1` (`y = 0` traps, so a result means `y >= 1`);
//!   `v = shr x k` (logical on unsigned) gives `v <= x`. Only when every constant operand read
//!   is a value of the op's type (a constant divisor: and not 0).
//! - the type of an int value `v` of `bits` bits, where `v` is defined: unsigned, `v <= 2^bits - 1`
//!   for `bits < 64` (a wider bound is no bound on any length) and `0 <= v`; signed,
//!   `v <= 2^(bits-1) - 1` and `-2^(bits-1) <= v`. A lower bound only for each value node another
//!   fact leaves (only such a node can be a step between two other facts).
//!
//! Every fact is about the current values of its operands wherever its region dominates the query
//! (an SSA value redefined in a later loop iteration passes the region's head again). A query
//! `i < n` searches for a path `i → .. → n` of total weight `≤ n.k - i.k - 1`, from `i`, on
//! demand, over facts whose region dominates the point (right before a `check.bounds`, or the end
//! of the block branching on a contract check's condition), and gives up after
//! [`BOUNDS_STEP_LIMIT`] edges. Any path found is a chain of true inequalities, so the proof is
//! sound however the search ends; giving up only keeps a check.
//!
//! The facts are one table sorted by their start node. Every fact keeps the place it was read
//! from and its two operands as written, so a path found can be written out as a witness
//! ([`crate::witness`]): the folding steps from an operand to its term ([`Cx::fold`]) become
//! edges of their own.

use std::cell::{OnceCell, RefCell};
use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{BOUNDS_STEP_LIMIT, Cx, Proof, preds_of};
use crate::ir::{Block, Body, CastKind, Func, FuncId, Op, Operand, Term as IrTerm, Value, ValueDef};
use crate::types::{TypeId, TypeKey};
use crate::verify::Dominators;
use crate::witness::{
    ASSERT_BLOCKS, Edge, MAX_CONST, NO_BLOCK, NOT_DEPTH, Node, Row, Source, Summary, Term as RowTerm,
};

/// Folding depth of `add`/`sub` chains, and of `slice.len (slice.make ..)` nesting.
const DEPTH: u32 = 8;
/// The most nodes a search for contradictory facts starts from.
const CYCLE_STARTS: usize = 16;
/// Constants and offsets beyond this are opaque (keeps every weight sum far from `i128` overflow).
/// Every `u64` constant is below it, so a constant index or length is never opaque.
const MAX_K: i128 = MAX_CONST;

pub(super) const ZERO: u32 = 0;

/// Parameters and fields past this have no [`Node::Field`] term node (their loads stay opaque).
const FIELD_SPAN: u32 = 1 << 12;

/// The term node of a witness node (`None`: past the numbering). With [`node_of`] the only code
/// that knows how nodes are numbered: everything else goes through the two.
pub(super) fn node_id(n: Node) -> Option<u32> {
    match n {
        Node::Zero => Some(ZERO),
        Node::Val(v) => v.0.checked_mul(3)?.checked_add(1),
        Node::Len(s) => s.0.checked_mul(3)?.checked_add(2),
        Node::Field(p, k) => (p.0 < FIELD_SPAN && k < FIELD_SPAN).then(|| 3 + 3 * (p.0 * FIELD_SPAN + k)),
    }
}

/// The witness node of a term node: the inverse of [`node_id`].
pub(super) fn node_of(node: u32) -> Node {
    match node {
        ZERO => Node::Zero,
        n if n % 3 == 1 => Node::Val(Value((n - 1) / 3)),
        n if n % 3 == 2 => Node::Len(Value((n - 2) / 3)),
        n => {
            let f = (n - 3) / 3;
            Node::Field(Value(f / FIELD_SPAN), f % FIELD_SPAN)
        }
    }
}

// A body has far fewer values than a third of `u32::MAX`.
pub(super) fn val_node(v: Value) -> u32 {
    node_id(Node::Val(v)).expect("value index within the node numbering")
}

pub(super) fn len_node(s: Value) -> u32 {
    node_id(Node::Len(s)).expect("value index within the node numbering")
}

/// One side of an inequality as written: the constant zero, an int operand, or the length of a
/// slice value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    Zero,
    Val(Operand),
    Len(Value),
}

/// `node + k`; `root` is the value the term was read from after folding (`None`: a constant).
#[derive(Clone, Copy, Debug)]
pub(super) struct Term {
    node: u32,
    k: i128,
    root: Option<Value>,
}

impl Term {
    pub(super) fn constant(c: u64) -> Term {
        Term { node: ZERO, k: i128::from(c), root: None }
    }

    /// The term's node and offset.
    pub(super) fn parts(self) -> (u32, i128) {
        (self.node, self.k)
    }

    /// `self + c`.
    pub(super) fn plus(self, c: i128) -> Option<Term> {
        let k = self.k.checked_add(c).filter(|k| k.abs() <= MAX_K)?;
        Some(Term { k, ..self })
    }
}

/// Where a fact holds: block `block` and every block it dominates, from instruction `after`
/// on (exclusive) when `after` is set.
#[derive(Clone, Copy)]
struct Region {
    block: u32,
    after: Option<u32>,
}

/// `from <= to + w` between two term nodes.
#[derive(Clone, Copy)]
struct Fact {
    from: u32,
    to: u32,
    w: i128,
    at: Region,
    /// The `eq`/`ne` instruction when the fact is one half of a proved equality.
    eq: Option<u32>,
    /// Where the fact was read, and what it says there: `a <= b + wop` between the two sides as
    /// written, before folding.
    src: Source,
    a: Side,
    b: Side,
    wop: i128,
}

/// The facts of a function, sorted by `from` (facts of one node in the order they were found).
#[derive(Default)]
pub(super) struct Graph {
    facts: Vec<Fact>,
}

impl Graph {
    /// The facts starting at `node`, as a range of the table.
    fn from(&self, node: u32) -> std::ops::Range<usize> {
        let lo = self.facts.partition_point(|f| f.from < node);
        let hi = lo + self.facts[lo..].partition_point(|f| f.from == node);
        lo..hi
    }
}

/// One improvement of a node's distance during a query: reached over `fact` from the step
/// `parent` (`u32::MAX`: the start).
#[derive(Clone, Copy)]
struct Step {
    parent: u32,
    fact: u32,
}

/// The lazily built per-function graph and what it needed of the callees.
#[derive(Default)]
pub(super) struct Facts {
    graph: OnceCell<Graph>,
    asserted: RefCell<FxHashMap<FuncId, u64>>,
    /// The summaries of callees computed here (no table of them was given).
    pub(super) claims: RefCell<FxHashMap<FuncId, Option<Arc<Summary>>>>,
}

impl Cx<'_> {
    fn unsigned(&self, v: Value) -> bool {
        matches!(self.int_ty(v), Some((false, _)))
    }

    /// `(signed, bits)` of `v`'s type when it is an int.
    pub(super) fn int_ty(&self, v: Value) -> Option<(bool, u8)> {
        match self.b.value_ty(v).and_then(|t| self.key(t)) {
            Some(TypeKey::Int { signed, bits }) => Some((*signed, bits.width())),
            _ => None,
        }
    }

    /// A cast from `x`'s type to `v`'s keeps every value: the target holds the source's range.
    fn widens(&self, x: Value, v: Value) -> bool {
        match (self.int_ty(x), self.int_ty(v)) {
            (Some((sx, wx)), Some((sv, wv))) => (sx == sv && wx <= wv) || (!sx && sv && wx < wv),
            _ => false,
        }
    }

    pub(super) fn small_const(&self, o: Operand) -> Option<i128> {
        self.b.int_const(o).filter(|c| (-MAX_K..=MAX_K).contains(c))
    }

    /// The width in bits of `v`'s type when it is an unsigned int.
    fn width(&self, v: Value) -> Option<u8> {
        match self.b.value_ty(v).and_then(|t| self.key(t)) {
            Some(TypeKey::Int { signed: false, bits }) => Some(bits.width()),
            _ => None,
        }
    }

    /// Where `v` has its value: after its instruction, or in the block it is a parameter of
    /// (`None`: unreachable).
    fn defined_in(&self, v: Value) -> Option<Region> {
        match self.b.values.get(v.index())?.def {
            ValueDef::Inst(i) => {
                self.reachable_inst(i).then(|| Region { block: self.block_of[i as usize], after: Some(i) })
            }
            ValueDef::Param { block, .. } => {
                self.dom.reachable(block.index()).then_some(Region { block: block.0, after: None })
            }
        }
    }

    /// The term of an int operand.
    pub(super) fn term(&self, o: Operand) -> Option<Term> {
        self.fold(o, &mut |_| {})
    }

    /// The node and offset of an operand as written: a constant is `zero + c`, a value itself.
    pub(super) fn written(&self, o: Operand) -> Option<(Node, i128)> {
        match o {
            Operand::Value(v) if self.int_ty(v).is_some() => Some((Node::Val(v), 0)),
            _ => self.small_const(o).map(|c| (Node::Zero, c)),
        }
    }

    /// [`Cx::term`], handing `sink` every folding step as the equality it uses, from the operand
    /// as written down to the term's node.
    pub(super) fn fold(&self, o: Operand, sink: &mut impl FnMut(Edge)) -> Option<Term> {
        self.fold_to(o, None, sink)
    }

    /// [`Cx::fold`] that reads value `stop` as a node of its own: nothing is folded through its
    /// definition, nor through a `slice.make` that defines it.
    pub(super) fn fold_to(
        &self,
        o: Operand,
        stop: Option<Value>,
        sink: &mut impl FnMut(Edge),
    ) -> Option<Term> {
        let mut o = o;
        let mut k: i128 = 0;
        let mut nested = 0;
        'term: loop {
            for _ in 0..DEPTH {
                if let Some(c) = self.small_const(o) {
                    return Some(Term { node: ZERO, k: k.checked_add(c)?, root: None });
                }
                let Operand::Value(v) = o else { return None };
                self.int_ty(v)?;
                let opaque = Term { node: val_node(v), k, root: Some(v) };
                if stop == Some(v) {
                    return Some(opaque);
                }
                let Some(d) = self.def(v) else { return Some(opaque) };
                if let Some((p, field, src)) = self.stable_field(d) {
                    let Some(node) = node_id(Node::Field(p, field)) else { return Some(opaque) };
                    sink(Edge { from: Node::Val(v), to: Node::Field(p, field), w: 0, src, mul: 1 });
                    return Some(Term { node, k, root: Some(v) });
                }
                let args = self.b.args(d);
                match (self.b.insts.op[d as usize], args) {
                    (Op::Add, &[x, y]) => {
                        let (x, c) = match (self.small_const(x), self.small_const(y)) {
                            (_, Some(c)) => (x, c),
                            (Some(c), _) => (y, c),
                            _ => return Some(opaque),
                        };
                        let (to, kx) = self.written(x)?;
                        sink(Edge {
                            from: Node::Val(v),
                            to,
                            w: kx + c,
                            src: Source::Arith { inst: d },
                            mul: 1,
                        });
                        k = k.checked_add(c)?;
                        o = x;
                    }
                    (Op::Sub, &[x, y]) => {
                        let Some(c) = self.small_const(y) else { return Some(opaque) };
                        let (to, kx) = self.written(x)?;
                        sink(Edge {
                            from: Node::Val(v),
                            to,
                            w: kx - c,
                            src: Source::Arith { inst: d },
                            mul: 1,
                        });
                        k = k.checked_sub(c)?;
                        o = x;
                    }
                    (
                        Op::Cast(CastKind::Exact | CastKind::Wrap | CastKind::Sat),
                        &[x @ Operand::Value(xv)],
                    ) if self.widens(xv, v) => {
                        sink(Edge {
                            from: Node::Val(v),
                            to: Node::Val(xv),
                            w: 0,
                            src: Source::Cast { inst: d },
                            mul: 1,
                        });
                        o = x;
                    }
                    (Op::SliceLen, &[s]) => {
                        let Operand::Value(sv) = s else { return None };
                        let len = Node::Len(sv);
                        sink(Edge {
                            from: Node::Val(v),
                            to: len,
                            w: 0,
                            src: Source::SliceLen { inst: d },
                            mul: 1,
                        });
                        // `slice.len (slice.make p n)` is `n`.
                        if stop != Some(sv)
                            && let Some(ds) = self.def(sv)
                            && self.b.insts.op[ds as usize] == Op::SliceMake
                            && let Some(&n) = self.b.args(ds).get(1)
                        {
                            let (to, kn) = self.written(n)?;
                            sink(Edge { from: len, to, w: kn, src: Source::SliceMake { inst: ds }, mul: 1 });
                            nested += 1;
                            if nested > DEPTH {
                                return None;
                            }
                            o = n;
                            continue 'term;
                        }
                        return Some(Term { node: len_node(sv), k, root: Some(v) });
                    }
                    _ => return Some(opaque),
                }
                if k.abs() > MAX_K {
                    return None;
                }
            }
            return None;
        }
    }

    /// `(p, k, source)` when instruction `d` reads int field `k` behind `p`, a `&T` parameter
    /// of the entry block with T not `UnsafeCell` (one value while the function runs,
    /// [`Node::Field`]): a `load` of `addr.field p k`, or a call of a getter of that field. The
    /// caller has checked that `d`'s result is an int.
    fn stable_field(&self, d: u32) -> Option<(Value, u32, Source)> {
        match (self.b.insts.op[d as usize], self.b.args(d)) {
            (Op::Load, &[Operand::Value(a)]) => {
                let fd = self.def(a)?;
                let (Op::FieldAddr, &[Operand::Value(p), k]) =
                    (self.b.insts.op[fd as usize], self.b.args(fd))
                else {
                    return None;
                };
                self.shared_param(p)?;
                Some((p, u32::try_from(self.b.int_const(k)?).ok()?, Source::FieldLoad { inst: d }))
            }
            (Op::Call, &[Operand::Func(g), Operand::Value(p)]) => {
                let pt = self.shared_param(p)?;
                let k = getter_field(self.funcs.get(g.index())?, pt)?;
                Some((p, k, Source::Getter { call: d }))
            }
            _ => None,
        }
    }

    /// The type of `p` when it is a parameter of the entry block of type `&T`, T not
    /// `UnsafeCell`, in a module that names its `UnsafeCell`.
    fn shared_param(&self, p: Value) -> Option<TypeId> {
        let cell = self.m.cell?;
        if !self.b.params(Block(0)).contains(&p) {
            return None;
        }
        let t = self.b.value_ty(p)?;
        let &TypeKey::Ref { mutable: false, inner } = self.key(t)? else { return None };
        match self.key(inner)? {
            TypeKey::Adt { def, .. } if *def == cell => None,
            _ => Some(t),
        }
    }

    /// The length term of the slice `p` points to (`p` itself a slice reference).
    pub(super) fn slice_len_term(&self, p: Operand) -> Option<Term> {
        let Operand::Value(pv) = p else { return None };
        if let Some(d) = self.def(pv)
            && self.b.insts.op[d as usize] == Op::SliceMake
            && let Some(&n) = self.b.args(d).get(1)
        {
            return self.term(n);
        }
        Some(Term { node: len_node(pv), k: 0, root: None })
    }

    fn graph(&self) -> &Graph {
        self.facts.graph.get_or_init(|| self.build())
    }

    fn build(&self) -> Graph {
        let mut g = Graph::default();
        for t in 1..self.b.blocks.len() {
            if let Some((cond, pol)) = edge_cond(self.b, &self.preds, t) {
                let at = Region { block: t as u32, after: None };
                self.cond_facts(&mut g, cond, pol, at, Source::Branch { block: t as u32 }, NOT_DEPTH);
            }
        }
        for i in 0..self.b.insts.len() as u32 {
            if self.b.insts.op[i as usize] != Op::Call || !self.reachable_inst(i) {
                continue;
            }
            let args = self.b.args(i);
            let Some(&Operand::Func(f)) = args.first() else { continue };
            // Only a comparison argument can give a fact: skip the callee summary otherwise.
            if !args[1..].iter().any(|&a| self.is_cmp(a)) {
                continue;
            }
            let mask = self.asserted(f);
            for (k, &a) in args[1..].iter().enumerate().take(64) {
                if mask & (1u64 << k) != 0 {
                    let at = Region { block: self.block_of[i as usize], after: Some(i) };
                    let src = Source::Assert { call: i, arg: k as u32 };
                    self.cond_facts(&mut g, a, true, at, src, NOT_DEPTH);
                }
            }
        }
        for i in 0..self.b.insts.len() as u32 {
            if self.b.insts.op[i as usize] == Op::BoundsCheck && self.reachable_inst(i) {
                self.check_facts(&mut g, i);
            }
        }
        for d in 0..self.b.insts.len() as u32 {
            if matches!(self.b.insts.op[d as usize], Op::And | Op::Rem | Op::Shr) && self.reachable_inst(d) {
                self.bit_facts(&mut g, d);
            }
        }
        for i in 0..self.b.insts.len() as u32 {
            if self.b.insts.op[i as usize] == Op::Call && self.reachable_inst(i) {
                self.call_facts(&mut g, i);
            }
        }
        // The lower end of the type, `lo <= v`, for the value nodes a fact leaves toward a node
        // other than zero, collected before the upper ends, which only lead to zero.
        let mut leave: Vec<u32> = g.facts.iter().filter(|f| f.to != ZERO).map(|f| f.from).collect();
        leave.sort_unstable();
        leave.dedup();
        for node in leave {
            let Node::Val(v) = node_of(node) else { continue };
            if let Some((signed, bits)) = self.int_ty(v)
                && let Some(at) = self.defined_in(v)
            {
                let lo = if signed { -(1i128 << (bits - 1)) } else { 0 };
                self.add_le(
                    &mut g,
                    at,
                    Source::Range { value: v },
                    (None, Some(Operand::Value(v)), -lo),
                    None,
                );
            }
        }
        for v in (0..self.b.values.len() as u32).map(Value) {
            let top = match self.int_ty(v) {
                Some((false, w)) if w < 64 => (1i128 << w) - 1,
                Some((true, w)) => (1i128 << (w - 1)) - 1,
                _ => continue,
            };
            if let Some(at) = self.defined_in(v) {
                self.add_le(
                    &mut g,
                    at,
                    Source::Range { value: v },
                    (Some(Operand::Value(v)), None, top),
                    None,
                );
            }
        }
        // Stable: the facts of one node stay in the order they were found, which is the order a
        // query tries them in.
        g.facts.sort_by_key(|f| f.from);
        g
    }

    /// The fact a `check.bounds x (n -| c)` states after it: `x < n -| c`, so `x + c < n`
    /// (`n -| c` is `max(n - c, 0)`, and `x < 0` is false for an unsigned `x`). A check on any
    /// other length is no fact here: the `Dominated` proof reads it.
    fn check_facts(&self, g: &mut Graph, i: u32) {
        let &[x, y] = self.b.args(i) else { return };
        // `check.bounds` compares unsigned (its operands are unsigned values or constants ≥ 0)
        if let Operand::Value(xv) = x
            && !self.unsigned(xv)
        {
            return;
        }
        let Some((n, c)) = self.sat_bound(y) else { return };
        let at = Region { block: self.block_of[i as usize], after: Some(i) };
        self.add_le(g, at, Source::Check { inst: i }, (Some(x), Some(n), -c - 1), None);
    }

    /// The facts a returned `call @f ..` states by `f`'s summary: its `post` rows about the
    /// arguments and the result, and its `pre` rows that read a `requires` check (checked at
    /// entry, or proved by this call) about the arguments.
    fn call_facts(&self, g: &mut Graph, i: u32) {
        let Some(&Operand::Func(f)) = self.b.args(i).first() else { return };
        let Some(s) = self.claim(f) else { return };
        let at = Region { block: self.block_of[i as usize], after: Some(i) };
        for (k, row) in s.post.iter().enumerate() {
            if let Some((a, b)) = self.call_sides(i, row) {
                self.add_sides(g, at, Source::Ensures { call: i, clause: k as u32 }, (a, b, row.w), None);
            }
        }
        for (k, p) in s.pre.iter().enumerate() {
            if p.entry != NO_BLOCK
                && let Some((a, b)) = self.call_sides(i, &p.row)
            {
                self.add_sides(g, at, Source::Arg { call: i, clause: k as u32 }, (a, b, p.row.w), None);
            }
        }
    }

    /// The two sides of summary row `row` of the callee of call `i`, about its operands and
    /// result.
    pub(super) fn call_sides(&self, i: u32, row: &Row) -> Option<(Side, Side)> {
        let args = self.b.args(i);
        let result = self.b.insts.result.get(i as usize).copied().flatten();
        let side = |t: RowTerm| -> Option<Side> {
            let arg = |k: u32| args.get(k as usize + 1).copied();
            match t {
                RowTerm::Zero => Some(Side::Zero),
                RowTerm::Param(k) => arg(k).map(Side::Val),
                RowTerm::LenParam(k) => match arg(k)? {
                    Operand::Value(v) => Some(Side::Len(v)),
                    _ => None,
                },
                RowTerm::Result => result.map(|v| Side::Val(Operand::Value(v))),
                RowTerm::LenResult => result.map(Side::Len),
            }
        };
        Some((side(row.a)?, side(row.b)?))
    }

    /// The facts bit op `d` (`and`, `rem`, `shr`) states about its unsigned result.
    fn bit_facts(&self, g: &mut Graph, d: u32) {
        let Some(Some(v)) = self.b.insts.result.get(d as usize).copied() else { return };
        let Some(w) = self.width(v) else { return };
        let top = (1i128 << w) - 1;
        // A constant operand is read as itself: only one that is a value of the type.
        let typed = |o: Operand| self.b.int_const(o).is_none_or(|c| (0..=top).contains(&c));
        let at = Region { block: self.block_of[d as usize], after: Some(d) };
        let src = Source::Bits { inst: d };
        let mut le =
            |o: Operand, w: i128| self.add_le(g, at, src, (Some(Operand::Value(v)), Some(o), w), None);
        match (self.b.insts.op[d as usize], self.b.args(d)) {
            (Op::And, &[x, y]) if typed(x) && typed(y) => {
                le(x, 0);
                le(y, 0);
            }
            (Op::Rem, &[x, y]) if typed(x) && typed(y) && self.b.int_const(y) != Some(0) => {
                le(x, 0);
                le(y, -1);
            }
            (Op::Shr, &[x, _]) if typed(x) => le(x, 0),
            _ => {}
        }
    }

    fn is_cmp(&self, o: Operand) -> bool {
        let Operand::Value(v) = o else { return false };
        self.def(v).is_some_and(|d| {
            matches!(
                self.b.insts.op[d as usize],
                Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne | Op::Not
            )
        })
    }

    /// The facts of condition `c` being `pol`, read at `src`.
    fn cond_facts(&self, g: &mut Graph, c: Operand, pol: bool, at: Region, src: Source, depth: u32) {
        let Operand::Value(v) = c else { return };
        let Some(d) = self.def(v) else { return };
        let (op, args) = (self.b.insts.op[d as usize], self.b.args(d));
        if op == Op::Not {
            if let (&[x], 1..) = (args, depth) {
                self.cond_facts(g, x, !pol, at, src, depth - 1);
            }
            return;
        }
        let &[x, y] = args else { return };
        // `lo < hi` (or `lo >= hi`) with `hi = n -| c`: `lo < n -| c` iff `lo + c < n`, both false
        // when `n < c`. `<=` has no such form: `0 <= n -| c` holds for every `n`.
        let strict = match (op, pol) {
            (Op::Lt, true) | (Op::Ge, false) => Some((x, y, true)),
            (Op::Lt, false) | (Op::Ge, true) => Some((x, y, false)),
            (Op::Gt, true) | (Op::Le, false) => Some((y, x, true)),
            (Op::Gt, false) | (Op::Le, true) => Some((y, x, false)),
            _ => None,
        };
        let mut le = |a: Operand, b: Operand, w: i128, eq: Option<u32>| {
            self.add_le(g, at, src, (Some(a), Some(b), w), eq)
        };
        if let Some((lo, hi, lt)) = strict
            && let Some((n, c)) = self.sat_bound(hi)
        {
            if lt {
                // lo + c < n
                le(lo, n, -c - 1, None);
            } else {
                // n <= lo + c
                le(n, lo, c, None);
            }
        }
        match (op, pol) {
            // x < y
            (Op::Lt, true) | (Op::Ge, false) => le(x, y, -1, None),
            // x >= y
            (Op::Lt, false) | (Op::Ge, true) => le(y, x, 0, None),
            // x > y
            (Op::Gt, true) | (Op::Le, false) => le(y, x, -1, None),
            // x <= y
            (Op::Gt, false) | (Op::Le, true) => le(x, y, 0, None),
            (Op::Eq, true) | (Op::Ne, false) => {
                le(x, y, 0, Some(d));
                le(y, x, 0, Some(d));
            }
            _ => {}
        }
    }

    /// `(n, c)` when `o` is `sub.sat n c` on unsigned values with a constant `c`.
    pub(super) fn sat_bound(&self, o: Operand) -> Option<(Operand, i128)> {
        let Operand::Value(v) = o else { return None };
        if !self.unsigned(v) {
            return None;
        }
        let d = self.def(v)?;
        match (self.b.insts.op[d as usize], self.b.args(d)) {
            (Op::SubSat, &[n, c]) => Some((n, self.small_const(c)?)),
            _ => None,
        }
    }

    /// Records `a <= b + w` (the triple `le`; `None` is the constant zero), read at `src` and valid
    /// in `at`.
    fn add_le(
        &self,
        g: &mut Graph,
        at: Region,
        src: Source,
        le: (Option<Operand>, Option<Operand>, i128),
        eq: Option<u32>,
    ) {
        let side = |o: Option<Operand>| o.map_or(Side::Zero, Side::Val);
        self.add_sides(g, at, src, (side(le.0), side(le.1), le.2), eq);
    }

    /// [`Cx::add_le`] between two [`Side`]s.
    fn add_sides(&self, g: &mut Graph, at: Region, src: Source, le: (Side, Side, i128), eq: Option<u32>) {
        let (a, b, w) = le;
        let (Some(ta), Some(tb)) = (self.side_term(a), self.side_term(b)) else { return };
        if ta.node == tb.node {
            return;
        }
        let (Some((_, ka)), Some((_, kb))) = (self.side_written(a), self.side_written(b)) else { return };
        // ta.node + ta.k <= tb.node + tb.k + w
        let Some(folded) = tb.k.checked_add(w).and_then(|s| s.checked_sub(ta.k)) else { return };
        // as written: a's node + ka <= b's node + kb + w
        let Some(wop) = kb.checked_add(w).and_then(|s| s.checked_sub(ka)) else { return };
        g.facts.push(Fact { from: ta.node, to: tb.node, w: folded, at, eq, src, a, b, wop });
    }

    /// The bit set of `f`'s parameters that are true whenever `f` returns.
    pub(super) fn asserted(&self, f: FuncId) -> u64 {
        if let Some(&m) = self.facts.asserted.borrow().get(&f) {
            return m;
        }
        let m = self.funcs.get(f.index()).and_then(|f| f.body.as_ref()).map_or(0, asserted_params);
        self.facts.asserted.borrow_mut().insert(f, m);
        m
    }

    fn holds(&self, at: Region, block: u32, inst: u32) -> bool {
        if at.block == block {
            at.after.is_none_or(|a| a < inst)
        } else {
            self.dom.dominates(at.block as usize, block as usize)
        }
    }

    /// The `LenEq` or `Offset` proof that `idx < n` holds right before instruction `at`. With
    /// `path`, the facts used are appended to it in order (indices for [`Cx::fact_edges`]).
    pub(super) fn prove_lt(&self, at: u32, idx: Term, n: Term, path: Option<&mut Vec<u32>>) -> Option<Proof> {
        self.prove_at(self.block_of[at as usize], at, idx, n, path)
    }

    /// [`Cx::prove_lt`] right before instruction `at` of `block` (`u32::MAX`: at the block's end).
    pub(super) fn prove_at(
        &self,
        block: u32,
        at: u32,
        idx: Term,
        n: Term,
        path: Option<&mut Vec<u32>>,
    ) -> Option<Proof> {
        let bound = n.k.checked_sub(idx.k)?.checked_sub(1)?;
        let offset = Proof::Offset { base: idx.root.map(|v| v.0), k: idx.k };
        // One node on both sides: holds by the constants alone, or by a walk back to the node
        // of negative weight (contradictory facts) found below.
        if idx.node == n.node && bound >= 0 {
            return Some(offset);
        }
        let g = self.graph();
        const START: u32 = u32::MAX;
        // Per node: its best distance so far, whether an equality was used, and the step that
        // reached it. A step is never rewritten, so its parents spell out one walk of exactly
        // the distance it was entered with, whatever improves later.
        let mut taken: Vec<Step> = Vec::new();
        let mut dist: FxHashMap<u32, (i128, Option<u32>, u32)> = FxHashMap::default();
        dist.insert(idx.node, (0, None, START));
        let mut work = VecDeque::from([idx.node]);
        let mut steps = 0u32;
        while let Some(u) = work.pop_front() {
            let Some(&(du, eq_u, step_u)) = dist.get(&u) else { continue };
            for fi in g.from(u) {
                let f = &g.facts[fi];
                steps += 1;
                if steps > BOUNDS_STEP_LIMIT {
                    return None;
                }
                if !self.holds(f.at, block, at) {
                    continue;
                }
                let Some(d) = du.checked_add(f.w) else { continue };
                let eq = eq_u.or(f.eq);
                if f.to == n.node && d <= bound {
                    if let Some(path) = path {
                        let from = path.len();
                        path.push(fi as u32);
                        let mut s = step_u;
                        while let Some(step) = taken.get(s as usize) {
                            path.push(step.fact);
                            s = step.parent;
                        }
                        path[from..].reverse();
                    }
                    return Some(match eq {
                        Some(cmp) => Proof::LenEq { cmp },
                        None => offset,
                    });
                }
                if dist.get(&f.to).is_none_or(|&(old, _, _)| d < old) {
                    dist.insert(f.to, (d, eq, taken.len() as u32));
                    taken.push(Step { parent: step_u, fact: fi as u32 });
                    work.push_back(f.to);
                }
            }
        }
        None
    }

    /// [`Cx::prove_lt`] for `idx <= n`.
    pub(super) fn prove_le(&self, at: u32, idx: Term, n: Term) -> Option<Proof> {
        self.prove_lt(at, idx, Term { k: n.k.checked_add(1)?, ..n }, None)
    }

    /// Facts that contradict each other right before instruction `at` of `block` (`u32::MAX`: at
    /// the block's end): a node and a walk of facts back to it of negative total weight
    /// (`u <= u - 1`). Only the start nodes of branch and assert facts holding there are tried
    /// (a contradiction needs a condition), at most [`CYCLE_STARTS`] of them, each a bounded
    /// query. For a call the point is before it: what the call itself states on return (its
    /// callee's `requires`) contradicts the very facts that refute it.
    pub(super) fn contradiction(&self, block: u32, at: u32) -> Option<(u32, Vec<u32>)> {
        let g = self.graph();
        let mut starts: Vec<u32> = Vec::new();
        for f in &g.facts {
            if matches!(f.src, Source::Branch { .. } | Source::Assert { .. })
                && self.holds(f.at, block, at)
                && !starts.contains(&f.from)
            {
                starts.push(f.from);
                if starts.len() == CYCLE_STARTS {
                    break;
                }
            }
        }
        starts.into_iter().find_map(|u| {
            let t = Term { node: u, k: 0, root: None };
            let mut path = Vec::new();
            self.prove_at(block, at, t, t, Some(&mut path)).map(|_| (u, path))
        })
    }

    /// A walk of facts holding at the end of `block` from the node of `from` to node `to`, of any
    /// weight (empty when they are the same node).
    pub(super) fn walk(&self, block: u32, from: Term, to: u32) -> Option<Vec<u32>> {
        let mut path = Vec::new();
        let far = Term { node: to, k: MAX_K, root: None };
        self.prove_at(block, u32::MAX, Term { node: from.node, k: 0, root: None }, far, Some(&mut path))?;
        Some(path)
    }

    /// The node of term `t`.
    pub(super) fn node_of(t: Term) -> u32 {
        t.node
    }

    /// The term of node `u` itself.
    pub(super) fn node_term(u: u32) -> Term {
        Term { node: u, k: 0, root: None }
    }

    /// The folding steps of operand `o`, from the operand as written down to its term's node.
    pub(super) fn down(&self, o: Operand, out: &mut Vec<Edge>) {
        let _ = self.fold(o, &mut |e| out.push(e));
    }

    /// The folding steps of operand `o` backwards, from its term's node up to the operand as
    /// written. Every step is an equality, so it holds in this direction too.
    pub(super) fn up(&self, o: Operand, out: &mut Vec<Edge>) {
        let from = out.len();
        self.down(o, out);
        out[from..].reverse();
        for e in &mut out[from..] {
            *e = Edge { from: e.to, to: e.from, w: -e.w, ..*e };
        }
    }

    /// The edges of fact `fi` between its two term nodes: up to its left operand, the fact as
    /// read at its source, down from its right operand.
    pub(super) fn fact_edges(&self, fi: u32, out: &mut Vec<Edge>) -> Option<()> {
        let f = self.graph().facts.get(fi as usize)?;
        self.side_up(f.a, None, out);
        let ((from, _), (to, _)) = (self.side_written(f.a)?, self.side_written(f.b)?);
        out.push(Edge { from, to, w: f.wop, src: f.src, mul: 1 });
        self.side_down(f.b, None, out);
        Some(())
    }

    /// The term of a side, with value `stop` read as a node of its own ([`Cx::fold_to`]).
    pub(super) fn side_term_to(&self, s: Side, stop: Option<Value>) -> Option<Term> {
        match s {
            Side::Zero => Some(Term::constant(0)),
            Side::Val(o) => self.fold_to(o, stop, &mut |_| {}),
            Side::Len(v) => {
                let mut t = None;
                self.len_down(v, stop, &mut |_| {}, &mut t);
                t
            }
        }
    }

    fn side_term(&self, s: Side) -> Option<Term> {
        self.side_term_to(s, None)
    }

    /// The node and offset of a side as written.
    pub(super) fn side_written(&self, s: Side) -> Option<(Node, i128)> {
        match s {
            Side::Zero => Some((Node::Zero, 0)),
            Side::Val(o) => self.written(o),
            Side::Len(v) => Some((Node::Len(v), 0)),
        }
    }

    /// The folding steps of the length of slice value `s`, from `Len(s)` down to its term's
    /// node (`term`): none, or through the `slice.make` that defines `s` to its length operand.
    fn len_down(&self, s: Value, stop: Option<Value>, sink: &mut impl FnMut(Edge), term: &mut Option<Term>) {
        if stop != Some(s)
            && let Some(d) = self.def(s)
            && self.b.insts.op[d as usize] == Op::SliceMake
            && let Some(&n) = self.b.args(d).get(1)
        {
            let Some((to, kn)) = self.written(n) else { return };
            sink(Edge { from: Node::Len(s), to, w: kn, src: Source::SliceMake { inst: d }, mul: 1 });
            *term = self.fold_to(n, stop, sink);
            return;
        }
        *term = Some(Term { node: len_node(s), k: 0, root: None });
    }

    /// The folding steps of side `s`, from the side as written down to its term's node.
    pub(super) fn side_down(&self, s: Side, stop: Option<Value>, out: &mut Vec<Edge>) {
        match s {
            Side::Zero => {}
            Side::Val(o) => {
                let _ = self.fold_to(o, stop, &mut |e| out.push(e));
            }
            Side::Len(v) => self.len_down(v, stop, &mut |e| out.push(e), &mut None),
        }
    }

    /// [`Cx::side_down`] backwards, from the term's node up to the side as written.
    pub(super) fn side_up(&self, s: Side, stop: Option<Value>, out: &mut Vec<Edge>) {
        let from = out.len();
        self.side_down(s, stop, out);
        out[from..].reverse();
        for e in &mut out[from..] {
            *e = Edge { from: e.to, to: e.from, w: -e.w, ..*e };
        }
    }

    /// The proof of `a <= b + w` right before instruction `at` of `block` (`u32::MAX`: at the
    /// block's end), appended to `out`: both sides folded, then a path of facts or the type
    /// range of the one value node against a constant.
    pub(super) fn side_goal(
        &self,
        block: u32,
        at: u32,
        (a, b, w): (Side, Side, i128),
        out: &mut Vec<Edge>,
    ) -> Option<()> {
        let (ta, tb) = (self.side_term(a)?, self.side_term(b)?);
        let bound = tb.k.checked_add(w)?.checked_sub(ta.k)?;
        let from = out.len();
        self.side_down(a, None, out);
        let mut path = Vec::new();
        let limit = Term { k: tb.k.checked_add(w)?.checked_add(1)?, ..tb };
        let mid = if self.prove_at(block, at, ta, limit, Some(&mut path)).is_some() {
            path.iter().try_for_each(|&f| self.fact_edges(f, out))
        } else {
            self.range_edge(ta.node, tb.node, bound, out)
        };
        if mid.is_none() {
            out.truncate(from);
            return None;
        }
        self.side_up(b, None, out);
        Some(())
    }

    /// `from <= to + bound` between a value node and zero when the value's type range alone
    /// gives it: one `Range` edge.
    fn range_edge(&self, from: u32, to: u32, bound: i128, out: &mut Vec<Edge>) -> Option<()> {
        let range = |node: u32| -> Option<(Value, i128, i128)> {
            let Node::Val(v) = node_of(node) else { return None };
            let (signed, bits) = self.int_ty(v)?;
            Some(if signed {
                (v, -(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
            } else {
                (v, 0, (1i128 << bits) - 1)
            })
        };
        let (value, edge) = if to == ZERO {
            let (v, _, hi) = range(from)?;
            (v, (Node::Val(v), Node::Zero, hi))
        } else if from == ZERO {
            let (v, lo, _) = range(to)?;
            (v, (Node::Zero, Node::Val(v), -lo))
        } else {
            return None;
        };
        let (from, to, w) = edge;
        (w <= bound).then(|| out.push(Edge { from, to, w, src: Source::Range { value }, mul: 1 }))
    }
}

/// `(cond, pol)` when block `t` (not the entry) is entered only through the `pol` edge of a
/// `cbr cond`.
pub(super) fn edge_cond(b: &Body, preds: &[Vec<(u32, bool)>], t: usize) -> Option<(Operand, bool)> {
    let &[(d, pol)] = preds.get(t)?.as_slice() else { return None };
    if t == 0 {
        return None;
    }
    let IrTerm::Cbr { cond, then, els } = b.blocks.get(d as usize)?.term else { return None };
    let (then, els) = (then.block.index(), els.block.index());
    let ok = if pol { then == t && els != t } else { els == t && then != t };
    ok.then_some((cond, pol))
}

/// The parameters (bit `k` = parameter `k`) every `ret` of `b` is reached only through a branch
/// on that parameter being true: `cbr p T F` with only `T` leading there, or `cbr (not p) T F`
/// with only `F`.
fn asserted_params(b: &Body) -> u64 {
    if b.blocks.len() > ASSERT_BLOCKS {
        return 0;
    }
    let params = b.params(Block(0));
    let param_bit = |o: Operand| match o {
        Operand::Value(v) => params.iter().position(|&p| p == v).filter(|&k| k < 64).map(|k| 1u64 << k),
        _ => None,
    };
    let dom = Dominators::compute(b);
    let preds = preds_of(b, &dom);
    let mut mask = u64::MAX;
    let mut returns = false;
    for (r, bd) in b.blocks.iter().enumerate() {
        if !dom.reachable(r) || !matches!(bd.term, IrTerm::Ret(_)) {
            continue;
        }
        returns = true;
        let mut here = 0u64;
        let mut walk = Some(r);
        while let Some(t) = walk {
            if let Some((cond, pol)) = edge_cond(b, &preds, t) {
                let negated = match cond {
                    Operand::Value(v) => match b.values.get(v.index()).map(|d| d.def) {
                        Some(ValueDef::Inst(i)) if b.insts.op.get(i as usize) == Some(&Op::Not) => {
                            b.args(i).first().copied()
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let bit = match (pol, negated) {
                    (true, None) => param_bit(cond),
                    (false, Some(p)) => param_bit(p),
                    _ => None,
                };
                here |= bit.unwrap_or(0);
            }
            walk = dom.idom(t).filter(|&p| p != t);
        }
        mask &= here;
    }
    if returns { mask } else { 0 }
}

/// Field `k` when `g` has the one parameter type `pt` and the body `a = addr.field %0 k;
/// r = load a; ret r`.
fn getter_field(g: &Func, pt: TypeId) -> Option<u32> {
    let b = g.body.as_ref()?;
    if g.params[..] != [pt]
        || b.blocks.len() != 1
        || b.insts.len() != 2
        || b.blocks[0].insts.indices() != (0..2)
    {
        return None;
    }
    let &[p] = b.params(Block(0)) else { return None };
    let (Some(a), Some(r)) = (b.insts.result[0], b.insts.result[1]) else { return None };
    match (b.insts.op[0], b.args(0), b.insts.op[1], b.args(1), b.blocks[0].term) {
        (
            Op::FieldAddr,
            &[Operand::Value(x), k],
            Op::Load,
            &[Operand::Value(y)],
            IrTerm::Ret(Some(Operand::Value(z))),
        ) if x == p && y == a && z == r => u32::try_from(b.int_const(k)?).ok(),
        _ => None,
    }
}
