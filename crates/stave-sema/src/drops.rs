//! The drop schedule: which value is dropped where, the one answer the checks and the generated
//! code share.
//!
//! **Rows.** A body's schedule ([`Schedule`]) has one [`Row`] per drop the body may run: the
//! point ([`Point`]: where its owner ends, an exit that leaves it, the binding point of a pattern,
//! an assignment) and what is dropped there ([`Owner`]: a local, a temporary of a statement, a
//! part a moving pattern ignores, the value an assignment overwrites). The rows are made once per
//! body by the builder of the borrow CFG (stave-borrow `cfg`), each together with the CFG
//! statement that models the drop, so every check that reads the CFG (loans, moves, lock ranks,
//! joins) reads the schedule. A row is one value: a temporary is named by its statement and
//! the token that ends the expression whose value it holds ([`Owner::Temp`], [`value_key`]),
//! which both stages compute from the syntax where they make it. The lowering asks the
//! schedule for every drop it emits ([`Schedule::has`]), each row at most once, and for the
//! order of the drops of each end and exit ([`Schedule::in_order`]): a drop without a row, a
//! second drop of one row, or a drop in another order is code no check has seen, an internal
//! error of the build. The other way round, the rows of code that can run are due
//! ([`Schedule::require`]): where the lowering lowers their point it must have decided each
//! one, so a drop it leaves out, or a value the checks keep alive that the code never made,
//! fails the build too. Whether a scheduled drop runs is the owner's init bit: the CFG's
//! dataflow and the lowering's owner states (or dynamic flags) both follow the moves.
//!
//! **Parts of a pattern.** A pattern that binds a non-Copy part by value takes its whole value;
//! the parts it does not bind are dropped at its binding point, in the order of
//! [`ignored_parts`], which both stages call ([`pat_shape`] reads the pattern for both).
//!
//! **Glue.** What the drop of a value of a type runs, in order ([`glue`]): the type's `Drop`
//! impl, then the glue of its parts in declaration order. The glue function, the call graph's
//! glue node, and the lock check of a value's own guards ([`crate::ranks`]) are made from it.

use rustc_hash::FxHashMap;
use stave_syntax::{NodeKind, Term};

use crate::Program;
use crate::decls::DeclKind;
use crate::ids::DeclId;
use crate::itemcx::ItemCx;
use crate::mono::InstKey;
use crate::resolve::Res;
use crate::sigs::Sigs;
use crate::traits::World;
use crate::types::{ConstArg, Interner, TypeId, TypeKind};

/// Where a drop runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Point {
    /// Where the owner ends: the end of its scope, of its statement, or of its loop.
    End,
    /// On the way out of the fn by the `return` at this token.
    Return(u32),
    /// On the way out of the fn by the `?` at this token.
    Try(u32),
    /// On the way to a loop's head or exit by the `break` or `continue` at this token.
    Jump(u32),
    /// The binding point of the pattern with this node.
    Bind(u32),
    /// The assignment whose destination is the expression with this node, before its write.
    Assign(u32),
}

/// What a drop drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Owner {
    /// The local of this binding node (a param: its node).
    Local(u32),
    /// A temporary of the statement with this node ([`crate::temps`]) holding the value of the
    /// expression that ends at this token (a step of a postfix chain: the step's last token).
    /// Both stages name a temporary so, each from the syntax where it makes the temporary: one
    /// row is one value.
    Temp(u32, u32),
    /// The part with this index in [`ignored_parts`] of the pattern at the point.
    Part(u32),
    /// The value the assignment at the point overwrites.
    Old,
}

/// The outermost constructor of a dropped value's type: the same in a generic body and in each
/// of its instances, unless the body's type is a type parameter ([`Head::Any`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Head {
    Any,
    Adt(DeclId),
    Tuple(u32),
    Array,
}

/// The token naming a temporary that holds the value of expression `n` ([`Owner::Temp`]): the
/// expression's last token, looking through parentheses (`(e)` is the value of `e`).
pub fn value_key(icx: &ItemCx<'_>, mut n: u32) -> u32 {
    for _ in 0..1_000_000 {
        let a = icx.a(n);
        if icx.kind(n) != Some(NodeKind::Primary)
            || !icx.is(a, Term::LParen)
            || icx.is(icx.close(a) - 1, Term::Comma)
        {
            break;
        }
        match icx.only_child(n) {
            Some(c) => n = c,
            None => break,
        }
    }
    icx.end_tok(n)
}

/// The [`Head`] of type `t`.
pub fn head<I: Interner + ?Sized>(cx: &I, t: TypeId) -> Head {
    if t.is_none() {
        return Head::Any;
    }
    match cx.kind(t) {
        TypeKind::Adt { def, .. } => Head::Adt(*def),
        TypeKind::Tuple(ts) => Head::Tuple(ts.len() as u32),
        TypeKind::Array { .. } => Head::Array,
        _ => Head::Any,
    }
}

/// One scheduled drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row {
    pub point: Point,
    pub owner: Owner,
    pub head: Head,
}

/// The drops of one body (its closures and `spawn` bodies included: nodes and tokens are the
/// item's).
#[derive(Clone, Debug, Default)]
pub struct Schedule {
    rows: Vec<Row>,
    /// The drops in the order they were scheduled; once sealed, grouped by point.
    order: Vec<(Point, Owner)>,
    /// The drops the code must decide ([`Schedule::require`]).
    due: Vec<(Point, Owner)>,
    sorted: bool,
}

impl Schedule {
    /// Adds a row. The rows of one point are added in the order their drops run.
    pub fn add(&mut self, row: Row) {
        self.rows.push(row);
        self.order.push((row.point, row.owner));
        self.sorted = false;
    }

    /// The scheduled drop of `owner` at `point` is in code that can run: where the lowering
    /// lowers that point it must decide the drop (emit it, or leave it out because the owner
    /// holds no value or its type has no drop glue there).
    pub fn require(&mut self, point: Point, owner: Owner) {
        self.due.push((point, owner));
    }

    /// The drops the code must decide.
    pub fn due(&self) -> &[(Point, Owner)] {
        &self.due
    }

    pub fn extend(&mut self, other: &Schedule) {
        self.rows.extend_from_slice(&other.rows);
        self.order.extend_from_slice(&other.order);
        self.due.extend_from_slice(&other.due);
        self.sorted = false;
    }

    /// Sorts the rows for [`Schedule::has`] and [`Schedule::in_order`].
    pub fn seal(&mut self) {
        self.rows.sort_unstable();
        self.rows.dedup();
        // stable: the drops of a point stay in the order they run
        self.order.sort_by_key(|x| x.0);
        self.due.sort_unstable();
        self.due.dedup();
        self.sorted = true;
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Whether drops of `owners`, run one after the other at `point`, are scheduled there in
    /// that order (other scheduled drops may lie between them: a drop of a value that was moved
    /// out on the way does not run).
    pub fn in_order(&self, point: Point, owners: &[Owner]) -> bool {
        let (lo, hi) = if self.sorted {
            let lo = self.order.partition_point(|x| x.0 < point);
            (lo, lo + self.order[lo..].partition_point(|x| x.0 == point))
        } else {
            (0, self.order.len())
        };
        let mut rest = self.order[lo..hi].iter().filter(|x| x.0 == point).map(|x| x.1);
        owners.iter().all(|o| rest.any(|x| x == *o))
    }

    /// Whether a drop of `owner` at `point` is scheduled for a value whose type has head `head`.
    pub fn has(&self, point: Point, owner: Owner, head: Head) -> bool {
        let fits = |r: &Row| {
            r.point == point
                && r.owner == owner
                && (r.head == head || r.head == Head::Any || head == Head::Any)
        };
        if !self.sorted {
            return self.rows.iter().any(fits);
        }
        let lo = self.rows.partition_point(|r| (r.point, r.owner) < (point, owner));
        self.rows[lo..].iter().take_while(|r| r.point == point && r.owner == owner).any(fits)
    }
}

/// The schedules of a program's checked bodies, by fn decl.
pub type Schedules = FxHashMap<DeclId, Schedule>;

// ---------- glue ----------

/// The parts a value's drop glue drops after the type's own `Drop` impl ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parts {
    None,
    /// A struct or a tuple: its fields, dropped in declaration order.
    Fields(Vec<TypeId>),
    /// An enum: the fields of each variant; the fields of the value's variant are dropped, in
    /// declaration order.
    Variants(Vec<Vec<TypeId>>),
    /// An array: its elements, dropped in index order.
    Elems(TypeId, u64),
}

impl Parts {
    /// Every part type, in order (an enum: variant by variant).
    pub fn flat(&self) -> Vec<TypeId> {
        match self {
            Parts::None => Vec::new(),
            Parts::Fields(fs) => fs.clone(),
            Parts::Variants(vs) => vs.iter().flatten().copied().collect(),
            Parts::Elems(e, _) => vec![*e],
        }
    }
}

/// What the drop of a value of the concrete type `t` runs: `method` (the instance of its
/// `Drop.drop`, on the whole value) first, then the glue of each of `parts` that has any, in
/// order. A part is alive until its own turn: while `method` runs every part is, while a field
/// is dropped the fields after it are.
#[derive(Clone, Debug)]
pub struct Glue {
    pub method: Option<InstKey>,
    pub parts: Parts,
}

/// The [`Glue`] of `t`.
pub fn glue<I: Interner + ?Sized>(w: &World<'_>, cx: &mut I, t: TypeId) -> Glue {
    let method = crate::mono::drop_method(w, cx, t);
    let parts = match cx.kind(t).clone() {
        TypeKind::Adt { def, .. } => {
            let is_enum = w.sigs.adt(def).is_some_and(|a| a.is_enum);
            let vs = crate::mono::adt_fields(w.sigs, cx, t).unwrap_or_default();
            if is_enum {
                Parts::Variants(vs)
            } else {
                Parts::Fields(vs.into_iter().next().unwrap_or_default())
            }
        }
        TypeKind::Tuple(ts) => Parts::Fields(ts.to_vec()),
        TypeKind::Array { elem, len: ConstArg::Value(n) } => Parts::Elems(elem, n),
        TypeKind::Array { elem, .. } => Parts::Elems(elem, 0),
        _ => Parts::None,
    };
    Glue { method, parts }
}

// ---------- patterns ----------

/// How a pattern binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Move,
    Ref(bool),
}

/// A field's sub-pattern: a pattern, or a field named without one (a binding of that name).
#[derive(Clone, Copy, Debug)]
pub enum Sub {
    Pat(u32),
    Name(u32),
}

/// What a pattern is against a value.
#[derive(Clone, Debug)]
pub enum Shape {
    /// `_`.
    Wild,
    /// One name: the binding node.
    Name(u32),
    /// A literal, a range, a constant: a test that binds nothing.
    Test,
    /// A tuple, struct or variant pattern against (peeled) type `ty`: per field of the tuple,
    /// struct or variant its type and its sub-pattern (a pattern names every field, once).
    /// `mode`: how names inside bind.
    Parts { ty: TypeId, mode: Mode, variant: Option<u32>, fields: Vec<(TypeId, Sub)> },
}

/// One step from a value to a part of it: field `field` (of variant `variant` of an enum), of
/// type `ty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub variant: Option<u32>,
    pub field: u32,
    pub ty: TypeId,
}

/// The types and answers a stage reads patterns with: the lowering with an instance's types,
/// the borrow checker with the definition's.
pub trait PatCx<'a> {
    fn icx(&self) -> ItemCx<'a>;
    fn prog(&self) -> &'a Program;
    fn sigs(&self) -> &'a Sigs;
    fn kind(&self, t: TypeId) -> TypeKind;
    /// Field types of the variants of ADT type `t` (struct: one variant).
    fn fields(&mut self, t: TypeId) -> Option<Vec<Vec<TypeId>>>;
    /// Whether type `t` has a `Drop` impl of its own.
    fn drop_impl(&mut self, t: TypeId) -> bool;
    /// Whether the binding at node `b`, bound by value, takes a non-Copy value: the answer of the
    /// definition ([`crate::typeck::TypedBody::binds_by_move`]).
    fn by_move(&mut self, b: u32) -> bool;
    /// A pattern the stages before accept cannot have this shape: a compiler bug at node `n`.
    fn bug(&mut self, n: u32, why: &'static str);
}

/// The type a pattern that is not one name sees: references peeled, the binding mode becoming
/// by reference under one.
pub fn peel_ty<'a, C: PatCx<'a> + ?Sized>(cx: &C, mut t: TypeId, mut mode: Mode) -> (TypeId, Mode) {
    for _ in 0..1_000_000 {
        match cx.kind(t) {
            TypeKind::Ref { mutable, inner } => {
                mode = match mode {
                    Mode::Move => Mode::Ref(mutable),
                    Mode::Ref(m) => Mode::Ref(m && mutable),
                };
                t = inner;
            }
            _ => break,
        }
    }
    (t, mode)
}

/// A pattern [`pat_shape`] cannot read. Taking it for a test would leave its value whole while
/// its names took parts of it, so it is a bug.
fn no_shape<'a, C: PatCx<'a> + ?Sized>(cx: &mut C, p: u32) -> Shape {
    cx.bug(p, "a pattern of no known shape");
    Shape::Test
}

/// The shape of pattern `p` against a value of type `t`.
pub fn pat_shape<'a, C: PatCx<'a> + ?Sized>(cx: &mut C, p: u32, t: TypeId, mode: Mode) -> Shape {
    let icx = cx.icx();
    let a = icx.a(p);
    if icx.is(a, Term::Underscore) {
        return Shape::Wild;
    }
    if icx.is(a, Term::KwMut) {
        return Shape::Name(p);
    }
    let kids = icx.kids_vec(p);
    if kids.first().is_some_and(|&l0| icx.kind(l0) == Some(NodeKind::Literal)) {
        return Shape::Test;
    }
    if icx.is(a, Term::LParen) {
        let close = icx.close(a);
        if kids.len() == 1 && !icx.is(close - 1, Term::Comma) {
            return pat_shape(cx, kids[0], t, mode);
        }
        let (dt, pm) = peel_ty(cx, t, mode);
        let ets = match cx.kind(dt) {
            TypeKind::Tuple(ts) => ts.to_vec(),
            _ => return no_shape(cx, p),
        };
        if kids.len() != ets.len() {
            return no_shape(cx, p);
        }
        let fields = ets.iter().zip(&kids).map(|(&et, &k)| (et, Sub::Pat(k))).collect();
        return Shape::Parts { ty: dt, mode: pm, variant: None, fields };
    }
    if !icx.is(a, Term::Ident) {
        return no_shape(cx, p);
    }
    let e = icx.chain_end(a);
    let has_paren = icx.is(e + 1, Term::LParen);
    let has_rec = icx.is(e + 1, Term::DotLBrace);
    if e == a && !has_paren && !has_rec {
        return Shape::Name(p);
    }
    let Some(Res::Decl(d)) = icx.uses(p).last().map(|u| u.res) else { return no_shape(cx, p) };
    let prog = cx.prog();
    let is_variant = match prog.decl(d).map(|x| x.kind) {
        Some(DeclKind::Variant) => true,
        Some(DeclKind::Struct) => false,
        Some(DeclKind::Const) => return Shape::Test,
        _ => return no_shape(cx, p),
    };
    let (dt, pm) = peel_ty(cx, t, mode);
    let vi = if is_variant { prog.decl(d).map_or(0, |x| x.index) } else { 0 };
    let Some(fts) = cx.fields(dt).and_then(|v| v.get(vi as usize).cloned()) else {
        return no_shape(cx, p);
    };
    let mut subs: Vec<Option<Sub>> = vec![None; fts.len()];
    if has_rec {
        let names: Vec<crate::ids::Sym> = match cx.kind(dt) {
            TypeKind::Adt { def, .. } => cx
                .sigs()
                .adt(def)
                .and_then(|ad| ad.variants.get(vi as usize))
                .map(|v| v.fields.iter().map(|f| f.name).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        for fp in kids {
            if icx.kind(fp) != Some(NodeKind::FieldPat) {
                continue;
            }
            let name = icx.sym(icx.a(fp));
            if let Some(fi) = names.iter().position(|&x| x == name)
                && fi < subs.len()
            {
                if subs[fi].is_some() {
                    cx.bug(fp, "a pattern names a field twice");
                }
                subs[fi] = Some(match icx.child(fp, NodeKind::Pattern) {
                    Some(sub) => Sub::Pat(sub),
                    None => Sub::Name(fp),
                });
            }
        }
    } else if has_paren {
        for (fi, &k) in kids.iter().enumerate().take(subs.len()) {
            subs[fi] = Some(Sub::Pat(k));
        }
    }
    // the type checker rejects a pattern that leaves a field out (E0208)
    let Some(subs) = subs.into_iter().collect::<Option<Vec<Sub>>>() else {
        cx.bug(p, "a pattern leaves a field out");
        return Shape::Test;
    };
    let fields = fts.into_iter().zip(subs).collect();
    Shape::Parts { ty: dt, mode: pm, variant: is_variant.then_some(vi), fields }
}

/// Whether sub-pattern `sub` of a field of type `ft` binds a non-Copy part by value.
fn sub_moves<'a, C: PatCx<'a> + ?Sized>(cx: &mut C, sub: Sub, ft: TypeId, mode: Mode) -> bool {
    match sub {
        Sub::Pat(q) => pat_moves(cx, q, ft, mode),
        Sub::Name(b) => mode == Mode::Move && cx.by_move(b),
    }
}

/// Whether pattern `p` against a value of type `t` binds a non-Copy part by value: such a
/// pattern moves its scrutinee, any other only reads it.
pub fn pat_moves<'a, C: PatCx<'a> + ?Sized>(cx: &mut C, p: u32, t: TypeId, mode: Mode) -> bool {
    match pat_shape(cx, p, t, mode) {
        Shape::Wild | Shape::Test => false,
        Shape::Name(b) => mode == Mode::Move && cx.by_move(b),
        Shape::Parts { mode, fields, .. } => {
            let mut any = false;
            for (ft, sub) in fields {
                any |= sub_moves(cx, sub, ft, mode);
            }
            any
        }
    }
}

/// The parts of a value of type `t` that pattern `p`, which moves it, does not bind by value:
/// the path of each from the value, in field order, a nested part in place of its field. Empty
/// when the pattern does not move the value. The pattern alone decides the list (a part whose
/// type has no drop glue is on it and drops nothing), so a part has the same index for the
/// definition and for every instance.
pub fn ignored_parts<'a, C: PatCx<'a> + ?Sized>(cx: &mut C, p: u32, t: TypeId) -> Vec<Vec<Step>> {
    let mut out = Vec::new();
    if pat_moves(cx, p, t, Mode::Move) {
        ignored(cx, p, t, Mode::Move, &mut Vec::new(), &mut out);
    }
    out
}

fn ignored<'a, C: PatCx<'a> + ?Sized>(
    cx: &mut C,
    p: u32,
    t: TypeId,
    mode: Mode,
    path: &mut Vec<Step>,
    out: &mut Vec<Vec<Step>>,
) {
    // a part behind a reference is not the scrutinee's
    if mode != Mode::Move {
        return;
    }
    if !pat_moves(cx, p, t, mode) {
        out.push(path.clone());
        return;
    }
    let Shape::Parts { ty, mode, variant, fields } = pat_shape(cx, p, t, mode) else { return };
    if mode != Mode::Move {
        return;
    }
    if cx.drop_impl(ty) {
        cx.bug(p, "a pattern takes apart a value whose type has a `Drop` impl");
        return;
    }
    for (fi, (ft, sub)) in fields.into_iter().enumerate() {
        path.push(Step { variant, field: fi as u32, ty: ft });
        match sub {
            Sub::Pat(q) => ignored(cx, q, ft, mode, path, out),
            Sub::Name(b) if mode == Mode::Move && cx.by_move(b) => {}
            Sub::Name(_) => out.push(path.clone()),
        }
        path.pop();
    }
}

/// Field types of the variants of ADT type `t` (struct: one variant).
pub fn adt_fields<I: Interner + ?Sized>(sigs: &Sigs, cx: &mut I, t: TypeId) -> Option<Vec<Vec<TypeId>>> {
    crate::mono::adt_fields(sigs, cx, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(point: Point, owner: Owner, head: Head) -> Row {
        Row { point, owner, head }
    }

    #[test]
    fn a_row_is_found_by_point_owner_and_head() {
        let mut s = Schedule::default();
        s.add(row(Point::End, Owner::Local(7), Head::Tuple(2)));
        s.add(row(Point::Try(40), Owner::Temp(12, 13), Head::Any));
        s.add(row(Point::Bind(9), Owner::Part(1), Head::Array));
        for sealed in [false, true] {
            if sealed {
                s.seal();
            }
            assert!(s.has(Point::End, Owner::Local(7), Head::Tuple(2)));
            // an instance's type under a definition's type parameter, and the reverse
            assert!(s.has(Point::Try(40), Owner::Temp(12, 13), Head::Array));
            assert!(s.has(Point::End, Owner::Local(7), Head::Any));
            assert!(!s.has(Point::End, Owner::Local(7), Head::Tuple(3)));
            assert!(!s.has(Point::End, Owner::Local(8), Head::Tuple(2)));
            assert!(!s.has(Point::Return(40), Owner::Temp(12, 13), Head::Any));
            assert!(!s.has(Point::Bind(9), Owner::Part(0), Head::Array));
        }
    }

    #[test]
    fn drops_of_one_point_keep_their_order() {
        let mut s = Schedule::default();
        let (a, b, c) = (Owner::Temp(3, 4), Owner::Local(5), Owner::Local(4));
        s.add(row(Point::Jump(20), a, Head::Any));
        s.add(row(Point::End, c, Head::Any));
        s.add(row(Point::Jump(20), a, Head::Any));
        s.add(row(Point::Jump(20), b, Head::Any));
        s.add(row(Point::Jump(20), c, Head::Any));
        s.seal();
        assert!(s.in_order(Point::Jump(20), &[a, a, b, c]));
        // a scheduled drop of a value that was moved out does not run
        assert!(s.in_order(Point::Jump(20), &[a, c]));
        assert!(s.in_order(Point::Jump(20), &[]));
        assert!(!s.in_order(Point::Jump(20), &[b, a]));
        assert!(!s.in_order(Point::Jump(20), &[a, a, a]));
        assert!(!s.in_order(Point::End, &[c, c]));
        assert!(!s.in_order(Point::Return(20), &[a]));
    }
}
