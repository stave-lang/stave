//! Function builder: a convenient, panic-free API for the front end's lowering.
//!
//! The builder keeps one instruction list per block, so instructions can be appended to any
//! block in any order; [`FuncBuilder::finish`] lays the body out as flat arrays in block order.
//! Blocks print in creation order (`b0` first), so the lowering creates blocks in the order of
//! the checked-IR construct table. `slot`s always go to the front of `b0`, in creation order.
//!
//! Misuse (emitting into a terminated block, no current block, ...) never panics: the first
//! error is recorded and returned by `finish`.

use crate::ir::{
    Block, BlockData, Body, CastKind, Const, ConstId, FuncId, Insts, IntrinsicId, Op, Operand, PanicKind,
    Site, Span, StrId, SwitchCase, Target, Term, Value, ValueData, ValueDef,
};
use crate::types::TypeId;

/// A builder misuse, reported by [`FuncBuilder::finish`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// An instruction or terminator was emitted with no current block.
    NoCurrentBlock,
    /// An instruction or second terminator was emitted into a terminated block.
    Terminated(Block),
    /// A block id that this builder did not create.
    UnknownBlock(Block),
    /// A block has no terminator at `finish`.
    Unterminated(Block),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::NoCurrentBlock => write!(f, "no current block"),
            BuildError::Terminated(b) => write!(f, "b{} is already terminated", b.0),
            BuildError::UnknownBlock(b) => write!(f, "unknown block b{}", b.0),
            BuildError::Unterminated(b) => write!(f, "b{} has no terminator", b.0),
        }
    }
}

impl std::error::Error for BuildError {}

#[derive(Clone, Copy)]
struct RawInst {
    op: Op,
    ty: Option<TypeId>,
    result: Option<Value>,
    args: Span,
    site: Site,
}

#[derive(Default)]
struct RawBlock {
    params: Vec<Value>,
    insts: Vec<RawInst>,
    term: Option<RawTerm>,
}

#[derive(Clone)]
enum RawTerm {
    Br(Target),
    Cbr(Operand, Target, Target),
    Switch(Operand, Vec<SwitchCase>, Target),
    Ret(Option<Operand>),
    Unreachable,
}

/// Builds one function body. Independent of the module, so bodies can be built in parallel.
pub struct FuncBuilder {
    blocks: Vec<RawBlock>,
    values: Vec<(TypeId, Option<(Block, u32)>)>,
    operands: Vec<Operand>,
    consts: Vec<Const>,
    cur: Option<Block>,
    site: Site,
    nslots: usize,
    /// The stores [`FuncBuilder::entry_store`] placed after the slots.
    nentry: usize,
    error: Option<BuildError>,
}

impl FuncBuilder {
    /// A builder whose entry block `b0` has one param per fn param; `b0` is current.
    pub fn new(params: &[TypeId]) -> FuncBuilder {
        let mut b = FuncBuilder {
            blocks: Vec::new(),
            values: Vec::new(),
            operands: Vec::new(),
            consts: Vec::new(),
            cur: None,
            site: Site::NONE,
            nslots: 0,
            nentry: 0,
            error: None,
        };
        let entry = b.create_block();
        for &t in params {
            b.add_param(entry, t);
        }
        b.cur = Some(entry);
        b
    }

    fn fail(&mut self, e: BuildError) {
        self.error.get_or_insert(e);
    }

    pub fn entry(&self) -> Block {
        Block(0)
    }

    /// The fn params (params of `b0`).
    pub fn params(&self) -> &[Value] {
        self.blocks.first().map_or(&[], |b| &b.params)
    }

    pub fn param(&self, i: usize) -> Option<Value> {
        self.params().get(i).copied()
    }

    /// Creates an empty block (not current).
    pub fn create_block(&mut self) -> Block {
        self.blocks.push(RawBlock::default());
        Block(self.blocks.len() as u32 - 1)
    }

    /// Creates a block with params of the given types.
    pub fn create_block_with(&mut self, tys: &[TypeId]) -> (Block, Vec<Value>) {
        let b = self.create_block();
        let vs = tys.iter().map(|&t| self.add_param(b, t)).collect();
        (b, vs)
    }

    /// Appends a param to block `b`.
    pub fn add_param(&mut self, b: Block, ty: TypeId) -> Value {
        let v = Value(self.values.len() as u32);
        match self.blocks.get_mut(b.index()) {
            Some(blk) => {
                self.values.push((ty, Some((b, blk.params.len() as u32))));
                blk.params.push(v);
            }
            None => {
                self.values.push((ty, None));
                self.fail(BuildError::UnknownBlock(b));
            }
        }
        v
    }

    pub fn block_params(&self, b: Block) -> &[Value] {
        self.blocks.get(b.index()).map_or(&[], |b| &b.params)
    }

    /// Makes `b` current.
    pub fn switch_to(&mut self, b: Block) {
        if b.index() >= self.blocks.len() {
            self.fail(BuildError::UnknownBlock(b));
            self.cur = None;
        } else {
            self.cur = Some(b);
        }
    }

    pub fn current(&self) -> Option<Block> {
        self.cur
    }

    /// True if the current block exists and has no terminator yet.
    pub fn is_open(&self) -> bool {
        self.cur.and_then(|b| self.blocks.get(b.index())).is_some_and(|b| b.term.is_none())
    }

    pub fn is_terminated(&self, b: Block) -> bool {
        self.blocks.get(b.index()).is_some_and(|b| b.term.is_some())
    }

    /// The site recorded on subsequently emitted instructions.
    pub fn set_site(&mut self, s: Site) {
        self.site = s;
    }

    /// The site [`FuncBuilder::set_site`] last set.
    pub fn site(&self) -> Site {
        self.site
    }

    pub fn value_type(&self, v: Value) -> Option<TypeId> {
        self.values.get(v.index()).map(|x| x.0)
    }

    // ---- constants ----

    pub fn konst(&mut self, c: Const) -> Operand {
        self.consts.push(c);
        Operand::Const(ConstId(self.consts.len() as u32 - 1))
    }

    /// An int constant (also bool `0`/`1` and char code points). Typed by its operand position.
    pub fn int(&mut self, v: i128) -> Operand {
        self.konst(Const::Int(v))
    }

    pub fn f32(&mut self, v: f32) -> Operand {
        self.konst(Const::F32(v.to_bits()))
    }

    pub fn f64(&mut self, v: f64) -> Operand {
        self.konst(Const::F64(v.to_bits()))
    }

    pub fn str(&self, s: StrId) -> Operand {
        Operand::Str(s)
    }

    pub fn func(&self, f: FuncId) -> Operand {
        Operand::Func(f)
    }

    // ---- instructions ----

    fn push_operands(&mut self, args: &[Operand]) -> Span {
        let start = self.operands.len() as u32;
        self.operands.extend_from_slice(args);
        Span { start, len: args.len() as u32 }
    }

    fn new_result(&mut self, ty: Option<TypeId>) -> Option<Value> {
        ty.map(|t| {
            self.values.push((t, None));
            Value(self.values.len() as u32 - 1)
        })
    }

    /// Emits a generic instruction into the current block. Returns the result value iff `ty` is
    /// `Some`.
    pub fn inst(&mut self, op: Op, ty: Option<TypeId>, args: &[Operand]) -> Option<Value> {
        let result = self.new_result(ty);
        let args = self.push_operands(args);
        let raw = RawInst { op, ty, result, args, site: self.site };
        match self.cur {
            None => self.fail(BuildError::NoCurrentBlock),
            Some(b) => match self.blocks.get_mut(b.index()) {
                None => self.fail(BuildError::UnknownBlock(b)),
                Some(blk) if blk.term.is_some() => self.fail(BuildError::Terminated(b)),
                Some(blk) => blk.insts.push(raw),
            },
        }
        result
    }

    /// Emits an instruction that has a result.
    pub fn op(&mut self, op: Op, ty: TypeId, args: &[Operand]) -> Value {
        match self.inst(op, Some(ty), args) {
            Some(v) => v,
            None => Value(u32::MAX),
        }
    }

    pub fn binary(&mut self, op: Op, ty: TypeId, a: impl Into<Operand>, b: impl Into<Operand>) -> Value {
        self.op(op, ty, &[a.into(), b.into()])
    }

    pub fn unary(&mut self, op: Op, ty: TypeId, a: impl Into<Operand>) -> Value {
        self.op(op, ty, &[a.into()])
    }

    /// A comparison (`eq ne lt le gt ge`) with result type `bool_ty`.
    pub fn cmp(&mut self, op: Op, bool_ty: TypeId, a: impl Into<Operand>, b: impl Into<Operand>) -> Value {
        self.op(op, bool_ty, &[a.into(), b.into()])
    }

    pub fn select(
        &mut self,
        ty: TypeId,
        c: impl Into<Operand>,
        a: impl Into<Operand>,
        b: impl Into<Operand>,
    ) -> Value {
        self.op(Op::Select, ty, &[c.into(), a.into(), b.into()])
    }

    pub fn cast(&mut self, kind: CastKind, ty: TypeId, x: impl Into<Operand>) -> Value {
        self.op(Op::Cast(kind), ty, &[x.into()])
    }

    /// A stack slot; `ptr_ty` is a `ref mut T` / `ptr mut T` type. Always placed at the front of
    /// `b0` (after earlier slots), whatever the current block.
    pub fn slot(&mut self, ptr_ty: TypeId) -> Value {
        self.slot_at(ptr_ty, usize::MAX)
    }

    /// The number of slots made so far.
    pub fn slot_count(&self) -> usize {
        self.nslots
    }

    /// A stack slot placed at position `at` among the slots of `b0` (the end when there are
    /// fewer), whatever the current block.
    pub fn slot_at(&mut self, ptr_ty: TypeId, at: usize) -> Value {
        let v = Value(self.values.len() as u32);
        self.values.push((ptr_ty, None));
        let raw = RawInst {
            op: Op::Slot,
            ty: Some(ptr_ty),
            result: Some(v),
            args: Span::default(),
            site: self.site,
        };
        if let Some(b0) = self.blocks.first_mut() {
            let n = self.nslots.min(b0.insts.len());
            b0.insts.insert(at.min(n), raw);
            self.nslots = n + 1;
        }
        v
    }

    /// `store p, v` placed in `b0` right after the slots and the entry stores made before it,
    /// whatever the current block: the value slot `p` holds when the fn is entered.
    pub fn entry_store(&mut self, p: impl Into<Operand>, v: impl Into<Operand>) {
        let args = self.push_operands(&[p.into(), v.into()]);
        let raw = RawInst { op: Op::Store, ty: None, result: None, args, site: self.site };
        if let Some(b0) = self.blocks.first_mut() {
            let at = (self.nslots + self.nentry).min(b0.insts.len());
            b0.insts.insert(at, raw);
            self.nentry += 1;
        }
    }

    pub fn load(&mut self, ty: TypeId, p: impl Into<Operand>) -> Value {
        self.op(Op::Load, ty, &[p.into()])
    }

    pub fn store(&mut self, p: impl Into<Operand>, v: impl Into<Operand>) {
        self.inst(Op::Store, None, &[p.into(), v.into()]);
    }

    /// A direct call; `ret` = the callee's return type.
    pub fn call(&mut self, f: FuncId, ret: Option<TypeId>, args: &[Operand]) -> Option<Value> {
        let mut all = Vec::with_capacity(args.len() + 1);
        all.push(Operand::Func(f));
        all.extend_from_slice(args);
        self.inst(Op::Call, ret, &all)
    }

    /// The fn value `(code, env)` of closure `f` (type `fn_ty`; `f`'s last param is `env`).
    pub fn fn_make(&mut self, fn_ty: TypeId, f: FuncId, env: impl Into<Operand>) -> Value {
        self.op(Op::FnMake, fn_ty, &[Operand::Func(f), env.into()])
    }

    /// A call through a fn value.
    pub fn call_ind(
        &mut self,
        f: impl Into<Operand>,
        ret: Option<TypeId>,
        args: &[Operand],
    ) -> Option<Value> {
        let mut all = Vec::with_capacity(args.len() + 1);
        all.push(f.into());
        all.extend_from_slice(args);
        self.inst(Op::CallInd, ret, &all)
    }

    pub fn intrinsic(&mut self, i: IntrinsicId, ret: Option<TypeId>, args: &[Operand]) -> Option<Value> {
        self.inst(Op::Intrinsic(i), ret, args)
    }

    /// An unconditional trap. The block still needs a terminator (normally `unreachable`).
    pub fn panic(&mut self, kind: PanicKind) {
        self.inst(Op::Panic(kind), None, &[]);
    }

    /// `check.bounds i n`.
    pub fn bounds_check(&mut self, i: impl Into<Operand>, n: impl Into<Operand>) {
        self.inst(Op::BoundsCheck, None, &[i.into(), n.into()]);
    }

    // ---- terminators ----

    fn target(&mut self, b: Block, args: &[Operand]) -> Target {
        if b.index() >= self.blocks.len() {
            self.fail(BuildError::UnknownBlock(b));
        }
        Target { block: b, args: self.push_operands(args) }
    }

    fn terminate(&mut self, t: RawTerm) {
        match self.cur {
            None => self.fail(BuildError::NoCurrentBlock),
            Some(b) => match self.blocks.get_mut(b.index()) {
                None => self.fail(BuildError::UnknownBlock(b)),
                Some(blk) if blk.term.is_some() => self.fail(BuildError::Terminated(b)),
                Some(blk) => blk.term = Some(t),
            },
        }
    }

    pub fn br(&mut self, b: Block, args: &[Operand]) {
        let t = self.target(b, args);
        self.terminate(RawTerm::Br(t));
    }

    pub fn cbr(
        &mut self,
        cond: impl Into<Operand>,
        t: Block,
        targs: &[Operand],
        e: Block,
        eargs: &[Operand],
    ) {
        let t = self.target(t, targs);
        let e = self.target(e, eargs);
        self.terminate(RawTerm::Cbr(cond.into(), t, e));
    }

    /// `switch scrut (value: target(args))* _: default(args)`.
    pub fn switch(
        &mut self,
        scrut: impl Into<Operand>,
        cases: &[(i128, Block, &[Operand])],
        default: Block,
        dargs: &[Operand],
    ) {
        let cs: Vec<SwitchCase> =
            cases.iter().map(|(v, b, a)| SwitchCase { value: *v, target: self.target(*b, a) }).collect();
        let d = self.target(default, dargs);
        self.terminate(RawTerm::Switch(scrut.into(), cs, d));
    }

    pub fn ret(&mut self, v: Option<Operand>) {
        self.terminate(RawTerm::Ret(v));
    }

    pub fn unreachable(&mut self) {
        self.terminate(RawTerm::Unreachable);
    }

    /// Ends with `unreachable` every open block that control cannot reach from the entry: a
    /// block made for code after an operand that turned out to diverge. An open block that is
    /// reachable stays an error of [`FuncBuilder::finish`].
    pub fn seal_unreached(&mut self) {
        let mut seen = vec![false; self.blocks.len()];
        let mut work: Vec<usize> = Vec::new();
        if let Some(s) = seen.first_mut() {
            *s = true;
            work.push(0);
        }
        let mut succs: Vec<Block> = Vec::new();
        while let Some(b) = work.pop() {
            succs.clear();
            match &self.blocks[b].term {
                Some(RawTerm::Br(t)) => succs.push(t.block),
                Some(RawTerm::Cbr(_, t, e)) => succs.extend([t.block, e.block]),
                Some(RawTerm::Switch(_, cases, d)) => {
                    succs.extend(cases.iter().map(|c| c.target.block));
                    succs.push(d.block);
                }
                Some(RawTerm::Ret(_) | RawTerm::Unreachable) | None => {}
            }
            for s in &succs {
                if let Some(x) = seen.get_mut(s.index())
                    && !*x
                {
                    *x = true;
                    work.push(s.index());
                }
            }
        }
        for (blk, reached) in self.blocks.iter_mut().zip(seen) {
            if !reached && blk.term.is_none() {
                blk.term = Some(RawTerm::Unreachable);
            }
        }
    }

    /// Lays out the body. Errors on recorded misuse or an unterminated block.
    pub fn finish(self) -> Result<Body, BuildError> {
        if let Some(e) = self.error {
            return Err(e);
        }
        let mut body = Body {
            operands: self.operands,
            consts: self.consts,
            values: self
                .values
                .iter()
                .map(|(t, p)| ValueData {
                    ty: *t,
                    def: match p {
                        Some((b, i)) => ValueDef::Param { block: *b, index: *i },
                        None => ValueDef::Inst(u32::MAX),
                    },
                })
                .collect(),
            ..Body::default()
        };
        let mut insts = Insts::default();
        let n: usize = self.blocks.iter().map(|b| b.insts.len()).sum();
        insts.op.reserve(n);
        insts.ty.reserve(n);
        insts.result.reserve(n);
        insts.args.reserve(n);
        insts.site.reserve(n);
        body.blocks.reserve(self.blocks.len());
        for (bi, blk) in self.blocks.into_iter().enumerate() {
            let Some(term) = blk.term else { return Err(BuildError::Unterminated(Block(bi as u32))) };
            let params = Span { start: body.block_params.len() as u32, len: blk.params.len() as u32 };
            body.block_params.extend_from_slice(&blk.params);
            let start = insts.op.len() as u32;
            for r in &blk.insts {
                let idx = insts.op.len() as u32;
                if let Some(v) = r.result
                    && let Some(vd) = body.values.get_mut(v.index())
                {
                    vd.def = ValueDef::Inst(idx);
                }
                insts.op.push(r.op);
                insts.ty.push(r.ty);
                insts.result.push(r.result);
                insts.args.push(r.args);
                insts.site.push(r.site);
            }
            let ispan = Span { start, len: blk.insts.len() as u32 };
            let term = match term {
                RawTerm::Br(t) => Term::Br(t),
                RawTerm::Cbr(c, t, e) => Term::Cbr { cond: c, then: t, els: e },
                RawTerm::Switch(s, cs, d) => {
                    let span = Span { start: body.cases.len() as u32, len: cs.len() as u32 };
                    body.cases.extend(cs);
                    Term::Switch { scrut: s, cases: span, default: d }
                }
                RawTerm::Ret(v) => Term::Ret(v),
                RawTerm::Unreachable => Term::Unreachable,
            };
            body.blocks.push(BlockData { params, insts: ispan, term });
        }
        body.insts = insts;
        Ok(body)
    }
}
