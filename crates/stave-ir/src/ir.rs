//! IR data structures (contracts/checked-ir.md).
//!
//! SSA with block parameters (no phi nodes). Per function, everything is flat arrays indexed by
//! `u32` ids: values, instructions (struct of arrays), operands, constants, switch cases.
//!
//! # Op set (`IrOp` names, as printed)
//!
//! Integer ops take the signedness from the result type; trapping ops record their trap kind and
//! are single instructions (no blocks):
//!
//! | Name | Operands | Semantics |
//! |---|---|---|
//! | `add sub mul` | `a b` | int: trap `Overflow`; float: IEEE |
//! | `div rem` | `a b` | int: `b == 0` traps `DivByZero`, `MIN / -1` and `MIN % -1` trap `Overflow`; float: IEEE, `rem` = truncated (fmod) |
//! | `neg` | `a` | signed int: `-MIN` traps `Overflow`; float: sign flip |
//! | `add.wrap sub.wrap mul.wrap` | `a b` | `+% -% *%` |
//! | `add.sat sub.sat mul.sat` | `a b` | `+\| -\| *\|` |
//! | `shl shr` | `a amount` | amount (any int type) ≥ bits traps `Overflow`; `shr` arithmetic iff signed |
//! | `shl.wrap` | `a amount` | `<<%`: amount masked by bits-1 |
//! | `and or xor` / `not` | `a b` / `a` | bitwise on ints, logical on bool |
//! | `popcount clz ctz` | `a` | bit counts within `a`'s type width; result type = operand type |
//! | `mul.wide` | `a b` | i32/i64/u32/u64 → `tuple (hi, lo)`: the exact double-width product; `hi` signed iff the type is, `lo` its low bits |
//! | `eq ne lt le gt ge` | `a b` | result bool; ints by signedness, floats IEEE (unordered: only `ne` true) |
//! | `select` | `c a b` | `c ? a : b` for a `bool` `c` and any one type of `a`, `b`; branch-free in every backend |
//! | `cast cast.wrap cast.sat cast.round cast.try` | `x` | `as as% as\| as~ as?`; `cast.try` yields the `Option` instance |
//! | `make` | elems.. | tuple / struct / array value |
//! | `get` / `set` | `agg idx` / `agg idx v` | field or element; a dynamic array index traps `Bounds` |
//! | `enum.make` / `enum.tag` / `enum.get` | `v fields..` / `e` / `e v f` | enum value, its variant index, a variant field |
//! | `slot` | — | stack slot of the pointee type, in `b0` only; result is a pointer |
//! | `load` / `store` | `p` / `p v` | memory read / write |
//! | `addr.field` / `addr.enum_field` / `addr.elem` | `p i` / `p v f` / `p i` | address arithmetic, unchecked |
//! | `slice.make` / `slice.len` / `slice.ptr` | `p n` / `s` / `s` | fat pointer (`&[T]`, `&str`) parts |
//! | `check.bounds` | `i n` | `i ≥ n` (unsigned) traps `Bounds` |
//! | `mem.prefetch_<rw>_<loc>` | `p i` | a hint for the address `p + i * size` (wrapping, as `addr.elem`); no access, never traps |
//! | `mem.store_stream` / `mem.fence_stream` | `p v` / — | a `store` with a non-temporal hint / orders earlier streaming stores |
//! | `fn.make` | `@f env` | fn value `(code, environment)` of closure `@f` (its last param is `env`) |
//! | `call` / `call.ind` | `@f args..` / `f args..` | direct / fn-value call (`call.ind` passes the value's environment as a trailing argument) |
//! | `intrinsic.<name>` | args.. | contracts/intrinsics.md |
//! | `panic.<kind>` | — | unconditional trap (`panic.unwrap`, `panic.assert`, ...) |
//! | `vec.<name>` / `mask.<name>` | | vector and mask ops ([`crate::simd`], `Op::Vec`) |

use stave_target::FeatureSet;

use crate::simd::VecOp;
use crate::types::{GenericArg, TypeId, TypeTable};

macro_rules! id {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id!(
    /// An SSA value of one function.
    Value
);
id!(
    /// A block of one function. `Block(0)` is the entry.
    Block
);
id!(
    /// A function of the module (index into `Module::funcs`).
    FuncId
);
id!(
    /// A string of the module (canonical string id S); printed `s<n>`.
    StrId
);
id!(
    /// A constant in a function's constant pool.
    ConstId
);
id!(
    /// An intrinsic name of the module (index into `Module::intrinsics`).
    IntrinsicId
);
id!(
    /// An opaque source site (file, line, column, fn) for trap reports; owned by the front end.
    Site
);

impl Site {
    pub const NONE: Site = Site(u32::MAX);
}

/// The canonical instance id: `41`, or `41.2` for a closure / `spawn` body, or `41.u` for the
/// unchecked entry of `41` (its body without the `requires` checks, made by the witness
/// checker). Lexicographic order is the print order (a closure follows its enclosing fn, the
/// unchecked entry follows the closures).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct InstanceId(pub Vec<u32>);

impl InstanceId {
    /// The last number of an unchecked entry's id, printed `u`.
    pub const UNCHECKED_ENTRY: u32 = u32::MAX;
}

impl std::fmt::Display for InstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, n) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(".")?;
            }
            if i > 0 && *n == InstanceId::UNCHECKED_ENTRY {
                f.write_str("u")?;
            } else {
                write!(f, "{n}")?;
            }
        }
        Ok(())
    }
}

/// Panic kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PanicKind {
    Bounds,
    Overflow,
    DivByZero,
    Assert,
    OutOfMemory,
    StackOverflow,
    Unwrap,
    /// The CPU lacks a feature of the build target's baseline (runtime start code only).
    UnsupportedCpu,
    /// A wait that can never end, e.g. a `Once` init reached again from its own initializer.
    Deadlock,
    /// A false `requires` / `ensures` clause. Lowering reports it through
    /// `runtime.panic.requires_failed` / `ensures_failed` (which print the clause), not this op.
    Contract,
}

impl PanicKind {
    pub const ALL: [PanicKind; 10] = [
        PanicKind::Bounds,
        PanicKind::Overflow,
        PanicKind::DivByZero,
        PanicKind::Assert,
        PanicKind::OutOfMemory,
        PanicKind::StackOverflow,
        PanicKind::Unwrap,
        PanicKind::UnsupportedCpu,
        PanicKind::Deadlock,
        PanicKind::Contract,
    ];

    /// The report name (`panic: <Kind> at ...`).
    pub fn name(self) -> &'static str {
        match self {
            PanicKind::Bounds => "Bounds",
            PanicKind::Overflow => "Overflow",
            PanicKind::DivByZero => "DivByZero",
            PanicKind::Assert => "Assert",
            PanicKind::OutOfMemory => "OutOfMemory",
            PanicKind::StackOverflow => "StackOverflow",
            PanicKind::Unwrap => "Unwrap",
            PanicKind::UnsupportedCpu => "UnsupportedCpu",
            PanicKind::Deadlock => "Deadlock",
            PanicKind::Contract => "Contract",
        }
    }

    /// The op name of `panic` with this kind.
    pub fn op_name(self) -> &'static str {
        match self {
            PanicKind::Bounds => "panic.bounds",
            PanicKind::Overflow => "panic.overflow",
            PanicKind::DivByZero => "panic.div_by_zero",
            PanicKind::Assert => "panic.assert",
            PanicKind::OutOfMemory => "panic.out_of_memory",
            PanicKind::StackOverflow => "panic.stack_overflow",
            PanicKind::Unwrap => "panic.unwrap",
            PanicKind::UnsupportedCpu => "panic.unsupported_cpu",
            PanicKind::Deadlock => "panic.deadlock",
            PanicKind::Contract => "panic.contract",
        }
    }
}

/// The cache level a prefetch targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefetchLoc {
    L1,
    L2,
    L3,
    /// Non-temporal: data used once, kept from displacing other cached data.
    Nta,
}

/// A prefetch hint: read or write intent and the cache level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Prefetch {
    pub write: bool,
    pub loc: PrefetchLoc,
}

impl Prefetch {
    /// Every hint, read hints first, in the order `l1 l2 l3 nta`.
    pub const ALL: [Prefetch; 8] = {
        use PrefetchLoc::*;
        [
            Prefetch { write: false, loc: L1 },
            Prefetch { write: false, loc: L2 },
            Prefetch { write: false, loc: L3 },
            Prefetch { write: false, loc: Nta },
            Prefetch { write: true, loc: L1 },
            Prefetch { write: true, loc: L2 },
            Prefetch { write: true, loc: L3 },
            Prefetch { write: true, loc: Nta },
        ]
    };

    /// The op name, also the source method name after `mem.` (`mem.prefetch_read_l1`).
    pub fn name(self) -> &'static str {
        use PrefetchLoc::*;
        match (self.write, self.loc) {
            (false, L1) => "mem.prefetch_read_l1",
            (false, L2) => "mem.prefetch_read_l2",
            (false, L3) => "mem.prefetch_read_l3",
            (false, Nta) => "mem.prefetch_read_nta",
            (true, L1) => "mem.prefetch_write_l1",
            (true, L2) => "mem.prefetch_write_l2",
            (true, L3) => "mem.prefetch_write_l3",
            (true, Nta) => "mem.prefetch_write_nta",
        }
    }

    /// The hint of source method `name` (`prefetch_read_l1`).
    pub fn from_method(name: &str) -> Option<Prefetch> {
        Prefetch::ALL.into_iter().find(|p| p.name().strip_prefix("mem.") == Some(name))
    }
}

/// The `as` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastKind {
    /// `as`: lossless for all values.
    Exact,
    /// `as%`: wraps / truncates (int to int).
    Wrap,
    /// `as|`: saturates (float to int: NaN → 0).
    Sat,
    /// `as~`: rounds to nearest even (to a float type).
    Round,
    /// `as?`: `Option.Some(v)` iff the value converts exactly, else `Option.None`.
    Try,
}

/// Instruction opcodes. See the module doc for operands and semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    AddWrap,
    SubWrap,
    MulWrap,
    ShlWrap,
    AddSat,
    SubSat,
    MulSat,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Not,
    Popcount,
    Clz,
    Ctz,
    MulWide,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Select,
    Cast(CastKind),
    Make,
    Get,
    Set,
    EnumMake,
    EnumTag,
    EnumGet,
    Slot,
    Load,
    Store,
    FieldAddr,
    EnumFieldAddr,
    ElemAddr,
    SliceMake,
    SliceLen,
    SlicePtr,
    FnMake,
    BoundsCheck,
    /// A prefetch hint: no access, no effect on results; kept in program order.
    Prefetch(Prefetch),
    /// A store with a non-temporal hint (same semantics as `Store`).
    StoreStream,
    /// Orders the streaming stores before it with every later store.
    FenceStream,
    Call,
    CallInd,
    Intrinsic(IntrinsicId),
    /// A vector or mask op.
    Vec(VecOp),
    Panic(PanicKind),
}

impl Op {
    /// The printed op name; `None` for `Intrinsic` (printed `intrinsic.<name>` from the module).
    pub fn name(self) -> Option<&'static str> {
        Some(match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Div => "div",
            Op::Rem => "rem",
            Op::Neg => "neg",
            Op::AddWrap => "add.wrap",
            Op::SubWrap => "sub.wrap",
            Op::MulWrap => "mul.wrap",
            Op::ShlWrap => "shl.wrap",
            Op::AddSat => "add.sat",
            Op::SubSat => "sub.sat",
            Op::MulSat => "mul.sat",
            Op::Shl => "shl",
            Op::Shr => "shr",
            Op::And => "and",
            Op::Or => "or",
            Op::Xor => "xor",
            Op::Not => "not",
            Op::Popcount => "popcount",
            Op::Clz => "clz",
            Op::Ctz => "ctz",
            Op::MulWide => "mul.wide",
            Op::Eq => "eq",
            Op::Ne => "ne",
            Op::Lt => "lt",
            Op::Le => "le",
            Op::Gt => "gt",
            Op::Ge => "ge",
            Op::Select => "select",
            Op::Cast(CastKind::Exact) => "cast",
            Op::Cast(CastKind::Wrap) => "cast.wrap",
            Op::Cast(CastKind::Sat) => "cast.sat",
            Op::Cast(CastKind::Round) => "cast.round",
            Op::Cast(CastKind::Try) => "cast.try",
            Op::Make => "make",
            Op::Get => "get",
            Op::Set => "set",
            Op::EnumMake => "enum.make",
            Op::EnumTag => "enum.tag",
            Op::EnumGet => "enum.get",
            Op::Slot => "slot",
            Op::Load => "load",
            Op::Store => "store",
            Op::FieldAddr => "addr.field",
            Op::EnumFieldAddr => "addr.enum_field",
            Op::ElemAddr => "addr.elem",
            Op::SliceMake => "slice.make",
            Op::SliceLen => "slice.len",
            Op::SlicePtr => "slice.ptr",
            Op::FnMake => "fn.make",
            Op::BoundsCheck => "check.bounds",
            Op::Prefetch(p) => p.name(),
            Op::StoreStream => "mem.store_stream",
            Op::FenceStream => "mem.fence_stream",
            Op::Call => "call",
            Op::CallInd => "call.ind",
            Op::Panic(k) => k.op_name(),
            Op::Vec(v) => v.name(),
            Op::Intrinsic(_) => return None,
        })
    }

    /// The trap kinds this op can raise.
    pub fn trap_kinds(self) -> &'static [PanicKind] {
        use PanicKind as P;
        match self {
            Op::Add | Op::Sub | Op::Mul | Op::Neg | Op::Shl | Op::Shr => &[P::Overflow],
            Op::Div | Op::Rem => &[P::DivByZero, P::Overflow],
            Op::Get | Op::Set | Op::BoundsCheck => &[P::Bounds],
            Op::Call | Op::CallInd | Op::Slot => &[P::StackOverflow],
            Op::Panic(P::Bounds) => &[P::Bounds],
            Op::Panic(P::Overflow) => &[P::Overflow],
            Op::Panic(P::DivByZero) => &[P::DivByZero],
            Op::Panic(P::Assert) => &[P::Assert],
            Op::Panic(P::OutOfMemory) => &[P::OutOfMemory],
            Op::Panic(P::StackOverflow) => &[P::StackOverflow],
            Op::Panic(P::Unwrap) => &[P::Unwrap],
            Op::Panic(P::UnsupportedCpu) => &[P::UnsupportedCpu],
            Op::Panic(P::Deadlock) => &[P::Deadlock],
            Op::Panic(P::Contract) => &[P::Contract],
            Op::Vec(v) => v.trap_kinds(),
            _ => &[],
        }
    }
}

/// A constant in a function's pool. Ints are typed by context (the operand position).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Const {
    Int(i128),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
}

/// An instruction or terminator operand. Constants are inline (never values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    Value(Value),
    Const(ConstId),
    /// A string literal (`&str` to static bytes).
    Str(StrId),
    /// A fn item used as a value (code, null environment) or as a direct callee.
    Func(FuncId),
    /// A type operand.
    Type(TypeId),
}

impl From<Value> for Operand {
    fn from(v: Value) -> Operand {
        Operand::Value(v)
    }
}

/// A `u32` range into one of a body's pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

impl Span {
    /// The slice of `v` this span covers; empty if it is out of range.
    pub fn of<T>(self, v: &[T]) -> &[T] {
        let s = self.start as usize;
        v.get(s..s.saturating_add(self.len as usize)).unwrap_or(&[])
    }

    pub fn in_range(self, n: usize) -> bool {
        (self.start as usize).checked_add(self.len as usize).is_some_and(|e| e <= n)
    }

    pub fn indices(self) -> std::ops::Range<u32> {
        self.start..self.start.saturating_add(self.len)
    }
}

/// A branch target with its block arguments (a span of the operand pool).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub block: Block,
    pub args: Span,
}

/// One `switch` case: an integer constant and its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwitchCase {
    pub value: i128,
    pub target: Target,
}

/// Block terminators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    Br(Target),
    Cbr {
        cond: Operand,
        then: Target,
        els: Target,
    },
    /// Cases are a span of `Body::cases`.
    Switch {
        scrut: Operand,
        cases: Span,
        default: Target,
    },
    Ret(Option<Operand>),
    Unreachable,
}

/// A block: its params (span of `Body::block_params`), its instructions (span of the
/// instruction arrays), and its terminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockData {
    pub params: Span,
    pub insts: Span,
    pub term: Term,
}

/// Where a value is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueDef {
    Param { block: Block, index: u32 },
    Inst(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueData {
    pub ty: TypeId,
    pub def: ValueDef,
}

/// Instructions, struct of arrays. Instruction `i` belongs to the block whose `insts` span
/// contains `i`; blocks' spans are disjoint and in block order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Insts {
    pub op: Vec<Op>,
    /// Result type; `None` for no result (printed `()`).
    pub ty: Vec<Option<TypeId>>,
    pub result: Vec<Option<Value>>,
    /// Operands, a span of `Body::operands`.
    pub args: Vec<Span>,
    pub site: Vec<Site>,
}

impl Insts {
    pub fn len(&self) -> usize {
        self.op.len()
    }

    pub fn is_empty(&self) -> bool {
        self.op.is_empty()
    }
}

/// A function body.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Body {
    pub blocks: Vec<BlockData>,
    pub block_params: Vec<Value>,
    pub insts: Insts,
    pub operands: Vec<Operand>,
    pub consts: Vec<Const>,
    pub cases: Vec<SwitchCase>,
    pub values: Vec<ValueData>,
    /// Borrow events the instructions do not show (contracts/borrow-model.md, "Side table").
    /// Empty unless the lowering was asked for them; not part of the canonical dump.
    pub borrows: Vec<BorrowMark>,
    /// The other events of the side table, by the instruction that carries each.
    pub events: Vec<EventMark>,
}

impl Body {
    pub fn block(&self, b: Block) -> Option<&BlockData> {
        self.blocks.get(b.index())
    }

    pub fn params(&self, b: Block) -> &[Value] {
        self.block(b).map_or(&[], |d| d.params.of(&self.block_params))
    }

    /// Operands of instruction `i`.
    pub fn args(&self, i: u32) -> &[Operand] {
        self.insts.args.get(i as usize).map_or(&[], |s| s.of(&self.operands))
    }

    pub fn target_args(&self, t: &Target) -> &[Operand] {
        t.args.of(&self.operands)
    }

    pub fn switch_cases(&self, cases: Span) -> &[SwitchCase] {
        cases.of(&self.cases)
    }

    pub fn value_ty(&self, v: Value) -> Option<TypeId> {
        self.values.get(v.index()).map(|d| d.ty)
    }

    pub fn konst(&self, c: ConstId) -> Option<Const> {
        self.consts.get(c.index()).copied()
    }

    /// The int constant an operand denotes, if it is one.
    pub fn int_const(&self, o: Operand) -> Option<i128> {
        match o {
            Operand::Const(c) => match self.konst(c) {
                Some(Const::Int(v)) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    /// Successor targets of a block's terminator, in order (cases, then default).
    pub fn successors(&self, b: Block) -> Vec<Target> {
        let Some(d) = self.block(b) else { return Vec::new() };
        match d.term {
            Term::Br(t) => vec![t],
            Term::Cbr { then, els, .. } => vec![then, els],
            Term::Switch { cases, default, .. } => {
                let mut v: Vec<Target> = self.switch_cases(cases).iter().map(|c| c.target).collect();
                v.push(default);
                v
            }
            Term::Ret(_) | Term::Unreachable => Vec::new(),
        }
    }

    /// Calls `f` with every successor target of block `b`, in the order of [`Body::successors`],
    /// without building the list.
    pub fn for_each_successor(&self, b: Block, mut f: impl FnMut(Target)) {
        let Some(d) = self.block(b) else { return };
        match d.term {
            Term::Br(t) => f(t),
            Term::Cbr { then, els, .. } => {
                f(then);
                f(els);
            }
            Term::Switch { cases, default, .. } => {
                for c in self.switch_cases(cases) {
                    f(c.target);
                }
                f(default);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }

    /// Blocks reachable from `b0`, as a flag per block.
    pub fn reachable(&self) -> Vec<bool> {
        let n = self.blocks.len();
        let mut seen = vec![false; n];
        if n == 0 {
            return seen;
        }
        let mut stack = vec![Block(0)];
        seen[0] = true;
        while let Some(b) = stack.pop() {
            self.for_each_successor(b, |t| {
                if let Some(s) = seen.get_mut(t.block.index())
                    && !*s
                {
                    *s = true;
                    stack.push(t.block);
                }
            });
        }
        seen
    }
}

/// A borrow of the side table: the result of the instruction that defines `value` is a new
/// reference to the place its address names, reached through the tag that address carries.
/// The place is the pointee of the value's type; of a fat reference, its elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BorrowMark {
    pub value: Value,
    /// `&mut` (else `&`).
    pub mutable: bool,
}

/// The `from` clauses of a function: what its signature says of the references its result
/// holds and of those it stores into the views its params point to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Clauses {
    pub kind: ClauseKind,
    /// The result's clause: (source param, `copy`).
    pub result: Vec<(u32, bool)>,
    /// The params' clauses: (target param, source param, `copy`).
    pub params: Vec<(u32, u32, bool)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClauseKind {
    /// No signature to check a call against (drop glue).
    #[default]
    Unchecked,
    /// A fn item or method: its clauses are the rows.
    Item,
    /// A closure body: no clause; every param and capture is a source of its result.
    Closure,
}

/// An event of the side table that is no borrow: it happens when the instruction that defines
/// `value` runs. For the kinds with a place, `value` is a pointer to it and the place is the
/// pointee of the value's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EventMark {
    pub value: Value,
    pub kind: EventKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A local or temporary begins in the slot the place is: the allocation that was there
    /// ends, with every tag of it.
    Begin,
    /// The local or temporary in the slot ends.
    End,
    /// The value left the place (a move, or the end of a drop): the bytes end through the tag
    /// of the pointer and are moved-out.
    Gone,
    /// An array repeat literal makes a second owner of a value that is not Copy.
    Repeat,
    /// `fn.make` is to copy the environment the pointer names into a slot of its own, so that
    /// each evaluation of a closure literal holds its own capture references.
    Env,
}

/// A function: signature plus an optional body (`None` = declared only, e.g. an `extern "C"`
/// import; not printed, not executable by the interpreter).
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub instance: InstanceId,
    /// Module path and item name joined by `.`.
    pub path: String,
    /// Generic arguments of the instance: types and const values (printed `<t3,t9,4>`).
    pub type_args: Vec<GenericArg>,
    pub params: Vec<TypeId>,
    /// `None` = unit.
    pub ret: Option<TypeId>,
    pub features: FeatureSet,
    pub body: Option<Body>,
    /// Site of the fn's name token (the PC-table row of contracts/intrinsics.md); `Site::NONE`
    /// if unknown.
    pub name_site: Site,
    /// The link symbol of an `extern "C"` import or export: its fn name, unmangled (sema
    /// rejects clashes and reserved names, E0812). `None` for Stave fns, named by [`crate::link`].
    pub symbol: Option<String>,
    /// The compiler-known item this function is an instance of (the front end resolved it in a
    /// built-in package); not printed. Stages find known functions by it, never by `path`.
    pub known: Option<crate::Known>,
    /// The function runs on its thread's alternate signal stack: its entry moves the stack
    /// pointer there unless it is there already (contracts/intrinsics.md "Alternate stack").
    /// Set by lowering from [`crate::known::reports_on_alt_stack`]; not printed.
    pub alt_stack: bool,
}

/// A source site row (contracts/intrinsics.md site table): what a [`Site`] indexes in
/// [`Module::sites`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SiteInfo {
    /// The file path diagnostics use.
    pub file: String,
    /// 1-based line.
    pub line: u32,
    /// 1-based column, in bytes.
    pub col: u32,
    /// Checked-IR path of the enclosing fn.
    pub func: String,
}

/// A `static` or `static mut` global: zero or more initialized bytes (little-endian, the [`crate::layout`]
/// layout of `ty`), addressed by `intrinsic.static_addr <index>` (result: a pointer to `ty`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Static {
    pub path: String,
    pub ty: TypeId,
    /// Exactly the type's size in bytes.
    pub init: Vec<u8>,
    /// The address words inside `init`, by ascending offset: each 8-byte word at `offset` holds
    /// the address of other program data (its bytes in `init` are zero). Every target resolves
    /// them when it lays out the image, so a static never runs code to initialize.
    pub refs: Vec<DataRef>,
}

/// An absolute 64-bit address inside a static's bytes: `target`'s first byte plus `addend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataRef {
    /// Byte offset of the word in [`Static::init`]; the word is 8-aligned in the static.
    pub offset: u32,
    pub target: DataTarget,
    /// At most the target's size (one past the end is a valid address).
    pub addend: u32,
}

/// What a [`DataRef`] points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataTarget {
    /// The bytes of string constant `s<S>` (read-only).
    Str(StrId),
    /// Static number `i` (`intrinsic.static_addr i`).
    Static(u32),
}

/// A whole program (or the part of it being printed / run).
#[derive(Clone, Debug, Default)]
pub struct Module {
    pub types: TypeTable,
    pub funcs: Vec<Func>,
    /// String bytes by `StrId`.
    pub strings: Vec<Vec<u8>>,
    /// Intrinsic names by `IntrinsicId` (`syscall3`, `cpuid`, ...).
    pub intrinsics: Vec<String>,
    /// The site table: row `i` is what `Site(i)` means.
    pub sites: Vec<SiteInfo>,
    /// `static` and `static mut` globals by index (`intrinsic.static_addr <index>`).
    pub statics: Vec<Static>,
    /// Per function, the summary the witness checker certified when it removed the checks
    /// ([`crate::witness::check::remove`]; empty before that). A check removed by a fact about a
    /// call is no longer provable from the bodies alone once the callee's own checks are gone:
    /// the bounds audit reads calls with these rows.
    pub stated: Vec<Option<std::sync::Arc<crate::witness::Summary>>>,
    /// The declaration of `std.cell.UnsafeCell` (`None`: the program has none, or it is
    /// unknown): the bytes of a value of such a type are not read-only behind a `&`. The
    /// bounds pass reads a field behind a `&` parameter as one value only when this is known.
    pub cell: Option<crate::types::DeclId>,
    /// Per `static`, in a module lowered with the borrow side table: it is not `static mut`,
    /// so nothing writes its bytes outside an `UnsafeCell`.
    pub frozen: Vec<bool>,
    /// Per function, in a module lowered with the borrow side table: its `from` clauses.
    pub clauses: Vec<Clauses>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    /// Declares a function (body set later with [`Module::define`]); ids can be used by calls
    /// before the body exists.
    pub fn declare(
        &mut self,
        instance: InstanceId,
        path: impl Into<String>,
        params: Vec<TypeId>,
        ret: Option<TypeId>,
    ) -> FuncId {
        let id = FuncId(self.funcs.len() as u32);
        self.funcs.push(Func {
            instance,
            path: path.into(),
            type_args: Vec::new(),
            params,
            ret,
            features: FeatureSet::EMPTY,
            body: None,
            name_site: Site::NONE,
            symbol: None,
            known: None,
            alt_stack: false,
        });
        id
    }

    /// Sets a function's body. Returns false for an unknown id.
    pub fn define(&mut self, f: FuncId, body: Body) -> bool {
        match self.funcs.get_mut(f.index()) {
            Some(func) => {
                func.body = Some(body);
                true
            }
            None => false,
        }
    }

    pub fn func(&self, f: FuncId) -> Option<&Func> {
        self.funcs.get(f.index())
    }

    pub fn func_mut(&mut self, f: FuncId) -> Option<&mut Func> {
        self.funcs.get_mut(f.index())
    }

    /// Finds a function by path (first match).
    pub fn find_func(&self, path: &str) -> Option<FuncId> {
        self.funcs.iter().position(|f| f.path == path).map(|i| FuncId(i as u32))
    }

    /// The first function stamped as known item `k`.
    pub fn known_func(&self, k: crate::Known) -> Option<FuncId> {
        self.funcs.iter().position(|f| f.known == Some(k)).map(|i| FuncId(i as u32))
    }

    /// The first function with a body stamped as known item `k`.
    pub fn known_body(&self, k: crate::Known) -> Option<FuncId> {
        self.funcs.iter().position(|f| f.known == Some(k) && f.body.is_some()).map(|i| FuncId(i as u32))
    }

    /// Whether function `f` is stamped as known item `k`.
    pub fn is_known(&self, f: FuncId, k: crate::Known) -> bool {
        self.func(f).is_some_and(|x| x.known == Some(k))
    }

    pub fn add_string(&mut self, bytes: impl Into<Vec<u8>>) -> StrId {
        self.strings.push(bytes.into());
        StrId(self.strings.len() as u32 - 1)
    }

    /// The id of an intrinsic name (deduplicated).
    pub fn intrinsic(&mut self, name: &str) -> IntrinsicId {
        if let Some(i) = self.intrinsics.iter().position(|n| n == name) {
            return IntrinsicId(i as u32);
        }
        self.intrinsics.push(name.to_string());
        IntrinsicId(self.intrinsics.len() as u32 - 1)
    }

    pub fn intrinsic_name(&self, i: IntrinsicId) -> Option<&str> {
        self.intrinsics.get(i.index()).map(|s| s.as_str())
    }

    /// Renumbers the type table canonically ([`TypeTable::canonicalize`]) and rewrites every type
    /// id in the functions.
    pub fn canonicalize_types(&mut self) -> Result<(), crate::types::CanonError> {
        let remap = self.types.canonicalize()?;
        let m = |t: TypeId| remap.get(t.index()).copied().unwrap_or(t);
        for f in &mut self.funcs {
            for a in &mut f.type_args {
                if let GenericArg::Type(t) = a {
                    *t = m(*t);
                }
            }
            f.params.iter_mut().for_each(|t| *t = m(*t));
            f.ret = f.ret.map(m);
            if let Some(b) = &mut f.body {
                b.values.iter_mut().for_each(|v| v.ty = m(v.ty));
                b.insts.ty.iter_mut().for_each(|t| *t = t.map(m));
                for o in &mut b.operands {
                    if let Operand::Type(t) = o {
                        *t = m(*t);
                    }
                }
            }
        }
        for st in &mut self.statics {
            st.ty = m(st.ty);
        }
        Ok(())
    }
}

impl TypeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
