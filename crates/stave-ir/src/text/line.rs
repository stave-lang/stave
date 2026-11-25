//! One line of IR text: the table-driven parse on the generated LL(1) table
//! (`generated::ir_text_ll1_table`, from contracts/ir-text.ebnf) and the typed line it denotes.
//! Everything here depends on the line alone, so lines can be parsed in any order or in
//! parallel; what relates lines to each other is the structure pass ([`super::build`]).

use std::sync::OnceLock;

use stave_target::{Feature, FeatureSet, Triple};

use super::lex::{LexError, Tok, is_word, lex, show, unescape};
use super::{EdgeKind, Level, Mode};
use crate::generated::ir_text_ll1_table as lt;
use crate::generated::ir_text_node_kinds::NodeKind as K;
use crate::generated::ir_text_terminals::{NTERM, TERM_IS_WORD, Term};
use crate::ir::{CastKind, Const, InstanceId, Op, PanicKind, Prefetch};
use crate::simd::VecOp;
use crate::types::{AdtKind, AdtShape, DeclId, FloatBits, GenericArg, IntBits, Lanes, TypeId, TypeKey};
use crate::witness::{Row, Term as RowTerm};

/// An error of one line, at a 1-based byte column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LineErr {
    pub col: u32,
    pub msg: String,
}

fn fail<T>(col: u32, msg: impl Into<String>) -> Result<T, LineErr> {
    Err(LineErr { col, msg: msg.into() })
}

/// The kinds of ids a line names that are checked against the module's sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefKind {
    Type,
    Str,
    Site,
    Static,
    Decl,
    Node,
}

/// One id a line names, with its column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IdRef {
    pub kind: RefKind,
    pub n: u32,
    pub col: u32,
}

/// An operand as written: a fn-local value number, a constant, or a module id.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ArgV {
    Val(u32),
    Const(Const),
    Str(u32),
    Func(InstanceId),
    Type(TypeId),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arg {
    pub v: ArgV,
    pub col: u32,
}

/// A branch target as written.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tgt {
    pub label: u32,
    pub col: u32,
    pub args: Vec<Arg>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TermL {
    Br(Tgt),
    Cbr(Arg, Tgt, Tgt),
    Switch(Arg, Vec<(i128, Tgt)>, Tgt),
    Ret(Option<Arg>),
    Unreachable,
}

/// The op of an instruction line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OpL {
    Op(Op),
    /// `intrinsic.<name>`: the name after the prefix.
    Intrinsic(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InstL {
    /// The result value number and its column.
    pub result: Option<(u32, u32)>,
    pub op: OpL,
    pub ty: Option<TypeId>,
    pub args: Vec<Arg>,
    pub site: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FnHead {
    pub inst: InstanceId,
    pub path: String,
    pub path_col: u32,
    pub type_args: Vec<GenericArg>,
    pub params: Vec<TypeId>,
    pub ret: Option<TypeId>,
    pub features: FeatureSet,
    pub symbol: Option<String>,
    pub variadic: bool,
    pub known: bool,
    pub known_col: u32,
    pub alt_stack: bool,
    pub unbuilt: bool,
    pub name_site: Option<u32>,
    pub body: bool,
}

/// The `pre` / `post` part of a claim line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClaimKindL {
    Pre { checked: bool },
    Post,
}

/// A line, typed. Ids that name other lines are kept as written; [`Parsed::refs`] lists the
/// section ids among them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Line {
    Header { target: Triple, mode: Mode, level: Level },
    Decl { id: u32, kind: AdtKind, path: String },
    Type { id: u32, key: TypeKey },
    Shape { id: u32, shape: AdtShape },
    Str { id: u32, bytes: Vec<u8> },
    Site { row: u32, file: String, line: u32, col: u32, func: String },
    Static { index: u32, ty: TypeId, path: String },
    Data { index: u32, offset: u32, bytes: Vec<u8> },
    Ref { index: u32, offset: u32, target: RefTargetL, addend: u32 },
    Main { inst: InstanceId, col: u32 },
    Cell(DeclId),
    RunStack(u64),
    Truncated,
    Node { n: u32, rec: bool },
    Edge { from: u32, to: u32, kind: EdgeKind },
    Nodes { inst: InstanceId, col: u32, nodes: Vec<u32> },
    Fn(Box<FnHead>),
    Claim { inst: InstanceId, col: u32, kind: ClaimKindL, row: Row },
    Block { label: u32, params: Vec<(u32, u32, TypeId)> },
    Inst(Box<InstL>),
    Term(TermL),
    Close,
}

/// What an address word points at, as written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefTargetL {
    Str(u32),
    Static(u32),
}

/// A parsed line: the typed line, the column of its first token, and the section ids it names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Parsed {
    pub line: Line,
    pub col: u32,
    pub refs: Vec<IdRef>,
}

/// Parses one line (without its `\n`): `None` for a blank or comment-only line.
pub(crate) fn parse_line(src: &[u8]) -> Result<Option<Parsed>, LineErr> {
    let toks = lex(src).map_err(|LexError { at, msg }| LineErr { col: at as u32 + 1, msg })?;
    let Some(first) = toks.first() else { return Ok(None) };
    let col = first.start + 1;
    let tree = drive(src, toks)?;
    let mut x = Extract { t: &tree, refs: Vec::new() };
    let line = x.line()?;
    Ok(Some(Parsed { line, col, refs: x.refs }))
}

// ---------------------------------------------------------------------------------------------
// Table driver

#[derive(Clone, Copy, Debug)]
enum Kid {
    Tok(u32),
    Node(u32),
}

struct Node {
    kind: K,
    prod: u16,
    kids: Vec<Kid>,
}

struct Tree<'a> {
    src: &'a [u8],
    toks: Vec<Tok>,
    nodes: Vec<Node>,
}

#[derive(Clone, Copy)]
enum Sym {
    T(u16),
    N(u16),
    End,
}

/// Runs the LL(1) table over the tokens of one line (`Line` is the start rule, the end of the
/// line is `EOF`) and builds the tree of named rules.
fn drive<'a>(src: &'a [u8], toks: Vec<Tok>) -> Result<Tree<'a>, LineErr> {
    let n = src.len() as u32;
    let eof = Tok { term: Term::Eof, start: n, end: n };
    let mut nodes: Vec<Node> = Vec::new();
    let mut open: Vec<u32> = Vec::new();
    let mut stack = vec![Sym::N(lt::START)];
    let mut i = 0usize;
    while let Some(top) = stack.pop() {
        let tok = toks.get(i).copied().unwrap_or(eof);
        match top {
            Sym::End => {
                open.pop();
            }
            Sym::T(x) => {
                let any = x == Term::AnyWord as u16 && TERM_IS_WORD[tok.term as usize];
                if x != tok.term as u16 && !any {
                    return Err(syntax_error(src, &toks, &nodes, &open, tok, Expect::Term(x)));
                }
                if let Some(&p) = open.last() {
                    nodes[p as usize].kids.push(Kid::Tok(i as u32));
                }
                i += 1;
            }
            Sym::N(nt) => {
                let p = lt::action(nt, tok.term as u16);
                if p < 0 {
                    return Err(syntax_error(src, &toks, &nodes, &open, tok, Expect::Row(nt)));
                }
                if lt::NT_NAMED[nt as usize] {
                    let Some(kind) = K::from_id(nt as u8) else {
                        return fail(tok.start + 1, "internal: unnamed rule");
                    };
                    let id = nodes.len() as u32;
                    nodes.push(Node { kind, prod: p as u16, kids: Vec::new() });
                    if let Some(&par) = open.last() {
                        nodes[par as usize].kids.push(Kid::Node(id));
                    }
                    open.push(id);
                    stack.push(Sym::End);
                }
                for &s in lt::rhs_rev(p as u16) {
                    stack.push(if s & lt::NT_BIT != 0 { Sym::N(s & !lt::NT_BIT) } else { Sym::T(s) });
                }
            }
        }
    }
    if let Some(t) = toks.get(i) {
        return fail(t.start + 1, format!("unexpected `{}` after a complete line", show(text(src, *t))));
    }
    Ok(Tree { src, toks, nodes })
}

enum Expect {
    Term(u16),
    Row(u16),
}

fn text(src: &[u8], t: Tok) -> &[u8] {
    src.get(t.start as usize..t.end as usize).unwrap_or(&[])
}

/// Tokens that begin an operand.
fn starts_operand(t: Term) -> bool {
    matches!(t, Term::Val | Term::Int | Term::Hex | Term::KwBf16 | Term::SId | Term::FnRef | Term::TId)
}

/// The op token and the fixed operand count of the instruction node `id` (`None`: the op takes
/// a variable number).
fn op_arity(src: &[u8], toks: &[Tok], nodes: &[Node], id: u32) -> Option<(String, Option<usize>)> {
    let node = nodes.get(id as usize)?;
    let first = match node.kids.first()? {
        Kid::Tok(t) => *t,
        Kid::Node(n) => match nodes.get(*n as usize)?.kids.first()? {
            Kid::Tok(t) => *t,
            Kid::Node(_) => return None,
        },
    };
    let name = show(text(src, *toks.get(first as usize)?));
    let rhs = lt::rhs_rev(node.prod);
    let mut count = 0usize;
    let mut fixed = true;
    // reversed: operands first, then the result-type slot and the op
    let skip = if node.kind == K::ValueOp { 2 } else { 3 };
    for &s in rhs.iter().take(rhs.len().saturating_sub(skip)) {
        if s & lt::NT_BIT != 0 {
            let nt = (s & !lt::NT_BIT) as usize;
            if lt::NT_NAMED[nt] {
                count += 1;
            } else {
                fixed = false;
            }
        } else {
            count += 1;
        }
    }
    Some((name, fixed.then_some(count)))
}

fn syntax_error(src: &[u8], toks: &[Tok], nodes: &[Node], open: &[u32], tok: Tok, want: Expect) -> LineErr {
    let col = tok.start + 1;
    let found = if tok.term == Term::Eof {
        "the end of the line".to_string()
    } else {
        format!("`{}`", show(text(src, tok)))
    };
    if let Expect::Row(nt) = want {
        if nt == lt::START {
            return LineErr { col, msg: format!("unknown line kind {found}") };
        }
        if nt == K::ValueOp as u16 {
            return LineErr { col, msg: format!("unknown op {found}") };
        }
    }
    // An instruction with the wrong number of operands: too few when its op node is still open
    // at the end of the operands, too many when an operand follows the closed node.
    if let Some(op) = nodes.iter().rposition(|n| matches!(n.kind, K::ValueOp | K::EffectOp)) {
        let op = op as u32;
        let inside = open.contains(&op);
        let ends = matches!(tok.term, Term::Eof | Term::Bang);
        if let Some((name, Some(n))) = op_arity(src, toks, nodes, op)
            && ((inside && ends) || (!inside && starts_operand(tok.term)))
        {
            let s = if n == 1 { "" } else { "s" };
            return LineErr {
                col,
                msg: format!("wrong arity: `{name}` takes {n} operand{s}, found {found}"),
            };
        }
    }
    let expected = match want {
        Expect::Term(x) => Term::from_id(x).map_or_else(String::new, describe),
        Expect::Row(nt) => row_expected(nt),
    };
    LineErr { col, msg: format!("expected {expected}, found {found}") }
}

fn describe(t: Term) -> String {
    match t {
        Term::Val => "a value `%N`".into(),
        Term::Blk => "a block label `bN`".into(),
        Term::TId => "a type `tN`".into(),
        Term::SId => "a string id `sN`".into(),
        Term::DId => "a decl id `dN`".into(),
        Term::FnRef => "a fn `@N`".into(),
        Term::Intr => "an intrinsic op".into(),
        Term::Int => "an integer".into(),
        Term::Hex => "a float constant".into(),
        Term::Bytes => "bytes `x..`".into(),
        Term::String => "a quoted string".into(),
        Term::Eof => "the end of the line".into(),
        Term::AnyWord => "a path".into(),
        t => format!("`{}`", t.text()),
    }
}

/// What row `nt` of the table accepts, for a message.
fn row_expected(nt: u16) -> String {
    match K::from_id(nt as u8).filter(|_| lt::NT_NAMED.get(nt as usize) == Some(&true)) {
        Some(K::Opnd) => return "an operand".into(),
        Some(K::Feature) => return "a feature name".into(),
        Some(K::Key) => return "a type key".into(),
        Some(K::Path) => return "a path".into(),
        _ => {}
    }
    let mut terms: Vec<Term> =
        (0..NTERM as u16).filter(|&t| lt::action(nt, t) >= 0).filter_map(Term::from_id).collect();
    if terms.contains(&Term::AnyWord) {
        terms.retain(|&t| t == Term::AnyWord || !TERM_IS_WORD[t as usize]);
    }
    let mut parts: Vec<String> = terms.iter().take(10).map(|&t| describe(t)).collect();
    if terms.len() > 10 {
        parts.push("...".into());
    }
    match parts.len() {
        0 => "nothing".into(),
        1 => parts.remove(0),
        _ => {
            let last = parts.pop().unwrap_or_default();
            format!("{} or {last}", parts.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Ops

/// Every op with a fixed name (all but `Op::Intrinsic`).
pub(crate) fn named_ops() -> Vec<Op> {
    let mut v = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Rem,
        Op::Neg,
        Op::AddWrap,
        Op::SubWrap,
        Op::MulWrap,
        Op::ShlWrap,
        Op::AddSat,
        Op::SubSat,
        Op::MulSat,
        Op::Shl,
        Op::Shr,
        Op::And,
        Op::Or,
        Op::Xor,
        Op::Not,
        Op::Popcount,
        Op::Clz,
        Op::Ctz,
        Op::MulWide,
        Op::Eq,
        Op::Ne,
        Op::Lt,
        Op::Le,
        Op::Gt,
        Op::Ge,
        Op::Select,
        Op::Cast(CastKind::Exact),
        Op::Cast(CastKind::Wrap),
        Op::Cast(CastKind::Sat),
        Op::Cast(CastKind::Round),
        Op::Cast(CastKind::Try),
        Op::Make,
        Op::Get,
        Op::Set,
        Op::EnumMake,
        Op::EnumTag,
        Op::EnumGet,
        Op::Slot,
        Op::Load,
        Op::Store,
        Op::FieldAddr,
        Op::EnumFieldAddr,
        Op::ElemAddr,
        Op::SliceMake,
        Op::SliceLen,
        Op::SlicePtr,
        Op::FnMake,
        Op::BoundsCheck,
        Op::StoreStream,
        Op::FenceStream,
        Op::Call,
        Op::CallInd,
    ];
    v.extend(Prefetch::ALL.iter().map(|&p| Op::Prefetch(p)));
    v.extend(PanicKind::ALL.iter().map(|&k| Op::Panic(k)));
    v.extend(VecOp::ALL.iter().map(|&o| Op::Vec(o)));
    v
}

/// The op a terminal spells, by terminal id.
fn op_of(t: Term) -> Option<Op> {
    static TABLE: OnceLock<Vec<Option<Op>>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut v = vec![None; NTERM];
        for op in named_ops() {
            if let Some(name) = op.name()
                && let Some(i) = Term::ALL.iter().position(|t| t.text() == name && !t.is_class())
            {
                v[i] = Some(op);
            }
        }
        v
    });
    table.get(t as usize).copied().flatten()
}

// ---------------------------------------------------------------------------------------------
// Typed extraction

#[derive(Clone, Copy)]
struct Cur<'a> {
    t: &'a Tree<'a>,
    kids: &'a [Kid],
    i: usize,
    /// Column for a shape error (the start of the node).
    col: u32,
}

fn shape_err(col: u32) -> LineErr {
    LineErr { col, msg: "internal: the line does not match its rule".into() }
}

impl<'a> Cur<'a> {
    fn of(t: &'a Tree<'a>, node: u32, col: u32) -> Result<Cur<'a>, LineErr> {
        let n = t.nodes.get(node as usize).ok_or_else(|| shape_err(col))?;
        Ok(Cur { t, kids: &n.kids, i: 0, col })
    }

    fn peek(&self) -> Option<Kid> {
        self.kids.get(self.i).copied()
    }

    fn peek_term(&self) -> Option<Term> {
        match self.peek()? {
            Kid::Tok(t) => self.t.toks.get(t as usize).map(|t| t.term),
            Kid::Node(_) => None,
        }
    }

    fn peek_kind(&self) -> Option<K> {
        match self.peek()? {
            Kid::Node(n) => self.t.nodes.get(n as usize).map(|n| n.kind),
            Kid::Tok(_) => None,
        }
    }

    fn tok(&mut self) -> Result<Tok, LineErr> {
        match self.peek() {
            Some(Kid::Tok(t)) => {
                self.i += 1;
                self.t.toks.get(t as usize).copied().ok_or_else(|| shape_err(self.col))
            }
            _ => Err(shape_err(self.col)),
        }
    }

    fn opt_tok(&mut self, term: Term) -> Option<Tok> {
        if self.peek_term() == Some(term) { self.tok().ok() } else { None }
    }

    fn node(&mut self, kind: K) -> Result<Cur<'a>, LineErr> {
        match self.peek() {
            Some(Kid::Node(n)) if self.peek_kind() == Some(kind) => {
                self.i += 1;
                let col = self.first_col(n);
                Cur::of(self.t, n, col)
            }
            _ => Err(shape_err(self.col)),
        }
    }

    fn opt_node(&mut self, kind: K) -> Option<Cur<'a>> {
        if self.peek_kind() == Some(kind) { self.node(kind).ok() } else { None }
    }

    /// The column of the first token under node `n`.
    fn first_col(&self, n: u32) -> u32 {
        let mut n = n;
        loop {
            match self.t.nodes.get(n as usize).and_then(|x| x.kids.first()) {
                Some(Kid::Tok(t)) => return self.t.toks.get(*t as usize).map_or(self.col, |t| t.start + 1),
                Some(Kid::Node(m)) => n = *m,
                None => return self.col,
            }
        }
    }

    fn done(&self) -> bool {
        self.i >= self.kids.len()
    }
}

struct Extract<'a> {
    t: &'a Tree<'a>,
    refs: Vec<IdRef>,
}

fn col(t: Tok) -> u32 {
    t.start + 1
}

impl<'a> Extract<'a> {
    fn text(&self, t: Tok) -> &'a [u8] {
        text(self.t.src, t)
    }

    /// An `Int` token's value.
    fn int(&self, t: Tok) -> Result<i128, LineErr> {
        let b = self.text(t);
        let (neg, digits) = match b.split_first() {
            Some((b'-', rest)) => (true, rest),
            _ => (false, b),
        };
        let mut v: i128 = 0;
        for &d in digits {
            let d = i128::from(d.wrapping_sub(b'0'));
            v = v
                .checked_mul(10)
                .and_then(|v| if neg { v.checked_sub(d) } else { v.checked_add(d) })
                .ok_or(LineErr { col: col(t), msg: format!("integer `{}` is out of range", show(b)) })?;
        }
        Ok(v)
    }

    /// An `Int` token that must lie in `0..=max`.
    fn uint(&self, t: Tok, max: u128, what: &str) -> Result<u128, LineErr> {
        let v = self.int(t)?;
        if v < 0 || v as u128 > max {
            return fail(col(t), format!("{what} `{}` is out of range 0..={max}", show(self.text(t))));
        }
        Ok(v as u128)
    }

    fn u32(&mut self, c: &mut Cur<'a>, what: &str) -> Result<u32, LineErr> {
        let t = c.tok()?;
        Ok(self.uint(t, u32::MAX as u128, what)? as u32)
    }

    /// The number after the one-letter prefix of a `Val`, `Blk`, `TId`, `SId` or `DId` token.
    fn id(&self, t: Tok) -> Result<u32, LineErr> {
        let b = self.text(t);
        let digits = b.get(1..).unwrap_or(&[]);
        let mut v: u32 = 0;
        for &d in digits {
            v = v
                .checked_mul(10)
                .and_then(|v| v.checked_add(u32::from(d.wrapping_sub(b'0'))))
                .ok_or(LineErr { col: col(t), msg: format!("`{}` is out of range", show(b)) })?;
        }
        Ok(v)
    }

    fn reference(&mut self, kind: RefKind, t: Tok) -> Result<u32, LineErr> {
        let n = self.id(t)?;
        self.refs.push(IdRef { kind, n, col: col(t) });
        Ok(n)
    }

    fn ty(&mut self, c: &mut Cur<'a>) -> Result<TypeId, LineErr> {
        let t = c.tok()?;
        Ok(TypeId(self.reference(RefKind::Type, t)?))
    }

    fn inst(&self, t: Tok) -> Result<InstanceId, LineErr> {
        let b = self.text(t);
        let body = b.get(1..).unwrap_or(&[]);
        let mut out = Vec::new();
        for (k, part) in body.split(|&c| c == b'.').enumerate() {
            if part == b"u" && k > 0 {
                out.push(InstanceId::UNCHECKED_ENTRY);
                continue;
            }
            let mut v: u32 = 0;
            for &d in part {
                v = v
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(u32::from(d.wrapping_sub(b'0'))))
                    .ok_or(LineErr { col: col(t), msg: format!("`{}` is out of range", show(b)) })?;
            }
            if k > 0 && v == InstanceId::UNCHECKED_ENTRY {
                return fail(col(t), format!("`{}`: the number {v} is spelled `u`", show(b)));
            }
            out.push(v);
        }
        Ok(InstanceId(out))
    }

    fn string(&self, t: Tok) -> Vec<u8> {
        unescape(self.text(t))
    }

    fn utf8(&self, t: Tok, what: &str) -> Result<String, LineErr> {
        String::from_utf8(self.string(t)).or_else(|_| fail(col(t), format!("{what} is not UTF-8")))
    }

    fn path(&self, c: &mut Cur<'a>) -> Result<(String, u32), LineErr> {
        let mut p = c.node(K::Path)?;
        let t = p.tok()?;
        if t.term == Term::String {
            let s = self.utf8(t, "a path")?;
            if is_word(s.as_bytes()) {
                return fail(col(t), format!("path `{s}` has the Word shape: it is written bare"));
            }
            return Ok((s, col(t)));
        }
        let b = self.text(t);
        Ok((String::from_utf8_lossy(b).into_owned(), col(t)))
    }

    fn features(&self, toks: &[Tok], sep: &str) -> Result<FeatureSet, LineErr> {
        let mut fs = Vec::new();
        for &t in toks {
            let name = String::from_utf8_lossy(self.text(t)).into_owned();
            fs.push(
                Feature::from_name(&name)
                    .ok_or(LineErr { col: col(t), msg: format!("unknown feature `{name}`") })?,
            );
        }
        let set = FeatureSet::from_features(fs.iter().copied());
        let canon = crate::types::feature_generators(set);
        if fs != canon {
            let want: Vec<&str> = canon.iter().map(|f| f.name()).collect();
            let at = toks.first().map_or(1, |&t| col(t));
            return fail(
                at,
                format!(
                    "feature list is not canonical (implied members left out, in feature order): write `{}`",
                    want.join(sep)
                ),
            );
        }
        Ok(set)
    }

    fn line(&mut self) -> Result<Line, LineErr> {
        let mut root = Cur::of(self.t, 0, 1)?;
        if root.opt_tok(Term::RBrace).is_some() {
            return Ok(Line::Close);
        }
        let kind = root.peek_kind().ok_or_else(|| shape_err(1))?;
        let mut c = root.node(kind)?;
        let line = match kind {
            K::HeaderLine => self.header(&mut c)?,
            K::DeclLine => {
                c.tok()?;
                let t = c.tok()?;
                let id = self.id(t)?;
                let kind = match c.node(K::DeclKind)?.tok()?.term {
                    Term::KwStruct => AdtKind::Struct,
                    _ => AdtKind::Enum,
                };
                let (path, _) = self.path(&mut c)?;
                Line::Decl { id, kind, path }
            }
            K::TypeLine => {
                c.tok()?;
                let t = c.tok()?;
                let id = self.id(t)?;
                let key = self.key(&mut c.node(K::Key)?)?;
                Line::Type { id, key }
            }
            K::ShapeLine => {
                c.tok()?;
                let t = c.tok()?;
                let id = self.id(t)?;
                let mut body = c.node(K::ShapeBody)?;
                let shape = match body.tok()?.term {
                    Term::KwStruct => {
                        let mut fields = Vec::new();
                        while !body.done() {
                            fields.push(self.ty(&mut body)?);
                        }
                        AdtShape::structure(fields)
                    }
                    _ => {
                        let mut variants = Vec::new();
                        while let Some(mut v) = body.opt_node(K::Variant) {
                            v.tok()?;
                            let mut fields = Vec::new();
                            while !v.done() {
                                fields.push(self.ty(&mut v)?);
                            }
                            variants.push(fields);
                        }
                        AdtShape::enumeration(variants)
                    }
                };
                Line::Shape { id, shape }
            }
            K::StrLine => {
                c.tok()?;
                let t = c.tok()?;
                let id = self.id(t)?;
                let s = c.tok()?;
                Line::Str { id, bytes: self.string(s) }
            }
            K::SiteLine => {
                c.tok()?;
                let row = self.u32(&mut c, "a site row")?;
                let f = c.tok()?;
                let file = self.utf8(f, "a site file")?;
                let line = self.u32(&mut c, "a line")?;
                let column = self.u32(&mut c, "a column")?;
                let (func, _) = self.path(&mut c)?;
                Line::Site { row, file, line, col: column, func }
            }
            K::StaticLine => {
                c.tok()?;
                let index = self.u32(&mut c, "a static index")?;
                let ty = self.ty(&mut c)?;
                let (path, _) = self.path(&mut c)?;
                Line::Static { index, ty, path }
            }
            K::DataLine => {
                c.tok()?;
                let index = self.u32(&mut c, "a static index")?;
                let ot = c.tok()?;
                let offset = self.uint(ot, u32::MAX as u128, "an offset")? as u32;
                let bt = c.tok()?;
                let hex = self.text(bt).get(1..).unwrap_or(&[]);
                let bytes: Vec<u8> = hex
                    .chunks(2)
                    .map(|p| {
                        let v = |d: u8| if d.is_ascii_digit() { d - b'0' } else { d.wrapping_sub(b'a') + 10 };
                        v(p[0]) * 16 + v(p.get(1).copied().unwrap_or(b'0'))
                    })
                    .collect();
                canonical_data_block(offset, &bytes).map_err(|msg| LineErr { col: col(ot), msg })?;
                Line::Data { index, offset, bytes }
            }
            K::RefLine => {
                c.tok()?;
                let index = self.u32(&mut c, "a static index")?;
                let offset = self.u32(&mut c, "an offset")?;
                let mut tc = c.node(K::RefTarget)?;
                let target = match tc.peek_term() {
                    Some(Term::SId) => {
                        let t = tc.tok()?;
                        RefTargetL::Str(self.reference(RefKind::Str, t)?)
                    }
                    _ => {
                        tc.tok()?;
                        let t = tc.tok()?;
                        let j = self.uint(t, u32::MAX as u128, "a static index")? as u32;
                        self.refs.push(IdRef { kind: RefKind::Static, n: j, col: col(t) });
                        RefTargetL::Static(j)
                    }
                };
                let addend = self.u32(&mut c, "an addend")?;
                Line::Ref { index, offset, target, addend }
            }
            K::MainLine => {
                c.tok()?;
                let t = c.tok()?;
                Line::Main { inst: self.inst(t)?, col: col(t) }
            }
            K::CellLine => {
                c.tok()?;
                let t = c.tok()?;
                Line::Cell(DeclId(self.reference(RefKind::Decl, t)?))
            }
            K::RunStackLine => {
                c.tok()?;
                let t = c.tok()?;
                Line::RunStack(self.uint(t, u64::MAX as u128, "a stack size")? as u64)
            }
            K::TruncatedLine => Line::Truncated,
            K::NodeLine => {
                c.tok()?;
                let n = self.u32(&mut c, "a node number")?;
                let rec = c.opt_tok(Term::KwRec).is_some();
                Line::Node { n, rec }
            }
            K::EdgeLine => {
                c.tok()?;
                let ft = c.tok()?;
                let from = self.uint(ft, u32::MAX as u128, "a node number")? as u32;
                self.refs.push(IdRef { kind: RefKind::Node, n: from, col: col(ft) });
                let tt = c.tok()?;
                let to = self.uint(tt, u32::MAX as u128, "a node number")? as u32;
                self.refs.push(IdRef { kind: RefKind::Node, n: to, col: col(tt) });
                let kind = match c.node(K::EdgeKind)?.tok()?.term {
                    Term::KwCall => EdgeKind::Call,
                    Term::KwIndirect => EdgeKind::Indirect,
                    _ => EdgeKind::Spawn,
                };
                Line::Edge { from, to, kind }
            }
            K::NodesLine => {
                c.tok()?;
                let t = c.tok()?;
                let inst = self.inst(t)?;
                let mut nodes = Vec::new();
                while !c.done() {
                    let nt = c.tok()?;
                    let n = self.uint(nt, u32::MAX as u128, "a node number")? as u32;
                    if nodes.last().is_some_and(|&l| l >= n) {
                        return fail(col(nt), "the nodes of a fn are listed ascending, without repeats");
                    }
                    self.refs.push(IdRef { kind: RefKind::Node, n, col: col(nt) });
                    nodes.push(n);
                }
                Line::Nodes { inst, col: col(t), nodes }
            }
            K::FnLine => Line::Fn(Box::new(self.fn_line(&mut c)?)),
            K::ClaimLine => {
                c.tok()?;
                let t = c.tok()?;
                let inst = self.inst(t)?;
                let mut kc = c.node(K::ClaimKind)?;
                let kind = match kc.tok()?.term {
                    Term::KwPre => ClaimKindL::Pre { checked: kc.opt_tok(Term::KwChecked).is_some() },
                    _ => ClaimKindL::Post,
                };
                let a = self.cterm(&mut c.node(K::CTerm)?)?;
                c.tok()?;
                let b = self.cterm(&mut c.node(K::CTerm)?)?;
                let wt = c.tok()?;
                let w = self.int(wt)?;
                Line::Claim { inst, col: col(t), kind, row: Row { a, b, w } }
            }
            K::BlockLine => {
                let t = c.tok()?;
                let label = self.id(t)?;
                let mut params = Vec::new();
                while !c.done() {
                    let Some(mut p) = c.opt_node(K::BlockParam) else {
                        c.tok()?;
                        continue;
                    };
                    let vt = p.tok()?;
                    let v = self.id(vt)?;
                    p.tok()?;
                    let ty = self.ty(&mut p)?;
                    params.push((v, col(vt), ty));
                }
                Line::Block { label, params }
            }
            K::ValueInst => {
                let vt = c.tok()?;
                let v = self.id(vt)?;
                c.tok()?;
                let mut inst = self.op(&mut c.node(K::ValueOp)?)?;
                inst.result = Some((v, col(vt)));
                inst.site = self.at(&mut c)?;
                Line::Inst(Box::new(inst))
            }
            K::EffectInst => {
                let mut inst = self.op(&mut c.node(K::EffectOp)?)?;
                inst.site = self.at(&mut c)?;
                Line::Inst(Box::new(inst))
            }
            K::TermLine => Line::Term(self.term(&mut c)?),
            _ => return Err(shape_err(c.col)),
        };
        Ok(line)
    }

    fn header(&mut self, c: &mut Cur<'a>) -> Result<Line, LineErr> {
        c.tok()?;
        let vt = c.tok()?;
        let version = self.int(vt)?;
        if version != 4 {
            return fail(
                col(vt),
                format!("IR text version {version} is not read here (this reader reads version 4)"),
            );
        }
        let tt = c.tok()?;
        let name = String::from_utf8_lossy(&self.string(tt)).into_owned();
        let Some(target) = Triple::ALL.into_iter().find(|t| t.name() == name) else {
            return match Triple::parse(&name) {
                Ok(t) => {
                    fail(col(tt), format!("target `{name}` is written by its canonical name `{}`", t.name()))
                }
                Err(e) => fail(col(tt), e.to_string()),
            };
        };
        let mode = match c.node(K::Mode)?.tok()?.term {
            Term::KwDebug => Mode::Debug,
            _ => Mode::Release,
        };
        let level_t = c.node(K::Level)?.tok()?;
        let level = match level_t.term {
            Term::KwLowered => Level::Lowered,
            Term::KwChecked => Level::Checked,
            _ => Level::Opt,
        };
        if level == Level::Opt && mode != Mode::Release {
            return fail(col(level_t), "level `opt` exists only with mode `release`");
        }
        Ok(Line::Header { target, mode, level })
    }

    fn key(&mut self, c: &mut Cur<'a>) -> Result<TypeKey, LineErr> {
        let kw = c.tok()?;
        Ok(match kw.term {
            Term::KwInt => {
                let signed = c.node(K::Sign)?.tok()?.term == Term::KwS;
                let w = c.node(K::IntW)?.tok()?;
                let bits = if w.term == Term::KwSize {
                    IntBits::Size
                } else {
                    match self.int(w)? {
                        8 => IntBits::B8,
                        16 => IntBits::B16,
                        32 => IntBits::B32,
                        64 => IntBits::B64,
                        _ => return fail(col(w), "an int width is 8, 16, 32, 64 or `size`"),
                    }
                };
                TypeKey::Int { signed, bits }
            }
            Term::KwFloat => {
                let w = c.node(K::FloatW)?.tok()?;
                let bits = match (w.term, self.text(w)) {
                    (Term::Blk, b"b16") => FloatBits::B16,
                    (Term::Int, b"16") => FloatBits::F16,
                    (Term::Int, b"32") => FloatBits::F32,
                    (Term::Int, b"64") => FloatBits::F64,
                    _ => return fail(col(w), "a float width is 16, 32, 64 or `b16`"),
                };
                TypeKey::Float(bits)
            }
            Term::KwBool => TypeKey::Bool,
            Term::KwChar => TypeKey::Char,
            Term::KwStr => TypeKey::Str,
            Term::KwRef => {
                let mutable = c.opt_tok(Term::KwMut).is_some();
                TypeKey::Ref { mutable, inner: self.ty(c)? }
            }
            Term::KwPtr => {
                let mutable = c.node(K::PtrMut)?.tok()?.term == Term::KwMut;
                TypeKey::RawPtr { mutable, inner: self.ty(c)? }
            }
            Term::KwArray => {
                let elem = self.ty(c)?;
                let len_t = c.tok()?;
                TypeKey::Array { elem, len: self.uint(len_t, u64::MAX as u128, "an array length")? as u64 }
            }
            Term::KwSlice => {
                let elem = self.ty(c)?;
                let align = match c.opt_tok(Term::KwAlign) {
                    Some(_) => {
                        let at = c.tok()?;
                        Some(self.uint(at, u32::MAX as u128, "an alignment")? as u32)
                    }
                    None => None,
                };
                TypeKey::Slice { elem, align }
            }
            Term::KwTuple => {
                let mut ts = Vec::new();
                while !c.done() {
                    ts.push(self.ty(c)?);
                }
                TypeKey::Tuple(ts)
            }
            Term::KwAdt => {
                let dt = c.tok()?;
                let def = DeclId(self.reference(RefKind::Decl, dt)?);
                let mut args = Vec::new();
                while let Some(mut g) = c.opt_node(K::GArg) {
                    args.push(self.garg(&mut g)?);
                }
                TypeKey::Adt { def, args }
            }
            Term::KwFntype => {
                let mut params = Vec::new();
                while c.peek_term() == Some(Term::TId) {
                    params.push(self.ty(c)?);
                }
                c.tok()?;
                let ret = self.ty(c)?;
                let mut fs = Vec::new();
                while let Some(mut f) = c.opt_node(K::Feature) {
                    fs.push(f.tok()?);
                }
                let features = self.features(&fs, " ")?;
                TypeKey::FnType { params, ret, features }
            }
            Term::KwVec => {
                let lane = self.ty(c)?;
                TypeKey::Vec { lane, n: self.lanes(c)? }
            }
            Term::KwMask => {
                let wt = c.tok()?;
                let width = self.uint(wt, u8::MAX as u128, "a mask width")? as u8;
                TypeKey::Mask { width, n: self.lanes(c)? }
            }
            _ => return Err(shape_err(col(kw))),
        })
    }

    fn lanes(&mut self, c: &mut Cur<'a>) -> Result<Lanes, LineErr> {
        let t = c.node(K::Lanes)?.tok()?;
        if t.term == Term::KwN {
            return Ok(Lanes::Native);
        }
        Ok(Lanes::Fixed(self.uint(t, u8::MAX as u128, "a lane count")? as u8))
    }

    fn garg(&mut self, g: &mut Cur<'a>) -> Result<GenericArg, LineErr> {
        let t = g.tok()?;
        if t.term == Term::TId {
            return Ok(GenericArg::Type(TypeId(self.reference(RefKind::Type, t)?)));
        }
        Ok(GenericArg::Const(self.int(t)?))
    }

    fn cterm(&mut self, c: &mut Cur<'a>) -> Result<RowTerm, LineErr> {
        let t = c.tok()?;
        Ok(match t.term {
            Term::KwZero => RowTerm::Zero,
            Term::KwResult => RowTerm::Result,
            Term::KwLenresult => RowTerm::LenResult,
            Term::KwP => RowTerm::Param(self.u32(c, "a param index")?),
            _ => RowTerm::LenParam(self.u32(c, "a param index")?),
        })
    }

    fn fn_line(&mut self, c: &mut Cur<'a>) -> Result<FnHead, LineErr> {
        c.tok()?;
        let it = c.tok()?;
        let inst = self.inst(it)?;
        let (path, path_col) = self.path(c)?;
        let mut type_args = Vec::new();
        if let Some(mut g) = c.opt_node(K::GenericArgs) {
            while !g.done() {
                if let Some(mut a) = g.opt_node(K::GArg) {
                    type_args.push(self.garg(&mut a)?);
                } else {
                    g.tok()?;
                }
            }
        }
        c.tok()?;
        let mut params = Vec::new();
        if let Some(mut l) = c.opt_node(K::TypeList) {
            while !l.done() {
                if l.peek_term() == Some(Term::TId) {
                    params.push(self.ty(&mut l)?);
                } else {
                    l.tok()?;
                }
            }
        }
        c.tok()?;
        let ret = match c.opt_node(K::FnRet) {
            Some(mut r) => {
                r.tok()?;
                Some(self.ty(&mut r)?)
            }
            None => None,
        };
        let features = match c.opt_node(K::Needs) {
            Some(mut n) => {
                let mut fs = Vec::new();
                while !n.done() {
                    if let Some(mut f) = n.opt_node(K::Feature) {
                        fs.push(f.tok()?);
                    } else {
                        n.tok()?;
                    }
                }
                self.features(&fs, ", ")?
            }
            None => FeatureSet::EMPTY,
        };
        let symbol = match c.opt_node(K::Symbol) {
            Some(mut s) => {
                s.tok()?;
                let t = s.tok()?;
                Some(self.utf8(t, "a symbol")?)
            }
            None => None,
        };
        let variadic = c.opt_tok(Term::KwVariadic).is_some();
        let known_tok = c.opt_tok(Term::KwKnown);
        let alt_stack = c.opt_tok(Term::KwAltstack).is_some();
        let unbuilt = c.opt_tok(Term::KwUnbuilt).is_some();
        let name_site = match c.opt_node(K::NameSite) {
            Some(mut s) => {
                s.tok()?;
                let t = s.tok()?;
                let row = self.uint(t, u32::MAX as u128, "a site row")? as u32;
                self.refs.push(IdRef { kind: RefKind::Site, n: row, col: col(t) });
                Some(row)
            }
            None => None,
        };
        let body = c.opt_tok(Term::LBrace).is_some();
        Ok(FnHead {
            inst,
            path,
            path_col,
            type_args,
            params,
            ret,
            features,
            symbol,
            variadic,
            known: known_tok.is_some(),
            known_col: known_tok.map_or(path_col, col),
            alt_stack,
            unbuilt,
            name_site,
            body,
        })
    }

    /// The `!<row>` of an instruction.
    fn at(&mut self, c: &mut Cur<'a>) -> Result<Option<u32>, LineErr> {
        match c.opt_node(K::At) {
            Some(mut a) => {
                a.tok()?;
                let t = a.tok()?;
                let row = self.uint(t, u32::MAX as u128, "a site row")? as u32;
                self.refs.push(IdRef { kind: RefKind::Site, n: row, col: col(t) });
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    fn op(&mut self, c: &mut Cur<'a>) -> Result<InstL, LineErr> {
        let op_tok = match c.peek() {
            Some(Kid::Node(_)) => {
                let kind = c.peek_kind().ok_or_else(|| shape_err(c.col))?;
                c.node(kind)?.tok()?
            }
            _ => c.tok()?,
        };
        let op = if op_tok.term == Term::Intr {
            let name = self.text(op_tok).get("intrinsic.".len()..).unwrap_or(&[]);
            OpL::Intrinsic(String::from_utf8_lossy(name).into_owned())
        } else {
            OpL::Op(
                op_of(op_tok.term)
                    .ok_or(LineErr { col: col(op_tok), msg: "internal: op without an IR op".into() })?,
            )
        };
        let ty = if c.peek_term() == Some(Term::TId) {
            Some(self.ty(c)?)
        } else {
            c.tok()?;
            c.tok()?;
            None
        };
        let mut args = Vec::new();
        while !c.done() {
            if let Some(mut o) = c.opt_node(K::Opnd) {
                args.push(self.opnd(&mut o)?);
            } else {
                let t = c.tok()?;
                args.push(self.loose(t)?);
            }
        }
        Ok(InstL { result: None, op, ty, args, site: None })
    }

    /// An `Int` or `FnRef` operand written outside `Opnd`.
    fn loose(&mut self, t: Tok) -> Result<Arg, LineErr> {
        let v = match t.term {
            Term::Int => ArgV::Const(Const::Int(self.int(t)?)),
            Term::FnRef => ArgV::Func(self.inst(t)?),
            _ => return Err(shape_err(col(t))),
        };
        Ok(Arg { v, col: col(t) })
    }

    fn opnd(&mut self, c: &mut Cur<'a>) -> Result<Arg, LineErr> {
        let t = c.tok()?;
        let hex = |x: &Self, t: Tok| -> u64 {
            x.text(t).get(2..).unwrap_or(&[]).iter().fold(0u64, |a, &d| {
                a.wrapping_mul(16).wrapping_add(u64::from(if d.is_ascii_digit() {
                    d - b'0'
                } else {
                    d.wrapping_sub(b'a') + 10
                }))
            })
        };
        let v = match t.term {
            Term::Val => ArgV::Val(self.id(t)?),
            Term::Int => ArgV::Const(Const::Int(self.int(t)?)),
            Term::Hex => {
                let v = hex(self, t);
                ArgV::Const(match t.end - t.start - 2 {
                    4 => Const::F16(v as u16),
                    8 => Const::F32(v as u32),
                    _ => Const::F64(v),
                })
            }
            Term::KwBf16 => {
                let h = c.tok()?;
                if h.end - h.start - 2 != 4 {
                    return fail(col(h), "a `bf16` constant has 4 hex digits");
                }
                ArgV::Const(Const::BF16(hex(self, h) as u16))
            }
            Term::SId => ArgV::Str(self.reference(RefKind::Str, t)?),
            Term::FnRef => ArgV::Func(self.inst(t)?),
            Term::TId => ArgV::Type(TypeId(self.reference(RefKind::Type, t)?)),
            _ => return Err(shape_err(col(t))),
        };
        Ok(Arg { v, col: col(t) })
    }

    fn target(&mut self, c: &mut Cur<'a>) -> Result<Tgt, LineErr> {
        let mut tc = c.node(K::Target)?;
        let bt = tc.tok()?;
        let label = self.id(bt)?;
        let mut args = Vec::new();
        while !tc.done() {
            if let Some(mut o) = tc.opt_node(K::Opnd) {
                args.push(self.opnd(&mut o)?);
            } else {
                tc.tok()?;
            }
        }
        Ok(Tgt { label, col: col(bt), args })
    }

    fn term(&mut self, c: &mut Cur<'a>) -> Result<TermL, LineErr> {
        let kw = c.tok()?;
        Ok(match kw.term {
            Term::KwBr => TermL::Br(self.target(c)?),
            Term::KwCbr => {
                let cond = self.opnd(&mut c.node(K::Opnd)?)?;
                let then = self.target(c)?;
                TermL::Cbr(cond, then, self.target(c)?)
            }
            Term::KwSwitch => {
                let scrut = self.opnd(&mut c.node(K::Opnd)?)?;
                let mut cases = Vec::new();
                while let Some(mut k) = c.opt_node(K::Case) {
                    let vt = k.tok()?;
                    let v = self.int(vt)?;
                    k.tok()?;
                    cases.push((v, self.target(&mut k)?));
                }
                c.tok()?;
                c.tok()?;
                TermL::Switch(scrut, cases, self.target(c)?)
            }
            Term::KwRet => match c.opt_node(K::Opnd) {
                Some(mut o) => TermL::Ret(Some(self.opnd(&mut o)?)),
                None => TermL::Ret(None),
            },
            _ => TermL::Unreachable,
        })
    }
}

/// The canonical cut of a static's bytes (checked-ir.md "static, data, ref"): one line per
/// 64-byte block, from its first to its last nonzero byte.
fn canonical_data_block(offset: u32, bytes: &[u8]) -> Result<(), String> {
    let end = u64::from(offset) + bytes.len() as u64;
    if u64::from(offset) / 64 != (end - 1) / 64 {
        return Err("a `data` line lies within one 64-byte block of its static".into());
    }
    if bytes.first() == Some(&0) || bytes.last() == Some(&0) {
        return Err("a `data` line begins and ends with a nonzero byte".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every op with a fixed name is a terminal of the grammar, and every op terminal of the
    /// instruction rules names an op.
    #[test]
    fn op_names_match_the_grammar() {
        for op in named_ops() {
            let name = op.name().unwrap_or("?");
            let t = Term::ALL.iter().find(|t| t.text() == name && !t.is_class());
            assert!(t.is_some_and(|&t| op_of(t) == Some(op)), "op `{name}` has no terminal");
        }
        let groups = [
            K::Op0,
            K::Op1,
            K::Op2,
            K::Op3,
            K::VecOp0,
            K::VecOp1,
            K::VecOp2,
            K::VecOp3,
            K::EffOp0,
            K::EffOp2,
            K::VecStore3,
        ];
        for p in 0..lt::NPROD as u16 {
            if groups.iter().any(|g| lt::PROD_LHS[p as usize] == *g as u16) {
                for &s in lt::rhs_rev(p) {
                    let t = Term::from_id(s).expect("op group productions hold terminals");
                    assert!(op_of(t).is_some(), "terminal `{}` names no op", t.text());
                }
            }
        }
    }
}
