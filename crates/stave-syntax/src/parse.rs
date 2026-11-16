//! The table-driven parser.
//!
//! An explicit-stack PDA over the generated LL(1) table (`generated::ll1_table`), one item at a
//! time; the items of a file are parsed in parallel (rayon) and the results are assembled in item
//! order, so the output never depends on the thread count.
//!
//! **Start state.** An item is parsed from the stack `[EOF, Item]` (`[EOF, Import]` for an import
//! item); its tokens are followed by a virtual `EOF`, so the item must end exactly at its cut.
//!
//! **Split actions** (generated `SPLIT_FIRST` / `SPLIT_REST`, grammar header): a split token (D3
//! `>>` `>=` `>>=`, unary `&&`) that has no entry in a nonterminal's row uses the entry of its
//! first terminal (`>` / `&`); an expected first terminal consumes that part of the token and
//! leaves the rest (`>` / `=` / `>=`, or `&`) as the lookahead.
//!
//! **Errors.** Each item reports only its first error and is then dropped. The code is
//! refined per contracts/parse-errors.md (E0013, E0011, E0014, E0012, then the Rust forms
//! E0019–E0023 and E0025–E0028, in that order), otherwise E0004. The E0004 expected set is the
//! set of terminals with an action in the failing state: the terminal on top of the stack, or
//! every terminal with a table entry in the top nonterminal's row, listed in table terminal
//! order. At the virtual `EOF` the reported (found) token is the real token after the item (the
//! next item's first token, or `EOF`).
//!
//! **AST.** Each named-rule expansion opens a node at the current token and closes it when its
//! right-hand side is done (a close marker under the pushed symbols), which yields postorder
//! directly (see [`crate::ast`]). A pass-through rule's node (generated `NT_PASS`, grammar
//! header) that closes with exactly one child and no terminal of its own is not emitted: its
//! child already sits where the node would be, and every ancestor's subtree size is counted from
//! the emitted nodes. The open stack, and so error reporting and the printer's trace, still see it.

use std::sync::OnceLock;

use rayon::prelude::*;
use stave_diag::{Diagnostic, FileId, Span};

use crate::ast::{Ast, ItemAst, Node};
use crate::generated::ll1_table as lt;
use crate::generated::node_kinds::NodeKind;
use crate::generated::terminals::{NTERM, SPLIT_FIRST, SPLIT_REST, TERM_TEXT, Term};
use crate::items::{ItemKind, ItemSpan, ItemTable, items};
use crate::lex::{KIND_ERROR, LexStop, Token, kind_name, lex, token_end};
use crate::make_diag;

/// Row width of the parser's copy of the table: the terminals plus one column for error tokens.
const W: usize = NTERM + 1;
/// The column used for error tokens (no entry anywhere).
const ERR_COL: u16 = NTERM as u16;

const EOF: u16 = Term::Eof as u16;
const NONE: u16 = u16::MAX;

// Stack entry packing (u32).
const SYM_MASK: u32 = 0xFFFF;
const CLOSE: u32 = 0xFFFF;
const PROD_SHIFT: u32 = 16;
const PROD_MASK: u32 = 0xFFF;
/// The entry is on the right edge of a cast's type (D2 tracking).
const F_CAST: u32 = 1 << 28;
/// The `>` that closes a D2 TypeArgs entry.
const F_D2: u32 = 1 << 29;
const NT_BIT: u32 = lt::NT_BIT as u32;

const _: () = assert!(lt::NNT < 4096, "nonterminal ids must fit the stack entry");

/// Tables derived once from the generated ones.
struct Tables {
    table: Vec<i16>,
    /// Per production: the reversed right-hand side as packed stack entries.
    rhs: Vec<u32>,
    off: Vec<u32>,
    named: Vec<bool>,
    /// Per nonterminal: a pass-through rule (generated `NT_PASS`).
    pass: Vec<bool>,
    nullable: Vec<bool>,
    /// Per production: 1 + index (in the reversed rhs) of the cast's `Type`, or 0.
    cast_seed: Vec<u8>,
    /// Per lookahead: the first terminal of a split token, or [`NONE`].
    split_first: [u16; W],
    /// Per lookahead: what is left of a split token after its first terminal, or [`NONE`].
    split_rest: [u16; W],
    generic_args: usize,
    /// Per cell `nt * W + term`: index into `macros`, or [`SLOW`].
    mcell: Vec<u32>,
    macros: Vec<Macro>,
    mops: Vec<u32>,
    mpush: Vec<u32>,
}

/// A precomputed expansion of one table cell `(X, t)`: everything the PDA does from popping `X`
/// with lookahead `t` until a terminal matching `t` is on top (or `X` derived ε). The steps are
/// deterministic given `(X, t)`, so applying the macro is exactly equivalent to running them one
/// by one. Node events: `OPEN kind` (the node stays open: pushed on the open stack) or
/// `EMIT kind b` (opened and closed inside the macro: emitted directly, at the current token).
/// Cells that reach an error, or involve the D2 cast tracking, have no macro ([`SLOW`]) and run
/// step by step.
#[derive(Clone, Copy, Debug)]
struct Macro {
    ops: u32,
    ops_end: u32,
    push: u32,
    push_end: u32,
    /// The matching terminal entry left on top (consumed directly), or [`NO_TERM`].
    term: u32,
}

const SLOW: u32 = u32::MAX;
/// A cell whose macro does nothing but pop the nonterminal (an ε production of a helper).
const EPS: u32 = u32::MAX - 1;
const NO_TERM: u32 = u32::MAX;
const EMIT: u32 = 1 << 31;

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(build_tables)
}

fn build_tables() -> Tables {
    let nnt = lt::NNT;
    let mut table = vec![-1i16; nnt * W];
    for nt in 0..nnt {
        for t in 0..NTERM {
            table[nt * W + t] = lt::TABLE[nt * NTERM + t];
        }
    }
    let mut rhs = Vec::with_capacity(lt::NRHS);
    let mut off = Vec::with_capacity(lt::NPROD + 1);
    let mut cast_seed = vec![0u8; lt::NPROD];
    let cast = NodeKind::Cast.nt();
    let ty = NodeKind::Type.nt();
    for (p, seed) in cast_seed.iter_mut().enumerate() {
        off.push(rhs.len() as u32);
        let lhs = lt::PROD_LHS[p] as u32;
        let syms = lt::rhs_rev(p as u16);
        for (k, &s) in syms.iter().enumerate() {
            rhs.push(s as u32 | ((lhs & PROD_MASK) << PROD_SHIFT));
            if lt::NT_ORIGIN[lhs as usize] == cast && s == lt::NT_BIT | ty && k < 255 {
                *seed = k as u8 + 1;
            }
        }
    }
    off.push(rhs.len() as u32);
    let mut nullable = vec![false; nnt];
    loop {
        let mut changed = false;
        for p in 0..lt::NPROD {
            let lhs = lt::PROD_LHS[p] as usize;
            if nullable[lhs] {
                continue;
            }
            let all = lt::rhs_rev(p as u16)
                .iter()
                .all(|&s| s & lt::NT_BIT != 0 && nullable[(s & !lt::NT_BIT) as usize]);
            if all {
                nullable[lhs] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let (mut split_first, mut split_rest) = ([NONE; W], [NONE; W]);
    for (t, (&f, &r)) in SPLIT_FIRST.iter().zip(&SPLIT_REST).enumerate() {
        if f >= 0 && r >= 0 {
            split_first[t] = f as u16;
            split_rest[t] = r as u16;
        }
    }
    let mut t = Tables {
        table,
        rhs,
        off,
        named: lt::NT_NAMED.to_vec(),
        pass: lt::NT_PASS.to_vec(),
        nullable,
        cast_seed,
        split_first,
        split_rest,
        generic_args: NodeKind::GenericArgs.nt() as usize,
        mcell: Vec::new(),
        macros: Vec::new(),
        mops: Vec::new(),
        mpush: Vec::new(),
    };
    build_macros(&mut t);
    t
}

/// The production for nonterminal `nt` on lookahead `c`, with the split fallback (a split token
/// uses its first terminal's entry).
#[inline(always)]
fn cell_prod(t: &Tables, nt: usize, c: u16) -> Option<usize> {
    let row = nt * W;
    let mut p = t.table[row + c as usize];
    if p < 0 && t.split_first[c as usize] != NONE {
        p = t.table[row + t.split_first[c as usize] as usize];
    }
    (p >= 0).then_some(p as usize)
}

/// Whether terminal `sym` on top of the stack accepts lookahead `c` (exactly or by a split).
#[inline(always)]
fn term_accepts(t: &Tables, sym: u16, c: u16) -> bool {
    sym == c || t.split_first[c as usize] == sym
}

fn build_macros(t: &mut Tables) {
    let mut mcell = vec![SLOW; lt::NNT * W];
    let (mut macros, mut mops, mut mpush) = (Vec::new(), Vec::new(), Vec::new());
    for nt in 0..lt::NNT {
        for c in 0..W {
            if let Some((ops, push, term)) = simulate(t, nt, c as u16) {
                if ops.is_empty() && push.is_empty() && term == NO_TERM {
                    mcell[nt * W + c] = EPS;
                    continue;
                }
                mcell[nt * W + c] = macros.len() as u32;
                let (o, pp) = (mops.len() as u32, mpush.len() as u32);
                mops.extend_from_slice(&ops);
                mpush.extend_from_slice(&push);
                macros.push(Macro {
                    ops: o,
                    ops_end: mops.len() as u32,
                    push: pp,
                    push_end: mpush.len() as u32,
                    term,
                });
            }
        }
    }
    t.mcell = mcell;
    t.macros = macros;
    t.mops = mops;
    t.mpush = mpush;
}

/// Runs the PDA steps of cell `(nt, c)` symbolically; `None` when the cell must run step by step.
fn simulate(t: &Tables, nt: usize, c: u16) -> Option<(Vec<u32>, Vec<u32>, u32)> {
    #[derive(Clone, Copy)]
    enum Ev {
        Open(u8, bool),
        Emit(u8, u32),
    }
    let encode = |ev: &[Ev]| -> Option<Vec<u32>> {
        let mut out = Vec::new();
        for &x in ev {
            match x {
                Ev::Open(k, true) => out.push(k as u32),
                Ev::Open(_, false) => {}
                Ev::Emit(k, b) if b < (1 << 23) => out.push(EMIT | (b << 8) | k as u32),
                Ev::Emit(..) => return None,
            }
        }
        Some(out)
    };
    let mut ev: Vec<Ev> = Vec::new();
    let mut stack: Vec<u32> = vec![NT_BIT | nt as u32];
    let mut open: Vec<(u8, u32, usize)> = Vec::new();
    let mut emitted = 0u32;
    for _ in 0..1024 {
        let Some(e) = stack.pop() else {
            return Some((encode(&ev)?, stack, NO_TERM));
        };
        let sym = e & SYM_MASK;
        if sym == CLOSE {
            let (k, at, evi) = open.pop()?;
            if t.pass[k as usize] {
                // never taken (pass-through rules do not derive ε); the omission rule runs in `run`
                return None;
            }
            ev[evi] = Ev::Open(k, false);
            ev.push(Ev::Emit(k, emitted - at));
            emitted += 1;
            continue;
        }
        if sym & NT_BIT == 0 {
            return term_accepts(t, sym as u16, c)
                .then_some(())
                .and_then(|()| Some((encode(&ev)?, stack, e)));
        }
        let y = (sym & !NT_BIT) as usize;
        let p = cell_prod(t, y, c)?;
        if t.cast_seed[p] != 0 {
            return None;
        }
        if t.named[y] {
            open.push((y as u8, emitted, ev.len()));
            ev.push(Ev::Open(y as u8, true));
            stack.push(CLOSE);
        }
        stack.extend_from_slice(&t.rhs[t.off[p] as usize..t.off[p + 1] as usize]);
    }
    None
}

/// Observer of the PDA (used by the printer); [`NoTrace`] costs nothing. A tracing run takes
/// every step one by one (no macros).
pub(crate) trait Trace {
    const ON: bool;
    /// A named node opens at token `tok`.
    fn open(&mut self, _kind: u8, _tok: usize) {}
    /// The innermost open named node closes.
    fn close(&mut self) {}
    /// A terminal is consumed from token `tok` (twice for a split token); `producer` is the
    /// nonterminal whose production pushed it.
    fn term(&mut self, _tok: usize, _term: u16, _producer: u16) {}
}

pub(crate) struct NoTrace;

impl Trace for NoTrace {
    const ON: bool = false;
}

/// Forces the step-by-step PDA (the reference the macros are checked against).
pub(crate) struct Plain;

impl Trace for Plain {
    const ON: bool = true;
}

/// Reusable per-thread buffers.
#[derive(Default)]
pub(crate) struct Scratch {
    stack: Vec<u32>,
    open: Vec<OpenNode>,
    d2: Vec<u32>,
}

/// A named node whose right-hand side is not done yet.
#[derive(Clone, Copy)]
struct OpenNode {
    kind: u8,
    /// A terminal was consumed while this node was the innermost open one.
    own: bool,
    /// File index of its first token.
    tok: u32,
    /// Item-relative count of the nodes emitted before it opened.
    at: u32,
}

impl OpenNode {
    fn new(kind: u8, tok: usize, at: u32) -> OpenNode {
        OpenNode { kind, own: false, tok: tok as u32, at }
    }
}

/// The PDA state at an item's first error.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fail {
    /// The stack entry on top.
    top: u32,
    /// File index of the lookahead token (`end` for the virtual EOF).
    la: usize,
    /// Lookahead kind as the PDA saw it (after splits; `ERR_COL` for error tokens).
    la_kind: u16,
    /// Bytes of a split token already consumed.
    shift: u32,
    /// The innermost open D2 `<` token.
    d2_lt: Option<u32>,
    /// The innermost open named node.
    open_kind: Option<u8>,
    /// Its first token.
    open_tok: Option<u32>,
}

#[inline(always)]
fn push_all(v: &mut Vec<u32>, xs: &[u32]) {
    v.reserve(xs.len());
    for &x in xs {
        v.push(x);
    }
}

/// Runs the PDA over `toks[begin..end]` from `[EOF, start]`, appending the item's nodes.
pub(crate) fn run<T: Trace>(
    toks: &[Token],
    begin: usize,
    end: usize,
    start: u16,
    nodes: &mut Vec<Node>,
    s: &mut Scratch,
    trace: &mut T,
) -> Result<(), Fail> {
    let t = tables();
    let end = end.min(toks.len());
    let kind_at = |i: usize| -> u16 { if i < end { (toks[i].kind as u16).min(ERR_COL) } else { EOF } };
    s.stack.clear();
    s.open.clear();
    s.d2.clear();
    s.stack.push(EOF as u32);
    s.stack.push(NT_BIT | start as u32);
    let base = nodes.len();
    let mut i = begin;
    let mut cur = kind_at(i);
    let mut shift = 0u32;
    while let Some(mut e) = s.stack.pop() {
        let sym = e & SYM_MASK;
        if sym == CLOSE {
            if let Some(o) = s.open.pop() {
                let b = (nodes.len() - base) as u32 - o.at;
                // a pass-through node over exactly one child (its last node, of subtree size
                // b - 1) and no terminal of its own is left out
                let bare =
                    t.pass[o.kind as usize] && !o.own && b > 0 && nodes.last().is_some_and(|c| c.b + 1 == b);
                if !bare {
                    nodes.push(Node { kind: o.kind, a: o.tok, b });
                }
                if T::ON {
                    trace.close();
                }
            }
            continue;
        }
        if sym & NT_BIT != 0 {
            let nt = (sym & !NT_BIT) as usize;
            let mi = if T::ON || e & (F_CAST | F_D2) != 0 { SLOW } else { t.mcell[nt * W + cur as usize] };
            if mi == EPS {
                continue;
            }
            if mi != SLOW {
                let m = t.macros[mi as usize];
                let count = (nodes.len() - base) as u32;
                let mut emitted = 0;
                for &op in &t.mops[m.ops as usize..m.ops_end as usize] {
                    if op & EMIT != 0 {
                        nodes.push(Node { kind: op as u8, a: i as u32, b: (op & !EMIT) >> 8 });
                        emitted += 1;
                    } else {
                        s.open.push(OpenNode::new(op as u8, i, count + emitted));
                    }
                }
                push_all(&mut s.stack, &t.mpush[m.push as usize..m.push_end as usize]);
                if m.term == NO_TERM {
                    continue;
                }
                e = m.term; // the matching terminal: consumed below without a push and pop
            } else {
                let Some(p) = cell_prod(t, nt, cur) else {
                    return Err(fail(e, i, cur, shift, s));
                };
                if t.named[nt] {
                    s.open.push(OpenNode::new(nt as u8, i, (nodes.len() - base) as u32));
                    s.stack.push(CLOSE);
                    if T::ON {
                        trace.open(nt as u8, i);
                    }
                }
                let rhs = &t.rhs[t.off[p] as usize..t.off[p + 1] as usize];
                if e & F_CAST == 0 && t.cast_seed[p] == 0 {
                    push_all(&mut s.stack, rhs);
                } else {
                    push_flagged(t, e, p, nt, rhs, i, s);
                }
                continue;
            }
        }
        // A terminal on top.
        let sym = (e & SYM_MASK) as u16;
        if let Some(o) = s.open.last_mut() {
            o.own = true;
        }
        if sym == cur {
            if T::ON {
                trace.term(i, sym, ((e >> PROD_SHIFT) & PROD_MASK) as u16);
            }
            if e & F_D2 != 0 {
                s.d2.pop();
            }
            i += 1;
            cur = kind_at(i);
            shift = 0;
            continue;
        }
        if term_accepts(t, sym, cur) {
            if T::ON {
                trace.term(i, sym, ((e >> PROD_SHIFT) & PROD_MASK) as u16);
            }
            if e & F_D2 != 0 {
                s.d2.pop();
            }
            cur = t.split_rest[cur as usize];
            shift += 1;
            continue;
        }
        return Err(fail(e, i, cur, shift, s));
    }
    Ok(())
}

/// Pushes a right-hand side while tracking D2 (contracts/parse-errors.md "d2"): the `Type` of a
/// cast is flagged, the flag follows the right edge of the type (past nullable suffixes), and a
/// flagged `GenericArgs` (entered through the `(TypeBase, "<")` override) opens a D2 entry closed
/// by its `>`.
#[cold]
fn push_flagged(t: &Tables, e: u32, p: usize, nt: usize, rhs: &[u32], i: usize, s: &mut Scratch) {
    let start = s.stack.len();
    s.stack.extend_from_slice(rhs);
    let seed = t.cast_seed[p] as usize;
    if seed > 0 {
        s.stack[start + seed - 1] |= F_CAST;
    }
    if e & F_CAST != 0 {
        for (k, &x) in rhs.iter().enumerate() {
            let sym = x & SYM_MASK;
            if sym & NT_BIT == 0 {
                break;
            }
            s.stack[start + k] |= F_CAST;
            if !t.nullable[(sym & !NT_BIT) as usize] {
                break;
            }
        }
        if nt == t.generic_args && !rhs.is_empty() {
            s.d2.push(i as u32);
            s.stack[start] |= F_D2;
        }
    }
}

#[cold]
fn fail(top: u32, la: usize, la_kind: u16, shift: u32, s: &Scratch) -> Fail {
    Fail {
        top,
        la,
        la_kind,
        shift,
        d2_lt: s.d2.last().copied(),
        open_kind: s.open.last().map(|o| o.kind),
        open_tok: s.open.last().map(|o| o.tok),
    }
}

/// Terminals with an entry in nonterminal `nt`'s row, in terminal order.
fn row_terms(nt: usize) -> Vec<u16> {
    let t = tables();
    (0..NTERM as u16).filter(|&x| t.table[nt * W + x as usize] >= 0).collect()
}

fn join_terms(ts: &[u16]) -> String {
    let mut s = String::new();
    for (k, &x) in ts.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        s.push_str(TERM_TEXT.get(x as usize).copied().unwrap_or("ERROR"));
    }
    s
}

fn tok_span(src: &[u8], toks: &[Token], file: FileId, i: usize, shift: u32) -> Span {
    let Some(t) = toks.get(i) else { return Span::new(file, src.len().min(u32::MAX as usize) as u32, 0) };
    let off = t.start.saturating_add(shift);
    let end = token_end(src, toks, i).max(off);
    Span::new(file, off, end - off)
}

/// E0004 at token `i` with the given expected set.
fn e0004(
    src: &[u8],
    toks: &[Token],
    file: FileId,
    i: usize,
    shift: u32,
    found: &str,
    expected: &[u16],
) -> Option<Diagnostic> {
    let expected = join_terms(expected);
    make_diag(
        "E0004",
        None,
        &[("found", found), ("expected", &expected)],
        tok_span(src, toks, file, i, shift),
        &[],
    )
}

/// The diagnostic for an item's first error (contracts/parse-errors.md).
fn error_diag(
    src: &[u8],
    toks: &[Token],
    file: FileId,
    begin: usize,
    end: usize,
    f: &Fail,
) -> Option<Diagnostic> {
    let sym = f.top & SYM_MASK;
    let top_term = (sym & NT_BIT == 0).then_some(sym as u16);
    let top_nt = (sym & NT_BIT != 0).then_some((sym & !NT_BIT) as usize);
    let kind = |j: usize| -> Option<u8> { (j >= begin && j < end).then(|| toks[j].kind) };
    let is = |j: Option<usize>, t: Term| j.and_then(kind) == Some(t as u8);
    let li = f.la;
    let back = |k: usize| li.checked_sub(k);
    // 1. E0013: a terminal of rule Unsafe (`(` or StrLit) on top.
    if let Some(tt) = top_term
        && (tt == Term::LParen as u16 || tt == Term::StrLit as u16)
        && f.open_kind == Some(NodeKind::Unsafe.id())
    {
        let at = (1..=3).filter_map(back).find(|&j| kind(j) == Some(Term::KwUnsafe as u8)).unwrap_or(li);
        return make_diag("E0013", None, &[], tok_span(src, toks, file, at, 0), &[]);
    }
    // 2. E0011: inside TypeArgs opened by D2 after a cast type.
    if let Some(lt_tok) = f.d2_lt {
        return make_diag("E0011", None, &[], tok_span(src, toks, file, lt_tok as usize, 0), &[]);
    }
    // 3. E0014: `Name { … }` struct literal or pattern.
    if f.shift == 0 {
        let la = f.la_kind;
        let ident = |j: Option<usize>| is(j, Term::Ident);
        let shape_a = la == Term::LBrace as u16
            && ident(back(1))
            && (is(Some(li + 1), Term::RBrace)
                || (ident(Some(li + 1))
                    && (is(Some(li + 2), Term::Colon)
                        || is(Some(li + 2), Term::Comma)
                        || is(Some(li + 2), Term::RBrace))));
        let shape_b =
            la == Term::Colon as u16 && ident(back(1)) && is(back(2), Term::LBrace) && ident(back(3));
        let at = if shape_a {
            Some(li)
        } else if shape_b {
            back(2)
        } else {
            None
        };
        if let Some(brace) = at {
            let path = name_chain(src, toks, begin, brace - 1);
            return make_diag("E0014", None, &[("path", &path)], tok_span(src, toks, file, brace, 0), &[]);
        }
    }
    // 4. E0012: a second CmpOp after a complete CmpExpr.
    let cmp = [Term::EqEq, Term::BangEq, Term::Lt, Term::LtEq, Term::Gt, Term::GtEq];
    if let Some(nt) = top_nt
        && cmp.iter().any(|&c| c as u16 == f.la_kind)
        && lt::NT_ORIGIN[nt] == NodeKind::AndExpr.nt()
    {
        return make_diag("E0012", None, &[], tok_span(src, toks, file, li, f.shift), &[]);
    }
    // 5–13: Rust forms.
    if f.shift == 0
        && let Some(d) = rust_form(src, toks, file, begin, end, f)
    {
        return Some(d);
    }
    // E0004.
    let expected = match (top_term, top_nt) {
        (Some(tt), _) => vec![tt],
        (None, Some(nt)) => row_terms(nt),
        _ => Vec::new(),
    };
    let found = if li < end {
        TERM_TEXT.get(f.la_kind as usize).copied().unwrap_or("ERROR")
    } else {
        toks.get(li).map_or("EOF", |t| kind_name(t.kind))
    };
    e0004(src, toks, file, li, f.shift, found, &expected)
}

/// Rules 5–13 of contracts/parse-errors.md: the Rust forms with their own codes.
fn rust_form(
    src: &[u8],
    toks: &[Token],
    file: FileId,
    begin: usize,
    end: usize,
    f: &Fail,
) -> Option<Diagnostic> {
    let sym = f.top & SYM_MASK;
    let top_term = (sym & NT_BIT == 0).then_some(sym as u16);
    // the origin rule of the nonterminal on top / of the rule that pushed the terminal on top
    let origin = |nt: u32| lt::NT_ORIGIN.get(nt as usize).copied();
    let top_origin = if sym & NT_BIT != 0 { origin(sym & !NT_BIT) } else { None };
    let term_origin = if sym & NT_BIT == 0 { origin((f.top >> PROD_SHIFT) & PROD_MASK) } else { None };
    let of = |o: Option<u16>, k: NodeKind| o == Some(k.nt());
    let li = f.la;
    let la = f.la_kind;
    let kind = |j: usize| -> Option<u8> { (j >= begin && j < end).then(|| toks[j].kind) };
    let is = |j: Option<usize>, t: Term| j.and_then(kind) == Some(t as u8);
    let back = |k: usize| li.checked_sub(k);
    let word = |j: Option<usize>, w: &[u8]| {
        is(j, Term::Ident) && j.is_some_and(|j| crate::lex::token_text(src, toks, j) == w)
    };
    let at = |j: usize| tok_span(src, toks, file, j, 0);
    // 5. E0019: `if let` / `while let`.
    if la == Term::KwLet as u16 {
        for (t, kw) in [(Term::KwIf, "if"), (Term::KwWhile, "while")] {
            if is(back(1), t) {
                return make_diag("E0019", None, &[("kw", kw)], at(li - 1), &[]);
            }
        }
    }
    // 6. E0020: `let P = e else`.
    if la == Term::KwElse as u16
        && top_term == Some(Term::Semi as u16)
        && of(term_origin, NodeKind::Stmt)
        && is(f.open_tok.map(|t| t as usize), Term::KwLet)
    {
        return make_diag("E0020", None, &[], at(li), &[]);
    }
    // 7. E0021: an expression statement with no AssignOp and no `;` before `}`.
    if la == Term::RBrace as u16
        && of(top_origin, NodeKind::ExprStmt)
        && let Some(first) = f.open_tok
    {
        return make_diag("E0021", None, &[], at(first as usize), &[]);
    }
    // 8. E0022: `where` in the item header (before the item's first depth-0 `{`).
    if word(Some(li), b"where") {
        let mut depth = 0i32;
        let mut header = true;
        for t in &toks[begin..li] {
            match t.term() {
                Some(Term::LBrace) if depth == 0 => {
                    header = false;
                    break;
                }
                Some(Term::LParen | Term::LBracket | Term::LBrace | Term::DotLBrace) => depth += 1,
                Some(Term::RParen | Term::RBracket | Term::RBrace) => depth -= 1,
                _ => {}
            }
        }
        if header {
            return make_diag("E0022", None, &[], at(li), &[]);
        }
    }
    // 9. E0023: `::` not before generic args.
    if is(back(1), Term::ColonColon) && la != Term::Lt as u16 {
        return make_diag("E0023", None, &[], at(li - 1), &[]);
    }
    if la == Term::ColonColon as u16 && !is(Some(li + 1), Term::Lt) {
        return make_diag("E0023", None, &[], at(li), &[]);
    }
    // 10. E0025: `mut` before a parameter.
    if la == Term::KwMut as u16 && (is(Some(li + 1), Term::Ident) || is(Some(li + 1), Term::KwSelf)) {
        let list = [
            NodeKind::Param,
            NodeKind::Params,
            NodeKind::XParams,
            NodeKind::FnDecl,
            NodeKind::TraitFn,
            NodeKind::KernelDecl,
            NodeKind::DispatchDecl,
            NodeKind::ExternItem,
            NodeKind::ExternFn,
        ];
        let closure =
            of(top_origin, NodeKind::Primary) && is(back(1), Term::LParen) && is(back(2), Term::KwFn);
        if closure || list.iter().any(|&k| of(top_origin, k)) {
            return make_diag("E0025", None, &[], at(li), &[]);
        }
    }
    // 11. E0026: tuple struct / unit struct.
    if (la == Term::LParen as u16 || la == Term::Semi as u16)
        && is(back(1), Term::Ident)
        && is(back(2), Term::KwStruct)
        && of(top_origin, NodeKind::AdtDecl)
    {
        let name = String::from_utf8_lossy(crate::lex::token_text(src, toks, li - 1)).into_owned();
        let case = (la == Term::Semi as u16).then_some("unit");
        return make_diag("E0026", case, &[("name", &name)], at(li), &[]);
    }
    let in_type = of(top_origin, NodeKind::Type) || of(top_origin, NodeKind::TypeBase);
    // 12. E0027: `!` as a type.
    if la == Term::Bang as u16 && (in_type || of(top_origin, NodeKind::RetClause)) {
        // after `->` the fix is exact: `never` is a type there
        let case = is(back(1), Term::MinusGt).then_some("ret");
        return make_diag("E0027", case, &[], at(li), &[]);
    }
    // 13. E0028: `impl Trait` / `dyn Trait`.
    if la == Term::KwImpl as u16 && in_type {
        return make_diag("E0028", None, &[("kw", "impl")], at(li), &[]);
    }
    if la == Term::Ident as u16 && word(back(1), b"dyn") {
        return make_diag("E0028", None, &[("kw", "dyn")], at(li - 1), &[]);
    }
    None
}

/// Source text of the dotted name chain `Ident ('.' Ident)*` ending at token `last`.
fn name_chain(src: &[u8], toks: &[Token], begin: usize, last: usize) -> String {
    let mut first = last;
    while first >= begin + 2
        && toks[first - 1].kind == Term::Dot as u8
        && toks[first - 2].kind == Term::Ident as u8
    {
        first -= 2;
    }
    let s = toks[first].start as usize;
    let e = token_end(src, toks, last) as usize;
    String::from_utf8_lossy(src.get(s..e).unwrap_or_default()).into_owned()
}

/// One parsed item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedItem {
    /// Postorder nodes (empty when dropped).
    pub nodes: Vec<Node>,
    /// False when the item is dropped (a syntax error, an error token, a pre-item run, a late
    /// import).
    pub ok: bool,
    /// The item's syntax diagnostic (none when ok, or when it holds an error token).
    pub diagnostic: Option<Diagnostic>,
}

/// Parses one item of a file's cut: `tokens` is the whole file's token list.
pub fn parse_item(src: &[u8], file: FileId, tokens: &[Token], item: ItemSpan) -> ParsedItem {
    let mut nodes = Vec::new();
    let (ok, diagnostic) =
        parse_item_into::<NoTrace>(src, file, tokens, item, &mut nodes, &mut Scratch::default());
    ParsedItem { nodes, ok, diagnostic }
}

/// Parses one item, appending its nodes to `nodes` (nothing is appended when it is dropped).
fn parse_item_into<T: Trace + TraceNew>(
    src: &[u8],
    file: FileId,
    tokens: &[Token],
    item: ItemSpan,
    nodes: &mut Vec<Node>,
    s: &mut Scratch,
) -> (bool, Option<Diagnostic>) {
    let begin = item.start as usize;
    let end = (item.end as usize).min(tokens.len());
    if begin >= end {
        return (false, None);
    }
    if tokens[begin..end].iter().any(|t| t.kind == KIND_ERROR) {
        return (false, None); // lexical diagnostics only, no E0004
    }
    let start = match item.kind {
        ItemKind::PreItem => {
            let expected = row_terms(NodeKind::File.nt() as usize);
            let found = kind_name(tokens[begin].kind);
            return (false, e0004(src, tokens, file, begin, 0, found, &expected));
        }
        ItemKind::LateImport => {
            let mut expected = row_terms(NodeKind::File.nt() as usize);
            expected.retain(|&t| t != Term::KwImport as u16);
            return (false, e0004(src, tokens, file, begin, 0, "import", &expected));
        }
        ItemKind::Import => NodeKind::Import.nt(),
        ItemKind::Item => NodeKind::Item.nt(),
    };
    let before = nodes.len();
    match run(tokens, begin, end, start, nodes, s, &mut T::new()) {
        Ok(()) => match empty_obligation(src, tokens, begin, end) {
            // E0013 case `empty` (contracts/parse-errors.md "Empty obligation"): dropped
            Some(at) => {
                nodes.truncate(before);
                (false, make_diag("E0013", Some("empty"), &[], tok_span(src, tokens, file, at, 0), &[]))
            }
            None => (true, None),
        },
        Err(f) => {
            nodes.truncate(before);
            (false, error_diag(src, tokens, file, begin, end, &f))
        }
    }
}

/// The first `unsafe` token of `tokens[begin..end]` followed by `(` and the empty string literal
/// `""` (an empty obligation is E0013).
fn empty_obligation(src: &[u8], tokens: &[Token], begin: usize, end: usize) -> Option<usize> {
    (begin..end.saturating_sub(2)).find(|&i| {
        tokens[i].kind == Term::KwUnsafe as u8
            && tokens[i + 1].kind == Term::LParen as u8
            && tokens[i + 2].kind == Term::StrLit as u8
            && token_end(src, tokens, i + 2).saturating_sub(tokens[i + 2].start) == 2
    })
}

/// Builds a fresh trace per item.
pub(crate) trait TraceNew {
    fn new() -> Self;
}

impl TraceNew for NoTrace {
    fn new() -> Self {
        NoTrace
    }
}

impl TraceNew for Plain {
    fn new() -> Self {
        Plain
    }
}

/// Items are parsed in chunks of consecutive items of about this many tokens (one node buffer
/// per chunk). Chunk bounds depend only on the token counts, never on the thread count.
const CHUNK_TOKENS: u32 = 8192;

/// The results of one chunk: its nodes and, per item, (ok, node count, diagnostic).
struct ChunkOut {
    nodes: Vec<Node>,
    items: Vec<(bool, u32, Option<Diagnostic>)>,
}

fn chunks(table: &ItemTable) -> Vec<&[ItemSpan]> {
    let mut out = Vec::new();
    let (mut lo, mut acc) = (0, 0u32);
    for (k, it) in table.items.iter().enumerate() {
        acc = acc.saturating_add(it.end.saturating_sub(it.start));
        if acc >= CHUNK_TOKENS {
            out.push(&table.items[lo..=k]);
            lo = k + 1;
            acc = 0;
        }
    }
    if lo < table.items.len() {
        out.push(&table.items[lo..]);
    }
    out
}

fn parse_chunks<T: Trace + TraceNew>(
    src: &[u8],
    file: FileId,
    tokens: &[Token],
    table: &ItemTable,
) -> Vec<ChunkOut> {
    chunks(table)
        .into_par_iter()
        .map_init(Scratch::default, |s, items| {
            let ntok: usize = items.iter().map(|it| it.end.saturating_sub(it.start) as usize).sum();
            let mut out = ChunkOut {
                nodes: Vec::with_capacity(ntok * 3 / 5 + 16),
                items: Vec::with_capacity(items.len()),
            };
            for &it in items {
                let before = out.nodes.len();
                let (ok, d) = parse_item_into::<T>(src, file, tokens, it, &mut out.nodes, s);
                out.items.push((ok, (out.nodes.len() - before) as u32, d));
            }
            out
        })
        .collect()
}

/// An item of a parsed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemInfo {
    pub kind: ItemKind,
    /// Token range `[start, end)`.
    pub start: u32,
    pub end: u32,
    /// Parsed without error (not dropped).
    pub ok: bool,
    /// The buffer of [`Ast::chunks`] holding the item's nodes.
    pub chunk: u32,
    /// The item's nodes in that buffer (empty when dropped).
    pub nodes: std::ops::Range<u32>,
}

/// Why a file has no items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileStop {
    /// E0002.
    TooLarge,
    /// E0001.
    InvalidUtf8,
    /// E0016.
    Unbalanced,
    /// E0005.
    TooDeep,
}

/// A lexed and parsed file.
#[derive(Clone, Debug)]
pub struct ParsedFile {
    pub file: FileId,
    /// All tokens, ending in `EOF` (see [`crate::Lexed::tokens`]).
    pub tokens: Vec<Token>,
    /// The items in file order (none when [`ParsedFile::stop`] is set).
    pub items: Vec<ItemInfo>,
    /// Nodes of the ok items.
    pub ast: Ast,
    /// Lexical and syntax diagnostics, sorted by (offset, code, message).
    pub diagnostics: Vec<Diagnostic>,
    pub stop: Option<FileStop>,
}

impl ParsedFile {
    /// The AST of item `i` (empty for a dropped item or an index out of range).
    pub fn item_ast(&self, i: usize) -> ItemAst<'_> {
        let nodes = self
            .items
            .get(i)
            .and_then(|it| {
                self.ast.chunks.get(it.chunk as usize)?.get(it.nodes.start as usize..it.nodes.end as usize)
            })
            .unwrap_or_default();
        ItemAst { nodes }
    }

    /// Byte range `[O_start, O_end)` of item `i`: its first token's start and its last token's
    /// end (the `items` dump fields).
    pub fn item_offsets(&self, src: &[u8], i: usize) -> Option<(u32, u32)> {
        let it = self.items.get(i)?;
        let first = self.tokens.get(it.start as usize)?.start;
        let last = it.end.checked_sub(1)? as usize;
        Some((first, token_end(src, &self.tokens, last)))
    }

    /// The source bytes of token `i`.
    pub fn token_text<'a>(&self, src: &'a [u8], i: usize) -> &'a [u8] {
        crate::lex::token_text(src, &self.tokens, i)
    }
}

/// Parses the items of a lexed, balanced file in parallel (rayon, current pool).
pub fn parse_items(src: &[u8], file: FileId, tokens: &[Token], table: &ItemTable) -> Vec<ParsedItem> {
    let mut out = Vec::with_capacity(table.items.len());
    for c in parse_chunks::<NoTrace>(src, file, tokens, table) {
        let mut at = 0usize;
        for (ok, n, diagnostic) in c.items {
            let nodes = c.nodes[at..at + n as usize].to_vec();
            at += n as usize;
            out.push(ParsedItem { nodes, ok, diagnostic });
        }
    }
    out
}

/// Lexes, cuts, and parses one file. Never panics.
pub fn parse_file(src: &[u8], file: FileId) -> ParsedFile {
    parse_file_with::<NoTrace>(src, file)
}

/// [`parse_file`] without the precomputed expansion macros: every PDA step one by one. The
/// result is identical (tests check this); it exists only as the reference for that check.
#[doc(hidden)]
pub fn parse_file_stepwise(src: &[u8], file: FileId) -> ParsedFile {
    parse_file_with::<Plain>(src, file)
}

fn parse_file_with<T: Trace + TraceNew>(src: &[u8], file: FileId) -> ParsedFile {
    let lexed = lex(src, file);
    let mut diagnostics = lexed.diagnostics;
    let tokens = lexed.tokens;
    let mut out = ParsedFile {
        file,
        tokens: Vec::new(),
        items: Vec::new(),
        ast: Ast::default(),
        diagnostics: Vec::new(),
        stop: None,
    };
    if let Some(stop) = lexed.stop {
        out.stop = Some(match stop {
            LexStop::TooLarge => FileStop::TooLarge,
            LexStop::InvalidUtf8 { .. } => FileStop::InvalidUtf8,
        });
    } else {
        match items(&tokens, file) {
            Err(e) => {
                out.stop = Some(match e {
                    crate::items::ItemsError::Unbalanced(_) => FileStop::Unbalanced,
                    crate::items::ItemsError::TooDeep(_) => FileStop::TooDeep,
                });
                diagnostics.extend(e.into_diagnostic());
            }
            Ok(table) => {
                let parsed = parse_chunks::<T>(src, file, &tokens, &table);
                let mut spans = table.items.iter();
                for (k, c) in parsed.into_iter().enumerate() {
                    let mut lo = 0;
                    for ((ok, n, d), span) in c.items.into_iter().zip(spans.by_ref()) {
                        out.items.push(ItemInfo {
                            kind: span.kind,
                            start: span.start,
                            end: span.end,
                            ok,
                            chunk: k as u32,
                            nodes: lo..lo + n,
                        });
                        lo += n;
                        diagnostics.extend(d);
                    }
                    out.ast.chunks.push(c.nodes);
                }
            }
        }
    }
    diagnostics.sort_by(|a, b| {
        let key = |d: &Diagnostic| d.primary().map(|s| s.offset);
        key(a).cmp(&key(b)).then_with(|| a.code().cmp(b.code())).then_with(|| a.message().cmp(b.message()))
    });
    out.tokens = tokens;
    out.diagnostics = diagnostics;
    out
}
