//! The whole-program call graph over the monomorphized program ([`CallGraph`], built by
//! [`crate::check`] after the instantiation depth check: [`crate::Typeck::call_graph`]).
//!
//! **Nodes.** A fn or method instance (decl and flat generic args, [`InstKey`]), a closure body of
//! an instance (`@<enclosing>.<k>`: a closure literal or a `spawn` body), one drop glue per value
//! type that needs dropping, and one fn-value target table per fn type. The instances are those
//! reachable from every non-generic fn and method with a checked body (every package, `test fn`s
//! included), so the node set depends neither on the build mode nor on `main`. A `dispatch fn` is
//! not a node: a call of it, or its use as a fn value, stands for its arm targets (those the build
//! target's arch owns, [`crate::dispatch`]). A fn without a checked body (an `extern` import, a
//! body with errors) is a node without out-edges.
//!
//! **Edges**, one per (caller, callee) at the first site in source order:
//! - [`EdgeKind::Call`]: a call the source spells (fn, method, trait method through a bound or
//!   qualified, user-type operator) or one it implies: a `for` loop's `next`, an f-string's
//!   `String.new` and `String.push_str` and each hole's `Display.fmt`, and `String.as_str` for a
//!   `&String` or f-string used as `&str`;
//! - [`EdgeKind::Dispatch`]: a call of a dispatch fn, to each of its arm targets;
//! - [`EdgeKind::Indirect`]: a call through a fn value (or a built-in method's call of a fn value
//!   it is passed), to each value that can flow there, or to the table of its fn type where an
//!   untracked value can (below);
//! - [`EdgeKind::Member`]: a table, to each fn and closure of its type made into a fn value
//!   anywhere (site: the first place the walk finds it made);
//! - [`EdgeKind::Spawn`]: the fn or closure holding a `spawn` statement, to the spawn body, and
//!   the one passing a closure literal to `runtime.thread.start` (which runs it on a new
//!   thread), to that closure;
//! - [`EdgeKind::Drop`]: a fn or closure to the glue of each value type it may drop
//!   ([`crate::typeck::TypedBody::drops`]: its locals and the values it makes; a value it drops is
//!   one of them or a part of one), and a glue to its type's `Drop` impl and to the glue of each
//!   part (field, element) that needs dropping, as lowering's drop glue does. A recursive value's
//!   drop is then a cycle through glue: [`CallGraph::is_drop_cycle`].
//!
//! Calls of runtime helpers lowering inserts are not edges (`set_user_site`, the error-trace
//! hooks `runtime.errtrace.push` / `restart`, `scope_spawn` / `scope_join`, traps, CPU
//! detection), nor is the backend's thread start stub calling
//! `runtime.thread.entry`.
//!
//! **Tables.** A fn value is made by a fn item used as a value (coercion `FnItem`), a closure
//! literal, or a `spawn` body (which `runtime.thread` runs as a `fn()` value). The only coercion
//! between fn types keeps the params and the return type and changes the features or the deny
//! clause (typeck), so a table is keyed by the params and the return type, features and deny
//! clause cleared, and holds every value of any fn type with that key: the upper bound of what a
//! call through such a value reaches. Every `Indirect` edge of a call carries the deny clause of
//! the called value's type ([`Edge::deny`]).
//!
//! **Fn-value flow.** A fn value lives only in a local, a param, or a view field, so the values a
//! call through one reaches are tracked by slots: each local and param of an instance (a closure's
//! belong to its enclosing instance), and each struct field. A slot holds what flows in: `let x =
//! e` and `x = e`, `s.f = e` and `S.{ f: e }`, the argument at that param position of every direct
//! call of the instance, and, for a call through a fn value, of every value the call reaches; the
//! value of `e` is a fn item, a closure literal, another slot (a local or a field read), or
//! untracked. Copy constraints between slots are solved to a fixpoint on flat arrays; a call
//! through a fn value then reaches the values its slot holds, or, if an untracked value flows in,
//! its table. Untracked: a fn value returned by a call, chosen by a value `if` / `match`, read from
//! a tuple, array, or enum payload, through a reference or raw pointer, or bound by a pattern
//! other than `let x`; each passes the whole table on.
//!
//! **Specialization.** A direct call that passes known fn values (fn items, closure literals, a
//! local set once to one, or a param that is itself specialized) reaches a node of the callee for
//! those values ([`NodeKind::Fn`] `spec`), so a helper such as `apply(f, x)` called with `g` from
//! one place and, inside `g`, with a closure from another is two nodes and no cycle. Calls whose
//! values are not known share the plain node; at most [`SPEC_LIMIT`] specialized nodes per
//! instance, then the plain node.
//!
//! **Roots** ([`Root`]): `main` of the root package, each `spawn` body, and each fn item or closure
//! literal passed to `std.thread.join` (both args), `par_for`, or `par_for_mut` (`body`).
//!
//! **Derived data**, on flat arrays: CSR out-edges and in-edges; strongly connected components by
//! trimming (peeling nodes without in- or without out-edges), then forward max-label propagation
//! and one backward search per label, repeated on what is left (no depth-first search); SCC ids
//! ordered by level, where level 0 has no edge to another SCC and every edge between two SCCs goes
//! from a higher level to a lower one (longest path to a sink, by level-synchronous peeling). A
//! pass over levels 0, 1, .. sees every callee's SCC before its callers'.
//!
//! The walk is sequential (one type overlay, discovery order = node order); extracting each
//! body's call events runs per body in parallel. Node ids are deterministic.

use std::fmt::Write as _;

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use stave_diag::{FileId, Span};
use stave_syntax::{NodeKind as Syn, Term};

use crate::Program;
use crate::closures::ClosureKind;
use crate::decls::{DeclFlags, DeclKind};
use crate::ids::DeclId;
use crate::itemcx::{Brackets, ItemCx};
use crate::mono::{InstKey, Resolved};
use crate::sigs::Sigs;
use crate::traits::{MethodRef, World};
use crate::typeck::{Coercion, Target, Typeck, TypedBody};
use crate::types::{GArg, Interner, Overlay, TypeId, TypeKind, TypeTable};

/// What a node stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A fn or method instance. `body`: it has a checked body (else no out-edges). `spec`: per
    /// param, the fn values every call of this node passes there (empty: any), so a fn that takes
    /// a fn value has one node per known value it is called with ([`CallGraph`] docs).
    Fn { inst: InstKey, rec: bool, body: bool, spec: Vec<Vec<u32>> },
    /// Closure body `index` (`@<enclosing>.<index>`) of the instance node `parent`.
    Closure { parent: u32, index: u32, spawn: bool },
    /// Fn-value target table `table` ([`CallGraph::tables`]).
    Table { table: u32 },
    /// The drop of a value of type `ty`: its `Drop` impl, then the drop of each part.
    Glue { ty: TypeId },
}

/// One node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    /// A fn: its name token; a closure: its `fn` / `spawn` token; a table: the site of its first
    /// member, else of its first call (every table has one or the other).
    pub span: Span,
}

impl Node {
    /// Declared `rec fn` (closures and tables never are).
    pub fn rec(&self) -> bool {
        matches!(self.kind, NodeKind::Fn { rec: true, .. })
    }
}

/// How an edge arises (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Call,
    /// A call of dispatch fn `decl`, to one of its arm targets.
    Dispatch {
        decl: DeclId,
    },
    Indirect,
    Member,
    Spawn,
    /// A drop: an owner to the glue of a value type it may drop, a glue to its `Drop` impl and to
    /// the glue of its parts.
    Drop,
}

/// One out-edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    pub to: u32,
    pub kind: EdgeKind,
    /// The first site in source order: the callee's name token, an operator, the `for` token,
    /// the f-string, the coerced expression's first token; for [`EdgeKind::Member`] where the
    /// fn value is made.
    pub site: Span,
    /// Effects ([`crate::effects::Effects`] bits) no target reached through this edge has: for
    /// an [`EdgeKind::Indirect`] edge, the deny clause of the fn type of its calls (of several
    /// calls to one table, what all of them deny); 0 for every other edge.
    pub deny: u8,
}

/// A fn-value target table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    /// The key: a fn type with no features ([`CallGraph::kind`] reads it).
    pub ty: TypeId,
    pub node: u32,
}

/// Why a node starts a thread's stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootKind {
    Main,
    Spawn,
    /// Argument `arg` of a call of `std.thread` fn `callee` (`join`, `par_for`, `par_for_mut`).
    ThreadArg {
        callee: DeclId,
        arg: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Root {
    pub node: u32,
    pub kind: RootKind,
    /// `main`'s name token, the `spawn` token, the argument.
    pub site: Span,
}

/// The call graph (module docs). Every array is flat and indexed by node, edge, SCC, or level.
#[derive(Clone, Debug, Default)]
pub struct CallGraph {
    pub nodes: Vec<Node>,
    /// Out-edges of node `n`: `out[out_start[n]..out_start[n + 1]]`, sorted by target.
    pub out_start: Vec<u32>,
    pub out: Vec<Edge>,
    /// In-edges of node `n`: indices into `out`, `in_edges[in_start[n]..in_start[n + 1]]`, sorted by
    /// caller.
    pub in_start: Vec<u32>,
    pub in_edges: Vec<u32>,
    /// Sorted by first appearance; [`Table::node`] is its node.
    pub tables: Vec<Table>,
    /// In discovery order, without duplicates.
    pub roots: Vec<Root>,
    /// The SCC of each node.
    pub scc_of: Vec<u32>,
    /// Nodes of SCC `s`: `scc_nodes[scc_start[s]..scc_start[s + 1]]`, ascending.
    pub scc_start: Vec<u32>,
    pub scc_nodes: Vec<u32>,
    /// The level of each SCC; SCC ids ascend with the level.
    pub scc_level: Vec<u32>,
    /// SCCs of level `l`: `level_start[l]..level_start[l + 1]`.
    pub level_start: Vec<u32>,
    /// Per SCC: a drop cycle ([`CallGraph::is_drop_cycle`]).
    pub scc_drop: Vec<bool>,
    /// Types the instances needed beyond typeck's table: [`TypeId`]s with the local bit index
    /// this list ([`CallGraph::types`]).
    pub local_types: Vec<TypeKind>,
    /// Instances were left out: an instantiation chain too deep (E0902 reported, or a chain only
    /// fn values or implied calls make), or too many instances. The graph is then incomplete.
    pub truncated: bool,
    /// Every call site with each node it reaches: (calling fn or closure node, call token, callee
    /// node), sorted. The token is the one typeck keys the call by (a call's `(`, a method's
    /// name, an operator, the `for` token, an f-string); a fn-value call reaches the values that
    /// flow there or its table. [`CallGraph::out`] keeps one edge per (caller, callee) only.
    pub calls: Vec<(u32, u32, u32)>,
}

/// Most instances of one generic on one discovery chain (lowering's limit).
const CHAIN_LIMIT: u32 = 4 * crate::mono::INST_DEPTH;

/// Most nodes.
const NODE_BUDGET: usize = 1 << 21;

/// Most specialized nodes of one instance; further calls with other known fn values share its
/// unspecialized node.
const SPEC_LIMIT: u32 = 16;

/// Marks a table index in an edge target before the table nodes exist.
const TABLE_BIT: u32 = 1 << 31;

impl CallGraph {
    pub fn succ(&self, n: u32) -> &[Edge] {
        let (a, b) = (self.out_start[n as usize], self.out_start[n as usize + 1]);
        &self.out[a as usize..b as usize]
    }

    /// The in-edges of `n`: (caller, edge).
    pub fn pred(&self, n: u32) -> impl Iterator<Item = (u32, &Edge)> {
        let (a, b) = (self.in_start[n as usize], self.in_start[n as usize + 1]);
        self.in_edges[a as usize..b as usize].iter().map(move |&e| (self.caller(e), &self.out[e as usize]))
    }

    /// The node an out-edge index leaves.
    pub fn caller(&self, e: u32) -> u32 {
        (self.out_start.partition_point(|&s| s <= e) - 1) as u32
    }

    pub fn scc_count(&self) -> usize {
        self.scc_level.len()
    }

    pub fn scc(&self, s: u32) -> &[u32] {
        let (a, b) = (self.scc_start[s as usize], self.scc_start[s as usize + 1]);
        &self.scc_nodes[a as usize..b as usize]
    }

    /// Whether SCC `s` is a cycle: more than one node, or one node with an edge to itself.
    pub fn is_cycle(&self, s: u32) -> bool {
        match self.scc(s) {
            [n] => self.succ(*n).iter().any(|e| e.to == *n),
            ns => ns.len() > 1,
        }
    }

    /// Whether SCC `s` is a drop cycle: a cycle with a glue node whose every member is a glue
    /// node, a `Drop.drop` impl, or a fn of a built-in package (a user fn that is not a `Drop`
    /// impl, a closure, or a table makes it an ordinary cycle). The recursion of dropping a
    /// recursive value, as glue(N) -> `Vec<N>.drop` -> `Vec<N>.clear` -> glue(N).
    pub fn is_drop_cycle(&self, s: u32) -> bool {
        self.scc_drop.get(s as usize).copied().unwrap_or(false)
    }

    pub fn level_count(&self) -> usize {
        self.level_start.len().saturating_sub(1)
    }

    /// The SCC ids of level `l`.
    pub fn level(&self, l: usize) -> std::ops::Range<u32> {
        self.level_start[l]..self.level_start[l + 1]
    }

    /// The enclosing fn instance node of `n` (itself for a fn; `None` for a table or glue).
    pub fn fn_of(&self, n: u32) -> Option<u32> {
        match self.nodes.get(n as usize)?.kind {
            NodeKind::Fn { .. } => Some(n),
            NodeKind::Closure { parent, .. } => Some(parent),
            NodeKind::Table { .. } | NodeKind::Glue { .. } => None,
        }
    }

    /// The node of fn instance `k`, if it is one.
    pub fn find(&self, k: &InstKey) -> Option<u32> {
        self.nodes
            .iter()
            .position(|n| matches!(&n.kind, NodeKind::Fn { inst, .. } if inst == k))
            .map(|i| i as u32)
    }

    /// An interner reading the graph's type ids: `global` is typeck's table.
    pub fn types<'g>(&self, global: &'g TypeTable) -> Overlay<'g> {
        let mut o = Overlay::new(global);
        for k in &self.local_types {
            o.intern(k.clone());
        }
        o
    }

    /// The names of nodes `ns`, as [`CallGraph::dump`] prints them.
    pub fn node_names(&self, prog: &Program, tc: &Typeck, ns: &[u32]) -> Vec<String> {
        let cx = self.types(&tc.types);
        let names = crate::sigs::NameCx { prog, generics: &[] };
        ns.iter().map(|&n| node_name(self, prog, &tc.sigs, &cx, &names, n, true)).collect()
    }

    /// The names of nodes `ns` in source terms: as [`CallGraph::node_names`], without the fn
    /// values a node is specialized on (`main.apply`, not `main.apply{0=main.twice}`).
    pub fn source_names(&self, prog: &Program, tc: &Typeck, ns: &[u32]) -> Vec<String> {
        let cx = self.types(&tc.types);
        let names = crate::sigs::NameCx { prog, generics: &[] };
        ns.iter().map(|&n| node_name(self, prog, &tc.sigs, &cx, &names, n, false)).collect()
    }

    /// A text form for tests: one line per node, edge, table, root, and SCC (cycles only), in id
    /// order. Fn instances print as checked-IR paths with their type args, closures as
    /// `<fn>.<k>`, tables as `table <fn type>`, sites as `line:col`.
    pub fn dump(&self, prog: &Program, tc: &Typeck) -> String {
        let cx = self.types(&tc.types);
        let names = crate::sigs::NameCx { prog, generics: &[] };
        let name = |n: u32| -> String { node_name(self, prog, &tc.sigs, &cx, &names, n, true) };
        let at = |s: Span| -> String {
            prog.source_map.locate(s).map_or_else(|_| "?".to_string(), |l| format!("{}:{}", l.line, l.col))
        };
        let mut s = String::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let flags = match &n.kind {
                NodeKind::Fn { rec, body, .. } => {
                    format!("{}{}", if *rec { " rec" } else { "" }, if *body { "" } else { " nobody" })
                }
                NodeKind::Closure { spawn: true, .. } => " spawn".to_string(),
                _ => String::new(),
            };
            let _ = writeln!(s, "node {i} {}{flags}", name(i as u32));
        }
        for i in 0..self.nodes.len() as u32 {
            for e in self.succ(i) {
                let kind = match e.kind {
                    EdgeKind::Call => "call".to_string(),
                    EdgeKind::Dispatch { decl } => format!("dispatch {}", prog.decl_path(decl)),
                    EdgeKind::Indirect if e.deny != 0 => {
                        format!("indirect without({})", crate::effects::Effects(e.deny).names().join(", "))
                    }
                    EdgeKind::Indirect => "indirect".to_string(),
                    EdgeKind::Member => "member".to_string(),
                    EdgeKind::Spawn => "spawn".to_string(),
                    EdgeKind::Drop => "drop".to_string(),
                };
                let _ = writeln!(s, "edge {} -> {} {kind} @{}", name(i), name(e.to), at(e.site));
            }
        }
        for r in &self.roots {
            let kind = match r.kind {
                RootKind::Main => "main".to_string(),
                RootKind::Spawn => "spawn".to_string(),
                RootKind::ThreadArg { callee, arg } => format!("{} arg {arg}", prog.decl_path(callee)),
            };
            let _ = writeln!(s, "root {} {kind} @{}", name(r.node), at(r.site));
        }
        for c in 0..self.scc_count() as u32 {
            if self.is_cycle(c) {
                let ns: Vec<String> = self.scc(c).iter().map(|&n| name(n)).collect();
                let drop = if self.is_drop_cycle(c) { " drop" } else { "" };
                let _ = writeln!(s, "cycle level {} {{{}}}{drop}", self.scc_level[c as usize], ns.join(", "));
            }
        }
        let _ = writeln!(s, "levels {}", self.level_count());
        if self.truncated {
            s.push_str("truncated\n");
        }
        s
    }
}

fn node_name(
    g: &CallGraph,
    prog: &Program,
    sigs: &Sigs,
    cx: &Overlay<'_>,
    names: &crate::sigs::NameCx<'_>,
    n: u32,
    spec_too: bool,
) -> String {
    match g.nodes.get(n as usize).map(|x| &x.kind) {
        Some(NodeKind::Fn { inst, spec, .. }) => {
            let mut s = crate::lower::fn_path(prog, sigs, inst.decl);
            if !inst.args.is_empty() {
                let args: Vec<String> = inst
                    .args
                    .iter()
                    .map(|a| match a {
                        GArg::Type(t) => crate::types::display(cx, names, *t),
                        GArg::Const(c) => c.value().map_or_else(|| "_".to_string(), |v| v.to_string()),
                    })
                    .collect();
                let _ = write!(s, "<{}>", args.join(", "));
            }
            let known: Vec<String> = spec
                .iter()
                .enumerate()
                .filter(|(_, ns)| spec_too && !ns.is_empty())
                .map(|(p, ns)| {
                    let vs: Vec<String> =
                        ns.iter().map(|&n| node_name(g, prog, sigs, cx, names, n, true)).collect();
                    format!("{p}={}", vs.join("|"))
                })
                .collect();
            if !known.is_empty() {
                let _ = write!(s, "{{{}}}", known.join(", "));
            }
            s
        }
        Some(NodeKind::Closure { parent, index, .. }) => {
            format!("{}.{index}", node_name(g, prog, sigs, cx, names, *parent, spec_too))
        }
        Some(NodeKind::Table { table }) => {
            let t = g.tables.get(*table as usize).map_or(TypeId::NONE, |t| t.ty);
            format!("table {}", crate::types::display(cx, names, t))
        }
        Some(NodeKind::Glue { ty }) => format!("drop_glue {}", crate::types::display(cx, names, *ty)),
        None => "?".to_string(),
    }
}

// ---------- per-body events ----------

/// A closure body of a fn body: its kind, token range, owner (0: the fn, `k + 1`: closure `k`),
/// fn type (typeck's, unsubstituted; `NONE` for a `spawn` body), and param binding nodes.
struct Closure {
    kind: ClosureKind,
    lo: u32,
    hi: u32,
    owner: u32,
    ty: TypeId,
    span: Span,
    params: Vec<u32>,
}

/// Where a fn value used in a body comes from.
#[derive(Clone, Copy)]
enum Src<'b> {
    /// A local of the body (a param or a binding, of the fn or of one of its closures).
    Local(u32),
    /// A struct field: (struct decl, field index).
    Field(DeclId, u32),
    /// A fn item made into a value.
    FnItem(DeclId, &'b [GArg]),
    /// Closure body `k`.
    Closure(u32),
    /// Not tracked: any value of its fn type.
    Top,
}

enum What<'b> {
    /// A direct call recorded by typeck (call or implied).
    Call(&'b Target),
    /// A call through a fn value of type `ty` coming from `callee`.
    ValueCall { ty: TypeId, callee: Src<'b> },
    /// A built-in method calling a fn value it is passed (a slice predicate).
    ValueUse { ty: TypeId, src: Src<'b> },
    /// Argument `idx` (receiver = 0 for methods) of the call at the same token is a fn value.
    Arg { idx: u32, src: Src<'b> },
    /// Local `b` is set to a fn value.
    Init { b: u32, src: Src<'b> },
    /// A struct field is set to a fn value.
    Store { decl: DeclId, index: u32, src: Src<'b> },
    /// A fn item made into a fn value; `root`: it is a thread fn's argument.
    FnItem { decl: DeclId, args: &'b [GArg], root: Option<RootKind> },
    /// Closure `k` made into a fn value (a `spawn` body, or `started` by `runtime.thread.start`:
    /// also spawned).
    Closure { k: u32, root: Option<RootKind>, started: bool },
    /// `String.as_str` of a coercion.
    AsStr,
    /// `String.new` and `String.push_str` of an f-string.
    Fstring,
}

/// Where an event happens: a token, made a [`Span`] only for the edges kept (re-lexing every call
/// token would cost more than the walk).
#[derive(Clone, Copy)]
enum Site {
    Tok { file: FileId, tok: u32 },
    At(Span),
}

impl Site {
    fn span(self, prog: &Program) -> Span {
        match self {
            Site::At(s) => s,
            Site::Tok { file, tok } => prog.file(file).map_or(Span::new(file, 0, 0), |f| {
                crate::scan::tok_span(file, &f.bytes, &f.parsed.tokens, tok as usize)
            }),
        }
    }
}

struct Ev<'b> {
    owner: u32,
    tok: u32,
    site: Site,
    what: What<'b>,
}

impl Ev<'_> {
    /// Events at one token: the call first, then its arguments.
    fn order(&self) -> (u32, u8) {
        (self.tok, matches!(self.what, What::Arg { .. }) as u8)
    }
}

struct BodyEvents<'b> {
    closures: Vec<Closure>,
    /// The fn's param binding nodes.
    params: Vec<u32>,
    evs: Vec<Ev<'b>>,
    /// (owner, value type, site token): each type an owner may drop once, at its first site.
    drops: Vec<(u32, TypeId, u32)>,
    file: FileId,
}

/// The owner of token `t`: the innermost closure containing it (`k + 1`), else 0. `cl` is sorted
/// by `lo` and properly nested, so it is `cl[i]` (the last starting at or before `t`) or one of
/// its owners.
fn owner_of(cl: &[Closure], t: u32) -> u32 {
    let mut j = cl.partition_point(|c| c.lo <= t) as u32;
    while j > 0 {
        let c = &cl[j as usize - 1];
        if t <= c.hi {
            return j;
        }
        j = c.owner;
    }
    0
}

/// The callee's last token of a call whose `(` is `paren`: before the turbofish, if any.
fn callee_tok(icx: &ItemCx<'_>, paren: u32) -> u32 {
    let t = paren.saturating_sub(1);
    if icx.is(t, Term::Gt) || icx.is(t, Term::GtGt) {
        let stop = t.saturating_sub(512);
        let mut k = t;
        while k > stop && !icx.is(k, Term::ColonColon) {
            k -= 1;
        }
        if icx.is(k, Term::ColonColon) {
            return k.saturating_sub(1);
        }
    }
    t
}

/// The param binding nodes of the fn or closure node `n`.
fn params_of(icx: &ItemCx<'_>, n: u32) -> Vec<u32> {
    icx.child(n, Syn::Params).map_or_else(Vec::new, |p| icx.kids_vec(p).iter().copied().collect())
}

/// The fn items and closure literals passed to the thread fns `fns[..3]` in `body`: (core token,
/// root). A closure literal's core token is its `fn` token; a fn item's, the token typeck keys its
/// value target by. Also the `fn` tokens of the closure literals passed to `fns[3]`
/// (`runtime.thread.start`), which runs them on a new thread.
fn thread_args(
    icx: &ItemCx<'_>,
    body: &TypedBody,
    fns: &[Option<DeclId>; 4],
) -> (Vec<(u32, RootKind)>, Vec<u32>) {
    let mut out = Vec::new();
    let mut started = Vec::new();
    let calls_one =
        body.targets.iter().any(|(_, t)| matches!(t, Target::Fn { decl, .. } if fns.contains(&Some(*decl))));
    if !calls_one {
        return (out, started);
    }
    let lo = icx.ast.descendants(body.node).start;
    for n in lo..=body.node {
        if icx.kind(n) != Some(Syn::Postfix) {
            continue;
        }
        for &k in icx.kids_vec(n).iter().skip(1) {
            let lp = icx.a(k).saturating_sub(1);
            if !icx.is(lp, Term::LParen) {
                continue;
            }
            let Some(Target::Fn { decl, .. }) = body.target(lp) else { continue };
            let Some(which) = fns.iter().position(|f| *f == Some(*decl)) else { continue };
            let args: &[u32] = match which {
                0 => &[0, 1],
                1 => &[2],
                2 => &[1],
                _ => {
                    if let Some(&arg) = icx.kids_vec(k).first()
                        && crate::closures::closure_kind(icx, arg) == Some(ClosureKind::Literal)
                    {
                        started.push(icx.a(arg));
                    }
                    continue;
                }
            };
            let kids = icx.kids_vec(k);
            for &i in args {
                let Some(&arg) = kids.get(i as usize) else { continue };
                let chain = crate::threads::wrapper_chain(icx, arg);
                let core = chain.last().copied().unwrap_or(arg);
                let root = RootKind::ThreadArg { callee: *decl, arg: i };
                if crate::closures::closure_kind(icx, core) == Some(ClosureKind::Literal) {
                    out.push((icx.a(core), root));
                } else if let Some(&c) = chain.iter().find(|&&c| body.coercion(c) == Some(&Coercion::FnItem))
                {
                    out.push((icx.end_tok(c), root));
                }
            }
        }
    }
    (out, started)
}

/// What the fn-valued expression `e` of `body` is (module docs, "Fn-value flow").
fn classify<'b>(icx: &ItemCx<'_>, body: &'b TypedBody, closures: &[Closure], e: u32) -> Src<'b> {
    let chain = crate::threads::wrapper_chain(icx, e);
    let end = icx.end_tok(e);
    for &c in &chain {
        // `FnItem` also marks a fn value whose features drop: only a fn item has a target here
        if body.coercion(c) == Some(&Coercion::FnItem)
            && let Some(Target::Fn { decl, args }) = body.target(icx.end_tok(c))
        {
            return Src::FnItem(*decl, args);
        }
        if icx.kind(c) == Some(Syn::Postfix)
            && let Some(Target::Field { decl, index, .. }) = body.target(icx.end_tok(c))
        {
            return Src::Field(*decl, *index);
        }
    }
    let core = chain.last().copied().unwrap_or(e);
    if crate::closures::closure_kind(icx, core) == Some(ClosureKind::Literal) {
        let lo = icx.a(core);
        return closures.iter().position(|c| c.lo == lo).map_or(Src::Top, |k| Src::Closure(k as u32));
    }
    if icx.kind(core) != Some(Syn::Primary) {
        return Src::Top;
    }
    let a = icx.a(core);
    if (icx.is(a, Term::Ident) || icx.is(a, Term::KwSelf)) && end == a {
        if let Some(u) = icx.use_at(core, a).or_else(|| icx.uses(core).first())
            && let crate::resolve::Res::Local(b) | crate::resolve::Res::SelfValue(b) = u.res
        {
            return Src::Local(b.node);
        }
        return Src::Top;
    }
    if icx.is(a, Term::LParen) && icx.close(a) == end && !icx.is(end.saturating_sub(1), Term::Comma) {
        let kids = icx.kids_vec(core);
        if kids.len() == 1 {
            return classify(icx, body, closures, kids[0]);
        }
    }
    Src::Top
}

/// The fn-value flow events of `body` (module docs). `fnish(t)`: `t` is a fn type or a generic
/// param (which an instance may make a fn type).
fn flow_events<'b>(
    icx: &ItemCx<'_>,
    body: &'b TypedBody,
    sigs: &Sigs,
    closures: &[Closure],
    params: &[u32],
    fnish: &dyn Fn(TypeId) -> bool,
    push: &mut dyn FnMut(u32, u32, What<'b>),
) {
    let mut simple: Vec<u32> = Vec::new();
    let lo = icx.ast.descendants(body.node).start;
    for n in lo..=body.node {
        match icx.kind(n) {
            Some(Syn::Stmt) if icx.is(icx.a(n), Term::KwLet) => {
                let kids = icx.kids_vec(n);
                let (Some(&pat), Some(&init)) = (kids.first(), kids.last()) else { continue };
                if let Some(t) = body.local(crate::ids::BindingRef { node: pat })
                    && fnish(t)
                {
                    simple.push(pat);
                    push(
                        icx.a(pat),
                        icx.a(pat),
                        What::Init { b: pat, src: classify(icx, body, closures, init) },
                    );
                }
            }
            Some(Syn::ExprStmt) => {
                let kids = icx.kids_vec(n);
                if kids.len() < 3 || !icx.is(icx.a(kids[1]), Term::Eq) || !fnish(body.ty(kids[2])) {
                    continue;
                }
                let src = classify(icx, body, closures, kids[2]);
                match classify(icx, body, closures, kids[0]) {
                    Src::Local(b) => push(icx.a(n), icx.a(n), What::Init { b, src }),
                    Src::Field(decl, index) => push(icx.a(n), icx.a(n), What::Store { decl, index, src }),
                    _ => {}
                }
            }
            Some(Syn::StructLit) => {
                let tok = icx.a(n);
                let Some(Target::Struct { decl }) = body.target(tok) else { continue };
                let Some(fields) = sigs.adt(*decl).and_then(|a| a.variants.first()).map(|v| &v.fields) else {
                    continue;
                };
                for &e in icx.kids_vec(n).iter() {
                    if !fnish(body.ty(e)) {
                        continue;
                    }
                    let name = icx.sym(icx.a(e).saturating_sub(2));
                    if let Some(index) = fields.iter().position(|f| f.name == name) {
                        let src = classify(icx, body, closures, e);
                        push(tok, tok, What::Store { decl: *decl, index: index as u32, src });
                    }
                }
            }
            Some(Syn::Postfix) => {
                let kids = icx.kids_vec(n);
                let Some(&prim) = kids.first() else { continue };
                let prim_end = icx.end_tok(prim);
                // the chain's call parens (a call without args has no `Args` node)
                let end = icx.end_tok(n);
                let mut t = prim_end + 1;
                while t <= end {
                    let tk = icx.tk(t);
                    if tk == Term::LBracket as u8 || tk == Term::DotLBrace as u8 {
                        t = icx.close(t) + 1;
                        continue;
                    }
                    if tk != Term::LParen as u8 {
                        t += 1;
                        continue;
                    }
                    let lp = t;
                    t = icx.close(lp) + 1;
                    let seg = callee_tok(icx, lp);
                    let (tok, offset, builtin) = match (body.target(lp), body.target(seg)) {
                        (Some(Target::Fn { .. } | Target::TraitMethod { .. }), _) => (lp, 0, false),
                        (Some(Target::FnValue { ty }), _) => {
                            let callee = match body.target(seg) {
                                Some(Target::Field { decl, index, .. }) => Src::Field(*decl, *index),
                                _ if seg == prim_end => classify(icx, body, closures, prim),
                                _ => Src::Top,
                            };
                            push(lp, seg, What::ValueCall { ty: *ty, callee });
                            (lp, 0, false)
                        }
                        (_, Some(Target::Method { .. })) => (seg, 1, false),
                        (_, Some(Target::Builtin { .. })) => (seg, 1, true),
                        _ => continue,
                    };
                    let Some(&k) = kids.iter().skip(1).find(|&&k| icx.a(k) == lp + 1) else {
                        continue;
                    };
                    for (i, &e) in icx.kids_vec(k).iter().enumerate() {
                        let ty = body.ty(e);
                        if !fnish(ty) {
                            continue;
                        }
                        let src = classify(icx, body, closures, e);
                        if builtin {
                            push(tok, tok, What::ValueUse { ty, src });
                        } else {
                            push(tok, tok, What::Arg { idx: i as u32 + offset, src });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // any other fn-valued local (a pattern binding but `let x`) is not tracked
    for &(b, t) in &body.locals {
        let is_param = params.contains(&b) || closures.iter().any(|c| c.params.contains(&b));
        if fnish(t) && !is_param && !simple.contains(&b) {
            push(icx.a(b), icx.a(b), What::Init { b, src: Src::Top });
        }
    }
}

fn extract<'b>(
    icx: &ItemCx<'_>,
    body: &'b TypedBody,
    sigs: &Sigs,
    fns: &[Option<DeclId>; 4],
    fnish: &dyn Fn(TypeId) -> bool,
    is_fn: &dyn Fn(TypeId) -> bool,
) -> BodyEvents<'b> {
    // one pass over the block's tokens: f-strings, and whether any closure body can start
    let mut fstrings: Vec<u32> = Vec::new();
    let mut any_closure = false;
    if let Some(b) = icx.child(body.node, Syn::Block) {
        let open = icx.a(b);
        if let Some(toks) = icx.toks.get(open as usize..=icx.close(open) as usize) {
            for (i, t) in toks.iter().enumerate() {
                if t.kind == Term::FmtLit as u8 {
                    fstrings.push(open + i as u32);
                }
                any_closure |= t.kind == Term::KwFn as u8 || t.kind == Term::KwSpawn as u8;
            }
        }
    }
    let mut closures: Vec<Closure> = Vec::new();
    let found = if any_closure { crate::closures::closure_nodes(icx, body.node) } else { Vec::new() };
    for (node, kind) in found {
        let lo = icx.a(node);
        let hi = icx.child(node, Syn::Block).map_or_else(|| icx.end_tok(node), |b| icx.close(icx.a(b)));
        let owner = owner_of(&closures, lo);
        let (ty, params) = if kind == ClosureKind::Literal {
            (body.ty(node), params_of(icx, node))
        } else {
            (TypeId::NONE, Vec::new())
        };
        closures.push(Closure { kind, lo, hi, owner, ty, span: icx.tok_span(lo), params });
    }
    let params = params_of(icx, body.node);
    let (roots, started) = thread_args(icx, body, fns);
    let root_at = |t: u32| roots.iter().find(|r| r.0 == t).map(|r| r.1);
    let mut evs: Vec<Ev> = Vec::new();
    let file = icx.fid();
    let mut push = |tok: u32, site_tok: u32, what: What<'b>| {
        let site = Site::Tok { file, tok: site_tok };
        evs.push(Ev { owner: owner_of(&closures, tok), tok, site, what });
    };
    for (tok, t) in &body.targets {
        let tok = *tok;
        match t {
            Target::Fn { decl, args } => {
                if icx.is(tok, Term::LParen) {
                    push(tok, callee_tok(icx, tok), What::Call(t));
                } else {
                    let what = What::FnItem { decl: *decl, args, root: root_at(tok) };
                    push(tok, tok, what);
                }
            }
            Target::Method { .. } | Target::Op { .. } => push(tok, tok, What::Call(t)),
            Target::TraitMethod { .. } => push(tok, callee_tok(icx, tok), What::Call(t)),
            _ => {}
        }
    }
    for (tok, t) in &body.implicit {
        push(*tok, *tok, What::Call(t));
    }
    for (n, c) in &body.coercions {
        if matches!(c, Coercion::StringToStr | Coercion::FmtToStr) {
            let a = icx.a(*n);
            push(a, a, What::AsStr);
        }
    }
    for &tok in &fstrings {
        push(tok, tok, What::Fstring);
    }
    // a body without a fn-typed value makes no flow: a generic-typed value there reaches a fn
    // type only through a param or field declared generic, which is untracked (`Walk::expand`,
    // `Walk::val`)
    let flows = body.node_types.iter().any(|&t| is_fn(t)) || body.locals.iter().any(|&(_, t)| is_fn(t));
    if flows {
        flow_events(icx, body, sigs, &closures, &params, fnish, &mut push);
    }
    drop(push);
    for (k, c) in closures.iter().enumerate() {
        let root = if c.kind == ClosureKind::Spawn { Some(RootKind::Spawn) } else { root_at(c.lo) };
        let started = started.contains(&c.lo);
        evs.push(Ev {
            owner: c.owner,
            tok: c.lo,
            site: Site::At(c.span),
            what: What::Closure { k: k as u32, root, started },
        });
    }
    evs.sort_by_key(|e| e.order());
    // a value type once per owner, at its first site
    let mut drops: Vec<(u32, TypeId, u32)> = body
        .drops
        .iter()
        .map(|&(tok, ty)| {
            let site = if icx.is(tok, Term::LParen) { callee_tok(icx, tok) } else { tok };
            (owner_of(&closures, tok), ty, site)
        })
        .collect();
    drops.sort_unstable();
    drops.dedup_by_key(|d| (d.0, d.1));
    BodyEvents { closures, params, evs, drops, file }
}

// ---------- the walk ----------

struct NodeB {
    kind: NodeKind,
    span: Span,
    /// The node whose body first requested it (`u32::MAX` for a starting fn).
    from: u32,
    /// Index into the bodies (and events) when it has events.
    body: Option<usize>,
}

/// A fn-value flow slot: a local of an instance (keyed by the instance node and the binding
/// node; a closure's locals and params belong to its enclosing instance), or a struct field.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SlotKey {
    Local(u32, u32),
    Field(DeclId, u32),
}

/// A fn value at walk time: a slot's content, known nodes, or untracked.
#[derive(Clone)]
enum Val {
    Slot(u32),
    Nodes(Vec<u32>),
    Top,
}

/// A call through a fn value.
struct ValueSite {
    from: u32,
    table: Option<u32>,
    callee: Val,
    /// (param index, argument)
    args: Vec<(u32, Val)>,
    site: Site,
    /// The call token ([`CallGraph::calls`]).
    tok: u32,
    /// The deny clause of the called value's fn type: every value that reaches the call was
    /// coerced to that type, so none has those effects.
    deny: u8,
}

/// The fn-value flow constraints the walk collects.
#[derive(Default)]
struct Flow {
    slots: FxHashMap<SlotKey, u32>,
    /// (from slot, to slot): the second holds every value the first does
    copies: Vec<(u32, u32)>,
    /// (slot, node)
    consts: Vec<(u32, u32)>,
    tops: Vec<u32>,
    sites: Vec<ValueSite>,
}

impl Flow {
    fn slot(&mut self, k: SlotKey) -> u32 {
        let n = self.slots.len() as u32;
        *self.slots.entry(k).or_insert(n)
    }

    fn add(&mut self, v: &Val, to: u32) {
        match v {
            Val::Slot(s) => self.copies.push((*s, to)),
            Val::Nodes(ns) => self.consts.extend(ns.iter().map(|&n| (to, n))),
            Val::Top => self.tops.push(to),
        }
    }
}

struct Walk<'a> {
    prog: &'a Program,
    sigs: &'a Sigs,
    bodies: &'a [TypedBody],
    evs: &'a [Option<BodyEvents<'a>>],
    brackets: &'a [Brackets],
    cx: Overlay<'a>,
    nodes: Vec<NodeB>,
    keys: FxHashMap<(InstKey, Vec<Vec<u32>>), u32>,
    /// Specializations made per instance ([`SPEC_LIMIT`]).
    specs: FxHashMap<InstKey, u32>,
    /// The fn values the call being resolved passes per param (empty: none known).
    spec_hint: Vec<Vec<u32>>,
    /// The known fn values of the params of the node being expanded: (binding, nodes).
    consts: Vec<(u32, Vec<u32>)>,
    glue: FxHashMap<TypeId, u32>,
    drop_memo: FxHashMap<TypeId, bool>,
    /// [`Walk::needs_drop`] of typeck's types by id: 0 unknown, 1 no, 2 yes.
    drop_known: Vec<u8>,
    /// (from, to or `TABLE_BIT | table`, kind, site, deny)
    edges: Vec<(u32, u32, EdgeKind, Site, u8)>,
    /// (from, call token, to or `TABLE_BIT | table`)
    calls: Vec<(u32, u32, u32)>,
    tables: Vec<TypeId>,
    table_of: FxHashMap<TypeId, u32>,
    /// (table, node, site)
    members: Vec<(u32, u32, Site)>,
    roots: Vec<(u32, RootKind, Site)>,
    arms: FxHashMap<DeclId, Vec<DeclId>>,
    /// `String.as_str`, `String.new`, `String.push_str`.
    string: [Option<DeclId>; 3],
    /// `runtime.thread.scope_spawn`: lowering passes each `spawn` body to it (param 1).
    scope_spawn: Option<DeclId>,
    flow: Flow,
    truncated: bool,
}

impl<'a> Walk<'a> {
    fn body_index(&self, d: DeclId) -> Option<usize> {
        let i = self.bodies.binary_search_by_key(&d, |b| b.decl).ok()?;
        self.evs[i].is_some().then_some(i)
    }

    /// Instances of `decl` on the discovery chain ending at node `from`.
    fn chain_count(&self, mut from: u32, decl: DeclId) -> u32 {
        let mut n = 0;
        while let Some(b) = self.nodes.get(from as usize) {
            if matches!(&b.kind, NodeKind::Fn { inst, .. } if inst.decl == decl) {
                n += 1;
            }
            from = b.from;
        }
        n
    }

    fn node_for(&mut self, key: InstKey, from: u32) -> Option<u32> {
        let mut spec = self.spec_hint.clone();
        if !spec.is_empty() {
            let known = self.keys.contains_key(&(key.clone(), spec.clone()));
            let made = self.specs.get(&key).copied().unwrap_or(0);
            if !known && made >= SPEC_LIMIT {
                spec = Vec::new();
            }
        }
        let k = (key, spec);
        if let Some(&n) = self.keys.get(&k) {
            return Some(n);
        }
        let (key, spec) = k;
        let d = *self.prog.decl(key.decl)?;
        if !matches!(d.kind, DeclKind::Fn | DeclKind::Method) {
            return None;
        }
        if self.nodes.len() >= NODE_BUDGET
            || (!key.args.is_empty() && self.chain_count(from, key.decl) >= CHAIN_LIMIT)
        {
            self.truncated = true;
            return None;
        }
        let body = self.body_index(key.decl);
        let id = self.nodes.len() as u32;
        if !spec.is_empty() {
            *self.specs.entry(key.clone()).or_insert(0) += 1;
        }
        let kind = NodeKind::Fn {
            inst: key.clone(),
            rec: d.flags.has(DeclFlags::REC),
            body: body.is_some(),
            spec: spec.clone(),
        };
        self.nodes.push(NodeB { kind, span: d.name_span, from, body });
        self.keys.insert((key, spec), id);
        Some(id)
    }

    /// Whether values of the concrete type `t` need dropping.
    fn needs_drop(&mut self, t: TypeId) -> bool {
        let global = !t.is_local() && !t.is_none();
        if global
            && let Some(&k) = self.drop_known.get(t.idx())
            && k != 0
        {
            return k == 2;
        }
        let w = World { prog: self.prog, sigs: self.sigs };
        let r = crate::mono::needs_drop(&w, &mut self.cx, t, &mut self.drop_memo);
        if global && let Some(k) = self.drop_known.get_mut(t.idx()) {
            *k = 1 + r as u8;
        }
        r
    }

    /// The glue node of `t` (first seen at `site`).
    fn glue_node(&mut self, t: TypeId, from: u32, site: Site) -> Option<u32> {
        if let Some(&g) = self.glue.get(&t) {
            return Some(g);
        }
        if self.nodes.len() >= NODE_BUDGET {
            self.truncated = true;
            return None;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(NodeB {
            kind: NodeKind::Glue { ty: t },
            span: site.span(self.prog),
            from,
            body: None,
        });
        self.glue.insert(t, id);
        Some(id)
    }

    /// Glue node `i` of `t`: its `Drop` impl, then the glue of every part that needs dropping.
    fn expand_glue(&mut self, i: u32, t: TypeId) {
        let site = Site::At(self.nodes[i as usize].span);
        let w = World { prog: self.prog, sigs: self.sigs };
        if let Some(k) = crate::mono::drop_method(&w, &mut self.cx, t)
            && let Some(n) = self.node_for(k, i)
        {
            self.edges.push((i, n, EdgeKind::Drop, site, 0));
        }
        let parts: Vec<TypeId> = match self.cx.kind(t).clone() {
            TypeKind::Adt { .. } => crate::mono::adt_fields(self.sigs, &mut self.cx, t)
                .map_or_else(Vec::new, |vs| vs.into_iter().flatten().collect()),
            TypeKind::Tuple(ts) => ts.to_vec(),
            TypeKind::Array { elem, .. } => vec![elem],
            _ => Vec::new(),
        };
        for p in parts {
            if self.needs_drop(p)
                && let Some(g) = self.glue_node(p, i, site)
            {
                self.edges.push((i, g, EdgeKind::Drop, site, 0));
            }
        }
    }

    /// The nodes a use of fn `decl` with `args` stands for: the instance, or a dispatch fn's arm
    /// targets.
    fn fn_targets(&mut self, decl: DeclId, args: Vec<GArg>, from: u32) -> Vec<u32> {
        let is_dispatch = self.prog.decl(decl).is_some_and(|d| d.flags.has(DeclFlags::DISPATCH));
        if !is_dispatch {
            return self.node_for(InstKey { decl, args }, from).into_iter().collect();
        }
        if !self.arms.contains_key(&decl) {
            let arms = self
                .prog
                .decl(decl)
                .and_then(|d| self.brackets.get(d.item.idx()).and_then(|b| ItemCx::new(self.prog, d.item, b)))
                .map_or_else(Vec::new, |icx| crate::dispatch::arm_targets(self.prog, self.sigs, &icx, decl));
            self.arms.insert(decl, arms);
        }
        let arms = self.arms.get(&decl).cloned().unwrap_or_default();
        arms.into_iter().filter_map(|a| self.node_for(InstKey { decl: a, args: Vec::new() }, from)).collect()
    }

    fn subst_args(&mut self, ga: &[GArg], args: &[GArg]) -> Vec<GArg> {
        if args.is_empty() {
            return ga.to_vec();
        }
        ga.iter().map(|&a| crate::mono::subst_garg(&mut self.cx, a, args)).collect()
    }

    fn subst(&mut self, t: TypeId, args: &[GArg]) -> TypeId {
        if args.is_empty() { t } else { self.cx.subst(t, args, None) }
    }

    /// The deny clause of fn type `t` (0 for any other type).
    fn deny_of(&self, t: TypeId) -> u8 {
        match self.cx.kind(t) {
            TypeKind::Fn { deny, .. } => *deny,
            _ => 0,
        }
    }

    /// The table of fn type `t` (`None` if `t` is not a fn type).
    fn table(&mut self, t: TypeId) -> Option<u32> {
        let TypeKind::Fn { params, ret, .. } = self.cx.kind(t).clone() else { return None };
        let key = self.cx.intern(TypeKind::Fn { params, ret, features: 0, deny: 0 });
        if let Some(&i) = self.table_of.get(&key) {
            return Some(i);
        }
        let i = self.tables.len() as u32;
        self.tables.push(key);
        self.table_of.insert(key, i);
        Some(i)
    }

    /// The table of fn item `decl`'s type with `args`.
    fn fn_item_table(&mut self, decl: DeclId, args: &[GArg]) -> Option<u32> {
        let sig = self.sigs.fn_sig(decl)?;
        let (ps, r) = (sig.params.clone(), sig.ret);
        let params: Vec<TypeId> = ps.iter().map(|&p| self.subst(p, args)).collect();
        let ret = self.subst(r, args);
        let t = self.cx.intern(TypeKind::Fn { params: params.into_boxed_slice(), ret, features: 0, deny: 0 });
        self.table(t)
    }

    /// Edges to the nodes `decl` with `args` stands for; returns them.
    fn call_fn(&mut self, from: u32, inst: u32, decl: DeclId, args: Vec<GArg>, site: Site) -> Vec<u32> {
        let kind = match self.prog.decl(decl) {
            Some(d) if d.flags.has(DeclFlags::DISPATCH) => EdgeKind::Dispatch { decl },
            _ => EdgeKind::Call,
        };
        let to = self.fn_targets(decl, args, inst);
        for &t in &to {
            self.edges.push((from, t, kind, site, 0));
        }
        to
    }

    #[allow(clippy::too_many_arguments)]
    fn call_trait(
        &mut self,
        from: u32,
        inst: u32,
        method: DeclId,
        self_ty: TypeId,
        ta: &[GArg],
        own: &[GArg],
        site: Site,
    ) -> Vec<u32> {
        let w = World { prog: self.prog, sigs: self.sigs };
        match crate::mono::resolve_trait_method(&w, &mut self.cx, method, self_ty, ta, own) {
            Some(Resolved::Inst(k)) => self.call_fn(from, inst, k.decl, k.args, site),
            _ => Vec::new(),
        }
    }

    /// A direct call target of instance node `inst` (args `args`) made in node `from`: the callee
    /// nodes.
    fn call(&mut self, from: u32, inst: u32, t: &Target, args: &[GArg], site: Site) -> Vec<u32> {
        match t {
            Target::Fn { decl, args: ga } => {
                let ga = self.subst_args(ga, args);
                if self.prog.decl(*decl).is_some_and(|d| d.kind == DeclKind::TraitMethod) {
                    // a trait method by path: `Self` is the last arg
                    let Some((&GArg::Type(st), rest)) = ga.split_last() else { return Vec::new() };
                    let np = self.sigs.fn_sig(*decl).map_or(0, |s| s.n_parent as usize).min(rest.len());
                    self.call_trait(from, inst, *decl, st, &rest[..np], &rest[np..], site)
                } else {
                    self.call_fn(from, inst, *decl, ga, site)
                }
            }
            Target::Method { method, args: ga, .. } | Target::Op { method, args: ga } => {
                let ga = self.subst_args(ga, args);
                match method {
                    MethodRef::Impl { method, .. } => self.call_fn(from, inst, *method, ga, site),
                    MethodRef::Bound { method, trait_args, self_ty } => {
                        let st = self.subst(*self_ty, args);
                        let ta = self.subst_args(trait_args, args);
                        let np = self.sigs.fn_sig(*method).map_or(0, |s| s.n_parent as usize);
                        let own = ga.get(np..).unwrap_or(&[]).to_vec();
                        self.call_trait(from, inst, *method, st, &ta, &own, site)
                    }
                    MethodRef::Builtin(_) => Vec::new(),
                }
            }
            Target::TraitMethod { method, self_ty, args: ga } => {
                let st = self.subst(*self_ty, args);
                let ga = self.subst_args(ga, args);
                let np = self.sigs.fn_sig(*method).map_or(0, |s| s.n_parent as usize).min(ga.len());
                self.call_trait(from, inst, *method, st, &ga[..np], &ga[np..], site)
            }
            _ => Vec::new(),
        }
    }

    fn root(&mut self, node: u32, kind: RootKind, site: Site) {
        if !self.roots.iter().any(|r| r.0 == node && r.1 == kind) {
            self.roots.push((node, kind, site));
        }
    }

    /// A fn value of instance node `inst` (closure nodes from `base`) at walk time.
    fn val(&mut self, src: &Src<'_>, inst: u32, base: u32, args: &[GArg]) -> Val {
        match *src {
            Src::Local(b) => match self.consts.iter().find(|c| c.0 == b) {
                Some((_, ns)) => Val::Nodes(ns.clone()),
                None => Val::Slot(self.flow.slot(SlotKey::Local(inst, b))),
            },
            Src::Field(d, i) => {
                // a field declared generic is untracked: generic code stores into it unseen
                let declared =
                    self.sigs.adt(d).and_then(|a| a.variants.first()).and_then(|v| v.fields.get(i as usize));
                match declared {
                    Some(f) if matches!(self.cx.kind(f.ty), TypeKind::Fn { .. }) => {
                        Val::Slot(self.flow.slot(SlotKey::Field(d, i)))
                    }
                    _ => Val::Top,
                }
            }
            Src::FnItem(decl, ga) => {
                let ga = self.subst_args(ga, args);
                Val::Nodes(self.fn_targets(decl, ga, inst))
            }
            Src::Closure(k) => Val::Nodes(vec![base + k]),
            Src::Top => Val::Top,
        }
    }

    fn expand(&mut self, i: u32) {
        let (args, bi, spec) = match &self.nodes[i as usize] {
            NodeB { kind: NodeKind::Fn { inst, spec, .. }, body: Some(b), .. } => {
                (inst.args.clone(), *b, spec.clone())
            }
            NodeB { kind: NodeKind::Glue { ty }, .. } => {
                let t = *ty;
                self.expand_glue(i, t);
                return;
            }
            _ => return,
        };
        let evs = self.evs;
        let Some(be) = evs[bi].as_ref() else { return };
        self.consts = spec
            .iter()
            .enumerate()
            .filter(|(_, ns)| !ns.is_empty())
            .filter_map(|(p, ns)| be.params.get(p).map(|&b| (b, ns.clone())))
            .collect();
        let base = self.nodes.len() as u32;
        for (k, c) in be.closures.iter().enumerate() {
            let kind = NodeKind::Closure { parent: i, index: k as u32, spawn: c.kind == ClosureKind::Spawn };
            self.nodes.push(NodeB { kind, span: c.span, from: i, body: None });
        }
        // a local set once, to a fn item or closure literal, holds that value everywhere
        let mut sets: Vec<(u32, &Src<'_>, usize)> = Vec::new();
        for ev in &be.evs {
            if let What::Init { b, src } = &ev.what {
                match sets.iter_mut().find(|s| s.0 == *b) {
                    Some(s) => s.2 += 1,
                    None => sets.push((*b, src, 1)),
                }
            }
        }
        for (b, src, n) in sets {
            if n == 1
                && matches!(src, Src::FnItem(..) | Src::Closure(_))
                && let Val::Nodes(ns) = self.val(src, i, base, &args)
            {
                self.consts.push((b, ns));
            }
        }
        let node_of = |o: u32| if o == 0 { i } else { base + o - 1 };
        if !args.is_empty() {
            self.generic_params(i, &args, be);
        }
        // the callees of the call at the last call token: direct nodes, or a value-call site
        let mut last: (u32, Vec<u32>, Option<usize>) = (u32::MAX, Vec::new(), None);
        for (ei, ev) in be.evs.iter().enumerate() {
            let from = node_of(ev.owner);
            match &ev.what {
                What::Call(t) => {
                    // the fn values the call passes: the callee is specialized on them
                    let mut hint: Vec<Vec<u32>> = Vec::new();
                    for a in be.evs[ei + 1..].iter().take_while(|a| a.tok == ev.tok) {
                        if let What::Arg { idx, src } = &a.what
                            && let Val::Nodes(ns) = self.val(src, i, base, &args)
                        {
                            let idx = *idx as usize;
                            if hint.len() <= idx {
                                hint.resize(idx + 1, Vec::new());
                            }
                            let mut ns = ns;
                            ns.sort_unstable();
                            ns.dedup();
                            hint[idx] = ns;
                        }
                    }
                    self.spec_hint = hint;
                    let to = self.call(from, i, t, &args, ev.site);
                    self.spec_hint = Vec::new();
                    self.calls.extend(to.iter().map(|&c| (from, ev.tok, c)));
                    last = (ev.tok, to, None);
                }
                What::ValueCall { ty, callee } => {
                    let t = self.subst(*ty, &args);
                    let table = self.table(t);
                    let deny = self.deny_of(t);
                    let callee = self.val(callee, i, base, &args);
                    let (site, tok) = (ev.site, ev.tok);
                    self.flow.sites.push(ValueSite {
                        from,
                        table,
                        callee,
                        args: Vec::new(),
                        site,
                        tok,
                        deny,
                    });
                    last = (ev.tok, Vec::new(), Some(self.flow.sites.len() - 1));
                }
                What::ValueUse { ty, src } => {
                    let t = self.subst(*ty, &args);
                    let table = self.table(t);
                    let deny = self.deny_of(t);
                    let callee = self.val(src, i, base, &args);
                    let (site, tok) = (ev.site, ev.tok);
                    self.flow.sites.push(ValueSite {
                        from,
                        table,
                        callee,
                        args: Vec::new(),
                        site,
                        tok,
                        deny,
                    });
                }
                What::Arg { idx, src } => {
                    if last.0 != ev.tok {
                        continue;
                    }
                    let v = self.val(src, i, base, &args);
                    match last.2 {
                        Some(s) => self.flow.sites[s].args.push((*idx, v)),
                        None => {
                            for c in last.1.clone() {
                                if let Some(p) = self.param_slot(c, *idx) {
                                    self.flow.add(&v, p);
                                }
                            }
                        }
                    }
                }
                What::Init { b, src } => {
                    let v = self.val(src, i, base, &args);
                    let s = self.flow.slot(SlotKey::Local(i, *b));
                    self.flow.add(&v, s);
                }
                What::Store { decl, index, src } => {
                    let v = self.val(src, i, base, &args);
                    let s = self.flow.slot(SlotKey::Field(*decl, *index));
                    self.flow.add(&v, s);
                }
                What::FnItem { decl, args: ga, root } => {
                    let ga = self.subst_args(ga, &args);
                    let tb = self.fn_item_table(*decl, &ga);
                    for n in self.fn_targets(*decl, ga, i) {
                        if let Some(tb) = tb {
                            self.members.push((tb, n, ev.site));
                        }
                        if let Some(r) = root {
                            self.root(n, *r, ev.site);
                        }
                    }
                }
                What::Closure { k, root, started } => {
                    let n = base + k;
                    let c = &be.closures[*k as usize];
                    if *started {
                        self.edges.push((from, n, EdgeKind::Spawn, ev.site, 0));
                    }
                    let tb = if c.kind == ClosureKind::Spawn {
                        self.edges.push((from, n, EdgeKind::Spawn, ev.site, 0));
                        if let Some(d) = self.scope_spawn
                            && let Some(ss) = self.node_for(InstKey { decl: d, args: Vec::new() }, i)
                            && let Some(p) = self.param_slot(ss, 1)
                        {
                            self.flow.add(&Val::Nodes(vec![n]), p);
                        }
                        let unit = self.cx.mk_tuple(Vec::new());
                        let t = self.cx.intern(TypeKind::Fn {
                            params: Box::new([]),
                            ret: unit,
                            features: 0,
                            deny: 0,
                        });
                        self.table(t)
                    } else {
                        let t = self.subst(c.ty, &args);
                        self.table(t)
                    };
                    if let Some(tb) = tb {
                        self.members.push((tb, n, ev.site));
                    }
                    if let Some(r) = root {
                        self.root(n, *r, ev.site);
                    }
                }
                What::AsStr => {
                    if let Some(d) = self.string[0] {
                        let to = self.call_fn(from, i, d, Vec::new(), ev.site);
                        self.calls.extend(to.iter().map(|&c| (from, ev.tok, c)));
                    }
                }
                What::Fstring => {
                    for d in [self.string[1], self.string[2]].into_iter().flatten() {
                        let to = self.call_fn(from, i, d, Vec::new(), ev.site);
                        self.calls.extend(to.iter().map(|&c| (from, ev.tok, c)));
                    }
                }
            }
        }
        for &(o, ty, tok) in &be.drops {
            let t = self.subst(ty, &args);
            if !self.needs_drop(t) {
                continue;
            }
            let site = Site::Tok { file: be.file, tok };
            if let Some(g) = self.glue_node(t, i, site) {
                self.edges.push((node_of(o), g, EdgeKind::Drop, site, 0));
            }
        }
        self.consts.clear();
    }

    /// A param declared with a generic type that this instance makes a fn type is untracked:
    /// generic code passes such values unseen (bodies without a fn-typed value make no flow).
    fn generic_params(&mut self, i: u32, args: &[GArg], be: &BodyEvents<'_>) {
        let decl = match &self.nodes[i as usize].kind {
            NodeKind::Fn { inst, .. } => inst.decl,
            _ => return,
        };
        let declared: Vec<TypeId> = self.sigs.fn_sig(decl).map_or_else(Vec::new, |s| s.params.clone());
        let mut lists: Vec<(Vec<TypeId>, &[u32])> = vec![(declared, &be.params)];
        for c in &be.closures {
            if let TypeKind::Fn { params, .. } = self.cx.kind(c.ty).clone() {
                lists.push((params.to_vec(), &c.params));
            }
        }
        for (types, binds) in lists {
            for (&t, &b) in types.iter().zip(binds.iter()) {
                if matches!(self.cx.kind(t), TypeKind::Fn { .. }) {
                    continue;
                }
                let st = self.subst(t, args);
                if matches!(self.cx.kind(st), TypeKind::Fn { .. }) {
                    let s = self.flow.slot(SlotKey::Local(i, b));
                    self.flow.tops.push(s);
                }
            }
        }
    }

    /// The slot of param `idx` of fn or closure node `n` (`None` without a checked body).
    fn param_slot(&mut self, n: u32, idx: u32) -> Option<u32> {
        let evs = self.evs;
        let (inst, b) = match &self.nodes.get(n as usize)?.kind {
            NodeKind::Fn { .. } => {
                let bi = self.nodes[n as usize].body?;
                (n, *evs[bi].as_ref()?.params.get(idx as usize)?)
            }
            NodeKind::Closure { parent, index, .. } => {
                let bi = self.nodes[*parent as usize].body?;
                let c = evs[bi].as_ref()?.closures.get(*index as usize)?;
                (*parent, *c.params.get(idx as usize)?)
            }
            _ => return None,
        };
        Some(self.flow.slot(SlotKey::Local(inst, b)))
    }

    /// Solves the fn-value flow and adds the edges of every call through a fn value: to each
    /// value that reaches it, or to its table when an untracked value does. Copy constraints are
    /// propagated to a fixpoint over a CSR of copy edges; a call's targets then pass its
    /// arguments to their params, which adds constraints, until none is new.
    fn solve_flow(&mut self) {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); self.tables.len()];
        for &(t, n, _) in &self.members {
            members[t as usize].push(n);
        }
        for m in &mut members {
            m.sort_unstable();
            m.dedup();
        }
        let mut sets: Vec<Vec<u32>> = Vec::new();
        let mut top: Vec<bool> = Vec::new();
        let mut done_consts = 0;
        let mut done_tops = 0;
        let mut passed: rustc_hash::FxHashSet<(u32, u32)> = rustc_hash::FxHashSet::default();
        loop {
            let n = self.flow.slots.len();
            sets.resize(n, Vec::new());
            top.resize(n, false);
            let mut work: Vec<u32> = Vec::new();
            for &(s, v) in &self.flow.consts[done_consts..] {
                if let Err(p) = sets[s as usize].binary_search(&v) {
                    sets[s as usize].insert(p, v);
                }
                work.push(s);
            }
            done_consts = self.flow.consts.len();
            for &s in &self.flow.tops[done_tops..] {
                top[s as usize] = true;
                work.push(s);
            }
            done_tops = self.flow.tops.len();
            // copy edges by source slot
            let mut start = vec![0u32; n + 1];
            for &(a, _) in &self.flow.copies {
                start[a as usize + 1] += 1;
            }
            for i in 0..n {
                start[i + 1] += start[i];
            }
            let mut fill = start.clone();
            let mut dst = vec![0u32; self.flow.copies.len()];
            for &(a, b) in &self.flow.copies {
                dst[fill[a as usize] as usize] = b;
                fill[a as usize] += 1;
            }
            work.extend(self.flow.copies.iter().map(|c| c.0));
            let mut queued = vec![false; n];
            for &s in &work {
                queued[s as usize] = true;
            }
            while let Some(s) = work.pop() {
                let s = s as usize;
                queued[s] = false;
                for &d in &dst[start[s] as usize..start[s + 1] as usize] {
                    let d = d as usize;
                    let mut changed = top[s] && !top[d];
                    top[d] |= top[s];
                    if !sets[s].is_empty() {
                        let merged = union(&sets[d], &sets[s]);
                        changed |= merged.len() != sets[d].len();
                        sets[d] = merged;
                    }
                    if changed && !queued[d] {
                        queued[d] = true;
                        work.push(d as u32);
                    }
                }
            }
            // a call's targets take its arguments
            let mut added = false;
            for si in 0..self.flow.sites.len() {
                if self.flow.sites[si].args.is_empty() {
                    continue;
                }
                let targets = site_targets(&self.flow.sites[si], &sets, &top, &members);
                for t in targets {
                    if !passed.insert((si as u32, t)) {
                        continue;
                    }
                    let args = self.flow.sites[si].args.clone();
                    for (idx, v) in args {
                        if let Some(p) = self.param_slot(t, idx) {
                            self.flow.add(&v, p);
                            added = true;
                        }
                    }
                }
            }
            if !added {
                break;
            }
        }
        let sites = std::mem::take(&mut self.flow.sites);
        for s in &sites {
            let untracked = match &s.callee {
                Val::Top => true,
                Val::Slot(x) => top[*x as usize],
                Val::Nodes(_) => false,
            };
            if untracked {
                if let Some(tb) = s.table {
                    self.edges.push((s.from, TABLE_BIT | tb, EdgeKind::Indirect, s.site, s.deny));
                    self.calls.push((s.from, s.tok, TABLE_BIT | tb));
                }
                continue;
            }
            for t in site_targets(s, &sets, &top, &members) {
                self.edges.push((s.from, t, EdgeKind::Indirect, s.site, s.deny));
                self.calls.push((s.from, s.tok, t));
            }
        }
    }
}

/// The nodes a value-call site reaches: its callee's values, or its whole table when untracked.
fn site_targets(s: &ValueSite, sets: &[Vec<u32>], top: &[bool], members: &[Vec<u32>]) -> Vec<u32> {
    let table = || s.table.and_then(|t| members.get(t as usize)).cloned().unwrap_or_default();
    match &s.callee {
        Val::Top => table(),
        Val::Nodes(ns) => ns.clone(),
        Val::Slot(x) if top[*x as usize] => table(),
        Val::Slot(x) => sets[*x as usize].clone(),
    }
}

/// The sorted union of two sorted sets.
fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// The root package's `fn main`.
fn main_decl(prog: &Program) -> Option<DeclId> {
    let pkg = prog.packages.iter().find(|p| p.is_root)?;
    let d = prog.module_item(pkg.root_module.get()?, prog.strings.lookup("main")?)?;
    (prog.decl(d)?.kind == DeclKind::Fn).then_some(d)
}

/// Builds the call graph of a checked program. `bodies` sorted by decl.
pub(crate) fn build(
    prog: &Program,
    sigs: &Sigs,
    table: &TypeTable,
    bodies: &[TypedBody],
    brackets: &[Brackets],
) -> CallGraph {
    let [join, par_for, par_for_mut] =
        ["join", "par_for", "par_for_mut"].map(|n| crate::mono::item_at(prog, "std.thread", n));
    let fns = [join, par_for, par_for_mut, crate::mono::item_at(prog, "runtime.thread", "start")];
    // 2: a fn type, 1: a generic param
    let fn_like: Vec<u8> = table
        .iter()
        .map(|(_, k)| match k {
            TypeKind::Fn { .. } => 2,
            TypeKind::Param(_) => 1,
            _ => 0,
        })
        .collect();
    let like =
        |t: TypeId| if t.is_none() || t.is_local() { 0 } else { fn_like.get(t.idx()).copied().unwrap_or(0) };
    let fnish = |t: TypeId| like(t) > 0;
    let is_fn = |t: TypeId| like(t) == 2;
    let evs: Vec<Option<BodyEvents<'_>>> = bodies
        .par_iter()
        .map(|b| {
            let d = prog.decl(b.decl)?;
            if b.has_error || !matches!(d.kind, DeclKind::Fn | DeclKind::Method) {
                return None;
            }
            let icx = brackets.get(b.item.idx()).and_then(|br| ItemCx::new(prog, b.item, br))?;
            Some(extract(&icx, b, sigs, &fns, &fnish, &is_fn))
        })
        .collect();
    let string = if sigs.lang.string.is_some() {
        ["as_str", "new", "push_str"].map(|n| crate::mono::inherent_method(prog, sigs, sigs.lang.string, n))
    } else {
        [None; 3]
    };
    let mut w = Walk {
        prog,
        sigs,
        bodies,
        evs: &evs,
        brackets,
        cx: Overlay::new(table),
        nodes: Vec::new(),
        keys: FxHashMap::default(),
        specs: FxHashMap::default(),
        spec_hint: Vec::new(),
        consts: Vec::new(),
        glue: FxHashMap::default(),
        drop_memo: FxHashMap::default(),
        drop_known: vec![0; table.len()],
        edges: Vec::new(),
        calls: Vec::new(),
        tables: Vec::new(),
        table_of: FxHashMap::default(),
        members: Vec::new(),
        roots: Vec::new(),
        arms: FxHashMap::default(),
        string,
        scope_spawn: crate::mono::item_at(prog, "runtime.thread", "scope_spawn"),
        flow: Flow::default(),
        truncated: false,
    };
    for (i, b) in bodies.iter().enumerate() {
        if evs[i].is_some() && sigs.fn_sig(b.decl).is_some_and(|s| s.generics.is_empty()) {
            w.node_for(InstKey { decl: b.decl, args: Vec::new() }, u32::MAX);
        }
    }
    if let Some(m) = main_decl(prog)
        && let Some(&n) = w.keys.get(&(InstKey { decl: m, args: Vec::new() }, Vec::new()))
    {
        let site = Site::At(w.nodes[n as usize].span);
        w.root(n, RootKind::Main, site);
    }
    let mut next = 0;
    while next < w.nodes.len() {
        w.expand(next as u32);
        next += 1;
    }
    w.solve_flow();
    finish(w)
}

/// Whether fn decl `d` is a `Drop.drop` impl method.
fn is_drop_impl(prog: &Program, sigs: &Sigs, d: DeclId) -> bool {
    prog.decl(d).is_some_and(|x| x.kind == DeclKind::Method)
        && sigs
            .impl_def(prog.decl(d).map_or(DeclId::NONE, |x| x.parent))
            .and_then(|i| i.trait_ref.as_ref())
            .is_some_and(|t| t.trait_decl == sigs.lang.drop)
}

/// Whether decl `d` is declared in a built-in package (`std`, `runtime`, and their dependencies).
fn in_builtin_package(prog: &Program, d: DeclId) -> bool {
    prog.decl(d).and_then(|x| prog.module(x.module)).is_some_and(|m| {
        prog.packages.get(m.package.idx()).is_some_and(|p| p.kind == crate::package::PackageKind::Builtin)
    })
}

fn finish(w: Walk<'_>) -> CallGraph {
    let prog = w.prog;
    let sigs = w.sigs;
    let n_fn = w.nodes.len() as u32;
    let mut nodes: Vec<Node> = w.nodes.into_iter().map(|b| Node { kind: b.kind, span: b.span }).collect();
    let mut table_site: Vec<Option<Site>> = vec![None; w.tables.len()];
    for &(t, _, s) in &w.members {
        table_site[t as usize].get_or_insert(s);
    }
    for &(_, t, _, s, _) in &w.edges {
        if t & TABLE_BIT != 0 {
            table_site[(t & !TABLE_BIT) as usize].get_or_insert(s);
        }
    }
    let tables: Vec<Table> =
        w.tables.iter().enumerate().map(|(i, &ty)| Table { ty, node: n_fn + i as u32 }).collect();
    for (i, s) in table_site.iter().enumerate() {
        let span = s.map_or(Span::new(FileId(u32::MAX), 0, 0), |s| s.span(prog));
        nodes.push(Node { kind: NodeKind::Table { table: i as u32 }, span });
    }
    let n = nodes.len();
    let fix = |t: u32| if t & TABLE_BIT != 0 { n_fn + (t & !TABLE_BIT) } else { t };
    let mut calls: Vec<(u32, u32, u32)> = w.calls.iter().map(|&(f, k, t)| (f, k, fix(t))).collect();
    calls.par_sort_unstable();
    calls.dedup();
    let raw: Vec<(u32, u32, EdgeKind, Site, u8)> = w
        .edges
        .into_iter()
        .map(|(f, t, k, s, d)| (f, fix(t), k, s, d))
        .chain(w.members.iter().map(|&(t, m, s)| (n_fn + t, m, EdgeKind::Member, s, 0)))
        .collect();
    // counting sort by caller keeps each caller's edges in recording order (source order per
    // body); per caller, a stable sort by callee keeps the first edge of each pair
    let mut out_start = vec![0u32; n + 1];
    for e in &raw {
        out_start[e.0 as usize + 1] += 1;
    }
    for i in 0..n {
        out_start[i + 1] += out_start[i];
    }
    let mut fill = out_start.clone();
    let mut slots: Vec<u32> = vec![0; raw.len()];
    for (i, e) in raw.iter().enumerate() {
        slots[fill[e.0 as usize] as usize] = i as u32;
        fill[e.0 as usize] += 1;
    }
    let mut kept: Vec<u32> = Vec::with_capacity(raw.len());
    let mut kept_deny: Vec<u8> = Vec::with_capacity(raw.len());
    let mut new_start = vec![0u32; n + 1];
    for v in 0..n {
        let list = &mut slots[out_start[v] as usize..out_start[v + 1] as usize];
        list.sort_by_key(|&i| raw[i as usize].1);
        let mut last = u32::MAX;
        for &i in list.iter() {
            let (_, to, _, _, deny) = raw[i as usize];
            if to != last {
                last = to;
                kept.push(i);
                kept_deny.push(deny);
            } else if let Some(d) = kept_deny.last_mut() {
                *d &= deny;
            }
        }
        new_start[v + 1] = kept.len() as u32;
    }
    let out_start = new_start;
    let out: Vec<Edge> = kept
        .par_iter()
        .zip(kept_deny.par_iter())
        .map(|(&i, &deny)| {
            let (_, to, kind, site, _) = raw[i as usize];
            Edge { to, kind, site: site.span(prog), deny }
        })
        .collect();
    let callers: Vec<u32> = kept.iter().map(|&i| raw[i as usize].0).collect();
    let (in_start, in_edges) = transpose(n, &out_start, &out);
    let to: Vec<u32> = out.iter().map(|e| e.to).collect();
    let in_from: Vec<u32> = in_edges.iter().map(|&e| callers[e as usize]).collect();
    let (comp, count) = sccs(n, &out_start, &to, &in_start, &in_from);
    let (scc_of, scc_start, scc_nodes, scc_level, level_start) = levels(n, &comp, count, &callers, &to);
    // members a drop cycle may have: glue, `Drop` impls, built-in fns (module docs)
    let drop_path: Vec<bool> = nodes
        .iter()
        .map(|x| match &x.kind {
            NodeKind::Glue { .. } => true,
            NodeKind::Fn { inst, .. } => {
                is_drop_impl(prog, sigs, inst.decl) || in_builtin_package(prog, inst.decl)
            }
            _ => false,
        })
        .collect();
    let mut g = CallGraph {
        nodes,
        out_start,
        out,
        in_start,
        in_edges,
        tables,
        roots: w.roots.iter().map(|&(node, kind, site)| Root { node, kind, site: site.span(prog) }).collect(),
        scc_of,
        scc_start,
        scc_nodes,
        scc_level,
        level_start,
        scc_drop: Vec::new(),
        local_types: w.cx.local,
        truncated: w.truncated,
        calls,
    };
    g.scc_drop = (0..g.scc_count() as u32)
        .map(|s| {
            let ns = g.scc(s);
            g.is_cycle(s)
                && ns.iter().any(|&v| matches!(g.nodes[v as usize].kind, NodeKind::Glue { .. }))
                && ns.iter().all(|&v| drop_path[v as usize])
        })
        .collect();
    g
}

/// In-edges by callee (edge indices; ascending, so by caller), CSR.
fn transpose(n: usize, out_start: &[u32], out: &[Edge]) -> (Vec<u32>, Vec<u32>) {
    let mut start = vec![0u32; n + 1];
    for e in out {
        start[e.to as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut idx = vec![0u32; out.len()];
    for v in 0..n {
        for e in out_start[v]..out_start[v + 1] {
            let t = out[e as usize].to as usize;
            idx[fill[t] as usize] = e;
            fill[t] += 1;
        }
    }
    (start, idx)
}

/// Strongly connected components: a component id per node and the count. Trimming peels nodes
/// with no live in-edge or no live out-edge (each its own component); then each label is the
/// largest node id reaching a node (forward max propagation to a fixpoint), a node whose label is
/// itself is a pivot, and its component is the nodes of its label reaching it (backward search
/// within the label); those are removed and the rest is trimmed and labelled again.
fn sccs(n: usize, out_start: &[u32], to: &[u32], in_start: &[u32], from: &[u32]) -> (Vec<u32>, u32) {
    const NONE: u32 = u32::MAX;
    let succ = |v: usize| &to[out_start[v] as usize..out_start[v + 1] as usize];
    let pred = |v: usize| &from[in_start[v] as usize..in_start[v + 1] as usize];
    let mut comp = vec![NONE; n];
    let mut count = 0u32;
    let mut alive = vec![true; n];
    let mut live: Vec<u32> = (0..n as u32).collect();
    let mut label = vec![0u32; n];
    let mut queued = vec![false; n];
    while !live.is_empty() {
        // trim
        let mut indeg = vec![0u32; n];
        let mut outdeg = vec![0u32; n];
        for &v in &live {
            let v = v as usize;
            indeg[v] = pred(v).iter().filter(|&&u| alive[u as usize]).count() as u32;
            outdeg[v] = succ(v).iter().filter(|&&w| alive[w as usize]).count() as u32;
        }
        let mut stack: Vec<u32> =
            live.iter().copied().filter(|&v| indeg[v as usize] == 0 || outdeg[v as usize] == 0).collect();
        while let Some(v) = stack.pop() {
            let v = v as usize;
            if !alive[v] {
                continue;
            }
            alive[v] = false;
            comp[v] = count;
            count += 1;
            for &w in succ(v) {
                let w = w as usize;
                if alive[w] {
                    indeg[w] -= 1;
                    if indeg[w] == 0 {
                        stack.push(w as u32);
                    }
                }
            }
            for &u in pred(v) {
                let u = u as usize;
                if alive[u] {
                    outdeg[u] -= 1;
                    if outdeg[u] == 0 {
                        stack.push(u as u32);
                    }
                }
            }
        }
        live.retain(|&v| alive[v as usize]);
        if live.is_empty() {
            break;
        }
        // forward max-label propagation
        for &v in &live {
            label[v as usize] = v;
            queued[v as usize] = true;
        }
        let mut work = live.clone();
        while let Some(v) = work.pop() {
            let v = v as usize;
            queued[v] = false;
            for &w in succ(v) {
                let w = w as usize;
                if alive[w] && label[w] < label[v] {
                    label[w] = label[v];
                    if !queued[w] {
                        queued[w] = true;
                        work.push(w as u32);
                    }
                }
            }
        }
        // backward search per pivot within its label
        for &r in &live {
            if label[r as usize] != r {
                continue;
            }
            let c = count;
            count += 1;
            comp[r as usize] = c;
            let mut st = vec![r];
            while let Some(v) = st.pop() {
                for &u in pred(v as usize) {
                    let u = u as usize;
                    if alive[u] && comp[u] == NONE && label[u] == r {
                        comp[u] = c;
                        st.push(u as u32);
                    }
                }
            }
        }
        for &v in &live {
            if comp[v as usize] != NONE {
                alive[v as usize] = false;
            }
        }
        live.retain(|&v| alive[v as usize]);
    }
    (comp, count)
}

type Levels = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

/// Levels of the condensed graph (longest path to a sink, by level-synchronous peeling from the
/// sinks) and the SCC renumbering by (level, smallest node): `(scc_of, scc_start, scc_nodes,
/// scc_level, level_start)`.
fn levels(n: usize, comp: &[u32], count: u32, callers: &[u32], to: &[u32]) -> Levels {
    let c = count as usize;
    // condensed edges with multiplicity, by callee component
    let mut outdeg = vec![0u32; c];
    let mut pstart = vec![0u32; c + 1];
    for (e, &t) in to.iter().enumerate() {
        let (a, b) = (comp[callers[e] as usize], comp[t as usize]);
        if a != b {
            outdeg[a as usize] += 1;
            pstart[b as usize + 1] += 1;
        }
    }
    for i in 0..c {
        pstart[i + 1] += pstart[i];
    }
    let mut fill = pstart.clone();
    let mut preds = vec![0u32; pstart[c] as usize];
    for (e, &t) in to.iter().enumerate() {
        let (a, b) = (comp[callers[e] as usize], comp[t as usize]);
        if a != b {
            preds[fill[b as usize] as usize] = a;
            fill[b as usize] += 1;
        }
    }
    let mut level = vec![0u32; c];
    let mut frontier: Vec<u32> = (0..count).filter(|&s| outdeg[s as usize] == 0).collect();
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &s in &frontier {
            for &p in &preds[pstart[s as usize] as usize..pstart[s as usize + 1] as usize] {
                let p = p as usize;
                level[p] = level[p].max(level[s as usize] + 1);
                outdeg[p] -= 1;
                if outdeg[p] == 0 {
                    next.push(p as u32);
                }
            }
        }
        frontier = next;
    }
    let mut min_node = vec![u32::MAX; c];
    for (v, &s) in comp.iter().enumerate() {
        let s = s as usize;
        min_node[s] = min_node[s].min(v as u32);
    }
    let mut order: Vec<u32> = (0..count).collect();
    order.sort_by_key(|&s| (level[s as usize], min_node[s as usize]));
    let mut renum = vec![0u32; c];
    for (new, &old) in order.iter().enumerate() {
        renum[old as usize] = new as u32;
    }
    let scc_of: Vec<u32> = comp.iter().map(|&s| renum[s as usize]).collect();
    let mut scc_start = vec![0u32; c + 1];
    for &s in &scc_of {
        scc_start[s as usize + 1] += 1;
    }
    for i in 0..c {
        scc_start[i + 1] += scc_start[i];
    }
    let mut fill = scc_start.clone();
    let mut scc_nodes = vec![0u32; n];
    for (v, &s) in scc_of.iter().enumerate() {
        let s = s as usize;
        scc_nodes[fill[s] as usize] = v as u32;
        fill[s] += 1;
    }
    let scc_level: Vec<u32> = order.iter().map(|&s| level[s as usize]).collect();
    let n_levels = scc_level.last().map_or(0, |&l| l as usize + 1);
    let mut level_start = vec![0u32; n_levels + 1];
    for &l in &scc_level {
        level_start[l as usize + 1] += 1;
    }
    for i in 0..n_levels {
        level_start[i + 1] += level_start[i];
    }
    (scc_of, scc_start, scc_nodes, scc_level, level_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (comp, count) of an edge list.
    fn run(n: usize, es: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut es = es.to_vec();
        es.sort();
        let mut out_start = vec![0u32; n + 1];
        for e in &es {
            out_start[e.0 as usize + 1] += 1;
        }
        for i in 0..n {
            out_start[i + 1] += out_start[i];
        }
        let site = Span::new(stave_diag::FileId(0), 0, 0);
        let out: Vec<Edge> =
            es.iter().map(|&(_, t)| Edge { to: t, kind: EdgeKind::Call, site, deny: 0 }).collect();
        let callers: Vec<u32> = es.iter().map(|e| e.0).collect();
        let (in_start, in_edges) = transpose(n, &out_start, &out);
        let to: Vec<u32> = out.iter().map(|e| e.to).collect();
        let from: Vec<u32> = in_edges.iter().map(|&e| callers[e as usize]).collect();
        let (comp, count) = sccs(n, &out_start, &to, &in_start, &from);
        let (scc_of, _, _, scc_level, _) = levels(n, &comp, count, &callers, &to);
        let lv: Vec<u32> = scc_of.iter().map(|&s| scc_level[s as usize]).collect();
        (scc_of, lv, comp)
    }

    #[test]
    fn components_and_levels() {
        // 0 -> 1 -> 2 -> 1, 2 -> 3, 4 -> 4, 5 -> 0, 6 isolated
        let (scc, lv, _) = run(7, &[(0, 1), (1, 2), (2, 1), (2, 3), (4, 4), (5, 0)]);
        assert_eq!(scc[1], scc[2]);
        assert_ne!(scc[0], scc[1]);
        assert_ne!(scc[2], scc[3]);
        assert_eq!(lv, vec![2, 1, 1, 0, 0, 3, 0]);
        // ids ascend with level
        assert!(scc[3] < scc[1] && scc[1] < scc[0] && scc[0] < scc[5]);
    }

    #[test]
    fn nested_cycles_need_several_rounds() {
        // two cycles joined one way, and a cycle whose largest node is reached from another cycle
        let es = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2), (3, 9), (9, 8), (8, 3), (5, 6), (6, 5), (6, 0)];
        let (scc, lv, _) = run(10, &es);
        assert_eq!(scc[0], scc[1]);
        assert_eq!(scc[2], scc[3]);
        assert_eq!(scc[3], scc[8]);
        assert_eq!(scc[8], scc[9]);
        assert_eq!(scc[5], scc[6]);
        assert_ne!(scc[0], scc[2]);
        assert_ne!(scc[5], scc[0]);
        assert_eq!((lv[2], lv[0], lv[5]), (0, 1, 2));
    }

    #[test]
    fn long_chain_and_ring() {
        let n = 2000;
        let mut es: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
        let (_, lv, _) = run(n, &es);
        assert_eq!(lv[0], n as u32 - 1);
        es.push((n as u32 - 1, 0));
        let (scc, lv, _) = run(n, &es);
        assert!(scc.iter().all(|&s| s == scc[0]));
        assert!(lv.iter().all(|&l| l == 0));
    }
}
