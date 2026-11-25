//! The structure pass (contracts/checked-ir.md "Loading" step 2) over the parsed lines, and the
//! build of the module from them: sections and their order, the nesting of fn bodies, id
//! resolution (blocks and values renumbered in text order, fns sorted by instance id), and every
//! datum of the text placed in its [`Module`] field.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use super::line::{
    Arg, ArgV, ClaimKindL, FnHead, IdRef, InstL, Line, OpL, Parsed, RefKind, RefTargetL, TermL, Tgt,
};
use super::{ErrorKind, FnLines, Header, Level, LineMap, MAX_ERRORS, ReadError, StackEdge, StackInput};
use crate::Known;
use crate::ir::{
    Block, BlockData, Body, Const, ConstId, DataRef, DataTarget, Func, FuncId, InstanceId, Module, Op,
    Operand, Site, SiteInfo, Span, StrId, SwitchCase, Target, Term, Value, ValueData, ValueDef,
};
use crate::types::{AdtKind, DeclId, DeclInfo, TypeId, TypeKey, TypeTable};
use crate::witness::{NO_BLOCK, Pre, Row, Summary};

/// A parsed line and its 1-based line number.
pub(crate) struct Ln {
    pub no: u32,
    pub p: Parsed,
}

/// A static as its lines give it; its bytes are laid out once the type table is verified.
pub(crate) struct StaticG {
    pub line: u32,
    pub ty: TypeId,
    pub path: String,
    /// (line, column, offset, bytes) per `data` line.
    pub data: Vec<(u32, u32, u32, Vec<u8>)>,
    pub refs: Vec<DataRef>,
}

/// What the structure pass builds: the module without its statics' bytes, and the rest of the
/// text's data.
pub(crate) struct Built {
    pub header: Header,
    pub module: Module,
    pub statics: Vec<StaticG>,
    pub stack: Option<StackInput>,
    pub lines: LineMap,
}

struct BlockG {
    line: u32,
    col: u32,
    label: u32,
    params: Vec<(u32, u32, TypeId)>,
    insts: Vec<(u32, InstL)>,
    term: Option<(u32, TermL)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BodyState {
    Bodiless,
    First,
    InBlock,
    AfterTerm,
    Closed,
}

struct FnG {
    line: u32,
    col: u32,
    head: FnHead,
    blocks: Vec<BlockG>,
    /// (line, column, kind, row) per `claim` line.
    claims: Vec<(u32, u32, ClaimKindL, Row)>,
    state: BodyState,
}

struct Errs(Vec<ReadError>);

impl Errs {
    fn at(&mut self, line: u32, col: u32, msg: impl Into<String>) {
        if self.0.len() < MAX_ERRORS {
            self.0.push(ReadError { line, col: Some(col), kind: ErrorKind::Structure, msg: msg.into() });
        }
    }
}

/// The section of a line kind (contracts/checked-ir.md "Sections"; the stack section's kinds in
/// their listed order) and its name for messages.
fn section(l: &Line) -> (u8, &'static str) {
    match l {
        Line::Header { .. } => (0, "`stave.ir` header"),
        Line::Decl { .. } => (1, "`decl`"),
        Line::Type { .. } => (2, "`type`"),
        Line::Shape { .. } => (3, "`shape`"),
        Line::Str { .. } => (4, "`str`"),
        Line::Site { .. } => (5, "`site`"),
        Line::Static { .. } => (6, "`static`"),
        Line::Data { .. } => (6, "`data`"),
        Line::Ref { .. } => (6, "`ref`"),
        Line::Main { .. } => (7, "`main`"),
        Line::Cell(_) => (8, "`cell`"),
        Line::RunStack(_) => (9, "`runstack`"),
        Line::Truncated => (10, "`truncated`"),
        Line::Node { .. } => (11, "`node`"),
        Line::Edge { .. } => (12, "`edge`"),
        Line::Nodes { .. } => (13, "`nodes`"),
        Line::Fn(_) => (14, "`fn`"),
        Line::Claim { .. } => (14, "`claim`"),
        Line::Block { .. } => (14, "block header"),
        Line::Inst(_) => (14, "instruction"),
        Line::Term(_) => (14, "terminator"),
        Line::Close => (14, "`}`"),
    }
}

/// The structure pass and the module build. Lines are in file order, blank lines dropped.
pub(crate) fn structure(lines: Vec<Ln>) -> Result<Built, Vec<ReadError>> {
    let mut e = Errs(Vec::new());
    let mut header: Option<Header> = None;
    let mut header_line = 1;
    let mut decls: Vec<(u32, u32, DeclInfo)> = Vec::new();
    let mut types: Vec<(u32, u32, TypeKey)> = Vec::new();
    let mut shapes: Vec<(u32, u32, u32, crate::types::AdtShape)> = Vec::new();
    let mut strings: Vec<Vec<u8>> = Vec::new();
    let mut sites: Vec<SiteInfo> = Vec::new();
    let mut statics: Vec<StaticG> = Vec::new();
    let mut main: Option<(u32, u32, InstanceId)> = None;
    let mut cell: Option<(u32, DeclId)> = None;
    let mut run_stack: Option<u64> = None;
    let mut truncated: Option<u32> = None;
    let mut nodes: Vec<bool> = Vec::new();
    let mut edges: Vec<StackEdge> = Vec::new();
    let mut nodes_lines: Vec<(u32, u32, InstanceId, Vec<u32>)> = Vec::new();
    let mut stack_lines = false;
    let mut fns: Vec<FnG> = Vec::new();
    let mut refs: Vec<(u32, IdRef)> = Vec::new();
    let mut cur: (u8, &'static str) = (0, "`stave.ir` header");

    if lines.is_empty() {
        e.at(1, 1, "the file is empty: it begins with the `stave.ir` header line");
    }
    for (k, Ln { no, p }) in lines.into_iter().enumerate() {
        let Parsed { line, col, refs: r } = p;
        let (rank, name) = section(&line);
        if k == 0 && rank != 0 {
            e.at(no, col, "the file begins with the `stave.ir` header line");
        }
        if rank < cur.0 {
            e.at(
                no,
                col,
                format!("{name} line out of section order: {name} lines come before {} lines", cur.1),
            );
            continue;
        }
        cur = (rank, name);
        refs.extend(r.into_iter().map(|r| (no, r)));
        let open_static = statics.len();
        if matches!(line, Line::Fn(_))
            && let Some(f) = fns.last()
            && matches!(f.state, BodyState::First | BodyState::InBlock | BodyState::AfterTerm)
        {
            e.at(
                no,
                col,
                format!("`fn` line inside the body of fn @{}: the body is not closed by `}}`", f.head.inst),
            );
        }
        match line {
            Line::Header { target, mode, level } => {
                if header.is_some() {
                    e.at(no, col, "duplicate `stave.ir` header line");
                } else {
                    header = Some(Header { target, mode, level });
                    header_line = no;
                }
            }
            Line::Decl { id, kind, path } => {
                if let Some((_, _, last)) = decls.last()
                    && last.id.0 >= id
                {
                    e.at(
                        no,
                        col,
                        format!(
                            "decl d{id} out of order: decl lines ascend by id, without repeats (after d{})",
                            last.id.0
                        ),
                    );
                }
                decls.push((no, col, DeclInfo { id: DeclId(id), kind, path }));
            }
            Line::Type { id, key } => {
                if id as usize != types.len() {
                    e.at(
                        no,
                        col,
                        format!(
                            "type t{id} out of order: type lines are t0, t1, ... (expected t{})",
                            types.len()
                        ),
                    );
                }
                types.push((no, col, key));
            }
            Line::Shape { id, shape } => {
                if let Some((_, _, last, _)) = shapes.last()
                    && *last >= id
                {
                    e.at(
                        no,
                        col,
                        format!("shape of t{id} out of order: shape lines ascend by type, without repeats"),
                    );
                }
                shapes.push((no, col, id, shape));
            }
            Line::Str { id, bytes } => {
                if id as usize != strings.len() {
                    e.at(
                        no,
                        col,
                        format!(
                            "str s{id} out of order: str lines are s0, s1, ... (expected s{})",
                            strings.len()
                        ),
                    );
                }
                strings.push(bytes);
            }
            Line::Site { row, file, line, col: c, func } => {
                if row as usize != sites.len() {
                    e.at(
                        no,
                        col,
                        format!(
                            "site {row} out of order: site rows are 0, 1, ... (expected {})",
                            sites.len()
                        ),
                    );
                }
                sites.push(SiteInfo { file, line, col: c, func });
            }
            Line::Static { index, ty, path } => {
                if index as usize != statics.len() {
                    e.at(
                        no,
                        col,
                        format!(
                            "static {index} out of order: statics are 0, 1, ... (expected {})",
                            statics.len()
                        ),
                    );
                }
                statics.push(StaticG { line: no, ty, path, data: Vec::new(), refs: Vec::new() });
            }
            Line::Data { index, offset, bytes } => match statics.last_mut() {
                Some(s) if index as usize + 1 == open_static => {
                    if !s.refs.is_empty() {
                        e.at(no, col, format!("`data` line of static {index} after its `ref` lines"));
                    } else if let Some((_, _, last, _)) = s.data.last()
                        && last / 64 >= offset / 64
                    {
                        e.at(
                            no,
                            col,
                            format!("`data` lines of static {index} ascend by offset, one per 64-byte block"),
                        );
                    }
                    s.data.push((no, col, offset, bytes));
                }
                _ => e.at(
                    no,
                    col,
                    format!("`data` line of static {index} outside the lines of static {index}"),
                ),
            },
            Line::Ref { index, offset, target, addend } => match statics.last_mut() {
                Some(s) if index as usize + 1 == open_static => {
                    if s.refs.last().is_some_and(|r| r.offset >= offset) {
                        e.at(
                            no,
                            col,
                            format!("`ref` lines of static {index} ascend by offset, without repeats"),
                        );
                    }
                    let target = match target {
                        RefTargetL::Str(sid) => DataTarget::Str(StrId(sid)),
                        RefTargetL::Static(j) => DataTarget::Static(j),
                    };
                    s.refs.push(DataRef { offset, target, addend });
                }
                _ => {
                    e.at(no, col, format!("`ref` line of static {index} outside the lines of static {index}"))
                }
            },
            Line::Main { inst, col: c } => {
                if main.is_some() {
                    e.at(no, col, "duplicate `main` line");
                }
                main = Some((no, c, inst));
            }
            Line::Cell(d) => {
                if cell.is_some() {
                    e.at(no, col, "duplicate `cell` line");
                }
                cell = Some((no, d));
            }
            Line::RunStack(n) => {
                stack_lines = true;
                if run_stack.is_some() {
                    e.at(no, col, "duplicate `runstack` line");
                }
                run_stack = Some(n);
            }
            Line::Truncated => {
                stack_lines = true;
                if truncated.is_some() {
                    e.at(no, col, "duplicate `truncated` line");
                }
                truncated = Some(no);
            }
            Line::Node { n, rec } => {
                stack_lines = true;
                if n as usize != nodes.len() {
                    e.at(
                        no,
                        col,
                        format!("node {n} out of order: node lines are 0, 1, ... (expected {})", nodes.len()),
                    );
                }
                nodes.push(rec);
            }
            Line::Edge { from, to, kind } => {
                stack_lines = true;
                if edges.last().is_some_and(|l| (l.from, l.to) >= (from, to)) {
                    e.at(no, col, "`edge` lines ascend by (from, to), without repeats");
                }
                edges.push(StackEdge { from, to, kind });
            }
            Line::Nodes { inst, col: c, nodes: ns } => {
                stack_lines = true;
                if nodes_lines.last().is_some_and(|l| l.2 >= inst) {
                    e.at(no, col, "`nodes` lines ascend by instance id, one per fn");
                }
                nodes_lines.push((no, c, inst, ns));
            }
            Line::Fn(head) => {
                let state = if head.body { BodyState::First } else { BodyState::Bodiless };
                fns.push(FnG { line: no, col, head: *head, blocks: Vec::new(), claims: Vec::new(), state });
            }
            Line::Claim { inst, col: c, kind, row } => match fns.last_mut() {
                Some(f) if matches!(f.state, BodyState::Closed | BodyState::Bodiless) => {
                    if f.head.inst != inst {
                        e.at(no, c, format!("claim of @{inst} after fn @{}: claim lines follow the body of the fn they name", f.head.inst));
                    } else if f.state == BodyState::Bodiless {
                        e.at(no, col, format!("claim of fn @{inst}, which has no body"));
                    } else if matches!(kind, ClaimKindL::Pre { .. })
                        && f.claims.iter().any(|c| c.2 == ClaimKindL::Post)
                    {
                        e.at(no, col, "a `pre` claim after a `post` claim: the `pre` rows come first");
                    }
                    f.claims.push((no, col, kind, row));
                }
                Some(f) if f.state != BodyState::Closed => {
                    e.at(no, col, format!("`claim` line inside the body of fn @{}", f.head.inst))
                }
                _ => e.at(no, col, "`claim` line before the first fn"),
            },
            Line::Block { label, params } => match fns.last_mut() {
                Some(f) => match f.state {
                    BodyState::First | BodyState::AfterTerm => {
                        if f.state == BodyState::First && label != 0 {
                            e.at(
                                no,
                                col,
                                format!("the body of fn @{} begins with block b0, not b{label}", f.head.inst),
                            );
                        }
                        f.blocks.push(BlockG { line: no, col, label, params, insts: Vec::new(), term: None });
                        f.state = BodyState::InBlock;
                    }
                    BodyState::InBlock => {
                        let last = f.blocks.last().map_or(0, |b| b.label);
                        e.at(
                            no,
                            col,
                            format!("block b{label} begins before block b{last} has its terminator"),
                        );
                    }
                    _ => e.at(no, col, "block header outside a fn body"),
                },
                None => e.at(no, col, "block header outside a fn body"),
            },
            Line::Inst(inst) => match fns.last_mut() {
                Some(f) if f.state == BodyState::InBlock => {
                    if let Some(b) = f.blocks.last_mut() {
                        b.insts.push((no, *inst));
                    }
                }
                Some(f) if f.state == BodyState::AfterTerm => {
                    e.at(no, col, "instruction after the terminator of its block")
                }
                Some(f) if f.state == BodyState::First => {
                    e.at(no, col, "instruction before the first block header")
                }
                _ => e.at(no, col, "instruction outside a fn body"),
            },
            Line::Term(t) => match fns.last_mut() {
                Some(f) if f.state == BodyState::InBlock => {
                    if let Some(b) = f.blocks.last_mut() {
                        b.term = Some((no, t));
                    }
                    f.state = BodyState::AfterTerm;
                }
                Some(f) if f.state == BodyState::AfterTerm => {
                    e.at(no, col, "a second terminator: a block has one")
                }
                Some(f) if f.state == BodyState::First => {
                    e.at(no, col, "terminator before the first block header")
                }
                _ => e.at(no, col, "terminator outside a fn body"),
            },
            Line::Close => match fns.last_mut() {
                Some(f) => match f.state {
                    BodyState::AfterTerm => f.state = BodyState::Closed,
                    BodyState::First => {
                        e.at(no, col, format!("the body of fn @{} has no block", f.head.inst))
                    }
                    BodyState::InBlock => e.at(no, col, "`}` before the terminator of the last block"),
                    _ => e.at(no, col, "`}` outside a fn body"),
                },
                None => e.at(no, col, "`}` outside a fn body"),
            },
        }
    }
    if let Some(f) = fns.last()
        && matches!(f.state, BodyState::First | BodyState::InBlock | BodyState::AfterTerm)
    {
        e.at(f.line, f.col, format!("the body of fn @{} is not closed by `}}`", f.head.inst));
    }
    let Some(header) = header else {
        if e.0.is_empty() {
            e.at(1, 1, "the `stave.ir` header line is missing");
        }
        return Err(e.0);
    };
    if !e.0.is_empty() {
        return Err(e.0);
    }

    // Ids that name lines.
    let decl_ids: Vec<u32> = decls.iter().map(|d| d.2.id.0).collect();
    for (no, r) in &refs {
        let (have, what) = match r.kind {
            RefKind::Type => ((r.n as usize) < types.len(), format!("t{} names no type line", r.n)),
            RefKind::Str => ((r.n as usize) < strings.len(), format!("s{} names no str line", r.n)),
            RefKind::Site => ((r.n as usize) < sites.len(), format!("site row {} names no site line", r.n)),
            RefKind::Static => {
                ((r.n as usize) < statics.len(), format!("static {} names no static line", r.n))
            }
            RefKind::Decl => (decl_ids.binary_search(&r.n).is_ok(), format!("d{} names no decl line", r.n)),
            RefKind::Node => ((r.n as usize) < nodes.len(), format!("node {} names no node line", r.n)),
        };
        if !have {
            e.at(*no, r.col, what);
        }
    }
    let mut named: Vec<u32> =
        refs.iter().filter(|(_, r)| r.kind == RefKind::Decl).map(|(_, r)| r.n).collect();
    named.sort_unstable();
    for (no, col, d) in &decls {
        if named.binary_search(&d.id.0).is_err() {
            e.at(*no, *col, format!("decl d{} is named by no `adt` type and no `cell` line", d.id.0));
        }
    }
    if !e.0.is_empty() {
        return Err(e.0);
    }

    // Types and shapes.
    let mut table = TypeTable::new();
    let mut first: HashMap<&TypeKey, usize> = HashMap::new();
    for (i, (no, col, key)) in types.iter().enumerate() {
        if let Some(&j) = first.get(key) {
            e.at(*no, *col, format!("type t{i} has the key of t{j}: no two type lines have equal keys"));
        } else {
            first.insert(key, i);
            table.intern(key.clone());
        }
    }
    if !e.0.is_empty() {
        return Err(e.0);
    }
    let decl_kind = |d: DeclId| decls.iter().find(|x| x.2.id == d).map(|x| x.2.kind);
    for (no, col, id, shape) in shapes {
        let t = TypeId(id);
        let def = match table.key(t) {
            Some(TypeKey::Adt { def, .. }) => Some(*def),
            _ => None,
        };
        match def {
            Some(def) => {
                if decl_kind(def) != Some(shape.kind) {
                    let k = |k: Option<AdtKind>| match k {
                        Some(AdtKind::Struct) => "struct",
                        Some(AdtKind::Enum) => "enum",
                        None => "?",
                    };
                    e.at(
                        no,
                        col,
                        format!(
                            "shape of t{id} is a {}, its decl d{} a {}",
                            k(Some(shape.kind)),
                            def.0,
                            k(decl_kind(def))
                        ),
                    );
                }
                table.set_shape(t, shape);
            }
            None => e.at(no, col, format!("shape of t{id}, which is not an `adt` type")),
        }
    }

    // Functions, by instance id.
    let mut order: Vec<usize> = (0..fns.len()).collect();
    order.sort_by(|&a, &b| fns[a].head.inst.cmp(&fns[b].head.inst).then(a.cmp(&b)));
    for w in order.windows(2) {
        if fns[w[0]].head.inst == fns[w[1]].head.inst {
            let (a, b) = if fns[w[0]].line < fns[w[1]].line { (w[0], w[1]) } else { (w[1], w[0]) };
            e.at(
                fns[b].line,
                fns[b].col,
                format!("fn @{} is defined twice (first at line {})", fns[b].head.inst, fns[a].line),
            );
        }
    }
    let fids: HashMap<InstanceId, FuncId> =
        order.iter().enumerate().map(|(k, &i)| (fns[i].head.inst.clone(), FuncId(k as u32))).collect();
    let main = match main {
        Some((no, col, inst)) => match fids.get(&inst) {
            Some(&f) => Some(f),
            None => {
                e.at(no, col, format!("`main` names @{inst}, which heads no fn"));
                None
            }
        },
        None => None,
    };
    let mut nodes_of_func: Vec<Vec<u32>> = vec![Vec::new(); fns.len()];
    for (no, col, inst, ns) in nodes_lines {
        match fids.get(&inst) {
            Some(&f) => nodes_of_func[f.index()] = ns,
            None => e.at(no, col, format!("`nodes` names @{inst}, which heads no fn")),
        }
    }

    let mut module = Module::new();
    let intrinsics = intrinsic_table(&fns);
    let mut map = LineMap {
        header: header_line,
        types: types.iter().map(|t| t.0).collect(),
        statics: Vec::new(),
        funcs: Vec::new(),
    };
    let mut stated: Vec<Option<Arc<Summary>>> = Vec::new();
    let level = header.level;
    for &i in &order {
        let f = &fns[i];
        let known = if f.head.known {
            let k = known_by_path(&f.head.path);
            if k.is_none() {
                e.at(
                    f.line,
                    f.head.known_col,
                    format!("`known`: `{}` is the path of no compiler-known item", f.head.path),
                );
            }
            k
        } else {
            None
        };
        let mut lines = FnLines { header: f.line, blocks: Vec::new(), terms: Vec::new(), insts: Vec::new() };
        let body = if f.head.body {
            let mut b = BodyCx { e: &mut e, fids: &fids, intrinsics: &intrinsics, inst: &f.head.inst };
            b.body(f, &mut lines)
        } else {
            None
        };
        if !f.claims.is_empty() {
            if level == Level::Lowered {
                for (no, col, _, _) in &f.claims {
                    e.at(*no, *col, "no `claim` line at level `lowered`");
                }
            }
            stated.resize(module.funcs.len() + 1, None);
            stated[module.funcs.len()] = Some(Arc::new(summary(&f.claims)));
        }
        let h = &f.head;
        module.funcs.push(Func {
            instance: h.inst.clone(),
            path: h.path.clone(),
            type_args: h.type_args.clone(),
            params: h.params.clone(),
            ret: h.ret,
            features: h.features,
            body,
            name_site: h.name_site.map_or(Site::NONE, Site),
            symbol: h.symbol.clone(),
            variadic: h.variadic,
            known,
            alt_stack: h.alt_stack,
            unbuilt: h.unbuilt,
        });
        map.funcs.push(lines);
    }
    if !e.0.is_empty() {
        return Err(e.0);
    }
    if level != Level::Lowered {
        stated.resize(module.funcs.len(), None);
    }
    module.types = table;
    module.strings = strings;
    module.sites = sites;
    module.intrinsics = intrinsics;
    module.stated = stated;
    module.cell = cell.map(|c| c.1);
    module.main = main;
    module.decls = decls.into_iter().map(|d| d.2).collect();
    map.statics = statics.iter().map(|s| s.line).collect();
    let stack = stack_lines.then(|| StackInput {
        run_stack,
        truncated: truncated.is_some(),
        nodes,
        edges,
        nodes_of_func,
    });
    Ok(Built { header, module, statics, stack, lines: map })
}

/// The compiler-known item whose path is `path`.
fn known_by_path(path: &str) -> Option<Known> {
    Known::ALL.iter().copied().find(|k| k.path() == path)
}

/// The summary that loaded `claim` rows state: rows only, a `checked` `pre` row marked as one
/// that reads a `requires` check.
fn summary(claims: &[(u32, u32, ClaimKindL, Row)]) -> Summary {
    let mut s = Summary::default();
    for (_, _, kind, row) in claims {
        match kind {
            ClaimKindL::Pre { checked } => {
                s.pre.push(Pre { row: *row, entry: if *checked { 0 } else { NO_BLOCK }, half: 0 })
            }
            ClaimKindL::Post => s.post.push(*row),
        }
    }
    s
}

struct BodyCx<'a> {
    e: &'a mut Errs,
    fids: &'a HashMap<InstanceId, FuncId>,
    intrinsics: &'a [String],
    inst: &'a InstanceId,
}

impl BodyCx<'_> {
    /// The body of `f`: blocks numbered by their headers, values by their definitions, both in
    /// text order; every pool in print order.
    fn body(&mut self, f: &FnG, lines: &mut FnLines) -> Option<Body> {
        let n0 = self.e.0.len();
        let mut labels: HashMap<u32, u32> = HashMap::new();
        for (k, b) in f.blocks.iter().enumerate() {
            if labels.insert(b.label, k as u32).is_some() {
                self.e.at(b.line, b.col, format!("block b{} is defined twice in fn @{}", b.label, self.inst));
            }
        }
        let mut vals: HashMap<u32, Value> = HashMap::new();
        let mut values: Vec<ValueData> = Vec::new();
        let mut define = |e: &mut Errs, v: u32, line: u32, col: u32, d: ValueData| {
            let id = Value(values.len() as u32);
            if vals.insert(v, id).is_some() {
                e.at(line, col, format!("value %{v} is defined twice in fn @{}", self.inst));
            }
            values.push(d);
        };
        let mut idx = 0u32;
        for (k, b) in f.blocks.iter().enumerate() {
            for (j, &(v, col, ty)) in b.params.iter().enumerate() {
                let def = ValueDef::Param { block: Block(k as u32), index: j as u32 };
                define(self.e, v, b.line, col, ValueData { ty, def });
            }
            for (line, inst) in &b.insts {
                if let Some((v, col)) = inst.result {
                    let ty = inst.ty.unwrap_or(TypeId(u32::MAX));
                    define(self.e, v, *line, col, ValueData { ty, def: ValueDef::Inst(idx) });
                }
                idx += 1;
            }
        }
        let mut body = Body { values, ..Body::default() };
        let mut pool = Pool::default();
        let r = Resolve { vals: &vals, labels: &labels, fids: self.fids, inst: self.inst };
        for b in &f.blocks {
            let pstart = body.block_params.len() as u32;
            for &(v, _, _) in &b.params {
                body.block_params.push(vals.get(&v).copied().unwrap_or(Value(u32::MAX)));
            }
            let istart = body.insts.len() as u32;
            for (line, inst) in &b.insts {
                let op = match &inst.op {
                    OpL::Op(op) => *op,
                    OpL::Intrinsic(name) => match intrinsic_id(self.intrinsics, name) {
                        Some(id) => Op::Intrinsic(id),
                        None => {
                            self.e.at(*line, 1, format!("`intrinsic.{name}` is not in the intrinsic table"));
                            Op::Intrinsic(crate::ir::IntrinsicId(u32::MAX))
                        }
                    },
                };
                let start = body.operands.len() as u32;
                for a in &inst.args {
                    let o = r.operand(self.e, *line, a, &mut pool);
                    body.operands.push(o);
                }
                let args = Span { start, len: body.operands.len() as u32 - start };
                body.insts.op.push(op);
                body.insts.ty.push(inst.ty);
                body.insts.result.push(inst.result.and_then(|(v, _)| vals.get(&v).copied()));
                body.insts.args.push(args);
                body.insts.site.push(inst.site.map_or(Site::NONE, Site));
                lines.insts.push(*line);
            }
            let insts = Span { start: istart, len: body.insts.len() as u32 - istart };
            let (tline, term) = match &b.term {
                Some((no, t)) => (*no, r.term(self.e, *no, t, &mut body, &mut pool)),
                None => (b.line, Term::Unreachable),
            };
            body.blocks.push(BlockData {
                params: Span { start: pstart, len: b.params.len() as u32 },
                insts,
                term,
            });
            lines.blocks.push(b.line);
            lines.terms.push(tline);
        }
        body.consts = pool.consts;
        if self.e.0.len() != n0 {
            return None;
        }
        reject_unreachable_blocks(self.e, &body, f, self.inst);
        (self.e.0.len() == n0).then_some(body)
    }
}

/// A canonical body has no block unreachable from `b0` (checked-ir.md "Canonical module"): the
/// printer could not print it back.
fn reject_unreachable_blocks(e: &mut Errs, body: &Body, f: &FnG, inst: &InstanceId) {
    for (k, reached) in body.reachable().into_iter().enumerate() {
        if !reached && let Some(b) = f.blocks.get(k) {
            e.at(b.line, b.col, format!("block b{} of fn @{inst} is unreachable from b0", b.label));
        }
    }
}

/// The module's intrinsic table: the names the bodies use, each once, sorted by bytes
/// (contracts/checked-ir.md "Canonical module").
fn intrinsic_table(fns: &[FnG]) -> Vec<String> {
    let mut names: BTreeSet<&str> = BTreeSet::new();
    for f in fns {
        for b in &f.blocks {
            for (_, inst) in &b.insts {
                if let OpL::Intrinsic(name) = &inst.op {
                    names.insert(name);
                }
            }
        }
    }
    names.into_iter().map(str::to_string).collect()
}

/// The id of `name` in `table`, a name of a table [`intrinsic_table`] made.
fn intrinsic_id(table: &[String], name: &str) -> Option<crate::ir::IntrinsicId> {
    let i = table.binary_search_by(|n| n.as_bytes().cmp(name.as_bytes())).ok()?;
    Some(crate::ir::IntrinsicId(i as u32))
}

/// The constant pool of one body: one entry per distinct constant, numbered by first use.
#[derive(Default)]
struct Pool {
    consts: Vec<Const>,
    ids: HashMap<(u8, i128), ConstId>,
}

impl Pool {
    fn id(&mut self, c: Const) -> ConstId {
        let key = const_key(&c);
        *self.ids.entry(key).or_insert_with(|| {
            self.consts.push(c);
            ConstId(self.consts.len() as u32 - 1)
        })
    }
}

/// What makes two constants the same: the kind and the bits, never the value (`0.0` and `-0.0`
/// differ, `f16` and `bf16` of equal bits differ).
fn const_key(c: &Const) -> (u8, i128) {
    match *c {
        Const::Int(v) => (0, v),
        Const::F16(b) => (1, i128::from(b)),
        Const::BF16(b) => (2, i128::from(b)),
        Const::F32(b) => (3, i128::from(b)),
        Const::F64(b) => (4, i128::from(b)),
    }
}

struct Resolve<'a> {
    vals: &'a HashMap<u32, Value>,
    labels: &'a HashMap<u32, u32>,
    fids: &'a HashMap<InstanceId, FuncId>,
    inst: &'a InstanceId,
}

impl Resolve<'_> {
    fn operand(&self, e: &mut Errs, line: u32, a: &Arg, pool: &mut Pool) -> Operand {
        match &a.v {
            ArgV::Val(v) => match self.vals.get(v) {
                Some(&x) => Operand::Value(x),
                None => {
                    e.at(line, a.col, format!("%{v} is not a value of fn @{}", self.inst));
                    Operand::Value(Value(u32::MAX))
                }
            },
            ArgV::Const(c) => Operand::Const(pool.id(*c)),
            ArgV::Str(s) => Operand::Str(StrId(*s)),
            ArgV::Func(inst) => match self.fids.get(inst) {
                Some(&f) => Operand::Func(f),
                None => {
                    e.at(line, a.col, format!("@{inst} heads no fn"));
                    Operand::Func(FuncId(u32::MAX))
                }
            },
            ArgV::Type(t) => Operand::Type(*t),
        }
    }

    fn target(&self, e: &mut Errs, line: u32, t: &Tgt, body: &mut Body, pool: &mut Pool) -> Target {
        let block = match self.labels.get(&t.label) {
            Some(&k) => Block(k),
            None => {
                e.at(line, t.col, format!("b{} is not a block of fn @{}", t.label, self.inst));
                Block(u32::MAX)
            }
        };
        let start = body.operands.len() as u32;
        for a in &t.args {
            let o = self.operand(e, line, a, pool);
            body.operands.push(o);
        }
        Target { block, args: Span { start, len: body.operands.len() as u32 - start } }
    }

    fn term(&self, e: &mut Errs, line: u32, t: &TermL, body: &mut Body, pool: &mut Pool) -> Term {
        match t {
            TermL::Br(t) => Term::Br(self.target(e, line, t, body, pool)),
            TermL::Cbr(c, a, b) => {
                let cond = self.operand(e, line, c, pool);
                let then = self.target(e, line, a, body, pool);
                Term::Cbr { cond, then, els: self.target(e, line, b, body, pool) }
            }
            TermL::Switch(s, cases, d) => {
                let scrut = self.operand(e, line, s, pool);
                let mut new = Vec::with_capacity(cases.len());
                for (v, t) in cases {
                    new.push(SwitchCase { value: *v, target: self.target(e, line, t, body, pool) });
                }
                let default = self.target(e, line, d, body, pool);
                let span = Span { start: body.cases.len() as u32, len: new.len() as u32 };
                body.cases.extend(new);
                Term::Switch { scrut, cases: span, default }
            }
            TermL::Ret(o) => Term::Ret(o.as_ref().map(|o| self.operand(e, line, o, pool))),
            TermL::Unreachable => Term::Unreachable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_outside_the_table_has_no_id() {
        let table: Vec<String> = ["copy_bytes", "cpuid", "syscall1"].map(String::from).to_vec();
        assert_eq!(intrinsic_id(&table, "cpuid"), Some(crate::ir::IntrinsicId(1)));
        for name in ["", "copy", "copy_bytes ", "syscall0", "zz", "Syscall1"] {
            assert_eq!(intrinsic_id(&table, name), None, "{name:?}");
        }
        assert_eq!(intrinsic_id(&[], "cpuid"), None);
    }

    #[test]
    fn constants_differ_by_kind_and_bits() {
        let mut p = Pool::default();
        let ids = [
            Const::Int(0),
            Const::F16(0),
            Const::BF16(0),
            Const::F32(0),
            Const::F64(0),
            Const::F32(0x8000_0000),
            Const::Int(-1),
            Const::F64(0x8000_0000_0000_0000),
            Const::Int(i128::MIN),
            Const::Int(i128::MAX),
        ]
        .map(|c| p.id(c));
        assert_eq!(ids.map(|c| c.0), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(p.id(Const::F32(0)), ConstId(3));
        assert_eq!(p.id(Const::Int(i128::MIN)), ConstId(8));
        assert_eq!(p.consts.len(), 10);
    }
}
