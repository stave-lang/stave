//! `stave dup` (contracts/cli.md): the classes of source fns whose code is the same, and the
//! constants declared again; the baseline of the `dup-gate` step (contracts/cli.md `dup-gate`).
//!
//! **What is compared.** Every fn instance of the call graph with a checked body is lowered as
//! every build lowers it ([`stave_sema::lower_graph_nodes`]), verified, and made canonical: the
//! level `lowered`, before the removal of proven checks and before any optimization, so a fn's
//! form depends on its own source, its signature, and the identities of what it names, never on
//! what an analysis of other fns proved. Each instance gets its duplicate form
//! ([`stave_ir::form`]), with each callee spelled by its class of equal code
//! ([`stave_ir::form::same_code`], every instance with a body taking part); closures are part of
//! their fn's form, never compared alone.
//!
//! **Classes.** Instances with equal forms make a class when they come from at least two source
//! fns (the instances of one generic fn are equal by construction and never a class by
//! themselves). A generic fn is compared per instance: the class lists the instances that are
//! equal, which is exactly true of the code the program builds; two generic fns that differ at
//! a type args tuple both are built with are left out of each other's classes (their equal
//! instances are a coincidence of those args, `size_of::<T>()` against `align_of::<T>()` at
//! `u32`). Classes with the same source fns merge, listing every instance. A second relation
//! compares methods up to `Self` ([`stave_ir::form::SelfTypes`]): such a class needs two
//! different `Self` types among its members. Fns below `min_size` (instructions plus blocks),
//! `test fn`s, drop glue, and fns the target does not build are not compared. A class is kept
//! when one member is outside the built-in packages, or with `all`.
//!
//! **Constants.** `const` items with one name, type, and value in more than one module; and the
//! aliases, `const X: T = m.Y;` (an optional `as`-family cast), grouped by the const they name.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use rayon::prelude::*;
use stave_ir::form::{Callees, MethodOf, SelfTypes, form, same_code};
use stave_ir::{FuncId, TypeId, TypeKey};
use stave_sema::callgraph::NodeKind;
use stave_sema::consteval::Val;
use stave_sema::sigs::SelfKind;
use stave_sema::{DeclFlags, DeclId, DeclKind, PackageKind, Program, Target, Typeck};
use stave_syntax::Term;

use crate::build::{self, Failure};
use crate::pipeline::{Output, Stage};

/// The default `--min-size`: instructions plus blocks of the canonical body. On the shipped trees
/// every class below it is a one-step accessor or forward (one field load, one call, or one
/// constant, then `ret`: sizes 1 to 4), which repeats per type and is cheaper to write than to
/// look up; from 5 on, each class has logic of its own (a bounds-checked index, a compare).
pub const DEFAULT_MIN_SIZE: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Keep the classes and groups whose members are all in built-in packages.
    pub all: bool,
    pub min_size: u32,
}

impl Default for Options {
    fn default() -> Options {
        Options { all: false, min_size: DEFAULT_MIN_SIZE }
    }
}

/// Which relation a class is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// Equal forms.
    Exact,
    /// Methods of different `Self` types, equal forms up to `Self`.
    UpToSelf,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Exact => "exact",
            Kind::UpToSelf => "self",
        }
    }

    fn parse(s: &str) -> Option<Kind> {
        [Kind::Exact, Kind::UpToSelf].into_iter().find(|k| k.name() == s)
    }
}

/// A source fn of a class.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Member {
    /// The checked-IR path of the source fn (`a.b.f`, a method `a.b.T.m`).
    pub path: String,
    /// A generic fn: the instances of the class (`a.b.f<u32>`); else empty.
    pub instances: Vec<String>,
    /// The file as diagnostics print it inside the root package, else `<package>/<path>`.
    pub file: String,
    pub line: u32,
    /// In the order `pub`, `trait-impl`, `export`, `known`.
    pub marks: Vec<&'static str>,
}

impl Member {
    /// `pub` and none of the marks of a fn that must exist where it is.
    fn callable(&self) -> bool {
        self.marks.first() == Some(&"pub") && self.marks.len() == 1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub kind: Kind,
    pub size: u32,
    /// Sorted by (path, instances).
    pub members: Vec<Member>,
}

impl Class {
    /// What one definition for all members saves: the size of every member but one.
    pub fn saving(&self) -> u64 {
        u64::from(self.size) * (self.members.len() as u64).saturating_sub(1)
    }
}

/// A declared constant.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstSite {
    pub path: String,
    pub file: String,
    pub line: u32,
}

/// One name, type, and value declared in more than one module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstGroup {
    pub name: String,
    pub ty: String,
    pub value: String,
    pub members: Vec<ConstSite>,
}

/// The consts declared as another const (`const X: T = m.Y;`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasGroup {
    /// The path of the const they name.
    pub of: String,
    pub members: Vec<ConstSite>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Fn instances compared (at least `min_size`, not `test fn`s).
    pub compared: usize,
    pub min_size: u32,
    /// Largest saving first, then kind, then members.
    pub classes: Vec<Class>,
    pub consts: Vec<ConstGroup>,
    pub aliases: Vec<AliasGroup>,
}

/// One fn instance with a body.
struct Unit {
    func: FuncId,
    decl: DeclId,
    node: u32,
    builtin: bool,
    test: bool,
    self_ty: Option<TypeId>,
}

/// The report of a checked program without error diagnostics.
pub fn report(out: &Output, opts: &Options) -> Result<Report, Failure> {
    let prog = &out.program;
    let Some(tc) = out.typeck.as_ref() else {
        return Err(Failure::Internal {
            stage: Stage::Lower,
            message: "dup: no type-checked program".into(),
        });
    };
    let g = &tc.call_graph;
    let nodes: Vec<u32> = (0..g.nodes.len() as u32)
        .filter(|&n| matches!(g.nodes[n as usize].kind, NodeKind::Fn { body: true, .. }))
        .collect();
    let drops = out.borrowck.as_ref().map(|b| &b.schedules);
    let (r, _) = stave_sema::lower_graph_nodes(prog, tc, &nodes, drops);
    let mut gl = r.map_err(|es| build::lower_failure(&es))?;
    build::verified(&gl.module, Stage::Lower)?;
    let remap = build::canonical(&mut gl.module);
    stave_ir::canon::permute_by_func(&mut gl.nodes_of_func, &remap);
    let m = &gl.module;

    let builtin_decl = |d: DeclId| -> bool {
        prog.decl(d)
            .and_then(|x| prog.module(x.module))
            .and_then(|x| prog.packages.get(x.package.idx()))
            .is_none_or(|p| p.kind == PackageKind::Builtin)
    };
    let mut units: Vec<Unit> = Vec::new();
    let mut methods: Vec<Option<MethodOf>> = vec![None; m.funcs.len()];
    for (i, f) in m.funcs.iter().enumerate() {
        if f.instance.0.len() != 1 || f.unbuilt {
            continue;
        }
        let fn_node = gl.nodes_of_func.get(i).and_then(|ns| {
            ns.iter().copied().find_map(|n| match &g.nodes[n as usize].kind {
                NodeKind::Fn { inst, .. } => Some((n, inst.decl)),
                _ => None,
            })
        });
        let Some((node, decl)) = fn_node else { continue };
        let Some(d) = prog.decl(decl) else { continue };
        let self_ty = self_type(prog, tc, m, f, decl);
        if let Some(t) = self_ty {
            let impl_args = tc.sigs.fn_sig(decl).map_or(0, |s| s.n_parent as usize);
            methods[i] = method_label(prog, tc, decl).map(|label| MethodOf { self_ty: t, label, impl_args });
        }
        if f.body.is_none() {
            continue;
        }
        units.push(Unit {
            func: FuncId(i as u32),
            decl,
            node,
            builtin: builtin_decl(decl),
            test: d.flags.has(DeclFlags::TEST),
            self_ty,
        });
    }

    let mut self_tys: Vec<TypeId> = units.iter().filter_map(|u| u.self_ty).collect();
    self_tys.sort_unstable();
    self_tys.dedup();
    let views: HashMap<TypeId, SelfTypes> = self_tys.par_iter().map(|&t| (t, SelfTypes::new(m, t))).collect();
    let funcs: Vec<FuncId> = units.iter().map(|u| u.func).collect();
    let same = same_code(m, &funcs, &methods, |fs, one| fs.par_iter().map(|&f| one(f)).collect());
    let callees = Callees { methods, same };
    let forms: Vec<(Option<stave_ir::form::Form>, Option<String>)> = units
        .par_iter()
        .map(|u| {
            let exact = form(m, u.func, None, &callees);
            let up = u.self_ty.and_then(|t| form(m, u.func, views.get(&t), &callees)).map(|x| x.text);
            (exact, up)
        })
        .collect();

    let kept: Vec<usize> = (0..units.len())
        .filter(|&i| !units[i].test && forms[i].0.as_ref().is_some_and(|x| x.size >= opts.min_size))
        .collect();
    let mut exact: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut up: HashMap<&str, Vec<usize>> = HashMap::new();
    for &i in &kept {
        if let Some(x) = &forms[i].0 {
            exact.entry(x.text.as_str()).or_default().push(i);
        }
        if let Some(t) = &forms[i].1 {
            up.entry(t.as_str()).or_default().push(i);
        }
    }
    // The instances of each generic source fn, by type args.
    let mut instances: HashMap<DeclId, Vec<(&[stave_ir::GenericArg], usize)>> = HashMap::new();
    for (i, u) in units.iter().enumerate() {
        if let Some(f) = m.func(u.func).filter(|f| !f.type_args.is_empty()) {
            instances.entry(u.decl).or_default().push((f.type_args.as_slice(), i));
        }
    }
    let text_of = |kind: Kind, i: usize| match kind {
        Kind::Exact => forms[i].0.as_ref().map(|x| x.text.as_str()),
        Kind::UpToSelf => forms[i].1.as_deref(),
    };
    // Two generic source fns that differ at a type args tuple both are built with: their equal
    // instances are not the same source fn twice.
    let conflict = |kind: Kind, a: DeclId, b: DeclId| -> bool {
        let (Some(xa), Some(xb)) = (instances.get(&a), instances.get(&b)) else { return false };
        xa.iter().any(|(args, i)| {
            xb.iter().any(|(bargs, j)| args == bargs && text_of(kind, *i) != text_of(kind, *j))
        })
    };
    let names = Names { prog, out, tc, m, units: &units };
    // (kind, source fns) -> (size, units)
    let mut found: HashMap<(Kind, Vec<DeclId>), (u32, Vec<usize>)> = HashMap::new();
    for (kind, groups) in [(Kind::Exact, exact), (Kind::UpToSelf, up)] {
        for (_, mut us) in groups {
            let decls: HashSet<DeclId> = us.iter().map(|&i| units[i].decl).collect();
            let dropped: HashSet<DeclId> = decls
                .iter()
                .copied()
                .filter(|&a| decls.iter().any(|&b| a != b && conflict(kind, a, b)))
                .collect();
            us.retain(|&i| !dropped.contains(&units[i].decl));
            let mut ds: Vec<DeclId> = us.iter().map(|&i| units[i].decl).collect();
            ds.sort_unstable();
            ds.dedup();
            if ds.len() < 2 {
                continue;
            }
            if kind == Kind::UpToSelf {
                let selfs: HashSet<TypeId> = us.iter().filter_map(|&i| units[i].self_ty).collect();
                if selfs.len() < 2 {
                    continue;
                }
            }
            if !opts.all && us.iter().all(|&i| units[i].builtin) {
                continue;
            }
            let size = forms[us[0]].0.as_ref().map_or(0, |x| x.size);
            let slot = found.entry((kind, ds)).or_insert((0, Vec::new()));
            slot.0 = slot.0.max(size);
            slot.1.extend(us);
        }
    }
    let mut classes: Vec<Class> = Vec::with_capacity(found.len());
    for ((kind, _), (size, us)) in found {
        classes.push(Class { kind, size, members: names.members(&us)? });
    }
    classes.sort_by(|a, b| {
        b.saving().cmp(&a.saving()).then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.members.cmp(&b.members))
    });
    let (consts, aliases) = consts(out, tc, opts)?;
    Ok(Report { compared: kept.len(), min_size: opts.min_size, classes, consts, aliases })
}

/// The `Self` type of a method instance, read off its signature where the declared signature
/// has the impl's self type: the receiver (`self`, or the referent of `&self` / `&mut self`),
/// else a return type that is the self type. `None`: not a method, or no such position.
fn self_type(
    prog: &Program,
    tc: &Typeck,
    m: &stave_ir::Module,
    f: &stave_ir::Func,
    decl: DeclId,
) -> Option<TypeId> {
    let d = prog.decl(decl)?;
    if d.kind != DeclKind::Method {
        return None;
    }
    let sig = tc.sigs.fn_sig(decl)?;
    match sig.self_kind {
        Some(SelfKind::Value) => f.params.first().copied(),
        Some(SelfKind::Ref | SelfKind::RefMut) => match m.types.key(*f.params.first()?)? {
            TypeKey::Ref { inner, .. } => Some(*inner),
            _ => None,
        },
        None => {
            let imp = tc.sigs.impl_def(d.parent)?;
            if sig.ret == imp.self_ty { f.ret } else { None }
        }
    }
}

/// `<trait path>.<name>` of a trait impl's method, `-.<name>` of an inherent one; `None` for a
/// method of an impl of a trait with args (`impl Tr<u32> for T`), whose path alone does not tell
/// the impls of one type apart.
fn method_label(prog: &Program, tc: &Typeck, decl: DeclId) -> Option<String> {
    let d = prog.decl(decl)?;
    let imp = tc.sigs.impl_def(d.parent)?;
    let name = prog.strings.get(d.name);
    match &imp.trait_ref {
        None => Some(format!("-.{name}")),
        Some(t) if t.args.is_empty() => Some(format!("{}.{name}", prog.decl_path(t.trait_decl))),
        Some(_) => None,
    }
}

/// The file name of `span` as the reports print it, and its line.
fn place(out: &Output, span: stave_diag::Span) -> Result<(String, u32), Failure> {
    let prog = &out.program;
    let loc = out
        .files()
        .locate(span)
        .map_err(|e| Failure::Internal { stage: Stage::Lower, message: format!("dup: {e}") })?;
    let file = prog.file(span.file).map_or(String::new(), |f| {
        if f.display_path.starts_with('/') {
            let pkg = prog.packages.get(f.package.0 as usize).map_or("", |p| p.name.as_str());
            format!("{pkg}/{}", f.rel_path)
        } else {
            f.display_path.clone()
        }
    });
    Ok((file, loc.line))
}

struct Names<'a> {
    prog: &'a Program,
    out: &'a Output,
    tc: &'a Typeck,
    m: &'a stave_ir::Module,
    units: &'a [Unit],
}

impl Names<'_> {
    /// The members of a class of units: one per source fn.
    fn members(&self, us: &[usize]) -> Result<Vec<Member>, Failure> {
        let g = &self.tc.call_graph;
        let mut by_decl: HashMap<DeclId, Vec<usize>> = HashMap::new();
        for &i in us {
            by_decl.entry(self.units[i].decl).or_default().push(i);
        }
        let mut members = Vec::with_capacity(by_decl.len());
        for (decl, is) in by_decl {
            let Some(d) = self.prog.decl(decl) else { continue };
            let first = &self.units[is[0]];
            let path = self.m.func(first.func).map_or(String::new(), |f| f.path.clone());
            let generic = self.m.func(first.func).is_some_and(|f| !f.type_args.is_empty());
            let mut instances = if generic {
                let ns: Vec<u32> = is.iter().map(|&i| self.units[i].node).collect();
                g.source_names(self.prog, self.tc, &ns)
            } else {
                Vec::new()
            };
            instances.sort();
            instances.dedup();
            let (file, line) = place(self.out, d.name_span)?;
            let mut marks = Vec::new();
            if d.is_pub() {
                marks.push("pub");
            }
            if d.flags.has(DeclFlags::TRAIT_IMPL) {
                marks.push("trait-impl");
            }
            if d.flags.has(DeclFlags::EXTERN_EXPORT) {
                marks.push("export");
            }
            if self.m.func(first.func).is_some_and(|f| f.known.is_some()) {
                marks.push("known");
            }
            members.push(Member { path, instances, file, line, marks });
        }
        members.sort();
        Ok(members)
    }
}

/// The surface spelling of types for the const report.
struct TyNames<'a>(&'a Program);

impl stave_sema::types::Names for TyNames<'_> {
    fn adt_name(&self, d: DeclId) -> String {
        self.0.decl_path(d)
    }

    fn param_name(&self, i: u32) -> String {
        format!("T{i}")
    }
}

/// The text of a const value, without recursion: ints in decimal, floats by their bits.
fn value_text(v: &Val) -> String {
    enum Step<'a> {
        Val(&'a Val),
        Text(&'static str),
    }
    let mut s = String::new();
    let mut stack = vec![Step::Val(v)];
    while let Some(step) = stack.pop() {
        let v = match step {
            Step::Text(t) => {
                s.push_str(t);
                continue;
            }
            Step::Val(v) => v,
        };
        match v {
            Val::Int(x) => {
                let _ = write!(s, "{x}");
            }
            Val::Float(x) => {
                let _ = write!(s, "f{:016x}", x.to_bits());
            }
            Val::Bool(b) => {
                let _ = write!(s, "{b}");
            }
            Val::Char(c) => {
                let _ = write!(s, "'\\u{{{c:x}}}'");
            }
            Val::Str(b) => {
                s.push_str(&stave_diag::json_string(&String::from_utf8_lossy(b)));
            }
            Val::Rep(x, n) => {
                let _ = write!(s, "[{n} x ");
                stack.push(Step::Text("]"));
                stack.push(Step::Val(x));
            }
            Val::Agg(xs) | Val::Variant(_, xs) => {
                if let Val::Variant(k, _) = v {
                    let _ = write!(s, "#{k}");
                }
                s.push('(');
                stack.push(Step::Text(")"));
                for (i, x) in xs.iter().enumerate().rev() {
                    stack.push(Step::Val(x));
                    if i > 0 {
                        stack.push(Step::Text(", "));
                    }
                }
            }
        }
    }
    s
}

/// The const that const `d` is declared as: its initializer is a path to a const, with an
/// optional `as`-family cast after it.
fn alias_of(prog: &Program, tc: &Typeck, d: DeclId) -> Option<DeclId> {
    let decl = prog.decl(d)?;
    let file = prog.file(decl.name_span.file)?;
    let toks = &file.parsed.tokens;
    let name = toks.partition_point(|t| t.start < decl.name_span.offset);
    let is = |i: usize, t: Term| toks.get(i).is_some_and(|x| x.kind == t as u8);
    let mut i = name;
    while i < toks.len() && !is(i, Term::Eq) {
        if is(i, Term::Semi) {
            return None;
        }
        i += 1;
    }
    let lo = i + 1;
    let mut j = lo;
    loop {
        if !is(j, Term::Ident) {
            return None;
        }
        j += 1;
        if !is(j, Term::Dot) {
            break;
        }
        j += 1;
    }
    let hi = j;
    let cast = [Term::KwAs, Term::AsPercent, Term::AsPipe, Term::AsTilde, Term::AsQuestion];
    if !is(j, Term::Semi) && !cast.iter().any(|&c| is(j, c)) {
        return None;
    }
    if hi - lo < 3 {
        return None;
    }
    let body = tc.body(d)?;
    body.targets.iter().find_map(|(tok, t)| match t {
        Target::Global { decl: s }
            if (lo as u32..hi as u32).contains(tok)
                && prog.decl(*s).is_some_and(|x| x.kind == DeclKind::Const) =>
        {
            Some(*s)
        }
        _ => None,
    })
}

type ConstKey = (String, stave_sema::TypeId, String);

fn consts(out: &Output, tc: &Typeck, opts: &Options) -> Result<(Vec<ConstGroup>, Vec<AliasGroup>), Failure> {
    let prog = &out.program;
    let mut by_value: HashMap<ConstKey, Vec<DeclId>> = HashMap::new();
    let mut by_source: HashMap<DeclId, Vec<DeclId>> = HashMap::new();
    let mut builtin = vec![false; prog.decls.len()];
    for (k, d) in prog.decls.iter().enumerate() {
        if d.kind != DeclKind::Const {
            continue;
        }
        let id = DeclId(k as u32);
        builtin[k] = prog
            .module(d.module)
            .and_then(|x| prog.packages.get(x.package.idx()))
            .is_none_or(|p| p.kind == PackageKind::Builtin);
        let Some(c) = tc.sigs.konst(id) else { continue };
        let Some(v) = &c.value else { continue };
        let key = (prog.strings.get(d.name).to_string(), c.ty, value_text(v));
        by_value.entry(key).or_default().push(id);
        if let Some(s) = alias_of(prog, tc, id) {
            by_source.entry(s).or_default().push(id);
        }
    }
    let keep = |ds: &[DeclId]| opts.all || ds.iter().any(|d| !builtin[d.idx()]);
    let site = |d: DeclId| -> Result<ConstSite, Failure> {
        let decl = prog.decl(d).copied().ok_or_else(|| Failure::Internal {
            stage: Stage::Lower,
            message: "dup: a const without a declaration".into(),
        })?;
        let (file, line) = place(out, decl.name_span)?;
        Ok(ConstSite { path: prog.decl_path(d), file, line })
    };
    let sites = |ds: &[DeclId]| -> Result<Vec<ConstSite>, Failure> {
        let mut v = ds.iter().map(|&d| site(d)).collect::<Result<Vec<_>, _>>()?;
        v.sort();
        Ok(v)
    };
    let mut groups = Vec::new();
    for ((name, ty, value), ds) in by_value {
        if ds.len() < 2 || !keep(&ds) {
            continue;
        }
        let ty = stave_sema::types::display(&tc.types, &TyNames(prog), ty);
        groups.push(ConstGroup { name, ty, value, members: sites(&ds)? });
    }
    groups.sort_by(|a, b| {
        b.members.len().cmp(&a.members.len()).then_with(|| (&a.name, &a.members).cmp(&(&b.name, &b.members)))
    });
    let mut aliases = Vec::new();
    for (s, ds) in by_source {
        if !keep(&ds) {
            continue;
        }
        aliases.push(AliasGroup { of: prog.decl_path(s), members: sites(&ds)? });
    }
    aliases.sort_by(|a, b| b.members.len().cmp(&a.members.len()).then_with(|| a.of.cmp(&b.of)));
    Ok((groups, aliases))
}

fn member_name(m: &Member) -> String {
    if m.instances.is_empty() { m.path.clone() } else { m.instances.join(", ") }
}

/// The human report (contracts/cli.md `stave dup`).
pub fn render_human(r: &Report) -> String {
    let mut s = String::new();
    for (i, c) in r.classes.iter().enumerate() {
        let what = match c.kind {
            Kind::Exact => "exact",
            Kind::UpToSelf => "up to Self",
        };
        let _ = writeln!(
            s,
            "class {}: {what}, size {}, {} fns, saves {}",
            i + 1,
            c.size,
            c.members.len(),
            c.saving()
        );
        for m in &c.members {
            let _ = write!(s, "  {}:{}: {}", m.file, m.line, member_name(m));
            if !m.marks.is_empty() {
                let _ = write!(s, " ({})", m.marks.join(", "));
            }
            s.push('\n');
        }
    }
    for g in &r.consts {
        let mut value = g.value.clone();
        if value.len() > 80 {
            let cut = (0..=77).rev().find(|&k| value.is_char_boundary(k)).unwrap_or(0);
            value.truncate(cut);
            value.push_str("...");
        }
        let _ = writeln!(s, "const {}: {} = {}: {} modules", g.name, g.ty, value, g.members.len());
        for m in &g.members {
            let _ = writeln!(s, "  {}:{}: {}", m.file, m.line, m.path);
        }
    }
    for a in &r.aliases {
        let _ = writeln!(s, "alias of {}: {}", a.of, a.members.len());
        for m in &a.members {
            let _ = writeln!(s, "  {}:{}: {}", m.file, m.line, m.path);
        }
    }
    let exact = r.classes.iter().filter(|c| c.kind == Kind::Exact).count();
    let n_alias: usize = r.aliases.iter().map(|a| a.members.len()).sum();
    let _ = writeln!(
        s,
        "dup: {} classes ({exact} exact, {} up to Self) over {} fns of size >= {}; {} const groups, {n_alias} aliases",
        r.classes.len(),
        r.classes.len() - exact,
        r.compared,
        r.min_size,
        r.consts.len()
    );
    s
}

fn json_list(xs: impl Iterator<Item = String>) -> String {
    let v: Vec<String> = xs.collect();
    format!("[{}]", v.join(","))
}

fn json_site(m: &ConstSite) -> String {
    let q = stave_diag::json_string;
    format!("{{\"const\":{},\"file\":{},\"line\":{}}}", q(&m.path), q(&m.file), m.line)
}

/// `--json`: one object per class, const group, and alias group, in the human order.
pub fn render_json(r: &Report) -> String {
    let q = stave_diag::json_string;
    let mut s = String::new();
    for c in &r.classes {
        let members = json_list(c.members.iter().map(|m| {
            format!(
                "{{\"fn\":{},\"instances\":{},\"file\":{},\"line\":{},\"marks\":{}}}",
                q(&m.path),
                json_list(m.instances.iter().map(|x| q(x))),
                q(&m.file),
                m.line,
                json_list(m.marks.iter().map(|x| q(x)))
            )
        }));
        let _ = writeln!(
            s,
            "{{\"kind\":\"class\",\"form\":\"{}\",\"size\":{},\"saving\":{},\"members\":{members}}}",
            c.kind.name(),
            c.size,
            c.saving()
        );
    }
    for g in &r.consts {
        let _ = writeln!(
            s,
            "{{\"kind\":\"const\",\"name\":{},\"type\":{},\"value\":{},\"members\":{}}}",
            q(&g.name),
            q(&g.ty),
            q(&g.value),
            json_list(g.members.iter().map(json_site))
        );
    }
    for a in &r.aliases {
        let _ = writeln!(
            s,
            "{{\"kind\":\"alias\",\"of\":{},\"members\":{}}}",
            q(&a.of),
            json_list(a.members.iter().map(json_site))
        );
    }
    s
}

/// The baseline header line.
const BASELINE_HEAD: &str =
    "# stave dup baseline: one class per line, `<form> <fn path>...` (contracts/cli.md `dup-gate`)";

/// The baseline of report `r`: one line per class, `<form> <path> <path>...`, the paths of its
/// source fns sorted, the lines sorted; no line numbers, sizes, or instances, so an edit
/// elsewhere does not change it.
pub fn baseline_text(r: &Report) -> String {
    let mut lines: Vec<String> = r
        .classes
        .iter()
        .map(|c| {
            let mut paths: Vec<&str> = c.members.iter().map(|m| m.path.as_str()).collect();
            paths.sort_unstable();
            paths.dedup();
            format!("{} {}", c.kind.name(), paths.join(" "))
        })
        .collect();
    lines.sort();
    lines.dedup();
    let mut s = String::from(BASELINE_HEAD);
    s.push('\n');
    for l in lines {
        s.push_str(&l);
        s.push('\n');
    }
    s
}

/// A parsed baseline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Baseline {
    pub classes: Vec<(Kind, Vec<String>)>,
}

/// Reads a baseline: `#` lines and empty lines are skipped; every other line is a form name
/// and at least two fn paths.
pub fn parse_baseline(text: &str) -> Result<Baseline, String> {
    let mut classes = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut words = l.split_whitespace();
        let kind = words
            .next()
            .and_then(Kind::parse)
            .ok_or_else(|| format!("line {}: expected `exact` or `self`", i + 1))?;
        let mut paths: Vec<String> = words.map(str::to_string).collect();
        if paths.len() < 2 {
            return Err(format!("line {}: a class has at least two fn paths", i + 1));
        }
        paths.sort();
        paths.dedup();
        classes.push((kind, paths));
    }
    Ok(Baseline { classes })
}

/// The gate's verdict on a report against a baseline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// One message per class no baseline class covers (exit 1 when not empty).
    pub new: Vec<String>,
    /// Baseline classes that share no two fns with a class of the report: their duplicate is gone
    /// (removing them is allowed).
    pub stale: usize,
}

/// A class passes when one baseline class of its form holds all its fn paths (a class that
/// lost members still passes; a new class, or a new member of an old one, does not). The
/// message names the new fns and the existing fn to call instead: the first member the baseline
/// already had, a callable one (`pub`, no mark of a fn that must exist where it is) first; else
/// the first callable member; else the first member.
pub fn check(r: &Report, b: &Baseline) -> Verdict {
    let mut used = vec![false; b.classes.len()];
    let mut new = Vec::new();
    for c in &r.classes {
        let mut paths: Vec<&str> = c.members.iter().map(|m| m.path.as_str()).collect();
        paths.sort_unstable();
        paths.dedup();
        let mut best: Option<(usize, usize)> = None;
        let mut covered = false;
        for (k, (kind, bp)) in b.classes.iter().enumerate() {
            if *kind != c.kind {
                continue;
            }
            let common = paths.iter().filter(|p| bp.binary_search_by(|x| x.as_str().cmp(p)).is_ok()).count();
            if common == paths.len() {
                covered = true;
            }
            if common >= 2 {
                used[k] = true;
            }
            if common > 0 && best.is_none_or(|(_, n)| common > n) {
                best = Some((k, common));
            }
        }
        if covered {
            continue;
        }
        let old = |m: &Member| {
            best.is_some_and(|(k, _)| b.classes[k].1.binary_search_by(|x| x.as_str().cmp(&m.path)).is_ok())
        };
        let existing = c
            .members
            .iter()
            .find(|m| old(m) && m.callable())
            .or_else(|| c.members.iter().find(|m| old(m)))
            .or_else(|| c.members.iter().find(|m| m.callable()))
            .unwrap_or(&c.members[0]);
        let how = match c.kind {
            Kind::Exact => "is the same code as",
            Kind::UpToSelf => "is the same code up to `Self` as",
        };
        let advice = match c.kind {
            Kind::Exact => "call it instead",
            Kind::UpToSelf => "write it once, generic over a trait the types implement",
        };
        for m in c.members.iter().filter(|m| m.path != existing.path && !old(m)) {
            new.push(format!(
                "{}:{}: new duplicate ({}, size {}): `{}` {how} `{}` ({}:{}): {advice}",
                m.file,
                m.line,
                c.kind.name(),
                c.size,
                member_name(m),
                member_name(existing),
                existing.file,
                existing.line
            ));
        }
    }
    let stale = used.iter().filter(|u| !**u).count();
    Verdict { new, stale }
}
