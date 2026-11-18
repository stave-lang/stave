//! Stave semantic analysis, stage 1: packages and modules, declarations and name
//! resolution, canonical ids.
//!
//! [`Program::load`] builds a whole program from a root package directory (or one `.stv` file):
//!
//! 1. packages: `stave.toml` (E1002), path and per-target dependencies, built-in packages and
//!    their optional manifests ([`package`]);
//! 2. module files are read (E1003 if unreadable), lexed, cut into items, and parsed
//!    (`stave-syntax`), each stage over all files in parallel ([`Timings::stages`]); a parse-only
//!    load ([`Config::parse_only`]) stops here;
//! 3. intern: every identifier, per file in parallel, then ranked canonically ([`Strings`]);
//! 4. modules and imports: import paths (E0101, E0107), the acyclic import graph (E0102)
//!    ([`modules`]);
//! 5. declarations ([`decls`]), module scopes (E0105), the prelude (the `pub` items of
//!    `std.core`, seen by modules of `std` and of every package depending on it; then the
//!    built-in type names and `size_of` / `align_of`, seen everywhere), impl headers and method
//!    tables;
//! 6. name resolution, one item at a time in parallel ([`resolve`]): E0101 E0103 E0105 E0106;
//! 7. canonical renumbering of decls.
//!
//! Every id the API exposes (files, items, strings, modules, packages, decls) is canonical, so
//! dumps (stage-gates `strings`, `names`) and later stages never see an insertion or thread
//! order. Diagnostics are in canonical order.
//!
//! Dumps (stage-gates): [`dump_strings`], [`dump_names`], [`dump_types`].
//!
//! Type checking: [`check`] ([`typeck`]), which also builds the whole-program call graph
//! ([`callgraph`]). Monomorphization and lowering to the checked IR:
//! [`lower()`] ([`lower`], [`mono`]).
//!
//! For the type checker: [`Program::decl`] (kind, name, node, flags), [`Program::members`]
//! (fields, variants, trait methods, impl methods in declaration order), [`Program::member`],
//! [`Program::impls`] / [`Program::impls_of`] (impl blocks per type head), [`Program::methods`]
//! (method tables), [`Program::uses`] (resolutions per node) and [`ItemRes::deferred`] (field and
//! method steps that need types).

pub mod builtins;
pub mod callgraph;
pub mod closures;
pub mod consteval;
pub mod contracts;
pub mod decls;
mod dispatch;
pub mod effects;
pub mod gates;
pub mod itemcx;
pub mod kinds;
pub mod literals;
pub mod lower;
mod lower_expr;
mod lower_fn;
mod lower_post;
mod lower_simd;
mod lower_stmt;
pub mod modules;
pub mod mono;
pub mod origins;
pub mod package;
pub mod patterns;
pub mod prim;
pub mod ranks;
pub mod resolve;
pub mod sigs;
mod threads;
pub mod traits;
pub mod typeck;
pub mod types;
pub mod unsafety;

pub mod expr;

mod canon;
mod diag;
mod dump;
mod ids;
mod intern;
mod pool;
mod recursion;
mod scan;
mod toml;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::{Diagnostic, FileId, SourceFiles, Span};
use stave_syntax::{ItemAst, ItemKind, NodeKind, ParsedFile, Term};
use stave_target::{Triple, X86Level};

pub use decls::{Decl, DeclFlags, DeclKind};
pub use diag::failures as diagnostic_build_failures;
pub use dump::{dump_names, dump_strings, names_file_into, strings_into};
pub use ids::{BindingRef, DeclId, GenericRef, ItemId, ModuleId, PackageId, Sym};
pub use intern::Strings;
pub use lower::{
    GraphLowering, LowerError, LowerErrorKind, LowerTimings, Mode, lower, lower_graph_nodes, lower_timed,
};
pub use modules::{Import, Module};
pub use package::{BuiltinPackage, FileBytes, Fs, Package, PackageKind};
pub use prim::{Builtin, Lanes, Prim, Scalar};
pub use resolve::{
    BindKind, Binding, DeferKind, Deferred, ImplInfo, ItemRes, MethodTable, Res, TypeHead, Use,
};
pub use typeck::{CallSite, Coercion, Instance, Target, Typeck, TypeckTimings, TypedBody, check, dump_types};
pub use types::{TypeId, TypeKind, TypeTable};

use crate::diag::mk;
use crate::resolve::{ScopeEntry, Tables};

/// Environment variable: when set, [`Program::load`] prints the wall time of each resolve
/// sub-phase on stderr.
pub const PROFILE_ENV: &str = "STAVE_SEMA_PROFILE";

/// Environment variable of the shuffle test knob: a seed that perturbs every internal order
/// (string interning order, decl creation order). Canonical output must not change.
pub const SHUFFLE_ENV: &str = "STAVE_SEMA_SHUFFLE";

/// What to build.
#[derive(Clone, Debug)]
pub struct Config {
    /// A package directory (with `stave.toml`) or one `.stv` file.
    pub root: PathBuf,
    /// The build target (dependency selection). `None`: the host baseline.
    pub target: Option<Triple>,
    /// Built-in packages (`std`, `runtime`), supplied by the driver.
    pub builtins: Vec<BuiltinPackage>,
    /// The prelude module: its `pub` items (not its imports) are in scope,
    /// after the module's own names, in every module of that package and of every package
    /// depending on it: (package, path in it). Default `("std", ["core"])`. `Vec`, `String`, and
    /// the rest of std need an import.
    pub prelude: (String, Vec<String>),
    pub fs: Fs,
    /// Contents that stand in for source files: a `.stv` file whose [`File::abs_path`] is a key
    /// is not read from `fs`. The set of files and the manifests still come from `fs`.
    pub sources: std::collections::BTreeMap<PathBuf, Vec<u8>>,
    /// Shuffle test knob (see [`SHUFFLE_ENV`]); `None` reads the environment.
    pub shuffle_seed: Option<u64>,
    /// Parse only (`//~ PARSE-ONLY`, see [`Config::parse_only`]).
    pub parse_only: bool,
}

impl Config {
    pub fn new(root: impl Into<PathBuf>) -> Config {
        Config {
            root: root.into(),
            target: None,
            builtins: Vec::new(),
            prelude: ("std".into(), vec!["core".into()]),
            fs: Fs::Disk,
            sources: std::collections::BTreeMap::new(),
            shuffle_seed: None,
            parse_only: false,
        }
    }

    /// Parse only (the `//~ PARSE-ONLY` conformance annotation): [`Program::load`] reads, lexes,
    /// cuts, and parses the root package's files (and its path dependencies'), with no built-in
    /// packages (`builtins` is ignored), and stops: no interning, modules, decls, or resolution.
    /// The program has packages (no root modules), files, items (no modules or decls), the source
    /// map, and the read, manifest, lexical, and syntax diagnostics; the other tables are empty.
    pub fn parse_only(mut self) -> Config {
        self.parse_only = true;
        self
    }

    /// Adds `dir` as the built-in `std` package (implicit dependency of every package).
    pub fn with_std(mut self, dir: impl Into<PathBuf>) -> Config {
        self.builtins.push(BuiltinPackage {
            name: "std".into(),
            dir: dir.into(),
            deps: Vec::new(),
            implicit: true,
        });
        self
    }

    /// Adds the shipped `std` (depending on `runtime`, implicit for every package) and `runtime`
    /// (its own manifest binds the per-arch `arch` package) built-in packages.
    pub fn with_shipped(mut self, std_dir: impl Into<PathBuf>, runtime_dir: impl Into<PathBuf>) -> Config {
        self.builtins.push(BuiltinPackage {
            name: "std".into(),
            dir: std_dir.into(),
            deps: vec!["runtime".into()],
            implicit: true,
        });
        self.builtins.push(BuiltinPackage {
            name: "runtime".into(),
            dir: runtime_dir.into(),
            deps: Vec::new(),
            implicit: false,
        });
        self
    }
}

/// A source file (one module).
#[derive(Debug)]
pub struct File {
    pub id: FileId,
    pub package: PackageId,
    pub module: ModuleId,
    /// Relative to the package directory (`src/a/b.stv`); with the package name, the canonical
    /// file key.
    pub rel_path: String,
    /// Path in diagnostics (relative to the root package directory when inside it).
    pub display_path: String,
    pub abs_path: PathBuf,
    pub bytes: FileBytes,
    /// XXH3-128 of the bytes.
    pub hash: [u64; 2],
    pub parsed: ParsedFile,
    /// Per token: the string id of an `Ident` token, `Sym::NONE` otherwise.
    pub syms: Vec<Sym>,
    /// Id of the file's first item (items are numbered file after file).
    pub first_item: u32,
}

/// An item (stage-gates `I`): an entry of a file's item cut, dropped ones included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub file: FileId,
    /// Index in the file's `ParsedFile::items`.
    pub index: u32,
    pub module: ModuleId,
    pub kind: ItemKind,
    pub ok: bool,
    /// The item's main decl (fn, ADT, trait, impl, const, alias; `NONE` for imports, extern
    /// blocks, allocators, and dropped items).
    pub decl: DeclId,
}

/// Wall time per phase. Read, lex, items, and parse each run over all files in parallel, one
/// after the other, so each has its own wall time ([`Timings::stages`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timings {
    /// Manifests, file discovery.
    pub load: Duration,
    /// Read + lex + items + parse, and the line index of the file table (the front end total).
    pub parse: Duration,
    /// Interning (with canonical string ranking).
    pub intern: Duration,
    /// Modules, imports, decls, scopes, resolution, canonical decl ids.
    pub resolve: Duration,
    /// Reading the module files (and their hashes).
    pub read: Duration,
    /// Lexing.
    pub lex: Duration,
    /// Cutting the token streams into items.
    pub items: Duration,
    /// Parsing the items (the `parse` stage proper; [`Timings::parse`] is the front-end total).
    pub parse_items: Duration,
}

impl Timings {
    /// The loader's stages under their diagnostics-schema names (`timings[].stage`), in pipeline
    /// order: `read` (manifests, discovery, and file reads), `lex`, `items`, `parse`, `intern`,
    /// `resolve`. A parse-only load reports zero for `intern` and `resolve`.
    pub fn stages(&self) -> [(&'static str, Duration); 6] {
        [
            ("read", self.load + self.read),
            ("lex", self.lex),
            ("items", self.items),
            ("parse", self.parse_items),
            ("intern", self.intern),
            ("resolve", self.resolve),
        ]
    }
}

/// A whole program after name resolution.
#[derive(Debug)]
pub struct Program {
    pub packages: Vec<Package>,
    /// By [`FileId`] (canonical: rank of (package name, relative path)).
    pub files: Vec<File>,
    /// The file table for diagnostics: the source files (same ids), then the manifests.
    pub source_map: SourceFiles,
    pub modules: Vec<Module>,
    pub items: Vec<Item>,
    pub strings: Strings,
    pub decls: Vec<Decl>,
    /// Per item.
    pub res: Vec<ItemRes>,
    pub impls: Vec<ImplInfo>,
    pub methods: MethodTable,
    /// The prelude: (name, decl) of each `pub` item of the prelude module, sorted by name id.
    pub prelude: Vec<(Sym, DeclId)>,
    /// Errors in canonical order.
    pub diagnostics: Vec<Diagnostic>,
    pub timings: Timings,
    /// The build target (dependency selection, intrinsic arch checks).
    pub target: Triple,
    member_list: Vec<DeclId>,
    member_range: Vec<(u32, u32)>,
    member_by_name: FxHashMap<(DeclId, Sym), DeclId>,
    impls_by_head: FxHashMap<TypeHead, Vec<u32>>,
    module_items: Vec<FxHashMap<Sym, DeclId>>,
}

fn default_target() -> Triple {
    Triple::host().unwrap_or(Triple::X86_64(X86Level::V2))
}

/// A file during loading (before ids are final).
struct Loading {
    package: PackageId,
    pf: package::PendingFile,
}

impl Program {
    pub fn decl(&self, d: DeclId) -> Option<&Decl> {
        self.decls.get(d.idx())
    }

    pub fn item(&self, i: ItemId) -> Option<&Item> {
        self.items.get(i.idx())
    }

    pub fn file(&self, f: FileId) -> Option<&File> {
        self.files.get(f.0 as usize)
    }

    pub fn module(&self, m: ModuleId) -> Option<&Module> {
        self.modules.get(m.idx())
    }

    /// The AST of item `i` (empty for a dropped item).
    pub fn item_ast(&self, i: ItemId) -> ItemAst<'_> {
        match self.item(i).and_then(|it| Some((it, self.file(it.file)?))) {
            Some((it, f)) => f.parsed.item_ast(it.index as usize),
            None => ItemAst { nodes: &[] },
        }
    }

    /// Resolutions of item `i`.
    pub fn item_res(&self, i: ItemId) -> Option<&ItemRes> {
        self.res.get(i.idx())
    }

    /// The uses keyed at `node` of item `i`, in offset order (a chain: head first).
    pub fn uses(&self, i: ItemId, node: u32) -> &[Use] {
        self.res.get(i.idx()).map_or(&[], |r| r.uses_at(node))
    }

    /// Members of `d` in declaration order: struct/variant fields, enum variants, trait
    /// methods, impl methods.
    pub fn members(&self, d: DeclId) -> &[DeclId] {
        match self.member_range.get(d.idx()) {
            Some(&(lo, hi)) => &self.member_list[lo as usize..hi as usize],
            None => &[],
        }
    }

    /// The member of `d` named `name` (the first one if duplicated).
    pub fn member(&self, d: DeclId, name: Sym) -> Option<DeclId> {
        self.member_by_name.get(&(d, name)).copied()
    }

    /// Impl blocks whose self type has head `h`, in item order.
    pub fn impls_of(&self, h: TypeHead) -> impl Iterator<Item = &ImplInfo> {
        self.impls_by_head.get(&h).into_iter().flatten().filter_map(|&i| self.impls.get(i as usize))
    }

    /// The impl block of impl decl `d`.
    pub fn impl_info(&self, d: DeclId) -> Option<&ImplInfo> {
        let i = self.impls.partition_point(|x| x.decl < d);
        self.impls.get(i).filter(|x| x.decl == d)
    }

    /// An item of module `m` by name (imports excluded).
    pub fn module_item(&self, m: ModuleId, name: Sym) -> Option<DeclId> {
        self.module_items.get(m.idx())?.get(&name).copied()
    }

    /// Dotted path text of a decl (`a.b.S.f`; a module: its module path).
    pub fn decl_path(&self, d: DeclId) -> String {
        let Some(decl) = self.decl(d) else { return String::new() };
        if decl.kind == DeclKind::Module {
            return self.module(decl.module).map_or(String::new(), |m| m.path_text.clone());
        }
        let own = match decl.kind {
            DeclKind::Impl => "impl".to_string(),
            _ => self.strings.get(decl.name).to_string(),
        };
        let prefix = match decl.parent.get() {
            Some(p) => self.decl_path(p),
            None => self.module(decl.module).map_or(String::new(), |m| m.path_text.clone()),
        };
        if prefix.is_empty() { own } else { format!("{prefix}.{own}") }
    }

    /// Whether there are error diagnostics.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.severity() == stave_diag::Severity::Error)
    }

    /// The first use (in item order) of syntax that parses but has no semantics in stage0 yet
    /// ([`ItemRes::unimplemented`]), as `<what> at <path>:<line>:<col>`.
    pub fn unimplemented(&self) -> Option<String> {
        let (i, &(tok, what)) =
            self.res.iter().enumerate().find_map(|(i, r)| Some((i, r.unimplemented.as_ref()?)))?;
        let file = self.item(ItemId(i as u32)).and_then(|it| self.file(it.file))?;
        let span = scan::tok_span(file.id, &file.bytes, &file.parsed.tokens, tok as usize);
        Some(match self.source_map.locate(span) {
            Ok(l) => format!("{what} at {}:{}:{}", l.path, l.line, l.col),
            Err(_) => what.to_string(),
        })
    }

    /// Builds the program.
    pub fn load(cfg: &Config) -> Program {
        let shuffle =
            cfg.shuffle_seed.or_else(|| std::env::var(SHUFFLE_ENV).ok().and_then(|s| s.parse().ok()));
        let target = cfg.target.unwrap_or_else(default_target);
        let t0 = Instant::now();
        let builtins: &[BuiltinPackage] = if cfg.parse_only { &[] } else { &cfg.builtins };
        let loaded = package::load(&cfg.fs, &cfg.root, target, builtins);

        // Canonical package order: (name, directory).
        let mut porder: Vec<usize> = (0..loaded.packages.len()).collect();
        porder.sort_by(|&a, &b| {
            let (pa, pb) = (&loaded.packages[a], &loaded.packages[b]);
            (&pa.name, &pa.canonical).cmp(&(&pb.name, &pb.canonical))
        });
        let mut pnew = vec![PackageId::NONE; porder.len()];
        for (new, &old) in porder.iter().enumerate() {
            pnew[old] = PackageId(new as u32);
        }
        // Canonical file order: (package name, relative path).
        let mut pending: Vec<Loading> =
            loaded.files.iter().map(|pf| Loading { package: pnew[pf.package], pf: pf.clone() }).collect();
        pending.sort_by(|a, b| {
            let na = &loaded.packages[a.pf.package].name;
            let nb = &loaded.packages[b.pf.package].name;
            (na, &a.pf.rel_path, a.package).cmp(&(nb, &b.pf.rel_path, b.package))
        });
        let t_load = t0.elapsed();

        // Read, lex, cut, parse: each stage over every file in parallel (own wall times).
        let t1 = Instant::now();
        let fs = &cfg.fs;
        let reads: Vec<(FileBytes, [u64; 2], Option<&'static str>)> = pending
            .par_iter()
            .map(|l| {
                let (bytes, err) = match cfg.sources.get(&l.pf.abs_path) {
                    Some(text) => (FileBytes::Owned(text.clone()), None),
                    None => match fs.read(&l.pf.abs_path) {
                        Ok(b) => (b, None),
                        Err(e) => (FileBytes::Owned(Vec::new()), Some(read_reason(&e))),
                    },
                };
                let hash = stave_intern::hash::xxh3_128_parts(&bytes);
                (bytes, hash, err)
            })
            .collect();
        let t_read = t1.elapsed();
        let ts = Instant::now();
        let lexed: Vec<stave_syntax::Lexed> =
            reads.par_iter().enumerate().map(|(i, r)| stave_syntax::lex(&r.0, FileId(i as u32))).collect();
        let t_lex = ts.elapsed();
        let ts = Instant::now();
        let tables: Vec<Option<Result<stave_syntax::ItemTable, stave_syntax::ItemsError>>> = lexed
            .par_iter()
            .enumerate()
            .map(|(i, lx)| lx.stop.is_none().then(|| stave_syntax::items(&lx.tokens, FileId(i as u32))))
            .collect();
        let t_items = ts.elapsed();
        let ts = Instant::now();
        let parsed: Vec<ParsedFile> = reads
            .par_iter()
            .zip(lexed)
            .zip(tables)
            .enumerate()
            .map(|(i, ((r, lx), tb))| parse_cut(&r.0, FileId(i as u32), lx, tb))
            .collect();
        let t_parse_items = ts.elapsed();
        let mut read_errors: Vec<(FileId, &'static str)> = Vec::new();
        let mut files: Vec<File> = Vec::with_capacity(reads.len());
        for (i, ((bytes, hash, err), parsed)) in reads.into_iter().zip(parsed).enumerate() {
            let (id, l) = (FileId(i as u32), &pending[i]);
            if let Some(r) = err {
                read_errors.push((id, r));
            }
            files.push(File {
                id,
                package: l.package,
                module: ModuleId::NONE,
                rel_path: l.pf.rel_path.clone(),
                display_path: l.pf.display_path.clone(),
                abs_path: l.pf.abs_path.clone(),
                bytes,
                hash,
                parsed,
                syms: Vec::new(),
                first_item: 0,
            });
        }
        let mut diags: Vec<Diagnostic> = Vec::new();
        let mut source_map = SourceFiles::new();
        for f in &files {
            let pname = &loaded.packages[pending[f.id.0 as usize].pf.package].name;
            let _ = source_map.add(pname, &f.display_path, &f.bytes);
        }
        for f in &files {
            diags.extend(f.parsed.diagnostics.iter().cloned());
        }
        for &(fid, reason) in &read_errors {
            diags.extend(mk("E1003", None, &[("reason", reason)], Span::new(fid, 0, 0), &[]));
        }
        let t_parse = t1.elapsed();
        // Manifests (E1002), in package order.
        let mut manifest_ids: FxHashMap<usize, FileId> = FxHashMap::default();
        let mut morder: Vec<usize> = (0..loaded.manifests.len()).collect();
        morder.sort_by_key(|&m| pnew[loaded.manifests[m].package]);
        for &m in &morder {
            let mf = &loaded.manifests[m];
            let pname = &loaded.packages[mf.package].name;
            let Ok(fid) = source_map.add(pname, &mf.display_path, &mf.text) else { continue };
            manifest_ids.insert(m, fid);
            for is in &mf.issues {
                let span = Span::new(fid, is.span.lo, is.span.hi.saturating_sub(is.span.lo));
                let holes: Vec<(&str, &str)> = is.holes.iter().map(|(k, v)| (*k, v.as_str())).collect();
                diags.extend(mk("E1002", is.case, &holes, span, &[]));
            }
        }
        let mut timings = Timings {
            load: t_load,
            parse: t_parse,
            read: t_read,
            lex: t_lex,
            items: t_items,
            parse_items: t_parse_items,
            ..Timings::default()
        };
        if cfg.parse_only {
            let pk = (&loaded, &porder[..], &pnew[..], &manifest_ids);
            return Program::parsed_only(pk, files, source_map, diags, timings, target);
        }

        // Intern.
        let t2 = Instant::now();
        let pool = intern::Pool::new();
        let module_segs: Vec<Vec<String>> = pending.iter().map(|l| l.pf.module_rel.clone()).collect();
        if let Some(seed) = shuffle {
            let mut all: Vec<String> = Vec::new();
            for f in &files {
                for (i, t) in f.parsed.tokens.iter().enumerate() {
                    if t.kind == Term::Ident as u8 {
                        all.push(String::from_utf8_lossy(f.parsed.token_text(&f.bytes, i)).into_owned());
                    }
                }
            }
            all.sort();
            all.dedup();
            intern::shuffle(&mut all, seed);
            for s in &all {
                pool.intern(s);
            }
        }
        files.par_iter_mut().for_each(|f| intern_file(&pool, f));
        let pkg_syms: Vec<Sym> = loaded.packages.iter().map(|p| pool.intern(&p.name)).collect();
        let seg_syms: Vec<Vec<Sym>> =
            module_segs.iter().map(|segs| segs.iter().map(|s| pool.intern(s)).collect()).collect();
        let (strings, old_to_new) = pool.canonicalize();
        files.par_iter_mut().for_each(|f| intern::remap_syms(&old_to_new, &mut f.syms));
        let remap1 = |s: Sym| -> Sym {
            let mut v = [s];
            intern::remap_syms(&old_to_new, &mut v);
            v[0]
        };
        let pkg_syms: Vec<Sym> = pkg_syms.into_iter().map(remap1).collect();
        let seg_syms: Vec<Vec<Sym>> =
            seg_syms.into_iter().map(|v| v.into_iter().map(remap1).collect()).collect();
        let t_intern = t2.elapsed();

        // Modules: one per file, canonical by (module path text, package).
        let t3 = Instant::now();
        let mut tp = (Instant::now(), std::env::var(PROFILE_ENV).is_ok());
        let mut mods: Vec<Module> = Vec::with_capacity(files.len());
        for (i, f) in files.iter().enumerate() {
            let rp = &loaded.packages[pending[i].pf.package];
            let rel = seg_syms[i].clone();
            let psym = pkg_syms[pending[i].pf.package];
            let path: Vec<Sym> = match rp.kind {
                PackageKind::Builtin => std::iter::once(psym).chain(rel.iter().copied()).collect(),
                _ if rp.is_root => rel.clone(),
                _ => {
                    if Some(&pending[i].pf.module_rel) == rp.root_rel.as_ref() {
                        vec![psym]
                    } else {
                        std::iter::once(psym).chain(rel.iter().copied()).collect()
                    }
                }
            };
            let path_text = modules::path_text(&strings, &path);
            mods.push(Module {
                id: ModuleId::NONE,
                package: f.package,
                file: f.id,
                rel,
                path,
                path_text,
                decl: DeclId::NONE,
                imports: Vec::new(),
            });
        }
        let mut morder: Vec<usize> = (0..mods.len()).collect();
        morder.sort_by(|&a, &b| {
            (&mods[a].path_text, mods[a].package).cmp(&(&mods[b].path_text, mods[b].package))
        });
        let mut modules: Vec<Module> = Vec::with_capacity(mods.len());
        let mut mods_opt: Vec<Option<Module>> = mods.into_iter().map(Some).collect();
        for (new, &old) in morder.iter().enumerate() {
            if let Some(mut m) = mods_opt[old].take() {
                m.id = ModuleId(new as u32);
                files[old].module = m.id;
                modules.push(m);
            }
        }

        prof(&mut tp, "modules");
        // Packages (public form).
        let midx = modules::ModuleIndex::new(&modules);
        let mut packages: Vec<Package> = Vec::with_capacity(porder.len());
        for &old in &porder {
            let rp = &loaded.packages[old];
            let root_module = match &rp.root_rel {
                Some(rr) => {
                    let syms: Option<Vec<Sym>> = rr.iter().map(|s| strings.lookup(s)).collect();
                    syms.and_then(|s| midx.get(pnew[old], &s)).unwrap_or(ModuleId::NONE)
                }
                None => ModuleId::NONE,
            };
            let manifest = rp.manifest.and_then(|m| manifest_ids.get(&m).copied());
            let mut deps: Vec<(String, PackageId)> =
                rp.deps.iter().map(|(n, p)| (n.clone(), pnew[*p])).collect();
            deps.sort();
            packages.push(Package {
                name: rp.name.clone(),
                kind: rp.kind,
                version: rp.version.clone(),
                dir: rp.dir.clone(),
                manifest,
                deps,
                root_module,
                is_root: rp.is_root,
            });
        }

        // Items.
        let mut items: Vec<Item> = Vec::new();
        for f in &mut files {
            f.first_item = items.len() as u32;
            for (k, it) in f.parsed.items.iter().enumerate() {
                items.push(Item {
                    file: f.id,
                    index: k as u32,
                    module: f.module,
                    kind: it.kind,
                    ok: it.ok,
                    decl: DeclId::NONE,
                });
            }
        }

        prof(&mut tp, "packages+items");
        // Decls: module decls, then per-file collection.
        let mut decls: Vec<Decl> = Vec::new();
        for m in &mut modules {
            m.decl = DeclId(decls.len() as u32);
            decls.push(Decl {
                kind: DeclKind::Module,
                name: m.path.last().copied().unwrap_or(Sym::NONE),
                module: m.id,
                parent: DeclId::NONE,
                item: ItemId::NONE,
                node: u32::MAX,
                name_span: Span::new(m.file, 0, 0),
                flags: DeclFlags(DeclFlags::PUB),
                index: 0,
            });
        }
        let collected: Vec<decls::FileDecls> = files
            .par_iter()
            .map(|f| {
                let mp = &modules[f.module.idx()].path_text;
                decls::collect_file(f, f.module, mp, &strings)
            })
            .collect();
        let mut raw_impls: Vec<ImplInfo> = Vec::new();
        let mut raw_imports: Vec<Vec<decls::RawImport>> = vec![Vec::new(); modules.len()];
        let mut dropped: Vec<Vec<(Sym, Span)>> = vec![Vec::new(); modules.len()];
        let mut poisoned: FxHashSet<Sym> = FxHashSet::default();
        let mut allocators: Vec<(ItemId, u32, u32)> = Vec::new();
        let mut order: Vec<usize> = (0..collected.len()).collect();
        if shuffle.is_some() {
            order.reverse();
        }
        let mut collected: Vec<Option<decls::FileDecls>> = collected.into_iter().map(Some).collect();
        for fi in order {
            let Some(c) = collected[fi].take() else { continue };
            let base = decls.len() as u32;
            for mut d in c.decls {
                if d.parent.is_some() {
                    d.parent = DeclId(d.parent.0 + base);
                }
                decls.push(d);
            }
            let first_item = files[fi].first_item;
            for (li, ld) in c.item_decl {
                if let Some(it) = items.get_mut((first_item + li) as usize) {
                    it.decl = DeclId(base + ld);
                }
            }
            for imp in c.impls {
                raw_impls.push(ImplInfo {
                    decl: DeclId(base + imp.decl),
                    item: imp.item,
                    self_ty: imp.self_ty,
                    trait_ty: imp.trait_ty,
                    head: None,
                    trait_decl: DeclId::NONE,
                    methods: imp.methods.iter().map(|&m| DeclId(base + m)).collect(),
                });
            }
            let m = files[fi].module.idx();
            raw_imports[m].extend(c.imports);
            dropped[m].extend(c.dropped);
            poisoned.extend(c.poisoned);
            allocators.extend(c.allocators);
            diags.extend(c.diags);
        }
        prof(&mut tp, "collect");
        raw_impls.sort_by_key(|i| i.item);

        // Imports.
        let dep_syms: Vec<Vec<(Sym, PackageId)>> = packages
            .iter()
            .map(|p| p.deps.iter().filter_map(|(n, id)| Some((strings.lookup(n)?, *id))).collect())
            .collect();
        let pviews: Vec<modules::PkgView<'_>> = packages
            .iter()
            .enumerate()
            .map(|(k, p)| modules::PkgView {
                name: strings.lookup(&p.name).unwrap_or(Sym::NONE),
                deps: &dep_syms[k],
                root: p.root_module,
            })
            .collect();
        let mut import_uses: Vec<(ItemId, Use)> = Vec::new();
        for mi in 0..modules.len() {
            let file = &files[modules[mi].file.0 as usize];
            let mut imps = Vec::new();
            for ri in std::mem::take(&mut raw_imports[mi]) {
                let seg_syms: Vec<Sym> = ri
                    .segs
                    .iter()
                    .map(|&t| file.syms.get(t as usize).copied().unwrap_or(Sym::NONE))
                    .collect();
                let pr = modules::resolve_path(&midx, &pviews, modules[mi].package, &seg_syms);
                let tspan = |t: u32| scan::tok_span(file.id, &file.bytes, &file.parsed.tokens, t as usize);
                if let Some(k) = pr.failed_at {
                    let name = strings.get(seg_syms[k]);
                    diags.extend(mk("E0101", None, &[("name", name)], tspan(ri.segs[k]), &[]));
                }
                if pr.ambiguous {
                    let pkg = &packages[modules[mi].package.idx()].name;
                    diags.extend(e0107(strings.get(seg_syms[0]), pkg, tspan(ri.segs[0])));
                }
                let root = file.parsed.item_ast((ri.item.0 - file.first_item) as usize).root().unwrap_or(0);
                for (k, &sm) in pr.seg_modules.iter().enumerate() {
                    if let Some(m) = sm.get() {
                        let t = ri.segs[k];
                        let sp = tspan(t);
                        import_uses.push((
                            ri.item,
                            Use {
                                node: root,
                                token: t,
                                offset: sp.offset,
                                res: Res::Decl(modules[m.idx()].decl),
                                in_fmt: false,
                            },
                        ));
                    }
                }
                imps.push(Import {
                    item: ri.item,
                    target: pr.target,
                    bound: file.syms.get(ri.bound_tok as usize).copied().unwrap_or(Sym::NONE),
                    bound_span: tspan(ri.bound_tok),
                    span: ri.span,
                    segs: ri.segs,
                    seg_modules: pr.seg_modules,
                });
            }
            modules[mi].imports = imps;
        }
        prof(&mut tp, "imports");
        diags.extend(modules::cycle_diagnostics(&modules));

        // Module scopes (E0105) and error decls.
        let mut scopes: Vec<FxHashMap<Sym, ScopeEntry>> = vec![FxHashMap::default(); modules.len()];
        let mut top_by_module: Vec<Vec<DeclId>> = vec![Vec::new(); modules.len()];
        for (k, d) in decls.iter().enumerate() {
            if d.parent.is_none() && !matches!(d.kind, DeclKind::Module | DeclKind::Impl | DeclKind::Error) {
                top_by_module[d.module.idx()].push(DeclId(k as u32));
            }
        }
        for mi in 0..modules.len() {
            let mut first_span: FxHashMap<Sym, Span> = FxHashMap::default();
            let mpath = modules[mi].path_text.clone();
            let mut entries: Vec<(Sym, Span, DeclId, bool)> = Vec::new();
            for imp in &modules[mi].imports {
                let decl = match imp.target.get() {
                    Some(t) => modules[t.idx()].decl,
                    None => DeclId::NONE,
                };
                entries.push((imp.bound, imp.bound_span, decl, true));
            }
            let mut tops = std::mem::take(&mut top_by_module[mi]);
            tops.sort_by_key(|d| (decls[d.idx()].name_span.offset, d.0));
            for d in tops {
                entries.push((decls[d.idx()].name, decls[d.idx()].name_span, d, false));
            }
            for (sym, span, decl, import) in entries {
                if sym.is_none() {
                    continue;
                }
                if let Some(&first) = first_span.get(&sym) {
                    let n = strings.get(sym);
                    diags.extend(mk(
                        "E0105",
                        None,
                        &[("name", n), ("scope", &mpath)],
                        span,
                        &[("first", first)],
                    ));
                    continue;
                }
                first_span.insert(sym, span);
                let decl = if decl.is_none() {
                    // A failed import binds an error decl.
                    let id = DeclId(decls.len() as u32);
                    decls.push(error_decl(sym, ModuleId(mi as u32), span));
                    id
                } else {
                    decl
                };
                scopes[mi].insert(sym, ScopeEntry { decl, import });
            }
            for (sym, span) in std::mem::take(&mut dropped[mi]) {
                if let std::collections::hash_map::Entry::Vacant(e) = scopes[mi].entry(sym) {
                    let id = DeclId(decls.len() as u32);
                    decls.push(error_decl(sym, ModuleId(mi as u32), span));
                    e.insert(ScopeEntry { decl: id, import: false });
                }
            }
        }

        prof(&mut tp, "scopes");
        // Prelude, built-in type names, members.
        let prelude_module = packages.iter().position(|p| p.name == cfg.prelude.0).and_then(|p| {
            let rel: Option<Vec<Sym>> = cfg.prelude.1.iter().map(|s| strings.lookup(s)).collect();
            midx.get(PackageId(p as u32), &rel?)
        });
        let mut prelude: FxHashMap<Sym, DeclId> = FxHashMap::default();
        if let Some(pm) = prelude_module {
            for (&s, e) in &scopes[pm.idx()] {
                if !e.import && decls[e.decl.idx()].is_pub() {
                    prelude.insert(s, e.decl);
                }
            }
        }
        // Modules that see the prelude: those of the prelude's package and of every package
        // depending on it (the runtime, which does not depend on std, sees none).
        let prelude_pkg = prelude_module.map(|m| modules[m.idx()].package);
        let prelude_ok: Vec<bool> = modules
            .iter()
            .map(|m| {
                prelude_pkg.is_some_and(|pp| {
                    m.package == pp || packages[m.package.idx()].deps.iter().any(|(_, d)| *d == pp)
                })
            })
            .collect();
        let prims: FxHashMap<Sym, Prim> =
            strings.iter().filter_map(|(s, text)| Prim::from_name(text).map(|p| (s, p))).collect();
        let builtins: FxHashMap<Sym, Builtin> =
            strings.iter().filter_map(|(s, text)| Builtin::from_name(text).map(|b| (s, b))).collect();
        let mut members: FxHashMap<(DeclId, Sym), DeclId> = FxHashMap::default();
        let mut variant_names: FxHashMap<Sym, Vec<DeclId>> = FxHashMap::default();
        for (k, d) in decls.iter().enumerate() {
            if matches!(d.kind, DeclKind::Variant | DeclKind::Field | DeclKind::TraitMethod)
                && d.parent.is_some()
            {
                members.entry((d.parent, d.name)).or_insert(DeclId(k as u32));
                if d.kind == DeclKind::Variant {
                    variant_names.entry(d.name).or_default().push(d.parent);
                }
            }
        }

        for v in variant_names.values_mut() {
            v.sort_by_key(|e| (decls[e.idx()].module, decls[e.idx()].name_span.offset));
        }

        prof(&mut tp, "prelude+members");
        // Impl headers, method tables.
        let empty_methods = MethodTable::default();
        let impl_by_decl: FxHashMap<DeclId, u32> =
            raw_impls.iter().enumerate().map(|(k, i)| (i.decl, k as u32)).collect();
        {
            let t = Tables {
                files: &files,
                modules: &modules,
                items: &items,
                decls: &decls,
                strings: &strings,
                scopes: &scopes,
                prelude: &prelude,
                prims: &prims,
                builtins: &builtins,
                prelude_ok: &prelude_ok,
                members: &members,
                methods: &empty_methods,
                impls: &[],
                impl_by_decl: &impl_by_decl,
                poisoned: &poisoned,
                variant_names: &variant_names,
            };
            let heads: Vec<(Option<TypeHead>, DeclId)> = raw_impls
                .par_iter()
                .map(|i| resolve::impl_header(&t, i.item, i.self_ty, i.trait_ty))
                .collect();
            for (i, (h, tr)) in raw_impls.iter_mut().zip(heads) {
                i.head = h;
                i.trait_decl = tr;
            }
        }
        let methods = method_table(&raw_impls, &decls);
        diags.extend(resolve::method_duplicates(&decls, &raw_impls, &strings, &modules));

        prof(&mut tp, "impls");
        // Resolve every ok item in parallel.
        let mut res: Vec<ItemRes>;
        {
            let t = Tables {
                files: &files,
                modules: &modules,
                items: &items,
                decls: &decls,
                strings: &strings,
                scopes: &scopes,
                prelude: &prelude,
                prims: &prims,
                builtins: &builtins,
                prelude_ok: &prelude_ok,
                members: &members,
                methods: &methods,
                impls: &raw_impls,
                impl_by_decl: &impl_by_decl,
                poisoned: &poisoned,
                variant_names: &variant_names,
            };
            let out: Vec<(ItemRes, Vec<Diagnostic>)> = (0..items.len() as u32)
                .into_par_iter()
                .map(|i| resolve::resolve_item(&t, ItemId(i)))
                .collect();
            res = Vec::with_capacity(out.len());
            for (r, d) in out {
                res.push(r);
                diags.extend(d);
            }
            // Allocators: the path names a module (resolved as an import path).
            let mut alloc_items: Vec<ItemId> = Vec::new();
            for &(item, first, n) in &allocators {
                let Some(it) = items.get(item.idx()) else { continue };
                let file = &files[it.file.0 as usize];
                let toks: Vec<u32> = (0..n).map(|k| first + 2 * k).collect();
                let seg_syms: Vec<Sym> =
                    toks.iter().map(|&t| file.syms.get(t as usize).copied().unwrap_or(Sym::NONE)).collect();
                let pr = modules::resolve_path(&midx, &pviews, file.package, &seg_syms);
                let tspan = |t: u32| scan::tok_span(file.id, &file.bytes, &file.parsed.tokens, t as usize);
                if let Some(k) = pr.failed_at {
                    diags.extend(mk(
                        "E0101",
                        None,
                        &[("name", strings.get(seg_syms[k]))],
                        tspan(toks[k]),
                        &[],
                    ));
                }
                if pr.ambiguous {
                    let pkg = &packages[file.package.idx()].name;
                    diags.extend(e0107(strings.get(seg_syms[0]), pkg, tspan(toks[0])));
                }
                let ast = file.parsed.item_ast(it.index as usize);
                let node = ast.root().map_or(0, |r| {
                    ast.descendants(r).find(|&n| ast.kind(n) == Some(NodeKind::AllocatorDecl)).unwrap_or(r)
                });
                for (k, &sm) in pr.seg_modules.iter().enumerate() {
                    if let Some(m) = sm.get() {
                        let sp = tspan(toks[k]);
                        import_uses.push((
                            item,
                            Use {
                                node,
                                token: toks[k],
                                offset: sp.offset,
                                res: Res::Decl(modules[m.idx()].decl),
                                in_fmt: false,
                            },
                        ));
                    }
                }
                alloc_items.push(item);
            }
            alloc_items.sort();
            if alloc_items.len() > 1 {
                let span_of = |i: ItemId| {
                    let it = &items[i.idx()];
                    let f = &files[it.file.0 as usize];
                    let start = f.parsed.items[it.index as usize].start as usize;
                    scan::tok_span(f.id, &f.bytes, &f.parsed.tokens, start)
                };
                let first = span_of(alloc_items[0]);
                for &i in &alloc_items[1..] {
                    diags.extend(mk("E0104", None, &[], span_of(i), &[("first", first)]));
                }
            }
        }
        prof(&mut tp, "resolve-items");
        let mut touched: Vec<ItemId> = Vec::new();
        for (item, u) in import_uses {
            if let Some(r) = res.get_mut(item.idx()) {
                r.uses.push(u);
                touched.push(item);
            }
        }
        touched.dedup();
        for item in touched {
            res[item.idx()].uses.sort_by_key(|u| (u.node, u.offset));
        }

        prof(&mut tp, "sort-uses");
        // Canonical decl ids.
        let mut impls = raw_impls;
        let mut prelude_list: Vec<(Sym, DeclId)> = prelude.iter().map(|(&s, &d)| (s, d)).collect();
        let (decls, methods) = match canon::decl_remap(&decls, &modules) {
            Some(remap) => {
                for p in &mut prelude_list {
                    p.1 = canon::map_decl(&remap, p.1);
                }
                let mut decls = remap.permute(decls).unwrap_or_default();
                for d in &mut decls {
                    d.parent = canon::map_decl(&remap, d.parent);
                }
                for m in &mut modules {
                    m.decl = canon::map_decl(&remap, m.decl);
                }
                for it in &mut items {
                    it.decl = canon::map_decl(&remap, it.decl);
                }
                for i in &mut impls {
                    i.decl = canon::map_decl(&remap, i.decl);
                    i.trait_decl = canon::map_decl(&remap, i.trait_decl);
                    if let Some(TypeHead::Adt(d)) = i.head {
                        i.head = Some(TypeHead::Adt(canon::map_decl(&remap, d)));
                    }
                    for m in &mut i.methods {
                        *m = canon::map_decl(&remap, *m);
                    }
                }
                res.par_iter_mut().for_each(|r| canon::remap_item_res(&remap, r));
                let methods = method_table(&impls, &decls);
                (decls, methods)
            }
            None => (decls, methods),
        };
        impls.sort_by_key(|i| i.decl);
        prelude_list.sort();

        prof(&mut tp, "canon");
        // Public lookup tables.
        let mut member_lists: Vec<Vec<(u32, DeclId)>> = vec![Vec::new(); decls.len()];
        let mut member_by_name: FxHashMap<(DeclId, Sym), DeclId> = FxHashMap::default();
        let mut module_items: Vec<FxHashMap<Sym, DeclId>> = vec![FxHashMap::default(); modules.len()];
        for (k, d) in decls.iter().enumerate() {
            let id = DeclId(k as u32);
            if let Some(p) = d.parent.get() {
                if let Some(l) = member_lists.get_mut(p.idx()) {
                    l.push((d.index, id));
                }
                member_by_name.entry((p, d.name)).or_insert(id);
            }
        }
        // Module items from the canonical table (the scope maps hold raw ids). Error decls are
        // items too (declared-with-error names).
        for (k, d) in decls.iter().enumerate() {
            if d.parent.is_none() && !matches!(d.kind, DeclKind::Module | DeclKind::Impl) {
                module_items[d.module.idx()].entry(d.name).or_insert(DeclId(k as u32));
            }
        }
        let mut member_list = Vec::new();
        let mut member_range = Vec::with_capacity(decls.len());
        for mut l in member_lists {
            l.sort();
            let lo = member_list.len() as u32;
            member_list.extend(l.into_iter().map(|x| x.1));
            member_range.push((lo, member_list.len() as u32));
        }
        let mut impls_by_head: FxHashMap<TypeHead, Vec<u32>> = FxHashMap::default();
        let mut by_item: Vec<(ItemId, u32)> =
            impls.iter().enumerate().map(|(k, i)| (i.item, k as u32)).collect();
        by_item.sort();
        for (_, k) in by_item {
            if let Some(h) = impls[k as usize].head {
                impls_by_head.entry(h).or_default().push(k);
            }
        }
        prof(&mut tp, "tables");
        let t_resolve = t3.elapsed();
        timings.intern = t_intern;
        timings.resolve = t_resolve;

        let diagnostics = stave_diag::canonical_order(diags, &source_map);
        Program {
            packages,
            files,
            source_map,
            modules,
            items,
            strings,
            decls,
            res,
            impls,
            methods,
            prelude: prelude_list,
            diagnostics,
            timings,
            target,
            member_list,
            member_range,
            member_by_name,
            impls_by_head,
            module_items,
        }
    }
}

/// The loader's packages, their canonical order (`porder[new] = old`), the old → new id map, and
/// the manifests' file ids.
type PackageIds<'a> = (&'a package::Loaded, &'a [usize], &'a [PackageId], &'a FxHashMap<usize, FileId>);

impl Program {
    /// The program of a parse-only load ([`Config::parse_only`]).
    fn parsed_only(
        (loaded, porder, pnew, manifest_ids): PackageIds<'_>,
        mut files: Vec<File>,
        source_map: SourceFiles,
        diags: Vec<Diagnostic>,
        timings: Timings,
        target: Triple,
    ) -> Program {
        let packages: Vec<Package> = porder
            .iter()
            .map(|&old| {
                let rp = &loaded.packages[old];
                let mut deps: Vec<(String, PackageId)> =
                    rp.deps.iter().map(|(n, p)| (n.clone(), pnew[*p])).collect();
                deps.sort();
                Package {
                    name: rp.name.clone(),
                    kind: rp.kind,
                    version: rp.version.clone(),
                    dir: rp.dir.clone(),
                    manifest: rp.manifest.and_then(|m| manifest_ids.get(&m).copied()),
                    deps,
                    root_module: ModuleId::NONE,
                    is_root: rp.is_root,
                }
            })
            .collect();
        let mut items: Vec<Item> = Vec::new();
        for f in &mut files {
            f.first_item = items.len() as u32;
            for (k, it) in f.parsed.items.iter().enumerate() {
                items.push(Item {
                    file: f.id,
                    index: k as u32,
                    module: ModuleId::NONE,
                    kind: it.kind,
                    ok: it.ok,
                    decl: DeclId::NONE,
                });
            }
        }
        let diagnostics = stave_diag::canonical_order(diags, &source_map);
        Program {
            packages,
            files,
            source_map,
            modules: Vec::new(),
            items,
            strings: Strings::default(),
            decls: Vec::new(),
            res: Vec::new(),
            impls: Vec::new(),
            methods: MethodTable::default(),
            prelude: Vec::new(),
            diagnostics,
            timings,
            target,
            member_list: Vec::new(),
            member_range: Vec::new(),
            member_by_name: FxHashMap::default(),
            impls_by_head: FxHashMap::default(),
            module_items: Vec::new(),
        }
    }
}

/// The [`ParsedFile`] of one file from its lexing and item cut: what [`stave_syntax::parse_file`]
/// returns (same tokens, items, item ASTs, diagnostics, and stop), with each item's nodes in a
/// chunk of their own (`ItemInfo::chunk` = item index).
fn parse_cut(
    src: &[u8],
    file: FileId,
    lexed: stave_syntax::Lexed,
    table: Option<Result<stave_syntax::ItemTable, stave_syntax::ItemsError>>,
) -> ParsedFile {
    let mut diagnostics = lexed.diagnostics;
    let mut out = ParsedFile {
        file,
        tokens: Vec::new(),
        items: Vec::new(),
        ast: stave_syntax::Ast::default(),
        diagnostics: Vec::new(),
        stop: None,
    };
    match (lexed.stop, table) {
        (Some(stop), _) => {
            out.stop = Some(match stop {
                stave_syntax::LexStop::TooLarge => stave_syntax::FileStop::TooLarge,
                stave_syntax::LexStop::InvalidUtf8 { .. } => stave_syntax::FileStop::InvalidUtf8,
            });
        }
        (None, Some(Err(e))) => {
            out.stop = Some(match e {
                stave_syntax::ItemsError::Unbalanced(_) => stave_syntax::FileStop::Unbalanced,
                stave_syntax::ItemsError::TooDeep(_) => stave_syntax::FileStop::TooDeep,
            });
            diagnostics.extend(e.into_diagnostic());
        }
        (None, Some(Ok(table))) => {
            let parsed = stave_syntax::parse_items(src, file, &lexed.tokens, &table);
            out.items.reserve(parsed.len());
            out.ast.chunks.reserve(parsed.len());
            for (k, (p, span)) in parsed.into_iter().zip(&table.items).enumerate() {
                out.items.push(stave_syntax::ItemInfo {
                    kind: span.kind,
                    start: span.start,
                    end: span.end,
                    ok: p.ok,
                    chunk: k as u32,
                    nodes: 0..p.nodes.len() as u32,
                });
                diagnostics.extend(p.diagnostic);
                out.ast.chunks.push(p.nodes);
            }
        }
        (None, None) => {}
    }
    diagnostics.sort_by(|a, b| {
        let key = |d: &Diagnostic| d.primary().map(|s| s.offset);
        key(a).cmp(&key(b)).then_with(|| a.code().cmp(b.code())).then_with(|| a.message().cmp(b.message()))
    });
    out.tokens = lexed.tokens;
    out.diagnostics = diagnostics;
    out
}

/// E0107: the first import segment is both a dependency and a module of the package.
fn e0107(name: &str, package: &str, at: Span) -> Option<Diagnostic> {
    mk("E0107", None, &[("name", name), ("package", package)], at, &[])
}

/// E1003 reason text of a read error.
fn read_reason(e: &std::io::Error) -> &'static str {
    match e.kind() {
        std::io::ErrorKind::NotFound => "not found",
        std::io::ErrorKind::PermissionDenied => "permission denied",
        std::io::ErrorKind::IsADirectory => "is a directory",
        _ => "read error",
    }
}

fn prof(tp: &mut (Instant, bool), name: &str) {
    if tp.1 {
        eprintln!("  {name}: {:.3} ms", tp.0.elapsed().as_secs_f64() * 1e3);
    }
    tp.0 = Instant::now();
}

fn error_decl(name: Sym, module: ModuleId, span: Span) -> Decl {
    Decl {
        kind: DeclKind::Error,
        name,
        module,
        parent: DeclId::NONE,
        item: ItemId::NONE,
        node: u32::MAX,
        name_span: span,
        flags: DeclFlags(DeclFlags::PUB),
        index: 0,
    }
}

/// Method tables from impls (in item order).
fn method_table(impls: &[ImplInfo], decls: &[Decl]) -> MethodTable {
    let mut order: Vec<&ImplInfo> = impls.iter().collect();
    order.sort_by_key(|i| i.item);
    let mut t = MethodTable::default();
    for imp in order {
        let Some(h) = imp.head else { continue };
        let inherent = imp.trait_ty == u32::MAX;
        for &m in &imp.methods {
            let Some(d) = decls.get(m.idx()) else { continue };
            let map = if inherent { &mut t.inherent } else { &mut t.via_trait };
            map.entry((h, d.name)).or_default().push(m);
        }
    }
    t
}

/// Interns the identifiers of one file (local dedup first, then the shared pool), including
/// f-string hole identifiers.
fn intern_file(pool: &intern::Pool, f: &mut File) {
    let toks = &f.parsed.tokens;
    let mut syms = vec![Sym::NONE; toks.len()];
    let mut local: FxHashMap<&[u8], Sym> = FxHashMap::default();
    let src: &[u8] = &f.bytes;
    fn get<'s>(local: &mut FxHashMap<&'s [u8], Sym>, pool: &intern::Pool, text: &'s [u8]) -> Sym {
        if let Some(&s) = local.get(text) {
            return s;
        }
        let s = pool.intern(std::str::from_utf8(text).unwrap_or(""));
        local.insert(text, s);
        s
    }
    for (i, t) in toks.iter().enumerate() {
        if t.kind == Term::Ident as u8 {
            syms[i] = get(&mut local, pool, stave_syntax::token_text(src, toks, i));
        } else if t.kind == Term::FmtLit as u8 {
            let text = stave_syntax::token_text(src, toks, i);
            for h in scan::fmt_holes(text, t.start) {
                for (off, _) in h.segs {
                    let s = (off - t.start) as usize;
                    let e = text[s..]
                        .iter()
                        .position(|b| !(b.is_ascii_alphanumeric() || *b == b'_'))
                        .map_or(text.len(), |p| s + p);
                    if &text[s..e] != b"self" {
                        get(&mut local, pool, &text[s..e]);
                    }
                }
            }
        }
    }
    f.syms = syms;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stagewise front end (lex, items, parse as separate passes) gives what
    /// `stave_syntax::parse_file` gives.
    #[test]
    fn stagewise_parse_equals_parse_file() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut srcs: Vec<Vec<u8>> = Vec::new();
        for dir in ["std", "runtime", "tests/conformance/run/json_lite/src"] {
            for e in std::fs::read_dir(root.join(dir)).into_iter().flatten().flatten() {
                if e.path().extension().is_some_and(|x| x == "stv") {
                    srcs.extend(std::fs::read(e.path()));
                }
            }
        }
        assert!(srcs.len() > 5, "{} sources", srcs.len());
        srcs.push(b"fn f() { let x = ; }\nfn g( {\n".to_vec());
        srcs.push(b"fn f() -> i32 { return 1 }\n}\n".to_vec());
        srcs.push(b"fn f() { \"\xff\" }\n".to_vec());
        srcs.push(format!("fn f() {{ {} }}\n", "(".repeat(5000)).into_bytes());
        srcs.push(
            format!("fn f() -> i32 {{ return {}1{}; }}\n", "(".repeat(300), ")".repeat(300)).into_bytes(),
        );
        srcs.push(b"x y z\nimport a;\nfn ok() {}\n".to_vec());
        srcs.push(Vec::new());
        for (i, src) in srcs.iter().enumerate() {
            let id = FileId(i as u32);
            let want = stave_syntax::parse_file(src, id);
            let lx = stave_syntax::lex(src, id);
            let tb = lx.stop.is_none().then(|| stave_syntax::items(&lx.tokens, id));
            let got = parse_cut(src, id, lx, tb);
            assert_eq!(got.tokens, want.tokens, "file {i}");
            assert_eq!(got.stop, want.stop, "file {i}");
            assert_eq!(got.diagnostics, want.diagnostics, "file {i}");
            assert_eq!(got.items.len(), want.items.len(), "file {i}");
            for (k, (a, b)) in got.items.iter().zip(&want.items).enumerate() {
                assert_eq!(
                    (a.kind, a.start, a.end, a.ok),
                    (b.kind, b.start, b.end, b.ok),
                    "file {i} item {k}"
                );
                assert_eq!(got.item_ast(k).nodes, want.item_ast(k).nodes, "file {i} item {k}");
            }
        }
    }
}
