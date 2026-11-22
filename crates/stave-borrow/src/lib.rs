//! Borrow checking.
//!
//! One fn body at a time, from signatures alone, in parallel (rayon, the caller's thread count)
//! with deterministic output:
//!
//! 1. [`cfg`]: a MIR-like CFG from the typed AST ([`stave_sema::TypedBody`]): places
//!    `local.(field|deref|index)*`, statements, loans;
//! 2. [`loans`]: origin sets per tracked local, solved by forward dataflow to a fixpoint;
//! 3. [`check`]: loan liveness by path exploration, then every access against the live
//!    overlapping loans (E0501), moved places (E0502), mutability (E0503), view writes (E0513),
//!    and returns against the `from` clause (E0510, E0512);
//! 4. [`report`]: diagnostics from the registry templates (E0501 loan choice);
//! 5. [`locks`]: the locks held at each call and join, for the rank check
//!    ([`stave_sema::ranks`]).
//!
//! Bodies with any type error are skipped (no cascades). Bodies the builder cannot model
//! ([`Borrowck::unsupported`]) get no diagnostics and are not checked: a program with one is not
//! known to be sound, so the caller must not accept it ([`Borrowck::unchecked`]). Const and `static mut`
//! initializers hold no borrows that outlive them and are not checked.
//!
//! **Contract clauses** are code of the body at the points where they run, whether or not a
//! check for them stays in the program: the `requires` clauses and then the `old(e)` values at
//! entry, the `ensures` clauses at each return after the scopes are left, over the params as the
//! body left them and the returned value (`result`). An `old(e)` value and `result` are live to
//! the clauses, so what they borrow stays borrowed until then.
//!
//! **Rules LANG.md leaves open**: a `&mut` borrow (explicit,
//! auto-referenced receiver, or capture) of a non-`mut` binding is E0503 (base case), and of a
//! place behind `&` is E0503 `shared-ref`; a move out of a dereference, an index, or a closure
//! capture is E0504 and is then checked as a copy (no E0502 cascade); temporaries without
//! drop glue live to the end of the fn;
//! `&mut` receivers passed on (`reborrow`) are two-phase like auto-referenced ones; the `borrow`
//! dump's `loan` column is the creation node of the reported loan.

pub mod bits;
pub mod cfg;
pub mod check;
pub mod live;
pub mod loans;
pub mod locks;
pub mod old;
pub mod report;
pub mod set;
pub mod ty;

use std::cell::Cell;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use stave_diag::Diagnostic;
use stave_sema::itemcx::{Brackets, ItemCx};
use stave_sema::{Program, Typeck, TypedBody};
use stave_syntax::NodeKind;

pub use check::DumpAccess;
pub use report::failures as diagnostic_build_failures;

/// The borrow checker's result.
#[derive(Debug, Default)]
pub struct Borrowck {
    /// In body order; the driver puts them into canonical order.
    pub diagnostics: Vec<Diagnostic>,
    /// `borrow` dump lines per item: (item, access), sorted.
    pub dump: Vec<(u32, DumpAccess)>,
    /// Bodies checked.
    pub checked: u64,
    /// Bodies skipped for a type error.
    pub skipped: u64,
    /// Bodies the builder could not model (nothing reported for them), with the reason.
    pub unsupported: Vec<(stave_sema::DeclId, &'static str)>,
    /// Bodies not checked because the builder found it disagrees with stave-sema (a compiler
    /// bug), with the reason.
    pub internal: Vec<(stave_sema::DeclId, &'static str)>,
    /// The largest number of dataflow rounds of one body.
    pub max_rounds: u32,
    pub time: Duration,
    /// Summed per-body time of the phases (CFG build, dataflow, checks + reports), all threads.
    pub phases: [Duration; 3],
    /// The locks each body that may hold one holds at its calls, joins, and drops, in body order.
    pub locks: Vec<stave_sema::ranks::BodyLocks>,
    /// The types each checked body may drop, in body order.
    pub drops: Vec<stave_sema::ranks::BodyDrops>,
    /// What the sparse dataflow did, summed over the bodies.
    pub cost: loans::Cost,
}

impl Borrowck {
    /// The first body that was not borrow checked because the builder cannot model it, with the
    /// reason. `Some` means the program must be rejected (it was not proven sound).
    pub fn unchecked(&self) -> Option<(stave_sema::DeclId, &'static str)> {
        self.unsupported.first().copied()
    }
}

/// Which dataflow checks the bodies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dataflow {
    /// [`loans`] and [`check`].
    #[default]
    Sparse,
    /// [`old`].
    Dense,
    /// Both: a body for which they give different diagnostics or dump lines is not checked and
    /// fails the run as an internal error ([`Borrowck::internal`]).
    Both,
}

/// The environment variable that names the dataflow.
pub const DATAFLOW_ENV: &str = "STAVE_BORROW_DATAFLOW";

impl Dataflow {
    /// The dataflow a value of [`DATAFLOW_ENV`] names: unset, empty or `new` the sparse one, `old`
    /// the dense one, `both` both. Any other value is an error.
    pub fn named(v: Option<&str>) -> Result<Dataflow, String> {
        match v.map(str::trim) {
            None | Some("" | "new") => Ok(Dataflow::Sparse),
            Some("old") => Ok(Dataflow::Dense),
            Some("both") => Ok(Dataflow::Both),
            Some(s) => Err(format!("{DATAFLOW_ENV} must be `new`, `old` or `both`, not `{s}`")),
        }
    }

    /// The choice of [`DATAFLOW_ENV`], read once.
    ///
    /// # Panics
    /// On a value [`Dataflow::named`] rejects: a run must not go on with another dataflow than
    /// the one asked for. `stave` checks the value first and reports it as a usage error.
    pub fn from_env() -> Dataflow {
        static CHOICE: OnceLock<Dataflow> = OnceLock::new();
        *CHOICE.get_or_init(|| match Dataflow::named(std::env::var(DATAFLOW_ENV).ok().as_deref()) {
            Ok(d) => d,
            Err(msg) => panic!("{msg}"),
        })
    }
}

/// Options.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Collect the `borrow` dump.
    pub dump: bool,
    pub dataflow: Dataflow,
}

struct One {
    diags: Vec<Diagnostic>,
    dump: Vec<(u32, DumpAccess)>,
    unsupported: Option<&'static str>,
    internal: Option<&'static str>,
    rounds: u32,
    phases: [Duration; 3],
    locks: locks::Scan,
    cost: loans::Cost,
}

/// Borrow-checks every fn body of the program without a type error.
pub fn check(prog: &Program, tc: &Typeck, opts: Options) -> Borrowck {
    let t0 = Instant::now();
    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let run = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let jobs: Vec<&TypedBody> = tc.bodies.iter().filter(|b| tc.sigs.fn_sig(b.decl).is_some()).collect();
    let skipped = jobs.iter().filter(|b| b.has_error).count() as u64;
    let todo: Vec<&TypedBody> = jobs.into_iter().filter(|b| !b.has_error).collect();
    let results: Vec<Option<One>> = on_big_stacks(|| {
        // bracket tables once per item (an impl's methods share theirs)
        let mut items: Vec<u32> = todo.iter().map(|b| b.item.0).collect();
        items.sort_unstable();
        items.dedup();
        let brs: Vec<(u32, Option<Brackets>)> = items
            .par_iter()
            .map(|&i| {
                let br = prog.item(stave_sema::ItemId(i)).and_then(|it| {
                    let f = prog.file(it.file)?;
                    let info = f.parsed.items.get(it.index as usize)?;
                    Some(Brackets::new(&f.parsed.tokens, info.start, info.end))
                });
                (i, br)
            })
            .collect();
        todo.par_iter()
            .map(|tb| {
                let k = brs.partition_point(|x| x.0 < tb.item.0);
                let br = brs.get(k).filter(|x| x.0 == tb.item.0).and_then(|x| x.1.as_ref())?;
                check_body(prog, tc, tb, br, opts, run)
            })
            .collect()
    });
    let mut out = Borrowck { skipped, ..Borrowck::default() };
    for (tb, r) in todo.iter().zip(results) {
        let Some(r) = r else { continue };
        out.checked += 1;
        out.max_rounds = out.max_rounds.max(r.rounds);
        out.cost.add(r.cost);
        for (a, b) in out.phases.iter_mut().zip(r.phases) {
            *a += b;
        }
        if let Some(why) = r.internal {
            out.internal.push((tb.decl, why));
            continue;
        }
        if let Some(why) = r.unsupported {
            out.unsupported.push((tb.decl, why));
            continue;
        }
        out.diagnostics.extend(r.diags);
        out.dump.extend(r.dump);
        out.drops.push(stave_sema::ranks::BodyDrops { decl: tb.decl, types: r.locks.drops.clone() });
        if r.locks.holds {
            let l = r.locks;
            out.locks.push(stave_sema::ranks::BodyLocks {
                decl: tb.decl,
                sites: l.sites,
                any: l.any,
                calls: l.calls,
                types: l.types,
            });
        }
    }
    out.dump.sort();
    out.dump.dedup();
    out.time = t0.elapsed();
    out
}

fn check_body(
    prog: &Program,
    tc: &Typeck,
    tb: &TypedBody,
    br: &Brackets,
    opts: Options,
    run: u64,
) -> Option<One> {
    let sig = tc.sigs.fn_sig(tb.decl)?;
    let icx = ItemCx::new(prog, tb.item, br)?;
    icx.child(tb.node, NodeKind::Block)?;
    let mut tys = ty::Tys::new(prog, &tc.sigs, &tc.types, &sig.generics, run);
    // A ref/view return is checked against the declared origin set (no `from` clause:
    // the empty set); a signature with an invalid `from` entry (E0520, E0521) is not (no cascade)
    let so = stave_sema::origins::of_sig(prog, &tys.ov, sig);
    let bad_from = so.error && !sig.origins.is_empty();
    let ret = if tys.ref_like(sig.ret) && !bad_from {
        // the `from` clause; without one, the return type (E0510 `no-from` case)
        let clause = icx
            .child(tb.node, NodeKind::RetClause)
            .and_then(|rc| icx.child(rc, NodeKind::From))
            .map(|n| icx.node_span(n));
        let from = clause.or(sig.ret_span);
        let list: Vec<(u32, bool)> = sig.origins.iter().map(|o| (u32::from(o.param), o.copy)).collect();
        match from {
            Some(s) => cfg::RetCheck::From(list, s, clause.is_some()),
            None => cfg::RetCheck::None,
        }
    } else {
        cfg::RetCheck::None
    };
    let declared: Vec<cfg::Store> = sig
        .param_origins
        .iter()
        .map(|o| cfg::Store {
            target: u32::from(o.target),
            source: u32::from(o.source),
            copy: o.copy,
            into_view: true,
        })
        .collect();
    let t0 = Instant::now();
    let body = cfg::build_fn(icx, tb, &mut tys, tb.node, (&sig.params, sig.ret), ret, declared);
    let mut one = One {
        diags: Vec::new(),
        dump: Vec::new(),
        unsupported: body.unsupported,
        internal: body.internal,
        rounds: 0,
        phases: [t0.elapsed(), Duration::ZERO, Duration::ZERO],
        locks: locks::Scan { drops: Some(Vec::new()), ..locks::Scan::default() },
        cost: loans::Cost::default(),
    };
    if body.unsupported.is_none() && body.internal.is_none() {
        check_tree(&body, &mut tys, opts, tb.item.0, &mut one);
        if one.locks.holds {
            one.locks.types = tys.ov.local.clone();
        }
    }
    Some(one)
}

/// Checks `root` and then the bodies nested in it, each before the ones nested in it, in order.
fn check_tree(root: &cfg::Body, tys: &mut ty::Tys<'_>, opts: Options, item: u32, one: &mut One) {
    let mut todo: Vec<&cfg::Body> = vec![root];
    while let Some(body) = todo.pop() {
        check_body_flow(body, tys, opts, item, one);
        todo.extend(body.nested.iter().rev());
    }
}

/// What a dataflow found in one body.
#[derive(PartialEq, Eq)]
struct Found {
    diags: Vec<Diagnostic>,
    dump: Vec<DumpAccess>,
}

/// One body under the sparse dataflow.
fn sparse<'b>(
    body: &'b cfg::Body,
    tys: &mut ty::Tys<'_>,
    dump: bool,
    phases: &mut [Duration; 3],
) -> (Found, loans::Flow<'b>) {
    let t0 = Instant::now();
    let mut flow = loans::Flow::new(body, tys);
    flow.solve();
    let t1 = Instant::now();
    let c = check::check(&mut flow, dump);
    let diags =
        report::diagnostics(body, tys, c.findings, &mut |at, loan| check::later_use(&mut flow, at, loan));
    phases[1] += t1 - t0;
    phases[2] += t1.elapsed();
    (Found { diags, dump: c.dump }, flow)
}

/// One body under the dense dataflow.
fn dense<'b>(
    body: &'b cfg::Body,
    tys: &mut ty::Tys<'_>,
    dump: bool,
    phases: &mut [Duration; 3],
) -> (Found, old::loans::Flow<'b>) {
    let t0 = Instant::now();
    let mut flow = old::loans::Flow::new(body, tys);
    flow.solve();
    let t1 = Instant::now();
    let c = old::check::check(&mut flow, dump);
    let diags = report::diagnostics(body, tys, c.findings, &mut |at, loan| {
        old::check::later_use(&mut flow, at, loan)
    });
    phases[1] += t1 - t0;
    phases[2] += t1.elapsed();
    (Found { diags, dump: c.dump }, flow)
}

fn check_body_flow(body: &cfg::Body, tys: &mut ty::Tys<'_>, opts: Options, item: u32, one: &mut One) {
    let (found, rounds, l) = match opts.dataflow {
        Dataflow::Sparse | Dataflow::Both => {
            let (found, flow) = sparse(body, tys, opts.dump, &mut one.phases);
            if opts.dataflow == Dataflow::Both {
                let (other, _) = dense(body, tys, opts.dump, &mut [Duration::ZERO; 3]);
                if other != found {
                    one.internal.get_or_insert("the sparse and the dense borrow dataflow disagree");
                }
            }
            one.cost.add(flow.cost());
            let l = locks::scan(body, &flow.rpo, &|t| flow.is_ref_like(t), tys);
            (found, flow.rounds, l)
        }
        Dataflow::Dense => {
            let (found, flow) = dense(body, tys, opts.dump, &mut one.phases);
            let l = locks::scan(body, &flow.rpo, &|t| flow.is_ref_like(t), tys);
            (found, flow.rounds, l)
        }
    };
    one.rounds = one.rounds.max(rounds);
    one.diags.extend(found.diags);
    one.dump.extend(found.dump.into_iter().map(|d| (item, d)));
    one.locks.holds |= l.holds;
    one.locks.sites.extend(l.sites);
    one.locks.any.extend(l.any);
    one.locks.calls.extend(l.calls);
    one.locks.drops = match (one.locks.drops.take(), l.drops) {
        (Some(mut a), Some(b)) => {
            a.extend(b);
            Some(a)
        }
        _ => None,
    };
}

// ---------- big-stack workers (the builder recurses over expression nesting) ----------

const STACK: usize = 512 << 20;

thread_local! {
    static BIG: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` on big-stack workers, as many as the current rayon pool has (the caller's count).
fn on_big_stacks<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if BIG.with(Cell::get) {
        return f();
    }
    let n = rayon::current_num_threads().max(1);
    let run = move || {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(STACK)
            .start_handler(|_| BIG.with(|b| b.set(true)))
            .build();
        match pool {
            Ok(p) => p.install(f),
            Err(_) => f(),
        }
    };
    // a worker of another pool waiting in `install` would run that pool's other jobs on its own
    // stack meanwhile; a plain thread waits without running anything
    if rayon::current_thread_index().is_some() {
        std::thread::scope(|s| s.spawn(run).join()).unwrap_or_else(|e| std::panic::resume_unwind(e))
    } else {
        run()
    }
}
