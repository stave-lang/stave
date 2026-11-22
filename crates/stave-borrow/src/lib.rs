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
pub mod loans;
pub mod locks;
pub mod report;
pub mod ty;

use std::cell::Cell;
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
    /// The drop schedule of each body a CFG was built for ([`stave_sema::drops`]): the lowering
    /// checks every drop it emits against it.
    pub schedules: stave_sema::drops::Schedules,
}

impl Borrowck {
    /// The first body that was not borrow checked because the builder cannot model it, with the
    /// reason. `Some` means the program must be rejected (it was not proven sound).
    pub fn unchecked(&self) -> Option<(stave_sema::DeclId, &'static str)> {
        self.unsupported.first().copied()
    }
}

/// Options.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Collect the `borrow` dump.
    pub dump: bool,
}

struct One {
    diags: Vec<Diagnostic>,
    dump: Vec<(u32, DumpAccess)>,
    unsupported: Option<&'static str>,
    internal: Option<&'static str>,
    rounds: u32,
    phases: [Duration; 3],
    locks: locks::Scan,
    schedule: stave_sema::drops::Schedule,
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
        let Some(mut r) = r else { continue };
        out.checked += 1;
        out.schedules.insert(tb.decl, std::mem::take(&mut r.schedule));
        out.max_rounds = out.max_rounds.max(r.rounds);
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
        schedule: schedule_of(&body),
    };
    if body.unsupported.is_none() && body.internal.is_none() {
        check_tree(&body, &mut tys, opts, tb.item.0, &mut one);
        if one.locks.holds {
            one.locks.types = tys.ov.local.clone();
        }
    }
    Some(one)
}

/// The drop schedule of `body` and the bodies nested in it, sorted.
fn schedule_of(body: &cfg::Body) -> stave_sema::drops::Schedule {
    fn add(b: &cfg::Body, out: &mut stave_sema::drops::Schedule) {
        out.extend(&b.schedule);
        for nb in &b.nested {
            add(nb, out);
        }
    }
    let mut out = stave_sema::drops::Schedule::default();
    add(body, &mut out);
    out.seal();
    out
}

fn check_tree(body: &cfg::Body, tys: &mut ty::Tys<'_>, opts: Options, item: u32, one: &mut One) {
    let t0 = Instant::now();
    let mut flow = loans::Flow::new(body, tys);
    flow.solve();
    one.rounds = one.rounds.max(flow.rounds);
    let t1 = Instant::now();
    let c = check::check(&mut flow, opts.dump);
    one.diags.extend(report::diagnostics(&mut flow, tys, c.findings));
    one.phases[1] += t1 - t0;
    one.phases[2] += t1.elapsed();
    one.dump.extend(c.dump.into_iter().map(|d| (item, d)));
    let l = locks::scan(&flow, tys);
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
    for nb in &body.nested {
        check_tree(nb, tys, opts, item, one);
    }
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
