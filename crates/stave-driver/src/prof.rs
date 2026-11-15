//! Step profile of one compile, for compile-speed work.
//!
//! With the environment variable [`PROFILE_ENV`] set, the pipeline and the CLI record the wall
//! time of every step between two [`mark`]s on the driver thread and the CLI prints the table on
//! stderr after the summary line ([`report`]). It is finer than `--timings` and not part of any
//! contract: steps and format may change. `bench/run.sh --stages` reads it.
//!
//! Lines: `profile: <step> <ms> <allocation calls> <allocated bytes>`. A step whose name starts
//! with `.` is a part of the step above it, reported by that step itself ([`part`]): wall time of
//! a sequential part, or time summed over the worker threads where the name ends in `-cpu`.
//! Allocation counts are zero unless the binary was built with [`CountingAlloc`] as its global
//! allocator (stave-cli's `alloc-count` feature; the counters slow parallel runs down, so time
//! and count in separate runs).
//!
//! One profile per process: concurrent pipeline runs (tests) must leave the variable unset.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Set (to anything) to record and print the profile.
pub const PROFILE_ENV: &str = "STAVE_PROFILE";

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// The system allocator, counting calls and requested bytes (a `realloc` counts as one call of
/// its new size).
pub struct CountingAlloc;

// SAFETY: every method forwards its arguments unchanged to `System`, whose contract is the
// caller's; the counters do not touch the allocation.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `l`.
        unsafe { System.alloc(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` was allocated by `System` with layout `l` (every method forwards to it).
        unsafe { System.dealloc(p, l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract for `l`.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(n as u64, Ordering::Relaxed);
        // SAFETY: `p` was allocated by `System` with layout `l`; the caller upholds the rest of
        // `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(p, l, n) }
    }
}

struct Row {
    name: &'static str,
    ms: f64,
    allocs: u64,
    bytes: u64,
}

struct State {
    start: Instant,
    last: Instant,
    allocs: u64,
    bytes: u64,
    rows: Vec<Row>,
}

static ON: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn counters() -> (u64, u64) {
    (ALLOCS.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
}

/// Starts the profile clock if [`PROFILE_ENV`] is set.
pub fn start() {
    if std::env::var_os(PROFILE_ENV).is_none() {
        return;
    }
    let now = Instant::now();
    let (allocs, bytes) = counters();
    if let Ok(mut s) = STATE.lock() {
        *s = Some(State { start: now, last: now, allocs, bytes, rows: Vec::new() });
        ON.store(true, Ordering::Relaxed);
    }
}

pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Ends step `name`: everything since the previous mark (or [`start`]).
pub fn mark(name: &'static str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let (allocs, bytes) = counters();
    if let Ok(mut g) = STATE.lock()
        && let Some(s) = g.as_mut()
    {
        s.rows.push(Row {
            name,
            ms: (now - s.last).as_secs_f64() * 1e3,
            allocs: allocs - s.allocs,
            bytes: bytes - s.bytes,
        });
        // the bookkeeping above is not part of the next step
        s.last = Instant::now();
        (s.allocs, s.bytes) = counters();
    }
}

/// Records part `name` (starting with `.`) of the step whose [`mark`] comes next.
pub fn part(name: &'static str, d: Duration) {
    if !enabled() {
        return;
    }
    if let Ok(mut g) = STATE.lock()
        && let Some(s) = g.as_mut()
    {
        s.rows.push(Row { name, ms: d.as_secs_f64() * 1e3, allocs: 0, bytes: 0 });
    }
}

/// The table (empty when the profile is off), ending the profile. The steps are in execution
/// order with each step's parts after it; the last line is the total since [`start`].
pub fn report() -> String {
    if !enabled() {
        return String::new();
    }
    ON.store(false, Ordering::Relaxed);
    let Some(s) = STATE.lock().ok().and_then(|mut g| g.take()) else { return String::new() };
    let mut out = String::new();
    let mut line = |name: &str, ms: f64, allocs: u64, bytes: u64| {
        out.push_str(&format!("profile: {name} {ms:.3} {allocs} {bytes}\n"));
    };
    // parts are recorded before their step's mark
    let mut parts: Vec<&Row> = Vec::new();
    let (mut allocs, mut bytes) = (0, 0);
    for r in &s.rows {
        if r.name.starts_with('.') {
            parts.push(r);
            continue;
        }
        line(r.name, r.ms, r.allocs, r.bytes);
        allocs += r.allocs;
        bytes += r.bytes;
        for p in parts.drain(..) {
            line(p.name, p.ms, p.allocs, p.bytes);
        }
    }
    line("total", (s.last - s.start).as_secs_f64() * 1e3, allocs, bytes);
    out
}
