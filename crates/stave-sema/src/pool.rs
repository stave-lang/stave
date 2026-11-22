//! Worker threads for the recursive phases (type checking, lowering).
//!
//! Both descend expression trees recursively (up to [`crate::typeck::MAX_DEPTH`] levels), which
//! needs more stack than a default rayon worker has. They run on a pool of big-stack threads with
//! as many workers as the caller's current rayon pool ([`rayon::current_num_threads`]:
//! the pool the call is `install`ed in, else the global one), so the caller's thread count
//! (`STAVE_THREADS`, `RAYON_NUM_THREADS`) is the phase's thread count. One pool per thread
//! count is made, at the first phase that asks for it, and every later phase of the process
//! (the checker, the borrow checker, the lowering) runs on it: its stacks are mapped once,
//! before the program's data fills the address space. Nested calls run on the pool they are
//! already on.
//!
//! A worker of another pool that waits in `install` keeps running its own pool's jobs meanwhile
//! (rayon's cross-pool wait), each on top of the waiting frames: callers that start one phase per
//! item of a `par_iter` would nest those items on one stack without bound. Such a caller waits on
//! a plain thread instead, which runs nothing else while it waits.
//!
//! Where the address space cannot hold that many stacks (a limit on it), the pool has half as
//! many workers, down to one; a pool of one that cannot be created (after a few attempts)
//! stops the run with [`NoThreads`] (a plain error, exit 2): the phases never run on a stack
//! smaller than [`STACK`].

use std::cell::Cell;

/// Stack size of a worker (address space only; pages are touched on demand).
const STACK: usize = 512 << 20;

thread_local! {
    static BIG: Cell<bool> = const { Cell::new(false) };
}

/// The unwind payload of a run whose worker threads cannot be created (the address space or
/// the thread limit is exhausted). The CLI reports it as a plain error, not as a bug.
#[derive(Debug)]
pub struct NoThreads(pub String);

/// Stops the run with [`NoThreads`]: an unwind without the panic hook, so nothing is printed
/// as an internal error.
pub fn no_threads(e: impl std::fmt::Display) -> ! {
    std::panic::resume_unwind(Box::new(NoThreads(e.to_string())))
}

/// Runs `f` on big-stack workers, as many as the current rayon pool has.
pub fn on_big_stacks<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if BIG.with(Cell::get) {
        return f();
    }
    let n = rayon::current_num_threads().max(1);
    let run = move || shared_pool(n).install(f);
    if rayon::current_thread_index().is_some() { wait_off_pool(run) } else { run() }
}

/// The big-stack pool for a caller pool of `n` threads, made once ([`big_pool`]).
fn shared_pool(n: usize) -> std::sync::Arc<rayon::ThreadPool> {
    static POOLS: std::sync::Mutex<Vec<(usize, std::sync::Arc<rayon::ThreadPool>)>> =
        std::sync::Mutex::new(Vec::new());
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = pools.iter().find(|(k, _)| *k == n) {
        return p.clone();
    }
    let p = std::sync::Arc::new(big_pool(n, STACK, &BIG));
    pools.push((n, p.clone()));
    p
}

/// Attempts to create threads before giving up (or halving a pool): the address space may be
/// short for a moment while other threads come and go.
const TRIES: u32 = 50;

/// A pool of `n` workers with `stack`-byte stacks (each marks `flag`). When the address space
/// cannot hold them, half as many are tried, down to one; none at all stops the run
/// ([`NoThreads`]).
pub fn big_pool(
    n: usize,
    stack: usize,
    flag: &'static std::thread::LocalKey<Cell<bool>>,
) -> rayon::ThreadPool {
    use std::sync::atomic::Ordering;
    // the most workers a pool had room for so far: later pools start there
    static FIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
    let mut k = n.max(1).min(FIT.load(Ordering::SeqCst).max(1));
    let mut tries = 0u32;
    loop {
        // a pool whose build fails keeps the workers it started (and their stacks): test the
        // room with threads that are joined first
        if !room_for(k, stack) {
            if again(&mut tries) {
                continue;
            }
            if k > 1 {
                k /= 2;
                tries = 0;
                continue;
            }
        }
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(k)
            .stack_size(stack)
            .start_handler(move |_| flag.with(|b| b.set(true)))
            .build();
        match built {
            Ok(p) => {
                if k < n {
                    FIT.store(k, Ordering::SeqCst);
                }
                return p;
            }
            Err(_) if again(&mut tries) => {}
            Err(_) if k > 1 => {
                k /= 2;
                tries = 0;
            }
            Err(e) => no_threads(e),
        }
    }
}

/// Runs `f` on a plain thread and waits for it (a panic in `f` resumes here). A thread that
/// cannot be created is tried again a few times ([`TRIES`]).
pub fn wait_off_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let job = std::sync::Mutex::new(Some(f));
    let mut tries = 0u32;
    loop {
        let spawned = std::thread::scope(|s| {
            let run = || {
                let f = job.lock().ok().and_then(|mut j| j.take());
                f.map(|f| f())
            };
            std::thread::Builder::new().spawn_scoped(s, run).map(|h| h.join())
        });
        match spawned {
            Ok(Ok(Some(v))) => return v,
            Ok(Ok(None)) => no_threads("the job of a worker thread was lost"),
            Ok(Err(e)) => std::panic::resume_unwind(e),
            Err(e) => {
                if !again(&mut tries) {
                    no_threads(e)
                }
            }
        }
    }
}

/// Whether `k` threads with `stack`-byte stacks can exist at once: each is started, waits until
/// all are, and is joined.
fn room_for(k: usize, stack: usize) -> bool {
    let gate = std::sync::Mutex::new(());
    let held = gate.lock();
    std::thread::scope(|s| {
        let mut ok = true;
        for _ in 0..k {
            let started = std::thread::Builder::new().stack_size(stack).spawn_scoped(s, || drop(gate.lock()));
            if started.is_err() {
                ok = false;
                break;
            }
        }
        drop(held);
        ok
    })
}

/// Waits a millisecond before another attempt: `false` after [`TRIES`] attempts.
fn again(tries: &mut u32) -> bool {
    if *tries >= TRIES {
        return false;
    }
    *tries += 1;
    std::thread::sleep(std::time::Duration::from_millis(1));
    true
}

#[cfg(test)]
mod tests {
    use super::on_big_stacks;

    #[test]
    fn follows_the_callers_pool() {
        for n in [1usize, 3] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            let (outer, inner) = pool.install(|| {
                on_big_stacks(|| (rayon::current_num_threads(), on_big_stacks(rayon::current_num_threads)))
            });
            assert_eq!((outer, inner), (n, n));
        }
    }

    /// Phases started from the items of a `par_iter` run one at a time on each caller's stack: a
    /// caller waiting for its phase does not take up the next item.
    #[test]
    fn parallel_callers_do_not_nest() {
        use rayon::prelude::*;
        use std::cell::Cell;
        thread_local! {
            static DEPTH: Cell<u32> = const { Cell::new(0) };
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let most = pool.install(|| {
            (0..32u32)
                .into_par_iter()
                .map(|_| {
                    let d = DEPTH.with(|c| {
                        c.set(c.get() + 1);
                        c.get()
                    });
                    on_big_stacks(|| std::thread::sleep(std::time::Duration::from_millis(2)));
                    DEPTH.with(|c| c.set(c.get() - 1));
                    d
                })
                .max()
        });
        assert_eq!(most, Some(1));
    }

    #[test]
    fn deep_recursion_fits() {
        fn down(n: u32) -> u32 {
            let pad = std::hint::black_box([n as u8; 256]);
            if n == 0 { u32::from(pad[0]) } else { down(n - 1) + u32::from(pad[1] & 1) }
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(256 << 10).build().unwrap();
        let r = pool.install(|| on_big_stacks(|| down(100_000)));
        assert_eq!(r, 50_000);
    }

    /// Workers whose stacks the address space cannot hold stop the run with `NoThreads`, an
    /// unwind the CLI reports as a plain error; the job never runs on a smaller stack.
    #[test]
    fn no_room_is_no_threads() {
        thread_local! {
            static FLAG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        let r = std::panic::catch_unwind(|| super::big_pool(2, 1 << 62, &FLAG).install(|| 1));
        let e = r.expect_err("a pool of 4 EiB stacks");
        assert!(e.downcast_ref::<super::NoThreads>().is_some());
    }
}
