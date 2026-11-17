//! Worker threads for the recursive phases (type checking, lowering).
//!
//! Both descend expression trees recursively (up to [`crate::typeck::MAX_DEPTH`] levels), which
//! needs more stack than a default rayon worker has. They run on a pool of big-stack threads with
//! exactly as many workers as the caller's current rayon pool ([`rayon::current_num_threads`]:
//! the pool the call is `install`ed in, else the global one), so the caller's thread count
//! (`STAVE_THREADS`, `RAYON_NUM_THREADS`) is the phase's thread count. The pool lives for one call;
//! there is no private global pool. Nested calls run on the pool they are already on.
//!
//! A worker of another pool that waits in `install` keeps running its own pool's jobs meanwhile
//! (rayon's cross-pool wait), each on top of the waiting frames: callers that start one phase per
//! item of a `par_iter` would nest those items on one stack without bound. Such a caller waits on
//! a plain thread instead, which runs nothing else while it waits.

use std::cell::Cell;

/// Stack size of a worker (address space only; pages are touched on demand).
const STACK: usize = 512 << 20;

thread_local! {
    static BIG: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` on big-stack workers, as many as the current rayon pool has.
pub(crate) fn on_big_stacks<R: Send>(f: impl FnOnce() -> R + Send) -> R {
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
    if rayon::current_thread_index().is_some() { wait_off_pool(run) } else { run() }
}

/// Runs `f` on a plain thread and waits for it (a panic in `f` resumes here).
fn wait_off_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    std::thread::scope(|s| s.spawn(f).join()).unwrap_or_else(|e| std::panic::resume_unwind(e))
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
}
