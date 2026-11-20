//! gen-corpus: deterministic, semantically plausible Stave v0.4 packages of a given size, for the
//! stage budgets and the stage1-vs-stage0 dump comparison.
//!
//! Output is byte-identical per `(lines, seed, mix)`, whatever the thread count: phase 1 plans
//! every module (paths, imports, item signatures, exact line budgets) from per-module random
//! streams; phase 2 generates each module's text from the plans alone, in parallel.
//!
//! `--pair` instead writes one smaller-featured program twice, in Stave and in C99, with the
//! same fns, control flow, and checksum (`pair`): the compile and run time benchmark input.
//!
//! [`classes`] generates single-file programs around borrow-checker constructs, each with the
//! verdict the checker must give: the input of the `fuzz-classes` oracle.

mod body;
pub mod classes;
mod emit;
pub mod mix;
mod model;
mod names;
mod pair;
mod plan;
mod rng;

pub use mix::Mix;
pub use pair::{PairOptions, PairSummary, PairTexts, generate_pair, pair_texts};

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Generation request.
pub struct Options {
    pub lines: u64,
    pub seed: u64,
    pub out: PathBuf,
    pub mix: Mix,
    /// Worker threads (0 = available parallelism). Does not change the output.
    pub threads: usize,
    /// Replace `stave.toml` and `src/` of an existing output directory.
    pub force: bool,
}

/// What was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    /// Lines of all `.stv` files.
    pub lines: u64,
    pub bytes: u64,
    /// Modules besides `main`.
    pub modules: u32,
    pub fns: u64,
    pub items: u64,
}

fn prepare(out: &Path, force: bool) -> Result<(), String> {
    if out.exists() {
        let mut entries = std::fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
        if entries.next().is_some() {
            if !force {
                return Err(format!(
                    "{}: not empty (use --force to replace stave.toml and src/)",
                    out.display()
                ));
            }
            let src = out.join("src");
            if src.exists() {
                std::fs::remove_dir_all(&src).map_err(|e| format!("{}: {e}", src.display()))?;
            }
            let m = out.join("stave.toml");
            if m.exists() {
                std::fs::remove_file(&m).map_err(|e| format!("{}: {e}", m.display()))?;
            }
        }
    }
    std::fs::create_dir_all(out.join("src")).map_err(|e| format!("{}: {e}", out.display()))
}

fn threads(n: usize) -> usize {
    if n > 0 { n } else { std::thread::available_parallelism().map_or(1, |x| x.get()).min(64) }
}

/// Runs `f(k)` for `k in 0..n` on `t` threads; results in index order.
fn par_map<T: Send, F: Fn(u32) -> Result<T, String> + Sync>(
    n: u32,
    t: usize,
    f: F,
) -> Result<Vec<T>, String> {
    let next = AtomicUsize::new(0);
    let err: Mutex<Option<String>> = Mutex::new(None);
    let parts: Vec<Vec<(u32, T)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t.max(1))
            .map(|_| {
                sc.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= n as usize {
                            break;
                        }
                        match f(k as u32) {
                            Ok(v) => got.push((k as u32, v)),
                            Err(e) => {
                                if let Ok(mut g) = err.lock() {
                                    g.get_or_insert(e);
                                }
                                next.store(n as usize, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    got
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });
    if let Some(e) = err.into_inner().ok().flatten() {
        return Err(e);
    }
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for (k, v) in parts.into_iter().flatten() {
        slots[k as usize] = Some(v);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(k, v)| v.ok_or_else(|| format!("internal: module {k} missing")))
        .collect()
}

fn module_file(out: &Path, path: &[String]) -> PathBuf {
    let mut f = out.join("src");
    for s in path {
        f.push(s);
    }
    f.set_extension("stv");
    f
}

/// Writes the package; returns what was written.
pub fn generate(o: &Options) -> Result<Summary, String> {
    prepare(&o.out, o.force)?;
    let lay = plan::layout(o.lines, o.seed, &o.mix)?;
    let m = lay.paths.len() as u32;
    let t = threads(o.threads);
    let plans = par_map(m, t, |k| Ok(plan::plan_module(k, &lay, o.seed, &o.mix)))?;
    let mut dirs: Vec<PathBuf> =
        plans.iter().filter_map(|p| module_file(&o.out, &p.path).parent().map(Path::to_path_buf)).collect();
    dirs.sort_unstable();
    dirs.dedup();
    for d in &dirs {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let (lines, bytes) = (AtomicU64::new(0), AtomicU64::new(0));
    par_map(m, t, |k| {
        let text = emit::module_text(&plans, k, &o.mix, o.seed);
        let p = &plans[k as usize];
        let n = text.bytes().filter(|&b| b == b'\n').count() as u64;
        if n != u64::from(p.budget) {
            return Err(format!("internal: module {} has {n} lines, planned {}", p.dotted(), p.budget));
        }
        lines.fetch_add(n, Ordering::Relaxed);
        bytes.fetch_add(text.len() as u64, Ordering::Relaxed);
        let f = module_file(&o.out, &p.path);
        std::fs::write(&f, text).map_err(|e| format!("{}: {e}", f.display()))
    })?;
    let main = emit::main_text(&plans, &lay.sinks, lay.main_pad, o.lines, o.seed);
    lines.fetch_add(main.bytes().filter(|&b| b == b'\n').count() as u64, Ordering::Relaxed);
    bytes.fetch_add(main.len() as u64, Ordering::Relaxed);
    let mf = o.out.join("src/main.stv");
    std::fs::write(&mf, main).map_err(|e| format!("{}: {e}", mf.display()))?;
    let tf = o.out.join("stave.toml");
    std::fs::write(&tf, emit::manifest()).map_err(|e| format!("{}: {e}", tf.display()))?;
    Ok(Summary {
        lines: lines.into_inner(),
        bytes: bytes.into_inner(),
        modules: m,
        fns: plans.iter().map(|p| p.fns.len() as u64).sum(),
        items: plans.iter().map(|p| p.items.len() as u64).sum(),
    })
}
