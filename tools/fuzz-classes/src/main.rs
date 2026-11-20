//! `fuzz-classes`: runs the programs of `gen_corpus::classes` through the compiler and judges
//! each one.
//!
//! ```text
//! fuzz-classes [--seed N] [--count M] [--class C[,C..]] [--time-budget S] [--jobs J]
//!              [--timeout S] [--release] [--out DIR] [--repo DIR]
//! fuzz-classes --print --class C --seed N      the program of (C, N)
//! fuzz-classes --exec FILE [--release]         one program, the verdict and runs as lines
//! ```
//!
//! Program `k` of a run is class `k % classes` at seed `N + k / classes`. Each program is
//! compiled and run by a child process of this binary (`--exec`), so a compiler panic, abort, or
//! hang is caught and the run goes on. Findings:
//! - `ice`: the compiler panicked or died, or failed on a program without error diagnostics;
//! - `hang`: no answer within `--timeout`;
//! - `hole`: accepted, but the program must be rejected;
//! - `precision`: rejected with borrow diagnostics only (for `generic-sites`: also bound and
//!   view-kind ones, `classes::RULE_CODES`), but the program is valid;
//! - `generator`: rejected for another reason (the generator wrote invalid Stave), or a bad
//!   program rejected without any expected code;
//! - `mismatch`: accepted, and a native build prints or exits other than the interpreter;
//! - `trap`: accepted, and the run trapped (no generated program traps);
//! - `oracle`: accepted, and the interpreter in checking mode (`STAVE_INTERP_CHECK=1` in the
//!   environment) reported a memory error, or in borrow-tracking mode (`STAVE_INTERP_CHECK=borrow`)
//!   a violation of the borrow model or a `from` clause a call did not keep.
//!
//! Each finding's program is saved under `DIR/findings/<kind>/<class>-<seed>.stv` with the
//! reason in its head comment; `DIR/summary.txt` gets the table printed at the end,
//! `DIR/programs.tsv` one line per program (class, seed, expected and actual verdict, finding,
//! case labels). Exit 1 when
//! a finding other than `precision` was made; precision notes are counted in the summary line.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use gen_corpus::classes::{self, Class, Verdict};
use stave_diag::Severity;
use stave_driver::build::Failure;
use stave_driver::exec::{self, Ran, RunSpec};
use stave_driver::interp::{self, Limits};
use stave_driver::pipeline::{self, Goal, Options, Stage};
use stave_driver::shipped::Library;
use stave_ir::Module;
use stave_target::Triple;

const USAGE: &str =
    "usage: fuzz-classes [--seed N] [--count M] [--class C[,C..]] [--time-budget S] [--jobs J]
                    [--timeout S] [--release] [--out DIR] [--repo DIR]
       fuzz-classes --print --class C --seed N
       fuzz-classes --exec FILE [--release] [--repo DIR]";

/// Finding kinds that fail a run.
const FAILING: [&str; 7] = ["ice", "hang", "hole", "generator", "mismatch", "trap", "oracle"];

struct Args {
    seed: u64,
    count: Option<u64>,
    classes: Vec<Class>,
    budget: Option<Duration>,
    jobs: usize,
    timeout: Duration,
    release: bool,
    out: PathBuf,
    repo: PathBuf,
    print: bool,
    exec: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        seed: 0,
        count: None,
        classes: Class::ALL.to_vec(),
        budget: None,
        jobs: 4,
        timeout: Duration::from_secs(60),
        release: false,
        out: PathBuf::from("fuzz-out"),
        repo: Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        print: false,
        exec: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        let num = |name: &str, s: String| s.parse::<u64>().map_err(|e| format!("{name}: {e}"));
        match arg.as_str() {
            "--seed" => a.seed = num("--seed", val("--seed")?)?,
            "--count" => a.count = Some(num("--count", val("--count")?)?),
            "--class" => {
                let v = val("--class")?;
                a.classes = if v == "all" {
                    Class::ALL.to_vec()
                } else {
                    v.split(',')
                        .map(|c| Class::parse(c).ok_or_else(|| format!("unknown class {c:?}")))
                        .collect::<Result<_, _>>()?
                };
            }
            "--time-budget" => {
                a.budget = Some(Duration::from_secs(num("--time-budget", val("--time-budget")?)?))
            }
            "--jobs" => a.jobs = num("--jobs", val("--jobs")?)?.max(1) as usize,
            "--timeout" => a.timeout = Duration::from_secs(num("--timeout", val("--timeout")?)?),
            "--release" => a.release = true,
            "--out" | "-o" => a.out = PathBuf::from(val("--out")?),
            "--repo" => a.repo = PathBuf::from(val("--repo")?),
            "--print" => a.print = true,
            "--exec" => a.exec = Some(PathBuf::from(val("--exec")?)),
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if a.count.is_none() && a.budget.is_none() && !a.print && a.exec.is_none() {
        a.count = Some(1);
    }
    // A built program runs with its source's directory as the working directory, so its path
    // must not depend on this process's.
    for p in [Some(&mut a.out), a.exec.as_mut()].into_iter().flatten() {
        *p = std::path::absolute(&*p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(a)
}

fn main() -> ExitCode {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("fuzz-classes: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if let Some(file) = &a.exec {
        return exec_one(file, &a.repo, a.release);
    }
    if a.print {
        for &c in &a.classes {
            print!("{}", classes::generate(c, a.seed).text);
        }
        return ExitCode::SUCCESS;
    }
    run(&a)
}

// ---- child: one program ----

/// The per-program dynamic oracle on the checked IR of an accepted program: a report of a
/// run-time borrow or memory violation, or `None`. Memory errors of the interpreter's checking
/// mode are reported by the interpreter leg itself; nothing else is checked here yet.
fn dynamic_oracle(_m: &Module, _main: &str, _target: Triple) -> Option<String> {
    None
}

fn host_target() -> Triple {
    Triple::parse(if cfg!(target_arch = "aarch64") { "aarch64-linux" } else { "x86_64-linux" })
        .expect("stage0 target")
}

fn show(r: &Ran) -> String {
    format!(
        "exit {} stdout {:?} stderr {:?}",
        r.exit,
        String::from_utf8_lossy(&r.stdout),
        String::from_utf8_lossy(&r.stderr)
    )
}

/// Prints `rejected CODES`, `unsupported MSG`, `failure MSG`, or `accepted` followed by
/// `interp-error`, `oracle`, `trap`, `mismatch` lines, then `done`.
fn exec_one(file: &Path, repo: &Path, release: bool) -> ExitCode {
    let target = host_target();
    let lib = Library::in_dir(repo);
    let build = |release: bool| {
        let mut o = Options::new(Goal::Build).threads(Some(1));
        o.target = Some(target);
        o.release = release;
        pipeline::run(file, &lib, &o)
    };
    let out = match build(false) {
        Ok(o) => o,
        Err(e) => {
            println!("failure pipeline: {e:?}");
            println!("done");
            return ExitCode::SUCCESS;
        }
    };
    let mut codes: Vec<&str> =
        out.diagnostics.iter().filter(|d| d.severity() == Severity::Error).map(|d| d.code()).collect();
    codes.dedup();
    if !codes.is_empty() {
        println!("rejected {}", codes.join(" "));
        println!("done");
        return ExitCode::SUCCESS;
    }
    // A failure after the borrow checker comes after its verdict: the program was accepted.
    let checked = out.failure.as_ref().is_none_or(|f| f.stage() > Stage::Borrowck);
    if checked {
        println!("accepted");
    }
    match &out.failure {
        None => {}
        Some(Failure::Unsupported { message, .. }) => {
            println!("unsupported {message:?}");
            println!("done");
            return ExitCode::SUCCESS;
        }
        Some(f) => {
            println!("failure {:?}", f.to_string());
            println!("done");
            return ExitCode::SUCCESS;
        }
    }
    let (Some(m), Some(main), Some(img)) = (&out.module, out.main, &out.image) else {
        println!("failure \"no module, main, or image without diagnostics\"");
        println!("done");
        return ExitCode::SUCCESS;
    };
    let main_path = m.func(main).map(|f| f.path.clone()).unwrap_or_default();
    let limits = Limits { max_steps: 200_000_000, ..Limits::default() };
    let want = match interp::run(out.for_interp().unwrap_or(m), &main_path, target, &[], &[], limits) {
        Ok(r) => r.ran,
        // Checking mode ends the run at the first memory error, borrow tracking at the first
        // violation of the borrow model: a report, not a crash.
        Err(e) if e.starts_with("interp: memory error") || e.starts_with("interp: borrow ") => {
            println!("oracle {e:?}");
            println!("done");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            println!("interp-error {e:?}");
            println!("done");
            return ExitCode::SUCCESS;
        }
    };
    if let Some(r) = dynamic_oracle(m, &main_path, target) {
        println!("oracle {r:?}");
    }
    if want.exit != 0 {
        println!("trap {:?}", show(&want));
    }
    // The interpreter's status is the full `exit_group` argument; the kernel keeps 8 bits.
    let want = Ran { exit: want.exit & 0xff, ..want };
    if exec::can_run(target) {
        let mut legs = vec![("debug", img.clone())];
        if release {
            match build(true) {
                Ok(o) => match (o.image, o.failure) {
                    (Some(i), None) => legs.push(("release", i)),
                    (_, f) => println!("failure {:?}", format!("release: {f:?}")),
                },
                Err(e) => println!("failure {:?}", format!("release: {e:?}")),
            }
        }
        for (leg, img) in legs {
            let bin = file.with_extension(format!("{leg}.bin"));
            if let Err(e) = img.write_file(&bin) {
                println!("failure {:?}", format!("{}: {e}", bin.display()));
                continue;
            }
            let spec = RunSpec { args: &[], runner: None, cwd: file.parent(), inherit: false };
            let got = exec::run(&bin, &spec);
            let _ = std::fs::remove_file(&bin);
            match got {
                Ok(got) if got == want => {}
                Ok(got) => println!(
                    "mismatch {:?}",
                    format!("{leg}: native {} / interpreter {}", show(&got), show(&want))
                ),
                Err(e) => println!("mismatch {:?}", format!("{leg}: native did not run: {e}")),
            }
        }
    } else {
        println!("note {:?}", format!("{target} binaries cannot run here: interpreter only"));
    }
    println!("done");
    ExitCode::SUCCESS
}

// ---- parent: the loop ----

struct Judged {
    accepted: Option<bool>,
    /// The error codes of a rejection.
    codes: String,
    unsupported: bool,
    kind: Option<&'static str>,
    detail: String,
}

/// Runs the child on `file` and judges its answer against the expected verdict.
fn judge(a: &Args, p: &classes::Program, file: &Path) -> Judged {
    let exe = std::env::current_exe().expect("own path");
    let (so, se) = (file.with_extension("out"), file.with_extension("err"));
    let mut cmd = Command::new(exe);
    cmd.arg("--exec").arg(file).arg("--repo").arg(&a.repo);
    if a.release {
        cmd.arg("--release");
    }
    let spawn = (|| {
        cmd.stdin(Stdio::null())
            .stdout(std::fs::File::create(&so)?)
            .stderr(std::fs::File::create(&se)?)
            .spawn()
    })();
    let mut child = match spawn {
        Ok(c) => c,
        Err(e) => {
            return Judged {
                accepted: None,
                codes: String::new(),
                unsupported: false,
                kind: Some("ice"),
                detail: format!("spawn: {e}"),
            };
        }
    };
    let t0 = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t0.elapsed() > a.timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break None,
        }
    };
    let stdout = std::fs::read_to_string(&so).unwrap_or_default();
    let stderr = std::fs::read_to_string(&se).unwrap_or_default();
    let _ = std::fs::remove_file(&so);
    let _ = std::fs::remove_file(&se);
    let mut j = Judged {
        accepted: None,
        codes: String::new(),
        unsupported: false,
        kind: None,
        detail: String::new(),
    };
    let Some(status) = status else {
        j.kind = Some("hang");
        j.detail = format!("no answer within {} s", a.timeout.as_secs());
        return j;
    };
    if !stdout.lines().any(|l| l == "done") {
        let tail: Vec<&str> = stderr.lines().rev().take(12).collect();
        j.kind = Some("ice");
        j.detail =
            format!("compiler child {status}\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
        return j;
    }
    fn set(j: &mut Judged, k: &'static str, d: String) {
        if j.kind.is_none() {
            j.kind = Some(k);
            j.detail = d;
        }
    }
    for line in stdout.lines() {
        let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
        match head {
            "rejected" => {
                j.accepted = Some(false);
                j.codes = rest.to_string();
                let codes: Vec<&str> = rest.split_whitespace().collect();
                // The rule codes of the class: a valid program rejected with them is a
                // precision note, with others the generator wrote invalid Stave.
                let borrow_only = codes.iter().all(|c| {
                    c.starts_with("E05")
                        || c.starts_with("E04")
                        || (p.class == Class::GenericSites && classes::RULE_CODES.contains(c))
                });
                match p.expect {
                    Verdict::Accept if borrow_only => {
                        set(&mut j, "precision", format!("valid program rejected: {rest}"))
                    }
                    Verdict::Accept => set(&mut j, "generator", format!("valid program rejected: {rest}")),
                    Verdict::Reject if !codes.iter().any(|c| p.codes.contains(c)) && !borrow_only => set(
                        &mut j,
                        "generator",
                        format!("rejected with {rest}, expected one of {:?}", p.codes),
                    ),
                    Verdict::Reject => {}
                }
            }
            "accepted" => {
                j.accepted = Some(true);
                if p.expect == Verdict::Reject {
                    set(&mut j, "hole", format!("accepted; expected a rejection with one of {:?}", p.codes));
                }
            }
            "unsupported" => j.unsupported = true,
            "failure" | "interp-error" => set(&mut j, "ice", format!("{head} {rest}")),
            "mismatch" => set(&mut j, "mismatch", rest.to_string()),
            "trap" => set(&mut j, "trap", rest.to_string()),
            "oracle" => set(&mut j, "oracle", rest.to_string()),
            _ => {}
        }
    }
    j
}

#[derive(Default)]
struct Tally {
    programs: u64,
    accepted: u64,
    rejected: u64,
    unsupported: u64,
    expect_accept: u64,
    expect_reject: u64,
    findings: BTreeMap<&'static str, u64>,
}

fn save(out: &Path, p: &classes::Program, kind: &str, detail: &str) -> PathBuf {
    let dir = out.join("findings").join(kind);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}-{}.stv", p.class.name(), p.seed));
    let mut head = format!("// {kind}: class {} seed {} expect {:?}\n", p.class.name(), p.seed, p.expect);
    for c in &p.cases {
        let _ = writeln!(head, "// case {c}");
    }
    for l in detail.lines() {
        let _ = writeln!(head, "// {l}");
    }
    let _ = std::fs::write(&path, head + &p.text);
    path
}

fn run(a: &Args) -> ExitCode {
    let work = a.out.join("work");
    if let Err(e) = std::fs::create_dir_all(&work) {
        eprintln!("fuzz-classes: {}: {e}", work.display());
        return ExitCode::from(2);
    }
    let nc = a.classes.len() as u64;
    let total = a.count.map(|c| c * nc);
    let next = AtomicU64::new(0);
    let tallies: Mutex<BTreeMap<Class, Tally>> =
        Mutex::new(a.classes.iter().map(|&c| (c, Tally::default())).collect());
    let log = Mutex::new(String::new());
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..a.jobs {
            s.spawn(|| {
                loop {
                    if a.budget.is_some_and(|b| t0.elapsed() >= b) {
                        break;
                    }
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if total.is_some_and(|t| k >= t) {
                        break;
                    }
                    let class = a.classes[(k % nc) as usize];
                    let p = classes::generate(class, a.seed + k / nc);
                    let file = work.join(format!("{}-{}.stv", class.name(), p.seed));
                    if std::fs::write(&file, &p.text).is_err() {
                        break;
                    }
                    let j = judge(a, &p, &file);
                    let _ = std::fs::remove_file(&file);
                    let got = match j.accepted {
                        Some(true) => "accepted".to_string(),
                        Some(false) => format!("rejected {}", j.codes),
                        None => "-".to_string(),
                    };
                    let _ = writeln!(
                        log.lock().unwrap_or_else(|e| e.into_inner()),
                        "{}\t{}\t{:?}\t{got}\t{}\t{}",
                        class.name(),
                        p.seed,
                        p.expect,
                        j.kind.unwrap_or("-"),
                        p.cases.join(" | ")
                    );
                    let mut ts = tallies.lock().unwrap_or_else(|e| e.into_inner());
                    let t = ts.get_mut(&class).expect("class tally");
                    t.programs += 1;
                    match p.expect {
                        Verdict::Accept => t.expect_accept += 1,
                        Verdict::Reject => t.expect_reject += 1,
                    }
                    match j.accepted {
                        Some(true) => t.accepted += 1,
                        Some(false) => t.rejected += 1,
                        None => {}
                    }
                    t.unsupported += u64::from(j.unsupported);
                    if let Some(kind) = j.kind {
                        *t.findings.entry(kind).or_default() += 1;
                        let path = save(&a.out, &p, kind, &j.detail);
                        let first = j.detail.lines().next().unwrap_or("");
                        eprintln!("{kind}: {} seed {}: {first} ({})", class.name(), p.seed, path.display());
                    }
                }
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    let ts = tallies.into_inner().unwrap_or_else(|e| e.into_inner());
    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "{:<15} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8}  findings",
        "class", "programs", "accepted", "rate", "unsup", "exp-acc", "exp-rej"
    );
    let mut all = Tally::default();
    for (c, t) in &ts {
        let rate = if t.programs == 0 { 0.0 } else { 100.0 * t.accepted as f64 / t.programs as f64 };
        let f: Vec<String> = t.findings.iter().map(|(k, n)| format!("{k} {n}")).collect();
        let _ = writeln!(
            rep,
            "{:<15} {:>8} {:>8} {:>7.1}% {:>6} {:>8} {:>8}  {}",
            c.name(),
            t.programs,
            t.accepted,
            rate,
            t.unsupported,
            t.expect_accept,
            t.expect_reject,
            f.join(", ")
        );
        all.programs += t.programs;
        all.accepted += t.accepted;
        for (k, n) in &t.findings {
            *all.findings.entry(k).or_default() += n;
        }
    }
    let _ = writeln!(
        rep,
        "{} programs in {secs:.1} s ({:.2}/s, {} jobs), {} accepted, {} precision notes",
        all.programs,
        all.programs as f64 / secs.max(1e-9),
        a.jobs,
        all.accepted,
        all.findings.get("precision").copied().unwrap_or(0)
    );
    print!("{rep}");
    let _ = std::fs::write(a.out.join("summary.txt"), &rep);
    let _ = std::fs::write(a.out.join("programs.tsv"), log.into_inner().unwrap_or_else(|e| e.into_inner()));
    let failing = all.findings.iter().any(|(k, &n)| n > 0 && FAILING.contains(k));
    if failing { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
