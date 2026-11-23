//! agent-eval: scores a directory of Stave answers against the task set in `tests/agent-eval/`.
//!
//! A task is a directory `<tasks>/<name>/` holding `task.md` (the whole prompt an agent gets
//! besides `LANG.md`) and `expected.stdout`; `<tasks>/index.txt` lists every task once with its
//! difficulty and feature areas. An answer is the file `<answers>/<name>/<answer file>`
//! (default `answer.stv`). Who wrote the answers is outside the scorer.
//!
//! Each answer gets one try: `stave check` (verdict, and the first error code on failure), then
//! `stave build --release`, then one run of the executable, compared byte for byte with
//! `expected.stdout` (exit status 0 required). Totals: the first-try check rate and run-correct
//! rate over all listed tasks (a missing answer counts as a failure of both), the check failures
//! per diagnostic code (top 10), and the answer size in tokens, `ceil(bytes / 4)` as in
//! `lang-budget`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub const DEFAULT_ANSWER_FILE: &str = "answer.stv";
const INDEX_FILE: &str = "index.txt";
const BYTES_PER_TOKEN: usize = 4;
const TOP_CODES: usize = 10;
pub const DIFFICULTIES: [&str; 3] = ["easy", "medium", "hard"];
pub const AREAS: [&str; 15] = [
    "control",
    "numeric",
    "adt",
    "patterns",
    "generics",
    "views",
    "std",
    "errors",
    "closures",
    "drops",
    "threads",
    "simd",
    "contracts",
    "arena",
    "statics",
];

/// One line of `index.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub name: String,
    pub difficulty: String,
    pub areas: Vec<String>,
}

/// Reads and validates `<tasks>/index.txt`: known difficulty and areas, each name once, and
/// every listed directory holding `task.md` and `expected.stdout`, every task directory listed.
pub fn read_index(tasks: &Path) -> Result<Vec<Task>, String> {
    let path = tasks.join(INDEX_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out: Vec<Task> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = || format!("{}:{}", path.display(), i + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        let [name, difficulty, areas] = f[..] else {
            return Err(format!("{}: want `<dir> <difficulty> <areas>`", at()));
        };
        if !DIFFICULTIES.contains(&difficulty) {
            return Err(format!("{}: unknown difficulty {difficulty:?}", at()));
        }
        let areas: Vec<String> = areas.split(',').map(str::to_string).collect();
        if let Some(a) = areas.iter().find(|a| !AREAS.contains(&a.as_str())) {
            return Err(format!("{}: unknown area {a:?}", at()));
        }
        if out.iter().any(|t| t.name == name) {
            return Err(format!("{}: task {name} listed twice", at()));
        }
        for f in ["task.md", "expected.stdout"] {
            if !tasks.join(name).join(f).is_file() {
                return Err(format!("{}: {name}/{f} missing", at()));
            }
        }
        out.push(Task { name: name.to_string(), difficulty: difficulty.to_string(), areas });
    }
    let rd = std::fs::read_dir(tasks).map_err(|e| format!("{}: {e}", tasks.display()))?;
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() && !out.iter().any(|t| t.name == n) {
            return Err(format!("{}: directory {n} is not listed in {INDEX_FILE}", tasks.display()));
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct Options {
    /// The `stave` binary.
    pub stave: PathBuf,
    /// The task directory (`tests/agent-eval`).
    pub tasks: PathBuf,
    /// The answer directory: `<answers>/<task>/<answer_file>`.
    pub answers: PathBuf,
    pub answer_file: String,
    /// Scratch directory for executables; created, never removed.
    pub work: PathBuf,
    /// Only these tasks (empty: all).
    pub only: Vec<String>,
    pub jobs: usize,
    /// Per compiler call and per run.
    pub timeout: Duration,
}

impl Options {
    pub fn new(stave: PathBuf, tasks: PathBuf, answers: PathBuf, work: PathBuf) -> Options {
        let jobs = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
        Options {
            stave,
            tasks,
            answers,
            answer_file: DEFAULT_ANSWER_FILE.to_string(),
            work,
            only: Vec::new(),
            jobs,
            timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Check {
    Ok,
    /// Exit 1: the first error diagnostic's code (`none` when the output held no code).
    Fail(String),
    Missing,
    /// Any other exit (3 is an internal compiler error), a signal, or a timeout.
    Crash(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Build {
    Ok,
    /// The first error code, or `exit <n>` (2: a construct the back end cannot build yet).
    Fail(String),
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Run {
    Ok,
    /// Exit 0, other stdout.
    Wrong,
    /// A non-zero exit (101: a panic), a signal, or a timeout.
    Exit(String),
    Skipped,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub task: Task,
    pub check: Check,
    pub build: Build,
    pub run: Run,
    /// `ceil(bytes / 4)` of the answer; None when it is missing.
    pub tokens: Option<usize>,
    /// The first lines of the failing step's diagnostics or output, for the reader.
    pub detail: String,
}

pub struct Report {
    pub outcomes: Vec<Outcome>,
}

struct Proc {
    /// None: killed by a signal or by the timeout.
    code: Option<i32>,
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_proc(cmd: &mut Command, timeout: Duration) -> Result<Proc, String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{:?}: {e}", cmd.get_program()))?;
    let drain = |r: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut b);
            }
            b
        })
    };
    let out = drain(child.stdout.take().map(|r| Box::new(r) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|r| Box::new(r) as Box<dyn Read + Send>));
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if start.elapsed() > timeout {
            timed_out = true;
            let _ = child.kill();
            break child.wait().map_err(|e| e.to_string())?;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Ok(Proc {
        code: if timed_out { None } else { status.code() },
        timed_out,
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
    })
}

fn exit_text(p: &Proc) -> String {
    match p.code {
        _ if p.timed_out => "timeout".to_string(),
        Some(c) => format!("exit {c}"),
        None => "signal".to_string(),
    }
}

/// The value of `"key": "..."` in one JSON object line (no escapes in the values read here).
fn json_str<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\"");
    let at = line.find(&pat)? + pat.len();
    let rest = line[at..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}

/// The code and message of the first error diagnostic in a `--diagnostics json` stream.
fn first_error(stderr: &[u8]) -> Option<(String, String)> {
    String::from_utf8_lossy(stderr)
        .lines()
        .filter(|l| json_str(l, "kind") == Some("diagnostic") && json_str(l, "severity") == Some("error"))
        .find_map(|l| {
            Some((json_str(l, "code")?.to_string(), json_str(l, "message").unwrap_or("").to_string()))
        })
}

fn first_error_code(stderr: &[u8]) -> Option<String> {
    first_error(stderr).map(|(c, _)| c)
}

/// The first error's code and message, else the first lines of the output.
fn failure_detail(stderr: &[u8]) -> String {
    first_error(stderr).map_or_else(|| head(stderr, 4), |(c, m)| format!("{c}: {m}"))
}

fn head(b: &[u8], lines: usize) -> String {
    String::from_utf8_lossy(b).lines().take(lines).collect::<Vec<_>>().join("\n")
}

pub fn tokens(bytes: usize) -> usize {
    bytes.div_ceil(BYTES_PER_TOKEN)
}

fn score_one(opts: &Options, task: &Task) -> Result<Outcome, String> {
    let mut o = Outcome {
        task: task.clone(),
        check: Check::Missing,
        build: Build::Skipped,
        run: Run::Skipped,
        tokens: None,
        detail: String::new(),
    };
    let answer = opts.answers.join(&task.name).join(&opts.answer_file);
    let Ok(src) = std::fs::read(&answer) else { return Ok(o) };
    o.tokens = Some(tokens(src.len()));
    let work = opts.work.join(&task.name);
    std::fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let stave = |args: &[&str]| {
        let mut c = Command::new(&opts.stave);
        c.args(args).arg(&answer).current_dir(&work);
        run_proc(&mut c, opts.timeout)
    };

    let p = stave(&["check", "--compiler", "cpu", "--diagnostics", "json"])?;
    o.check = match p.code {
        Some(0) => Check::Ok,
        Some(1) => Check::Fail(first_error_code(&p.stderr).unwrap_or_else(|| "none".to_string())),
        _ => Check::Crash(exit_text(&p)),
    };
    if o.check != Check::Ok {
        o.detail = failure_detail(&p.stderr);
        return Ok(o);
    }

    let exe = work.join("prog");
    let _ = std::fs::remove_file(&exe);
    let exe_s = exe.to_string_lossy().into_owned();
    let p = stave(&["build", "--release", "--compiler", "cpu", "--diagnostics", "json", "-o", &exe_s])?;
    if p.code != Some(0) {
        o.build = Build::Fail(first_error_code(&p.stderr).unwrap_or_else(|| exit_text(&p)));
        o.detail = failure_detail(&p.stderr);
        return Ok(o);
    }
    o.build = Build::Ok;

    let expected = std::fs::read(opts.tasks.join(&task.name).join("expected.stdout"))
        .map_err(|e| format!("{}: {e}", task.name))?;
    let p = run_proc(Command::new(&exe).current_dir(&work), opts.timeout)?;
    o.run = match p.code {
        Some(0) if p.stdout == expected => Run::Ok,
        Some(0) => Run::Wrong,
        _ => Run::Exit(exit_text(&p)),
    };
    if o.run != Run::Ok {
        let diff =
            expected.split(|&b| b == b'\n').zip(p.stdout.split(|&b| b == b'\n')).position(|(a, b)| a != b);
        o.detail = match (&o.run, diff) {
            (Run::Wrong, Some(n)) => format!(
                "line {}: want {:?}, got {:?}",
                n + 1,
                head(expected.split(|&b| b == b'\n').nth(n).unwrap_or_default(), 1),
                head(p.stdout.split(|&b| b == b'\n').nth(n).unwrap_or_default(), 1)
            ),
            (Run::Wrong, None) => format!("{} bytes, want {}", p.stdout.len(), expected.len()),
            _ => head(&p.stderr, 4),
        };
    }
    Ok(o)
}

/// Scores every listed task (or `opts.only`), `opts.jobs` at a time, in index order.
pub fn score(opts: &Options) -> Result<Report, String> {
    // Every command runs in its task's work directory, so no path may stay relative; a bare
    // `stave` is looked up on PATH.
    let abs = |p: &Path| std::path::absolute(p).map_err(|e| format!("{}: {e}", p.display()));
    let mut o = opts.clone();
    o.tasks = abs(&opts.tasks)?;
    o.answers = abs(&opts.answers)?;
    o.work = abs(&opts.work)?;
    if opts.stave.components().count() > 1 {
        o.stave = abs(&opts.stave)?;
    }
    let opts = &o;
    let mut tasks = read_index(&opts.tasks)?;
    if !opts.only.is_empty() {
        if let Some(n) = opts.only.iter().find(|n| !tasks.iter().any(|t| &t.name == *n)) {
            return Err(format!("no task {n}"));
        }
        tasks.retain(|t| opts.only.contains(&t.name));
    }
    if !opts.answers.is_dir() {
        return Err(format!("{}: not a directory", opts.answers.display()));
    }
    std::fs::create_dir_all(&opts.work).map_err(|e| format!("{}: {e}", opts.work.display()))?;
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Result<Outcome, String>>>> = Mutex::new(vec![None; tasks.len()]);
    std::thread::scope(|s| {
        for _ in 0..opts.jobs.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(t) = tasks.get(i) else { break };
                    let r = score_one(opts, t);
                    slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(r);
                }
            });
        }
    });
    let mut outcomes = Vec::new();
    for r in slots.into_inner().unwrap_or_else(|e| e.into_inner()) {
        outcomes.push(r.ok_or("a task was not scored")??);
    }
    Ok(Report { outcomes })
}

/// Counts over a group of outcomes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub tasks: usize,
    pub answered: usize,
    pub check_ok: usize,
    pub run_ok: usize,
}

impl Tally {
    fn add(&mut self, o: &Outcome) {
        self.tasks += 1;
        self.answered += usize::from(o.check != Check::Missing);
        self.check_ok += usize::from(o.check == Check::Ok);
        self.run_ok += usize::from(o.run == Run::Ok);
    }
}

fn rate(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { n as f64 / d as f64 }
}

impl Report {
    pub fn total(&self) -> Tally {
        let mut t = Tally::default();
        self.outcomes.iter().for_each(|o| t.add(o));
        t
    }

    /// Tallies per key; a task with several areas counts in each.
    fn by(&self, keys: impl Fn(&Task) -> Vec<String>) -> BTreeMap<String, Tally> {
        let mut m: BTreeMap<String, Tally> = BTreeMap::new();
        for o in &self.outcomes {
            for k in keys(&o.task) {
                m.entry(k).or_default().add(o);
            }
        }
        m
    }

    pub fn by_difficulty(&self) -> Vec<(String, Tally)> {
        let m = self.by(|t| vec![t.difficulty.clone()]);
        DIFFICULTIES.iter().filter_map(|d| m.get(*d).map(|t| (d.to_string(), *t))).collect()
    }

    pub fn by_area(&self) -> Vec<(String, Tally)> {
        let m = self.by(|t| t.areas.clone());
        AREAS.iter().filter_map(|a| m.get(*a).map(|t| (a.to_string(), *t))).collect()
    }

    /// Check failures per diagnostic code, most frequent first (ties by code), at most 10.
    pub fn top_codes(&self) -> Vec<(String, usize)> {
        let mut m: BTreeMap<&str, usize> = BTreeMap::new();
        for o in &self.outcomes {
            if let Check::Fail(c) = &o.check {
                *m.entry(c.as_str()).or_default() += 1;
            }
        }
        let mut v: Vec<(String, usize)> = m.into_iter().map(|(c, n)| (c.to_string(), n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(TOP_CODES);
        v
    }

    /// (sum, mean) of the answer tokens over the answered and over the run-correct tasks.
    pub fn token_stats(&self) -> ((usize, f64), (usize, f64)) {
        let stat = |f: &dyn Fn(&Outcome) -> bool| {
            let v: Vec<usize> = self.outcomes.iter().filter(|o| f(o)).filter_map(|o| o.tokens).collect();
            let s: usize = v.iter().sum();
            (s, rate(s, v.len()))
        };
        (stat(&|_| true), stat(&|o| o.run == Run::Ok))
    }

    pub fn json(&self) -> String {
        let mut j = String::from("{\n  \"schema\": \"stave-agent-eval/v1\",\n  \"tasks\": [\n");
        for (i, o) in self.outcomes.iter().enumerate() {
            let (check, code) = match &o.check {
                Check::Ok => ("ok", None),
                Check::Fail(c) => ("fail", Some(c.as_str())),
                Check::Missing => ("missing", None),
                Check::Crash(c) => ("crash", Some(c.as_str())),
            };
            let (build, build_why) = match &o.build {
                Build::Ok => ("ok", None),
                Build::Fail(c) => ("fail", Some(c.as_str())),
                Build::Skipped => ("skipped", None),
            };
            let (run, run_why) = match &o.run {
                Run::Ok => ("ok", None),
                Run::Wrong => ("wrong", None),
                Run::Exit(e) => ("exit", Some(e.as_str())),
                Run::Skipped => ("skipped", None),
            };
            let areas: Vec<String> = o.task.areas.iter().map(|a| esc(a)).collect();
            j += &format!(
                "    {{\"name\": {}, \"difficulty\": {}, \"areas\": [{}], \"check\": {}, \"code\": {}, \
                 \"build\": {}, \"build_detail\": {}, \"run\": {}, \"run_detail\": {}, \"tokens\": {}, \
                 \"detail\": {}}}{}\n",
                esc(&o.task.name),
                esc(&o.task.difficulty),
                areas.join(", "),
                esc(check),
                opt(code),
                esc(build),
                opt(build_why),
                esc(run),
                opt(run_why),
                o.tokens.map_or("null".to_string(), |t| t.to_string()),
                esc(&o.detail),
                if i + 1 < self.outcomes.len() { "," } else { "" }
            );
        }
        let t = self.total();
        let ((ts, tm), (ss, sm)) = self.token_stats();
        let codes: Vec<String> = self
            .top_codes()
            .iter()
            .map(|(c, n)| format!("{{\"code\": {}, \"count\": {n}}}", esc(c)))
            .collect();
        let groups = |v: Vec<(String, Tally)>| {
            v.iter().map(|(k, t)| format!("{}: {}", esc(k), tally_json(t))).collect::<Vec<_>>().join(", ")
        };
        j += &format!(
            "  ],\n  \"totals\": {},\n  \"first_try_check_rate\": {:.4},\n  \"first_try_run_rate\": {:.4},\n  \
             \"top_codes\": [{}],\n  \"tokens\": {{\"answered_sum\": {ts}, \"answered_mean\": {tm:.1}, \
             \"solved_sum\": {ss}, \"solved_mean\": {sm:.1}}},\n  \"by_difficulty\": {{{}}},\n  \
             \"by_area\": {{{}}}\n}}\n",
            tally_json(&t),
            rate(t.check_ok, t.tasks),
            rate(t.run_ok, t.tasks),
            codes.join(", "),
            groups(self.by_difficulty()),
            groups(self.by_area()),
        );
        j
    }

    pub fn table(&self) -> String {
        let mut s = format!(
            "{:<20} {:<6} {:<12} {:<10} {:<14} {:>6}\n",
            "task", "diff", "check", "build", "run", "tokens"
        );
        for o in &self.outcomes {
            let check = match &o.check {
                Check::Ok => "ok".to_string(),
                Check::Fail(c) => c.clone(),
                Check::Missing => "missing".to_string(),
                Check::Crash(c) => format!("crash {c}"),
            };
            let build = match &o.build {
                Build::Ok => "ok".to_string(),
                Build::Fail(c) => c.clone(),
                Build::Skipped => "-".to_string(),
            };
            let run = match &o.run {
                Run::Ok => "ok".to_string(),
                Run::Wrong => "wrong".to_string(),
                Run::Exit(e) => e.clone(),
                Run::Skipped => "-".to_string(),
            };
            let tok = o.tokens.map_or("-".to_string(), |t| t.to_string());
            s += &format!(
                "{:<20} {:<6} {:<12} {:<10} {:<14} {:>6}\n",
                o.task.name, o.task.difficulty, check, build, run, tok
            );
        }
        let t = self.total();
        let ((_, tm), (_, sm)) = self.token_stats();
        s += &format!(
            "first try: check {}/{} ({:.1}%), run correct {}/{} ({:.1}%), answered {}\n",
            t.check_ok,
            t.tasks,
            100.0 * rate(t.check_ok, t.tasks),
            t.run_ok,
            t.tasks,
            100.0 * rate(t.run_ok, t.tasks),
            t.answered
        );
        for (k, t) in self.by_difficulty() {
            s += &format!("  {k:<9} check {}/{} run {}/{}\n", t.check_ok, t.tasks, t.run_ok, t.tasks);
        }
        let codes: Vec<String> = self.top_codes().iter().map(|(c, n)| format!("{c} x{n}")).collect();
        s += &format!(
            "check failures by code: {}\n",
            if codes.is_empty() { "none".to_string() } else { codes.join(", ") }
        );
        s += &format!("answer tokens (bytes / 4): mean {tm:.1} answered, {sm:.1} solved\n");
        s
    }
}

fn tally_json(t: &Tally) -> String {
    format!(
        "{{\"tasks\": {}, \"answered\": {}, \"check_ok\": {}, \"run_ok\": {}}}",
        t.tasks, t.answered, t.check_ok, t.run_ok
    )
}

fn opt(s: Option<&str>) -> String {
    s.map_or("null".to_string(), esc)
}

/// A JSON string literal.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_matches_lang_budget() {
        assert_eq!(tokens(0), 0);
        assert_eq!(tokens(4), 1);
        assert_eq!(tokens(5), 2);
    }

    #[test]
    fn first_error_code_skips_warnings() {
        let s = b"{\"kind\":\"diagnostic\",\"code\":\"E1411\",\"severity\":\"warning\",\"message\":\"m\"}\n\
                  {\"kind\":\"diagnostic\",\"code\":\"E0202\",\"severity\":\"error\",\"message\":\"m\"}\n";
        assert_eq!(first_error_code(s).as_deref(), Some("E0202"));
        assert_eq!(first_error_code(b"compiled on CPU\n"), None);
    }

    #[test]
    fn json_escapes() {
        assert_eq!(esc("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
    }

    #[test]
    fn the_task_index_is_valid() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/agent-eval");
        let ts = read_index(&dir).unwrap();
        assert_eq!(ts.len(), 50);
        let n = |d: &str| ts.iter().filter(|t| t.difficulty == d).count();
        assert_eq!((n("easy"), n("medium"), n("hard")), (10, 25, 15));
        for a in AREAS {
            assert!(ts.iter().any(|t| t.areas.iter().any(|x| x == a)), "no task covers area {a}");
        }
    }
}
