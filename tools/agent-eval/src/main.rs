//! `agent-eval --stave BIN --answers DIR [--tasks DIR] [--answer-file NAME] [--out FILE]
//! [--work DIR] [--only A,B] [--jobs N] [--timeout SECS] [--require-all]`: scores the answers
//! `DIR/<task>/NAME` (default `answer.stv`) against `tests/agent-eval`, prints the text table,
//! and writes the JSON result to FILE (default `agent-eval.json`). Exit 0 when scored; with
//! `--require-all`, 1 unless every task runs correctly; 2 on a usage or setup error.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

const USAGE: &str = "usage: agent-eval --stave BIN --answers DIR [--tasks DIR] [--answer-file NAME] \
                     [--out FILE] [--work DIR] [--only A,B] [--jobs N] [--timeout SECS] [--require-all]";

fn fail(msg: &str) -> ExitCode {
    eprintln!("agent-eval: {msg}\n{USAGE}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut stave: Option<PathBuf> = None;
    let mut answers: Option<PathBuf> = None;
    let mut tasks = PathBuf::from("tests/agent-eval");
    let mut answer_file: Option<String> = None;
    let mut out = PathBuf::from("agent-eval.json");
    let mut work: Option<PathBuf> = None;
    let mut only: Vec<String> = Vec::new();
    let mut jobs: Option<usize> = None;
    let mut timeout: Option<u64> = None;
    let mut require_all = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().filter(|v| !v.starts_with("--"));
        match a.as_str() {
            "--stave" => stave = val().map(PathBuf::from),
            "--answers" => answers = val().map(PathBuf::from),
            "--tasks" => match val() {
                Some(v) => tasks = PathBuf::from(v),
                None => return fail("--tasks needs a directory"),
            },
            "--answer-file" => answer_file = val(),
            "--out" => match val() {
                Some(v) => out = PathBuf::from(v),
                None => return fail("--out needs a file"),
            },
            "--work" => work = val().map(PathBuf::from),
            "--only" => match val() {
                Some(v) => only = v.split(',').map(str::to_string).collect(),
                None => return fail("--only needs task names"),
            },
            "--jobs" => match val().map(|v| v.parse::<usize>()) {
                Some(Ok(n)) if n > 0 => jobs = Some(n),
                _ => return fail("--jobs needs a positive integer"),
            },
            "--timeout" => match val().map(|v| v.parse::<u64>()) {
                Some(Ok(n)) if n > 0 => timeout = Some(n),
                _ => return fail("--timeout needs a positive number of seconds"),
            },
            "--require-all" => require_all = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            _ => return fail(&format!("unexpected argument {a:?}")),
        }
    }
    let (Some(stave), Some(answers)) = (stave, answers) else {
        return fail("--stave and --answers are required");
    };
    let work =
        work.unwrap_or_else(|| std::env::temp_dir().join(format!("agent-eval-{}", std::process::id())));
    let mut opts = agent_eval::Options::new(stave, tasks, answers, work);
    if let Some(f) = answer_file {
        opts.answer_file = f;
    }
    opts.only = only;
    if let Some(n) = jobs {
        opts.jobs = n;
    }
    if let Some(s) = timeout {
        opts.timeout = Duration::from_secs(s);
    }
    let report = match agent_eval::score(&opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("agent-eval: {e}");
            return ExitCode::from(2);
        }
    };
    print!("{}", report.table());
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty());
    if let Err(e) =
        parent.map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&out, report.json()))
    {
        eprintln!("agent-eval: {}: {e}", out.display());
        return ExitCode::from(2);
    }
    println!("result: {}", out.display());
    let t = report.total();
    if require_all && t.run_ok != t.tasks {
        eprintln!("agent-eval: {} of {} tasks not run correctly", t.tasks - t.run_ok, t.tasks);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
