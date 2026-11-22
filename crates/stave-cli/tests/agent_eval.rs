//! The agent-eval scorer (tools/agent-eval) on the task set `tests/agent-eval`: every
//! `reference.stv` checks, builds in release mode, and prints its `expected.stdout`, so the
//! scorer gives full marks; and a broken answer set gets the right verdicts and tallies.

use std::path::{Path, PathBuf};

use agent_eval::{Build, Check, Options, Run};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn host_runs() -> bool {
    cfg!(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("agent_eval").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn opts(answers: PathBuf, work: PathBuf) -> Options {
    Options::new(PathBuf::from(env!("CARGO_BIN_EXE_stave")), repo().join("tests/agent-eval"), answers, work)
}

#[test]
fn references_get_full_marks() {
    if !host_runs() {
        return;
    }
    let mut o = opts(repo().join("tests/agent-eval"), tmp("references"));
    o.answer_file = "reference.stv".to_string();
    let r = agent_eval::score(&o).unwrap();
    let t = r.total();
    let bad: Vec<String> = r
        .outcomes
        .iter()
        .filter(|o| o.run != Run::Ok)
        .map(|o| format!("{}: {:?} {:?} {:?}\n{}", o.task.name, o.check, o.build, o.run, o.detail))
        .collect();
    assert!(bad.is_empty(), "{}\n{}", bad.join("\n"), r.table());
    assert_eq!((t.tasks, t.check_ok, t.run_ok), (50, 50, 50));
}

#[test]
fn broken_answers_get_their_verdicts() {
    if !host_runs() {
        return;
    }
    let answers = tmp("broken");
    let put = |task: &str, src: &str| {
        std::fs::create_dir_all(answers.join(task)).unwrap();
        std::fs::write(answers.join(task).join("answer.stv"), src).unwrap();
    };
    // A literal without a type: E0202.
    put(
        "hello",
        "import std.io;\n\nfn main() -> i32 {\n    let a = 2;\n    io.print(\"x\\n\");\n    return 0;\n}\n",
    );
    put("fizzbuzz", "import std.io;\n\nfn main() -> i32 {\n    io.print(\"1\\n2\\n\");\n    return 0;\n}\n");
    let mut o = opts(answers, tmp("broken-work"));
    o.only = vec!["hello".to_string(), "fizzbuzz".to_string(), "collatz".to_string()];
    let r = agent_eval::score(&o).unwrap();
    let get = |n: &str| r.outcomes.iter().find(|o| o.task.name == n).unwrap();
    assert_eq!(get("hello").check, Check::Fail("E0202".to_string()), "{}", get("hello").detail);
    assert_eq!(get("hello").build, Build::Skipped);
    assert_eq!((&get("fizzbuzz").check, &get("fizzbuzz").build), (&Check::Ok, &Build::Ok));
    assert_eq!(get("fizzbuzz").run, Run::Wrong);
    assert_eq!((&get("collatz").check, get("collatz").tokens), (&Check::Missing, None));
    let t = r.total();
    assert_eq!((t.tasks, t.answered, t.check_ok, t.run_ok), (3, 2, 1, 0));
    assert_eq!(r.top_codes(), vec![("E0202".to_string(), 1)]);
    let j = r.json();
    assert!(j.contains("\"first_try_check_rate\": 0.3333"), "{j}");
    assert!(j.contains("\"first_try_run_rate\": 0.0000"), "{j}");
}
