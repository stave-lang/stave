//! `stave loops` (contracts/cli.md): the `tests/conformance/driver/loops_forms.expected.json`
//! golden, built against the pinned stand-in library of the dump goldens; the human form;
//! columns, vector views, and f-string holes with the shipped std; `--all`; errors and usage.
//!
//! `STAVE_BLESS=1` rewrites the golden from the current output (review the diff!).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn stave(args: &[&str], lib: Option<&Path>) -> (i32, String, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_stave"));
    c.args(args).env_remove("STAVE_THREADS").env_remove("STAVE_STD_DIR").env_remove("STAVE_SEMA_SHUFFLE");
    if let Some(l) = lib {
        c.env("STAVE_STD_DIR", l);
    }
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn pinned() -> PathBuf {
    repo().join("tests/conformance/dumps/_lib")
}

fn case() -> PathBuf {
    repo().join("tests/conformance/driver/loops_forms.stv")
}

fn json(l: &str) -> serde_json::Value {
    serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}"))
}

#[test]
fn json_golden() {
    let file = case();
    let (c, out, err) = stave(&["loops", file.to_str().unwrap(), "--json"], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    let golden = file.with_extension("expected.json");
    if std::env::var_os("STAVE_BLESS").is_some() {
        std::fs::write(&golden, &out).unwrap();
    }
    let want = std::fs::read_to_string(&golden).unwrap();
    assert_eq!(out, want, "--- got\n{out}--- want\n{want}");
    // Structure: one object per loop in (line, col) order, then the summary that counts them.
    let lines: Vec<&str> = out.lines().collect();
    let (summary, rows) = lines.split_last().unwrap();
    let mut last = (0, 0);
    let mut counts = std::collections::BTreeMap::new();
    for l in rows {
        let v = json(l);
        let key = (v["line"].as_u64().unwrap(), v["col"].as_u64().unwrap());
        assert!(last < key, "not in (line, col) order: {l}");
        last = key;
        *counts.entry(v["class"].as_str().unwrap().to_string()).or_insert(0u64) += 1;
    }
    let s = json(summary);
    assert_eq!(s["summary"], true, "{summary}");
    for class in ["chunk", "indexed", "sequential", "unread"] {
        assert_eq!(s[class].as_u64(), Some(counts.get(class).copied().unwrap_or(0)), "{class}: {summary}");
    }
}

#[test]
fn human_form() {
    let file = case();
    let (c, out, err) = stave(&["loops", file.to_str().unwrap()], Some(&pinned()));
    assert_eq!(c, 0, "{err}");
    for want in [
        " main.product range indexed",
        " main.plain range sequential not-indexed",
        " main.others while unread",
        " main.others loop unread",
        " main.others element sequential not-indexed",
        " main.others iter unread",
        " main.others chunk chunk",
        " main.Cols.total range indexed",
    ] {
        assert!(out.lines().any(|l| l.ends_with(want)), "{want}:\n{out}");
    }
    let rows = out.lines().count() - 1;
    let last = out.lines().last().unwrap();
    assert!(last.starts_with("loops: ") && last.ends_with(" unread"), "{last}");
    let total: usize = last["loops: ".len()..]
        .split(", ")
        .map(|f| f.split(' ').next().unwrap().parse::<usize>().unwrap())
        .sum();
    assert_eq!(total, rows, "{out}");
}

#[test]
fn columns_views_holes_and_all() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("loops_report");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("columns.stv");
    let src = "\
import std.io;
import std.soa;
import std.vec;
struct P {
    x: u32,
    y: u32,
}
fn columns(ps: &mut soa.Soa<P>) {
    for i in 0..ps.x.len() {
        ps.y[i] = ps.x[i];
    }
}
struct Q {
    xs: [u32; 4],
    n: u32,
}
fn holes(q: &Q, ys: &[u32]) {
    for i in 0..4usize {
        let y = ys[i] +% q.xs[i];
        io.print(f\"{y} {q.n}\\n\");
    }
}
struct R {
    a: vec.Vec<u32>,
    b: vec.Vec<u32>,
    c: vec.Vec<u32>,
}
fn views(r: &mut R, w: &vec.Vec<u32>) {
    for i in 0..w.len() {
        let x = r.a.as_slice()[i] +% w.as_slice()[i];
        r.a.as_mut_slice()[i] = x;
        r.b.as_mut_slice()[i] = r.c.as_slice()[i] +% r.c.as_slice().len() as% u32;
    }
}
struct Own {
    xs: [u32; 4],
}
impl Own {
    // a user method of the same name is a call like any other
    fn as_slice(&self) -> &[u32] from self {
        return &self.xs;
    }
}
fn own(o: &Own) -> u32 {
    let mut s = 0u32;
    for i in 0..4usize {
        s +%= o.as_slice()[i];
    }
    return s;
}
fn main() {
    let mut ps = soa.Soa::<P>.new();
    columns(&mut ps);
    holes(&Q.{ xs: [0u32; 4], n: 0 }, &ps.y);
    let mut r = R.{ a: vec.Vec::<u32>.new(), b: vec.Vec::<u32>.new(), c: vec.Vec::<u32>.new() };
    let w = vec.Vec::<u32>.new();
    views(&mut r, &w);
    let _ = own(&Own.{ xs: [0u32; 4] });
}
";
    std::fs::write(&file, src).unwrap();
    let path = file.to_str().unwrap();
    let (c, out, err) = stave(&["loops", path, "--json"], None);
    assert_eq!(c, 0, "{err}");
    let rows: Vec<serde_json::Value> = out.lines().map(json).collect();
    assert_eq!(rows.len(), 5, "{out}");
    assert_eq!(rows[0]["sources"], serde_json::json!(["ps.x", "&mut ps.y"]), "{out}");
    // an f-string hole uses the name its path starts at whole: `{q.n}` leaves no `q.xs`
    assert_eq!(rows[1]["sources"], serde_json::json!(["ys"]), "{out}");
    // both views of a vector are one place, spelled as its write; `len` on a view uses it whole
    assert_eq!(
        rows[2]["sources"],
        serde_json::json!(["w.as_slice()", "&mut r.a.as_mut_slice()", "&mut r.b.as_mut_slice()"]),
        "{out}"
    );
    assert_eq!((rows[3]["class"].as_str(), &rows[3]["sources"]), (Some("indexed"), &serde_json::json!([])));
    // `--all`: the loops of std and runtime too, theirs named by package
    let (c, all, err) = stave(&["loops", path, "--all", "--json"], None);
    assert_eq!(c, 0, "{err}");
    let rows: Vec<serde_json::Value> = all.lines().map(json).collect();
    let files: Vec<&str> = rows.iter().filter_map(|r| r["file"].as_str()).collect();
    assert!(files.iter().any(|f| f.starts_with("std/")), "{files:?}");
    assert_eq!(files.iter().filter(|f| f.ends_with("columns.stv")).count(), 4, "{files:?}");
    assert!(files.iter().all(|f| !f.starts_with('/') || f.ends_with("columns.stv")), "{files:?}");
}

#[test]
fn errors_and_usage() {
    let bad = repo().join("tests/conformance/effects/unknown_effect.stv");
    let (c, out, err) = stave(&["loops", bad.to_str().unwrap()], None);
    assert_eq!((c, out.as_str()), (1, ""), "{err}");
    assert!(err.contains("E1404"), "{err}");
    let (c, out, err) = stave(&["loops", bad.to_str().unwrap(), "--json"], None);
    assert_eq!((c, out.as_str()), (1, ""), "{err}");
    assert!(err.contains("\"code\":\"E1404\""), "{err}");
    assert_eq!(stave(&["loops", "--release"], None).0, 2);
    assert_eq!(stave(&["loops", "/no/such/package"], None).0, 2);
}
