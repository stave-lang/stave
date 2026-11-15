//! Lex + parse throughput (budget: >= 3M lines/s per thread).
//!
//! `cargo run --release -p stave-syntax --example syntax_bench [LINES]`
//!
//! The input is the LANG.md ```stave blocks (imports dropped after the first copy, so every item
//! parses) repeated to about LINES lines (default 2,000,000). Prints the median of 5 runs for
//! lex alone and for `parse_file` (lex + item cut + parse) on 1 thread and on all threads.

use stave_diag::FileId;
use stave_syntax::{lex, parse_file};
use std::time::Instant;

fn input(lines: usize) -> String {
    let md = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../LANG.md")).expect("LANG.md");
    let mut blocks = Vec::new();
    let mut rest = md.as_str();
    while let Some(s) = rest.find("```stave\n") {
        let body = &rest[s + "```stave\n".len()..];
        let e = body.find("```").expect("unclosed block");
        blocks.push(&body[..e]);
        rest = &body[e + 3..];
    }
    let mut imports = String::new();
    let mut body = String::new();
    for b in &blocks {
        for l in b.lines() {
            if l.starts_with("import ") {
                if !imports.contains(l) {
                    imports.push_str(l);
                    imports.push('\n');
                }
            } else {
                body.push_str(l);
                body.push('\n');
            }
        }
    }
    let per = body.lines().count().max(1);
    let mut s = imports;
    for _ in 0..lines.div_ceil(per) {
        s.push_str(&body);
    }
    s
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> f64 {
    median(
        (0..5)
            .map(|_| {
                let t = Instant::now();
                std::hint::black_box(f());
                t.elapsed().as_secs_f64()
            })
            .collect(),
    )
}

fn main() {
    let lines: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2_000_000);
    let src = input(lines);
    let n_lines = src.bytes().filter(|&b| b == b'\n').count();
    let bytes = src.len();
    let pf = parse_file(src.as_bytes(), FileId(0));
    assert!(pf.diagnostics.is_empty(), "bench input must parse: {:?}", pf.diagnostics.first());
    let tokens = pf.tokens.len();
    let nodes = pf.ast.node_count();
    println!(
        "input: {n_lines} lines, {:.1} MB, {tokens} tokens, {} items, {nodes} nodes",
        bytes as f64 / 1e6,
        pf.items.len()
    );
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");
    let t_lex = time(|| lex(src.as_bytes(), FileId(0)).tokens.len());
    let t_1 = one.install(|| time(|| parse_file(src.as_bytes(), FileId(0)).ast.node_count()));
    let t_all = time(|| parse_file(src.as_bytes(), FileId(0)).ast.node_count());
    let report = |what: &str, t: f64| {
        println!(
            "{what:<28} {:>8.2} ms  {:>7.2} M lines/s  {:>7.1} MB/s  {:>6.1} M tokens/s",
            t * 1e3,
            n_lines as f64 / t / 1e6,
            bytes as f64 / t / 1e6,
            tokens as f64 / t / 1e6
        );
    };
    report("lex, 1 thread", t_lex);
    report("lex + parse, 1 thread", t_1);
    report(&format!("lex + parse, {} threads", rayon::current_num_threads()), t_all);
}
