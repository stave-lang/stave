//! The std.slice methods that must be branchless per element (`sum min max count fill
//! copy_from`) compile to checked IR whose loops branch only on the loop test: every loop
//! (natural loop of a back edge) holds exactly one conditional terminator, and its condition does
//! not depend on loaded element data or on the result of a fn-value call (a predicate). The program calls every method on every element
//! type, so each instance is lowered; it is checked for both CPU architectures.
//!
//! The search methods (`find find_byte find_any_byte eq any all`) return early once per vector
//! or chunk: one of their loops must show a data-dependent branch, which shows the check sees one.

use std::collections::HashSet;
use std::fmt::Write as _;

use stave_driver::exec::TempDir;
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_ir::{Block, Body, Module, Op, Operand, Term, Value, VecOp};
use stave_target::Triple;

const BRANCHLESS: &[&str] = &["sum", "min", "max", "count", "fill", "copy_from"];
const SEARCH: &[&str] = &["find", "find_byte", "find_any_byte", "eq", "any", "all"];

/// Element types and a literal of each.
const ELEMS: &[(&str, &str)] = &[
    ("u8", "0"),
    ("u16", "0"),
    ("u32", "0"),
    ("u64", "0"),
    ("usize", "0"),
    ("i8", "0"),
    ("i16", "0"),
    ("i32", "0"),
    ("i64", "0"),
    ("isize", "0"),
    ("f32", "0.0"),
    ("f64", "0.0"),
    ("bool", "false"),
];

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source() -> String {
    let mut s = String::new();
    let mut main = String::from("fn main() -> i32 {\n");
    for &(t, lit) in ELEMS {
        let numeric = t != "bool";
        let _ = writeln!(s, "fn use_{t}(xs: &[{t}], ys: &mut [{t}], x: {t}) {{");
        if numeric {
            s.push_str("    let a = xs.sum();\n    let b = xs.min();\n    let c = xs.max();\n");
        }
        if t == "u8" {
            s.push_str("    let d = xs.find_byte(x);\n    let e = xs.find_any_byte(xs);\n");
        }
        let _ = writeln!(s, "    let f = xs.find(x);\n    let g = xs.eq(ys);");
        let _ = writeln!(s, "    let h = xs.count(fn(v: {t}) -> bool {{ return v == x; }});");
        let _ = writeln!(s, "    let i = xs.any(fn(v: {t}) -> bool {{ return v == x; }});");
        let _ = writeln!(s, "    let j = xs.all(fn(v: {t}) -> bool {{ return v == x; }});");
        s.push_str("    ys.fill(x);\n    ys.copy_from(xs);\n}\n\n");
        let _ = writeln!(
            main,
            "    let a_{t}: [{t}; 1] = [{lit}];\n    let mut b_{t}: [{t}; 1] = [{lit}];\n    use_{t}(&a_{t}[0..1], &mut b_{t}[0..1], {lit});"
        );
    }
    main.push_str("    return 0;\n}\n");
    s.push_str(&main);
    s
}

/// Values that depend on element data: loads, fn-value calls, and everything computed from them
/// (through instructions and block arguments), to a fixed point. Direct calls are std helpers
/// that compute slices and indices from their arguments.
fn data_values(b: &Body) -> HashSet<Value> {
    let mut t: HashSet<Value> = HashSet::new();
    let tainted = |t: &HashSet<Value>, o: &Operand| matches!(o, Operand::Value(v) if t.contains(v));
    loop {
        let before = t.len();
        for (i, op) in b.insts.op.iter().enumerate() {
            let Some(r) = b.insts.result[i] else { continue };
            let source = matches!(
                op,
                Op::Load
                    | Op::CallInd
                    | Op::Vec(VecOp::Load | VecOp::LoadAligned | VecOp::LoadMasked | VecOp::Gather)
            );
            if source || b.args(i as u32).iter().any(|o| tainted(&t, o)) {
                t.insert(r);
            }
        }
        for k in 0..b.blocks.len() {
            for tg in b.successors(Block(k as u32)) {
                let args = b.target_args(&tg);
                for (p, a) in b.params(tg.block).iter().zip(args) {
                    if tainted(&t, a) {
                        t.insert(*p);
                    }
                }
            }
        }
        if t.len() == before {
            return t;
        }
    }
}

/// The natural loops of `b`: for each back edge (found by DFS from the entry) the header and
/// every block that reaches the edge's source without passing the header.
fn loops(b: &Body) -> Vec<Vec<usize>> {
    let n = b.blocks.len();
    let succ: Vec<Vec<usize>> =
        (0..n).map(|k| b.successors(Block(k as u32)).iter().map(|t| t.block.index()).collect()).collect();
    let mut pred = vec![Vec::new(); n];
    for (k, ss) in succ.iter().enumerate() {
        for &s in ss {
            pred[s].push(k);
        }
    }
    // Iterative DFS; state 1 = on the stack, 2 = done.
    let mut state = vec![0u8; n];
    let mut back = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    state[0] = 1;
    while let Some(top) = stack.last_mut() {
        let v = top.0;
        if top.1 < succ[v].len() {
            let s = succ[v][top.1];
            top.1 += 1;
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => back.push((v, s)),
                _ => {}
            }
        } else {
            state[v] = 2;
            stack.pop();
        }
    }
    back.into_iter()
        .map(|(src, head)| {
            let mut body = vec![head];
            let mut work = vec![src];
            while let Some(x) = work.pop() {
                if !body.contains(&x) {
                    body.push(x);
                    work.extend(pred[x].iter().copied());
                }
            }
            body
        })
        .collect()
}

/// Per loop: (conditional terminators, of which data-dependent).
fn loop_branches(b: &Body) -> Vec<(usize, usize)> {
    let data = data_values(b);
    loops(b)
        .iter()
        .map(|l| {
            let mut conds = 0;
            let mut on_data = 0;
            for &k in l {
                let c = match b.blocks[k].term {
                    Term::Cbr { cond, .. } => cond,
                    Term::Switch { scrut, .. } => scrut,
                    _ => continue,
                };
                conds += 1;
                if matches!(c, Operand::Value(v) if data.contains(&v)) {
                    on_data += 1;
                }
            }
            (conds, on_data)
        })
        .collect()
}

fn lower(target: &str) -> Module {
    let tmp = TempDir::new("combinators-branchless").unwrap();
    let p = tmp.path().join("main.stv");
    std::fs::write(&p, source()).unwrap();
    let mut o = Options::new(Goal::Check).threads(Some(4));
    o.lower = true;
    o.target = Some(Triple::parse(target).unwrap());
    let out = pipeline::run(&p, &Library::in_dir(&repo()), &o).unwrap();
    assert!(!out.has_errors(), "{target}: diagnostics {:?}", out.diagnostics);
    assert!(out.failure.is_none(), "{target}: {}", out.failure.as_ref().unwrap());
    out.module.expect("checked IR")
}

#[test]
fn per_element_combinators_branch_only_on_the_loop_test() {
    for target in ["x86_64-linux", "aarch64-linux"] {
        let m = lower(target);
        let mut checked = 0;
        let mut loops_seen = 0;
        let mut search_on_data = 0;
        let mut bad = Vec::new();
        for f in &m.funcs {
            let Some(body) = &f.body else { continue };
            let Some(rest) = f.path.strip_prefix("std.slice.") else { continue };
            let name = rest.rsplit('.').next().unwrap_or("");
            let per_loop = loop_branches(body);
            if BRANCHLESS.contains(&name) {
                checked += 1;
                loops_seen += per_loop.len();
                for (k, &(conds, on_data)) in per_loop.iter().enumerate() {
                    if conds != 1 || on_data != 0 {
                        bad.push(format!(
                            "{} {:?}: loop {k}: {conds} conditional branches, {on_data} on data",
                            f.path, f.type_args
                        ));
                    }
                }
            } else if SEARCH.contains(&name) {
                search_on_data += per_loop.iter().map(|&(_, d)| d).sum::<usize>();
            }
        }
        assert!(bad.is_empty(), "{target}:\n{}", bad.join("\n"));
        // sum, min, max: 12 element types; count, fill, copy_from: 13.
        assert!(checked >= 3 * 12 + 3 * 13, "{target}: only {checked} std.slice fns checked");
        assert!(loops_seen >= 40, "{target}: only {loops_seen} loops seen");
        assert!(search_on_data > 0, "{target}: no data-dependent branch found in the searches");
    }
}
