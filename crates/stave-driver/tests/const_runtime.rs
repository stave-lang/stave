//! Const evaluation against the run time. Generated integer expressions (every integer type;
//! `+ - * / %`, wrapping `+% -% *%`, saturating `+| -| *|`, `& | ^`, `<< >> <<%`, unary `-` and
//! `~`, `as as% as|`, comparisons, `&&` / `||`, `if then else`), with unsuffixed literals in
//! every position of their chain, each written twice: as a `const` initializer and as the body
//! of a fn whose literals are typed and passed through an identity fn (nothing is folded). A
//! model of the operators (LANG.md: `+ - *` overflow, `/ %` by 0, `MIN / -1`, `MIN % -1`, and a
//! shift amount outside `0..bits` trap; `+% -% *% <<%` wrap; `+| -| *|` saturate) decides each
//! case. The const must have the twin's value, and be rejected (E0203 / E0205) exactly when the
//! twin traps; every twin runs (a trapping one in a run of its own) and must agree with the
//! model.
//!
//! The always-on test runs a small sample. `cargo test --release -p stave-driver --test
//! const_runtime -- --ignored` runs the full set: `STAVE_CONST_CASES` cases (default 500,000)
//! on `STAVE_CONST_THREADS` threads (default 4).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use stave_driver::exec::{self, RunSpec, TempDir};
use stave_driver::pipeline::{self, Goal, Options};
use stave_driver::shipped::Library;
use stave_target::Triple;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// name, signed, bits
const TYPES: [(&str, bool, u32); 10] = [
    ("i8", true, 8),
    ("i16", true, 16),
    ("i32", true, 32),
    ("i64", true, 64),
    ("isize", true, 64),
    ("u8", false, 8),
    ("u16", false, 16),
    ("u32", false, 32),
    ("u64", false, 64),
    ("usize", false, 64),
];

fn range(t: usize) -> (i128, i128) {
    let (_, s, w) = TYPES[t];
    if s { (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1) } else { (0, (1i128 << w) - 1) }
}

fn wrap(v: i128, t: usize) -> i128 {
    let (_, s, w) = TYPES[t];
    let m = (v as u128) & ((1u128 << w) - 1);
    if s && m >> (w - 1) == 1 { m as i128 - (1i128 << w) } else { m as i128 }
}

/// Operator levels of one chain node (LANG.md precedence: `|` < `^` < `&` < shifts < `+` < `*`).
const LEVELS: [&[&str]; 6] = [
    &["|"],
    &["^"],
    &["&"],
    &["<<", ">>", "<<%"],
    &["+", "-", "+%", "-%", "+|", "-|"],
    &["*", "/", "%", "*%", "*|"],
];

const CMPS: [&str; 6] = ["==", "!=", "<", "<=", ">", ">="];

/// `None`: the operation traps.
fn binop(op: &str, a: i128, b: i128, t: usize) -> Option<i128> {
    let (lo, hi) = range(t);
    let (_, s, w) = TYPES[t];
    let inr = |v: i128| (lo..=hi).contains(&v).then_some(v);
    let sat = |v: i128| v.clamp(lo, hi);
    Some(match op {
        "+" => inr(a + b)?,
        "-" => inr(a - b)?,
        "*" => inr(a.checked_mul(b)?)?,
        "/" | "%" => {
            if b == 0 || s && a == lo && b == -1 {
                return None;
            }
            if op == "/" { a / b } else { a % b }
        }
        "+%" => wrap(a + b, t),
        "-%" => wrap(a - b, t),
        "*%" => wrap(a.wrapping_mul(b), t),
        "+|" => sat(a + b),
        "-|" => sat(a - b),
        "*|" => a.checked_mul(b).map_or(if (a < 0) != (b < 0) { lo } else { hi }, sat),
        "&" => wrap(a & b, t),
        "|" => wrap(a | b, t),
        "^" => wrap(a ^ b, t),
        "<<" | ">>" => {
            if b < 0 || b >= w as i128 {
                return None;
            }
            if op == "<<" { wrap(a << b, t) } else { a >> b }
        }
        "<<%" => wrap(a << (b & (w as i128 - 1)), t),
        _ => unreachable!("{op}"),
    })
}

#[derive(Clone, Debug)]
enum E {
    Lit {
        v: i128,
        suffix: bool,
    },
    Chain {
        ops: Vec<&'static str>,
        items: Vec<E>,
    },
    Neg(Box<E>),
    Not(Box<E>),
    /// `(e) op to`; `from`: the operand's type.
    Cast {
        e: Box<E>,
        op: &'static str,
        from: usize,
    },
    Cmp {
        l: Box<E>,
        op: &'static str,
        r: Box<E>,
        ty: usize,
    },
    Logic {
        and: bool,
        items: Vec<E>,
    },
    If {
        c: Box<E>,
        a: Box<E>,
        b: Box<E>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    I(i128),
    B(bool),
}

/// The value of `e` of type `t` (`None`: it traps), lazily where the run time is.
fn eval(e: &E, t: usize) -> Option<V> {
    Some(match e {
        E::Lit { v, .. } => V::I(*v),
        E::Chain { ops, items } => {
            let V::I(mut acc) = eval(&items[0], t)? else { unreachable!() };
            for (op, x) in ops.iter().zip(&items[1..]) {
                let V::I(b) = eval(x, t)? else { unreachable!() };
                acc = binop(op, acc, b, t)?;
            }
            V::I(acc)
        }
        E::Neg(x) => {
            let V::I(v) = eval(x, t)? else { unreachable!() };
            V::I(if v == range(t).0 { return None } else { -v })
        }
        E::Not(x) => {
            let V::I(v) = eval(x, t)? else { unreachable!() };
            V::I(wrap(!v, t))
        }
        E::Cast { e, op, from } => {
            let V::I(v) = eval(e, *from)? else { unreachable!() };
            let (lo, hi) = range(t);
            V::I(match *op {
                "as" => v,
                "as%" => wrap(v, t),
                _ => v.clamp(lo, hi),
            })
        }
        E::Cmp { l, op, r, ty } => {
            let (V::I(a), V::I(b)) = (eval(l, *ty)?, eval(r, *ty)?) else { unreachable!() };
            V::B(match *op {
                "==" => a == b,
                "!=" => a != b,
                "<" => a < b,
                "<=" => a <= b,
                ">" => a > b,
                _ => a >= b,
            })
        }
        E::Logic { and, items } => {
            for x in items {
                if eval(x, t)? == V::B(!*and) {
                    return Some(V::B(!*and));
                }
            }
            V::B(*and)
        }
        E::If { c, a, b } => {
            if eval(c, t)? == V::B(true) {
                eval(a, t)?
            } else {
                eval(b, t)?
            }
        }
    })
}

/// A deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

fn interesting(t: usize) -> Vec<i128> {
    let (lo, hi) = range(t);
    let (_, s, w) = TYPES[t];
    let w = w as i128;
    let mut v = vec![0, 1, 2, 3, 7, w - 1, w, w + 1, 100, hi, hi - 1, hi / 2 + 1];
    if s {
        v.extend([lo, lo + 1, -1, -2, -7, -w]);
    }
    v.retain(|x| (lo..=hi).contains(x));
    v.sort();
    v.dedup();
    v
}

struct Gen {
    rng: Rng,
}

impl Gen {
    fn lit(&mut self, t: usize, shift_amount: bool) -> E {
        let vs = interesting(t);
        let v = if shift_amount && self.rng.chance(80) {
            let w = TYPES[t].2 as i128;
            let v = self.rng.below(w as usize + 2) as i128;
            if TYPES[t].1 && self.rng.chance(10) { -v } else { v }
        } else {
            vs[self.rng.below(vs.len())]
        };
        let (lo, hi) = range(t);
        E::Lit { v: v.clamp(lo, hi), suffix: self.rng.chance(25) }
    }

    /// An integer expression of type `t`, `depth` levels of nesting left.
    fn int(&mut self, t: usize, depth: u32) -> E {
        let pick = if depth == 0 { 0 } else { self.rng.below(10) };
        match pick {
            0 | 1 => self.lit(t, false),
            2..=5 => {
                let lvl = self.rng.below(LEVELS.len());
                let n = 2 + self.rng.below(3);
                let mut items = vec![self.int(t, depth - 1)];
                let mut ops = Vec::new();
                for _ in 1..n {
                    let op = LEVELS[lvl][self.rng.below(LEVELS[lvl].len())];
                    ops.push(op);
                    let shift = matches!(op, "<<" | ">>" | "<<%");
                    items.push(if shift && self.rng.chance(70) {
                        self.lit(t, true)
                    } else {
                        self.int(t, depth - 1)
                    });
                }
                E::Chain { ops, items }
            }
            6 if TYPES[t].1 => E::Neg(Box::new(self.int(t, depth - 1))),
            6 => E::Not(Box::new(self.int(t, depth - 1))),
            7 => {
                let from = self.rng.below(TYPES.len());
                let ops = ["as", "as%", "as|"];
                let mut op = ops[self.rng.below(3)];
                if op == "as" && !lossless(from, t) {
                    op = "as%";
                }
                let mut e = self.int(from, depth - 1);
                closed(&mut e, &mut self.rng);
                E::Cast { e: Box::new(e), op, from }
            }
            8 => {
                let c = self.boolean(depth - 1);
                E::If {
                    c: Box::new(c),
                    a: Box::new(self.int(t, depth - 1)),
                    b: Box::new(self.int(t, depth - 1)),
                }
            }
            _ => E::Not(Box::new(self.int(t, depth - 1))),
        }
    }

    fn boolean(&mut self, depth: u32) -> E {
        let pick = if depth == 0 { 0 } else { self.rng.below(4) };
        match pick {
            0 | 1 => {
                let ty = self.rng.below(TYPES.len());
                let l = self.int(ty, depth.saturating_sub(1).max(1));
                let r = self.int(ty, depth.saturating_sub(1));
                let mut e = E::Cmp { l: Box::new(l), op: CMPS[self.rng.below(6)], r: Box::new(r), ty };
                closed(&mut e, &mut self.rng);
                e
            }
            2 => {
                let n = 2 + self.rng.below(2);
                let and = self.rng.chance(50);
                E::Logic { and, items: (0..n).map(|_| self.boolean(depth - 1)).collect() }
            }
            _ => E::If {
                c: Box::new(self.boolean(depth - 1)),
                a: Box::new(self.boolean(depth - 1)),
                b: Box::new(self.boolean(depth - 1)),
            },
        }
    }
}

/// Whether a plain `as` from `f` to `t` is lossless (the checker's rule).
fn lossless(f: usize, t: usize) -> bool {
    let ((_, fs, fb), (_, ts, tb)) = (TYPES[f], TYPES[t]);
    (fs == ts && tb >= fb) || (!fs && ts && tb > fb)
}

/// The literals of the typing region of `e` (its operands, not inside a cast or a comparison).
fn region_lits<'a>(e: &'a mut E, out: &mut Vec<&'a mut bool>) {
    match e {
        E::Lit { suffix, .. } => out.push(suffix),
        E::Chain { items, .. } => items.iter_mut().for_each(|x| region_lits(x, out)),
        E::Neg(x) | E::Not(x) => region_lits(x, out),
        E::If { a, b, .. } => {
            region_lits(a, out);
            region_lits(b, out);
        }
        E::Cmp { l, r, .. } => {
            region_lits(l, out);
            region_lits(r, out);
        }
        E::Cast { .. } | E::Logic { .. } => {}
    }
}

/// A region no expected type reaches (a cast operand, a comparison) needs a typed literal.
fn closed(e: &mut E, rng: &mut Rng) {
    let mut lits = Vec::new();
    region_lits(e, &mut lits);
    if !lits.is_empty() && !lits.iter().any(|s| **s) {
        let k = rng.below(lits.len());
        *lits[k] = true;
    }
}

fn lit_text(v: i128, t: Option<usize>) -> String {
    let suf = t.map_or("", |t| TYPES[t].0);
    if v < 0 { format!("(-{}{suf})", -v) } else { format!("{v}{suf}") }
}

/// The source of `e` of type `t`: `twin` passes every literal through `id_T`, typed.
fn show(e: &E, t: usize, twin: bool, out: &mut String) {
    match e {
        E::Lit { v, suffix } => {
            if twin {
                let _ = write!(out, "id_{}({})", TYPES[t].0, lit_text(*v, Some(t)));
            } else {
                out.push_str(&lit_text(*v, suffix.then_some(t)));
            }
        }
        E::Chain { ops, items } => {
            out.push('(');
            show(&items[0], t, twin, out);
            for (op, x) in ops.iter().zip(&items[1..]) {
                let _ = write!(out, " {op} ");
                match x {
                    // a literal shift amount stays one in the twin: the rule on it (E0205 at
                    // or past the width) is a rule of the text, in a fn as in a const
                    E::Lit { v, .. } if twin && matches!(*op, "<<" | ">>" | "<<%") => {
                        out.push_str(&lit_text(*v, Some(t)))
                    }
                    _ => show(x, t, twin, out),
                }
            }
            out.push(')');
        }
        E::Neg(x) => {
            out.push_str("-(");
            show(x, t, twin, out);
            out.push(')');
        }
        E::Not(x) => {
            out.push_str("~(");
            show(x, t, twin, out);
            out.push(')');
        }
        E::Cast { e, op, from } => {
            out.push('(');
            show(e, *from, twin, out);
            let _ = write!(out, " {op} {})", TYPES[t].0);
        }
        E::Cmp { l, op, r, ty } => {
            out.push('(');
            show(l, *ty, twin, out);
            let _ = write!(out, " {op} ");
            show(r, *ty, twin, out);
            out.push(')');
        }
        E::Logic { and, items } => {
            out.push('(');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(if *and { " && " } else { " || " });
                }
                show(x, t, twin, out);
            }
            out.push(')');
        }
        E::If { c, a, b } => {
            out.push_str("(if ");
            show(c, t, twin, out);
            out.push_str(" then ");
            show(a, t, twin, out);
            out.push_str(" else ");
            show(b, t, twin, out);
            out.push(')');
        }
    }
}

struct Case {
    ty: String,
    text: String,
    twin: String,
    want: Option<V>,
}

fn case(g: &mut Gen) -> Case {
    let boolean = g.rng.chance(35);
    let (ty, e, t) = if boolean {
        ("bool".to_string(), g.boolean(3), 0)
    } else {
        let t = g.rng.below(TYPES.len());
        (TYPES[t].0.to_string(), g.int(t, 3), t)
    };
    let (mut text, mut twin) = (String::new(), String::new());
    show(&e, t, false, &mut text);
    show(&e, t, true, &mut twin);
    Case { ty, text, twin, want: eval(&e, t) }
}

fn fmt_v(v: V) -> String {
    match v {
        V::I(i) => i.to_string(),
        V::B(b) => b.to_string(),
    }
}

#[derive(Default, Debug)]
struct Tally {
    cases: usize,
    values: usize,
    rejected: usize,
    static_rejected: usize,
    twins_run: usize,
    traps_run: usize,
    bad: Vec<String>,
}

/// The batch program: const `C<i>` on line [`const_line`], its twin `r<i>` on the next line.
/// A const or twin the checker rejects is dropped (`drop` / `rdrop`). `main` prints every const
/// and every twin the model gives a value; with `i + 1` extra arguments it runs trapping twin
/// `r<i>` alone.
fn program(cases: &[Case], drop: &[bool], rdrop: &[bool]) -> String {
    let mut s = String::from("import std.io;\nimport std.env;\n");
    for (n, ..) in TYPES {
        let _ = writeln!(s, "fn id_{n}(x: {n}) -> {n} {{ return x; }}");
    }
    for (i, c) in cases.iter().enumerate() {
        if drop[i] {
            s.push_str("// dropped\n");
        } else {
            let _ = writeln!(s, "const C{i}: {} = {};", c.ty, c.text);
        }
        if rdrop[i] {
            s.push_str("// dropped\n");
        } else {
            let _ = writeln!(s, "fn r{i}() -> {} {{ return {}; }}", c.ty, c.twin);
        }
    }
    s.push_str("fn main() -> i32 {\n    let k = env.count();\n    if k == 1 {\n");
    for (i, c) in cases.iter().enumerate() {
        if !drop[i] {
            let _ = writeln!(s, "        let c{i} = C{i}; io.print(f\"C{i} {{c{i}}}\\n\");");
        }
        if c.want.is_some() && !rdrop[i] {
            let _ = writeln!(s, "        let v{i} = r{i}(); io.print(f\"R{i} {{v{i}}}\\n\");");
        }
    }
    s.push_str("        return 0;\n    }\n");
    for (i, c) in cases.iter().enumerate() {
        if c.want.is_none() && !rdrop[i] {
            let _ = writeln!(s, "    if k == {} {{ let v = r{i}(); io.print(f\"T {{v}}\\n\"); }}", i + 2);
        }
    }
    s.push_str("    return 0;\n}\n");
    s
}

fn const_line(i: usize) -> u32 {
    (2 + TYPES.len() + 2 * i + 1) as u32
}

fn run_batch(cases: &[Case], dir: &Path, name: &str, tally: &Mutex<Tally>) {
    let path = dir.join(format!("{name}.stv"));
    let lib = Library::in_dir(&repo());
    let target = Triple::parse("x86_64-linux").unwrap();
    let n = cases.len();
    let (mut drop, mut rdrop) = (vec![false; n], vec![false; n]);
    let mut codes: Vec<Vec<&'static str>> = vec![Vec::new(); n];
    let mut rcodes: Vec<Vec<&'static str>> = vec![Vec::new(); n];
    let mut bad = Vec::new();
    // check, drop the lines with errors, check again
    for _ in 0..3 {
        std::fs::write(&path, program(cases, &drop, &rdrop)).unwrap();
        let mut o = Options::new(Goal::Check).threads(Some(1));
        o.target = Some(target);
        let out = pipeline::run(&path, &lib, &o).unwrap();
        if let Some(f) = &out.failure {
            bad.push(format!("{name}: check failure: {f}"));
            break;
        }
        let mut any = false;
        for d in out.diagnostics.iter().filter(|d| d.severity() == stave_diag::Severity::Error) {
            let line = d.primary().and_then(|sp| out.files().locate(sp).ok()).map(|l| l.line);
            let at = line.and_then(|l| (0..n).find(|&i| const_line(i) == l || const_line(i) + 1 == l));
            match (at, line) {
                (Some(i), Some(l)) if l == const_line(i) => {
                    drop[i] = true;
                    codes[i].push(d.code());
                    any = true;
                }
                (Some(i), _) => {
                    rdrop[i] = true;
                    rcodes[i].push(d.code());
                    any = true;
                }
                _ => bad.push(format!("{name}: {} at line {line:?}", d.code())),
            }
        }
        if !any {
            break;
        }
    }
    std::fs::write(&path, program(cases, &drop, &rdrop)).unwrap();
    let mut o = Options::new(Goal::Build).threads(Some(1));
    o.target = Some(target);
    let out = pipeline::run(&path, &lib, &o).unwrap();
    if out.has_errors() || out.failure.is_some() || out.image.is_none() {
        let mut m = tally.lock().unwrap();
        m.bad.extend(bad);
        m.bad.push(format!(
            "{name}: build: {:?} {:?}",
            out.diagnostics.first().map(|d| d.code()),
            out.failure
        ));
        return;
    }
    let bin = dir.join(name);
    out.image.as_ref().unwrap().write_file(&bin).unwrap();
    let spec = RunSpec { args: &[], runner: None, cwd: None, inherit: false };
    let ran = exec::run(&bin, &spec).unwrap();
    let stdout = String::from_utf8_lossy(&ran.stdout);
    let got: std::collections::HashMap<&str, &str> =
        stdout.lines().filter_map(|l| l.split_once(' ')).collect();
    if ran.exit != 0 {
        bad.push(format!("{name}: run exit {}: {}", ran.exit, String::from_utf8_lossy(&ran.stderr)));
    }
    let mut t = Tally { cases: n, ..Tally::default() };
    for (i, c) in cases.iter().enumerate() {
        let what = format!("{name} C{i}: `const C: {} = {};`", c.ty, c.text);
        let cval = got.get(format!("C{i}").as_str()).copied();
        // the twin's text breaks a rule of the checker (a literal shift amount past the width):
        // the const breaks the same one
        if rdrop[i] {
            t.static_rejected += 1;
            if !drop[i] || !same_codes(&codes[i], &rcodes[i]) {
                bad.push(format!("{what}: const {:?} ({cval:?}), twin rejected {:?}", codes[i], rcodes[i]));
            }
            continue;
        }
        match (c.want, drop[i]) {
            (Some(w), false) => {
                t.values += 1;
                if cval != Some(fmt_v(w).as_str()) {
                    bad.push(format!("{what} is {cval:?}, run time {}", fmt_v(w)));
                }
            }
            (Some(w), true) => bad.push(format!("{what} rejected ({:?}), run time {}", codes[i], fmt_v(w))),
            (None, false) => bad.push(format!("{what} accepted ({cval:?}), run time traps")),
            (None, true) => {
                t.rejected += 1;
                if !codes[i].iter().all(|c| matches!(*c, "E0203" | "E0205")) {
                    bad.push(format!("{what} rejected with {:?}, run time traps", codes[i]));
                }
            }
        }
        match c.want {
            Some(w) => {
                t.twins_run += 1;
                let g = got.get(format!("R{i}").as_str()).copied();
                if g != Some(fmt_v(w).as_str()) {
                    bad.push(format!("{name} r{i} `{}` is {g:?}, the model says {}", c.twin, fmt_v(w)));
                }
            }
            None => {
                t.traps_run += 1;
                let args: Vec<String> = (0..=i).map(|_| "x".to_string()).collect();
                let spec = RunSpec { args: &args, runner: None, cwd: None, inherit: false };
                let r = exec::run(&bin, &spec).unwrap();
                let err = String::from_utf8_lossy(&r.stderr);
                if r.exit == 0 || !err.contains("panic") {
                    bad.push(format!("{name} r{i} `{}` does not trap: exit {}, {err}", c.twin, r.exit));
                }
            }
        }
    }
    t.bad = bad;
    let mut m = tally.lock().unwrap();
    m.cases += t.cases;
    m.values += t.values;
    m.rejected += t.rejected;
    m.static_rejected += t.static_rejected;
    m.twins_run += t.twins_run;
    m.traps_run += t.traps_run;
    m.bad.extend(t.bad);
}

fn differential(total: usize, threads: usize, seed: u64) {
    let target = Triple::parse("x86_64-linux").unwrap();
    if !exec::can_run(target) {
        eprintln!("note: x86-64 Linux binaries cannot run here: skipped");
        return;
    }
    const BATCH: usize = 400;
    let nb = total.div_ceil(BATCH);
    let tmp = TempDir::new("constrt").unwrap();
    let tally = Mutex::new(Tally::default());
    let next = Mutex::new(0usize);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let b = {
                        let mut n = next.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    if b >= nb {
                        break;
                    }
                    let mut g = Gen { rng: Rng(seed ^ (b as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)) };
                    let n = BATCH.min(total - b * BATCH);
                    let cases: Vec<Case> = (0..n).map(|_| case(&mut g)).collect();
                    run_batch(&cases, tmp.path(), &format!("b{b}"), &tally);
                }
            });
        }
    });
    let t = tally.into_inner().unwrap();
    eprintln!(
        "const vs run time: {} cases, {} const values, {} rejected (twin traps), {} rejected by a rule of the text (twin too), {} twins run, {} trapping twins run, {} mismatches",
        t.cases,
        t.values,
        t.rejected,
        t.static_rejected,
        t.twins_run,
        t.traps_run,
        t.bad.len()
    );
    assert!(
        t.bad.is_empty(),
        "{} mismatches, first 40:\n{}",
        t.bad.len(),
        t.bad[..t.bad.len().min(40)].join("\n")
    );
}

/// The const has the twin's errors `b`, and besides them only a trap of its evaluation (an
/// operation before the rejected one traps at run time).
fn same_codes(a: &[&str], b: &[&str]) -> bool {
    b.iter().all(|c| a.contains(c)) && a.iter().all(|c| b.contains(c) || matches!(*c, "E0203" | "E0205"))
}

#[test]
fn const_values_are_run_time_values_sample() {
    differential(800, 2, 0x5eed);
}

#[test]
#[ignore]
fn const_values_are_run_time_values_full() {
    let n = std::env::var("STAVE_CONST_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let th = std::env::var("STAVE_CONST_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    differential(n, th, 0x00c0_de57);
}
