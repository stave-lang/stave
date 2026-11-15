//! Paired corpus: one program written twice, in Stave and in C99, from one `(lines, seed)`.
//!
//! Both texts come from the same generator calls, so they have the same fns, statements and
//! expressions, and the program prints the same checksum in both languages. The constructs are
//! chosen so that every operation is defined and means the same in both:
//! - integers are `u32`/`u64` (`uint32_t`/`uint64_t`, never promoted to `int`): `+% -% *%` are
//!   C's unsigned `+ - *`; constant shift amounts stay below the width; `a <<% b` is
//!   `a << (b & (bits - 1))`; `/` and `%` divide by `b | 1`;
//! - floats are `f64` with `+ - * /` only (C99 mode: no FMA contraction); a divisor is
//!   `x * x + 1.0`; `as| u32` is `stv_f2u` (truncate, saturate, NaN to 0); int to float
//!   converts a `u32` only (exact in both);
//! - expressions have no side effects and fns are pure, so C's unspecified evaluation order
//!   cannot change a result;
//! - every `for` runs fewer than 8 times and every `while` has a counter limit; a fn calls only
//!   earlier fns of lower rank, and inside loops only fns of rank 0 (which call nothing), so
//!   the run time is bounded;
//! - integer literals are never 0, and a comparison's right operand always reads a variable, so
//!   C compilers find no constant or tautological comparison to warn about.
//!
//! Every local is folded into the fn's accumulator `h` at the end of its block: nothing is dead
//! code for an optimizer to drop, and no C variable is unused.

use crate::rng::Rng;
use std::fmt::Write;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    U32,
    U64,
    F64,
}

const TYS: [Ty; 3] = [Ty::U32, Ty::U64, Ty::F64];
const TY_WEIGHTS: [u32; 3] = [35, 45, 20];

/// FNV-1a 64-bit prime and offset basis: the checksum mix of the drivers and `main`.
const FNV_PRIME: u64 = 1_099_511_628_211;
const FNV_BASIS: u64 = 14_695_981_039_346_656_037;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

impl Ty {
    fn s(self) -> &'static str {
        match self {
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::F64 => "f64",
        }
    }

    fn c(self) -> &'static str {
        match self {
            Ty::U32 => "uint32_t",
            Ty::U64 => "uint64_t",
            Ty::F64 => "double",
        }
    }

    fn bits(self) -> u32 {
        if self == Ty::U32 { 32 } else { 64 }
    }

    /// Integer literal suffixes (Stave, C).
    fn suffix(self) -> (&'static str, &'static str) {
        if self == Ty::U32 { ("u32", "u") } else { ("u64", "ull") }
    }
}

/// One expression in both languages.
struct X {
    s: String,
    c: String,
}

impl X {
    fn same(t: String) -> X {
        X { c: t.clone(), s: t }
    }

    /// A parenthesised binary operation; `so`/`co` are the Stave and C operators.
    fn bin(a: &X, so: &str, co: &str, b: &X) -> X {
        X { s: format!("({} {so} {})", a.s, b.s), c: format!("({} {co} {})", a.c, b.c) }
    }

    /// Reads a variable or calls a generated fn: C front ends cannot fold it to a constant.
    fn reads_var(&self) -> bool {
        let b = self.c.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_alphanumeric() || b[i] == b'_' {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let tok = &self.c[start..i];
                let typeish = matches!(tok, "uint32_t" | "uint64_t" | "double" | "stv_f2u");
                if !b[start].is_ascii_digit() && !typeish {
                    return true;
                }
            } else {
                i += 1;
            }
        }
        false
    }
}

struct Sig {
    unit: u32,
    idx: u32,
    params: Vec<Ty>,
    ret: Ty,
    rank: u8,
    public: bool,
}

impl Sig {
    fn name(&self) -> String {
        format!("f_{}_{}", self.unit, self.idx)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum K {
    Sc(Ty),
    /// Element type, length (a power of two).
    Arr(Ty, u32),
}

struct Local {
    name: String,
    k: K,
    assignable: bool,
    /// Folded into `h` when its block ends.
    fold: bool,
}

/// Generates one fn body into both buffers.
struct Gen<'a> {
    r: &'a mut Rng,
    sigs: &'a [Sig],
    visible: &'a [usize],
    unit: u32,
    rank: u8,
    ret: Ty,
    locals: Vec<Local>,
    scopes: Vec<usize>,
    loops: u32,
    next: u32,
    s: String,
    c: String,
    lines: u32,
}

const MAX_IND: u32 = 4;

impl Gen<'_> {
    fn line(&mut self, ind: u32, s: &str, c: &str) {
        for _ in 0..ind {
            self.s.push_str("    ");
            self.c.push_str("    ");
        }
        self.s.push_str(s);
        self.s.push('\n');
        self.c.push_str(c);
        self.c.push('\n');
        self.lines += 1;
    }

    fn fresh(&mut self, p: char) -> String {
        let n = self.next;
        self.next += 1;
        format!("{p}{n}")
    }

    fn ty(&mut self) -> Ty {
        TYS[self.r.weighted(&TY_WEIGHTS)]
    }

    fn int_lit(&mut self, t: Ty) -> u64 {
        if self.r.chance(850) {
            1 + u64::from(self.r.below(9999))
        } else if t == Ty::U32 {
            (self.r.next_u64() >> 33) | 1
        } else {
            (self.r.next_u64() >> 1) | 1
        }
    }

    fn lit(&mut self, t: Ty) -> X {
        match t {
            Ty::F64 => {
                const FR: [&str; 6] = ["0", "5", "25", "125", "75", "0625"];
                let a = self.r.below(100);
                let f = FR[self.r.below(6) as usize];
                X { s: format!("{a}.{f}f64"), c: format!("{a}.{f}") }
            }
            _ => {
                let v = self.int_lit(t);
                let (ss, cs) = t.suffix();
                X { s: format!("{v}{ss}"), c: format!("{v}{cs}") }
            }
        }
    }

    /// A scalar local (or parameter) of type `t`, if any.
    fn pick_var(&mut self, t: Ty, assignable: bool) -> Option<usize> {
        let n = self.locals.len() as u32;
        if n == 0 {
            return None;
        }
        let start = self.r.below(n);
        (0..n).map(|k| ((start + k) % n) as usize).find(|&i| {
            let l = &self.locals[i];
            l.k == K::Sc(t) && (!assignable || l.assignable)
        })
    }

    fn pick_arr(&mut self, elem: Option<Ty>) -> Option<usize> {
        let n = self.locals.len() as u32;
        if n == 0 {
            return None;
        }
        let start = self.r.below(n);
        (0..n).map(|k| ((start + k) % n) as usize).find(|&i| match self.locals[i].k {
            K::Arr(e, _) => elem.is_none_or(|t| t == e),
            K::Sc(_) => false,
        })
    }

    /// An expression of type `t` that reads a variable (`h` always exists).
    fn var(&mut self, t: Ty) -> X {
        if let Some(i) = self.pick_var(t, false) {
            return X::same(self.locals[i].name.clone());
        }
        match t {
            Ty::U64 => X::same("h".into()),
            Ty::U32 => X { s: "(h as% u32)".into(), c: "((uint32_t)h)".into() },
            Ty::F64 => X { s: "((h as% u32) as f64)".into(), c: "((double)((uint32_t)h))".into() },
        }
    }

    /// `name[(i & (len - 1)) as usize]`.
    fn index(&mut self, a: usize, d: u32) -> X {
        let (name, len) = match self.locals[a].k {
            K::Arr(_, len) => (self.locals[a].name.clone(), len),
            K::Sc(_) => unreachable!("index of a scalar"),
        };
        let i = if d == 0 && self.r.chance(500) { self.lit(Ty::U32) } else { self.var_or_expr(d) };
        X {
            s: format!("{name}[({} & {}u32) as usize]", i.s, len - 1),
            c: format!("{name}[{} & {}u]", i.c, len - 1),
        }
    }

    /// A `u32` index operand: below depth 1 a plain variable, so array reads do not nest.
    fn var_or_expr(&mut self, d: u32) -> X {
        if d == 0 { self.var(Ty::U32) } else { self.expr(Ty::U32, d) }
    }

    fn leaf(&mut self, t: Ty) -> X {
        let w = self.r.below(100);
        if w < 12
            && let Some(a) = self.pick_arr(Some(t))
        {
            return self.index(a, 0);
        }
        if w < 70
            && let Some(i) = self.pick_var(t, false)
        {
            return X::same(self.locals[i].name.clone());
        }
        self.lit(t)
    }

    fn expr(&mut self, t: Ty, d: u32) -> X {
        if d == 0 {
            return self.leaf(t);
        }
        match self.r.weighted(&[24, 44, 10, 3, 12, 7]) {
            0 => self.leaf(t),
            1 => self.binop(t, d),
            2 => self.cast(t, d),
            3 => {
                let mut a = self.expr(t, d - 1);
                if !a.reads_var() {
                    a = self.var(t);
                }
                let op = if t == Ty::F64 { "-" } else { "~" };
                X { s: format!("({op}{})", a.s), c: format!("({op}{})", a.c) }
            }
            4 => match self.call(t, d) {
                Some(x) => x,
                None => self.binop(t, d),
            },
            _ => {
                let c = self.cond(d - 1);
                let a = self.expr(t, d - 1);
                let b = self.expr(t, d - 1);
                X {
                    s: format!("(if {} then {} else {})", c.s, a.s, b.s),
                    c: format!("({} ? {} : {})", c.c, a.c, b.c),
                }
            }
        }
    }

    /// A right operand: a variable when the left one is constant, a literal when it would repeat
    /// the left one, so no operation folds to a constant in C (Stave folds only `const` items, so
    /// folding would favour C).
    fn rhs(&mut self, a: &X, t: Ty, d: u32) -> X {
        let b = self.expr(t, d);
        if b.c == a.c {
            return self.lit(t);
        }
        if a.reads_var() || b.reads_var() { b } else { self.var(t) }
    }

    fn binop(&mut self, t: Ty, d: u32) -> X {
        let mut a = self.expr(t, d - 1);
        if t == Ty::F64 {
            return match self.r.below(4) {
                0 => X::bin(&a, "+", "+", &self.rhs(&a, t, d - 1)),
                1 => X::bin(&a, "-", "-", &self.rhs(&a, t, d - 1)),
                2 => X::bin(&a, "*", "*", &self.rhs(&a, t, d - 1)),
                _ => {
                    let mut l = self.leaf(t);
                    if !a.reads_var() && !l.reads_var() {
                        l = self.var(t);
                    }
                    X {
                        s: format!("({} / (({} * {}) + 1.0f64))", a.s, l.s, l.s),
                        c: format!("({} / (({} * {}) + 1.0))", a.c, l.c, l.c),
                    }
                }
            };
        }
        let (ss, _) = t.suffix();
        let mask = t.bits() - 1;
        let op = self.r.weighted(&[14, 10, 10, 8, 6, 14, 7, 7, 3, 3, 3, 3]);
        if (op == 6 || op == 7) && !a.reads_var() {
            a = self.var(t);
        }
        match op {
            0 => X::bin(&a, "+%", "+", &self.rhs(&a, t, d - 1)),
            1 => X::bin(&a, "-%", "-", &self.rhs(&a, t, d - 1)),
            2 => X::bin(&a, "*%", "*", &self.rhs(&a, t, d - 1)),
            3 => X::bin(&a, "&", "&", &self.rhs(&a, t, d - 1)),
            4 => X::bin(&a, "|", "|", &self.rhs(&a, t, d - 1)),
            5 => X::bin(&a, "^", "^", &self.rhs(&a, t, d - 1)),
            // Below 32 also for u64: a u64 shifted left by 32 or more truncates to a constant 0.
            6 => {
                let k = X::same(self.r.range(1, 31).to_string());
                X::bin(&a, "<<", "<<", &k)
            }
            7 => {
                let k = X::same(self.r.range(1, mask).to_string());
                X::bin(&a, ">>", ">>", &k)
            }
            8 => {
                let b = self.rhs(&a, t, d - 1);
                X { s: format!("({} <<% {})", a.s, b.s), c: format!("({} << ({} & {mask}u))", a.c, b.c) }
            }
            9 => {
                let b = self.rhs(&a, t, d - 1);
                X {
                    s: format!("({} >> ({} & {mask}{ss}))", a.s, b.s),
                    c: format!("({} >> ({} & {mask}u))", a.c, b.c),
                }
            }
            10 => {
                let b = self.rhs(&a, t, d - 1);
                X { s: format!("({} / ({} | 1{ss}))", a.s, b.s), c: format!("({} / ({} | 1u))", a.c, b.c) }
            }
            _ => {
                let b = self.rhs(&a, t, d - 1);
                X { s: format!("({} % ({} | 1{ss}))", a.s, b.s), c: format!("({} % ({} | 1u))", a.c, b.c) }
            }
        }
    }

    fn cast(&mut self, t: Ty, d: u32) -> X {
        let from = loop {
            let f = self.ty();
            if f != t {
                break f;
            }
        };
        let a = self.expr(from, d - 1);
        let (s, c) = match (from, t) {
            (Ty::U32, Ty::U64) => (format!("({} as u64)", a.s), format!("((uint64_t){})", a.c)),
            (Ty::U64, Ty::U32) => (format!("({} as% u32)", a.s), format!("((uint32_t){})", a.c)),
            (Ty::F64, Ty::U32) => (format!("({} as| u32)", a.s), format!("stv_f2u({})", a.c)),
            (Ty::U32, Ty::F64) => (format!("({} as f64)", a.s), format!("((double){})", a.c)),
            (Ty::F64, Ty::U64) => {
                (format!("(({} as| u32) as u64)", a.s), format!("((uint64_t)stv_f2u({}))", a.c))
            }
            (Ty::U64, Ty::F64) => {
                (format!("(({} as% u32) as f64)", a.s), format!("((double)((uint32_t){}))", a.c))
            }
            _ => unreachable!("cast to the same type"),
        };
        X { s, c }
    }

    fn callee_ok(&self, callee: u8) -> bool {
        if self.loops > 0 { callee == 0 && self.rank > 0 } else { callee < self.rank }
    }

    fn call(&mut self, t: Ty, d: u32) -> Option<X> {
        let n = self.visible.len() as u32;
        if n == 0 || self.rank == 0 {
            return None;
        }
        let sigs = self.sigs;
        for _ in 0..4 {
            let f = &sigs[self.visible[self.r.below(n) as usize]];
            if f.ret != t || !self.callee_ok(f.rank) {
                continue;
            }
            let qual = if f.unit == self.unit { String::new() } else { format!("m{}.", f.unit) };
            let name = f.name();
            let params = f.params.clone();
            let (mut s, mut c) = (format!("{qual}{name}("), format!("{name}("));
            for (k, &p) in params.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                    c.push_str(", ");
                }
                let a = self.expr(p, (d - 1).min(1));
                s.push_str(&a.s);
                c.push_str(&a.c);
            }
            s.push(')');
            c.push(')');
            return Some(X { s, c });
        }
        None
    }

    fn cond(&mut self, d: u32) -> X {
        match self.r.weighted(&[70, 12, 12, 6]) {
            1 if d > 0 => {
                let (a, b) = (self.cond(d - 1), self.cond(d - 1));
                X::bin(&a, "&&", "&&", &b)
            }
            2 if d > 0 => {
                let (a, b) = (self.cond(d - 1), self.cond(d - 1));
                X::bin(&a, "||", "||", &b)
            }
            3 if d > 0 => {
                let a = self.cond(d - 1);
                X { s: format!("(!{})", a.s), c: format!("(!{})", a.c) }
            }
            _ => {
                let t = self.ty();
                let b = self.var(t);
                let mut a = self.expr(t, d);
                if a.s == b.s {
                    a = self.lit(t);
                }
                // `lit < x || lit2 != x` is a tautology clang warns about: no constant operands.
                if !a.reads_var() {
                    let v = self.var(t);
                    let so = if t == Ty::F64 { "+" } else { "+%" };
                    a = X::bin(&v, so, "+", &a);
                }
                const OPS: [&str; 6] = ["<", "<=", ">", ">=", "==", "!="];
                let op = OPS[self.r.below(6) as usize];
                X::bin(&a, op, op, &b)
            }
        }
    }

    fn depth(&mut self) -> u32 {
        1 + self.r.below(3)
    }

    fn declare(&mut self, name: String, k: K, assignable: bool, fold: bool) {
        self.locals.push(Local { name, k, assignable, fold });
    }

    fn open(&mut self) {
        self.scopes.push(self.locals.len());
    }

    /// Folds the block's locals into `h`, then forgets them.
    fn close(&mut self, ind: u32) {
        let start = self.scopes.pop().unwrap_or(0);
        let gone: Vec<Local> = self.locals.drain(start..).collect();
        for l in gone.iter().filter(|l| l.fold) {
            let n = &l.name;
            match l.k {
                K::Sc(Ty::U64) => self.line(ind, &format!("h = h ^ {n};"), &format!("h = h ^ {n};")),
                K::Sc(Ty::U32) => {
                    self.line(ind, &format!("h = h +% ({n} as u64);"), &format!("h = h + ((uint64_t){n});"))
                }
                K::Sc(Ty::F64) => self.line(
                    ind,
                    &format!("h = h ^ (({n} as| u32) as u64);"),
                    &format!("h = h ^ ((uint64_t)stv_f2u({n}));"),
                ),
                K::Arr(Ty::U64, _) => {
                    self.line(ind, &format!("h = h ^ {n}[0];"), &format!("h = h ^ {n}[0];"))
                }
                K::Arr(Ty::U32, _) => self.line(
                    ind,
                    &format!("h = h +% ({n}[0] as u64);"),
                    &format!("h = h + ((uint64_t){n}[0]);"),
                ),
                K::Arr(Ty::F64, _) => self.line(
                    ind,
                    &format!("h = h ^ (({n}[0] as| u32) as u64);"),
                    &format!("h = h ^ ((uint64_t)stv_f2u({n}[0]));"),
                ),
            }
        }
    }

    fn block(&mut self, ind: u32, target: u32) {
        self.open();
        self.stmts(ind, target.max(1));
        self.close(ind);
    }

    fn stmts(&mut self, ind: u32, target: u32) {
        let start = self.lines;
        while self.lines - start < target {
            let room = target - (self.lines - start);
            self.stmt(ind, room);
        }
    }

    fn ret_expr(&mut self) -> X {
        let d = self.r.below(2);
        let e = self.expr(self.ret, d);
        match self.ret {
            Ty::U64 => X { s: format!("h ^ {}", e.s), c: format!("h ^ {}", e.c) },
            Ty::U32 => X { s: format!("(h as% u32) ^ {}", e.s), c: format!("((uint32_t)h) ^ {}", e.c) },
            Ty::F64 => X {
                s: format!("((h as% u32) as f64) + {}", e.s),
                c: format!("((double)((uint32_t)h)) + {}", e.c),
            },
        }
    }

    fn stmt(&mut self, ind: u32, room: u32) {
        let nest = room >= 3 && ind < MAX_IND;
        let in_loop = self.loops > 0;
        let w = [
            30,                                         // let
            22,                                         // assign
            12,                                         // mix into h
            3,                                          // array
            8,                                          // array store
            if nest { 14 } else { 0 },                  // if
            if nest { 7 } else { 0 },                   // if-else
            if nest && self.loops < 2 { 9 } else { 0 }, // for
            if nest && self.loops < 2 { 3 } else { 0 }, // while
            if room >= 3 { 2 } else { 0 },              // early return
            if room >= 3 && in_loop { 3 } else { 0 },   // break
        ];
        match self.r.weighted(&w) {
            0 => {
                let t = self.ty();
                let name = self.fresh('v');
                let d = self.depth();
                let e = self.expr(t, d);
                self.line(
                    ind,
                    &format!("let mut {name}: {} = {};", t.s(), e.s),
                    &format!("{} {name} = {};", t.c(), e.c),
                );
                self.declare(name, K::Sc(t), true, true);
            }
            1 => {
                let t = self.ty();
                let Some(i) = self.pick_var(t, true) else { return self.mix(ind) };
                let name = self.locals[i].name.clone();
                let d = self.depth();
                let mut e = self.expr(t, d);
                if e.s == name {
                    e = self.lit(t);
                }
                self.line(ind, &format!("{name} = {};", e.s), &format!("{name} = {};", e.c));
            }
            2 => self.mix(ind),
            3 => {
                let t = self.ty();
                let len = if self.r.chance(500) { 8 } else { 16 };
                let name = self.fresh('a');
                let zero = match t {
                    Ty::F64 => "0.0f64".to_string(),
                    _ => format!("0{}", t.suffix().0),
                };
                self.line(
                    ind,
                    &format!("let mut {name}: [{}; {len}] = [{zero}; {len}];", t.s()),
                    &format!("{} {name}[{len}] = {{0}};", t.c()),
                );
                self.declare(name, K::Arr(t, len), true, true);
            }
            4 => {
                let Some(a) = self.pick_arr(None) else { return self.mix(ind) };
                let K::Arr(t, _) = self.locals[a].k else { return };
                let place = self.index(a, 1);
                let d = self.depth();
                let e = self.expr(t, d);
                self.line(ind, &format!("{} = {};", place.s, e.s), &format!("{} = {};", place.c, e.c));
            }
            5 | 6 => {
                let d = self.depth().min(2);
                let c = self.cond(d);
                self.line(ind, &format!("if {} {{", c.s), &format!("if {} {{", c.c));
                let inner = room - 2;
                let t1 = self.r.range(1, inner.min(8));
                self.block(ind + 1, t1);
                let left = inner.saturating_sub(t1 + 1);
                if left >= 1 && self.r.chance(500) {
                    self.line(ind, "} else {", "} else {");
                    let t2 = self.r.range(1, left.min(6));
                    self.block(ind + 1, t2);
                }
                self.line(ind, "}", "}");
            }
            7 => {
                let n = self.fresh('n');
                let i = self.fresh('i');
                let d = self.depth().min(2);
                let e = self.expr(Ty::U32, d);
                self.line(
                    ind,
                    &format!("let {n}: u32 = {} & 7u32;", e.s),
                    &format!("uint32_t {n} = {} & 7u;", e.c),
                );
                self.declare(n.clone(), K::Sc(Ty::U32), false, false);
                self.line(
                    ind,
                    &format!("for {i} in 0u32..{n} {{"),
                    &format!("for (uint32_t {i} = 0u; {i} < {n}; {i}++) {{"),
                );
                self.loops += 1;
                self.open();
                self.declare(i, K::Sc(Ty::U32), false, false);
                let t = self.r.range(1, room.saturating_sub(3).clamp(1, 8));
                self.block(ind + 1, t);
                self.close(ind + 1);
                self.loops -= 1;
                self.line(ind, "}", "}");
            }
            8 => {
                let w = self.fresh('w');
                let lim = self.r.range(1, 6);
                self.line(ind, &format!("let mut {w}: u32 = 0u32;"), &format!("uint32_t {w} = 0u;"));
                let d = self.depth().min(2);
                self.loops += 1;
                let c = self.cond(d);
                self.line(
                    ind,
                    &format!("while ({w} < {lim}u32) && {} {{", c.s),
                    &format!("while (({w} < {lim}u) && {}) {{", c.c),
                );
                self.line(ind + 1, &format!("{w} = {w} + 1u32;"), &format!("{w} = {w} + 1u;"));
                self.declare(w, K::Sc(Ty::U32), false, false);
                let t = self.r.range(1, room.saturating_sub(4).clamp(1, 8));
                self.block(ind + 1, t);
                self.loops -= 1;
                self.line(ind, "}", "}");
            }
            9 | 10 => {
                let d = self.depth().min(2);
                let c = self.cond(d);
                self.line(ind, &format!("if {} {{", c.s), &format!("if {} {{", c.c));
                if self.r.weighted(&[w[9], w[10]]) == 0 {
                    let e = self.ret_expr();
                    self.line(ind + 1, &format!("return {};", e.s), &format!("return {};", e.c));
                } else {
                    self.line(ind + 1, "break;", "break;");
                }
                self.line(ind, "}", "}");
            }
            _ => self.mix(ind),
        }
    }

    fn mix(&mut self, ind: u32) {
        let d = self.depth();
        let mut e = self.expr(Ty::U64, d);
        if e.c == "h" {
            e = self.lit(Ty::U64);
        }
        let k = self.int_lit(Ty::U64) | 1;
        self.line(ind, &format!("h = (h ^ {}) *% {k}u64;", e.s), &format!("h = (h ^ {}) * {k}ull;", e.c));
    }
}

/// The program in both languages.
pub struct PairTexts {
    /// `(path relative to the package root, text)`: `stave.toml`, `src/main.stv`, `src/m<k>.stv`.
    pub stave: Vec<(String, String)>,
    /// The whole C program in one file.
    pub c_single: String,
    /// `(file name, text)`: `corpus.h`, `m<k>.h`, `m<k>.c`, `main.c`, `files.txt`.
    pub c_split: Vec<(String, String)>,
    pub units: u32,
    /// Generated fns, without the per-unit drivers and `main`.
    pub fns: u64,
}

/// Request for [`pair_texts`].
#[derive(Clone, Copy, Debug)]
pub struct PairOptions {
    /// Approximate line count of the Stave program.
    pub lines: u64,
    pub seed: u64,
    /// Modules / translation units (0 = one per 1250 lines).
    pub units: u32,
    /// Passes of `main` over every driver: scales the run time only.
    pub reps: u32,
}

const C_PRELUDE: &str = "\
#include <stdint.h>

/* f64 -> u32 like Stave's `as| u32`: truncate toward zero, saturate, NaN -> 0. */
static inline uint32_t stv_f2u(double d) {
    if (!(d >= 0.0)) return 0u;
    if (d >= 4294967295.0) return 4294967295u;
    return (uint32_t)d;
}
";

struct Unit {
    s: String,
    c_body: String,
    c_header: String,
    imports: Vec<u32>,
}

fn unit_text(k: u32, target: u64, seed: u64, sigs: &mut Vec<Sig>) -> Unit {
    let mut r = Rng::derive(seed, u64::from(k), 0x5A17);
    let mut imports: Vec<u32> = Vec::new();
    if k > 0 {
        for _ in 0..r.range(1, 3) {
            let m = r.below(k);
            if !imports.contains(&m) {
                imports.push(m);
            }
        }
        imports.sort_unstable();
    }
    let mut visible: Vec<usize> =
        (0..sigs.len()).filter(|&i| sigs[i].public && imports.contains(&sigs[i].unit)).collect();
    let mut s = format!("// unit {k} (generated by gen-corpus --pair)\n");
    let mut c_body = String::new();
    let mut c_header = String::new();
    for m in &imports {
        let _ = writeln!(s, "import m{m};");
    }
    let mut lines = s.bytes().filter(|&b| b == b'\n').count() as u64 + 5;
    let first = sigs.len();
    let mut idx = 0u32;
    while lines < target || idx == 0 {
        let params: Vec<Ty> = (0..r.range(1, 4)).map(|_| TYS[r.weighted(&TY_WEIGHTS)]).collect();
        let ret = TYS[r.weighted(&TY_WEIGHTS)];
        let mut rank = [0u8, 1, 2, 3][r.weighted(&[40, 30, 20, 10])];
        if rank > 0 && !visible.iter().any(|&i| sigs[i].rank < rank) {
            rank = 0;
        }
        let public = !r.chance(200);
        let body_lines = r.lognormal(12.0, 0.6).clamp(3.0, 120.0) as u32;
        let sig = Sig { unit: k, idx, params, ret, rank, public };
        let mut g = Gen {
            r: &mut r,
            sigs: &sigs[..],
            visible: &visible,
            unit: k,
            rank,
            ret,
            locals: Vec::new(),
            scopes: Vec::new(),
            loops: 0,
            next: 0,
            s: String::new(),
            c: String::new(),
            lines: 0,
        };
        let (mut sp, mut cp, mut init_s, mut init_c) =
            (String::new(), String::new(), format!("{}u64", g.int_lit(Ty::U64)), String::new());
        init_c.push_str(&init_s.replace("u64", "ull"));
        for (i, &t) in sig.params.iter().enumerate() {
            if i > 0 {
                sp.push_str(", ");
                cp.push_str(", ");
            }
            let _ = write!(sp, "p{i}: {}", t.s());
            let _ = write!(cp, "{} p{i}", t.c());
            match t {
                Ty::U64 => {
                    let _ = write!(init_s, " ^ p{i}");
                    let _ = write!(init_c, " ^ p{i}");
                }
                Ty::U32 => {
                    let _ = write!(init_s, " ^ (p{i} as u64)");
                    let _ = write!(init_c, " ^ ((uint64_t)p{i})");
                }
                Ty::F64 => {
                    let _ = write!(init_s, " ^ ((p{i} as| u32) as u64)");
                    let _ = write!(init_c, " ^ ((uint64_t)stv_f2u(p{i}))");
                }
            }
            g.declare(format!("p{i}"), K::Sc(t), false, false);
        }
        let name = sig.name();
        let (pubs, statc) = if public { ("pub ", "") } else { ("", "static ") };
        g.line(
            0,
            &format!("{pubs}fn {name}({sp}) -> {} {{", ret.s()),
            &format!("{statc}{} {name}({cp}) {{", ret.c()),
        );
        g.line(1, &format!("let mut h: u64 = {init_s};"), &format!("uint64_t h = {init_c};"));
        g.declare("h".into(), K::Sc(Ty::U64), true, false);
        g.block(1, body_lines);
        let e = g.ret_expr();
        g.line(1, &format!("return {};", e.s), &format!("return {};", e.c));
        g.line(0, "}", "}");
        lines += u64::from(g.lines) + 2;
        s.push('\n');
        s.push_str(&g.s);
        c_body.push('\n');
        c_body.push_str(&g.c);
        if public {
            let _ = writeln!(c_header, "{} {name}({cp});", ret.c());
        }
        visible.push(sigs.len());
        sigs.push(sig);
        idx += 1;
    }
    // Driver: calls every fn of the unit once.
    let mut r = Rng::derive(seed, u64::from(k), 0xD21E);
    s.push('\n');
    c_body.push('\n');
    let _ = writeln!(s, "pub fn run_{k}(s: u64) -> u64 {{\n    let mut h: u64 = s;");
    let _ = writeln!(c_body, "uint64_t run_{k}(uint64_t s) {{\n    uint64_t h = s;");
    let _ = writeln!(c_header, "uint64_t run_{k}(uint64_t s);");
    for f in &sigs[first..] {
        let name = f.name();
        let (mut sa, mut ca) = (String::new(), String::new());
        for (i, &t) in f.params.iter().enumerate() {
            if i > 0 {
                sa.push_str(", ");
                ca.push_str(", ");
            }
            let sh = r.range(1, 31);
            match t {
                Ty::U64 => {
                    let v = r.next_u64() | 1;
                    let _ = write!(sa, "(s ^ {v}u64)");
                    let _ = write!(ca, "(s ^ {v}ull)");
                }
                Ty::U32 => {
                    let _ = write!(sa, "((s >> {sh}) as% u32)");
                    let _ = write!(ca, "((uint32_t)(s >> {sh}))");
                }
                Ty::F64 => {
                    let _ = write!(sa, "(((s >> {}) as% u32) as f64)", sh + 32);
                    let _ = write!(ca, "((double)((uint32_t)(s >> {})))", sh + 32);
                }
            }
        }
        let (cs, cc) = match f.ret {
            Ty::U64 => (format!("{name}({sa})"), format!("{name}({ca})")),
            Ty::U32 => (format!("({name}({sa}) as u64)"), format!("((uint64_t){name}({ca}))")),
            Ty::F64 => {
                (format!("(({name}({sa}) as| u32) as u64)"), format!("((uint64_t)stv_f2u({name}({ca})))"))
            }
        };
        let _ = writeln!(s, "    h = (h ^ {cs}) *% {FNV_PRIME}u64;");
        let _ = writeln!(c_body, "    h = (h ^ {cc}) * {FNV_PRIME}ull;");
    }
    s.push_str("    return h;\n}\n");
    c_body.push_str("    return h;\n}\n");
    Unit { s, c_body, c_header, imports }
}

/// Generates the program in both languages. Byte-identical per options.
pub fn pair_texts(o: &PairOptions) -> PairTexts {
    let units = if o.units > 0 { o.units } else { (o.lines / 1250).clamp(1, 4096) as u32 };
    let reps = o.reps.max(1);
    let main_lines = 2 * u64::from(units) + 12;
    let per_unit = o.lines.saturating_sub(main_lines) / u64::from(units);
    let mut sigs: Vec<Sig> = Vec::new();
    let mut stave = vec![("stave.toml".to_string(), "name = \"corpus\"\nkind = \"bin\"\n".to_string())];
    let mut split = vec![(
        "corpus.h".to_string(),
        format!(
            "/* generated by gen-corpus --pair: shared prelude */\n#ifndef CORPUS_H\n#define CORPUS_H\n{C_PRELUDE}#endif\n"
        ),
    )];
    let mut single = format!(
        "/* generated by gen-corpus --pair --lines {} --seed {} */\n#include <stdio.h>\n{C_PRELUDE}",
        o.lines, o.seed
    );
    let mut files = String::new();
    for k in 0..units {
        let u = unit_text(k, per_unit, o.seed, &mut sigs);
        stave.push((format!("src/m{k}.stv"), u.s));
        let _ = write!(single, "\n/* unit {k} */\n{}", u.c_body);
        let mut c = format!(
            "/* unit {k} (generated by gen-corpus --pair) */\n#include \"corpus.h\"\n#include \"m{k}.h\"\n"
        );
        for m in &u.imports {
            let _ = writeln!(c, "#include \"m{m}.h\"");
        }
        c.push_str(&u.c_body);
        split.push((
            format!("m{k}.h"),
            format!("#ifndef M{k}_H\n#define M{k}_H\n#include \"corpus.h\"\n{}#endif\n", u.c_header),
        ));
        split.push((format!("m{k}.c"), c));
        let _ = writeln!(files, "m{k}.c");
    }
    // main: `reps` passes over every unit's driver.
    let (mut ms, mut mc) =
        (String::from("// generated by gen-corpus --pair\nimport std.io;\n"), String::new());
    for k in 0..units {
        let _ = writeln!(ms, "import m{k};");
    }
    let _ = write!(
        ms,
        "\nfn main() -> i32 {{\n    let mut c: u64 = {FNV_BASIS}u64;\n    for r in 0u32..{reps}u32 {{\n        let s: u64 = (r as u64) *% {GOLDEN}u64;\n"
    );
    let _ = write!(
        mc,
        "\nint main(void) {{\n    uint64_t c = {FNV_BASIS}ull;\n    for (uint32_t r = 0u; r < {reps}u; r++) {{\n        uint64_t s = ((uint64_t)r) * {GOLDEN}ull;\n"
    );
    for k in 0..units {
        let _ = writeln!(ms, "        c = (c ^ m{k}.run_{k}(s)) *% {FNV_PRIME}u64;");
        let _ = writeln!(mc, "        c = (c ^ run_{k}(s)) * {FNV_PRIME}ull;");
    }
    ms.push_str("    }\n    io.print(f\"checksum {c}\\n\");\n    return 0;\n}\n");
    mc.push_str("    }\n    printf(\"checksum %llu\\n\", (unsigned long long)c);\n    return 0;\n}\n");
    stave.push(("src/main.stv".to_string(), ms));
    single.push_str(&mc);
    let mut main_c =
        String::from("/* generated by gen-corpus --pair */\n#include <stdio.h>\n#include \"corpus.h\"\n");
    for k in 0..units {
        let _ = writeln!(main_c, "#include \"m{k}.h\"");
    }
    main_c.push_str(&mc);
    split.push(("main.c".to_string(), main_c));
    files.push_str("main.c\n");
    split.push(("files.txt".to_string(), files));
    PairTexts { stave, c_single: single, c_split: split, units, fns: sigs.len() as u64 }
}

/// What [`generate_pair`] wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairSummary {
    pub stave_lines: u64,
    pub c_lines: u64,
    pub units: u32,
    pub fns: u64,
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn count_lines(t: &str) -> u64 {
    t.bytes().filter(|&b| b == b'\n').count() as u64
}

/// Writes `out/stave/` (a package), `out/c/single.c`, and `out/c/split/`.
pub fn generate_pair(o: &PairOptions, out: &Path, force: bool) -> Result<PairSummary, String> {
    if out.exists() {
        let mut entries = std::fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
        if entries.next().is_some() {
            if !force {
                return Err(format!("{}: not empty (use --force to replace stave/ and c/)", out.display()));
            }
            for sub in ["stave", "c"] {
                let d = out.join(sub);
                if d.exists() {
                    std::fs::remove_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
                }
            }
        }
    }
    let t = pair_texts(o);
    let mut stave_lines = 0;
    for (p, text) in &t.stave {
        if p.ends_with(".stv") {
            stave_lines += count_lines(text);
        }
        write(&out.join("stave").join(p), text)?;
    }
    write(&out.join("c/single.c"), &t.c_single)?;
    for (p, text) in &t.c_split {
        write(&out.join("c/split").join(p), text)?;
    }
    Ok(PairSummary { stave_lines, c_lines: count_lines(&t.c_single), units: t.units, fns: t.fns })
}
