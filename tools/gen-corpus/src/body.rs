//! Phase 2: statements and expressions of fn bodies.
//!
//! Invariants that keep the output semantically plausible (LANG.md v0.4):
//! - every local is declared with a type annotation and an initializer; names never repeat
//!   within a fn (no shadowing);
//! - an unsuffixed literal only appears where the typing rules give it a type (annotation, parameter,
//!   field, return, or the other operand of a binary operator / match or if arm); elsewhere
//!   it carries a suffix;
//! - struct values are read in place, borrowed (`&s`) for one call, or moved once by a
//!   statement at the loop depth where they were declared; `&mut` borrows appear only in
//!   statements whose other operands read no struct;
//! - references returned with `from` live inside a block of their own;
//! - a body calls only fns and methods of lower rank (inside loops: rank 0 only), declared
//!   earlier in its module or in an imported module, so call graphs are acyclic and shallow.

use crate::mix::Mix;
use crate::model::*;
use crate::names::{NOTES, NOUNS, push_pascal};
use crate::rng::Rng;
use std::collections::HashMap;
use std::fmt::Write;

/// `(module, fn index)`.
pub type FnRef = (u32, u32);

/// Callee pools by return shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Key {
    Sc(Sc),
    Opt(Sc),
    Res(Sc),
    ResOwn(u32),
    Tup,
    RefSc(Sc),
    RefS,
    MutSc(Sc),
    E,
    S,
    Unit,
    MutUnit,
    Pick,
    RefPick,
    Bound,
    Const,
    GenCtor,
}

#[derive(Default)]
pub struct Pool {
    /// Own fns, ascending (a visible prefix).
    own: Vec<u32>,
    ext: Vec<FnRef>,
}

#[derive(Default)]
pub struct Pools(HashMap<Key, Pool>);

impl Pools {
    fn keys(f: &FnDef, own: bool) -> ([Option<Key>; 2], usize) {
        let k = match f.kind {
            FnKind::Pick => Key::Pick,
            FnKind::RefPick => Key::RefPick,
            FnKind::Bound(_) => Key::Bound,
            FnKind::Const => Key::Const,
            FnKind::GenCtor(_) => Key::GenCtor,
            FnKind::Plain => match f.ret {
                Ret::Unit if f.params.iter().any(|p| matches!(p, PTy::Mut(_))) => Key::MutUnit,
                Ret::Unit => Key::Unit,
                Ret::Sc(t) => Key::Sc(t),
                Ret::S(_) => Key::S,
                Ret::E(_) => Key::E,
                Ret::Opt(t) => Key::Opt(t),
                Ret::Res(t, e) => return ([Some(Key::Res(t)), own.then_some(Key::ResOwn(e))], 2),
                Ret::Tup(..) => Key::Tup,
                Ret::RefSc(t) => Key::RefSc(t),
                Ret::RefS(_) => Key::RefS,
                Ret::MutSc(t) => Key::MutSc(t),
                Ret::T | Ret::RefT | Ret::Cell(_) => return ([None, None], 0),
            },
        };
        ([Some(k), None], 1)
    }

    pub fn build(plans: &[ModPlan], me: u32) -> Pools {
        let mut p = Pools::default();
        for (i, f) in plans[me as usize].fns.iter().enumerate() {
            let (ks, n) = Pools::keys(f, true);
            for k in ks[..n].iter().flatten() {
                p.0.entry(*k).or_default().own.push(i as u32);
            }
        }
        for &m in &plans[me as usize].imports {
            for (i, f) in plans[m as usize].fns.iter().enumerate() {
                if !f.public {
                    continue;
                }
                let (ks, n) = Pools::keys(f, false);
                for k in ks[..n].iter().flatten() {
                    p.0.entry(*k).or_default().ext.push((m, i as u32));
                }
            }
        }
        p
    }
}

/// Per-module generation context.
pub struct Ctx<'a> {
    pub plans: &'a [ModPlan],
    pub me: u32,
    pub mix: &'a Mix,
    pub pools: Pools,
    pub ext_structs: Vec<(u32, u32)>,
    pub ext_enums: Vec<(u32, u32)>,
    pub ext_consts: Vec<(u32, u32)>,
    pub io: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(plans: &'a [ModPlan], me: u32, mix: &'a Mix) -> Ctx<'a> {
        let p = &plans[me as usize];
        let (mut es, mut ee, mut ec) = (Vec::new(), Vec::new(), Vec::new());
        for &m in &p.imports {
            let q = &plans[m as usize];
            es.extend((0..q.structs.len() as u32).map(|i| (m, i)));
            ee.extend((0..q.enums.len() as u32).map(|i| (m, i)));
            ec.extend((0..q.consts.len() as u32).map(|i| (m, i)));
        }
        Ctx {
            plans,
            me,
            mix,
            pools: Pools::build(plans, me),
            ext_structs: es,
            ext_enums: ee,
            ext_consts: ec,
            io: p.uses_io,
        }
    }
}

/// Own-module items declared so far.
#[derive(Clone, Copy, Default, Debug)]
pub struct Vis {
    pub fns: u32,
    pub structs: u32,
    pub enums: u32,
    pub impls: u32,
    pub timpls: u32,
    pub consts: u32,
}

/// How a struct place is held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum How {
    Owned,
    Ref,
    Mut,
    SelfRef,
    SelfMut,
    SelfVal,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LK {
    Sc(Sc),
    /// Struct place: module, struct, instantiation (None inside a generic impl), holding.
    P(u32, u32, Option<Sc>, How),
    E(u32, u32),
    Arr(Sc, u32),
    Hidden,
}

#[derive(Clone, Debug)]
struct Local {
    name: String,
    k: LK,
    mutable: bool,
    assignable: bool,
    moved: bool,
    depth: u32,
}

/// What a `return` needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RetCtx {
    Unit,
    Sc(Sc),
    S(u32),
    E(u32),
    Opt(Sc),
    Res(Sc, u32),
    Tup(Sc, Sc),
    /// The final line is a template written by the caller; no early returns.
    Fixed,
}

/// Expression flags.
#[derive(Clone, Copy, Debug)]
struct E {
    /// The context types an unsuffixed literal.
    lit: bool,
    /// Read no struct place (operands of `&mut` statements and compound assignments).
    ns: bool,
    /// Remaining depth.
    d: u32,
}

impl E {
    fn sub(self) -> E {
        E { d: self.d.saturating_sub(1), ..self }
    }
    fn typed(self) -> E {
        E { lit: true, ..self }
    }
    fn untyped(self) -> E {
        E { lit: false, ..self }
    }
}

#[derive(Clone, Copy)]
enum Src {
    Local(usize),
    Field(usize, u32),
    Field2(usize, u32, u32),
    Elem(usize, u32),
    Const(u32, u32),
}

const SPACES: &str = "                                                                                ";

pub struct Body<'o, 'a> {
    cx: &'o Ctx<'a>,
    out: &'o mut String,
    r: &'o mut Rng,
    vis: Vis,
    locals: Vec<Local>,
    loop_depth: u32,
    nest: u32,
    next: u32,
    rank: u8,
    ret: RetCtx,
}

impl<'o, 'a> Body<'o, 'a> {
    pub fn new(
        cx: &'o Ctx<'a>,
        out: &'o mut String,
        r: &'o mut Rng,
        vis: Vis,
        rank: u8,
        ret: RetCtx,
    ) -> Self {
        Body { cx, out, r, vis, locals: Vec::new(), loop_depth: 0, nest: 0, next: 0, rank, ret }
    }

    fn plans(&self) -> &'a [ModPlan] {
        self.cx.plans
    }

    fn mix(&self) -> &'a Mix {
        self.cx.mix
    }

    fn depth(&mut self) -> u32 {
        let d = self.mix().expr.max_depth;
        1 + self.r.below(d)
    }

    fn e(&mut self) -> E {
        E { lit: true, ns: false, d: self.depth() }
    }

    fn ind(&mut self, n: u32) {
        let n = (n as usize * 4).min(SPACES.len());
        self.out.push_str(&SPACES[..n]);
    }

    fn fresh(&mut self, prefix: char) -> String {
        let mut s = String::with_capacity(6);
        s.push(prefix);
        let _ = write!(s, "{}", self.next);
        self.next += 1;
        s
    }

    fn push_local(&mut self, name: String, k: LK, mutable: bool) {
        let depth = self.loop_depth;
        self.locals.push(Local { name, k, mutable, assignable: mutable, moved: false, depth });
    }

    /// Registers parameter `p<i>`.
    pub fn param(&mut self, i: usize, p: PTy) {
        let me = self.cx.me;
        let k = match p {
            PTy::Sc(t) => LK::Sc(t),
            PTy::Ref(s) => LK::P(me, s, None, How::Ref),
            PTy::Mut(s) => LK::P(me, s, None, How::Mut),
            PTy::Val(s) => LK::P(me, s, None, How::Owned),
            PTy::EVal(e) => LK::E(me, e),
            PTy::T | PTy::RefT => LK::Hidden,
        };
        self.push_local(format!("p{i}"), k, false);
    }

    /// Registers `self` of own struct `s`.
    pub fn self_param(&mut self, s: u32, how: How) {
        let me = self.cx.me;
        self.push_local("self".into(), LK::P(me, s, None, how), false);
    }

    fn qual(&mut self, m: u32) {
        if m != self.cx.me {
            let _ = write!(self.out, "m{m}.");
        }
    }

    fn struct_ty(&mut self, m: u32, s: u32, inst: Option<Sc>) {
        self.qual(m);
        self.out.push_str(&self.plans()[m as usize].structs[s as usize].name);
        if let Some(t) = inst {
            self.out.push('<');
            self.out.push_str(t.name());
            self.out.push('>');
        }
    }

    fn enum_ty(&mut self, m: u32, e: u32) {
        self.qual(m);
        self.out.push_str(&self.plans()[m as usize].enums[e as usize].name);
    }

    fn rank_ok(&self, callee: u8) -> bool {
        if self.loop_depth > 0 { callee == 0 && self.rank > 0 } else { callee < self.rank }
    }

    fn fn_def(&self, f: FnRef) -> &'a FnDef {
        &self.plans()[f.0 as usize].fns[f.1 as usize]
    }

    fn pick_fn(&mut self, key: Key) -> Option<FnRef> {
        let pool = self.cx.pools.0.get(&key)?;
        let own_vis = pool.own.partition_point(|&f| f < self.vis.fns);
        let total = own_vis + pool.ext.len();
        if total == 0 {
            return None;
        }
        for _ in 0..3 {
            let i = self.r.below(total as u32) as usize;
            let f = if i < own_vis { (self.cx.me, pool.own[i]) } else { pool.ext[i - own_vis] };
            if self.rank_ok(self.fn_def(f).rank) {
                return Some(f);
            }
        }
        None
    }

    fn impl_visible(&self, m: u32, s: u32) -> Option<&'a ImplDef> {
        let st = &self.plans()[m as usize].structs[s as usize];
        let i = st.imp?;
        if m != self.cx.me || i < self.vis.impls {
            Some(&self.plans()[m as usize].impls[i as usize])
        } else {
            None
        }
    }

    fn sc(&mut self) -> Sc {
        Sc::ALL[self.r.weighted(&Sc::WEIGHTS)]
    }

    fn int_sc(&mut self) -> Sc {
        Sc::INTS[self.r.weighted(&Sc::WEIGHTS[..6])]
    }

    // ---------- literals and sources ----------

    /// Writes a literal; returns true when it is negative (needs parentheses as an operand).
    fn lit(&mut self, ty: Sc, typed: bool) -> bool {
        let mut neg = false;
        match ty {
            Sc::Bool => self.out.push_str(if self.r.chance(500) { "true" } else { "false" }),
            Sc::F64 => {
                const FR: [&str; 6] = ["0", "5", "25", "125", "75", "0625"];
                let a = if self.r.chance(800) { self.r.below(10) } else { self.r.below(1000) };
                let f = FR[self.r.below(6) as usize];
                if self.r.chance(30) {
                    let _ = write!(self.out, "{}.{f}e{}", a % 10, self.r.below(4));
                } else {
                    let _ = write!(self.out, "{a}.{f}");
                }
                if !typed {
                    self.out.push_str("f64");
                }
            }
            _ => {
                let max = match ty {
                    Sc::U8 => 200,
                    Sc::U16 => 5000,
                    _ => 9000,
                };
                let v = if self.r.chance(800) { self.r.below(17) } else { self.r.below(max) };
                neg = ty.signed() && v > 0 && self.r.chance(150);
                if neg {
                    self.out.push('-');
                }
                let form = self.r.below(100);
                if form < 5 && !neg {
                    let _ = write!(self.out, "0x{v:x}");
                } else if form < 7 && v < 64 && !neg {
                    let _ = write!(self.out, "0b{v:b}");
                } else if v >= 1000 && form < 50 {
                    let _ = write!(self.out, "{}_{:03}", v / 1000, v % 1000);
                } else {
                    let _ = write!(self.out, "{v}");
                }
                if !typed {
                    self.out.push_str(ty.name());
                }
            }
        }
        neg
    }

    /// A readable scalar of type `ty`. `fstr`: names and field paths only (f-string holes).
    fn find_src(&mut self, ty: Sc, ns: bool, fstr: bool) -> Option<Src> {
        let plans = self.plans();
        let n = self.locals.len() as u32;
        for _ in 0..if n == 0 { 0 } else { 6 } {
            let li = self.r.below(n) as usize;
            let l = &self.locals[li];
            if l.moved {
                continue;
            }
            match l.k {
                LK::Sc(t) if t == ty => return Some(Src::Local(li)),
                LK::P(m, s, inst, _) if !ns => {
                    let st = &plans[m as usize].structs[s as usize];
                    if st.fields.is_empty() {
                        continue;
                    }
                    let fi = self.r.below(st.fields.len() as u32);
                    match st.fields[fi as usize].ty {
                        FTy::Sc(t) if t == ty => return Some(Src::Field(li, fi)),
                        FTy::T if inst == Some(ty) => return Some(Src::Field(li, fi)),
                        FTy::S(inner) => {
                            let is = &plans[m as usize].structs[inner as usize];
                            let gi = self.r.below(is.fields.len() as u32);
                            if is.fields[gi as usize].ty == FTy::Sc(ty) {
                                return Some(Src::Field2(li, fi, gi));
                            }
                        }
                        _ => {}
                    }
                }
                LK::Arr(t, len) if t == ty && !fstr => return Some(Src::Elem(li, self.r.below(len))),
                _ => {}
            }
        }
        if fstr {
            return None;
        }
        let own = self.vis.consts;
        let total = own + self.cx.ext_consts.len() as u32;
        for _ in 0..if total == 0 { 0 } else { 2 } {
            let i = self.r.below(total);
            let (m, c) = if i < own { (self.cx.me, i) } else { self.cx.ext_consts[(i - own) as usize] };
            if plans[m as usize].consts[c as usize].ty == ty {
                return Some(Src::Const(m, c));
            }
        }
        None
    }

    fn write_src(&mut self, s: Src) {
        let plans = self.plans();
        match s {
            Src::Local(li) => self.out.push_str(&self.locals[li].name),
            Src::Field(li, fi) | Src::Field2(li, fi, _) => {
                let LK::P(m, st, _, _) = self.locals[li].k else { return };
                let sd = &plans[m as usize].structs[st as usize];
                self.out.push_str(&self.locals[li].name);
                self.out.push('.');
                self.out.push_str(sd.fields[fi as usize].name);
                if let (Src::Field2(_, _, gi), FTy::S(inner)) = (s, sd.fields[fi as usize].ty) {
                    self.out.push('.');
                    self.out.push_str(plans[m as usize].structs[inner as usize].fields[gi as usize].name);
                }
            }
            Src::Elem(li, k) => {
                self.out.push_str(&self.locals[li].name);
                let _ = write!(self.out, "[{k}usize]");
            }
            Src::Const(m, c) => {
                self.qual(m);
                self.out.push_str(&plans[m as usize].consts[c as usize].name);
            }
        }
    }

    /// A place (struct local) of `(m, s)`: `shared` = borrowable as `&S`, else `&mut S` (owned, mut).
    fn find_place(&mut self, m: u32, s: u32, shared: bool) -> Option<usize> {
        let n = self.locals.len();
        if n == 0 {
            return None;
        }
        let start = self.r.below(n as u32) as usize;
        (0..n).map(|j| (start + j) % n).find(|&li| {
            let l = &self.locals[li];
            match l.k {
                LK::P(lm, ls, _, how) if lm == m && ls == s && !l.moved => {
                    if shared {
                        matches!(how, How::Owned | How::Ref | How::SelfRef | How::SelfVal)
                    } else {
                        how == How::Owned && l.mutable
                    }
                }
                _ => false,
            }
        })
    }

    /// `&s`, `p0`, `self` or `&self`: a shared borrow of place `li`.
    fn write_shared(&mut self, li: usize) {
        let LK::P(_, _, _, how) = self.locals[li].k else { return };
        if matches!(how, How::Owned | How::SelfVal) {
            self.out.push('&');
        }
        self.out.push_str(&self.locals[li].name);
    }

    /// A random unmoved place local.
    fn any_place(&mut self) -> Option<usize> {
        let n = self.locals.len() as u32;
        for _ in 0..if n == 0 { 0 } else { 4 } {
            let li = self.r.below(n) as usize;
            if matches!(self.locals[li].k, LK::P(..)) && !self.locals[li].moved {
                return Some(li);
            }
        }
        None
    }

    fn scalar_local(&mut self, ty: Sc) -> Option<usize> {
        let n = self.locals.len() as u32;
        for _ in 0..if n == 0 { 0 } else { 6 } {
            let li = self.r.below(n) as usize;
            if self.locals[li].k == LK::Sc(ty) {
                return Some(li);
            }
        }
        None
    }

    // ---------- expressions ----------

    /// Wraps a compound sub-expression in parentheses.
    fn operand(&mut self, ty: Sc, e: E) {
        let at = self.out.len();
        if self.expr(ty, e) {
            self.out.insert(at, '(');
            self.out.push(')');
        }
    }

    fn leaf(&mut self, ty: Sc, e: E) -> bool {
        if self.r.chance(650)
            && let Some(s) = self.find_src(ty, e.ns, false)
        {
            self.write_src(s);
            return false;
        }
        self.lit(ty, e.lit)
    }

    /// Writes an expression of type `ty`; returns true when it is not a primary/postfix form.
    fn expr(&mut self, ty: Sc, e: E) -> bool {
        if ty == Sc::Bool {
            return self.bool_expr(e);
        }
        if e.d == 0 {
            return self.leaf(ty, e);
        }
        let w = &self.mix().expr;
        match self.r.weighted(&[w.leaf, w.binop, w.cast, w.unary, w.call, w.r#if, w.r#match]) {
            1 => {
                self.binop(ty, e);
                true
            }
            2 => {
                self.cast(ty, e);
                true
            }
            3 => self.unary(ty, e),
            4 => match self.call_expr(ty, e) {
                Some(c) => c,
                None => self.leaf(ty, e),
            },
            5 => {
                self.vif(ty, e);
                true
            }
            6 => self.vmatch(ty, e) || self.leaf(ty, e),
            _ => self.leaf(ty, e),
        }
    }

    fn binop(&mut self, ty: Sc, e: E) {
        let le = e.sub();
        let re = le.typed();
        if ty == Sc::F64 {
            const OPS: [&str; 4] = [" + ", " - ", " * ", " / "];
            self.operand(ty, le);
            self.out.push_str(OPS[self.r.below(4) as usize]);
            self.operand(ty, re);
            return;
        }
        // Trapping `+ - *`: a variable lhs and a small literal rhs, so no operation
        // overflows by construction (constants) and unsigned `-` is always written `-%`/`-|`.
        let trap = self.mix().expr.trapping_permille;
        if self.r.chance(trap)
            && let Some(src) = self.find_src(ty, e.ns, false)
        {
            let ops: &[&str] = if ty.signed() { &[" + ", " - ", " * "] } else { &[" + ", " * "] };
            self.write_src(src);
            self.out.push_str(ops[self.r.below(ops.len() as u32) as usize]);
            let k = 1 + self.r.below(9);
            let _ = write!(self.out, "{k}");
            return;
        }
        match self.r.weighted(&[40, 15, 20, 10, 5]) {
            0 => {
                const OPS: [&str; 3] = [" +% ", " -% ", " *% "];
                self.operand(ty, le);
                self.out.push_str(OPS[self.r.below(3) as usize]);
                self.operand(ty, re);
            }
            1 => {
                const OPS: [&str; 3] = [" +| ", " -| ", " *| "];
                self.operand(ty, le);
                self.out.push_str(OPS[self.r.below(3) as usize]);
                self.operand(ty, re);
            }
            2 => {
                const OPS: [&str; 3] = [" & ", " | ", " ^ "];
                self.operand(ty, le);
                self.out.push_str(OPS[self.r.below(3) as usize]);
                self.operand(ty, re);
            }
            3 => {
                self.operand(ty, le);
                match self.r.below(3) {
                    0 => self.out.push_str(" << "),
                    1 => self.out.push_str(" >> "),
                    _ => {
                        self.out.push_str(" <<% ");
                        if self.r.chance(300) {
                            self.operand(ty, re);
                            return;
                        }
                    }
                }
                let k = 1 + self.r.below(ty.bits() - 1);
                let _ = write!(self.out, "{k}");
            }
            _ => {
                self.operand(ty, le);
                self.out.push_str(if self.r.chance(500) { " / " } else { " % " });
                let k = 1 + self.r.below(9);
                let _ = write!(self.out, "{k}");
            }
        }
    }

    fn cast(&mut self, ty: Sc, e: E) {
        let inner = e.sub().untyped();
        let ll = ty.lossless_from();
        if ty == Sc::F64 {
            if self.r.chance(600) {
                let src = ll[self.r.below(ll.len() as u32) as usize];
                self.operand(src, inner);
                self.out.push_str(" as f64");
            } else {
                let src = if self.r.chance(500) { Sc::I64 } else { Sc::U64 };
                self.operand(src, inner);
                self.out.push_str(" as~ f64");
            }
            return;
        }
        let c = self.r.below(100);
        if c < 50 && !ll.is_empty() {
            let src = ll[self.r.below(ll.len() as u32) as usize];
            self.operand(src, inner);
            self.out.push_str(" as ");
        } else if c < 75 {
            let mut src = self.int_sc();
            if src == ty {
                src = if ty == Sc::I64 { Sc::U64 } else { Sc::I64 };
            }
            self.operand(src, inner);
            self.out.push_str(" as% ");
        } else {
            let src = if self.r.chance(400) {
                Sc::F64
            } else if ty == Sc::I64 {
                Sc::U64
            } else {
                Sc::I64
            };
            self.operand(src, inner);
            self.out.push_str(" as| ");
        }
        self.out.push_str(ty.name());
    }

    fn unary(&mut self, ty: Sc, e: E) -> bool {
        let inner = e.sub();
        if ty == Sc::F64 || (ty.signed() && self.r.chance(300)) {
            self.out.push('-');
        } else {
            self.out.push('~');
        }
        self.operand(ty, inner);
        true
    }

    fn vif(&mut self, ty: Sc, e: E) {
        let s = e.sub();
        self.out.push_str("if ");
        self.operand(Sc::Bool, s.typed());
        self.out.push_str(" then ");
        self.operand(ty, s);
        self.out.push_str(" else ");
        self.operand(ty, s.typed());
    }

    fn bool_expr(&mut self, e: E) -> bool {
        if e.d == 0 {
            return self.leaf(Sc::Bool, e);
        }
        let s = e.sub();
        match self.r.weighted(&[15, 50, 8, 15, 12]) {
            1 => {
                self.cmp(s);
                true
            }
            2 => {
                self.out.push('!');
                self.operand(Sc::Bool, s);
                true
            }
            3 => {
                self.operand(Sc::Bool, s);
                self.out.push_str(if self.r.chance(500) { " && " } else { " || " });
                self.operand(Sc::Bool, s);
                true
            }
            4 => match self.call_expr(Sc::Bool, e) {
                Some(c) => c,
                None => self.leaf(Sc::Bool, e),
            },
            _ => self.leaf(Sc::Bool, e),
        }
    }

    fn cmp(&mut self, e: E) {
        const OPS: [&str; 6] = [" == ", " != ", " < ", " <= ", " > ", " >= "];
        let n = self.locals.len() as u32;
        let mut t = None;
        for _ in 0..if n == 0 { 0 } else { 3 } {
            let li = self.r.below(n) as usize;
            if let LK::Sc(x) = self.locals[li].k
                && x != Sc::Bool
            {
                t = Some(x);
                break;
            }
        }
        let t = match t {
            Some(t) => t,
            None => self.int_sc(),
        };
        self.operand(t, e.untyped());
        self.out.push_str(OPS[self.r.below(6) as usize]);
        self.operand(t, e.typed());
    }

    /// Value-form match; false (nothing written) when no scrutinee is available.
    fn vmatch(&mut self, ty: Sc, e: E) -> bool {
        let s = e.sub();
        let at = self.out.len();
        let ok = match self.r.below(3) {
            0 => self.int_vmatch(ty, e, s),
            1 => self.opt_vmatch(ty, s),
            _ => self.enum_vmatch(ty, s),
        };
        if !ok {
            self.out.truncate(at);
        }
        ok
    }

    fn int_vmatch(&mut self, ty: Sc, e: E, s: E) -> bool {
        let st = self.int_sc();
        let Some(src) = self.find_src(st, e.ns, false) else { return false };
        self.out.push_str("match ");
        self.write_src(src);
        self.out.push_str(" { ");
        let n = 1 + self.r.below(3);
        let base = self.r.below(4);
        for i in 0..n {
            let _ = write!(self.out, "{} => ", base + i * 2);
            self.expr(ty, if i == 0 { s } else { s.typed() });
            self.out.push_str(", ");
        }
        self.out.push_str("_ => ");
        self.expr(ty, s.typed());
        self.out.push_str(" }");
        true
    }

    fn opt_vmatch(&mut self, ty: Sc, s: E) -> bool {
        let opt = self.r.chance(600);
        let Some(f) = self.pick_fn(if opt { Key::Opt(ty) } else { Key::Res(ty) }) else { return false };
        self.out.push_str("match ");
        if !self.try_call(f, None, s) {
            return false;
        }
        let b = self.fresh('b');
        let _ = write!(self.out, " {{ {}({b}) => ", if opt { "Option.Some" } else { "Result.Ok" });
        let mark = self.locals.len();
        self.push_local(b.clone(), LK::Sc(ty), false);
        if self.r.chance(600) {
            self.out.push_str(&b);
        } else {
            self.expr(ty, s);
        }
        self.locals.truncate(mark);
        self.out.push_str(if opt { ", Option.None => " } else { ", Result.Err(_) => " });
        self.expr(ty, s.typed());
        self.out.push_str(" }");
        true
    }

    fn enum_vmatch(&mut self, ty: Sc, s: E) -> bool {
        self.out.push_str("match ");
        let Some((m, en)) = self.enum_scrutinee(s) else { return false };
        self.out.push_str(" { ");
        let nv = self.plans()[m as usize].enums[en as usize].variants.len();
        for vi in 0..nv {
            let mark = self.locals.len();
            if vi > 0 {
                self.out.push_str(", ");
            }
            if vi + 1 == nv && nv >= 3 && self.r.chance(200) {
                self.out.push('_');
            } else {
                self.variant_pat(m, en, vi);
            }
            self.out.push_str(" => ");
            self.expr(ty, if vi == 0 { s } else { s.typed() });
            self.locals.truncate(mark);
        }
        self.out.push_str(" }");
        true
    }

    fn visible_enum(&mut self) -> Option<(u32, u32)> {
        let own = self.vis.enums;
        let total = own + self.cx.ext_enums.len() as u32;
        if total == 0 {
            return None;
        }
        let i = self.r.below(total);
        Some(if i < own { (self.cx.me, i) } else { self.cx.ext_enums[(i - own) as usize] })
    }

    fn visible_struct(&mut self) -> Option<(u32, u32)> {
        let own = self.vis.structs;
        let total = own + self.cx.ext_structs.len() as u32;
        if total == 0 {
            return None;
        }
        let i = self.r.below(total);
        Some(if i < own { (self.cx.me, i) } else { self.cx.ext_structs[(i - own) as usize] })
    }

    /// Writes an enum-valued scrutinee (a call or a variant literal; never a local).
    fn enum_scrutinee(&mut self, s: E) -> Option<(u32, u32)> {
        if self.r.chance(500)
            && let Some(f) = self.pick_fn(Key::E)
            && let Ret::E(en) = self.fn_def(f).ret
        {
            let at = self.out.len();
            if self.try_call(f, None, s) {
                return Some((f.0, en));
            }
            self.out.truncate(at);
        }
        let (m, en) = self.visible_enum()?;
        self.enum_lit(m, en, s);
        Some((m, en))
    }

    /// A call producing `ty`: `Some(compound)` on success; nothing written on failure.
    fn call_expr(&mut self, ty: Sc, e: E) -> Option<bool> {
        let s = e.sub();
        for _ in 0..3 {
            let at = self.out.len();
            match self.r.below(10) {
                0..=3 => {
                    if let Some(f) = self.pick_fn(Key::Sc(ty))
                        && self.try_call(f, None, s)
                    {
                        return Some(false);
                    }
                }
                4 => {
                    if let Some(f) = self.pick_fn(Key::Pick)
                        && self.try_call(f, Some(ty), s)
                    {
                        return Some(false);
                    }
                }
                5 => {
                    let key = if ty == Sc::U32 && self.r.chance(500) { Key::Const } else { Key::Bound };
                    if let Some(f) = self.pick_fn(key)
                        && self.fn_def(f).ret == Ret::Sc(ty)
                        && self.try_call(f, None, s)
                    {
                        return Some(false);
                    }
                }
                6 | 7 => {
                    if let Some(c) = self.method_call(ty, s) {
                        return Some(c);
                    }
                }
                8 => {
                    self.out.push('*');
                    if self.r.chance(600) {
                        if let Some(f) = self.pick_fn(Key::RefSc(ty))
                            && self.try_call(f, None, s)
                        {
                            return Some(true);
                        }
                    } else if let Some(f) = self.pick_fn(Key::RefPick)
                        && self.try_call(f, Some(ty), s)
                    {
                        return Some(true);
                    }
                }
                _ => {
                    if self.r.chance(500) {
                        if let Some(f) = self.pick_fn(Key::Tup)
                            && let Ret::Tup(a, b) = self.fn_def(f).ret
                            && (a == ty || b == ty)
                            && self.try_call(f, None, s)
                        {
                            self.out.push_str(if a == ty { ".0" } else { ".1" });
                            return Some(false);
                        }
                    } else if let Some(f) = self.pick_fn(Key::RefS)
                        && let Ret::RefS(st) = self.fn_def(f).ret
                    {
                        let sd = &self.plans()[f.0 as usize].structs[st as usize];
                        if let Some(fl) = sd.fields.iter().find(|x| x.ty == FTy::Sc(ty))
                            && self.try_call(f, None, s)
                        {
                            self.out.push('.');
                            self.out.push_str(fl.name);
                            return Some(false);
                        }
                    }
                }
            }
            self.out.truncate(at);
        }
        None
    }

    /// A shared place of a struct of module `m` implementing trait `tr` (visible impl).
    fn bound_place(&mut self, m: u32, tr: u32) -> Option<usize> {
        let plans = self.plans();
        let n = self.locals.len();
        if n == 0 {
            return None;
        }
        let start = self.r.below(n as u32) as usize;
        let (me, vt) = (self.cx.me, self.vis.timpls);
        (0..n).map(|j| (start + j) % n).find(|&li| {
            let l = &self.locals[li];
            match l.k {
                LK::P(lm, ls, _, how) if lm == m && !l.moved => {
                    matches!(how, How::Owned | How::Ref | How::SelfRef | How::SelfVal)
                        && plans[m as usize].structs[ls as usize]
                            .traits
                            .iter()
                            .any(|&(t, ti)| t == tr && (m != me || ti < vt))
                }
                _ => false,
            }
        })
    }

    /// Writes a call of `f` (generic fns instantiated with `inst`); false (nothing written)
    /// when an argument cannot be formed here.
    fn try_call(&mut self, f: FnRef, inst: Option<Sc>, s: E) -> bool {
        let fd = self.fn_def(f);
        let mut picks = [usize::MAX; 8];
        let mut bound_ty = None;
        for (j, p) in fd.params.iter().enumerate().take(8) {
            match *p {
                PTy::Ref(st) => {
                    if s.ns {
                        return false;
                    }
                    match self.find_place(f.0, st, true) {
                        Some(li) => picks[j] = li,
                        None => return false,
                    }
                }
                PTy::Mut(_) => return false,
                PTy::RefT => match (fd.kind, inst) {
                    (FnKind::RefPick, Some(t)) => match self.scalar_local(t) {
                        Some(li) => picks[j] = li,
                        None => return false,
                    },
                    (FnKind::Bound(tr), _) => {
                        if s.ns {
                            return false;
                        }
                        match self.bound_place(f.0, tr) {
                            Some(li) => {
                                picks[j] = li;
                                if let LK::P(m, st, _, _) = self.locals[li].k {
                                    bound_ty = Some((m, st));
                                }
                            }
                            None => return false,
                        }
                    }
                    _ => return false,
                },
                PTy::T if inst.is_none() => return false,
                _ => {}
            }
        }
        self.qual(f.0);
        self.out.push_str(&fd.name);
        match fd.kind {
            FnKind::Pick | FnKind::RefPick | FnKind::GenCtor(_) => {
                let t = inst.unwrap_or(Sc::U32);
                let _ = write!(self.out, "::<{}>", t.name());
            }
            FnKind::Bound(_) => {
                if let Some((m, st)) = bound_ty {
                    self.out.push_str("::<");
                    self.struct_ty(m, st, None);
                    self.out.push('>');
                }
            }
            FnKind::Const => {
                let n = 1 + self.r.below(8);
                let _ = write!(self.out, "::<{n}>");
            }
            FnKind::Plain => {}
        }
        self.out.push('(');
        for (j, p) in fd.params.iter().enumerate() {
            if j > 0 {
                self.out.push_str(", ");
            }
            match *p {
                PTy::Sc(t) => {
                    self.expr(t, s.typed());
                }
                PTy::Ref(_) => self.write_shared(picks[j]),
                PTy::Val(st) => self.struct_lit(f.0, st, None, s),
                PTy::EVal(en) => self.enum_lit(f.0, en, s),
                PTy::T => {
                    self.expr(inst.unwrap_or(Sc::U32), s.untyped());
                }
                PTy::RefT => {
                    if fd.kind == FnKind::RefPick {
                        self.out.push('&');
                        self.out.push_str(&self.locals[picks[j]].name);
                    } else {
                        self.write_shared(picks[j]);
                    }
                }
                PTy::Mut(_) => {}
            }
        }
        self.out.push(')');
        true
    }

    /// A method call on a place producing `ty`: `Some(compound)`; nothing written on failure.
    fn method_call(&mut self, ty: Sc, s: E) -> Option<bool> {
        if s.ns {
            return None;
        }
        let li = self.any_place()?;
        let LK::P(m, st, inst, _) = self.locals[li].k else { return None };
        let plans = self.plans();
        let sd = &plans[m as usize].structs[st as usize];
        if m == self.cx.me && !sd.traits.is_empty() && self.rank_ok(0) && self.r.chance(400) {
            let (tr, ti) = sd.traits[self.r.below(sd.traits.len() as u32) as usize];
            let td = &plans[m as usize].traits[tr as usize];
            let tm = &td.methods[self.r.below(td.methods.len() as u32) as usize];
            if ti < self.vis.timpls && tm.ret == ty {
                self.out.push_str(&self.locals[li].name);
                self.out.push('.');
                self.out.push_str(&tm.name);
                self.out.push('(');
                if let Some(a) = tm.arg {
                    self.expr(a, s.typed());
                }
                self.out.push(')');
                return Some(false);
            }
        }
        let imp = self.impl_visible(m, st)?;
        let mt = &imp.methods[self.r.below(imp.methods.len() as u32) as usize];
        if !self.rank_ok(mt.rank) {
            return None;
        }
        let fty = |fi: u32| sd.fields[fi as usize].ty;
        let (deref, arg) = match mt.kind {
            MKind::Get(fi) if fty(fi) == FTy::Sc(ty) => (false, None),
            MKind::Mix(fi) if fty(fi) == FTy::Sc(ty) => (false, Some(self.find_place(m, st, true)?)),
            MKind::RefGet(fi) if fty(fi) == FTy::Sc(ty) => (true, None),
            MKind::GCount if ty == Sc::U32 => (false, None),
            MKind::GGet if inst == Some(ty) => (true, None),
            _ => return None,
        };
        if deref {
            self.out.push('*');
        }
        self.out.push_str(&self.locals[li].name);
        self.out.push('.');
        self.out.push_str(&mt.name);
        self.out.push('(');
        if let Some(a) = arg {
            self.write_shared(a);
        }
        self.out.push(')');
        Some(deref)
    }

    fn struct_lit(&mut self, m: u32, st: u32, inst: Option<Sc>, e: E) {
        let sd = &self.plans()[m as usize].structs[st as usize];
        self.qual(m);
        self.out.push_str(&sd.name);
        self.out.push_str(".{ ");
        let fe = E { d: e.d.min(1), ..e }.typed();
        for (j, f) in sd.fields.iter().enumerate() {
            if j > 0 {
                self.out.push_str(", ");
            }
            self.out.push_str(f.name);
            self.out.push_str(": ");
            match f.ty {
                FTy::Sc(t) => {
                    self.expr(t, fe);
                }
                FTy::S(inner) => self.struct_lit(m, inner, None, fe),
                FTy::T => {
                    self.expr(inst.unwrap_or(Sc::U32), fe.untyped());
                }
            }
        }
        self.out.push_str(" }");
    }

    fn variant_name(&mut self, m: u32, en: u32, vi: usize) {
        self.enum_ty(m, en);
        self.out.push('.');
        let name = self.plans()[m as usize].enums[en as usize].variants[vi].name;
        push_pascal(self.out, name);
    }

    fn enum_lit(&mut self, m: u32, en: u32, e: E) {
        let ed = &self.plans()[m as usize].enums[en as usize];
        let vi = self.r.below(ed.variants.len() as u32) as usize;
        self.variant_name(m, en, vi);
        let fe = E { d: e.d.min(1), ..e };
        match &ed.variants[vi].kind {
            VKind::Unit => {}
            VKind::Tup(ts) => {
                self.out.push('(');
                for (j, &t) in ts.iter().enumerate() {
                    if j > 0 {
                        self.out.push_str(", ");
                    }
                    self.expr(t, fe.untyped());
                }
                self.out.push(')');
            }
            VKind::Rec(fs) => {
                self.out.push_str(".{ ");
                for (j, &(n, t)) in fs.iter().enumerate() {
                    if j > 0 {
                        self.out.push_str(", ");
                    }
                    self.out.push_str(n);
                    self.out.push_str(": ");
                    self.expr(t, fe.typed());
                }
                self.out.push_str(" }");
            }
        }
    }

    /// Writes the pattern of variant `vi` and binds its scalars in the current scope.
    fn variant_pat(&mut self, m: u32, en: u32, vi: usize) {
        self.variant_name(m, en, vi);
        match &self.plans()[m as usize].enums[en as usize].variants[vi].kind {
            VKind::Unit => {}
            VKind::Tup(ts) => {
                self.out.push('(');
                for (j, &t) in ts.iter().enumerate() {
                    if j > 0 {
                        self.out.push_str(", ");
                    }
                    if self.r.chance(700) {
                        let b = self.fresh('b');
                        self.out.push_str(&b);
                        self.push_local(b, LK::Sc(t), false);
                    } else {
                        self.out.push('_');
                    }
                }
                self.out.push(')');
            }
            VKind::Rec(fs) => {
                self.out.push_str(".{ ");
                for (j, &(n, t)) in fs.iter().enumerate() {
                    if j > 0 {
                        self.out.push_str(", ");
                    }
                    if self.r.chance(250) {
                        self.out.push_str(n);
                        self.out.push_str(": _");
                    } else if self.r.chance(500) && !self.locals.iter().any(|l| l.name == n) {
                        self.out.push_str(n);
                        self.push_local(n.to_string(), LK::Sc(t), false);
                    } else {
                        let b = self.fresh('b');
                        let _ = write!(self.out, "{n}: {b}");
                        self.push_local(b, LK::Sc(t), false);
                    }
                }
                self.out.push_str(" }");
            }
        }
    }

    // ---------- statements ----------

    /// Exactly `lines` lines of statements at indentation `ind`, in a new scope.
    pub fn block(&mut self, lines: u32, ind: u32) {
        let mark = self.locals.len();
        let mut rem = lines;
        while rem > 0 {
            let used = self.stmt(rem, ind);
            rem -= used.min(rem);
        }
        self.locals.truncate(mark);
    }

    fn stmt(&mut self, rem: u32, ind: u32) -> u32 {
        let s = &self.mix().stmt;
        if rem >= 3 && self.nest < s.max_nest && self.r.chance(s.compound_permille) {
            let span = rem.min(3 + self.r.below(14));
            self.nest += 1;
            let used = self.compound(span, ind);
            self.nest -= 1;
            if let Some(n) = used {
                return n;
            }
        }
        self.ind(ind);
        self.simple(false);
        self.out.push('\n');
        1
    }

    fn compound(&mut self, t: u32, ind: u32) -> Option<u32> {
        let s = &self.mix().stmt;
        let w = [
            s.r#if,
            s.r#while,
            s.r#for,
            s.r#loop,
            s.match_int,
            s.match_enum,
            s.block,
            s.ref_block,
            s.closure,
        ];
        match self.r.weighted(&w) {
            0 => Some(self.if_stmt(t, ind)),
            1 if t >= 4 => Some(self.while_stmt(t, ind)),
            2 => Some(self.for_stmt(t, ind)),
            3 if t >= 5 => Some(self.loop_stmt(t, ind)),
            4 if t >= 4 => self.match_int(t, ind),
            5 => self.match_enum(t, ind),
            6 => Some(self.block_stmt(t, ind)),
            7 if t >= 5 => self.ref_block(ind),
            8 if t >= 7 => self.closure_block(ind),
            _ => None,
        }
    }

    /// `k` parts, each at least 1, summing to `n` (`n >= k`).
    fn split(&mut self, n: u32, k: u32) -> Vec<u32> {
        let mut v = vec![1u32; k as usize];
        let mut left = n.saturating_sub(k);
        let last = v.len().saturating_sub(1);
        for (i, x) in v.iter_mut().enumerate() {
            let a = if i == last { left } else { self.r.below(left + 1) };
            *x += a;
            left -= a;
        }
        v
    }

    /// A condition (bool expression, statement position).
    pub fn cond(&mut self) {
        let e = self.e();
        self.bool_expr(e);
    }

    fn if_stmt(&mut self, t: u32, ind: u32) -> u32 {
        let form = if t >= 7 && self.r.chance(250) {
            2
        } else if t >= 5 && self.r.chance(600) {
            1
        } else {
            0
        };
        let parts = self.split(t - 2 - form, form + 1);
        self.ind(ind);
        self.out.push_str("if ");
        self.cond();
        self.out.push_str(" {\n");
        self.block(parts[0], ind + 1);
        if form == 2 {
            self.ind(ind);
            self.out.push_str("} else if ");
            self.cond();
            self.out.push_str(" {\n");
            self.block(parts[1], ind + 1);
        }
        if form >= 1 {
            self.ind(ind);
            self.out.push_str("} else {\n");
            self.block(parts[form as usize], ind + 1);
        }
        self.ind(ind);
        self.out.push_str("}\n");
        t
    }

    fn counter(&mut self, ind: u32) -> String {
        let c = self.fresh('i');
        self.ind(ind);
        let _ = writeln!(self.out, "let mut {c}: u32 = 0;");
        self.push_local(c.clone(), LK::Sc(Sc::U32), true);
        if let Some(l) = self.locals.last_mut() {
            l.assignable = false;
        }
        c
    }

    fn while_stmt(&mut self, t: u32, ind: u32) -> u32 {
        let c = self.counter(ind);
        let k = self.r.range(2, 8);
        self.ind(ind);
        let _ = writeln!(self.out, "while {c} < {k} {{");
        self.loop_depth += 1;
        self.block(t - 4, ind + 1);
        self.ind(ind + 1);
        let _ = writeln!(self.out, "{c} = {c} + 1;");
        self.loop_depth -= 1;
        self.ind(ind);
        self.out.push_str("}\n");
        t
    }

    fn loop_stmt(&mut self, t: u32, ind: u32) -> u32 {
        let c = self.counter(ind);
        let k = self.r.range(2, 8);
        self.ind(ind);
        self.out.push_str("loop {\n");
        self.loop_depth += 1;
        self.ind(ind + 1);
        let _ = writeln!(self.out, "if {c} >= {k} {{ break; }}");
        self.ind(ind + 1);
        let _ = writeln!(self.out, "{c} = {c} + 1;");
        let mut inner = t - 5;
        if inner >= 2 && self.r.chance(300) {
            self.ind(ind + 1);
            let _ = writeln!(self.out, "if ({c} % 2) == 0 {{ continue; }}");
            inner -= 1;
        }
        self.block(inner, ind + 1);
        self.loop_depth -= 1;
        self.ind(ind);
        self.out.push_str("}\n");
        t
    }

    fn for_stmt(&mut self, t: u32, ind: u32) -> u32 {
        let k = self.fresh('k');
        let mark = self.locals.len();
        let arr = if t >= 3 && self.r.chance(250) {
            self.locals.iter().enumerate().rev().find_map(|(li, l)| match l.k {
                LK::Arr(et, len) if l.mutable => Some((li, et, len)),
                _ => None,
            })
        } else {
            None
        };
        self.ind(ind);
        let mut body = t - 2;
        if let Some((ai, et, len)) = arr {
            let _ = writeln!(self.out, "for {k} in 0..{len}usize {{");
            self.loop_depth += 1;
            self.push_local(k.clone(), LK::Hidden, false);
            self.ind(ind + 1);
            let a = self.locals[ai].name.clone();
            let op = if et == Sc::F64 { "+" } else { "+%" };
            let _ = write!(self.out, "{a}[{k}] = {a}[{k}] {op} ");
            let e = self.e();
            self.operand(et, e);
            self.out.push_str(";\n");
            body -= 1;
        } else {
            let ty = self.int_sc();
            let src = if self.r.chance(500) { self.find_src(ty, false, false) } else { None };
            self.out.push_str("for ");
            self.out.push_str(&k);
            self.out.push_str(" in ");
            match src {
                Some(s) => {
                    self.out.push_str("0..(");
                    self.write_src(s);
                    self.out.push_str(" & 7)");
                }
                None => {
                    let hi = self.r.range(2, 8);
                    if self.r.chance(200) {
                        let _ = write!(self.out, "1{}..={hi}", ty.name());
                    } else {
                        let _ = write!(self.out, "0{}..{hi}", ty.name());
                    }
                }
            }
            self.out.push_str(" {\n");
            self.loop_depth += 1;
            self.push_local(k, LK::Sc(ty), false);
        }
        self.block(body, ind + 1);
        self.locals.truncate(mark);
        self.loop_depth -= 1;
        self.ind(ind);
        self.out.push_str("}\n");
        t
    }

    /// A match arm body of `b` lines (`{ stmt }` on the pattern line when `b == 1`).
    fn arm(&mut self, b: u32, ind: u32) {
        if b == 1 {
            self.out.push_str("{ ");
            let mark = self.locals.len();
            self.simple(true);
            self.locals.truncate(mark);
            self.out.push_str(" }\n");
        } else {
            self.out.push_str("{\n");
            self.block(b - 2, ind + 1);
            self.ind(ind);
            self.out.push_str("}\n");
        }
    }

    fn match_int(&mut self, t: u32, ind: u32) -> Option<u32> {
        let ty = self.int_sc();
        let src = self.find_src(ty, false, false)?;
        let na = self.r.range(1, 4.min(t - 3));
        let parts = self.split(t - 2, na + 1);
        self.ind(ind);
        self.out.push_str("match ");
        self.write_src(src);
        self.out.push_str(" {\n");
        let base = self.r.below(4);
        let step = 1 + self.r.below(3);
        for (i, &b) in parts.iter().enumerate() {
            self.ind(ind + 1);
            let v = base + i as u32 * step;
            if i as u32 == na {
                self.out.push('_');
            } else if i as u32 + 1 == na && self.r.chance(250) {
                let _ = write!(self.out, "{v}..={}", v + 2);
            } else {
                let _ = write!(self.out, "{v}");
            }
            self.out.push_str(" => ");
            self.arm(b, ind + 1);
        }
        self.ind(ind);
        self.out.push_str("}\n");
        Some(t)
    }

    fn match_enum(&mut self, t: u32, ind: u32) -> Option<u32> {
        let at = self.out.len();
        self.ind(ind);
        self.out.push_str("match ");
        // Scrutinee: an owned enum local (moved), an Option/Result call, or an enum value.
        let mut moved = None;
        let mut opt: Option<(bool, Sc)> = None;
        let mut en = None;
        let n = self.locals.len();
        let local = (0..n).rev().take(8).find(|&li| {
            let l = &self.locals[li];
            matches!(l.k, LK::E(..)) && !l.moved && l.depth == self.loop_depth
        });
        match (local, self.r.below(3)) {
            (Some(li), 0 | 1) => {
                if let LK::E(m, e) = self.locals[li].k {
                    self.out.push_str(&self.locals[li].name);
                    moved = Some(li);
                    en = Some((m, e));
                }
            }
            (_, 2) => {
                let o = self.r.chance(600);
                let ty = self.sc();
                if let Some(f) = self.pick_fn(if o { Key::Opt(ty) } else { Key::Res(ty) }) {
                    let e = self.e();
                    if self.try_call(f, None, e) {
                        opt = Some((o, ty));
                    }
                }
            }
            _ => {}
        }
        if en.is_none() && opt.is_none() {
            let e = self.e();
            en = self.enum_scrutinee(e);
        }
        let narms = match (en, opt) {
            (Some((m, e)), _) => self.plans()[m as usize].enums[e as usize].variants.len() as u32,
            (None, Some(_)) => 2,
            (None, None) => {
                self.out.truncate(at);
                return None;
            }
        };
        if t < 2 + narms {
            self.out.truncate(at);
            return None;
        }
        if let Some(li) = moved {
            self.locals[li].moved = true;
        }
        self.out.push_str(" {\n");
        let parts = self.split(t - 2, narms);
        for (i, &b) in parts.iter().enumerate() {
            let mark = self.locals.len();
            self.ind(ind + 1);
            match (en, opt) {
                (Some((m, e)), _) => {
                    if i + 1 == narms as usize && narms >= 3 && self.r.chance(200) {
                        self.out.push('_');
                    } else {
                        self.variant_pat(m, e, i);
                    }
                }
                (None, Some((o, ty))) => {
                    if i == 0 {
                        let b = self.fresh('b');
                        let _ = write!(self.out, "{}({b})", if o { "Option.Some" } else { "Result.Ok" });
                        self.push_local(b, LK::Sc(ty), false);
                    } else {
                        self.out.push_str(if o { "Option.None" } else { "Result.Err(_)" });
                    }
                }
                (None, None) => {}
            }
            self.out.push_str(" => ");
            self.arm(b, ind + 1);
            self.locals.truncate(mark);
        }
        self.ind(ind);
        self.out.push_str("}\n");
        Some(t)
    }

    fn block_stmt(&mut self, t: u32, ind: u32) -> u32 {
        self.ind(ind);
        self.out.push_str("{\n");
        self.block(t - 2, ind + 1);
        self.ind(ind);
        self.out.push_str("}\n");
        t
    }

    /// `let mut v = ..; { let r: &T = <ref>; v = *r; }`: a returned reference used in its own block.
    fn ref_block(&mut self, ind: u32) -> Option<u32> {
        let at = self.out.len();
        let v = self.fresh('v');
        let r = self.fresh('r');
        let e = self.e();
        let mut ty = self.sc();
        // Decide the reference source first: 0 RefSc call, 1 RefGet method, 2 generic ref pick, 3 RefS call.
        let choice = self.r.below(4);
        let mut field: Option<&'static str> = None;
        let mut rs: Option<(u32, u32)> = None;
        let mut method: Option<(usize, &'a str)> = None;
        let mut f = None;
        match choice {
            0 => f = Some((self.pick_fn(Key::RefSc(ty))?, None)),
            1 => {
                let li = self.any_place()?;
                let LK::P(m, st, _, _) = self.locals[li].k else { return None };
                let imp = self.impl_visible(m, st)?;
                let mt = &imp.methods[self.r.below(imp.methods.len() as u32) as usize];
                let MKind::RefGet(fi) = mt.kind else { return None };
                if !self.rank_ok(mt.rank) {
                    return None;
                }
                let FTy::Sc(t) = self.plans()[m as usize].structs[st as usize].fields[fi as usize].ty else {
                    return None;
                };
                ty = t;
                method = Some((li, mt.name.as_str()));
            }
            2 => f = Some((self.pick_fn(Key::RefPick)?, Some(ty))),
            _ => {
                let g = self.pick_fn(Key::RefS)?;
                let Ret::RefS(st) = self.fn_def(g).ret else { return None };
                let sd = &self.plans()[g.0 as usize].structs[st as usize];
                let fl = &sd.fields[self.r.below(sd.fields.len() as u32) as usize];
                let FTy::Sc(t) = fl.ty else { return None };
                ty = t;
                field = Some(fl.name);
                rs = Some((g.0, st));
                f = Some((g, None));
            }
        }
        self.ind(ind);
        let _ = write!(self.out, "let mut {v}: {} = ", ty.name());
        self.lit(ty, true);
        self.out.push_str(";\n");
        self.ind(ind);
        self.out.push_str("{\n");
        self.ind(ind + 1);
        let _ = write!(self.out, "let {r}: &");
        match rs {
            Some((m, st)) => self.struct_ty(m, st, None),
            None => self.out.push_str(ty.name()),
        }
        self.out.push_str(" = ");
        if let Some((li, name)) = method {
            self.out.push_str(&self.locals[li].name);
            self.out.push('.');
            self.out.push_str(name);
            self.out.push_str("()");
        } else if let Some((g, inst)) = f
            && !self.try_call(g, inst, e)
        {
            self.out.truncate(at);
            return None;
        }
        self.out.push_str(";\n");
        self.ind(ind + 1);
        match field {
            Some(fl) => {
                let _ = writeln!(self.out, "{v} = {r}.{fl};");
            }
            None => {
                let _ = writeln!(self.out, "{v} = *{r};");
            }
        }
        self.ind(ind);
        self.out.push_str("}\n");
        self.push_local(v, LK::Sc(ty), true);
        Some(5)
    }

    /// `[m2]` closures: a closure capturing a local by `&` called through a nested
    /// closure that captures it and an outer local by `&mut` (7 lines):
    /// `let mut v: T = lit; { let c: T = lit; let q = fn(x: T) -> T { return x +% c; };
    /// let h = fn() { v = q(v); }; h(); }`.
    fn closure_block(&mut self, ind: u32) -> Option<u32> {
        let ty = self.int_sc();
        let v = self.fresh('v');
        let c = self.fresh('c');
        let q = self.fresh('q');
        let x = self.fresh('x');
        let h = self.fresh('h');
        self.ind(ind);
        let _ = write!(self.out, "let mut {v}: {} = ", ty.name());
        self.lit(ty, true);
        self.out.push_str(";\n");
        self.ind(ind);
        self.out.push_str("{\n");
        self.ind(ind + 1);
        let _ = write!(self.out, "let {c}: {} = ", ty.name());
        self.lit(ty, true);
        self.out.push_str(";\n");
        self.ind(ind + 1);
        let t = ty.name();
        let _ = writeln!(self.out, "let {q} = fn({x}: {t}) -> {t} {{ return {x} +% {c}; }};");
        self.ind(ind + 1);
        let _ = writeln!(self.out, "let {h} = fn() {{ {v} = {q}({v}); }};");
        self.ind(ind + 1);
        let _ = writeln!(self.out, "{h}();");
        self.ind(ind);
        self.out.push_str("}\n");
        self.push_local(v, LK::Sc(ty), true);
        Some(7)
    }

    // ---------- simple statements (one line, no indentation or newline) ----------

    fn simple(&mut self, inline: bool) {
        let s = &self.mix().stmt;
        let w = [
            s.r#let,
            s.assign,
            s.print,
            s.let_struct,
            s.field_write,
            s.mut_call,
            s.call,
            s.consume,
            s.tuple,
            s.early_return,
            s.let_enum,
            s.r#try,
            s.array,
            if inline { 0 } else { s.comment },
        ];
        for _ in 0..3 {
            let at = self.out.len();
            let ok = match self.r.weighted(&w) {
                0 => {
                    self.let_sc();
                    true
                }
                1 => self.assign(),
                2 => self.print(),
                3 => self.let_struct(),
                4 => self.field_write(),
                5 => self.mut_call(),
                6 => self.call_stmt(),
                7 => self.consume(),
                8 => self.let_tuple(),
                9 => self.early_return(),
                10 => self.let_enum(),
                11 => self.try_stmt(),
                12 => self.array(),
                _ => {
                    self.comment();
                    true
                }
            };
            if ok {
                return;
            }
            self.out.truncate(at);
        }
        self.let_sc();
    }

    fn let_sc(&mut self) {
        let ty = self.sc();
        let mutable = self.r.chance(400);
        let v = self.fresh('v');
        let _ = write!(self.out, "let {}{v}: {} = ", if mutable { "mut " } else { "" }, ty.name());
        let e = self.e();
        self.expr(ty, e);
        self.out.push(';');
        self.push_local(v, LK::Sc(ty), mutable);
    }

    fn assign(&mut self) -> bool {
        let n = self.locals.len() as u32;
        for _ in 0..if n == 0 { 0 } else { 6 } {
            let li = self.r.below(n) as usize;
            let l = &self.locals[li];
            let LK::Sc(ty) = l.k else { continue };
            if !l.assignable || l.moved {
                continue;
            }
            self.out.push_str(&self.locals[li].name);
            let op = if ty.is_int() {
                const OPS: [&str; 10] =
                    [" = ", " +%= ", " -%= ", " *%= ", " +|= ", " -|= ", " ^= ", " |= ", " &= ", " += "];
                OPS[self.r.weighted(&[30, 15, 8, 6, 8, 5, 6, 5, 5, 3])]
            } else if ty == Sc::F64 {
                const OPS: [&str; 4] = [" = ", " += ", " -= ", " *= "];
                OPS[self.r.below(4) as usize]
            } else {
                " = "
            };
            self.out.push_str(op);
            let e = self.e();
            self.expr(ty, e);
            self.out.push(';');
            return true;
        }
        false
    }

    fn print(&mut self) -> bool {
        if !self.cx.io {
            return false;
        }
        let word = NOUNS[self.r.below(NOUNS.len() as u32) as usize];
        let mut holes = [None, None];
        for h in holes.iter_mut() {
            let ty = Sc::ALL[self.r.weighted(&Sc::WEIGHTS[..7])];
            *h = self.find_src(ty, false, true);
        }
        if holes.iter().all(Option::is_none) {
            let _ = write!(self.out, "io.print(\"{word}\\n\");");
            return true;
        }
        let _ = write!(self.out, "io.print(f\"{word}:");
        for h in holes.into_iter().flatten() {
            self.out.push_str(" {");
            self.write_src(h);
            self.out.push('}');
        }
        self.out.push_str("\\n\");");
        true
    }

    fn let_struct(&mut self) -> bool {
        let v = self.fresh('s');
        let mutable = self.r.chance(500);
        let pre = if mutable { "let mut " } else { "let " };
        let e = self.e();
        if self.r.chance(250)
            && let Some(f) = self.pick_fn(Key::S)
        {
            let Ret::S(st) = self.fn_def(f).ret else { return false };
            self.out.push_str(pre);
            self.out.push_str(&v);
            self.out.push_str(": ");
            self.struct_ty(f.0, st, None);
            self.out.push_str(" = ");
            if !self.try_call(f, None, e) {
                return false;
            }
            self.out.push(';');
            self.push_local(v, LK::P(f.0, st, None, How::Owned), mutable);
            return true;
        }
        if self.r.chance(150)
            && let Some(f) = self.pick_fn(Key::GenCtor)
        {
            let Ret::Cell(gs) = self.fn_def(f).ret else { return false };
            let inst = self.sc();
            self.out.push_str(pre);
            self.out.push_str(&v);
            self.out.push_str(": ");
            self.struct_ty(f.0, gs, Some(inst));
            self.out.push_str(" = ");
            if !self.try_call(f, Some(inst), e) {
                return false;
            }
            self.out.push(';');
            self.push_local(v, LK::P(f.0, gs, Some(inst), How::Owned), mutable);
            return true;
        }
        let Some((m, st)) = self.visible_struct() else { return false };
        let sd = &self.plans()[m as usize].structs[st as usize];
        let inst = if sd.generic { Some(self.sc()) } else { None };
        self.out.push_str(pre);
        self.out.push_str(&v);
        self.out.push_str(": ");
        self.struct_ty(m, st, inst);
        self.out.push_str(" = ");
        let ctor = if sd.generic || !self.r.chance(300) {
            None
        } else {
            self.impl_visible(m, st)
                .and_then(|imp| imp.methods.iter().find(|x| x.kind == MKind::Ctor))
                .filter(|mt| self.rank_ok(mt.rank))
        };
        match ctor {
            Some(mt) => {
                self.struct_ty(m, st, None);
                self.out.push('.');
                self.out.push_str(&mt.name);
                self.out.push('(');
                let mut first = true;
                for f in &sd.fields {
                    if let FTy::Sc(t) = f.ty {
                        if !first {
                            self.out.push_str(", ");
                        }
                        first = false;
                        self.expr(t, e.sub().typed());
                    }
                }
                self.out.push(')');
            }
            None => self.struct_lit(m, st, inst, e),
        }
        self.out.push(';');
        self.push_local(v, LK::P(m, st, inst, How::Owned), mutable);
        true
    }

    fn writable_place(&mut self) -> Option<usize> {
        let n = self.locals.len() as u32;
        for _ in 0..if n == 0 { 0 } else { 6 } {
            let li = self.r.below(n) as usize;
            let l = &self.locals[li];
            if let LK::P(_, _, _, how) = l.k
                && !l.moved
                && ((how == How::Owned && l.mutable) || how == How::Mut || how == How::SelfMut)
            {
                return Some(li);
            }
        }
        None
    }

    fn field_write(&mut self) -> bool {
        let Some(li) = self.writable_place() else { return false };
        let LK::P(m, st, inst, _) = self.locals[li].k else { return false };
        let plans = self.plans();
        let sd = &plans[m as usize].structs[st as usize];
        if sd.fields.is_empty() {
            return false;
        }
        let fi = self.r.below(sd.fields.len() as u32) as usize;
        let f = &sd.fields[fi];
        let (path2, ty) = match f.ty {
            FTy::Sc(t) => (None, t),
            FTy::T => match inst {
                Some(t) => (None, t),
                None => return false,
            },
            FTy::S(inner) => {
                let is = &plans[m as usize].structs[inner as usize];
                let g = &is.fields[self.r.below(is.fields.len() as u32) as usize];
                let FTy::Sc(t) = g.ty else { return false };
                (Some(g.name), t)
            }
        };
        self.out.push_str(&self.locals[li].name);
        self.out.push('.');
        self.out.push_str(f.name);
        if let Some(g) = path2 {
            self.out.push('.');
            self.out.push_str(g);
        }
        let compound = ty.is_int() && self.r.chance(400);
        self.out.push_str(if !compound {
            " = "
        } else if self.r.chance(600) {
            " +%= "
        } else {
            " ^= "
        });
        let e = self.e();
        self.expr(ty, E { ns: compound, ..e });
        self.out.push(';');
        true
    }

    fn mut_call(&mut self) -> bool {
        let e = self.e();
        let ns = E { ns: true, ..e };
        match self.r.below(3) {
            0 => {
                let Some(f) = self.pick_fn(Key::MutUnit) else { return false };
                let fd = self.fn_def(f);
                let Some(PTy::Mut(st)) = fd.params.first().copied() else { return false };
                let Some(li) = self.find_place(f.0, st, false) else { return false };
                self.qual(f.0);
                self.out.push_str(&fd.name);
                self.out.push_str("(&mut ");
                self.out.push_str(&self.locals[li].name);
                for p in &fd.params[1..] {
                    self.out.push_str(", ");
                    let PTy::Sc(t) = *p else { return false };
                    self.expr(t, ns);
                }
                self.out.push_str(");");
                true
            }
            1 => {
                let Some(li) = self.writable_place() else { return false };
                let LK::P(m, st, inst, _) = self.locals[li].k else { return false };
                let Some(imp) = self.impl_visible(m, st) else { return false };
                let mt = &imp.methods[self.r.below(imp.methods.len() as u32) as usize];
                if !self.rank_ok(mt.rank) {
                    return false;
                }
                let sd = &self.plans()[m as usize].structs[st as usize];
                let ty = match (mt.kind, inst) {
                    (MKind::Bump(fi), _) => match sd.fields[fi as usize].ty {
                        FTy::Sc(t) => t,
                        _ => return false,
                    },
                    (MKind::GPut, Some(t)) => t,
                    _ => return false,
                };
                self.out.push_str(&self.locals[li].name);
                self.out.push('.');
                self.out.push_str(&mt.name);
                self.out.push('(');
                self.expr(ty, if mt.kind == MKind::GPut { ns.untyped() } else { ns });
                self.out.push_str(");");
                true
            }
            _ => {
                let ty = self.sc();
                let Some(f) = self.pick_fn(Key::MutSc(ty)) else { return false };
                let fd = self.fn_def(f);
                let Some(PTy::Mut(st)) = fd.params.first().copied() else { return false };
                let Some(li) = self.find_place(f.0, st, false) else { return false };
                self.out.push('*');
                self.qual(f.0);
                self.out.push_str(&fd.name);
                self.out.push_str("(&mut ");
                self.out.push_str(&self.locals[li].name);
                self.out.push_str(") = ");
                self.expr(ty, ns);
                self.out.push(';');
                true
            }
        }
    }

    fn call_stmt(&mut self) -> bool {
        let Some(f) = self.pick_fn(Key::Unit) else { return false };
        let e = self.e();
        if !self.try_call(f, None, e) {
            return false;
        }
        self.out.push(';');
        true
    }

    fn consume(&mut self) -> bool {
        let n = self.locals.len();
        let Some(li) = (0..n).rev().find(|&li| {
            let l = &self.locals[li];
            matches!(l.k, LK::P(_, _, None, How::Owned)) && !l.moved && l.depth == self.loop_depth
        }) else {
            return false;
        };
        let LK::P(m, st, _, _) = self.locals[li].k else { return false };
        let Some(imp) = self.impl_visible(m, st) else { return false };
        let Some(mt) = imp.methods.iter().find(|x| matches!(x.kind, MKind::Consume(_))) else { return false };
        let MKind::Consume(fi) = mt.kind else { return false };
        if !self.rank_ok(mt.rank) {
            return false;
        }
        let FTy::Sc(ty) = self.plans()[m as usize].structs[st as usize].fields[fi as usize].ty else {
            return false;
        };
        let v = self.fresh('v');
        let _ = write!(self.out, "let {v}: {} = {}.{}();", ty.name(), self.locals[li].name, mt.name);
        self.locals[li].moved = true;
        self.push_local(v, LK::Sc(ty), false);
        true
    }

    fn let_tuple(&mut self) -> bool {
        let Some(f) = self.pick_fn(Key::Tup) else { return false };
        let Ret::Tup(a, b) = self.fn_def(f).ret else { return false };
        let (va, vb) = (self.fresh('v'), self.fresh('v'));
        let _ = write!(self.out, "let ({va}, {vb}): ({}, {}) = ", a.name(), b.name());
        let e = self.e();
        if !self.try_call(f, None, e) {
            return false;
        }
        self.out.push(';');
        self.push_local(va, LK::Sc(a), false);
        self.push_local(vb, LK::Sc(b), false);
        true
    }

    fn early_return(&mut self) -> bool {
        if self.ret == RetCtx::Fixed {
            return false;
        }
        self.out.push_str("if ");
        self.cond();
        self.out.push_str(" { return");
        if self.ret != RetCtx::Unit {
            self.out.push(' ');
            self.ret_expr();
        }
        self.out.push_str("; }");
        true
    }

    fn let_enum(&mut self) -> bool {
        let Some((m, en)) = self.visible_enum() else { return false };
        let v = self.fresh('e');
        let _ = write!(self.out, "let {v}: ");
        self.enum_ty(m, en);
        self.out.push_str(" = ");
        let e = self.e();
        self.enum_lit(m, en, e);
        self.out.push(';');
        self.push_local(v, LK::E(m, en), false);
        true
    }

    fn try_stmt(&mut self) -> bool {
        let (f, ty) = match self.ret {
            RetCtx::Opt(_) => {
                let ty = self.sc();
                let Some(f) = self.pick_fn(Key::Opt(ty)) else { return false };
                (f, ty)
            }
            RetCtx::Res(_, en) => {
                let Some(f) = self.pick_fn(Key::ResOwn(en)) else { return false };
                let Ret::Res(ty, _) = self.fn_def(f).ret else { return false };
                (f, ty)
            }
            _ => return false,
        };
        let v = self.fresh('v');
        let _ = write!(self.out, "let {v}: {} = ", ty.name());
        let e = self.e();
        if !self.try_call(f, None, e) {
            return false;
        }
        self.out.push_str("?;");
        self.push_local(v, LK::Sc(ty), false);
        true
    }

    fn array(&mut self) -> bool {
        let e = self.e();
        if self.r.chance(500) {
            let n = self.locals.len();
            if let Some(li) = (0..n).rev().find(|&li| matches!(self.locals[li].k, LK::Arr(..))) {
                let LK::Arr(t, len) = self.locals[li].k else { return false };
                let k = self.r.below(len);
                let _ = write!(self.out, "{}[{k}usize] = ", self.locals[li].name);
                self.expr(t, e);
                self.out.push(';');
                return true;
            }
        }
        let t = if self.r.chance(800) { self.int_sc() } else { Sc::F64 };
        let len = [2u32, 4, 8][self.r.below(3) as usize];
        let v = self.fresh('a');
        let _ = write!(self.out, "let mut {v}: [{}; {len}] = [", t.name());
        if len == 8 || self.r.chance(500) {
            self.expr(t, e);
            let _ = write!(self.out, "; {len}]");
        } else {
            for j in 0..len {
                if j > 0 {
                    self.out.push_str(", ");
                }
                self.expr(t, e.sub());
            }
            self.out.push(']');
        }
        self.out.push(';');
        self.push_local(v, LK::Arr(t, len), true);
        true
    }

    fn comment(&mut self) {
        self.out.push_str("//");
        let n = self.r.range(3, 6);
        for _ in 0..n {
            self.out.push(' ');
            self.out.push_str(NOTES[self.r.below(NOTES.len() as u32) as usize]);
        }
    }

    // ---------- returns ----------

    /// The value of a `return` for the current return context.
    fn ret_expr(&mut self) {
        let e = self.e();
        let me = self.cx.me;
        match self.ret {
            RetCtx::Sc(t) => {
                self.expr(t, e);
            }
            RetCtx::S(st) => self.struct_lit(me, st, None, e),
            RetCtx::E(en) => {
                if self.r.chance(400) {
                    self.out.push_str("if ");
                    self.operand(Sc::Bool, e.sub());
                    self.out.push_str(" then ");
                    self.enum_lit(me, en, e);
                    self.out.push_str(" else ");
                    self.enum_lit(me, en, e);
                } else {
                    self.enum_lit(me, en, e);
                }
            }
            RetCtx::Opt(t) => match self.r.below(3) {
                0 => {
                    self.out.push_str("Option.Some(");
                    self.expr(t, e.untyped());
                    self.out.push(')');
                }
                1 => self.out.push_str("Option.None"),
                _ => {
                    self.out.push_str("if ");
                    self.operand(Sc::Bool, e.sub());
                    self.out.push_str(" then Option.Some(");
                    self.expr(t, e.sub().untyped());
                    self.out.push_str(") else Option.None");
                }
            },
            RetCtx::Res(t, en) => match self.r.below(3) {
                0 => {
                    self.out.push_str("Result.Ok(");
                    self.expr(t, e.untyped());
                    self.out.push(')');
                }
                1 => {
                    self.out.push_str("Result.Err(");
                    self.enum_lit(me, en, e);
                    self.out.push(')');
                }
                _ => {
                    self.out.push_str("if ");
                    self.operand(Sc::Bool, e.sub());
                    self.out.push_str(" then Result.Ok(");
                    self.expr(t, e.sub().untyped());
                    self.out.push_str(") else Result.Err(");
                    self.enum_lit(me, en, e);
                    self.out.push(')');
                }
            },
            RetCtx::Tup(a, b) => {
                self.out.push('(');
                self.expr(a, e.untyped());
                self.out.push_str(", ");
                self.expr(b, e.untyped());
                self.out.push(')');
            }
            RetCtx::Unit | RetCtx::Fixed => {}
        }
    }

    /// The final `return ..;` line of a non-unit body with a generated return value.
    pub fn final_return(&mut self, ind: u32) {
        self.ind(ind);
        self.out.push_str("return ");
        if let RetCtx::Sc(t) = self.ret {
            let n = self.locals.len();
            let el = (0..n).find(|&li| {
                let l = &self.locals[li];
                matches!(l.k, LK::E(..)) && !l.moved && l.depth == self.loop_depth
            });
            if let (Some(li), true) = (el, self.r.chance(600))
                && let LK::E(m, en) = self.locals[li].k
            {
                self.out.push_str("match ");
                self.out.push_str(&self.locals[li].name);
                self.locals[li].moved = true;
                self.out.push_str(" { ");
                let nv = self.plans()[m as usize].enums[en as usize].variants.len();
                let e = self.e();
                for vi in 0..nv {
                    if vi > 0 {
                        self.out.push_str(", ");
                    }
                    let mark = self.locals.len();
                    self.variant_pat(m, en, vi);
                    self.out.push_str(" => ");
                    self.expr(t, e);
                    self.locals.truncate(mark);
                }
                self.out.push_str(" };\n");
                return;
            }
        }
        self.ret_expr();
        self.out.push_str(";\n");
    }

    // ---------- template helpers for emit ----------

    pub fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    pub fn indent(&mut self, n: u32) {
        self.ind(n);
    }

    /// `Name.{ .. }` of own struct `st`, scalar fields taken from params `p0, p1, ..` in order.
    pub fn ctor_lit(&mut self, st: u32) {
        let me = self.cx.me;
        let sd = &self.plans()[me as usize].structs[st as usize];
        self.out.push_str(&sd.name);
        self.out.push_str(".{ ");
        let mut p = 0;
        let e = E { lit: true, ns: false, d: 1 };
        for (j, f) in sd.fields.iter().enumerate() {
            if j > 0 {
                self.out.push_str(", ");
            }
            self.out.push_str(f.name);
            self.out.push_str(": ");
            match f.ty {
                FTy::Sc(_) => {
                    let _ = write!(self.out, "p{p}");
                    p += 1;
                }
                FTy::S(inner) => self.struct_lit(me, inner, None, e),
                FTy::T => self.out.push_str("p0"),
            }
        }
        self.out.push_str(" }");
    }

    /// `Cell.{ a: p0, n: .., .. }` of own generic struct `gs`.
    pub fn gen_lit(&mut self, gs: u32) {
        let me = self.cx.me;
        let sd = &self.plans()[me as usize].structs[gs as usize];
        self.out.push_str(&sd.name);
        self.out.push_str(".{ ");
        for (j, f) in sd.fields.iter().enumerate() {
            if j > 0 {
                self.out.push_str(", ");
            }
            self.out.push_str(f.name);
            self.out.push_str(": ");
            match f.ty {
                FTy::T => self.out.push_str("p0"),
                FTy::Sc(t) => {
                    let e = E { lit: true, ns: false, d: 1 };
                    self.expr(t, e);
                }
                FTy::S(inner) => self.struct_lit(me, inner, None, E { lit: true, ns: false, d: 1 }),
            }
        }
        self.out.push_str(" }");
    }
}
