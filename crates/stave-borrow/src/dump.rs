//! The `mir` dump (stage-gates.md): each borrow-checkable body's CFG as the builder leaves it,
//! one section per body.
//!
//! Every id the text shows is the rank of a key the printed text itself fixes, never a creation
//! or recursion order of the builder, so a second builder with other data structures prints the
//! same bytes for the same CFG:
//! - blocks: the preorder of a depth-first walk from the entry that takes the successors in
//!   order; a block the entry does not reach is not printed;
//! - locals: the params by index, then the captures by binding node, then every other local in
//!   the order the printed lines first name it (blocks, then findings); a local no line names
//!   is not printed;
//! - loans: in the order the printed lines first name them;
//! - types: the rank of the key among the types the section names and the components of their
//!   keys, in the canonical type-key order (as `types`);
//! - nested bodies (closures, `spawn`) by `k`, the index of their `fn` / `spawn` token among
//!   the fn's closure bodies (stave-sema `closures`, the `@<inst>.<k>` of the IR text).

use std::fmt::Write;

use rustc_hash::{FxHashMap, FxHashSet};
use stave_diag::Span;
use stave_sema::DeclId;
use stave_sema::drops::{Head, Made, Owner, Point, Runs};
use stave_sema::itemcx::ItemCx;
use stave_sema::temps::Bound;
use stave_sema::types::{Interner, Overlay, TypeId};

use crate::cfg::{
    Body, GuardEffect, LoanId, LocalId, MoveOut, OpKind, Operand, PartFix, Place, Proj, Rvalue, StmtKind,
    TempEnd, Terminator,
};
use crate::ty::{DerefKind, Flows};

/// What a section's header says of its body.
#[derive(Clone, Copy, Debug)]
pub enum State {
    Ok,
    /// A type error: the body is not borrow checked.
    Error,
    Unsupported(&'static str),
    Internal(&'static str),
}

/// `body d<D>[.<k>] I N STATE`.
pub fn header(out: &mut String, decl: DeclId, k: Option<u32>, item: u32, node: u32, state: State) {
    let _ = write!(out, "body d{}", decl.0);
    if let Some(k) = k {
        let _ = write!(out, ".{k}");
    }
    let _ = write!(out, " {item} {node} ");
    let _ = match state {
        State::Ok => writeln!(out, "ok"),
        State::Error => writeln!(out, "error"),
        State::Unsupported(why) => writeln!(out, "unsupported \"{why}\""),
        State::Internal(why) => writeln!(out, "internal \"{why}\""),
    };
}

/// The sections of top body `body` of decl `decl` (fn node `fn_node`): its own, then, when it is
/// checked, those of the bodies nested in it in `k` order. A body that is not checked has only
/// its header: what the builder made of it is partial.
pub fn sections(
    out: &mut String,
    icx: &ItemCx<'_>,
    ov: &Overlay<'_>,
    decl: DeclId,
    fn_node: u32,
    body: &Body,
) {
    let item = icx.item.0;
    if let Some(why) = body.internal {
        return header(out, decl, None, item, fn_node, State::Internal(why));
    }
    if let Some(why) = body.unsupported {
        return header(out, decl, None, item, fn_node, State::Unsupported(why));
    }
    header(out, decl, None, item, fn_node, State::Ok);
    body_text(out, ov, body);
    if body.nested.is_empty() {
        return;
    }
    // k: the closure body's index by its first token; a body nested in a nested body is
    // numbered among the fn's closure bodies too
    let nodes = stave_sema::closures::closure_nodes(icx, fn_node);
    let at: Vec<Span> = nodes.iter().map(|&(n, _)| icx.tok_span(icx.a(n))).collect();
    let mut all: Vec<(u32, &Body)> = Vec::new();
    let mut todo: Vec<&Body> = body.nested.iter().rev().collect();
    while let Some(b) = todo.pop() {
        let k = b.at.and_then(|s| at.iter().position(|x| *x == s)).map_or(u32::MAX, |k| k as u32);
        all.push((k, b));
        todo.extend(b.nested.iter().rev());
    }
    all.sort_by_key(|x| x.0);
    for (k, b) in all {
        let node = nodes.get(k as usize).map_or(u32::MAX, |x| x.0);
        header(out, decl, Some(k), item, node, State::Ok);
        body_text(out, ov, b);
    }
}

/// Canonical numbers of one body's blocks, locals, and loans.
struct Num {
    block: Vec<u32>,
    blocks: Vec<u32>,
    local: Vec<u32>,
    locals: Vec<LocalId>,
    loan: Vec<u32>,
    loans: Vec<LoanId>,
}

impl Num {
    fn local(&mut self, l: LocalId) {
        if let Some(x) = self.local.get_mut(l as usize)
            && *x == u32::MAX
        {
            *x = self.locals.len() as u32;
            self.locals.push(l);
        }
    }

    fn loan(&mut self, body: &Body, mut l: LoanId) {
        while let Some(x) = self.loan.get_mut(l as usize) {
            if *x != u32::MAX {
                return;
            }
            *x = self.loans.len() as u32;
            self.loans.push(l);
            let loan = &body.loans[l as usize];
            self.local(loan.place.local);
            match loan.whole {
                Some(w) => l = w,
                None => return,
            }
        }
    }

    fn op(&mut self, o: &Operand) {
        if o.kind != OpKind::Const {
            self.local(o.place.local);
        }
    }

    fn stmt(&mut self, body: &Body, k: &StmtKind) {
        match k {
            StmtKind::Assign { dest, rv, .. } => {
                self.local(dest.local);
                match rv {
                    Rvalue::Use(o) => self.op(o),
                    Rvalue::Ref { place, loan, .. } => {
                        self.local(place.local);
                        self.loan(body, *loan);
                    }
                    Rvalue::Agg(os) | Rvalue::Pure(os) | Rvalue::Call { args: os, .. } => {
                        os.iter().for_each(|o| self.op(o))
                    }
                }
            }
            StmtKind::Activate { local, reserved, active } => {
                self.local(*local);
                self.loan(body, *reserved);
                self.loan(body, *active);
            }
            StmtKind::Read(o) | StmtKind::Return(o) => self.op(o),
            StmtKind::Drop(p) => self.local(p.local),
            StmtKind::StorageDead(l) => self.local(*l),
        }
    }
}

/// One builder finding, with its sort key (kind, file, offset).
enum Find<'b> {
    Move(&'b Place, Span, &'b MoveOut),
    Guard(Span, &'b GuardEffect),
    TempWrite(Span),
}

impl Find<'_> {
    fn key(&self) -> (u8, u32, u32) {
        match self {
            Find::Move(_, s, _) => (0, s.file.0, s.offset),
            Find::Guard(s, _) => (1, s.file.0, s.offset),
            Find::TempWrite(s) => (2, s.file.0, s.offset),
        }
    }

    fn places(&self) -> [Option<&Place>; 2] {
        match self {
            Find::Move(p, _, why) => match why {
                MoveOut::Drop(q) | MoveOut::Part(q, _) => [Some(*p), Some(q)],
                _ => [Some(*p), None],
            },
            Find::Guard(_, e) => match e {
                GuardEffect::Move(p) | GuardEffect::Assign(p) | GuardEffect::MutBorrow(p) => [Some(p), None],
                GuardEffect::Try => [None, None],
            },
            Find::TempWrite(_) => [None, None],
        }
    }
}

/// The lines of one body after its header.
fn body_text(out: &mut String, ov: &Overlay<'_>, b: &Body) {
    let mut num = Num {
        block: vec![u32::MAX; b.blocks.len()],
        blocks: Vec::new(),
        local: vec![u32::MAX; b.locals.len()],
        locals: Vec::new(),
        loan: vec![u32::MAX; b.loans.len()],
        loans: Vec::new(),
    };
    let mut stack: Vec<u32> = vec![0];
    while let Some(x) = stack.pop() {
        match num.block.get_mut(x as usize) {
            Some(n) if *n == u32::MAX => *n = num.blocks.len() as u32,
            _ => continue,
        }
        num.blocks.push(x);
        stack.extend(b.blocks[x as usize].term.succs().iter().rev());
    }
    let mut params: Vec<(u32, LocalId)> = Vec::new();
    let mut caps: Vec<(u32, LocalId)> = Vec::new();
    for (i, l) in b.locals.iter().enumerate() {
        match (l.param, l.capture) {
            (Some(_), Some(_)) => caps.push((l.node, i as LocalId)),
            (Some(p), None) => params.push((p, i as LocalId)),
            _ => {}
        }
    }
    params.sort_unstable();
    caps.sort_unstable();
    for &(_, l) in params.iter().chain(&caps) {
        num.local(l);
    }
    for k in 0..num.blocks.len() {
        let blk = &b.blocks[num.blocks[k] as usize];
        for s in &blk.stmts {
            num.stmt(b, &s.kind);
        }
    }
    let mut finds: Vec<Find<'_>> = Vec::new();
    finds.extend(b.bad_moves.iter().map(|(p, s, w)| Find::Move(p, *s, w)));
    finds.extend(b.guard_effects.iter().map(|(s, e)| Find::Guard(*s, e)));
    finds.extend(b.temp_writes.iter().map(|(s, _)| Find::TempWrite(*s)));
    finds.sort_by_key(|f| f.key());
    for f in &finds {
        for p in f.places().into_iter().flatten() {
            num.local(p.local);
        }
    }

    // the type table: what the local lines and the `deep` line name, closed over key components
    let mut deep: Vec<TypeId> = b.deep_drops.iter().copied().collect();
    let mut roots: Vec<TypeId> = num.locals.iter().map(|&l| b.locals[l as usize].ty).collect();
    roots.extend_from_slice(&deep);
    let mut table: Vec<TypeId> = Vec::new();
    let mut seen: FxHashSet<TypeId> = FxHashSet::default();
    let mut kids = Vec::new();
    while let Some(t) = roots.pop() {
        if t.is_none() || ov.is_error_ty(t) || !seen.insert(t) {
            continue;
        }
        table.push(t);
        kids.clear();
        ov.kind(t).children(&mut kids);
        roots.extend_from_slice(&kids);
    }
    table.sort_by(|&x, &y| stave_sema::types::cmp_type_keys(ov, x, y));
    let tnum: FxHashMap<TypeId, u32> = table.iter().enumerate().map(|(j, &t)| (t, j as u32)).collect();
    let next = table.len() as u32;
    let tn = |t: TypeId| tnum.get(&t).copied().unwrap_or(next);
    for (j, &t) in table.iter().enumerate() {
        let _ = writeln!(out, "t{j} {}", stave_sema::types::type_key_text_as(ov, t, &tn));
    }
    let ty = |out: &mut String, t: TypeId| match tnum.get(&t) {
        Some(j) => {
            let _ = write!(out, "t{j}");
        }
        None => out.push('-'),
    };

    for (j, &l) in num.locals.iter().enumerate() {
        let loc = &b.locals[l as usize];
        let _ = write!(out, "local l{j} ");
        match (loc.param, loc.capture) {
            (Some(_), Some(_)) => out.push_str("capture"),
            (Some(p), None) => {
                let _ = write!(out, "param{p}");
            }
            _ if loc.env => out.push_str("env"),
            _ if loc.temp && loc.static_item => out.push_str("static"),
            _ if loc.temp && b.joins.iter().any(|j| j.0 == l) => out.push_str("join"),
            _ if loc.temp => out.push_str("temp"),
            _ if loc.name == "result" => out.push_str("result"),
            _ if loc.node == u32::MAX => out.push_str("anon"),
            _ => out.push_str("bind"),
        }
        out.push(' ');
        opt(out, loc.node);
        out.push(' ');
        ty(out, loc.ty);
        if loc.temp {
            out.push(' ');
            opt(out, loc.stmt);
            out.push_str(match loc.end {
                None => " -",
                Some(TempEnd::Loop) => " loop",
                Some(TempEnd::Stmt(Bound::Stmt)) => " stmt",
                Some(TempEnd::Stmt(Bound::Cond)) => " cond",
                Some(TempEnd::Stmt(Bound::Guard)) => " guard",
                Some(TempEnd::Stmt(Bound::Branch)) => " branch",
                Some(TempEnd::Stmt(Bound::Arm)) => " arm",
                Some(TempEnd::Stmt(Bound::Clause)) => " clause",
                Some(TempEnd::Stmt(Bound::Old)) => " old",
            });
            out.push(' ');
            match loc.made {
                Some(m) => made(out, m),
                None => out.push('-'),
            }
        } else {
            out.push_str(" - - -");
        }
        flags(
            out,
            &[
                (loc.mutable, "mut"),
                (loc.capture_mut, "capture-mut"),
                (loc.guard, "guard"),
                (loc.matched, "matched"),
                (loc.static_root, "static-root"),
            ],
        );
        out.push('\n');
    }

    for (j, &l) in num.loans.iter().enumerate() {
        let loan = &b.loans[l as usize];
        let _ = write!(out, "loan L{j} ");
        opt(out, loan.node);
        out.push(' ');
        place(out, &num, &loan.place);
        out.push(' ');
        match loan.whole {
            Some(w) => {
                let _ = write!(out, "L{}", num.loan[w as usize]);
            }
            None => out.push('-'),
        }
        flags(
            out,
            &[
                (loan.mutable, "mut"),
                (loan.reserved, "reserved"),
                (loan.capture, "capture"),
                (loan.cast, "cast"),
            ],
        );
        out.push('\n');
    }

    for (j, &x) in num.blocks.iter().enumerate() {
        let blk = &b.blocks[x as usize];
        let _ = write!(out, "block b{j} ");
        match &blk.term {
            Terminator::Goto(s) => {
                let _ = write!(out, "goto b{}", num.block[*s as usize]);
            }
            Terminator::Branch(ss) => {
                out.push_str("branch");
                for s in ss {
                    let _ = write!(out, " b{}", num.block[*s as usize]);
                }
            }
            Terminator::Return => out.push_str("return"),
            Terminator::Unreachable => out.push_str("unreachable"),
        }
        out.push('\n');
        for s in &blk.stmts {
            out.push_str("  ");
            opt(out, s.node);
            stmt(out, &num, &s.kind);
            out.push('\n');
        }
    }

    let mut loops: Vec<(u32, u32, Vec<u32>)> = Vec::new();
    for &(head, exit, end) in &b.loops {
        let Some(&h) = num.block.get(head as usize).filter(|h| **h != u32::MAX) else { continue };
        let e = num.block.get(exit as usize).copied().unwrap_or(u32::MAX);
        let mut inside: Vec<u32> = (head..end)
            .filter(|&x| x != exit)
            .filter_map(|x| num.block.get(x as usize).copied())
            .filter(|&x| x != u32::MAX)
            .collect();
        inside.sort_unstable();
        loops.push((h, e, inside));
    }
    loops.sort();
    for (h, e, inside) in loops {
        let _ = write!(out, "loop b{h} ");
        if e == u32::MAX {
            out.push('-');
        } else {
            let _ = write!(out, "b{e}");
        }
        out.push_str(" [");
        for (i, x) in inside.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let _ = write!(out, "b{x}");
        }
        out.push_str("]\n");
    }

    let mut sched = b.schedule.clone();
    sched.seal();
    for r in sched.rows() {
        out.push_str("drop ");
        match r.point {
            Point::End => out.push_str("end"),
            Point::Return(t) => tagged(out, "return", t),
            Point::Try(t) => tagged(out, "try", t),
            Point::Jump(t) => tagged(out, "jump", t),
            Point::Bind(n) => tagged(out, "bind", n),
            Point::Assign(n) => tagged(out, "assign", n),
        }
        out.push(' ');
        match r.owner {
            Owner::Local(n) => tagged(out, "local", n),
            Owner::Temp(s, m) => {
                tagged(out, "temp", s);
                out.push(' ');
                made(out, m);
            }
            Owner::Part(i) => tagged(out, "part", i),
            Owner::Old => out.push_str("old"),
        }
        let _ = match r.head {
            Head::Any => write!(out, " any"),
            Head::Adt(d) => write!(out, " d{}", d.0),
            Head::Tuple(n) => write!(out, " tuple{n}"),
            Head::Array => write!(out, " array"),
        };
        out.push_str(match sched.runs(r.point, r.owner) {
            Some(Runs::Always) => " always\n",
            Some(Runs::Never) => " never\n",
            Some(Runs::Flag) => " flag\n",
            None => " -\n",
        });
    }

    let mut lines: Vec<((u8, u32, u32), String)> = Vec::new();
    for f in &finds {
        let mut s = String::new();
        let (kind, file, off) = f.key();
        let _ = write!(s, "find {} {file} {off}", ["move", "guard", "temp-write"][kind as usize]);
        match f {
            Find::Move(p, _, why) => {
                s.push(' ');
                place(&mut s, &num, p);
                match why {
                    MoveOut::Behind => s.push_str(" behind"),
                    MoveOut::Capture(c) => {
                        let _ = write!(s, " capture {} {}", c.file.0, c.offset);
                    }
                    MoveOut::Drop(q) => {
                        s.push_str(" drop ");
                        place(&mut s, &num, q);
                    }
                    MoveOut::Static => s.push_str(" static"),
                    MoveOut::Part(q, fix) => {
                        s.push_str(" part ");
                        place(&mut s, &num, q);
                        s.push_str(match fix {
                            PartFix::Reborrow => " reborrow",
                            PartFix::Pattern(_) => " pattern",
                            PartFix::Replace => " replace",
                        });
                    }
                }
            }
            Find::Guard(_, e) => match e {
                GuardEffect::Move(p) => {
                    s.push_str(" move ");
                    place(&mut s, &num, p);
                }
                GuardEffect::Assign(p) => {
                    s.push_str(" assign ");
                    place(&mut s, &num, p);
                }
                GuardEffect::MutBorrow(p) => {
                    s.push_str(" mut-borrow ");
                    place(&mut s, &num, p);
                }
                GuardEffect::Try => s.push_str(" try"),
            },
            Find::TempWrite(_) => {}
        }
        lines.push((f.key(), s));
    }
    lines.sort();
    lines.dedup();
    for (_, s) in lines {
        out.push_str(&s);
        out.push('\n');
    }

    if !deep.is_empty() {
        deep.sort_by_key(|&t| tnum.get(&t).copied().unwrap_or(u32::MAX));
        out.push_str("deep");
        for t in deep {
            out.push(' ');
            ty(out, t);
        }
        out.push('\n');
    }
}

/// A number, or `-` for none.
fn opt(out: &mut String, n: u32) {
    if n == u32::MAX {
        out.push('-');
    } else {
        let _ = write!(out, "{n}");
    }
}

fn tagged(out: &mut String, tag: &str, n: u32) {
    out.push_str(tag);
    out.push(' ');
    opt(out, n);
}

fn made(out: &mut String, m: Made) {
    opt(out, m.from);
    out.push(':');
    opt(out, m.to);
}

/// ` FLAG,FLAG` of the flags that hold, or ` -`.
fn flags(out: &mut String, fs: &[(bool, &str)]) {
    let mut any = false;
    for &(on, name) in fs {
        if on {
            out.push(if any { ',' } else { ' ' });
            out.push_str(name);
            any = true;
        }
    }
    if !any {
        out.push_str(" -");
    }
}

fn place(out: &mut String, num: &Num, p: &Place) {
    let _ = write!(out, "l{}", num.local.get(p.local as usize).copied().unwrap_or(u32::MAX));
    for pr in &p.proj {
        let _ = match pr {
            Proj::Deref(DerefKind::Shared) => write!(out, ".*"),
            Proj::Deref(DerefKind::Mut) => write!(out, ".*m"),
            Proj::Deref(DerefKind::Raw) => write!(out, ".*r"),
            Proj::Field(f) => write!(out, ".f{f}"),
            Proj::EnumField(v, f) => write!(out, ".v{v}:{f}"),
            Proj::Index => write!(out, ".[]"),
            Proj::Column(c) => write!(out, ".c{c}"),
        };
    }
}

fn operand(out: &mut String, num: &Num, o: &Operand) {
    match o.kind {
        OpKind::Const => out.push_str("const "),
        OpKind::Copy | OpKind::Move => {
            out.push_str(if o.kind == OpKind::Copy { "copy " } else { "move " });
            place(out, num, &o.place);
            out.push(' ');
        }
    }
    opt(out, o.node);
}

fn operands(out: &mut String, num: &Num, os: &[Operand]) {
    out.push('[');
    for (i, o) in os.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        operand(out, num, o);
    }
    out.push(']');
}

fn stmt(out: &mut String, num: &Num, k: &StmtKind) {
    match k {
        StmtKind::Assign { dest, rv, init } => {
            out.push_str(" assign ");
            place(out, num, dest);
            out.push_str(if *init { " init " } else { " set " });
            match rv {
                Rvalue::Use(o) => {
                    out.push_str("use ");
                    operand(out, num, o);
                }
                Rvalue::Ref { place: p, mutable, loan } => {
                    out.push_str(if *mutable { "ref-mut " } else { "ref " });
                    place(out, num, p);
                    let _ = write!(out, " L{}", num.loan[*loan as usize]);
                }
                Rvalue::Agg(os) => {
                    out.push_str("agg");
                    operands(out, num, os);
                }
                Rvalue::Pure(os) => {
                    out.push_str("pure");
                    operands(out, num, os);
                }
                Rvalue::Call { args, flows, stores, tok } => {
                    out.push_str("call");
                    operands(out, num, args);
                    out.push_str(" tok ");
                    opt(out, *tok);
                    out.push_str(" flows ");
                    match flows {
                        Flows::None => out.push('-'),
                        Flows::AllArgs => out.push_str("all"),
                        Flows::Named(v) => {
                            out.push('[');
                            for (i, (a, copy)) in v.iter().enumerate() {
                                if i > 0 {
                                    out.push_str(", ");
                                }
                                let _ = write!(out, "{}{a}", if *copy { "copy " } else { "" });
                            }
                            out.push(']');
                        }
                    }
                    out.push_str(" stores[");
                    for (i, s) in stores.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let _ = write!(
                            out,
                            "{}<{}{}{}",
                            s.target,
                            if s.copy { "copy " } else { "" },
                            s.source,
                            if s.into_view { " view" } else { "" }
                        );
                    }
                    out.push(']');
                }
            }
        }
        StmtKind::Activate { local, reserved, active } => {
            let _ = write!(
                out,
                " activate l{} L{} L{}",
                num.local[*local as usize], num.loan[*reserved as usize], num.loan[*active as usize]
            );
        }
        StmtKind::Read(o) => {
            out.push_str(" read ");
            operand(out, num, o);
        }
        StmtKind::Drop(p) => {
            out.push_str(" drop ");
            place(out, num, p);
        }
        StmtKind::StorageDead(l) => {
            let _ = write!(out, " dead l{}", num.local[*l as usize]);
        }
        StmtKind::Return(o) => {
            out.push_str(" return ");
            operand(out, num, o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_states() {
        let mut s = String::new();
        header(&mut s, DeclId(7), None, 3, 40, State::Ok);
        header(&mut s, DeclId(7), Some(2), 3, 31, State::Ok);
        header(&mut s, DeclId(8), None, 4, 9, State::Error);
        header(&mut s, DeclId(9), None, 5, 12, State::Unsupported("nested body"));
        header(&mut s, DeclId(10), None, 6, 15, State::Internal("a value without a type"));
        let want = [
            "body d7 3 40 ok",
            "body d7.2 3 31 ok",
            "body d8 4 9 error",
            "body d9 5 12 unsupported \"nested body\"",
            "body d10 6 15 internal \"a value without a type\"",
        ];
        assert_eq!(s, want.map(|l| format!("{l}\n")).concat());
    }
}
