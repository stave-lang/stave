//! The printer of contracts/checked-ir.md: sections in contract order, every line spelled as
//! `ir-text.ebnf` says, from the canonical module only.
//!
//! The output is a function of the module, header and stack section alone: every
//! table is walked by index, none by hash order. A module that is not canonical
//! ([`crate::canon`]) is canonicalized (a copy) first, so the text is the canonical module's.
//! A module whose text would not load in the reader's first two steps is refused
//! ([`super::spell`]): every printed id names a line, every key has a spelling (no generic
//! parameter, every `adt` key's decl recorded).

use std::fmt::Write as _;

use super::{Header, StackInput, VERSION};
use crate::canon;
use crate::ir::{Body, Const, DataTarget, Func, FuncId, Module, Op, Operand, Site, Target, Term};
use crate::types::{GenericArg, TypeId, feature_generators};
use crate::verify::VerifyError;
use crate::witness::{NO_BLOCK, Row, Summary, Term as RowTerm};

/// The text of `m` (checked-ir.md): the gate form, or with `stack` the emit form, which is the
/// gate form plus the stack section. A module whose text would not load is not printed: its
/// faults ([`super::spell`]).
pub fn print_module(
    m: &Module,
    header: &Header,
    stack: Option<&StackInput>,
) -> Result<String, Vec<VerifyError>> {
    let errs = super::spell::faults(m, stack);
    if !errs.is_empty() {
        return Err(errs);
    }
    Ok(print_verified(m, header, stack))
}

fn print_verified(m: &Module, header: &Header, stack: Option<&StackInput>) -> String {
    if canon::is_canonical(m) {
        return Printer::new(m, header).print(stack);
    }
    let mut c = m.clone();
    let remap = canon::canonicalize(&mut c);
    let stack = stack.map(|s| {
        let mut s = s.clone();
        canon::permute_by_func(&mut s.nodes_of_func, &remap);
        s
    });
    Printer::new(&c, header).print(stack.as_ref())
}

/// The checked-level text of a bare module, for callers that have only a module: header
/// `x86_64-linux-v2 debug checked`, no stack section.
pub fn print_checked(m: &Module) -> Result<String, Vec<VerifyError>> {
    use super::{Level, Mode};
    use stave_target::{Triple, X86Level};
    let header = Header { target: Triple::X86_64(X86Level::V2), mode: Mode::Debug, level: Level::Checked };
    print_module(m, &header, None)
}

/// How [`body_text`] spells what a body names outside its own values and blocks. The module
/// printer spells a function `@<instance>` and a type `t<N>`.
pub(crate) trait BodyNames {
    fn func(&self, out: &mut String, f: FuncId);
    fn ty(&self, out: &mut String, t: TypeId);
}

/// The lines of body `b` of a function of `m` as the module printer writes them (block headers,
/// instructions, terminators), with functions and types spelled by `names`, without the `!row`
/// site suffixes, and with each site row passed as an argument ([`crate::known::site_param`])
/// spelled `site`.
pub(crate) fn body_text(m: &Module, b: &Body, names: &dyn BodyNames) -> String {
    let header = Header {
        target: stave_target::Triple::X86_64(stave_target::X86Level::V2),
        mode: super::Mode::Debug,
        level: super::Level::Checked,
    };
    let mut p = Printer { m, header: &header, out: String::new(), names: Some(names) };
    p.body(b);
    p.out
}

struct Printer<'a> {
    m: &'a Module,
    header: &'a Header,
    out: String,
    /// Set: the body form of [`body_text`].
    names: Option<&'a dyn BodyNames>,
}

impl<'a> Printer<'a> {
    fn new(m: &'a Module, header: &'a Header) -> Printer<'a> {
        Printer { m, header, out: String::new(), names: None }
    }

    fn tref(&mut self, t: TypeId) {
        match self.names {
            Some(n) => n.ty(&mut self.out, t),
            None => {
                let _ = write!(self.out, "t{}", t.0);
            }
        }
    }

    fn print(mut self, stack: Option<&StackInput>) -> String {
        self.header_line();
        self.decls();
        self.types();
        self.shapes();
        self.strings();
        self.sites();
        self.statics();
        if let Some(f) = self.m.main.and_then(|f| self.m.func(f)) {
            let _ = writeln!(self.out, "main @{}", f.instance);
        }
        if let Some(d) = self.m.cell {
            let _ = writeln!(self.out, "cell d{}", d.0);
        }
        if let Some(s) = stack {
            self.stack(s);
        }
        for (i, f) in self.m.funcs.iter().enumerate() {
            self.func(f);
            if self.header.level != super::Level::Lowered {
                self.claims(f, self.m.stated.get(i).and_then(|s| s.as_deref()));
            }
        }
        self.out
    }

    fn header_line(&mut self) {
        let h = self.header;
        let _ = write!(self.out, "stave.ir {VERSION} ");
        quote(&mut self.out, h.target.name().as_bytes());
        let _ = writeln!(self.out, " {} {}", h.mode.name(), h.level.name());
    }

    fn decls(&mut self) {
        for d in &self.m.decls {
            let kind = match d.kind {
                crate::types::AdtKind::Struct => "struct",
                crate::types::AdtKind::Enum => "enum",
            };
            let _ = write!(self.out, "decl d{} {kind} ", d.id.0);
            path(&mut self.out, &d.path);
            self.out.push('\n');
        }
    }

    fn types(&mut self) {
        for i in 0..self.m.types.len() {
            let _ = writeln!(self.out, "type t{i} {}", self.m.types.key_text(TypeId(i as u32)));
        }
    }

    fn shapes(&mut self) {
        for i in 0..self.m.types.len() {
            let t = TypeId(i as u32);
            let Some(shape) = self.m.types.shape(t) else { continue };
            let _ = write!(self.out, "shape t{i}");
            match shape.kind {
                crate::types::AdtKind::Struct => {
                    self.out.push_str(" struct");
                    for f in shape.variants.first().into_iter().flatten() {
                        let _ = write!(self.out, " t{}", f.0);
                    }
                }
                crate::types::AdtKind::Enum => {
                    self.out.push_str(" enum");
                    for v in &shape.variants {
                        self.out.push_str(" |");
                        for f in v {
                            let _ = write!(self.out, " t{}", f.0);
                        }
                    }
                }
            }
            self.out.push('\n');
        }
    }

    fn strings(&mut self) {
        for (i, s) in self.m.strings.iter().enumerate() {
            let _ = write!(self.out, "str s{i} ");
            quote(&mut self.out, s);
            self.out.push('\n');
        }
    }

    fn sites(&mut self) {
        for (i, s) in self.m.sites.iter().enumerate() {
            let _ = write!(self.out, "site {i} ");
            quote(&mut self.out, s.file.as_bytes());
            let _ = write!(self.out, " {} {} ", s.line, s.col);
            path(&mut self.out, &s.func);
            self.out.push('\n');
        }
    }

    /// A static's `data` lines: its bytes cut into 64-byte blocks at offsets 0, 64, ..., one
    /// line per block with a nonzero byte, from its first to its last nonzero byte.
    fn statics(&mut self) {
        for (i, st) in self.m.statics.iter().enumerate() {
            let _ = write!(self.out, "static {i} t{} ", st.ty.0);
            path(&mut self.out, &st.path);
            self.out.push('\n');
            for (b, block) in st.init.chunks(64).enumerate() {
                let (Some(first), Some(last)) =
                    (block.iter().position(|&x| x != 0), block.iter().rposition(|&x| x != 0))
                else {
                    continue;
                };
                let _ = write!(self.out, "data {i} {} x", b * 64 + first);
                for x in &block[first..=last] {
                    let _ = write!(self.out, "{x:02x}");
                }
                self.out.push('\n');
            }
            for r in &st.refs {
                let _ = write!(self.out, "ref {i} {} ", r.offset);
                match r.target {
                    DataTarget::Str(s) => {
                        let _ = write!(self.out, "s{}", s.0);
                    }
                    DataTarget::Static(j) => {
                        let _ = write!(self.out, "static {j}");
                    }
                }
                let _ = writeln!(self.out, " {}", r.addend);
            }
        }
    }

    /// The stack section; the lines of `nodes` follow the fns' instance order.
    fn stack(&mut self, s: &StackInput) {
        if let Some(n) = s.run_stack {
            let _ = writeln!(self.out, "runstack {n}");
        }
        if s.truncated {
            self.out.push_str("truncated\n");
        }
        for (n, rec) in s.nodes.iter().enumerate() {
            let _ = writeln!(self.out, "node {n}{}", if *rec { " rec" } else { "" });
        }
        for e in &s.edges {
            let _ = writeln!(self.out, "edge {} {} {}", e.from, e.to, e.kind.name());
        }
        for (f, nodes) in self.m.funcs.iter().zip(&s.nodes_of_func) {
            if nodes.is_empty() {
                continue;
            }
            let _ = write!(self.out, "nodes @{}", f.instance);
            for n in nodes {
                let _ = write!(self.out, " {n}");
            }
            self.out.push('\n');
        }
    }

    fn func(&mut self, f: &Func) {
        let _ = write!(self.out, "fn @{} ", f.instance);
        path(&mut self.out, &f.path);
        if !f.type_args.is_empty() {
            self.out.push('<');
            for (i, a) in f.type_args.iter().enumerate() {
                if i > 0 {
                    self.out.push(',');
                }
                match a {
                    GenericArg::Type(t) => {
                        let _ = write!(self.out, "t{}", t.0);
                    }
                    GenericArg::Const(c) => {
                        let _ = write!(self.out, "{c}");
                    }
                }
            }
            self.out.push('>');
        }
        self.out.push('(');
        for (i, t) in f.params.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let _ = write!(self.out, "t{}", t.0);
        }
        self.out.push(')');
        if let Some(r) = f.ret {
            let _ = write!(self.out, " -> t{}", r.0);
        }
        let needs = feature_generators(f.features);
        if !needs.is_empty() {
            self.out.push_str(" needs(");
            for (i, x) in needs.iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                self.out.push_str(x.name());
            }
            self.out.push(')');
        }
        if let Some(sym) = &f.symbol {
            self.out.push_str(" symbol ");
            quote(&mut self.out, sym.as_bytes());
        }
        for (on, word) in [(f.variadic, "variadic"), (f.known.is_some(), "known"), (f.alt_stack, "altstack")]
        {
            if on {
                self.out.push(' ');
                self.out.push_str(word);
            }
        }
        if f.unbuilt {
            self.out.push_str(" unbuilt");
        }
        if f.name_site != Site::NONE {
            let _ = write!(self.out, " site {}", f.name_site.0);
        }
        let Some(b) = &f.body else {
            self.out.push('\n');
            return;
        };
        self.out.push_str(" {\n");
        self.body(b);
        self.out.push_str("}\n");
    }

    fn body(&mut self, b: &Body) {
        for (bi, bd) in b.blocks.iter().enumerate() {
            let _ = write!(self.out, "b{bi}(");
            for (i, v) in bd.params.of(&b.block_params).iter().enumerate() {
                if i > 0 {
                    self.out.push_str(", ");
                }
                match b.value_ty(*v) {
                    Some(t) => {
                        let _ = write!(self.out, "%{}: ", v.0);
                        self.tref(t);
                    }
                    None => {
                        let _ = write!(self.out, "%{}: ?", v.0);
                    }
                }
            }
            self.out.push_str("):\n");
            for i in bd.insts.indices() {
                self.inst(b, i as usize);
            }
            self.out.push_str("  ");
            self.term(b, bd.term);
            self.out.push('\n');
        }
    }

    fn inst(&mut self, b: &Body, i: usize) {
        self.out.push_str("  ");
        if let Some(Some(v)) = b.insts.result.get(i) {
            let _ = write!(self.out, "%{} = ", v.0);
        }
        match b.insts.op.get(i).copied() {
            Some(Op::Intrinsic(id)) => {
                let _ = write!(self.out, "intrinsic.{}", self.m.intrinsic_name(id).unwrap_or("?"));
            }
            Some(op) => self.out.push_str(op.name().unwrap_or("?")),
            None => self.out.push('?'),
        }
        match b.insts.ty.get(i).copied().flatten() {
            Some(t) => {
                self.out.push(' ');
                self.tref(t);
            }
            None => self.out.push_str(" ()"),
        }
        let args = b.args(i as u32);
        // The body form spells a site row passed as an argument `site`, as it drops `!row`.
        let site_param = match (self.names, b.insts.op.get(i), args.first()) {
            (Some(_), Some(Op::Call), Some(Operand::Func(f))) => {
                self.m.func(*f).and_then(|g| g.known).and_then(crate::known::site_param)
            }
            _ => None,
        };
        for (j, o) in args.iter().enumerate() {
            self.out.push(' ');
            if matches!(o, Operand::Const(_)) && site_param.is_some_and(|p| j == p + 1) {
                self.out.push_str("site");
                continue;
            }
            self.operand(b, *o);
        }
        if let Some(s) = b.insts.site.get(i).filter(|s| **s != Site::NONE && self.names.is_none()) {
            let _ = write!(self.out, " !{}", s.0);
        }
        self.out.push('\n');
    }

    fn term(&mut self, b: &Body, t: Term) {
        match t {
            Term::Br(t) => {
                self.out.push_str("br ");
                self.target(b, &t);
            }
            Term::Cbr { cond, then, els } => {
                self.out.push_str("cbr ");
                self.operand(b, cond);
                self.out.push(' ');
                self.target(b, &then);
                self.out.push(' ');
                self.target(b, &els);
            }
            Term::Switch { scrut, cases, default } => {
                self.out.push_str("switch ");
                self.operand(b, scrut);
                for c in b.switch_cases(cases) {
                    let _ = write!(self.out, " {}:", c.value);
                    self.target(b, &c.target);
                }
                self.out.push_str(" _:");
                self.target(b, &default);
            }
            Term::Ret(None) => self.out.push_str("ret"),
            Term::Ret(Some(o)) => {
                self.out.push_str("ret ");
                self.operand(b, o);
            }
            Term::Unreachable => self.out.push_str("unreachable"),
        }
    }

    fn target(&mut self, b: &Body, t: &Target) {
        if t.block.index() < b.blocks.len() {
            let _ = write!(self.out, "b{}(", t.block.0);
        } else {
            self.out.push_str("b?(");
        }
        for (i, o) in b.target_args(t).iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            self.operand(b, *o);
        }
        self.out.push(')');
    }

    fn operand(&mut self, b: &Body, o: Operand) {
        match o {
            Operand::Value(v) if v.index() < b.values.len() => {
                let _ = write!(self.out, "%{}", v.0);
            }
            Operand::Value(_) => self.out.push_str("%?"),
            Operand::Const(c) => match b.konst(c) {
                Some(Const::Int(x)) => {
                    let _ = write!(self.out, "{x}");
                }
                Some(Const::F16(x)) => {
                    let _ = write!(self.out, "0x{x:04x}");
                }
                Some(Const::BF16(x)) => {
                    let _ = write!(self.out, "bf16 0x{x:04x}");
                }
                Some(Const::F32(x)) => {
                    let _ = write!(self.out, "0x{x:08x}");
                }
                Some(Const::F64(x)) => {
                    let _ = write!(self.out, "0x{x:016x}");
                }
                None => self.out.push('?'),
            },
            Operand::Str(s) => {
                let _ = write!(self.out, "s{}", s.0);
            }
            Operand::Func(f) => match (self.m.func(f), self.names) {
                (Some(_), Some(n)) => n.func(&mut self.out, f),
                (Some(g), None) => {
                    let _ = write!(self.out, "@{}", g.instance);
                }
                (None, _) => self.out.push_str("@?"),
            },
            Operand::Type(t) => self.tref(t),
        }
    }

    /// The `claim` lines of a fn: the `pre` rows in the summary's order, then the `post` rows.
    /// `checked` marks a `pre` row that reads a `requires` check.
    fn claims(&mut self, f: &Func, s: Option<&Summary>) {
        let Some(s) = s else { return };
        for p in &s.pre {
            let _ = write!(self.out, "claim @{} pre", f.instance);
            if p.entry != NO_BLOCK {
                self.out.push_str(" checked");
            }
            self.row(p.row);
        }
        for r in &s.post {
            let _ = write!(self.out, "claim @{} post", f.instance);
            self.row(*r);
        }
    }

    fn row(&mut self, r: Row) {
        self.out.push(' ');
        row_term(&mut self.out, r.a);
        self.out.push_str(" <= ");
        row_term(&mut self.out, r.b);
        let _ = writeln!(self.out, " {}", r.w);
    }
}

fn row_term(out: &mut String, t: RowTerm) {
    match t {
        RowTerm::Zero => out.push_str("zero"),
        RowTerm::Param(i) => {
            let _ = write!(out, "p {i}");
        }
        RowTerm::LenParam(i) => {
            let _ = write!(out, "lenp {i}");
        }
        RowTerm::Result => out.push_str("result"),
        RowTerm::LenResult => out.push_str("lenresult"),
    }
}

/// A path: bare when it has the `Word` shape of ir-text.ebnf, else a string.
fn path(out: &mut String, p: &str) {
    let b = p.as_bytes();
    let word = b.first().is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
        && b.iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.'));
    if word {
        out.push_str(p);
    } else {
        quote(out, b);
    }
}

/// A string between double quotes with the `strings` dump escape rule (stage-gates.md): bytes
/// 0x20 to 0x7E as themselves except `\` and `"`, `\n` and `\t` for line feed and tab, `\xHH`
/// (lowercase) for every other byte.
fn quote(out: &mut String, bytes: &[u8]) {
    out.push('"');
    for &c in bytes {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(c as char),
            _ => {
                let _ = write!(out, "\\x{c:02x}");
            }
        }
    }
    out.push('"');
}
