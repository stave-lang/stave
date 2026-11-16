//! Emits `crates/stave-syntax/src/generated/{mod,ll1_table,terminals,node_kinds}.rs`.
//! The files are excluded from rustfmt (`#[rustfmt::skip] pub mod generated;` in stave-syntax).

use crate::naming::Names;
use crate::{Output, RUST_DIR, Sym, Tables};
use std::fmt::Write;

/// Packs `items` into lines of at most `width` columns, each line indented by `indent` and every
/// item followed by `,`.
pub fn pack<I: IntoIterator<Item = String>>(items: I, indent: &str, width: usize) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for it in items {
        if !line.is_empty() && indent.len() + line.len() + it.len() + 2 > width {
            out += indent;
            out += line.trim_end();
            out.push('\n');
            line.clear();
        }
        line += &it;
        line += ", ";
    }
    if !line.is_empty() {
        out += indent;
        out += line.trim_end();
        out.push('\n');
    }
    out
}

/// Right-hand sides reversed (push order), `NT_BIT | nt` for a nonterminal; and offsets.
pub fn packed_rhs(t: &Tables) -> (Vec<u16>, Vec<u32>) {
    let mut rhs = Vec::new();
    let mut off = vec![0u32];
    for p in &t.prods {
        for s in p.rhs.iter().rev() {
            rhs.push(match *s {
                Sym::T(x) => x,
                Sym::N(n) => 0x8000 | n,
            });
        }
        off.push(rhs.len() as u32);
    }
    (rhs, off)
}

fn nums<T: ToString>(xs: &[T]) -> impl Iterator<Item = String> + '_ {
    xs.iter().map(|x| x.to_string())
}

const W: usize = 100;

fn mod_rs(header: &str) -> String {
    format!(
        "// {header}\n\
         //! LL(1) parse tables generated from the grammar by tools/ll1gen.\n\
         \n\
         pub mod ll1_table;\n\
         pub mod node_kinds;\n\
         pub mod terminals;\n"
    )
}

fn terminals_rs(t: &Tables, names: &Names, header: &str) -> String {
    let n = t.nterm();
    let mut o = format!("// {header}\n");
    o += "//! Terminals in sorted spelling order (byte order), which is the E0004 expected-set order.\n";
    o += "//! `Term as u16` is the terminal id of `ll1_table`.\n\n";
    o += "/// Number of terminals.\n";
    let _ = writeln!(o, "pub const NTERM: usize = {n};\n");
    o += "/// A terminal of the grammar.\n#[repr(u8)]\n";
    o += "#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]\npub enum Term {\n";
    for (i, (s, v)) in t.terms.iter().zip(&names.term).enumerate() {
        let _ = writeln!(o, "    /// `{s}`\n    {v} = {i},");
    }
    o += "}\n\nimpl Term {\n    /// Every terminal, by id.\n    pub const ALL: [Term; NTERM] = [\n";
    o += &pack(names.term.iter().map(|v| format!("Term::{v}")), "        ", W);
    o += "    ];\n\n";
    o += "    /// The terminal id.\n    pub const fn id(self) -> u16 {\n        self as u16\n    }\n\n";
    o += "    /// The terminal with id `id`.\n    pub const fn from_id(id: u16) -> Option<Term> {\n";
    o += "        if (id as usize) < NTERM { Some(Self::ALL[id as usize]) } else { None }\n    }\n\n";
    o += "    /// Spelling for diagnostics: the token text, or the class name (`Ident`, `EOF`).\n";
    o += "    pub const fn text(self) -> &'static str {\n        TERM_TEXT[self as usize]\n    }\n\n";
    o += "    /// A lexical class (`Ident`, `IntLit`, ..., `EOF`) rather than a fixed spelling.\n";
    o += "    pub const fn is_class(self) -> bool {\n        TERM_IS_CLASS[self as usize]\n    }\n}\n\n";
    o += "/// Spelling per terminal id.\npub const TERM_TEXT: [&str; NTERM] = [\n";
    o += &pack(t.terms.iter().map(|s| format!("{s:?}")), "    ", W);
    o += "];\n\n/// Lexical class flag per terminal id.\npub const TERM_IS_CLASS: [bool; NTERM] = [\n";
    o += &pack(nums(&t.term_is_class), "    ", W);
    o += "];\n\n";
    let (first, rest) = split_arrays(t);
    o += "// Split actions (grammar header): D3 `>>` `>=` `>>=` start with the `>` that closes type\n";
    o += "// arguments; the token `&&` in unary position is two `&`. A nonterminal with no action on a\n";
    o += "// split token uses the action of the token's first terminal; an expected first terminal\n";
    o += "// consumes that part of the token and leaves the rest as the lookahead.\n\n";
    let pairs = split_pairs(t, 1);
    let _ = writeln!(o, "/// Split: the first terminal of a split token ({pairs}); -1 otherwise.");
    o += "pub const SPLIT_FIRST: [i16; NTERM] = [\n";
    o += &pack(nums(&first), "    ", W);
    o += "];\n\n";
    let pairs = split_pairs(t, 2);
    let _ = writeln!(o, "/// Split: the terminal left after the first one ({pairs}); -1 otherwise.");
    o += "pub const SPLIT_REST: [i16; NTERM] = [\n";
    o += &pack(nums(&rest), "    ", W);
    o += "];\n";
    o
}

/// Per terminal: the first terminal and the rest of a split token, -1 for other terminals.
pub fn split_arrays(t: &Tables) -> (Vec<i16>, Vec<i16>) {
    let n = t.nterm();
    let (mut first, mut rest) = (vec![-1i16; n], vec![-1i16; n]);
    for &(tok, f, r) in &t.split {
        first[tok as usize] = f as i16;
        rest[tok as usize] = r as i16;
    }
    (first, rest)
}

/// `>> → >, ...` for the doc comments: `part` 1 = first terminal, 2 = rest.
pub fn split_pairs(t: &Tables, part: usize) -> String {
    t.split
        .iter()
        .map(|&(a, f, r)| {
            let to = if part == 1 { f } else { r };
            format!("`{}` → `{}`", t.terms[a as usize], t.terms[to as usize])
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn node_kinds_rs(t: &Tables, names: &Names, header: &str) -> String {
    let mut o = format!("// {header}\n");
    o += "//! AST node kinds: one per named grammar rule. Helper nonterminals from\n";
    o += "//! `? * +` and groups make no node. A kind's id equals its nonterminal id in `ll1_table`.\n\n";
    o += "/// Number of node kinds (named rules).\n";
    let _ = writeln!(o, "pub const NKIND: usize = {};\n", t.nkind);
    o += "/// A node kind (a named grammar rule).\n#[repr(u8)]\n";
    o += "#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]\npub enum NodeKind {\n";
    for (i, v) in names.kind.iter().enumerate() {
        let _ = writeln!(o, "    {v} = {i},");
    }
    o += "}\n\nimpl NodeKind {\n    /// Every node kind, by id.\n    pub const ALL: [NodeKind; NKIND] = [\n";
    o += &pack(names.kind.iter().map(|v| format!("NodeKind::{v}")), "        ", W);
    o += "    ];\n\n";
    o += "    /// The node kind id.\n    pub const fn id(self) -> u8 {\n        self as u8\n    }\n\n";
    o += "    /// The rule's nonterminal id in `ll1_table` (equal to the kind id).\n";
    o += "    pub const fn nt(self) -> u16 {\n        self as u16\n    }\n\n";
    o += "    /// The node kind with id `id`.\n    pub const fn from_id(id: u8) -> Option<NodeKind> {\n";
    o += "        if (id as usize) < NKIND { Some(Self::ALL[id as usize]) } else { None }\n    }\n\n";
    o += "    /// The rule name.\n    pub const fn name(self) -> &'static str {\n";
    o += "        KIND_NAME[self as usize]\n    }\n}\n\n";
    o += "/// Rule name per node kind id.\npub const KIND_NAME: [&str; NKIND] = [\n";
    o += &pack(names.kind.iter().map(|s| format!("{s:?}")), "    ", W);
    o += "];\n";
    o
}

fn ll1_table_rs(t: &Tables, names: &Names, header: &str) -> String {
    let (rhs, off) = packed_rhs(t);
    let nnt = t.nts.len();
    let mut o = format!("// {header}\n");
    o += "//! LL(1) parse table. Nonterminal ids `0..NKIND` are the named rules (start rule\n";
    o += "//! first, id = `NodeKind` id); `NKIND..NNT` are helpers from `? * +` and groups, each with\n";
    o += "//! its origin rule. Every conflict is one of the listed greedy overrides, resolved to the\n";
    o += "//! first listed alternative.\n\n";
    o += "use super::terminals::NTERM;\n\n";
    o += "/// Number of nonterminals.\n";
    let _ = writeln!(o, "pub const NNT: usize = {nnt};");
    o += "/// Number of productions.\n";
    let _ = writeln!(o, "pub const NPROD: usize = {};", t.prods.len());
    o += "/// Total length of all right-hand sides.\n";
    let _ = writeln!(o, "pub const NRHS: usize = {};", rhs.len());
    o += "/// The start nonterminal (`File`).\npub const START: u16 = 0;\n";
    o += "/// Marks a nonterminal in `PROD_RHS`; the low bits are its id.\npub const NT_BIT: u16 = 0x8000;\n\n";
    o += "/// Name per nonterminal (`Rule#n` for helpers).\npub static NT_NAME: [&str; NNT] = [\n";
    o += &pack(t.nts.iter().map(|n| format!("{:?}", n.name)), "    ", W);
    o += "];\n\n/// Origin rule per nonterminal: the nonterminal id (= node kind) of the rule it comes from.\n";
    o += "pub static NT_ORIGIN: [u16; NNT] = [\n";
    o += &pack(t.nts.iter().map(|n| n.origin.to_string()), "    ", W);
    o += "];\n\n/// True for a named rule (makes an AST node), false for a helper.\n";
    o += "pub static NT_NAMED: [bool; NNT] = [\n";
    o += &pack(t.nts.iter().map(|n| n.named.to_string()), "    ", W);
    o += "];\n\n/// True for a pass-through rule: its node is omitted when it holds exactly one child node\n";
    o += "/// and consumed no terminal itself (the child takes its place).\n";
    o += "pub static NT_PASS: [bool; NNT] = [\n";
    o += &pack(t.nts.iter().map(|n| n.pass.to_string()), "    ", W);
    o += "];\n\n// Productions:\n";
    for i in 0..t.prods.len() {
        let _ = writeln!(o, "//   {i}: {}", t.show_prod(i));
    }
    o += "\n/// Left-hand side per production.\npub static PROD_LHS: [u16; NPROD] = [\n";
    o += &pack(t.prods.iter().map(|p| p.lhs.to_string()), "    ", W);
    o += "];\n\n/// `PROD_RHS[PROD_OFF[p]..PROD_OFF[p + 1]]` is production p's right-hand side.\n";
    o += "pub static PROD_OFF: [u16; NPROD + 1] = [\n";
    o += &pack(nums(&off), "    ", W);
    o +=
        "];\n\n/// Right-hand sides, each reversed (push order): a terminal id, or `NT_BIT | nonterminal`.\n";
    o += "pub static PROD_RHS: [u16; NRHS] = [\n";
    o += &pack(nums(&rhs), "    ", W);
    o += "];\n\n/// `TABLE[nt * NTERM + term]`: the production to expand, or -1 (syntax error).\n";
    o += "pub static TABLE: [i16; NNT * NTERM] = [\n";
    for row in t.table.chunks(t.nterm()) {
        o += &pack(nums(row), "    ", W);
    }
    o += "];\n\n";
    o += "/// The production for nonterminal `nt` on terminal `term`, or -1.\n#[inline]\n";
    o += "pub fn action(nt: u16, term: u16) -> i16 {\n    TABLE[nt as usize * NTERM + term as usize]\n}\n\n";
    o += "/// Production `prod`'s right-hand side, reversed (push order).\n#[inline]\n";
    o += "pub fn rhs_rev(prod: u16) -> &'static [u16] {\n";
    o += "    &PROD_RHS[PROD_OFF[prod as usize] as usize..PROD_OFF[prod as usize + 1] as usize]\n}\n\n";
    o += "/// Number of override cells (the grammar's listed LL(1) conflicts).\n";
    let _ = writeln!(o, "pub const NOVERRIDE: usize = {};\n", t.overrides.len());
    for (i, ov) in t.overrides.iter().enumerate() {
        let alts: Vec<String> =
            ov.alternatives.iter().map(|&p| format!("`{}`", t.show_prod(p as usize))).collect();
        let _ = writeln!(o, "/// `({}, \"{}\")`: {}.", ov.rule, ov.token, ov.why);
        let _ = writeln!(o, "/// Chooses {} over {}.", alts[0], alts[1..].join(", "));
        let _ = writeln!(o, "pub const {}: usize = {i};", names.override_const[i]);
    }
    o += "\n/// Origin rule (nonterminal id) per override cell.\n";
    let _ = writeln!(o, "pub const OVERRIDE_ORIGIN: [u16; NOVERRIDE] = [{}];", list(t, |c| c.origin));
    o += "/// Nonterminal per override cell.\n";
    let _ = writeln!(o, "pub const OVERRIDE_NT: [u16; NOVERRIDE] = [{}];", list(t, |c| c.nt));
    o += "/// Lookahead terminal per override cell.\n";
    let _ = writeln!(o, "pub const OVERRIDE_TERM: [u16; NOVERRIDE] = [{}];", list(t, |c| c.term));
    o += "/// Chosen production per override cell (`TABLE[OVERRIDE_NT * NTERM + OVERRIDE_TERM]`).\n";
    let _ = writeln!(o, "pub const OVERRIDE_PROD: [u16; NOVERRIDE] = [{}];", list(t, |c| c.prod));
    o
}

/// Comma-separated values of one override field.
pub fn list(t: &Tables, f: impl Fn(&crate::OverrideCell) -> u16) -> String {
    t.overrides.iter().map(|c| f(c).to_string()).collect::<Vec<_>>().join(", ")
}

pub fn emit(t: &Tables, names: &Names, header: &str) -> Vec<Output> {
    let out = |file: &str, content: String| Output { path: format!("{RUST_DIR}/{file}"), content };
    vec![
        out("mod.rs", mod_rs(header)),
        out("ll1_table.rs", ll1_table_rs(t, names, header)),
        out("terminals.rs", terminals_rs(t, names, header)),
        out("node_kinds.rs", node_kinds_rs(t, names, header)),
    ]
}
